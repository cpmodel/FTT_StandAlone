"""
=========================================
model_class.py
=========================================

Model class for operating the FTT family of models.

The :class:`RunFTT` class loads model inputs, solves each configured scenario
over the simulation period, and stores the resulting variable arrays.  It is
also the interface used by the GUI for progress reporting and cancellation.
"""

import configparser
import time

import numpy as np

# Separate FTT modules
import ftt_source.Freight.ftt_fr_main as ftt_fr
import ftt_source.Heat.ftt_h_main as ftt_h
import ftt_source.Power.ftt_p_main as ftt_p
import ftt_source.Transport.ftt_tr_main as ftt_tr
import ftt_source.Industrial_Heat.ftt_chi_main as ftt_indhe_chi
import ftt_source.Industrial_Heat.ftt_fbt_main as ftt_indhe_fbt
import ftt_source.Industrial_Heat.ftt_mtm_main as ftt_indhe_mtm
import ftt_source.Industrial_Heat.ftt_nmm_main as ftt_indhe_nmm
import ftt_source.Industrial_Heat.ftt_ois_main as ftt_indhe_ois
# Sector coupling
from ftt_source.sector_coupling.electricity_demand import (
    electricity_demand_feedback,
)
from ftt_source.sector_coupling.electricity_price import (
    electricity_price_feedback,
)
# Support modules
from ftt_source.support import dimensions_functions as dims_f
from ftt_source.support import titles_functions as titles_f
from ftt_source.support import data_loading
from ftt_source.support.cross_section import cross_section


class RunCancelledError(RuntimeError):
    """Raised when a model run is cancelled by the frontend."""


class RunFTT:
    """
    Class to run the FTT model.

    A class instance represents one model run and stores:

    Attributes
    ----------
    titles : dict
        Classification titles used by the sector modules.
    dims : dict
        Active dimensions for each model variable.
    histend : dict
        Last historical year for each variable.
    domain : dict
        FTT model domain associated with each variable.
    forstart : dict
        Forecast start year for each variable.
    unit : dict
        Units associated with each variable.
    input : dict
        Input arrays for each configured scenario.
    variables : dict
        Current-year arrays being solved.
    lags : dict
        Arrays from the preceding model year.
    output : dict
        Solved arrays for each configured scenario.
    """

    def __init__(
        self,
        inputs_path=None,
        utilities_path=None,
        settings_path=None,
        settings=None,
        progress_callback=None,
        log_callback=None,
        stop_callback=None,
    ):
        """Instantiate a model run from settings and input files.

        Parameters
        ----------
        inputs_path : str or Path, optional
            Path to the ``Inputs`` directory.  The configured repository path
            is used when omitted.
        utilities_path : str or Path, optional
            Path to the ``Utilities`` directory.
        settings_path : str or Path, optional
            Path to a ``settings.ini`` file.
        settings : dict, optional
            Values overriding entries in the ``[settings]`` section, such as
            ``{"scenarios": "S0, S1", "simulation_end": "2040"}``.
        progress_callback : callable, optional
            Called as ``progress_callback(completed, total)`` after each
            scenario-year step.
        log_callback : callable, optional
            Called with status messages during loading and solving.
        stop_callback : callable, optional
            Called before each solve step; returning ``True`` cancels the run.
        """
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.stop_callback = stop_callback

        from ftt_source.paths import set_paths

        # Configure data paths before loading titles or input files.
        set_paths(inputs_path=inputs_path, utilities_path=utilities_path)
        config = configparser.ConfigParser()
        if settings_path is None:
            from ftt_source.paths import _PACKAGE_ROOT

            package_settings = _PACKAGE_ROOT / "settings.ini"
            settings_path = package_settings if package_settings.is_file() else "settings.ini"
        config.read(str(settings_path))
        if settings:
            if not config.has_section("settings"):
                config.add_section("settings")
            for key, value in settings.items():
                config.set("settings", key, str(value))

        section = config["settings"]
        self.name = section["name"]
        self.model_start = int(section["model_start"])
        self.model_end = int(section["model_end"])
        self.simulation_start = int(section["simulation_start"])
        self.simulation_end = int(section["simulation_end"])
        self.current = self.model_start
        self.years = np.arange(self.model_start, self.model_end + 1)
        self.timeline = np.arange(self.simulation_start, self.simulation_end + 1)
        self.ftt_modules = section["enable_modules"]
        self.scenarios = section["scenarios"]

        # Load classification titles and variable metadata.
        self.titles = titles_f.load_titles()
        self.dims, self.histend, self.domain, self.forstart, self.unit = (
            dims_f.load_dims()
        )
        # Retrieve inputs for every configured scenario.
        self.input = data_loading.load_data(
            self.titles,
            self.dims,
            self.timeline,
            self.scenarios,
            self.ftt_modules,
            self.forstart,
            log_callback=log_callback,
        )
        self.variables = {}
        self.lags = {}
        self.output = {}

        # Pre-compute FTT:Power constants once per run.
        if "FTT-P" in self.ftt_modules:
            self.power_settings = ftt_p.build_power_settings(self.titles, config)

    def run(self):
        """Solve all configured scenarios and retain their output arrays."""
        self.solve_all()

    def _check_stop_requested(self):
        """Raise :class:`RunCancelledError` when the frontend requests stop."""
        if self.stop_callback and self.stop_callback():
            raise RunCancelledError("Run cancelled by user")

    def solve_all(self):
        """Solve every configured scenario over the simulation timeline.

        The output arrays are allocated from the loaded input arrays so that
        every variable retains the dimensions declared in the variable
        listing.  Progress callbacks receive the total number of
        scenario-year steps and the number completed so far.
        """
        # Define the output container before solving any scenario.
        self.output = {
            scenario: {
                variable: np.zeros_like(values)
                for variable, values in scenario_data.items()
            }
            for scenario, scenario_data in self.input.items()
        }
        total_steps = len(self.input) * len(self.timeline)
        if self.progress_callback:
            self.progress_callback(0, total_steps)

        started = time.time()
        for scenario_index, scenario in enumerate(self.input):
            # Solve each year in the active scenario.
            self._check_stop_requested()
            if self.log_callback:
                self.log_callback(f"Running scenario {scenario}")
            for y, year in enumerate(self.timeline):
                self._check_stop_requested()
                self.variables, self.lags = self.solve_year(year, y, scenario)
                self._store_output(scenario, y)
                if self.progress_callback:
                    self.progress_callback(
                        scenario_index * len(self.timeline) + y + 1,
                        total_steps,
                    )

        message = f"Total elapsed time is {time.time() - started:.2f} seconds"
        if self.log_callback:
            self.log_callback(message)

    def _store_output(self, scenario, year_index):
        """Store one solved model year in the scenario output container.

        Variables with a ``TIME`` dimension are written at the corresponding
        time-axis index.  Variables without ``TIME`` are stored after the last
        solved year because the output retains their final solved state.
        """
        for variable, values in self.variables.items():
            if "TIME" in self.dims[variable]:
                time_axis = self.dims[variable].index("TIME")
                target = self.output[scenario][variable]
                target_index = [slice(None)] * target.ndim
                target_index[time_axis] = year_index
                target[tuple(target_index)] = values
            elif year_index == len(self.timeline) - 1:
                self.output[scenario][variable][...] = values

    def solve_year(self, year, y, scenario, max_iteration=1):
        """Solve one model year for a scenario.

        Demand sectors are solved before the Power sector.  For non-baseline
        scenarios, electricity demand and price feedback are applied around
        the Power solve when the feedback conditions are active.
        """
        variables, time_lags = self.update(year, y, scenario)

        for _ in range(max_iteration):
            # 1. Run demand sectors before Power.
            if "FTT-Tr" in self.ftt_modules:
                variables = ftt_tr.solve(
                    variables, time_lags, self.titles, self.histend, year, self.domain
                )
            if "FTT-Fr" in self.ftt_modules:
                variables = ftt_fr.solve(
                    variables, time_lags, self.titles, self.histend, year, self.domain
                )
            if "FTT-H" in self.ftt_modules:
                variables = ftt_h.solve(
                    variables, time_lags, self.titles, self.histend, year, self.domain
                )

            # Industrial heat modules use the same solver interface.
            industrial_modules = {
                "FTT-IH-CHI": ftt_indhe_chi,
                "FTT-IH-FBT": ftt_indhe_fbt,
                "FTT-IH-MTM": ftt_indhe_mtm,
                "FTT-IH-NMM": ftt_indhe_nmm,
                "FTT-IH-OIS": ftt_indhe_ois,
            }
            for module_name, module in industrial_modules.items():
                if module_name in self.ftt_modules:
                    variables = module.solve(
                        variables,
                        time_lags,
                        self.titles,
                        self.histend,
                        year,
                        self.domain,
                    )

            # 2. Aggregate demand-sector electricity feedback.
            if scenario != "S0" and year > self.histend["MEWG"]:
                variables = electricity_demand_feedback(
                    variables, self.output["S0"], y, self.titles, self.unit
                )

            # 3. Run Power after electricity demand has been updated.
            if "FTT-P" in self.ftt_modules:
                variables = ftt_p.solve(
                    variables,
                    time_lags,
                    self.titles,
                    self.histend,
                    year,
                    self.domain,
                    self.power_settings,
                )
                # 4. Update electricity fuel costs for the following step.
                if year > self.histend["MEWG"]:
                    variables = electricity_price_feedback(variables, time_lags)

        return variables, time_lags

    def update(self, year, y, scenario):
        """Create current-year arrays and the lagged arrays used by solvers.

        The first simulation year uses the corresponding input cross-section
        as its lag.  Subsequent years use the variables solved in the
        preceding year.
        """
        self.current = year
        # Read the required variables for the current model year.
        current = cross_section(self.input, self.dims, year, y, scenario)
        # Keep the starting-year lag independent because solvers mutate current.
        lags = (
            {variable: values.copy() for variable, values in current.items()}
            if y == 0
            else self.variables
        )
        return current, lags
