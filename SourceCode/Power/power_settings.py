"""One-time FTT:Power settings assembly.

The :class:`PowerSettings` object is constructed once in ``RunFTT.__init__``
and then reused for every call to the Power solver. Keeping the config parsing
and static lookup construction here avoids rebuilding the same Power metadata
for every simulation year.
"""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PowerSettings:
    """Static Power config and lookups reused across a model run."""

    prsc_base_year: int
    ex_base_year: int
    usd_exchange_region: str
    rldc_start_year: int
    bcet_copy_range_end: int
    gamma_mode: str
    sector_coupling: bool
    mset_coupling: bool
    elec_idx: int
    usd_idx: int
    nuclear_tech_idx: int
    prsc_var: str
    ex_var: str
    rex_var: str
    tech_to_resource: tuple[int, ...]
    erti_jti_map: Mapping[int, tuple[int, ...]]
    cf_multipliers: Mapping[int, float]
    wind_solar_indices: Mapping[str, object]
    gen_tech_indices: Mapping[str, tuple[tuple[int, ...], int]]


def _title_index(titles, dimension, label):
    return list(titles[dimension]).index(label)


def _build_tech_to_resource(titles):
    tech_resource_labels = (
        ("1 Nuclear", "1 Nuclear"),
        ("2 Oil", "2 Oil"),
        ("3 Coal", "3 Coal"),
        ("4 Coal + CCS", "3 Coal"),
        ("5 Waste", "6 Biogas"),
        ("6 Waste + CCS", "6 Biogas"),
        ("7 CCGT", "4 Gas"),
        ("8 CCGT + CCS", "4 Gas"),
        ("9 OCGT", "4 Gas"),
        ("10 OCGT + CCS", "4 Gas"),
        ("11 Biomass", "5 Biomass"),
        ("12 Biomass + CCS", "5 Biomass"),
        ("13 Large Hydro", "9 Large Hydro"),
        ("14 Pumped Hydro", "9 Large Hydro"),
        ("15 Geothermal", "13 Geothermal"),
        ("16 Marine", "8 Tidal"),
        ("17 Onshore", "10 Onshore"),
        ("18 Offshore", "11 Offshore"),
        ("19 Solar PV", "12 Solar PV"),
        ("20 CSP", "12 Solar PV"),
        ("21 Fuel cells / Turbine", "7 None (storage)"),
        ("22 Lithium-ion", "7 None (storage)"),
    )
    return tuple(
        _title_index(titles, "ERTI", resource_label)
        for _, resource_label in tech_resource_labels
    )


def _build_erti_jti_map(titles):
    return {
        _title_index(titles, "ERTI", "2 Oil"): (
            _title_index(titles, "JTI", "3 Crude oil etc"),
            _title_index(titles, "JTI", "4 Heavy fuel oil"),
            _title_index(titles, "JTI", "5 Middle distillates"),
        ),
        _title_index(titles, "ERTI", "3 Coal"): (
            _title_index(titles, "JTI", "1 Hard coal"),
            _title_index(titles, "JTI", "2 Other coal etc"),
        ),
        _title_index(titles, "ERTI", "4 Gas"): (
            _title_index(titles, "JTI", "6 Other gas"),
            _title_index(titles, "JTI", "7 Natural gas"),
        ),
    }


def _build_cf_multipliers(titles):
    return {
        _title_index(titles, "T2TI", "20 CSP"): 2.0,
    }


def _build_wind_solar_indices(titles):
    return {
        "wind": (
            _title_index(titles, "T2TI", "17 Onshore"),
            _title_index(titles, "T2TI", "18 Offshore"),
        ),
        "solar": _title_index(titles, "T2TI", "19 Solar PV"),
        "csp": _title_index(titles, "T2TI", "20 CSP"),
    }


def _build_gen_tech_indices(titles):
    return {
        "nuclear": (
            (_title_index(titles, "T2TI", "1 Nuclear"),),
            _title_index(titles, "ERTI", "1 Nuclear"),
        ),
        "biomass": (
            (
                _title_index(titles, "T2TI", "11 Biomass"),
                _title_index(titles, "T2TI", "12 Biomass + CCS"),
            ),
            _title_index(titles, "ERTI", "5 Biomass"),
        ),
        "biogas": (
            (
                _title_index(titles, "T2TI", "5 Waste"),
                _title_index(titles, "T2TI", "6 Waste + CCS"),
            ),
            _title_index(titles, "ERTI", "6 Biogas"),
        ),
        "tidal": (
            (_title_index(titles, "T2TI", "16 Marine"),),
            _title_index(titles, "ERTI", "8 Tidal"),
        ),
        "hydro": (
            (
                _title_index(titles, "T2TI", "13 Large Hydro"),
                _title_index(titles, "T2TI", "14 Pumped Hydro"),
            ),
            _title_index(titles, "ERTI", "9 Large Hydro"),
        ),
        "onshore": (
            (_title_index(titles, "T2TI", "17 Onshore"),),
            _title_index(titles, "ERTI", "10 Onshore"),
        ),
        "offshore": (
            (_title_index(titles, "T2TI", "18 Offshore"),),
            _title_index(titles, "ERTI", "11 Offshore"),
        ),
        "solar": (
            (
                _title_index(titles, "T2TI", "19 Solar PV"),
                _title_index(titles, "T2TI", "20 CSP"),
            ),
            _title_index(titles, "ERTI", "12 Solar PV"),
        ),
        "geothermal": (
            (_title_index(titles, "T2TI", "15 Geothermal"),),
            _title_index(titles, "ERTI", "13 Geothermal"),
        ),
    }


def load_power_settings(titles, config):
    """Build the Power settings object once per run and reuse it in ``solve()``."""

    prsc_base_year = int(config.get("settings", "prsc_base_year", fallback="2013"))
    ex_base_year = int(config.get("settings", "ex_base_year", fallback="2013"))
    usd_exchange_region = config.get(
        "settings",
        "usd_exchange_region",
        fallback="34 USA (US)",
    )

    return PowerSettings(
        prsc_base_year=prsc_base_year,
        ex_base_year=ex_base_year,
        usd_exchange_region=usd_exchange_region,
        rldc_start_year=int(config.get("settings", "rldc_start_year", fallback="2013")),
        bcet_copy_range_end=int(
            config.get("settings", "bcet_copy_range_end", fallback="22")
        ),
        gamma_mode=config.get(
            "settings",
            "gamma_mode",
            fallback="multiplicative",
        ),
        sector_coupling=config.getboolean(
            "settings",
            "sector_coupling",
            fallback=True,
        ),
        mset_coupling=config.getboolean(
            "settings",
            "mset_coupling",
            fallback=False,
        ),
        elec_idx=_title_index(titles, "JTI", "8 Electricity"),
        usd_idx=_title_index(titles, "RTI", usd_exchange_region),
        nuclear_tech_idx=_title_index(titles, "T2TI", "1 Nuclear"),
        prsc_var=f"PRSC{str(prsc_base_year)[2:]}",
        ex_var=f"EX{str(ex_base_year)[2:]}",
        rex_var=f"REX{str(ex_base_year)[2:]}",
        tech_to_resource=_build_tech_to_resource(titles),
        erti_jti_map=_build_erti_jti_map(titles),
        cf_multipliers=_build_cf_multipliers(titles),
        wind_solar_indices=_build_wind_solar_indices(titles),
        gen_tech_indices=_build_gen_tech_indices(titles),
    )
