"""
Module contains functions to generate plots for C40 cities e-truck report.

Functions and classes included:

"""

# Standard library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import configparser

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

FIGURE_WIDTH = 16
ROW_HEIGHT = 4.2

# Local library imports

def plot_mandate_tco(regions, baseline_name, scenario_names, pickle_name='Results'):
    """
    Function to plot yearly ZTTC lines for BEV and Diesel freight trucks.
    Creates one subplot per region and compares BEV across baseline/scenarios
    against a single Diesel baseline line.

    Parameters
    -----------
    regions: list or dict
        Either a list of region numbers or a dictionary mapping
        region numbers to display names.
    baseline_name: str or dict
        Baseline scenario key, or a one-item dictionary
        {scenario_key: friendly_label}.
    scenario_names: str, list, or dict
        Scenario keys to compare, or a dictionary
        {scenario_key: friendly_label}.
    pickle_name: str
        Name of the pickle file containing the results.
    """
    with open(f'Output/{pickle_name}.pickle', 'rb') as f:
        results = pickle.load(f)

    if isinstance(baseline_name, dict):
        if len(baseline_name) != 1:
            raise ValueError("baseline_name dictionary must have exactly one entry: {scenario_key: friendly_label}")
        baseline_key, baseline_label = next(iter(baseline_name.items()))
    else:
        baseline_key = baseline_name
        baseline_label = str(baseline_name)

    if isinstance(scenario_names, dict):
        scenario_label_map = {str(k): str(v) for k, v in scenario_names.items()}
    elif isinstance(scenario_names, str):
        scenario_label_map = {scenario_names: scenario_names}
    else:
        scenario_label_map = {str(s): str(s) for s in scenario_names}

    # Ensure any percent signs in labels are escaped for safe rendering
    scenario_label_map = {k: v.replace('%', '%%') for k, v in scenario_label_map.items()}
    baseline_label = str(baseline_label).replace('%', '%%')

    if baseline_key not in results:
        raise KeyError(f"Baseline scenario '{baseline_key}' not found in results.")

    # Always include baseline BEV line, then additional scenarios
    scenario_keys = [baseline_key] + [k for k in scenario_label_map.keys() if k != baseline_key]
    scenario_labels = {baseline_key: str(baseline_label)}
    scenario_labels.update(scenario_label_map)

    missing_scenarios = [s for s in scenario_keys if s not in results]
    if missing_scenarios:
        raise KeyError(f"Scenario(s) not found in results: {missing_scenarios}")

    zttc_baseline = results[baseline_key]['ZTTC']

    if zttc_baseline.ndim != 4:
        raise ValueError(f"Expected ZTTC to be 4D (region, technology, cost_component, year), got shape {zttc_baseline.shape}")

    if isinstance(regions, dict):
        region_ids = list(regions.keys())
        region_labels = [str(regions[region_id]) for region_id in region_ids]
    else:
        region_ids = list(regions)
        region_labels = [f'Region {region_id}' for region_id in region_ids]

    n_regions = zttc_baseline.shape[0]
    region_indices = np.asarray(region_ids, dtype=int) - 1

    if np.any(region_indices < 0) or np.any(region_indices >= n_regions):
        raise ValueError(f"Region indices must be in 1..{n_regions}. Received: {region_ids}")

    n_techs = zttc_baseline.shape[1]

    # User-provided zero-based indices
    tech_indices = {
        'MDT': {'diesel': 12, 'bev': 32},
        'HDT': {'diesel': 13, 'bev': 33},
    }

    for vehicle_class, index_map in tech_indices.items():
        for fuel, idx in index_map.items():
            if idx < 0 or idx >= n_techs:
                raise ValueError(
                    f"{fuel.upper()} {vehicle_class} index {idx} is out of bounds for {n_techs} technologies"
                )

    diesel_series_by_class = {
        vehicle_class: zttc_baseline[region_indices, index_map['diesel'], 0, :]
        for vehicle_class, index_map in tech_indices.items()
    }
    bev_series_by_class_scenario = {
        vehicle_class: {
            scenario: results[scenario]['ZTTC'][region_indices, index_map['bev'], 0, :]
            for scenario in scenario_keys
        }
        for vehicle_class, index_map in tech_indices.items()
    }

    # Build year axis from settings.ini and validate against results timeline length
    config = configparser.ConfigParser()
    config.read('settings.ini')
    simulation_start = int(config['settings']['simulation_start'])
    simulation_end = int(config['settings']['simulation_end'])
    years = np.arange(simulation_start, simulation_end + 1)
    n_years = zttc_baseline.shape[3]
    if len(years) != n_years:
        years = np.arange(simulation_start, simulation_start + n_years)

    plot_start_year = 2025
    year_mask = years >= plot_start_year
    if not np.any(year_mask):
        raise ValueError(f"No years available at or after {plot_start_year}.")
    years = years[year_mask]

    diesel_series_by_class = {
        vehicle_class: series[:, year_mask]
        for vehicle_class, series in diesel_series_by_class.items()
    }
    bev_series_by_class_scenario = {
        vehicle_class: {
            scenario: series[:, year_mask]
            for scenario, series in scenario_map.items()
        }
        for vehicle_class, scenario_map in bev_series_by_class_scenario.items()
    }

    def get_strict_crossover_year(bev_line, diesel_line, year_axis):
        diff = bev_line - diesel_line

        # Strict parity event: transition from BEV > Diesel to BEV <= Diesel
        transition_indices = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0]
        if transition_indices.size == 0:
            return None

        i = int(transition_indices[0])
        y0, y1 = float(year_axis[i]), float(year_axis[i + 1])
        d0, d1 = float(diff[i]), float(diff[i + 1])

        if d1 == d0:
            return y1

        # Linear interpolation for diff == 0 crossing
        fraction = -d0 / (d1 - d0)
        return y0 + fraction * (y1 - y0)

    # Plot: one row per region, MDT left and HDT right
    n_regions_selected = len(region_ids)
    fig, axes = plt.subplots(
        n_regions_selected,
        2,
        figsize=(FIGURE_WIDTH, ROW_HEIGHT * n_regions_selected),
        sharex=True,
        sharey=False
    )
    # Increase vertical spacing between subplot rows
    fig.subplots_adjust(hspace=10)
    axes = np.atleast_2d(axes)
    legend_handles = {}

    col_order = ['MDT', 'HDT']
    for i in range(n_regions_selected):
        for j, vehicle_class in enumerate(col_order):
            ax = axes[i, j]
            diesel_line = diesel_series_by_class[vehicle_class][i]
            diesel_plot = ax.plot(
                years,
                diesel_line,
                label=f'Diesel ({scenario_labels[baseline_key]})',
                linewidth=2,
                linestyle='--',
                color='black'
            )[0]
            legend_handles[f'Diesel'] = diesel_plot
            plt.margins(x=0)  # Remove horizontal margins to align with year ticks

            crossover_years = {}
            scenario_colors = {}
            bev_lines_by_scenario = {}

            for scenario in scenario_keys:
                bev_line = bev_series_by_class_scenario[vehicle_class][scenario][i]
                bev_plot = ax.plot(years, bev_line, label=f'BEV ({scenario_labels[scenario]})', linewidth=2)[0]
                legend_handles[f'BEV ({scenario_labels[scenario]})'] = bev_plot
                scenario_colors[scenario] = bev_plot.get_color()
                bev_lines_by_scenario[scenario] = bev_line

                cross_year = get_strict_crossover_year(bev_line, diesel_line, years)
                crossover_years[scenario] = cross_year
                if cross_year is not None:
                    ax.axvline(
                        cross_year,
                        linestyle=':',
                        linewidth=2,
                        color=bev_plot.get_color(),
                        alpha=0.85
                    )

            baseline_cross_year = crossover_years.get(baseline_key)
            annotation_index = 0
            for scenario in scenario_keys:
                if scenario == baseline_key:
                    continue
                scenario_cross_year = crossover_years.get(scenario)
                if baseline_cross_year is None or scenario_cross_year is None:
                    continue

                delta_years = scenario_cross_year - baseline_cross_year
                y_at_cross = float(np.interp(scenario_cross_year, years, bev_lines_by_scenario[scenario]))

                ax.annotate(
                    f'{delta_years:+.2f} y',
                    xy=(scenario_cross_year, y_at_cross),
                    xytext=(25, -20 * annotation_index),
                    textcoords='offset points',
                    color=scenario_colors[scenario],
                    fontsize=12,
                    weight='bold',
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7)
                )
                annotation_index += 1

            ax.set_title(f'{region_labels[i]} - {vehicle_class}', weight='bold')
            if j == 0:
                # ax.set_ylabel('Levelized Cost ($/tkm)')
                pass
            ax.grid(True, alpha=0.3)

    fig.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=max(1, len(legend_handles)),
        frameon=True
    )
    fig.supylabel('Levelized Cost ($/tkm)', ha='left', va='center', x=0.01)

    # Ensure adequate vertical padding when using tight layout
    plt.tight_layout(rect=(0, 0.08, 1, 0.97), h_pad=1.2)
    plt.show()


def plot_ztvt_timeseries(regions, scenario_name, pickle_name='Results', tech_indices=(32, 33)):
    """
    Plot ZTVT subsidy-rate time series (2025-2050) for one scenario
    in two panels: MDT (left) and HDT (right).

    Parameters
    -----------
    regions: dict
        Dictionary mapping region numbers to display names.
    scenario_name: str or dict
        Scenario key, or a one-item dictionary {scenario_key: friendly_label}.
    pickle_name: str
        Name of the pickle file containing the results.
    tech_indices: iterable of int
        Two zero-based technology indices in order (MDT, HDT).
        Default is BEV MDT and BEV HDT: (32, 33).
    """
    if not isinstance(regions, dict):
        raise ValueError("regions must be a dictionary of {region_id: region_label}.")

    if isinstance(scenario_name, dict):
        if len(scenario_name) != 1:
            raise ValueError("scenario_name dictionary must have exactly one entry: {scenario_key: friendly_label}")
        scenario_key, scenario_label = next(iter(scenario_name.items()))
    else:
        scenario_key = scenario_name
        scenario_label = str(scenario_name)

    with open(f'Output/{pickle_name}.pickle', 'rb') as f:
        results = pickle.load(f)

    if scenario_key not in results:
        raise KeyError(f"Scenario '{scenario_key}' not found in results.")

    ztvt = results[scenario_key]['ZTVT']
    if ztvt.ndim != 4:
        raise ValueError(f"Expected ZTVT to be 4D (region, technology, cost_component, year), got shape {ztvt.shape}")

    region_ids = list(regions.keys())
    region_labels = [str(regions[region_id]) for region_id in region_ids]
    region_indices = np.asarray(region_ids, dtype=int) - 1

    n_regions = ztvt.shape[0]
    if np.any(region_indices < 0) or np.any(region_indices >= n_regions):
        raise ValueError(f"Region indices must be in 1..{n_regions}. Received: {region_ids}")

    tech_indices = np.asarray(list(tech_indices), dtype=int)
    n_techs = ztvt.shape[1]
    if tech_indices.size != 2:
        raise ValueError("tech_indices must contain exactly two indices in order: (MDT, HDT).")
    if np.any(tech_indices < 0) or np.any(tech_indices >= n_techs):
        raise ValueError(f"Technology indices must be in 0..{n_techs - 1}. Received: {tech_indices.tolist()}")

    mdt_idx, hdt_idx = int(tech_indices[0]), int(tech_indices[1])

    config = configparser.ConfigParser()
    config.read('settings.ini')
    simulation_start = int(config['settings']['simulation_start'])
    simulation_end = int(config['settings']['simulation_end'])
    years = np.arange(simulation_start, simulation_end + 1)
    n_years = ztvt.shape[3]
    if len(years) != n_years:
        years = np.arange(simulation_start, simulation_start + n_years)

    year_mask = (years >= 2025) & (years <= 2050)
    if not np.any(year_mask):
        raise ValueError("No years available in the requested 2025-2050 range.")
    years = years[year_mask]

    mdt_series = ztvt[region_indices, mdt_idx, 0, :][:, year_mask]
    hdt_series = ztvt[region_indices, hdt_idx, 0, :][:, year_mask]

    # Increase font sizes locally for this figure
    with plt.rc_context({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, ROW_HEIGHT), sharex=True, sharey=True)
        cmap = plt.get_cmap('tab10')
        legend_handles = []
        legend_labels = []

        import matplotlib.ticker as mtick
        for i, region_label in enumerate(region_labels):
            color = cmap(i % 10)
            mdt_line = axes[0].plot(years, mdt_series[i], linewidth=2, color=color, label=region_label)[0]
            axes[1].plot(years, hdt_series[i], linewidth=2, color=color)
            legend_handles.append(mdt_line)
            legend_labels.append(region_label)
            # add to y margin
            axes[0].set_ylim(axes[0].get_ylim()[0] * 1.07, axes[0].get_ylim()[1] * 1.07)
            axes[1].set_ylim(axes[1].get_ylim()[0] * 1.07, axes[1].get_ylim()[1] * 1.07)

        # Format y-axes as percent (assumes data in fractional units, e.g. 0.1 == 10%)
        percent_formatter = mtick.PercentFormatter(xmax=1.0)
        axes[0].yaxis.set_major_formatter(percent_formatter)
        axes[1].yaxis.set_major_formatter(percent_formatter)

        axes[0].set_title('MDT', weight='bold')
        axes[1].set_title('HDT', weight='bold')
        axes[0].set_ylabel('Subsidy required to achieve TCO parity')
        for ax in axes:
            ax.grid(True, alpha=0.3)

        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=max(1, len(region_labels)),
            frameon=True
        )
        # plt.tight_layout(rect=(0, 0.08, 1, 0.95))
        plt.show()


def plot_zest_hdt_stacked(regions, scenarios, pickle_name='Results'):
    """
    Plot stacked area charts of ZEST (service area MTKM) for HDT technology groups.

    Layout:
    - Scenarios across columns
    - Regions down rows

    Parameters
    ----------
    regions : dict
        Dictionary mapping region numbers to region names.
    scenarios : dict
        Dictionary mapping scenario keys to scenario display names.
    pickle_name : str
        Name of pickle file in Output/ (without extension).
    """
    if not isinstance(regions, dict) or len(regions) == 0:
        raise ValueError("regions must be a non-empty dictionary of {region_id: region_label}.")
    if not isinstance(scenarios, dict) or len(scenarios) == 0:
        raise ValueError("scenarios must be a non-empty dictionary of {scenario_key: scenario_label}.")

    with open(f'Output/{pickle_name}.pickle', 'rb') as f:
        results = pickle.load(f)

    scenario_keys = list(scenarios.keys())
    scenario_labels = {str(k): str(v) for k, v in scenarios.items()}
    missing_scenarios = [scenario for scenario in scenario_keys if scenario not in results]
    if missing_scenarios:
        raise KeyError(f"Scenario(s) not found in results: {missing_scenarios}")

    region_ids = list(regions.keys())
    region_labels = [str(regions[region_id]) for region_id in region_ids]

    # HDT indices from user spec (0-based): 3,8,13,18,23,28,33,38,43
    group_indices = {
        'Diesel': [13, 18],
        'CNG/LPG': [23],
        'BEV': [33],
        'Other': [3, 8, 28, 38, 43],
    }
    group_order = ['Diesel', 'CNG/LPG', 'BEV', 'Other']
    group_colors = ['#264653', '#2a9d8f', '#4cc9f0', '#8d99ae']

    sample = results[scenario_keys[0]]['ZEWK']
    if sample.ndim != 4:
        raise ValueError(f"Expected ZEWK to be 4D (region, technology, cost_component, year), got shape {sample.shape}")

    n_regions_total = sample.shape[0]
    n_techs = sample.shape[1]
    region_indices = np.asarray(region_ids, dtype=int) - 1
    if np.any(region_indices < 0) or np.any(region_indices >= n_regions_total):
        raise ValueError(f"Region indices must be in 1..{n_regions_total}. Received: {region_ids}")

    for label, idxs in group_indices.items():
        arr = np.asarray(idxs)
        if np.any(arr < 0) or np.any(arr >= n_techs):
            raise ValueError(f"Group {label} has out-of-bounds tech indices for n_techs={n_techs}: {idxs}")

    config = configparser.ConfigParser()
    config.read('settings.ini')
    simulation_start = int(config['settings']['simulation_start'])
    simulation_end = int(config['settings']['simulation_end'])
    years = np.arange(simulation_start, simulation_end + 1)
    n_years = sample.shape[3]
    if len(years) != n_years:
        years = np.arange(simulation_start, simulation_start + n_years)

    year_mask = (years >= 2025) & (years <= 2050)
    if not np.any(year_mask):
        raise ValueError("No years available in the requested 2025-2050 range.")
    years = years[year_mask]

    n_rows = len(region_ids)
    n_cols = len(scenario_keys)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIGURE_WIDTH + 1 * n_cols, ROW_HEIGHT * n_rows),
        sharex=True,
        sharey=False,
        layout='constrained'
    )
    axes = np.atleast_2d(axes)

    stack_handles = None
    for r, (region_id, region_label) in enumerate(zip(region_ids, region_labels)):
        region_idx = int(region_id) - 1
        for c, scenario_key in enumerate(scenario_keys):
            ax = axes[r, c]
            zewk = results[scenario_key]['ZEWK']

            grouped = []
            for group in group_order:
                idxs = group_indices[group]
                grouped.append(np.sum(zewk[region_idx, idxs, 0, :], axis=0)[year_mask] / 1000.0)

            stack_handles = ax.stackplot(
                years,
                grouped,
                labels=group_order,
                colors=group_colors,
                alpha=0.9
            )
            ax.margins(x=0)
            ax.set_xlim(years[0], years[-1])

            if r == 0:
                ax.set_title(scenario_labels[scenario_key], weight='bold')
            if c == 0:
                    ax.set_ylabel(region_label, labelpad=5, weight='bold')

    if stack_handles is not None:
        fig.legend(
            handles=stack_handles,
            labels=group_order,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(group_order),
            frameon=True
        )

        fig.supylabel('Number of trucks (thousands)', ha='left', va='center', x=0.01)
        plt.tight_layout(rect=(0.01, 0.08, 1, 0.95), h_pad=1.1)

    plt.show()




if __name__ == '__main__':
    # plot_mandate_tco(
    #     regions={41:"China", 3:"Germany", 42:"India", 10:"Netherlands"},
    #     baseline_name={'S0': 'Baseline'},
    #     scenario_names={'city_mandates_real': 'City mandates (real targets)',
    #                     'city_mandates_75': 'City mandates (75pct by 2040)',
    #                     'city_mandates_100': 'City mandates (100pct by 2040)'},
    #     pickle_name='Results'
    # )
    
    plot_ztvt_timeseries(
        regions={82:"Berlin", 115:"Delhi", 87:"Milan", 110:"London", 95:"Rotterdam"},
        scenario_name={'tco_parity': 'TCO Parity'},
        pickle_name='Results_tco_subsidy',
        tech_indices=(32, 33)
    )
    
    # plot_zest_hdt_stacked(
    #     regions={82:"Berlin", 115:"Delhi", 110:"London", 87:"Milan", 95:"Rotterdam", 75:"Shanghai"},
    #     scenarios={'S0': 'Baseline',
    #                "carbon_tax": "Carbon tax",
    #                "emissions_regulation": "Emissions regulation",
    #                'city_mandates_100': 'City mandates (100pct by 2040)'},
    #     pickle_name='Results'
    # )
    