# Global imports
import pickle
import configparser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

def plot_mandate_tco(
    regions, 
    baseline_name, 
    scenario_names, 
    pickle_name='Results',
    FIGURE_WIDTH=7,
    ROW_HEIGHT=3.6,
    output_name='tco_parity'):
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
    output_name: str
        Name of the output file (without extension).
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
    fig.subplots_adjust(hspace=.2, wspace=.3)
    axes = np.atleast_2d(axes)
    legend_handles = {}

    col_order = ['MDT', 'HDT']
    col_titles = {'MDT': 'Medium duty truck', 'HDT': 'Heavy duty truck'}

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

            color_dict = {
                'S0': '#4cc9f0',
                'city_mandates_2030': '#1f77b4',
                'city_mandates_2035': '#ff7f0e',
                'city_mandates_2040': '#2ca02c',
            }
            crossover_years = {}
            scenario_colors = {}
            bev_lines_by_scenario = {}

            for scenario in scenario_keys:
                bev_line = bev_series_by_class_scenario[vehicle_class][scenario][i]
                bev_plot = ax.plot(years, bev_line, color=color_dict.get(scenario), label=f'BEV ({scenario_labels[scenario]})', linewidth=2)[0]
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
            annotation_entries = []
            for scenario in scenario_keys:
                if scenario == baseline_key:
                    continue
                scenario_cross_year = crossover_years.get(scenario)
                if baseline_cross_year is None or scenario_cross_year is None:
                    continue

                delta_years = scenario_cross_year - baseline_cross_year
                y_at_cross = float(np.interp(scenario_cross_year, years, bev_lines_by_scenario[scenario]))

                annotation_entries.append((f'{delta_years:+.2f} yrs', scenario_colors[scenario]))

            if annotation_entries:
                annotation_boxes = [
                    TextArea(
                        text,
                        textprops=dict(color=color, fontsize=10, weight='bold')
                    )
                    for text, color in annotation_entries
                ]
                annotation_box = VPacker(
                    children=annotation_boxes,
                    align='right',
                    pad=0,
                    sep=4
                )
                anchored_box = AnchoredOffsetbox(
                    loc='upper right',
                    child=annotation_box,
                    pad=0.2,
                    borderpad=0.2,
                    frameon=True,
                    bbox_to_anchor=(0.98, 0.98),
                    bbox_transform=ax.transAxes
                )
                anchored_box.patch.set_boxstyle('round,pad=0.2')
                anchored_box.patch.set_facecolor('white')
                anchored_box.patch.set_edgecolor('gray')
                anchored_box.patch.set_alpha(0.85)
                ax.add_artist(anchored_box)

            if i == 0:
                ax.set_title(col_titles[vehicle_class], weight='bold')
            if j == 0:
                ax.set_ylabel(region_labels[i], fontsize=14, labelpad=5, weight='bold')
            # ax.set_title(f'{region_labels[i]} - {vehicle_class}', weight='bold')
            # ax.grid(True, alpha=0.3)

    fig.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc='lower center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        frameon=True
    )
    fig.supylabel('Levelized Cost ($/tkm)', ha='left', va='center', fontsize=14, x=-0.04)

    # Ensure adequate vertical padding when using tight layout
    plt.savefig(f'Figures/output/{output_name}.png', dpi=300, bbox_inches="tight")
    
    
def plot_tco_years(
    regions, 
    pickle_name='Results',
    FIGURE_WIDTH=7,
    ROW_HEIGHT=3.6,
    output_name='tco_parity_years'):
    """
    Function to create a scatter plot, showing year TCO partity is achieved on the x-axis, 
    and the y-axis showing the BEV market share in 2050.

    Parameters
    -----------
    regions: list or dict
        Either a list of region numbers or a dictionary mapping
        region numbers to display names.
    pickle_name: str
        Name of the pickle file containing the results.
    output_name: str
        Name of the output file (without extension).
    """
    # Load results and extract ZEWS and ZTTC data
    with open(f'Output/{pickle_name}.pickle', 'rb') as f:
        results = pickle.load(f)
    zttc_baseline = results['S0']['ZTTC']
    zews_baseline = results['S0']['ZEWS']
    
    # Build year axis from settings.ini and validate against results timeline length
    config = configparser.ConfigParser()
    config.read('settings.ini')
    simulation_start = int(config['settings']['simulation_start'])
    simulation_end = int(config['settings']['simulation_end'])
    years = np.arange(simulation_start, simulation_end + 1)
    n_years = zttc_baseline.shape[3]
    if len(years) != n_years:
        years = np.arange(simulation_start, simulation_start + n_years)
        
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
    bev_series_by_class = {
        vehicle_class: zttc_baseline[region_indices, index_map['bev'], 0, :]
        for vehicle_class, index_map in tech_indices.items()
    }
    zews_series_by_class = {
        vehicle_class: zews_baseline[region_indices, index_map['bev'], 0, :]
        for vehicle_class, index_map in tech_indices.items()
    }
    # Define crossover year function
    def get_strict_crossover_year(bev_line, diesel_line, year_axis):
        diff = bev_line - diesel_line
        transition_indices = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0]
        if transition_indices.size == 0:
            return None
        i = int(transition_indices[0])
        y0, y1 = float(year_axis[i]), float(year_axis[i + 1])
        d0, d1 = float(diff[i]), float(diff[i + 1])
        if d1 == d0:
            return y1
        fraction = -d0 / (d1 - d0)
        return y0 + fraction * (y1 - y0)
    # Determine crossover years and 2050 market shares
    crossover_years = {}
    market_shares_2050 = {}
    for vehicle_class in tech_indices.keys():
        crossover_years[vehicle_class] = []
        market_shares_2050[vehicle_class] = []
        for i in range(len(region_indices)):
            diesel_line = diesel_series_by_class[vehicle_class][i]
            bev_line = bev_series_by_class[vehicle_class][i]
            zews_line = zews_series_by_class[vehicle_class][i]
            cross_year = get_strict_crossover_year(bev_line, diesel_line, years)
            crossover_years[vehicle_class].append(cross_year)
            market_share_2050 = float(zews_line[-1])
            market_shares_2050[vehicle_class].append(market_share_2050)
        
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, ROW_HEIGHT))
    markers = {'MDT': 'o', 'HDT': 's'}
    colors = {'MDT': '#1f77b4', 'HDT': '#ff7f0e'}
    col_titles = {'MDT': 'Medium duty truck', 'HDT': 'Heavy duty truck'}
    for vehicle_class in tech_indices.keys():
        ax.scatter(
            crossover_years[vehicle_class],
            market_shares_2050[vehicle_class],
            label=col_titles[vehicle_class],
            marker=markers[vehicle_class],
            color=colors[vehicle_class],
            s=100,
            edgecolor='black',
            alpha=0.8
        )
        # Add region labels to points
        for i, (cross_year, market_share) in enumerate(zip(crossover_years[vehicle_class], market_shares_2050[vehicle_class])):
            if cross_year is not None:
                ax.annotate(
                    f'{region_labels[i]}',
                    xy=(cross_year, market_share),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.7
                )
    ax.set_xlabel('Year of BEV TCO parity', fontsize=12)
    ax.set_ylabel('BEV market share in 2050', fontsize=12)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.set_xlim(left=2025, right=2050)
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f'Figures/output/{output_name}.png', dpi=300, bbox_inches="tight")
    # plt.savefig(f'Figures/output/{output_name}.svg', bbox_inches="tight")