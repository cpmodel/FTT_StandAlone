"""
Script contains fucntions to generate plots mapping current trajectories of shares
and costs with sensitivities.
"""
# Global imports
import pickle
import configparser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_costs(
    regions, 
    scenario_name_map,
    scenario_type_map = {"S0": "main",
                         "S1": "positive",
                         "S2": "negative"}, 
    pickle_name='Results',
    FIGURE_WIDTH=7,
    ROW_HEIGHT=3.3,
    output_name='current_traj_costs'):
    """
    Plot line charts of costs for HDT and MDT technology groups.
    Function uses the ZTTC variable.

    Layout:
    - Two cols, MDT on left and HDT on right
    - Regions down rows

    Parameters
    ----------
    regions : dict
        Dictionary mapping region numbers to region names.
    scenario_name_map : dict
        Dictionary mapping scenario keys to scenario display names.
    scenario_type_map : dict
        Dictionary mapping scenario keys to whether scenario is main estimate or positive/negative sensitivity.
    pickle_name : str
        Name of pickle file in Output/ (without extension).
    output_name : str
        Name of the output file (without extension).
    """
    if not isinstance(regions, dict) or len(regions) == 0:
        raise ValueError("regions must be a non-empty dictionary of {region_id: region_label}.")
    if not isinstance(scenario_name_map, dict) or len(scenario_name_map) == 0:
        raise ValueError("scenario_name_map must be a non-empty dictionary of {scenario_key: scenario_label}.")

    with open(f'Output/{pickle_name}.pickle', 'rb') as f:
        results = pickle.load(f)

    # Identify main, positive, and negative scenario keys
    main_key = next((k for k, v in scenario_type_map.items() if v == "main"), None)
    pos_key  = next((k for k, v in scenario_type_map.items() if v == "positive"), None)
    neg_key  = next((k for k, v in scenario_type_map.items() if v == "negative"), None)

    if main_key is None:
        raise ValueError("scenario_type_map must contain a 'main' entry.")

    required_keys = [k for k in [main_key, pos_key, neg_key] if k is not None]
    missing = [k for k in required_keys if k not in results]
    if missing:
        raise KeyError(f"Scenario(s) not found in results: {missing}")

    zttc_main = results[main_key]['ZTTC']
    if zttc_main.ndim != 4:
        raise ValueError(
            f"Expected ZTTC to be 4D (region, technology, cost_component, year), got shape {zttc_main.shape}"
        )

    region_ids = list(regions.keys())
    region_labels = [str(regions[r]) for r in region_ids]
    region_indices = np.asarray(region_ids, dtype=int) - 1
    n_regions_total = zttc_main.shape[0]
    if np.any(region_indices < 0) or np.any(region_indices >= n_regions_total):
        raise ValueError(f"Region indices must be in 1..{n_regions_total}. Received: {region_ids}")

    # Zero-based technology indices (consistent with rest of codebase)
    tech_indices = {
        'MDT': {'diesel': 12, 'cng': 22, 'bev': 32},
        'HDT': {'diesel': 13, 'cng': 23, 'bev': 33},
    }
    n_techs = zttc_main.shape[1]
    for vc, idx_map in tech_indices.items():
        for fuel, idx in idx_map.items():
            if idx < 0 or idx >= n_techs:
                raise ValueError(
                    f"{fuel.upper()} {vc} index {idx} is out of bounds for n_techs={n_techs}"
                )

    config = configparser.ConfigParser()
    config.read('settings.ini')
    simulation_start = int(config['settings']['simulation_start'])
    simulation_end = int(config['settings']['simulation_end'])
    years = np.arange(simulation_start, simulation_end + 1)
    n_years = zttc_main.shape[3]
    if len(years) != n_years:
        years = np.arange(simulation_start, simulation_start + n_years)

    year_mask = (years >= 2025) & (years <= 2050)
    if not np.any(year_mask):
        raise ValueError("No years available in 2025-2050 range.")
    years = years[year_mask]

    col_order = ['MDT', 'HDT']
    col_titles = {'MDT': 'Medium duty truck', 'HDT': 'Heavy duty truck'}
    bev_color    = '#4cc9f0'
    cng_color    = '#f4a261'
    diesel_color = 'black'

    n_rows = len(region_ids)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(FIGURE_WIDTH, ROW_HEIGHT * n_rows),
        sharex=True,
        sharey=False
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.3)
    axes = np.atleast_2d(axes)

    legend_handles = {}

    for r, (region_id, region_label) in enumerate(zip(region_ids, region_labels)):
        ri = int(region_id) - 1

        for c, vc in enumerate(col_order):
            ax = axes[r, c]
            comparison_fuel = 'diesel'
            comparison_color = diesel_color
            comparison_linestyle = '--'
            if vc == 'MDT' and str(region_label).strip().lower() == 'delhi':
                comparison_fuel = 'cng'
                comparison_color = cng_color
                comparison_linestyle = '-.'

            comparison_idx = tech_indices[vc].get(comparison_fuel, tech_indices[vc]['diesel'])
            bev_idx    = tech_indices[vc]['bev']

            # Comparison line from main scenario
            comparison_line = results[main_key]['ZTTC'][ri, comparison_idx, 0, :][year_mask]
            h_comparison, = ax.plot(
                years, comparison_line,
                color=comparison_color, linewidth=2, linestyle=comparison_linestyle,
                label=comparison_fuel.upper()
            )

            # Comparison sensitivity band
            if pos_key is not None and neg_key is not None:
                comparison_pos = results[pos_key]['ZTTC'][ri, comparison_idx, 0, :][year_mask]
                comparison_neg = results[neg_key]['ZTTC'][ri, comparison_idx, 0, :][year_mask]
                ax.fill_between(
                    years,
                    np.minimum(comparison_pos, comparison_neg),
                    np.maximum(comparison_pos, comparison_neg),
                    color=comparison_color, alpha=0.15
                )

            # BEV main estimate line
            bev_main_line = results[main_key]['ZTTC'][ri, bev_idx, 0, :][year_mask]
            h_bev, = ax.plot(
                years, bev_main_line,
                color=bev_color, linewidth=2, linestyle='-',
                label='BEV'
            )

            # BEV sensitivity band
            if pos_key is not None and neg_key is not None:
                bev_pos = results[pos_key]['ZTTC'][ri, bev_idx, 0, :][year_mask]
                bev_neg = results[neg_key]['ZTTC'][ri, bev_idx, 0, :][year_mask]
                h_bev_band = ax.fill_between(
                    years,
                    np.minimum(bev_pos, bev_neg),
                    np.maximum(bev_pos, bev_neg),
                    color=bev_color, alpha=0.2)

            comparison_label = 'Diesel' if comparison_fuel == 'diesel' else comparison_fuel.upper()
            bev_label = 'BEV'
            if bev_label not in legend_handles:
                legend_handles[bev_label] = h_bev
            if comparison_label not in legend_handles:
                legend_handles[comparison_label] = h_comparison

            ax.margins(x=0)
            ax.set_xlim(years[0], years[-1])
            ax.grid(True, alpha=0.3)

            if r == 0:
                ax.set_title(col_titles[vc], weight='bold')
            if c == 0:
                ax.set_ylabel(region_label, fontsize=14, labelpad=5, weight='bold')

    legend_labels = sorted(legend_handles)
    fig.legend(
        handles=[legend_handles[label] for label in legend_labels],
        labels=legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.06),
        ncol=len(legend_handles),
        frameon=True
    )
    fig.supylabel('Levelized Cost ($/tkm)', ha='left', va='center', fontsize=14, x=-0.02)

    plt.savefig(f'Figures/output/{output_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'Figures/output/svg/{output_name}.svg', bbox_inches='tight')


def plot_shares(
    regions, 
    scenario_name_map,
    scenario_type_map = {"S0": "main",
                         "S1": "positive",
                         "S2": "negative"}, 
    pickle_name='Results',
    FIGURE_WIDTH=7,
    ROW_HEIGHT=3.4,
    output_name='current_traj_shares'):
    """
    Plot line charts of shares for HDT and MDT technology groups.
    Function uses the ZEWS variable.

    Layout:
    - Two cols, MDT on left and HDT on right
    - Regions down rows

    Parameters
    ----------
    regions : dict
        Dictionary mapping region numbers to region names.
    scenario_name_map : dict
        Dictionary mapping scenario keys to scenario display names.
    scenario_type_map : dict
        Dictionary mapping scenario keys to whether scenario is main estimate or positive/negative sensitivity.
    pickle_name : str
        Name of the pickle file containing the results.
    FIGURE_WIDTH : float
        Width of the figure in inches.
    ROW_HEIGHT : float
        Height of each row in inches.
    output_name : str
        Name of the output file (without extension).
    """
    if not isinstance(regions, dict) or len(regions) == 0:
        raise ValueError("regions must be a non-empty dictionary of {region_id: region_label}.")
    if not isinstance(scenario_name_map, dict) or len(scenario_name_map) == 0:
        raise ValueError("scenario_name_map must be a non-empty dictionary of {scenario_key: scenario_label}.")

    with open(f'Output/{pickle_name}.pickle', 'rb') as f:
        results = pickle.load(f)

    # Identify main, positive, and negative scenario keys
    main_key = next((k for k, v in scenario_type_map.items() if v == "main"), None)
    pos_key  = next((k for k, v in scenario_type_map.items() if v == "positive"), None)
    neg_key  = next((k for k, v in scenario_type_map.items() if v == "negative"), None)

    if main_key is None:
        raise ValueError("scenario_type_map must contain a 'main' entry.")

    required_keys = [k for k in [main_key, pos_key, neg_key] if k is not None]
    missing = [k for k in required_keys if k not in results]
    if missing:
        raise KeyError(f"Scenario(s) not found in results: {missing}")

    zews_main = results[main_key]['ZEWS']
    zttc_main = results[main_key]['ZTTC']
    if zews_main.ndim != 4:
        raise ValueError(
            f"Expected ZEWS to be 4D (region, technology, share_component, year), got shape {zews_main.shape}"
        )
    if zttc_main.ndim != 4:
        raise ValueError(
            f"Expected ZTTC to be 4D (region, technology, cost_component, year), got shape {zttc_main.shape}"
        )

    region_ids = list(regions.keys())
    region_labels = [str(regions[r]) for r in region_ids]
    region_indices = np.asarray(region_ids, dtype=int) - 1
    n_regions_total = zews_main.shape[0]
    if np.any(region_indices < 0) or np.any(region_indices >= n_regions_total):
        raise ValueError(f"Region indices must be in 1..{n_regions_total}. Received: {region_ids}")

    tech_indices = {
        'MDT': {'diesel': 12, 'cng': 22, 'bev': 32},
        'HDT': {'diesel': 13, 'cng': 23, 'bev': 33},
    }
    n_techs = zews_main.shape[1]
    for vc, idx_map in tech_indices.items():
        for fuel, idx in idx_map.items():
            if idx < 0 or idx >= n_techs:
                raise ValueError(
                    f"{fuel.upper()} {vc} index {idx} is out of bounds for n_techs={n_techs}"
                )

    config = configparser.ConfigParser()
    config.read('settings.ini')
    simulation_start = int(config['settings']['simulation_start'])
    simulation_end = int(config['settings']['simulation_end'])
    years = np.arange(simulation_start, simulation_end + 1)
    n_years = zews_main.shape[3]
    if len(years) != n_years:
        years = np.arange(simulation_start, simulation_start + n_years)

    year_mask = (years >= 2025) & (years <= 2050)
    if not np.any(year_mask):
        raise ValueError("No years available in 2025-2050 range.")
    years = years[year_mask]

    def get_strict_crossover_year(bev_line, comparison_line, year_axis):
        diff = bev_line - comparison_line
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

    col_order = ['MDT', 'HDT']
    col_titles = {'MDT': 'Medium duty truck', 'HDT': 'Heavy duty truck'}
    bev_color = '#4cc9f0'
    cng_color = '#f4a261'
    diesel_color = 'black'

    n_rows = len(region_ids)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(FIGURE_WIDTH, ROW_HEIGHT * n_rows),
        sharex=True,
        sharey=True
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    axes = np.atleast_2d(axes)

    legend_handles = {}

    for r, (region_id, region_label) in enumerate(zip(region_ids, region_labels)):
        ri = int(region_id) - 1

        for c, vc in enumerate(col_order):
            ax = axes[r, c]
            comparison_fuel = 'diesel'
            if vc == 'MDT' and str(region_label).strip().lower() == 'delhi':
                comparison_fuel = 'cng'

            comparison_idx = tech_indices[vc][comparison_fuel]
            diesel_idx = tech_indices[vc]['diesel']
            cng_idx = tech_indices[vc]['cng']
            bev_idx = tech_indices[vc]['bev']

            diesel_main_line = results[main_key]['ZEWS'][ri, diesel_idx, 0, :][year_mask]
            h_diesel, = ax.plot(
                years, diesel_main_line,
                color=diesel_color, linewidth=2, linestyle='--',
                label='Diesel'
            )

            if pos_key is not None and neg_key is not None:
                diesel_pos = results[pos_key]['ZEWS'][ri, diesel_idx, 0, :][year_mask]
                diesel_neg = results[neg_key]['ZEWS'][ri, diesel_idx, 0, :][year_mask]
                ## Turning off shading for diesel and cng for now 
                # ax.fill_between(
                #     years,
                #     np.minimum(diesel_pos, diesel_neg),
                #     np.maximum(diesel_pos, diesel_neg),
                #     color=diesel_color, alpha=0.15
                # )

            cng_main_line = results[main_key]['ZEWS'][ri, cng_idx, 0, :][year_mask]
            h_cng, = ax.plot(
                years, cng_main_line,
                color=cng_color, linewidth=2, linestyle='-.',
                label='CNG'
            )

            if pos_key is not None and neg_key is not None:
                cng_pos = results[pos_key]['ZEWS'][ri, cng_idx, 0, :][year_mask]
                cng_neg = results[neg_key]['ZEWS'][ri, cng_idx, 0, :][year_mask]
                # ax.fill_between(
                #     years,
                #     np.minimum(cng_pos, cng_neg),
                #     np.maximum(cng_pos, cng_neg),
                #     color=cng_color, alpha=0.2
                # )

            bev_main_line = results[main_key]['ZEWS'][ri, bev_idx, 0, :][year_mask]
            h_bev, = ax.plot(
                years, bev_main_line,
                color=bev_color, linewidth=2, linestyle='-',
                label='BEV'
            )

            if pos_key is not None and neg_key is not None:
                bev_pos = results[pos_key]['ZEWS'][ri, bev_idx, 0, :][year_mask]
                bev_neg = results[neg_key]['ZEWS'][ri, bev_idx, 0, :][year_mask]
                ax.fill_between(
                    years,
                    np.minimum(bev_pos, bev_neg),
                    np.maximum(bev_pos, bev_neg),
                    color=bev_color, alpha=0.2
                )

            bev_tco_line = results[main_key]['ZTTC'][ri, bev_idx, 0, :][year_mask]
            comparison_tco_line = results[main_key]['ZTTC'][ri, comparison_idx, 0, :][year_mask]
            cross_year = get_strict_crossover_year(bev_tco_line, comparison_tco_line, years)
            if cross_year is not None:
                ax.axvline(
                    cross_year,
                    linestyle=':',
                    linewidth=4,
                    color=bev_color,
                    alpha=0.6
                )

            if 'BEV' not in legend_handles:
                legend_handles['BEV'] = h_bev
            if 'CNG' not in legend_handles:
                legend_handles['CNG'] = h_cng
            if 'Diesel' not in legend_handles:
                legend_handles['Diesel'] = h_diesel
            if 'TCO parity' not in legend_handles and cross_year is not None:
                legend_handles['TCO parity'] = Line2D(
                    [0], [0],
                    color=bev_color,
                    linewidth=4,
                    linestyle=':',
                    alpha=0.6
                )

            ax.grid(True, alpha=0.3)
            ax.margins(x=0)
            ax.set_xlim(years[0], years[-1])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

            if r == 0:
                ax.set_title(col_titles[vc], weight='bold')
            if c == 0:
                ax.set_ylabel(region_label, fontsize=14, labelpad=5, weight='bold')

    legend_labels = sorted(legend_handles)
    fig.legend(
        handles=[legend_handles[label] for label in legend_labels],
        labels=legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.07),
        ncol=len(legend_handles),
        frameon=True
    )
    fig.supylabel('Truck share', ha='left', va='center', fontsize=14, x=-0.02)

    plt.savefig(f'Figures/output/{output_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'Figures/output/svg/{output_name}.svg', bbox_inches='tight')