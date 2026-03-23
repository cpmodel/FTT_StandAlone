"""
Script contains fucntions to generate plots mapping current trajectories of shares
and costs with sensitivities.
"""
# Global imports
import pickle
import configparser
import numpy as np
import matplotlib.pyplot as plt

def plot_costs(
    regions, 
    scenario_name_map,
    scenario_type_map = {"S0": "main",
                         "S1": "positive",
                         "S2": "negative"}, 
    pickle_name='Results',
    FIGURE_WIDTH=7,
    ROW_HEIGHT=2.7):
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
        'MDT': {'diesel': 12, 'bev': 32},
        'HDT': {'diesel': 13, 'bev': 33},
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
            diesel_idx = tech_indices[vc]['diesel']
            bev_idx    = tech_indices[vc]['bev']

            # Diesel line from main scenario
            diesel_line = results[main_key]['ZTTC'][ri, diesel_idx, 0, :][year_mask]
            h_diesel, = ax.plot(
                years, diesel_line,
                color=diesel_color, linewidth=2, linestyle='--',
                label='Diesel'
            )

            # Diesel sensitivity band
            if pos_key is not None and neg_key is not None:
                diesel_pos = results[pos_key]['ZTTC'][ri, diesel_idx, 0, :][year_mask]
                diesel_neg = results[neg_key]['ZTTC'][ri, diesel_idx, 0, :][year_mask]
                h_diesel_band = ax.fill_between(
                    years,
                    np.minimum(diesel_pos, diesel_neg),
                    np.maximum(diesel_pos, diesel_neg),
                    color=diesel_color, alpha=0.15
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

            if 'Diesel' not in legend_handles:
                legend_handles['Diesel'] = h_diesel
            bev_label = 'BEV'
            if bev_label not in legend_handles:
                legend_handles[bev_label] = h_bev

            ax.margins(x=0)
            ax.set_xlim(years[0], years[-1])
            # ax.grid(True, alpha=0.3)

            if r == 0:
                ax.set_title(col_titles[vc], weight='bold')
            if c == 0:
                ax.set_ylabel(region_label, fontsize=14, labelpad=5, weight='bold')

    fig.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc='lower center',
        bbox_to_anchor=(0.5, 0.06),
        ncol=len(legend_handles),
        frameon=True
    )
    fig.supylabel('Levelized Cost ($/tkm)', ha='left', va='center', fontsize=14, x=-0.02)

    plt.savefig('Figures/output/current_traj_costs.png', dpi=300, bbox_inches='tight')
