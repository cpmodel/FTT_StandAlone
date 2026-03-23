import pickle
import configparser
import numpy as np
import matplotlib.pyplot as plt

def plot_zewk_hdt_stacked(
    regions, 
    scenarios, 
    pickle_name='Results',
    FIGURE_WIDTH=8,
    ROW_HEIGHT=2.4,
    output_name='zewk_hdt_stacked'):
    """
    Plot stacked area charts of ZEWK (number of trucks) for HDT and MDT technology groups.

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
        'Diesel': [12, 13, 17, 18],
        'CNG/LPG': [22, 23],
        'BEV': [32, 33],
        'Other': [2, 3, 7, 8, 27, 28, 37, 38, 42, 43],
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
        raise ValueError("No years available in 2025-2050 range.")
    years = years[year_mask]

    # Require 2050 explicitly for the annotation
    if 2050 not in years:
        raise ValueError("Year 2050 is not available in the selected time range/data.")
    year_2050_idx = int(np.where(years == 2050)[0][0])

    n_rows = len(region_ids)
    n_cols = len(scenario_keys)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIGURE_WIDTH + 1 * n_cols, ROW_HEIGHT * n_rows),
        sharex=True,
        sharey=False
    )

    axes = np.atleast_2d(axes)

    # Baseline scenario is first scenario/first column
    baseline_key = scenario_keys[0]
    bev_idxs = group_indices['BEV']

    stack_handles = None
    for r, (region_id, region_label) in enumerate(zip(region_ids, region_labels)):
        region_idx = int(region_id) - 1

        # Baseline BEV value in 2050 for this region (thousands of trucks)
        baseline_bev_2050 = (
            np.sum(results[baseline_key]['ZEWK'][region_idx, bev_idxs, 0, :], axis=0)[year_mask] / 1000.0
        )[year_2050_idx]

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

            # BEV increase annotation in 2050 vs baseline
            scenario_bev_2050 = grouped[group_order.index('BEV')][year_2050_idx]
            if baseline_bev_2050 == 0:
                if scenario_bev_2050 == 0:
                    pct_text = ""
                else:
                    pct_text = "n/a"
            else:
                pct_change = (scenario_bev_2050 - baseline_bev_2050) / baseline_bev_2050 * 100.0
                pct_text = f"{pct_change:+.1f}%"
                
            if not c == 0:  # Don't annotate baseline column
                
                ax.text(
                    0.96, 0.8, pct_text,
                    transform=ax.transAxes,
                    ha='right', va='top',
                    fontsize=11, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.75)
                )

            if r == 0:
                ax.set_title(scenario_labels[scenario_key], weight='bold')
            if c == 0:
                ax.set_ylabel(region_label, fontsize = 14, labelpad=5, weight='bold')
            else:
                ax.set_yticklabels([])

    fig.subplots_adjust(wspace=.1, hspace=.1)

    if stack_handles is not None:
        fig.legend(
            handles=stack_handles,
            labels=group_order,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.05),
            ncol=len(group_order),
            frameon=True
        )

        fig.supylabel('Number of trucks (thousands)', ha='left', va='center', fontsize=14, x=0.04)
        plt.savefig(f'Figures/output/{output_name}.png', dpi=300, bbox_inches="tight")