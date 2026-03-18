# Global imports
import pickle
import configparser
import numpy as np
import matplotlib.pyplot as plt    
import matplotlib.ticker as mtick

def plot_ztvt_timeseries(
    regions, 
    scenario_name, 
    pickle_name='Results', 
    tech_indices=(32, 33),
    FIGURE_WIDTH=9,
    ROW_HEIGHT=3.3
    ):
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
        raise ValueError("No years available in 2025-2050 range.")
    years = years[year_mask]

    mdt_series = ztvt[region_indices, mdt_idx, 0, :][:, year_mask]
    hdt_series = ztvt[region_indices, hdt_idx, 0, :][:, year_mask]

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, ROW_HEIGHT), sharex=True, sharey=True)
    cmap = plt.get_cmap('tab10')
    fig.subplots_adjust(wspace=.1, hspace=.1)
    legend_handles = []
    legend_labels = []

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

        axes[0].set_title('Medium duty truck', weight='bold')
        axes[1].set_title('Heavy duty truck', weight='bold')
        axes[0].set_ylabel('Subsidy to achieve TCO parity')

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=max(1, len(region_labels)),
        frameon=True
    )
    plt.savefig('Figures/output/tco_subsidy.png', dpi=300, bbox_inches="tight")