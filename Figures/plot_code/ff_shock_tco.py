"""
Script for functions to calculate by how many years a diesel price spike can 
advance TCO parity, and the resulting market share impact in 2050. 
"""

import configparser
import csv
import math
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def plot_ff_shock_tco_impact(
    regions: dict[int, str],
    scenario_name_map: dict[str, str],
    pickle_name: str,
    output_name: str,
    FIGURE_WIDTH=7.6,
    ROW_HEIGHT=4
) -> None:
    """
    Plots the tco parity years (x axis) cities on the y axis. Main plot points
    indicate baseline tco parity year, and then arrows indicate advancement 
    under fossil fuel price shock scenario. 
    
    Parameters
    ----------
    regions: dict[int, str]
        Dictionary mapping region IDs to region names.
    scenario_name_map: dict[str, str]
        Dictionary mapping scenario keys to human-readable names for labeling.
    pickle_name: str
        Name of the pickle file containing the results.
    output_name: str
        Name of the output file (without extension).  
    """
    if not isinstance(regions, dict) or len(regions) == 0:
        raise ValueError("regions must be a non-empty dictionary of {region_id: region_label}.")
    if not isinstance(scenario_name_map, dict) or len(scenario_name_map) < 2:
        raise ValueError("scenario_name_map must contain at least baseline and shock scenarios.")

    scenario_keys = list(scenario_name_map.keys())
    baseline_key = next(
        (k for k, v in scenario_name_map.items() if 'baseline' in str(v).strip().lower()),
        scenario_keys[0]
    )
    shock_key = next(
        (
            k
            for k, v in scenario_name_map.items()
            if k != baseline_key and (
                'shock' in str(v).strip().lower()
                or 'spike' in str(v).strip().lower()
            )
        ),
        next((k for k in scenario_keys if k != baseline_key), None)
    )
    if shock_key is None:
        raise ValueError("Could not infer diesel shock scenario from scenario_name_map.")

    with open(f'Output/{pickle_name}.pickle', 'rb') as f:
        results = pickle.load(f)

    for scenario in (baseline_key, shock_key):
        if scenario not in results:
            raise KeyError(f"Scenario '{scenario}' not found in results.")

    zttc_baseline = results[baseline_key]['ZTTC']
    zttc_shock = results[shock_key]['ZTTC']

    config = configparser.ConfigParser()
    config.read('settings.ini')
    simulation_start = int(config['settings']['simulation_start'])
    simulation_end = int(config['settings']['simulation_end'])
    years = np.arange(simulation_start, simulation_end + 1)
    n_years = zttc_baseline.shape[3]
    if len(years) != n_years:
        years = np.arange(simulation_start, simulation_start + n_years)

    region_ids = list(regions.keys())
    region_labels = [str(regions[r]) for r in region_ids]
    region_indices = np.asarray(region_ids, dtype=int) - 1
    n_regions_total = zttc_baseline.shape[0]
    if np.any(region_indices < 0) or np.any(region_indices >= n_regions_total):
        raise ValueError(
            f"Region indices must be in 1..{n_regions_total}. Received: {region_ids}"
        )

    tech_indices = {
        'MDT': {'diesel': 12, 'bev': 32},
        'HDT': {'diesel': 13, 'bev': 33},
    }

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

    crossover_years = {
        'MDT': {'baseline': [], 'shock': []},
        'HDT': {'baseline': [], 'shock': []},
    }
    for vehicle_class, idx_map in tech_indices.items():
        diesel_idx = idx_map['diesel']
        bev_idx = idx_map['bev']
        for ri in region_indices:
            bev_base = zttc_baseline[ri, bev_idx, 0, :]
            diesel_base = zttc_baseline[ri, diesel_idx, 0, :]
            bev_shock = zttc_shock[ri, bev_idx, 0, :]
            diesel_shock = zttc_shock[ri, diesel_idx, 0, :]

            cross_base = get_strict_crossover_year(bev_base, diesel_base, years)
            cross_shock = get_strict_crossover_year(bev_shock, diesel_shock, years)
            if cross_base is not None:
                cross_base = math.floor(cross_base)
            if cross_shock is not None:
                cross_shock = math.floor(cross_shock)

            crossover_years[vehicle_class]['baseline'].append(cross_base)
            crossover_years[vehicle_class]['shock'].append(cross_shock)

    def _draw_gradient_arrow(ax, x0, x1, y, c_start, c_end,
                               n_seg=60, linewidth=1.6, head_length=0.5, head_width=0.16,
                               tip_gap=0.6, zorder=2):
        """Draw a horizontal arrow from (x0,y) to (x1,y) with a color gradient.
        tip_gap: data-unit gap left between the arrowhead tip and x1."""
        direction = 1 if x1 >= x0 else -1
        x1_adj = x1 - direction * tip_gap
        x_shaft_end = x1_adj - direction * head_length
        xs = np.linspace(x0, x_shaft_end, n_seg + 1)
        points = np.stack([xs, np.full_like(xs, y)], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        rgba_start = np.array(mcolors.to_rgba(c_start))
        rgba_end   = np.array(mcolors.to_rgba(c_end))
        seg_colors = [
            rgba_start + (rgba_end - rgba_start) * t
            for t in np.linspace(0, 1, n_seg)
        ]
        lc = LineCollection(segments, colors=seg_colors, linewidth=linewidth, zorder=zorder)
        ax.add_collection(lc)
        # Filled arrowhead triangle at endpoint
        ax.fill(
            [x_shaft_end, x1_adj, x_shaft_end],
            [y + head_width / 2, y, y - head_width / 2],
            color=rgba_end,
            zorder=zorder + 1,
        )

    y_positions = np.arange(len(region_labels))
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, ROW_HEIGHT), sharey=True)
    axes = np.atleast_1d(axes)
    class_order = ['MDT', 'HDT']
    class_titles = {'MDT': 'Medium duty truck', 'HDT': 'Heavy duty truck'}
    baseline_color = "#264653"
    shock_color = '#4cc9f0'

    year_min = min(years)
    year_max = max(years)
    x_left = max(2025, year_min)
    x_right = min(2050, year_max)
    # Use a start point at the right edge to indicate an off-axis baseline without drawing outside axes.
    x_off_axis = x_right + 0.45

    for ax, vehicle_class in zip(axes, class_order):
        base_vals = crossover_years[vehicle_class]['baseline']
        shock_vals = crossover_years[vehicle_class]['shock']

        for y, base_year, shock_year in zip(y_positions, base_vals, shock_vals):
            if base_year is not None:
                ax.scatter(base_year, y, color=baseline_color, s=55, zorder=3)

            if base_year is not None and shock_year is not None:
                _draw_gradient_arrow(ax, base_year, shock_year, y, baseline_color, shock_color)
                ax.scatter(shock_year, y, color=shock_color, s=40, zorder=4)
            elif base_year is None and shock_year is not None:
                # Baseline parity is beyond axis horizon; arrow enters from the right edge.
                _draw_gradient_arrow(ax, x_off_axis, shock_year, y, baseline_color, shock_color)
                ax.scatter(shock_year, y, color=shock_color, s=40, zorder=4)
            elif shock_year is not None:
                ax.scatter(shock_year, y, color=shock_color, s=40, marker='D', zorder=4)
            else:
                ax.text(
                    x_left + 12.5,
                    y,
                    'No TCO parity point',
                    va='center',
                    ha='center',
                    fontsize=8,
                    color='gray',
                )

        ax.set_title(class_titles[vehicle_class], weight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlim(x_left - 0.5, x_right + 0.5)
        ax.set_xticks(np.arange(x_left, x_right + 1, 5))

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(region_labels)
    axes[0].invert_yaxis()

    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=baseline_color, markersize=7,
               label=f"Average prices"),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=shock_color, markersize=7,
               label=f"Fossil fuel price shock"),
    ]
    fig.subplots_adjust(bottom=0.22)
    fig.supxlabel('Year of BEV TCO parity', y=0.08)
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.035))

    plt.savefig(f'Figures/output/{output_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'Figures/output/svg/{output_name}.svg', bbox_inches='tight')
    