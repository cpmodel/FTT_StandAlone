# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:57:05 2025

@author: Femke Nijsse
"""

# Import the results pickle file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from preprocessing import get_output, get_metadata, save_fig, save_data

variable_to_validate = "ZEWS"


output_2022 = "Results_starting_2022.pickle"
output_2022_new = "Results_starting_2022_new_turnover2.pickle"
output_2023 = "Results_starting_2023.pickle"
output_2023_new = "Results_starting_2023_new_turnover2.pickle"
output_2024 = "Results_starting_2024.pickle"

def read_in_variable(input_file, variable_to_validate):
    '''Reading in the new variable, and selecting right subset'''
    output_S0 = get_output(input_file, "S0")
    zews_results = output_S0[variable_to_validate][:, :, 0, :]
    
    # Germany, US, China, India
    regions_with_data = np.array([2, 33, 40, 41])
    BEV_indices = np.array(range(31, 34))
    index_2023 = 13
    
    zews_results_subset = zews_results[:, :, index_2023]
    zews_results_subset = zews_results_subset[np.ix_(regions_with_data, BEV_indices)]
    
    return zews_results_subset

zews_results_2022_subset = read_in_variable(output_2022, variable_to_validate)
zews_results_2022_subset_new = read_in_variable(output_2022_new, variable_to_validate)
zews_results_2023_subset = read_in_variable(output_2023, variable_to_validate)
zews_results_2023_subset_new = read_in_variable(output_2023_new, variable_to_validate)
zews_results_2024_subset = read_in_variable(output_2024, variable_to_validate)

unequal = (zews_results_2023_subset == zews_results_2024_subset)



labels = ["LDV", "MDV", "HDV"]

colors = ['r', 'b', 'g']

markers1 = ['o', '.']
markers2 = ['v', 'x']
descriptions1 = [
    "2023 start, TR≈0.125", 
    "2023 start, TR≈0.2"
]
descriptions2 = ["2022 start, TR≈0.125",
                 "2022 start, TR≈0.2"]

# Create proxy artists for the second legend
legend_markers1 = [mlines.Line2D([], [], marker=m, linestyle='None', markersize=8, label=desc,
                                 alpha=0.5 if m == 'o' else 1.0)
                  for m, desc in zip(markers1, descriptions1)]
# Create proxy artists for the second legend
legend_markers2 = [mlines.Line2D([], [], marker=m, linestyle='None', markersize=8, label=desc,
                                 alpha=0.5 if m == 'v' else 1)
                  for m, desc in zip(markers2, descriptions2)]

fig, axes = plt.subplots(nrows=2, figsize=(6, 9))  

for tech in range(3):
    axes[0].scatter(zews_results_2024_subset[:, tech], zews_results_2023_subset_new[:, tech],
               marker='.', color=colors[tech])
    
    axes[0].scatter(zews_results_2024_subset[:, tech], zews_results_2023_subset[:, tech],
               label = labels[tech], color=colors[tech], alpha=0.35)
    
    # Second legend
    second_legend = axes[0].legend(handles=legend_markers1, loc='lower right')
    
    # Add it to the plot, so it's not overwritten
    axes[0].add_artist(second_legend)
    

for tech in range(3):
    axes[1].scatter(zews_results_2024_subset[:, tech], zews_results_2022_subset_new[:, tech],
               marker='x', color=colors[tech])
    
    axes[1].scatter(zews_results_2024_subset[:, tech], zews_results_2022_subset[:, tech],
               marker='v', color=colors[tech], alpha=0.35)
    
    # Second legend
    second_legend = axes[1].legend(handles=legend_markers2, loc='lower right')
    
    # Add it to the plot, so it's not overwritten
    axes[1].add_artist(second_legend)
    

for ax in axes:  
    ax.set_ylabel("Results when simulating 2023")
    ax.set_xlabel("True 2023 shares BEVs")
    
    
    # Get current axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Plot 1:1 line within the existing limits
    ax.plot([x_min, x_max], [x_min, x_max], 'k--', label="1:1 Line")  # Dashed black line
    
    # Restore original limits (to prevent auto-scaling from affecting them)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add legend
    first_legend = ax.legend(loc='upper left')
    
    # Add the second legend to the plot
    ax.add_artist(first_legend)
    