# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:35:01 2024

@author: Femke Nijsse
"""

# Import the results pickle file
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from preprocessing import get_output, get_metadata


# Set global font size
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})

output_file = "Results_sectors.pickle"

output_S0_FTTP = get_output(output_file, "S0")
output_S0_FTTH  = get_output(output_file, "S0")
output_S0_FTTTr = get_output(output_file, "S0")
output_S0_FTTFr = get_output(output_file, "S0")

titles, fig_dir, tech_titles, models = get_metadata()
models = ["FTT:H", "FTT:Tr", "FTT:Fr", "FTT:P"]


# Define the shares, prices of interest
shares_names = {"FTT:P": "MEWS", "FTT:Tr": "TEWS", "FTT:H": "HEWS", "FTT:Fr": "ZEWS"}
emissions_names = {"FTT:P": "MEWE", "FTT:Tr": "TEWE", "FTT:H": "HEWE", "FTT:Fr": "ZEWE"}

models_to_scenarios = {"FTT:H": ["FTT-P", "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-H", "All minus FTT-P"],
                       "FTT:Tr": ["FTT-P",  "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-Tr", "All minus FTT-P"],
                       "FTT:Fr": ["FTT-P",  "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-Fr", "All minus FTT-P"],
                       "FTT:P": ["FTT-P", "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-P"] }

outputs_baseline = {"FTT:P": output_S0_FTTP, "FTT:H": output_S0_FTTH, 
                    "FTT:Tr": output_S0_FTTTr, "FTT:Fr": output_S0_FTTFr}



def get_total_emissions(output, model):
    """Sum over regions and technologies"""
    if model in ["FTT:P", "FTT:H"]:
        emission_m = np.sum(output[emissions_names[model]], axis=(0, 1, 2))
    elif model == "FTT:Tr":
        emission_by_tech = output[emissions_names[model]]
        all_vehicles = list(range(emission_by_tech.shape[1]))
        non_EVs = [x for x in all_vehicles if x not in [18, 19, 20]]
        emission_m = np.sum(emission_by_tech[:, non_EVs], axis=(0, 1, 2))
    elif model == "FTT:Fr":
        emission_by_tech = output[emissions_names[model]]
        all_vehicles = list(range(emission_by_tech.shape[1]))
        non_EVs = [x for x in all_vehicles if x not in [12, 13]]
        emission_m = np.sum(emission_by_tech[:, non_EVs], axis=(0, 1, 2))
    return emission_m


# Compute overall CO2 reductions per sector
emissions = {}
emissions_abs_diff = {}

def flatten(xss):
    return [x for xs in xss for x in xs]

all_policy_scens = list(set(flatten(models_to_scenarios.values())))
for model in models:
    emissions[model] = {}
    emissions_abs_diff[model] = {}
    output_baseline = outputs_baseline[model]
    emissions[model]["Baseline"] = get_total_emissions(output_baseline, model)
    for scenario in models_to_scenarios[model]:
        output = get_output(output_file, scenario)
        emissions[model][scenario] = get_total_emissions(output, model)
        emissions_abs_diff[model][scenario] = emissions[model][scenario] - emissions[model]["Baseline"]  
    

#%% Plot the figure
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, figure=fig)

title_labels = {"FTT:P": "Power sector emissions", "FTT:H": "Heating emissions",
                "FTT:Tr": "Emissions from cars", "FTT:Fr": "Emissions from freight"}
scenario_labels = {"FTT-P": "Power policies", "FTT-H": "Heating policies", "FTT-Tr": "Transport policies",
                   "FTT-Fr": "Freight policies", "All minus FTT-P": "Other sectors combined", 
                   "All minus FTT-H": "Other sectors combined", "All minus FTT-Tr": "Other sectors combined", 
                   "All minus FTT-Fr": "Other sectors combined"}
# Define a harmonious color palette
palette = sns.color_palette("Blues_r", 3)
palette_green =  sns.color_palette("Greens_r", 3)

year_ind = 2050 - 2010


def combined_or_not(scenario_name):
    if "combined" in scenario_name:
        combined = 0
    else:
        combined = 1
    return combined
    

# Create a list of axes using flattening
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[1, :])
]

def compare_strings(str1, str2):
    """Check if strings are the same except for :/-"""
    
    # Replace colons with hyphens in both strings
    str1_normalized = str1.replace(":", "-")
    str2_normalized = str2.replace(":", "-")
    
    # Compare the normalized strings
    return str1_normalized == str2_normalized

for mi, model in enumerate(models):
    ax = axs[mi]
    
    for si, scenario in enumerate(models_to_scenarios[model][::-1]):
        emission_diff = emissions_abs_diff[model][scenario][year_ind]
        
        # Put in a dash for within-sector emissions reductions
        if compare_strings(model, scenario):
            
            ax.barh(scenario_labels[scenario], 0)
            y_position = si - 1  # Get the y position for the bar
            ax.plot([-0.8, 0.8], [y_position, y_position], color='black', linewidth=1)  # Small dash
            
            emission_diff
        
        # Cross-sectoral emission reductions + emissions red. from power sector.
        else:
            
            # Normal cross-sectoral emissions
            if model == "FTT:P" or scenario != "All minus FTT-P":
                
                print(model, scenario)
                ax.barh(scenario_labels[scenario], emission_diff, color=palette[combined_or_not(scenario_labels[scenario])])
                ax.set_title(title_labels[model], pad=15)
                
            
            if model == "FTT:P" and scenario != "All minus FTT-P":
                scen_to_model = scenario.replace("-", ":")
                emission_red_own_sector = emissions_abs_diff[scen_to_model][scenario][year_ind]
                axs[3].barh(scenario_labels[scenario], emission_red_own_sector, color=palette_green[1])
                
                
            elif model == "FTT:P" and scenario == "All minus FTT-P":
                models_to_add = ["FTT:Fr", "FTT:Tr", "FTT:H"]                
                total_emissions = np.sum([
                            emissions_abs_diff[model_to_add][scenario][year_ind]
                            for model_to_add in models_to_add] )
                axs[3].barh(scenario_labels[scenario], total_emissions, color=palette_green[(si+1)//4])
            
    
    
    
    ax.axvline(0, color='black', linewidth=0.8)  # Add vertical line at zero

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.grid(which='both', axis='x', color='grey', linestyle='-', linewidth=0.5)
    
    if mi == 0 or mi == 3:
        ax.set_yticklabels(ax.get_yticklabels())  # Keep y-tick labels for axs[0] and axs[3]
    else:
        ax.set_yticklabels([])  # Remove y-tick labels for other plots
    
    if mi > 1:
        ax.set_xlabel(r"MtCO$_2$ emission")
    ax.tick_params(axis='both', which='both', length=0)  # Remove ticks in both axes

    # Align policies labels left
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('left')
        
    # Adjust the position of the y-tick labels to avoid overlap with the bars
    ax.tick_params(axis='y', pad=150)
    if mi < 3:
        ax.set_xlim(-18, 18)

fig.subplots_adjust(wspace=0.6, hspace=0.4)  # Increase horizontal space between subplots
    
# Save the graph as an editable svg file
output_file = os.path.join(fig_dir, "Emission_reduction_by_sector.svg")
output_file2 = os.path.join(fig_dir, "Emission_reduction_by_sector.png")
fig.savefig(output_file, format="svg", bbox_inches='tight')
fig.savefig(output_file2, format="png", bbox_inches='tight')


