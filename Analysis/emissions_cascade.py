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
output_baseline = get_output(output_file, "S0")
output_all_policies = get_output(output_file, "sxp - All policies")

titles, fig_dir, tech_titles, models, shares_vars = get_metadata()
models = ["FTT:H", "FTT:Tr", "FTT:Fr", "FTT:P"]


# Define the shares, prices of interest
emissions_names = {"FTT:P": "MEWE", "FTT:Tr": "TEWE", "FTT:H": "HEWE", "FTT:Fr": "ZEWE"}

models_to_scenarios = {"FTT:H": ["FTT-P", "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-H", "All minus FTT-P"],
                       "FTT:Tr": ["FTT-P",  "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-Tr", "All minus FTT-P"],
                       "FTT:Fr": ["FTT-P",  "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-Fr", "All minus FTT-P"],
                       "FTT:P": ["FTT-P", "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-P"] }



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
    emissions[model]["Baseline"] = get_total_emissions(output_baseline, model)
    emissions[model]["sxp - All policies"] = get_total_emissions(output_all_policies, model)
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

#%% ===========================================================================
# Donut chart figure cumulative emissions figures
# =============================================================================

# Data

emissions_from_2025 = {
    scenario: {
        policy: values[15:] for policy, values in policies.items()
    } for scenario, policies in emissions.items()
}

emissions_cum_2050 = np.sum([emissions_from_2025[model]["Baseline"] for model in models])

def cumulative_saved_emissions(model, policy):
    '''Get the difference of emissions between policy and baseline for only that sector'''
    
    cum_emissions_baseline = np.sum(emissions_from_2025[model][policy])
    baseline_sector_emissions = np.sum(emissions_from_2025[model]["Baseline"])
    return baseline_sector_emissions - cum_emissions_baseline

def combined_policies_saved_emissions():
    total_emissions_all_policies = np.sum([emissions_from_2025[model]["sxp - All policies"] for model in models])
    return  emissions_cum_2050 - total_emissions_all_policies

sectoral_saved_emissions = [cumulative_saved_emissions("FTT:P", "FTT-P"),
                            cumulative_saved_emissions("FTT:H", "FTT-H"),
                            cumulative_saved_emissions("FTT:Tr", "FTT-Tr"),
                            cumulative_saved_emissions("FTT:Fr", "FTT-Fr")]

inner_data = {"No policy": emissions_cum_2050}
data = {
    "Power policy": cumulative_saved_emissions("FTT:P", "FTT-P"),
    "Heat policy": cumulative_saved_emissions("FTT:H", "FTT-H"),
    "Transport policy": cumulative_saved_emissions("FTT:Fr", "FTT-Fr"),
    "Freight policy": cumulative_saved_emissions("FTT:Tr", "FTT-Tr"),
    "Combined policies": max(combined_policies_saved_emissions() - np.sum(sectoral_saved_emissions), 0)
}


# Calculate the remaining emissions after policies
remaining = inner_data["No policy"] - sum(data.values())

# Add the remaining emissions to the outer data
data["Remaining emissions"] = remaining

labels = list(data.keys())
sizes = list(data.values())

# Create a pie chart with higher DPI
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)


# Create custom autopct function
def custom_autopct(pct):
    if pct > 35:
        return f'{pct:.1f}%'
    else:
        return f'–{pct:.1f}%'

# Donut
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct=lambda pct: custom_autopct(pct),
    startangle=90, pctdistance=0.65 ,
    radius=1.2, wedgeprops=dict(width=0.3, edgecolor='w'))


# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')

# Set font properties
plt.setp(texts, size=12)
plt.setp(autotexts, size=12)

# Add title
plt.title("Global Cumulative Emissions 2025–2050", fontsize=16)

# Display the chart
plt.show()

