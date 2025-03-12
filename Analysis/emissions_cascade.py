# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:35:01 2024

@author: Femke Nijsse
"""
#%%
# Import the results pickle file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd

from preprocessing import get_output, get_metadata, save_fig, save_data
import config


titles, fig_dir, tech_titles, models, shares_vars = get_metadata()

models = ["FTT:P", "FTT:H", "FTT:Tr", "FTT:Fr"]
repl_dict = config.REPL_DICT
model_names = [repl_dict[model] for model in models]

# Define the shares, prices of interest
emissions_names = {"FTT:P": "MEWE", "FTT:Tr": "TEWE", "FTT:H": "HEWE", "FTT:Fr": "ZEWE"}

all_policies_or_mandates = "All policies"
if all_policies_or_mandates == "All policies":
    output_file = "Results_sectors.pickle"
    output_baseline = get_output(output_file, "S0")
    #output_all_policies = get_output(output_file, "sxp half - All policies")
    output_all_policies = get_output(output_file, "sxp - All policies")




    models_to_scenarios = {"FTT:H": ["FTT-P", "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-H", "All minus FTT-P"],
                           "FTT:Tr": ["FTT-P",  "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-Tr", "All minus FTT-P"],
                           "FTT:Fr": ["FTT-P",  "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-Fr", "All minus FTT-P"],
                           "FTT:P": ["FTT-P", "FTT-H", "FTT-Tr", "FTT-Fr", "All minus FTT-P"] }
    
    single_sector_list = ["half FTT-P", "half FTT-H", "half FTT-Tr", "half FTT-Fr"]
    single_sector_list = ["FTT-P", "FTT-H", "FTT-Tr", "FTT-Fr"]
    models_to_scenarios = {"FTT:H": single_sector_list,
                           "FTT:Tr": single_sector_list,
                           "FTT:Fr": single_sector_list,
                           "FTT:P": single_sector_list }

if all_policies_or_mandates == "Mandates":
    output_file = "Results_sxp.pickle"
    output_baseline = get_output(output_file, "S0")
    #output_all_policies = get_output(output_file, "Mandates half")
    output_all_policies = get_output(output_file, "Mandates")

    single_mand_list = ["sxp half - P mand", "sxp half - H mand", "sxp half - Tr mand", "sxp half - Fr mand"]
    single_mand_list = ["sxp - P mand", "sxp - H mand", "sxp - Tr mand", "sxp - Fr mand"]

    models_to_scenarios =  {"FTT:H": single_mand_list,
                           "FTT:Tr": single_mand_list,
                           "FTT:Fr": single_mand_list,
                           "FTT:P": single_mand_list}



def get_total_emissions(output, model):
    """Sum over regions and technologies"""
    if model in ["FTT:P", "FTT:H", "FTT:Fr"]:
        emission_m = np.sum(output[emissions_names[model]], axis=(0, 1, 2))
    elif model == "FTT:Tr":
        emission_by_tech = output[emissions_names[model]]
        all_vehicles = list(range(emission_by_tech.shape[1]))
        non_EVs = [x for x in all_vehicles if x not in [18, 19, 20]]
        emission_m = np.sum(emission_by_tech[:, non_EVs], axis=(0, 1, 2))
    
    # Rescale so that 2022 emissions line up with actual emissions
    emission_m = scale_total_emissions(emission_m, model)
    
    return emission_m

def scale_total_emissions(emissions_m, model):
    """ Total emissions are scaled to 2022 IEA numbers.
    https://www.iea.org/data-and-statistics/charts/global-co2-emissions-by-sector-2019-2022
    - Power sector is 14650 Mt CO2
    - "Space and water heating: https://www.iea.org/energy-system/buildings/heating (2400 Mt CO2)"
    - "Road transport emissions: 5870 Mt CO2 total (https://www.iea.org/energy-system/transport ), 
        of which 39% freight and remaining transport (https://www.nature.com/articles/s41598-024-52682-4)"""
    
    emissions_2022 = {"FTT:P" : 14650,
                      "FTT:H" : 2400,
                      "FTT:Tr": 3522,
                      "FTT:Fr": 2289 }
    
    ind_2022 = 12
    
    scaling_factor = emissions_2022[model] / emissions_m[ind_2022]
    emissions_m_rescaled = emissions_m * scaling_factor
    #emissions_m_rescaled = emissions_m
   
    
    return emissions_m_rescaled
    


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
plot_extra = False
if plot_extra:
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
        
    save_fig(fig, fig_dir, "Emission_reduction_by_sector")


#%% ===========================================================================
# Data wrangling
# =============================================================================

# Emissions timeseries from 2025
emissions_from_2025 = {
    scenario: {
        policy: np.sum(values[15:]) for policy, values in policies.items()
    } for scenario, policies in emissions.items()
}

# Emissions in 2050
emissions_2050 = {
    scenario: {
        policy: values[-1] for policy, values in policies.items()
        } for scenario, policies in emissions.items()
    }

emissions_from_2025_S0 = [emissions_from_2025[model]["Baseline"] for model in models]
emissions_from_2025_combined_policies = [emissions_from_2025[model]["sxp - All policies"] for model in models]
emissions_cum_2050_S0 = np.sum(emissions_from_2025_S0)  # Sum over sectors
emissions_2050_S0 = [emissions_2050[model]["Baseline"] for model in models]
emissions_tot_2050_S0 = np.sum(emissions_2050_S0)
emissions_2050_combined_policies = [emissions_2050[model]["sxp - All policies"] for model in models]


print(f"Total cumulative emissions S0 2025-2050 is {emissions_cum_2050_S0/1000:.1f} GtCO₂")
print(f"Total emissions 2050 S0 is {emissions_tot_2050_S0/1000:.1f} GtCO₂")


def get_saved_emissions(model, policy, emissions_cum_or_2050):
    '''First way to calculate combination gain. 
    1. Compute baseline emissions
    2. Compute the economy-wide emissions per sectoral policy
    3. Compute the economy-wide emissions in the baseline
    '''
    
    baseline =  emissions_cum_or_2050[model]["Baseline"]  
    cum_emissions_policy = np.sum([emissions_cum_or_2050[model][policy] for model in models])
    baseline_sector_emissions = np.sum([emissions_cum_or_2050[model]["Baseline"] for model in models])
    
    return baseline - (baseline_sector_emissions - cum_emissions_policy)

def get_sectoral_emissions_policy(model, policy, emissions_cum_or_2050):
    '''Second way to calculation combination gain.
    1. Simply compute the emissions in the relevant sector based on that sector's policy 
    '''
    emissions_sector_only = emissions_cum_or_2050[model][policy]
    return emissions_sector_only

def combined_policies_saved_emissions():
    total_emissions_all_policies = np.sum([emissions_from_2025[model]["sxp - All policies"] for model in models])
    return  emissions_cum_2050_S0 - total_emissions_all_policies

def combined_policies_saved_emissions_2050():
    total_emissions_all_policies = np.sum([emissions_2050[model]["sxp - All policies"] for model in models])
    return  emissions_tot_2050_S0 - total_emissions_all_policies

def set_up_list_saved_emissions(emissions, function):
    # if all_policies_or_mandates == "All policies":
    #     sectoral_saved_emissions = [function("FTT:P", "half FTT-P", emissions),
    #                                 function("FTT:H", "half FTT-H", emissions),
    #                                 function("FTT:Tr", "half FTT-Tr", emissions),
    #                                 function("FTT:Fr", "half FTT-Fr", emissions)]
    # elif all_policies_or_mandates == "Mandates":
    #     sectoral_saved_emissions = [function("FTT:P", "sxp half - P mand", emissions),
    #                                 function("FTT:H", "sxp half - H mand", emissions),
    #                                 function("FTT:Tr", "sxp half - Tr mand", emissions),
    #                                 function("FTT:Fr", "sxp half - Fr mand", emissions)]
        
    if all_policies_or_mandates == "All policies":
        sectoral_saved_emissions = [function("FTT:P", "FTT-P", emissions),
                                    function("FTT:H", "FTT-H", emissions),
                                    function("FTT:Tr", "FTT-Tr", emissions),
                                    function("FTT:Fr", "FTT-Fr", emissions)]
    elif all_policies_or_mandates == "Mandates":
        sectoral_saved_emissions = [function("FTT:P", "sxp - P mand", emissions),
                                    function("FTT:H", "sxp - H mand", emissions),
                                    function("FTT:Tr", "sxp - Tr mand", emissions),
                                    function("FTT:Fr", "sxp - Fr mand", emissions)]        
        

    return sectoral_saved_emissions


def set_up_data_dict(sectoral_saved_emissions, combined_policies_function):
    data = {
        "Power policies": sectoral_saved_emissions[0],
        "Heat policies": sectoral_saved_emissions[1],
        "Transport policies": sectoral_saved_emissions[2],
        "Freight policies": sectoral_saved_emissions[3],
        "Combined policies": max(combined_policies_function() - np.sum(sectoral_saved_emissions), 0)
    }
    return data

sectoral_saved_emissions = set_up_list_saved_emissions(emissions_from_2025, get_saved_emissions)
sectoral_saved_emissions_2050 = set_up_list_saved_emissions(emissions_2050, get_saved_emissions)
data_cum = set_up_data_dict(sectoral_saved_emissions, combined_policies_saved_emissions)
data_2050 = set_up_data_dict(sectoral_saved_emissions_2050, combined_policies_saved_emissions_2050)

# Calculate the remaining emissions after policies
remaining_cum = emissions_cum_2050_S0 - sum(data_cum.values())
remaining_2050 = emissions_tot_2050_S0 - sum(data_2050.values())

# Add the remaining emissions to the data
data_cum["Remaining emissions"] = remaining_cum
data_2050["Remaining emissions"] = remaining_2050





#%% ===========================================================================
# Bar chart figure cumulative emissions figures
# =============================================================================

df_cumulative = pd.DataFrame({
        "Combined policies": emissions_from_2025_combined_policies,
         "One sector at a time": np.array(sectoral_saved_emissions),
         "Baseline": emissions_from_2025_S0
         }).transpose()

df_2050 = pd.DataFrame({
        "Combined policies": emissions_2050_combined_policies,
         "One sector at a time": np.array(sectoral_saved_emissions_2050),
         "Baseline": emissions_2050_S0
         }).transpose()


# Create subplots (2 rows, 1 column)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.2, 2.6), dpi=300)  # Adjust figsize to fit both plots

colors = ['#f47e7a', '#b71f5c', '#621237', '#dbbaa7']

def plot_stacked_bar_chart(data, axis):
    
    data.columns = model_names
    data = data/1000  # Convert to GtCO2
    data.iloc[:, 0:4].plot.barh(align='center', stacked=True, ax=axis,
                                              color=colors, width=0.75)
    plt.tight_layout()
    
    # Create a bolded title for each subplot (axes[0] --> 'Cumulative 2025–2050 emissions', axes[1] --> 'Emissions 2050')
    if axis == axes[0]:
        axis.set_title(r"Cumulative 2025–2050 emissions (GtCO$_{\mathbf{2}}$)", fontweight='bold', pad=-20, loc='right')
    else:
        axis.set_title(r"Emissions 2050 (GtCO$_{\mathbf{2}}$)", fontweight='bold', pad=-20, loc='right')
    
    # Adjust the subplot so that the title fits
    plt.subplots_adjust(top=1.05, left=0.26)
    
    # Remove the legend
    if axis == axes[0]:
        axis.legend().remove()
    
    # Remove the top and right spine
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    
   
    
    # Calculate the x-coordinates for the end of the middle and bottom bars
    mid_bar_x = data.iloc[1, :].sum()  # Middle of the second bar
    bottom_bar_x = data.iloc[0, :].sum()  # Middle of the bottom bar
    #print(bottom_bar_x)
    
    # Add annotation with a 90-degree turn arrow
    axis.annotate(
        "",
        xy=(bottom_bar_x, 0),              # Start at the middle of the second bar
        xytext=(mid_bar_x, 1),    # End at the middle of the lowest bar with vertical offset
        ha="left", va="center",
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.15,head_length=0.1",
            connectionstyle="angle,angleA=90,angleB=0,rad=0",  # 90-degree turn
            color="black",
            lw=1.5,
            shrinkA=0,shrinkB=0
        ))

    # Position the text label slightly to the right of the arrow
    axis.text(mid_bar_x *1.03, 0.5, "Combination gain", ha="left", va="center")


plot_stacked_bar_chart(df_cumulative, axes[0])
plot_stacked_bar_chart(df_2050, axes[1])

# Create nice dataframe for printing
data = pd.DataFrame(
    [sectoral_saved_emissions_2050, emissions_2050_combined_policies],
    index=["One at a time", "Combined"],
    columns=["Power", "Heat", "Transport", "Freight"]
)

data = data.round(2)
# Print the table nicely
print(data)

total_saved_emissions_2050 = np.sum(sectoral_saved_emissions_2050) - np.sum(emissions_2050_combined_policies)
print(f"The additional emissions savings in 2050: {total_saved_emissions_2050:.0f} MtCO2")


# %%
