# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:13:22 2024

@author: Femke Nijsse
"""

# Import the results pickle file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import get_output, get_metadata, save_fig, save_data
import config


output_file = "Results_sxp.pickle"
titles, fig_dir, tech_titles, models, shares_vars = get_metadata()


price_names = {"FTT:P": "MECW battery only", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
tech_variable = {"FTT:P": 18, "FTT:Tr": 19, "FTT:H": 10, "FTT:Fr": 13}
operation_cost_name = {"FTT:P": "MLCO"}

tech_name = {"FTT:P": "Solar PV", "FTT:Tr": "EV (mid-range)",
             "FTT:H": "Air-air heat pump", "FTT:Fr": "Small electric truck"}

year_inds = [year - 2010 for year in [2024, 2035, 2050]]


def get_weighted_costs(output, model, tech_variable, year_inds):
    """Get the weighted cost based on the scenario (output), model,
    tech_variable and the indices of the years of interest.
    """
    
    if model == "FTT:P" and tech_variable in [2, 6]:
        prices = output[operation_cost_name[model]][:, tech_variable, 0, year_inds]
    else:
        prices = output[price_names[model]][:, tech_variable, 0, year_inds]
    
    # Way by total size of the market per region
    weights = np.sum(output[shares_vars[model]][:, :, 0, year_inds], axis=1)
   
    
    weighted_prices = np.average(prices, weights=weights, axis=0)
    
    return weighted_prices

output_S0 = get_output(output_file, "S0")


#%% =============================================================================
# Cost declines wrt to 2024 at 2035 and 2050 (omitted in favour of different graph)
# ==================================================================================

plot_dot_graph = False

if plot_dot_graph:
    df_dict = {}         # Creates a dict later filled with dataframes

    for model in models:
        df_dict[model] = pd.DataFrame()
        rows = []
        
        # Get the bit of the model name after the colon (like Fr)
        model_abb = model.split(':')[1]
        output_ct = get_output(output_file, f"sxp - {model_abb} CT")
        output_sub = get_output(output_file, f"sxp - {model_abb} subs")
        output_man = get_output(output_file, f"sxp - {model_abb} mand")
        
        scenarios = {"Current traj.": output_S0, "Carbon tax": output_ct,
                     "Subsidies": output_sub, "Mandates": output_man}
        
        for scen, output in scenarios.items():
            weighted_prices = get_weighted_costs(output, model, tech_variable[model], year_inds)
            normalised_prices = weighted_prices / weighted_prices[0]
            
            row = {"Scenario": scen, "Price 2035": normalised_prices[1], "Price 2050": normalised_prices[2]}
            rows.append(row)
        
        df_dict[model] = pd.DataFrame(rows)
        
    
    # Cost declines 2035/2050 -- plotting
    
    fig, axs = plt.subplots(2, 2, figsize=(7, 10), sharey=True)
    axs = axs.flatten()
    palette = sns.color_palette("Blues_r", 3)
    
    
    for mi, model in enumerate(models):
        df = df_dict[model]
        ax = axs[mi]
        ax.plot(df["Scenario"], df["Price 2035"], 'o', label='Price 2035', markersize=15, color=palette[0])
        ax.plot(df["Scenario"], df["Price 2050"], 'o', label='Price 2050', markersize=15, color=palette[1])
    
        # Add labels and title
        ax.set_xticklabels(df["Scenario"], rotation=90)
        if mi % 2 == 0:
            ax.set_ylabel('Cost relative to 2024')
        ax.set_title(tech_name[model], pad=20)
        
        # Remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.xaxis.set_ticks_position('none') 
        ax.yaxis.set_ticks_position('none') 
        
        ax.set_ylim(0.45, 1.)
        ax.grid(axis='y')
        
    
    # Add legend only to the last plot
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(hspace=1, wspace=200)
    fig.legend(handles, labels, loc='upper right', frameon=False, ncol=2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend
    plt.show()
       
    # Save the graph as an editable svg file
    save_fig(fig, fig_dir, "Cost_declines_by_policy")



#%% =====================================================================
# Globally averaged cost difference over time by within-sector policy
# ========================================================================

clean_tech_variable = {"FTT:P": [18], "FTT:Tr": [19], "FTT:H": [10], "FTT:Fr": [13]}
fossil_tech_variable = {"FTT:P": [2], "FTT:Tr": [1], "FTT:H": [3], "FTT:Fr": [5]} # Note 4 for transport gives an error
graph_label = {"FTT:P": "New solar + battery vs existing coal", "FTT:H": "Water-air HP vs gas boiler",
               "FTT:Tr": "EVs vs petrol cars", "FTT:Fr": "EV trucks vs diesel"}


# Define the percentage difference function
def get_percentage_difference(clean_price, dirty_price):
    return 100 * (clean_price - dirty_price) / dirty_price

def find_lowest_cost_tech(output, model, tech_list, year_ind):
    """Given model output, the model, and a list of techs to consider,
    find the technology with the lowest costs in 2030"""
    
    lowest_cost = 10000
    for tech in tech_list:
        cost = get_weighted_costs(output, model, tech, year_ind)
        if cost < lowest_cost:
            tech_ind_cheapest = tech
            lowest_cost = cost
    return tech_ind_cheapest

            
   
year_inds = list(range(13, 41))
timeseries_dict = {}


for model in models:
    timeseries_by_policy = []   
    
    # Get the bit of the model name after the colon (like Fr)
    model_abb = model.split(':')[1]
    output_ct = get_output(output_file, f"sxp - {model_abb} CT")
    output_sub = get_output(output_file, f"sxp - {model_abb} subs")
    output_man = get_output(output_file, f"sxp - {model_abb} mand")
    
    scenarios = {"Current traj.": output_S0, "Carbon tax": output_ct,
                 "Subsidies": output_sub, "Mandates": output_man}
    
    
    clean_tech_variable[model] = find_lowest_cost_tech(output_S0, model, clean_tech_variable[model], 20)
    fossil_tech_variable[model] = find_lowest_cost_tech(output_S0, model, fossil_tech_variable[model], 20)
    
    for scen, output in scenarios.items():
        weighted_prices_clean = get_weighted_costs(output, model, clean_tech_variable[model], year_inds)
        weighted_prices_fossil = get_weighted_costs(output, model, fossil_tech_variable[model], year_inds)
        price_diff_perc = get_percentage_difference(weighted_prices_clean, weighted_prices_fossil)
        timeseries_by_policy.append(price_diff_perc)
        
    
    timeseries_dict[model] = timeseries_by_policy


#%% Global cost difference -- plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
axs = axs.flatten()

# Get 4 colours from the "rocket" palette
colours = sns.color_palette()

for mi, model in enumerate(models):
    ax = axs[mi]

    ax.axhline(y=0, color="grey", linewidth=2)    
    for si, (scen, colour) in enumerate(zip(scenarios.keys(), colours)):
        ax.plot(range(2023, 2051), timeseries_dict[model][si], label=scen, color=colour, linewidth=2.5)
    
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    
    ax.set_xlim(2024, 2034)
    ax.set_ylim(-5.4, 25.6)
    
    ax.grid(True, which='major', linewidth=0.7)
    ax.grid(True, axis='y', linewidth=0.7)
    
    if mi == 0:
        ax.legend(loc='best')
    
    if mi in [0, 2]:
        ax.set_ylabel("Levelised cost difference (%)")
    
    ax.text(2033, 26, graph_label[model], ha="right")
    

# Initialize an empty DataFrame to collect the results
df_list = []

years = list(range(2023, 2051))
# Iterate over the dictionary to create the DataFrame
for model, arrays in timeseries_dict.items():
    for i, scenario in enumerate(scenarios.keys()):
        # Convert the array to a DataFrame
        temp_df = pd.DataFrame(arrays[i].reshape(1, -1), columns=years)
        temp_df['Model'] = model
        temp_df['Scenario'] = scenario
        df_list.append(temp_df)
        
# Combine all DataFrames into one
final_df = pd.concat(df_list, ignore_index=True)

# Reorder columns to have 'Model' and 'Scenario' first
final_df = final_df[['Model', 'Scenario'] + years]

# Save the graph and its data
save_fig(fig, fig_dir, "Figure 3 - Global_price_perc_diff_timeseries_by_policy")
save_data(final_df, fig_dir, "Figure 3 - Global_price_perc_diff_timeseries_by_policy")






