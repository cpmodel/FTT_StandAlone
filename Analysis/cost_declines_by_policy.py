# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:13:22 2024

@author: Femke Nijsse
"""

# Import the results pickle file
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import get_output, get_metadata

# Set global font size
plt.rcParams.update({'font.size': 14})

# Set global font size for tick labels
plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})

output_file = "Results_sxp.pickle"
titles, fig_dir, tech_titles, models = get_metadata()


price_names = {"FTT:P": "MEWC", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
shares_variables = {"FTT:P": "MEWG", "FTT:Tr": "TEWK", "FTT:Fr": "ZEWK", "FTT:H": "HEWG"}
tech_variable = {"FTT:P": 18, "FTT:Tr": 19, "FTT:H": 11, "FTT:Fr": 12}


tech_name = {"FTT:P": "Solar PV", "FTT:Tr": "EV (mid-range)",
             "FTT:H": "Air-air heat pump", "FTT:Fr": "Small electric truck"}


year_inds = list(np.array([2024, 2035, 2050]) - 2010)


def get_weighted_costs(output, model, tech_variable, year_inds):
    """Get the weighted cost based on the scenario (output), model,
    tech_variable and the indices of the years of interest.
    """
    prices = output[price_names[model]][:, tech_variable, 0, year_inds]
    weights = output[shares_variables[model]][:, tech_variable, 0, year_inds]
    weighted_prices = np.average(prices, weights=weights, axis=0)
    
    return weighted_prices

df_dict = {}         # Creates a dict later filled with dataframes
output_S0 = get_output(output_file, "S0")

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
    

#%% Plot the figure with cost declines by policy
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
figure_output_file = os.path.join(fig_dir, "Cost_declines_by_policy.svg")
figure_output_file2 = os.path.join(fig_dir, "Cost_declines_by_policy.png")

fig.savefig(figure_output_file, format="svg")
fig.savefig(figure_output_file2, format="png")



#%% globally averaged costs over time by policy.   # TODO get final ones from Rishi
clean_tech_variable = {"FTT:P": [18], "FTT:Tr": [19], "FTT:H": [10], "FTT:Fr": [12]}
fossil_tech_variable = {"FTT:P": [6], "FTT:Tr": [1], "FTT:H": [2], "FTT:Fr": [4]} # Note 4 for transport gives an error
timeseries_dict = {}
graph_label = {"FTT:P": "Solar vs gas", "FTT:H": "Water-air HP vs gas boiler",
               "FTT:Tr": "petrol vs EV", "FTT:Fr": "Diesel truck vs EV"}

# TODO: figure out why the cost difference becomes "BETTER" when adding 7 to transport

# Define the percentage difference function
def get_percentage_difference(clean_price, dirty_price):
    return 100 * (clean_price - dirty_price) / dirty_price

def find_lowest_cost_tech(output, model, tech_list, year_ind):
    """Given model output, the model, and a list of techs to consider,
    find the technology with the lowest costs in 2030"""
    
    lowest_cost_start = 10000
    for tech in tech_list:
        cost = get_weighted_costs(output, model, tech, year_ind)
        if cost < lowest_cost_start:
            tech_ind_cheapest = tech
            lowest_costs = cost
    return tech_ind_cheapest
            
   

year_inds = list(range(14, 41))

for model in models:
    timeseries_by_policy = []   
    # Get the bit of the model name after the colon (like Fr)
    model_abb = model.split(':')[1]
    output_ct = get_output(output_file, f"sxp - {model_abb} CT")
    output_sub = get_output(output_file, f"sxp - {model_abb} subs")
    output_man = get_output(output_file, f"sxp - {model_abb} mand")
    
    scenarios = {"Current traj.": output_S0, "Carbon tax": output_ct,
                 "Subsidies": output_sub, "Mandates": output_man}
    
    
    clean_tech_variable[model] = find_lowest_cost_tech(output, model, clean_tech_variable[model], 20)
    fossil_tech_variable[model] = find_lowest_cost_tech(output, model, fossil_tech_variable[model], 20)
    
    for scen, output in scenarios.items():
        weighted_prices_clean = get_weighted_costs(output, model, clean_tech_variable[model], year_inds)
        weighted_prices_fossil = get_weighted_costs(output, model, fossil_tech_variable[model], year_inds)
        price_diff_perc = get_percentage_difference(weighted_prices_clean, weighted_prices_fossil)
        timeseries_by_policy.append(price_diff_perc)
        
        
    
    timeseries_dict[model] = timeseries_by_policy

fig, axs = plt.subplots(2, 2, figsize=(7, 10), sharey=True)
axs = axs.flatten()

for mi, model in enumerate(models):
    df = df_dict[model]
    ax = axs[mi]
    for si, scen in enumerate(scenarios.keys()):
        ax.plot(range(2024, 2051), timeseries_dict[model][si], label=scen)
    ax.axhline(y=0, color="grey")    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    ax.grid()
    if mi == 0:
        ax.legend(loc='best')
    
    if mi in [0, 2]:
        ax.set_ylabel("levelised cost difference (%)")
    
    ax.text(2050, 2, graph_label[model], ha="right")


# Save the graph as an editable svg file
figure_output_file = os.path.join(fig_dir, "Price_diff_timeseries_by_policy_global.svg")
figure_output_file2 = os.path.join(fig_dir, "Price_diff_timeseries_by_policy_global.png")

fig.savefig(figure_output_file, format="svg")
fig.savefig(figure_output_file2, format="png")



