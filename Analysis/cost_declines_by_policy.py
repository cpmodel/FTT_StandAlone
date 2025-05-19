# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:13:22 2024

This script produces one figure. That is, the bare levelised cost difference,
based on the policy. 

@author: Femke Nijsse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from preprocessing import get_output, get_metadata, save_fig, save_data

import config

#%%
# Import the results pickle file
output_file = "Results_sxp.pickle"
titles, fig_dir, tech_titles, models, shares_vars = get_metadata()


price_names = {"FTT:P": "MECW battery only", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
operation_cost_name = {"FTT:P": "MLCO"}

tech_name = {"FTT:P": "Solar PV", "FTT:Tr": "EV (mid-range)",
             "FTT:H": "Air-air heat pump", "FTT:Fr": "Small electric truck"}

year_inds = [year - 2010 for year in [2024, 2035, 2050]]

# Define the percentage difference function
def get_percentage_difference(clean_price, dirty_price):
    return 100 * (clean_price - dirty_price) / dirty_price


def get_weighted_costs(output, model, tech_variable, year_inds):
    """Get the weighted cost based on the scenario (output), model,
    tech_variable and the indices of the years of interest.
    """
    
    if model == "FTT:P" and tech_variable in [2, 6]:
        prices = output[operation_cost_name[model]][:, tech_variable, 0, year_inds]
    else:
        prices = output[price_names[model]][:, tech_variable, 0, year_inds]
    
    # Weigh by total size of the market per region
    weights = np.sum(output[shares_vars[model]][:, :, 0, year_inds], axis=1)    
    weighted_prices = np.average(prices, weights=weights, axis=0)

    
    return weighted_prices

def get_costs(output, model, tech_variable, year_inds):
    """Get the  cost based on the scenario (output), model,
    tech_variable and the indices of the years of interest.
    """
    
    if model == "FTT:P" and tech_variable in [2, 6]:
        prices = output[operation_cost_name[model]][:, tech_variable, 0, year_inds]
    else:
        prices = output[price_names[model]][:, tech_variable, 0, year_inds]

    return prices

def get_weighted_percentage_difference(output, model, 
                                       prices_clean, prices_fossil, year_inds):
    
    """First compute the percentage difference in each region,
    Then compute the weighted difference based on overall market share"""
    
    percentage_difference = get_percentage_difference(prices_clean, prices_fossil)
    weights = np.sum(output[shares_vars[model]][:, :, 0, year_inds], axis=1)    
    weighted_difference = np.average(percentage_difference, weights=weights, axis=0)
    
    return weighted_difference


output_S0 = get_output(output_file, "S0")



#%% =====================================================================
# Globally averaged cost difference over time by within-sector policy
# ========================================================================

clean_tech_variable = {"FTT:P": 18, "FTT:Tr": 19, "FTT:H": 10, "FTT:Fr": 33}
fossil_tech_variable = {"FTT:P": 2, "FTT:Tr": 1, "FTT:H": 3, "FTT:Fr": 13} 
graph_label = {"FTT:P": "New solar + battery vs existing coal", "FTT:H": "Air-to-water HP vs gas boiler",
               "FTT:Tr": "Electric vehicles vs petrol cars", "FTT:Fr": "Electric trucks vs diesel trucks"}

price_diff_perc_by_region = {}
weights_by_region = {} 
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
     
    
    for scen, output in scenarios.items():
               
        prices_clean = get_costs(output, model, clean_tech_variable[model], year_inds)
        prices_fossil = get_costs(output, model, fossil_tech_variable[model], year_inds)
        price_diff_perc_by_region[model] = get_percentage_difference(prices_clean, prices_fossil)
        weights_by_region[model] = np.sum(output[shares_vars[model]][:, :, 0, year_inds], axis=1)   
        price_diff_perc = get_weighted_percentage_difference(output, model, 
                                               prices_clean, prices_fossil, year_inds)
        timeseries_by_policy.append(price_diff_perc)
        
    
    timeseries_dict[model] = timeseries_by_policy

#%% Global cost difference -- plotting
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.8), sharey=True)
axs = axs.flatten()

# Get 4 colours from the "mako" palette
colours = sns.color_palette("mako", 4)

def custom_xaxis_formatter(x, pos):
    if x in [2030, 2040, 2050]:
        return f'{int(x)}'
    else:
        return ''

for mi, model in enumerate(models):
    ax = axs[mi]

    ax.axhline(y=0, color="grey", linewidth=2)    
    
    
    for si, (scen, colour) in enumerate(zip(scenarios.keys(), colours)):
        x_vals = range(2023, 2051)
        y_vals = timeseries_dict[model][si]
        ax.plot(x_vals, y_vals, label=scen, color=colour, linewidth=1)
        
        # Find zero crossings
        for i in range(len(y_vals) - 1):
            if y_vals[i] * y_vals[i + 1] < 0:  # Sign change indicates a zero crossing
                # Linear interpolation for exact crossing point
                x_cross = x_vals[i] + (0 - y_vals[i]) * (x_vals[i + 1] - x_vals[i]) / (y_vals[i + 1] - y_vals[i])
                ax.axvline(x_cross, color=colour, linestyle="--", linewidth=1)
    
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    if mi in [0, 2, 3]:
        ax.set_xlim(2024, 2035)
        ax.grid(True, which='major', linewidth=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    else:
        ax.set_xlim(2024, 2050)
        ax.grid(True, which='major', linewidth=0.3)
        ax.grid(True, which='minor', linewidth=0.05)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

    ax.set_ylim(-15, 40)
    
    # Set the major x-axis ticks to be at set intervals for heat
    if mi == 1:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_xaxis_formatter))
    
    ax.grid(True, axis='y', linewidth=0.3)
    
    if mi == 3:
        ax.legend(loc='upper right')
    
    if mi in [0, 2]:
        ax.set_ylabel("Levelised cost difference (%)")
    
    ax.set_title(graph_label[model], ha="right")
    ax.title.set_position((1.07, ax.title.get_position()[1]))
    fig.subplots_adjust(hspace=.4)


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

