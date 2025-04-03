# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:13:22 2024

This script produces one figure. That is, the bare levelised cost difference,
based on the policy. 

@author: Femke Nijsse
"""

# Import the results pickle file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import get_output, get_metadata, save_fig, save_data
import config


#%%
output_file = "Results_mand_coordination.pickle"
titles, fig_dir, tech_titles, _, _ = get_metadata()
output_S0 = get_output(output_file, "S0")


# Change this to "ZTLC" for levelised cost, and "ZWIC" for upfront cost. 
# Note, I'm taking off subsidies for ZWIC.

price_name = "ZTLC"
tech_variable = {"LDV": 31, "MDV": 32, "HDV": 33}

if price_name == "ZTLC":
    cats = ["LDV", "MDV", "HDV"]
else:
    cats = ["MDV", "HDV"]

tech_name = {"LDV": "Electric van", "MDV": "Small electric truck", 
             "FTT:Fr": "Large electric truck"}

if price_name == "ZWIC":
    regions = {"China": 40}
else:
    regions = {"India": 41, "Germany": 2, "United States": 33}

year_inds = [year - 2010 for year in [2024, 2035, 2050]]

# Define the percentage difference function
def get_percentage_difference(clean_price, dirty_price):
    return 100 * (clean_price - dirty_price) / dirty_price


def get_weighted_costs(output, cat, tech_variable, year_inds):
    """Get the weighted cost based on the scenario (output), cat,
    tech_variable and the indices of the years of interest.
    """
    
    prices = get_costs(output, cat, tech_variable, year_inds)
    # Weigh by total size of the market per region
    weights = np.sum(output["ZEWK"][:, :, 0, year_inds], axis=1)    
    weighted_prices = np.average(prices, weights=weights, axis=0)

    
    return weighted_prices

def get_costs(output, cat, tech_variable, year_inds):
    """Get the  cost based on the scenario (output), cat,
    tech_variable and the indices of the years of interest.
    """
    if price_name == "ZTLC":
        prices = output[price_name][:, tech_variable, 0, year_inds]
    elif price_name == "ZWIC":
        prices = ( output[price_name][:, tech_variable, 0, year_inds] 
                / (1 + output["ZTVT"][:, tech_variable, 0, year_inds]) )
        
    return prices

def get_weighted_percentage_difference(output, cat, 
                                       prices_clean, prices_fossil, year_inds):
    
    """First compute the percentage difference in each region,
    Then compute the weighted difference based on overall market share"""
    
    percentage_difference = get_percentage_difference(prices_clean, prices_fossil)
    weights = np.sum(output["ZEWK"][:, :, 0, year_inds], axis=1)    
    weighted_difference = np.average(percentage_difference, weights=weights, axis=0)
    
    return weighted_difference





#%% =====================================================================
# Globally averaged cost difference over time by within-sector policy
# ========================================================================

clean_tech_variable = {"LDV": 31, "MDV": 32, "HDV": 33}
fossil_tech_variable = {"LDV": 11, "MDV": 12, "HDV": 13} 
graph_label = {"LDV": "Electric van vs diesel", "MDV": "Small electric truck vs diesel",
               "HDV": "Large electric truck vs diesel"}

price_diff_perc_by_region = {}
weights_by_region = {} 
year_inds = list(range(13, 41))
timeseries_dict = {}

for cat in cats:
    timeseries_dict[cat] = {}
    timeseries_by_policy = []
    
    # Get the bit of the cat name after the colon (like Fr)
    output_eur = get_output(output_file, "Fr mand EUR")
    output_p_CN = get_output(output_file, "Fr mand EUR + CN")
    #output_p_US = get_output(output_file, "Fr mand EUR + CN + US") # Not used in first draft
    output_p_IN = get_output(output_file, "Fr mand EUR + CN + US + IN + CA")
    output_p_RoW = get_output(output_file, "sxp - Fr mand")


    
    scenarios = {"Current traj.": output_S0, "+ Europe": output_eur, "+ China": output_p_CN,
                 "+ IN, CA and ½US": output_p_IN, "+ RoW": output_p_RoW}
     
    
    for scen, output in scenarios.items():
               
        prices_clean = get_costs(output, cat, clean_tech_variable[cat], year_inds)
        prices_fossil = get_costs(output, cat, fossil_tech_variable[cat], year_inds)
        price_diff_perc_by_region[cat] = get_percentage_difference(prices_clean, prices_fossil)
        timeseries_by_policy.append(price_diff_perc_by_region[cat][list(regions.values())])

    timeseries_by_policy = np.array(timeseries_by_policy)
    for ri, region in enumerate(regions.keys()):
        timeseries_dict[cat][region] = timeseries_by_policy[:, ri, :]

#%% Global cost difference -- plotting
fig, axs = plt.subplots(len(cats), len(regions), figsize=(4 + 1.2 * len(regions), 2* len(cats)), sharey=True)

# First, use 3 grey colors, and then 3 teal/blue ones
colours = ["#33333C", "#5A6065", "#7B848C", "#004C6D", "#2C7FB8"]

def custom_xaxis_formatter(x, pos):
    if x%5 == 0: # Show multiples of 5
        return f'{int(x)}'
    else:
        return ''

for ci, cat in enumerate(cats): # Loop over rows (vehicle class)
    for ri, region in enumerate(regions): # Loop over columns (regions)
        if len(regions) > 1:
            ax = axs[ci, ri]
        else:
            ax = axs[ci]
        ax.axhline(y=0, color="grey", linewidth=2)    
        
        xmin = 2024
        xmax = 2033 if price_name == "ZTLC" else 2039
        for si, (scen, colour) in enumerate(zip(scenarios.keys(), colours)):
            x_vals = range(2023, 2051)
            y_vals = timeseries_dict[cat][region][si]
            ax.plot(x_vals, y_vals, label=scen, color=colour, linewidth=1)
            
            # Find zero crossings
            for i in range(len(y_vals) - 1):
                if y_vals[i] * y_vals[i + 1] < 0:  # Sign change indicates a zero crossing
                    # Linear interpolation for exact crossing point
                    x_cross = x_vals[i] + (0 - y_vals[i]) * (x_vals[i + 1] - x_vals[i]) / (y_vals[i + 1] - y_vals[i])
                    ax.axvline(x_cross, color=colour, linestyle="--", linewidth=1)
                    
                    # Annotation baseline on the right, others on the left
                    offset = 0.4 if price_name == "ZTLC" else 0.4
                    # Add annotation if x_cross > 2025
                    if xmin < x_cross < xmax:
                        ax.text(x_cross + offset, 10, scen, color=colour,
                                fontsize=7, ha='center', va='bottom', rotation=90)
        
        # Remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.xaxis.set_ticks_position('none') 
        ax.yaxis.set_ticks_position('none') 
        
        ax.set_xlim(xmin, xmax)
        ax.grid(True, which='major', linewidth=0.3)
    
        ax.set_ylim(0, 30)  
        #ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_xaxis_formatter))     
        ax.grid(True, axis='y', linewidth=0.3)
        
        
        if ri == 0:
            if price_name == "ZWIC":
                ax.set_ylabel(f"Price difference {cat} (%)")
            else:
                ax.set_ylabel(f"Cost difference {cat} (%)")

        
        if ci == 0:
            ax.set_title(region)
            
        if ri == 0 and ci == 0:
            ax.annotate("TCO of BEV already lower \n than ICE vehicles", (2025, 10))
        if ri == 0 and ci == 2:
            ax.annotate("TCO of short-haul HDV already \n lower than ICE trucks.\n No data for long-haul HDVS", (2025, 10))
        if ri == 2 and ci == 1:
            ax.annotate("TCO of MDVs already lower \n than ICE trucks", (2025, 10))
        fig.subplots_adjust(hspace=.4)

    
save_fig(fig, fig_dir, "Figure 5 - Policy coordination gains")


#%% Initialize an empty DataFrame to collect the results
df_list = []

years = list(range(2023, 2051))
# Iterate over the dictionary to create the DataFrame
for cat, arrays in timeseries_dict.items(): # Category, region, scenario
    for region in regions:
        for i, scenario in enumerate(scenarios.keys()):
            # Convert the array to a DataFrame
            temp_df = pd.DataFrame([arrays[region][i]], columns=years)
            temp_df['cat'] = cat
            temp_df['Scenario'] = scenario
            temp_df['Region'] = region
            df_list.append(temp_df)
            
# Combine all DataFrames into one
final_df = pd.concat(df_list, ignore_index=True)

# Reorder columns to have 'cat' and 'Scenario' first
final_df = final_df[['cat', 'Scenario'] + years]

# Save the graph and its data
save_data(final_df, fig_dir, f"Figure 8 - Policy coordination {price_name}")


