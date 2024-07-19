# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:39:28 2024

@author: fjmn202
"""

# Import the results pickle file
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess_data

# Set global font size
plt.rcParams.update({'font.size': 14})
# Set global font size
plt.rcParams.update({'xtick.labelsize': 12, 'ytick.labelsize': 12})

output, titles, fig_dir, tech_titles = preprocess_data("Results_scens.pickle", "S0")


# Define the regions and the region numbers of interest
regions = {'India': 41, "China": 40, "Brazil": 43, "United States": 33, "UK": 14, "Germany": 2}

# Define the clean technology list by model
clean_techs = {"FTT:P": [16, 18], "FTT:Tr": [18, 19, 20], "FTT:H": [10, 11], "FTT:Fr": [12]}
dirty_techs = {"FTT:P": [0, 2, 6], "FTT:Tr": list(range(12)), "FTT:H": [2, 3], "FTT:Fr": [0, 2, 4, 6, 8]}

# Define the year of interest
year = 2030

# Define the shares, prices of interest
models = {"FTT:P", "FTT:Tr", "FTT:H", "FTT:Fr"}
price_names = {"FTT:P": "MECW", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
shares_names = {"FTT:P": "MEWS", "FTT:Tr": "TEWS", "FTT:H": "HEWS", "FTT:Fr": "ZEWS"}
operation_cost_name = {"FTT:P": "MLCO"}



# Construct a dataframe with the biggest clean and dirty technologies.
# The dataframe will have the following columns:
# - Region
# - Sector
# - Clean technology
# - Dirty technology
# - Clean price (2030)
# - Dirty price (2030)
# - Cross-over year if any

# Find the biggest clean or fossil technology:
def find_biggest_tech(output, tech_lists, year, model, regions):
    """Find the biggest technology in each region for a given model."""
    shares_var = shares_names[model]
    tech_list = tech_lists[model]
    max_techs = {}
    for r, ri in regions.items():
        max_share = 0
        for tech in tech_list:
            share = output[shares_var][ri, tech, 0, year - 2010 + 1] 
            if share >= max_share:
                max_share = share
                max_techs[r] = tech
    return max_techs


# Find the biggest clean or fossil technology:
def find_biggest_tech_dirty(output, dirty_techs, biggest_techs_clean, year, model):
    """Find the biggest technology in each region for a given model."""
    
    if model != "FTT:Tr":
        max_techs = find_biggest_tech(output, dirty_techs, year, model, regions)
        return max_techs
    
    max_techs = {}
    for r, ri in regions.items():
        # For FTT:Tr, only compare vehicles in the same luxery category
        biggest_tech_clean = (r, biggest_techs_clean[r])
        dirty_techs = remove_vehicles_from_list(dirty_techs, biggest_tech_clean)
        max_techs.update(find_biggest_tech(output, dirty_techs, year, model, {r: ri}))
    return max_techs

def remove_vehicles_from_list(dirty_techs, biggest_techs_clean):
    """ Remove cars in different luxery categories from consideration."""
    if model != "FTT:Tr":
        return dirty_techs
    
    r, tech = biggest_techs_clean
    if tech == 18:
        dirty_techs["FTT:Tr"] = [0, 3, 6, 9]
    elif tech == 19:
        dirty_techs["FTT:Tr"] = [1, 4, 7, 10]
    elif tech == 20:
        dirty_techs["FTT:Tr"] = [2, 5, 8, 11]
    return dirty_techs
        
        

def get_prices(output, year, model, biggest_technologies):
    """Get the prices of the biggest technologies."""
    price_var = price_names[model]
    prices = {}
    for r, tech in biggest_technologies.items():
        try:
            prices[r] = output[price_var][regions[r], tech, 0, year - 2010]
        except IndexError as e:
            print(regions[r])
            print(model)
            #print(output[price_var])
            print(tech)
            print(r)
            raise e
    return prices

def get_crossover_year(output, model, biggest_techs_clean, biggest_techs_fossil, price_names):
    """ Get the year when the clean technology becomes cheaper than the dirty technology."""
    crossover_years = {}
    for r, ri in regions.items():
        tech_clean = biggest_techs_clean[r]
        tech_fossil = biggest_techs_fossil[r]
        try:
            price_series_clean = output[price_names[model]][ri, tech_clean, 0, 10:]
            price_series_fossil = output[price_names[model]][ri, tech_fossil, 0, 10:]
            crossover_years[r] = np.where(price_series_clean <= price_series_fossil)[0][0] + 2020
        except IndexError:
            crossover_years[r] = None
    return crossover_years

def get_crossover_operational_vs_new(output, model, biggest_techs_clean, biggest_techs_fossil, price_names, op_cost_name):
    crossover_years = {}
    if model != "FTT:P":
        for r, ri in regions.items():
            crossover_years[r] = None
        return crossover_years
    
    for r, ri in regions.items():
        tech_clean = biggest_techs_clean[r]
        tech_fossil = biggest_techs_fossil[r]
        try:
            price_series_clean = output[price_names[model]][ri, tech_clean, 0, 10:]
            price_series_fossil = output[op_cost_name[model]][ri, tech_fossil, 0, 10:]
            crossover_years[r] = np.where(price_series_clean <= price_series_fossil)[0][0] + 2020
        except IndexError:
            crossover_years[r] = None
    return crossover_years
  


rows = []
for model in models:
    biggest_techs_clean = find_biggest_tech(output, clean_techs, year, model, regions)
    biggest_techs_fossil = find_biggest_tech_dirty(output, dirty_techs, biggest_techs_clean, year, model)
    clean_tech_names = {key: titles[tech_titles[model]][index] for key, index in biggest_techs_clean.items()}
    fossil_tech_names = {key: titles[tech_titles[model]][index] for key, index in biggest_techs_fossil.items()}
    prices_clean = get_prices(output, year, model, biggest_techs_clean)
    prices_dirty = get_prices(output, year, model, biggest_techs_fossil)
    crossover_years = get_crossover_year(output, model, biggest_techs_clean, biggest_techs_fossil, price_names)
    crossover_years_op = get_crossover_operational_vs_new(output, model, biggest_techs_clean, biggest_techs_fossil, price_names, operation_cost_name)
    for r in regions:
        row = {
        "Region": r, 
        "Sector": model, 
        "Clean technology": biggest_techs_clean[r], 
        "Clean tech name": clean_tech_names[r],
        "Clean price (2030)": prices_clean[r], 
        "Fossil technology": biggest_techs_fossil[r], 
        "Fossil tech name": fossil_tech_names[r],
        "Fossil price (2030)": prices_dirty[r],
        "Cross-over year": crossover_years[r],
        "Cross-over operational": crossover_years_op[r]
        }
        rows.append(row)

# Construct the DataFrame from the list of dictionaries
df = pd.DataFrame(rows, columns=["Region", "Sector",
                                 "Clean technology", "Clean tech name", "Clean price (2030)", 
                                 "Fossil technology", "Fossil tech name", "Fossil price (2030)", 
                                 "Cross-over year", "Cross-over operational"])


#%% Making a graph with four subplots for each sector.
# Each graph should show the percentage difference between the clean and dirty technology in 2025, 2035 and 2050. 
# The x-axis should the year and the y-axis the percentage difference.
# The title should be the sector name. Each line in the subplot will have a different region. 


# Define the years of interest
years = [2025, 2035, 2050]

# Define the percentage difference function
def get_percentage_difference(clean_price, dirty_price):
    return 100 * (clean_price - dirty_price) / dirty_price

# Define the data for the plot
def compute_percentage_difference(model, years):
    """Compute percentage price difference per year per region in top techs."""
    
    biggest_techs_clean = find_biggest_tech(output, clean_techs, 2030, model, regions)
    biggest_techs_fossil = find_biggest_tech_dirty(output, dirty_techs, biggest_techs_clean, 2030, model)
    percentage_difference = np.zeros((len(regions), len(years)))
    for ri, r in enumerate(regions):
    
        for yi, year in enumerate(years):
            clean_prices = get_prices(output, year, model, biggest_techs_clean)
            fossil_prices = get_prices(output, year, model, biggest_techs_fossil)
               
            percentage_difference[ri, yi] = get_percentage_difference(clean_prices[r], fossil_prices[r])
    
    return percentage_difference

#%% Plot the figure
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
axs = axs.flatten()

for mi, model in enumerate(models):
    percentage_difference = compute_percentage_difference(model, years)      
    ax = axs[mi]  
    
    for ri, r in enumerate(regions):
        ax.plot(years, percentage_difference[ri], label=r, marker='o', markersize=8)
    
    ax.axhline(0, color='grey', linestyle='--', linewidth=2)  # Adding horizontal line at y=0
    ax.set_title(f"{model}")
    
    if mi % 2 == 0:  # Add y-label only to the leftmost subplots
        ax.set_ylabel("Levelised costs difference (%)")
    if mi > 1:  # Add y-label only to the leftmost subplots
        ax.set_xlabel("Year")
  
   
    
    # Remove the top and right frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
            
# Extract handles and labels from the first subplot
handles, labels = axs[0].get_legend_handles_labels()

# Add a single legend for the entire figure, positioned beneath the graph
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3)

plt.tight_layout()
plt.tight_layout()
plt.show()


# Save the graph as an editable svg file

output_file = os.path.join(fig_dir, "Figure1_baseline_price_differences.svg")
fig.savefig(output_file, format="svg")


