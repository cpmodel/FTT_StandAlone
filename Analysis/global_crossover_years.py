# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 20:48:15 2024

@author: Owner
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import csv

from preprocessing import get_output, get_metadata

# Set global font size
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'xtick.labelsize': 13, 'ytick.labelsize': 13})

output_file = "Results.pickle"

output = get_output("Results.pickle", "FTT-Tr")

titles, fig_dir, tech_titles, models = get_metadata()

# Define the regions and the region numbers of interest
current_dir = os.getcwd()
print(current_dir)

top_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
input_dir = os.path.join(top_dir, "Inputs")

df = pd.read_csv(os.path.join(current_dir, "Analysis/e3me_regions_jan23.csv"))

# Convert the DataFrame to a dictionary
regions = dict(zip(df['Country'], df['Value']))


# Define the clean technology list by model
clean_techs = {"FTT:P": [16, 18], "FTT:Tr": [18, 19, 20], "FTT:H": [10, 11], "FTT:Fr": [12]}
dirty_techs = {"FTT:P": [0, 2, 6], "FTT:Tr": list(range(12)), "FTT:H": [2, 3], "FTT:Fr": [0, 2, 4, 6, 8]}


# Define the shares, prices of interest
model_names_r = ["Trucks", "Cars", "Heating", "Power"]
price_names = {"FTT:P": "MEWC", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
shares_names = {"FTT:P": "MEWS", "FTT:Tr": "TEWS", "FTT:H": "HEWS", "FTT:Fr": "ZEWS"}
operation_cost_name = {"FTT:P": "MLCO"}
# TODO: should carbon tax be part of this? Probably not, right?

shares_variables = {"FTT:P": "MEWG", "FTT:Tr": "TEWK", "FTT:Fr": "ZEWK", "FTT:H": "HEWG"}
tech_variable = {"FTT:P": 18, "FTT:Tr": 19, "FTT:H": 11, "FTT:Fr": 12}

# Define the year of interest
year = 2030

year_inds = list(np.array([2030]) - 2010)

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

def interpolate_crossover_year(price_series_clean, price_series_fossil):
    """Interpolate based on price difference in cross-over year and previous year
    Returns None if the prices of the clean technology are higher than 
    the fossil technolgy throughout."""
    
    crossover_index = np.where(price_series_clean <= price_series_fossil)[0][0]
    year_before = 2020 + crossover_index - 1
    
    # Interpolating between the year_before and the crossover year
    price_before = price_series_clean[crossover_index - 1]
    price_after = price_series_clean[crossover_index]
    
    fossil_price_before = price_series_fossil[crossover_index - 1]
    fossil_price_after = price_series_fossil[crossover_index]
    
    # Linear interpolation formula to find the fraction of the year
    fraction = (fossil_price_before - price_before) / ((price_after - price_before) - (fossil_price_after - fossil_price_before))
    
    crossover_year = year_before + fraction
        
    if crossover_year < 2021:
        crossover_year = 2021
    
    return crossover_year

def get_crossover_year(output, model, biggest_techs_clean, biggest_techs_fossil, price_names):
    """ Get the year when the clean technology becomes cheaper than the fossil technology."""
    crossover_years = {}
    for r, ri in regions.items():
        tech_clean = biggest_techs_clean[r]
        tech_fossil = biggest_techs_fossil[r]
        try:
            price_series_clean = output[price_names[model]][ri, tech_clean, 0, 10:]
            price_series_fossil = output[price_names[model]][ri, tech_fossil, 0, 10:]
            crossover_years[r] = interpolate_crossover_year(price_series_clean, price_series_fossil)
    
            
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
    

#%%
# Calculate weighted average costs for clean and fossils
def calculate_weighted_average_costs(output, models, regions, price_names, shares_variables, tech_variable):
    weighted_avg_costs_clean = {}
    weighted_avg_costs_fossil = {}

    for model in models:
        clean_costs = []
        fossil_costs = []
        weights = []
        
        for r, ri in regions.items():
            try:
                # Get price series for clean and fossil technologies
                tech_clean = clean_techs[model]
                price_series_clean = output[price_names[model]][ri, tech_clean, 0, 10:]
                tech_fossil = dirty_techs[model]
                price_series_fossil = output[price_names[model]][ri, tech_fossil, 0, 10:]
                
                # Get shares as weights
                share_data_clean = output[shares_variables[model]][ri, tech_clean, :]
                share_data_fossil = output[shares_variables[model]][ri, tech_fossil, :]
                weight = np.sum(share_data_clean) + np.sum(share_data_fossil)
                
                clean_costs.append(price_series_clean)
                fossil_costs.append(price_series_fossil)
                weights.append(weight)
                
            except KeyError as e:
                print(f"Invalid key access: {e}")
                continue
        
        if weights:
            # Calculate weighted average costs
            weighted_avg_costs_clean[model] = np.average(clean_costs, axis=0, weights=weights)
            weighted_avg_costs_fossil[model] = np.average(fossil_costs, axis=0, weights=weights)
        else:
            weighted_avg_costs_clean[model] = None
            weighted_avg_costs_fossil[model] = None

    return weighted_avg_costs_clean, weighted_avg_costs_fossil

weighted_avg_costs_clean, weighted_avg_costs_fossil = calculate_weighted_average_costs(output, models, regions, price_names, shares_variables, tech_variable)


def calculate_global_crossover_year_with_costs(output, models, regions, price_names, shares_variables, tech_variable):
    # Calculate weighted average costs for clean and fossil technologies
    weighted_avg_costs_clean, weighted_avg_costs_fossil = calculate_weighted_average_costs(
        output, models, regions, price_names, shares_variables, tech_variable
    )

    global_crossover_years = {}
    for model in models:
        # Get the weighted average costs for the current model
        clean_costs = weighted_avg_costs_clean.get(model)
        fossil_costs = weighted_avg_costs_fossil.get(model)

        if clean_costs is not None and fossil_costs is not None:
            # Find the minimum costs across regions for each year
            min_clean_costs = np.min(clean_costs, axis=0)
            min_fossil_costs = np.min(fossil_costs, axis=0)
            
            # Identify the first year where clean costs are less than fossil costs
            crossover_year_index = np.where(min_clean_costs < min_fossil_costs)[0]
            
            if crossover_year_index.size > 0:
                global_crossover_year = crossover_year_index[0] + 2020
                global_crossover_years[model] = global_crossover_year
            else:
                global_crossover_years[model] = None
        else:
            global_crossover_years[model] = None

    return global_crossover_years


global_crossover_years = calculate_global_crossover_year_with_costs(
    output=output,
    models=models,
    regions=regions,
    price_names=price_names,
    shares_variables=shares_variables,
    tech_variable=tech_variable
)

print(global_crossover_years)



#%%
output_csv_path = "Analysis/global_crossover_years_for_Tr.csv"

with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Crossover Year"])
    for model_name, crossover_year in global_crossover_years.items():
        writer.writerow([model_name, crossover_year])

print(f"Data saved to {output_csv_path}")

