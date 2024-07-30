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

from preprocessing import get_output, get_metadata

# Set global font size
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'xtick.labelsize': 13, 'ytick.labelsize': 13})

output_file = "Results.pickle"
output = get_output("Results.pickle", "S0")
#output_ct = get_output(output_file, "Carbon tax")
#output_sub = get_output(output_file, "Subsidies")
#output_man = get_output(output_file, "Mandates")

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
    

def calculate_global_crossover_year(output, models, regions, price_names, shares_variables, tech_variable):
    global_crossover_years = {}
    for model in models:
        # The correct function call
        crossover_years = get_crossover_year(output, model, biggest_techs_clean, biggest_techs_fossil, price_names)
        
        # Extract valid years and corresponding regions
        valid_years = [(year, region) for region, year in crossover_years.items() if year is not None]
        
        if not valid_years:
            global_crossover_years[model] = None
        else:
            # Separate years and regions
            years, region_list = zip(*valid_years)
            
            # Calculate weights based on share variables
            weights = []
            for region in region_list:
                try:
                    share_data = output[shares_variables[model]][regions[region], tech_variable[model], :]
                    weight = np.sum(share_data)
                    weights.append(weight)
                except KeyError as e:
                    print(f"Invalid key access: {e}")
                    weights.append(0)
            
            if weights:
                global_crossover_year = np.average(years, weights=weights)
                global_crossover_years[model] = global_crossover_year
            else:
                global_crossover_years[model] = None

    return global_crossover_years


global_crossover_years = calculate_global_crossover_year(output=output, models=models, regions=regions, price_names=price_names, shares_variables=shares_variables, tech_variable=tech_variable)


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


def find_min_costs_for_year(dict1, dict2, year_index):
    min_costs = {}
    for key in dict1.keys():
        costs1 = dict1[key][:, year_index] if len(dict1[key].shape) > 1 else dict1[key][year_index]
        costs2 = dict2[key][:, year_index] if len(dict2[key].shape) > 1 else dict2[key][year_index]
        min_costs[key] = min(np.min(costs1), np.min(costs2))
    return min_costs

# Example usage with your dictionaries and year index
year_index = 30  # Index for the year 2030
min_costs = find_min_costs_for_year(weighted_avg_costs_clean, weighted_avg_costs_fossil, year_index)
print(min_costs)


def find_crossover_point(costs1, costs2, years):
    differences = costs1 - costs2
    for i in range(1, len(differences)):
        if differences[i-1] >= 0 and differences[i] < 0:
            return years[i]
    return None  # Return None if no crossover point is found


# Determine the index for the year 2030
start_year = 2020
target_year = 2030
year_index = target_year - start_year

# Find the minimum costs for the year 2030
min_costs_2030 = find_min_costs_for_year(weighted_avg_costs_clean, weighted_avg_costs_fossil, year_index)

# Example years array
years = np.arange(start_year, start_year + len(weighted_avg_costs_clean['FTT:Fr'][0]))

# Find crossover point for FTT:Fr
costs_solar = weighted_avg_costs_clean['FTT:Tr'][0]
costs_coal = weighted_avg_costs_fossil['FTT:Tr'][0]
crossover_year = find_crossover_point(costs_solar, costs_coal, years)

print(f"Minimum costs in 2030: {min_costs_2030}")
print(f"Crossover year for Solar and Coal: {crossover_year}")


"""
def convert_to_years_and_months(global_crossover_years):
    converted = {}
    for model, value in global_crossover_years.items():
        year = int(value)
        fractional_part = value - year
        months = round(fractional_part * 12)  # Convert to months and round to nearest integer
        converted[model] = (year, months)
    return converted


converted_years_months = convert_to_years_and_months(global_crossover_years)
print(converted_years_months)


#%%
# Convert to years and months
def convert_to_years_and_months(global_crossover_years):
    converted = {}
    for model, value in global_crossover_years.items():
        year = int(value)
        fractional_part = value - year
        months = round(fractional_part * 12)
        if months == 12:
            year += 1
            months = 0
        converted[model] = (year, months)
    return converted

# Example data (replace this with your actual data)
global_crossover_years = {
    'FTT:P': np.float64(2030.4601304505902),
    'FTT:H': np.float64(2021.6858193655391),
    'FTT:Tr': np.float64(2021.0000000000002),
    'FTT:Fr': np.float64(2027.3177332464743)
}

# Convert the years and months
converted_years_months = convert_to_years_and_months(global_crossover_years)

# Prepare data for the table with actual outputs
data = [['-'],
        ['-'],
        ['-'],
        ['-']]

# Map converted data to the correct table cell
for model, (year, months) in converted_years_months.items():
    if year > 2021:
        time_str = f'{year - 2021} years, {months} months' if months > 0 else f'{year - 2021} years'
    else:
        time_str = f'{months} months'
    
    if model == 'FTT:Tr':
        data[0][0] = time_str
    elif model == 'FTT:P':
        data[1][0] = time_str
    elif model == 'FTT:H':
        data[2][0] = time_str
    elif model == 'FTT:Fr':
        data[3][0] = time_str

# Plotting the table with enhanced formatting
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=data, colLabels=["Fr", ""], rowLabels=["Tr", "P", "H", "Fr"], loc='center', cellLoc='center')

# Enhance the table appearance
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.5, 1.5)  # Scale to make the table bigger
for key, cell in table.get_celld().items():
    cell.set_edgecolor('black')  # Border color
    cell.set_linewidth(1.2)  # Border line width
    cell.set_text_props(ha='center', va='center')  # Center text
    cell.set_text_props(ha='center', va='center', fontsize=14)  # Larger text

plt.title("Transition in Sector brings Cost-parity forward by", pad=20)
plt.show()
"""