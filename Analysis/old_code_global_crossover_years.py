# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 20:48:15 2024

@author: Rishi
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

scenarios = ["S0", "FTT-Tr", "FTT-Fr", "FTT-P", "FTT-H"]

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
clean_techs = {"FTT:P": [18], "FTT:Tr": [19], "FTT:H": [10], "FTT:Fr": [12]}
dirty_techs = {"FTT:P": [2], "FTT:Tr": list(range(12)), "FTT:H": [2], "FTT:Fr": [4]}


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
        dirty_techs["FTT:Tr"] = [1]
    elif tech == 19:
        dirty_techs["FTT:Tr"] = [1]
    elif tech == 20:
        dirty_techs["FTT:Tr"] = [1]
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
    """Interpolate based on price difference in cross-over year and previous year.
    Returns inf if the prices of the clean technology are higher than the fossil technology throughout.
    Returns -inf if the clean technology has always been cheaper.
    """
    if all(price_series_clean > price_series_fossil):
        return np.inf  # Clean technology never becomes cheaper
    
    if all(price_series_clean < price_series_fossil):
        return -np.inf  # Clean technology is already cheaper
    
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
    """Get the year when the clean technology becomes cheaper than the fossil technology."""
    crossover_years = {}
    for r, ri in regions.items():
        tech_clean = biggest_techs_clean[r]
        tech_fossil = biggest_techs_fossil[r]
        try:
            price_series_clean = output[price_names[model]][ri, tech_clean, 0, 10:]
            price_series_fossil = output[price_names[model]][ri, tech_fossil, 0, 10:]
            crossover_year = interpolate_crossover_year(price_series_clean, price_series_fossil)
            crossover_years[r] = crossover_year
            
        except IndexError:
            crossover_years[r] = None
    return crossover_years


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
    

def calculate_global_crossover_year(output, models, regions, price_names, shares_variables, tech_variable):
    global_crossover_years = {}
    for model in models:
        
        crossover_years = get_crossover_year(output, model, biggest_techs_clean, biggest_techs_fossil, price_names)
        
        # Separate years and regions
        finite_years = [(year, region) for region, year in crossover_years.items() if year not in [None, np.inf, -np.inf]]
        always_cheaper_regions = [region for region, year in crossover_years.items() if year == -np.inf]
        never_cheaper_regions = [region for region, year in crossover_years.items() if year == np.inf]
        
        if not finite_years and not always_cheaper_regions:
            global_crossover_years[model] = None
        else:
            # Separate years and regions
            years, region_list = zip(*finite_years) if finite_years else ([], [])
            
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
                global_crossover_year = np.average(years, weights=weights) if years else None
                global_crossover_years[model] = global_crossover_year
            else:
                global_crossover_years[model] = None

            # Handle regions where clean technology is already cheaper
            if always_cheaper_regions:
                for region in always_cheaper_regions:
                    try:
                        share_data = output[shares_variables[model]][regions[region], tech_variable[model], :]
                        weight = np.sum(share_data)
                        if global_crossover_years[model] is None:
                            global_crossover_years[model] = 2020  # Assume the earliest possible year
                        global_crossover_years[model] = (global_crossover_years[model] * sum(weights) + 2020 * weight) / (sum(weights) + weight)
                    except KeyError as e:
                        print(f"Invalid key access for always cheaper region: {e}")
                        
    return global_crossover_years

global_crossover_years = {}

for scenario in scenarios:
    output_file = "Results.pickle"
    output = get_output(output_file, scenario)
    rows = []
    for model in models:
        biggest_techs_clean = find_biggest_tech(output, clean_techs, year, model, regions)
        biggest_techs_fossil = find_biggest_tech_dirty(output, dirty_techs, biggest_techs_clean, year, model)
        clean_tech_names = {key: titles[tech_titles[model]][index] for key, index in biggest_techs_clean.items()}
        fossil_tech_names = {key: titles[tech_titles[model]][index] for key, index in biggest_techs_fossil.items()}
        prices_clean = get_prices(output, year, model, biggest_techs_clean)
        prices_dirty = get_prices(output, year, model, biggest_techs_fossil)
        crossover_years = get_crossover_year(output, model, biggest_techs_clean, biggest_techs_fossil, price_names)
        global_crossover_years[scenario] = calculate_global_crossover_year(output=output, models=models, regions=regions, price_names=price_names, shares_variables=shares_variables, tech_variable=tech_variable)

print(global_crossover_years)


# Calculate the difference from the baseline scenario (S0)
baseline_scenario = "S0"
baseline_crossover_years = global_crossover_years[baseline_scenario]
differences_from_baseline = {}

for scenario in scenarios:
    if scenario != baseline_scenario:
        scenario_differences = {}
        for model in models:
            # Ensure the model exists in both the current scenario and the baseline
            if model in global_crossover_years[scenario] and model in baseline_crossover_years:
                # Check if both values are not None
                if global_crossover_years[scenario][model] is not None and baseline_crossover_years[model] is not None:
                    # Calculate the difference for the corresponding model
                    scenario_differences[model] = global_crossover_years[scenario][model] - baseline_crossover_years[model]
                else:
                    # If either value is None, set the difference to None
                    scenario_differences[model] = None
        # Store the differences for the current scenario
        differences_from_baseline[scenario] = scenario_differences

# Print the differences from the baseline scenario
print("\nDifferences from Baseline Scenario (S0):")
for scenario, differences in differences_from_baseline.items():
    print(f"{scenario}: {differences}")

#%%    

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to convert fractional years to "years and months"
def convert_to_years_months(value):
    if np.isnan(value):
        return '0 months'
    years = int(value)
    months = int((value - years) * 12)
    return f'{years} years {months} months' if years > 0 else f'{months} months'

# Convert dictionary to DataFrame
df = pd.DataFrame(differences_from_baseline, index=['FTT:P', 'FTT:H', 'FTT:Tr', 'FTT:Fr'])

# Apply conversion to DataFrame
df_converted = df.applymap(convert_to_years_months)

# Mapping dictionary for policy names
policy_names = {
    'FTT:P': 'Power policies',
    'FTT:H': 'Heat policies',
    'FTT:Tr': 'Transport policies',
    'FTT:Fr': 'Forestry policies'
}

# Update row labels
df_converted.rename(index=policy_names, inplace=True)

# Plotting the table with enhanced aesthetics
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed

# Hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Create table plot
table = ax.table(cellText=df_converted.values,
                 colLabels=df.columns,
                 rowLabels=df_converted.index,
                 cellLoc='center',
                 loc='center',
                 edges='BRLT')

# Adjust table properties
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.1, 2.0)  # Adjust the scaling as needed

# Aesthetic improvements
header_color = '#40466e'
header_text_color = 'w'
row_colors = ['#f2f2f2', 'w']

# Set header properties
for (i, j), cell in table.get_celld().items():
    if j == -1:
        cell.set_text_props(fontweight='bold', color='black')  # Row labels bold
    if i == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text_color, fontweight='bold')  # Column labels white text and bold
    if i > 0 and j > -1:
        cell.set_facecolor(row_colors[i % 2])  # Alternating row colors

# Add titles as text annotations
plt.suptitle('Global Crossover Years by Policy', fontsize=16, fontweight='bold', x=0.6, y=0.8)
ax.text(0.5, 0.8, 'Sectors', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')

plt.subplots_adjust(left=0.25, top=0.8, bottom=0.2, right=0.95)

plt.show()
