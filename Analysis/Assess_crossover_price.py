# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:39:28 2024

@author: fjmn202
"""

# Import the results pickle file
import pickle
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your script is in a subdirectory of the project root, adjust the relative path accordingly.
# For example, if your script is in 'src/scripts', you might use '../..' to reach the project root.
project_root_relative_path = '..'  # Adjust this path to match your project structure

# Get the directory of the current script & the path to the pickle file
script_dir = os.path.dirname(__file__)

# Calculate the absolute path to the project root
project_root_absolute_path = os.path.abspath(os.path.join(script_dir, project_root_relative_path))

# Change the current working directory to the project root
os.chdir(project_root_absolute_path)

# # Add the SourceCode directory to Python's search path
# source_code_path = os.path.join(project_root_absolute_path, 'SourceCode')
# sys.path.append(source_code_path)

pickle_path = os.path.join(project_root_absolute_path, 'Output\\Results.pickle') 

with open(pickle_path, 'rb') as f:
    results = pickle.load(f)


# Attempt to import again
try:
    from SourceCode.support.titles_functions import load_titles
    print("Import successful")
except ModuleNotFoundError as e:
    print(f"Import failed: {e}")
    # Troubleshooting step 4: Use an absolute path for verification
    sys.path.append(r"C:\Users\fjmn202\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone_laptop_repos\FTT_StandAlone\SourceCode")
    sys.path.append(r"C:\Users\fjmn202\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone\FTT_StandAlone\SourceCode")
    from support.titles_functions import load_titles
# Local library imports



# Extract the results for S0
output = results['S0']

# Define the regions and the region numbers of interest
regions = {'India': 41, "China": 40, "Brazil": 43, "United States": 33, "UK": 14, "Germany": 2, "France": 6}

# Import classification titles from utilities
titles = load_titles()

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

# Get names of the technologies of interest
tech_titles = {"FTT:P": "T2TI", "FTT:Tr": "VTTI", "FTT:Fr": "FTTI", "FTT:H": "HTTI"}

# Construct a dataframe with the biggest clean and dirty technologies.
# The dataframe will have the following columns:
# - Region
# - Sector
# - Clean technology
# - Dirty technology
# - Clean price (2020)
# - Dirty price (2020)
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
        prices[r] = output[price_var][regions[r], tech, 0, year - 2010 + 1]
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