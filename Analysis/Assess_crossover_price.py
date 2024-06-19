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

pickle_path = os.path.join(project_root_absolute_path, 'Output\\Results2.pickle') 

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
regions = {'India': 41, "China": 40, "Brazil": 43, "United States": 33, "UK": 14, "Germany": 2}

# For the clean technology, we choose the clean technology with the largest share in 2030. We choose between:
# FTT:P --> wind or solar (16/18)
# FTT:Tr --> One of the three categories of electric transport (18, 19, 20)
# FTT:H --> Either air-air heat pumps or water-air (10, 11)
# FTT:Fr --> Only look at small trucks, 12 is the electric vehicle one. 

# For the fossil technology, we similarly choose the dominant for with the largest market share in 2030. We choose between:
# FTT:P --> coal, gas or nuclear (0, 2, 6)
# FTT:Tr --> petrol vs diesel (0-11). Compare like to like, so if the biggest category is mid, compare with mid
# FTT:H --> Gas everywhere, right? Or also electric heating and coal?

# Import classification titles from utilities
titles = load_titles()


# Define the clean technology list by model
clean_techs = {"FTT:P": [16, 18], "FTT:Tr": [18, 19, 20], "FTT:H": [10, 11], "FTT:Fr": [12]}
dirty_techs = {"FTT:P": [0, 2, 6], "FTT:Tr": list(range(12)), "FTT:H": [3, 4], "FTT:Fr": [0, 2, 4, 6, 8]}

# Define the year of interest
year = 2030

# Define the price of interest
models = {"FTT:P", "FTT:Tr", "FTT:H", "FTT:Fr"}
price_names = {"FTT:P": "MECW", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
shares_names = {"FTT:P": "MEWS", "FTT:Tr": "TEWS", "FTT:H": "HEWS", "FTT:Fr": "ZEWS"}

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
def find_biggest_tech(output, tech_lists, year, model):
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

def remove_vehicles_from_list(dirty_techs, biggest_techs_clean):
    """ Remove cars in different luxery categories from consideration."""
    if model != "FTT:Tr":
        return dirty_techs
    
    for r, tech in biggest_techs_clean.items():
        if tech == 18:
            dirty_techs["FTT:Tr"] = [0, 3, 6, 9]
        if tech == 19:
            dirty_techs["FTT:Tr"] = [1, 4, 7, 10]
        if tech == 20:
            dirty_techs["FTT:Tr"] = [2, 5, 8, 11]
    return dirty_techs
        
        

def get_prices(output, year, model, biggest_technologies):
    """Get the prices of the biggest technologies."""
    price_var = price_names[model]
    prices = {}
    for r, tech in biggest_technologies.items():
        prices[r] = output[price_var][regions[r], tech, 0, year - 2010 + 1]
    return prices

def get_crossover_year(output, model, biggest_techs_clean, biggest_techs_dirty):
    """ Get the year when the clean technology becomes cheaper than the dirty technology."""
    crossover_years = {}
    for r, ri in regions.items():
        tech_clean = biggest_techs_clean[r]
        tech_dirty = biggest_techs_dirty[r]
        try:
            price_series_clean = output[price_names[model]][ri, tech_clean, 0, :]
            price_series_dirty = output[price_names[model]][ri, tech_dirty, 0, :]
            crossover_years[r] = np.where(price_series_clean <= price_series_dirty)[0][0] + 2010
        except IndexError as e:
            crossover_years[r] = None
    return crossover_years


corss_overs = get_crossover_year(
    output, "FTT:P", 
    find_biggest_tech(output, clean_techs, year, "FTT:P"), find_biggest_tech(output, dirty_techs, year, "FTT:P"))

# Construct the dataframe
df = pd.DataFrame(columns=["Region", "Sector", "Clean technology", "Dirty technology", "Clean price (2020)", "Dirty price (2020)", "Cross-over year"])
rows = []
for model in models:
    biggest_techs_clean = find_biggest_tech(output, clean_techs, year, model)
    dirty_techs = remove_vehicles_from_list(dirty_techs, biggest_techs_clean)
    biggest_techs_dirty = find_biggest_tech(output, biggest_techs_clean, dirty_techs, year, model)
    prices_clean = get_prices(output, year, model, biggest_techs_clean)
    prices_dirty = get_prices(output, year, model, biggest_techs_dirty)
    crossover_years = get_crossover_year(output, model, biggest_techs_clean, biggest_techs_dirty)
    for r in regions:
        row = {
        "Region": r, 
        "Sector": model, 
        "Clean technology": biggest_techs_clean[r], 
        "Dirty technology": biggest_techs_dirty[r], 
        "Clean price (2020)": prices_clean[r], 
        "Dirty price (2020)": prices_dirty[r],
        "Cross-over year": crossover_years[r]
        }
        rows.append(row)

# Construct the DataFrame from the list of dictionaries
df = pd.DataFrame(rows, columns=["Region", "Sector", "Clean technology", "Dirty technology", "Clean price (2020)", "Dirty price (2020)", "Cross-over year"])