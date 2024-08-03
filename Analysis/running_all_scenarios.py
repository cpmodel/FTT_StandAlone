# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:22:52 2024

@author: rs1132
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from preprocessing import get_output, get_metadata

# Set global font size
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'xtick.labelsize': 13, 'ytick.labelsize': 13})

def run_scenario(scenario):
    output_file = "Results.pickle"
    output = get_output(output_file, scenario)

    titles, fig_dir, tech_titles, models = get_metadata()

    current_dir = os.getcwd()
    top_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    input_dir = os.path.join(top_dir, "Inputs")

    df = pd.read_csv(os.path.join(current_dir, "Analysis/e3me_regions_jan23.csv"))
    regions = dict(zip(df['Country'], df['Value']))

    clean_techs = {"FTT:P": [16, 18], "FTT:Tr": [18, 19, 20], "FTT:H": [10, 11], "FTT:Fr": [12]}
    dirty_techs = {"FTT:P": [0, 2, 6], "FTT:Tr": list(range(12)), "FTT:H": [2, 3], "FTT:Fr": [0, 2, 4, 6, 8]}
    
    price_names = {"FTT:P": "MEWC", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
    shares_names = {"FTT:P": "MEWS", "FTT:Tr": "TEWS", "FTT:H": "HEWS", "FTT:Fr": "ZEWS"}
    shares_variables = {"FTT:P": "MEWG", "FTT:Tr": "TEWK", "FTT:Fr": "ZEWK", "FTT:H": "HEWG"}
    tech_variable = {"FTT:P": 18, "FTT:Tr": 19, "FTT:H": 11, "FTT:Fr": 12}
    year = 2030

    year_inds = list(np.array([2030]) - 2010)

    def find_biggest_tech(output, tech_lists, year, model, regions):
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

    def find_biggest_tech_dirty(output, dirty_techs, biggest_techs_clean, year, model):
        if model != "FTT:Tr":
            max_techs = find_biggest_tech(output, dirty_techs, year, model, regions)
            return max_techs

        max_techs = {}
        for r, ri in regions.items():
            biggest_tech_clean = (r, biggest_techs_clean[r])
            dirty_techs = remove_vehicles_from_list(dirty_techs, biggest_tech_clean)
            max_techs.update(find_biggest_tech(output, dirty_techs, year, model, {r: ri}))
        return max_techs

    def remove_vehicles_from_list(dirty_techs, biggest_techs_clean):
        if model != "FTT:Tr":
            return dirty_techs

        r, tech = biggest_techs_clean
        if tech == 18:
            dirty_techs["FTT:Tr"] = [0, 3, 6, 9]
        elif tech == 19:
            dirty_techs["FTT:Tr"] = [1, 4, 7, 10]
        elif tech == 20:
            dirty_techs["FTT:Tr"] = [1, 2, 5, 8, 11]
        return dirty_techs

    def get_prices(output, year, model, biggest_technologies):
        price_var = price_names[model]
        prices = {}
        for r, tech in biggest_technologies.items():
            try:
                prices[r] = output[price_var][regions[r], tech, 0, year - 2010]
            except IndexError as e:
                print(regions[r])
                print(model)
                print(tech)
                print(r)
                raise e
        return prices

    def interpolate_crossover_year(price_series_clean, price_series_fossil):
        crossover_index = np.where(price_series_clean <= price_series_fossil)[0][0]
        year_before = 2020 + crossover_index - 1

        price_before = price_series_clean[crossover_index - 1]
        price_after = price_series_clean[crossover_index]

        fossil_price_before = price_series_fossil[crossover_index - 1]
        fossil_price_after = price_series_fossil[crossover_index]

        fraction = (fossil_price_before - price_before) / ((price_after - price_before) - (fossil_price_after - fossil_price_before))

        crossover_year = year_before + fraction

        if crossover_year < 2021:
            crossover_year = 2021

        return crossover_year

    def get_crossover_year(output, model, biggest_techs_clean, biggest_techs_fossil, price_names):
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

    def calculate_weighted_average_costs(output, models, regions, price_names, shares_variables, tech_variable):
        weighted_avg_costs_clean = {}
        weighted_avg_costs_fossil = {}

        for model in models:
            clean_costs = []
            fossil_costs = []
            weights = []

            for r, ri in regions.items():
                try:
                    tech_clean = clean_techs[model]
                    price_series_clean = output[price_names[model]][ri, tech_clean, 0, 10:]
                    tech_fossil = dirty_techs[model]
                    price_series_fossil = output[price_names[model]][ri, tech_fossil, 0, 10:]

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
                weighted_avg_costs_clean[model] = np.average(clean_costs, axis=0, weights=weights)
                weighted_avg_costs_fossil[model] = np.average(fossil_costs, axis=0, weights=weights)
            else:
                weighted_avg_costs_clean[model] = None
                weighted_avg_costs_fossil[model] = None

        return weighted_avg_costs_clean, weighted_avg_costs_fossil

    weighted_avg_costs_clean, weighted_avg_costs_fossil = calculate_weighted_average_costs(output, models, regions, price_names, shares_variables, tech_variable)

    def extract_specific_series_for_each_key(original_data, series_indices):
        specific_series = {}
        for key, index in series_indices.items():
            if key in original_data and index < len(original_data[key]):
                specific_series[key] = original_data[key][index]
            else:
                specific_series[key] = None
        return specific_series

    series_indices_for_clean = {'FTT:Fr': 0, 'FTT:H': 0, 'FTT:P': 1, 'FTT:Tr': 1}
    series_indices_for_fossil = {'FTT:Fr': 2, 'FTT:H': 0, 'FTT:P': 2, 'FTT:Tr': 0}

    specific_series_from_clean = extract_specific_series_for_each_key(weighted_avg_costs_clean, series_indices_for_clean)
    specific_series_from_fossil = extract_specific_series_for_each_key(weighted_avg_costs_fossil, series_indices_for_fossil)

    sector_crossover_year = {}

    for key in specific_series_from_fossil.keys():
        ftt_lowest_series_clean = specific_series_from_clean.get(key)
        ftt_lowest_series_fossil = specific_series_from_fossil.get(key)

        sector_crossover_year[key] = interpolate_crossover_year(ftt_lowest_series_clean, ftt_lowest_series_fossil)

    return sector_crossover_year

def convert_fractional_years_to_years_and_months(fractional_year):
    year = int(fractional_year)
    fraction = fractional_year - year
    months = round(fraction * 12)
    if months == 12:
        year += 1
        months = 0
    return year, months

# List of scenarios to run
scenarios = ["S0", "FTT-H", "FTT-P", "FTT-Tr", "FTT-Fr"]

# Dictionary to store fractional year results for each scenario
fractional_year_results = {}

for scenario in scenarios:
    results = run_scenario(scenario)
    fractional_year_results[scenario] = results

# Get the fractional years for FTT-P scenario to subtract from others
ftt_p_fractional_years = fractional_year_results["S0"]

# Dictionary to store the adjusted results
adjusted_results = {}

for scenario, results in fractional_year_results.items():
    if scenario == "S0":
        adjusted_results[scenario] = results
    else:
        adjusted_results[scenario] = {region: results[region] - ftt_p_fractional_years[region] for region in results.keys()}

# Convert the adjusted fractional years to years and months
converted_results = {}

for scenario, results in adjusted_results.items():
    converted_results[scenario] = {region: convert_fractional_years_to_years_and_months(fractional_year) for region, fractional_year in results.items()}

# Save the results dictionary to a file
with open('adjusted_results.pickle', 'wb') as handle:
    pickle.dump(converted_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Print the results for each scenario
for scenario, result in converted_results.items():
    print(f"Scenario: {scenario}")
    for key, value in result.items():
        print(f"{key}: Year = {value[0]}, Month = {value[1]}")

#%%

import matplotlib.pyplot as plt
import pandas as pd

# Load the adjusted results from the pickle file
with open('adjusted_results.pickle', 'rb') as handle:
    converted_results = pickle.load(handle)

# Define the policy names corresponding to the scenarios
policy_names = {
    "FTT-H": "Heat policies",
    "FTT-P": "Power policies",
    "FTT-Tr": "Transport policies",
    "FTT-Fr": "Freight policies"
}

# Convert the results into a pandas DataFrame, excluding the S0 scenario
data = []
for scenario, results in converted_results.items():
    if scenario != "S0":  # Exclude the reference scenario
        for region, (year, month) in results.items():
            if year == 0:
                data.append({'Policy': policy_names[scenario], 'Region': region, 'YearMonth': f'{month} months'})
            else:
                data.append({'Policy': policy_names[scenario], 'Region': region, 'YearMonth': f'{year} years, {month} months'})

df = pd.DataFrame(data)

# Pivot the DataFrame to get the desired format
table_df = df.pivot(index='Policy', columns='Region', values='YearMonth')

# Reorder the rows according to the desired model ordering
model_order = ["Power policies", "Heat policies", "Transport policies", "Freight policies"]
table_df = table_df.reindex(model_order)

# Plotting the table using matplotlib
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the size as needed

# Add titles as text annotations
plt.suptitle('Cost Parity Brought Forward by', fontsize=16, fontweight='bold', x=0.6, y=0.7)
ax.text(0.5, 0.7, 'Sectors', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')

ax.axis('tight')
ax.axis('off')

# Create a table
table = ax.table(cellText=table_df.values, colLabels=table_df.columns, rowLabels=table_df.index, cellLoc='center', loc='center', edges='BRLT')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.1, 2.5)  # Increase the vertical size of rows

header_color = '#40466e'
header_text_color = 'w'
row_colors = ['#f2f2f2', 'w']

for (i, j), cell in table.get_celld().items():
    if j == -1:
        cell.set_text_props(fontweight='bold', color='black')  # Row labels bold
    if i == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text_color, fontweight='bold')  # Column labels white text and bold
    if i > 0 and j > -1:
        cell.set_facecolor(row_colors[i % 2])  # Alternating row colors

plt.subplots_adjust(left=0.25, top=0.8, bottom=0.2, right=0.95)
