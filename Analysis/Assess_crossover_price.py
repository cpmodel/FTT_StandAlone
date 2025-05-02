# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:39:28 2024

This script produces three graphs and a table, showing the cross-over price, 
including the cross-sectoral effects.

@author: fjmn202
"""

# Import the results pickle file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn 
from matplotlib.lines import Line2D

from matplotlib.font_manager import FontProperties
from preprocessing import get_output, get_metadata, save_fig, save_data
import config

output_file = "Results_sxp.pickle"
output_S0 = get_output(output_file, "S0")
titles, fig_dir, tech_titles, models, cap_vars = get_metadata()

# Define the regions and the region numbers of interest
regions = {'India': 41, "China": 40, "Brazil": 43,
           "United States": 33, "Canada": 35, "Germany": 2, "UK": 14,
           "South Africa": 56, "Japan": 34}

regions_all = {i: i - 1 for i in range(1, 72)}

# Define the clean technology list by model
clean_techs = {"FTT:P": [16, 18], "FTT:Tr": [18, 19, 20], "FTT:H": [10, 11], "FTT:Fr": [33]}
dirty_techs = {"FTT:P": [1, 2, 6], "FTT:Tr": list(range(12)), "FTT:H": [2, 3], "FTT:Fr": [13, 23]}

# Define the shares, prices of interest
model_names_r = ["Trucks", "Cars", "Heating", "Power"]
price_names = config.PRICE_NAMES
repl_dict = config.REPL_DICT2
shares_vars = {"FTT:P": "MEWS", "FTT:Tr": "TEWS", "FTT:H": "HEWS", "FTT:Fr": "ZEWS"}
operation_cost_name = {"FTT:P": "MLCO"}
# TODO: should carbon tax be part of this? Probably not, right?

# Define the year of interest
year = 2030

def skip_model_region_combo(model, ri):
    # For figure one, skip those where we have significant proxy data
    skip = False
    if model == "FTT:H":
        # Skip if not EU, Canada, US, China, Korea
        skip = ri not in list(range(30)) + [33, 35, 40, 47]
    if model == "FTT:Tr":
        skip = ri not in list(range(30)) + [33, 34, 40, 41]
    if model == "FTT:Fr":
        skip = ri not in list(range(30)) + [33, 40, 41]
    
    return skip
    


# Find the biggest clean or fossil technology:
def find_biggest_tech(output, tech_lists, year, model, regions):
    """Find the biggest technology in each region for a given model."""
    shares_var = shares_vars[model]
    tech_list = tech_lists[model]
    max_techs = {}
    try:
        for r, ri in regions.items():
            max_share = 0
            for tech in tech_list:
                share = output[shares_var][ri, tech, 0, year - 2010 + 1] 
                if share >= max_share:
                    max_share = share
                    max_techs[r] = tech
    except AttributeError as e:
        print(regions)
        raise e
    for r, ri in regions.items():
        max_share = 0
        for tech in tech_list:
            share = output[shares_var][ri, tech, 0, year - 2010 + 1] 
            if share >= max_share:
                max_share = share
                max_techs[r] = tech
    return max_techs


# Find the biggest clean or fossil technology:
def find_biggest_tech_fossil(output, dirty_techs, biggest_techs_clean, year, model, regions):
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
        
        

def get_prices(output, year, model, biggest_technologies, regions):
    """Get the prices of the biggest technologies."""
    price_var = price_names[model]
    if model == "FTT:P" and (2 in biggest_technologies.values() or 6 in biggest_technologies.values()):
        price_var = operation_cost_name[model]
    prices = {}
    for r, tech in biggest_technologies.items():
        try:
            prices[r] = output[price_var][regions[r], tech, 0, year - 2010]
        except (IndexError, KeyError) as e:
            print(regions)
            print(model)
            print(tech)
            print(r)
            print(biggest_technologies)
            raise e
        
    return prices

def interpolate_crossover_year(price_series_clean, price_series_fossil):
    """ First, find cross-over year. 
    Then, interpolate based on price difference in cross-over year and previous year
    Returns -inf if cost-parity in past, inf if cost-parity not reached before 2050."""
    
    # First check if cost-parity occurs
    # Set crossover year to -inf if cost-parity already achieved and inf if it will not.
    if (price_series_clean <= price_series_fossil).all():
        return float('-inf')
    elif (price_series_clean > price_series_fossil).all():
        return float('inf')
    
    # Then, if we start with cost-parity, but don't have it consistently, also return -inf
    if price_series_clean[0] < price_series_fossil[0]:
        return float('-inf')
    
    crossover_index = np.argmax(price_series_clean <= price_series_fossil)
    year_before = 2020 + crossover_index - 1
    
    # Interpolating between the year_before and the crossover year of clean tech
    price_before = price_series_clean[crossover_index - 1]
    price_after = price_series_clean[crossover_index]
    
    # Same for the fossil price
    fossil_price_before = price_series_fossil[crossover_index - 1]
    fossil_price_after = price_series_fossil[crossover_index]
    
    # Linear interpolation formula to find the fraction of the year
    fraction = (fossil_price_before - price_before) / ((price_after - price_before) - (fossil_price_after - fossil_price_before))
    
    crossover_year = year_before + fraction
      
    try:
        if crossover_year < 2021:
            crossover_year = 2021
    except ValueError as e:
        print(crossover_index)
        print(price_series_clean.shape)
        print(price_series_fossil.shape)
        print(fossil_price_before)
        print(price_before)
        raise e
 
    return crossover_year

def get_crossover_year(output, model, biggest_techs_clean, 
                       biggest_techs_fossil, price_names, regions):
    """ Get the year when the clean technology becomes cheaper than the fossil technology."""
    crossover_years = {}
    for r, ri in regions.items():
        
        tech_clean = biggest_techs_clean[r]
        tech_fossil = biggest_techs_fossil[r]
        
        price_series_clean = output[price_names[model]][ri, tech_clean, 0, 10:]
        price_series_fossil = output[price_names[model]][ri, tech_fossil, 0, 10:]
        crossover_years[r] = interpolate_crossover_year(price_series_clean, price_series_fossil)
    
    return crossover_years

#%% =========================================================================
# First figure: Baseline costs difference over time by region
# ============================================================================

rows = []
for model in models:
    # Get the bit of the model name after the colon (like Fr)
    model_abb = model.split(':')[1]
    output_ct = get_output(output_file, f"sxp - {model_abb} CT")
    output_sub = get_output(output_file, f"sxp - {model_abb} subs")
    output_man = get_output(output_file, f"sxp - {model_abb} mand")
    
    biggest_techs_clean = find_biggest_tech(output_S0, clean_techs, year, model, regions)
    biggest_techs_fossil = find_biggest_tech_fossil(output_S0, dirty_techs, biggest_techs_clean, year, model, regions)
    clean_tech_names = {reg: titles[tech_titles[model]][index] for reg, index in biggest_techs_clean.items()}
    fossil_tech_names = {reg: titles[tech_titles[model]][index] for reg, index in biggest_techs_fossil.items()}
    prices_clean = get_prices(output_S0, year, model, biggest_techs_clean, regions)
    prices_dirty = get_prices(output_S0, year, model, biggest_techs_fossil, regions)
    
    crossover_years = get_crossover_year(output_S0, model, biggest_techs_clean, biggest_techs_fossil, price_names, regions)
    crossover_years_ct = get_crossover_year(output_ct, model, biggest_techs_clean, biggest_techs_fossil, price_names, regions)
    crossover_years_sub = get_crossover_year(output_sub, model, biggest_techs_clean, biggest_techs_fossil, price_names, regions)
    crossover_years_man = get_crossover_year(output_man, model, biggest_techs_clean, biggest_techs_fossil, price_names, regions)

    
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
        "Cross-over": crossover_years[r],
        "Cross-over carbon tax": crossover_years_ct[r],
        "Cross-over subsidies": crossover_years_sub[r],
        "Cross-over mandates": crossover_years_man[r],
         }
        rows.append(row)

# Construct the DataFrame from the list of dictionaries
df_cy = pd.DataFrame(rows, columns=["Region", "Sector",
                                 "Clean technology", "Clean tech name", "Clean price (2030)", 
                                 "Fossil technology", "Fossil tech name", "Fossil price (2030)", 
                                 "Cross-over", "Cross-over carbon tax",
                                 "Cross-over subsidies", "Cross-over mandates"])


def comparison_str(clean_tech, fossil_tech):
    if clean_tech == "19 Solar PV" and fossil_tech == "3 Coal":
        output_str = "New solar vs existing coal"
    elif clean_tech == "19 Solar PV" and fossil_tech == "7 CCGT":
        output_str = "New solar vs existing gas"
    elif clean_tech == "17 Onshore" and fossil_tech == "3 Coal":
        output_str = "New onshore wind vs existing coal"
    elif clean_tech == "17 Onshore" and fossil_tech == "7 CCGT":
        output_str = "New onshore wind vs existing gas"
    
    elif clean_tech == "12 Heatpump AirAir" and fossil_tech in ["3 Gas", "4 Gas condensing"]:
        output_str = "Air-air HP vs gas"
    elif clean_tech == "11 Heatpump AirWater" and fossil_tech in ["3 Gas", "4 Gas condensing"]:
        output_str = "Water-air HP vs gas"
        
    

    elif clean_tech in  ["19 Electric Econ", "20 Electric Mid", "21 Electric Lux"] and fossil_tech in ["1 Petrol Econ", "2 Petrol Mid", "3 Petrol Lux"]:
        output_str = "Electric car vs petrol"
    elif clean_tech in  ["20 Electric Mid", "21 Electric Lux"] and fossil_tech in ["8 Diesel Mid", "9 Diesel Lux"]:
        output_str = "Electric car vs diesel"
        
    elif clean_tech == "BEV HDT" and fossil_tech == "Diesel HDT":
        output_str = "Heavy-duty BEV truck vs diesel"
    elif clean_tech == "BEV MHT" and fossil_tech == "Diesel MHT":
        output_str = "Medium-duty BEV truck vs diesel"
    elif clean_tech == "BEV HDT" and fossil_tech == "Petrol HDT":
        output_str = "Heavy-duty BEV truck vs diesel"
    elif clean_tech == "BEV MDT" and fossil_tech == "Petrol MDT":
        output_str = "Medium-duty BEV truck vs diesel"
    elif clean_tech == "BEV HDT" and fossil_tech == "CNG/LPG HDT":
        output_str = "Heavy-duty BEV truck vs diesel"
        
    else:
        output_str = "TBD"
        print(f"Missing technology combo is {clean_tech}, {fossil_tech}")
        
    return output_str

# Define the mapping of comparison types to linestyles
linestyle_mapping = {
    'New solar vs existing coal': '-',
    'New solar vs existing gas': '--',
    'New onshore wind vs existing coal': '-.',
    'New onshore wind vs existing gas': ':',
    
    'Air-air HP vs gas': '-',
    'Water-air HP vs gas': '--',

    'Electric car vs diesel': '-',
    'Electric car vs petrol': '--',
    
    'EV truck vs diesel': '-',
    'EV truck vs petrol': '--',
}


# Define the years of interest
years = np.arange(2025, 2051)

# Define the percentage difference function
def get_percentage_difference(clean_price, dirty_price):
    return 100 * (clean_price - dirty_price) / dirty_price

# Define the data for the plot
def get_price_differences_percentage(model, years, regions):
    """Compute percentage price difference per year per region in top techs."""
    
    biggest_techs_clean = find_biggest_tech(output_S0, clean_techs, 2030, model, regions)
    biggest_techs_fossil = find_biggest_tech_fossil(output_S0, dirty_techs, biggest_techs_clean, 
                                                    2030, model, regions)
    price_difference_percentage = np.zeros((len(regions), len(years)))
    
    for ri, r in enumerate(regions):
        for yi, year in enumerate(years):
            clean_prices = get_prices(output_S0, year, model, biggest_techs_clean, regions)
            fossil_prices = get_prices(output_S0, year, model, biggest_techs_fossil, regions)
               
            price_difference_percentage[ri, yi] = get_percentage_difference(clean_prices[r], fossil_prices[r])
    
    return price_difference_percentage


def find_intersections(x, y):
    """Find the x-values where the line crosses y=0."""
    intersections = []
    for i in range(1, len(y)):
        if y[i-1] * y[i] < 0:  # Check for sign change
            # Linear interpolation to find the exact crossing point
            x_cross = x[i-1] - y[i-1] * (x[i] - x[i-1]) / (y[i] - y[i-1])
            intersections.append(x_cross)
    return intersections

# Setup for the plot
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.2), sharey=True)
axs = axs.flatten()

# Create custom legend handles and labels for the linestyles
custom_lines = [Line2D([0], [0], color='black', linestyle=linestyle_mapping[key]) for key in linestyle_mapping]
custom_labels = list(linestyle_mapping.keys())
region_colors = [plt.rcParams['axes.prop_cycle'].by_key()['color'][i] for i in range(len(regions))]

perc_diff_out = pd.DataFrame(columns=["Model", "Region", "Clean tech", "Fossil tech"] + list(years))

for mi, model in enumerate(models):
    percentage_difference = get_price_differences_percentage(model, years, regions)      
    ax = axs[mi]  
    
    # Track the linestyles used in this subplot
    used_linestyles = {}
    
    for ri, r in enumerate(regions):
        
        # We don't have very strong cost data for all regions. Skipping some. 
        if skip_model_region_combo(model, regions[r]):
            continue
        
        pr_diff = percentage_difference[ri]
        
        # Generate the output_str for the current comparison
        clean_tech = df_cy.loc[df_cy['Region'] == r, 'Clean tech name'].values[mi]
        fossil_tech = df_cy.loc[df_cy['Region'] == r, 'Fossil tech name'].values[mi]
        clean_vs_fossil_str = comparison_str(clean_tech, fossil_tech)
        
        perc_diff_out.loc[len(perc_diff_out)] = [model, r, clean_tech, fossil_tech] + list(pr_diff)

        # Get the linestyle for the current comparison, and save it
        linestyle = linestyle_mapping.get(clean_vs_fossil_str, '-')
        used_linestyles[clean_vs_fossil_str] = linestyle
        
        # Plot the price difference
        line, = ax.plot(years, pr_diff, label=r, linestyle=linestyle, c=region_colors[ri])
        
        # Find intersections with y=0
        intersections = find_intersections(years, pr_diff)

        # Plot markers at intersections with the same color as the line
        for x_cross in intersections:
            ax.plot(x_cross, 0, 'o', color=line.get_color(), markersize=4)  # Use the line color for the markers
    
    
    ax.axhline(0, color='grey', linewidth=1)  # Adding horizontal line at y=0
    ax.set_title(f"{repl_dict[model]}")
    
    if mi % 2 == 0:  # Add y-label only to the leftmost subplots
        ax.set_ylabel("Levelised costs difference (%)")

    # Set xlim between 2025 and 2050
    ax.set_xlim(2025, 2050)   
    ax.set_ylim(-100, 100)
    
    
    # Apply green and grey shading from 0 to ylimits
    ymin, ymax = ax.get_ylim() 
    ax.fill_between(years, ymin, 0, where=(np.array(years) >= 2025), facecolor='lightgreen', alpha=0.35)
    ax.fill_between(years, 0, ymax, where=(np.array(years) >= 2025), facecolor='lightgrey', alpha=0.35)
    
    # Reset the y-limits after shading
    ax.set_ylim(ymin, ymax)
    
    # Create custom legend handles and labels for the linestyles used in this subplot
    custom_lines = [Line2D([0], [0], color='black', linestyle=used_linestyles[key]) for key in used_linestyles]
    custom_labels = list(used_linestyles.keys())
    
    # Add the custom legend to the subplot
    ax.legend(custom_lines, custom_labels, loc='upper right',
              title="Leading technologies", title_fontproperties=FontProperties(weight='bold'))

# Extract handles and labels from the first subplot
handles, labels = axs[0].get_legend_handles_labels()

# Create custom legend handles and labels for the colors with continuous line style
color_patches = [Line2D([0], [0], color=r_color, linestyle='-') for r_color in region_colors]
color_labels = regions.keys()

# Add the color legend to the figure
fig.legend(color_patches, color_labels, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=5)

plt.tight_layout()

# Save the graph as an editable svg file
save_fig(fig, fig_dir, "Figure 2 - Baseline_price_differences")
save_data(perc_diff_out, fig_dir, "Figure 2 - Baseline_price_difference")




#%% =========================================================================
# Table: 5x4 table with difference in crossover year
# ============================================================================

clean_tech_variable = {"FTT:P": 18, "FTT:Tr": 19, "FTT:H": 10, "FTT:Fr": 33}
fossil_tech_variable = {"FTT:P": 2, "FTT:Tr": 1, "FTT:H": 2, "FTT:Fr": 13}       # Note 4 for transport gives an error

output_ppolicies = get_output(output_file, "sxp - P mand")
output_hpolicies = get_output(output_file, "sxp - H mand")
output_trpolicies = get_output(output_file, "sxp - Tr mand")
output_frpolicies = get_output(output_file, "sxp - Fr mand")
output_all_mandates = get_output(output_file, "Mandates")

#output_all_policies = get_output(output_file, "sxp - All policies")

output_files = [output_S0, output_ppolicies, output_hpolicies, output_trpolicies, output_frpolicies, output_all_mandates]
policy_names = ["Baseline", "Coal phase-out", "Heat pump mandates", "EV mandates", "EV truck mandates", "All mandates"]


def convert_fractional_years_to_years_and_months(fractional_year):
    ''' Extract the integer part (year) and the fractional part (month) '''
    year = int(fractional_year)
    fraction = fractional_year - year

    # Convert the fractional part to months
    months = round(fraction * 12)

    # If months is 12, increment the year and reset months
    if months == -12:
        year -= 1
        months = 0
       
    return -year, -months


def get_weighted_costs(output, model, tech_variable, year_inds):
    """Get the weighted cost based on the scenario (output), model,
    tech_variable and the indices of the years of interest.
    """
    
    if model == "FTT:P" and tech_variable in [2, 6, [2], [6]]:
        prices = output[operation_cost_name[model]][:, tech_variable, 0, year_inds]
    else:
        prices = output[price_names[model]][:, tech_variable, 0, year_inds]
    
    # Way by total size of the market per region
    weights = np.sum(output[shares_vars[model]][:, :, 0, year_inds], axis=1)
    weighted_prices = np.average(prices, weights=weights, axis=0)
        
    return weighted_prices



def format_monthyear_str(years, months):
    '''Is it plural or singular month/year?'''
    
    if years == 0 and months != 1 and months != -1:
        yearmonth_str = f'{months} months'
    elif years == 0 and (months == 1 or months == -1):
        yearmonth_str = f'{months} month'
    elif years == 1 and months == 1:
        yearmonth_str = f'{years} year, {months} month'
    elif years == 1 and months != 0:
        yearmonth_str = f'{years} year, {months} months'
    elif years == 1 and months == 0:
        yearmonth_str = f'{years} year'
    elif years != 0 and months == 0:
        yearmonth_str = f'{years} years'
    else:
        yearmonth_str = f'{years} years, {months} months'
    return yearmonth_str



def compute_average_crossover_diff(df_crossovers, policy_name, model):
    """ Compute the average cross-over year diff, only for those regions where 
    there is a finite saving."""
        
    cy_arrays = []
    for index, row in df_crossovers.iterrows():
        cy_arrays.append(np.array(list(row["Crossover years"].values())))    
    
    # Create a boolean mask for each array where values are not inf or nan
    masks = [np.isfinite(arr) for arr in cy_arrays]
    
    # Combine masks to get indices where all arrays are finite
    combined_mask = np.all(masks, axis=0)
    valid_indices = list(np.where(combined_mask)[0])
    print(f"There is a crossover in {len(valid_indices)} regions in {model}")
    
    cy_S0 = cy_arrays[0]
    cy_rows = []
    
    for pi, policy in enumerate(policy_name[1:]):
        cy_diff = [cy_arrays[pi+1][ind] - cy_S0[ind] for ind in valid_indices]
        weights = df_crossovers["Weights"].iloc[0][valid_indices]
        averaged_cy = np.average(cy_diff, weights=weights)
        
        years, months = convert_fractional_years_to_years_and_months(averaged_cy)
        yearmonth_str = format_monthyear_str(years, months)
        row = {
            "Model": model,
            "Policy": policy,
            "Crossover year diff": yearmonth_str}
        cy_rows.append(row)
    
    return cy_rows

def get_weights(output, model, cap_vars):
    weights = np.sum(output[cap_vars[model]][:, :, 0, 20], axis=1)
    return weights

crossover_list = []
crossover_diff_list = []

# Compute it first by region, and average cross-over years, or first average prices
average_prices_first = False

if average_prices_first == False:

    for model in models:
    
        
        clean_tech_dict = {i: clean_tech_variable[model] for i in range(1, 72)}
        fossil_tech_dict = {i: fossil_tech_variable[model] for i in range(1, 72)}
        
        
        for policy, output in zip(policy_names, output_files):    
            crossover_years = get_crossover_year(output, model,
                                    clean_tech_dict, fossil_tech_dict, price_names, regions_all)
            weights = get_weights(output, model, cap_vars)
            row = {
                "Model": model,
                "Policy": policy,
                "Crossover years": crossover_years, 
                "Weights": weights}
            crossover_list.append(row)
    
        df_crossovers = pd.DataFrame(crossover_list, 
                          columns=["Model", "Policy", "Crossover years", "Weights"])
        
    for model in models:
         average_crossover_rows = compute_average_crossover_diff(
                         df_crossovers[df_crossovers["Model"]==model], policy_names, model)
         crossover_diff_list.extend(average_crossover_rows)
     
    df = pd.DataFrame(crossover_diff_list)
        

elif average_prices_first == True: 
    
    crossover_diff_list = []
    clean_tech_variable = {"FTT:P": [18], "FTT:Tr": [19], "FTT:H": [10], "FTT:Fr": [33]}
    fossil_tech_variable = {"FTT:P": [2], "FTT:Tr": [1], "FTT:H": [3], "FTT:Fr": [13]} # Note 4 for transport gives an error
    year_inds = list(range(10, 41))
    
    for model in models:
        
        fossil_costs_S0 = get_weighted_costs(output_S0, model, fossil_tech_variable[model], year_inds)
        clean_costs_S0 = get_weighted_costs(output_S0, model, clean_tech_variable[model], year_inds)
        year_S0 = interpolate_crossover_year(clean_costs_S0, fossil_costs_S0)
        if model == "FTT:H":
            print(fossil_costs_S0)
            print(clean_costs_S0)
        
        for policy, output in zip(policy_names[1:], output_files[1:]):  
            fossil_costs = get_weighted_costs(output, model, fossil_tech_variable[model], year_inds)
            clean_costs = get_weighted_costs(output, model, clean_tech_variable[model], year_inds)
            year = interpolate_crossover_year(clean_costs, fossil_costs)
            try:
                years, months = convert_fractional_years_to_years_and_months(year-year_S0)
            except ValueError:
                print(f"model is {model}")
                print(f"policy is {policy}")
                print(f"year_S0: {year_S0}")
                print(f'year: {year}')
            yearmonth_str = format_monthyear_str(years, months)
            row = {
                "Model": model,
                "Policy": policy,
                "Crossover year diff": yearmonth_str}
            crossover_diff_list.extend(row)
            print(row)
    df = pd.DataFrame(crossover_diff_list)
          

# Pivot the DataFrame to get a 5x4 table
table = df.pivot(index='Policy', columns='Model', values='Crossover year diff')
# Reindex the table to ensure the policy order matches policy_names
table = table.reindex(index=policy_names[1:], columns=models)

# Plot the table as a figure
fig, ax = plt.subplots(figsize=(6, 2), dpi=300)
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=table.values, rowLabels=table.index,
                 colLabels=["Power sector", "Heating", "Cars", "Trucks"],
                 cellLoc='center', loc='center')

table.auto_set_font_size(False)
#table.set_fontsize(10)
table.scale(1.2, 1.2)

header_color = '#40466e'
header_text_color = 'w'
row_colors = ['#f2f2f2', 'w']

# Formatting of the table
for (i, j), cell in table.get_celld().items():
    if j == -1:
        cell.set_text_props(fontweight='bold', color='black')  # Row labels bold
    if i == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text_color, fontweight='bold')  # Column labels white text and bold
    if i > 0 and j > -1:
        cell.set_facecolor(row_colors[i % 2])  # Alternating row colors


plt.subplots_adjust(top=1, bottom=0)  # Adjust whitespace
plt.title("How much is cost-parity brought forward?", fontweight='bold')


#%% ==============================================================================
#             Figure 3: Same as table above, but now in box plot. 
# ===============================================================================

fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.5), sharey=True)
axs = axs.flatten()
#seaborn.set(style = 'whitegrid') 

# Expand the dataframe to make it suitable for the boxplot
# Initialize an empty DataFrame to collect the results
expanded_df = pd.DataFrame()

# Iterate through each row and expand the dictionary
for idx, row in df_crossovers.iterrows():
    
    temp_df = pd.DataFrame(list(row['Crossover years'].items()), columns=['Region', 'Crossover year'])
    # Add the other columns to the temporary DataFrame
    temp_df['Model'] = row['Model']
    temp_df['Policy'] = row['Policy']
    # Append to the result DataFrame
    expanded_df = pd.concat([expanded_df, temp_df], ignore_index=True)

# Rearrange columns to the desired order
expanded_df = expanded_df[['Model', 'Policy', 'Region', 'Crossover year']]

def subtract_baseline(filtered_df_model):
    '''Subtract the baseline crossover year'''
    # Step 1: Filter baseline data
    baseline_df = filtered_df_model[filtered_df_model['Policy'] == 'Baseline'][['Region', 'Crossover year']]
    
    # Step 2: Merge with the original DataFrame on 'Region'
    merged_df = filtered_df_model.merge(baseline_df, on='Region', suffixes=('', '_Baseline'))
    
    # Step 3: Calculate the difference from baseline
    merged_df['Crossover year difference'] = -(merged_df['Crossover year'] - merged_df['Crossover year_Baseline'])
    
    # Step 4: Filter out the baseline policy rows and drop the baseline crossover year column
    new_df = merged_df[merged_df['Policy'] != 'Baseline'].copy()
    
    return new_df

    
def all_finite(group):
    return np.isfinite(group["Crossover year"]).all()
   
for mi, model in enumerate(models):
    
    ax = axs[mi] 
    df_model = expanded_df[expanded_df["Model"]==model]
    
    # Group by "Region" and filter out groups with any non-finite "Crossover year" values
    filtered_df_model = df_model.groupby("Region").filter(all_finite)  
    df_difference = subtract_baseline(filtered_df_model)
    
    seaborn.boxplot(y ='Policy', x ='Crossover year difference', data = df_difference, ax=ax)
    # Remove the word policy from the y-axis
    ax.set_ylabel('')
    
    ax.set_title(repl_dict[model])
    ax.set_xlim(-2, 12)   
    #ax.tick_params(axis='x', rotation=45)   # Rotate x-axis labels
    ax.set_xlabel('')                       # Remove x-axis label
    
    # Remove the top and right frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set vertical grid lines
    ax.xaxis.grid(True)
    
    save_data(df_difference, fig_dir, f"Figure 6- Boxplot crossover years {repl_dict[model]}")

    
# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.4, wspace=0.2)    
save_fig(fig, fig_dir, "Figure 6 - Boxplot crossover years")

#%% =========================================================================
#  Fourth figure: timelines by sector by country
# ============================================================================

def determine_lims_crossover(row):
    xmin = np.min(row)
    xmax = np.max(row)
    
    # If there is a inf, but not all are inf
    if (row == np.inf).any() and not (row == np.inf).all():
        xmax = 2050
    
    return xmin, xmax
 
# Define lighter colours for better distinction
colors = {
    'Cross-over': '#003f7f',  # Dark blue
    'Cross-over carbon tax': '#547bb5',  # Medium-dark blue
    'Cross-over subsidies': '#7a5195',  # Purple
    'Cross-over mandates': '#bc5090',  # Pink-purple
}

policy_names = {
    'Cross-over': 'Current traject.',
    'Cross-over carbon tax': 'Carbon tax',
    'Cross-over subsidies': 'Subsidies',
    'Cross-over mandates': 'Mandates',
    }   
    

# Plot the timelines
y_base = 0
y_gap = 1.0
model_gap = -5
offset_op = 0.3


# Define the timeline range
timeline_start = 2020
timeline_end = 2050

# Create the plot
fig, axs = plt.subplots(nrows=len(models), figsize=(8, 8), sharex=True)
axs = axs.flatten()



# Go over models in reverse order
for mi, model in enumerate(models):
    ax = axs[mi]
    model_data = df_cy[df_cy['Sector'] == model]
    
    yticks = []
    yticklabels = []
    
    
    for index, row in model_data.iterrows():
        y_position = len(df) - index     
        
        # Write down what we're comparing
        ax.text(2051, y_position - 0.25, comparison_str(row["Clean tech name"], row["Fossil tech name"]))

        
        # Plot the lines connecting the crossover years
        xmin, xmax = determine_lims_crossover(row.iloc[-4:])
        if xmin is not None and xmax is not None:
            ax.hlines(y_position, xmin, xmax, color='black', alpha=1.0)
       
                
        # Plot the crossover year (points)
        for pi, policy in enumerate(colors.keys()):
            if row[policy] < 2021:
                pass 
            ax.plot(row[policy], y_position, 'o', color=colors[policy], markersize=10)
                    
        # Plot arrows when the crossover point in past of after 2050 (-inf or inf):
        if (row.iloc[-4:] == np.inf).all():
            ax.arrow(2049, y_position, 0.8, 0, head_width=0.3, head_length=0.3, fc='#003f5c', ec='#003f5c')
        elif (row.iloc[-4:] == -np.inf).all():
            ax.arrow(2021, y_position, -0.8, 0, head_width=0.3, head_length=0.3, fc='#003f5c', ec='#003f5c')                
       
        
        # Set the region label
        
        yticks.append(y_position)
        yticklabels.append(row['Region'])
    
    ax.text(2046, max(yticks)+0.5, model_names_r[3-mi], weight='bold')
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Remove ticks and move tick labels
    ax.tick_params(axis='y', length=0, pad=15)
    ax.tick_params(axis='x', length=0, pad=10)
            
    ax.set_xlim(2019.95, 2050.1)
    
    # Disable default vertical grid lines
    ax.grid(False, axis='x')
    # Draw custom vertical grid lines within y-tick range
    for x in np.arange(2020, 2051, 5):  # Example for every 5 years
        ax.plot([x, x], [min(yticks), max(yticks)], color='gray', linestyle='--', linewidth=0.5)
        
    # Hide excess grid lines by setting major ticks only at yticks positions
    ax.yaxis.set_major_locator(plt.FixedLocator(yticks))
        
    
# Create custom legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[policy],
                      markersize=12, label=policy_names[policy]) for policy in colors]
ax.legend(handles=handles, loc='upper right',  ncol=4, 
                          bbox_to_anchor=(0.9, -0.2), framealpha=0.8)


fig.suptitle("Cost-parity point: costs without policy \n Comparison largest clean and fossil technology in each country", 
             x=0.45, y=0.95, ha='center')

# Save the graph as an png/svg files
save_fig(fig, fig_dir, "Horizontal_timeline_crossover_year")
df_cy['Sector'] = df_cy['Sector'].replace(repl_dict)
save_data(df_cy, fig_dir, "Figure X - Horizontal_timeline_crossover_year")
