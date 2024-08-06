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

from preprocessing import get_output, get_metadata

# Set global font size
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'xtick.labelsize': 13, 'ytick.labelsize': 13})

output_file = "Results_sxp.pickle"
output_S0 = get_output(output_file, "S0")


titles, fig_dir, tech_titles, models = get_metadata()

# Define the regions and the region numbers of interest
regions = {'India': 41, "China": 40, "Brazil": 43, "United States": 33, "Germany": 2, "UK": 14}

# Define the clean technology list by model
clean_techs = {"FTT:P": [16, 18], "FTT:Tr": [18, 19, 20], "FTT:H": [10, 11], "FTT:Fr": [12]}
dirty_techs = {"FTT:P": [0, 2, 6], "FTT:Tr": list(range(12)), "FTT:H": [2, 3], "FTT:Fr": [0, 2, 4, 6, 8]}


# Define the shares, prices of interest
model_names_r = ["Trucks", "Cars", "Heating", "Power"]
price_names = {"FTT:P": "MEWC", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
shares_names = {"FTT:P": "MEWS", "FTT:Tr": "TEWS", "FTT:H": "HEWS", "FTT:Fr": "ZEWS"}
operation_cost_name = {"FTT:P": "MLCO"}
# TODO: should carbon tax be part of this? Probably not, right?

# Define the year of interest
year = 2030



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
        
        

def get_prices(output, year, model, biggest_technologies, regions):
    """Get the prices of the biggest technologies."""
    price_var = price_names[model]
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
    """Interpolate based on price difference in cross-over year and previous year
    Returns -inf if cost-parity in past, inf if cost-parity not reached before 2050."""
    
    # First check if cost-parity occurs
    # Set crossover year to -inf if cost-parity already achieved and inf if it will not.
    if (price_series_clean <= price_series_fossil).all():
        return float('-inf')
    elif (price_series_clean > price_series_fossil).all():
        return float('inf')
    
    # Then, if we start with cost-parity, but don't have it consistently, also return -inf
    if price_series_clean[0] <= price_series_fossil[0]:
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

def get_crossover_year(output, model, biggest_techs_clean, biggest_techs_fossil, price_names, regions):
    """ Get the year when the clean technology becomes cheaper than the fossil technology."""
    crossover_years = {}
    for r, ri in regions.items():
        
        tech_clean = biggest_techs_clean[r]
        tech_fossil = biggest_techs_fossil[r]
        
        price_series_clean = output[price_names[model]][ri, tech_clean, 0, 10:]
        price_series_fossil = output[price_names[model]][ri, tech_fossil, 0, 10:]
        crossover_years[r] = interpolate_crossover_year(price_series_clean, price_series_fossil)
    
    return crossover_years



rows = []
for model in models:
    # Get the bit of the model name after the colon (like Fr)
    model_abb = model.split(':')[1]
    output_ct = get_output(output_file, f"sxp - {model_abb} CT")
    output_sub = get_output(output_file, f"sxp - {model_abb} subs")
    output_man = get_output(output_file, f"sxp - {model_abb} mand")
    
    biggest_techs_clean = find_biggest_tech(output_S0, clean_techs, year, model, regions)
    biggest_techs_fossil = find_biggest_tech_dirty(output_S0, dirty_techs, biggest_techs_clean, year, model)
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
df = pd.DataFrame(rows, columns=["Region", "Sector",
                                 "Clean technology", "Clean tech name", "Clean price (2030)", 
                                 "Fossil technology", "Fossil tech name", "Fossil price (2030)", 
                                 "Cross-over", "Cross-over carbon tax",
                                 "Cross-over subsidies", "Cross-over mandates"])

# %% Compute how much, globally, we're brought forward. 

clean_tech_variable = {"FTT:P": 18, "FTT:Tr": 19, "FTT:H": 10, "FTT:Fr": 12}
fossil_tech_variable = {"FTT:P": 6, "FTT:Tr": 1, "FTT:H": 2, "FTT:Fr": 4}       # Note 4 for transport gives an error

output_file_sectors = "Results_sectors.pickle"


output_S0 = get_output(output_file_sectors, "S0")
output_ppolicies = get_output(output_file_sectors, "FTT-P")
output_hpolicies = get_output(output_file_sectors, "FTT-H")
output_trpolicies = get_output(output_file_sectors, "FTT-Tr")
output_frpolicies = get_output(output_file_sectors, "FTT-Fr")

output_files = [output_S0, output_ppolicies, output_hpolicies, output_trpolicies, output_frpolicies]
policy_names = ["Baseline", "Power policies", "Heat policies", "Transport policies", "Freight policies"]

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
    
    print(f"There is a crossover in {len(valid_indices)} number of regions in {model}")
    
    cy_S0 = cy_arrays[0]
    cy_rows = []
    
    for pi, policy in enumerate(policy_name[1:]):
        averaged_cy = np.average([cy_arrays[pi+1][ind] - cy_S0[ind] for ind in valid_indices])
        row = {
            "Model": model,
            "Policy": policy,
            "Crossover years": averaged_cy}
        cy_rows.append(row)
    
    return cy_rows

crossover_list = []
crossover_diff_list = []

for model in models:
    
    clean_tech_dict = {i: clean_tech_variable[model] for i in range(1, 72)}
    fossil_tech_dict = {i: fossil_tech_variable[model] for i in range(1, 72)}
    
    regions = {i: i - 1 for i in range(1, 72)}
    
    for policy, output in zip(policy_names, output_files):    
    # CrossoverYear_XPolicy
        crossover_years = get_crossover_year(output, model, clean_tech_dict, fossil_tech_dict, price_names, regions)
        row = {
            "Model": model,
            "Policy": policy,
            "Crossover years": crossover_years}
        crossover_list.append(row)

df_crossovers = pd.DataFrame(crossover_list, columns = ["Model", "Policy", "Crossover years"])

for model in models:
    average_crossover_rows = compute_average_crossover_diff(df_crossovers[df_crossovers["Model"]==model], policy_names, model)
    



#%% Making a graph with four subplots for each sector.
# Each graph shows the percentage difference between the clean and fossil technology in 2025, 2035 and 2050. 
# The x-axis is the year and the y-axis the percentage difference.
# The title is the sector name. Each region has its own line.


# Define the years of interest
years = [2025, 2035, 2050]

# Define the percentage difference function
def get_percentage_difference(clean_price, dirty_price):
    return 100 * (clean_price - dirty_price) / dirty_price

# Define the data for the plot
def compute_percentage_difference(model, years):
    """Compute percentage price difference per year per region in top techs."""
    
    biggest_techs_clean = find_biggest_tech(output_S0, clean_techs, 2030, model, regions)
    biggest_techs_fossil = find_biggest_tech_dirty(output_S0, dirty_techs, biggest_techs_clean, 2030, model)
    percentage_difference = np.zeros((len(regions), len(years)))
    for ri, r in enumerate(regions):
    
        for yi, year in enumerate(years):
            clean_prices = get_prices(output_S0, year, model, biggest_techs_clean, regions)
            fossil_prices = get_prices(output_S0, year, model, biggest_techs_fossil, regions)
               
            percentage_difference[ri, yi] = get_percentage_difference(clean_prices[r], fossil_prices[r])
    
    return percentage_difference


#%% Plot the figure of levelised cost difference fossil vs clean tech
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
   
    # Remove the top and right frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
            
# Extract handles and labels from the first subplot
handles, labels = axs[0].get_legend_handles_labels()

# Add a single legend for the entire figure, positioned beneath the graph
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3)

plt.tight_layout()
plt.tight_layout()

# Save the graph as an editable svg file
output_file = os.path.join(fig_dir, "Figure1_baseline_price_differences.svg")
output_file2 = os.path.join(fig_dir, "Figure1_baseline_price_differences.png")

fig.savefig(output_file, format="svg", bbox_inches='tight')
fig.savefig(output_file2, format="png", bbox_inches='tight')



#%% Cross-over year by policy

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
y_gap = 1.08
model_gap = -5
offset_op = 0.3


# Define the timeline range
timeline_start = 2020
timeline_end = 2050

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))

yticks = []
yticklabels = []

# Go over models in reverse order
for mi, model in enumerate(models[::-1]):
    
    ax.text(2045, mi * 7.9 + 7, model_names_r[mi], fontsize=13)
    model_data = df[df['Sector'] == model]
    for index, row in model_data.iterrows():
        y_position = y_base + (len(df)-index) * y_gap
        ax.hlines(y_position, timeline_start, timeline_end, color='grey', alpha=0.5)
        
        
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
            ax.arrow(2049.3, y_position, 1, 0, head_width=0.3, head_length=0.2, fc='#003f5c', ec='#003f5c')
        elif (row.iloc[-4:] == -np.inf).all():
            ax.arrow(2020.7, y_position, -1, 0, head_width=0.3, head_length=0.2, fc='#003f5c', ec='#003f5c')                
       
        yticks.append(y_position)
        yticklabels.append(row['Region'])
    y_base += len(model_data) * y_gap + model_gap

# Set the region label
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

# Remove frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Remove ticks and move tick labels
ax.tick_params(axis='y', length=0, pad=-5)
ax.tick_params(axis='x', length=0, pad=-10)


# Create custom legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[policy],
                      markersize=12, label=policy_names[policy]) for policy in colors]
legend_y_position = 10
ax.legend(handles=handles, loc='upper right', frameon=True, ncol=4, bbox_to_anchor=(1.0, -0.03), framealpha=0.8)


fig.suptitle("When does the clean technology becomes the lowest costs, when excluding policy costs \n Comparison largest clean and fossil technology in each country", 
             x=0.45, y=0.95, ha='center')

# Save the graph as an editable svg file
output_file = os.path.join(fig_dir, "Horizontal_timeline_crossover_year.png")
output_file2 = os.path.join(fig_dir, "Horizontal_timeline_crossover_year.svg")
fig.savefig(output_file, format="png", bbox_inches='tight')
fig.savefig(output_file2, format="svg", bbox_inches='tight')



