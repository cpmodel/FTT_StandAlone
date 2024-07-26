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
from matplotlib.lines import Line2D

from preprocessing import get_output, get_metadata

# Set global font size
plt.rcParams.update({'font.size': 14})
# Set global font size
plt.rcParams.update({'xtick.labelsize': 12, 'ytick.labelsize': 12})

output_file = "Results_policies.pickle"
output = get_output("Results_scens.pickle", "S0")
output_ct = get_output(output_file, "Carbon tax")
output_sub = get_output(output_file, "Subsidies")
output_man = get_output(output_file, "Mandates")

titles, fig_dir, tech_titles, models = get_metadata()

# Define the regions and the region numbers of interest
regions = {'India': 41, "China": 40, "Brazil": 43, "United States": 33, "UK": 14, "Germany": 2}

# Define the clean technology list by model
clean_techs = {"FTT:P": [16, 18], "FTT:Tr": [18, 19, 20], "FTT:H": [10, 11], "FTT:Fr": [12]}
dirty_techs = {"FTT:P": [0, 2, 6], "FTT:Tr": list(range(12)), "FTT:H": [2, 3], "FTT:Fr": [0, 2, 4, 6, 8]}

# Define the year of interest
year = 2030

# Define the shares, prices of interest
model_names_r = ["Trucks", "Cars", "Heating", "Power"]
price_names = {"FTT:P": "MEWC", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
shares_names = {"FTT:P": "MEWS", "FTT:Tr": "TEWS", "FTT:H": "HEWS", "FTT:Fr": "ZEWS"}
operation_cost_name = {"FTT:P": "MLCO"}
# TODO: should carbon tax be part of this? Probably not, right?


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

def interpolate_crossover_year(price_series_clean, price_series_fossil):
    # Return None if the prices are under the threshold throughout the timeseries
    
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
    """ Get the year when the clean technology becomes cheaper than the dirty technology."""
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
    crossover_years_ct = get_crossover_year(output_ct, model, biggest_techs_clean, biggest_techs_fossil, price_names)
    crossover_years_sub = get_crossover_year(output_sub, model, biggest_techs_clean, biggest_techs_fossil, price_names)
    crossover_years_man = get_crossover_year(output_man, model, biggest_techs_clean, biggest_techs_fossil, price_names)

    crossover_years_op = get_crossover_operational_vs_new(output, model, biggest_techs_clean, biggest_techs_fossil, price_names, operation_cost_name)
    crossover_years_op_ct = get_crossover_operational_vs_new(output_ct, model, biggest_techs_clean, biggest_techs_fossil, price_names, operation_cost_name)
    crossover_years_op_sub = get_crossover_operational_vs_new(output_sub, model, biggest_techs_clean, biggest_techs_fossil, price_names, operation_cost_name)
    crossover_years_op_man = get_crossover_operational_vs_new(output_man, model, biggest_techs_clean, biggest_techs_fossil, price_names, operation_cost_name)

    
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
        "Cross-over operat.": crossover_years_op[r],
        "Cross-over operat. carbon tax": crossover_years_op_ct[r],
        "Cross-over operat. subsidies": crossover_years_op_sub[r],
        "Cross-over operat. mandates": crossover_years_op_man[r]
        }
        rows.append(row)

# Construct the DataFrame from the list of dictionaries
df = pd.DataFrame(rows, columns=["Region", "Sector",
                                 "Clean technology", "Clean tech name", "Clean price (2030)", 
                                 "Fossil technology", "Fossil tech name", "Fossil price (2030)", 
                                 "Cross-over", "Cross-over carbon tax",
                                 "Cross-over subsidies", "Cross-over mandates",
                                 "Cross-over operat.", "Cross-over operat. carbon tax",
                                 "Cross-over operat. subsidies", "Cross-over operat. mandates"])


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
    
    if row.isna().any() and not row.isna().all():
        xmax = 2050
    
    return xmin, xmax
    
    

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the timelines
y_base = 0
y_gap = 1
model_gap = -5
offset_op = 0.3



# Define lighter colours for better distinction
colors = {
    'Cross-over': '#005f99',  # Lighter blue
    'Cross-over carbon tax': '#3f5b8c',  # Medium-dark blue
    'Cross-over subsidies': '#7a5195',  # Lighter medium blue
    'Cross-over mandates': '#bc5090',  # Light blue
    'Cross-over operat.': '#009688',  # Lighter dark green
    'Cross-over operat. carbon tax': '#2e7d32',  # Medium-dark green
    'Cross-over operat. subsidies': '#66bb6a',  # Medium green
    'Cross-over operat. mandates': '#a5d6a7'  # Light green
}

policy_names = {
    'Cross-over': 'Current traject.',
    'Cross-over carbon tax': 'Carbon tax',
    'Cross-over subsidies': 'Subsidies',
    'Cross-over mandates': 'Mandates',
    'Cross-over operat.': 'Operational costs',
    'Cross-over operat. carbon tax': 'Operational costs',
    'Cross-over operat. subsidies': 'Operational costs',
    'Cross-over operat. mandates': 'Operational costs'
    }

# Define the timeline range
timeline_start = 2020
timeline_end = 2050

# # Plot the timelines
# for index, row in df.iterrows():
#     y_position = index  # Adjust the vertical position for each region
#     ax.hlines(y_position, timeline_start, timeline_end, color='grey', alpha=0.5)
#     for policy in colors.keys():
#         if row["Sector"] == "FTT:P":
#             ax.plot(row[policy], y_position, 'o', color=colors[policy], label=row['Region'])  # Plot the dots
#         else:
#             ax.plot(row[policy], y_position, 'o', color=colors[policy], label=row['Region'])  # Plot the dots

yticks = []
yticklabels = []

for model in models[::-1]:
    model_data = df[df['Sector'] == model]
    for index, row in model_data.iterrows():
        y_position = y_base + (len(df)-index) * y_gap
        ax.hlines(y_position, timeline_start, timeline_end, color='grey', alpha=0.5)
        
        # Plot the lines connecting the crossover years
        xmin, xmax = determine_lims_crossover(row.iloc[-8:-4])
        if xmin is not None and xmax is not None:
            ax.hlines(y_position, xmin, xmax, color='black', alpha=1.0)
        if row["Sector"] == "FTT:P":
            xmin, xmax = determine_lims_crossover(row.iloc[-4:])
            if xmin is not None and xmax is not None:
                ax.hlines(y_position, xmin, xmax, color='black', alpha=1.0)
                
        # Plot the crossover year (points)
        for pi, policy in enumerate(colors.keys()):
            if row[policy] <= 2021:
                pass
            elif row["Sector"] == "FTT:P" and pi >= 4:
                ax.plot(row[policy], y_position, 'o', color=colors[policy])
            elif pi < 4:
                ax.plot(row[policy], y_position, 'o', color=colors[policy])
        
        
                
        # Plot arrows when the crossover poinnt in past of after 2050:
        if row.iloc[-8:-4].isna().all():
            ax.arrow(2049.3, y_position, 1, 0, head_width=0.2, head_length=0.2, fc='#003f5c', ec='#003f5c')
        elif (row.iloc[-8:-4] < 2021.1).all():
            ax.arrow(2020.7, y_position, -1, 0, head_width=0.2, head_length=0.2, fc='#003f5c', ec='#003f5c')                
       
        yticks.append(y_position)
        yticklabels.append(row['Region'])
    y_base += len(model_data) * y_gap + model_gap


# Remove frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(axis='y', length=0, pad=-5)
ax.tick_params(axis='x', length=0, pad=-10)

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

# # Create custom legend
# handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[policy],
#                       markersize=10, label=policy_names[policy]) for policy in colors]
# legend_y_position = 10
# ax.legend(handles=handles, loc='upper right', frameon=True, ncol=4, bbox_to_anchor=(1.0, -0.03), framealpha=0.8)





# Define the policy legend elements
policy_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Cross-over'], markersize=10, label='Current traject.'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Cross-over carbon tax'], markersize=10, label='Carbon tax'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Cross-over subsidies'], markersize=10, label='Subsidies'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Cross-over mandates'], markersize=10, label='Mandates')
]

# Define the comparison legend elements
comparison_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#005f99', markersize=10, label='New green tech vs new fossil', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#009688', markersize=10, label='New green tech vs existing fossil', linestyle='None')
]

# Create custom handles for comparisons
custom_lines = [
    Line2D([0], [0], color=colors['Cross-over'], lw=4),
    Line2D([0], [0], color=colors['Cross-over carbon tax'], lw=4),
    Line2D([0], [0], color=colors['Cross-over subsidies'], lw=4),
    Line2D([0], [0], color=colors['Cross-over mandates'], lw=4),
    Line2D([0], [0], color=colors['Cross-over operat.'], lw=4),
    Line2D([0], [0], color=colors['Cross-over operat. carbon tax'], lw=4),
    Line2D([0], [0], color=colors['Cross-over operat. subsidies'], lw=4),
    Line2D([0], [0], color=colors['Cross-over operat. mandates'], lw=4)
]

# Place the policy legend
first_legend = ax.legend(handles=policy_legend_elements, loc='upper left', frameon=True, ncol=4, bbox_to_anchor=(-0.1, -0.05), framealpha=0.8)
ax.add_artist(first_legend)

# Place the comparison legend below the first one
ax.legend(handles=comparison_legend_elements, loc='upper left', frameon=True, ncol=2, bbox_to_anchor=(-0.1, -0.10), framealpha=0.8)

# Add the custom lines for comparison under the correct labels
for line in custom_lines:
    ax.add_line(line)




# Add secondary y-axis for models
secax = ax.secondary_yaxis('left')
secax.set_yticks([2.5 + i*7 for i, model in enumerate(models[::-1])])
secax.set_yticklabels(model_names_r, rotation=90, va='center', ha='center')
secax.tick_params(length=0, pad=100)
secax.spines['left'].set_visible(False)

fig.suptitle("Crossover year by policy \n Comparison largest clean and fossil technology in each country", 
             x=0.45, y=0.95, ha='center')

# Save the graph as an editable svg file
output_file = os.path.join(fig_dir, "Horizontal_timeline_crossover_year.png")
# fig.savefig(output_file, format="svg", bbox_inches='tight')
fig.savefig(output_file, format="png", bbox_inches='tight')


