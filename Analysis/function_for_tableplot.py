# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:25:57 2024

@author: Rishi
"""

# Import the results pickle file
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from preprocessing import get_output, get_metadata

current_dir = os.path.dirname(os.path.realpath(__file__))
# The input dir is found by going up a directly to the parent directory and then going to the Input folder
top_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
input_dir = os.path.join(top_dir, "Inputs")

# Set global font size
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'xtick.labelsize': 13, 'ytick.labelsize': 13})

output_file = "Results.pickle"
output = get_output(output_file, "S0")

titles, fig_dir, tech_titles, models = get_metadata()

# Define the regions and the region numbers of interest
#regions = {'India': 41, "China": 40, "Brazil": 43, "United States": 33, "UK": 14, "Germany": 2}

all_regions = pd.read_csv(os.path.join(current_dir, "e3me_regions_jan23.csv"))

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
         }
        rows.append(row)


# Construct the DataFrame from the list of dictionaries
df = pd.DataFrame(rows, columns=["Region", "Sector",
                                 "Clean technology", "Clean tech name", "Clean price (2030)", 
                                 "Fossil technology", "Fossil tech name", "Fossil price (2030)", 
                                 "Cross-over", "Cross-over carbon tax",
                                 "Cross-over subsidies", "Cross-over mandates"])

def determine_lims_crossover(row):
    xmin = np.min(row)
    xmax = np.max(row)
    
    if row.isna().any() and not row.isna().all():
        xmax = 2050
    
    return xmin, xmax
    


#%%

# Import the results pickle file
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import get_output, get_metadata

# Set global font size
plt.rcParams.update({'font.size': 14})

# Set global font size for tick labels
plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})

output_file = "Results.pickle"

output_S0 = get_output(output_file, "S0")

titles, fig_dir, tech_titles, models = get_metadata()

price_names = {"FTT:P": "MEWC", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
shares_variables = {"FTT:P": "MEWG", "FTT:Tr": "TEWK", "FTT:Fr": "ZEWK", "FTT:H": "HEWG"}
tech_variable = {"FTT:P": 18, "FTT:Tr": 19, "FTT:H": 11, "FTT:Fr": 12}
scenarios = {"Current traj.": output_S0}

tech_name = {"FTT:P": "Solar PV", "FTT:Tr": "EV (mid-range)",
             "FTT:H": "Air-air heat pump", "FTT:Fr": "Small electric truck"}


year_inds = list(np.array([2024, 2035, 2050]) - 2010)


df_dict = {}         # Creates a new dataframe that's empty

for model in models:
    df_dict[model] = pd.DataFrame()
    rows = []
    for scen, output in scenarios.items():
        prices = output[price_names[model]][:, tech_variable[model], 0, year_inds]
        weights = output[shares_variables[model]][:, tech_variable[model], 0, year_inds]
        weighted_prices = np.average(prices, weights=weights, axis=0)
        normalised_prices = weighted_prices / weighted_prices[0]
        
        row = {"Scenario": scen, "Price 2035": normalised_prices[1], "Price 2050": normalised_prices[2]}
        rows.append(row)
    
    df_dict[model] = pd.DataFrame(rows)

share = output[shares_variables][regions, tech_variable[model], 0]
weights = output[shares_variables[model]][:, tech_variable[model], 0]

"""

df_dict_1 = {}         # Creates a new dataframe that's empty

for model in models:
    df_dict_1[model] = pd.DataFrame()
    rows = []
    for scen, output in scenarios.items():
        prices = output[price_names[model]][:, tech_variable[model], 0, year_inds]
        weights = output[shares_variables[model]][:, tech_variable[model], 0, year_inds]
        weighted_prices = np.average(prices, weights=weights, axis=0)
        normalised_prices = weighted_prices / weighted_prices[0]
        
        row = {"Scenario": scen, "Price 2035": normalised_prices[1], "Price 2050": normalised_prices[2]}
        rows.append(row)
    
    df_dict[model] = pd.DataFrame(rows)
    
"""    

"""
#%% Plot the figure
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
axs = axs.flatten()


for mi, model in enumerate(models):
    df = df_dict[model]
    ax = axs[mi]
    ax.plot(df["Scenario"], df["Price 2035"], 'o', label='Price 2035', markersize=15)
    ax.plot(df["Scenario"], df["Price 2050"], 'o', label='Price 2050', markersize=15)

    # Add labels and title
    ax.set_xticklabels(df["Scenario"], rotation=90)
    if mi%2 == 0:
        ax.set_ylabel('Cost relative to 2024')
    ax.set_title(tech_name[model], pad=20)
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(axis='y')
    

# Add legend only to the last plot
handles, labels = ax.get_legend_handles_labels()
plt.subplots_adjust(hspace=2)
fig.legend(handles, labels, loc='upper right', frameon=False, ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend
plt.show()


   
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
    'Cross-over subsidies': '#7a5195',  # Purple
    'Cross-over mandates': '#bc5090',  # Pink-purple
}

policy_names = {
    'Cross-over': 'Current traject.',
    'Cross-over carbon tax': 'Carbon tax',
    'Cross-over subsidies': 'Subsidies',
    'Cross-over mandates': 'Mandates',
    }

# Define the timeline range
timeline_start = 2020
timeline_end = 2050


yticks = []
yticklabels = []

for model in models[::-1]:
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
            
            elif pi < 4:
                ax.plot(row[policy], y_position, 'o', color=colors[policy], markersize=10)
        
                
        # Plot arrows when the crossover poinnt in past of after 2050:
        if row.iloc[-4:].isna().all():
            ax.arrow(2049.3, y_position, 1, 0, head_width=0.3, head_length=0.2, fc='#003f5c', ec='#003f5c')
        elif (row.iloc[-4:] < 2021.1).all():
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


policy_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Cross-over'], markersize=10, label='Current traject.'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Cross-over carbon tax'], markersize=10, label='Carbon tax'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Cross-over subsidies'], markersize=10, label='Subsidies'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Cross-over mandates'], markersize=10, label='Mandates')
]
# Create custom legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[policy],
                      markersize=10, label=policy_names[policy]) for policy in colors]
legend_y_position = 10
ax.legend(handles=handles, loc='upper right', frameon=True, ncol=4, bbox_to_anchor=(1.0, -0.03), framealpha=0.8)


# Add secondary y-axis for models (sectors)
secax = ax.secondary_yaxis('left')
secax.set_yticks([2.8 + i*7.2 for i, model in enumerate(models[::-1])])
secax.set_yticklabels(model_names_r, rotation=90, va='center', ha='center')
secax.tick_params(length=0, pad=100)
secax.spines['left'].set_visible(False)

fig.suptitle("Crossover year by policy \n Comparison largest clean and fossil technology in each country", 
             x=0.45, y=0.95, ha='center')
"""