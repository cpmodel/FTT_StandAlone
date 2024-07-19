# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:13:22 2024

@author: Femke Nijsse
"""

# Import the results pickle file
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess_data

# Set global font size
plt.rcParams.update({'font.size': 14})

# Set global font size for tick labels
plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})

output_file = "Results_policies.pickle"

# TODO: run with the policies separately
output_S0, titles, fig_dir, tech_titles = preprocess_data(output_file, "S0")
output_ct, _, _, _  = preprocess_data(output_file, "Carbon tax")
output_sub, _, _, _ = preprocess_data(output_file, "Subsidies")
output_man, _, _, _ = preprocess_data(output_file, "Mandates")

models = ["FTT:P", "FTT:H", "FTT:Tr", "FTT:Fr"]
price_names = {"FTT:P": "MECW", "FTT:Tr": "TEWC", "FTT:H": "HEWC", "FTT:Fr": "ZTLC"}
shares_variables = {"FTT:P": "MEWG", "FTT:Tr": "TEWK", "FTT:Fr": "ZEWK", "FTT:H": "HEWG"}
tech_variable = {"FTT:P": 18, "FTT:Tr": 19, "FTT:H": 11, "FTT:Fr": 12}
scenarios = {"Current traj.": output_S0, "Carbon tax": output_ct,
             "Subsidies": output_sub, "Mandates": output_man}

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
   
# Save the graph as an editable svg file
output_file = os.path.join(fig_dir, "Cost_declines_by_policy.svg")
fig.savefig(output_file, format="svg")
