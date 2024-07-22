# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:35:01 2024

@author: Femke Nijsse
"""

# Import the results pickle file
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import get_output, get_metadata

# Set global font size
plt.rcParams.update({'font.size': 14})
# Set global font size
plt.rcParams.update({'xtick.labelsize': 12, 'ytick.labelsize': 12})

output_file = "Results_policies.pickle"

output = get_output("Results_scens.pickle", "S0")
output_ct  = get_output(output_file, "Carbon tax")
output_sub = get_output(output_file, "Subsidies")
output_man = get_output(output_file, "Mandates")

titles, fig_dir, tech_titles, models = get_metadata()

# Define the clean technology list by model
clean_techs = {"FTT:P": [16, 18], "FTT:Tr": [18, 19, 20], "FTT:H": [10, 11], "FTT:Fr": [12]}
dirty_techs = {"FTT:P": [0, 2, 6], "FTT:Tr": list(range(12)), "FTT:H": [2, 3], "FTT:Fr": [0, 2, 4, 6, 8]}

# Define the year of interest
year = 2030

# Define the shares, prices of interest
shares_names = {"FTT:P": "MEWS", "FTT:Tr": "TEWS", "FTT:H": "HEWS", "FTT:Fr": "ZEWS"}
emissions_names = {"FTT:P": "MEWE", "FTT:Tr": "TEWE", "FTT:H": "HEWE", "FTT:Fr": "ZEWE"}

models_to_scenarios = {"FTT:P": ["FTT:H", "FTT:Tr", "FTT:Fr", "All minus FTT:P"],
                       "FTT:H": ["FTT:P",  "FTT:Tr", "FTT:Fr", "All minus FTT:H"],
                       "FTT:Tr": ["FTT:P",  "FTT:H", "FTT:Fr", "All minus FTT:Tr"],
                       "FTT:Rr": ["FTT:P",  "FTT:H", "FTT:Tr", "All minus FTT:Fr"] }

# # Compute overall CO2 reductions per sector
# for model in models:
#%% Plot the figure
fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharey='row')
axs = axs.flatten()



