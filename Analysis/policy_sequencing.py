# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:35:01 2024

@author: Femke Nijsse
"""
#%%
# Import the results pickle file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd

from preprocessing import get_output, get_metadata, save_fig, save_data
import config

# To do: create scenarios with a newly defined baseline of only temporary mandates
titles, fig_dir, tech_titles, models, shares_vars = get_metadata()


output_file = "Results_sequencing.pickle"
output_baseline = get_output(output_file, "S0")
scenarios = ["sxp - Fr CT", "CT_from_2027", "CT_from_2030", "CT_from_2035"]
scenario_names = ["Carbon tax only", "Mandate before 2027", "Mandate before 2030", "Mandate before 2035"]

scenario_counterfactuals = ["CT_from_2024_counterfactual_2030", 
                            "CT_from_2024_counterfactual_2035",
                            "CT_from_2027_counterfactual_2030",
                            "CT_from_2027_counterfactual_2035"
                            "CT_from_2030_counterfactual_2030",
                            "CT_from_2030_counterfactual_2035",
                            "CT_from_2035_counterfactual_2035"]

# Dataframe to save EV truck shares and sales as function of scenario
fleet_stock_EVs = pd.DataFrame(index=np.arange(2024, 2051),
                               columns=scenario_names + scenario_counterfactuals)
fleet_growth_EVs = pd.DataFrame(index=np.arange(2025, 2051),
                                columns=scenario_names + scenario_counterfactuals)
fleet_growth_EVs_difference = pd.DataFrame(index=np.array([2030, 2035]), columns=scenario_names)


# Extract the results of variable TEWK from all carbon tax scenarios
for scenario, scenario_name in zip(scenarios, scenario_names):
    output = get_output(output_file, scenario)
    # TODO: switch this when there is a new classification
    fleet_stock_EVs[scenario_name] = output['ZEWK'][:, 13, 0, 14:].sum(axis=0)
    fleet_growth_EVs[scenario_name] = fleet_stock_EVs[scenario_name].diff()[1:]
    
# Extract the results of variable TEWK from all counterfactual scenarios
for scenario in scenario_counterfactuals:
    output = get_output(output_file, scenario)
    fleet_stock_EVs[scenario] = output['ZEWK'][:, 13, 0, 14:].sum(axis=0)
    fleet_growth_EVs[scenario] = fleet_stock_EVs[scenario].diff()[1:]
    
def get_diff(year, scenario, counterfactual):
    return fleet_growth_EVs.loc[year, scenario] - fleet_growth_EVs.loc[year, counterfactual]

# For each of the seven year/combination combination, combine the difference between the scenerio and counterfactual
fleet_growth_EVs_difference.loc[2030, "Carbon tax only"] = get_diff(2030, "Carbon tax only", "CT_from_2024_counterfactual_2030")
fleet_growth_EVs_difference.loc[2030, "Mandate before 2027"] = get_diff(2030, "Mandate before 2027", "CT_from_2027_counterfactual_2030")
fleet_growth_EVs_difference.loc[2030, "Mandate before 2030"] = get_diff(2030, "Mandate before 2030", "CT_from_2030_counterfactual_2030")
fleet_growth_EVs_difference.loc[2035, "Carbon tax only"] = get_diff(2035, "Carbon tax only", "CT_from_2024_counterfactual_2035")
fleet_growth_EVs_difference.loc[2035, "Mandate before 2027"] = get_diff(2035, "Mandate before 2027", "CT_from_2027_counterfactual_2035")
fleet_growth_EVs_difference.loc[2035, "Mandate before 2030"] = get_diff(2035, "Mandate before 2030", "CT_from_2030_counterfactual_2035")
fleet_growth_EVs_difference.loc[2035, "Mandate before 2035"] = get_diff(2035, "Mandate before 2035", "CT_from_2035_counterfactual_2035")

    
#%% ===========================================================================
# Bar chart figure for policy effectiveness freight
# =============================================================================


# Create subplots (2 rows, 1 column)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7.2, 1.8), dpi=300, sharey=True)  # Adjust figsize to fit both plots

# Try again differently with more teal/green
colors = ['#a1dab4', '#41ae76', '#238b45', '#005824']

# Reverse those colours
colors = colors[::-1]


# Create two horizontal bar charts for the growth in EVs in 2030 and 2035. 

# Plot the growth in EVs in 2030, omitting the "Switch in 2035" scenario
fleet_growth_EVs.drop(columns="Mandate before 2035").loc[2030].plot(
                kind='barh', color=colors, ax=axes[0], width=0.75)
#fleet_growth_EVs.loc[2030].plot(kind='barh', color=colors, ax=axes[0])
axes[0].set_title("EV fleet growth under a carbon tax in 2030", weight='bold')
axes[0].set_xlabel("Growth in EVs")
axes[0].set_ylabel("Year")


# Plot the growth in EVs in 2035
fleet_growth_EVs.loc[2035].plot(kind='barh', color=colors, ax=axes[1], width=0.75)
axes[1].set_title("EV fleet growth under a carbon tax in 2035", weight='bold')
axes[1].set_xlabel("Growth in EVs")
axes[1].set_ylabel("Year")

# Remove spines and add commas to the x-axis
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Remove year as x-axis label
axes[0].set_ylabel("")

# Decrease space slightly between two graphs
plt.subplots_adjust(wspace=0.15)
        
# Switch the order of the bars
axes[0].invert_yaxis()

# Compute the ratio of effectiveness between the Mandate before 2035 and Carbon tax only  scenario
ratio = fleet_growth_EVs.loc[2035, "Mandate before 2035"] / fleet_growth_EVs.loc[2035, "Carbon tax only"] 
print(f"The carbon tax is {ratio:.0f}x as effective after 10 years of mandates, compared to a carbon tax only scenario")

# Save sequencing figure as a svg file and a png file
save_fig(fig, fig_dir, "EV_growth_sequencing")
save_data(fleet_growth_EVs, fig_dir, "EV_growth_sequencing")









# %%
