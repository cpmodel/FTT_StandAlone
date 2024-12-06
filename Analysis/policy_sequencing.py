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
                            "CT_from_2027_counterfactual_2035",
                            "CT_from_2030_counterfactual_2030",
                            "CT_from_2030_counterfactual_2035",
                            "CT_from_2035_counterfactual_2035"]

# Dataframe to save EV truck shares and sales as function of scenario
sales = pd.DataFrame(index=np.arange(2025, 2051),
                                columns=scenario_names + scenario_counterfactuals)
total_sales = pd.DataFrame(index=np.arange(2025, 2051),
                                columns=scenario_names + scenario_counterfactuals)
sales_difference = pd.DataFrame(index=np.array(["2030 no policy", "2030 CT", "2035 no policy", "2035 CT"]),
                                columns=scenario_names)
relative_sales_difference = pd.DataFrame(index=np.array(["2030 no policy", "2030 CT", "2035 no policy", "2035 CT"]),
                                columns=scenario_names)


# Extract the results of variable ZEWK and ZEWI from all carbon tax scenarios
for scenario, scenario_name in zip(scenarios, scenario_names):
    output = get_output(output_file, scenario)
    # TODO: switch this when there is a new classification (for total and normal sales)
    sales[scenario_name] = output["ZEWI"][:, 13, 0, 15:].sum(axis=0)
    total_sales[scenario_name] =  output["ZEWI"][:, 1::2, 0, 15:].sum(axis=(0, 1))

#%%    
# Extract the results of variable ZEWK and ZEWI from all counterfactual scenarios
for scenario in scenario_counterfactuals:
    output = get_output(output_file, scenario)
    sales[scenario] = output["ZEWI"][:, 13, 0, 15:].sum(axis=0)
    total_sales[scenario] =  output["ZEWI"][:, 1::2, 0, 15:].sum(axis=(0, 1))

relative_sales = sales / total_sales

def get_diff(output_var, year, scenario, counterfactual, variable):
    
    difference = variable.loc[year, scenario] - variable.loc[year, counterfactual]
    counterfactual = variable.loc[year, counterfactual]
    output_var.loc[str(year) + " CT", scenario] = difference
    output_var.loc[str(year) + " no policy", scenario] = counterfactual
    return output_var

def match_scenarios_for_diff(variable_diff, variable_all):                    
    # For each of the seven year/combination combination, combine the difference between the scenario and counterfactual
    variable_diff = get_diff(variable_diff, 2030, "Carbon tax only", "CT_from_2024_counterfactual_2030", variable_all)
    variable_diff = get_diff(variable_diff, 2030, "Mandate before 2027", "CT_from_2027_counterfactual_2030", variable_all)
    variable_diff = get_diff(variable_diff, 2030, "Mandate before 2030", "CT_from_2030_counterfactual_2030", variable_all)
    variable_diff = get_diff(variable_diff, 2035, "Carbon tax only", "CT_from_2024_counterfactual_2035", variable_all)
    variable_diff = get_diff(variable_diff, 2035, "Mandate before 2027", "CT_from_2027_counterfactual_2035", variable_all)
    variable_diff = get_diff(variable_diff, 2035, "Mandate before 2030", "CT_from_2030_counterfactual_2035", variable_all)
    variable_diff = get_diff(variable_diff, 2035, "Mandate before 2035", "CT_from_2035_counterfactual_2035", variable_all)

    return variable_diff

sales_difference = match_scenarios_for_diff(sales_difference, sales)
relative_sales_difference = match_scenarios_for_diff(relative_sales_difference, relative_sales)



#%% ===========================================================================
# Bar chart figure for policy effectiveness freight
# =============================================================================


# Create subplots (2 rows, 1 column)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7.2, 1.8), dpi=300, sharey=True)  # Adjust figsize to fit both plots

# Try again differently with more teal/green
colors = ['#a1dab4', '#41ae76', '#238b45', '#005824']

# Reverse those colours
colors = colors[::-1]

# Create two horizontal bar charts for the sales in EVs in 2030 and 2035. 

relative_sales_difference.T.loc[:, [ "2030 no policy", "2030 CT"]].plot(
                kind='barh', stacked=True, color=colors, ax=axes[0], width=0.75)

#fleet_growth_EVs.loc[2030].plot(kind='barh', color=colors, ax=axes[0])
axes[0].set_title("Preceding policy", weight='bold', loc='left')
axes[0].set_xlabel("Additional sales share in large electric trucks")
axes[0].set_ylabel("Year")


# Plot the growth in EVs in 2035
relative_sales_difference.T.loc[:, ["2035 no policy", "2035 CT"]].plot(
        kind='barh', stacked=True, color=colors, ax=axes[1], width=0.75)
axes[1].set_xlabel("Additional sales share in large electric trucks")
axes[1].set_ylabel("Year")

# Remove spines and add commas to the x-axis
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
# Remove year as x-axis label
axes[0].set_ylabel("")

# Decrease space slightly between two graphs
plt.subplots_adjust(wspace=0.15)
        
# Switch the order of the bars
axes[0].invert_yaxis()


# Save sequencing figure as a svg file and a png file
save_fig(fig, fig_dir, "EV_sales_sequencing")
save_data(sales_difference, fig_dir, "EV_sales_sequencing")

ratio = sales_difference.loc["2035 CT", "Mandate before 2035"] / sales_difference.loc["2035 CT", "Carbon tax only"] 
print(f"The carbon tax is {ratio:.0f}x as effective after 10 years of mandates, compared to a carbon tax only scenario")





# %%
