# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:35:01 2024

@author: Femke Nijsse
"""
#%%
# Import the results pickle file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import get_output, get_metadata, save_fig, save_data
import config


# To do: create scenarios with a newly defined baseline of only temporary mandates
titles, fig_dir, tech_titles, models, shares_vars = get_metadata()



# %%

output_file = "Results_sequencing2.pickle"
output_baseline = get_output(output_file, "S0")
scenarios_ct = [f"CT_from_{year}" for year in range(2025, 2041)]
scenarios_cf = [f"CT_from_{year}_counterfactual_{year}" for year in range(2025, 2041)]
truck_cats = ["Medium-duty trucks", "Heavy-duty trucks"]

years = np.arange(2025, 2041)
years2x = np.concatenate([years, years])

# Dataframe to save EV truck shares and sales as function of scenario
sales = pd.DataFrame(index=years2x, columns=scenarios_ct + scenarios_cf + ["Truck cat"])
total_sales = pd.DataFrame(index=years2x, columns=scenarios_ct + scenarios_cf + ["Truck cat"])
rel_sales = pd.DataFrame(index=years2x, columns=scenarios_ct + scenarios_cf + ['Truck cat'])

sales_diff = pd.DataFrame(index=years, columns = truck_cats)
relative_sales_diff = pd.DataFrame(index=years, columns = truck_cats)
sales_ratio = pd.DataFrame(index=years, columns = truck_cats)

for scenario in scenarios_ct + scenarios_cf:
    # Extract the results of variable ZEWK and ZEWI from all carbon tax scenarios
   
    output = get_output(output_file, scenario)
    
    for truck_cat in enumerate(truck_cats):
        sales[scenario] = output["ZEWI"][:, 32:34, 0, 15:31].sum(axis=0).flatten()
        sales["Truck cat"] =  ["Medium-duty trucks"] * 16 + ["Heavy-duty trucks"] * 16
        total_sales_mdv = output["ZEWI"][:, 2::5, 0, 15:31].sum(axis=(0, 1))
        total_sales_hdv = output["ZEWI"][:, 3::5, 0, 15:31].sum(axis=(0, 1))
        total_sales[scenario] =  np.concatenate([total_sales_mdv, total_sales_hdv])
        total_sales["Truck cat"] = ["Medium-duty trucks"] * 16 + ["Heavy-duty trucks"] * 16
        rel_sales[scenario] = sales[scenario] / total_sales[scenario]
        rel_sales["Truck cat"] = ["Medium-duty trucks"] * 16 + ["Heavy-duty trucks"] * 16


for truck_cat in truck_cats:
    for year in years:
        #sales_diff.loc[year, "No policy"] = sales[f"CT_from_{year}_counterfactual_{year}"].loc[year]
        sales_diff.loc[year, truck_cat] = (
            sales[sales['Truck cat'] == truck_cat][f"CT_from_{year}"] - sales[sales['Truck cat'] == truck_cat][f"CT_from_{year}_counterfactual_{year}"]).loc[year]
        relative_sales_diff.loc[year, truck_cat] = (
            rel_sales[rel_sales['Truck cat'] == truck_cat][f"CT_from_{year}"] - rel_sales[rel_sales['Truck cat'] == truck_cat][f"CT_from_{year}_counterfactual_{year}"]).loc[year]
        sales_ratio.loc[year, truck_cat] = (
            rel_sales[rel_sales['Truck cat'] == truck_cat][f"CT_from_{year}"] / rel_sales[rel_sales['Truck cat'] == truck_cat][f"CT_from_{year}_counterfactual_{year}"]).loc[year]

sales_diff["Medium-duty trucks"] /= 1000  
sales_diff["Heavy-duty trucks"] /= 1000  


#%%    
# Create subplots (2 rows, 1 column)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7.2, 1.8), dpi=300)


sales_diff.plot(
    ax=axes[0],   
    kind='bar', 
    width=0.6,
    color=["#a6cee3", "#1f78b4"],  # optional: use accessible colours
)

axes[0].set_ylabel("Additional sales (1000)")
#axes[0].set_xlabel("Year")


sales_ratio.plot(
    color=["#a6cee3", "#1f78b4"],  # optional: use accessible colours
    kind='bar', 
    width=0.6,
    ax = axes[1]
)

axes[1].set_ylim(bottom=1)
axes[1].set_ylabel("Sales ratio")
#axes[1].set_xlabel("Year")

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
axes[0].text(-0.1, 1.08, 'a', transform=axes[0].transAxes, fontweight='bold', va='top', ha='right')
axes[1].text(-0.1, 1.08, 'b', transform=axes[1].transAxes, fontweight='bold', va='top', ha='right')

fig.tight_layout()
save_fig(fig, fig_dir, "Figure 5 - Policy sequencing.svg")

sales_diff = sales_diff.reset_index()
sales_diff = sales_diff.rename(columns={'index': 'Year'})

sales_ratio = sales_ratio.reset_index()
sales_ratio = sales_ratio.rename(columns={'index': 'Year'})

save_data(sales_diff, fig_dir, "Figure 5a - Additional sales")
save_data(sales_ratio, fig_dir, 'Figure 5b - Ratio sales carbon price vs no policy')





