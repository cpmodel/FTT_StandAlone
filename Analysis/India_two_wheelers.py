# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:18:35 2024

@author: Femke Nijsse
"""

# Import the results pickle file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import get_metadata, get_output, save_fig, save_data
import config

# To do: change this to output Amir is finding. 
output_file = "Results_S0.pickle"

# Change to the name of the scenario
output_S0 = get_output(output_file, "S0")


titles, fig_dir, tech_titles, cats, shares_vars = get_metadata()

tech_names = {}             # The titles of the technology names
cats = ["FTT:twv"]

# Group the technologies in a coarser classification:                     
grouping_twv = {"Diesel": [10, 15], "CNG/LPG": [20], "Electric": [30]
                    }


green_techs = ["Electric"]

colours_freight = {
    "Diesel": "slategrey",  # Slate blue for traditional diesel vehicles
    "CNG/LPG": "mediumturquoise",  # Medium turquoise to differentiate it as cleaner than petrol/diesel
    "Electric": "mediumseagreen",  # Medium sea green to be consistent with above
}


groupings = {"FTT:twv": grouping_twv}


# Define the text to be added to the left of the y-labels
left_labels = ["Two- and three wheelers"]

# First define the global number of vehicles (millions):
def sum_vehicles(output):
    total_vehicles_all = {}           # Total vehicles
    total_vehicles = {}               # Total vehicles, summed over categories
    for cat in cats:
        total_vehicles_all[cat] = np.sum(output[shares_vars["FTT:Fr"]], axis=(0, 2))
        tech_names[cat] = list(groupings[cat].keys())
        
        indices = groupings[cat].values()
               
        total_vehicles[cat] =  np.array(
                [[np.sum(total_vehicles_all[cat][idx, year]) 
                  for idx in indices]
                     for year in range(41)]
                )
        total_vehicles[cat] = total_vehicles[cat].T / 10**6     #   Convert to millions
    return total_vehicles
        
# Create the dataframes
def create_dataframes(total_shares):
    cat_dfs = {}              # Dictionary with the DataFrame for each cat
    for cat, data in total_shares.items():
        # Create a DataFrame for the current cat
        df = pd.DataFrame(data, columns=range(2010, 2051))  # Years from 2010 to 2050
        # Add the technology names as a column
        df['Technology'] = tech_names[cat]
        # Set the Technology column as the index
        df.set_index('Technology', inplace=True)
        # Only plot from 2020 onwards
        df = df.loc[:, 2020:]
        # Save the DataFrame in the dictionary
        cat_dfs[cat] = df
    return cat_dfs


total_shares_S0 = sum_vehicles(output_S0)
cat_dfs_S0 = create_dataframes(total_shares_S0)



#%% Plot the figure
fig, ax = plt.subplots(figsize=(2.6, 2.3))
#axs = axs.flatten()


left_most_indices = [0]  # For a 4x4 grid
text_y_values = {"FTT:twv": 5}

def get_sum_greens_2050(cat_dfs):
    """Take the sum over all green technologies 2050"""
    green_sum = np.sum([cat_dfs[2050][tech]
                      for tech in green_techs])
    return green_sum

def get_sum_all(cat_dfs, cat):
    """Take the sum over all green technologies"""
    sum_all = np.sum(cat_dfs[2050])
    return sum_all

def green_growth(cat_df_scen, cat):
    "Percentage difference in proper green techs from baseline"
    baseline_green = get_sum_greens_2050(cat_dfs_S0[cat])
    scenario_green = get_sum_greens_2050(cat_df_scen)
    green_growth = (scenario_green - baseline_green)/baseline_green * 100
    
    green_growth_pp = (scenario_green - baseline_green)/np.sum(cat_dfs_S0[cat][2050]) * 100    
    return green_growth, green_growth_pp


def plot_column(cat_dfs, col, col_title):
    for idx, (cat, cat_df) in enumerate(cat_dfs.items()):
        # Figure 1: Plot global shares of generation      
        #ax = axs[left_most_indices[idx] + col]
        cat_df_T = cat_df.T
        cat_df_T.plot(kind="area", ax=ax, color = colours_freight, linewidth=0)
        
        ax.set_ylabel("Vehicles (millions)")
        ax.get_yaxis().set_label_coords(-0.2, 0.5)
        
        # Set data and demand labels and ticks
    
        years_labels = [2020, 2030, 2040, 2050]
        years_labels_loc = [1, 10.5, 19.5, 29]
        ax.set_xticks(years_labels_loc)
        ax.set_xticklabels(years_labels)
    
        ax.xaxis.set_ticks_position('none') 
        ax.yaxis.set_ticks_position('none') 
        ax.tick_params(axis='both', which='major')
        
        # Remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Compute difference from baseline
        green_growth_out, green_growth_pp = green_growth(cat_df, cat)
        
        
        if col == 0:
            ax.text(-0.32, 0.5, left_labels[idx], transform=ax.transAxes,
                    ha='right', va='center', rotation=90, fontweight='bold')
        
        if col != 0:
            ax.text(x=29.5, y=text_y_values[cat], s=f'+{green_growth_pp:.0f}%pt', 
                    horizontalalignment='right', fontweight='bold')
        
        if col == 3:
            # Create individual legend to the right of each plot
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles, labels, title="Technology",
                      labelspacing=0.1, frameon=False,
                      bbox_to_anchor=(1.0, 0.9), loc='upper left')
            
        
            # Access the legend's title and increase vertical padding
            legend.get_title().set_position((0, 5))  # The numbers are x, y offsets
            
        else:
            # Explicitly do not create a legend for the first column
            ax.legend([], [], frameon=False)
            
        # For the top row
        if idx == 0:
            ax.set_title(col_title, fontweight='bold')
            
# Change the column names here
plot_column(cat_dfs_S0, 0, "A. Current trajectory")
# plot_column(cat_dfs_ct, 1, "B. Carbon tax")
# plot_column(cat_dfs_sub, 2, "C. Subsidies")
# plot_column(cat_dfs_man, 3, "D. Mandate")

fig.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the width spacing
save_fig(fig, fig_dir, "Figure 5 - Shares_twv_India")

#%% Save the data for the design team

def combine_dfs(dict_of_dfs):
    
    # Combine the DataFrames into a single DataFrame
    combined_df = pd.concat(dict_of_dfs.values(), keys=dict_of_dfs.keys())
    
    # Reset the index, converting both levels of the index into columns
    combined_df = combined_df.reset_index()

    # Rename the newly created level_0 column to "cat"
    combined_df = combined_df.rename(columns={'level_0': 'cat', "Level_1": "Technology"})
    
    return combined_df

save_data(combine_dfs(cat_dfs_S0), fig_dir, "Figure 5 - Shares_India_twv")
save_fig(fig, fig_dir, "Figure 12a - shares two- and three whelers India")
