# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:18:35 2024

@author: Femke Nijsse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import get_metadata, get_output, save_fig, save_data
import config

# Import the results pickle file
output_file = "Results_sxp.pickle"

output_S0 = get_output(output_file, "S0")
output_S0_all = {"FTT:P": output_S0, "FTT:H": output_S0, "FTT:Tr": output_S0, "FTT:Fr": output_S0}
output_ct_all = {"FTT:P": get_output(output_file, "sxp - P CT"),
                 "FTT:H": get_output(output_file, "sxp - H CT"),
                 "FTT:Tr": get_output(output_file, "sxp - Tr CT"),
                 "FTT:Fr": get_output(output_file, "sxp - Fr CT")}
output_sub_all = {"FTT:P": get_output(output_file, "sxp - P subs"),
                 "FTT:H": get_output(output_file, "sxp - H subs"),
                 "FTT:Tr": get_output(output_file, "sxp - Tr subs"),
                 "FTT:Fr": get_output(output_file, "sxp - Fr subs")}
output_man_all = {"FTT:P": get_output(output_file, "sxp - P mand"),
                 "FTT:H": get_output(output_file, "sxp - H mand"),
                 "FTT:Tr": get_output(output_file, "sxp - Tr mand"),
                 "FTT:Fr": get_output(output_file, "sxp - Fr mand")}

titles, fig_dir, tech_titles, models, shares_vars = get_metadata()

tech_names = {}             # The titles of the technology names


# Group the technologies in a coarser classification:
grouping_power = {"Coal": [2,3], "Gas and oil": [1, 6, 7, 8, 9], 
                  "Nuclear": [0], "Biomass + other": [4, 5, 8, 9, 12, 13],
                  "Hydropower": [12, 13], "Offshore wind": [17], "Onshore wind": [16], "Solar": [18, 19]}

grouping_heat =  { "Coal": [6], "Oil": [0, 1], "Gas": [2, 3], "Biomass": [4, 5],
                  "District heating": [7], "Electric": [8],
                  "Hydronic heat pump": [9, 10], "Air-air heat pump": [11], "Solar thermal": [12]}

grouping_transport = {"Petrol": [0, 1, 2, 4, 5, 6], "Diesel": [6, 7, 8, 9, 10, 11], 
                      "CNG": [12, 13, 14], "Hybrid": [15, 16, 17],
                      "Plug-in hybrid": [21, 22, 23], "Electric": [18, 19, 20],  "Hydrogen": [24, 25, 26]
                      }

grouping_freight = {"Petrol": [2, 3, 7, 8], "Diesel": [12, 13, 17, 18],
                    "CNG/LPG": [22, 23], "Hybrid": [27, 28],
                    "Electric": [32, 33], "Hydrogen": [42, 43], 
                    "Biofuel": [37, 38]
                    }

green_power = ["Offshore wind", "Onshore wind", "Solar", "Hydropower", "Biomass + other"]
green_heat = ["Hydronic heat pump", "Air-air heat pump"]
green_transport = ["Electric"]
green_freight = ["Electric"]

green_all = {"FTT:P": green_power, "FTT:H": green_heat, "FTT:Tr": green_transport, "FTT:Fr": green_freight}



colours_power = {
    "Nuclear": "lightcoral",  # Light coral to indicate its clean but debated nature
    "Coal": "black",  # Black for traditional, high carbon emission sources
    "Gas and oil": "silver",  # Slate grey to differentiate from coal but still fossil fuel
    "Biomass + other": "saddlebrown",  # brown for consistency
    "Hydropower": "royalblue",  
    "Offshore wind": "palegreen",  # Sky blue to signify wind, clean and free like the sky
    "Onshore wind": "mediumseagreen",  # Sky blue to signify wind, clean and free like the sky
    "Solar": "goldenrod"  # Goldenrod, rich and vibrant for solar energy
}

colours_heat = {
    "Oil": "grey",  # Darker grey to represent oil as a dense fossil fuel
    "Gas": "silver",  # Light grey for gas, distinguishing it slightly from oil
    "Biomass": "saddlebrown",  # Brown represents the organic nature of biomass
    "Coal": "black",  # Black for coal, keeping it strong and distinct
    "District heating": "tomato",  # Warm color to represent heat distribution
    "Electric": "royalblue",  # Bright blue to signify electric energy
    "Hydronic heat pump": "mediumseagreen",  # Medium sea green for water-based clean technology
    "Air-air heat pump": "palegreen",  # Pale green to signify air-based clean technology
    "Solar thermal": "goldenrod"  # Golden for the sun's energy, vibrant yet different from gold
}

colours_transport = {
    "Petrol": "dimgray",  # Dim gray for traditional gasoline vehicles
    "Diesel": "slategrey",  # Slate blue for traditional diesel vehicles
    "CNG": "mediumturquoise",  # Medium turquoise to differentiate it as cleaner than petrol/diesel
    "Hybrid": "lightgreen",  # Light green to show a mix of traditional and electric technologies
    "Plug-in hybrid": "yellowgreen",  # Yellow green to represent a better environmental choice over standard hybrids
    "Electric": "mediumseagreen",  # # Medium sea green to be consistent with heating
    "Hydrogen": "orange"  # Bright orange for the innovation and energy potential of hydrogen
}
colours_freight ={
    "Petrol": "dimgray",  # Dim gray for traditional gasoline vehicles
    "Petrol adv": "darkgrey",  # Light gray to indicate slight improvements in petrol technology
    "Diesel": "slategrey",  # Slate blue for traditional diesel vehicles
    "Diesel adv": "lightsteelblue",  # Cornflower blue for advanced diesel with better emissions
    "CNG/LPG": "mediumturquoise",  # Medium turquoise to differentiate it as cleaner than petrol/diesel
    "Hybrid": "lightgreen",  # Light green to show a mix of traditional and electric technologies
    "Electric": "mediumseagreen",  # Medium sea green to be consistent with above
    "Biofuel": "saddlebrown",
    "Hydrogen": "orange"  # Bright orange for the innovation and energy potential of hydrogen
}

groupings = {"FTT:P": grouping_power, "FTT:H": grouping_heat,
             "FTT:Tr": grouping_transport, "FTT:Fr": grouping_freight}

colours ={"FTT:P": colours_power, "FTT:H": colours_heat,
             "FTT:Tr": colours_transport, "FTT:Fr": colours_freight}

ylabels = {"FTT:P": "Generation (PWh)", "FTT:H": "Useful demand (PWh)",
           "FTT:Tr": "Cars (billions)", "FTT:Fr": "Trucks (millions)"}

legend_titles = {"FTT:P": "Power technology", "FTT:H": "Heating technology", 
              "FTT:Tr": "Transport technology", "FTT:Fr": "Freight technology"}

# Define the text to be added to the left of the y-labels
left_labels = ["Power generation", "Residential heating", "Personal transport", "Road freight"]

# First define the globally summed generation, or number of vehicles:
def sum_vehicles_or_gen(output):
    total_shares_all = {}           # The share of each technology (or total capacity, not yet decided)
    total_shares = {}
    for model in models:
        total_shares_all[model] = np.sum(output[model][shares_vars[model]], axis=(0, 2))
        tech_names[model] = list(groupings[model].keys())
        
        indices = groupings[model].values()
               
        total_shares[model] =  np.array(
                [[np.sum(total_shares_all[model][idx, year]) 
                  for idx in indices]
                     for year in range(41)]
                )
        total_shares[model] = total_shares[model].T / 10**6
    return total_shares
        
# Create the dataframes
def create_dataframes(total_shares):
    model_dfs = {}              # Dictionary with the DataFrame for each model
    for model, data in total_shares.items():
        # Create a DataFrame for the current model
        df = pd.DataFrame(data, columns=range(2010, 2051))  # Years from 2010 to 2050
        # Add the technology names as a column
        df['Technology'] = tech_names[model]
        # Set the Technology column as the index
        df.set_index('Technology', inplace=True)
        # Only plot from 2020 onwards
        df = df.loc[:, 2020:]
        # Save the DataFrame in the dictionary
        model_dfs[model] = df
    return model_dfs


total_shares_S0 = sum_vehicles_or_gen(output_S0_all)
model_dfs_S0 = create_dataframes(total_shares_S0)

total_shares_ct = sum_vehicles_or_gen(output_ct_all)
model_dfs_ct = create_dataframes(total_shares_ct)

total_shares_sub = sum_vehicles_or_gen(output_sub_all)
model_dfs_sub = create_dataframes(total_shares_sub)

total_shares_man = sum_vehicles_or_gen(output_man_all)
model_dfs_man = create_dataframes(total_shares_man)


#%% Plot the figure
fig, axs = plt.subplots(4, 4, figsize=(7.2, 6.8), sharey='row')
axs = axs.flatten()

left_most_indices = [0, 4, 8, 12]  # For a 4x4 grid
text_y_values = {"FTT:P": 32, "FTT:H": 7, "FTT:Tr": 1.2, "FTT:Fr": 120}

def get_sum_greens_2050(model_dfs, model):
    """Take the sum over all green technologies"""
    green_sum = np.sum([model_dfs[2050][tech]
                      for tech in 
                      green_all[model]])
    green_share = green_sum / np.sum(model_dfs[2050])
    return green_sum, green_share


def green_growth(model_df_scen, model):
    "Percentage difference in proper green techs from baseline"
    _, baseline_green = get_sum_greens_2050(model_dfs_S0[model], model)
    _, scenario_green = get_sum_greens_2050(model_df_scen, model)
    green_growth = (scenario_green - baseline_green)/baseline_green * 100
    
    green_growth_pp = (scenario_green - baseline_green)/np.sum(model_dfs_S0[model][2050]) * 100    
    green_growth_pp = (scenario_green - baseline_green) * 100   

    return green_growth, green_growth_pp


def plot_column(model_dfs, col, col_title):
    for idx, (model, model_df) in enumerate(model_dfs.items()):
        # Figure 1: Plot global shares of generation      
        ax = axs[left_most_indices[idx] + col]
        model_df_T = model_df.T
        model_df_T.plot(kind="area", ax=ax, color = colours[model], linewidth=0)
        
        ax.set_ylabel(ylabels[model])
        ax.get_yaxis().set_label_coords(-0.2, 0.5)
        
        # Set data and demand labels and ticks
        if idx == 3:
            # Turn off for rows above
            years_labels = [2020, 2030, 2040, 2050]
            years_labels_loc = [1, 10.5, 19.5, 29]
            ax.set_xticks(years_labels_loc)
            ax.set_xticklabels(years_labels)
        else: 
            ax.set_xticks([])
        ax.xaxis.set_ticks_position('none') 
        ax.yaxis.set_ticks_position('none') 
        ax.tick_params(axis='both', which='major', pad=-3)
        
        # Remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Compute difference from baseline
        green_growth_out, green_growth_pp = green_growth(model_df, model)
        
        
        if col == 0:
            ax.text(-0.32, 0.5, left_labels[idx], transform=ax.transAxes,
                    ha='right', va='center', rotation=90, fontweight='bold')
        
        if col != 0:
            ax.text(x=29.5, y=text_y_values[model], s=f'+{green_growth_pp:.0f}%pt', 
                    horizontalalignment='right', fontweight='bold')
        
        if col == 3:
            # Create individual legend to the right of each plot
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles, labels, title=legend_titles[model],
                      labelspacing=0.1, frameon=False,
                      bbox_to_anchor=(1.0, 1.0), loc='upper left')
        
            # Access the legend's title and increase vertical padding
            legend.get_title().set_position((0, 5))  # The numbers are x, y offsets
            
        else:
            # Explicitly do not create a legend for the first column
            ax.legend([], [], frameon=False)
            
        # For the top row
        if idx == 0:
            ax.set_title(col_title, fontweight='bold')
            

plot_column(model_dfs_S0, 0, "A. Current trajectory")
plot_column(model_dfs_ct, 1, "B. Carbon tax")
plot_column(model_dfs_sub, 2, "C. Subsidies")
plot_column(model_dfs_man, 3, "D. Mandates / phase-out")



def combine_dfs(dict_of_dfs):
    
    # Combine the DataFrames into a single DataFrame
    combined_df = pd.concat(dict_of_dfs.values(), keys=dict_of_dfs.keys())
    
    # Reset the index, converting both levels of the index into columns
    combined_df = combined_df.reset_index()

    # Rename the newly created level_0 column to "Model"
    combined_df = combined_df.rename(columns={'level_0': 'Model', "Level_1": "Technology"})
    
    return combined_df


fig.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the width spacing

save_fig(fig, fig_dir, "Figure 4 - Shares_graph_4x4")
save_data(combine_dfs(model_dfs_S0), fig_dir, "Figure 4 - Shares_graph_4x4_Baseline")
save_data(combine_dfs(model_dfs_ct), fig_dir, "Figure 4 - Shares_graph_4x4_Carbontax")
save_data(combine_dfs(model_dfs_sub), fig_dir, "Figure 4 - Shares_graph_4x4_Subsidies")
save_data(combine_dfs(model_dfs_man), fig_dir, "Figure 4 - Shares_graph_4x4_Mandates")

