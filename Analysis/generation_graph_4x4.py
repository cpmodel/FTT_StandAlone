# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:18:35 2024

@author: Femke Nijsse
"""

# Import the results pickle file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from preprocessing import get_metadata, get_output

# Set global font size
plt.rcParams.update({'font.size': 12})

# Set global font size for tick labels
plt.rcParams.update({'xtick.labelsize': 10, 'ytick.labelsize': 10})


output_file = "Results_scens.pickle"

output_S0 = get_output(output_file, "S0")
output_ct = get_output(output_file, "Carbon tax")
output_sub = get_output(output_file, "and_subsidies")
output_man = get_output(output_file, "and_mandates")

titles, fig_dir, tech_titles, models = get_metadata()

shares_variables = {"FTT:P": "MEWG", "FTT:Tr": "TEWK", "FTT:Fr": "ZEWK", "FTT:H": "HEWG"}

tech_names = {}             # The titles of the technology names


# Group the technologies in a coarser classification:
    
grouping_power = {"Coal and oil": [1, 2, 4], "Coal + CCS": [3, 5], "Gas": [6], "Gas + CCS": [7], 
                  "Nuclear": [0], "Biomass + other": [8, 10, 12, 14, 20, 21, 22, 23], "Biomass + CCS": [9, 11, 13],
                  "Hydropower": [15], "Wind": [16, 17], "Solar": [18, 19]}


grouping_heat = { "Coal": [6], "Oil": [0, 1], "Gas": [2, 3], "Biomass": [4, 5],
                  "District heating": [7], "Electric": [8],
                  "Hydronic heat pump": [9, 10], "Air-air heat pump": [11], "Solar thermal": [12]}

grouping_transport = {"Petrol": [0, 1, 2], "Petrol adv": [3, 4, 5], "Diesel": [6, 7, 8], 
                      "Diesel adv": [9, 10, 11], "CNG": [12, 13, 14], "Hybrid": [15, 16, 17],
                      "Plug-in hybrid": [21, 22, 23], "Electric": [18, 19, 20],  "Hydrogen": [24, 25, 26]
                      }
grouping_freight = {"Petrol": [0, 2], "Diesel": [4, 6],
                    "CNG/LPG": [8], "Hybrid": [10], "Electric": [12], "Hydrogen": [18], "Biofuel": [14, 16]
                    }



colours_power = {
    "Nuclear": "lightcoral",  # Light coral to indicate its clean but debated nature
    "Coal and oil": "black",  # Black for traditional, high carbon emission sources
    "Coal + CCS": "dimgray",  # Dim gray to represent coal with carbon capture, slightly cleaner
    "Gas": "silver",  # Slate grey to differentiate from coal but still fossil fuel
    "Gas + CCS": "gainsboro",  # Light slate gray for gas with carbon capture, transitional
    "Biomass + other": "darkolivegreen",  # Dark olive green to represent biomass, a renewable but complex source
    "Biomass + CCS": "yellowgreen",  # Yellow green for biomass with carbon capture, enhancing its clean aspects
    "Hydropower": "royalblue",  # Sky blue to signify wind, clean and free like the sky
    "Wind": "skyblue",  # Sky blue to signify wind, clean and free like the sky
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
    "Petrol adv": "darkgrey",  # Light gray to indicate slight improvements in petrol technology
    "Diesel": "slategrey",  # Slate blue for traditional diesel vehicles
    "Diesel adv": "lightsteelblue",  # Cornflower blue for advanced diesel with better emissions
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
           "FTT:Tr": "Cars (billions)", "FTT:Fr": "Light trucks (millions)"}

legend_titles = {"FTT:P": "Power technology", "FTT:H": "Heating technology", 
              "FTT:Tr": "Transport technology", "FTT:Fr": "Freight technology"}

# First define the globally summed generation, or number of vehicles:
def sum_vehicles_or_gen(output):
    total_shares_all = {}           # The share of each technology (or total capacity, not yet decided)
    total_shares = {}
    for model in models:
        total_shares_all[model] = np.sum(output[shares_variables[model]], axis=(0, 2))
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

total_shares_S0 = sum_vehicles_or_gen(output_S0)
model_dfs_S0 = create_dataframes(total_shares_S0)

total_shares_ct = sum_vehicles_or_gen(output_ct)
model_dfs_ct = create_dataframes(total_shares_ct)

total_shares_sub = sum_vehicles_or_gen(output_sub)
model_dfs_sub = create_dataframes(total_shares_sub)

total_shares_man = sum_vehicles_or_gen(output_man)
model_dfs_man = create_dataframes(total_shares_man)


#%% Plot the figure
fig, axs = plt.subplots(4, 4, figsize=(12, 12), sharey='row')
axs = axs.flatten()

left_most_indices = [0, 4, 8, 12]  # For a 4x4 grid

def plot_column(model_dfs, col, col_title):
    for idx, (model, model_df) in enumerate(model_dfs.items()):
        # Figure 1: Plot global shares of generation      
        ax = axs[left_most_indices[idx] + col]
        model_df_T = model_df.T
        model_df_T.plot(kind="area", ax=ax, color = colours[model], linewidth=0)
        
        ax.set_ylabel(ylabels[model])
        years_labels = [2020, 2030, 2040, 2050]
        years_labels_loc = [1, 10.5, 19.5, 29]
        ax.set_xticks(years_labels_loc)
        ax.set_xticklabels(years_labels)
        ax.xaxis.set_ticks_position('none') 
        ax.yaxis.set_ticks_position('none') 
        ax.tick_params(axis='both', which='major', pad=-3)
        
        # Remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Add faint white horizontal grid lines
        #ax.grid(axis='y', color='white', linestyle='-', linewidth=0.5)
        
        if col == 3:
            # Create individual legend to the right of each plot
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles, labels, title=legend_titles[model],
                      labelspacing=0.1, frameon=False,
                      bbox_to_anchor=(1.05, 1.1), loc='upper left')
            
           # Manually set the alignment of each text object in the legend
            for text in legend.get_texts():
                text.set_horizontalalignment('left')
        
        
            # Access the legend's title and increase the pad manually
            legend.get_title().set_position((0, 5))  # The numbers are x, y offsets
            
        else:
            # Explicitly do not create a legend for the first column
            ax.legend([], [], frameon=False)
            
        
        if idx == 0:
            ax.set_title(col_title)

plot_column(model_dfs_S0, 0, "A. Current trajectory")
plot_column(model_dfs_ct, 1, "B. + Carbon tax")
plot_column(model_dfs_sub, 2, "C. + Subsidies")
plot_column(model_dfs_man, 3, "D. + Mandates")

fig.subplots_adjust(wspace=0.08)  # Adjust the height spacing

output_file = os.path.join(fig_dir, "Shares_graph_4x4.svg")
output_file2 = os.path.join(fig_dir, "Shares_graph_4x4.png")

fig.savefig(output_file2, bbox_inches='tight')#, bbox_extra_artists=(lgd,), bbox_inches='tight')