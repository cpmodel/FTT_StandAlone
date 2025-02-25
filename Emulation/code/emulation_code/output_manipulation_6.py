import pickle
import pandas as pd
import os
import numpy as np
import sys

# Set root directory
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
os.chdir(root_dir)

import SourceCode.support.titles_functions as titles_f
titles = titles_f.load_titles()

#%%

# Function to load the pickle file
def load_pickle_file(output_path):
    with open(output_path, 'rb') as file:
        output = pickle.load(file)
    return output

# Function to filter the output data
def filter_output_data(output, vars_to_compare):
    filtered_output = {key: value for key, value in output.items() if key in vars_to_compare}
    return filtered_output

# Function to convert output data to DataFrame
def convert_output_to_dataframe(output_data, scen, titles):
    scenario_list = []
    variable_list = []
    country_list = []
    country_short_list = []
    technology_list = []
    value_list = []
    year_list = []
    

    for variable, dimensions in output_data.items():

        print(f'Converting {variable} for {scen}')
        if variable == 'MEWW':
            indices = np.indices(dimensions.shape).reshape(dimensions.ndim, -1).T

            # Iterate over the indices and extract values
            for index in indices:
                # Index corresponds to dimension in the np array
                value = dimensions[tuple(index)]
                
                # Append data to lists as though accessing dimensions of vars
                scenario_list.append(scen)
                variable_list.append(variable)
                country_list.append('Global') 
                country_short_list.append('GBL')
                tech = index[1] 
                technology_list.append(titles['T2TI'][tech])  
                year = index[3] + 2010
                year_list.append(year)
                
                # Append value to the value list
                value_list.append(value) 
                
        else:
            # Flatten the array and get the indices
            indices = np.indices(dimensions.shape).reshape(dimensions.ndim, -1).T
    
            # Iterate over the indices and extract values
            for index in indices:
                
                # Index corresponds to dimension in the np array
                value = dimensions[tuple(index)]
                # Append data to lists as though accessing dimensions of vars
                scenario_list.append(scen)
                variable_list.append(variable)
                country = index[0]
                country_list.append(titles['RTI'][country])  
                country_short_list.append(titles['RTI_short'][country])
                tech = index[1] 
                technology_list.append(titles['T2TI'][tech])  
                year = index[3] + 2010
                year_list.append(year)
                
                # Append value to the value list
                value_list.append(value)
    
    df = pd.DataFrame({
    'scenario': scenario_list,
    'variable': variable_list,
    'country': country_list,
    'country_short' : country_short_list,
    'technology': technology_list,
    'year': year_list,
    'value': value_list
    })
    
    return df

# Function to save the DataFrame to a CSV file
def save_dataframe_to_csv(df, output_dir, scen):
    output_file = os.path.join(output_dir, f'{scen}_batch.csv')
    df.to_csv(output_file, index=False)
    print(f'Batch {scen} saved')

# Main function to process one results file at a time and save as a batch
def process_and_save_results_file(scen, vars_to_compare, output_dir, titles):
    output_path = f'Output/Results_{scen}_core.pickle'
    
    # Load the pickle file
    output = load_pickle_file(output_path)
    
    # Filter the output data
    filtered_output = filter_output_data(output, vars_to_compare)
    
    # Convert the filtered output data to DataFrame
    df = convert_output_to_dataframe(filtered_output, scen, titles)
    
    # Save the DataFrame to a CSV file
    save_dataframe_to_csv(df, output_dir, scen)

# Example usage
if __name__ == "__main__":
    # Define the scenarios and variables to compare
    scen_levels = pd.read_csv('Emulation/data/scenarios/S3_scen_levels.csv')
    emulation_scens = scen_levels['scenario']
    vars_to_compare = ['MEWS', 'MEWK', 'MEWG', 'MEWE', 'MEWW', 'METC', 'MEWC', 'MECW', "MEWP"]
    output_dir = 'Emulation/data/runs'
    
    # Process each results file one at a time and save as a batch
    for scen in scen_levels['scenario']:
        process_and_save_results_file(scen, vars_to_compare, output_dir, titles)