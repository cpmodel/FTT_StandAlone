import os
import pandas as pd
import xlsxwriter

os.chdir("C:/Users/ib400/OneDrive - University of Exeter/Documents/GitHub/FTT_StandAlone")

# Change depending on scenario
#scen = 'S2' ## this needs sorting and the scens need to be generalised
scenario_list = ['S0', 'S3']

# create dictionaries to store input files
# dictionaries used for generalisaiotn and multiple scenarios
file_paths = {}
xls = {}
# Load the Excel workbook
for scen in scenario_list:
    file_paths[scen] = f"Inputs/_Masterfiles/FTT-P/FTT-P-24x70_2021_{scen}.xlsx"
    xls[scen] = pd.ExcelFile(file_paths[scen])

# Dictionary to store DataFrames with variable names based on scenarios
dataframes = {}

# initialise output workbook
#output_workbook_S2 = []
for scen in scenario_list:
    dataframes[f"df_{scen}"] = []

# Specify the sheet names to process
sheets_to_process = ['BCET', 'MEWA', 'MGAM', 'MEWT', 'MEWR', 'MWKA', 'MEFI']  # Add the desired sheet names here
# MWDD done seperately due to difference in structure

#%% ## take input masterfiles and wrangle to remove unneccessary info. 
###### Will need recreating by reversing process after variation

# Iterate over the desired sheets and apply the transformations
for scen in scenario_list:
    for sheet_name in sheets_to_process:
        # Load the sheet into a DataFrame
        df = pd.read_excel(file_paths[scen], sheet_name=sheet_name, header=None)
    
        # Remove the index 0
        df = df.iloc[4:]
        # Reset the index of the DataFrame
        df = df.reset_index(drop=True)
    
        # Extract every 36th row from the column
        numb_col = df[0].iloc[::36].reset_index(drop=True)
        country_col = df[1].iloc[::36].reset_index(drop=True)
        
        # Repeat the extracted values for the next 36 rows
        numb_col = numb_col.repeat(36).reset_index(drop=True)
        country_col = country_col.repeat(36).reset_index(drop=True)
        
        # Insert the new column as the first 2 columns in the DataFrame
        df[0] = numb_col
        df.insert(1, 'Country', country_col)
        
        # Delete the rows that were used to create the new column
        rows_to_delete = df.index[36::36]
        df = df.drop(rows_to_delete)
        
        # Set the first row as column names
        df.columns = df.iloc[0]
        df.columns.values[0:3] = ['Code', 'Country', 'Technology']
        
        # Delete the first row from the DataFrame
        df = df.iloc[1:].reset_index(drop=True)
        
        dataframes[f"df_{scen}"].append(df)
        
    #### Code for adding MWDD sheet which has different structure
    df = pd.read_excel(file_paths[scen], sheet_name= 'MWDD', header=None)
    # remove sheer titles
    df = df.iloc[4:].reset_index(drop=True).drop(df.columns[0], axis=1)
    # drop and rename cols
    df.columns = df.iloc[0]
    df.columns.values[0] = 'Technology'
    # Delete the first row from the DataFrame
    df = df.iloc[1:].reset_index(drop=True)
    
    # Add to output
    dataframes[f"df_{scen}"].append(df)

#%%
#### create new excels for inspection

sheets_to_output = ['BCET', 'MEWA', 'MGAM', 'MEWT', 
                    'MEWR', 'MWKA', 'MEFI', 'MWDD']  # Add the desired sheet names here

for scen in scenario_list:
    # Create a new Excel writer object
    with pd.ExcelWriter(f'Emulation/data/output_workbook_{scen}.xlsx') as writer:
    # Iterate through the sheet names and output list
        for sheet_name, output in zip(sheets_to_output, dataframes[f"df_{scen}"]):
        
            # Create a new DataFrame from the output
            df = pd.DataFrame(output)
            
            # Write the DataFrame to a new sheet in the output workbook
            df.to_excel(writer, sheet_name=sheet_name, index=False)


#%% Possible developments

### This code needs streamlining to be just a function that can be called in the run file
## Can the below code be used as a simpler way of manipulating data?
##  master = pd.read_excel(master_path, sheet_name = sheet_name, 
                        usecols=lambda col: col not in [1], skiprows=4) # get this into input_wrangle.py
 #master.columns = ['Technology'] + list(master.columns[1:]) # do we need to rename, can we search by index later??








