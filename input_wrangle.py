import os
import pandas as pd

os.chdir("C:/Users/ib400/OneDrive - University of Exeter/Documents/GitHub/FTT_StandAlone")
# Load the Excel workbook
# Change depending on scenario
scen = 'S2'
#file_path = "Inputs/_Masterfiles/FTT-P/FTT-P-24x70_2021_S0.xlsx"
file_path = "Inputs/_Masterfiles/FTT-P/FTT-P-24x70_2021_S2.xlsx"

xl = pd.ExcelFile(file_path)


#%%
#### Cell for adding main sheets of master input file to trimmed output
# initialise output workbook
output_workbook = []
# Specify the sheet names to process
sheets_to_process = ['BCET', 'MEWA', 'MGAM', 'MEWT', 'MEWR', 'MWKA', 'MEFI']  # Add the desired sheet names here
#sheets_to_process = ['BCET']
# Initialize an empty list to store the modified DataFrames

# Iterate over the desired sheets and apply the transformations
for sheet_name in sheets_to_process:
    # Load the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

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
    
    output_workbook.append(df)

#%%
#### Code for adding MWDD sheet which has different structure
df = pd.read_excel(file_path, sheet_name= 'MWDD', header=None)
# remove sheer titles
df = df.iloc[4:].reset_index(drop=True).drop(df.columns[0], axis=1)
# drop and rename cols
df.columns = df.iloc[0]
df.columns.values[0] = 'Technology'
# Delete the first row from the DataFrame
df = df.iloc[1:].reset_index(drop=True)

# Add to output
output_workbook.append(df)


#%%
#### create new excel for inspection

sheets_to_output = ['BCET', 'MEWA', 'MGAM', 'MEWT', 
                    'MEWR', 'MWKA', 'MEFI', 'MWDD']  # Add the desired sheet names here

# Create a new Excel writer object
writer = pd.ExcelWriter('output_workbook_S2.xlsx', engine='xlsxwriter')

# Iterate through the sheet names and output list
for sheet_name, output in zip(sheets_to_output, output_workbook):
    # Create a new DataFrame from the output
    df = pd.DataFrame(output)

    # Write the DataFrame to a new sheet in the output workbook
    df.to_excel(writer, sheet_name=sheet_name, index=False)

# Save the output workbook
writer.save()





