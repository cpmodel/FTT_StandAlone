# %%
# Import the needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import itertools
# %%

######################################################################################################
################################### Import and format all databases ##################################
######################################################################################################

script_path = r'C:\Users\WB585192\OneDrive - WBG\GitHub_CPAT\FTT_StandAlone_v0.8\Inputs'
#script_path = r'D:\WB\GitHub\FTT_StandAlone\Inputs'
working_dir = os.path.dirname(script_path)
os.chdir(working_dir)

# ______________________________________________________________________________
#
# Extract the Emission Factor data from Master file
# ______________________________________________________________________________

# Import EF data from the Master file
# EFs in tCO2/GWh

name_sheet_FTTp_Master = 'BCET'
name_excel_file_EF = r'Inputs\_MasterFiles\FTT-P\FTT-P-24x70_2021_S0.xlsx'
df_EF = pd.read_excel (name_excel_file_EF, name_sheet_FTTp_Master)
df_EF.info()

# Extract only the vector EF
EF = df_EF.iloc[4:28, 16].to_list()
print(EF)
print(len(EF)) 

# ______________________________________________________________________________
# Create carbon tax trajectories
# ______________________________________________________________________________

time_FTT = np.arange(2010, 2061)
time_FTT = time_FTT.tolist() #I change to_list() to tolist() and it works

# Factor to convert from 2021 to 2013 dollars. NGDP_D in CPAT
factor_dollars = 101.751156110395/118.369897000613

# Define the x and y coordinates of the two points
x1 = [2023, 2035]  # initial and final year of the tax
y1 = [5, 25]       # initial and final value of the tax, $/ton CO2 in 2021 dollars

# Linear interpolation to estimate the tax in time
tax = pd.DataFrame({'Time': time_FTT, 'Tax1': np.interp(time_FTT, x1, y1)})
tax['Tax1']=  np.where(tax['Time'] < 2023, 0, tax['Tax1'])
tax['Tax1_US_2013']=tax['Tax1']*factor_dollars

print(tax)

# ______________________________________________________________________________
# Convert the unit in USD/MWh
# ______________________________________________________________________________

# Extract the column as a Series
Tax1_US_2013 = pd.DataFrame(tax['Tax1_US_2013'])

# Transpose the Series into a row DataFrame
T_Tax1_US_2013 = Tax1_US_2013.T
# Repeat the row 28 times to match the number of technologies
tax_data = pd.concat([T_Tax1_US_2013] * 24, ignore_index=True)
# Check the length of the dataframe tax_data
print(len(tax_data))
# Convert the unit of the CT (USD/tCO2) in USD/MWh
CT = tax_data.apply(lambda x: x * EF / 1000)
# Set column indices from 2010 to 2060
CT.columns = range(2010, 2061) 
print(CT)

# ______________________________________________________________________________
# Create MCOCX files with CT in USD/MWh and add the CT trajectory
# ______________________________________________________________________________

## CHANGE HERE ##
path_directory = 'C:/Users/WB585192/OneDrive - WBG/GitHub_CPAT/FTT_StandAlone_v0.8/Inputs'
files_mcocx = sorted(glob.glob(path_directory + "/MCOCX/MCOCX_*.csv"))        #To recognize/select each file that starts with 'MCOCX_'

output_path = 'C:/Users/WB585192/OneDrive - WBG/GitHub_CPAT/FTT_StandAlone_v0.8/Inputs/S0/FTT-P/'

tech_column = pd.read_csv(files_mcocx[0]).iloc[:,0]
result = pd.DataFrame()
result['Technology'] =  tech_column

# For each MCOCX file multiply existing CT in USD/tCO2 by EF in USD/GWh divided by 1000 to have MWh
for file in sorted(files_mcocx):                               
    file_name = file.split("\\")[1]                                     # Get the file name without extension
    output_file = output_path + file_name                               # Output CSV file name
    file_mcocx = pd.read_csv(file)                                      # Select only the CT data across time and technology (exclude year and technologies)
    years = range(2010, 2061) 

    file_mcocx = file_mcocx.iloc[:,1:]                                  # Add the dataframe to file_mcocx, starting from column 2
    file_mcocx.columns = years                                          # Set column indices from 2010 to 2060

    CT_x, file_mcocx = CT.align(file_mcocx, axis=0, fill_value=0)       # Align CT and file_mcocx indices
    file_mcocx = file_mcocx.add(CT_x)                                   # Sum the CT trajectory to the original dataset

    result1 = pd.concat([result, file_mcocx], axis=1)                    # Merge the first column with techno + the dataset
    result1.to_csv(output_file, index=False)                             # Create csv files
    print(f"Processed file: {file}. Output saved as: {output_file}")    # Print output

    