import pandas as pd
import os

# Get the path of the current script file
# script_path = os.path.abspath(__file__)

script_path = 'D:\WB\GitHub\FTT_StandAlone\Inputs'

# Set the working directory to the folder containing the script
working_dir = os.path.dirname(script_path)
os.chdir(working_dir)


# Specify the path to your Excel file
excel_file_path = '_MasterFiles\FTT-P\FTT-P-24x70_2021_S0.xlsx'

# Read the Excel file using pandas
data_frame = pd.read_excel(excel_file_path, sheet_name='BCET')

# Print the contents of the Excel file
print(data_frame)


# EFs in tCO2/GWh, constant accross countries, from tab BCET, in FTT-P-24x70_2021_S0.xlsx  
EF_file_path = 'EmissionFactors.csv'

# Read the CSV file using pandas
EF = pd.read_csv(EF_file_path)

# Print the contents of the CSV file
print(EF)


# ______________________________________________________________________________
# Create carbon tax trajectories
# ______________________________________________________________________________

import numpy as np

time_FTT = np.arange(2010, 2061)
time_int = np.arange(2023, 2036)

# Factor to convert from 2021 to 2013 dollars. NGDP_D in CPAT

factor_dollars = 101.751156110395/118.369897000613


# Define the x and y coordinates of the two points
x1 = [2023, 2035]  # initial and final year of the tax
y1 = [5, 25]      # initial and final value of the tax, $/ton CO2 in 2021 dollars



# Linear interpolation to estimate the tax in time

tax = pd.DataFrame({'Time': time_FTT, 'Tax1': np.interp(time_FTT, x1, y1)})

tax['Tax1']=  np.where(tax['Time'] < 2023, 0, tax['Tax1'])

tax['Tax1_US_2013']=tax['Tax1']*factor_dollars

# ______________________________________________________________________________
# ...
# ______________________________________________________________________________

final = pd.DataFrame({'Time': time_FTT, 'Category': EF['Category']})

test=pd.DataFrame(data= 0,columns=EF['Category'], index=time_FTT)

#data= EF.T['15 Emissions (tCO2/GWh)']


# ==================================================================
# T E S T S
# ==================================================================

# Create the first DataFrame
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': ['X', 'Y', 'Z']})
print("DataFrame 1:")
print(df1)

# Create the second DataFrame
df2 = pd.DataFrame({'A': [1, 2, 4], 'C': ['P', 'Q', 'R']})
print("\nDataFrame 2:")
print(df2)

# Perform a left join on 'A' column
result = pd.merge(df1, df2, on='A', how='left')

# Print the result
print("\nLeft Join Result:")
print(result)

# Otro

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
df
# Perform if statement using value condition
if_condition = df['A'] > 3

# Apply the condition to the DataFrame
filtered_df = df[if_condition]

# Print the filtered DataFrame
print(filtered_df)
