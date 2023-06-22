import pandas as pd
import os
import numpy as np

# Get the path of the current script file
# script_path = os.path.abspath(__file__)

script_path = 'D:\WB\GitHub\FTT_StandAlone\Inputs'

# Set the working directory to the folder containing the script
working_dir = os.path.dirname(script_path)
os.chdir(working_dir)


# Specify the path to your Excel file
excel_file_path = 'Inputs\_MasterFiles\FTT-P\FTT-P-24x70_2021_S0.xlsx'

# Read the Excel file using pandas
data_frame = pd.read_excel(excel_file_path, sheet_name='BCET')

# Print the contents of the Excel file
print(data_frame)


# EFs in tCO2/GWh, constant accross countries, from tab BCET, in FTT-P-24x70_2021_S0.xlsx  
EF_file_path = 'Inputs\EmissionFactors.csv'

# Read the CSV file using pandas
EF = pd.read_csv(EF_file_path)

# Print the contents of the CSV file
print(EF)


# ______________________________________________________________________________
# Create carbon tax trajectories
# ______________________________________________________________________________



time_FTT = np.arange(2010, 2061).to_list()

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
# Calculating the tax
# ______________________________________________________________________________


categories_p = EF['Category'].to_list()

categories_df = pd.DataFrame (EF['Category'], columns=['Category'])



# Generate all possible combinations of categories and times
combinations = list(itertools.product(categories_p, time_FTT))

# Create the DataFrame
time_categories = pd.DataFrame(combinations, columns=['Category', 'Time'])

tax_categories = pd.merge(time_categories, tax, on=['Time'])

tax_cat_EF = pd.merge(tax_categories, EF, on=['Category'])

tax_cat_EF['Tax in 2013 $ per MWh']= tax_cat_EF['Tax1_US_2013']*tax_cat_EF['15 Emissions (tCO2/GWh)']/1000

# Wide format Only the tax expressed as 2013 USD per MWh:

# tax_cat_EF_wide =tax_cat_EF.melt(id_vars=['Time','Category'], value_vars ='Tax in 2013 $ per MWh')

tax_cat_EF_wide = tax_cat_EF.pivot(index='Category', columns='Time', values='Tax in 2013 $ per MWh')

tax_fin =pd.merge(categories_df, tax_cat_EF_wide, on=['Category'])

#iloc[0:24, 0:52]

