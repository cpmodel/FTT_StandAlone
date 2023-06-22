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

script_path = 'D:\WB\GitHub\FTT_StandAlone\Inputs'
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


# ______________________________________________________________________________
# Create carbon tax trajectories
# ______________________________________________________________________________


time_FTT = np.arange(2010, 2061) #.to_list()

time_FTT = time_FTT.to_list() # This line used to work, but not anymore


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


#categories_p = EF['Category'].to_list()

categories_p=df_EF.iloc[4:28, 1].to_list()

categories_df = pd.DataFrame (categories_p, columns=['Category'])

EF_df = df_EF.iloc[4:28, [1,16]] # ***************

EF_df.columns = ['Category','EF']

EF_df.columns

# Generate all possible combinations of categories and times
combinations = list(itertools.product(categories_p, time_FTT))

# Create the DataFrame
time_categories = pd.DataFrame(combinations, columns=['Category', 'Time'])

tax_categories = pd.merge(time_categories, tax, on=['Time'])

tax_cat_EF = pd.merge(tax_categories, EF_df, on=['Category'])

tax_cat_EF['Tax in 2013 $ per MWh']= tax_cat_EF['Tax1_US_2013']*tax_cat_EF['EF']/1000

# Wide format Only the tax expressed as 2013 USD per MWh:

tax_cat_EF_wide = tax_cat_EF.pivot(index='Category', columns='Time', values='Tax in 2013 $ per MWh') # **** I'm loosing a column for some reason 

tax_fin =pd.merge(categories_df, tax_cat_EF_wide, on=['Category'])  # **** I'm loosing another column 

# tax_fin needs to be added to the mcocx files and saved in the folder with the cpat1 scenario ****


# ______________________________________________________________________________
#
# Create MCOCX file for each country/region with new CT in USD/MWh
# ______________________________________________________________________________

path_directory = 'Inputs/S0/FTT-P'
files_mcocx = glob.glob(path_directory + "/MCOCX_*.csv") #To recognize/select each file that starts with 'MCOCX_'

# For each MCOCX file multiply existing CT in USD/tCO2 by EF in USD/GWh divided by 1000 to have MWh
for file in files_mcocx:
    #print(file)
    file_mcocx = pd.read_csv(file).iloc[0:24, 1:53] #Select only the CT data across time and technology (exclude year and technologies)
    CT = file_mcocx.apply(lambda x: x * EF / 1000) #For each column of file_mcocx multiply it by EF/1000
    print(CT)



