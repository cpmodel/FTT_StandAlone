# %%
# Import the needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
# %%

######################################################################################################
################################### Import and format all databases ##################################
######################################################################################################

script_path = 'D:\WB\GitHub\FTT_StandAlone\Inputs'
working_dir = os.path.dirname(script_path)
os.chdir(working_dir)

# ______________________________________________________________________________
# Create carbon tax trajectories
# ______________________________________________________________________________
time_FTT = np.arange(2010, 2061)

# Factor to convert from 2021 to 2013 dollars. NGDP_D in CPAT

factor_dollars = 101.751156110395/118.369897000613


# Define the x and y coordinates of the two points
x1 = [2023, 2035]  # initial and final year of the tax
y1 = [5, 25]      # initial and final value of the tax, $/ton CO2 in 2021 dollars



# Linear interpolation to estimate the tax in time

tax = pd.DataFrame({'Time': time_FTT, 'Tax1': np.interp(time_FTT, x1, y1)})

tax['Tax1']=  np.where(tax['Time'] < 2023, 0, tax['Tax1'])

tax['Tax1_US_2013']=tax['Tax1']*factor_dollars

tax_fin = tax['Tax1_US_2013'] # Tax in US2013 $/ tCO2


#-------------------------------------------------------------------------------

# Import EF data from the Master file
# EFs in tCO2/GWh

name_sheet_FTTp_Master = 'BCET'
name_excel_file_EF = r'Inputs\_MasterFiles\FTT-P\FTT-P-24x70_2021_S0.xlsx'
df_EF = pd.read_excel (name_excel_file_EF, name_sheet_FTTp_Master)
df_EF.info()

#%%
EF = df_EF.iloc[4:28, 16].to_list()
print(EF)
path_directory = 'Inputs/S0/FTT-P'
files_mcocx = glob.glob( path_directory +"/MCOCX_*.csv")
for file in files_mcocx:
    #print(file)
    file_mcocx = pd.read_csv(file).iloc[0:24, 1:53]
    CT = file_mcocx.apply(lambda x: x+tax_cat_EF_wide / 1000)
    print(CT)

    #for c in file_mcocx:

