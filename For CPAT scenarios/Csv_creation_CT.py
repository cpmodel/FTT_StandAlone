# %%
# Import the needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import glob

# %%

# ______________________________________________________________________________
#
# Extract the Emission Factor data from Master file
# ______________________________________________________________________________

# Import EF data from the Master file
name_sheet_FTTp_Master = 'BCET'
name_excel_file_EF = r'C:\Users\WB585192\OneDrive - WBG\GitHub_CPAT\FTT_StandAlone_v0.8\Inputs\_MasterFiles\FTT-P\FTT-P-24x70_2021_S0.xlsx' #Read path
df_EF = pd.read_excel (name_excel_file_EF, name_sheet_FTTp_Master) #Extract the info from BCET spreadsheet
df_EF.info()

# Extract only the vector EF
EF = df_EF.iloc[4:28, 16].to_list()
print(EF)

# ______________________________________________________________________________
#
# Create MCOCX file for each country/region with new CT in USD/MWh
# ______________________________________________________________________________

path_directory = 'C:/Users/WB585192/OneDrive - WBG/GitHub_CPAT/FTT_StandAlone_v0.8/Inputs/S0/FTT-P'
files_mcocx = glob.glob(path_directory + "/MCOCX_*.csv") #To recognize/select each file that starts with 'MCOCX_'

# For each MCOCX file multiply existing CT in USD/tCO2 by EF in USD/GWh divided by 1000 to have MWh
for file in files_mcocx:
    #print(file)
    file_mcocx = pd.read_csv(file).iloc[0:24, 1:53] #Select only the CT data across time and technology (exclude year and technologies)
    CT = file_mcocx.apply(lambda x: x * EF / 1000) #For each column of file_mcocx multiply it by EF/1000
    print(CT)

