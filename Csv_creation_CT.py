# %%
# Import the needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import glob

# %%

######################################################################################################
################################### Import and format all databases ##################################
######################################################################################################

# Import EF data from the Excel file
name_sheet_FTTp_Master = 'BCET'
name_excel_file_EF = r'C:\Users\WB585192\OneDrive - WBG\GitHub_CPAT\FTT_StandAlone_v0.8\Inputs\_MasterFiles\FTT-P\FTT-P-24x70_2021_S0.xlsx'
df_EF = pd.read_excel (name_excel_file_EF, name_sheet_FTTp_Master)
df_EF.info()

#%%
EF = df_EF.iloc[4:28, 16].to_list()
print(EF)
path_directory = 'C:/Users/WB585192/OneDrive - WBG/GitHub_CPAT/FTT_StandAlone_v0.8/Inputs/S0/FTT-P'
files_mcocx = glob.glob( path_directory +"/MCOCX_*.csv")
for file in files_mcocx:
    #print(file)
    file_mcocx = pd.read_csv(file).iloc[0:24, 1:53]
    CT = file_mcocx.apply(lambda x: x * EF / 1000)
    print(CT)

    #for c in file_mcocx:

