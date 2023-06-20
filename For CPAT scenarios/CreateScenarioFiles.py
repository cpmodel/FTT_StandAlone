import pandas as pd

# Specify the path to your Excel file
excel_file_path = 'D:\WB\GitHub\FTT_StandAlone\Inputs\_MasterFiles\FTT-P\FTT-P-24x70_2021_S0.xlsx'

# Read the Excel file using pandas
data_frame = pd.read_excel(excel_file_path, sheet_name='BCET')

# Print the contents of the Excel file
print(data_frame)
