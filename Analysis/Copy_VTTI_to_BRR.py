# This script copies all the TVTT csv files in the C:\Users\Work profile\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone\Inputs\S0\FTT-Tr\ folder
# And rename them to Base registration tax.csv

import os
import shutil

# Define the source and destination folders
source_folder = r"C:\Users\Work profile\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone\Inputs\S0\FTT-Tr"

# Get the list of files in the source folder
files = os.listdir(source_folder)

# Loop through the files, find the files starting with TVTT, and copy them. The new name is Base registration rate_CC.csv, where CC is the country code
# The country code is the same last two characters of the file name
for file in files:
    if file.startswith("TTVT"):
        country_code = file[-6:-4]
        new_name = "Base registration rate_" + country_code + ".csv"
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(source_folder, new_name)
        shutil.copyfile(source_path, destination_path)
        print(f"File {file} copied to {new_name}")


