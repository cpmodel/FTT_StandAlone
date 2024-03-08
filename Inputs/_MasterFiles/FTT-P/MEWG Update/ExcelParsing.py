# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:24:41 2024

@author: Jasleen
"""

import pandas as pd
import os

def RunOverDirectory(dir_path):
    try:
        # Iterate over all files in the folder
        for file_name in os.listdir(dir_path + "\RawFiles"):
            # Check if the file is a regular file (not a directory)
            if os.path.isfile(os.path.join(dir_path+ "\RawFiles", file_name)):
                namewoext, extension = os.path.splitext(file_name)
                file_path = os.path.join(dir_path+ "\RawFiles", file_name) 
                output_file_name = namewoext + ".xlsx"
                output_file_path = os.path.join(dir_path, output_file_name)
                AddRegions(file_path, sheet_name, output_file_path)


    except FileNotFoundError:
        print(f"Folder '{dir_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def ReadFile(file_path, sheet_name):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=5)

        # Iterate through each row
        timeRow = 0
        first_column_values = df.iloc[:, 0]
        for index, row in df.iterrows():
            if (row.iloc[0])=="Time":
                timeRow = index
                break
        
        return df

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def AddRegions(file_path, sheet_name, output_file_path):
    df = ReadFile(file_path, sheet_name)
    
    region_countries = ["Belarus"]
    region_name = "Rest of Annex I"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = ["Aruba", "Barbados", "El Salvador", "Dominican Republic", "Belize", "BES Islands", "Br Virgin Is", "Suriname", "Costa Rica", "Cuba", 
                        "Curacao", "Dominica", "Dominican Republic", "Grenada", "Guadeloupe", "Guatemala", "Haiti", "Honduras", "Jamaica", "Martinique",
                        "Nicaragua", "Paraguay", "Panama", "Puerto Rico", "St Kitts Nevis", "Bolivia", "Plurinational State of Bolivia", "Guyana", "Peru",
                        "South Georgia", "Uruguay", "Chile", "Ecuador", "Falklands Malv", "Trinidad Tobago", "Trinidad and Tobago"]
    region_name = "Rest Latin America"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = ["Brunei Darussalam", "Brunei", "Cambodia", "Lao PDR", "Laos", "Lao People's Democratic Republic", "Myanmar", 
                        "Philippines", "Singapore", "Thailand", "Viet Nam", "Vietnam"]
    region_name = "Rest of ASEAN"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = ["Algeria", "Libya"]
    region_name = "NorthAfrica OPEC"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = ["Angola", "Republic of the Congo", "Congo", "Gabon", "Equatorial Guinea", "Eq Guinea", "Eq. Guinea"] 
    region_name = "CentralAfricaOPEC"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = ["Cabo Verde", "Mauritania", "Morocco", "W. Sahara", "Tunisia", "Sudan"]
    region_name = "Rest North Africa"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = ["Cent Afr Rep", "Central African Rep.", "Cameroon", "Chad", "Sao Tome Prn"]
    region_name = "Rest Central Africa"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = ["Benin", "Burkina Faso", "Cote d Ivoire", "Côte d'Ivoire", "Guinea", "Guinea Bissau", "Guinea-Bissau", "Liberia",
                        "Mali", "Niger", "Gambia", "Ghana", "Togo", "Sierra Leone", "Côte d'Ivoire",  "Senegal"]
    region_name = "Rest West Africa"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = ["Burundi", "Djibouti", "Ethiopia", "Malawi", "Mozambique", "Rwanda", "Eritrea", "Somalia", "Somaliland", "South Sudan", 
                        "S. Sudan", "Uganda", "Zambia"]
    region_name = "Rest East Africa"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = ["Botswana", "Eswatini", "eSwatini", "Kingdom of Eswatini", "Lesotho",  "Madagascar", "Namibia", "Seychelles", "Tanzania", 
                        "United Republic of Tanzania", "Zimbabwe", "Mauritius"]
    region_name = "Rest South Africa"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = [ "North Korea", "Korea DPR", "Democratic People's Republic of Korea", "Timor Leste", "Timor-Leste", "Anguilla", "Antigua Barb", "Bahamas", "Cayman Is", 
                        "Montserrat", "St Barth", "St Lucia", "St Martin", "St Vincent Gren", "Turks Caicos", "Albania", "Andorra", 
                        "Palestine", "Qatar", "Yemen", "Amer Samoa", "Kiribati", "Nauru", "Niue", "Palau", "Papua N Guin", 
                        "Papua New Guinea", "Solomon Is", "Solomon Is.", "Tuvalu", "Maldives", "Mongolia", "Nepal", "Afghanistan", 
                        "Sri Lanka", "Uzbekistan", "Bangladesh", "Bhutan", "Kyrgyzstan", "Kirghizistan", "Turkmenistan", 
                        "Turkménistan", "Syria", "Syrian Arab Republic", "Moldova", "Republic of Moldova", "Armenia", "Tajikistan", "Tadjikistan", 
                        "Azerbaijan", "Georgia", "Bosnia Herzg", 'Bosnia and Herzegovina', "Bosnia and Herz.", "Kosovo*", "Kosovo", 
                        "Fiji", "Fr Polynesia", "Marshall Is", "Moldova Rep", "Montenegro", "Serbia", "Samoa", "Israel", "Jordan",  
                        "Mayotte",  "Micronesia", "Lebanon",  "Syrian AR", "St Pierre Mq", "Cook Is", "Tokelau", "Tonga", "Vanuatu",
                        "Fr. S. Antarctic Lands"]
    region_name = "Rest of World"
    df = GetRegionValue(df, region_countries, region_name)
    
    region_countries = ["Iraq", "Bahrain", "Iran IR", "Iran",  "Islamic Republic of Iran", "Kuwait", "Oman", "Venezuela", "Bolivarian Republic of Venezuela"]       
    region_name = "Rest of OPEC"
    df = GetRegionValue(df, region_countries, region_name)
    
    # Write the updated DataFrame to a new Excel file
    df.to_excel(output_file_path, index=False)



def GetRegionValue(df, region_countries, region_name):
    # Search for rows which match the region_countries list
    mask = df["Time"].isin(region_countries)
    matching_rows = df[mask]
    
    # Add the filtered rows
    result = matching_rows.sum(axis=0)
    
    # Replace non-number values to ..
    for idx, val in result.items():
        if not pd.api.types.is_numeric_dtype(type(val)):
            result[idx] = ".."
    
    # Change the name of the fiest column to the region_name
    result.iloc[0] = region_name
    result.iloc[1] = ""
    
    # Append the resultant row in the dataframe
    df = df.append(result, ignore_index=True)
    
    return df


# Main script
sheet_name = "UKDS.Stat export"  
current_directory = os.getcwd()

RunOverDirectory(current_directory + "\DataFiles")





