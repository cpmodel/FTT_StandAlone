# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:42:53 2023

@author: Jasleen Kaur
"""
import pandas as pd
import csv
import os

# Load the Excel file into a pandas DataFrame
file_path = r"C:\Users\wb614170\FTT_StandAlone\Inputs\_MasterFiles\FTT-Tr\Jasleen\RTFS_All_Countries.xlsx"
df = pd.read_excel(file_path)


fuel_list = df.columns[2:]

first_line_rtfs = [""]
for year in range(2001,2101):
    first_line_rtfs.append(year)

for index, row in df.iterrows():
    rtfs_file_name = "./RTFS_Files/RTFS_" + row['Region Code'] + ".csv"
    old_bttc_file_name = "./Old_BTTC_Files/BTTC_" + row['Region Code'] + ".csv"
    new_bttc_file_name = "./New_BTTC_Files/BTTC_" + row['Region Code'] + ".csv"

    if os.path.exists(old_bttc_file_name):
        with open(old_bttc_file_name,'r') as old_bttc_file:  
            with open(rtfs_file_name, 'w', newline='') as rtfs_file, open(new_bttc_file_name, 'w', newline='') as new_bttc_file:
                bttc_writer = csv.writer(new_bttc_file, lineterminator='\n')
                bttc_reader = csv.reader(old_bttc_file)
                bttc_row = next(bttc_reader)
                bttc_row.append('18 Fuel subsidy (USD/km)')
                bttc_writer.writerow(bttc_row)
    
                rtfs_writer = csv.writer(rtfs_file)
                rtfs_writer.writerow(first_line_rtfs)
                
                for fuel in fuel_list:
                    bttc_row = next(bttc_reader)
                    bttc_row.append(row[fuel])
                    bttc_writer.writerow(bttc_row)
                        
                    line = [fuel]
                    for year in range(2001,2023):
                        line.append(round(row[fuel], 10))
                    
                    reductionAmount = 0.2*row[fuel]
                    prevValue = row[fuel]
                    for year in range(2023,2101):
                        line.append(round(prevValue, 10))
                        if prevValue > 0:
                            prevValue = prevValue - reductionAmount
                        
                        if prevValue < 0:
                            prevValue = 0
                        
                        
                    rtfs_writer.writerow(line)
