# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:18:52 2024

@author: pv + ib
"""

from collections import OrderedDict
import configparser
import csv
import datetime
import json
import os
from pathlib import Path
import pickle
import sys
import time
from threading import Timer, Thread
import webbrowser

from bottle import (route, run, request, response, static_file)
import numpy as np
import pandas as pd

from SourceCode.support import titles_functions

#%%

if __name__ == '__main__':

    dirmain = os.path.dirname(os.path.realpath(__file__))
    diroutput = os.path.join(dirmain, "Output")
    
    # Read titles to eventually create 2D dataframes
    titles = titles_functions.load_titles()
    
    # time horizon
    tl = np.arange(2010, 2027+1)
    
    # Read pickle file
    with open('Output\Gamma.pickle', 'rb') as f:
        output = pickle.load(f)
        
    # Scenarios
    scens = list(output.keys())
        
    # FTT:P gamma values
    mgam = dict()
    mgamx = dict()
    bcet = dict()
    mewg =  dict()
    
    for scen in scens:
        
        mgam[scen] = dict()
        bcet[scen] = dict()
        mewg[scen] = dict()
        
        for r, reg in enumerate(titles["RTI"]):
            
            bcet[scen][reg] = pd.DataFrame(output[scen]["BCET"][r, :, :, 0],
                                           index=titles["T2TI"],
                                           columns=titles["C2TI"])
            
            mgam[scen][reg] = pd.DataFrame(output[scen]["MGAM"][r, :, 0, :], 
                                           index=titles["T2TI"], 
                                           columns=tl)
        
            # mgamx[scen][reg] = pd.DataFrame(output[scen]["MGAMX"][r, :, 0, :], 
            #                                index=titles["T2TI"], 
            #                                columns=tl)    
    
            mewg[scen][reg] = pd.DataFrame(output[scen]["MEWG"][r, :, 0, :], 
                                           index=titles["T2TI"], 
                                           columns=tl)    

#%%

for r, reg in enumerate(titles["RTI"]):

    country = titles['RTI_short'][r]
    
    
    df = pd.DataFrame(columns=range(2001,2101), index = mgam['Gamma'][reg].index)
    df.index.name = country
    # Merge the blank DataFrame with the DataFrame from gamma
    merged_df = df.combine_first(mgam['Gamma'][reg])
    # Forward and back fill values for whole row
    merged_df = merged_df.ffill(axis=1)
    merged_df = merged_df.bfill(axis=1) 
    
    # Save the merged DataFrame to a CSV file
    file_path = f'Inputs/S0/FTT-P/MGAM_{country}.csv'
    merged_df.to_csv(file_path, index=True)


#%% Save to masterfile
n_regions = 71

 # Model location, assuming you have the support repo and main repo in same folder
model_to_change = "..\FTT_StandAlone" 
dir_excel_S0 = os.path.join(model_to_change, "Inputs\_Masterfiles\FTT-P\FTT-P-24x71_2023_S0.xlsx")
dir_excel_S1 = os.path.join(model_to_change, "Inputs\_Masterfiles\FTT-P\FTT-P-24x71_2023_S1.xlsx")
dir_excel_S2 = os.path.join(model_to_change, "Inputs\_Masterfiles\FTT-P\FTT-P-24x71_2023_S2.xlsx")
excel_sheets_to_change = [dir_excel_S0] #, dir_excel_S1, dir_excel_S2]

# Note: the indexing will be changed from 2024 onwards. 
onshore_ind = 21
offshore_ind = 22
solar_ind = 23
thermal_ind = 24

print_to_excel  = True
if print_to_excel:
    print("Writing data to excel files, overwriting old files")
 
    for dir_excel in excel_sheets_to_change:
        book = load_workbook(dir_excel)    
        

        MGAM_excel = np.array(pd.read_excel(dir_excel, sheet_name='MGAM'))
            
        # Write generation and capacity to excel sheets
        with pd.ExcelWriter(dir_excel, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:

            for i in np.arange(0, n_regions):
                onshore_generation.iloc[[i]].to_excel(writer, sheet_name="MGAM", \
                                                      startcol=12, startrow=onshore_ind+i*36, header=False, index=False)
                offshore_generation.iloc[:,:-1].iloc[[i]].to_excel(writer, sheet_name="MEWG", \
                                                      startcol=12, startrow=offshore_ind+i*36, header=False, index=False)
                solar_generation.iloc[:,:-1].iloc[[i]].to_excel(writer, sheet_name="MEWG", \
                                                      startcol=12, startrow=solar_ind+i*36, header=False, index=False)
                csp_generation.iloc[:,:-1].iloc[[i]].to_excel(writer, sheet_name="MEWG",\
                                                      startcol=12, startrow=thermal_ind+i*36, header=False, index=False)
                