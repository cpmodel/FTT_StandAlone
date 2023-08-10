# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:29:35 2023

Code for varying ambition between S0 and S2.

This can then be exported to country files using 
changed_input_export(compare_path, master_path) from country_inputs.py



@author: ib400
"""
#%%

import os
import pandas as pd
from openpyxl import load_workbook
from tqdm import tqdm
import time

os.chdir("C:/Users/ib400/OneDrive - University of Exeter/Documents/GitHub/FTT_StandAlone")



def regional_ambition(regions = {}, scenarios = ['S0', 'S2']):
    
    regions = regions
    scenarios = scenarios
    
    # trial values
    regions = {'US': 0.5,
               'CN': 0.5}
    scenarios = ['S0', 'S2']
    ambition = 0.5
    # seperate scenarios
    scenario_base = scenarios[0]
    scenario_compare = scenarios[1] 
    
    # load the comparison to establish bounds to vary within 
    comparison_path = f'Emulation/data/{scenario_base}_{scenario_compare}_comparison.xlsx'
    
    # laoding wb takes a lot of time and we need to load spreadsheets anyway
    compare_wb = load_workbook(comparison_path) 
    sheet_names = compare_wb.sheetnames
    
    # seperation in variable types
    simple_vars = ['MEWT', 'MEFI']
    complex_vars = ['MEWR', 'MWKA']
    new_sheets = []
    for sheet_name in sheet_names:
        
        
        #sheet_name = 'MEFI' # check for generalisability
        var_df = pd.read_excel(comparison_path, sheet_name=sheet_name)
        
        if sheet_name in simple_vars:
            
            for row in var_df[var_df['Scenario'] == 'S2'].index: # just chose scenario for readability
                print(row)
                meta = var_df.iloc[row, 0:5]
                #lower_bound = var_df.iloc[row]
                upper_bound = var_df.iloc[row]
                new_level = (upper_bound[5:] * ambition) # this is currently not GEn, neg numbers
                new_level_meta = pd.concat([meta, new_level])
                new_level_meta.loc['Scenario'] = f'{scenario_base}<{ambition}>{scenario_compare}'
                new_sheets.append(new_level_meta)
            
    new_sheet = pd.DataFrame(new_sheets)
            
#%% ## Possible developments

# Need to think about MGAMs - value range is a little different
# Also background variables - BCET etc. how do we vary these? Randomly, peturbation?



#### Argument additions:
    # variables of interest?
    





