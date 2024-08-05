# -*- coding: utf-8 -*-
"""
This script copies all the files from the S0 folder to the subfolders
in case the files do not exist yet.

It also allows copying the MSAL files around. Note there are types
@author: Femke
"""

import shutil, os, glob
import os.path
import numpy as np
import pandas as pd


current_dir = os.path.dirname(os.path.realpath(__file__))
# The input dir is found by going up a directly to the parent directory and then going to the Input folder
top_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
input_dir = os.path.join(top_dir, "Inputs")



def create_file_list(variable, source_dir):
    "Create list of files to be copied from S0 for a specific variable"
    GLOB_PARMS = f"{variable}*.csv"
    file_list = glob.glob(os.path.join(source_dir, GLOB_PARMS))
    return file_list

def get_source_dir(input_dir, base_scen, model):
    source_dir = os.path.join(input_dir, "S0", model)
    return source_dir

def copy_csv_files_to_scen(model, variable, scen_name, source_dir):
    """Copy all country files for a single variable to the new scenario"""
    file_list = create_file_list(variable, source_dir)
    desti_dir = os.path.join(input_dir, scen_name, model) 
    
    for file in file_list:
        # Create desti_dir if it does not exist
        if not os.path.exists(desti_dir):  
            os.makedirs(desti_dir)
        shutil.copy(file, desti_dir)

#%% Copying the policy files
source_dir = get_source_dir(input_dir, "S0", "FTT-P")

def save_new_file(model, scen_name, file, df):
    "Create destination directory, and save the file"
    
    desti_dir = os.path.join(input_dir, scen_name, model)
    # Extract the filename from the original file path
    filename = os.path.basename(file)
    # Create the full destination file path
    dest_file_path = os.path.join(desti_dir, filename)
    # Save the DataFrame to the new file path
    df.to_csv(dest_file_path, index=False)


def change_csv_files(model, scen_name, source_dir, variable, policy):
    file_list = create_file_list(variable, source_dir)
    for file in file_list:
        # Read in file
        df = pd.read_csv(file)
        df = policy_change(df, policy)
        save_new_file(model, scen_name, file, df)


def policy_change(df, policy):
    carbon_price = 200.0     # Constant €200 per tonne CO2 
    match policy:
        
        case "REPP":  # A linearly increasing price to €200 per tonne CO2, i.e.  
            price_2050 = 200.0
            price_2023 = df.iloc[:, 14] / 3.667 # Note, REPP is given per tC, rather than tCO2
            
            # Reshape the price_2023 to a column vector
            price_2023 = price_2023.values.reshape(-1, 1)
            
            # Linearly increase the price from 2023 to 2050 values. 
            df.iloc[:, 14:42] = ( price_2023 + (price_2050 - price_2023) / 27.0 * np.arange(28) ) * 3.667 
            # After 2050, continue everywhere with equal yearly increases, equal to price_2050/27
            df.iloc[:, 42:] = ( price_2050 + price_2050 / 27.0 * np.arange(1, 21) ) * 3.667      
            
        case "REPP2":  # A linearly increasing price to €200 per tonne CO2, i.e.  
            price_2050 = 200.0
                        
            # Linearly increase to €200 per tonne CO2 
            df.iloc[:, 14:42] = (price_2050) / 27.0 * np.arange(28) * 3.667 
            # After 2050, continue everywhere with equal yearly increases, equal to price_2050/27
            df.iloc[:, 42:] = ( price_2050 + price_2050 / 27.0 * np.arange(1, 21) ) * 3.667       
    
        case "Power REPP":
            df.iloc[:, 15:] = carbon_price * 3.667 
        
        
        # Power sector policies
        case "MEWR strong":     # Completely outregulate fossil technologies from 2024
            df.iloc[1:8, 24:] = 0
        case "MEWT":           # Subsidize all renewables
            #df.iloc[ 8:18, 24:] = -0.3
            #df.iloc[19:22, 24:] = -0.3
            df.iloc[ 8:22, 24:] = -0.3
        case "Coal phase-out":
            df.iloc[0, 1] = 1       # The coal phase-out is coded as a function; this switch turns it on 
      
        
        # Transport policies
        case "TREG strong":
            df.iloc[:15, 24:] = 0
        case "TWSA strong":
            df.iloc[18:21, 24:] = 0 # (exogenous sales in k-veh) TODO: I will need to figure out what a reasonable mandate is. 
        case "BRR strong tax": 
            df.iloc[:15, 24:] = 0.3
        case "BRR strong subsidy":
            df.iloc[18:21, 24:] = -0.3
        case "BRR strong combo":
            df.iloc[:15, 24:] = 0.3
            df.iloc[18:21, 24:] = -0.3
        case "EV mandate regulation":
            df.iloc[:15, 35:] = 0
        case "EV mandate exogenous sales":
            df.iloc[0, 1] = 1       # The EV mandates are coded as a function; this switch turns it on
        case "Transport REPP":
             df.iloc[:, 15:] = carbon_price * 3.667    
  
                   
            
        # Freight policies
        case "ZREG strong":
            df.iloc[[0, 2, 4, 6, 8], 7:] = 0
        case "ZWSA strong":
            df.iloc[12, 7:] = 0 # TODO: I will need to figure out what a reasonable mandate is. 
        case "ZTVT strong tax":
            df.iloc[[0, 2, 4, 6, 8], 7:] = 0.3
        case "ZTVT strong subsidy":
            df.iloc[12, 7:] = -0.3
        case "ZTVT strong combo":
            df.iloc[[0, 2, 4, 6, 8], 7:] = 0.3
            df.iloc[12, 7:] = -0.3
        case "EV truck mandate regulation":
            df.iloc[[0, 2, 4, 6, 8], 23:] = 0
        case "EV truck mandate exogenous sales":
            df.iloc[0, 1] = 1       # The EV mandates are coded as a function; this switch turns it on
        case "Freight REPP":
            df.iloc[:, 15:] = carbon_price * 3.667 
            
        # Heat policies
        case "HREG strong":
            df.iloc[:4, 24:] = 0
            df.iloc[6, 24:] = 0
        case "HWSA strong": 
            df.iloc[10:12, 24:35] = 0.005     # Air-source heat pumps
            df.iloc[9, 24:35] = 0.002         # Ground-source heat pump
            df.iloc[2:4, 24:35] = -0.005      # Gas (note, that this probably won't work everywhere). TODO: Does the code already stop this?
            df.iloc[7, 24:35] = -0.002        # Electric heating
        case "HTVS strong tax":  # Strong tax
            df.iloc[:4, 24:] = 0.3
            df.iloc[6, 24:] = 0.3
        case "HTVS strong subsidy":  # Strong tax
            df.iloc[9:12, 24:] = -0.3         # 30% subsidy on heat pumps
        case "HTVS strong combo":  # Strong tax
            df.iloc[:4, 24:] = 0.3
            df.iloc[6, 24:] = 0.3
            df.iloc[9:12, 24:] = -0.3         # 30% subsidy on heat pumps
        case "Heat pump mandate 2035 regulation":
            df.iloc[:4, 35:] = 0
            df.iloc[6, 35:] = 0
        case "Heat pump mandate exogenous sales":
            df.iloc[0, 1] = 1       # The heat pump mandates are coded as a function; this switch turns it on
        case "Heat REPP":
            df.iloc[:, 15:] = carbon_price * 3.667 
            
        
        # Sector coupling
        case "Sector coupling":
            df.iloc[3, 1] = 0.5         # 50% cost savings on second-hand batteries
            
    return df
        
        
# Import policies from policies.csv in same folder
policies = pd.read_csv(os.path.join(current_dir, "Policies_sector_by_policy.csv"))

policy_packages = list(policies.keys()[9:])
#policy_packages = ["Carbon tax", "and_subsidies", "and_mandates", "Subsidies", "Mandates"]

for policy_package in policy_packages:
    print(policy_package)
    policies_turned_on = policies[policy_package]

    for pi, row in enumerate(policies.iterrows()):
        policy = row[1]         # Get the row as a dictionary

        if policies_turned_on[pi]:
            print(f"{policy['Model']}: {policy['Policy']}")
            
            source_dir = get_source_dir(input_dir, "S0", policy["Model"])
            copy_csv_files_to_scen(policy["Model"], policy["Variable"], policy_package, source_dir)
            change_csv_files(policy["Model"], policy_package, source_dir, policy["Variable"], policy["Policy"])
        


