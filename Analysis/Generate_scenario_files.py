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

from scipy.stats import poisson, binom, uniform, norm, randint
np.random.seed(123)

MSAL = binom(1, 0.5)                                 # Which MSAL switch to use
WACC_range = uniform()                               # Scale between unequal and equal access to finance
learning_rate_solar = norm(-0.303, 0.047)            # Learning rate solar
learning_rate_wind = norm(-0.158, 0.045)             # Learning rate wind
lifetime = randint(25, 35)                           # Lifetime of solar panel
gamma = norm(loc=1, scale=0.2)                       # Scaling factor of gamma
fuel_costs = norm(loc=1, scale=0.2)                  # Scaling factor of gamma
grid_expansion_duration = poisson(0.6)               # The lead time of solar

Nsample = 200

MC_samples = np.vstack([MSAL.rvs(Nsample),     
           WACC_range.rvs(Nsample),                             # BCET
           learning_rate_solar.rvs(Nsample),                    # BCET
           learning_rate_wind.rvs(Nsample),                     # BCET
           lifetime.rvs(Nsample),                               # BCET & MEWA
           gamma.rvs(Nsample),                                  # MGAM
           fuel_costs.rvs(Nsample),                             # BCET
           grid_expansion_duration.rvs(Nsample)+1]).transpose() # BCET & MEWA

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
copy_csv_files_to_scen("FTT-P", "MEWR", "S0_001", source_dir)

def save_new_file(model, scen_name, file, df):
    "Create destination directory, and save the file"
    
    desti_dir = os.path.join(input_dir, scen_name, model)
    # Extract the filename from the original file path
    filename = os.path.basename(file)
    # Create the full destination file path
    dest_file_path = os.path.join(desti_dir, filename)
    # Save the DataFrame to the new file path
    df.to_csv(dest_file_path, index=False)




#%% define all the policies that we want to compare

# In the power sector, we would like to change the following policies:
# 1. MEWT - Renewable subsidies (say 20% wind subsidies) or feed-in-tariffs
# 2. MEWR - (100% renewables by 2035)
# 3. MEWRb - Efficiency regulation (say regulation on coal only?) This is a bonus
# 4. MCOCX - carbon tax

# In the transport & freight sectors, we would like to change the following policies:
# 1. TREG and ZREG - Efficiency regulations (so going to advanced ones)
# 2. TWSA and ZWSA - ZEV mandates (taking the same as the policy brief --> 2035 has 50% ZEV in slow, 100% in fast. Say 2040 for freigth). NOTE: freight has it in abs. values
# 3. Fuel regulation --> it seems Aileen used this against all fossil cars?
# 4. RTCO in % / TTVT abs and RZCO (registration tax per CO2) / ZTVT (absolute vehicle tax)
# 5.  RTCO in % / TTVT abs and ZTVT (absolute vehicle tax)

# In the heating sector, we would like to change the following policies:
# To be determined

def change_csv_files(model, scen_name, source_dir, variable, policy):
    file_list = create_file_list(variable, source_dir)
    for file in file_list:
        # Read in file
        df = pd.read_csv(file)
        # Change the values of row 1 (python indexing) to row 7 (including) to 0
        df = policy_change(df, policy)
        save_new_file(model, scen_name, file, df)


def policy_change(df, policy):
    match policy:
        # Power sector policies
        case "MEWR strong":     # Completely outregulate fossil technologies from 2024
            df.iloc[1:8, 24:] = 0
        case "MWKA strong":     # Set MWKA to 2021 MEWK values in 2022 and 2023, and then go down to 0 by 2035 --> #TODO
            # Easiest here is probably to semi-hard code this work. 
            pass
        case "MEWT strong":     # Subsidize all renewables, except for solar
            df.iloc[ 8:18, 24:] = -0.3
            df.iloc[19:22, 24:] = -0.3
        case "MCOCX strong":  # Say, a linearly increasing price to $500 dollar per CO2, i.e.  
            df.iloc[1:8, 24:] = df.iloc
       
        
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
            
        # Heat policies
        case "HREG strong":
            df.iloc[:4, 24:] = 0
            df.iloc[6, 24:] = 0
        case "HWSA strong": # To do: test this
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
            
            
    return df
        
        
# Import policies from policies.csv in same folder
policies = pd.read_csv(os.path.join(current_dir, "policies.csv"))

for row in policies.iterrows():
    policy = row[1]         # Get the row as a dictionary
    source_dir = get_source_dir(input_dir, "S0", policy["Model"])
    copy_csv_files_to_scen(policy["Model"], policy["Variable"], f"S0_{policy['Model']}", source_dir)
    change_csv_files(policy["Model"], f"S0_{policy['Model']}", source_dir, policy["Variable"], policy["Policy"])
        
change_csv_files("FTT-P", "S0_001", source_dir, "RTCO", "RTCO strong subsidy")     


