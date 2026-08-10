# -*- coding: utf-8 -*-
"""
This script copies all the files from the S0 folder to the subfolders
in case the files do not exist yet.

It also allows copying the MSAL files around. Note there are types
@author: Femke
"""

import shutil, os
import os.path
import numpy as np
import pandas as pd


current_dir = os.path.dirname(os.path.realpath(__file__))
# The input dir is found by going up a directly to the parent directory and then going to the Input folder
top_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
input_dir = os.path.join(top_dir, "Inputs")



def create_file_list(variable, source_dir):
    """Create list of source files for one variable.

    New input format stores all countries in a single file named
    ``<variable>.csv``. Prefer that file when present.
    """

    combined_file = os.path.join(source_dir, f"{variable}.csv")
    if os.path.exists(combined_file):
        return [combined_file]
    return []

def get_source_dir(input_dir, base_scen, model):
    source_dir = os.path.join(input_dir, "S0", model)
    return source_dir

def copy_csv_files_to_scen(model, variable, scen_name, source_dir):
    """Copy source file(s) for a single variable to the new scenario."""
    file_list = create_file_list(variable, source_dir)
    desti_dir = os.path.join(input_dir, scen_name, model) 

    if not file_list:
        raise FileNotFoundError(
            f"No input file found for variable '{variable}' in '{source_dir}'."
        )
    
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
    if not os.path.exists(desti_dir):
        os.makedirs(desti_dir)
    # Extract the filename from the original file path
    filename = os.path.basename(file)
    # Create the full destination file path
    dest_file_path = os.path.join(desti_dir, filename)
    # Save the DataFrame to the new file path
    df.to_csv(dest_file_path, index=False)


def change_csv_files(model, scen_name, source_dir, variable, policy):
    file_list = create_file_list(variable, source_dir)
    if not file_list:
        raise FileNotFoundError(
            f"No input file found for variable '{variable}' in '{source_dir}'."
        )
    for file in file_list:
        # Read in file
        df = pd.read_csv(file)
        df = policy_change(df, policy)
        save_new_file(model, scen_name, file, df)


def policy_change(df, policy):
    def _normalise_row_positions(row_positions):
        if isinstance(row_positions, slice):
            start = 0 if row_positions.start is None else row_positions.start
            stop = len(df) if row_positions.stop is None else row_positions.stop
            step = 1 if row_positions.step is None else row_positions.step
            return list(range(start, stop, step))
        if isinstance(row_positions, range):
            return list(row_positions)
        if isinstance(row_positions, (list, tuple, np.ndarray)):
            return [int(x) for x in row_positions]
        return [int(row_positions)]

    def _apply_rows_per_country(row_positions, col_selector, value):
        """Apply row-index policies to each RTI block in consolidated files."""
        positions = _normalise_row_positions(row_positions)

        if "RTI" not in df.columns:
            df.iloc[positions, col_selector] = value
            return

        for _, group in df.groupby("RTI", sort=False):
            group_idx = group.index.to_numpy()
            group_len = len(group_idx)
            valid_local = [p for p in positions if 0 <= p < group_len]
            if valid_local:
                df.iloc[group_idx[valid_local], col_selector] = value

    carbon_price = 200.0     # Constant €200 per tonne CO2 
    match policy:
        
        case "REPP":  # A linearly increasing price to €200 per tonne CO2, i.e.  
            price_2050 = carbon_price
            price_2024 = df.iloc[:, 15] / 3.667 # Note, REPP is given per tC, rather than tCO2
            
            # Reshape the price_2023 to a column vector
            price_2023 = price_2024.values.reshape(-1, 1)
            
            # Linearly increase the price from 2023 to 2050 values. 
            df.iloc[:, 15:42] = ( price_2023 + (price_2050 - price_2024) / 26.0 * np.arange(27) ) * 3.667 
            # After 2050, continue everywhere with equal yearly increases, equal to price_2050/27
            df.iloc[:, 42:] = ( price_2050 + price_2050 / 26.0 * np.arange(1, 21) ) * 3.667      
            
        case "REPP2":  # A linearly increasing price to €200 per tonne CO2, i.e.  
            price_2050 = carbon_price
                        
            # Linearly increase to €200 per tonne CO2 
            df.iloc[:, 15:42] = (price_2050) / 26.0 * np.arange(27) * 3.667 
            # After 2050, continue everywhere with equal yearly increases, equal to price_2050/27
            df.iloc[:, 42:] = ( price_2050 + price_2050 / 27.0 * np.arange(1, 21) ) * 3.667       
    
        case "Power REPP":
            df[df.columns[1:]] = df[df.columns[1:]].astype(float)
            df.iloc[:, 15:] = carbon_price * 3.667 
            
        case "Power REPP half":
            df[df.columns[1:]] = df[df.columns[1:]].astype(float)
            df.iloc[:, 15:] = carbon_price * 3.667 / 2
        
        
        # Power sector policies
        case "MEWR strong":     # Completely outregulate fossil technologies from 2024
            _apply_rows_per_country(slice(1, 10), slice(24, None), 0)
        case "MEWT":           # Subsidize all renewables
            _apply_rows_per_country(slice(12, 22), slice(25, None), -0.3)
        case "MEWT half":           # Subsidize all renewables
            _apply_rows_per_country(slice(12, 22), slice(25, None), -0.15)
        case "Coal phase-out":
            _apply_rows_per_country(0, 1, 1)       # The coal phase-out is coded as a function; this switch turns it on 
        case "Coal phase-out half":
            _apply_rows_per_country(0, 1, 0.5)       # TODO code the phase-out so it can be halved!
      
        
        # Transport policies
        case "TREG strong":
            _apply_rows_per_country(slice(0, 15), slice(24, None), 0)
        case "BRR strong tax": 
            _apply_rows_per_country(slice(0, 15), slice(25, None), 0.3)
        case "BRR strong subsidy":
            _apply_rows_per_country(slice(18, 21), slice(25, None), -0.3)
        case "BRR half subsidy":
            _apply_rows_per_country(slice(18, 21), slice(25, None), -0.15)
        case "BRR strong combo":
            _apply_rows_per_country(slice(0, 15), slice(25, None), 0.3)
            _apply_rows_per_country(slice(18, 21), slice(25, None), -0.3)
        case "EV mandate regulation":
            _apply_rows_per_country(slice(0, 15), slice(35, None), 0)
        case "EV mandate":
            df.iloc[:, 3] = 2026       # Start year EV mandate
            df.iloc[:, 3] = 2035       # End year EV mandate
            df.iloc[:, 3] = 1          # Maximum EV mandate
        case "EV mandate half":
            _apply_rows_per_country(0, 1, 2045)    # Half the speed of the mandate
        case "Transport REPP":
             df[df.columns[1:]] = df[df.columns[1:]].astype(float)
             df.iloc[:, 15:] = carbon_price * 3.667    
        case "Transport REPP half":
             df[df.columns[1:]] = df[df.columns[1:]].astype(float)
             df.iloc[:, 15:] = carbon_price * 3.667 / 2   
  
                   
            
        # Freight policies
        case "ZREG strong":
            _apply_rows_per_country(range(25), slice(7, None), 0)
        case "ZTVT strong tax":
            _apply_rows_per_country(range(25), slice(7, None), 0.3)
        case "ZTVT strong subsidy":
            _apply_rows_per_country([31, 32, 33], slice(8, None), -0.3)
        case "ZTVT half subsidy":
            _apply_rows_per_country([31, 32, 33], slice(8, None), -0.15)
        case "ZTVT strong combo":
            _apply_rows_per_country(range(25), slice(8, None), 0.3)
            _apply_rows_per_country([31, 32, 33], slice(8, None), -0.3)
        case "EV truck mandate regulation":
            _apply_rows_per_country(range(25), slice(23, None), 0)
        case "EV truck mandate":
            df.iloc[:, 1] = 2026
            df.iloc[:, 2] = 2040       # The EV mandates are coded as a function; this switch turns it on
            df.iloc[:, 3] = 1       # The EV mandates are coded as a function; this switch turns it on
        case "EV truck mandate half":
            df.iloc[:, 1] = 2026
            df.iloc[:, 2] = 2040       # The EV mandates are coded as a function; this switch turns it on
            df.iloc[:, 3] = 0.5       # The EV mandates are coded as a function; this switch turns it on
        case "Freight REPP":
            df[df.columns[1:]] = df[df.columns[1:]].astype(float)
            df.iloc[:, 15:] = carbon_price * 3.667 
        case "Freight REPP half":
            df[df.columns[1:]] = df[df.columns[1:]].astype(float)
            df.iloc[:, 15:] = carbon_price * 3.667 / 2
        
        # Carbon tax with start and end date
        case str(value) if value.startswith("Freight REPP 20"):
            df[df.columns[1:]] = df[df.columns[1:]].astype(float)
            years = value.split()[-1]  # Extract the range part, e.g., "2026-2050"
            start_year, end_year = map(int, years.split("-"))  # Extract start and end years
            base_year = 2025  # Shifted base year for simplicity
            base_column_index = 16  # Adjusted to match the base year
            start_column_index = base_column_index + (start_year - base_year)
            end_column_index = base_column_index + (end_year - base_year) + 1
            df.iloc[:, start_column_index:end_column_index] = carbon_price * 3.667
            
        case "EV truck mandate before 2027":
            _apply_rows_per_country(0, 1, 2027)       # #TODO: Check if this still works with new mandates
        case "EV truck mandate before 2030":
            _apply_rows_per_country(0, 1, 2030)       # 
        case "EV truck mandate before 2035":
            _apply_rows_per_country(0, 1, 2035)       # 
    
        
            
        # Heat policies
        case "HREG strong":
            _apply_rows_per_country(slice(0, 4), slice(24, None), 0)
            _apply_rows_per_country(6, slice(24, None), 0)
        case "HTVS strong tax": 
            _apply_rows_per_country(slice(0, 4), slice(24, None), 0.3)
            _apply_rows_per_country(6, slice(24, None), 0.3)
        case "HTVS strong subsidy":
            _apply_rows_per_country(slice(9, 12), slice(25, None), -0.3)         # 30% subsidy on heat pumps
        case "HTVS half subsidy":  # half the s tax
            _apply_rows_per_country(slice(9, 12), slice(25, None), -0.15)         # 30% subsidy on heat pumps
        case "HTVS strong combo":  # Strong tax
            _apply_rows_per_country(slice(0, 4), slice(25, None), 0.3)
            _apply_rows_per_country(6, slice(25, None), 0.3)
            _apply_rows_per_country(slice(9, 12), slice(25, None), -0.3)         # 30% subsidy on heat pumps
        case "Heat pump mandate 2035 regulation":
            _apply_rows_per_country(slice(0, 4), slice(35, None), 0)
            _apply_rows_per_country(6, slice(35, None), 0)
        case "Heat pump mandate":
            df.iloc[:, 2] = 2026       # Start heat pump mandate
            df.iloc[:, 2] = 2035       # End heat pump mandate
            df.iloc[:, 3] = 1           # Maximum heat pump mandate
        case "Heat pump mandate half":
            df.iloc[:, 1] = df.iloc[:, 1].astype(float)
            _apply_rows_per_country(0, 1, 2045)       # The heat pump mandates are coded as a function; this switch turns it on
        case "Heat REPP":
            df[df.columns[1:]] = df[df.columns[1:]].astype(float)
            df.iloc[:, 15:] = carbon_price * 3.667 
            
        
        # Sector coupling
        case "Sector coupling":
            _apply_rows_per_country(3, 1, 0.5)         # 50% cost savings on second-hand batteries
            
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
        


