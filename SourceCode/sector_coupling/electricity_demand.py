# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:55:17 2023

@author: Femke & authors referee projects
"""
# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd
import copy


def energy_demand_from_sectors(data, titles, histend, year, ftt_modules):
    """ Energy demand (FRET in E3ME-FTT),
    as a function of the heat, freight and transport sector
    FRET is given in thousands toe
    
    In E3ME, freight is assumed to move with Tr in terms of biofuel mandates,
    and TJEF includes both transport and freight. 
    """
    
    if "FTT-P" in ftt_modules:
        data["FRET"][:, :, 0] = copy.deepcopy(data['FRETX'][:, :, 0])
        #print(data["FRET"][0])
        for r in range(len(titles['RTI'])):  # Loop over world regions
           
                
            
            if "FTT-Tr" in ftt_modules:
                data["FRET"][r, 15, 0] = data["TJEF"][r, 7, 0]
            if "FTT-Fr" in ftt_modules:
                # TODO?: fix something to account for missing data if freight turned off
                data["FRET"][r, 15, 0] += data["ZJEF"][r, 7, 0]
                
                
            # TODO: not all countries are represented in FTT:Fr output. EU/Brazil/China are but not regions 50-70
            
            if "FTT-H" in ftt_modules:            
                data["FRET"][r, 18, 0] = data["HJEF"][r, 7, 0]
                # if r == 0 and year in range(2014, 2018):
                #     print(f"Electricity usage in heating in Belgium in {year}:")
                #     print(f'HJEF: {data["HJEF"][r, 7, 0]}')
                #     print(f'FRET: {data["FRET"][r, 18, 0]}')
                
           
    
                
            
            # Compute the changes in actual electricity demand (normally in FTT.f90, put here for testing purposes)
            
            data["MEWD"][r, 7, 0] = np.sum(data["FRET"][r, :, 0]) * 41.868/1000 #/3.6 electricity
    else:
        print("FTT-P not included, so MEWD unaffected")    
        
    return data




def electricity_demand_feedback(data, data_baseline, year, titles, units):
    """
        Calculate change in electricity demand from baseline S0 scenario from other models
    """

    jti = titles["JTI"]
    elec_index = jti.index("8 Electricity")
    
    # Electricity mapping for each model
    elec_map = pd.read_csv(os.path.join('Utilities', 'mappings',
                                        "Electricity_demand_mapping.csv"),
                           index_col=0)
    unit_conversion = pd.read_csv(os.path.join('Utilities', 'mappings', "Energy_unit_conversions.csv"),
                                  index_col=0)
    demand_unit = units["MEWD"]
    for model in elec_map.index:
    
        demand_var = elec_map.loc[model, "fuel_var"]
        
        # convert to same unit as MEWD
        unit = units[demand_var]
        conversion_factor = unit_conversion.loc[unit, demand_unit]
        
        # Compute change from baseline
        base_demand = data_baseline[demand_var][:, elec_index, 0, year]
        new_demand = data[demand_var][:, elec_index, 0]
        elec_change = new_demand - base_demand
        elec_change = elec_change * conversion_factor
        data["MEWDX"][:, elec_index, 0] += elec_change
        if model == "FTT-Fr" and year == 2025:
            breakie = 1
        
    return data



