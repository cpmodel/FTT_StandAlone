# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:55:17 2023

@author: Femke & authors referee projects
"""
# Standard library imports
import os

# Third-party imports
import pandas as pd



def electricity_demand_feedback(data, data_baseline, y, titles, units):
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
        
        # convert to same unit as MEWD (PJ)
        unit = units[demand_var]
        conversion_factor = unit_conversion.loc[unit, demand_unit]
        
        # Compute change from baseline
        base_demand = data_baseline[demand_var][:, elec_index, 0, y]
        new_demand = data[demand_var][:, elec_index, 0]
        elec_change = new_demand - base_demand
        elec_change = elec_change * conversion_factor
        data["MEWDX"][:, elec_index, 0] += elec_change
  
        
    return data



