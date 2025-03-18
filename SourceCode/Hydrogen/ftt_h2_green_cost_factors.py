# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:43:28 2025

@author: pv
"""

import numpy as np

# Local library imports
from SourceCode.support.divide import divide

def calc_green_cost_factors(data, titles, year):
    """
    This function estimates the combined energy and feedstock costs for each
    technology.

    Parameters
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution
    titles: dictionary of lists
        Dictionary containing all title classification
    year: int
        Curernt/active year of solution

    Returns
    -------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    """
    
    # Unpack hydrogen cost matrix 
    c7ti = {category: index for index, category in enumerate(titles['C7TI'])}
    grid = [t for t, tech in enumerate(titles['HYTI']) if 'grid' in tech]
    green = [t for t, tech in enumerate(titles['HYTI']) if 'green' in tech]
    elec_use = c7ti['Electricity demand, mean, kWh/kg H2']
    elec_capex = c7ti['Onsite electricity CAPEX, mean, €/kg H2 cap']
    elec_opex = c7ti['Additional OPEX, mean, €/kg H2 prod.']
    elec_loadfac = c7ti['Maximum capacity factor']
    
    
    # First, make sure shares of dedicated VRE technologies add up to 1
    scalar = data['WSSH'][:, 0, 0]+data['WOSH'][:, 0, 0]+data['WWSH'][:, 0, 0]
    scalar[np.isclose(scalar, 0.0)] = 1.0
    # Apply scalar
    data['WSSH'][:, 0, 0] /= scalar
    data['WOSH'][:, 0, 0] /= scalar
    data['WWSH'][:, 0, 0] /= scalar
    
    # CAPEX factors as a weighted average in EUR/kW. Convert to EUR/kWh
    vre_capex_factor = (data['WSSH'][:, 0, 0]*data['WSIC'][:, 0, 0]+
                        data['WOSH'][:, 0, 0]*data['WOIC'][:, 0, 0]+
                        data['WWSH'][:, 0, 0]*data['WWIC'][:, 0, 0])
    vre_capex_factor /= 8766 
    
    # OPEX factors as a weighted average in EUR/MWh. Convert to EUR/kWh
    vre_opex_factor = (data['WSSH'][:, 0, 0]*data['WSOM'][:, 0, 0]+
                       data['WOSH'][:, 0, 0]*data['WOOM'][:, 0, 0]+
                       data['WWSH'][:, 0, 0]*data['WWOM'][:, 0, 0])
    vre_opex_factor *= 1e-3 
    
    # Load factor as a weighted average
    vre_load_factor = (data['WSSH'][:, 0, 0]*data['WSLF'][:, 0, 0]+
                       data['WOSH'][:, 0, 0]*data['WOLF'][:, 0, 0]+
                       data['WWSH'][:, 0, 0]*data['WWLF'][:, 0, 0])
    
    # How much VRE CAPEX we need depends on the electricity use factor
    # Take electricity use factor from grid-based electrolysis
    data['BCHY'][:, green, elec_capex] = data['BCHY'][:, grid, elec_use] * vre_capex_factor[:, None] 
    data['BCHY'][:, green, elec_opex] = data['BCHY'][:, grid, elec_use] * vre_opex_factor[:, None] 
    data['BCHY'][:, green, elec_loadfac] = vre_load_factor[:, None] 

    if year > 2023:
        x=1
    
    return data