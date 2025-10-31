# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:43:28 2025

@author: pv
"""

import numpy as np

def calc_ener_cons(data, titles, year):
    """
    This function estimates the energy consumption

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
    
    # Conversion from kWh to toe
    conversion = 1/11630
    
    for tech in range(len(titles['HYTI'])):
        
        # Feedstock costs
        if tech in [0, 1, 4]:
            # Natural gas
            data['HYJF'][:, 6, 0] += np.multiply(data['BCHY'][:, tech, c7ti['Feedstock input, mean, kWh/kg H2 prod.']] * conversion,
                                                   data['HYG1'][:,tech,0]) * 1e6 * 1e-3
        elif tech in [2, 3]:
            # Coal
            data['HYJF'][:, 0, 0] += np.multiply(data['BCHY'][:, tech, c7ti['Feedstock input, mean, kWh/kg H2 prod.']] * conversion,
                                                   data['HYG1'][:,tech,0]) * 1e6 * 1e-3
        
        # Heat costs (assume it's always gas)
        data['HYJF'][:, 6, 0] += np.multiply(data['BCHY'][:, tech, c7ti['Heat demand, mean, kWh/kg H2']] * conversion,
                                               data['HYG1'][:,tech,0]) * 1e6 * 1e-3
        
        # Electricity costs
        data['HYJF'][:, 7, 0] += np.multiply(data['BCHY'][:, tech, c7ti['Electricity demand, mean, kWh/kg H2']] * conversion,
                                               data['HYG1'][:,tech,0]) * 1e6 * 1e-3
    return data

# %%
def calc_emis_rate(data, titles, year):
    """
    This function estimates the emission factors

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
    
    # Conversion from kWh to toe
    conversion = 1/11630
    
    for tech in range(len(titles['HYTI'])):
        
        # Feedstock emissions (assume 100% decomposition)
        # Pyrolysis breaks natural gas down into solid C, so no emissions from
        # feedstock
        if tech in [0, 1]:
            # Natural gas
            data['HYEF'][:, tech, 0] += (data['BCHY'][:, tech, c7ti['Feedstock input, mean, kWh/kg H2 prod.']] 
                                         * conversion # toe gas / kg H2
                                         * 0.0024299 # ktCO2 / toe gas
                                         * 1e6 * 0.65)  # ktCO2/kg H2 to kg CO2 / kg H2
        elif tech in [2, 3]:
            # Coal
            data['HYEF'][:, tech, 0] += (data['BCHY'][:, tech, c7ti['Feedstock input, mean, kWh/kg H2 prod.']] 
                                         * conversion # toe coal / kg H2
                                         * 0.00419804 # ktCO2 / toe coal
                                         * 1e6 * 0.65)  # ktCO2/kg H2 to kg CO2 / kg H2
    
        # Emissions due to heat generation (assume it's always gas)
        data['HYEF'][:, tech, 0] += (data['BCHY'][:, tech, c7ti['Heat demand, mean, kWh/kg H2']] 
                                     * conversion # toe gas / kg H2
                                     * 0.0024299 # ktCO2 / toe gas
                                     * 1e6)  # ktCO2/kg H2 to kg CO2 / kg H2
        
    # Apply CO2 capture rates to CCS techs
    data['HYEF'][:, [1,3], 0] *= 0.05
        
        

    return data