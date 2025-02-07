# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:43:28 2025

@author: pv
"""

import numpy as np

def calc_ener_cost(data, titles, year):
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
    
    # Conversion from kWh to ktoe
    conversion = 0.0859845
    
    for tech in range(len(titles['HYTI'])):
        
        # Feedstock costs
        if tech in [0, 1, 4]:
            # Natural gas
            data['HYFC'][:, tech, 0] += np.multiply(data['BCHY'][:, tech, c7ti['Feedstock input, mean, kWh/kg H2 prod.']] * conversion,
                                                   data['PFRG'][:,5,0]) / data['PRSC'][:, 0, 0]
        elif tech in [2, 3]:
            # Coal
            data['HYFC'][:, tech, 0] += np.multiply(data['BCHY'][:, tech, c7ti['Feedstock input, mean, kWh/kg H2 prod.']] * conversion,
                                                   data['PFRC'][:,5,0]) / data['PRSC'][:, 0, 0]
        
        # Heat costs (assume it's always gas)
        data['HYFC'][:, tech, 0] += np.multiply(data['BCHY'][:, tech, c7ti['Heat demand, mean, kWh/kg H2']] * conversion,
                                               data['PFRG'][:,5,0]) / data['PRSC'][:, 0, 0]
        
        # Electricity costs
        data['HYFC'][:, tech, 0] += np.multiply(data['BCHY'][:, tech, c7ti['Electricity demand, mean, kWh/kg H2']] * conversion,
                                               data['PFRE'][:,5,0]) / data['PRSC'][:, 0, 0]
    
    return data