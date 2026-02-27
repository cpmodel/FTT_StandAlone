"""
ftt_local_learning.py
=========================================
Script to calculate localised learning by doing for ftt-freight.

Functions and classes included in the file:
    - get_start_local_capacity
        Function to get the starting local capacity for each technology in each region.
    - add_local_capacity
        Function to calculate and add new local capacity in simulation period.
    - get_local_learning
        Function to calculate local learning factor for each technology in each region.
        
@author: Cormac Lynch
"""
import numpy as np

def get_start_local_capacity(data, year):
    """
    Function to get the starting local capacity for each technology in each region. This involves
    grouping the sales of for each technology across region groups

    Parameters
    -----------
    data: dict
        Dictionary containing the data for each region and technology.
    year: int

    Returns
    ----------
    freight_local_capacity: ndarray
        Local learning by doing for freight technologies, with shape (regions, vehicles, 1).
    """
    
    # Assume that all BEV trucks ever sold in region are still active and calculate based on ZEWK.
    # Apply technology spillovers using ZEWB while preserving (region, vehicle, 1) shape.
    veh_aggregated = data["ZEWK"][:, :, 0] @ data["ZEWB"][0, :, :]
    local_capacity = data["region spillover"][0, :, :] @ veh_aggregated
    freight_local_capacity = data["ZEWK"] * 0.0
    freight_local_capacity[:, :, 0] = local_capacity
    
    return freight_local_capacity

def add_local_capacity(data, data_dt, zewi_t, year):
    """
    Function to calculate and add new local capacity in simulation period.

    Parameters
    -----------
    data: dict
        Dictionary containing the data for each region and technology.
    data_dt: dict
        Dictionary containing the data for each region and technology from the previous time step.
    zewi_t: ndarray
        Array of new vehicle sales for each region and technology.
    year: int

    Returns
    ----------
    data: dict
        Updated data dictionary with local capacity for each technology in each region. 
    """
    # calcaulate capacity additions considering spillover matrix
    veh_aggregated = zewi_t[:, :, 0] @ data["ZEWB"][0, :, :]
    local_additions = data["region spillover"][0, :, :] @ veh_aggregated
    # local_additions = zewi_t[:, :, 0] @ data["ZEWB"][0, :, :] @ data["region spillover"][:, :, 0]
    # Add local additions to existing local capacity
    data["Freight local capacity"][:,:,0] = local_additions + data_dt["Freight local capacity"][:,:,0]
        
    return data

def get_local_learning(data, zewi_t, titles, tech):
    """
    Function to calculate local learning factor for each technology in each region.

    Parameters
    -----------
    data: dict
        Dictionary containing the data for each region and technology.
    zewi_t: ndarray
        Array of new vehicle sales for each region and technology.
    titles: dict
        Dictionary containing the titles for each category.
    tech: int
        Technology index for which to calculate local learning factor.

    Returns
    ----------
    local_learning_factor: ndarray
        Local learning factor for each technology in each region, with shape (regions, vehicles, 1).
    """
    c6ti = {category: index for index, category in enumerate(titles['C6TI'])}
    
    # calcaulate capacity additions considering spillover matrix
    veh_aggregated = zewi_t[:, :, 0] @ data["ZEWB"][0, :, :]
    local_additions = data["region spillover"][0, :, :] @ veh_aggregated
    local_learning_factor = (1.0 + data["BZTC"][:, tech, c6ti['13 Learning exponent']]
                            * local_additions[:, tech] / data['Freight local capacity'][:, tech, 0] )
    return local_learning_factor