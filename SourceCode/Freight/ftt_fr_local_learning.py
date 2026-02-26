"""
ftt_local_learning.py
=========================================
Script to calculate localised learning by doing for ftt-freight.

Functions and classes included in the file:
    - get_start_local_capacity
        Function to get the starting local capacity for each technology in each region.
        
@author: Cormac Lynch
"""

def get_start_local_capacity(data, year):
    """
    Function to get the starting local capacity for each technology in each region.

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
    freight_local_capacity = data["ZEWK"] * 0.0
    freight_local_capacity[:, :, 0] = data["ZEWK"][:, :, 0] @ data["ZEWB"][0, :, :]

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
    local_additions = zewi_t[:, :, 0] @ data["ZEWB"][0, :, :]
    # Add local additions to existing local capacity
    data["Freight local capacity"][:,:,0] = local_additions + data_dt["Freight local capacity"][:,:,0]
        
    return data
