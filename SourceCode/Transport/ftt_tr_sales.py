# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:09:50 2024

@author: Cormac Lynch & Femke Nijsse
"""

import numpy as np

# %% new sales
# -----------------------------------------------------------------------------
# -----------------------Calculation of new sales (TEWI)-----------------------
# -----------------------------------------------------------------------------
def get_sales(data, data_dt, time_lag, dt, c3ti, t):
    """
    Calculate new sales/additions in FTT-Transport.

    This function calculates the amount of new vehicle sales required (TEWI).
    TEWI is calculated in thousands of vehicles. This is based on:

    1. If capacity has grown, the difference between the new capacity and the
       old.
    2. The amount of existing capacity that is depreciated (retired) as it
       reaches its end of life.
    
    Capacity depreciation is based on the vehicle lifetime in the
    cost matrix. This means 1/vehicle lifetime of capacity is retired every
    year.

    Parameters
    -----------
    data: dictionary
        data holds the current state of all variables.
        Variable names are keys and the values are 3D NumPy arrays.
    data_dt: dictionary
        data_dt is a container that for the previous timestep of all variables.
    time_lag: dictionary
        time_lag holds all variables of previous year
    dt: float
        dt is timestep: 1 / (number of timesteps).

    Returns
    ----------
    data: dictionary
        data now includes an updated TEWI or sales variable.
    tewi_t: 
        a 3D NumPy array that holds the new sales/additions at time t.
    
    """
    # Find capacity growth between this timestep and last
    cap_growth = data['TEWK'][:, :, 0] - data_dt['TEWK'][:, :, 0]

    # Find share growth between this timestep and last
    share_growth = data["TEWS"][:, :, 0] - data_dt["TEWS"][:, :, 0]

    # Calculate share_depreciation
    share_depreciation = data_dt["TEWS"][:, :, 0] * dt / data['BTTC'][:, :, c3ti['8 lifetime']]

    # If there is growth, depreciation is fully replaced.
    # Where capacity has decreased, we only replace depreciated capacity
    # if the depreciation > capacity loss
    conditions = [
        (cap_growth > 0.0),
        (-share_depreciation < share_growth) & (share_growth < 0)
    ]

    outputs = [
        data_dt["TEWK"][:, :, 0] * dt / data['BTTC'][:, :, c3ti['8 lifetime']],
        (share_growth + share_depreciation) * time_lag["TEWK"][:, :, 0]
    ]

    # Use np.select to choose an output for each condition
    cap_drpctn = np.select(conditions, outputs, default=0)
            
    # Ensure no negative values
    cap_drpctn = np.maximum(cap_drpctn, 0)
    cap_drpctn = cap_drpctn[:, :, np.newaxis]

    # Find total additions at time t
    # If capacity has grown, additions equal the difference +
    # depreciation from the previous year.
    # Otherwise, additions just equal depreciations.
    tewi_t = np.zeros(data["TEWI"].shape)
    tewi_t[:, :, 0] = np.where(cap_growth[:, :] > 0.0,
                                cap_growth[:, :] + cap_drpctn[:, :, 0],
                                cap_drpctn[:, :, 0])
        
    # Reset new additions if first FTT iteration
    if (t == 1):
        data["TEWI"][:, :, 0] = 0
    # Add additions at iteration t to total annual additions
    data["TEWI"][:, :, 0] = data["TEWI"][:, :, 0] + tewi_t[:, :, 0]

    return data, tewi_t