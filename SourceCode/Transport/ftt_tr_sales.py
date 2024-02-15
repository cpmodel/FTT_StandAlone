# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:09:50 2024

@author: CL
"""

import numpy as np

# %% new sales
# -----------------------------------------------------------------------------
# -----------------------Calculation of new sales (TEWI)-----------------------
# -----------------------------------------------------------------------------
def get_sales(data, data_dt, time_lag, titles, dt, c3ti, t):
    """
    Calculate new sales/additions in FTT-Transport.

    This function calculates the amount of new vehicle sales required (TEWI).
    TEWI is calculated in thousands of vehicles. This is based on:

    1. If capacity has grown, the difference between the new capacity and the
       old.
    2. The amount of existing capacity that is depreciated (retired) as it
       reaches its end of life.
    
    Capacity depreciation is currently based on the vehicle lifetime in the
    cost matrix. This means 1/vehicle lifetime of capacity is retired every
    year.

    Parameters
    -----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) for all
        variables. Variable names are keys and the values are 3D NumPy arrays.
    data_dt: dictionary
        Data_dt is a container that holds all cross-sectional (of time) data
        for all variables of the previous iteration.
    time_lag: dictionary
        Time_lag is a container that holds all cross-sectional (of time) data
        for all variables of the previous year.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.
    dt: integer
        Dt is an integer - 1 / number of iterations.

    Returns
    ----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.
    
    """
    # Find the difference in capacity between this iteration and last
    cap_diff = data['TEWK'][:, :, 0] - data_dt['TEWK'][:, :, 0]
    # Initialise variable for amount of capacity depreciated
    cap_drpctn = np.zeros([len(titles['RTI']), len(titles['VTTI']), 1])
    # If capacity has grown, additions equal the difference +
    # depreciation from the previous year.
    # Otherwise, additions just equal depreciations.
    # Where capacity has decreased, we only add new capacity
    # if the depreciation > capacity loss
    for r in range(len(titles['RTI'])):
        for veh in range(len(titles['VTTI'])):
            cap_drpctn[r,veh,0] = np.where(cap_diff[r, veh] > 0.0,
                                data_dt["TEWK"][r, veh, 0] * dt / data['BTTC'][r, veh, c3ti['8 lifetime']],
                                np.where((-data_dt["TEWS"][r, veh, 0] * dt / data['BTTC'][r, veh, c3ti['8 lifetime']] <
                                            data["TEWS"][r, veh, 0] - data_dt["TEWS"][r, veh, 0] < 0),
                                            (data["TEWS"][r, veh, 0] - data_dt["TEWS"][r, veh, 0] +
                                            data_dt["TEWS"][r,veh,0] * dt / data['BTTC'][r, veh, c3ti['8 lifetime']])
                                            * time_lag["TEWK"][r, veh, 0], 0))
    # Find total additions at time t
    tewi_t = np.zeros([len(titles['RTI']), len(titles['VTTI']), 1],)
    tewi_t[:, :, 0] = np.where(cap_diff[:, :] > 0.0,
                                cap_diff[:, :] + cap_drpctn[:, :, 0],
                                cap_drpctn[:, :, 0])
    
    # Reset new additions if first FTT iteration
    if (t == 1):
        data["TEWI"][: ,: ,0] = 0
    # Add additions at iteration t to total annual additions
    data["TEWI"][:, :, 0] = data["TEWI"][:,:,0] + tewi_t[:,:,0]

    return data, tewi_t