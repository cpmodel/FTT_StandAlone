# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:09:26 2024

@author: AE

=========================================
ftt_h_sales.py
=========================================
Domestic Heat FTT module.
####################################

    Calculate new sales/additions in FTT-Heat.

    This function calculates the amount of new additional boilers (HEWI).
    This is based on:

    1. If capacity has grown, the difference between the new capacity and the
       old.
    2. The amount of existing capacity that is depreciated (retired) as it
       reaches its end of life.
    
    Capacity depreciation is currently based on the boiler lifetime in the
    cost matrix. This means 1/boiler lifetime of capacity is retired every
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
    hewi_t: 
    

"""

# Third party imports
import numpy as np

# %% new sales
# -----------------------------------------------------------------------------
# -----------------------Calculation of new sales (HEWI)-----------------------
# -----------------------------------------------------------------------------

def get_sales(data, data_dt, time_lag, titles, dt, t, endo_eol):

    eol_repl_t = np.zeros([len(titles['RTI']), len(titles['HTTI']), 1])

    eol_repl_t[:, :, 0] = np.where(endo_eol >= 0.0,
                                    time_lag['HEWK'][:, :, 0] * dt * data['HETR'][:, :, 0],
                                    0.0)
    eol_repl_t[:, :, 0] = np.where(np.logical_and(-data_dt['HEWS'][:,:,0] * dt * data['HETR'][:,:,0] < endo_eol ,  endo_eol < 0.0),
                                    ((data['HEWS'][:, :, 0] - data_dt['HEWS'][:, :, 0] + data_dt['HEWS'][:, :, 0] * dt * data['HETR'][:, :, 0]) * time_lag['HEWK'][:, :, 0]),
                                    eol_repl_t[:, :, 0])
    
    # Catch any negative values
    for r in range (len(titles['RTI'])):
        for b in range (len(titles['HTTI'])):
            if eol_repl_t[r, b, 0] < 0.0:
                eol_repl_t[r, b, 0] = 0.0

    # Capacity growth
    hewi_t = np.zeros([len(titles['RTI']), len(titles['HTTI']), 1])
    hewi_t[:, :, 0] = np.where(data['HEWK'][:, :, 0] - data_dt['HEWK'][:, :, 0] > 0.0,
                                        data['HEWK'][:, :, 0] - data_dt['HEWK'][:, :, 0] + eol_repl_t[:, :, 0],
                                        eol_repl_t[:, :, 0])

    # Capacity growth, add each time step to get total at end of loop
    data['HEWI'][:, :, 0] = data['HEWI'][:, :, 0] + hewi_t[:, :, 0]
    
    data['HEWI'][:, :, 0] = np.where(data['HEWI'][:, :, 0] < 0.0,
                                        0.0,
                                        data['HEWI'][:, :, 0])

    if (t == 1):
        data['HEWI'][:, :, 0] = 0.0

    return data, hewi_t