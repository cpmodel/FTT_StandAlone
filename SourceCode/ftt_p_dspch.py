# -*- coding: utf-8 -*-
"""
power_generation.py
=========================================
Power generation FTT module.

Functions included:
    - get_lcoe
        Calculate levelized costs
    - solve
        Main solution function for the module
"""

# Standard library imports
from math import sqrt
import os
import copy
import sys
import warnings

# Third party imports
import pandas as pd
import numpy as np
from numba import njit

# Local library imports
from support.divide import divide


# %% dispatch function
# -----------------------------------------------------------------------------
# -------------------------- DSPTCH of capacity ------------------------------
# -----------------------------------------------------------------------------
@njit(fastmath=True)
def dspch(MWDD, MEWS, MKLB, MCRT, MEWL, MWMC_lag, MMCD_lag, rti, t2ti, lbti):
    """
    Calculate dispatch of capacity.

    The function estimates an allocation of power production between
    technologies and load bands.

    Parameters
    -----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
    time_lag: dictionary
        Time_lag is a container that holds all cross-sectional (of time) data
        for all variables of the previous year.
        Variable names are keys and the values are 3D NumPy arrays.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.

    Returns
    ----------
    data: type
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.

    Notes
    ---------
    Additional notes if required.
    """

    # Shorthand for dd (load band suitability of different technologies)
    dd = MWDD[0,:,:]

    # Technology logic identifiers
    s_var = dd[:,5]

    # Convergence criterion
    crit = 0.0001

    # Initialize variables to return (for numba)
    MSLB = np.zeros((rti, t2ti, lbti))
    MLLB = np.zeros((rti, t2ti, lbti))
    MES1 = np.zeros((rti, t2ti, 1))
    MES2 = np.zeros((rti, t2ti, 1))

    for r in range(rti):

        if r == 40:
            x = 1+1
        p_tech = np.zeros((t2ti, lbti))
        p_grid = np.zeros((t2ti, lbti))
        slb = np.zeros((t2ti, lbti))
        d_slb = np.zeros((t2ti, lbti))
        cflb = np.zeros((t2ti, lbti))
        q = 0
        d_s_tot = 1
        # Shares of capacity by tech
        s_i = MEWS[r, :, 0]
        # Shares of capacity by load band
        klb = MKLB[r, :, 0]
        # Average marginal cost
        m0 = np.sum(s_i * MWMC_lag[r, :, 0])

        # Technologies bid for generation time: weighted MNL
        # Constrained by size of load bands and available tech shares,
        # so allocate in an iterative process to try to find allocation closest
        # to preferences. Probabilistic 'college admissions'/'stable marriage'
        while (d_s_tot > crit and q < 200):

            for i in range(lbti-1): # NOT intermittent renewables

                sig = np.sqrt(np.sum(dd[:,i]*MMCD_lag[r, :, 0]**2*MEWS[r, :, 0]))
                exponent = -(MWMC_lag[r, :, 0]-m0)/sig
                fn = np.zeros((t2ti))
                # Approximate exponential
                fn[np.abs(exponent)<20] = np.exp(exponent[np.abs(exponent)<20])
                fn[exponent>=20] = 1e9
                fn[exponent<=-20] = 1e-9
                # Multinomial logit or simplification

                if (np.sum(dd[:,i]*s_i)>0 and sig>0.001 and np.sum(dd[:,i]*s_i*fn)>0):

                    p_tech[:, i] = dd[:,i]*s_i*fn / np.sum(dd[:,i]*s_i*fn)

                else:

                    p_tech[:, i] = np.zeros((t2ti))

            # Grid operator has preferences amongst what is bid for
            for k in range(t2ti):

                # p_grid is likelihood grid accepts bid from tech k
                sig = np.sqrt(np.sum(dd[k,:]*MMCD_lag[r, k, 0]**2))

                if (dd[k,5]==0 and np.sum(dd[k,:]*klb)>0 and sig>0.001):

                    p_grid[k,:] = dd[k,:]*klb / np.sum(dd[k,:]*klb)

                else:

                    p_grid[k,:] = np.zeros((lbti))

                if r == 40 and k == 6:
                    x = 1+1

            # Increment q
            q += 1
            # Allocate one round of generation to load bands according to p_grid, p_tech
            for i in range(lbti-1):

                for k in range(t2ti):

                    d_slb[k, i] = np.abs(np.minimum(np.abs(s_i[k]), np.abs(klb[i]))*p_tech[k, i]*p_grid[k, i])

            d_slb[:, 5] = np.zeros((t2ti))
            # Cumulative allocations each round
            slb = slb + d_slb
            # Remove allocation from what's left per tech and load band
            s_i = s_i - np.sum(d_slb, axis=1)
            klb = klb - np.sum(d_slb, axis=0)
            d_s_tot = np.sum(d_slb)

        # Capacity of renewables
        for index in range(len(MWDD[0,:,5])):
            if MWDD[0,index,5]:
                slb[index, 5] = MEWS[r, index, 0]
        # slb[s_var, 5] = MEWS[r, s_var, 0]

        # Capacity factors by load band (definitionally)
        cflb[:, 0] = 7500/8766
        cflb[:, 1] = 4400/8766
        cflb[:, 2] = 2200/8766
        cflb[:, 3] = 700/8766
        cflb[:, 4] = 80/8766
        cflb[:, 5] = np.zeros((t2ti))
        # cflb[s_var, 5] = MEWL[r, s_var, 0]
        for index in range(len(MWDD[0,:,5])):
            if MWDD[0,index,5]:
                cflb[index, 5] = MEWL[r, index, 0] # * (1 - MCRT[r,index,0])
        cflb = np.where(cflb==0, 1, cflb)
        # cflb[~cflb.astype(bool)] = 1

        # Save in data dict
        MSLB[r,:,:] = slb
        MLLB[r,:,:] = cflb

        # Upper share limit: set to ones for now (no upper limit, but
        # resistance to 100% market shares for any single tech)
        MES1[r, :, 0] = np.ones(t2ti)

        # Lower share limits
        # TODO: Are these compatible with storage cost incorporation? Maybe not...
        # Difference between availability (minus taken by other LB) and requirement per LB
        # Note this quantity is essentially what Gmin(i,j) takes tanh of
        grid_lim = np.zeros((5))
        # grid_lim[4] = np.sum(dd[:, 4]*MEWS[r, :, 0]) - klb[4]
        # grid_lim[3] = np.sum(dd[:, 3]*MEWS[r, :, 0]) - np.sum(klb[3:5])
        # grid_lim[2] = np.sum(dd[:, 2]*MEWS[r, :, 0]) - np.sum(klb[2:5])
        # grid_lim[1] = np.sum(dd[:, 1]*MEWS[r, :, 0]) - np.sum(klb[1:5])
        # # No lower limit for baseload band
        # grid_lim[0] = np.sum(dd[:, 0]*MEWS[r, :, 0]) - np.sum(klb[0:5])
        
        grid_lim[4] = np.sum(dd[:, 4]*MEWS[r, :, 0]) - MKLB[r, 4, 0]
        grid_lim[3] = np.sum(dd[:, 3]*MEWS[r, :, 0]) - np.sum(MKLB[r, 3:5, 0])
        grid_lim[2] = np.sum(dd[:, 2]*MEWS[r, :, 0]) - np.sum(MKLB[r, 2:5, 0])
        grid_lim[1] = np.sum(dd[:, 1]*MEWS[r, :, 0]) - np.sum(MKLB[r, 1:5, 0])
        # No lower limit for baseload band
        grid_lim[0] = np.sum(dd[:, 0]*MEWS[r, :, 0]) - np.sum(MKLB[r, 0:5, 0])
        #Temp is shares minus grid_lim for suitable LB
        temp = np.zeros((t2ti, lbti))
        for i in range(1, lbti-1):
            temp[:,i] = dd[:,i]*(MEWS[r,:,0] - grid_lim[i])
        # For each tech, take largest value in temp or shares, whichever is less
        for i in range(t2ti):
            MES2[r, i, 0] = min(np.max(temp[i,:]), MEWS[r, i, 0])

        if r == 40:
            x = 1+1

    return MSLB, MLLB, MES1, MES2