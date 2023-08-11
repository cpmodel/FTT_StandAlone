# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_shares.py
=========================================
Power generation shares FTT module.

Functions included:
    - shares
        Calculate market shares

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
from SourceCode.support.divide import divide

# %% JIT-compiled shares equation
# -----------------------------------------------------------------------------
@njit(fastmath=True)
def shares(dt, t, T_Scal, mewdt, mwdlt, mews_dt, metc_dt, mtcd_dt,
           mwka, mes1_dt, mes2_dt, mewa, isReg, mewk_dt, mewk_lag, mewr,
           mewl_dt, mews_lag, mwlo, mwdl, rti, t2ti,no_it):

    """
    Function to calculate market share dynamics

    This function calculates market shares based on market shares of the
    previous iteration

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

    # Values to return
    mews = np.zeros((rti, t2ti, 1))
    mewl = np.zeros((rti, t2ti, 1))
    mewg = np.zeros((rti, t2ti, 1))
    mewk = np.zeros((rti, t2ti, 1))

    for r in range(rti):

        if mewdt[r] == 0.0:
            continue

        if r==1:
            x = 1+1
#            print('BE')

        # Initialise variables related to market share dynamics
        # DSiK contains the change in shares
        dSik = np.zeros((t2ti, t2ti))

        # F contains the preferences
        F = np.ones((t2ti, t2ti))*0.5

        # Market share constraints
        Gijmax = np.ones((t2ti))
        Gijmin = np.ones((t2ti))

        for t1 in range(t2ti):

            if not (mews_dt[r, t1, 0] > 0.0 and
                    metc_dt[r, t1, 0] != 0.0 and
                    mtcd_dt[r, t1, 0] != 0.0 and
                    mwka[r, t1, 0] < 0.0):
                continue

            Gijmax[t1] = np.tanh(1.25*(mes1_dt[r, t1, 0] - mews_dt[r, t1, 0])/0.1)
            Gijmin[t1] = np.tanh(1.25*(-mes2_dt[r, t1, 0] + mews_dt[r, t1, 0])/0.1)
            dSik[t1, t1] = 0
            S_i = mews_dt[r, t1, 0]
#                    Aki = 0.5 * data['PG_EOL'][r, t1, 0] / time_lag['MEWK'][r, t1, 0]

            for t2 in range(t1):

                if not (mews_dt[r, t1, 0] > 0.0 and
                        metc_dt[r, t1, 0] != 0.0 and
                        mtcd_dt[r, t1, 0] != 0.0 and
                        mwka[r, t1, 0] < 0.0):
                    continue

                S_k = mews_dt[r, t2, 0]
#                        Aik = 0.5 * data['PG_EOL'][r, t2, 0] / time_lag['MEWK'][r, t2, 0]

                # Use substitution rate matrix, instead of a
                # estimation based on EoL capacity
                # Aik = mewa[r, t1, t2]
                # Aki = mewa[r, t2, t1]

                # Propagating width of variations in perceived costs
                dFik = np.sqrt(2) * np.sqrt(mtcd_dt[r, t1, 0]*mtcd_dt[r, t1, 0] + mtcd_dt[r, t2, 0]*mtcd_dt[r, t2, 0])

                # Consumer preference incl. uncertainty
                Fik = 0.5*(1+np.tanh(1.25*(metc_dt[r, t2, 0]-metc_dt[r, t1, 0])/dFik))

                # Preferences are then adjusted for regulations
                F[t1, t2] = Fik*(1.0-isReg[r, t1]) * (1.0 - isReg[r, t2]) + isReg[r, t2]*(1.0-isReg[r, t1]) + 0.5*(isReg[r, t1]*isReg[r, t2])
                F[t2, t1] = (1.0-Fik)*(1.0-isReg[r, t2]) * (1.0 - isReg[r, t1]) + isReg[r, t1]*(1.0-isReg[r, t2]) + 0.5*(isReg[r, t2]*isReg[r, t1])

                
                #Runge-Kutta market share dynamics (do not remove the divide-by-6, it is part of the algorithm)
                k_1 = S_i*S_k * (mewa[r, t1, t2]*F[t1, t2]*Gijmax[t1]*Gijmin[t2] - mewa[r, t2, t1]*F[t2, t1]*Gijmax[t2]*Gijmin[t1])
                k_2 = (S_i+dt*k_1/2)* (S_k-dt*k_1/2)*(mewa[r, t1, t2]*F[t1, t2]*Gijmax[t1]*Gijmin[t2] - mewa[r, t2, t1]*F[t2, t1]*Gijmax[t2]*Gijmin[t1])
                k_3 = (S_i+dt*k_2/2)* (S_k-dt*k_2/2)*(mewa[r, t1, t2]*F[t1, t2]*Gijmax[t1]*Gijmin[t2] - mewa[r, t2, t1]*F[t2, t1]*Gijmax[t2]*Gijmin[t1])
                k_4 = (S_i+dt*k_3)*(S_k-dt*k_3)  *(mewa[r, t1, t2]*F[t1, t2]*Gijmax[t1]*Gijmin[t2] - mewa[r, t2, t1]*F[t2, t1]*Gijmax[t2]*Gijmin[t1])

                dSik[t1, t2] = (k_1+2*k_2+2*k_3+k_4)*dt/T_Scal/6
                dSik[t2, t1] = -dSik[t1, t2]

        # Add in exogenous sales figures. These are blended with
        # endogenous result! Note that it's different from the
        # ExogSales specification!

        dSk = np.zeros((t2ti))
        dUk = np.zeros((t2ti))
        dUkTK = np.zeros((t2ti))
        dUkREG = np.zeros((t2ti))


        endo_shares = mews_dt[r, :, 0] + np.sum(dSik, axis=1)

        # Copy over load factors that do not change
        # Only applies to baseload and variable technologies
        mewl[r, :, 0] = mewl_dt[r, :, 0].copy()
        # new_capacity_idx = np.logical_and(mews_lag[r, :, 0]==0, mews[r, :, 0] > 0)
        for tech_idx in range(t2ti):
            if np.logical_and(mews_lag[r, tech_idx, 0]==0, endo_shares[tech_idx] > 0):
                    mewl[r, tech_idx, 0] = mwlo[r, tech_idx, 0]

        endo_gen = endo_shares * (mewdt[r]*1000/3.6) * mewl[r, :, 0] / np.sum(endo_shares * mewl[r, :, 0])

        endo_capacity = endo_gen / mewl[r, :, 0] / 8766

        Utot = np.sum(endo_capacity)
        Utot_scaled = (mewdt[r]*1000/3.6)/(8766*np.sum(endo_shares * mewl[r, :, 0]))



        # PV: Added a term to check that exogenous capacity is smaller than regulated capacity.
        # Regulations have priority over exogenous capacity
        reg_vs_exog = ((mwka[r, :, 0]) > mewr[r, :, 0]) & (mewr[r, :, 0] >= 0.0)
        mwka[r, :, 0] = np.where(reg_vs_exog, -1.0, mwka[r, :, 0])
        MWKA_scalar = 1.0



        # Correct for regulations using difference between endogenous capacity and capacity from last time step with endo shares
            
        dUkREG = -(endo_capacity - endo_shares*np.sum(mewk_dt[r, :, 0]))* isReg[r, :] 

        #calculate capcity additions or subtractions after regulations, to prevent subtractions being too large and causing negatve shares.

        dUkTK = mwka[r, :, 0] - (endo_capacity + dUkREG)
        dUkTK[mwka[r, :, 0] < 0.0] = 0.0

        # Check that exogenous capacity isn't too large
        # As a proxy, the sum of exogenous capacities can't be greater
        # than 95% of last year's capacity level.
        if (dUkTK.sum() > 0.95 * Utot):

            MWKA_scalar = dUkTK.sum() / (0.95 * Utot)

            dUkTK = dUkTK / MWKA_scalar


        # Sum effect of exogenous sales additions (if any) with
        # effect of regulations
        dUk = dUkTK + dUkREG
        #dUtot = np.sum(dUk)*Utot_scaled/Utot
        dUtot = np.sum(dUk)
        # Convert to market shares and make sure sum is zero
        # Note that FTT Power's market shares are not shares of capacity, so this calculation is skewed. Worth rethinking.
        # This equation comes from S = U/Utot. 
        # If S = MEWK*8766*np.sum(endo_shares * mewl[r, :, 0])/(mewdt[r]*1000/3.6) we can instead write S = U/Utot_scaled to correct this
        # dSk = dUk/Utot - Uk dUtot/Utot^2  (Chain derivative)
        dSk = np.divide(dUk, Utot) - endo_capacity*np.divide(dUtot, (Utot*Utot))


        # New market shares
       
        mews[r, :, 0] = endo_shares + dSk

        # Copy over load factors that do not change
        # Only applies to baseload and variable technologies
        mewl[r, :, 0] = mewl_dt[r, :, 0].copy()
        # new_capacity_idx = np.logical_and(mews_lag[r, :, 0]==0, mews[r, :, 0] > 0)
        for tech_idx in range(t2ti):
            if np.logical_and(mews_lag[r, tech_idx, 0]==0, mews[r, tech_idx, 0] > 0):
                    mewl[r, tech_idx, 0] = mwlo[r, tech_idx, 0]

        # Grid operators guess-estimate expected generation based on LF from last step
       
        mewg[r, :, 0] = mews[r, :, 0] * (mewdt[r]*1000/3.6) * mewl[r, :, 0] / np.sum(mews[r, :, 0] * mewl[r, :, 0])
        mewk[r, :, 0] = mewg[r, :, 0] / mewl[r, :, 0] / 8766

        if r == 2:
            x = 1+1

    return mews, mewl, mewg, mewk
