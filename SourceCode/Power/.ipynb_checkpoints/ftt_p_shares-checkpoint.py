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


# %% JIT-compiled shares equation
# -----------------------------------------------------------------------------
#@njit(fastmath=True)
def shares(dt, T_Scal, e_demand, e_demand_step, mews_dt, metc_dt, mtcd_dt,
           mwka, mes1_dt, mes2_dt, mewa, isReg, mewk_dt, mewk_lag, mewr,
           mewl_dt, mews_lag, mwlo, lag_demand, rti, t2ti):

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

        if e_demand[r] == 0.0:
            continue

        if r==40:
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
                Aik = mewa[r, t1, t2]
                Aki = mewa[r, t2, t1]

                # Propagating width of variations in perceived costs
                dFik = np.sqrt(2) * np.sqrt((mtcd_dt[r, t1, 0]*mtcd_dt[r, t1, 0] + mtcd_dt[r, t2, 0]*mtcd_dt[r, t2, 0]))

                # Consumer preference incl. uncertainty
                Fik = 0.5*(1+np.tanh(1.25*(metc_dt[r, t2, 0]-metc_dt[r, t1, 0])/dFik))

                # Preferences are then adjusted for regulations
                F[t1, t2] = Fik*(1.0-isReg[r, t1]) * (1.0 - isReg[r, t2]) + isReg[r, t2]*(1.0-isReg[r, t1]) + 0.5*(isReg[r, t1]*isReg[r, t2])
                F[t2, t1] = (1.0-Fik)*(1.0-isReg[r, t2]) * (1.0 - isReg[r, t1]) + isReg[r, t1]*(1.0-isReg[r, t2]) + 0.5*(isReg[r, t2]*isReg[r, t1])

                # Market share dynamics
                dSik[t1, t2] = S_i*S_k * (Aik*F[t1, t2]*Gijmax[t1]*Gijmin[t2] - Aki*F[t2, t1]*Gijmax[t2]*Gijmin[t1])*dt/T_Scal
                dSik[t2, t1] = -dSik[t1, t2]

        # Add in exogenous sales figures. These are blended with
        # endogenous result! Note that it's different from the
        # ExogSales specification!
        Utot = np.sum(mewk_dt[:, :, 0], axis=1)
        Utot_lag = np.sum(mewk_lag[:, :, 0], axis=1)
        dSk = np.zeros((t2ti))
        dUk = np.zeros((t2ti))
        dUkTK = np.zeros((t2ti))
        dUkREG = np.zeros((t2ti))

#def shares(e_demand, mews_dt, metc_dt, mtcd_dt, mwka, mes1_dt, mes2_dt, mewa, isReg, mewk_dt, mewk_lag, mewr, rti, t2ti):
        # PV: Added a term to check that exogenous capacity is smaller than regulated capacity.
        # Regulations have priority over exogenous capacity
        reg_vs_exog = ((mwka[r, :, 0]) > mewr[r, :, 0]) & (mewr[r, :, 0] >= 0.0)
        mwka[r, :, 0] = np.where(reg_vs_exog, -1.0, mwka[r, :, 0])
        MWKA_scalar = 1.0

        dUkTK = mwka[r, :, 0] - mewk_dt[r, :, 0]
        dUkTK[mwka[r, :, 0] < 0.0] = 0.0
        #dUkTK[dUkTK < 0.0] = 0.0
        # Check that exogenous capacity isn't too large
        # As a proxy, the sum of exogenous capacities can't be greater
        # than 90% of last year's capacity level.
        if (dUkTK.sum() > 0.95 * Utot[r]):

            MWKA_scalar = dUkTK.sum() / (0.95 * Utot[r])

            dUkTK = dUkTK / MWKA_scalar

        # Correct for regulations
        if Utot_lag[r] > 0.0 and Utot[r] > 0.0:

            dUkREG = -(mewk_dt[r, :, 0]
                      * (e_demand_step[r] / (e_demand[r] - e_demand_step[r]))
                      * isReg[r, :].reshape((t2ti)))
        # Sum effect of exogenous sales additions (if any) with
        # effect of regulations
        dUk = dUkTK + dUkREG
        dUtot = np.sum(dUk)
        # Convert to market shares and make sure sum is zero
        # dSk = dUk/Utot - Uk dUtot/Utot^2  (Chain derivative)
        dSk = np.divide(dUk, Utot[r]) - mewk_dt[r, :, 0]*np.divide(dUtot, (Utot[r]*Utot[r]))

        # Correct for overestimation of reduction in share space due to regulation
        # dSk[data_dt['MEWS'][r, :, 0] + dSk < 0] = -data_dt['MEWS'][r, :, 0]
        dSk = np.where(mews_dt[r, :, 0] + dSk < 0, -mews_dt[r, :, 0], dSk)

        # New market shares
        mews[r, :, 0] = mews_dt[r, :, 0] + np.sum(dSik, axis=1) + dSk

        # Copy over load factors that do not change
        # Only applies to baseload and variable technologies
        mewl[r, :, 0] = mewl_dt[r, :, 0].copy()
        # new_capacity_idx = np.logical_and(mews_lag[r, :, 0]==0, mews[r, :, 0] > 0)
        for tech_idx in range(t2ti):
            if np.logical_and(mews_lag[r, tech_idx, 0]==0, mews[r, tech_idx, 0] > 0):
                    mewl[r, tech_idx, 0] = mwlo[r, tech_idx, 0]

        # Grid operators guess-estimate expected generation based on LF from last step
        mewg[r, :, 0] = mewk_dt[r, :, 0] * mewl[r, :, 0] * e_demand[r] / (lag_demand[r]) * 8766
        mewk[r, :, 0] = mewg[r, :, 0] / mewl[r, :, 0] / 8766

        if r == 40:
            x = 1+1

    return mews, mewl, mewg, mewk
