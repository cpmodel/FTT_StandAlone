# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 15:20:59 2025

@author: Femke

"""

import numpy as np

def policies_old(rti, mewl, t2ti, endo_shares, mewdt, mewk_dt,
                 reg_constr, mwka, t, dt, no_it, mewr, mewk_lag):
    
    
    # Values to return
    mews = np.zeros((rti, t2ti, 1))
    mewg = np.zeros((rti, t2ti, 1))
    mewk = np.zeros((rti, t2ti, 1))
    
     
    for r in range(rti):
       
        endo_gen = endo_shares[r] * (mewdt[r]*1000/3.6) * mewl[r, :, 0] / np.sum(endo_shares[r] * mewl[r, :, 0])
        endo_capacity = endo_gen / mewl[r, :, 0] / 8766

        # Regulations have priority over exogenous capacity        

        # Correct for regulations using difference between endogenous capacity and capacity from last time step with endo shares
        dUkREG = -(endo_capacity - endo_shares[r] * np.sum(mewk_dt[r, :, 0])) * reg_constr[r, :]    


        # If MWKA is a ban or removal, base removal on endogenous capacity after regulation to ensure no negative shares
        condition1 = mwka[r, :, 0] < endo_capacity
        dUkMK = np.where(condition1, (mwka[r, :, 0] - (endo_capacity + dUkREG)) * (t / no_it), 0)
        
        # If MWKA is a target beyond the last year's capacity, treat as a kick-start.
        # Small additions will help the target be met.
        # Only do for MWKA > MWKL to prevent oscillations
        condition2 = (mwka[r, :, 0] > endo_capacity) & (mwka[r, :, 0] > mewk_lag[r, :, 0])
        dUkMK = np.where(condition2, (mwka[r, :, 0] - endo_capacity) * (t / no_it), dUkMK)
        
        # Regulations have priority over exogenous capacity
        condition3 = (mwka[r, :, 0] < 0) | ((mewr[r, :, 0] >= 0.0) & (mwka[r, :, 0] > mewr[r, :, 0]))
        dUkMK = np.where(condition3, 0.0, dUkMK)


        # Sum effect of exogenous sales additions (if any) with
        # effect of regulations
        dUk = dUkREG + dUkMK

        dUtot = np.sum(dUk)
 

        # Use modified capacity and modified total capacity to recalulate market shares
        # This method will mean any capacities set to zero will result in zero shares
        # It avoids negative shares
        # All other capacities will be stretched, depending on the magnitude of dUtot and how much of a change this makes to total capacity
        # If dUtot is small and implemented in a way which will not under or over estimate capacity greatly, MWKA is fairly accurate

        # New market shares
        if np.sum(endo_capacity) + dUtot > 0:
            mews[r, :, 0] = (endo_capacity + dUk) / (np.sum(endo_capacity) + dUtot)
       
       
        mewg[r, :, 0] = mews[r, :, 0] * (mewdt[r]*1000/3.6) * mewl[r, :, 0] / np.sum(mews[r, :, 0] * mewl[r, :, 0])
        mewk[r, :, 0] = mewg[r, :, 0] / mewl[r, :, 0] / 8766
        
    
    return mews, mewl, mewg, mewk


