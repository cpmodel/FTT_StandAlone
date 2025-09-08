# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 15:20:59 2025

@author: Femke

"""

import numpy as np

def policies_old(rti, mewl_dt, t2ti, mwlo, mews_lag, endo_shares, mewdt, mewk_dt,
                 isReg, mwka, t, dt, no_it, mewr, mewk_lag):
    
    
    # Values to return
    mews = np.zeros((rti, t2ti, 1))
    mewl = np.zeros((rti, t2ti, 1))
    mewg = np.zeros((rti, t2ti, 1))
    mewk = np.zeros((rti, t2ti, 1))
    
    
    for r in range(rti):
        
        dUk = np.zeros((t2ti))
        dUkREG = np.zeros((t2ti))
        
        # Copy over load factors that do not change
        # Only applies to baseload and variable technologies
        mewl[r, :, 0] = mewl_dt[r, :, 0].copy()
        
        # new_capacity_idx = np.logical_and(mews_lag[r, :, 0]==0, mews[r, :, 0] > 0)
        for tech_idx in range(t2ti):
            if np.logical_and(mews_lag[r, tech_idx, 0]==0, endo_shares[r, tech_idx] > 0):
                    mewl[r, tech_idx, 0] = mwlo[r, tech_idx, 0]

        endo_gen = endo_shares[r] * (mewdt[r]*1000/3.6) * mewl[r, :, 0] / np.sum(endo_shares[r] * mewl[r, :, 0])

        endo_capacity = endo_gen / mewl[r, :, 0] / 8766

        


        # PV: Added a term to check that exogenous capacity is smaller than regulated capacity.
        # Regulations have priority over exogenous capacity
        
        #reg_vs_exog = ((mwka[r, :, 0]) > mewr[r, :, 0]) & (mewr[r, :, 0] >= 0.0)
        #mwka[r, :, 0] = np.where(reg_vs_exog, -1.0, mwka[r, :, 0])


        # Correct for regulations using difference between endogenous capacity and capacity from last time step with endo shares
            
        dUkREG = -(endo_capacity - endo_shares[r] * np.sum(mewk_dt[r, :, 0])) * isReg[r, :] 

        
        # =====================================================================
        # Old calculations. If we ensure that these are scaled with (t / no_it), they are more accurate.
        # Calculate capacity additions or subtractions after regulations, to prevent subtractions being too large and causing negatve shares.
        
        #Utot = np.sum(endo_capacity)
        # dUkTK = np.zeros((t2ti))
        # dUkTK = mwka[r, :, 0] - (endo_capacity + dUkREG)
        # dUkTK[mwka[r, :, 0] < 0.0] = 0.0

        # # Check that exogenous capacity isn't too large
        # # As a proxy, the sum of exogenous capacities can't be greater
        # # than 95% of last year's capacity level.
        # if (dUkTK.sum() > 0.95 * Utot):

        #     MWKA_scalar = dUkTK.sum() / (0.95 * Utot)
        #     dUkTK = dUkTK / MWKA_scalar

        


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
        
    
    # Then check the results
    check_shares_output(mews, mewl, mewg, mewk)
    
    return mews, mewl, mewg, mewk


def check_shares_output(mews, mewl, mewg, mewk):
    '''Check for nans, whether shares add up to 1, and whether there are
    negative shares '''
    
    # Check for NaN values in 'mewk'
    if np.isnan(mewk).any():
        nan_indices_mewk = np.where(np.isnan(mewk))
        raise ValueError(f"NaN values detected in 'mewk' at indices: {nan_indices_mewk}. Please check shares.")
        
    region_sums = mews.sum(axis=1)
    if not np.all(np.abs(region_sums - 1.0) < 1e-8):
        raise ValueError("Sum of MEWS does not add up to 1")
    
    if np.any(mews < 0.0):
        r_err, t_err = np.unravel_index(np.nanargmin(mews[:, :, 0]), mews[:, :, 0].shape)
        print(f'{mews[r_err, t_err, 0]}, Region: {r_err} and technology {t_err}')
        raise ValueError("Negative MEWS found")