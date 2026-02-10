# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:59:07 2024

@author: Femke Nijsse
"""

import numpy as np

       
def implement_shares_policies(endo_capacity, endo_shares, 
                              titles, zwsa, zreg, isReg,
                              sum_over_classes, n_veh_classes, Utot_dt, no_it):
        
    # Add in exogenous sales figures. These are blended with
    # endogenous result! Note that it's different from the
    # ExogSales specification!
    
    zews = np.zeros((len(titles['RTI']), len(titles['FTTI']), 1))

    
    for r in range(len(titles['RTI'])):
        
        dUk = np.zeros([len(titles['FTTI'])])
        dUkTK = np.zeros([len(titles['FTTI'])])
        dUkREG = np.zeros([len(titles['FTTI'])])
        
    
        # Check that exogenous sales additions aren't too large
        # As a proxy it can't be greater than 80% of the class fleet size
        # divided by 15 (the average lifetime of freight vehicles)
        sum_zwsa = sum_over_classes(zwsa)
        
        for veh_class in range(n_veh_classes):
            if (sum_zwsa[r, veh_class, 0] > 0.8 * Utot_dt[r, veh_class] / 15 and
                    Utot_dt[r, veh_class] > 0):
        
                # ZWSA_scalar[veh_class] = sum_zwsa[veh_class] / (0.8 * Utot_dt[r] / 15)
                zwsa[r, veh_class::n_veh_classes] /= (
                                sum_zwsa[r, veh_class, 0] / (0.8 * Utot_dt[r, veh_class] / 15) )

        # Check that exogenous capacity is smaller than regulated capacity
        # Regulations have priority over exogenous capacity
        reg_vs_exog = ((zwsa[r, :, 0] / no_it + endo_capacity[r]) 
                      > zreg[r, :, 0]) & (zreg[r, :, 0] >= 0.0)
     
        # ZWSA is yearly capacity additions. We need to split it up based on the number of time steps, and also scale it if necessary.
        dUkTK =  np.where(reg_vs_exog, 0.0, zwsa[r, :, 0] / no_it)
        
        Utot_dt_reshaped = np.tile(Utot_dt, (1, zews.shape[1] // Utot_dt.shape[1], 1))[:, :, 0] # Reshape to 71 x #tech (duplicate info)

        # Correct for regulations due to the stretching effect. This is the difference in capacity due only to rflt increasing.
        # This is the difference between capacity based on the endogenous capacity, and what the endogenous capacity would have been
        # if rflz (i.e. total vehicles) had not grown.
        dUkREG = -(endo_capacity[r] - endo_shares[r] * Utot_dt_reshaped[r]) \
                 * isReg[r, :].reshape([len(titles['FTTI'])])
                                   
        # Sum effect of exogenous sales additions (if any) with effect of regulations. 
        dUk = dUkTK + dUkREG
        dUtot_dt = [dUk[i::n_veh_classes].sum() for i in range(n_veh_classes)]
    
        # Calculate changes to endogenous capacity, and use to find new market shares
        # Zero capacity will result in zero shares
        # All other capacities will be streched
        for veh_class in range(n_veh_classes):
            denominator = (np.sum(endo_capacity[r, veh_class::n_veh_classes]) + dUtot_dt[veh_class]) 
            if denominator > 0:
                zews[r, veh_class::n_veh_classes, 0] = ( 
                        (endo_capacity[r, veh_class::n_veh_classes] + dUk[veh_class::n_veh_classes])
                        / denominator )
    
            
    return zews
