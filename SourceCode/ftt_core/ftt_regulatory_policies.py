# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:40:57 2025

@author: Femke
"""

import numpy as np

def implement_regulatory_policies(endo_shares, endo_capacity, regions, shares,
                              exog_sales, regulation, isReg,
                              demand, demand_dt, no_it, titles):
    
    shares_new = np.copy(shares)
        
    for r in regions:
        
        dUkTK = np.zeros([len(endo_shares[0])])
        dUkREG = np.zeros([len(endo_shares[0])])
        exog_sales_scalar = 1.0

        # Check that exogenous sales additions aren't too large
        # As a proxy it can't be greater than 80% of the fleet size
        # divided by 13 (the average lifetime of vehicles)
        if (exog_sales[r, :, 0].sum() > 0.8 * demand[r] / 13):

            exog_sales_scalar = exog_sales[r, :,
                                       0].sum() / (0.8 * demand[r] / 13)
        # Check endogenous capacity plus additions for a single time step does not exceed regulated capacity.
        reg_vs_exog = ((exog_sales[r, :, 0] / exog_sales_scalar / no_it + endo_capacity[r])
                       > regulation[r, :, 0]) & (regulation[r, :, 0] >= 0.0)

        # Exogenous sales are yearly capacity additions. 
        # We need to split it up based on the number of time steps, and also scale it if necessary.
        dUkTK = np.where(reg_vs_exog, 0.0,
                         exog_sales[r, :, 0] / exog_sales_scalar / no_it)

        # Correct for regulations due to the stretching effect. This is the difference in capacity due only to rflt increasing.
        # This is the difference between capacity based on the endogenous capacity, and what the endogenous capacity would have been
        # if rflt (i.e. total demand) had not grown.

        dUkREG = -(endo_capacity[r] - endo_shares[r] *
                   demand_dt[r, np.newaxis]) * isReg[r, :].reshape([len(titles['VTTI'])])

        # Sum effect of exogenous sales additions (if any) with effect of regulations.
        dUk = dUkTK + dUkREG
        dUtot = np.sum(dUk)

        # Calculate changes to endogenous capacity, and use to find new market shares
        # Zero capacity will result in zero shares
        # All other capacities will be streched

        shares_new[r, :, 0] = (
            endo_capacity[r] + dUk) / (np.sum(endo_capacity[r]) + dUtot)
    
            
    return shares_new