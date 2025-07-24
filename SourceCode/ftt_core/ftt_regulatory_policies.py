# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:40:57 2025

@author: Femke
"""

import numpy as np

def implement_regulatory_policies(endo_shares, endo_capacity, regions, shares,
                              exog_sales, regulation, isReg,
                              demand, demand_dt, no_it, titles):
    """
    Implement regulatory policies and calculate updated market shares.
    
    Parameters
    ----------
    endo_shares : ndarray
        Endogenous market shares (num_regions, num_techs)
    endo_capacity : ndarray  
        Endogenous capacity (num_regions, num_techs)
    regions : array_like
        List of region indices to process
    shares : ndarray
        Current market shares (num_regions, num_techs, 1)
    exog_sales : ndarray
        Exogenous sales additions (num_regions, num_techs, 1)
    regulation : ndarray
        Regulation capacity targets (num_regions, num_techs, 1)
    isReg : ndarray
        Regulation flags (num_regions, num_techs)
    demand : ndarray
        Current demand (num_regions,)
    demand_dt : ndarray
        Previous demand (num_regions,)
    no_it : int
        Number of time iterations
    titles : dict
        Title classifications
        
    Returns
    -------
    ndarray
        Updated market shares (num_regions, num_techs, 1)
    """
    
    shares_new = np.copy(shares)
    
    if len(regions) == 0:
        return shares_new
    
    # Convert regions to array for indexing
    regions_array = np.array(regions)
    
    # Extract data for processing regions only
    exog_sales_proc = exog_sales[regions_array, :, 0]  # (num_regions_proc, num_techs)
    regulation_proc = regulation[regions_array, :, 0]  # (num_regions_proc, num_techs)
    isReg_proc = isReg[regions_array, :]              # (num_regions_proc, num_techs)
    demand_proc = demand[regions_array]               # (num_regions_proc,)
    demand_dt_proc = demand_dt[regions_array]         # (num_regions_proc,)
    endo_capacity_proc = endo_capacity[regions_array] # (num_regions_proc, num_techs)
    endo_shares_proc = endo_shares[regions_array]     # (num_regions_proc, num_techs)
    
    # Calculate exogenous sales scalars for all regions
    exog_sales_sum = np.sum(exog_sales_proc, axis=1)  # (num_regions_proc,)
    max_allowed = 0.8 * demand_proc / 13               # (num_regions_proc,)
    
    # Check that exogenous sales additions aren't too large
    # As a proxy it can't be greater than 80% of the fleet size
    # divided by 13 (the average lifetime of vehicles)
    exog_sales_scalar = np.where(
        (exog_sales_sum > max_allowed) & (max_allowed > 0),
        exog_sales_sum / max_allowed,
        1.0
    )  # (num_regions_proc,)
    
    # Broadcast scalar for element-wise operations
    exog_sales_scalar_broadcast = exog_sales_scalar[:, np.newaxis]  # (num_regions_proc, 1)
    
    # Check endogenous capacity plus additions for a single time step does not exceed regulated capacity.
    scaled_exog = exog_sales_proc / exog_sales_scalar_broadcast / no_it  # (num_regions_proc, num_techs)
    reg_vs_exog = ((scaled_exog + endo_capacity_proc) > regulation_proc) & (regulation_proc >= 0.0)
    
    # Exogenous sales are yearly capacity additions. 
    # We need to split it up based on the number of time steps, and also scale it if necessary.
    dUkTK = np.where(reg_vs_exog, 0.0, scaled_exog)  # (num_regions_proc, num_techs)
    
    # Correct for regulations due to the stretching effect. This is the difference in capacity due only to rflt increasing.
    # This is the difference between capacity based on the endogenous capacity, and what the endogenous capacity would have been
    # if rflt (i.e. total demand) had not grown.
    demand_dt_broadcast = demand_dt_proc[:, np.newaxis]  # (num_regions_proc, 1)
    dUkREG = -(endo_capacity_proc - endo_shares_proc * demand_dt_broadcast) * isReg_proc
    
    # Sum effect of exogenous sales additions (if any) with effect of regulations.
    dUk = dUkTK + dUkREG  # (num_regions_proc, num_techs)
    dUtot = np.sum(dUk, axis=1)  # (num_regions_proc,)
    
    # Calculate changes to endogenous capacity, and use to find new market shares
    # Zero capacity will result in zero shares
    # All other capacities will be streched
    endo_capacity_sum = np.sum(endo_capacity_proc, axis=1)  # (num_regions_proc,)
    denominator = endo_capacity_sum + dUtot  # (num_regions_proc,)
    
    # Handle division by zero
    valid_denom = denominator != 0.0
    new_shares = np.zeros_like(endo_capacity_proc)
    
    if np.any(valid_denom):
        new_shares[valid_denom] = (
            (endo_capacity_proc[valid_denom] + dUk[valid_denom]) / 
            denominator[valid_denom, np.newaxis]
        )
    
    # Update shares_new for the processed regions
    shares_new[regions_array, :, 0] = new_shares
    
    return shares_new