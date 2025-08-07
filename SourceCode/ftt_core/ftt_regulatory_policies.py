# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:40:57 2025

@author: Femke
"""

import numpy as np

from SourceCode.support.divide import divide


def scale_exogenous_sales(
        exog_sales, demand, endo_capacity, regulation_cap, no_it, lifetimes):
    """
    Calculate the exogenous sales scalars and scaled exogenous sales.

    Parameters
    ----------
    exog_sales : ndarray
        Exogenous sales additions by region and tech (num_regions, num_techs).
    demand : ndarray
        Current demand (num_regions,).
    endo_capacity : ndarray
        Endogenous capacity by region and tech
    regulation_cap : ndarray
        Regulation capacity targets (num_regions, num_techs, 1)
    no_it : int
        Number of time iterations.
    lifetimes : ndarray
        Lifetimes of technologies (num_regions, num_techs).

    Returns
    -------
    scaled_exog_sales : ndarray
        Scaled exogenous sales (num_regions, num_techs).
    """
    avg_lifetime = np.mean(lifetimes, axis=1)  # Average lifetime per region

    exog_sales_sum = np.sum(exog_sales, axis=1)  
    max_allowed = 0.8 * demand / avg_lifetime 

    # Scaling down factor where exogenous sales exceed 80% of total sales
    exog_sales_scalar = np.where(
        (exog_sales_sum > max_allowed) & (max_allowed > 0),
        exog_sales_sum / max_allowed,
        1.0
    )  

    # Scale exogenous sales
    scaled_exog_sales = exog_sales / exog_sales_scalar[:, np.newaxis] / no_it 
    
    # Endogenous capacity + additions must not exceed regulated capacity.
    # If they do, regulations have priority over exogenous capacity
    reg_vs_exog = ((endo_capacity + scaled_exog_sales) > regulation_cap) & (regulation_cap >= 0.0)

    # Remove exogenous sales where regulations trump
    dUk_exog_sales = np.where(reg_vs_exog, 0.0, scaled_exog_sales)

    return dUk_exog_sales

def exogenous_capacity(exogenous_capacity, endo_capacity, dUk_other,
                       t, no_it
        ):
    """
    Calculate the change to endogenous capacity to reach exogenous capacities
    
    Parameters
    ----------
    exogenous_capacity : ndarray
    endo_capacity : ndarray
    dUk_other : changes to capacities from other policy instruments
    t : timestep
    no_it : number of iterations
    
    Returns
    ----------
    Changes in endogenous capacity to reach exogenous capacity at t=no_it
    """
    # TODO (temporary note: this is wrong in E3ME, as the old time step scaling 
    # t/no_it, led to too big a change at t=2 and t=3.)
    # Initially, you need 1/no_it th of the remaining step, then 1/(no_it - 1) etc
    # For instance, when no_it = 4, initially you need to close 1/4th of the remaining
    # gap (3/4 still to close). In the next step, you need 1/4th of the original step, which is
    # 1/3 of the remaining gap etc.
    share_remaining_gap_to_close = 1 / (no_it - t + 1)
    
    dUk_exog_cap = (exogenous_capacity - (endo_capacity + dUk_other)) * share_remaining_gap_to_close
                    
    return dUk_exog_cap
    

def regulation_correction(endo_capacity, endo_shares, demand_dt, reg_constr):
    """
    If there is demand growth, the shares equation may have underregulation.
    Correct for this underregulation.
    
    Parameters
    ----------
    endo_capacity : ndarray
        Endogenous capacity by region and tech
    endo_shares : ndarray
        Endogenous shares by region and tech
    demand_dt : ndarray
        Previous demand
    reg_constr : ndarray
            Share of investment in a certain technology stopped 
    """
    
    # Correct for the stretching effect: capacity added due to demand growth alone.
    # Equals actual endogenous capacity minus what it would be with constant demand.
    # TODO: Now aligned with E3ME, but no where statement before
    cap_growth_from_stretching = endo_capacity - endo_shares * demand_dt[:, np.newaxis]
    dUk_reg = np.where(cap_growth_from_stretching > 0,
                   -cap_growth_from_stretching * reg_constr,
                   0)
    return dUk_reg
    

def implement_regulatory_policies(
        endo_shares, endo_capacity, regions, shares,
        exog_sales, regulation_cap, reg_constr,
        demand, demand_dt, no_it, t, lifetimes):
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
    regulation_cap : ndarray
        Regulation capacity targets (num_regions, num_techs, 1)
    reg_constr : ndarray
        Share of investment in a certain technology stopped (num_regions, num_techs).
    demand : ndarray
        Current demand (num_regions,)
    demand_dt : ndarray
        Demand in previous timestep (num_regions,)
    no_it : int
        Number of time iterations
    t : int
        Current time step
    lifetimes : ndarray
        Lifetimes of technologies (num_regions, num_techs).
        
    Returns
    -------
    ndarray
        Updated market shares (num_regions, num_techs, 1)
    """
    
    shares_new = np.copy(shares)
    
    
    # Select all relevant regions
    exog_sales, regulation_cap, reg_constr, demand, demand_dt, endo_capacity, endo_shares, lifetimes = \
        slice_region_data(
        [exog_sales[:, :, 0], regulation_cap[:, :, 0], reg_constr, demand, demand_dt, endo_capacity, endo_shares, lifetimes],
        regions
    )

    # Calculate exogenous sales effects, capped at maximum sales
    dUk_exog_sales = scale_exogenous_sales(
        exog_sales, demand, no_it, lifetimes
    )
    
    # Calculate correction for possible underregulation
    dUk_reg = regulation_correction(
        endo_capacity, endo_shares, demand_dt, reg_constr)
    
    dUk_exog_cap = exogenous_capacity(
        exogenous_capacity, endo_capacity, dUk_exog_sales + dUk_reg, t, no_it)

    
    # Sum effect of exogenous sales with possible correction for underregulation
    dUk = dUk_exog_sales + dUk_reg  + dUk_exog_cap # (num_regions, num_techs)
    dUtot = np.sum(dUk, axis=1)  # (num_regions,)
    
    # Calculate total capacity in each region
    total_capacity = np.sum(endo_capacity, axis=1) + dUtot  
    
    # Compute new shares based on updated capacity (fill in zeroes for divide by zero)
    new_shares = divide(endo_capacity + dUk, total_capacity[:, None])
    
    # Update shares_new for the processed regions
    shares_new[regions, :, 0] = new_shares
    
    return shares_new

def slice_region_data(arrays, regions):
    """
    Slice data for the specified regions.

    Parameters
    ----------
    arrays : list of ndarray
        List of arrays to slice.
    regions : ndarray
        Array of region indices to slice.

    Returns
    -------
    list of ndarray
        Sliced arrays for the specified regions.
    """
    return [arr[regions] for arr in arrays]