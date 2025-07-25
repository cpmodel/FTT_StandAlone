# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:40:57 2025

@author: Femke
"""

import numpy as np

def scale_exogenous_sales(exog_sales, demand, no_it, lifetimes):
    """
    Calculate the exogenous sales scalars and scaled exogenous sales.

    Parameters
    ----------
    exog_sales : ndarray
        Exogenous sales additions for the processed regions (num_regions, num_techs).
    demand : ndarray
        Current demand for the processed regions (num_regions,).
    no_it : int
        Number of time iterations.
    lifetimes : ndarray
        Lifetimes of technologies (num_regions, num_techs).

    Returns
    -------
    scaled_exog : ndarray
        Scaled exogenous sales (num_regions, num_techs).
    exog_sales_scalar_broadcast : ndarray
        Broadcasted exogenous sales scalar (num_regions, 1).
    """
    avg_lifetime = np.mean(lifetimes, axis=1)  # Average lifetime per region (num_regions,)

    exog_sales_sum = np.sum(exog_sales, axis=1)  # (num_regions,)
    max_allowed = 0.8 * demand / avg_lifetime  # (num_regions,)

    # Check that exogenous sales additions aren't too large
    exog_sales_scalar = np.where(
        (exog_sales_sum > max_allowed) & (max_allowed > 0),
        exog_sales_sum / max_allowed,
        1.0
    )  # (num_regions,)

    # Scale exogenous sales
    scaled_exog = exog_sales / exog_sales_scalar[:, np.newaxis] / no_it  # (num_regions, num_techs)

    return scaled_exog

def calculate_regulation_effects(endo_capacity, endo_shares, demand, investment_stop_share, scaled_exog, regulation):
    """
    Calculate the effects of regulations on capacity and shares.

    Parameters
    ----------
    endo_capacity : ndarray
        Endogenous capacity for the processed regions (num_regions, num_techs).
    endo_shares : ndarray
        Endogenous shares for the processed regions (num_regions, num_techs).
    demand : ndarray
        Previous demand for the processed regions (num_regions,).
    investment_stop_share : ndarray
        Share of investment in a certain technology stopped (num_regions, num_techs).
    scaled_exog : ndarray
        Scaled exogenous sales (num_regions, num_techs).
    regulation : ndarray
        Regulation capacity targets for the processed regions (num_regions, num_techs).

    Returns
    -------
    dUk : ndarray
        Total capacity changes due to regulations and exogenous sales (num_regions, num_techs).
    dUtot : ndarray
        Total capacity changes summed across technologies (num_regions,).
    """
    # Check endogenous capacity plus additions for a single time step does not exceed regulated capacity.
    # Regulations have priority over exogenous capacity
    reg_vs_exog = ((scaled_exog + endo_capacity) > regulation) & (regulation >= 0.0)

    # Exogenous sales are yearly capacity additions.
    dUk_exog_sales = np.where(reg_vs_exog, 0.0, scaled_exog)  # (num_regions, num_techs)

    # Correct for regulations due to the stretching effect: the difference in capacity due to demand increasing only.
    # This is the difference between capacity based on the endogenous capacity, and what the endogenous capacity
    # would have been if total demand had not grown.
    demand_broadcast = demand[:, np.newaxis]  # (num_regions, 1)
    dUk_reg = -(endo_capacity - endo_shares * demand_broadcast) * investment_stop_share

    # Sum effect of exogenous sales additions with effect of regulations.
    dUk = dUk_exog_sales + dUk_reg  # (num_regions, num_techs)
    dUtot = np.sum(dUk, axis=1)  # (num_regions,)

    return dUk, dUtot

def implement_regulatory_policies(endo_shares, endo_capacity, regions, shares,
                              exog_sales, regulation, investment_stop_share,
                              demand, demand_dt, no_it, lifetimes):
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
    investment_stop_share : ndarray
        Share of investment in a certain technology stopped (num_regions, num_techs).
    demand : ndarray
        Current demand (num_regions,)
    demand : ndarray
        Demand in previous timestep (num_regions,)
    no_it : int
        Number of time iterations
    lifetimes : ndarray
        Lifetimes of technologies (num_regions, num_techs).
        
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
    
    # Select all relevant regions
    exog_sales, regulation, investment_stop_share, demand, demand_dt, endo_capacity, endo_shares, lifetimes = slice_region_data(
        [exog_sales[:, :, 0], regulation[:, :, 0], investment_stop_share, demand, demand_dt, endo_capacity, endo_shares, lifetimes],
        regions_array
    )

    # Calculate exogenous sales effects (ensuring they are not larger than possible in the system)
    scaled_exog_sales = scale_exogenous_sales(
        exog_sales, demand, no_it, lifetimes
    )

    # Calculate regulation and exog sales effects
    dUk, dUtot = calculate_regulation_effects(
        endo_capacity, endo_shares, demand_dt, investment_stop_share,
        scaled_exog_sales, regulation
    )
    
    # Calculate changes to endogenous capacity
    endo_capacity_sum = np.sum(endo_capacity, axis=1)  # (num_regions,)
    total_capacity = endo_capacity_sum + dUtot  # (num_regions,)
    
    # Compute new shares based on updated capacity
    new_shares = np.divide(
        endo_capacity + dUk,
        total_capacity[:, None],
        out=np.zeros_like(endo_capacity),
        where=total_capacity[:, None] != 0
    )
    
    # Update shares_new for the processed regions
    shares_new[regions_array, :, 0] = new_shares
    
    return shares_new

def slice_region_data(arrays, regions_array):
    """
    Slice data for the specified regions.

    Parameters
    ----------
    arrays : list of ndarray
        List of arrays to slice.
    regions_array : ndarray
        Array of region indices to slice.

    Returns
    -------
    list of ndarray
        Sliced arrays for the specified regions.
    """
    return [arr[regions_array] for arr in arrays]