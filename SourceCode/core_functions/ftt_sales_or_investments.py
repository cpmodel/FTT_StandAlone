# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:09:50 2024

@author: Cormac Lynch & Femke Nijsse
"""

import numpy as np


def get_sales(cap, cap_dt, cap_lag, shares, shares_dt, sales_or_investment, 
               timescales, dt, t):
    """
    Calculate new sales/investments for all FTT models

    This function calculates the amount of new vehicle sales required (e.g. TEWI).

    1. If capacity has grown, the difference between the new capacity and the
       old.
    2. The amount of existing capacity that is depreciated (retired) as it
       reaches its end of life.
    
    Capacity depreciation is based on the technology lifetime in the
    cost matrix. 1/lifetime is depreciated each year.

    Parameters
    -----------
    cap: np.array
        capacity (e.g. TEWK, MEWK)
    cap_dt: np.array
        capacity at the previous timestep dt
    cap_lag: np.array
        capacity at the previous year
    shares: np.array
        current shares (e.g. TEWS, MEWS)
    shares_dt: np.array
        shares at the previous timestep dt
    dt: float
        dt is timestep: 1 / (number of timesteps).
    t: int
        t is current time

    Returns
    ----------
    investment_or_sales: np.array
        investment or sales up to and including time t
    investment_or_sales_t: np.array
        investment or sales at time t
    
    """
    # Find capacity growth between this timestep and last
    cap_growth = cap[:, :, 0] - cap_dt[:, :, 0]

    # Find share growth between this timestep and last
    share_growth = shares[:, :, 0] - shares_dt[:, :, 0]

    # Calculate share_depreciation
    share_depreciation = shares_dt[:, :, 0] * dt / timescales
    
    # If there is growth, depreciation is fully replaced.
    # Where capacity has decreased, we only replace depreciated capacity
    # if the depreciation > capacity loss
    conditions = [
        (cap_growth > 0.0),
        (-share_depreciation < share_growth) & (share_growth < 0)
    ]

    outputs = [
        cap_dt[:, :, 0] * dt / timescales,
        (share_growth + share_depreciation) * cap_lag[:, :, 0]
    ]

    # Use np.select to choose an output for each condition
    cap_drpctn = np.select(conditions, outputs, default=0)
            
    # Ensure no negative values
    cap_drpctn = np.maximum(cap_drpctn, 0)
    cap_drpctn = cap_drpctn[:, :, np.newaxis]

    # Find total additions at time t
    # If capacity has grown, additions equal the difference +
    # depreciation from the previous year.
    # Otherwise, additions just equal depreciations.
    tewi_t = np.zeros(sales_or_investment.shape)
    tewi_t[:, :, 0] = np.where(cap_growth[:, :] > 0.0,
                                cap_growth[:, :] + cap_drpctn[:, :, 0],
                                cap_drpctn[:, :, 0])
        
    # Add additions at iteration t to total annual additions
    sales_or_investment[:, :, 0] = sales_or_investment[:, :, 0] + tewi_t[:, :, 0]

    return sales_or_investment, tewi_t
