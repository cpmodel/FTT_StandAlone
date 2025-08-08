# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:40:57 2025

@author: Femke

Contains three types of regulatory policies
1. Exogenous sales (which come in addition to endogenous sales)
2. Exogenous capacity
3. Correction for regulation under stretching

"""

import numpy as np


def exogenous_sales(
        exog_sales, demand, endo_capacity, regulation_cap, no_it, lifetimes):
    """
    Calculate exogenous sales, scaled down if they exceed 80% of overall sales.
    Where exogenous sales contradict regulatory constraints, regulation has priority.
    Returns changes in capacity per region and tech
    """
    avg_lifetime = np.mean(lifetimes, axis=1)  # Average lifetime per region

    exog_sales_sum = np.sum(exog_sales, axis=1)  
    max_allowed = 0.8 * demand / avg_lifetime 

    # Scale down exogenous sales if they exceed 80% of total sales
    scaled_exog_sales = np.where(
        ((exog_sales_sum > max_allowed) & (max_allowed > 0))[:, None],
        exog_sales * (max_allowed / exog_sales_sum)[:, None],
        exog_sales
        ) / no_it
    
    # Endogenous capacity + additions must not exceed regulated capacity.
    # If they do, regulations have priority over exogenous capacity.
    reg_overrides_exog = (
        ((endo_capacity + scaled_exog_sales) > regulation_cap) & 
        (regulation_cap >= 0.0) )
    dUk_exog_sales = np.where(reg_overrides_exog, 0.0, scaled_exog_sales)

    return dUk_exog_sales


def exogenous_capacity(
        exogenous_capacity, endo_capacity, dUk_other, regulation_cap,
        t, no_it
        ):
    """
    Calculate the change to endogenous capacity to reach exogenous capacities.
    The goal is to move linearly to the new capacity over one year.
    Regulation override exogenous capacity in case of a conflict
    
    """
    # Note: incorrect in E3ME
    
    # When no_it = 20, initially you need to close 1/20th of the remaining
    # gap. In the next step, you need 1/20th of the original step, which is
    # 1/19th of the remaining gap etc.
    share_remaining_gap_to_close = 1 / (no_it - t + 1)
    
    capacity_gap = exogenous_capacity - (endo_capacity + dUk_other)
    dUk_exog_cap = capacity_gap * share_remaining_gap_to_close
    
    reg_overrides_exog = (exogenous_capacity > regulation_cap) & (regulation_cap >= 0)
        
    # Only apply where exogenous capacities turned on
    dUk_exog_cap = np.where(exogenous_capacity >= 0, dUk_exog_cap, 0)
    
    # Do not apply where regulations override exogenous capacity
    dUk_exog_cap = np.where(reg_overrides_exog, 0, dUk_exog_cap)
                    
    return dUk_exog_cap
    

def regulation_correction(
        endo_capacity, endo_shares, cap_sum_demand, reg_constr):
    """
    Demand growth raises capacities while shares stay the same (stretching). 
    This extra capacity is not yet regulated. Correct for this underregulation.

    """
    # Note: now aligned with E3ME, but no where statement before
    cap_growth_from_stretching = endo_capacity - endo_shares * cap_sum_demand
    dUk_reg = np.where(cap_growth_from_stretching > 0,
                   -cap_growth_from_stretching * reg_constr,
                   0)
    
    return dUk_reg
    

