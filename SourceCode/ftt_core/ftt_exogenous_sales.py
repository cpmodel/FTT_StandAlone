# -*- coding: utf-8 -*-
"""
=========================================
ftt_exogenous_sales.py
=========================================
Contains the exogenous sales regulatory policy for FTT modules.

Functions included:
    - exogenous_sales
        Calculate change in capacity from exogenous sales
"""

import numpy as np


def exogenous_sales(
        exog_sales, demand, endo_capacity, regulation_cap, no_it, lifetimes):
    """
    Compute constrained exogenous sales (which are capacity additions)
    
    Exogenous sales are applied in addition to endogenous sales/capacity changes.
    
    Exogenous sales must sum to zero in each region
    # TODO: add input validation when data is read in    
    
    To limit distortions, exogenous sales are capped at 80% of replacement sales
    and scaled down if exceeded. Regulatory capacity limits take precedence.
    
    """
    # Technology-level annual replacement rate
    replacement_rate = 1.0 / np.where(lifetimes > 0, lifetimes, 1.0)

    # 80% of sum of replacement sales (approximation)
    max_allowed = 0.8 * np.sum(endo_capacity * replacement_rate, axis=1)

    # Sum over technologies with positive sales
    exog_sales_sum = np.sum(exog_sales * (exog_sales > 0), axis=1)
    safe_exog_sum = np.where(exog_sales_sum > 0, exog_sales_sum, 1.0)
    
    # Scale exog_sales down if total sales too large
    scaled_exog_sales = np.where(
        ((exog_sales_sum > max_allowed) & (max_allowed > 0))[:, None],
        exog_sales * (max_allowed / safe_exog_sum)[:, None],
        exog_sales
    ) / no_it

    # Remove exogenous sales where there is no demand
    has_demand = demand > 0
    scaled_exog_sales = np.where(has_demand[:, None], scaled_exog_sales, 0.0)

    # Regulatory override
    reg_overrides_exog = (
        ((endo_capacity + scaled_exog_sales) > regulation_cap) &
        (regulation_cap >= 0.0)
    )

    return np.where(reg_overrides_exog, 0.0, scaled_exog_sales)
