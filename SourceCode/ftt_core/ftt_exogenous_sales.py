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
    Calculate exogenous sales, scaled down if they exceed 80% of overall sales.
    Where exogenous sales contradict regulatory constraints, regulation has priority.
    Returns changes in capacity per region and tech
    """
    avg_lifetime = np.mean(lifetimes, axis=1)
    has_demand = demand > 0

    exog_sales_sum = np.sum(exog_sales, axis=1)
    max_allowed = 0.8 * demand / np.where(avg_lifetime > 0, avg_lifetime, 1.0)

    safe_exog_sum = np.where(exog_sales_sum > 0, exog_sales_sum, 1.0)
    scaled_exog_sales = np.where(
        ((exog_sales_sum > max_allowed) & (max_allowed > 0))[:, None],
        exog_sales * (max_allowed / safe_exog_sum)[:, None],
        exog_sales
    ) / no_it

    # Zero out regions with no demand before the regulation check,
    # so ZREG/TREG = 0 (maximum regulation) is not falsely triggered there
    scaled_exog_sales = np.where(has_demand[:, None], scaled_exog_sales, 0.0)

    reg_overrides_exog = (
        ((endo_capacity + scaled_exog_sales) > regulation_cap) &
        (regulation_cap >= 0.0) )
    return np.where(reg_overrides_exog, 0.0, scaled_exog_sales)
