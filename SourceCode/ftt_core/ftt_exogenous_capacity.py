# -*- coding: utf-8 -*-
"""
=========================================
ftt_exogenous_capacity.py
=========================================
Contains the exogenous capacity and regulation correction policies for FTT modules.

Functions included:
    - exogenous_capacity
        Calculate change in capacity to reach exogenous capacity targets
    - regulation_correction
        Correct for underregulation caused by demand growth (stretching)
"""

import numpy as np


def exogenous_capacity(
        exogenous_capacity, endo_capacity, dcap_reg_corr, regulation_cap,
        t, no_it
        ):
    """
    Adjust capacity towards a specified exogenous target.
    The goal is to move linearly to the new capacity over one year.
    Regulation overrides exogenous capacity in case of a conflict.

    At time step 1 (of no_it subannual steps), you close 1/no_it of the remaining gap.
    At each subsequent step, you close a similar absolute amount as in the first step, 
    which corresponds to an increasing share of the remaining gap (e.g. 1/(no_it - t + 1)
    at step 2, and so on.
    """
    share_remaining_gap_to_close = 1 / (no_it - t + 1)
    
    # Gap between endogenous capacity (computed from the shares equation) and exogenous target
    capacity_gap = exogenous_capacity - (endo_capacity + dcap_reg_corr)
    dcap_exog_cap = capacity_gap * share_remaining_gap_to_close

    reg_overrides_exog = (exogenous_capacity > regulation_cap) & (regulation_cap >= 0)

    # Only apply where exogenous capacities turned on
    dcap_exog_cap = np.where(exogenous_capacity >= 0, dcap_exog_cap, 0)

    # Do not apply where regulations override exogenous capacity
    dcap_exog_cap = np.where(reg_overrides_exog, 0, dcap_exog_cap)

    return dcap_exog_cap


def regulation_correction(
        endo_capacity, endo_shares, cap_sum_demand_dt, reg_constr):
    """
    Demand growth raises capacities while shares stay the same (stretching).
    This extra capacity is not yet regulated. Correct for this underregulation.
    """
    # First term is with current demand, second with demand at previous t
    cap_growth_from_stretching = endo_capacity - endo_shares * cap_sum_demand_dt
    # Apply regulation to excess capacity created by demand growth (rather than shares)
    dcap_reg_corr = np.where(cap_growth_from_stretching > 0,
                   -cap_growth_from_stretching * reg_constr,
                   0)

    return dcap_reg_corr
