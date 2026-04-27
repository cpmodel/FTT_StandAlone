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
        exogenous_capacity, endo_capacity, dcap_other, regulation_cap,
        t, no_it
        ):
    """
    Calculate the change to endogenous capacity to reach exogenous capacities.
    The goal is to move linearly to the new capacity over one year.
    Regulation overrides exogenous capacity in case of a conflict.

    When no_it = 20, initially you need to close 1/20th of the remaining
    gap. In the next step, you need 1/20th of the original step, which is
    1/19th of the remaining gap etc.
    """
    share_remaining_gap_to_close = 1 / (no_it - t + 1)

    capacity_gap = exogenous_capacity - (endo_capacity + dcap_other)
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
    cap_growth_from_stretching = endo_capacity - endo_shares * cap_sum_demand_dt
    dcap_reg_corr = np.where(cap_growth_from_stretching > 0,
                   -cap_growth_from_stretching * reg_constr,
                   0)

    return dcap_reg_corr
