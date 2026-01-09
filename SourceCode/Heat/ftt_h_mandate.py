# -*- coding: utf-8 -*-
"""
Heat-specific mandate functions.

This module contains seeding and mandate logic for heat pumps,
following the same pattern as ftt_fr_mandate.py for Freight.
Uses core functions from ftt_core.ftt_mandate.
"""

import numpy as np

from SourceCode.ftt_core.ftt_mandate import get_new_sales_under_mandate, get_mandate_share


# Heat pump technology indices (ground source, air-water, air-air)
GREEN_INDICES = [9, 10, 11]
MANDATE_START_YEAR = 2025
N_YEARS = 11  # Default mandate duration


def implement_seeding(cap, seeding, cum_sales_in, sales_in, year):
    """
    For regions without heat pump sales, introduce heat pumps at 3% to 15%
    of global sales share. Runs from 2025-2030.

    Parameters
    ----------
    cap : ndarray
        Capacity by region and technology (HEWK)
    seeding : int or ndarray
        Switch to enable/disable seeding (0 = off, nonzero = on)
    cum_sales_in : ndarray
        Cumulative sales (HEWI)
    sales_in : ndarray
        Current period sales (hewi_t)
    year : int
        Current year

    Returns
    -------
    tuple
        (cum_sales_after, sales_after, cap_after)
    """
    mandate_end_year = 2030

    # Handle both scalar and array seeding switch
    if isinstance(seeding, np.ndarray):
        seed_switch = seeding[0, 0, 0] if seeding.size > 0 else 0
    else:
        seed_switch = seeding

    # If seeding is off or outside seeding period, return inputs
    if seed_switch == 0 or year not in range(MANDATE_START_YEAR, mandate_end_year + 1):
        return cum_sales_in, sales_in, cap

    # Calculate global heat pump share
    total_sales = np.sum(sales_in)
    if total_sales == 0:
        return cum_sales_in, sales_in, cap

    green_share = np.sum(sales_in[:, GREEN_INDICES]) / total_sales

    # Seeding is 15% of global green share, ramping up from 2025-2030
    mandate_share = get_mandate_share(year, MANDATE_START_YEAR, mandate_end_year) * 0.15 * green_share

    if mandate_share > 0:
        sales_after_mandate = get_new_sales_under_mandate(
            sales_in, mandate_share, GREEN_INDICES
        )

        # Update capacity
        sales_difference = sales_after_mandate - sales_in
        cap = cap + sales_difference
        cap[:, :, 0] = np.maximum(cap[:, :, 0], 0)

        # Update cumulative sales
        cum_sales_after_mandate = np.copy(cum_sales_in)
        cum_sales_after_mandate[:, :, 0] += sales_difference[:, :, 0]

        return cum_sales_after_mandate, sales_after_mandate, cap

    return cum_sales_in, sales_in, cap


def implement_mandate(cap, hp_mandate, cum_sales_in, sales_in, year):
    """
    Implement heat pump mandate: linearly increasing required share of
    heat pump sales from 2025 to mandate end year.

    Parameters
    ----------
    cap : ndarray
        Capacity by region and technology (HEWK)
    hp_mandate : ndarray
        Mandate switch/end year. Values:
        - 0: mandate off
        - 2025-2060: mandate end year
    cum_sales_in : ndarray
        Cumulative sales (HEWI)
    sales_in : ndarray
        Current period sales (hewi_t)
    year : int
        Current year

    Returns
    -------
    tuple
        (cum_sales_after, sales_after, cap_after)
    """
    # Handle both scalar and array mandate switch
    if isinstance(hp_mandate, np.ndarray):
        mandate_switch = hp_mandate[0, 0, 0] if hp_mandate.size > 0 else 0
    else:
        mandate_switch = hp_mandate

    # If mandate is off, return inputs
    if mandate_switch == 0:
        return cum_sales_in, sales_in, cap

    # Determine mandate end year
    mandate_end_year = MANDATE_START_YEAR + N_YEARS  # Default: 2036

    if mandate_switch in range(2025, 2060):
        # Custom end year specified
        mandate_end_year = int(mandate_switch)

    # Calculate mandate share for this year
    mandate_share = get_mandate_share(year, MANDATE_START_YEAR, mandate_end_year)

    if mandate_share > 0:
        sales_after_mandate = get_new_sales_under_mandate(
            sales_in, mandate_share, GREEN_INDICES
        )

        # Update capacity
        sales_difference = sales_after_mandate - sales_in
        cap = cap + sales_difference
        cap[:, :, 0] = np.maximum(cap[:, :, 0], 0)

        # Update cumulative sales
        cum_sales_after_mandate = np.copy(cum_sales_in)
        cum_sales_after_mandate[:, :, 0] += sales_difference[:, :, 0]

        return cum_sales_after_mandate, sales_after_mandate, cap

    return cum_sales_in, sales_in, cap
