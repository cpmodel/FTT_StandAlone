# -*- coding: utf-8 -*-
"""
Centralized kickstarter functions for FTT sectors.

This module provides core kickstarter policy functionality. The kickstarter
is a time-limited EV promotion mechanism that boosts BEV market share to
specified targets during an initial adoption period (typically 2024-2028).

Unlike mandates which linearly increase to 100%, kickstarter policies set
fixed target shares for specific years to "jump-start" EV adoption.

Functions:
    get_kickstarter_target: Get target BEV share for a given year
    get_new_sales_under_kickstarter: Apply kickstarter target to sales
    implement_kickstarter: Full kickstarter implementation

@author: Amir Akther
"""

import numpy as np


# Default kickstarter configuration
# Kickstarter is a SHORT-TERM policy that completely turns off after 2027
KICKSTARTER_START_YEAR = 2024
KICKSTARTER_END_YEAR = 2027
DEFAULT_TARGETS = {
    2024: 0.03,  # 3% BEV share target (starting)
    2025: 0.06,  # 6% BEV share target
    2026: 0.10,  # 10% BEV share target
    2027: 0.20,  # 20% BEV share target (final year)
}


def get_kickstarter_target(year, targets=None, start_year=None, end_year=None):
    """
    Get the kickstarter target share for a given year.

    Unlike mandates which linearly increase, kickstarter uses fixed
    yearly targets to rapidly boost EV adoption. The kickstarter is
    a SHORT-TERM policy that completely turns off after the end year.

    Parameters
    ----------
    year : int
        Current simulation year
    targets : dict, optional
        Year-to-target mapping. If None, uses DEFAULT_TARGETS.
    start_year : int, optional
        First year of kickstarter (default 2024)
    end_year : int, optional
        Last year of kickstarter (default 2027)

    Returns
    -------
    float
        Target share for green technologies (0.0 to 1.0)
        Returns 0.0 outside the kickstarter period (policy is OFF)
    """
    if targets is None:
        targets = DEFAULT_TARGETS
    if start_year is None:
        start_year = KICKSTARTER_START_YEAR
    if end_year is None:
        end_year = KICKSTARTER_END_YEAR

    # Return target if within kickstarter period
    if start_year <= year <= end_year:
        return targets.get(year, 0.0)

    # Outside kickstarter period - policy is completely OFF
    return 0.0


def get_new_sales_under_kickstarter(sales_in, target_share, green_indices, regions=None):
    """
    Apply kickstarter target to sales distribution.

    Similar to mandate function but designed for fixed targets rather
    than linear progression. Boosts green technology sales to meet
    target and proportionally reduces non-green sales.

    Parameters
    ----------
    sales_in : ndarray
        Sales array (regions x techs x 1)
    target_share : float
        Target share for green technologies (0.0 to 1.0)
    green_indices : list
        Indices of green technologies (BEV)
    regions : list, optional
        Regions to apply kickstarter to. If None, applies to all.

    Returns
    -------
    ndarray
        Adjusted sales array
    """
    sales_after = np.copy(sales_in)

    if regions is None:
        regions = range(sales_in.shape[0])

    for r in regions:
        total_sales = np.sum(sales_in[r, :, 0])

        # Skip if no sales
        if total_sales == 0:
            continue

        current_green = np.sum(sales_in[r, green_indices, 0])
        current_share = current_green / total_sales

        # Skip if already meeting target
        if current_share >= target_share:
            continue

        target_green = total_sales * target_share

        # Scale up green sales
        if current_green > 0:
            scale_factor = target_green / current_green
            sales_after[r, green_indices, 0] *= scale_factor
        else:
            # Use global shares for distribution
            global_green_sales = np.sum(sales_in[:, green_indices, 0], axis=0)
            if np.sum(global_green_sales) > 0:
                global_shares = global_green_sales / np.sum(global_green_sales)
                sales_after[r, green_indices, 0] = target_green * global_shares
            else:
                # Equal distribution if no global data
                sales_after[r, green_indices, 0] = target_green / len(green_indices)

        # Scale down non-green sales proportionally
        non_green_indices = [i for i in range(sales_in.shape[1]) if i not in green_indices]
        remaining_sales = total_sales - target_green
        current_non_green = np.sum(sales_in[r, non_green_indices, 0])

        if current_non_green > 0:
            scale_factor = remaining_sales / current_non_green
            sales_after[r, non_green_indices, 0] *= scale_factor
        elif len(non_green_indices) > 0:
            sales_after[r, non_green_indices, 0] = remaining_sales / len(non_green_indices)

    return sales_after


def implement_kickstarter(cap, cum_sales_in, sales_in, year, green_indices,
                          kickstarter_switch, targets=None,
                          start_year=None, end_year=None):
    """
    Implement kickstarter policy for green technology promotion.

    The kickstarter policy rapidly boosts BEV adoption by setting
    fixed market share targets for specific years. Unlike mandates
    which linearly increase to 100%, kickstarter uses stepped targets
    to "jump-start" the market.

    Parameters
    ----------
    cap : ndarray
        Capacity by region and technology
    cum_sales_in : ndarray
        Cumulative sales
    sales_in : ndarray
        Current period sales
    year : int
        Current simulation year
    green_indices : list
        Indices of green technologies (BEV)
    kickstarter_switch : ndarray
        Controls kickstarter activation per region:
        - 0: kickstarter off
        - 1: kickstarter on with default targets
        - Other values can be used for custom configurations
    targets : dict, optional
        Year-to-target mapping. If None, uses DEFAULT_TARGETS.
    start_year : int, optional
        First year of kickstarter (default 2024)
    end_year : int, optional
        Last year of kickstarter (default 2028)

    Returns
    -------
    tuple
        (cum_sales_after, sales_after, cap_after)
    """
    # Check if kickstarter is active for any region
    if isinstance(kickstarter_switch, np.ndarray):
        if np.all(kickstarter_switch == 0):
            return cum_sales_in, sales_in, cap
    elif kickstarter_switch == 0:
        return cum_sales_in, sales_in, cap

    # Get target for this year
    target_share = get_kickstarter_target(year, targets, start_year, end_year)

    # If no target for this year, return unchanged
    if target_share == 0:
        return cum_sales_in, sales_in, cap

    # Identify regions with kickstarter enabled
    if isinstance(kickstarter_switch, np.ndarray):
        regions = np.where(kickstarter_switch[:, 0, 0] != 0)[0]
    else:
        regions = None  # Apply to all

    # Apply kickstarter to sales
    sales_after = get_new_sales_under_kickstarter(
        sales_in, target_share, green_indices, regions=regions
    )

    # Update capacity
    sales_difference = sales_after - sales_in
    cap_after = cap + sales_difference
    cap_after[:, :, 0] = np.maximum(cap_after[:, :, 0], 0)

    # Update cumulative sales
    cum_sales_after = np.copy(cum_sales_in)
    cum_sales_after[:, :, 0] += sales_difference[:, :, 0]

    return cum_sales_after, sales_after, cap_after
