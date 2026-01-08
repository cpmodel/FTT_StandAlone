# -*- coding: utf-8 -*-
"""
Freight-specific kickstarter policy implementation.

This module wraps the core kickstarter functions for use in the freight
sector, handling the multi-vehicle-class structure specific to freight.

The kickstarter is a SHORT-TERM policy (2024-2027) that boosts BEV truck
adoption by setting fixed NEW SALES share targets:
    - 2024: 3%
    - 2025: 6%
    - 2026: 10%
    - 2027: 20%
    - After 2027: Policy OFF

Functions:
    implement_kickstarter: Apply kickstarter policy to freight sales

@author: Amir Akther
"""

import numpy as np

from SourceCode.ftt_core.ftt_kickstarter import (
    get_kickstarter_target,
    get_new_sales_under_kickstarter,
    KICKSTARTER_START_YEAR,
    KICKSTARTER_END_YEAR,
    DEFAULT_TARGETS,
)


# BEV index within each vehicle class (index 6 = BEV in freight tech list)
BEV_INDEX_IN_CLASS = 6


def implement_kickstarter(cap, EV_truck_kickstarter, cum_sales_in, sales_in,
                          n_veh_classes, year):
    """
    Implement kickstarter policy for freight BEV adoption.

    Applies kickstarter targets to boost BEV truck market share across
    all vehicle classes (TWV, LCV, MDT, HDT, Bus). Each class is treated
    independently.

    Parameters
    ----------
    cap : ndarray
        Capacity by region and technology (regions x techs x 1)
    EV_truck_kickstarter : ndarray
        Kickstarter switch per region (regions x 1 x 1):
        - 0: kickstarter off for this region
        - 1: kickstarter on with default targets
    cum_sales_in : ndarray
        Cumulative sales (regions x techs x 1)
    sales_in : ndarray
        Current period sales (regions x techs x 1)
    n_veh_classes : int
        Number of vehicle classes (typically 5)
    year : int
        Current simulation year

    Returns
    -------
    tuple
        (cum_sales_after, sales_after, cap_after)
    """
    # Check if kickstarter is active for any region
    if np.all(EV_truck_kickstarter[:, 0, 0] == 0):
        return cum_sales_in, sales_in, cap

    # Get target share for this year
    target_share = get_kickstarter_target(year)

    # If no target for this year, return unchanged
    if target_share == 0:
        return cum_sales_in, sales_in, cap

    # Initialize output arrays
    cum_sales_after = np.copy(cum_sales_in)
    sales_after = np.copy(sales_in)

    # Identify regions with kickstarter enabled
    regions = np.where(EV_truck_kickstarter[:, 0, 0] != 0)[0]

    # Apply kickstarter to each vehicle class independently
    for veh_class in range(n_veh_classes):
        # Extract sales for this vehicle class
        # Technologies are interleaved: class0_tech0, class1_tech0, ..., class0_tech1, ...
        sales_in_class = sales_in[:, veh_class::n_veh_classes, :]

        # BEV index within this class slice
        green_indices_class = [BEV_INDEX_IN_CLASS]

        # Apply kickstarter to this class
        sales_after_class = get_new_sales_under_kickstarter(
            sales_in_class, target_share, green_indices_class, regions=regions
        )

        # Store results back into full array
        sales_after[:, veh_class::n_veh_classes] = sales_after_class

        # Update capacity for this class
        sales_difference = sales_after_class - sales_in_class
        cap[:, veh_class::n_veh_classes, :] += sales_difference
        cap[:, veh_class::n_veh_classes, 0] = np.maximum(
            cap[:, veh_class::n_veh_classes, 0], 0
        )

        # Update cumulative sales for this class
        cum_sales_after[:, veh_class::n_veh_classes, 0] += sales_difference[:, :, 0]

    return cum_sales_after, sales_after, cap
