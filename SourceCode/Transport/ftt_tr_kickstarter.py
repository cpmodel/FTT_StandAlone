# -*- coding: utf-8 -*-
"""
Transport-specific kickstarter policy implementation.

This module wraps the core kickstarter functions for use in the transport
sector. Unlike freight, transport has no vehicle class interleaving,
making the implementation simpler.

The kickstarter is a SHORT-TERM policy (2024-2027) that boosts BEV car
adoption by setting fixed NEW SALES share targets:
    - 2024: 3%
    - 2025: 6%
    - 2026: 10%
    - 2027: 20%
    - After 2027: Policy OFF

Functions:
    implement_kickstarter: Apply kickstarter policy to transport sales

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


# Green technology indices for Transport EVs (BEV Econ/Mid/Lux)
# These are the direct technology indices, NOT class-interleaved like freight
GREEN_INDICES_EV = [18, 19, 20]


def implement_kickstarter(cap, EV_kickstarter, cum_sales_in, sales_in, year):
    """
    Implement kickstarter policy for transport BEV adoption.

    Applies kickstarter targets to boost BEV car market share.
    Unlike freight, transport has no vehicle class interleaving,
    so we apply kickstarter directly to the full technology array.

    Parameters
    ----------
    cap : ndarray
        Capacity by region and technology (regions x techs x 1) - TEWK
    EV_kickstarter : ndarray
        Kickstarter switch per region (regions x 1 x 1):
        - 0: kickstarter off for this region
        - 1: kickstarter on with default targets
    cum_sales_in : ndarray
        Cumulative sales (regions x techs x 1) - TEWI
    sales_in : ndarray
        Current period sales (regions x techs x 1) - tewi_t
    year : int
        Current simulation year

    Returns
    -------
    tuple
        (cum_sales_after, sales_after, cap_after)
    """
    # Check if kickstarter is active for any region
    if np.all(EV_kickstarter[:, 0, 0] == 0):
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
    regions = np.where(EV_kickstarter[:, 0, 0] != 0)[0]

    # Apply kickstarter directly (no vehicle class loop needed for transport)
    sales_after = get_new_sales_under_kickstarter(
        sales_in, target_share, GREEN_INDICES_EV, regions=regions
    )

    # Update capacity
    sales_difference = sales_after - sales_in
    cap_after = cap + sales_difference
    cap_after[:, :, 0] = np.maximum(cap_after[:, :, 0], 0)

    # Update cumulative sales
    cum_sales_after[:, :, 0] += sales_difference[:, :, 0]

    return cum_sales_after, sales_after, cap_after
