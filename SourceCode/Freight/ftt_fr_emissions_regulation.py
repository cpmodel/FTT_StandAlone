# -*- coding: utf-8 -*-
"""
Freight-specific emissions regulation policy implementation.

This module wraps the core emissions regulation functions for use in the freight
sector, handling the multi-vehicle-class structure specific to freight.

The emissions regulation enforces declining CO2 targets per vehicle class:
    - LCV (class 1): 300 gCO2/km -> 0 by 2040
    - MDT (class 2): 800 gCO2/km -> 0 by 2040
    - HDT (class 3): 1200 gCO2/km -> 0 by 2040
    - TWV (class 0) and Bus (class 4): No limit

Functions:
    implement_emissions_regulation: Apply emissions regulation to freight sales

@author: Amir Akther
"""

import numpy as np

from SourceCode.ftt_core.ftt_emissions_regulation import (
    get_target_emissions,
    get_new_sales_under_emissions_regulation,
    REGULATION_START_YEAR,
    REGULATION_END_YEAR,
    DEFAULT_START_EMISSIONS,
)


# BEV index within each vehicle class (index 6 = BEV in freight tech list)
BEV_INDEX_IN_CLASS = 6


def implement_emissions_regulation(cap, emissions_regulation_switch, cum_sales_in,
                                    sales_in, n_veh_classes, year, emissions_intensity):
    """
    Implement emissions regulation policy for freight sector.

    Applies declining CO2 emission targets to each vehicle class independently,
    redistributing sales from high-emitting to low-emitting technologies
    to meet fleet-average targets.

    Parameters
    ----------
    cap : ndarray
        Capacity by region and technology (regions x techs x 1) - ZEWK
    emissions_regulation_switch : ndarray
        Regulation switch per region (regions x 1 x 1):
        - 0: regulation off for this region
        - non-zero: regulation on
    cum_sales_in : ndarray
        Cumulative sales (regions x techs x 1) - ZEWI
    sales_in : ndarray
        Current period sales (regions x techs x 1) - zewi_t
    n_veh_classes : int
        Number of vehicle classes (typically 5: TWV, LCV, MDT, HDT, Bus)
    year : int
        Current simulation year
    emissions_intensity : ndarray
        CO2 emissions by technology (regions x techs) in gCO2/km
        From data['BZTC'][:, :, c6ti['12 CO2 emissions (gCO2/km)']]

    Returns
    -------
    tuple
        (cum_sales_after, sales_after, cap_after)

    Notes
    -----
    Technologies are interleaved by vehicle class in freight:
    - Index 0, 5, 10, ... = class 0 (TWV)
    - Index 1, 6, 11, ... = class 1 (LCV)
    - etc.

    Each vehicle class has its own emissions target that declines
    linearly from 2025 to 2040:
    - LCV: 300 -> 0 gCO2/km
    - MDT: 800 -> 0 gCO2/km
    - HDT: 1200 -> 0 gCO2/km
    - TWV/Bus: No limit (infinity)
    """
    # Check if regulation is active for any region
    if np.all(emissions_regulation_switch[:, 0, 0] == 0):
        return cum_sales_in, sales_in, cap

    # Initialize output arrays
    cum_sales_after = np.copy(cum_sales_in)
    sales_after = np.copy(sales_in)

    # Identify regions with regulation enabled
    regions = list(np.where(emissions_regulation_switch[:, 0, 0] != 0)[0])

    # Apply regulation to each vehicle class independently
    for veh_class in range(n_veh_classes):
        # Extract sales for this vehicle class
        # Technologies are interleaved: class0_tech0, class1_tech0, ..., class0_tech1, ...
        sales_in_class = sales_in[:, veh_class::n_veh_classes, :]
        emissions_for_class = emissions_intensity[:, veh_class::n_veh_classes]

        # Get target emissions for this vehicle class and year
        target = get_target_emissions(year, veh_class)

        # Skip if no target (infinite means no regulation for this class)
        if target == float('inf'):
            continue

        # Process each region with active regulation
        for r in regions:
            # Apply emissions target redistribution
            new_sales = get_new_sales_under_emissions_regulation(
                sales_in_class[r, :, 0],
                emissions_for_class[r, :],
                target,
                zero_emission_index=BEV_INDEX_IN_CLASS
            )

            # Store results back
            sales_after[r, veh_class::n_veh_classes, 0] = new_sales

        # Update capacity for this class
        sales_difference = (
            sales_after[:, veh_class::n_veh_classes, :] -
            sales_in[:, veh_class::n_veh_classes, :]
        )
        cap[:, veh_class::n_veh_classes, :] += sales_difference
        cap[:, veh_class::n_veh_classes, 0] = np.maximum(
            cap[:, veh_class::n_veh_classes, 0], 0
        )

        # Update cumulative sales for this class
        cum_sales_after[:, veh_class::n_veh_classes, 0] += sales_difference[:, :, 0]

    return cum_sales_after, sales_after, cap
