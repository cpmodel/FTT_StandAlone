# -*- coding: utf-8 -*-
"""
Centralized emissions regulation functions for FTT sectors.

This module provides core emissions regulation policy functionality. The emissions
regulation enforces declining CO2 targets per vehicle class, redistributing sales
from high-emitting to low-emitting technologies to meet fleet-average targets.

Unlike mandates which set BEV market share targets, emissions regulation targets
fleet-average CO2 emissions (gCO2/km) that decline linearly to zero.

Functions:
    get_target_emissions: Get target emissions for a vehicle class in a given year
    get_fleet_emissions: Calculate weighted average fleet emissions
    find_advanced_counterpart: Find lower-emission variant of a technology
    ensure_minimum_bev_share: Enforce minimum BEV market share
    get_new_sales_under_emissions_regulation: Redistribute sales to meet emissions target
    implement_emissions_regulation: Full emissions regulation implementation

@author: Amir Akther
"""

import numpy as np


# Default emissions regulation configuration
REGULATION_START_YEAR = 2025
REGULATION_END_YEAR = 2040
MIN_BEV_SHARE = 0.03

# Default emissions targets by vehicle class (gCO2/km)
# These decline linearly from start values to 0 between 2025-2040
DEFAULT_START_EMISSIONS = {
    0: float('inf'),  # TWV - no limit
    1: 300,           # LCV - 300 gCO2/km
    2: 800,           # MDT - 800 gCO2/km
    3: 1200,          # HDT - 1200 gCO2/km
    4: float('inf')   # Bus - no limit
}


def get_target_emissions(year, veh_class, start_emissions=None,
                         start_year=None, end_year=None):
    """
    Calculate target emissions for a vehicle class in a given year.

    Linear reduction from start_emissions to 0 between start_year and end_year.
    Returns infinity before start_year (no limit) and 0 at/after end_year
    (full transition required).

    Parameters
    ----------
    year : int
        Current simulation year
    veh_class : int
        Vehicle class index (0=TWV, 1=LCV, 2=MDT, 3=HDT, 4=Bus)
    start_emissions : dict, optional
        Starting emissions by vehicle class in gCO2/km.
        If None, uses DEFAULT_START_EMISSIONS.
    start_year : int, optional
        Year regulation begins (default 2025)
    end_year : int, optional
        Year emissions must reach zero (default 2040)

    Returns
    -------
    float
        Target emissions in gCO2/km (infinity means no limit)
    """
    if start_emissions is None:
        start_emissions = DEFAULT_START_EMISSIONS
    if start_year is None:
        start_year = REGULATION_START_YEAR
    if end_year is None:
        end_year = REGULATION_END_YEAR

    # Before regulation starts - no limit
    if year < start_year:
        return float('inf')

    # At or after end year - full transition required
    if year >= end_year:
        return 0.0

    # Linear interpolation
    total_years = end_year - start_year
    years_in = year - start_year
    reduction_factor = 1 - (years_in / total_years)

    return start_emissions.get(veh_class, float('inf')) * reduction_factor


def get_fleet_emissions(sales, emissions_intensity):
    """
    Calculate weighted average fleet emissions.

    Parameters
    ----------
    sales : ndarray
        Sales by technology (1D array)
    emissions_intensity : ndarray
        Emissions by technology in gCO2/km (1D array)

    Returns
    -------
    float
        Weighted average emissions in gCO2/km
    """
    total_sales = np.sum(sales)

    if total_sales == 0:
        return 0.0

    total_emissions = np.sum(sales * emissions_intensity)
    return total_emissions / total_sales


def find_advanced_counterpart(tech_idx, emissions_intensity):
    """
    Find the advanced (lower-emission) version of a technology.

    Checks if the next technology index has lower but non-zero emissions,
    which would indicate it's an advanced variant of the current technology.

    Parameters
    ----------
    tech_idx : int
        Technology index to find counterpart for
    emissions_intensity : ndarray
        Emissions by technology in gCO2/km (1D array)

    Returns
    -------
    int or None
        Index of advanced counterpart, or None if not found
    """
    base_emission = emissions_intensity[tech_idx]

    # Check next technology to see if it's the advanced version
    if tech_idx + 1 < len(emissions_intensity):
        next_emission = emissions_intensity[tech_idx + 1]
        # If next technology has lower emissions and isn't zero-emission
        if 0 < next_emission < base_emission:
            return tech_idx + 1

    return None


def ensure_minimum_bev_share(sales_in, emissions_intensity, zero_emission_index=None,
                              min_share=None, regions=None):
    """
    Ensure BEV sales meet minimum share requirement.

    Scales up BEV (zero-emission) sales and proportionally reduces
    non-BEV sales to maintain total sales.

    Parameters
    ----------
    sales_in : ndarray
        Sales array. Can be 1D (techs) or 3D (regions x techs x 1)
    emissions_intensity : ndarray
        Emissions by technology in gCO2/km
    zero_emission_index : int, optional
        Index of zero-emission technology (BEV). If None, auto-detected.
    min_share : float, optional
        Minimum BEV share (default 0.03 = 3%)
    regions : list, optional
        Regions to apply to (only for 3D input). If None, applies to all.

    Returns
    -------
    ndarray
        Adjusted sales array
    """
    if min_share is None:
        min_share = MIN_BEV_SHARE

    sales = np.copy(sales_in)

    # Handle 1D input (single region/class)
    if sales.ndim == 1:
        total_sales = np.sum(sales)

        if total_sales == 0:
            return sales

        # Find BEV index (technology with zero emissions)
        if zero_emission_index is None:
            zero_indices = np.where(emissions_intensity == 0)[0]
            if len(zero_indices) == 0:
                return sales  # No zero-emission tech found
            bev_index = zero_indices[0]
        else:
            bev_index = zero_emission_index

        current_bev_share = sales[bev_index] / total_sales

        if current_bev_share < min_share:
            required_bev_sales = total_sales * min_share
            bev_sales_increase = required_bev_sales - sales[bev_index]

            non_bev_indices = [i for i in range(len(sales)) if i != bev_index]
            total_non_bev = np.sum(sales[non_bev_indices])

            if total_non_bev > 0:
                reduction_factor = (total_non_bev - bev_sales_increase) / total_non_bev
                for idx in non_bev_indices:
                    sales[idx] *= reduction_factor

            sales[bev_index] = required_bev_sales

        return sales

    # Handle 3D input (regions x techs x 1)
    if regions is None:
        regions = range(sales.shape[0])

    # Find BEV index
    if zero_emission_index is None:
        # Use first region to find zero-emission index
        if emissions_intensity.ndim == 1:
            zero_indices = np.where(emissions_intensity == 0)[0]
        else:
            zero_indices = np.where(emissions_intensity[0, :] == 0)[0]
        if len(zero_indices) == 0:
            return sales
        bev_index = zero_indices[0]
    else:
        bev_index = zero_emission_index

    for r in regions:
        total_sales = np.sum(sales[r, :, 0])

        if total_sales == 0:
            continue

        current_bev_share = sales[r, bev_index, 0] / total_sales

        if current_bev_share < min_share:
            required_bev_sales = total_sales * min_share
            bev_sales_increase = required_bev_sales - sales[r, bev_index, 0]

            non_bev_indices = [i for i in range(sales.shape[1]) if i != bev_index]
            total_non_bev = np.sum(sales[r, non_bev_indices, 0])

            if total_non_bev > 0:
                reduction_factor = (total_non_bev - bev_sales_increase) / total_non_bev
                for idx in non_bev_indices:
                    sales[r, idx, 0] *= reduction_factor

            sales[r, bev_index, 0] = required_bev_sales

    return sales


def get_new_sales_under_emissions_regulation(sales_in, emissions_intensity,
                                              target_emissions, zero_emission_index=None,
                                              regions=None):
    """
    Apply emissions regulation to sales distribution.

    Redistributes sales from high-emitting to low-emitting technologies
    to meet the target fleet-average emissions. Uses a strategy that:
    1. Shifts sales from highest-emitting technologies first
    2. Splits transfers 50/50 between advanced ICE and BEV when available
    3. Transfers 100% to BEV when no advanced counterpart exists

    Parameters
    ----------
    sales_in : ndarray
        Sales array. Can be 1D (techs) or 3D (regions x techs x 1)
    emissions_intensity : ndarray
        Emissions by technology in gCO2/km
    target_emissions : float
        Target fleet-average emissions in gCO2/km
    zero_emission_index : int, optional
        Index of zero-emission technology (BEV). If None, auto-detected.
    regions : list, optional
        Regions to apply regulation to (only for 3D input). If None, applies to all.

    Returns
    -------
    ndarray
        Adjusted sales array
    """
    sales = np.copy(sales_in)

    # Handle 1D input (single region/class)
    if sales.ndim == 1:
        total_sales = np.sum(sales)

        if total_sales == 0:
            return sales

        # Find BEV index (technology with zero emissions)
        if zero_emission_index is None:
            zero_indices = np.where(emissions_intensity == 0)[0]
            if len(zero_indices) == 0:
                return sales  # No zero-emission tech found
            bev_index = zero_indices[0]
        else:
            bev_index = zero_emission_index

        # Sort non-BEV technologies by emissions intensity (highest first)
        ice_indices = [i for i in range(len(sales)) if emissions_intensity[i] > 0]
        ice_indices.sort(key=lambda x: emissions_intensity[x], reverse=True)

        current_emissions = get_fleet_emissions(sales, emissions_intensity)

        # Iteratively redistribute until target met
        while current_emissions > target_emissions and current_emissions > 0:
            for ice_idx in ice_indices:
                if sales[ice_idx] > 0:
                    # Check for advanced counterpart
                    adv_idx = find_advanced_counterpart(ice_idx, emissions_intensity)

                    # Calculate transfer amount (limit to 25% of total sales per iteration)
                    transfer_amount = min(sales[ice_idx], total_sales * 0.25)

                    if adv_idx is not None and emissions_intensity[adv_idx] > 0:
                        # Split transfer 50/50 between advanced ICE and BEV
                        adv_amount = transfer_amount * 0.5
                        bev_amount = transfer_amount * 0.5

                        sales[ice_idx] -= transfer_amount
                        sales[adv_idx] += adv_amount
                        sales[bev_index] += bev_amount
                    else:
                        # No advanced version - shift all to BEV
                        sales[ice_idx] -= transfer_amount
                        sales[bev_index] += transfer_amount

                    # Recalculate fleet emissions
                    current_emissions = get_fleet_emissions(sales, emissions_intensity)

                    if current_emissions <= target_emissions:
                        break

        return sales

    # Handle 3D input (regions x techs x 1)
    if regions is None:
        regions = range(sales.shape[0])

    # Find BEV index
    if zero_emission_index is None:
        if emissions_intensity.ndim == 1:
            zero_indices = np.where(emissions_intensity == 0)[0]
        else:
            zero_indices = np.where(emissions_intensity[0, :] == 0)[0]
        if len(zero_indices) == 0:
            return sales
        bev_index = zero_indices[0]
    else:
        bev_index = zero_emission_index

    for r in regions:
        # Get emissions intensity for this region
        if emissions_intensity.ndim == 1:
            em_int = emissions_intensity
        else:
            em_int = emissions_intensity[r, :]

        total_sales = np.sum(sales[r, :, 0])

        if total_sales == 0:
            continue

        # Sort non-BEV technologies by emissions intensity (highest first)
        ice_indices = [i for i in range(sales.shape[1]) if em_int[i] > 0]
        ice_indices.sort(key=lambda x: em_int[x], reverse=True)

        current_emissions = get_fleet_emissions(sales[r, :, 0], em_int)

        # Iteratively redistribute until target met
        while current_emissions > target_emissions and current_emissions > 0:
            for ice_idx in ice_indices:
                if sales[r, ice_idx, 0] > 0:
                    # Check for advanced counterpart
                    adv_idx = find_advanced_counterpart(ice_idx, em_int)

                    # Calculate transfer amount
                    transfer_amount = min(sales[r, ice_idx, 0], total_sales * 0.25)

                    if adv_idx is not None and em_int[adv_idx] > 0:
                        # Split transfer 50/50
                        adv_amount = transfer_amount * 0.5
                        bev_amount = transfer_amount * 0.5

                        sales[r, ice_idx, 0] -= transfer_amount
                        sales[r, adv_idx, 0] += adv_amount
                        sales[r, bev_index, 0] += bev_amount
                    else:
                        # No advanced version - shift all to BEV
                        sales[r, ice_idx, 0] -= transfer_amount
                        sales[r, bev_index, 0] += transfer_amount

                    # Recalculate fleet emissions
                    current_emissions = get_fleet_emissions(sales[r, :, 0], em_int)

                    if current_emissions <= target_emissions:
                        break

    return sales


def implement_emissions_regulation(cap, cum_sales_in, sales_in, year,
                                    emissions_intensity, zero_emission_index,
                                    regulation_switch, veh_class=None,
                                    start_emissions=None, start_year=None,
                                    end_year=None, min_bev_share=None):
    """
    Implement emissions regulation policy.

    Full implementation combining target calculation, minimum BEV share
    enforcement, and sales redistribution to meet fleet-average emissions
    targets.

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
    emissions_intensity : ndarray
        Emissions by technology in gCO2/km
    zero_emission_index : int
        Index of zero-emission technology (BEV)
    regulation_switch : ndarray or scalar
        Controls regulation activation per region:
        - 0: regulation off
        - non-zero: regulation on
    veh_class : int, optional
        Vehicle class for target lookup (required if using class-based targets)
    start_emissions : dict, optional
        Starting emissions by vehicle class. If None, uses DEFAULT_START_EMISSIONS.
    start_year : int, optional
        Year regulation begins (default 2025)
    end_year : int, optional
        Year emissions must reach zero (default 2040)
    min_bev_share : float, optional
        Minimum BEV share (default 0.03)

    Returns
    -------
    tuple
        (cum_sales_after, sales_after, cap_after)
    """
    if start_year is None:
        start_year = REGULATION_START_YEAR
    if min_bev_share is None:
        min_bev_share = MIN_BEV_SHARE

    # Check if regulation is active for any region
    if isinstance(regulation_switch, np.ndarray):
        if np.all(regulation_switch == 0):
            return cum_sales_in, sales_in, cap
    elif regulation_switch == 0:
        return cum_sales_in, sales_in, cap

    # Get target emissions for this year and vehicle class
    if veh_class is not None:
        target = get_target_emissions(year, veh_class, start_emissions,
                                       start_year, end_year)
    else:
        # Default to infinity (no target) if no vehicle class specified
        target = float('inf')

    # Identify regions with regulation enabled
    if isinstance(regulation_switch, np.ndarray):
        regions = list(np.where(regulation_switch[:, 0, 0] != 0)[0])
    else:
        regions = None  # Apply to all

    # Step 1: Ensure minimum BEV share (if year >= start_year)
    if year >= start_year:
        sales_after = ensure_minimum_bev_share(
            sales_in, emissions_intensity, zero_emission_index,
            min_bev_share, regions
        )
    else:
        sales_after = np.copy(sales_in)

    # Step 2: Apply emissions regulation
    if target < float('inf'):
        sales_after = get_new_sales_under_emissions_regulation(
            sales_after, emissions_intensity, target,
            zero_emission_index, regions
        )

    # Update capacity
    sales_difference = sales_after - sales_in
    cap_after = cap + sales_difference
    cap_after[:, :, 0] = np.maximum(cap_after[:, :, 0], 0)

    # Update cumulative sales
    cum_sales_after = np.copy(cum_sales_in)
    cum_sales_after[:, :, 0] += sales_difference[:, :, 0]

    return cum_sales_after, sales_after, cap_after
