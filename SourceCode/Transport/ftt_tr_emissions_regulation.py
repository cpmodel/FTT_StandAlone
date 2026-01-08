# -*- coding: utf-8 -*-
"""
Transport-specific emissions regulation policy implementation.

This module implements emissions regulation for passenger transport with:
- Segment-specific targets (Economy, Mid, Luxury)
- Starting targets based on current fleet-average emissions at 2025
- Linear decline to 0 by 2040
- Proportional redistribution to eligible lower-emission technologies
- Excludes CNG and Hydrogen from receiving redistributed sales

Functions:
    get_fleet_emissions: Calculate weighted average fleet emissions
    get_proportional_redistribution: Redistribute sales proportionally
    implement_emissions_regulation: Main policy implementation

"""

import numpy as np


# Regulation timeline
REGULATION_START_YEAR = 2025
REGULATION_END_YEAR = 2040

# Module-level storage for baseline emissions (calculated once at start year)
# Using module-level to avoid polluting the data dictionary
_baseline_emissions_cache = {}

# Technology indices within full VTTI array (31 technologies)
# Segment definitions - indices in the flat VTTI array
SEGMENTS = {
    'econ': {
        'indices': [0, 3, 6, 9, 12, 15, 18, 21, 24],  # All econ technologies
        'eligible': [3, 9, 15, 18, 21],  # Adv Petrol, Adv Diesel, Hybrid, Electric, PHEV
        'excluded': [12, 24],  # CNG, Hydrogen
    },
    'mid': {
        'indices': [1, 4, 7, 10, 13, 16, 19, 22, 25],
        'eligible': [4, 10, 16, 19, 22],
        'excluded': [13, 25],
    },
    'lux': {
        'indices': [2, 5, 8, 11, 14, 17, 20, 23, 26],
        'eligible': [5, 11, 17, 20, 23],
        'excluded': [14, 26],
    },
}

# Technology names for reference (indices 0-30):
# 0: Petrol Econ, 1: Petrol Mid, 2: Petrol Lux
# 3: Adv Petrol Econ, 4: Adv Petrol Mid, 5: Adv Petrol Lux
# 6: Diesel Econ, 7: Diesel Mid, 8: Diesel Lux
# 9: Adv Diesel Econ, 10: Adv Diesel Mid, 11: Adv Diesel Lux
# 12: CNG Econ, 13: CNG Mid, 14: CNG Lux
# 15: Hybrid Econ, 16: Hybrid Mid, 17: Hybrid Lux
# 18: Electric Econ, 19: Electric Mid, 20: Electric Lux
# 21: PHEV Econ, 22: PHEV Mid, 23: PHEV Lux
# 24: Hydrogen Econ, 25: Hydrogen Mid, 26: Hydrogen Lux
# 27-30: Bikes


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


def get_proportional_redistribution(sales, emissions_intensity, target_emissions,
                                    eligible_indices):
    """
    Redistribute sales proportionally to eligible lower-emission technologies.

    Reduces sales from high-emitting technologies and distributes them
    proportionally among eligible lower-emission technologies.
    CNG and Hydrogen are excluded from receiving sales.

    Parameters
    ----------
    sales : ndarray
        Sales by technology (1D array for segment)
    emissions_intensity : ndarray
        Emissions by technology in gCO2/km (1D array for segment)
    target_emissions : float
        Target fleet-average emissions in gCO2/km
    eligible_indices : list
        Indices of technologies eligible to receive redistributed sales
        (relative to segment indices)

    Returns
    -------
    ndarray
        Adjusted sales array
    """
    sales_out = np.copy(sales)
    current_avg = get_fleet_emissions(sales, emissions_intensity)

    if current_avg <= target_emissions:
        return sales_out  # Already meeting target

    total_sales = np.sum(sales)
    if total_sales == 0:
        return sales_out

    # Identify high emitters (above target) and low emitters (at or below target)
    high_emitter_mask = emissions_intensity > target_emissions
    low_emitter_mask = emissions_intensity <= target_emissions

    # Filter eligible technologies (exclude CNG/Hydrogen)
    eligible_mask = np.zeros(len(sales), dtype=bool)
    eligible_mask[eligible_indices] = True
    eligible_low_mask = low_emitter_mask & eligible_mask

    # If no eligible low-emission technologies, return unchanged
    if not np.any(eligible_low_mask):
        return sales_out

    # Calculate how much sales need to shift
    # We need to reduce fleet average from current_avg to target_emissions
    # Formula: (high_sales * high_avg + low_sales * low_avg) / total = target
    # We shift sales from high to low until target is met

    # Iterative approach: shift sales gradually
    max_iterations = 100
    for _ in range(max_iterations):
        current_avg = get_fleet_emissions(sales_out, emissions_intensity)

        if current_avg <= target_emissions:
            break

        # Calculate excess emissions
        excess_rate = (current_avg - target_emissions) / current_avg
        shift_fraction = min(0.1, excess_rate)  # Shift up to 10% per iteration

        # Amount to shift from high emitters
        high_sales = np.sum(sales_out[high_emitter_mask])
        if high_sales == 0:
            break

        shift_amount = high_sales * shift_fraction

        # Reduce high emitters proportionally
        for i in np.where(high_emitter_mask)[0]:
            if sales_out[i] > 0:
                reduction = (sales_out[i] / high_sales) * shift_amount
                sales_out[i] -= reduction

        # Distribute to eligible low emitters proportionally
        eligible_low_sales = np.sum(sales_out[eligible_low_mask])
        if eligible_low_sales > 0:
            # Distribute based on current shares of eligible low emitters
            for i in np.where(eligible_low_mask)[0]:
                share = sales_out[i] / eligible_low_sales
                sales_out[i] += shift_amount * share
        else:
            # If no existing sales in eligible low emitters, distribute equally
            eligible_count = np.sum(eligible_low_mask)
            for i in np.where(eligible_low_mask)[0]:
                sales_out[i] += shift_amount / eligible_count

    return sales_out


def implement_emissions_regulation(cap, emissions_regulation_switch, cum_sales_in,
                                   sales_in, year, emissions_intensity):
    """
    Implement emissions regulation policy for transport sector.

    Applies declining CO2 emission targets to each vehicle segment independently,
    redistributing sales from high-emitting to eligible low-emitting technologies.

    Parameters
    ----------
    cap : ndarray
        Capacity by region and technology (regions x techs x 1) - TEWK
    emissions_regulation_switch : ndarray
        Regulation switch per region (regions x 1 x 1):
        - 0: regulation off for this region
        - non-zero: regulation on
    cum_sales_in : ndarray
        Cumulative sales (regions x techs x 1) - TEWI
    sales_in : ndarray
        Current period sales (regions x techs x 1) - tewi_t
    year : int
        Current simulation year
    emissions_intensity : ndarray
        CO2 emissions by technology (regions x techs) in gCO2/km
        From data['BTTC'][:, :, c3ti['14 CO2Emissions']]

    Returns
    -------
    tuple
        (cum_sales_after, sales_after, cap_after)

    Notes
    -----
    - Baseline emissions are calculated once at REGULATION_START_YEAR (2025)
      and cached in module-level variable for use in subsequent years
    - Sales only redistribute to: Adv Petrol, Adv Diesel, Hybrid, PHEV, Electric
    - Sales do NOT go to: CNG, Hydrogen (excluded)
    - Each segment (Econ/Mid/Lux) has independent targets
    """
    global _baseline_emissions_cache

    # Check if regulation is active for any region
    if np.all(emissions_regulation_switch[:, 0, 0] == 0):
        return cum_sales_in, sales_in, cap

    # Before regulation starts, return unchanged
    if year < REGULATION_START_YEAR:
        return cum_sales_in, sales_in, cap

    # Initialize output arrays
    cum_sales_after = np.copy(cum_sales_in)
    sales_after = np.copy(sales_in)

    # Identify regions with regulation enabled
    regions = list(np.where(emissions_regulation_switch[:, 0, 0] != 0)[0])

    # Calculate baseline emissions at start year (only once)
    if not _baseline_emissions_cache or year == REGULATION_START_YEAR:
        _baseline_emissions_cache = {}
        for segment_name, segment_config in SEGMENTS.items():
            indices = segment_config['indices']
            # Calculate fleet-average emissions for this segment across all regions
            segment_baselines = []
            for r in regions:
                segment_sales = sales_in[r, indices, 0]
                segment_em = emissions_intensity[r, indices]
                fleet_em = get_fleet_emissions(segment_sales, segment_em)
                segment_baselines.append(fleet_em)

            # Use average across regions as baseline (or could be region-specific)
            if segment_baselines:
                _baseline_emissions_cache[segment_name] = np.mean(segment_baselines)
            else:
                _baseline_emissions_cache[segment_name] = 100.0  # Default fallback

    # Calculate declining target based on fixed baseline
    years_elapsed = year - REGULATION_START_YEAR
    total_years = REGULATION_END_YEAR - REGULATION_START_YEAR
    reduction_factor = min(1.0, years_elapsed / total_years)

    # Apply regulation to each segment independently
    for segment_name, segment_config in SEGMENTS.items():
        indices = segment_config['indices']
        # Convert eligible indices from full array to segment-local indices
        eligible_full = segment_config['eligible']
        eligible_local = [indices.index(e) for e in eligible_full if e in indices]

        # Get target for this segment: baseline -> 0 by 2040
        baseline = _baseline_emissions_cache.get(segment_name, 100.0)
        target = baseline * (1 - reduction_factor)

        # Process each region with active regulation
        for r in regions:
            # Extract segment sales and emissions
            segment_sales = sales_in[r, indices, 0]
            segment_emissions = emissions_intensity[r, indices]

            # Apply proportional redistribution
            new_segment_sales = get_proportional_redistribution(
                segment_sales,
                segment_emissions,
                target,
                eligible_local
            )

            # Store results back
            sales_after[r, indices, 0] = new_segment_sales

    # Update capacity based on sales changes
    sales_difference = sales_after - sales_in
    cap_after = cap + sales_difference
    cap_after[:, :, 0] = np.maximum(cap_after[:, :, 0], 0)

    # Update cumulative sales
    cum_sales_after[:, :, 0] += sales_difference[:, :, 0]

    return cum_sales_after, sales_after, cap_after
