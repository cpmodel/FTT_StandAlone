# -*- coding: utf-8 -*-
"""
Centralized mandate functions for all FTT sectors.

This module provides core mandate functionality used by Transport, Heat, and Freight.
Sectors call these functions with their specific green technology indices.

Functions:
    get_mandate_share: Calculate year-based mandate share (linear progression)
    get_new_sales_under_mandate: Apply mandate to sales distribution
    implement_seeding: Small boost for low-adoption regions (first 5 years simulation)
    implement_mandate: Full mandate with flexible end year

@author: Amir Akther and Femke Nijsse
"""

import numpy as np


def get_mandate_share(year, mandate_start_year, mandate_end_year, max_mandate):
    """
    Calculate the mandate share based on the year.

    Linear increase from 0 to 1 between start and end years.
    Returns 0 before start year and after end year.

    Parameters
    ----------
    year : int
        Current simulation year
    mandate_start_year : int
        Year when mandate begins (e.g., 2025)
    mandate_end_year : int
        Year when mandate reaches 100% (e.g., 2036)
    max_mandate
        The mandate at the last year (usually 1, but lower for kick-start)

    Returns
    -------
    float
        Mandate share between 0.0 and 1.0
    """
    if (year < mandate_start_year).all():
        return np.full(mandate_start_year.shape[0], 0.0)
    elif (year >= mandate_end_year).all():
        return np.full(mandate_start_year.shape[0], 0.0)
    else:
        return (year + 1 - mandate_start_year) / (mandate_end_year - mandate_start_year) * max_mandate


def get_new_sales_under_mandate(sales_in, mandate_shares, green_indices):
    """
    Apply mandate to sales distribution.

    Adjusts sales so that green technologies reach the target mandate share.
    Non-green sales are reduced proportionally to maintain total sales.

    Parameters
    ----------
    sales_in : ndarray
        Sales array (regions x techs x 1)
    mandate_shares : float
        Target share for green technologies (0.0 to 1.0)
    green_indices : list
        Indices of green technologies
    regions : list, optional
        Regions to apply mandate to. If None, applies to all.

    Returns
    -------
    ndarray
        Adjusted sales array
    """
    sales_after_mandate = np.copy(sales_in)
    
    regions = np.where(mandate_shares > 0)[0] 

    for r in regions:
        total_sales = np.sum(sales_in[r, :, 0])
        mandate_share = mandate_shares[r]

        # Skip region if there are no sales
        if total_sales == 0:
            continue

        current_green = np.sum(sales_in[r, green_indices, 0])
        current_share = current_green / total_sales

        # Skip region if green share already meets mandate
        if current_share >= mandate_share:
            continue

        target_green = total_sales * mandate_share

        # Scale up green sales
        if current_green > 0:
            # Use local proportions
            scale_factor = target_green / current_green
            sales_after_mandate[r, green_indices, 0] *= scale_factor
        else:
            # Use global shares for distribution
            global_green_sales = np.sum(sales_in[:, green_indices, 0], axis=0)
            if np.sum(global_green_sales) > 0:
                global_shares = global_green_sales / np.sum(global_green_sales)
                sales_after_mandate[r, green_indices, 0] = target_green * global_shares
            else:
                # Equal distribution if no global data
                sales_after_mandate[r, green_indices, 0] = target_green / len(green_indices)

        # Scale down non-green sales to maintain total
        non_green_indices = [i for i in range(sales_in.shape[1]) if i not in green_indices]
        remaining_sales = total_sales - target_green
        current_non_green = np.sum(sales_in[r, non_green_indices, 0])

        if current_non_green > 0:
            scale_factor = remaining_sales / current_non_green
            sales_after_mandate[r, non_green_indices, 0] *= scale_factor
        elif len(non_green_indices) > 0:
            sales_after_mandate[r, non_green_indices, 0] = remaining_sales / len(non_green_indices)

    return sales_after_mandate


def implement_seeding(cap, cum_sales_in, sales_in, year, green_indices,
                      histend, seeding_years=5, seed_fraction=0.15):
    """
    Seed green technologies in low-adoption regions.

    Applies a small mandate (fraction of global green share) to bootstrap
    green technology adoption in regions with little or no adoption.
    Runs from start_year to end_year.
    
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
    histend : int
        Last historical year
    seeding_years : int, optional
        Number of years seeding applies
    green_indices : list
        Indices of green technologies for this sector
    seed_fraction : float, optional
        Fraction of global green share to use as target (default 0.15) #todo, make sense

    Returns
    -------
    tuple
        (cum_sales_after, sales_after, cap_after)
    """
    
    # Calculate global green share
    total_sales = np.sum(sales_in)
    if total_sales == 0:
        return cum_sales_in, sales_in, cap
    green_share = np.sum(sales_in[:, green_indices]) / total_sales

    start_years = np.full(sales_in.shape[0], histend)
    end_years = start_years + seeding_years
    
    seed_fraction = seed_fraction * (year - start_years + 1) / seeding_years
    
    # The max mandate is an increasing fraction of the green sales share
    max_mandate = np.full(sales_in.shape[0], seed_fraction * green_share)
    
    mandate_info = mandate_info = np.stack([start_years, end_years, max_mandate], axis=1)
    
    # Seeding works the same as a weak mandate, so reuse mandate function
    cum_sales_after_mandate, sales_after_mandate, cap = implement_mandate(
                    cap, cum_sales_in, sales_in, year, green_indices,
                                          mandate_info)
    
    return cum_sales_after_mandate, sales_after_mandate, cap



def implement_mandate(cap, cum_sales_in, sales_in, year, green_indices,
                      mandate_info):
    """
    Implement full green technology mandate.

    Linearly increases required green share from 0% to 100% over the mandate period.

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
        Indices of green technologies for this sector
    mandate_info:
        Contains start year, end year and share at end year per region

    Returns
    -------
    tuple
        (cum_sales_after, sales_after, cap_after)
    """
    
    start_years = mandate_info[:, 0]
    end_years = mandate_info[:, 1]
    max_mandate = mandate_info[:, 2]
    
    # Return if no mandates in scenario
    if (max_mandate == 0).all():
        return cum_sales_in, sales_in, cap

    # Calculate mandate share for this year
    mandate_shares = get_mandate_share(year, start_years, end_years, max_mandate)
    
    # Return if no mandates active this year
    if (mandate_shares == 0).all():
        return cum_sales_in, sales_in, cap
    
    # Compute new sales 
    if (mandate_shares > 0).any():
        sales_after_mandate = get_new_sales_under_mandate(sales_in, mandate_shares, green_indices)

        # Update capacity
        sales_difference = sales_after_mandate - sales_in
        cap = cap + sales_difference
        cap[:, :, 0] = np.maximum(cap[:, :, 0], 0)

        # Update cumulative sales
        cum_sales_after_mandate = np.copy(cum_sales_in)
        cum_sales_after_mandate[:, :, 0] += sales_difference[:, :, 0]

        return cum_sales_after_mandate, sales_after_mandate, cap

