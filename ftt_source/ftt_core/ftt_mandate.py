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

    The share increases linearly between the start and end years and is zero
    before the mandate starts or after it ends.
    """
    if (year < mandate_start_year).all() or (year >= mandate_end_year).all():
        return np.zeros(mandate_start_year.shape[0])
    return (
        (year + 1 - mandate_start_year)
        / (mandate_end_year - mandate_start_year)
        * max_mandate
    )


def get_new_sales_under_mandate(sales_in, mandate_shares, green_indices):
    """
    Apply mandates to the sales distribution.

    Green sales are increased to reach the target share and non-green sales
    are reduced proportionally to maintain total sales.
    """
    sales_after_mandate = np.copy(sales_in)
    non_green_indices = [
        index for index in range(sales_in.shape[1]) if index not in green_indices
    ]

    regions = np.flatnonzero(mandate_shares > 0)
    for region in regions:
        total_sales = np.sum(sales_in[region])
        mandate_share = mandate_shares[region]
        # Skip region if there are no sales.
        if total_sales == 0:
            continue

        current_green = np.sum(sales_in[region, green_indices])
        # Skip region if green share already meets mandate.
        if current_green / total_sales >= mandate_share:
            continue

        target_green = total_sales * mandate_share
        if current_green > 0:
            # Scale up green sales using local proportions.
            sales_after_mandate[region, green_indices] *= (
                target_green / current_green
            )
        else:
            # Use global shares when the region has no green sales.
            global_green_sales = np.sum(sales_in[:, green_indices], axis=0)
            if np.sum(global_green_sales) > 0:
                global_shares = global_green_sales / np.sum(global_green_sales)
                sales_after_mandate[region, green_indices] = (
                    target_green * global_shares
                )
            else:
                # Equal distribution if there is no global data.
                sales_after_mandate[region, green_indices] = (
                    target_green / len(green_indices)
                )

        # Scale down non-green sales to maintain total sales.
        remaining_sales = total_sales - target_green
        current_non_green = np.sum(sales_in[region, non_green_indices])
        if current_non_green > 0:
            sales_after_mandate[region, non_green_indices] *= (
                remaining_sales / current_non_green
            )
        elif non_green_indices:
            sales_after_mandate[region, non_green_indices] = (
                remaining_sales / len(non_green_indices)
            )

    return sales_after_mandate


def implement_seeding(
    cap, cum_sales_in, sales_in, year, green_indices, histend,
    seeding_years=5, seed_fraction=0.15,
):
    """
    Seed green technologies in low-adoption regions.

    Applies a small mandate based on the global green sales share to bootstrap
    green technology adoption during the first years of the simulation.
    """
    # Calculate global green share.
    total_sales = np.sum(sales_in)
    if total_sales == 0:
        return cum_sales_in, sales_in, cap

    green_share = np.sum(sales_in[:, green_indices]) / total_sales
    start_years = np.full(sales_in.shape[0], histend)
    end_years = start_years + seeding_years
    max_mandate = (
        seed_fraction
        * (year - start_years + 1)
        / seeding_years
        * green_share
    )
    mandate_info = np.stack([start_years, end_years, max_mandate], axis=1)
    result = implement_mandate(
        cap, cum_sales_in, sales_in, year, green_indices, mandate_info
    )
    return result


def implement_mandate(
    cap, cum_sales_in, sales_in, year, green_indices, mandate_info,
):
    """
    Implement a full green-technology mandate.

    The required green share increases linearly over the mandate period.
    """
    start_years = mandate_info[:, 0]
    end_years = mandate_info[:, 1]
    max_mandate = mandate_info[:, 2]
    # Return if no mandates are present in the scenario.
    if (max_mandate == 0).all():
        return cum_sales_in, sales_in, cap

    mandate_shares = get_mandate_share(
        year, start_years, end_years, max_mandate
    )
    # Return if no mandates are active this year.
    if (mandate_shares == 0).all():
        return cum_sales_in, sales_in, cap

    sales_after_mandate = get_new_sales_under_mandate(
        sales_in, mandate_shares, green_indices
    )
    sales_difference = sales_after_mandate - sales_in
    cap_after_mandate = np.maximum(cap + sales_difference, 0)
    cum_sales_after_mandate = np.copy(cum_sales_in)
    cum_sales_after_mandate += sales_difference
    result = cum_sales_after_mandate, sales_after_mandate, cap_after_mandate
    return result


def get_new_sales_under_mandate_old(sales_in, mandate_shares, green_indices):
    """Apply mandates to the legacy singleton-third-axis sales distribution."""
    sales_after_mandate = np.copy(sales_in)
    regions = np.where(mandate_shares > 0)[0]

    for region in regions:
        total_sales = np.sum(sales_in[region, :, 0])
        mandate_share = mandate_shares[region]
        # Skip region if there are no sales.
        if total_sales == 0:
            continue

        current_green = np.sum(sales_in[region, green_indices, 0])
        # Skip region if green share already meets mandate.
        if current_green / total_sales >= mandate_share:
            continue

        target_green = total_sales * mandate_share
        if current_green > 0:
            # Scale up green sales using local proportions.
            sales_after_mandate[region, green_indices, 0] *= (
                target_green / current_green
            )
        else:
            # Use global shares when the region has no green sales.
            global_green_sales = np.sum(sales_in[:, green_indices, 0], axis=0)
            if np.sum(global_green_sales) > 0:
                global_shares = global_green_sales / np.sum(global_green_sales)
                sales_after_mandate[region, green_indices, 0] = (
                    target_green * global_shares
                )
            else:
                # Equal distribution if there is no global data.
                sales_after_mandate[region, green_indices, 0] = (
                    target_green / len(green_indices)
                )

        # Scale down non-green sales to maintain total sales.
        non_green_indices = [
            index for index in range(sales_in.shape[1]) if index not in green_indices
        ]
        remaining_sales = total_sales - target_green
        current_non_green = np.sum(sales_in[region, non_green_indices, 0])
        if current_non_green > 0:
            sales_after_mandate[region, non_green_indices, 0] *= (
                remaining_sales / current_non_green
            )
        elif non_green_indices:
            sales_after_mandate[region, non_green_indices, 0] = (
                remaining_sales / len(non_green_indices)
            )

    return sales_after_mandate


def implement_seeding_old(
    cap, cum_sales_in, sales_in, year, green_indices, histend,
    seeding_years=5, seed_fraction=0.15,
):
    """
    Seed green technologies in low-adoption regions using the legacy shapes.

    Seeding works the same as a weak mandate, so reuse the mandate function.
    """
    total_sales = np.sum(sales_in)
    if total_sales == 0:
        return cum_sales_in, sales_in, cap

    green_share = np.sum(sales_in[:, green_indices]) / total_sales
    start_years = np.full(sales_in.shape[0], histend)
    end_years = start_years + seeding_years
    max_mandate = (
        seed_fraction
        * (year - start_years + 1)
        / seeding_years
        * green_share
    )
    mandate_info = np.stack([start_years, end_years, max_mandate], axis=1)
    return implement_mandate_old(
        cap, cum_sales_in, sales_in, year, green_indices, mandate_info
    )


def implement_mandate_old(
    cap, cum_sales_in, sales_in, year, green_indices, mandate_info,
):
    """Implement a full green-technology mandate using legacy array shapes."""
    start_years = mandate_info[:, 0]
    end_years = mandate_info[:, 1]
    max_mandate = mandate_info[:, 2]
    # Return if no mandates are present in the scenario.
    if (max_mandate == 0).all():
        return cum_sales_in, sales_in, cap

    mandate_shares = get_mandate_share(
        year, start_years, end_years, max_mandate
    )
    # Return if no mandates are active this year.
    if (mandate_shares == 0).all():
        return cum_sales_in, sales_in, cap

    sales_after_mandate = get_new_sales_under_mandate_old(
        sales_in, mandate_shares, green_indices
    )
    sales_difference = sales_after_mandate - sales_in
    cap_after_mandate = np.copy(cap)
    cap_after_mandate += sales_difference
    cap_after_mandate[:, :, 0] = np.maximum(cap_after_mandate[:, :, 0], 0)
    cum_sales_after_mandate = np.copy(cum_sales_in)
    cum_sales_after_mandate[:, :, 0] += sales_difference[:, :, 0]
    return cum_sales_after_mandate, sales_after_mandate, cap_after_mandate
