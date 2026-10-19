"""Mandate and seeding functions shared by FTT sectors."""

import numpy as np


def get_mandate_share(year, mandate_start_year, mandate_end_year, max_mandate):
    """Return the region-specific mandate share for ``year``."""
    if (year < mandate_start_year).all() or (year >= mandate_end_year).all():
        return np.zeros(mandate_start_year.shape[0])
    return (
        (year + 1 - mandate_start_year)
        / (mandate_end_year - mandate_start_year)
        * max_mandate
    )


def get_new_sales_under_mandate(sales_in, mandate_shares, green_indices):
    """Redistribute 2D sales to meet active green-technology mandates."""
    sales_after_mandate = np.copy(sales_in)
    non_green_indices = [
        index for index in range(sales_in.shape[1]) if index not in green_indices
    ]

    for region in np.flatnonzero(mandate_shares > 0):
        total_sales = np.sum(sales_in[region])
        mandate_share = mandate_shares[region]
        if total_sales == 0:
            continue

        current_green = np.sum(sales_in[region, green_indices])
        if current_green / total_sales >= mandate_share:
            continue

        target_green = total_sales * mandate_share
        if current_green > 0:
            sales_after_mandate[region, green_indices] *= (
                target_green / current_green
            )
        else:
            global_green_sales = np.sum(sales_in[:, green_indices], axis=0)
            if np.sum(global_green_sales) > 0:
                global_shares = global_green_sales / np.sum(global_green_sales)
                sales_after_mandate[region, green_indices] = (
                    target_green * global_shares
                )
            else:
                sales_after_mandate[region, green_indices] = (
                    target_green / len(green_indices)
                )

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
    seeding_years=5, seed_fraction=0.15, compare_with_old=False,
):
    """Seed green technologies using 2D capacity and sales arrays."""
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

    if compare_with_old:
        old_result = implement_seeding_old(
            cap[:, :, None],
            cum_sales_in[:, :, None],
            sales_in[:, :, None],
            year,
            green_indices,
            histend,
            seeding_years,
            seed_fraction,
        )
        _assert_mandate_results_match(result, old_result)

    return result


def implement_mandate(
    cap, cum_sales_in, sales_in, year, green_indices, mandate_info,
    compare_with_old=False,
):
    """Apply a green-technology mandate to 2D capacity and sales arrays."""
    start_years = mandate_info[:, 0]
    end_years = mandate_info[:, 1]
    max_mandate = mandate_info[:, 2]
    if (max_mandate == 0).all():
        return cum_sales_in, sales_in, cap

    mandate_shares = get_mandate_share(
        year, start_years, end_years, max_mandate
    )
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

    if compare_with_old:
        old_result = implement_mandate_old(
            cap[:, :, None],
            cum_sales_in[:, :, None],
            sales_in[:, :, None],
            year,
            green_indices,
            mandate_info,
        )
        _assert_mandate_results_match(result, old_result)

    return result


def _assert_mandate_results_match(new_result, old_result):
    """Assert equality between 2D and legacy singleton-third-axis results."""
    for new_value, old_value in zip(new_result, old_result):
        np.testing.assert_allclose(
            new_value, old_value[:, :, 0], rtol=1e-12, atol=1e-12
        )


def get_new_sales_under_mandate_old(sales_in, mandate_shares, green_indices):
    """Legacy singleton-third-axis sales redistribution."""
    sales_after_mandate = np.copy(sales_in)
    regions = np.where(mandate_shares > 0)[0]

    for region in regions:
        total_sales = np.sum(sales_in[region, :, 0])
        mandate_share = mandate_shares[region]
        if total_sales == 0:
            continue

        current_green = np.sum(sales_in[region, green_indices, 0])
        if current_green / total_sales >= mandate_share:
            continue

        target_green = total_sales * mandate_share
        if current_green > 0:
            sales_after_mandate[region, green_indices, 0] *= (
                target_green / current_green
            )
        else:
            global_green_sales = np.sum(sales_in[:, green_indices, 0], axis=0)
            if np.sum(global_green_sales) > 0:
                global_shares = global_green_sales / np.sum(global_green_sales)
                sales_after_mandate[region, green_indices, 0] = (
                    target_green * global_shares
                )
            else:
                sales_after_mandate[region, green_indices, 0] = (
                    target_green / len(green_indices)
                )

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
    """Legacy singleton-third-axis seeding implementation."""
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
    """Legacy singleton-third-axis mandate implementation."""
    start_years = mandate_info[:, 0]
    end_years = mandate_info[:, 1]
    max_mandate = mandate_info[:, 2]
    if (max_mandate == 0).all():
        return cum_sales_in, sales_in, cap

    mandate_shares = get_mandate_share(
        year, start_years, end_years, max_mandate
    )
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
