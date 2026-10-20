"""Sales and investment calculations for all FTT models."""

import numpy as np


def get_sales(
    cap, cap_dt, cap_lag, sales_or_investment_in, timescales, dt,
):
    """
    Calculate new sales/investments for all FTT models (e.g. TEWI).

    1. If capacity has grown, the difference between the new capacity and the
       old.
    2. The amount of existing capacity that is depreciated (retired) as it
       reaches its end of life.

    Capacity depreciation is based on the technology lifetime in the
    cost matrix. 1/lifetime is depreciated each year.
    """
    # Find capacity growth since last time step
    cap_growth = cap - cap_lag
    cap_growth_dt = cap - cap_dt
    # Calculate capacity depreciation.
    cap_depreciation = cap_lag * dt / timescales

    # If there is growth, depreciation is fully replaced.
    # Where capacity has decreased, we only replace depreciated capacity
    # if the depreciation > capacity loss
    eol_replacements = np.select(
        [
            cap_growth >= 0.0,
            (cap_depreciation > -cap_growth_dt) & (cap_growth < 0),
        ],
        [cap_depreciation, cap_depreciation + cap_growth_dt],
        default=0,
    )
    eol_replacements = np.maximum(eol_replacements, 0)

    # Three end-of-life (eol) replacement options, depending on conditions
    # Ensure no negative values
    # Find total additions at time t
    # If capacity has grown, additions equal the difference +
    # eol replacements from the previous year.
    # Otherwise, additions just equal replacements.
    sales_dt = np.where(
        cap_growth_dt > 0.0,
        cap_growth_dt + eol_replacements,
        eol_replacements,
    )
    sales_or_investment = sales_or_investment_in + sales_dt

    return sales_or_investment, sales_dt


def get_sales_yearly(
    cap, cap_lag, sales_or_investment_in, timescales,
):
    """
    Calculate new sales/investments for all FTT models before simulation starts
    (e.g. TEWI).

    1. If capacity has grown, the difference between the new capacity and the
       old.
    2. The amount of existing capacity that is depreciated (retired) as it
       reaches its end of life.

    Capacity depreciation is based on the technology lifetime in the
    cost matrix. 1/lifetime is depreciated each year.
    """
    # Find capacity growth since last year
    cap_growth = cap - cap_lag
    # Calculate capacity depreciation.
    cap_depreciation = cap_lag / timescales

    # If there is growth, end-of-life replacements take place fully
    # Where capacity has decreased, we only replace eol capacity
    # if the eol_replacement > capacity loss
    eol_replacements = np.select(
        [
            cap_growth >= 0.0,
            (cap_depreciation > -cap_growth) & (cap_growth < 0),
        ],
        [cap_depreciation, cap_depreciation + cap_growth],
        default=0,
    )
    eol_replacements = np.maximum(eol_replacements, 0)

    # Three end-of-life (eol) replacement options, depending on conditions
    # Ensure no negative values
    # If capacity has grown, additions equal the difference +
    # eol replacement from the previous year.
    # Otherwise, additions just equal eol replacement.
    sales_or_investment = np.where(
        cap_growth > 0.0,
        cap_growth + eol_replacements,
        eol_replacements,
    )

    return sales_or_investment


def get_sales_old(cap, cap_dt, cap_lag, sales_or_investment_in, timescales, dt):
    """Legacy singleton-third-axis implementation of :func:`get_sales`."""
    cap_growth = cap[:, :, 0] - cap_lag[:, :, 0]
    cap_growth_dt = cap[:, :, 0] - cap_dt[:, :, 0]
    cap_depreciation = cap_lag[:, :, 0] * dt / timescales

    eol_replacements = np.select(
        [
            cap_growth >= 0.0,
            (cap_depreciation > -cap_growth_dt) & (cap_growth < 0),
        ],
        [cap_depreciation, cap_depreciation + cap_growth_dt],
        default=0,
    )
    eol_replacements = np.maximum(eol_replacements, 0)

    sales_dt = np.zeros(sales_or_investment_in.shape)
    sales_dt[:, :, 0] = np.where(
        cap_growth_dt > 0.0,
        cap_growth_dt + eol_replacements,
        eol_replacements,
    )
    return sales_or_investment_in + sales_dt, sales_dt


def get_sales_yearly_old(cap, cap_lag, sales_or_investment_in, timescales):
    """Legacy singleton-third-axis implementation of :func:`get_sales_yearly`."""
    cap_growth = cap[:, :, 0] - cap_lag[:, :, 0]
    cap_depreciation = cap_lag[:, :, 0] / timescales

    eol_replacements = np.select(
        [
            cap_growth >= 0.0,
            (cap_depreciation > -cap_growth) & (cap_growth < 0),
        ],
        [cap_depreciation, cap_depreciation + cap_growth],
        default=0,
    )
    eol_replacements = np.maximum(eol_replacements, 0)

    sales_or_investment = np.zeros(sales_or_investment_in.shape)
    sales_or_investment[:, :, 0] = np.where(
        cap_growth > 0.0,
        cap_growth + eol_replacements,
        eol_replacements,
    )
    return sales_or_investment
