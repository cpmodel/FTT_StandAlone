"""Sales and investment calculations for FTT modules."""

import numpy as np


def get_sales(
    cap, cap_dt, cap_lag, sales_or_investment_in, timescales, dt,
    compare_with_old=False,
):
    """Calculate intra-year sales from 2D capacity and sales arrays.

    Set ``compare_with_old`` while converting a module to assert equivalence
    against :func:`get_sales_old`, which uses singleton-third-axis arrays.
    """
    cap_growth = cap - cap_lag
    cap_growth_dt = cap - cap_dt
    cap_depreciation = cap_lag * dt / timescales

    eol_replacements = np.select(
        [
            cap_growth >= 0.0,
            (cap_depreciation > -cap_growth_dt) & (cap_growth < 0),
        ],
        [cap_depreciation, cap_depreciation + cap_growth_dt],
        default=0,
    )
    eol_replacements = np.maximum(eol_replacements, 0)
    sales_dt = np.where(
        cap_growth_dt > 0.0,
        cap_growth_dt + eol_replacements,
        eol_replacements,
    )
    sales_or_investment = sales_or_investment_in + sales_dt

    if compare_with_old:
        old_sales, old_sales_dt = get_sales_old(
            cap[:, :, np.newaxis],
            cap_dt[:, :, np.newaxis],
            cap_lag[:, :, np.newaxis],
            sales_or_investment_in[:, :, np.newaxis],
            timescales,
            dt,
        )
        np.testing.assert_allclose(
            sales_or_investment, old_sales[:, :, 0], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            sales_dt, old_sales_dt[:, :, 0], rtol=1e-12, atol=1e-12
        )

    return sales_or_investment, sales_dt


def get_sales_yearly(
    cap, cap_lag, sales_or_investment_in, timescales, compare_with_old=False,
):
    """Calculate annual sales from 2D capacity and sales arrays."""
    cap_growth = cap - cap_lag
    cap_depreciation = cap_lag / timescales

    eol_replacements = np.select(
        [
            cap_growth >= 0.0,
            (cap_depreciation > -cap_growth) & (cap_growth < 0),
        ],
        [cap_depreciation, cap_depreciation + cap_growth],
        default=0,
    )
    eol_replacements = np.maximum(eol_replacements, 0)
    sales_or_investment = np.where(
        cap_growth > 0.0,
        cap_growth + eol_replacements,
        eol_replacements,
    )

    if compare_with_old:
        old_sales = get_sales_yearly_old(
            cap[:, :, np.newaxis],
            cap_lag[:, :, np.newaxis],
            sales_or_investment_in[:, :, np.newaxis],
            timescales,
        )
        np.testing.assert_allclose(
            sales_or_investment, old_sales[:, :, 0], rtol=1e-12, atol=1e-12
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
