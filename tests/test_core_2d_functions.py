import numpy as np

from ftt_source.ftt_core.ftt_sales_or_investments import (
    get_sales,
    get_sales_old,
    get_sales_yearly,
    get_sales_yearly_old,
)
from ftt_source.ftt_core.ftt_shares import (
    shares_change,
    shares_change_old,
    shares_change_premature,
    shares_change_premature_old,
)
from ftt_source.ftt_core.ftt_mandate import (
    implement_mandate,
    implement_mandate_old,
    implement_seeding,
    implement_seeding_old,
)
from ftt_source.ftt_core.ftt_emissions_regulation import (
    implement_emissions_regulation,
    implement_emissions_regulation_old,
    reset_baseline_cache,
)


def test_2d_sales_functions_match_legacy_functions():
    cap = np.array([[4.0, 1.0], [2.0, 3.0]])
    cap_dt = np.array([[3.0, 1.5], [2.5, 2.0]])
    cap_lag = np.array([[2.0, 2.0], [3.0, 1.0]])
    sales = np.array([[1.0, 0.5], [0.0, 2.0]])
    timescales = np.array([[10.0, 20.0], [5.0, 4.0]])

    sales_2d, sales_dt_2d = get_sales(
        cap, cap_dt, cap_lag, sales, timescales, 0.25
    )
    sales_3d, sales_dt_3d = get_sales_old(
        cap[:, :, None],
        cap_dt[:, :, None],
        cap_lag[:, :, None],
        sales[:, :, None],
        timescales,
        0.25,
    )
    yearly_2d = get_sales_yearly(
        cap, cap_lag, sales, timescales
    )
    yearly_3d = get_sales_yearly_old(
        cap[:, :, None], cap_lag[:, :, None], sales[:, :, None], timescales
    )

    np.testing.assert_allclose(sales_2d, sales_3d[:, :, 0])
    np.testing.assert_allclose(sales_dt_2d, sales_dt_3d[:, :, 0])
    np.testing.assert_allclose(yearly_2d, yearly_3d[:, :, 0])


def test_2d_shares_functions_match_legacy_functions():
    regions = np.array([0, 1])
    shares = np.array([[0.4, 0.6], [0.7, 0.3]])
    costs = np.array([[1.0, 1.5], [2.0, 1.2]])
    costs_sd = np.array([[0.2, 0.3], [0.25, 0.2]])
    subst = np.array(
        [
            [[0.0, 0.2], [0.3, 0.0]],
            [[0.0, 0.25], [0.35, 0.0]],
        ]
    )
    reg_constr = np.array([[0.0, 0.2], [0.1, 0.0]])
    upper_limit = np.full((2, 2), 1.0)
    lower_limit = np.zeros((2, 2))

    standard_2d = shares_change(
        0.25, regions, shares, costs, costs_sd, subst, reg_constr, 2, 2,
        upper_limit, lower_limit, True, 10.0,
    )
    standard_3d = shares_change_old(
        0.25, regions, shares[:, :, None], costs[:, :, None],
        costs_sd[:, :, None], subst, reg_constr, 2, 2,
        upper_limit[:, :, None], lower_limit[:, :, None], True, 10.0,
    )
    premature_2d = shares_change_premature(
        0.25, regions, shares, costs, costs_sd, costs * 1.1,
        costs_sd * 1.1, subst, reg_constr, 2, 2,
    )
    premature_3d = shares_change_premature_old(
        0.25, regions, shares[:, :, None], costs[:, :, None],
        costs_sd[:, :, None], (costs * 1.1)[:, :, None],
        (costs_sd * 1.1)[:, :, None], subst, reg_constr, 2, 2,
    )

    np.testing.assert_allclose(standard_2d, standard_3d)
    np.testing.assert_allclose(premature_2d, premature_3d)


def test_2d_mandate_and_seeding_match_active_legacy_policy():
    capacity = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 1.0]])
    cumulative_sales = np.array([[0.4, 0.2, 0.1], [0.3, 0.1, 0.2]])
    sales = np.array([[0.2, 0.1, 0.1], [0.1, 0.2, 0.1]])
    mandate_info = np.array([[2020.0, 2025.0, 1.0], [2020.0, 2025.0, 1.0]])
    green_indices = [1, 2]

    mandate_2d = implement_mandate(
        capacity,
        cumulative_sales,
        sales,
        2022,
        green_indices,
        mandate_info,
    )
    mandate_3d = implement_mandate_old(
        capacity[:, :, None],
        cumulative_sales[:, :, None],
        sales[:, :, None],
        2022,
        green_indices,
        mandate_info,
    )
    seeding_2d = implement_seeding(
        capacity,
        cumulative_sales,
        sales,
        2021,
        green_indices,
        2020,
    )
    seeding_3d = implement_seeding_old(
        capacity[:, :, None],
        cumulative_sales[:, :, None],
        sales[:, :, None],
        2021,
        green_indices,
        2020,
    )

    for result_2d, result_3d in zip(mandate_2d, mandate_3d):
        np.testing.assert_allclose(result_2d, result_3d[:, :, 0])
    for result_2d, result_3d in zip(seeding_2d, seeding_3d):
        np.testing.assert_allclose(result_2d, result_3d[:, :, 0])


def test_2d_emissions_regulation_matches_active_legacy_policy():
    capacity = np.full((2, 27), 10.0)
    cumulative_sales = np.full((2, 27), 2.0)
    sales = np.full((2, 27), 1.0)
    emissions_intensity = np.tile(np.linspace(300.0, 0.0, 27), (2, 1))
    regulation_info = np.array([[2020.0, 2025.0, 1.0]] * 2)

    reset_baseline_cache()
    result_2d = implement_emissions_regulation(
        capacity, cumulative_sales, sales, 2023, emissions_intensity,
        regulation_info, 'transport',
    )
    reset_baseline_cache()
    result_3d = implement_emissions_regulation_old(
        capacity[:, :, None], cumulative_sales[:, :, None], sales[:, :, None],
        2023, emissions_intensity, regulation_info, 'transport',
    )

    for values_2d, values_3d in zip(result_2d, result_3d):
        np.testing.assert_allclose(values_2d, values_3d[:, :, 0])
