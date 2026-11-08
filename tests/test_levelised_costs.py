import numpy as np

from ftt_source.ftt_core.ftt_get_levelised_costs import get_levelised_costs


def test_levelised_costs_return_distinct_policy_standard_deviations():
    lco, lco_with_policy, lco_sd, lco_with_policy_sd = get_levelised_costs(
        upfront=np.array([100.0]),
        upfront_policies=np.array([20.0]),
        upfront_sd=np.array([10.0]),
        annual=np.array([10.0]),
        annual_policies=np.array([5.0]),
        annual_sd=np.array([2.0]),
        service_delivered=np.array([1.0]),
        service_sd=np.array([0.2]),
        lifetimes=np.array([10.0]),
        r=np.array([0.05]),
    )

    assert lco_with_policy[0] > lco[0]
    assert lco_with_policy_sd[0] > lco_sd[0]
