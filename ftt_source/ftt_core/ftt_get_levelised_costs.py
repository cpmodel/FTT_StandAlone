# -*- coding: utf-8 -*-
"""
Levelised cost functions (with and without a construction/lead time).

Created on Fri Dec 19 10:13:40 2025
@author: Femke Nijsse
"""

import numpy as np


def _annuity_factor(r, n):
    """
    Today's value of receiving 1 every year for n years, i.e. the sum of the
    discount factors 1/(1+r) + 1/(1+r)^2 + ... + 1/(1+r)^n.
    Later money is worth less, so this is a bit less than n.
    Example: r = 5%, n = 3  ->  2.72

    Formula: (1 - (1+r)^-n) / r   (for r = 0 this is 0/0, and the answer is n)
    """
    is_zero = (r == 0)
    safe_r = np.where(is_zero, 1.0, r)   # avoids dividing by zero
    return np.where(is_zero, n, (1 - (1 + safe_r) ** -n) / safe_r)


# =============================================================================
# 1. Technologies WITHOUT a lead time
# =============================================================================
def get_levelised_costs(upfront, upfront_policies, upfront_sd,
                        annual, annual_policies, annual_sd,
                        service_delivered, service_sd,
                        lifetimes, r):
    """
    Levelised cost per unit of service (e.g. Eur/kWh), without and with
    policies, plus the     standard deviation (SD) of the cost without and with policies.

    See https://www.wikiwand.com/en/articles/Annuity_(finance_theory)

    Parameters
    ----------
    upfront : np.ndarray            Capital cost paid at year 0.
    upfront_policies : np.ndarray   Extra upfront cost (or subsidy if negative).
    upfront_sd : np.ndarray         Standard deviation of the upfront cost.
    annual : np.ndarray             Yearly running cost.
    annual_policies : np.ndarray    Extra yearly cost (or saving if negative) from policies.
    annual_sd : np.ndarray          Standard deviation of the yearly cost.
    service_delivered : np.ndarray  Service delivered per year (e.g. kWh, km, heat).
    service_sd : np.ndarray         Standard deviation of the yearly service.
    lifetimes : np.ndarray          Lifetime in years.
    r : np.ndarray                  Discount rate as a decimal.

    Returns
    -------
    lco : levelised cost without policies
    lco_with_policies : levelised cost including policies
    lco_sd : standard deviation of lco
    lco_with_policies_sd : standard deviation of lco_with_policies
    """

    # Today's value of 1 per year over the lifetime, and the same for squared
    # discount factors (needed for the SD, because independent errors add as
    # squares). Squared factors (1+r)^(-2t) are an annuity at rate (1+r)^2 - 1.
    annuity_factor = _annuity_factor(r, lifetimes)
    annuity_factor_var = _annuity_factor((1 + r) ** 2 - 1, lifetimes)

    # ---------------------------------------------------------------------
    # MEAN COST
    # ---------------------------------------------------------------------
    # Capital recovery factor: spreads a one-off cost into an equal cost per year
    crf = 1 / annuity_factor

    # Mid-year correction: yearly costs and service arrive mid-year (worth
    # (1+r)^0.5 more today), but the upfront cost is paid at year 0. Relative
    # to the service, the upfront cost therefore counts (1+r)^-0.5 less.
    mid_year_correction = (1 + r) ** -0.5

    total_cost = upfront * crf * mid_year_correction + annual
    total_cost_with_policies = ((upfront + upfront_policies) * crf * mid_year_correction
                                + (annual + annual_policies))

    lco = total_cost / service_delivered
    lco_with_policies = total_cost_with_policies / service_delivered

    # ---------------------------------------------------------------------
    # STANDARD DEVIATION (approximate, "first-order")
    # ---------------------------------------------------------------------
    # LCO = (total cost) / (total service), both in today's money. Then:
    #   - 1 extra unit of cost raises the LCO by 1 / service
    #   - 1 extra unit of service lowers the LCO by lco / service
    # The two effects are independent, so they add as squares:
    #
    #   sd = sqrt( (cost_sd / service)^2 + (lco * service_sd / service)^2 )
    #
    # Policies change the mean cost, not the cost uncertainty. The resulting
    # SD therefore differs only through the mean cost used for service risk.

    # Mid-year shift: yearly amounts are worth (1+r)^0.5 more (squared: 1+r)
    annuity_factor_mid = annuity_factor * (1 + r) ** 0.5
    annuity_factor_var_mid = annuity_factor_var * (1 + r)

    # SD of the total cost: upfront and yearly costs are independent, and the
    # yearly cost errors are independent between years.
    npv_cost_sd = np.sqrt(upfront_sd ** 2 + annual_sd ** 2 * annuity_factor_var_mid)

    # Total service and its SD. The service error is one shared error in every
    # year, so its SD scales exactly like the total service.
    npv_service = service_delivered * annuity_factor_mid
    npv_service_sd = service_sd * annuity_factor_mid

    lco_sd = np.sqrt((npv_cost_sd / npv_service) ** 2
                     + (lco * npv_service_sd / npv_service) ** 2)
    lco_with_policies_sd = np.sqrt((npv_cost_sd / npv_service) ** 2
                                   + (lco_with_policies * npv_service_sd / npv_service) ** 2)

    return lco, lco_with_policies, lco_sd, lco_with_policies_sd


# =============================================================================
# 2. Technologies WITH a lead time (construction period)
# =============================================================================
def get_levelised_costs_with_build(upfront, upfront_policies, upfront_sd,
                                   annual, annual_policies, annual_sd,
                                   service_delivered, service_sd,
                                   lifetimes, leadtimes, r):
    """
    Levelised cost (e.g. $/MWh) including a construction period.

    Extra parameter
    ---------------
    leadtimes : np.ndarray
        Construction time in years. Must be > 0 (use get_levelised_costs
        for technologies without a lead time).
    All other parameters and the returns are as in get_levelised_costs.

    Assumptions
    -----------
    - The upfront cost is spread evenly over the construction years
      (paid at the end of years 1..leadtimes), each payment discounted to today.
    - Yearly costs and service start once construction is finished.
    - No mid-year correction: build payments, yearly costs and service are all
      mid-year on average, so they shift together and this cancels out.
    - Uncertainties are independent, so variances add and we take the
      square root at the end (first-order approximation).
    """

    # Fail loudly instead of silently giving wrong numbers.
    if np.any(leadtimes <= 0):
        raise ValueError("leadtimes must be > 0 in get_levelised_costs_with_build")

    # Make sure these can be combined with the 'upfront' array.
    upfront_policies = np.atleast_1d(upfront_policies) * np.ones_like(upfront)
    upfront_sd = np.atleast_1d(upfront_sd) * np.ones_like(upfront)

    # ---------------------------------------------------------------------
    # CONSTRUCTION PHASE: value of the upfront cost in today's money
    # ---------------------------------------------------------------------
    # Sum of discount factors over the build years (and of the squared factors,
    # for the SD).
    sum_disc = _annuity_factor(r, leadtimes)
    sum_disc_sq = _annuity_factor((1 + r) ** 2 - 1, leadtimes)

    # Each build year pays upfront / leadtimes. Add up their present values.
    pv_upfront = (upfront / leadtimes) * sum_disc
    pv_upfront_policies = (upfront_policies / leadtimes) * sum_disc

    # Each payment has SD = upfront_sd / leadtimes. Payments are independent,
    # so their variances add.
    var_pv_upfront = (upfront_sd / leadtimes) ** 2 * sum_disc_sq

    # ---------------------------------------------------------------------
    # MEAN COST
    # ---------------------------------------------------------------------
    # Today's value of 1 per year over the lifetime, and the same for squared
    # discount factors (for the SD).
    annuity_factor = _annuity_factor(r, lifetimes)
    annuity_factor_var = _annuity_factor((1 + r) ** 2 - 1, lifetimes)

    # Capital recovery factor: spreads a one-off cost into an equal cost per year
    crf = 1 / annuity_factor

    # Yearly costs and service only start after construction, so discount them
    # back by the number of construction years.
    leadtimes_df = (1 + r) ** -leadtimes
    annual_after_leadtimes = annual * leadtimes_df
    annual_policies_after_leadtimes = annual_policies * leadtimes_df
    service_after_leadtimes = service_delivered * leadtimes_df

    total_cost = pv_upfront * crf + annual_after_leadtimes
    total_cost_with_policies = ((pv_upfront + pv_upfront_policies) * crf
                                + annual_after_leadtimes
                                + annual_policies_after_leadtimes)

    lco = total_cost / service_after_leadtimes
    lco_with_policies = total_cost_with_policies / service_after_leadtimes

    # ---------------------------------------------------------------------
    # STANDARD DEVIATION (approximate, "first-order")
    # ---------------------------------------------------------------------
    # Same idea as in get_levelised_costs:
    #   sd = sqrt( (cost_sd / service)^2 + (lco * service_sd / service)^2 )
    # using the LCO including policies.

    # SD of the total cost. Yearly costs start after construction, hence the
    # extra leadtimes_df.
    npv_cost_sd = np.sqrt(var_pv_upfront
                          + annual_sd ** 2 * leadtimes_df ** 2 * annuity_factor_var)

    # Total service and its SD (one shared error in every year, so the SD
    # scales like the total service).
    npv_service = service_delivered * leadtimes_df * annuity_factor
    npv_service_sd = service_sd * leadtimes_df * annuity_factor

    lco_sd = np.sqrt((npv_cost_sd / npv_service) ** 2
                     + (lco * npv_service_sd / npv_service) ** 2)
    lco_with_policies_sd = np.sqrt((npv_cost_sd / npv_service) ** 2
                                   + (lco_with_policies * npv_service_sd / npv_service) ** 2)

    return lco, lco_with_policies, lco_sd, lco_with_policies_sd