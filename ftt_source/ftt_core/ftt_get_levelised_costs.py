# -*- coding: utf-8 -*-
"""
Levelised cost functions (with and without a construction/lead time).

Created on Fri Dec 19 10:13:40 2025
@author: Femke Nijsse
"""

import numpy as np


# -----------------------------------------------------------------------------
# What is the "mid-year correction"?
# -----------------------------------------------------------------------------
# The standard annuity formulas assume every yearly payment happens at the END
# of the year. In reality, running costs and service are spread over the year,
# so on average they happen in the MIDDLE of the year (half a year earlier).
#
# Money that arrives half a year earlier is worth (1 + r) ** 0.5 times more
# today. This applies equally to the yearly costs AND the yearly service, so
# for those two it cancels out: annual / service stays the same.
#
# But the upfront cost is paid at a fixed moment, so it does NOT get that
# boost. Compared with the (now larger) discounted service, the upfront cost
# per unit of service becomes smaller by a factor (1 + r) ** -0.5.
# That factor is the "mid-year correction" and it is only applied to the
# upfront term.
#
# This is used in get_levelised_costs only. In get_levelised_costs_with_build
# the build-year payments are ALSO mid-year on average, so they shift together
# with the yearly costs and service, and nothing needs correcting.
# -----------------------------------------------------------------------------


# =============================================================================
# 1. Technologies WITHOUT a lead time
# =============================================================================
def get_levelised_costs(upfront, upfront_policies, upfront_sd,
                        annual, annual_policies, annual_sd,
                        service_delivered, service_sd,
                        lifetimes, r):
    """
    Levelised cost per unit of service (e.g. Eur/kWh), without and with
    policies, plus the standard deviation (SD) of the cost without policies.

    See https://www.wikiwand.com/en/articles/Annuity_(finance_theory)

    Parameters
    ----------
    upfront : np.ndarray            Capital cost paid at year 0.
    upfront_policies : np.ndarray   Extra upfront cost (or subsidy if negative).
    upfront_sd : np.ndarray         Standard deviation of the upfront cost.
    annual : np.ndarray             Yearly running cost (mid-year).
    annual_policies : np.ndarray    Extra yearly cost (or saving if negative) from policies.
    annual_sd : np.ndarray          Standard deviation of the yearly cost.
    service_delivered : np.ndarray  Service delivered per year (mid-year).
    service_sd : np.ndarray         Standard deviation of the yearly service.
    lifetimes : np.ndarray          Lifetime in years.
    r : np.ndarray                  Discount rate as a decimal (must be > 0).

    Returns
    -------
    lco : levelised cost without policies
    lco_with_policies : levelised cost including policies
    lco_sd : standard deviation of lco (no policies)
    """

    # ---------------------------------------------------------------------
    # MEAN COST
    # ---------------------------------------------------------------------
    # Capital recovery factor (CRF): turns a one-off cost into an equal cost
    # per year over the lifetime (assuming end-of-year payments).
    crf = r / (1 - (1 + r) ** -lifetimes)

    # Mid-year correction: yearly costs and service happen half a year
    # earlier than the CRF assumes. This shrinks the upfront term (see top).
    mid_year_correction = (1 + r) ** -0.5

    # Cost per year = (upfront cost spread over the years) + yearly cost
    total_cost = upfront * crf * mid_year_correction + annual
    total_cost_with_policies = ((upfront + upfront_policies) * crf * mid_year_correction
                                + (annual + annual_policies))

    # Levelised cost = cost per year / service per year
    lco = total_cost / service_delivered
    lco_with_policies = total_cost_with_policies / service_delivered

    # ---------------------------------------------------------------------
    # STANDARD DEVIATION (approximate, "first-order")
    # ---------------------------------------------------------------------
    # Idea: LCO = (total cost) / (total service), both in today's money.
    # For a ratio, the *relative* errors of top and bottom add up
    # "in quadrature":
    #
    #   rel. error of LCO = sqrt( (rel. error of cost)^2 + (rel. error of service)^2 )
    #
    # Cost and service are treated in exactly the same way.

    # Annuity factor: "how many years' worth" a yearly amount is worth today.
    # The mid-year shift makes yearly amounts worth (1 + r) ** 0.5 more.
    annuity_factor = (1 - (1 + r) ** -lifetimes) / r
    annuity_factor_mid = annuity_factor * (1 + r) ** 0.5

    # Same idea for uncertainties that are independent each year.
    # (Independent errors add as squares, so we sum discount factors squared.
    # Squared, the mid-year shift becomes (1 + r) ** 1.)
    annuity_factor_var = ((1 + r) ** -2
                          * (1 - (1 + r) ** (-2 * lifetimes))
                          / (1 - (1 + r) ** -2))
    annuity_factor_var_mid = annuity_factor_var * (1 + r)

    # Total cost in today's money, and its SD
    # (upfront and yearly uncertainties are assumed independent).
    npv_cost = upfront + annual * annuity_factor_mid
    npv_cost_sd = np.sqrt(upfront_sd ** 2 + annual_sd ** 2 * annuity_factor_var_mid)

    # Total service in today's money, and its SD.
    # The service error is one shared error for every year,
    # so its SD scales exactly like its mean.
    npv_service = service_delivered * annuity_factor_mid
    npv_service_sd = service_sd * annuity_factor_mid

    # Combine: relative errors of cost and service, times the LCO itself.
    # (With the mid-year correction, lco == npv_cost / npv_service exactly.)
    lco_sd = lco * np.sqrt((npv_cost_sd / npv_cost) ** 2
                           + (npv_service_sd / npv_service) ** 2)

    return lco, lco_with_policies, lco_sd


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
    All other parameters are as in get_levelised_costs.

    Assumptions
    -----------
    - The upfront cost is spread evenly over the construction years
      (paid at the end of years 1..leadtimes), each payment discounted to today.
    - Yearly costs and service start after construction is finished.
    - No mid-year correction is needed here: the build-year payments and the
      yearly costs/service all happen mid-year on average, so they shift by the
      same half year and this cancels in the cost/service ratio.
    - Uncertainties are independent, so variances add and we take the
      square root at the end (first-order approximation).
    """

    # This function only makes sense with a construction period.
    # Fail loudly instead of silently giving wrong numbers.
    if np.any(leadtimes <= 0):
        raise ValueError("leadtimes must be > 0 in get_levelised_costs_with_build")

    # Make sure these can be combined with the 'upfront' array.
    upfront_policies = np.atleast_1d(upfront_policies) * np.ones_like(upfront)
    upfront_sd = np.atleast_1d(upfront_sd) * np.ones_like(upfront)

    # ---------------------------------------------------------------------
    # CONSTRUCTION PHASE: value of the upfront cost in today's money
    # ---------------------------------------------------------------------
    q = (1 + r) ** -1      # discount factor for one year
    q2 = q ** 2            # same, but for squared amounts (used for the SD)

    # Shortcuts for adding up discount factors over the build years
    # (geometric series):
    #   q + q^2 + ... + q^B       = q  * (1 - q^B)  / (1 - q)
    #   q^2 + q^4 + ... + q^(2B)  = q2 * (1 - q2^B) / (1 - q2)
    sum_disc = q * (1 - q ** leadtimes) / (1 - q)
    sum_disc_sq = q2 * (1 - q2 ** leadtimes) / (1 - q2)

    # Each build year pays upfront / leadtimes. Add up their present values.
    pv_upfront = (upfront / leadtimes) * sum_disc
    pv_upfront_policies = (upfront_policies / leadtimes) * sum_disc

    # Variance of that sum: each payment has SD = upfront_sd / leadtimes.
    # Independent payments -> variances add.
    var_pv_upfront = (upfront_sd / leadtimes) ** 2 * sum_disc_sq

    # ---------------------------------------------------------------------
    # MEAN COST
    # ---------------------------------------------------------------------
    # Capital recovery factor: turns a one-off cost into an equal cost per year.
    crf = r / (1 - (1 + r) ** -lifetimes)

    # Yearly costs and service start after construction, so we discount them
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
    # Same idea as in the function above:
    #   rel. error of LCO = sqrt( (rel. error of cost)^2 + (rel. error of service)^2 )

    # "How many years' worth" a yearly amount is worth (annuity factor),
    # and the version for independent yearly errors.
    annuity_factor = (1 - (1 + r) ** -lifetimes) / r

    annuity_factor_var = ((1 + r) ** -2
                          * (1 - (1 + r) ** (-2 * lifetimes))
                          / (1 - (1 + r) ** -2))

    # Total cost in today's money, and its SD.
    # Yearly costs start after construction, so they get the extra leadtimes_df.
    npv_cost = pv_upfront + annual * leadtimes_df * annuity_factor
    npv_cost_sd = np.sqrt(var_pv_upfront
                          + annual_sd ** 2 * leadtimes_df ** 2 * annuity_factor_var)

    # Total service in today's money, and its SD (same scaling as the mean,
    # because the service error is one shared error for all years).
    npv_service = service_delivered * leadtimes_df * annuity_factor
    npv_service_sd = service_sd * leadtimes_df * annuity_factor

    # Combine: relative errors of cost and service, times the LCO itself
    lco_sd = lco * np.sqrt((npv_cost_sd / npv_cost) ** 2
                           + (npv_service_sd / npv_service) ** 2)

    return lco, lco_with_policies, lco_sd