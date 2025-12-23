# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:13:40 2025

@author: Femke Nijsse
"""

import numpy as np

def get_levelised_costs(upfront, upfront_policies, upfront_sd,
                        annual, annual_policies, annual_sd,
                        service_delivered, service_sd,
                        lifetimes, r
                        ):
    """
    Calculate levelised costs per unit with and without policy,
    and their standard deviation
    
    See https://www.wikiwand.com/en/articles/Annuity_(finance_theory)

    Parameters
    ----------
    upfront : np.ndarray
        Initial capital costs at year 0.
    upfront_policies : np.ndarray
        Additional upfront costs or subsidies.
    upfront_sd : np.ndarray
        Standard deviation of upfront costs
    annual : np.ndarray
        Annual recurring costs.
    annual_policies : np.ndarray
        Annual costs or savings from policies.
    annual_sd : np.ndarray
        Standard deviation of annual costs
    service_delivered : np.ndarray
        Annual service delivered (e.g., kWh, km, heat).    
    service_sd : np.ndarray
        Standard deviation of service delivered 
    lifetimes : np.ndarray
        Asset/project lifetime in years.
    r : np.ndarray
        Discount rate as a decimal.

    Returns
    -------
    lco : np.ndarray
        Levelised cost per unit without policies
    lco_with_policies : np.ndarray
        Levelised cost per unit including policies
    lco_std : np.ndarray
        Standard deviation of LCO without policies
    """

    # Capital recovery factor (CRF) converts a present value into an equivalent uniform 
    # annual cost. It is the inverse of the annuity factor from the geometric series 
    # summation of discounted cash flows.
    
    # The mid-year correction accounts for the fact that variable cost happen mid-year
    # mid_year_correction = (1 + r) ** -0.5   # TODO: switch to this one when we're happy with PR. This correctly accounts for timing
    mid_year_correction = (1 + r) ** -1     # This reproduces what we had before
    crf = r / (1 - (1 + r) ** -lifetimes) * mid_year_correction
    
    # Total discounted costs
    total_cost = upfront * crf + annual
    total_cost_with_policies = (upfront + upfront_policies) * crf + (annual + annual_policies)
    
    # Levelised cost per unit
    lco = total_cost / service_delivered
    lco_with_policies = total_cost_with_policies / service_delivered
    
    
    # -------------------------
    # Approximate variance (first-order), roughly 5% error
    # -------------------------
    # Compute LCO standard deviation accounting for uncertainty in upfront cost, 
    # annual cost, and service.
    # Discount factors for annual cost and its variance
    discount_factor_annual = (1 - (1 + r)**(-lifetimes)) / r
    discount_factor_annual_var = (1 - (1 + r)**(-2 * lifetimes)) / (1 - (1 + r)**-2)
    
    # NPV of costs and variance
    npv_total_cost = upfront + annual * discount_factor_annual
    var_npv_total_cost = upfront_sd**2 + annual_sd**2 * discount_factor_annual_var
    
    # Discounted service delivered
    discounted_service = service_delivered * discount_factor_annual
    
    # LCO standard deviation
    lco_sd = np.sqrt(
        var_npv_total_cost / discounted_service**2
        + (npv_total_cost / discounted_service**2)**2 * service_sd**2
    )
    
    return lco, lco_with_policies, lco_sd



def get_levelised_costs_with_build(upfront, upfront_policies, upfront_sd,
                                   annual, annual_policies, annual_sd,
                                   service_delivered, service_sd,
                                   lifetimes, leadtimes, r):
    """
    Levelised cost with leadtimes-time adjustment.
    
    Parameters
    ----------
    leadtimes : np.ndarray
        Construction time in years
    Other variables as above

    Assumptions
    - Upfront CAPEX is spread evenly across leadtimes (years 1..leadtimes),
      each tranche discounted to present value (PV) and summed.
    - Annual OPEX and service start after the leadtimes period (shifted by 'leadtimes').
    - Variances are combined assuming independence with a first-order Taylor
      approximation: variances add; take the square root at the end.
    """

    # Ensure policies and SD can broadcast with upfront arrays
    upfront_policies = np.atleast_1d(upfront_policies) * np.ones_like(upfront)
    upfront_sd = np.atleast_1d(upfront_sd) * np.ones_like(upfront)

    # -------------------------
    # build-phase present value of CAPEX and its variance (closed-form geometric sums)
    # -------------------------
    # Sum_{k=1..B} q^k  = q * (1 - q^B) / (1 - q)
    # Sum_{k=1..B} q^{2k} = q^2 * (1 - q^{2B}) / (1 - q^2)
    q = (1 + r) ** -1       # One-period discount factor
    q2 = q ** 2
    one_minus_q = 1 - q
    one_minus_q2 = 1 - q2
    
    # Vectorised closed-form geometric sums over 'leadtimes' build years:
    # Check for leadtimes == 0 to avoid division by zero
    sum_disc = np.where(leadtimes > 0, q * (1 - q ** leadtimes) / one_minus_q, 0.0)
    sum_disc_sq = np.where(leadtimes > 0, q2 * (1 - (q2 ** leadtimes)) / one_minus_q2, 0.0)

    # Present value of upfront and policies spread evenly over leadtimes years
    pv_upfront = np.where(leadtimes > 0, (upfront / leadtimes) * sum_disc, 0.0)
    pv_upfront_policies = np.where(leadtimes > 0, (upfront_policies / leadtimes) * sum_disc, 0.0)

    # Variance over leadtimes years: Var(c*X) = c^2 Var(X), summed over years
    var_pv_upfront = np.where(leadtimes > 0, (upfront_sd / leadtimes) ** 2 * sum_disc_sq, 0.0)

    # -------------------------
    # Mean LCO
    # -------------------------
    
    # Capital recovery factor (CRF) converts a present value into an uniform annual cost
    crf = r / (1 - (1 + r) ** -lifetimes)

    # OPEX and service start after construction; apply leadtimes discount to per-year quantities
    leadtimes_df = (1 + r) ** -leadtimes
    annual_after_leadtimes = annual * leadtimes_df
    annual_policies_after_leadtimes = annual_policies * leadtimes_df
    service_after_leadtimes = service_delivered * leadtimes_df

    total_cost = pv_upfront * crf + annual_after_leadtimes
    total_cost_with_policies = (pv_upfront + pv_upfront_policies) * crf + annual_after_leadtimes + annual_policies_after_leadtimes

    lco = total_cost / service_after_leadtimes
    lco_with_policies = total_cost_with_policies / service_after_leadtimes

    # -------------------------
    # Standard deviation (NPV-based, consistent with base; Taylor approximation)
    # -------------------------
    annuity_factor = (1 - (1 + r) ** (-lifetimes)) / r
    annuity_factor_var = (1 - (1 + r) ** (-2 * lifetimes)) / (1 - (1 + r) ** -2)

    # NPV of costs and service (operations start after leadtimes)
    npv_total_cost = pv_upfront + annual * leadtimes_df * annuity_factor
    var_npv_total_cost = var_pv_upfront + (annual_sd ** 2) * (leadtimes_df ** 2) * annuity_factor_var
    npv_service = service_delivered * leadtimes_df * annuity_factor

    # Taylor approximation for SD of the ratio A/B
    lco_sd = np.sqrt(
        var_npv_total_cost / npv_service ** 2
        + (npv_total_cost / npv_service ** 2) ** 2 * service_sd ** 2
    )

    return lco, lco_with_policies, lco_sd



