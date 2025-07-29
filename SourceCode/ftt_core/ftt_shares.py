# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 08:36:58 2025

@author: Femke
"""

import numpy as np
from numba import njit


def shares_change(dt, t, regions, shares_dt, costs_dt, costs_sd_dt,
           subst, isReg, num_regions, num_techs,
           upper_limit=None, lower_limit=None, limits_active=False):
    
    '''This is a wrapper function for the jitted shares function. We want
    to always give the same types into the function for rapid compile'''
    
    if not limits_active:
        upper_limit = np.empty((num_regions, num_techs, 1))
        lower_limit = np.empty((num_regions, num_techs, 1))
    
    change_in_shares = shares_change_jitted(dt, t, regions, shares_dt, costs_dt, costs_sd_dt,
               subst, isReg, num_regions, num_techs,
               upper_limit, lower_limit, limits_active)
    
    return change_in_shares
    

@njit(fastmath=True)
def shares_change_jitted(dt, t, regions, shares_dt, costs_dt, costs_sd_dt,
           subst, isReg, num_regions, num_techs,
           upper_limit, lower_limit, limits_active=False):

    """
    Function to calculate market share dynamics.

   This function calculates market shares based on market shares of the
   previous iteration.
    
    Parameters
    ----------
    dt : float
        The time step size.
    t : int
        The current time step.
    demand, shares_dt, costs_dt, costs_sd_dt, upper_limit_dt, lower_limit_dt, subst, isReg : ndarray
        Input arrays used in the calculation of market shares. 
    num_regions : float
        Number of regions
    num_techs : float
        Number of technologies.

    Returns
    -------
    ndarray
        The change in shares, taking into account regulation and endogenous limits (optional)
        
    Notes
    -----
    This function is decorated with `@njit(fastmath=True)` for performance optimization.
    """

    dSij_all = np.zeros((num_regions, num_techs, num_techs))

    for r in regions:

        # Initialise variables related to market share dynamics
        # dSij contains the change in shares
        dSij = np.zeros((num_techs, num_techs))

        # F contains the preferences
        F = np.ones((num_techs, num_techs))*0.5

        # Market share constraints (if any)
        Gijmax = np.ones((num_techs))
        Gijmin = np.ones((num_techs))

        for t1 in range(num_techs):

            if not (shares_dt[r, t1, 0] > 0.0 and
                    costs_dt[r, t1, 0] != 0.0 and
                    costs_sd_dt[r, t1, 0] != 0.0): 
                continue
            
            if limits_active:
                Gijmax[t1] = np.tanh(1.25 * (upper_limit[r, t1, 0] - shares_dt[r, t1, 0]) / 0.1)
                Gijmin[t1] = 0.5 + 0.5 * np.tanh(1.25 * (-lower_limit[r, t1, 0] + shares_dt[r, t1, 0]) / 0.1)
          
            dSij[t1, t1] = 0
            S_i = shares_dt[r, t1, 0]

            for t2 in range(t1):

                if not (shares_dt[r, t2, 0] > 0.0 and
                        costs_dt[r, t2, 0] != 0.0 and
                        costs_sd_dt[r, t2, 0] != 0.0): 
                    continue

                S_k = shares_dt[r, t2, 0]

                # Propagating width of variations in perceived costs
                dFij = np.sqrt(2) * np.sqrt(costs_sd_dt[r, t1, 0] * costs_sd_dt[r, t1, 0]
                                          + costs_sd_dt[r, t2, 0] * costs_sd_dt[r, t2, 0])

                # Consumer preference incl. uncertainty
                Fij = 0.5 * (1 + np.tanh(1.25 * (costs_dt[r, t2, 0] - costs_dt[r, t1, 0]) / dFij))

                # Preferences are then adjusted for regulations
                F[t1, t2] = Fij*(1.0-isReg[r, t1]) * (1.0 - isReg[r, t2]) + isReg[r, t2]*(1.0-isReg[r, t1]) + 0.5*(isReg[r, t1]*isReg[r, t2])
                F[t2, t1] = (1.0-Fij)*(1.0-isReg[r, t2]) * (1.0 - isReg[r, t1]) + isReg[r, t1]*(1.0-isReg[r, t2]) + 0.5*(isReg[r, t2]*isReg[r, t1])
                
                if limits_active:
                    # Runge-Kutta market share dynamics (do not remove the divide-by-6, it is part of the algorithm)
                    delta_AFG =  (subst[r, t1, t2] * F[t1, t2] * Gijmax[t1] * Gijmin[t2]
                                - subst[r, t2, t1] * F[t2, t1] * Gijmax[t2] * Gijmin[t1])
                else:
                    delta_AFG =  (subst[r, t1, t2] * F[t1, t2]
                                - subst[r, t2, t1] * F[t2, t1])
                
                # Change in shares = S_i * S_j * delta_AFG
                dSij[t1, t2] = _rk4_integration(S_i, S_k, delta_AFG, dt)
                dSij[t2, t1] = -dSij[t1, t2]
            
        
        dSij_all[r] = dSij
    
    dSij_sum = np.sum(dSij_all, axis=2)
    
    return dSij_sum
    


@njit(fastmath=True)
def _rk4_integration(S_i, S_k, delta_AFG, dt):
    """Helper function for RK4 calculation"""
    k_1 = S_i * S_k * delta_AFG
    k_2 = (S_i + dt * k_1/2) * (S_k - dt * k_1 / 2) * delta_AFG
    k_3 = (S_i + dt * k_2/2) * (S_k - dt * k_2 / 2) * delta_AFG
    k_4 = (S_i + dt * k_3) * (S_k - dt * k_3) * delta_AFG
    
    return (k_1 + 2*k_2 + 2*k_3 + k_4) * dt / 6

    
def shares_premature(dt, shares_dt, costs_marginal_dt, costs_marginal_sd_dt, 
                    costs_payback_dt, costs_payback_sd_dt, subst,
                    scrappage_rate, isReg, regions):
    """
    Vectorized shares function for premature replacements.
    
    This function implements the market share dynamics for premature replacements
    using replicator dynamics and Runge-Kutta integration.
    
    Parameters
    ----------
    dt : float
        Time step
    shares_dt : ndarray
        Market shares
    costs_marginal_dt : ndarray
        Marginal costs (HGC2)
    costs_marginal_sd_dt : ndarray
        Standard deviations of marginal costs (HGD2)
    costs_payback_dt : ndarray
        Payback costs (HGC3)
    costs_payback_sd_dt : ndarray
        Standard deviations of payback costs (HGD3)
    subst : ndarray
        Substitution matrix
    scrappage_rate : ndarray
        Technology scrappage rates (SR)
    isReg : ndarray
        Regulation indicators
    regions : array_like
        List of region indices to process
        
    Returns
    -------
    dSij_all : ndarray
        Change in market shares matrix for all regions
    """
    num_regions = len(shares_dt)
    num_techs = len(shares_dt[0])
    
    dSij_all = np.zeros((num_regions, num_techs, num_techs))
    
    if len(regions) == 0:
        return dSij_all
    
    b1, b2 = np.triu_indices(num_techs, k=1)
    skip_mask = _skip_criteria_premature(shares_dt, costs_marginal_dt, costs_marginal_sd_dt, 
                                        costs_payback_dt, costs_payback_sd_dt, 
                                        scrappage_rate, b1, b2)
    
    # The core FTT equations for premature replacements
    for r in regions:
        dSij_all[r] = shares_change_premature(
            r, b1, b2, skip_mask[r], dt,
            shares_dt, costs_marginal_dt, costs_marginal_sd_dt,
            costs_payback_dt, costs_payback_sd_dt, subst, 
            scrappage_rate, isReg, num_techs
        )
    
    return dSij_all


def shares_change_premature(r, b1, b2, skip_mask_r, dt,
                           shares_dt, costs_marginal_dt, costs_marginal_sd_dt,
                           costs_payback_dt, costs_payback_sd_dt, subst, 
                           scrappage_rate, isReg, num_techs):
    """
    Compute premature replacement share changes for a single region.
    
    This combines the core FTT market dynamics calculations for premature
    replacements in a single region.
    """
    # Filter valid technology pairs
    mask = ~skip_mask_r
    i1, i2 = b1[mask], b2[mask]
    
    if len(i1) == 0:  # No valid pairs
        return np.zeros((num_techs, num_techs))
    
    # Compute substitution rates using scrappage rates
    Aij = subst[0, i1, i2] * scrappage_rate[r, i2]  # SR[b2] in original
    Aji = subst[0, i2, i1] * scrappage_rate[r, i1]  # SR[b1] in original
    
    # Compute cost preferences for premature replacements
    # Compare marginal costs vs payback costs
    dFEij = 1.414 * np.sqrt(costs_payback_sd_dt[r, i1, 0]**2 + costs_marginal_sd_dt[r, i2, 0]**2)
    cost_difference_ij = costs_marginal_dt[r, i2, 0] - costs_payback_dt[r, i1, 0]
    
    # Compare marginal costs vs payback costs (reverse direction)
    dFEji = 1.414 * np.sqrt(costs_marginal_sd_dt[r, i1, 0]**2 + costs_payback_sd_dt[r, i2, 0]**2)
    cost_difference_ji = costs_marginal_dt[r, i1, 0] - costs_payback_dt[r, i2, 0]
    
    # Cost preference using tanh approximation (note: 1.25 not 1.25/sqrt(2))
    FEij = 0.5 * (1 + np.tanh(1.25 * cost_difference_ij / dFEij))
    FEji = 0.5 * (1 + np.tanh(1.25 * cost_difference_ji / dFEji))
    
    # Adjust for regulation (different from normal replacements)
    FEij_reg = FEij * (1.0 - isReg[r, i1])
    FEji_reg = FEji * (1.0 - isReg[r, i2])
    
    # Compute net preference-weighted substitution rate
    delta_F_A = Aij * FEij_reg - Aji * FEji_reg
    
    # Change is shares is S_i * S_j * delta_F_A
    # RK4 is used for accuracy and speed
    S_all = shares_dt[r, :, 0]
    S_i, S_j = S_all[i1], S_all[i2]
    dS = _rk4_integration(S_i, S_j, delta_F_A, dt)
    
    # Build dSij matrix
    dSij = np.zeros((num_techs, num_techs))
    dSij[i1, i2] = dS
    dSij[i2, i1] = -dS
    
    return dSij


def _skip_criteria_premature(shares_dt, costs_marginal_dt, costs_marginal_sd_dt, 
                            costs_payback_dt, costs_payback_sd_dt, 
                            scrappage_rate, b1, b2):
    """
    Skip technology pairs when shares, costs, or scrappage rates are invalid
    for premature replacements.
    
    Returns skip mask: (num_regions, num_pairs) boolean array where True = skip
    """
    # Extract data for all regions and pairs at once
    shares_b1 = shares_dt[:, b1, 0]
    shares_b2 = shares_dt[:, b2, 0]
    costs_marginal_b1 = costs_marginal_dt[:, b1, 0]
    costs_marginal_b2 = costs_marginal_dt[:, b2, 0]
    costs_marginal_sd_b1 = costs_marginal_sd_dt[:, b1, 0]
    costs_marginal_sd_b2 = costs_marginal_sd_dt[:, b2, 0]
    costs_payback_b1 = costs_payback_dt[:, b1, 0]
    costs_payback_b2 = costs_payback_dt[:, b2, 0]
    costs_payback_sd_b1 = costs_payback_sd_dt[:, b1, 0]
    costs_payback_sd_b2 = costs_payback_sd_dt[:, b2, 0]
    scrap_b1 = scrappage_rate[:, b1]
    scrap_b2 = scrappage_rate[:, b2]
    
    # Valid conditions: shares > 0, costs != 0, cost_sd != 0, scrappage > 0
    valid_b1 = (shares_b1 > 0.0) & (costs_marginal_b1 != 0.0) & (costs_marginal_sd_b1 != 0.0) & \
               (costs_payback_b1 != 0.0) & (costs_payback_sd_b1 != 0.0) & (scrap_b1 > 0.0)
    valid_b2 = (shares_b2 > 0.0) & (costs_marginal_b2 != 0.0) & (costs_marginal_sd_b2 != 0.0) & \
               (costs_payback_b2 != 0.0) & (costs_payback_sd_b2 != 0.0) & (scrap_b2 > 0.0)
    
    # Skip if any condition fails
    return ~(valid_b1 & valid_b2)





