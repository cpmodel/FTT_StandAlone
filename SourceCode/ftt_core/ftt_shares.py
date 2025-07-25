# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 08:36:58 2025

@author: Femke
"""

import numpy as np
from numba import njit


def shares(dt, t, shares_dt, costs_dt, costs_sd_dt, subst,
           turnover_rate, isReg, demand,
           regions='All', return_dSij=False):
    """
    Vectorized shares function.
    
    This function implements the market share dynamics using replicator
    dynamics and Runge-Kutta integration.
    
    Parameters
    ----------
    dt : float
        Time step
    t : int
        Current time step
    shares_dt : ndarray
        Market shares
    costs_dt : ndarray
        Technology costs
    costs_sd_dt : ndarray
        Cost standard deviations
    subst : ndarray
        Substitution matrix
    turnover_rate : ndarray
        Technology turnover rates
    isReg : ndarray
        Regulation indicators
    demand : ndarray
        Demand by region
    regions : array_like or str, optional
        List of region indices to process. If 'All', all regions are processed.
    return_dSij : bool, optional
        If True, return dSij matrix instead of final shares and capacity
        
    Returns
    -------
    result : ndarray or tuple
        If return_dSij=True: dSij_all matrix
        If return_dSij=False: (endo_shares, endo_capacity) tuple
    """
    num_regions = len(shares_dt)
    num_techs = len(shares_dt[0])
    
    regions_to_process = _get_regions_to_process(regions, num_regions)
    dSij_all = np.zeros((num_regions, num_techs, num_techs))
    
    if len(regions_to_process) == 0:
        return _handle_empty_regions(return_dSij, dSij_all, num_regions, num_techs)
    
    b1, b2 = np.triu_indices(num_techs, k=1)
    skip_mask = _skip_criteria(shares_dt, costs_dt, costs_sd_dt, subst, b1, b2)
    
    # The core FTT equations
    for r in regions_to_process:
        dSij_all[r] = shares_change(
            r, b1, b2, skip_mask[r], dt,
            shares_dt, costs_dt, costs_sd_dt, subst, 
            turnover_rate, isReg, num_techs
        )
    
    # Return results
    if return_dSij:
        return dSij_all
    
    # Compute final shares and capacity
    dSij_total = np.sum(dSij_all, axis=2)
    endo_shares = np.zeros((num_regions, num_techs))
    endo_shares[regions_to_process] = shares_dt[regions_to_process, :, 0] + dSij_total[regions_to_process]
    endo_capacity = endo_shares * demand[:, np.newaxis]
    
    return endo_shares, endo_capacity


def shares_change(r, b1, b2, skip_mask_r, dt,
                 shares_dt, costs_dt, costs_sd_dt, subst, 
                 turnover_rate, isReg, num_techs):
    """
    Compute share changes for a single region.
    
    This combines the core FTT market dynamics calculations for a single region,
    including substitution rates, cost preferences, regulation adjustments,
    and Runge-Kutta integration.
    """
    # Filter valid technology pairs
    mask = ~skip_mask_r
    i1, i2 = b1[mask], b2[mask]
    
    if len(i1) == 0:  # No valid pairs
        return np.zeros((num_techs, num_techs))
    
    # Compute substitution rates
    Aij = subst[0, i1, i2] * turnover_rate[r, i1]
    Aji = subst[0, i2, i1] * turnover_rate[r, i2]
    
    # Compute cost preferences
    # Width of cost distribution
    dFij = np.sqrt(costs_sd_dt[r, i1, 0]**2 + costs_sd_dt[r, i2, 0]**2)
    
    # Cost preference using tanh approximation of error function
    cost_difference = costs_dt[r, i2, 0] - costs_dt[r, i1, 0]
    const_CDF = 1.25 / np.sqrt(2)
    Fij = 0.5 * (1 + np.tanh(const_CDF * cost_difference / dFij))
    
    # Adjust for regulation
    Fij_reg, Fji_reg = _adjust_preferences_for_regulation(Fij, isReg[r], i1, i2)
    
    # Compute net preference-weighted substitution rate
    delta_F_A = Aij * Fij_reg - Aji * Fji_reg
    
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


@njit    # Use numba to compile. Comment out if you need to debug this function
def _adjust_preferences_for_regulation(Fij, isReg, i1, i2):
    """
    Adjust preferences based on regulation.
    
    Parameters
    ----------
    Fij : ndarray
        Consumer preference without regulation
    reg_i1, reg_i2 : ndarray
        Regulation values for technologies i1 and i2
        
    Returns
    -------
    Fij_reg, Fji_reg : tuple of ndarrays
        Adjusted preferences for both directions
    """
    
    reg_i1, reg_i2 = isReg[i1], isReg[i2]

    Fij_reg = (Fij * (1.0 - reg_i1) * (1.0 - reg_i2)
               + reg_i2 * (1.0 - reg_i1)
               + 0.5 * (reg_i1 * reg_i2))

    Fji_reg = ((1.0 - Fij) * (1.0 - reg_i2) * (1.0 - reg_i1)
               + reg_i1 * (1.0 - reg_i2)
               + 0.5 * (reg_i2 * reg_i1))
    
    return Fij_reg, Fji_reg

def _get_regions_to_process(regions, num_regions):
    """
    Determine the regions to process based on the input.

    Parameters
    ----------
    regions : array_like or str
        List of region indices to process. If 'All', all regions are processed.
    num_regions : int
        Total number of regions available.

    Returns
    -------
    regions_to_process : array_like
        List of region indices to process.
    """
    if isinstance(regions, str) and regions == 'All':
        return np.arange(num_regions)  # Process all regions
    elif len(regions) == 0:
        return []  # No regions to process
    else:
        return regions

    
def _handle_empty_regions(return_dSij, dSij_all, num_regions, num_techs):
    """Handle case when no regions are processed."""
    if return_dSij:
        return dSij_all
    else:
        return (np.zeros((num_regions, num_techs)), 
                np.zeros((num_regions, num_techs)))


def _skip_criteria(shares_dt, costs_dt, costs_sd_dt, subst, b1, b2):
    """
    Skip technology pairs when shares, costs, or substitution are invalid.
    
    Returns skip mask: (num_regions, num_pairs) boolean array where True = skip
    """
    # Extract data for all regions and pairs at once
    shares_b1 = shares_dt[:, b1, 0]
    shares_b2 = shares_dt[:, b2, 0]
    costs_b1 = costs_dt[:, b1, 0]
    costs_b2 = costs_dt[:, b2, 0]
    costs_sd_b1 = costs_sd_dt[:, b1, 0]
    costs_sd_b2 = costs_sd_dt[:, b2, 0]
    subst_b1_b2 = subst[0, b1, b2]
    subst_b2_b1 = subst[0, b2, b1]
    
    # Valid conditions: shares > 0, costs != 0, cost_sd != 0, substitution possible
    valid_b1 = (shares_b1 > 0.0) & (costs_b1 != 0.0) & (costs_sd_b1 != 0.0)
    valid_b2 = (shares_b2 > 0.0) & (costs_b2 != 0.0) & (costs_sd_b2 != 0.0)
    valid_subst = (subst_b1_b2 != 0.0) | (subst_b2_b1 != 0.0)
    
    # Skip if any condition fails
    return ~(valid_b1 & valid_b2 & valid_subst)

    
@njit     # Use numba to compile. Comment out if you need to debug this function
def _rk4_integration(S_i, S_j, delta_F_A, dt):
    """JIT-compiled Runge-Kutta 4th order integration for Lotka-Volterra equations.
    
    Parameters:
    -----------
    S_i : ndarray
        Market shares for technology i
    S_j : ndarray  
        Market shares for technology j
    delta_F_A : ndarray
        Net preference-weighted substitution rate (Aij * Fij_reg - Aji * Fji_reg)
    dt : float
        Time step
        
    Returns:
    --------
    dS : ndarray
        Change in market shares
    """
    # Pre-compute dt factors to avoid repeated multiplication
    dt_half = dt * 0.5
    dt_sixth = dt / 6.0
    
    # Runge-Kutta 4th order integration steps
    k1 = S_i * S_j * delta_F_A
    k2 = (S_i + dt_half * k1) * (S_j - dt_half * k1) * delta_F_A
    k3 = (S_i + dt_half * k2) * (S_j - dt_half * k2) * delta_F_A
    k4 = (S_i + dt * k3) * (S_j - dt * k3) * delta_F_A
    
    # Final integration step
    dS = dt_sixth * (k1 + 2 * (k2 + k3) + k4)
    
    return dS