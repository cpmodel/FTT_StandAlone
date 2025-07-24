# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 08:36:58 2025

@author: Femke
"""

import numpy as np
from numba import njit

def shares(dt, t, shares_dt, costs_dt, costs_sd_dt, subst,
           turnover_rate, isReg, demand, regions):
    """Vectorized shares function for improved performance.
    
    This function implements the market share dynamics using replicator
    dynamics and Runge-Kutta integration
    
    It is vectorised. Instead of having a matrix of tech1 by tech2, it
    puts everything in a big array to make calculations more rapid.
    
    Parameters
    ----------
    regions : array_like
        List of region indices to process. Can be empty array if no regions qualify.
    """

    num_regions = len(shares_dt)
    num_techs = len(shares_dt[0])
    
    endo_shares = np.zeros((num_regions, num_techs))
    endo_capacity = np.zeros((num_regions, num_techs))
    
    # Constant for the approximation of the CDF of the normal distribution
    const_CDF = 1.25 / np.sqrt(2)
    
    # Get pairwise indices. triu_indices returns the indices of the upper triangle of a matrix
    b1, b2 = np.triu_indices(num_techs, k=1)
    
    # Get skip mask for all regions
    skip_mask = _skip_criteria(shares_dt, costs_dt, costs_sd_dt, subst, b1, b2)
    
    # Convert regions to list for iteration
    regions_to_process = list(regions)
    
    for r in regions_to_process:
        
        # Get valid pairs for this region
        mask = ~skip_mask[r]
        i1, i2 = b1[mask], b2[mask]

        # Substitution rates
        Aij = subst[0, i1, i2] * turnover_rate[r, i1]  
        Aji = subst[0, i2, i1] * turnover_rate[r, i2]
        
        # Width of the cost distribution
        dFij = np.sqrt(
              costs_sd_dt[r, i1, 0] * costs_sd_dt[r, i1, 0]
            + costs_sd_dt[r, i2, 0] * costs_sd_dt[r, i2, 0]
        )
        
        # Cost comparison. Error function approximated by 0.5 * (1 + (tanh(1.25 / sqrt(2) * x))
        # TODO: check if this approximation is correct, per https://www.johndcook.com/blog/2025/03/06/gelu/
        Fij = 0.5 * (
            1 + np.tanh(
                const_CDF * (costs_dt[r, i2, 0] - costs_dt[r, i1, 0]) / dFij
            )
        )
        
        # Adjust the prefereces based on regulation
        reg_i1 = isReg[r, i1]
        reg_i2 = isReg[r, i2]

        Fij_reg = (Fij * (1.0 - reg_i1) * (1.0 - reg_i2)
            + reg_i2 * (1.0 - reg_i1)
            + 0.5 * (reg_i1 * reg_i2))

        Fji_reg = ((1.0 - Fij) * (1.0 - reg_i2) * (1.0 - reg_i1)
            + reg_i1 * (1.0 - reg_i2)
            + 0.5 * (reg_i2 * reg_i1))

        # Compute net preference-weighted substitution rate
        delta_F_A = Aij * Fij_reg - Aji * Fji_reg
        
        S_all = shares_dt[r, :, 0]          # All shares for this region
        S_i, S_j = S_all[i1], S_all[i2]     # Shares for the interacting pairs
        
        # dS = S_i * S_j * delta_F_A. 
        # Use JIT-compiled RK4 integration
        dS = _rk4_integration(S_i, S_j, delta_F_A, dt)
        
        # Update shares
        dSij = np.zeros((num_techs, num_techs))
        dSij[i1, i2] = dS           # Fill in the changes for i1 -> i2
        dSij[i2, i1] = -dS          # Fill in the changes for i2 -> i1

        # Sum changes for each technology
        dSij_total = np.sum(dSij, axis=1)

        # Update endogenous shares and capacity
        endo_shares[r] = S_all + dSij_total
        endo_capacity[r] = endo_shares[r] * demand[r]

    return endo_shares, endo_capacity


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

    
@njit
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