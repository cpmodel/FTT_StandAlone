# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 08:36:58 2025

@author: Femke Nijsse

Two versions of the shares equation are included in the file.
1. The normal shares equation as described by Mercure (2012)
2. The shares equation for premature replacements as described in Knobloch (2017)
"""

from math import sqrt

import numpy as np
from numba import njit

# Parameter to approximate the cumulative distribution function of the cost comparison
CDF_APPROX = 1.25 / sqrt(2) 

def shares_change(
    dt, regions,
    shares_dt,
    costs, costs_sd,
    subst, reg_constr, num_regions, num_techs,
    upper_limit=None, lower_limit=None, limits_active=False
    ):
    '''This is a wrapper function for the jitted shares function. We want
    to always give the same types into the function for rapid compile'''

    if not limits_active:
        upper_limit = np.empty((num_regions, num_techs, 1))
        lower_limit = np.empty((num_regions, num_techs, 1))

    change_in_shares = shares_change_jitted(dt, regions, shares_dt, costs, costs_sd,
               subst, reg_constr, num_regions, num_techs,
               upper_limit, lower_limit, limits_active)

    return change_in_shares


# Jit-in-time compilation. Comment this line out if you need to debug *in* the function
@njit(fastmath=True, cache=True)
def shares_change_jitted(
    dt, regions,
    shares_dt,
    costs, costs_sd,
    subst, reg_constr,
    num_regions, num_techs,
    upper_limit, lower_limit, limits_active=False
    ):

    """
    Function to calculate change in market shares, based on previous
    market shares, substitution rates and costs
     
    Parameters
    ----------
    dt : float
        Time step size
    shares_dt, costs, costs_sd : ndarray
        Shares, costs and standard deviation of costs at previous timestep
    subst, 
        Substitution matrix (determines speed), called Aij in paper
    reg_constr : ndarray
        Regulatory constraints: stops share changes to this tech
    num_regions, num_techs : int
        Number of regions and technologies
    upper_limit_dt, lower_limit_dt : ndarray
        Any minimum and maximum limits (e.g. for grid stability)
    limits_active : bool
        Whether limits are active or not
    
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

        # dSij contains the change in shares from each tech pair
        dSij = np.zeros((num_techs, num_techs))

        # F contains the preferences
        F = np.ones((num_techs, num_techs)) * 0.5

        # Market share constraints (if any)
        Gijmax = np.ones((num_techs))
        Gijmin = np.ones((num_techs))
        

        for tech_i in range(num_techs):

            S_i = shares_dt[r, tech_i, 0]
            if not (S_i > 0.0 and
                    costs[r, tech_i, 0] != 0.0):
                continue
            
            if limits_active:
                Gijmax[tech_i] = np.tanh(1.25 * (upper_limit[r, tech_i, 0] - S_i) / 0.1)
                Gijmin[tech_i] = 0.5 + 0.5 * np.tanh(1.25 * (-lower_limit[r, tech_i, 0] + S_i) / 0.1)
          
            for tech_j in range(tech_i):
                
                S_j = shares_dt[r, tech_j, 0]
                if not (S_j > 0.0 and
                        costs[r, tech_j, 0] != 0.0):
                    continue

                # Propagating width of variations in perceived costs
                dFij = np.sqrt(  costs_sd[r, tech_i, 0] * costs_sd[r, tech_i, 0]
                               + costs_sd[r, tech_j, 0] * costs_sd[r, tech_j, 0])

                # Consumer preference incl. uncertainty.
                # ERF is approximated with tanh, using the CDF_APPROX of 1.25/sqrt(2)
                Fij = 0.5 * (1 + np.tanh(CDF_APPROX * (costs[r, tech_j, 0] - costs[r, tech_i, 0]) / dFij))

                # Adjust preferences for regulated constraints
                F[tech_i, tech_j], F[tech_j, tech_i] = _apply_regulation_adjustment(
                    Fij, 1.0 - Fij, reg_constr[r, tech_i], reg_constr[r, tech_j])
                
                if limits_active:
                    # Minimum or maximum limits on technology, for instance for grid stability
                    delta_AFG =  (subst[r, tech_i, tech_j] * F[tech_j, tech_i] * Gijmax[tech_i] * Gijmin[tech_j]
                                - subst[r, tech_j, tech_i] * F[tech_i, tech_j] * Gijmax[tech_j] * Gijmin[tech_i])
                else:
                    delta_AFG =  (subst[r, tech_i, tech_j] * F[tech_j, tech_i]
                                - subst[r, tech_j, tech_i] * F[tech_i, tech_j])
                
                # Change in shares = S_i * S_j * delta_AFG
                dSij[tech_i, tech_j] = _rk4_integration(S_i, S_j, delta_AFG, dt)
                dSij[tech_j, tech_i] = -dSij[tech_i, tech_j]
        
        dSij_all[r] = dSij
    
    dSij_sum = np.sum(dSij_all, axis=2)
    
    return dSij_sum


# Jit-in-time compilation. Comment this line out if you need to debug *in* the function
@njit(fastmath=True, inline='always')
def _apply_regulation_adjustment(Fij, Fji, reg_constr_i, reg_constr_j):
    """
    Apply regulation constraint effects to base preferences.
    
    Logic:
    - Term 1: Base preference, scaled down when regulated
    - Term 2: Boost preference when target technology is regulated (blocked)
    - Term 3: Neutral preference (50%) when both technologies are regulated
    
    Parameters
    ----------
    Fij : float
        Base preference from technology i to j
    Fji : float  
        Base preference from technology j to i
    reg_constr_i, reg_constr_j : float
        reg_constr for technology i, j (0=no regulation, 1=fully regulated)
        
    Returns
    -------
    Fij_reg, Fji_reg : float
        Preferences adjusted for regulation effects
    """
    
    # i→j preference with regulation adjustment
    Fij_reg = (Fji * (1.0 - reg_constr_j) * (1.0 - reg_constr_i)  # Base preference term
               + reg_constr_i * (1.0 - reg_constr_j)              # i blocked → favor j
               + 0.5 * (reg_constr_j * reg_constr_i))             # Both blocked → neutral
    
    # j→i preference with regulation adjustment
    Fji_reg = (Fij * (1.0 - reg_constr_i) * (1.0 - reg_constr_j)  # Base preference term
               + reg_constr_j * (1.0 - reg_constr_i)              # j blocked → favor i
               + 0.5 * (reg_constr_i * reg_constr_j))             # Both blocked → neutral
    
    return Fij_reg, Fji_reg


# Jit-in-time compilation. Comment this line out if you need to debug *in* the function
@njit(fastmath=True)
def _rk4_integration(
    S_i, S_j, delta_AFG, dt
    ):
    """Helper function for RK4 calculation.
    
    We assume that within a timestep, the costs and the limits do not change"""
    
    k_1 = S_i * S_j * delta_AFG
    k_2 = (S_i + dt * k_1/2) * (S_j - dt * k_1 / 2) * delta_AFG
    k_3 = (S_i + dt * k_2/2) * (S_j - dt * k_2 / 2) * delta_AFG
    k_4 = (S_i + dt * k_3) * (S_j - dt * k_3) * delta_AFG
    
    return (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6


# Jit-in-time compilation. Comment this line out if you need to debug *in* the function
@njit(fastmath=True, cache=True)
def shares_change_premature(
    dt, regions,
    shares_dt,
    costs_marg, costs_marg_sd,
    costs_payb, costs_payb_sd,
    subst, reg_constr,
    num_regions, num_techs
    ):
    
    """
    Function to calculate change in market shares due to premature replacements,
    based on previous market shares, substitution rates and various costs
     
    Parameters
    ----------
    dt : float
        Time step size
    shares_dt costs, costs_sd : ndarray
        Shares in previous timestep
    costs_marg, costs_marg_sd : ndarray
        Marginal costs (fuel costs + O&M etc.)
    costs_payb, costs_payb_sd : ndarray
        Payback costs: more expensive than levelised cost, due to faster payback times
    subst, reg_constr : ndarray
        Substitution matrix (determines speed) and regulation (to slow growth)
    upper_limit_dt, lower_limit_dt : ndarray (implmentation still needed)
        Any minimum and maximum limits (e.g. for grid stability)
    num_regions, num_techs : int
        Number of regions and technologies
    
    Returns
    -------
    ndarray
        The change in shares due to premature replacements,
        taking into account regulation. Endogenous limits still need implementing
        
    Notes
    -----
    This function is decorated with `@njit(fastmath=True)` for performance optimization.
    """
    
    dSij_all = np.zeros((num_regions, num_techs, num_techs))

    for r in regions:
        dSij = np.zeros((num_techs, num_techs))
        F = np.ones((num_techs, num_techs)) * 0.5
        
        for tech_i in range(num_techs):
            if not (shares_dt[r, tech_i, 0] > 0.0 and
                    costs_marg[r, tech_i, 0] != 0.0 and
                    costs_payb[r, tech_i, 0] != 0.0):
                continue
    
            S_i = shares_dt[r, tech_i, 0]
    
            for tech_j in range(tech_i):
                if not (shares_dt[r, tech_j, 0] > 0.0 and
                        costs_marg[r, tech_j, 0] != 0.0 and
                        costs_payb[r, tech_j, 0] != 0.0):
                    continue
    
                S_j = shares_dt[r, tech_j, 0]
    
                # Original premature replacement calculations
                dFij = sqrt(  costs_payb_sd[r, tech_i, 0] * costs_payb_sd[r, tech_i, 0]
                            + costs_marg_sd[r, tech_j, 0] * costs_marg_sd[r, tech_j, 0])
                dFji = sqrt(  costs_marg_sd[r, tech_i, 0] * costs_marg_sd[r, tech_i, 0]
                            + costs_payb_sd[r, tech_j, 0] * costs_payb_sd[r, tech_j, 0])
                
                # ERF is approximated with tanh, using the CDF_APPROX of 1.25/sqrt(2)
                Fij = 0.5 * (1 + np.tanh(CDF_APPROX * (costs_marg[r, tech_j, 0] - costs_payb[r, tech_i, 0]) / dFij))
                Fji = 0.5 * (1 + np.tanh(CDF_APPROX * (costs_marg[r, tech_i, 0] - costs_payb[r, tech_j, 0]) / dFji))
    
                # Original regulation adjustment for premature replacements
                F[tech_j, tech_i] = Fij * (1.0 - reg_constr[r, tech_i])
                F[tech_i, tech_j] = Fji * (1.0 - reg_constr[r, tech_j])
                
                
                delta_AFG = (subst[r, tech_i, tech_j] * F[tech_j, tech_i]
                           - subst[r, tech_j, tech_i] * F[tech_i, tech_j])
                
                # Change in shares = S_i * S_j * delta_AFG
                dSij[tech_i, tech_j] = _rk4_integration(S_i, S_j, delta_AFG, dt)
                dSij[tech_j, tech_i] = -dSij[tech_i, tech_j]
                
        dSij_all[r] = dSij
        
    dSij_sum = np.sum(dSij_all, axis=2)
                
    return dSij_sum
