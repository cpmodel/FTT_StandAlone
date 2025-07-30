# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 08:36:58 2025

@author: Femke Nijsse

Two versions of the shares equation are included in the file.
1. The normal shares equation as described by Mercure (2012)
2. The shares equation for premature replacements as described in Knobloch (2017)
"""

import numpy as np
from numba import njit
from math import sqrt


def shares_change(
    dt, regions,
    shares_dt,
    costs, costs_sd,
    subst, isReg, num_regions, num_techs,
    upper_limit=None, lower_limit=None, limits_active=False
    ):
    '''This is a wrapper function for the jitted shares function. We want
    to always give the same types into the function for rapid compile'''
    
    if not limits_active:
        upper_limit = np.empty((num_regions, num_techs, 1))
        lower_limit = np.empty((num_regions, num_techs, 1))
    
    change_in_shares = shares_change_jitted(dt, regions, shares_dt, costs, costs_sd,
               subst, isReg, num_regions, num_techs,
               upper_limit, lower_limit, limits_active)
    
    return change_in_shares
    
# Jit-in-time compilation. Comment this line out if you need to debug *in* the function
@njit(fastmath=True, cache=True)
def shares_change_jitted(
    dt, regions,
    shares_dt,
    costs, costs_sd,
    subst, isReg,
    num_regions, num_techs,
    upper_limit, lower_limit, limits_active=False
    ):

    """
    Function to calculate change in market shares, based on previous
    market shares, substitution rates and costs
     
    Parameters
    ----------
    dt : float
        The time step size.
    shares_dt, costs, costs_sd : ndarray
        Shares and cost arrays used in the calculation of market shares. 
    subst, isReg : ndarray
        Substitution matrix (determines speed) and regulation (to slow growth)
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

        # Initialise variables related to market share dynamics
        # dSij contains the change in shares
        dSij = np.zeros((num_techs, num_techs))

        # F contains the preferences
        F = np.ones((num_techs, num_techs)) * 0.5

        # Market share constraints (if any)
        Gijmax = np.ones((num_techs))
        Gijmin = np.ones((num_techs))

        for t1 in range(num_techs):
            
            S_i = shares_dt[r, t1, 0]
            if not (S_i > 0.0 and
                    costs[r, t1, 0] != 0.0):
                continue
            
            if limits_active:
                Gijmax[t1] = np.tanh(1.25 * (upper_limit[r, t1, 0] - S_i) / 0.1)
                Gijmin[t1] = 0.5 + 0.5 * np.tanh(1.25 * (-lower_limit[r, t1, 0] + S_i) / 0.1)
          
            for t2 in range(t1):
                
                S_j = shares_dt[r, t2, 0]
                if not (S_j > 0.0 and
                        costs[r, t2, 0] != 0.0):
                    continue


                # Propagating width of variations in perceived costs
                dFij = np.sqrt(2) * np.sqrt(costs_sd[r, t1, 0] * costs_sd[r, t1, 0]
                                          + costs_sd[r, t2, 0] * costs_sd[r, t2, 0])

                # Consumer preference incl. uncertainty
                Fij = 0.5 * (1 + np.tanh(1.25 * (costs[r, t2, 0] - costs[r, t1, 0]) / dFij))

                # Preferences are then adjusted for regulations
                F[t1, t2] = Fij*(1.0-isReg[r, t1]) * (1.0 - isReg[r, t2]) + isReg[r, t2]*(1.0-isReg[r, t1]) + 0.5*(isReg[r, t1]*isReg[r, t2])
                F[t2, t1] = (1.0-Fij)*(1.0-isReg[r, t2]) * (1.0 - isReg[r, t1]) + isReg[r, t1]*(1.0-isReg[r, t2]) + 0.5*(isReg[r, t2]*isReg[r, t1])
                
                if limits_active:
                    # Minimum or maximum limits on technology, for instance for grid stability
                    delta_AFG =  (subst[r, t1, t2] * F[t1, t2] * Gijmax[t1] * Gijmin[t2]
                                - subst[r, t2, t1] * F[t2, t1] * Gijmax[t2] * Gijmin[t1])
                else:
                    delta_AFG =  (subst[r, t1, t2] * F[t1, t2]
                                - subst[r, t2, t1] * F[t2, t1])
                
                # Change in shares = S_i * S_j * delta_AFG
                dSij[t1, t2] = _rk4_integration(S_i, S_j, delta_AFG, dt)
                dSij[t2, t1] = -dSij[t1, t2]
        
        dSij_all[r] = dSij
    
    dSij_sum = np.sum(dSij_all, axis=2)
    
    return dSij_sum
    

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
def shares_premature(
    dt, regions,
    shares_dt,
    costs_marg, costs_marg_sd,
    costs_payb, costs_payb_sd,
    scrappage_rate, subst, isReg,
    num_regions, num_techs
    ):
    
    """
    Function to calculate change in market shares due to premature replacements,
    based on previous market shares, substitution rates and various costs
     
    Parameters
    ----------
    dt : float
        The time step size.
    shares_dt costs, costs_sd : ndarray
        Shares in previous timestep
    costs_marg, costs_marg_sd : ndarray
        Marginal costs (fuel costs + O&M etc.)
    costs_payb, costs_payb_sd : ndarray
        Payback costs: more expensive than levelised cost, due to faster payback times
    subst, isReg : ndarray
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
        
        for b1 in range(num_techs):
            if not (shares_dt[r, b1, 0] > 0.0 and
                    costs_marg[r, b1, 0] != 0.0 and
                    costs_payb[r, b1, 0] != 0.0 and
                    scrappage_rate[r, b1] > 0.0):
                continue
    
            S_i = shares_dt[r, b1, 0]
    
            for b2 in range(b1):
                if not (shares_dt[r, b2, 0] > 0.0 and
                        costs_marg[r, b2, 0] != 0.0 and
                        costs_payb[r, b2, 0] != 0.0 and
                        scrappage_rate[r, b2] > 0.0):
                    continue
    
                S_j = shares_dt[r, b2, 0]
    
                # Original premature replacement calculations
                dFij = 1.414 * sqrt((costs_payb_sd[r, b1, 0] * costs_payb_sd[r, b1, 0]
                                   + costs_marg_sd[r, b2, 0] * costs_marg_sd[r, b2, 0]))
                dFji = 1.414 * sqrt((costs_marg_sd[r, b1, 0] * costs_marg_sd[r, b1, 0]
                                   + costs_payb_sd[r, b2, 0] * costs_payb_sd[r, b2, 0]))
    
                Fij = 0.5 * (1 + np.tanh(1.25 * (costs_marg[r, b2, 0] - costs_payb[r, b1, 0]) / dFij))
                Fji = 0.5 * (1 + np.tanh(1.25 * (costs_marg[r, b1, 0] - costs_payb[r, b2, 0]) / dFji))
    
                # Original regulation adjustment for premature replacements
                F[b1, b2] = Fij * (1.0 - isReg[r, b1])
                F[b2, b1] = Fji * (1.0 - isReg[r, b2])
                
                delta_AFG = (subst[0, b1, b2] * F[b1, b2] * scrappage_rate[r, b2]
                           - subst[0, b2, b1] * F[b2, b1] * scrappage_rate[r, b1])
                
                # Change in shares = S_i * S_j * delta_AFG
                dSij[b1, b2] = _rk4_integration(S_i, S_j, delta_AFG, dt)
                dSij[b2, b1] = -dSij[b1, b2]
                
        dSij_all[r] = dSij
        
    dSij_sum = np.sum(dSij_all, axis=2)
                
    return dSij_sum
