# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 08:36:58 2025

@author: Femke
"""

import numpy as np
from numba import njit
from math import sqrt


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
        F = np.ones((num_techs, num_techs)) * 0.5

        # Market share constraints (if any)
        Gijmax = np.ones((num_techs))
        Gijmin = np.ones((num_techs))

        for t1 in range(num_techs):

            if not (shares_dt[r, t1, 0] > 0.0 and
                    costs_dt[r, t1, 0] != 0.0):
                continue
            
            if limits_active:
                Gijmax[t1] = np.tanh(1.25 * (upper_limit[r, t1, 0] - shares_dt[r, t1, 0]) / 0.1)
                Gijmin[t1] = 0.5 + 0.5 * np.tanh(1.25 * (-lower_limit[r, t1, 0] + shares_dt[r, t1, 0]) / 0.1)
          
            dSij[t1, t1] = 0
            S_i = shares_dt[r, t1, 0]

            for t2 in range(t1):

                if not (shares_dt[r, t2, 0] > 0.0 and
                        costs_dt[r, t2, 0] != 0.0):
                    continue

                S_j = shares_dt[r, t2, 0]

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
                dSij[t1, t2] = _rk4_integration(S_i, S_j, delta_AFG, dt)
                dSij[t2, t1] = -dSij[t1, t2]
            
        
        dSij_all[r] = dSij
    
    dSij_sum = np.sum(dSij_all, axis=2)
    
    return dSij_sum
    

@njit(fastmath=True)
def _rk4_integration(S_i, S_j, delta_AFG, dt):
    """Helper function for RK4 calculation"""
    k_1 = S_i * S_j * delta_AFG
    k_2 = (S_i + dt * k_1/2) * (S_j - dt * k_1 / 2) * delta_AFG
    k_3 = (S_i + dt * k_2/2) * (S_j - dt * k_2 / 2) * delta_AFG
    k_4 = (S_i + dt * k_3) * (S_j - dt * k_3) * delta_AFG
    
    return (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6


@njit(fastmath=True)
def shares_premature(dt, regions, shares_dt,
                     costs_marg, costs_marg_sd,
                     costs_payb, costs_payb_sd,
                     scrappage_rate, subst, isReg, num_regions, num_techs):
    
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
    
                # Original regulation adjustment for premature
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
