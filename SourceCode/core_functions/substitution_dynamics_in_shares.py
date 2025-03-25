# -*- coding: utf-8 -*-
"""
=========================================
substitution_dynamics_in_shares.py
=========================================
Substitution dynamics in market shares.
############################

Function to estimate the substitution dynamics in market share space.

Functions included:
    - substitution_in_shares
        Solves the substitution dynamics based on levelised cost estimates.
"""

# Standard library imports
from math import sqrt
import copy
import warnings

# Third party imports
import numpy as np


def substitution_in_shares(act_shares, shares, submat, lc, lcsd, r, dt, titles):
    """
    This function applies the adapted Lotka-Volterra equation to determine substitution
    dynamics between several options in market share space. The differential equations
    are estimated by applying the Runge-Kutta method.

    Parameters
    ----------
    shares : 3D NumPy Array
        Lagged market shares.
    submat : 3D NumPy Array
        Substitution frequencies.
    lc : 3D NumPy Array
        Levelised cost metric.
    lcsd : 3D NumPy Array
        Standard deviation of the levelised costs.
    r : integer
        Country indicator.
    titles : dictionary
        Collection of title classifications used inside the model.

    Returns
    -------
    dSij : 2D NumPy Arrsay
        Matrix of market share changes between all pairs.

    """

    # Initialise variables related to market share dynamics
    # DSiK contains the change in shares
    dSij = np.zeros([len(titles['HYTI']), len(titles['HYTI'])])

    # F contains the preferences
    F = np.ones([len(titles['HYTI']), len(titles['HYTI'])]) * 0.5

    for t1 in range(len(titles['HYTI'])):

        if (not shares[r, t1, 0] > 0.0):
            continue

        S_i = shares[r, t1, 0]
        
        corr_i = act_shares[r, t1, 0] / shares[r, t1, 0]
        if corr_i > 1.0: corr_i = 1

        for t2 in range(t1):

            if (not shares[r, t2, 0] > 0.0):
                continue

            S_j = shares[r, t2, 0]
            corr_j = act_shares[r, t2, 0] / shares[r, t2, 0]
            if corr_j > 1.0: corr_j = 1

            # Propagating width of variations in perceived costs
            dFij = 1.414 * sqrt((lcsd[r, t1, 0] * lcsd[r, t1, 0]
                                 + lcsd[r, t2, 0] * lcsd[r, t2, 0]))


            # Consumer preference incl. uncertainty
            Fij = 0.5 * (1 + np.tanh(1.25 * (lc[r, t2, 0]
                                       - lc[r, t1, 0]) / dFij))

            # Preferences are then adjusted for regulations
            F[t1, t2] = Fij
            F[t2, t1] = (1.0 - Fij)

            #Runge-Kutta market share dynamiccs
            k_1 = S_i*S_j * (submat[0,t1, t2]*F[t1,t2]*corr_j- submat[0,t2, t1]*F[t2,t1]*corr_i)
            k_2 = (S_i+dt*k_1/2)*(S_j-dt*k_1/2)* (submat[0,t1, t2]*F[t1,t2]*corr_j - submat[0,t2, t1]*F[t2,t1]*corr_i)
            k_3 = (S_i+dt*k_2/2)*(S_j-dt*k_2/2) * (submat[0,t1, t2]*F[t1,t2]*corr_j - submat[0,t2, t1]*F[t2,t1]*corr_i)
            k_4 = (S_i+dt*k_3)*(S_j-dt*k_3) * (submat[0,t1, t2]*F[t1,t2]*corr_j - submat[0,t2, t1]*F[t2,t1]*corr_i)
            
            if t1 == 9:
                x = 1
            dSij[t1, t2] = dt*(k_1+2*k_2+2*k_3+k_4)/6
            dSij[t2, t1] = -dSij[t1, t2]
            
    return dSij

# %%

def innovator_effect(shares, submat, lc, lcsd, r, dt, titles, year, tech_exclusion, 
                     year_active=2029, nucleation_lim=0.2, innovator_rate=0.3):
    """
    The typical decision-making core in FTT only deals with the imitator effect
    and it does not allow for nucleation of novel technologies in the system.
    This function seeks to remedy that. 
    
    The decision-making core to represent the innovator effect is similar, but 
    has notable changes. First of all, Si represents a non-existent technology
    that may nucleate into the system. The condition is that its market share
    should be lower than the nucleation limit. Secondly, Sj always represents
    an incumbent technology with a share above the nucleation limit. Thirdly, 
    market share changes are only calculated moving from Sj to Si:
        
        dSij = (nucleation_lim - Si) * Sj * (Fij * Aij) * innovator_rate
        
    The innovator rate is there to tweak the magnitude of change.

    Parameters
    ----------
    shares : 3D NumPy Array
        Lagged market shares.
    submat : 3D NumPy Array
        Substitution frequencies.
    lc : 3D NumPy Array
        Levelised cost metric.
    lcsd : 3D NumPy Array
        Standard deviation of the levelised costs.
    r : integer
        Country indicator.
    titles : dictionary
        Collection of title classifications used inside the model.
    year : Integer
        Current year of the model run.
    tech_exclusion : 2D NumPy Array
        Boolean that determines which technologies in which regions are permissible
        for the nucleation routine.
    year_active : Integer, optional
        The year from which nucleation of new technologies is permissible. The default is 2029.
    nucleation_lim : Float, optional
        The maximum market share after which the innovator effects stops. From that
        point forward, diffusion is controlled by the imitator effect. The default is 0.2.
    innovator_rate : Float, optional
        Adjustment factor to slow or speed up substitution rates across the board.
        The default is 0.3.

    Returns
    -------
    dSij : 2D NumPy Arrsay
        Matrix of market share changes between all pairs.

    """
    
    # Initialise variables related to market share dynamics
    # DSiK contains the change in shares
    dSij = np.zeros([len(titles['HYTI']), len(titles['HYTI'])])

    # F contains the preferences
    F = np.ones([len(titles['HYTI']), len(titles['HYTI'])]) * 0.5
   
    if year <= year_active:
        
        return dSij
    
    else:
     
        for t1 in range(len(titles['HYTI'])):
            
            # Skip if shares of i greater than nucleation limit or if the tech
            # is to be excluded from the innovator effect.
            if (shares[r, t1, 0] > nucleation_lim or
                np.isclose(tech_exclusion[t1], 1.0)):
                continue
            
            # We take the difference between the nucleation_limit and the current share
            # This means that technologies with 0% share can be taken up in the system
            S_i = nucleation_lim - shares[r, t1, 0]
            
            # In this case, we will need to loop over all combinations as t1
            # represents the new technology, while t2 represents the incumbant
            for t2 in range(len(titles['HYTI'])):
                
                # Skip if t1 == t2
                if t1==t2:
                    continue
                
                # t2 represents the incumbant technology, so it's share always
                # needs to be greater than the nucleation_limit.
                if (not shares[r, t2, 0] > nucleation_lim):
                    continue
     
                S_j = shares[r, t2, 0]
     
                # Propagating width of variations in perceived costs
                dFij = 1.414 * sqrt((lcsd[r, t1, 0] * lcsd[r, t1, 0]
                                     + lcsd[r, t2, 0] * lcsd[r, t2, 0]))
     
     
                # Consumer preference incl. uncertainty
                Fij = 0.5 * (1 + np.tanh(1.25 * (lc[r, t2, 0]
                                           - lc[r, t1, 0]) / dFij))
     
                # Preferences are then adjusted for regulations
                F[t1, t2] = Fij
                F[t2, t1] = (1.0 - Fij)
     
                #Runge-Kutta market share dynamiccs
                #One-sided only
                k_1 = S_i*S_j * (submat[0,t1, t2]*F[t1,t2]) * innovator_rate
                k_2 = (S_i+dt*k_1/2)*(S_j-dt*k_1/2)* (submat[0,t1, t2]*F[t1,t2]) * innovator_rate
                k_3 = (S_i+dt*k_2/2)*(S_j-dt*k_2/2) * (submat[0,t1, t2]*F[t1,t2]) * innovator_rate
                k_4 = (S_i+dt*k_3)*(S_j-dt*k_3) * (submat[0,t1, t2]*F[t1,t2]) * innovator_rate
     
                # Check that Sij[t1, t2] is always positive and Sij[t2, t1] is always negative
                dSij[t1, t2] = dt*(k_1+2*k_2+2*k_3+k_4)/6
                dSij[t2, t1] = -dSij[t1, t2]
                
        return dSij
    