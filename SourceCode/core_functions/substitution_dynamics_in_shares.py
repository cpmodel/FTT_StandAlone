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


def substitution_in_shares(shares, submat, lc, lcsd, r, dt, titles):
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

        for t2 in range(t1):

            if (not shares[r, t2, 0] > 0.0):
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
            k_1 = S_i*S_j * (submat[0,t1, t2]*F[t1,t2]- submat[0,t2, t1]*F[t2,t1])
            k_2 = (S_i+dt*k_1/2)*(S_j-dt*k_1/2)* (submat[0,t1, t2]*F[t1,t2] - submat[0,t2, t1]*F[t2,t1])
            k_3 = (S_i+dt*k_2/2)*(S_j-dt*k_2/2) * (submat[0,t1, t2]*F[t1,t2] - submat[0,t2, t1]*F[t2,t1])
            k_4 = (S_i+dt*k_3)*(S_j-dt*k_3) * (submat[0,t1, t2]*F[t1,t2] - submat[0,t2, t1]*F[t2,t1])

            dSij[t1, t2] = dt*(k_1+2*k_2+2*k_3+k_4)/6
            dSij[t2, t1] = -dSij[t1, t2]
            
    return dSij