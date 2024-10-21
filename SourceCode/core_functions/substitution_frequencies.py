# -*- coding: utf-8 -*-
"""
=========================================
substitution_frequencies.py
=========================================
FTT substitution frequency module.
#################################
"""

# Standard library imports
from math import sqrt
import os
import copy
import sys
import warnings

# Third party imports
import numpy as np
from numba import njit
from inspect import currentframe, getframeinfo
filename = getframeinfo(currentframe()).filename


def sub_freq(lifetimes, leadtimes, lifetime_adjust, leadtime_adjust, kappa,
             exclusions, techs, regions):
    """
    Calculate substitution frequencies used in the FTT models.

    The function calculates the substitution frequency on the basis of
    lifetimes, leadtimes, and a constant called "kappa". The user can plug in
    additional values to change the substitution frequency, such as adjustments
    to lifetimes (to reflect e.g. nuclear lifetime extension), adjustment to
    leadtimes (to reflect e.g. longer planning times for wind turbines), and
    exclusion matrices (to disallow certain substitutions between technologies)

    Parameters
    -----------
    lifetimes: NumPy Array
        Lifetimes for each technologies.
    leadtimes: NumPy Array
        Leadtimes for each technologies.
    lifetime_adjust: NumPy Array
        Changes to the lifetimes in the cost matrix.
    leadtime_adjust: NumPy Array
        Changes to the leadtimes in the cost matrix.
    kappa: float
        A sector specific constant.
    exclusions: NumPy Array
        The numbers herein are multiplied to the final substitution frequency
        matrix; it can be used to disallow specific interactions.
    techs: list
        List of technologies.
    regions: list
        list of regions


    Returns
    ----------
    Aij: NumPy Array
        Based on the inputs, a 3D matrix of substitution frequencies is
        returned to the main code (region x technology x technology)

    Notes
    ---------
    None.
    """

    # no. of technologies
    nt = len(techs)
    # no. of regions
    nr = len(regions)
    # matrix shape
    shape = (nt, nt)
    # Initialise Aij matrix
    Aij = np.zeros((nr, nt, nt))

    for r in range(nr):

        # Create technology by technology matices
        lt_mat = np.tile(lifetimes[r], nt).reshape(shape)
        bt_mat = np.tile(leadtimes[r], nt).reshape(shape)
        lt_mat_adj = np.tile(lifetime_adjust[r], nt).reshape(shape)
        bt_mat_adj = np.tile(leadtime_adjust[r], nt).reshape(shape)

        Aij[r] = kappa[r] / ( (lt_mat + lt_mat_adj) * (bt_mat.T + bt_mat_adj.T) ) * exclusions[r]

    return Aij