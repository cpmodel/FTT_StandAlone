# -*- coding: utf-8 -*-
"""
power_generation.py
=========================================
Power generation FTT module.

Functions included:
    - get_lcoe
        Calculate levelized costs
    - solve
        Main solution function for the module
"""

# Standard library imports
from math import sqrt
import os
import copy
import sys
import warnings

# Third party imports
import pandas as pd
import numpy as np
from numba import njit

# Local library imports
from support.divide import divide


# %% survival function
# -----------------------------------------------------------------------------
# -------------------------- Survival calcultion ------------------------------
# -----------------------------------------------------------------------------
def survival_function(data, time_lag, histend, year, titles):
    # In this function we calculate scrappage, sales, tracking of age, and
    # average efficiency.

    # Create a generic matrix of fleet-stock by age
    # Assume uniform distribution, but only do so when we still have historical
    # market share data. Afterwards it becomes endogeous
    if year < histend['MEWG']:

        # TODO: This needs to be replaced with actual data
        for age in range(len(titles['TYTI'])):

            data['MEKA'][:, :, age] = data['MSRV'][:, :, age] * data['MEWK'][:, :, 0]

    else:
        # Once we start to calculate the market shares and total fleet sizes
        # endogenously, we can update the techicle stock by age matrix and
        # calculate scrappage, sales, average age, and average efficiency.
        for r in range(len(titles['RTI'])):

            for t1 in range(len(titles['T2TI'])):
                # Move all t1icles one year up:
                # New sales will get added to the age-tracking matrix in the main
                # routine.
                data['MEKA'][r, t1, :-1] = copy.deepcopy(time_lag['MEKA'][r, t1, 1:])

                # Current age-tracking matrix:
                # Only retain the fleet that survives
                data['MEKA'][r, t1, :] = data['MEKA'][r, t1, :] * data['MSRV'][r, 0, :]

                # Total amount of t1icles MEWKA survive:
                survival = np.sum(data['MEKA'][r, t1, :])

                # EoL scrappage: previous year's stock minus what survived
                if time_lag['MEWK'][r, t1, 0] > survival:

                    data['MEOL'][r, t1, 0] = time_lag['MEWK'][r, t1, 0] - survival

    # calculate fleet size
    return data
