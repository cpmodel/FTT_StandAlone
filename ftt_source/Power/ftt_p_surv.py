# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_surv.py
=========================================
Power generation survival FTT module.
#####################################

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros

Functions included:
    - survival_function
        Calculate survival of technology

"""


# Third party imports
import numpy as np
from numba import njit


# %% survival function
# -----------------------------------------------------------------------------
# -------------------------- Survival calcultion ------------------------------
# -----------------------------------------------------------------------------
def survival_function(data, time_lag, histend, year, titles, c2ti):

    """
    In this function scrappage, sales, tracking of age, and average efficiency
    are calculated.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the current year
    time_lag: dictionary
        time_lag is the same, but with data for last year.
    histend: dict of integers
        Final year of histrorical data by variable
    year: int
        Curernt/active year of solution
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.
    c2ti: dictionary
        The names of the elements of the cost matrix

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the current year
        The values are updated and returned to the main routine.

    Notes
    ---------
    This function is currently unused.
    """
    
    # TODO: This is a generic survival function
    HalfLife = data['BCET'][:, :, c2ti['9 Lifetime (years)']]/2
    dLifeT = HalfLife/10

    for age in range(len(titles['TYTI'])):

        age_matrix = np.ones_like(data['MSRV'][:, :, age]) * age

        data['MSRV'][:, :, age] = 1.0 - 0.5*(1+np.tanh(1.25*(HalfLife-age_matrix)/dLifeT))

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
                data['MEKA'][r, t1, :-1] = time_lag['MEKA'][r, t1, 1:]

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
