# -*- coding: utf-8 -*-
"""
=========================================
cleafs_main.py
=========================================
Fertiliser CLEAFS module.
############################

This is the main file for the fertiliser module, CLEAFS.

Local library imports:

    CLEAFS functions:

    - `bass_model <bass_model.html>`__
        Bass diffusion model

    Support functions:

    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros

Functions included:
    - solve
        Main solution function for the module
"""

# Standard library imports
import copy

# Third party imports
import pandas as pd
import numpy as np
from numba import njit

# Local library imports
from SourceCode.support.divide import divide
from SourceCode.core_functions.ftt_sales_or_investments import get_sales
import SourceCode.Fertiliser.population_shares as pop_shares
import SourceCode.Fertiliser.bass_model as bm



# %% main function
# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lags, iter_lag, titles, histend, year, domain):
    """
    Main solution function for the module.

    Add an extended description in the future.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution
    time_lag: type
        Description
    iter_lag: type
        Description
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of histrorical data by variable
    year: int
        Curernt/active year of solution
    Domain: dictionary of lists
        Pairs variables to domains

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    Notes
    ---------
    """
    # Categories for the cost matrix (BCET)
    cfti = {category: index for index, category in enumerate(titles['CFTI'])}


    # data = npv_calc.npv_calculation(data, titles)
    # data = npv_calc.potential_population(data, titles)

    green_tech = 'Green fertiliser'
    grey_tech = 'Grey fertiliser'
    sim_var = 'FERTD'
    green_idx = titles['TFTI'].index(green_tech)
    grey_idx = titles['TFTI'].index(grey_tech)


    # Calculate growth in agricultural sector
    agri_growth = data['HYD1'] / time_lags['HYD1']
    # Replace NaNs
    agri_growth[np.isnan(agri_growth)] = 1
    # Project fertiliser demand
    proj_fert_demand = time_lags['FERTD'].sum(axis = 1) * agri_growth[:, 0, :]
    
    total_fert_demand = proj_fert_demand.sum(axis = 1)
    # Project fertiliser demand using agriculture growth
    if histend['FERTD'] > year:
        # Fill fertilsier demand where it is 0
        data['FERTD'][data['FERTD'] == 0] = proj_fert_demand[data['FERTD'][:,0, :] == 0]
    # Project maximum potential fertiliser demand similary
    # if histend['MFERTD'] < year:
    #     proj_mfert_demand = time_lags['MFERTD'] * agri_growth
    #     data['MFERTD'] = proj_mfert_demand

    if year > histend['FERTD']:
        data = pop_shares.green_population_share(data, time_lags, titles)
        data = bm.simulate_bass_diffusion(data, time_lags, titles, histend, green_tech, sim_var, total_fert_demand)
        # in N-equivalent fertiliser kt
        data['FERTD'][:, grey_idx, :] = proj_fert_demand - data['FERTD'][:, green_idx, :]

    return data
