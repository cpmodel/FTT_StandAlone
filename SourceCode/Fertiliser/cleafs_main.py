# -*- coding: utf-8 -*-
"""
=========================================
cleafs_main.py
=========================================
Fertiliser CLEAFS module.
############################

This is the main file for the power module, FTT: Power. The power
module models technological replacement of electricity generation technologies due
to simulated investor decision making. Investors compare the **levelised cost of
electricity**, which leads to changes in the market shares of different technologies.

After market shares are determined, the rldc function is called, which calculates
**residual load duration curves**. This function estimates how much power needs to be
supplied by flexible or baseload technologies to meet electricity demand at all times.
This function also returns load band heights, curtailment, and storage information,
including storage costs and marginal costs for wind and solar.

FTT: Power also includes **dispatchers decisions**; dispatchers decide when different technologies
supply the power grid. Investor decisions and dispatcher decisions are matched up, which is an
example of a stable marraige problem.

Costs in the model change due to endogenous learning curves, costs for electricity
storage, as well as increasing marginal costs of resources calculated using cost-supply
curves. **Cost-supply curves** are recalculated at the end of the routine.

Local library imports:

    FTT: Power functions:

    - `rldc <ftt_p_rldc.html>`__
        Residual load duration curves
    - `dspch <ftt_p_dspch.html>`__
        Dispatch of capcity
    - `get_lcoe <ftt_p_lcoe.html>`__
        Levelised cost calculation
    - `survival_function <ftt_p_surv.html>`__
        Calculation of scrappage, sales, tracking of age, and average efficiency.
    - `shares <ftt_p_shares.html>`__
        Market shares simulation (core of the model)
    - `cost_curves <ftt_p_costc.html>`__
        Calculates increasing marginal costs of resources

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
import SourceCode.Fertiliser.npv_calculation as npv_calc
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
    agri_growth = data['AQR'] / time_lags['AQR']
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
        data = bm.simulate_bass_diffusion(data, time_lags, titles, histend, green_tech, sim_var, total_fert_demand)
        data['FERTD'][:, grey_idx, :] = proj_fert_demand - data['FERTD'][:, green_idx, :]

    return data
