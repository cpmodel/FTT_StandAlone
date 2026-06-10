# -*- coding: utf-8 -*-
"""
============================================================
ftt_gmp_main.py
============================================================
Green Molecules Project FTT module.

This is the main file for FTT-Green Molecules.


Functions included:
    - solve
        Main solution function for the module

"""

# Third party imports
import numpy as np

# Local library imports
from SourceCode.ftt_core.ftt_sales_or_investments import get_sales
from SourceCode.ftt_core.ftt_shares import shares_change
from SourceCode.ftt_core.ftt_mandate import implement_seeding, implement_mandate
from SourceCode.ftt_core.ftt_exogenous_sales import exogenous_sales
from SourceCode.ftt_core.ftt_exogenous_capacity import regulation_correction

from SourceCode.support.divide import divide
from SourceCode.support.check_market_shares import check_market_shares


# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, titles, histend, year, domain):
    """
    Main solution function for the GMP module.

    This function simulates investor decision making for the production of 
    green molecules and the subsequent production of electricity for grid 
    balancing. 
    
    Levelised costs (from the get_gm_lc function) are taken and market shares
    for each pathway are simulated to ensure demand (curtailment) is met.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for given year of solution
    time_lag: type
        Model variables from the previous year
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of histrorical data by variable
    year: int
        Current year

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    """
    print("GMP module called.")
    pass