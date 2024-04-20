# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_revu.py
=========================================
Power revenue per unit FTT module.

Functions included:
    - revu
        Calculate revenue per unit
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

# Local library imports
from SourceCode.support.divide import divide
from SourceCode.Power.new.ftt_p_wacc import get_wacc
from SourceCode.Power.new.ftt_p_vf import get_vf

# -----------------------------------------------------------------------------
# --------------------------- revu function -----------------------------------
# -----------------------------------------------------------------------------


def get_revu(data,titles,year):
    """  
    Calculate levelized revenue per unit.

    Calculate levelized revenue per unit based on value factors and average electricity prices,
    discounted using WACC.

    Parameters
    -----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) for all
        variables. Variable names are keys and the values are 3D NumPy arrays.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.

    Returns
    ----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.

    Notes
    ---------
    Additional notes if required.
    """
    #loop over countries and technologies
    for r in range(len(titles['RTI'])):
        for tech in range(len(titles['T2TI'])):
        
            vf = get_vf(data, titles, year)
            price = data['DAEP'][r,0,0]
    
            undiscounted_revu = vf*price
            
            # Net present value calculations
            # Discount rate
            wacc = get_wacc(data, titles,year,histend)
            lifetime = data['MELF'][r,tech,0]    #lifetime is dependent on country
     
            denominator = (1+wacc)**lifetime
            # 
#           denominator of discounting (money value) from lcoe module
                npv_utility = (et)/denominator #et from lcoe module
            
#           discounting numerator            
            disc_revu_num = undiscounted_revu/denominator
            discounted_rev = disc_revu_num/npv_utility
            
            data['DRPU'][r, tech, 0] = discounted_rev


return data