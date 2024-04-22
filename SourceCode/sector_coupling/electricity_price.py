# -*- coding: utf-8 -*-
"""
Creates an electricity price feedback from the power sector into other sectors. 

This was developed for the REFEREE and EEIST II projects.

"""
# Standard library imports
import os

# Third party imports
import pandas as pd
import numpy as np
import copy


def electricity_price_feedback(data, time_lag):
    """ "This function takes as input the electricity price from FTT-P, and 
    returns the relative price of electricity at year x"""
    
    
    """
        Use MEWP from FTT-Power to grow fuel costs for electricity in each FTT model
        Calculation is done year on year based on the lag (previous year solution)
    """
    variables = copy.deepcopy(data)
    mewp_growth = data["MEWP"] / time_lag["MEWP"]
    elec_mewp_growth = mewp_growth[:, 7, 0][np.newaxis].T

    # Update each fuel cost variable

    # Electricity mapping for each model
    elec_map = pd.read_csv(os.path.join('Utilities', 'mappings', "Electricity_cost_mapping.csv"),
                           index_col=0)
    for model in elec_map.index:
        elec_index = [int(x) for x in elec_map.loc[model, "Electricity_index"].split(",")]
        cost_var = elec_map.loc[model, "Cost_var"]
        cost_index = elec_map.loc[model, "Cost index"]
        lag_cost = time_lag[cost_var][:, elec_index, cost_index]

        variables[cost_var][:, elec_index, cost_index] = lag_cost * elec_mewp_growth

    return variables

pass

def electricity_demand_price_elasticity(data, titles, histend, year, ftt_modules):
    """
    Compute electricity demand changes using the
    econometrically estimated elasticity. This is found in the X databank 
    under the BFRE estimated parameters. 
    
    The equation in E3ME is in COINT. I think it 
    """
    pass

