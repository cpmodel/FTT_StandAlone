# -*- coding: utf-8 -*-
"""
Creates an electricity price feedback from the power sector into other sectors. 

This was developed for the REFEREE and EEIST II projects.

"""
# Third party imports
import pandas as pd
import numpy as np

from ftt_source.support.divide import divide
from ftt_source.paths import get_utilities_path

def electricity_price_feedback(data, time_lag):
    """    
        Use MEWP from FTT-Power to grow fuel costs for electricity in each FTT model
        Calculation is done year on year based on the lag (previous year solution)
    """
    
    mewp_growth = divide(data["MEWP"], time_lag["MEWP"])
    elec_mewp_growth = mewp_growth[:, 7, np.newaxis]


    # Electricity mapping for each sector (find indices of cost matrix column and technology indices)
    elec_map_path = get_utilities_path() / 'mappings' / 'Electricity_cost_mapping.csv'
    elec_map = pd.read_csv(elec_map_path, index_col=0)
    
    # Update each fuel cost variable
    for model in elec_map.index:
        elec_index = [int(x) for x in elec_map.loc[model, "Electricity_index"].split(",")]
        cost_var = elec_map.loc[model, "Cost_var"]
        cost_index = elec_map.loc[model, "Cost index"]
        lag_cost = time_lag[cost_var][:, elec_index, cost_index]

        data[cost_var][:, elec_index, cost_index] = lag_cost * elec_mewp_growth

    return data


def electricity_price_feedback_old(data, time_lag):
    """Legacy boundary wrapper for old 3D sector data."""
    converted = dict(data)
    lag = dict(time_lag)
    converted["MEWP"] = np.asarray(data["MEWP"]).squeeze()
    lag["MEWP"] = np.asarray(time_lag["MEWP"]).squeeze()
    return electricity_price_feedback(converted, lag)

