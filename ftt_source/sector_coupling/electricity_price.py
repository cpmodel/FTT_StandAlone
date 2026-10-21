"""
Create electricity-price feedback from Power into the demand sectors.

This feedback was developed for the REFEREE and EEIST II projects.
"""

import numpy as np
import pandas as pd

from ftt_source.paths import get_utilities_path
from ftt_source.support.divide import divide


def electricity_price_feedback(data, time_lag):
    """
    Use ``MEWP`` from Power to update electricity fuel costs in each sector.

    The calculation is based on the previous year's price and cost values.
    """
    price_growth = divide(data["MEWP"], time_lag["MEWP"])
    electricity_growth = price_growth[:, 7, np.newaxis]

    # Electricity mapping for each sector: cost variable, column, and fuels.
    mapping_path = get_utilities_path() / "mappings" / "Electricity_cost_mapping.csv"
    mappings = pd.read_csv(mapping_path, index_col=0)

    # Update each fuel cost variable.
    for model in mappings.index:
        electricity_indices = [
            int(index)
            for index in mappings.loc[model, "Electricity_index"].split(",")
        ]
        cost_variable = mappings.loc[model, "Cost_var"]
        cost_index = mappings.loc[model, "Cost index"]
        lag_cost = time_lag[cost_variable][:, electricity_indices, cost_index]
        data[cost_variable][:, electricity_indices, cost_index] = (
            lag_cost * electricity_growth
        )
    return data
