"""
Construct a single-year cross-section of model input data.

Functions included:
    - cross_section
        Select one simulation year from the data dictionary
"""

import numpy as np


def cross_section(data_in, dimensions, year, y, scenario):
    """
    Construct the input dictionary for one model year.

    Time-dependent variables are selected at ``y``; variables without a time
    classification are copied in full.
    """
    data_out = {}
    # Loop through all variables in the selected scenario.
    for variable, values in data_in[scenario].items():
        if "TIME" in dimensions[variable]:
            time_axis = dimensions[variable].index("TIME")
            data_out[variable] = np.take(values, y, axis=time_axis)
        else:
            data_out[variable] = values.copy()
    return data_out
