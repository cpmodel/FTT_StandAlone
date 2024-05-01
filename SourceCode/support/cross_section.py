# -*- coding: utf-8 -*-
"""
=========================================
cross_section.py
=========================================
Function to construct cross section of data for given year.

Functions included in the file:
    - cross_section
        Cross-sectional data slicer, selects a single year of the data dictionary.
"""

# Standard library imports
import copy

# Third party imports
import numpy as np


def cross_section(data_in, dimensions, year, y, scenario, econometrics=None, lag=None, lag_sales=None):
    """ Construct cross section of data for given year.

    Parameters
    ----------
    data_in: dictionary of numpy arrays
        Data to take the cross-section from
    dimensions: dictionary of tuples
        Dictionary with the dimensions of all variables
    year: int
        The index of the year
    scenario: string
        The name of the scenario
    econometrics: dataframe
        Optional argument, variable names for the econometrics
    lag: int
        Optional argument, the lag to take the cross-section of
    lag_sales: dictionary of dataframes
        Optional argument, target country and number of lags
        for the lagged sales specification.

    Returns
    -----------
    data_out: dictionary of numpy arrays
        Dictionary of numpy array for a single given year


    """


    data_out = {}

    # Getting cross-section for all variable or only for the variables needed for the econometrics.
    if lag_sales is None:

        if econometrics is None:

            # Loop through all variables
            for var in data_in[scenario]:
                # If the variable has a time dimension take only the one year
                if dimensions[var][3] == 'TIME':
                    data_out[var] = copy.deepcopy(data_in[scenario][var][:, :, :, y])

                # If the variable does not have a time dimension take all
                else:
                    data_out[var] = copy.deepcopy(data_in[scenario][var][:, :, :, 0])

        # If a sheet with the econometric variables' name is fed, then fetch data only for those variables.
        else:

            # Loop through the variable needed for the econometrics
            for var in econometrics.index:

                # If the variable is present in the input data (to avoid an error for constant, dummies etc...)
                if var in data_in[scenario]:

                    # If the variable has a lag corresponding to the lag fed, take the values from that lag.
                    if lag in range(econometrics.loc[var, 'Lags'] + 1):

                        data_out[var] = copy.deepcopy(data_in[scenario][var][:, :, :, y-lag])

#                    # Otherwise take the year fed (should be the first one)
                    elif lag is None:

                        data_out[var] = copy.deepcopy(data_in[scenario][var][:, :, :, y])


    # Getting lagged cross-section for variables needed for the lagged sales specification, which are specified in the dictionary
    # fed into the positional argument lag_sales
    # TODO: add a warning saying that the input must be the correct one.
    # If the dataframe with the lagged sales target country and number of lags is fed
    else:

        # Loop through sales variables
        for var in lag_sales:

            # Create container of data
            data_out[var] = np.zeros_like(data_in[scenario][var][:,:,:,y])

            # Loop through each region
            for r in lag_sales[var].index:

                # Get the number of lags associated to the regions
                lag_year = lag_sales[var].loc[r, 'lag']
                # Get the inddex of the country the region should imitate
                target_reg = lag_sales[var].loc[r, 'target_ctry']

                # If there aren't enough years to take the desired lags, take the first year available
                if year - lag_year <= 0:

                    data_out[var][r, :, :] = copy.deepcopy(data_in[scenario][var][target_reg, :, :, 0])

                # Otherwise take the desired year.
                else:

                    data_out[var][r, :, :] = copy.deepcopy(data_in[scenario][var][target_reg, :, :, y-lag_year])



    return data_out
