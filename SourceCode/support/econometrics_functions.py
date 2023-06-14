# -*- coding: utf-8 -*-
"""
econometrics_functions.py
=========================================
Functions to read estimated coefficients,
names of the variable needed for
econometrics, and solve the econometrics
specification.

Functions included:
    - load_coefficients
        Load estimated coefficients and variable names.
    - estimation
        Solve the econometric specification.
"""

#Standard library import
import os
import copy
import re

#Third party import
import pandas as pd
import numpy as np

# Local library imports
from support.divide import divide

#%% Read coefficients and variable names

def load_coefficients(converter):

    """
    Read estimated coefficients from excel spreadsheet, and names of the variables
    needed for the econometric specification.

    Parameters
    -----------
    converter: dictionary of dataframes.
        Spreadsheet used to convert individual fuels into
        broad fuels categories.

    Returns
    -----------
    coeff_dict: dictionary of dataframes
        Dictionary of estimated coefficients.
    exog_var: Dataframe
        Dataframe of variables needed for the econometric
        specification and relative lags.
    """

#----------------------Read coefficients-----------------------------

    #Declare path to the spreadsheet with the estimated coefficients.
    coeff_path = os.path.join('parameter_estimation\output', 'coefficients.xlsx')

    #Read the spreadsheet into a dictionary and delete the first two tabs.
    coeff_dict = pd.read_excel(coeff_path, sheet_name = None, header= None)
    del coeff_dict['Specifications']
    # del coeff_dict['Cover'], coeff_dict['Specifications']
    #Organise coefficients for each module. A multiindex is applied to specify
    #whether the coefficients belong to the short-term or long-term equation.
    for key in coeff_dict:

        coeff_dict[key].loc[5,:] = coeff_dict[key].loc[5,:].fillna(method='ffill')
        coeff_dict[key] = coeff_dict[key].set_index(0)
        coeff_dict[key] = coeff_dict[key].iloc[5:,:]
        coeff_dict[key].columns = pd.MultiIndex.from_arrays([coeff_dict[key].iloc[0].values,
                                                            coeff_dict[key].iloc[1].values])
        coeff_dict[key] = coeff_dict[key].iloc[2:,:]
        coeff_dict[key] = coeff_dict[key].reset_index(drop = True)
        coeff_dict[key] = coeff_dict[key].fillna(0)



#--------------------Read variables' name-----------------------------

    #Declare path to the spreadsheet with the  variable names for all sectors
    var_path = os.path.join('parameter_estimation\\assumptions', 'variables.csv')

    #Read spreadsheet into a dataframe
    exog_var = pd.read_csv(var_path, header = 0, usecols = ['Variable', 'Lags'], index_col = ['Variable'])

    #Append detailed fuels with related lags number to the exogenous variables
    exog_var = exog_var.to_dict()
    conv = copy.deepcopy(converter)
    conv['broad_fuels'] = conv['broad_fuels'].replace(exog_var['Lags'])
    conv['broad_fuels'] = conv['broad_fuels'].rename(columns = {'broad_fuel':'Lags'})
    exog_var = pd.DataFrame(exog_var)
    exog_var = exog_var.append(conv['broad_fuels'])



    return coeff_dict, exog_var


#%% Solve econometric specification


def estimation(data, data_lag, region, module, fuel, coefficients, sector_index, converter):

    """
    Predict future values according to the estimated coefficients.

    Parameters
    ---------
    data: dictionary of numpy arrays
        Dictionary containing the cross section of data.
    data_lag: dictionary of numpy arrays
        Dictionary containing the lags of cross section of data
    region: int
        The index of the region
    module: string
        The name of the sector
    fuel: int
        The number of the fuel
    coefficients: dictionary of dataframes
        Coefficients to use to forecast future values
    sector_index: int
        The index of the sector
    converter: dictionary of dataframes
        Used to go from detailed to broad fuels

    Returns
    ----------
    new_value: float
        The forecast value for the next period.
    """

    #Strin that says to which broad fuel the detailed fuel belongs to.
    #TODO: maybe key and column name too hardcoded?
    fuel_group = converter['broad_fuels'].loc['FR_{}'.format(fuel), 'broad_fuel']

    #Select coefficients based on fuel group, and separate into long and short term coefficients
    coeff = { time: coefficients['{}_{}'.format(module,fuel_group)].loc[ :, time] for time in ['LR', 'SR'] }


    #Create container for the transformed variables that will go into the equations.
    data_eq = { time: pd.Series(index = coeff[time].columns, dtype = 'float64') for time in coeff}

#   Loop through short and long run
    for time in coeff:

        #Create containeer of the data to fetch
        data_all = {}
        adj_sector_index = sector_index

        #If the equation is the long-run one, take the lags. This is needed since the ECT is estimated
        #with one lag.
        if time == 'LR':

            for i in data_lag:

                data_all[i-1] = copy.deepcopy(data_lag[i])

        #If the equation is the short-run one, take both current values and the lags.
        else:

            #Current value
            data_all[0] = copy.deepcopy(data)


            #Lags
            for i in data_lag:

                data_all[i] = copy.deepcopy(data_lag[i])


        #Loop through the variables in the respective equation, in order to fetch the data.
        for i in coeff[time].columns:

            #Name of the econometric variable split in two. The first part tells if the variable is lagged, logged, differenced
            #or combinations. The second part is the name of the variable to fetch in the data.
            var_split = i.split('_',1)
            

            #If the variable is one of the four categories of fuel, then the variable to fetch is
            #one of the underlying detailed fuels.
            if len(var_split) > 1 and var_split[1] in list(converter['broad_fuels']['broad_fuel']):

                var_split[1] = 'FR_{}'.format(fuel)

            #Assign one to the intercept
            if var_split[0] == 'C':

                data_eq[time][i] = 1

            #Log
            elif var_split[0] == 'L':
                if data_all[0][var_split[1]].shape[1] == 1: 
                    adj_sector_index = 0

                a = data_all[0][var_split[1]][region, adj_sector_index, :]

                if a > 0.0:
                    data_eq[time][i] = np.log(a)

#                data_eq[time][i] = np.log(data_all[0][var_split[1]][region, sector_index, :])

            #Log difference
            elif var_split[0] == 'DL':

                if data_all[0][var_split[1]].shape[1] == 1: 
                    adj_sector_index = 0


                a = data_all[0][var_split[1]][region, adj_sector_index, :]
                b = data_all[1][var_split[1]][region, adj_sector_index, :]
                div = divide(a, b)

                if div > 0.0:
                    data_eq[time][i] = np.log(div)

#                data_eq[time][i] = (np.log(data_all[0][var_split[1]][region, sector_index, :]) -
#                                    np.log(data_all[1][var_split[1]][region, sector_index, :]))

            #Lagged log
            elif var_split[0][0] == 'L' and len(var_split[0]) >= 2:

                lag_nr = int(var_split[0][1:])
                
                if data_all[lag_nr][var_split[1]].shape[1] == 1: 
                    adj_sector_index = 0              

                a = data_all[lag_nr][var_split[1]][region, adj_sector_index, :]

                if a>0.0:
                    data_eq[time][i] = np.log(a)

#                data_eq[time][i] = np.log(data_all[lag_nr][var_split[1]][region, sector_index, :])

            #Lagged logged difference
            elif 'DL' in var_split[0] and len(var_split[0]) >= 3:

                lag_nr = int(var_split[0][2:])

                if  data_all[lag_nr][var_split[1]].shape[1] == 1: 
                    adj_sector_index = 0
                    
                a = data_all[lag_nr][var_split[1]][region, adj_sector_index, :]
                b = data_all[lag_nr + 1][var_split[1]][region, adj_sector_index, :]
                div = divide(a, b)

                if div>0.0:
                    data_eq[time][i] = np.log(div)

#                data_eq[time][i] = (np.log(data_all[lag_nr][var_split[1]][region, sector_index, :]) -
#                                   np.log(data_all[lag_nr + 1][var_split[1]][region, sector_index, :]))


        #Compute ECT from the long-run equation (in t-1)
        if time == 'LR' and np.all(coeff[time].loc[region, :]):

            if  data_all[0]['FR_{}'.format(fuel)].shape[1] == 1: 
                    adj_sector_index = 0
                    
            a = data_all[0]['FR_{}'.format(fuel)][region, adj_sector_index, :]

            if a>0.0:
                data_eq['SR']['ECT'] = np.log(a) - data_eq['LR'].multiply(coeff[time].loc[region, :]).sum()

#            data_eq['SR']['ECT'] = np.log(data_all[0]['FR_{}'.format(fuel)][region, sector_index, :]) - \
#                                      data_eq['LR'].multiply(coeff[time].loc[region, :]).sum()



    #Forecast he growth rate with short term equation, and apply that to dependent variable in t-1
    #The growth is exponentiated since it is expresed as a log difference.

    for_growth = data_eq['SR'].multiply(coeff['SR'].loc[region, :]).sum()

    adj_sector_index = sector_index
    if  data_all[1]['FR_{}'.format(fuel)].shape[1] == 1: 
        adj_sector_index = 1
        
    new_value = data_all[1]['FR_{}'.format(fuel)][region, adj_sector_index, :]*np.exp(for_growth)


    return new_value












