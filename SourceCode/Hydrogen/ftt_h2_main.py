# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:30 2024

@author: PV

=========================================
ftt_h2_main.py
=========================================
Hydrogen FTT module.
####################################

This is the main file for FTT: Hydrogen, which models technological
diffusion of hydrogen production technologies due to simulated investment decision making.
Producers compare the **levelised cost of hydrogen**, which leads to changes in the
market shares of different technologies.


Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros
    - `estimation <econometrics_functions.html>`__
        Predict future values according to the estimated coefficients.

Functions included:
    - solve
        Main solution function for the module
    - get_lcoh2
        Calculate levelised cost of hydrogen

"""
# Standard library imports
from math import sqrt
import copy
import warnings

# Third party imports
import numpy as np

# Local library imports
from SourceCode.support.divide import divide
from SourceCode.Hydrogen.ftt_h2_lcoh import get_lcoh as get_lcoh2
from SourceCode.Hydrogen.cost_supply_curve import calc_csc
from SourceCode.Hydrogen.ftt_h2_pooledtrade import pooled_trade
from SourceCode.core_functions.substitution_frequencies import sub_freq
from SourceCode.core_functions.substitution_dynamics_in_shares import substitution_in_shares
# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, iter_lag, titles, histend, year, domain):
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
    specs: dictionary of NumPy arrays
        Function specifications for each region and module

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    Notes
    ---------
    This function should be broken up into more elements in development.
    """

     # Categories for the cost matrix (BHTC)
    c7ti = {category: index for index, category in enumerate(titles['C7TI'])}
    jti = {category: index for index, category in enumerate(titles['JTI'])}
    hyti = {category: index for index, category in enumerate(titles['HYTI'])}

    fuelvars = ['FR_1', 'FR_2', 'FR_3', 'FR_4', 'FR_5', 'FR_6',
                'FR_7', 'FR_8', 'FR_9', 'FR_10', 'FR_11', 'FR_12']

    sector = 'hydrogen'
    #sector_index = titles['Sectors_short'].index(sector)
    print("FTT: Hydrogen under construction")

    # Estimate substitution frequencies
    data['HYWA'] = sub_freq(data['BCHY'][:, :, c7ti['Lifetime']],
                            data['BCHY'][:, :, c7ti['Buildtime']],
                            data['BCHY'][:, :, c7ti['Lifetime']]*0,
                            data['BCHY'][:, :, c7ti['Buildtime']]*0,
                            np.ones(len(titles['RTI']))*75,
                            np.ones([len(titles['RTI']), len(titles['HYTI']), len(titles['HYTI'])]),
                            titles['HYTI'], titles['RTI'])



    # %% Historical accounting
    if year < 2023:

        # data['HYWK'] = divide(data['HYG1'], data['BCHY'][:, :, c7ti['Capacity factor'], np.newaxis])
        data['HYWS'] = data['HYWK'] / data['HYWK'].sum(axis=1)[:, :, None]
        data['HYWS'][np.isnan(data['HYWS'])] = 0.0
        data['HYPD'][:, 0, 0] = data['HYG1'][:, :, 0].sum(axis=1)
        data = get_lcoh2(data, titles)
        
        # Total capacity
        data['HYKF'][:, 0, 0] = data['HYWK'][:, :, 0].sum(axis=1)
        
        # Total hydrogen demand
        data['HYDT'][:, 0, 0] = (data['HYD1'][:, 0, 0] +
                                 data['HYD2'][:, 0, 0] +
                                 data['HYD3'][:, 0, 0] +
                                 data['HYD4'][:, 0, 0] +
                                 data['HYD5'][:, 0, 0] ) / 1.6
        
        # System-wide capacity factor
        data['HYSC'][:, 0, 0] = divide(data['HYG1'][:, :, 0].sum(axis=1),
                                       data['HYWK'][:, :, 0].sum(axis=1))
        
        # Calculate the average lifetime across all technologies in each region
        # This determines the rate of decline when new capacity seems underutilised
        average_lifetime = data['BCHY'][:, :, c7ti['Lifetime']] * divide(data['HYWK'][:, :, 0],
                                                                         data['HYWK'][:, :, :].sum(axis=1))
        average_lifetime = average_lifetime.sum(axis=1)
        
        # Estimate capacity growth.
        # Assume a maximum of 15% growth when the system is operating near
        # it's full potential and assume total capacity decreases when the
        # opposite occurs when capacity factors drop below 65%  
        data['HYCG'][:,0,0] = np.tanh(1.25*(data['HYSC'][:, 0, 0] - 0.65)/0.1)
        
        data['HYCG'][:,0,0] = np.where( data['HYCG'][:,0,0] < 0.0,
                                       divide(data['HYCG'][:,0,0], average_lifetime),
                                       data['HYCG'][:,0,0] * 0.15)
        

    # %% Simulation period
    else:

        # Total hydrogen demand
        data['HYDT'][:, 0, 0] = (data['HYD1'][:, 0, 0] +
                                 data['HYD2'][:, 0, 0] +
                                 data['HYD3'][:, 0, 0] +
                                 data['HYD4'][:, 0, 0] +
                                 data['HYD5'][:, 0, 0] )
        
        dem_lag = (time_lag['HYD1'][:, 0, 0] +
                                 time_lag['HYD2'][:, 0, 0] +
                                 time_lag['HYD3'][:, 0, 0] +
                                 time_lag['HYD4'][:, 0, 0] +
                                 time_lag['HYD5'][:, 0, 0] )

        data_dt = {}

        # First, fill the time loop variables with the their lagged equivalents
        for var in time_lag.keys():

            if domain[var] == 'FTT-H2':

                data_dt[var] = copy.deepcopy(time_lag[var])

        data_dt['HYIY'] = np.zeros([len(titles['RTI']), len(titles['HYTI']), 1])

        # Factor used to create quarterly data from annual figures
        no_it = int(data['noit'][0, 0, 0])
        dt = 1 / float(no_it)

        # Project hydrogen production
        data['HYKF'][:, 0, 0] = time_lag['HYKF'][:, 0, 0] * (1 + time_lag['HYCG'][:, 0, 0])
        # Remove capacity additions in the medium-term outlook
        data['HYKF'][:, 0, 0] -= data['HYMT'][:, :, 0].sum(axis=1)
        
        # Split the market into a grey and green market
        # From CLEAFS
        
        # ALlocate permissible technologies to the green market
        
        # Remove green technologies in the green market from the overall market
        # This becomes the grey market

        # =====================================================================
        # Start of the quarterly time-loop
        # =====================================================================

        #Start the computation of shares
        for t in range(1, no_it+1):
            
            # Expected capacity expansion for the green market
            
            # Expected capacity expension for the grey market
            HYKFt = data_dt['HYKF'][:, 0, 0] + (data['HYKF'][:, 0, 0] - data_dt['HYKF'][:, 0, 0]) * t/no_it

            for r in range(len(titles['RTI'])):

                if np.isclose(HYKFt[r], 0.0):
                    continue
                
                dSij = substitution_in_shares(data_dt['HYWS'], data['HYWA'], 
                                              data_dt['HYLC'], data_dt['HYLD'], 
                                              r, dt, titles)


                #calculate temporary market shares and temporary capacity from endogenous results
                data['HYWS'][r, :, 0] = data_dt['HYWS'][r, :, 0] + np.sum(dSij, axis=1)
                data['HYWK'][r, :, 0] = data['HYWS'][r, :, 0] * HYKFt[r, np.newaxis]
                
                # Add medium-term capacity additions and rescale shares
                if data['HYWK'][r, :, 0].sum() > 0.0:
                    data['HYWK'][r, :, 0] += data['HYMT'][r, :, 0]
                    data['HYWS'][r, :, 0] = data['HYWK'][r, :, 0] / data['HYWK'][r, :, 0].sum()
                    
            # Call CSC function
            data['HYG1'][:, :, 0], data['HYEP'][0,0,0], data['HYCF'][:, :, 0] = calc_csc(data_dt['HYCC'], data_dt['HYLD'], data['HYDT'], data['HYWK'], data_dt['HYCF'], data['HYTC'], titles)

            data['HYWK'][np.isnan(data['HYWK'])] = 0.0
            data['HYG1'][np.isnan(data['HYWK'])] = 0.0
            
            # Capacity additions
            cap_diff = data['HYWK'][r, :, 0] - time_lag['HYWK'][r, :, 0]
            cap_drpctn = time_lag['HYWK'][r, :, 0] / time_lag['BCHY'][r, :, c7ti['Lifetime']]
            data['HYWI'][r, :, 0] = np.where(cap_diff > 0.0,
                                             cap_diff + cap_drpctn,
                                             cap_drpctn)
            
            # Spillover learning
            # Using a technological spill-over matrix (HYWB) together with capacity
            # additions (MEWI) we can estimate total global spillover of similar
            # techicals
            bi = np.zeros((len(titles['RTI']),len(titles['HYTI'])))
            hywi0 = np.sum(data['HYWI'][:, :, 0], axis=0)
            dw = np.zeros(len(titles["HYTI"]))
            
            for i in range(len(titles["HYTI"])):
                dw_temp = copy.deepcopy(hywi0)
                dw_temp[dw_temp > dw_temp[i]] = dw_temp[i]
                dw[i] = np.dot(dw_temp, data['HYWB'][0, i, :])

            # Cumulative capacity incl. learning spill-over effects
            data["HYWW"][0, :, 0] = time_lag['HYWW'][0, :, 0] + dw
            
 
            # Learning-by-doing effects on investment
            for tech in range(len(titles['HYTI'])):

                if data['HYWW'][0, tech, 0] > 0.1:

                    data['BCHY'][:, tech, c7ti['CAPEX, mean, €/tH2 cap']] = data_dt['BCHY'][:, tech, c7ti['CAPEX, mean, €/tH2 cap']] * \
                        (1.0 + data['BCHY'][:, tech, c7ti['Learning rate']] * dw[tech]/data['HYWW'][0, tech, 0])

        # System-wide capacity factor
        data['HYSC'][:, 0, 0] = divide(data['HYG1'][:, :, 0].sum(axis=1),
                                       data['HYWK'][:, :, 0].sum(axis=1))

        # Calculate the average lifetime across all technologies in each region
        # This determines the rate of decline when new capacity seems underutilised
        average_lifetime = data['BCHY'][:, :, c7ti['Lifetime']] * divide(data['HYWK'][:, :, 0],
                                                                         data['HYWK'][:, :, :].sum(axis=1))
        average_lifetime = average_lifetime.sum(axis=1)
        
        # Estimate capacity growth.
        # Assume a maximum of 15% growth when the system is operating near
        # it's full potential and assume total capacity decreases when the
        # opposite occurs when capacity factors drop below 65%  
        data['HYCG'][:,0,0] = np.tanh(1.25*(data['HYSC'][:, 0, 0] - 0.65)/0.1)
        
        data['HYCG'][:,0,0] = np.where( data['HYCG'][:,0,0] < 0.0,
                                       divide(data['HYCG'][:,0,0], average_lifetime),
                                       data['HYCG'][:,0,0] * 0.15)
                    
        # Call LCOH2 function
        data = get_lcoh2(data, titles)

        # Update lags
        for var in data_dt.keys():

            if domain[var] == 'FTT-H2':

                data_dt[var] = np.copy(data[var])            
        



    return data
