# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:30 2024

@author: AE & CL

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
from SourceCode.Hydrogen.ftt_h2_csc import get_csc
from SourceCode.core_functions.substitution_frequencies import sub_freq
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

    # Hard-coded exclusions
    data['HYWA'][:, -1, :] = 0.0
    data['HYWA'][:, :, -1] = 0.0


    # %% Historical accounting
    if year < 2023:

        data['HYWK'] = divide(data['HYG1'], data['BCHY'][:, :, c7ti['Capacity factor'], np.newaxis])
        data['HYWS'] = data['HYWK'] / data['HYWK'].sum(axis=1)[:, :, None]
        data['HYWS'][np.isnan(data['HYWS'])] = 0.0
        data['HYWS'][np.isnan(data['HYWS'])] = 0.0
        data['HYPD'][:, 0, 0] = data['HYG1'][:, :, 0].sum(axis=1)
        data = get_lcoh2(data, titles)

    # %% Simulation period
    else:

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
        data['HYPD'][:, 0, 0] = time_lag['HYG1'][:, :, 0].sum(axis=1) * 1.02

        # =====================================================================
        # Start of the quarterly time-loop
        # =====================================================================

        #Start the computation of shares
        for t in range(1, no_it+1):

            HYPDt = data_dt['HYPD'][:, 0, 0] + (data['HYPD'][:, 0, 0] - data_dt['HYPD'][:, 0, 0]) * t/no_it

            for r in range(len(titles['RTI'])):

                if np.isclose(HYPDt[r], 0.0):
                    continue

                # Initialise variables related to market share dynamics
                # DSiK contains the change in shares
                dSij = np.zeros([len(titles['HYTI']), len(titles['HYTI'])])

                # F contains the preferences
                F = np.ones([len(titles['HYTI']), len(titles['HYTI'])]) * 0.5

                for t1 in range(len(titles['HYTI'])):

                    if (not data_dt['HYWS'][r, t1, 0] > 0.0 or
                             list(hyti.keys())[t1] == "12 H2 by-production"):
                        continue

                    S_i = data_dt['HYWS'][r, t1, 0]

                    for t2 in range(t1):

                        if (not data_dt['HYWS'][r, t2, 0] > 0.0 or
                                 list(hyti.keys())[t1] == "12 H2 by-production"):
                            continue

                        S_j = data_dt['HYWS'][r, t2, 0]

                        # Propagating width of variations in perceived costs
                        dFij = 1.414 * sqrt((data_dt['HYLD'][r, t1, 0] * data_dt['HYLD'][r, t1, 0]
                                             + data_dt['HYLD'][r, t2, 0] * data_dt['HYLD'][r, t2, 0]))


                        # Consumer preference incl. uncertainty
                        Fij = 0.5 * (1 + np.tanh(1.25 * (data_dt['HYLC'][r, t2, 0]
                                                   - data_dt['HYLC'][r, t1, 0]) / dFij))

                        # Preferences are then adjusted for regulations
                        F[t1, t2] = Fij
                        F[t2, t1] = (1.0 - Fij)

                        #Runge-Kutta market share dynamiccs
                        k_1 = S_i*S_j * (data['HYWA'][0,t1, t2]*F[t1,t2]- data['HYWA'][0,t2, t1]*F[t2,t1])
                        k_2 = (S_i+dt*k_1/2)*(S_j-dt*k_1/2)* (data['HYWA'][0,t1, t2]*F[t1,t2] - data['HYWA'][0,t2, t1]*F[t2,t1])
                        k_3 = (S_i+dt*k_2/2)*(S_j-dt*k_2/2) * (data['HYWA'][0,t1, t2]*F[t1,t2] - data['HYWA'][0,t2, t1]*F[t2,t1])
                        k_4 = (S_i+dt*k_3)*(S_j-dt*k_3) * (data['HYWA'][0,t1, t2]*F[t1,t2] - data['HYWA'][0,t2, t1]*F[t2,t1])

                        dSij[t1, t2] = dt*(k_1+2*k_2+2*k_3+k_4)/6
                        dSij[t2, t1] = -dSij[t1, t2]

                #calculate temportary market shares and temporary capacity from endogenous results
                data['HYWS'][r, :, 0] = data_dt['HYWS'][r, :, 0] + np.sum(dSij, axis=1)

                data['HYWK'][r, :, 0] = data['HYWS'][r, :, 0] * HYPDt[r, np.newaxis]/data['BCHY'][r, :, c7ti["Capacity factor"]]

                data['HYG1'][r, :, 0] = data['HYWK'][r, :, 0] * data['BCHY'][r, :, c7ti["Capacity factor"]]

            data['HYWK'][np.isnan(data['HYWK'])] = 0.0
            data['HYG1'][np.isnan(data['HYWK'])] = 0.0

            # Call LCOH2 function
            data = get_lcoh2(data, titles)

            # Update lags
            for var in data_dt.keys():

                if domain[var] == 'FTT-H2':

                    data_dt[var] = np.copy(data[var])

    # Create cost-supply curve
    hylc = data['HYLC'][:, :, 0]
    hyld = data['HYLD'][:, :, 0]
    hycsc = data['HYCSC'][:, :, :]
    hyrcsc = data['HYRCSC'][:, :, :]
    cf_idx = c7ti['Capacity factor']
    data['BCHY'][:, -1, cf_idx] = 0.5
    hywk = data['HYG1'][:, :, 0] / data['BCHY'][:, :, cf_idx]
    hyd1 = data['HYD1'][:, 0, 0]
    hyexpsh = data['HYEXPSH'][:, 0, 0]
    hyexpsh[:] = 0.1


    hycsc, bins = get_csc(hylc, hyld, hywk, hycsc, titles)

    return data
