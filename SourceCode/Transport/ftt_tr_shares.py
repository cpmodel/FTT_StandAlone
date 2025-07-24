# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 09:53:20 2025

@author: Test Profile
"""

import numpy as np
from math import sqrt

def shares_transport(data_dt, data, year, rfltt, isReg, titles, c3ti, dt):

    num_regions = len(titles['RTI'])
    num_techs = len(titles['VTTI'])
    
    # Initialize output arrays for all regions
    endo_shares = np.zeros((num_regions, num_techs))
    endo_capacity = np.zeros((num_regions, num_techs))

    for r in range(len(titles['RTI'])):
        # Skip regions for which more recent data is available
        if data['TDA1'][r, 0, 0] >= year:
            continue

        if rfltt[r] == 0.0:
            continue

        ############################ FTT ##################################
        # Initialise variables related to market share dynamics
        # DSiK contains the change in shares
        dSik = np.zeros([len(titles['VTTI']), len(titles['VTTI'])])

        # F contains the preferences
        F = np.ones([len(titles['VTTI']), len(titles['VTTI'])])*0.5

    
        for v1 in range(len(titles['VTTI'])):

            # Skip technologies with zero market share or zero costs
            if not (data_dt['TEWS'][r, v1, 0] > 0.0 and
                    data_dt['TELC'][r, v1, 0] != 0.0 and
                    data_dt['TLCD'][r, v1, 0] != 0.0):
                continue

            S_veh_i = data_dt['TEWS'][r, v1, 0]

            for v2 in range(v1):

                # Skip technologies with zero market share or zero costs
                if not (data_dt['TEWS'][r, v2, 0] > 0.0 and
                        data_dt['TELC'][r, v2, 0] != 0.0 and
                        data_dt['TLCD'][r, v2, 0] != 0.0):
                    continue

                S_veh_k = data_dt['TEWS'][r, v2, 0]
                Aik = data['TEWA'][0, v1, v2] * \
                    data['BTTC'][r, v1, c3ti['17 Turnover rate']]
                Aki = data['TEWA'][0, v2, v1] * \
                    data['BTTC'][r, v2, c3ti['17 Turnover rate']]

                # Propagating width of variations in perceived costs
                dFik = np.sqrt(2) * sqrt((data_dt['TLCD'][r, v1, 0] * data_dt['TLCD'][r, v1, 0]
                                     + data_dt['TLCD'][r, v2, 0] * data_dt['TLCD'][r, v2, 0]))

                # Consumer preference incl. uncertainty
                Fik = 0.5 * (1 + np.tanh(1.25 * (data_dt['TELC'][r, v2, 0]
                                                 - data_dt['TELC'][r, v1, 0]) / dFik))
                # Preferences are then adjusted for regulations
                F[v1, v2] = (Fik * (1.0 - isReg[r, v1]) * (1.0 - isReg[r, v2]) + isReg[r, v2]
                             * (1.0 - isReg[r, v1]) + 0.5 * (isReg[r, v1] * isReg[r, v2]))
                F[v2, v1] = ((1.0 - Fik) * (1.0 - isReg[r, v2]) * (1.0 - isReg[r, v1]) + isReg[r, v1]
                             * (1.0-isReg[r, v2]) + 0.5 * (isReg[r, v2] * isReg[r, v1]))

                # Runge-Kutta market share dynamiccs
                k_1 = S_veh_i*S_veh_k * (Aik*F[v1, v2] - Aki*F[v2, v1])
                k_2 = (S_veh_i+dt*k_1/2)*(S_veh_k-dt*k_1/2) * \
                    (Aik*F[v1, v2] - Aki*F[v2, v1])
                k_3 = (S_veh_i+dt*k_2/2)*(S_veh_k-dt*k_2/2) * \
                    (Aik*F[v1, v2] - Aki*F[v2, v1])
                k_4 = (S_veh_i+dt*k_3)*(S_veh_k-dt*k_3) * \
                    (Aik*F[v1, v2] - Aki*F[v2, v1])

                # Market share dynamics
                # dSik[v1, v2] = S_veh_i*S_veh_k* (Aik*F[v1,v2] - Aki*F[v2,v1])*dt
                dSik[v1, v2] = dt*(k_1+2*k_2+2*k_3+k_4)/6
                dSik[v2, v1] = -dSik[v1, v2]

        # Calculate temporary market shares and temporary capacity from endogenous results
        endo_shares[r, :] = data_dt['TEWS'][r, :, 0] + np.sum(dSik, axis=1)
        endo_capacity[r, :] = endo_shares[r, :] * rfltt[r]
          
    return endo_shares, endo_capacity