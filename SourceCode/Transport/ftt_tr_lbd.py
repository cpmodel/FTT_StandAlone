# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:31:15 2024

@author: Rishi
"""

from SourceCode.support.learning_by_doing import generalized_learning_by_doing

def learning_by_doing_tr(data, data_dt, time_lag, titles, year, c3ti, dt, tewi_t, new_bat):
    return generalized_learning_by_doing(
        'transport', data, data_dt,
        time_lag=time_lag, titles=titles, year=year, c3ti=c3ti, dt=dt, tewi_t=tewi_t, new_bat=new_bat
    )

"""
import numpy as np
import copy

def learning_by_doing_tr(data, data_dt, time_lag, titles, year, c3ti, dt, tewi_t, new_bat):
    # Cumulative investment for learning cost reductions
    bi = np.zeros((len(titles['RTI']), len(titles['VTTI'])))
    for r in range(len(titles['RTI'])):
        # Investment spillover
        bi[r, :] = np.matmul(data['TEWB'][0, :, :], tewi_t[r, :, 0])
    
    # Total new investment
    dw = np.sum(bi, axis=0)
    # Total new battery investment (in MWh)
    dwev = np.sum(new_bat, axis=0)    

    # Copy over TWWB values
    data['TWWB'] = copy.deepcopy(time_lag['TWWB'])

    # Cumulative capacity for batteries first
    data['TEWW'][0, :, 0] = data_dt['TEWW'][0, :, 0] + dwev[:, 0]
    bat_cap = copy.deepcopy(data["TEWW"])

    # Cumulative capacity for ICE vehicles
    for veh in range(len(titles['VTTI'])):
        if (veh < 18) or (veh > 23):
            data['TEWW'][0, veh, 0] = data_dt['TEWW'][0, veh, 0] + dw[veh]
            # Make sure bat_cap for ICE vehicles is 0
            bat_cap[0, veh, 0] = 0

    # Copy over the technology cost categories that do not change 
    # (all except prices which are updated through learning-by-doing below)
    data['BTTC'] = copy.deepcopy(data_dt['BTTC'])
    # Copy over the initial cost matrix
    data["BTCI"] = copy.deepcopy(data_dt['BTCI'])


    # Battery learning
    for veh in range(len(titles['VTTI'])):
        if 17 < veh < 24:
            # Battery cost as a result of learning
            # Battery cost = energy density over time*rare metal price trend over time
            data["BTTC"][:, veh, c3ti['19 Battery cost ($/kWh)']] = (
                (data["BTTC"][:, veh, c3ti['22 Energy density']] ** (year - 2022)) 
                * (data["BTTC"][:, veh, c3ti['21 Rare metal price']] ** (year - 2022)) 
                * data['BTCI'][:, veh, c3ti['19 Battery cost ($/kWh)']] 
                * (np.sum(bat_cap, axis=1) / np.sum(data["TWWB"], axis=1))
                ** data["BTTC"][:, veh, c3ti['16 Learning exponent']])                 
    
    # Save battery cost
    data["TEBC"] = np.zeros([len(titles['RTI']), len(titles['VTTI']),1])
    data["TEBC"][:, :, 0] = data["BTTC"][:, :, c3ti['19 Battery cost ($/kWh)']]
    
    # Initialise variable for indirect EV/PHEV costs
    id_cost = np.zeros([len(titles['RTI']), len(titles['VTTI']),1])
    # Initialise variable for cost of EV/PHEV - battery
    i_cost = np.zeros([len(titles['RTI']), len(titles['VTTI']),1])
    
    # Learning-by-doing effects on investment
    for veh in range(len(titles['VTTI'])):
        if data['TEWW'][0, veh, 0] > 0.1:
            # Learning on indirect costs (only relevant for EVs and PHEVs)
            id_cost[:, veh, 0] = (data['BTCI'][:, veh, c3ti['1 Prices cars (USD/veh)']] 
                       * 0.15 * 0.993**(year - 2022))
            # Learning on the EV/PHEV (seperate from battery)
            i_cost[:, veh, 0] = data["TEVC"][:,veh,0] * ((np.sum(bat_cap, axis=1) / np.sum(data["TWWB"], axis=1)) 
                                              ** (data["BTTC"][:, veh, c3ti['16 Learning exponent']]/2))
            
            # Calculate new costs (seperate treatments for ICE vehicles and EVs/PHEVs)
            if 17 < veh < 24:
                # EVs and PHEVs HERE     
                # Cost (excluding the cost of battery) + Updated cost of battery
                # + updated indirect cost) * Markup factor
                data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] =  \
                        ((i_cost[:,veh,0] + (data["BTTC"][:, veh, c3ti['19 Battery cost ($/kWh)']] 
                         * data["BTTC"][:, veh, c3ti['18 Battery cap (kWh)']]) + id_cost[:, veh, 0]) 
                         * data['BTTC'][:, veh, c3ti['20 Markup factor']])
            else:
                # ICE HERE
                data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] = \
                    data_dt['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] \
                    * (1.0 + data['BTTC'][:, veh, c3ti['16 Learning exponent']] \
                    * dw[veh] / data['TEWW'][0, veh, 0])

    # Investment in terms of car purchases:
    for r in range(len(titles['RTI'])):

        data['TWIY'][r, :, 0] = (data_dt['TWIY'][r, :, 0] + data['TEWI'][r, :, 0] * dt 
                                * data['BTTC'][r, :, c3ti['1 Prices cars (USD/veh)']] / 1.33
                                )
        
    return data
"""