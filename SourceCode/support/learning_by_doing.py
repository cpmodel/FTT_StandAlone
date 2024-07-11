# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:30:35 2024

@author: Rishi
"""

import numpy as np
import copy
from SourceCode.Freight.ftt_fr_lcof import get_lcof
from SourceCode.Power.ftt_p_lcoe import set_carbon_tax

def generalized_learning_by_doing(model_type, data, data_dt, **kwargs):
    if model_type == 'freight':
        titles = kwargs['titles']
        c6ti = kwargs['c6ti']
        dt = kwargs['dt']
        dw = kwargs['dw']

        for tech in range(len(titles['FTTI'])):
            if data['ZEWW'][0, tech, 0] > 0.1:
                data['ZCET'][:, tech, c6ti['1 Price of vehicles (USD/vehicle)']] =  \
                        data_dt['ZCET'][:, tech, c6ti['1 Price of vehicles (USD/vehicle)']] * \
                        (1.0 + data['ZLER'][tech] * dw[tech]/data['ZEWW'][0, tech, 0])

        # Calculate total investment by technology in terms of truck purchases
        for r in range(len(titles['RTI'])):
            data['ZWIY'][r, :, 0] = data_dt['ZWIY'][r, :, 0] + \
            data['ZEWY'][r, :, 0]*dt*data['ZCET'][r, :, c6ti['1 Price of vehicles (USD/vehicle)']]*1.263

        # Calculate levelised cost again
        data = get_lcof(data, titles)

    elif model_type == 'heat':
        time_lag = kwargs['time_lag']
        titles = kwargs['titles']
        c4ti = kwargs['c4ti']
        dt = kwargs['dt']
        hewi_t = kwargs['hewi_t']

        bi = np.zeros((len(titles['RTI']),len(titles['HTTI'])))
        for r in range(len(titles['RTI'])):
            bi[r,:] = np.matmul(data['HEWB'][0, :, :],hewi_t[r, :, 0])
        dw = np.sum(bi, axis=0)

        # Cumulative capacity incl. learning spill-over effects
        data['HEWW'][0, :, 0] = data_dt['HEWW'][0, :, 0] + dw

        # Copy over the technology cost categories that do not change (all except prices which are updated through learning-by-doing below)
        data['BHTC'] = copy.deepcopy(data_dt['BHTC'])

        # Learning-by-doing effects on investment and efficiency
        for b in range(len(titles['HTTI'])):

            if data['HEWW'][0, b, 0] > 0.0001:

                data['BHTC'][:, b, c4ti['1 Inv cost mean (EUR/Kw)']] = (data_dt['BHTC'][:, b, c4ti['1 Inv cost mean (EUR/Kw)']]  \
                                                                         *(1.0 + data['BHTC'][:, b, c4ti['7 Investment LR']] * dw[b]/data['HEWW'][0, b, 0]))
                data['BHTC'][:, b, c4ti['2 Inv Cost SD']] = (data_dt['BHTC'][:, b, c4ti['2 Inv Cost SD']]  \
                                                                         *(1.0 + data['BHTC'][:, b, c4ti['7 Investment LR']] * dw[b]/data['HEWW'][0, b, 0]))
                data['BHTC'][:, b, c4ti['9 Conversion efficiency']] = (data_dt['BHTC'][:, b, c4ti['9 Conversion efficiency']] \
                                                                        * 1.0 / (1.0 + data['BHTC'][:, b, c4ti['20 Efficiency LR']] * dw[b]/data['HEWW'][0, b, 0]))


        #Total investment in new capacity in a year (m 2014 euros):
          #HEWI is the continuous time amount of new capacity built per unit time dI/dt (GW/y)
          #BHTC(:,:,1) are the investment costs (2014Euro/kW)
        data['HWIY'][:,:,0] = data['HWIY'][:,:,0] + data['HEWI'][:,:,0]*dt*data['BHTC'][:,:,0]/data['PRSC14'][:,0,0,np.newaxis]
        # Save investment cost for front end
        data["HWIC"][:, :, 0] = data["BHTC"][:, :, c4ti['1 Inv cost mean (EUR/Kw)']]
        # Save efficiency for front end
        data["HEFF"][:, :, 0] = data["BHTC"][:, :, c4ti['9 Conversion efficiency']]
        
    elif model_type == 'power':
        titles = kwargs['titles']
        time_lag = kwargs['time_lag']
        year = kwargs['year']
        c2ti = kwargs['c2ti']
        dt = kwargs['dt']
        
        bi = np.zeros((len(titles['RTI']),len(titles['T2TI'])))
        mewi0 = np.sum(data['MEWI'][:, :, 0], axis=0)
        dw = np.zeros(len(titles["T2TI"]))

        for i in range(len(titles["T2TI"])):
            dw_temp = copy.deepcopy(mewi0) * dt
            dw_temp[dw_temp > dw_temp[i]] = dw_temp[i]
            dw[i] = np.dot(dw_temp, data['MEWB'][0, i, :])

        # Cumulative capacity incl. learning spill-over effects
        data["MEWW"][0, :, 0] = data_dt['MEWW'][0, :, 0] + dw

        # Copy over the technology cost categories that do not change (all except prices which are updated through learning-by-doing below)
        data['BCET'][:, :, 1:17] = copy.deepcopy(time_lag['BCET'][:, :, 1:17])

        # Store gamma values in the cost matrix (in case it varies over time)
        data['BCET'][:, :, c2ti['21 Empty']] = copy.deepcopy(data['MGAM'][:, :, 0])

        # Add in carbon costs
        data['BCET'][:, :, c2ti['1 Carbon Costs ($/MWh)']] = set_carbon_tax(data, c2ti, year)

        # Learning-by-doing effects on investment
        for tech in range(len(titles['T2TI'])):
            if data['MEWW'][0, tech, 0] > 0.1:
                data['BCET'][:, tech, c2ti['3 Investment ($/kW)']] = data_dt['BCET'][:, tech, c2ti['3 Investment ($/kW)']] * \
                                                                      (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0])
                data['BCET'][:, tech, c2ti['4 std ($/MWh)']] = data_dt['BCET'][:, tech, c2ti['4 std ($/MWh)']] * \
                                                                      (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0])
                data['BCET'][:, tech, c2ti['7 O&M ($/MWh)']] = data_dt['BCET'][:, tech, c2ti['7 O&M ($/MWh)']] * \
                                                                      (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0])
                data['BCET'][:, tech, c2ti['8 std ($/MWh)']] = data_dt['BCET'][:, tech, c2ti['8 std ($/MWh)']] * \
                                                                      (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0])

        # Investment in terms of car purchases:
        for r in range(len(titles['RTI'])):
            data['MWIY'][r, :, 0] = data_dt['MWIY'][r, :, 0] + data['MEWI'][r, :, 0]*dt*data['BCET'][r, :, c2ti['3 Investment ($/kW)']]/1.33
            
    elif model_type == 'transport':
        titles = kwargs['titles']
        time_lag = kwargs['time_lag']
        year = kwargs['year']
        c3ti = kwargs['c3ti']
        dt = kwargs['dt']
        tewi_t = kwargs['tewi_t']
        new_bat = kwargs['new_bat']
        
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
