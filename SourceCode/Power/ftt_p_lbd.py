# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:10:38 2024

@author: Rishi
"""

# =============================================================
# Learning-by-doing
# =============================================================

# Cumulative global learning
# Using a technological spill-over matrix (PG_SPILL) together with capacity
# additions (PG_CA) we can estimate total global spillover of similar
# techicals

from SourceCode.support.learning_by_doing import generalized_learning_by_doing

def learning_by_doing(data, data_dt, time_lag, titles, year, c2ti, dt):
    return generalized_learning_by_doing(
        'power', data, data_dt, 
        time_lag=time_lag, titles=titles, year=year, c2ti=c2ti, dt=dt
    )


"""
import numpy as np
import copy
from SourceCode.Power.ftt_p_lcoe import set_carbon_tax

def learning_by_doing(data, data_dt, time_lag, titles, year, c2ti, dt):
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

    return data
"""