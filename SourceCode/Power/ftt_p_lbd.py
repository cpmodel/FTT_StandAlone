# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:58:20 2024

@author: Rishi
"""

import numpy as np

def learning_by_doing(mewi_t, titles, data, data_dt, time_lag, c2ti, set_carbon_tax, year):
    # Cumulative global learning
    mewi0 = np.sum(mewi_t[:, :, 0], axis=0)
    dw = np.zeros(len(titles["T2TI"]))
    
    for i in range(len(titles["T2TI"])):
        dw_temp = np.copy(mewi0)
        dw_temp[dw_temp > dw_temp[i]] = dw_temp[i]
        dw[i] = np.dot(dw_temp, data['MEWB'][0, i, :])

    # Cumulative capacity incl. learning spill-over effects
    data["MEWW"][0, :, 0] = data_dt['MEWW'][0, :, 0] + dw

    # Copy over the technology cost categories. We update the investment and capacity factors below
    data['BCET'][:, :, 1:17] = time_lag['BCET'][:, :, 1:17].copy()

    # Store gamma values in the cost matrix (in case it varies over time)
    data['BCET'][:, :, c2ti['21 Gamma ($/MWh)']] = data['MGAM'][:, :, 0]

    # Add in carbon costs
    data['BCET'][:, :, c2ti['1 Carbon Costs ($/MWh)']] = set_carbon_tax(data, c2ti, year)

    # Learning-by-doing effects on investment
    for tech in range(len(titles['T2TI'])):
        if data['MEWW'][0, tech, 0] > 0.001:
            data['BCET'][:, tech, c2ti['3 Investment ($/kW)']] = (
                data_dt['BCET'][:, tech, c2ti['3 Investment ($/kW)']]
                * (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech] / data['MEWW'][0, tech, 0]))
            data['BCET'][:, tech, c2ti['4 std ($/MWh)']] = (
                data_dt['BCET'][:, tech, c2ti['4 std ($/MWh)']]
                * (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech] / data['MEWW'][0, tech, 0]))
            data['BCET'][:, tech, c2ti['7 O&M ($/MWh)']] = (
                data_dt['BCET'][:, tech, c2ti['7 O&M ($/MWh)']]
                * (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech] / data['MEWW'][0, tech, 0]))
            data['BCET'][:, tech, c2ti['8 std ($/MWh)']] = (
                data_dt['BCET'][:, tech, c2ti['8 std ($/MWh)']]
                * (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech] / data['MEWW'][0, tech, 0]))
    
    return data

def calculate_cumulative_capacity_power(data, data_dt, get_sector_coupling_dict, titles):
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)
    
    battery_lifetime = 12  # assuming 12 years
    time_steps = 4  # per quarter in a year

    capacity_batteries_current_timestep = data["MSSC"] * sector_coupling_assumps["GW to GWh"]
    capacity_batteries_last_timestep = data_dt["MSSC"] * sector_coupling_assumps["GW to GWh"]

    quarterly_cap_additions = capacity_batteries_current_timestep - capacity_batteries_last_timestep

    cumulative_quarterly_capacity_power = quarterly_cap_additions + (capacity_batteries_current_timestep / (battery_lifetime / time_steps))
    
    return cumulative_quarterly_capacity_power
