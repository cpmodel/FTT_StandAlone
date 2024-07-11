# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 21:54:30 2024

@author: Owner
"""

def calculate_cumulative_storage(data_power, data_transport, data_heat, data_freight, c3ti, c4ti, c6ti):
    # Calculate total battery capacity for EVs and PHEVs in the transport model
    ev_phev_battery_capacity = np.sum(data_transport["TEWI"][:, 17:24, 0] * data_transport["BTTC"][:, 17:24, c3ti["18 Battery cap (kWh)"]])
    
    # Extract power model storage capacities
    power_storage_capacity = np.sum(data_power["MEWK"][:, :, 0])  # Example calculation
    
    # Calculate total new capacity added in the heat model
    total_heat_capacity_added = np.sum(data_heat["HEWI"][:, :, 0] * data_heat["BHTC"][:, :, c4ti["1 Inv cost mean (EUR/Kw)"]] / data_heat["PRSC14"][:, 0, 0, np.newaxis])
    
    # Calculate total new capacity added in the freight model
    bi = np.zeros((len(data_freight['ZEWY']), len(data_freight['ZEWY'][0])))
    for r in range(len(data_freight['ZEWY'])):
        bi[r, :] = np.matmul(data_freight['ZEWB'][0, :, :], data_freight['ZEWY'][r, :, 0])
    dw = np.sum(bi, axis=0)
    data_freight['ZEWW'][0, :, 0] += dw
    total_freight_capacity_added = np.sum(data_freight['ZEWW'][0, :, 0])
    
    # Combine the storage capacities
    cumulative_storage_capacity = ev_phev_battery_capacity + power_storage_capacity + total_heat_capacity_added + total_freight_capacity_added
    
    return cumulative_storage_capacity
