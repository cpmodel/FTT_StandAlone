# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:41:20 2024

@author: Femke Nijsse
"""

import numpy as np
from SourceCode.sector_coupling.transport_batteries_to_power import get_sector_coupling_dict, share_transport_batteries

def quarterly_bat_add_power(no_it, data, data_dt, titles):
    
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)

    battery_lifetime = 12  # Assuming 12 years

    capacity_batteries_current_timestep = data["MSSC"] * sector_coupling_assumps["GW to GWh"]
    capacity_batteries_last_timestep = data_dt["MSSC"] * sector_coupling_assumps["GW to GWh"]

    quarterly_cap_additions = capacity_batteries_current_timestep - capacity_batteries_last_timestep
    
    # New capacity + end-of-life replacements
    quarterly_deployment = (quarterly_cap_additions
                            + capacity_batteries_current_timestep / battery_lifetime / no_it )
    
    # Account for the fact some of these batteries are not new, and come from
    # repurposed batteries or V2G
    share_bat_transport = share_transport_batteries(data, titles)
    
    quarterly_deployment_new = (1 - share_bat_transport) * quarterly_deployment
    
    return np.sum(quarterly_deployment_new)



def get_cumulative_batcap(data, time_lag, year, titles):
    """Add all the quarterly additions together for true cumulative additions
    This function is called from the model_class, after the power sector"""
    
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)
    c6ti = {category: index for index, category in enumerate(titles['C6TI'])}
    
    
    # For simplicity, in 2020, the capacity is set to 2020 values of overall battery
    # capacity in the three models together. 
    
    if year <= 2020:
        data["Cumulative total batcap 2020"][0, 0, 0] = (
            np.sum(data["TWWB"]) / 1000
            + np.sum(data["MSSC"] * sector_coupling_assumps["GW to GWh"])
            + np.sum(data["ZEWK"][:, :, 0] * data["ZCET"][:, :, c6ti['21 Battery capacity (kWh)']] / 10e6)
            )
        data["Cumulative total batcap"] = np.copy(data["Cumulative total batcap 2020"])
    else:
        # Copy over the 2020 value from last year
        data["Cumulative total batcap 2020"] = time_lag["Cumulative total batcap 2020"]
        
        # Add battery capacity additions across models and across timesteps
        cum_batcap = time_lag["Cumulative total batcap"] + np.sum(data["Battery cap additions"])
        data["Cumulative total batcap"] = cum_batcap
        
    return data

# def shares_batcap_transport(battery_cap_additions, cum_additions):
#     "Each year, calculate the share of battery additions from transport"
    
def battery_costs(data, time_lag, year, titles):
    """Compute remaining fraction of costs for batteries, based on cumulative 
    capacities"""
    
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)
    battery_learning_exp = sector_coupling_assumps["Battery learning exponent"]
    battery_cost_fraction = (
        ( time_lag["Cumulative total batcap"] / time_lag["Cumulative total batcap 2020"] ) 
        ** battery_learning_exp )
    
    # No learning takes place before 2020
    if year <= 2020:
        battery_cost_fraction = 1
    
    return battery_cost_fraction
    
    