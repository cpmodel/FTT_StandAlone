# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:41:20 2024

@author: Femke Nijsse

Find the capacity addtions of batteries by sector, including conversions to ensure
the units are the same.

Compute the cost relative cost reductions via learning-by-doing (lbd, Wright's law')
"""

import numpy as np
from SourceCode.sector_coupling.transport_batteries_to_power import get_sector_coupling_dict, share_transport_batteries

def power_battery_additions_dt(no_it, data, data_dt, titles):
    "Get battery additions from the power sector each timestep"
    
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)

    battery_lifetime = 12  # Assuming 12 years

    capacity_batteries_current_timestep = data["MSSC"] * sector_coupling_assumps["GW to GWh"]
    capacity_batteries_last_timestep = data_dt["MSSC"] * sector_coupling_assumps["GW to GWh"]

    capacity_additions_dt = capacity_batteries_current_timestep - capacity_batteries_last_timestep
    
    # New capacity + end-of-life replacements
    deployment_dt = (capacity_additions_dt
                            + capacity_batteries_current_timestep / battery_lifetime / no_it )
    
    # Account for the fact some of these batteries are not new, and come from
    # repurposed batteries or V2G
    share_bat_transport = share_transport_batteries(data, titles)
    
    deployment_new_dt = (1 - share_bat_transport) * deployment_dt
    
    return np.sum(deployment_new_dt)

def get_start_cap(data, titles):
    '''Get initial capacity. Note that this may overestimate capacity, as 
    historical battery sizes are smaller than current ones in EVs'''
    
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)
    c6ti = {category: index for index, category in enumerate(titles['C6TI'])}
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}

    
    start_cap = (
        np.sum(data["TEWK"][:, :, 0] * data['BTTC'][:, :, c3ti['18 Battery cap (kWh)']] / 1e3)
        + np.sum(data["MSSC"] * sector_coupling_assumps["GW to GWh"])
        + np.sum(data["ZEWK"][:, :, 0] * data["BZTC"][:, :, c6ti['16 Battery capacity (kWh)']] / 1e6)
        )
    
    return start_cap


    
def battery_costs(data, time_lag, year, t, titles, histend):
    """Compute the battery cost, based on (estimated) cumulative capacity."""
   
    
    if year <= histend['Battery price']:
        # Set historical cumulative capacity
        data["Cumulative total batcap"] = get_start_cap(data, titles)

    
    if year > histend['Battery price']:
        # Update battery capacities
        if time_lag["Cumulative total batcap"] == 0:
            raise ValueError(
               f"Cumulative total battery capacity not set in {year}. "
               "Ensure that battery_costs is run at least once during histend['Battery price']."
               "Updating battery costs may resolve this issue too"
               )
                
        sector_coupling_assumps = get_sector_coupling_dict(data, titles)
        battery_learning_exp = sector_coupling_assumps["Battery learning exponent"]
        battery_additions, _ = update_cumulative_cap(data, time_lag, year, t)
        
        # Approximate Wright's law
        data['Battery price'] = (time_lag["Battery price"]
                    * (1.0 + battery_learning_exp * battery_additions / time_lag['Cumulative total batcap'])  
                       )
    
    return data

def update_cumulative_cap(data, time_lag, year, t):
    """Add all the sectoral additions together for cumulative additions
    This function is called from battery_costs below"""
        
    
    battery_additions = guess_battery_additions(data, time_lag, t)
    # Add battery capacity additions across models and across timesteps
    data["Cumulative total batcap"] = time_lag["Cumulative total batcap"] + battery_additions
        
        
    return battery_additions, data

def guess_battery_additions(data, time_lag, t):
    """Compute last year's battery additions share by sector.
    
    When only some of the sectors have run, it will impute total battery 
    additions based on partial data 
    """
    t = t - 1  # Indices start at zero, not one 
    
    # Share by sector
    if np.sum(time_lag["Battery cap additions"]) > 0:
        share_by_sector = ( np.sum(time_lag["Battery cap additions"], axis=(1,2)) 
                            / np.sum(time_lag["Battery cap additions"]) )
    else: # 45% for transport and power, 10% for freight if no former shares
        share_by_sector = np.array([0.45, 0.45, 0.1])
        
     
    # Check if data is complete for all sectors at the latest timestep
    def check_complete(array, t):
        """Check if all models have run and information is complete"""
        number_of_completed_sectors = np.count_nonzero(array[:, t])
        complete = False
        if number_of_completed_sectors == 3:
            complete = True
        return complete
        
    complete = check_complete(data["Battery cap additions"], t)
    
    if complete:
        total_cap_additions = np.sum(data["Battery cap additions"][:, :t+1, 0])
    
    else:
        cap_additions_latest = data["Battery cap additions"][:, t, 0]
        # Calculate the total of non-zero elements
        total_non_zero = np.sum(cap_additions_latest)
        # Calculate the imputed total
        sum_existing_shares = np.sum(share_by_sector[cap_additions_latest != 0])
        if sum_existing_shares > 0:
            imputed_total = total_non_zero / sum_existing_shares
        else:
            imputed_total = total_non_zero
    
        total_cap_additions = (imputed_total 
                               + np.sum(data["Battery cap additions"][:, :t]) )
    
    
    return total_cap_additions

