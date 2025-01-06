# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:41:20 2024

@author: Femke Nijsse
"""

import numpy as np
from SourceCode.sector_coupling.transport_batteries_to_power import get_sector_coupling_dict, share_transport_batteries

def quarterly_bat_add_power(no_it, data, data_dt, titles):
    "Add battery additions from the power sector each quarter"
    
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
    This function is called from battery_costs below"""
    
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)
    c6ti = {category: index for index, category in enumerate(titles['C6TI'])}
    
    # For simplicity, in 2022, the capacity is set to 2022 values of overall battery
    # capacity in the three models together. 
    # TODO: Deal with the fact that different models start at different times
    
    if year <= 2022:
        data["Cumulative total batcap start"][0, 0, 0] = (
            np.sum(data["TEWW"][0, 18:24]) / 1000
            + np.sum(data["MSSC"] * sector_coupling_assumps["GW to GWh"])
            + np.sum(data["ZEWK"][:, :, 0] * data["BZTC"][:, :, c6ti['16 Battery capacity (kWh)']] / 1e6)
            )
        data["Cumulative total batcap"] = np.copy(data["Cumulative total batcap start"])
    else:
        # Copy over the 2022 value from last year
        data["Cumulative total batcap start"] = time_lag["Cumulative total batcap start"]
        
        # Impute any missing data from the current timestep
        battery_additions = guess_battery_additions(data, time_lag)
        # Add battery capacity additions across models and across timesteps
        data["Cumulative total batcap"] = time_lag["Cumulative total batcap"] + battery_additions
        
    return data


    
def battery_costs(data, time_lag, year, titles):
    """Compute remaining fraction of costs for batteries, based on cumulative 
    capacities"""
    
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)
    battery_learning_exp = sector_coupling_assumps["Battery learning exponent"]
    
    get_cumulative_batcap(data, time_lag, year, titles)
    # No learning takes place before 2021
    if year <= 2022:
        battery_cost_fraction = 1
    else:
        # Add safety checks for both division and exponentiation
        safe_denominator = np.where(time_lag["Cumulative total batcap start"] <= 0,
                                  np.finfo(float).eps,
                                  time_lag["Cumulative total batcap start"])
        
        ratio = time_lag["Cumulative total batcap"] / safe_denominator
        # Ensure ratio is positive before applying power
        safe_ratio = np.where(ratio <= 0, np.finfo(float).eps, ratio)
        
        battery_cost_fraction = safe_ratio ** battery_learning_exp
    
    return battery_cost_fraction

def guess_battery_additions(data, time_lag):
    """ This function computes last year's battery additions share by sector, 
    and imputes the total battery additions based on partial data 
    or normally if there is complete data. 
    """
    
    # Share by sector
    if np.sum(time_lag["Battery cap additions"]) > 0:
        share_by_sector = ( np.sum(time_lag["Battery cap additions"], axis=(1,2)) 
                            / np.sum(time_lag["Battery cap additions"]) )
    else: # 45% for transport and power, 10% for freight if no former shares
        share_by_sector = np.array([0.45, 0.45, 0.1])
    
    # Find last timestep with at least some data:
    def find_current_timestep(array):
        """Find the latest column with at least one non-zero element"""
        for col in range(array.shape[1] - 1, -1, -1):
            if 1 <= np.count_nonzero(array[:, col]) <= 3:
                return col
        return None
    latest_timestep = find_current_timestep(data["Battery cap additions"])
    
    # Return zero if there is no data
    if latest_timestep is None:
        return 0
    
    
    
    # Check if data complete
    def check_complete(array, latest_timestep):
        """Check if all models have run and information is complete"""
        number_of_completed_sectors = np.count_nonzero(array[:, latest_timestep])
        complete = False
        if number_of_completed_sectors == 3:
            complete = True
        return complete
        
    complete = check_complete(data["Battery cap additions"], latest_timestep)
    
    if complete:
        total_cap_additions = np.sum(data["Battery cap additions"])
    else:
        cap_additions_latest = data["Battery cap additions"][:, latest_timestep, 0]
        # Calculate the total of non-zero elements
        total_non_zero = np.sum(cap_additions_latest)
        # Calculate the imputed total
        sum_existing_shares = np.sum(share_by_sector[cap_additions_latest != 0])
        if sum_existing_shares > 0:
            imputed_total = total_non_zero / sum_existing_shares
        else:
            imputed_total = total_non_zero
    
        total_cap_additions = (imputed_total 
                               + np.sum(data["Battery cap additions"][:, :latest_timestep]) )
    
    
    return total_cap_additions    
    
    