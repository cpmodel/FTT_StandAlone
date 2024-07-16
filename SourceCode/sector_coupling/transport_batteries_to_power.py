# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_2nd_hand_batteries.py
=========================================
Second-hand batteries repurposing module

@author: Femke Nijsse
"""

import numpy as np 

def get_sector_coupling_dict(data, titles):
    sector_coupling_assumps = dict(zip(
                list(titles['SCA']),
                data["SectorCouplingAssumps"][0, :, 0]
                ))
    return sector_coupling_assumps
    
def second_hand_batteries(data, time_lag, iter_lag, year, titles):
    """
    This function estimates the size of the second-hand battery market
    based on scrappage of electric vehicles from FTT:Tr. We use batteries
    from cars scrapped in the previous time step
    
    Numbers are taken from the Xu et al (2022) paper: 
    https://www.nature.com/articles/s41467-022-35393-0
    
    Differences with Xu can be partially explained by the smaller size of  
    batteries in FTT:Tr compared to their model. 

    Returns:
        data dictionary with updated battery capacity in GWh
        # Todo: check if units still correct
        
    """
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)
    
    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}
    
    # Units - REVS: 1000 cars, BTTC - kWh
    battery_capacity_lag = time_lag["REVS"] * data["BTTC"][:, :, c3ti['18 Battery cap (kWh)'], None]
    
    # Sum battery capacity in GWh
    battery_capacity_lag_sum = np.sum(battery_capacity_lag, axis=1) / 1000
    summed_battery_capacity = (battery_capacity_lag_sum  
                               * sector_coupling_assumps["Starting degradation"]
                              )
    used_battery_capacity = (summed_battery_capacity 
                            * sector_coupling_assumps["Utilisation rate"] 
                            )
    
    # Move all batteries one year up
    data['Second-hand batteries by age'][:, :-1, :] = (
            np.copy(time_lag['Second-hand batteries by age'][:, 1:, :]) 
            * sector_coupling_assumps["Yearly decay"] 
            )
    
    # Add newly scrapped vehicle batteries to tracking matrix
    data['Second-hand batteries by age'][:, -1, :] = used_battery_capacity
    
    # All all batteries together
    data['Second-hand battery stock'] = \
            np.sum(data['Second-hand batteries by age'], axis=1, keepdims=True)
                        
    
    return data

def share_repurposed_batteries(data, year, titles):
    """
    Estimate the battery needs in power from GW to GWh, and compute ratio with transport.
    The original Ueckerdt paper does not contain this information. We therefore estimate
    this from key numbers in the https://energy.mit.edu/research/future-of-energy-storage/
    
    In table 6.13, in 5 gCO2 scenario, there is a factor 3.8 between the two.
    In table C12, in the 5 gCO2 scenario, there is a factor 4.8. Average is
    4.3. 

    Returns
    -------
    Share repurposed batteries compared to short-term storage needs
    
    """
    
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)

    # Convert GW to GWh (estimate)
    capacity_batteries_power = data["MSSC"] * sector_coupling_assumps["GW to GWh"]
    
    storage_ratio = data["Second-hand battery stock"]  / capacity_batteries_power
    
    
    return storage_ratio

def update_costs_from_repurposing(data, storage_ratio, year, titles):
    """
    Compute the new costs for storage. Repurposing batteries have costs
    of roughly 20-80% of new batteries, or 30% to 70%, according to:
        https://www.sciencedirect.com/science/article/pii/S2589004223012725
        
    We take a mid-point value of 50%.
    
    Returns
    -------
    Battery costs and new marginal costs
    
    """ 
    
    sector_coupling_assumps = get_sector_coupling_dict(data, titles)
    
    share_repurposed = np.minimum(storage_ratio, 1)
    remaining_costs_fraction = ((1-share_repurposed)
                                + share_repurposed 
                                * (1-sector_coupling_assumps["Cost savings"]))
    # if year%10 == 0:
    #     print(f"In {year}, the remaining_cost_fraction is {remaining_costs_fraction[1]}")
    #     print(f"In {year}, region with the highest remaining fraction is {np.argmax(remaining_costs_fraction)}"
    #           f"at {remaining_costs_fraction[np.argmax(remaining_costs_fraction)]}")

    
    data["MSSP"] = data["MSSP"] * remaining_costs_fraction
    data["MSSM"] = data["MSSM"] * remaining_costs_fraction
    
    return data
    