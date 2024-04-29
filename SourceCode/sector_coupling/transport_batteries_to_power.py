# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_2nd_hand_batteries.py
=========================================
Second-hand batteries repurposing module

@author: Femke Nijsse
"""

import numpy as np 

from SourceCode.support.read_assumptions import read_sc_assumptions

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
    
    assumptions = read_sc_assumptions()
    utilisation_rate = 0.5       # Share of car batteries getting a second life in power
    yearly_decay = 0.98          # Assumption, very roughly based on Xu, but not really
    
    # Starting capacity of the batteries according to Xu et al. Conservative assumption
    starting_degredation = 0.74  

    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}
    
    # Units - REVS: 1000 cars, BTTC - kWh
    battery_capacity_lag = time_lag["REVS"] * data["BTTC"][:, :, c3ti['18 Battery cap (kWh)'], None]
    
    # Sum battery capacity in GWh
    battery_capacity_lag_sum = np.sum(battery_capacity_lag, axis=1)/1000
    summed_battery_capacity = battery_capacity_lag_sum * starting_degredation
    used_battery_capacity = summed_battery_capacity * utilisation_rate
    
    # Move all batteries one year up
    data['Second-hand batteries by age'][:, :-1, :] = \
            np.copy(time_lag['Second-hand batteries by age'][:, 1:, :]) * yearly_decay
    
    # Add newly scrapped vehicle batteries to tracking matrix
    data['Second-hand batteries by age'][:, -1, :] = used_battery_capacity
    
    # All all batteries together
    data['Second-hand battery stock'] = \
            np.sum(data['Second-hand batteries by age'], axis=1, keepdims=True)
                        
    
    return data

def share_repurposed_batteries(data, year):
    """
    Estimate the total storage demand (GWh) from capacity (GW). The original 
    Ueckerdt paper does not contain this information. We therefore estimate
    this from key numbers in the https://energy.mit.edu/research/future-of-energy-storage/
    
    In table 6.13, in 5 gCO2 scenario, there is a factor 3.8 between the two.
    In table C12, in the 5 gCO2 scenario, there is a factor 4.8. Average is
    4.3. 

    Returns
    -------
    Share repurposed batteries compared to short-term storage needs
    
    """
    # Convert from GW to GWh (estimate)
    storage_from_transport = data["Second-hand battery stock"] * 4.3
    storage_ratio = storage_from_transport / data["MSSC"]
    
    if year%10 == 0:
        print(f"Stock storage repurposed batteries in {year}: {np.sum(storage_from_transport)/1000:.3f} TWh")
        print(f"Storage ratio is {storage_ratio[0]}")
        print(f"Storage demand in the power sector: {np.sum(data['MSSC'])/1000:.3f} TWh")
    
    return storage_ratio

def update_costs_from_repurposing(data, storage_ratio, year):
    """
    Compute the new costs for storage. Repurposing batteries have costs
    of roughly 20-80% of new batteries, or 30% to 70%, according to:
        https://www.sciencedirect.com/science/article/pii/S2589004223012725
        
    We take a mid-point value of 50%.
    
    Returns
    -------
    Battery costs and new marginal costs
    
    """ 
    share_repurposed = np.minimum(storage_ratio, 1)
    cost_savings = 0.5
    remaining_costs_fraction = ((1-share_repurposed) + share_repurposed * (1-cost_savings))
    print(f"In year {year}, the remaining_cost_fraction is {remaining_costs_fraction[0]}")
    #data["MSSP"] = data["MSSP"] * remaining_costs_fraction
    #data["MSSM"] = data["MSSM"] * remaining_costs_fraction
    
    return data
    