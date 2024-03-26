# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_shares.py
=========================================
Second-hand batteries repurposing module

@author: Femke Nijsse
"""

import numpy as np 

def second_hand_batteries(data, time_lag, iter_lag, year, titles):
    """
    This function estimates the size of the second-hand battery market
    based on scrappage of electric vehicles from FTT:Tr. We use batteries
    from cars scrapped in the previous time step
    
    Returns:
        data dictionary with updated battery capacity in GWh
        # Todo: check if units still correct
    """
    utilisation_rate = 0.5  # Share of car batteries getting a second life in power

    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}
    
    # Units - REVS: 1000 cars, BTTC - kWh
    battery_capacity_last_year = time_lag["REVS"] * data["BTTC"][:, :, c3ti['18 Battery cap (kWh)'], None]
    # Sum battery capacity in GWh
    summed_battery_capacity = np.sum(battery_capacity_last_year, axis=1)/1000
    used_battery_capacity = summed_battery_capacity * utilisation_rate
    # Move all batteries one year up
    
    data['Second-hand batteries by age'][..., :-1] = np.copy(time_lag['Second-hand battery stock'][..., 1:])
    data['Second-hand batteries by age'][..., -1] = used_battery_capacity
    data['Second-hand battery in GWh'] = np.sum(data['Second-hand battery by age'], axis=-1)
    

    return data