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
    
    Numbers are taken from the Xu et al (2022) paper: https://www.nature.com/articles/s41467-022-35393-0

    Returns:
        data dictionary with updated battery capacity in GWh
        # Todo: check if units still correct
    """
    utilisation_rate = 0.5  # Share of car batteries getting a second life in power
    yearly_decay = 0.98  # Assumption, very roughly based on Xu, but not really
    starting_degredation = 0.74 # Starting capacity of the batteries according to Xu et al.

    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}
    
    # Units - REVS: 1000 cars, BTTC - kWh
    battery_capacity_last_year = time_lag["REVS"] * data["BTTC"][:, :, c3ti['18 Battery cap (kWh)'], None]
    # Sum battery capacity in GWh
    summed_prior_battery_capacity = np.sum(battery_capacity_last_year, axis=1)/1000
    summed_battery_capacity = summed_prior_battery_capacity * starting_degredation
    used_battery_capacity = summed_battery_capacity * utilisation_rate
    # Move all batteries one year up
    
    data['Second-hand batteries by age'][..., :-1] = np.copy(time_lag['Second-hand battery stock'][..., 1:]) * yearly_decay
    data['Second-hand batteries by age'][..., -1] = used_battery_capacity
    data['Second-hand battery stock'] = np.sum(data['Second-hand batteries by age'], axis=-1)
    br = 1

    return data