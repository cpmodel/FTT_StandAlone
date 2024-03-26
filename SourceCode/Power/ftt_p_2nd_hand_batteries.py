# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:07:01 2024

@author: Femke Nijsse
"""

def second_hand_batteries(time_lag, data, titles):
    """This function seeks to estimate the size of the second-hand battery
    market based on scrappage of electric vehicles from FTT:Tr. We use batteries
    from cars scrapped in the previous time step
    
    Returns:
        data dictionary with updated battery capacity in GWh
        # Todo: check if units still correct
    """
    
    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}
    
    # Units - REVS: 1000 cars, BTTC - kWh
    battery_capacity_last_year = time_lag["REVS"] * data["BTTC"][:, c3ti['18 Battery cap (kWh)']]
    
    
    return battery_capacity_last_year