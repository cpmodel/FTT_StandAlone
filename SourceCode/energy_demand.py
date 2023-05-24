# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:55:17 2023

@author: Femke Nijsse
"""
import numpy as np
import copy

def energy_demand(data, titles, histend, ftt_modules):
    """ Energy demand (FRET in E3ME-FTT),
    as a function of the heat and transport sector
    
    
    """
    data["FRET"][:, :, 0] = copy.deepcopy(data['FRETX'][:, :, 0])
    for r in range(len(titles['RTI'])):  # Loop over world regions
        if "FTT-Tr" in ftt_modules:
            data["FRET"][r, 15, 0] = data["TJEF"][r, 7, 0]
        if "FTT-H" in ftt_modules:            
            data["FRET"][r, 18, 0] = data["HJEF"][r, 7, 0]
        if r == 0:
            print("Electricity usage in heating in Belgium:")
            print(data["HJEF"][r, 7, 0])
        
        # Compute the changes in actual electricity demand (normally in FTT.f90, put here for testing purposes)
        
        data["MEWD"][r, 7, 0] = np.sum(data["FRET"][r, :, 0]) * 41.868/1000 #/3.6 electricity
        
    return data

