# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:55:17 2023

@author: Femke Nijsse
"""
import numpy as np
import copy

def energy_demand_from_sectors(data, titles, histend, year, ftt_modules):
    """ Energy demand (FRET in E3ME-FTT),
    as a function of the heat, freight and transport sector
    FRET is given in thousands toe
    
    In E3ME, freight is assumed to move with Tr in terms of biofuel mandates,
    and TJEF includes both transport and freight. 
    """
    
    if "FTT-P" in ftt_modules:
        data["FRET"][:, :, 0] = copy.deepcopy(data['FRETX'][:, :, 0])
        print("The data in FRET is:")
        print(data["FRET"][0])
        for r in range(len(titles['RTI'])):  # Loop over world regions
            if r == 0 and year in range(2012, 2016):
                print("Electricity usage in heating in Belgium:")
                print(f'HJEF: {data["HJEF"][r, 7, 0]}')
                print(f'FRET: {data["FRET"][r, 18, 0]}')
            
            if "FTT-Tr" in ftt_modules:
                data["FRET"][r, 15, 0] = data["TJEF"][r, 7, 0]
            if "FTT-Fr" in ftt_modules:
                # TODO: fix something to account for missing data if freight turned off
                data["FRET"][r, 15, 0] += data["ZJEF"][r, 7, 0]
            
            if "FTT-H" in ftt_modules:            
                data["FRET"][r, 18, 0] = data["HJEF"][r, 7, 0]
            
           
    
                
            
            # Compute the changes in actual electricity demand (normally in FTT.f90, put here for testing purposes)
            
            data["MEWD"][r, 7, 0] = np.sum(data["FRET"][r, :, 0]) * 41.868/1000 #/3.6 electricity
    else:
        print("FTT-P not included, so MEWD unaffected")    
        
    return data

def electricity_demand_price_elasticity(data, titles, histend, year, ftt_modules):
    """
    Compute electricity demand changes using the
    econometrically estimated elasticity. This is found in the X databank 
    under the BFRE estimated parameters. 
    
    The equation in E3ME is in COINT. I think it 
    """
    pass

