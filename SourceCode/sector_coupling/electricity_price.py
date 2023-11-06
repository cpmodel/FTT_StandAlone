# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:52:54 2023

@author: Femke Nijsse
"""

def relative_electricity_price(data, titles, histend, year, ftt_modules):
    """ "This function takes as input the electricity price from FTT-P, and 
    returns the relative price of electricity at year x"""
    
    if year >= histend["MEWG"]:
        print(year)
    
    pass

def electricity_demand_price_elasticity(data, titles, histend, year, ftt_modules):
    """
    Compute electricity demand changes using the
    econometrically estimated elasticity. This is found in the X databank 
    under the BFRE estimated parameters. 
    
    The equation in E3ME is in COINT. I think it 
    """
    pass

