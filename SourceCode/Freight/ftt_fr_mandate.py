# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:13:09 2024

@author: Femke Nijsse
"""



import numpy as np

def EV_truck_mandate(EV_mandate, zwsa, zews, rflz, year, n_years=16):
    """ 
    Sets a mandate of growing exogenous sales to 2035. At that point, separately,
    regulations comes into play that outregulates fossil technologies. 
    
    A few exceptions:
        * When not enough fossil cars are left, no mandate
        * When there are too many EVs, no mandate
    """
    
    if EV_mandate[0,0,0] > 1:
        years_on = int(EV_mandate[0,0,0] - 2025)
    else:
        years_on = n_years
    
    for veh_class in range(5):               # two-wheelers, LDV, MDV, HDV, buses
        
        fossil_techs = list(range(25))
        EV_techs = list(range(30, 35))    
        
        if EV_mandate[0,0,0] != 0:
            
            if year in range(2025, 2025 + years_on):
            
                n = year - 2024
                x = n/n_years
                yearly_replacement = 1 / 15
                exogenous_sigmoid = x / (x + 0.8)   # Sigmoid, as endogenous replacements take increasing share
                
                # In 2040, the sum should be 80% of sales.
                sum_zwsa_share = np.full(zwsa.shape[0], exogenous_sigmoid * yearly_replacement)
                
                # Sum of ICE/EVs in each vehicle class
                
                sum_ff = np.sum(zews[:, veh_class:fossil_techs[-1]:5], axis=(1, 2))
                sum_EV = np.sum(zews[:, veh_class+EV_techs[0]:EV_techs[-1]:5], axis=(1, 2))
                sum_zwsa_share = np.where(sum_ff < 1.8 * sum_zwsa_share, 0, sum_zwsa_share)          # Stop when there is too little fossil fuels to replace
                sum_zwsa_share = np.where(sum_EV > 1 - 2 * sum_zwsa_share, 0, sum_zwsa_share)        # Also stop if virtually all shares are EVs already
                
                # Compute fractions for each truck, ff technique, based on fraction of shares
                # Ensure no division by zero (note, fossil fuel second option doesn't matter, as we've already scaled sum_zwsa to sum_ff)
                backup_EV_shares = np.tile(np.array([1.0]), (zwsa.shape[0], 1))
                backup_fossil_shares = np.tile(np.array([0.08, 0.0, 0.9, 0, 0.02]), (zwsa.shape[0], 1))
                
                # Fraction of each EV type by region (backup_shares if dividing by zero)
                frac_EVs = np.divide(zews[:, EV_techs[0]+veh_class, 0],  sum_EV, 
                                     out=backup_EV_shares[:, 0], where=sum_EV > 0)
                frac_fossils =  np.divide(zews[:, veh_class:fossil_techs[-1]+1:5, 0],  sum_ff[:, None],
                                          out=backup_fossil_shares, where=sum_ff[:, None] > 0)
    
                zwsa[:, EV_techs[0]+veh_class, 0] = sum_zwsa_share * frac_EVs * rflz[:, veh_class, 0]
                zwsa[:, veh_class:fossil_techs[-1]+1:5, 0] = -sum_zwsa_share[:, None] * frac_fossils * rflz[:, veh_class]
               
            
    # Else: return zswa unchanged
    return zwsa
