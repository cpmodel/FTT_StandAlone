# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:13:09 2024

@author: Femke Nijsse
"""



import numpy as np

def EV_truck_mandate(EV_mandate, twsa, tews, rflt, year, n_years=11):
    """ 
    Sets a mandate of growing exogenous sales to 2035. At that point, separately,
    regulations comes into play that outregulates fossil technologies. 
    
    A few exceptions:
        * When not enough fossil cars are left, no mandate
        * When there are too many EVs, no mandate
    """
    
    
    fossil_techs = list(range(10))
    EV_techs = [12, 13]
    
    
    if EV_mandate[0,0,0] == 1:
        if year in range(2025, 2025 + n_years):
        
            frac = 1/n_years            # Fraction decrease per year
            n = year - 2024
            
            yearly_replacement = 1/15 * 0.8
            
            # In 2035, the sum should be 80% of sales. Lifetime = ~15y
            sum_twsa = np.full(twsa.shape[0], frac * n * yearly_replacement)   
            
            sum_ff = np.sum(tews[:, fossil_techs], axis=(1, 2))
            sum_EV = np.sum(tews[:, EV_techs], axis=(1, 2))
            sum_twsa = np.where(sum_ff < 1.8 * sum_twsa, 0, sum_twsa)          # Stop when there is too little fossil fuels to replace
            sum_twsa = np.where(sum_EV > 1 - 2 * sum_twsa, 0, sum_twsa)        # Also stop if virtually all shares are heat pumps already
            
            # Compute fractions for each heat pump, ff technique, based on fraction of shares
            # Ensure no division by zero (note, fossil fuel second option doesn' matter, as we've already scaled sum_twsa to sum_ff)
            backup_shares = np.tile(np.array([0.9, 0.1]), (twsa.shape[0], 1))
            frac_EVs = np.where(sum_EV[:, None] > 0, tews[:, EV_techs, 0] / sum_EV[:, None], backup_shares) 
            frac_fossils = np.where(sum_ff[:, None] > 0, tews[:, fossil_techs, 0] / sum_ff[:, None], tews[:, fossil_techs, 0])

            twsa[:, fossil_techs, 0] = -sum_twsa[:, None] * frac_fossils * rflt[:, :, 0]
            twsa[:, EV_techs, 0] = sum_twsa[:, None] * frac_EVs * rflt[:, :, 0]
            
            
            
    # Else: return hswa unchanged
    return twsa
