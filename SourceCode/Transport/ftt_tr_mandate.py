# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:25:47 2024

@author: Femke Nijsse
"""



import numpy as np

def EV_mandate(EV_mandate, twsa, tews, rflt, year, n_years=11):
    """ 
    Sets a mandate of growing exogenous sales to 2035. At that point, separately,
    regulations comes into play that outregulates fossil technologies. 
    
    A few exceptions:
        * When not enough fossil cars are left, no mandate
        * When there are too many EVs, no mandate
    """
    
    fossil_techs = list(range(15))
    EV_techs = [18, 19, 20]
    
    
    if EV_mandate[0,0,0] == 1:
        if year in range(2025, 2025 + n_years):
        
            frac = 1/n_years            # Fraction decrease per year
            n = year - 2024
            
            yearly_replacement = 1/13 * 0.8
            
            # In 2035, the sum should be 80% of sales.
            sum_twsa_share = np.full(twsa.shape[0], frac * n * yearly_replacement)   
            
            sum_ff = np.sum(tews[:, fossil_techs], axis=(1, 2))
            sum_EV = np.sum(tews[:, EV_techs], axis=(1, 2))
            sum_twsa_share = np.where(sum_ff < 1.8 * sum_twsa_share, 0, sum_twsa_share)          # Stop when there is too little fossil fuels to replace
            sum_twsa_share = np.where(sum_EV > 1 - 2 * sum_twsa_share, 0, sum_twsa_share)        # Also stop if virtually all shares are heat pumps already
            
            # Compute fractions for each heat pump, ff technique, based on fraction of shares
            # Ensure no division by zero (note, fossil fuel second option doesn' matter, as we've already scaled sum_twsa to sum_ff)
            backup_shares = np.tile(np.array([0.3, 0.4, 0.3]), (twsa.shape[0], 1))
            # fraction of each EV type by region (backup_shares if dividing by zero)
            frac_EVs = np.divide(tews[:, EV_techs, 0],  sum_EV[:, None], out=backup_shares, where=sum_EV[:, None] > 0)
            frac_fossils =  np.divide(tews[:, fossil_techs, 0],  sum_ff[:, None])

            twsa[:, fossil_techs, 0] = -sum_twsa_share[:, None] * frac_fossils * rflt[:, :, 0]
            twsa[:, EV_techs, 0] = sum_twsa_share[:, None] * frac_EVs * rflt[:, :, 0]
            test = 1
            
            
    # Else: return hswa unchanged
    return twsa
