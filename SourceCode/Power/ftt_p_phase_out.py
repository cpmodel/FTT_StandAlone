# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:32:26 2024

@author: Femke Nijsse

Linearly decreasing coal phase-out worldwide

We use recursive maths, as we do not store the initial MEWK value
The equation for a linearly decreasing line is:
    
    Xn = X_{n-1} * (1-0.1n)/(1-0.1(n-1)). After 10 year, this equals 0. 
"""

import numpy as np

def set_linear_coal_phase_out(coal_phaseout, mwka, mwka_lag, mewk_lag, year, techs=[2, 4]):
    '''For rich countries, coal phase-out happens up to 2035. For developing countries, 
    there is more time and they get to 2045.'''
    
    # Return unchanged if the coal phase-out is turned off
    if coal_phaseout[0,0,0] == 0:
        return mwka
    
    
    def set_MWKA(mwka, regions, n_years):
        
        frac = 1/n_years            # Fraction decrease per year
        n = year - 2024
        indices = np.ix_(regions, techs)
        
        if year == 2025:
            mwka[indices] = mewk_lag[indices] * (1 - frac * n)/(1 - frac * (n - 1))
        elif year in range(2026, 2025 + n_years):
            mwka[indices] = mwka_lag[indices] * (1 - frac * n)/(1 - frac * (n - 1))
        elif year > 2025 + n_years - 1:
            mwka[indices] = np.zeros_like(mwka_lag[indices])
        
        return mwka
        
    rich_region_indices = [i for i in range(38) if i != 31]
    other_regions = [i for i in range(71) if i not in rich_region_indices]
    mwka = set_MWKA(mwka, rich_region_indices, n_years=11)
    mwka = set_MWKA(mwka, other_regions, n_years=21)

    
    return mwka