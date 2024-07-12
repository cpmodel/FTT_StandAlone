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

def set_linear_coal_phase_out(coal_phaseout, mwka, mwka_lag, mewk_lag, year, n_years=11, techs=[2, 4]):
    
    
    if coal_phaseout[0,0,0] == 1:
        frac = 1/n_years            # Fraction decrease per year
        n = year - 2024
        if year == 2025:
            mwka[:, techs] = mewk_lag[:, techs] * (1 - frac * n)/(1 - frac * (n - 1))
        elif year in range(2026, 2025 + n_years):
            mwka[:, techs] = mwka_lag[:, techs] * (1 - frac * n)/(1 - frac * (n - 1))
        elif year > 2025 + n_years - 1:
            mwka[:, techs] = np.zeros_like(mwka_lag[:, techs])
    
    # Else: return mwka unchanged
    
    
    return mwka