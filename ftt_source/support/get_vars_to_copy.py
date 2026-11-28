# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 11:54:46 2025

@author: Femke
"""

def get_domain_vars_to_copy(time_lag, domain, sector):
    '''
    Get variables that need to be copied from the lagged variables in this domain
    
    '''
    all_vars = time_lag.keys()
    
    # Only copy over variables from the relevant sector
    domain_vars = [var for var in all_vars if domain[var] == sector]
    
    return domain_vars