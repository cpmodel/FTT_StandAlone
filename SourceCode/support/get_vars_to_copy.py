# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 11:54:46 2025

@author: Femke
"""

import numpy as np


def get_loop_vars_to_copy(data, data_dt, domain, sector):
    '''
    Get variables that need to be copied within the time loop
    
    Only copy domain variables, where data and data_dt are not the same
    '''
    all_vars = data_dt.keys()
    
    # Only copy over variables from the relevant sector
    domain_vars = [var for var in all_vars if domain[var] == sector]
    
    # Only copy over variables that differ
    vars_to_copy = [var for var in domain_vars if not np.array_equal(data_dt[var], data[var])]
    
    return vars_to_copy

def get_domain_vars_to_copy(time_lag, domain, sector):
    '''
    Get variables that need to be copied from the lagged variables in this domain
    
    '''
    all_vars = time_lag.keys()
    
    # Only copy over variables from the relevant sector
    domain_vars = [var for var in all_vars if domain[var] == sector]
    
    return domain_vars