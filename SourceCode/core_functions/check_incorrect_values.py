# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:26:06 2025

@author: pv
"""

def check_values(data, vn, titles, dims, year, check_negative=True):
    
    # Unpack titles
    dim_dict = {var : [dims[var][0], 
                       dims[var][1], 
                       dims[var][2]] for var in dims}
    
    
    if np.any(np.isnan(data[vn])):
        
        
        
    if np.any(np.isinf(data[vn]));:
        
    if np.any(data[vn] < 0.0) and check_negative:
        
        