# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:46:06 2023

@author: Femke Nijsse
"""

def input_functions_message(scen, var, dims, read, timeline="None"):
    print("Critical error reading input functions:")
    print(f'Variable is {var} with dimensions {dims[var]}')
    print(f"Scenario is {scen}")
    print(f"Timeline is {timeline}")
    print(f"file that's being read in is: {read}")
    
    