
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:09:17 2023

@author: Femke Nijsse
"""

from copy import deepcopy 

def compute_hjef(data, titles):
    """Computes the demand for fuels for the residential sector in thousand toe.
    This includes a heating part, and a non-heating part. 
    
    """
    # hjef = deepcopy(data["HJEF"])
    # for r in range(len(titles['RTI'])):          # Loop over regions
    #     for fuel in range(len(titles['JTI'])):   # Loop over fuels
    #         if data['HJFC'][r, fuel, 0] > 0.0:
    #             # TODO: for electricity, it's absurd if we scale non-heat pumps with HJFC. 
    #             # That would mean that non-heat electricity grows at the same pace as heat-pump electricity
                
    #             #data['HJEF'][r, fuel, 0] = data['HJHF'][r, fuel, 0] / data['HJFC'][r, fuel, 0]
    #             hjef[r, fuel, 0] = data["FU14B"][r, fuel, 0] * data['HJHF'][r, fuel, 0] / data["FU14A"][r, fuel, 0] / data['HJFC'][r, fuel, 0]
    pass
    
    return deepcopy(data["HJEF"])