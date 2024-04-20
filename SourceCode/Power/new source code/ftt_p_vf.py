# Standard library imports
from math import sqrt
import os
import copy
import sys
import warnings

# Third party imports
import pandas as pd
import numpy as np

# Local library imports
from SourceCode.support.divide import divide


# -----------------------------------------------------------------------------
# --------------------------- value factor function -----------------------------------
# -----------------------------------------------------------------------------

def get_vf(data,titles,year,histend):
    """
    Value Factor expresses the generation weighted price a specific technology
    can obtain compared to the general generation weigthed price

    Parameters
    ----------
    data : dictionary
        Data is a container that holds all cross-sectional (of time) for all
        variables. Variable names are keys and the values are 3D NumPy arrays.
    titles : TYPE
        DESCRIPTION.
    year : TYPE
        DESCRIPTION.
    histend : TYPE
        DESCRIPTION.

    Returns
    -------
    data : dictionary

    """
    for r in range(len(titles['RTI'])): #loop for regions
            for tech in range(len(titles['T2TI'])): #loop for technologies
                if year <= histend['DPVF']:
        
                    data['DPVF'][r, tech, 0] = data['DPVF'][r, tech, 0]
                
                else:
                    #value factor 2023 as base
                    DPVF2023 = data['DPVF2023'][r,tech,0]     #value factor in 2013
                    gensh2023 = data['DGEN2023'][r,tech,0]    #generation share in 2O23
                    gensh = data['MEWS'][r,tech,0]            #generation share
                    
                    #changes according to shares
                    estimate = data['DVFE'][r,tech,0] #value factor estimates
                    
                    #change in generation shares since 2013
                    # power share of capacity MEWS 
                    #shareschange function -> change: last data divided by shares 2023 -1
                    changeshares = (gensh/gensh2023)-1
                    
                    VF = DPVF2023*changeshares*estimate
                
                
                data['DPVF'][r,tech,0] = VF

return data