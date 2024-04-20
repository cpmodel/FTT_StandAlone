"""
=========================================
ftt_p_cod.py
=========================================
Power  FTT CoD module.
#################################

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros

Functions included:
    - get_cod
        Calculate wacc

"""

# Standard library imports
import numpy as np

# Local library imports
from SourceCode.support.divide import divide

# Local library imports
from SourceCode.support.divide import divide


# -----------------------------------------------------------------------------
# --------------------------- CoD function -----------------------------------
# -----------------------------------------------------------------------------

def get_CoD(data, titles,year,histend):
    """
    Calculate the cost of debt (CoD) for different technologies and regions.

    This function adjusts the CoD based on data from 2019 considering the interest rates.
    
    Parameters
    -----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) for all
        variables. Variable names are keys and the values are 3D NumPy arrays.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.
    year : int
        The current year of simulation.
    histend : dict
        A dictionary with historical end years for various parameters, e.g., 'DCOC'.

    Returns
    ----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.

    Notes
    ---------
    Additional notes if required.
    """
        
        
    if year <= histend['DCOD']:
        
        for r in range(len(titles['RTI'])): #loop for regions
            for tech in range(len(titles['T2TI'])): #loop for technologies
                
                data["DCOD"][r, tech, 0] = data["DCOD"][r, tech, 0]
        # data["BCET"][:, :, 16] = data["DCOC"][:, :, 0]  #equivalent loop for countries and technologies
    
    
    else:
        for r in range(len(titles['RTI'])): #loop for regions
            for tech in range(len(titles['T2TI'])): #loop for technologies
            #Renewables: every generation doubling WACC lowers with 5%
            #Non-renewable: estimates of WACC change
                DCOD2015 = data['DCOD2015'][:,:,0]
                DPIR2015 = data['DPIR2015'][:,0,0]
                DPIR = data["DPIR"][:,0,0]
                    
                DCOD2015_exclIR = DCOD2015 - DPIR2015
                
                DCOD = DCOD2015_exclIR + DPIR
                               
                data["DCOD"][r,tech,0] = DCOD
    return data