"""
=========================================
ftt_p_wacc.py
=========================================
Power  FTT WACC module.
#################################

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros

Functions included:
    - get_wacc
        Calculate wacc

"""

# Standard library imports
import numpy as np

# Local library imports
from SourceCode.support.divide import divide

# Local library imports
from SourceCode.support.divide import divide


# -----------------------------------------------------------------------------
# --------------------------- WACC function -----------------------------------
# -----------------------------------------------------------------------------

def get_wacc(data, titles,year,histend):
    """
    Calculate the weighted average cost of capital (WACC) for different technologies and regions.

    This function adjusts the WACC based on data from 2019 considering the interest rates, 
    the learning rate for renewables, and competition for non-renewables.
    
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

        
        
    if year <= histend['DCOC']:
        
        for r in range(len(titles['RTI'])): #loop for regions
            for tech in range(len(titles['T2TI'])): #loop for technologies
                
                data["BCET"][r, tech, 16] = data["DCOC"][r, tech, 0]
        # data["BCET"][:, :, 16] = data["DCOC"][:, :, 0]  #equivalent loop for countries and technologies
    
    
    else:
        
        WACC2015 = data['DCOC2015'][:,:,0]
        DPIR2015 = data['DPIR2015'][:,0,0]
        DPIR = data["DPIR"][:,0,0]
        gen = data['MEWG'][:,:,0] 
        gen2015 = data['MEWG2015'][:,:,0]    
        gensh = data['MEWS'][:,:,0]    #generation share
        gensh2015 = data['MEWS2015'][:,:,0]     
        WACC2015_exclIR = WACC2015 - DPIR2015
            
        for r in range(len(titles['RTI'])): #loop for regions
            for tech in range(len(titles['T2TI'])): #loop for technologies
            #Renewables: every generation doubling WACC lowers with 5%
            #Non-renewable: estimates of WACC change
            
                if tech in [16, 17, 18]:
                    doubling = np.floor(np.log2(gen[r,tech]/gen2015[r,tech]))
                    learning_rate = 0.05
                    WACC_change = -WACC2015_exclIR*doubling*learning_rate
                    
                else:
                    DCCE = data['DCCE'][r,0,0]  #WACC change estimates for non renewables
                    genshchange = (gensh[r,tech] / gensh2015[r,tech] - 1)
                    WACC_change = WACC2015_exclIR*genshchange*DCCE
                    
                    
                WACC =  WACC2015_exclIR + WACC_change + DPIR
                
                
                data["DCOC"][r,tech,0] = WACC
    return data