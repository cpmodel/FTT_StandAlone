# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:48:27 2025

@author: pv
"""

import numpy as np

def calc_h2_demand(data):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Split the market into a grey and green market
    # From CLEAFS
    gr_nh3_fert_share_cleafs = data['FERTD'][:, 0, 0] / data['FERTD'][:, :, 0].sum(axis=1)
    gr_nh3_fert_lvl_cleafs = gr_nh3_fert_share_cleafs * data['HYD1'][:, 0, 0]
    
    # Apply mandate
    gr_nh3_fert_lvl = np.maximum(data['WDM1'][:, 0, 0] * data['HYD1'][:, 0, 0], gr_nh3_fert_lvl_cleafs)
    gr_nh3_chem_lvl = data['WDM2'][:, 0, 0] * data['HYD2'][:, 0, 0]
    gr_meoh_chem_lvl = data['WDM3'][:, 0, 0] * data['HYD3'][:, 0, 0]
    gr_h2oil_chem_lvl = data['WDM4'][:, 0, 0] * data['HYD4'][:, 0, 0]
    gr_h2ener_chem_lvl = data['WDM5'][:, 0, 0] * data['HYD5'][:, 0, 0]
    
    # Total size of the green market
    data['WGRM'][:, 0, 0] = (gr_nh3_fert_lvl + gr_nh3_chem_lvl + gr_meoh_chem_lvl + 
                             gr_h2oil_chem_lvl + gr_h2ener_chem_lvl)

    
    return data