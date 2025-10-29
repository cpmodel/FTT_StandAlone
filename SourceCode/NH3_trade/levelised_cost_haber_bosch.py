# -*- coding: utf-8 -*-
"""


"""

# Third party imports
import numpy as np


# %% LOCNH3
# --------------------------------------------------------------------------
# -------------------------- LOCNH3 function -------------------------------
# --------------------------------------------------------------------------

def get_lchb(data, h2_input, titles):
    
    # Hard-coded inputs
    bt = 3      # Assumed
    lt = 30     # Assumed

    # Loop over regions
    for r in range(len(titles['RTI'])):
        
        # Loop over market (grey or green)
        for m in range(len(titles('TFTI'))):
        
            # CAPEX from IEA. Assume 2024 USD
            capex = 770 / (data['PRSC'][r, 0, 0] * data['EX'][r, 0, 0]) / 1.18
            opex = 0.03 * capex
            discount_rate = 0.08
            
            npv_in = 0.0
            dnpv_in = 0.0
            npv_out = 0.0
        
            for t in range(lt+bt+1):
                
                if t <= bt:
                    
                    ic = capex / 0.85
                    dic = 0.05 * ic
                    
                    ft = 0.0
                    dft = 0.0
                    
                    h2_cost = 0.0
                    dh2_cost = 0.0
                    
                    omt = 0.0
                    domt = 0.0
                    
                    pt = 0.0
                    
                else:
                    
                    ic = 0.0
                    dic = 0.0
                    
                    ft = 2.2 / 41.868 * data['PFRE'][r, 5] / (data['PRSC'][r, 0, 0] * data['EX'][r, 0, 0])
                    dft = 0.1 * ft
                    
                    h2_cost = h2_input * data['WPPR'][r, m, 0]
                    dh2_cost = 0.01*h2_cost
                    
                    omt = opex
                    domt = 0.1 * omt
                    
                    pt = 1.0
                    
                npv_in += (ic + omt + h2_cost + ft) / (1 + discount_rate) ** t
                dnpv_in += 1.414 * np.sqrt(dic**2 + domt**2 + dh2_cost**2 + dft**2) / (1 + discount_rate)**t
                npv_out += (pt) / (1 + discount_rate) ** t
                
            data['NH3LC'][r, m, 0] = npv_in/npv_out
            data['NH3LCSD'][r, m, 0] = dnpv_in/npv_out


    return data

# %% Delivery costs
# --------------------------------------------------------------------------
# --------------------------- DCNH3 function -------------------------------
# --------------------------------------------------------------------------
def get_delivery_cost(data, titles):
    
    delivery_cost = np.zeros((len(titles['RTI']), len(titles['RTI'])))
    
    # Stack production costs (=levelised cost) across matrix
    delivery_cost += data['NH3LC'][:, :, 0] * (1.0 + data['NH3TRF'][:, :, 0]) + data['NH3TCC'][:, :, 0] + data['NH3CBAM'][:, :, 0]
    
    # Add transportatition costs
    
    
    
    return data