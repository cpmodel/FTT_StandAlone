# -*- coding: utf-8 -*-
"""


"""

# Third party imports
import numpy as np
from SourceCode.support.divide import divide


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
        for m in range(len(titles['TFTI'])):
        
            # CAPEX from IEA. Assume 2024 USD
            capex = 770 / (data['PRSC'][r, 0, 0] * data['EX'][r, 0, 0]) / 1.18
            opex = 0.03 * capex
            discount_rate = data['DISCOUNTRATES'][r, 0, 0]
            
            npv_in = 0.0
            dnpv_in = 0.0
            npv_out = 0.0
        
            for t in range(lt+bt+1):
                
                if t <= bt:
                    
                    # 0.85 represents the expected capacity factor
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

# %% CBAM function
# --------------------------------------------------------------------------
# ----------------------- CBAM cost function -------------------------------
# --------------------------------------------------------------------------
def get_cbam(data, h2_input, titles):
    
    # Difference in carbon prices
    carbon_price = data['HYPR'][:, 0, 0].copy()
    
    # Create matrices for subtraction
    carbon_price_at_importer = np.zeros((len(titles['RTI']), len(titles['RTI'])))
    carbon_price_at_importer += carbon_price[None, :]
    carbon_price_at_exporter = np.zeros((len(titles['RTI']), len(titles['RTI'])))
    carbon_price_at_exporter += carbon_price[:, None]
    
    # Get the bilateral difference
    # in 2010Euro per tCO2
    cbam_penalty_rate = carbon_price_at_importer - carbon_price_at_exporter
    # Remove negative numbers
    cbam_penalty_rate[cbam_penalty_rate<0.0] = 0.0
    
    # Get emission intensity as a split by 
    emission_intensity_green = divide(np.sum(data['WGWG'][:, :, 0]  * data['HYEF'][:, :, 0] * h2_input, axis=1)
                                      , np.sum(data['WGWG'][:, :, 0]* h2_input, axis=1))
    emission_intensity_grey = divide(np.sum(data['WBWG'][:, :, 0]  * data['HYEF'][:, :, 0] * h2_input, axis=1)
                                     , np.sum(data['WBWG'][:, :, 0]* h2_input, axis=1))
    # Grey market emission intensity matrix
    # Get differences between importers and exporters
    emis_intensity_grey_at_importer = np.zeros((len(titles['RTI']), len(titles['RTI'])))
    emis_intensity_grey_at_importer += emission_intensity_grey[None, :]
    emis_intensity_grey_at_exporter = np.zeros((len(titles['RTI']), len(titles['RTI'])))
    emis_intensity_grey_at_exporter += emission_intensity_grey[:, None]    
    # Get the difference
    emis_intensity_grey_diff = emis_intensity_grey_at_importer - emis_intensity_grey_at_exporter
    # Remove negative numbers
    emis_intensity_grey_diff[emis_intensity_grey_diff<0.0] = 0.0    
    
    # Green market emission intensity matrix
    # Get differences between importers and exporters
    emis_intensity_green_at_importer = np.zeros((len(titles['RTI']), len(titles['RTI'])))
    emis_intensity_green_at_importer += emission_intensity_green[None, :]
    emis_intensity_green_at_exporter = np.zeros((len(titles['RTI']), len(titles['RTI'])))
    emis_intensity_green_at_exporter += emission_intensity_green[:, None]    
    # Get the difference
    emis_intensity_green_diff = emis_intensity_green_at_importer - emis_intensity_green_at_exporter
    # Remove negative numbers
    emis_intensity_green_diff[emis_intensity_green_diff<0.0] = 0.0    
    
    # Now apply carbon price to emission intensities
    # CBAM is only applied if the carbon price is higher in the importing region,
    # and if the emission intensity is also higher.
    data['NH3CBAM'][:, :, 1] = cbam_penalty_rate * emis_intensity_grey_diff
    data['NH3CBAM'][:, :, 0] = cbam_penalty_rate * emis_intensity_green_at_exporter
    
    return data
    

# %% Delivery costs
# --------------------------------------------------------------------------
# ------------------- Delivery cost function -------------------------------
# --------------------------------------------------------------------------
def get_delivery_cost(data, time_lag, titles):
    
    # Stack production costs (=levelised cost) across matrix
    
    # Green market delivery costs
    data['NH3DELIVCOST'][:, :, 0] = ((time_lag['NH3LC'][:, 0, 0, None] 
                                          * (1.0 + data['NH3TRF'][:, :, 0]))
                                     + data['NH3TCC'][:, :, 0] 
                                     + data['NH3CBAM'][:, :, 0]
                                     )
    # Grey market delivery costs
    data['NH3DELIVCOST'][:, :, 1] = ((time_lag['NH3LC'][:, 1, 0, None] 
                                          * (1.0 + data['NH3TRF'][:, :, 0]))
                                     + data['NH3TCC'][:, :, 0] 
                                     + data['NH3CBAM'][:, :, 1]
                                     )
    
    return data