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
            capex = 770 / (data['PRSC'][33, 0, 0] * data['EX'][33, 0, 0]) / 1.18
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
                    
                    h2_cost = h2_input * data['WPPR'][r, m, 0] * 1e3
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
    emission_intensity_green = divide(np.sum(data['WGWG'][:, :, 0]  *(data['HYEF'][:, :, 0] + data['HYEFINDIRECT'][:, :, 0]) * h2_input, axis=1)
                                      , np.sum(data['WGWG'][:, :, 0]* h2_input, axis=1)) + data['NH3EFINDIRECT'][:, 0, 0]
    emission_intensity_grey = divide(np.sum(data['WBWG'][:, :, 0]  * (data['HYEF'][:, :, 0] + data['HYEFINDIRECT'][:, :, 0]) * h2_input, axis=1)
                                     , np.sum(data['WBWG'][:, :, 0]* h2_input, axis=1)) + data['NH3EFINDIRECT'][:, 0, 0]
    # Grey market emission intensity matrix
    # Get differences between importers and exporters
    emis_intensity_grey_at_exporter = np.zeros((len(titles['RTI']), len(titles['RTI'])))
    emis_intensity_grey_at_exporter += emission_intensity_grey[:, None]    
    
    # Green market emission intensity matrix
    # Get differences between importers and exporters
    emis_intensity_green_at_exporter = np.zeros((len(titles['RTI']), len(titles['RTI'])))
    emis_intensity_green_at_exporter += emission_intensity_green[:, None]    

    
    # Now apply carbon price to emission intensities
    # CBAM is only applied if the carbon price is higher in the importing region,
    # and if the emission intensity is also higher.
    data['NH3CBAM'][:, :, 1] = cbam_penalty_rate * emis_intensity_grey_at_exporter * data['NH3CBAMSWITCH'][None, :, 0, 0]
    data['NH3CBAM'][:, :, 0] = cbam_penalty_rate * emis_intensity_green_at_exporter * data['NH3CBAMSWITCH'][None, :, 0, 0]
    
    return data
    

# %% Delivery costs
# --------------------------------------------------------------------------
# ------------------- Delivery cost function -------------------------------
# --------------------------------------------------------------------------
def get_delivery_cost(data, time_lag, titles):
    
    # Convert energy prices to the correct units
    # Electricity price in Euro/kWh 
    electricity_price = data['PFRE'][:, 5, 0]/(data['PRSC'][:, 0, 0]/data['EX'][:, 0, 0])/11630
    # Heavy oil price in Euro/toe
    heavy_oil_price = data['PFRO'][:, 5, 0]/(data['PRSC'][:, 0, 0]/data['EX'][:, 0, 0])
    
    # Adjust transportation costs by accounting for electricity consumption at
    # the export and import terminals
    # Import terminals consume 0.003 kWh/tNH3 and export terminals consume 
    # 0.001 kWh/tNH3
    
    # Caluculate heavy fuel oil costs
    # The fuel consumption estimates are per tanker
    # We need to know what the fuel consumption is per tNH3 traded
    # 1 tanker can hold around 54049 tNH3
    data['NH3TRANSPORTFUELCOST'][:, :, 0] = (data['NH3TRANSPORTFUELCONSUMPTION'][:, :, 0] 
                                             / 54049
                                             * heavy_oil_price[:, None]
                                             )
    
    # ============= Transport emission factors ================================
    data['NH3TRANSPORTEMISSIONFACTOR'][:, :, 0] = (data['NH3TRANSPORTFUELCONSUMPTION'][:, :, 0] # tHFO/tNH3
                                                   * 3.114  # kg CO2/ kg HFO = tCO2/tHFO
                                                   )
    
    # =============== Transport emission costs ================================
    
    # Apply carbon price to transport emission intensity
    data['NH3TRANSPORTEMISSIONCOST'][:, :, 0] = (data['NH3TRANSPORTEMISSIONFACTOR'][:, :, 0] 
                                                 * data['HYPR'][None, :, 0, 0]
                                                 * data['ETSEFFECTONTRANSPORT'][:, :, 0]
                                                 )
    # Remove carbon penalties for intra-EU trade
    data['NH3TRANSPORTEMISSIONCOST'][:31, :31, 0] = 0.0
    
    # Total transportation costs are a function of:
        # 1. NH3 production costs
        # 2. Electricity costs at the export terminal
        # 3. Electricity costs at the import terminal
        # 4. Heavy fuel oil costs due to shipping
        # 5. Non-energy related shipping costs
        # 6. Transport emissions costs
        
    data['NH3TCCout'][:, :, 0]  = (data['NH3TCC'][:, :, 0]                      # Ship costs and terminal costs
                                   + 0.003*0.179*electricity_price[None, :]     # Export terminal electricity costs
                                   + 0.001*0.179*electricity_price[:, None]     # Import terminal electricity costs
                                   + data['NH3TRANSPORTFUELCOST'][:, :, 0]      # Heavy fuel oil costs 
                                   + data['NH3SHIPPINGCOST'][:, :, 0]           # Shipping costs
                                   + data['NH3TRANSPORTEMISSIONCOST'][:, :, 0]
                                   )
    
    # Store shipping costs (excl fuel) in a interrogatable variables
    data['NH3SHIPPINGCOSTout'][:, :, 0] = np.copy(data['NH3SHIPPINGCOST'][:, :, 0])
    
    # ============= Green market Bilateral trade costs ========================
    # Green market delivery costs
    data['NH3DELIVCOST'][:, :, 0] = ((time_lag['NH3LC'][:, 0, 0, None] 
                                          * (1.0 + data['NH3TRF'][:, :, 0]))
                                     + data['NH3TCCout'][:, :, 0] 
                                     + data['NH3CBAM'][:, :, 0]
                                     )
    # Just the bilateral trade costs w/o production
    data['NH3BILACOST'][:, :, 0] = ((time_lag['NH3LC'][:, 0, 0, None] 
                                          * (data['NH3TRF'][:, :, 0]))
                                     + data['NH3TCCout'][:, :, 0] 
                                     + data['NH3CBAM'][:, :, 0]
                                     )    
    
    # ============= Grey market Bilateral trade costs ========================
    # Grey market delivery costs
    data['NH3DELIVCOST'][:, :, 1] = ((time_lag['NH3LC'][:, 1, 0, None] 
                                          * (1.0 + data['NH3TRF'][:, :, 0]))
                                     + data['NH3TCCout'][:, :, 0]
                                     + data['NH3CBAM'][:, :, 1]
                                     )
    
    # Just the bilateral trade costs w/o production
    data['NH3BILACOST'][:, :, 1] = ((time_lag['NH3LC'][:, 1, 0, None] 
                                          * (data['NH3TRF'][:, :, 0]))
                                     + data['NH3TCCout'][:, :, 0] 
                                     + data['NH3CBAM'][:, :, 1]
                                     )  
    

    
    
    
    return data