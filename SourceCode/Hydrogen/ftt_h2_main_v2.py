# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:30 2024

@author: PV

=========================================
ftt_h2_main.py
=========================================
Hydrogen FTT module.
####################################

This is the main file for FTT: Hydrogen, which models technological
diffusion of hydrogen production technologies due to simulated investment decision making.
Producers compare the **levelised cost of hydrogen**, which leads to changes in the
market shares of different technologies.


Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Element-wise divide which replaces divide-by-zeros with zeros
    - `estimation <econometrics_functions.html>`__
        Predict future values according to the estimated coefficients.

Functions included:
    - solve
        Main solution function for the module
    - get_lcoh2
        Calculate levelised cost of hydrogen

"""
# Third party imports
import numpy as np
import pandas as pd

# Local library imports
from SourceCode.support.divide import divide
from SourceCode.Hydrogen.ftt_h2_lcoh import get_lcoh as get_lcoh2
from SourceCode.Hydrogen.cost_supply_curve import calc_csc
from SourceCode.Hydrogen.ftt_h2_pooledtrade import pooled_trade
from SourceCode.core_functions.substitution_frequencies import sub_freq
from SourceCode.core_functions.substitution_dynamics_in_shares import substitution_in_shares, innovator_effect
from SourceCode.core_functions.capacity_growth_rate import calc_capacity_growthrate
from SourceCode.Hydrogen.ftt_h2_energy_costs import calc_ener_cost
from SourceCode.Hydrogen.h2_demand import calc_h2_demand
from SourceCode.Hydrogen.ftt_h2_green_cost_factors import calc_green_cost_factors
from SourceCode.Hydrogen.energy_and_emissions import calc_emis_rate, calc_ener_cons
from SourceCode.NH3_trade.nh3_cost_functions import get_lchb, get_cbam, get_delivery_cost
from SourceCode.NH3_trade.nh3_trade_dynamics import calculate_nh3_trade

# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, iter_lag, titles, histend, year, domain, dimensions, scenario):
    """
    Main solution function for the module.

    Add an extended description in the future.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution
    time_lag: type
        Description
    iter_lag: type
        Description
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of histrorical data by variable
    year: int
        Curernt/active year of solution
    specs: dictionary of NumPy arrays
        Function specifications for each region and module
    dimensions: dictionary of 

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    Notes
    ---------
    This function should be broken up into more elements in development.
    """

     # Categories for the cost matrix (BHTC)
    c7ti = {category: index for index, category in enumerate(titles['C7TI'])}
    jti = {category: index for index, category in enumerate(titles['JTI'])}
    hyti = {category: index for index, category in enumerate(titles['HYTI'])}
    
    green_idx = 0
    grey_idx = 1


    sector = 'hydrogen'
    #sector_index = titles['Sectors_short'].index(sector)
    
    # Adjustment to frequencies
    lifetime_adjust = data['BCHY'][:, :, c7ti['Lifetime']]*0
    buildtime_adjust = data['BCHY'][:, :, c7ti['Buildtime']]*0
    
    # Manual adjustment of lifetimes for new technologies
    idx = [i for i in range(len(titles['HYTI'])) if i not in [0, 2]]
    if 2021 < year < 2035:
        
        # Increase lifetime for novel technologies by a factor of ten, and
        # reduce over time.
        lifetime_adjust[:, idx] = 15 
        
    elif 2034 < year < 2045:
        
        lifetime_adjust[:, idx] = 15 * (1 - (2045-year)/(2045 - 2034))
        
    
    
    # Estimate substitution frequencies
    data['HYWA'] = sub_freq(data['BCHY'][:, :, c7ti['Lifetime']],
                            data['BCHY'][:, :, c7ti['Buildtime']],
                            lifetime_adjust,
                            buildtime_adjust,
                            np.ones(len(titles['RTI']))*50,
                            np.ones([len(titles['RTI']), len(titles['HYTI']), len(titles['HYTI'])]),
                            titles['HYTI'], titles['RTI'])
    
    # Sum demand
    # data['HYDT'][:, 0, 0] = (data['HYD1'][:, 0, 0] +
    #                          data['HYD2'][:, 0, 0] +
    #                          data['HYD3'][:, 0, 0] +
    #                          data['HYD4'][:, 0, 0] +
    #                          data['HYD5'][:, 0, 0] )
    
    # Energy price to technological energy cost mapping
    data = calc_ener_cost(data, titles, year)
    
    # Calculate cost factors for green electrolysis
    data = calc_green_cost_factors(data, titles, year)
    
    # Calculate emission factors
    data = calc_emis_rate(data, titles, year)
    # Overwrite for now
    data['HYEF'][:, :, 0] = np.copy(data['BCHY'][:,:, c7ti['Emission factor']])
    
    # H2 content in NH3 by mass
    h2_mass_content = 0.179
    
    # Substitution rate
    # For now we simply assume a value of 2.5
    sub_rate = 0.1


    # %% Historical accounting
    if year <= histend['HYG1']:
        
        # Note FTT:H2's last historical data point in 2022
        # NH3 trade is 2023
        # We assume that the supply map for 2023 is a close enough approximation
        # for 2022.
        
        data['NH3DEM'][:, grey_idx, 0] = data['NH3SM2023'][:, :, 0].sum(axis=0)
        data['NH3PROD'][:, grey_idx, 0] = data['NH3SM2023'][:, :, 0].sum(axis=1)
        data['NH3IMP'][:, grey_idx, 0] = (data['NH3SM2023'][:, :, 0] * (1.0-np.eye(len(titles['RTI'])))).sum(axis=0)
        data['NH3EXP'][:, grey_idx, 0] = (data['NH3SM2023'][:, :, 0] * (1.0-np.eye(len(titles['RTI'])))).sum(axis=1)
        
        # Store supply map in time-based variable
        data['NH3SMLVL'][:, :, grey_idx] = data['NH3SM2023'][:, :, 0].copy()
        
        # Convert NH3 production to H2 demand (which is always sourced locally)
        data['HYDT'][:, 0, 0] = data['NH3DEM'][:, grey_idx, 0] * h2_mass_content
        data['HYPD'][:, 0, 0] = data['HYDT'][:, 0, 0].copy()
        # Hydrogen production is provided in absolute levels and also includes
        # H2 production for non-NH3 purposes
        for r in range(len(titles['RTI'])):
            if data['HYG1'][r, :, 0].sum() > 0.0:
                
                # Capacity shares
                data['HYWS'][r, :, 0] = ((data['HYG1'][r, :, 0] * data['HYCF'][r, :, 0])  
                                         / (data['HYG1'][r, :, 0] * data['HYCF'][r, :, 0]).sum()
                                         )
                
                # Production shares
                prod_shares = ((data['HYG1'][r, :, 0])  
                               / (data['HYG1'][r, :, 0]).sum()
                               )
                
                # Overwrite production estimates
                data['HYG1'][r, :, 0] = prod_shares * data['HYPD'][r, 0, 0]
                
                # Re-estimate capacities
                data['HYWK'][r, :, 0] = divide(data['HYG1'][r, :, 0],
                                               data['HYCF'][r, :, 0]
                                               )
                
                # Get supply map in shares
                if data['NH3SMLVL'][r, :, grey_idx].sum() > 0.0:
                    data['NH3SMSHAR'][r, :, grey_idx] = data['NH3SMLVL'][r, :, grey_idx] / data['NH3SMLVL'][r, :, grey_idx].sum()
                    
                # Copy supply map in shares for the grey market and apply to the
                # green market
                data['NH3SMSHAR'][r, :, green_idx] = data['NH3SMSHAR'][r, :, grey_idx].copy()
        
        # Quick and dirty correction to historical capacities and util rates of
        # Green H2 technologies
        # TODO: Apply correction in the processing script
        data['HYWK'][:, :, 0] = np.where(data['HYCF'][:, :, 0] > data['BCHY'][:, :, c7ti['Maximum capacity factor']],
                                         divide(data['HYCF'][:, :, 0],
                                                data['BCHY'][:, :, c7ti['Maximum capacity factor']]) *
                                         data['HYWK'][:, :, 0],
                                         data['HYWK'][:, :, 0])
        
        # All capacity is "Bad" capacity
        data['WBWK'] = np.copy(data['HYWK'])
        # Adjust capacity factors
        data['HYCF'][:, :, 0] = divide(data['HYG1'][:, :, 0], data['HYWK'][:, :, 0])
        
        # LCOH calclation
        data = get_lcoh2(data, titles)
        
        # global price
        glob_h2_price = divide((data['HYCC'][:, :, 0] * data['HYG1'][:, :, 0]).sum()
                                               , data['HYG1'][:, :, 0].sum()
                                               )
        
        for r in range(len(titles['RTI'])):
            # Average producer prices for green and grey
            data['WPPR'][r, grey_idx, 0] = divide((data['HYCC'][r, :, 0] * data['HYG1'][r, :, 0]).sum()
                                                   , data['HYG1'][r, :, 0].sum()
                                                   )
            if np.isclose(data['WPPR'][r, green_idx, 0], 0.0):
                data['WPPR'][r, green_idx, 0] = glob_h2_price
            
            data['WPPR'][r, green_idx, 0] = divide((data['HYCC'][r, :, 0] * data['HYG1'][r, :, 0] * data['HYGR'][0, :, 0]).sum()
                                                  , (data['HYG1'][r, :, 0]* data['HYGR'][0, :, 0]).sum()
                                                  ) 
            
            if np.isclose(data['WPPR'][r, grey_idx, 0], 0.0):
                data['WPPR'][r, grey_idx, 0] = glob_h2_price + 5.0
        
        # Calculate NH3 LC
        data = get_lchb(data, h2_mass_content, titles)
        
        # Calculate CBAM
        data = get_cbam(data, h2_mass_content, titles)
        
        # Calculate delivery costs
        data = get_delivery_cost(data, data, titles)
        
        # Calculate energy use
        data = calc_ener_cons(data, titles, year)
            

    # %% Simulation period
    else:
        
        
        # Apply demand index to get future demand (grey market)
        data['NH3DEM'][:, grey_idx, 0] = time_lag['NH3DEM'][:, grey_idx, 0] * data['NH3DEMIDX'][:, 0, 0]
        # Check for medium-term inputs and if green techs, then set demand for
        # green market
        data['NH3DEM'][:, green_idx, 0] = np.sum(data['HYGR'][:, :, 0] 
                                         * data['HYMT'][:, :, 0] 
                                         * data['BCHY'][:, :, c7ti['Maximum capacity factor']]
                                         ,axis=1)
        
        # Remove demand in Taiwan
        data['NH3DEM'][48, green_idx, 0] = 0.0
        data['NH3DEM'][48, grey_idx, 0] = 0.0
        
        
        if year < 2029:
            # Set green ammonia demand to a floor level
            mask = data['NH3DEM'][:, green_idx, 0] < time_lag['NH3DEM'][:, green_idx, 0]
            data['NH3DEM'][:, green_idx, 0] = np.where(mask,
                                                      time_lag['NH3DEM'][:, green_idx, 0],
                                                      data['NH3DEM'][:, green_idx, 0])
            
            # data['NH3DEM'][:, green_idx, 0][mask] = time_lag['NH3DEM'][:, green_idx1, 0]
            
            # floor demand
            data['WGFL'][:, 0, 0] = data['NH3DEM'][:, green_idx, 0].copy()
            
            # Apply mandates
            mandated_demand = data['NH3DEM'][:, grey_idx, 0] * data['WDM1'][:, 0, 0]
            
            mask = data['NH3DEM'][:, green_idx, 0] < mandated_demand
            data['NH3DEM'][:, green_idx, 0] = np.where(mask,
                                                      mandated_demand,
                                                      data['NH3DEM'][:, green_idx, 0])
            
            
            # Remove green demand from grey demand
            data['NH3DEM'][:, grey_idx, 0] -= data['NH3DEM'][:, green_idx, 0]
            
            # Prevent negative values
            data['NH3DEM'][:, grey_idx, 0][data['NH3DEM'][:, grey_idx, 0]<0.0] = 0.0
            
        else:
            
            # Set green demand to floor level first
            data['WGFL'][:, 0, 0] = time_lag['WGFL'][:, 0, 0].copy()
            data['NH3DEM'][:, green_idx, 0] = data['WGFL'][:, 0, 0].copy()
            
            # Apply mandates
            mandated_demand = data['NH3DEM'][:, grey_idx, 0] * data['WDM1'][:, 0, 0]
            data['NH3DEM'][:, green_idx, 0][data['NH3DEM'][:, green_idx, 0] < mandated_demand] = mandated_demand
            
            # Remove green demand from grey demand
            data['NH3DEM'][:, grey_idx, 0] -= data['NH3DEM'][:, green_idx, 0]
            
            # Prevent negative values
            data['NH3DEM'][:, grey_idx, 0][data['NH3DEM'][:, grey_idx, 0]<0.0] = 0.0
            
        # Check if the supply map for the green market needs to be filled
        if time_lag['NH3SMLVL'][:, :, green_idx].sum() == 0:
            
            data['NH3SMLVL'][:, :, green_idx] = time_lag['NH3SMSHAR'][:, :, green_idx] * data['NH3DEM'][:, None, green_idx, 0]
            
            
        
        
        # First, fill the time loop variables with the their lagged equivalents
        data_dt = {}
        for var in time_lag.keys():

            if domain[var] == 'FTT-H2':

                data_dt[var] = np.copy(time_lag[var])

        data_dt['HYIY'] = np.zeros([len(titles['RTI']), len(titles['HYTI']), 1])
        data_dt['HYIT'] = np.zeros([len(titles['RTI']), len(titles['HYTI']), 1])

        # Factor used to create quarterly data from annual figures
        no_it = int(data['noit'][0, 0, 0])
        dt = 1 / float(no_it)
        
        # %% Start of differential loop
        
        for t in range(1, no_it+1):
            
            # split by market
            for m_idx in range(len(titles['TFTI'])):
                
                if t ==1: print("Calculating trade for the {} market".format(titles['TFTI'][m_idx]))
            
                # Get demand step
                demand_step = (data['NH3DEM'][:, m_idx, 0] - time_lag['NH3DEM'][:, m_idx, 0]) * t/no_it
                data_dt['NH3DEM'][:, m_idx, 0] = time_lag['NH3DEM'][:, m_idx, 0] + demand_step
                
                # Check the supply map
                if demand_step.sum() > 0.0 and np.isclose(data_dt['NH3SMLVL'][:, :, m_idx].sum(), 0.0):
                    
                    # Use the proxy supply map in shares
                    data_dt['NH3SMLVL'][:, :, m_idx] = data_dt['NH3SMSHAR'][:, :, m_idx] * data_dt['NH3DEM'][:, None, m_idx, 0]
                    
                    
                #---------------------------------------------------------- 
                # Call NH3 trade function
                data = calculate_nh3_trade(data, time_lag, demand_step, data_dt, year, sub_rate, m_idx, titles, t, no_it, dt)
                
                # Production of NH3 translates to production of H2, i.e. no trade is assumed
                data['H2DEMAND'][:, m_idx, 0] = data['NH3PROD'][:, m_idx, 0] * h2_mass_content
                
                for r in range(len(titles['RTI'])):
                    
                    if np.isclose(data['H2DEMAND'][r, m_idx, 0], 0.0):
                        continue
                    
                    #---------------------------------------------------------- 
                    # Green hydrogen market
                    if m_idx == green_idx:
                        
                        dSij_green = substitution_in_shares(data_dt['WGWS'], data_dt['WGWS'], data['HYWA'], 
                                                      data_dt['HYLC'], data_dt['HYLD'], 
                                                      r, dt, titles)
                        
                        
                        #calculate temporary market shares and temporary capacity from endogenous results
                        endo_shares = data_dt['WGWS'][r, :, 0] + np.sum(dSij_green, axis=1)
                        endo_cap = divide(endo_shares * data_dt['BCHY'][r, :, c7ti['Maximum capacity factor']] * data['H2DEMAND'][r, m_idx, 0],
                                          endo_shares * data_dt['BCHY'][r, :, c7ti['Maximum capacity factor']])
                        
                        # Add in medium-term capacity
                        cap_add = data['HYMT'][r, :, 0]/no_it * data['HYGR'][0, :, 0]
                        tot_cap_add = cap_add.sum()
                        
                        if (endo_cap.sum() + tot_cap_add > 0.0):
                            
                            data['WGWS'][r, :, 0] = (endo_cap + cap_add)/(np.sum(endo_cap)+tot_cap_add)
                            
                        # Get capacity
                        data['WGWK'][r, :, 0] = divide(data_dt['WGWS'][r, :, 0] * data['BCHY'][r, :, c7ti['Maximum capacity factor']] * data['H2DEMAND'][r, m_idx, 0],
                                                       data_dt['WGWS'][r, :, 0] * data['BCHY'][r, :, c7ti['Maximum capacity factor']])
                        
                        # Get production
                        data['WGWG'][r, :, 0] = data_dt['WGWK'][r, :, 0] * data['BCHY'][r, :, c7ti['Maximum capacity factor']]
                                                     
                    #----------------------------------------------------------    
                    # Grey hydrogen market    
                    else:
                        dSij_grey = substitution_in_shares(data_dt['WBWS'], data_dt['HYWS'], data['HYWA'], 
                                                           data_dt['HYLC'], data_dt['HYLD'], 
                                                           r, dt, titles)                    
        
                        #calculate temporary market shares and temporary capacity from endogenous results
                        endo_shares = data_dt['WBWS'][r, :, 0] + np.sum(dSij_grey, axis=1)
                        endo_cap = divide(endo_shares * data_dt['BCHY'][r, :, c7ti['Maximum capacity factor']] * data['H2DEMAND'][r, m_idx, 0],
                                          endo_shares * data_dt['BCHY'][r, :, c7ti['Maximum capacity factor']])
                        
                        # Add in medium-term capacity
                        cap_add = data['HYMT'][r, :, 0]/no_it * (1.0-data['HYGR'][0, :, 0])
                        tot_cap_add = cap_add.sum()
                        
                        if (endo_cap.sum() + tot_cap_add > 0.0):
                            
                            data['WBWS'][r, :, 0] = (endo_cap + cap_add)/(np.sum(endo_cap)+tot_cap_add)
                            
                        # Get capacity
                        data['WBWK'][r, :, 0] = divide(data_dt['WBWS'][r, :, 0] * data['BCHY'][r, :, c7ti['Maximum capacity factor']] * data['H2DEMAND'][r, m_idx, 0],
                                                       data_dt['WBWS'][r, :, 0] * data['BCHY'][r, :, c7ti['Maximum capacity factor']])
                        
                        # Get production
                        data['WBWG'][r, :, 0] = data_dt['WBWK'][r, :, 0] * data['BCHY'][r, :, c7ti['Maximum capacity factor']]
                        
            # H2 production by tech and region
            data['HYG1'][:, :, 0] = data['WGWG'][:, :, 0] + data['WBWG'][:, :, 0]
            # Capacity by tech and region
            data['HYWK'][:, :, 0] = data['WGWK'][:, :, 0] + data['WBWK'][:, :, 0]
            # Market shares across both segments
            data['HYWS'][:, :, 0] = divide(data['HYWK'][:, :, 0], data['HYWK'][:, :, None, 0].sum(axis=1))
            
            if (np.any(np.isnan(data['HYG1'][:, :, 0])) or
                np.any(np.isnan(data['HYWK'][:, :, 0])) or
                np.any(np.isnan(data['HYCF'][:, :, 0]))
                ):
                x=1
                raise ValueError("Error: The results contain NaN values.")
                
            if (np.any(data['HYG1'][:, :, 0]< 0.0) or
                        np.any(data['HYWK'][:, :, 0]<0.0) or
                        np.any(data['HYCF'][:, :, 0]<0.0)
                        ):
                x=1
                raise ValueError("Error: The results contain negative values.")
                
            # global price
            glob_h2_price = divide((data['HYCC'][:, :, 0] * data['HYG1'][:, :, 0]).sum()
                                                   , data['HYG1'][:, :, 0].sum()
                                                   )
            
            for r in range(len(titles['RTI'])):
                # Average producer prices for green and grey
                data['WPPR'][r, grey_idx, 0] = divide((data['HYCC'][r, :, 0] * data['HYG1'][r, :, 0]).sum()
                                                       , data['HYG1'][r, :, 0].sum()
                                                       )
                if np.isclose(data['WPPR'][r, green_idx, 0], 0.0):
                    data['WPPR'][r, green_idx, 0] = glob_h2_price 
                
                data['WPPR'][r, green_idx, 0] = divide((data['HYCC'][r, :, 0] * data['HYG1'][r, :, 0] * data['HYGR'][0, :, 0]).sum()
                                                      , (data['HYG1'][r, :, 0]* data['HYGR'][0, :, 0]).sum()
                                                      ) 
                
                if np.isclose(data['WPPR'][r, grey_idx, 0], 0.0):
                    data['WPPR'][r, grey_idx, 0] = glob_h2_price+ 5.0

            
            # Capacity additions
            cap_diff = (data['HYWK'][:, :, 0] - data_dt['HYWK'][:, :, 0])
            cap_drpctn = divide(data_dt['HYWK'][:, :, 0], time_lag['BCHY'][:, :, c7ti['Lifetime']])
            data['HYWI'][:, :, 0] = np.where(cap_diff > 0.0,
                                             cap_diff + cap_drpctn,
                                             cap_drpctn)
            
            # Spillover learning
            # Using a technological spill-over matrix (HYWB) together with capacity
            # additions (MEWI) we can estimate total global spillover of similar
            # techicals
            bi = np.zeros((len(titles['RTI']),len(titles['HYTI'])))
            hywi0 = np.sum(data['HYWI'][:, :, 0], axis=0)
            dw = np.zeros(len(titles["HYTI"]))
            dw = np.dot(hywi0, data['HYWB'][0, :, :])
            
            # for i in range(len(titles["HYTI"])):
            #     dw_temp = np.copy(hywi0)
            #     dw_temp[dw_temp > dw_temp[i]] = dw_temp[i]
            #     dw[i] = np.dot(dw_temp, data['HYWB'][0, i, :])

            # Cumulative capacity incl. learning spill-over effects
            data["HYWW"][0, :, 0] = time_lag['HYWW'][0, :, 0] + dw
            
            
            # Cost components to not copy over
            not_comps = ['Onsite electricity CAPEX, mean, €/kg H2 cap', 
                         'Onsite electricity CAPEX, % of mean',
                         'Additional OPEX, mean, €/kg H2 prod.', 
                         'Additional OPEX, std, % of mean']
            idx_to_copy = [i for i, comp in enumerate(titles['C7TI']) if comp not in not_comps]
            
            # Copy over the technology cost categories that do not change (all except prices which are updated through learning-by-doing below)
            data['BCHY'][:, :, idx_to_copy] = np.copy(time_lag['BCHY'][:, :, idx_to_copy])            
 
            # Learning-by-doing effects on investment
            for tech in range(len(titles['HYTI'])):

                if data['HYWW'][0, tech, 0] > 0.1:
                    
                    # CAPEX
                    data['BCHY'][:, tech, c7ti['CAPEX, mean, €/kg H2 cap']] = data_dt['BCHY'][:, tech, c7ti['CAPEX, mean, €/kg H2 cap']] * \
                        (1.0 + data['BCHY'][:, tech, c7ti['Learning rate']] * dw[tech]/data['HYWW'][0, tech, 0])
                    # Fixed OPEX
                    data['BCHY'][:, tech, c7ti['Fixed OPEX, mean, €/kg H2 cap/y']] = data_dt['BCHY'][:, tech, c7ti['Fixed OPEX, mean, €/kg H2 cap/y']] * \
                        (1.0 + data['BCHY'][:, tech, c7ti['Learning rate']] * dw[tech]/data['HYWW'][0, tech, 0])
                    # Variable OPEX
                    data['BCHY'][:, tech, c7ti['Variable OPEX, mean, €/kg H2 prod']] = data_dt['BCHY'][:, tech, c7ti['Variable OPEX, mean, €/kg H2 prod']] * \
                        (1.0 + data['BCHY'][:, tech, c7ti['Learning rate']] * dw[tech]/data['HYWW'][0, tech, 0])                        
            
            # Store the investment component
            data['HYIC'][:, :, 0] = data['BCHY'][:, :, c7ti['CAPEX, mean, €/kg H2 cap']]
            
            # Total investment in hydrogen technology
            data['HYIY'][:, :, 0] = data_dt['HYIY'][:, :, 0] + data['HYWI'][:, :, 0] * dt * data['BCHY'][:, :, c7ti['CAPEX, mean, €/kg H2 cap']]
            
            # Total CAPEX in hydrogen supply (incl dedicated power)
            data['HYIT'][:, :, 0] = (data_dt['HYIT'][:, :, 0] + data['HYWI'][:, :, 0] * 
                                     dt  * 
                                    (data['BCHY'][:, :, c7ti['Storage CAPEX, mean, €/kgH2 cap']] + 
                                     data['BCHY'][:, :, c7ti['Onsite electricity CAPEX, mean, €/kg H2 cap']])
                                     ) + data['HYIY'][:, :, 0]
            
            
            # Calculate energy use
            data = calc_ener_cons(data, titles, year)     
            
            # Total emissions
            data['HYWE'] = data['HYEF'] * data['HYG1'] * 1e6 * 1e-9
                        
            # Call LCOH2 function
            data = get_lcoh2(data, titles)
            
            # Calculate NH3 LC
            data = get_lchb(data, h2_mass_content, titles)
            
            # Calculate CBAM
            data = get_cbam(data, h2_mass_content, titles)
            
            # Calculate delivery costs
            data = get_delivery_cost(data, data, titles)
    
            # Update lags
            for var in data_dt.keys():
    
                if domain[var] == 'FTT-H2':
    
                    data_dt[var] = np.copy(data[var]) 

    return data
