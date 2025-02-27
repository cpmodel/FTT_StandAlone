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
        Bespoke element-wise divide which replaces divide-by-zeros with zeros
    - `estimation <econometrics_functions.html>`__
        Predict future values according to the estimated coefficients.

Functions included:
    - solve
        Main solution function for the module
    - get_lcoh2
        Calculate levelised cost of hydrogen

"""
# Standard library imports
from math import sqrt
import copy
import warnings

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

    fuelvars = ['FR_1', 'FR_2', 'FR_3', 'FR_4', 'FR_5', 'FR_6',
                'FR_7', 'FR_8', 'FR_9', 'FR_10', 'FR_11', 'FR_12']

    sector = 'hydrogen'
    #sector_index = titles['Sectors_short'].index(sector)
    # print("FTT: Hydrogen under construction")

    # Estimate substitution frequencies
    data['HYWA'] = sub_freq(data['BCHY'][:, :, c7ti['Lifetime']],
                            data['BCHY'][:, :, c7ti['Buildtime']],
                            data['BCHY'][:, :, c7ti['Lifetime']]*0,
                            data['BCHY'][:, :, c7ti['Buildtime']]*0,
                            np.ones(len(titles['RTI']))*75,
                            np.ones([len(titles['RTI']), len(titles['HYTI']), len(titles['HYTI'])]),
                            titles['HYTI'], titles['RTI'])
    
    # Sum demand
    data['HYDT'][:, 0, 0] = (data['HYD1'][:, 0, 0] +
                             data['HYD2'][:, 0, 0] +
                             data['HYD3'][:, 0, 0] +
                             data['HYD4'][:, 0, 0] +
                             data['HYD5'][:, 0, 0] )
    
    # Energy price to technological energy cost mapping
    data = calc_ener_cost(data, titles, year)
    
    # Calculate cost factors for green electrolysis
    data = calc_green_cost_factors(data, titles, year)
    
    # Calculate emission factors
    data = calc_emis_rate(data, titles, year)
    # Overwrite for now
    data['HYEF'][:, :, 0] = copy.deepcopy(data['BCHY'][:,:, c7ti['Emission factor']])


    # %% Historical accounting
    if year <= histend['HYG1']:
        
        # Quick and dirty correction to historical capacities and util rates of
        # Green H2 technologies
        # TODO: Apply correction in the processing script
        data['HYWK'][:, :, 0] = np.where(data['HYCF'][:, :, 0] > data['BCHY'][:, :, c7ti['Maximum capacity factor']],
                                         divide(data['HYCF'][:, :, 0],
                                                data['BCHY'][:, :, c7ti['Maximum capacity factor']]) *
                                         data['HYWK'][:, :, 0],
                                         data['HYWK'][:, :, 0])
        
        # Adjust capacity factors
        data['HYCF'][:, :, 0] = divide(data['HYG1'][:, :, 0], data['HYWK'][:, :, 0])

        # data['HYWK'] = divide(data['HYG1'], data['BCHY'][:, :, c7ti['Capacity factor'], np.newaxis])
        data['HYWS'] = data['HYWK'] / data['HYWK'].sum(axis=1)[:, :, None]
        data['HYWS'][np.isnan(data['HYWS'])] = 0.0
        data['HYPD'][:, 0, 0] = data['HYG1'][:, :, 0].sum(axis=1)
        
        # LCOH calclation
        data = get_lcoh2(data, titles)
        
        # Total capacity
        data['WBKF'][:, 0, 0] = data['HYWK'][:, :, 0].sum(axis=1)
        
        # System-wide capacity factor
        data['WBCF'][:, 0, 0] = divide(data['HYG1'][:, :, 0].sum(axis=1),
                                       np.sum(data['HYWK'][:, :, 0] * 
                                              data['BCHY'][:, :, c7ti['Maximum capacity factor']], axis=1)
                                       )
        
        # Calculate the average lifetime across all technologies in each region
        # This determines the rate of decline when new capacity seems underutilised
        average_lifetime = data['BCHY'][:, :, c7ti['Lifetime']] * divide(data['HYWK'][:, :, 0],
                                                                         data['HYWK'][:, :, :].sum(axis=1))
        average_lifetime = average_lifetime.sum(axis=1)
        
        # Running average over 5 years
        if year == 2022:
            
            # Estimate capacity growth.
            data['WBCG'][:,0,0] = calc_capacity_growthrate(data['WBCF'][:, 0, 0], average_lifetime)
            data['WBCG'][np.isinf(data['WBCG'])] = 1.0
            data['WBCG'][np.isnan(data['WBCG'])] = 1.0
            
            data['WBCG'][:,0,0] = 0.2 * data['WBCG'][:,0,0] + 0.8 * time_lag['WBCG'][:,0,0]
        else:
            data['WBCG'][:,0,0] = divide(data['HYWK'][:, :, 0].sum(axis=1), 
                                         time_lag['HYWK'][:, :, 0].sum(axis=1))-1.0
            
        # Calculate energy use
        data = calc_ener_cons(data, titles, year)
            
        

    # %% Simulation period
    else:

        # Total hydrogen demand        
        dem_lag = (time_lag['HYD1'][:, 0, 0] +
                                 time_lag['HYD2'][:, 0, 0] +
                                 time_lag['HYD3'][:, 0, 0] +
                                 time_lag['HYD4'][:, 0, 0] +
                                 time_lag['HYD5'][:, 0, 0] )

        data_dt = {}

        # First, fill the time loop variables with the their lagged equivalents
        for var in time_lag.keys():

            if domain[var] == 'FTT-H2':

                data_dt[var] = copy.deepcopy(time_lag[var])

        data_dt['HYIY'] = np.zeros([len(titles['RTI']), len(titles['HYTI']), 1])

        # Factor used to create quarterly data from annual figures
        no_it = int(data['noit'][0, 0, 0])
        dt = 1 / float(no_it)
        
        # Get hydrogen demand
        data = calc_h2_demand(data)
        

        
        # %% Initialise capacities for both markets
        
        # TODO: Reorganise this code
        
        # Project hydrogen production
        # Default market
        data['WBKF'][:, 0, 0] = time_lag['WBKF'][:, 0, 0] * (1 + time_lag['WBCG'][:, 0, 0])
        

        # Split technologies to their respective market
        if year == 2023 and data['WGRM'].sum() > 0.0 and  data['HYGR'].sum() > 0.0:
            
            # Allocate permissible technologies to the green market
            data_dt['WGWK'][:, :, 0] = time_lag['HYWK'][:, :, 0] * data['HYGR'][:, :, 0]
            data_dt['WGWS'] = data_dt['WGWK'] / data_dt['WGWK'].sum(axis=1)[:, :, None]
            data_dt['WGWS'][np.isnan(data_dt['WGWS'])] = 0.0
            # Remove green techs from the default market
            data_dt['WBWK'][:, :, 0] = time_lag['HYWK'][:, :, 0] * (1.0 - data['HYGR'][:, :, 0])
            data_dt['WBWS'] = data_dt['WBWK'] / data_dt['WBWK'].sum(axis=1)[:, :, None]
            data_dt['WBWS'][np.isnan(data_dt['WBWS'])] = 0.0
            
            # We need a green market forecasted capacity
            data['WGKF'][:, 0, 0] = data_dt['WGWK'][:, :, 0].sum(axis=1)*1.25
            # Fill historical estimate
            data_dt['WGKF'][:, 0, 0] = data_dt['WGWK'][:, :, 0].sum(axis=1)
            
            # Remove initialised green capacities from the grey/bad market
            # data_dt['WBWK'][:, :, 0] -= data_dt['WGWK'][:, :, 0]
            
        elif year > 2023 and data['WGRM'].sum() > 0.0 and  data['HYGR'].sum() > 0.0:
                
            # Project hydrogen production
            # Green market
            data['WGKF'][:, 0, 0] = time_lag['WGKF'][:, 0, 0] * (1 + time_lag['WGCG'][:, 0, 0])
            
        # Estimate the green capacity that is in the system
        # Calculate for all years
        green_cap = (data_dt['WGWK'] + data_dt['HYMT']) * data['HYGR']
        green_cap_prod_pot = green_cap[:, :, 0] * data['BCHY'][:, :, c7ti['Maximum capacity factor']]
        
        
        # Compare green technologies to green hydrogen demand
        # Typically the values will be higher than the green H2 demand at 
        # the start. Therefore, we set a floor demand level, which will remain
        # constant over time. We only do this for the period where the medium-
        # term capacity additions come into play          
        if year <= 2028:
            

            # Check if global green demand is less than 70% of green capacity
            if data['WGRM'].sum() < 0.7*green_cap_prod_pot.sum():
                
                green_h2_addition = 0.7*green_cap_prod_pot.sum() - data['WGRM'].sum()
                
                # Share out green demand where green capacity will arise
                green_cap_share = green_cap_prod_pot.sum(axis=1) / green_cap_prod_pot.sum()
                data['WGFL'][:, 0, 0] = green_h2_addition * green_cap_share
            
            # Adjust green market capacity forecast
            if year > 2023:
                
                data['WGKF'][:, 0, 0] = time_lag['WGKF'][:, 0, 0] * (1 + time_lag['WGCG'][:, 0, 0]) + np.sum(data_dt['HYMT'] * data['HYGR'], axis=1)[:, 0]
             
        else:
            
            data['WGFL'][:, 0, 0] = copy.deepcopy(time_lag['WGFL'][:, 0, 0])
            
        # Add floor green H2 demand levels to the green market
        data['WGRM'][:, 0, 0] += data['WGFL'][:, 0, 0]
        
        # Also add WRGM and WGFL to data_dt
        data_dt['WGRM'][:, 0, 0] = copy.deepcopy(data['WGRM'][:, 0, 0])
        data_dt['WGFL'][:, 0, 0] = copy.deepcopy(data['WGFL'][:, 0, 0])
        
        # Correction to capacity forecast of the green market
        # If demand grows more quickly than capacity, then we need to inflate
        # capacity needs. We also need to flag that rates are greater than what
        # seems reasonable
        if year > 2023:
            if green_cap_prod_pot.sum() < data_dt['WGRM'].sum():
            
                scalar = data_dt['WGRM'].sum() / green_cap_prod_pot.sum()
                data['WGKF'] *= scalar
            
            
            
        
        # %% Decision-making core -  split by market - Green market first
            

        # =====================================================================
        # Start of the quarterly time-loop
        # =====================================================================
        

        #Start the computation of shares
        for t in range(1, no_it+1):
            
            # Expected capacity expansion for the green market
            green_capacity_forecast = data_dt['WGKF'][:, 0, 0] + (data['WGKF'][:, 0, 0] - 
                                                                  data_dt['WGKF'][:, 0, 0]) * t/no_it
            # green_capacity_forecast += np.sum(data['HYMT'][:, :, 0] * np.isclose(data['HYGR'][:, :, 0], 1.0) * t/no_it, axis=1)
            
            # Green demand step
            green_demand_step = data_dt['WGRM'][:, 0, 0] + (data['WGRM'][:, 0, 0] - 
                                                                  data_dt['WGRM'][:, 0, 0]) * t/no_it
            
            # Decision-making core for the green market
            for r in range(len(titles['RTI'])):

                if np.isclose(green_capacity_forecast[r], 0.0):
                    continue
                
                dSij_green = substitution_in_shares(data_dt['WGWS'], data['HYWA'], 
                                              data_dt['HYLC'], data_dt['HYLD'], 
                                              r, dt, titles)
                
                #calculate temporary market shares and temporary capacity from endogenous results
                data['WGWS'][r, :, 0] = data_dt['WGWS'][r, :, 0] + np.sum(dSij_green, axis=1)
                data['WGWK'][r, :, 0] = data['WGWS'][r, :, 0] * green_capacity_forecast[r, np.newaxis]
                
                # Add medium-term capacity additions and rescale shares
                if data['WGWK'][r, :, 0].sum() > 0.0:
                    data['WGWK'][r, :, 0] += data['HYMT'][r, :, 0] * data['HYGR'][0, :, 0] * t/no_it
                    data['WGWS'][r, :, 0] = data['WGWK'][r, :, 0] / data['WGWK'][r, :, 0].sum()   
                    
            # Check where production will occur based on CSC and assumptions
            # Apply fixed domestic utilisation for domestic demand assumptions
            prod_pot = np.sum(data['WGWK'][:, :, 0] * data['BCHY'][:, :, c7ti['Maximum capacity factor']], axis=1)
            dem_pot = green_demand_step * data['WMUT'][:, 0, 0]
            fixed_production_green_potential = np.minimum(prod_pot, dem_pot)
            
            fixed_production_green = fixed_production_green_potential[:, None] * data['WGWS'][:, :, 0]
            unmet_demand_green = green_demand_step - fixed_production_green.sum(axis=1)
            
            # Unallocated capacity
            unalloc_cap_green = data['WGWK'][:, :, 0] - fixed_production_green
                    
            # Call CSC function
            (endo_production_green, 
             data['WCPR'][:,0,0], 
             green_cap_factor,
             data['WEPR'][:,0,0],
             data['WIPR'][:,0,0]) = calc_csc(data_dt['HYCC'], data_dt['HYLD'], 
                                            unmet_demand_green[:, None, None], 
                                            unalloc_cap_green[:, :, None], 
                                            data['BCHY'][:, :, c7ti['Maximum capacity factor']],
                                            data['HYTC'], 
                                            titles,
                                            'green',
                                            year) 

            # Combine fixed and endogenous production estimates
            data['WGWG'][:, :, 0] = endo_production_green + fixed_production_green 
                
            # %% Decision-making core -  split by market - Grey/default market second
                
            # Expected capacity expension for the grey market
            grey_capacity_forecast = data_dt['WBKF'][:, 0, 0] + (data['WBKF'][:, 0, 0] -
                                                                 data_dt['WBKF'][:, 0, 0]) * t/no_it
            
            # grey_capacity_forecast += np.sum(data['HYMT'][:, :, 0] * np.isclose(data['HYGR'][:, :, 0], 0.0) * t/no_it, axis=1)

            # Grey demand step
            grey_demand_step = data_dt['HYDT'][:, 0, 0] + ((data['HYDT'][:, 0, 0] - 
                                                           data_dt['HYDT'][:, 0, 0]) -  
                                                           (data['WGFL'][:, 0, 0] - 
                                                            data_dt['WGFL'][:, 0, 0])) * t/no_it
            grey_demand_step -= green_demand_step
            grey_demand_step[grey_demand_step<0.0] = 0.0
            
            for r in range(len(titles['RTI'])):

                if np.isclose(grey_capacity_forecast[r], 0.0):
                    continue
                
                dSij_grey = substitution_in_shares(data_dt['WBWS'], data['HYWA'], 
                                                   data_dt['HYLC'], data_dt['HYLD'], 
                                                   r, dt, titles)


                #calculate temporary market shares and temporary capacity from endogenous results
                data['WBWS'][r, :, 0] = data_dt['WBWS'][r, :, 0] + np.sum(dSij_grey, axis=1)
                data['WBWK'][r, :, 0] = data['WBWS'][r, :, 0] * grey_capacity_forecast[r, np.newaxis]
                
                # Add medium-term capacity additions and rescale shares
                if data['WBWK'][r, :, 0].sum() > 0.0:
                    data['WBWK'][r, :, 0] += data['HYMT'][r, :, 0] * np.isclose(data['HYGR'][0, :, 0], 0.0) *t/no_it
                    
                    # Add spill-over capacity from the green market - as it can
                    # still compete for the default market
                    # data['WBWK'][r, :, 0] += cap_from_green_to_grey[r, :]
                    
                    # Rescale market shares (it retains endogenous info as the rest 
                    # is added on top of capacities)
                    data['WBWS'][r, :, 0] = data['WBWK'][r, :, 0] / data['WBWK'][r, :, 0].sum()
                    
                    # Adjust forecasted capacity level
                    # data['WBKF'][:, 0, 0]  += np.sum(data['HYMT'][r, :, 0] * np.isclose(data['HYGR'][0, :, 0], 0.0))
                    # data_dt['WBKF'][:, 0, 0]  += np.sum(data['HYMT'][r, :, 0] * np.isclose(data['HYGR'][0, :, 0], 0.0)) * t/no_it
                    
            # Check where production will occur based on CSC and assumptions
            # Apply fixed domestic utilisation for domestic demand assumptions
            prod_pot = np.sum(data['WBWK'][:, :, 0] * data['BCHY'][:, :, c7ti['Maximum capacity factor']], axis=1)
            dem_pot = (grey_demand_step)  * data['WMUT'][:, 0, 0]
            fixed_production_grey_potential = np.minimum(prod_pot, dem_pot)
            fixed_production_grey = fixed_production_grey_potential[:, None] * data['WBWS'][:, :, 0]
            unmet_demand_grey = grey_demand_step - fixed_production_grey.sum(axis=1)
            
            # Unallocated capacity
            unalloc_cap_grey = data['WBWK'][:, :, 0] - fixed_production_grey
                    
            # Call CSC function
            (endo_production_grey, 
             data['WCPR'][:,1,0], 
             grey_cap_factor,
             data['WEPR'][:,1,0],
             data['WIPR'][:,1,0]) = calc_csc(data_dt['HYCC'], data_dt['HYLD'], 
                                            unmet_demand_grey[:, None, None], 
                                            unalloc_cap_grey[:, :, None], 
                                            data['BCHY'][:, :, c7ti['Maximum capacity factor']],
                                            data['HYTC'], 
                                            titles,
                                            'grey',
                                            year) 
            
            # Combine fixed and endogenous production estimates
            data['WBWG'][:, :, 0] = endo_production_grey + fixed_production_grey
            
            # %% Accounting section
            
            # Combine data from the grey and green markets
            # Production by tech and region
            data['HYG1'][:, :, 0] = data['WGWG'][:, :, 0] + data['WBWG'][:, :, 0]
            # Capacity by tech and region
            data['HYWK'][:, :, 0] = data['WGWK'][:, :, 0] + data['WBWK'][:, :, 0]
            # Capacity factors
            data['HYCF'][:, :, 0] = divide(data['HYG1'][:, :, 0], data['HYWK'][:, :, 0]) 
            
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
            
            # Capacity additions
            cap_diff = data['HYWK'][:, :, 0] - time_lag['HYWK'][:, :, 0]
            cap_drpctn = divide(time_lag['HYWK'][:, :, 0], time_lag['BCHY'][:, :, c7ti['Lifetime']])
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
            
            for i in range(len(titles["HYTI"])):
                dw_temp = copy.deepcopy(hywi0)
                dw_temp[dw_temp > dw_temp[i]] = dw_temp[i]
                dw[i] = np.dot(dw_temp, data['HYWB'][0, i, :])

            # Cumulative capacity incl. learning spill-over effects
            data["HYWW"][0, :, 0] = time_lag['HYWW'][0, :, 0] + dw
            
            
            # Cost components to not copy over
            not_comps = ['Onsite electricity CAPEX, mean, €/kg H2 cap', 
                         'Onsite electricity CAPEX, % of mean',
                         'Additional OPEX, mean, €/kg H2 prod.', 
                         'Additional OPEX, std, % of mean']
            idx_to_copy = [i for i, comp in enumerate(titles['C7TI']) if comp not in not_comps]
            
            # Copy over the technology cost categories that do not change (all except prices which are updated through learning-by-doing below)
            data['BCHY'][:, :, idx_to_copy] = copy.deepcopy(time_lag['BCHY'][:, :, idx_to_copy])            
 
            # Learning-by-doing effects on investment
            for tech in range(len(titles['HYTI'])):

                if data['HYWW'][0, tech, 0] > 0.1:

                    data['BCHY'][:, tech, c7ti['CAPEX, mean, €/kg H2 cap']] = data_dt['BCHY'][:, tech, c7ti['CAPEX, mean, €/kg H2 cap']] * \
                        (1.0 + data['BCHY'][:, tech, c7ti['Learning rate']] * dw[tech]/data['HYWW'][0, tech, 0])
                        
            # Store the investment component
            data['HYIC'][:, :, 0] = data['BCHY'][:, :, c7ti['CAPEX, mean, €/kg H2 cap']]
        
        # %% New capacity expansion forecast - grey market
        # System-wide capacity factor
        # Normalised to the maximum capacity factor
        data['WBCF'][:, 0, 0] = divide(data['WBWG'][:, :, 0].sum(axis=1),
                                       np.sum(data['WBWK'][:, :, 0] * data['BCHY'][:, :, c7ti['Maximum capacity factor']], axis=1)
                                       )
        
        
        
        # Initiase dataframe for checking
        df_check_grey_growth = pd.DataFrame(0.0, index=titles['RTI'], columns=['CFSys', 'Dem_CAGR_est','Cap_CAGR_est', 'Cap_CAGR_corr'])
        df_check_grey_growth.CFSys = data['WBCF'][:, 0, 0]
        
        # Calculate the average lifetime across all technologies in each region
        # This determines the rate of decline when new capacity seems underutilised
        average_lifetime_grey = data['BCHY'][:, :, c7ti['Lifetime']] * divide(data['WBWK'][:, :, 0],
                                                                         data['WBWK'][:, :, :].sum(axis=1))
        average_lifetime_grey = average_lifetime_grey.sum(axis=1)
        average_lifetime_grey[np.isclose(average_lifetime_grey, 0.0)] = 30.0
        
        # Estimate capacity growth.
        data['WBCG'][:,0,0] = calc_capacity_growthrate(data['WBCF'][:, 0, 0], average_lifetime_grey)
        data['WBCG'][np.isinf(data['WBCG'])] = 0.0
        data['WBCG'][np.isnan(data['WBCG'])] = 0.0
        data['WBCG'][:,0,0] = 0.2 * data['WBCG'][:,0,0] + 0.8 * time_lag['WBCG'][:,0,0]
        
        # Apply correction to growth rates
        grey_demand_glo = data['HYDT'].sum() - data['WGRM'].sum()
        grey_demand_glo_lag = time_lag['HYDT'].sum() - time_lag['WGRM'].sum()
        # Next year's growth rate is probably similar to last years (we add 1% on top)
        grey_growth = grey_demand_glo / grey_demand_glo_lag-1
        df_check_grey_growth.Dem_CAGR_est = divide((data['HYDT'][:,0,0] - data['WGRM'][:,0,0]),
                                                   (time_lag['HYDT'][:,0,0] - time_lag['WGRM'][:,0,0]))-1
        # Check what the capacity forecast would be if calculated growth rates
        # are used
        grey_est_cap_growth = (np.sum(data['WBKF'] * (1+data['WBCG'])) / data['WBKF'].sum())-1
        df_check_grey_growth.Cap_CAGR_est = data['WBCG'][:,0,0]
        # Scale growth rates accordingly
        scalar = grey_growth/grey_est_cap_growth
        # Rescale
        data['WBCG'] *= scalar
        df_check_grey_growth.Cap_CAGR_corr = data['WBCG'][:, 0,0]

    
        # %% New capacity expansion forecast - green market
        # System-wide capacity factor
        # Normalised to the maximum capacity factor
        data['WGCF'][:, 0, 0] = divide(data['WGWG'][:, :, 0].sum(axis=1),
                                       np.sum(data['WGWK'][:, :, 0] * data['BCHY'][:, :, c7ti['Maximum capacity factor']], axis=1)
                                       )
        
        # Calculate the average lifetime across all technologies in each region
        # This determines the rate of decline when new capacity seems underutilised
        average_lifetime_green = data['BCHY'][:, :, c7ti['Lifetime']] * divide(data['WGWK'][:, :, 0],
                                                                         data['WGWK'][:, :, :].sum(axis=1))
        average_lifetime_green = average_lifetime_green.sum(axis=1)
        average_lifetime_green[np.isclose(average_lifetime_green, 0.0)] = 20.0
        
        # To avoid green capacity to disappear too quickly set decline rates to
        # be low
        if year < 2035:
 
            average_lifetime_green *= (99 * (1-(2034-year)/11) +1)
            
            
        # Initiase dataframe for checking
        df_check_green_growth = pd.DataFrame(0.0, index=titles['RTI'], columns=['CFSys', 'Dem_CAGR_est','Cap_CAGR_est', 'Cap_CAGR_corr'])
        df_check_green_growth.CFSys = data['WGCF'][:, 0, 0]
        
        # Estimate capacity growth.
        data['WGCG'][:,0,0] = calc_capacity_growthrate(data['WGCF'][:, 0, 0], average_lifetime_green)
        data['WGCG'][np.isinf(data['WGCG'])] = 0.0
        data['WGCG'][np.isnan(data['WGCG'])] = 0.0
        data['WGCG'][:,0,0] = 0.5 * data['WGCG'][:,0,0] + 0.5 * time_lag['WGCG'][:,0,0]
        
        # Apply correction to growth rates
        green_demand_glo = data['WGRM'].sum()
        green_demand_glo_lag = time_lag['WGRM'].sum()
        # Next year's growth rate is probably similar to last years (we add 1% on top)
        if np.isclose(green_demand_glo_lag, 0.0):
            green_growth = divide(green_demand_glo, green_demand_glo_lag)-1 + 0.01
        else:
            green_growth = 0.01
        df_check_green_growth.Dem_CAGR_est = divide((data['WGRM'][:,0,0]),
                                                   (time_lag['WGRM'][:,0,0]))-1

        # Check what the capacity forecast would be if calculated growth rates
        # are used
        green_est_cap_growth = (np.sum(data['WGKF'] * (1+data['WGCG'])) / data['WGKF'].sum())-1
        df_check_green_growth.Cap_CAGR_est = data['WGCG'][:,0,0]
        # Scale growth rates accordingly
        scalar = green_growth/green_est_cap_growth
        # Rescale
        if year > 2023: 
            data['WGCG'] *= scalar
        df_check_green_growth.Cap_CAGR_corr = data['WGCG'][:, 0,0]
        # %%
        # Calculate energy use
        data = calc_ener_cons(data, titles, year)     
        
        # Total emissions
        data['HYWE'] = data['HYEF'] * data['HYG1'] * 1e6 * 1e-9
                    
        # Call LCOH2 function
        data = get_lcoh2(data, titles)

        # Update lags
        for var in data_dt.keys():

            if domain[var] == 'FTT-H2':

                data_dt[var] = np.copy(data[var])            
        



    return data
