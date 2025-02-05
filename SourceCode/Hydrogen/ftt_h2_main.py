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

# Local library imports
from SourceCode.support.divide import divide
from SourceCode.Hydrogen.ftt_h2_lcoh import get_lcoh as get_lcoh2
from SourceCode.Hydrogen.cost_supply_curve import calc_csc
from SourceCode.Hydrogen.ftt_h2_pooledtrade import pooled_trade
from SourceCode.core_functions.substitution_frequencies import sub_freq
from SourceCode.core_functions.substitution_dynamics_in_shares import substitution_in_shares
from SourceCode.core_functions.substitution_dynamics_in_shares import decision_making_core
from SourceCode.core_functions.capacity_growth_rate import calc_capacity_growthrate
# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, iter_lag, titles, histend, year, domain):
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
    print("FTT: Hydrogen under construction")

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
    

        



    # %% Historical accounting
    if year < 2023:
        
        # Quick and dirty correction to historical capacities and util rates of
        # Green H2 technologies
        # TODO: Apply correction in the processing script
        data['HYWK'][:, :, 0] = np.where(data['HYCF'][:, :, 0] > data['BCHY'][:, :, c7ti['Maximum capacity factor']],
                                         divide(data['HYCF'][:, :, 0],
                                                data['BCHY'][:, :, c7ti['Maximum capacity factor']]) *
                                         data['HYWK'][:, :, 0],
                                         data['HYWK'][:, :, 0])

        # data['HYWK'] = divide(data['HYG1'], data['BCHY'][:, :, c7ti['Capacity factor'], np.newaxis])
        data['HYWS'] = data['HYWK'] / data['HYWK'].sum(axis=1)[:, :, None]
        data['HYWS'][np.isnan(data['HYWS'])] = 0.0
        data['HYPD'][:, 0, 0] = data['HYG1'][:, :, 0].sum(axis=1)
        

        
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
        
        # Estimate capacity growth.
        data['WBCG'][:,0,0] = calc_capacity_growthrate(data['WBCF'][:, 0, 0], average_lifetime)
        data['WBCG'][np.isinf(data['WBCG'])] = 1.0
        data['WBCG'][np.isnan(data['WBCG'])] = 1.0
        # Running average over 5 years
        if year == 2022:
            data['WBCG'][:,0,0] = 0.2 * data['WBCG'][:,0,0] + 0.8 * time_lag['WBCG'][:,0,0]
        else:
            data['WBCG'][:,0,0] = divide(data['HYWK'][:, :, 0].sum(axis=1), 
                                         time_lag['HYWK'][:, :, 0].sum(axis=1))-1.0
            
        

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
        
        # %% Demand vectors
        # Split the market into a grey and green market
        # From CLEAFS
        gr_nh3_fert_share_cleafs = data['FERTD'][:, 0, 0] / data['FERTD'][:, :, 0].sum()
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
        
        # %% Initialise capacities for both markets
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
            data['WGKF'][:, 0, 0] = data_dt['WGWK'][:, :, 0].sum(axis=1)*1.1
            
            # Remove initialised green capacities from the grey/bad market
            data_dt['WBWK'][:, :, 0] -= data_dt['WGWK'][:, :, 0]
            
        else:
            
            # Project hydrogen production
            # Green market
            data['WGKF'][:, 0, 0] = time_lag['WGKF'][:, 0, 0] * (1 + time_lag['WGCG'][:, 0, 0])
            
            # Check if there is sufficient market size
            # initially, forecast may need to be inflated
            if data['WGKF'].sum() > 0.0 and data['WGRM'].sum() > 0.0:
                if data['WGKF'].sum()*0.5 < data['WGRM'].sum():
                    
                    data['WGKF'][:, 0, 0] *= data['WGRM'].sum()/data['WGKF'].sum()*0.5
        
        # %% Decision-making core -  split by market - Green market first
            

        # =====================================================================
        # Start of the quarterly time-loop
        # =====================================================================

        #Start the computation of shares
        for t in range(1, no_it+1):
            
            # Expected capacity expansion for the green market
            green_capacity_forecast = data_dt['WGKF'][:, 0, 0] + (data['WGKF'][:, 0, 0] - data_dt['WGKF'][:, 0, 0]) * t/no_it
            
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
                    data['WGWK'][r, :, 0] += data['HYMT'][r, :, 0] * data['HYGR'][0, :, 0]
                    data['WGWS'][r, :, 0] = data['WGWK'][r, :, 0] / data['WGWK'][r, :, 0].sum()   
                    
            # Check where production will occur based on CSC and assumptions
            # Apply fixed domestic utilisation for domestic demand assumptions
            prod_pot = np.sum(data['WGWK'][:, :, 0] * data['BCHY'][:, :, c7ti['Maximum capacity factor']], axis=1)
            dem_pot = data['WGRM'][:, 0, 0]  * data['WMUT'][:, 0, 0]
            fixed_production_green_potential = np.minimum(prod_pot, dem_pot)
            
            
            fixed_production_green = fixed_production_green_potential[:, None] * data['WGWS'][:, :, 0]
            unmet_demand_green = data['WGRM'][:, 0, 0] - fixed_production_green.sum(axis=1)
            
            # Unallocated capacity
            unalloc_cap_green = data['WGWK'][:, :, 0] - fixed_production_green
                    
            # Call CSC function
            endo_production_green, data['WCPR'][:,0,0], green_cap_factor = calc_csc(data_dt['HYCC'], data_dt['HYLD'], 
                                                                                    unmet_demand_green[:, None, None], 
                                                                                    unalloc_cap_green[:, :, None], 
                                                                                    data['BCHY'][:, :, c7ti['Maximum capacity factor']],
                                                                                    data['HYTC'], 
                                                                                    titles) 

            # Combine fixed and endogenous production estimates
            data['WGWG'][:, :, 0] = endo_production_green + fixed_production_green 
            
            # Estimate capacity to take back to the grey market
            if year == 2023:
                # Retain at least 2.5 times the capacity
                cond = data['WGWG'][:, :, 0]* 2.5 > data['WGWK'][:, :, 0] 
                cap_from_green_to_grey = np.where(cond,
                                                  data['WGWG'][:, :, 0]* 2.5,
                                                  0.0)
                # Remove unused green capacity
                data['WGWK'][:, :, 0] -= cap_from_green_to_grey
                # Rescale market shares (again)
                data['WGWS'][:, :, 0] = data['WGWK'][:, :, 0] / data['WGWK'][:, :, 0].sum() 
                data['WGWS'][np.isinf(data['WGWS'])] = 0.0
                data['WGWS'][np.isnan(data['WGWS'])] = 0.0
                
                # Add spill-over capacity from the green market - as it can
                # still compete for the default market
                data['WBWK'][:, :, 0] += cap_from_green_to_grey
                data['WBWS'][:, :, 0] = data['WBWK'][:, :, 0] / data['WBWK'][:, :, :].sum(axis=1)
                data['WBWS'][np.isinf(data['WBWS'])] = 0.0
                data['WBWS'][np.isnan(data['WBWS'])] = 0.0
                
            # %% Decision-making core -  split by market - Grey/default market second
                
            # Expected capacity expension for the grey market
            grey_capacity_forecast = data_dt['WBKF'][:, 0, 0] + (data['WBKF'][:, 0, 0] - data_dt['WBKF'][:, 0, 0]) * t/no_it
            
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
                    data['WBWK'][r, :, 0] += data['HYMT'][r, :, 0] * np.isclose(data['HYGR'][0, :, 0], 0.0)
                    
                    # Add spill-over capacity from the green market - as it can
                    # still compete for the default market
                    # data['WBWK'][r, :, 0] += cap_from_green_to_grey[r, :]
                    
                    # Rescale market shares (it retains endogenous info as the rest 
                    # is added on top of capacities)
                    data['WBWS'][r, :, 0] = data['WBWK'][r, :, 0] / data['WBWK'][r, :, 0].sum()
                    
            # Check where production will occur based on CSC and assumptions
            # Apply fixed domestic utilisation for domestic demand assumptions
            prod_pot = np.sum(data['WBWK'][:, :, 0] * data['BCHY'][:, :, c7ti['Maximum capacity factor']], axis=1)
            dem_pot = (data['HYDT'][:, 0, 0] - data['WGRM'][:, 0, 0])  * data['WMUT'][:, 0, 0]
            fixed_production_grey_potential = np.minimum(prod_pot, dem_pot)
            fixed_production_grey = fixed_production_grey_potential[:, None] * data['WBWS'][:, :, 0] * data['BCHY'][:, :, c7ti['Maximum capacity factor']]
            unmet_demand_grey = (data['HYDT'][:, 0, 0] - data['WGRM'][:, 0, 0]) - fixed_production_grey.sum(axis=1)
            
            # Unallocated capacity
            unalloc_cap_grey = data['WBWK'][:, :, 0] - fixed_production_grey
                    
            # Call CSC function
            endo_production_grey, data['WCPR'][:,1,0], grey_cap_factor = calc_csc(data_dt['HYCC'], data_dt['HYLD'], 
                                                                                    unmet_demand_grey[:, None, None], 
                                                                                    unalloc_cap_grey[:, :, None], 
                                                                                    data['BCHY'][:, :, c7ti['Maximum capacity factor']],
                                                                                    data['HYTC'], 
                                                                                    titles) 
            
            # Combine fixed and endogenous production estimates
            data['WBWG'][:, :, 0] = endo_production_grey + fixed_production_grey
            
            if np.any(data['WBWG'][:, :, 0] < 0.0) or np.any(np.isnan(data['WBWG'][:, :, 0])):
                
                cond = data['WBWG'][:, :, 0] < 0.0
                indices = list(zip(*np.where(cond)))
                indices_by_name = [ [titles['RTI'][xy[0]], titles['HYTI'][xy[1]]] for xy in indices]
                msg = "Negative production values found:\nVar: {}\nYear: {}\nIndices: {}".format('WBWG', year, indices_by_name)
                print(msg)
                
            
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
                raise ValueError("Error: The results contain NaN values.")
                
            if (np.any(data['HYG1'][:, :, 0]< 0.0) or
                        np.any(data['HYWK'][:, :, 0]<0.0) or
                        np.any(data['HYCF'][:, :, 0]<0.0)
                        ):
                raise ValueError("Error: The results contain negative values.")
            
            # Capacity additions
            cap_diff = data['HYWK'][r, :, 0] - time_lag['HYWK'][r, :, 0]
            cap_drpctn = time_lag['HYWK'][r, :, 0] / time_lag['BCHY'][r, :, c7ti['Lifetime']]
            data['HYWI'][r, :, 0] = np.where(cap_diff > 0.0,
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
            
 
            # Learning-by-doing effects on investment
            for tech in range(len(titles['HYTI'])):

                if data['HYWW'][0, tech, 0] > 0.1:

                    data['BCHY'][:, tech, c7ti['CAPEX, mean, €/tH2 cap']] = data_dt['BCHY'][:, tech, c7ti['CAPEX, mean, €/tH2 cap']] * \
                        (1.0 + data['BCHY'][:, tech, c7ti['Learning rate']] * dw[tech]/data['HYWW'][0, tech, 0])
        
        # %% New capacity expansion forecast - grey market
        # System-wide capacity factor
        data['WBCF'][:, 0, 0] = divide(data['WBWG'][:, :, 0].sum(axis=1),
                                       data['WBWK'][:, :, 0].sum(axis=1)
                                       )
        

        # Calculate the average lifetime across all technologies in each region
        # This determines the rate of decline when new capacity seems underutilised
        average_lifetime_grey = data['BCHY'][:, :, c7ti['Lifetime']] * divide(data['WBWK'][:, :, 0],
                                                                         data['WBWK'][:, :, :].sum(axis=1))
        average_lifetime_grey = average_lifetime_grey.sum(axis=1)
        
        # Estimate capacity growth.
        data['WBCG'][:,0,0] = calc_capacity_growthrate(data['WBCF'][:, 0, 0], average_lifetime_grey)
        data['WBCG'][np.isinf(data['WBCG'])] = 1.0
        data['WBCG'][np.isnan(data['WBCG'])] = 1.0
        data['WBCG'][:,0,0] = 0.2 * data['WBCG'][:,0,0] + 0.8 * time_lag['WBCG'][:,0,0]
        # %% New capacity expansion forecast - green market
        # System-wide capacity factor
        data['WGCF'][:, 0, 0] = divide(data['WGWG'][:, :, 0].sum(axis=1),
                                       data['WGWK'][:, :, 0].sum(axis=1)
                                       )
        
        # Calculate the average lifetime across all technologies in each region
        # This determines the rate of decline when new capacity seems underutilised
        average_lifetime_green = data['BCHY'][:, :, c7ti['Lifetime']] * divide(data['WGWK'][:, :, 0],
                                                                         data['WGWK'][:, :, :].sum(axis=1))
        average_lifetime_green = average_lifetime_green.sum(axis=1)
        
        # Estimate capacity growth.
        data['WGCG'][:,0,0] = calc_capacity_growthrate(data['WGCF'][:, 0, 0], average_lifetime_green)
        data['WGCG'][np.isinf(data['WGCG'])] = 1.0
        data['WGCG'][np.isnan(data['WGCG'])] = 1.0
        data['WGCG'][:,0,0] = 0.2 * data['WGCG'][:,0,0] + 0.8 * time_lag['WGCG'][:,0,0]
        # %%
        
        
                    
        # Call LCOH2 function
        data = get_lcoh2(data, titles)

        # Update lags
        for var in data_dt.keys():

            if domain[var] == 'FTT-H2':

                data_dt[var] = np.copy(data[var])            
        



    return data
