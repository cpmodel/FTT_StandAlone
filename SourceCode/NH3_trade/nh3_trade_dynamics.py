# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_shares.py
=========================================
Power generation shares FTT module.

Functions included:
    - shares
        Calculate market shares

"""

# Third party imports
import numpy as np
from numba import njit

# Local library imports
from SourceCode.support.divide import divide


def calculate_nh3_trade(data, time_lags, demand_step, data_dt, year, sub_rate, m_idx, titles, t, noit, dt):

    # Interpolate demand
    demand_step = time_lags['NH3DEM'][:, m_idx, 0] + (data['NH3DEM'][:, m_idx, 0]-time_lags['NH3DEM'][:, m_idx, 0]) * t/noit
    delta_demand_step = (data['NH3DEM'][:, m_idx, 0]-time_lags['NH3DEM'][:, m_idx, 0]) * 1/noit
        
    # Loop over importing regions
    for r_imp in range(len(titles['RTI'])):
        
        d_trade_competition = np.zeros((len(titles['RTI']), len(titles['RTI'])))
        
        if not demand_step[r_imp]>0.0: 
            continue         
        
        # The first exporter
        for r_exp1 in range(len(titles['RTI'])):
            
            if not data_dt['NH3SMLVL'][r_exp1, r_imp, m_idx]>0.0:
                continue
            
            # The competing other exporter
            for r_exp2 in range(r_exp1):
                
                if not data_dt['NH3SMLVL'][r_exp2, r_imp, m_idx]>0.0:
                    continue
                
                # Unpack delivery costs from exporters 1 and 2 to the importer
                c1 = data_dt['NH3DELIVCOST'][r_exp1, r_imp, m_idx]
                c2 = data_dt['NH3DELIVCOST'][r_exp2, r_imp, m_idx]
                
                # Error propagation of standard deviations
                dc12 = 1.414*np.sqrt( (data_dt['NH3LCSD'][r_exp1, 0, 0])**2 + (data_dt['NH3LCSD'][r_exp2, 0, 0])**2)
                
                # Preferences
                fij_d = 0.5+0.5*np.tanh(1.25*(c2-c1)/dc12)
                fji_d = 1 - fij_d
                                            
                
                # unpack variables
                d1_lvls = data_dt['NH3SMLVL'][r_exp1, r_imp, m_idx]
                d2_lvls = data_dt['NH3SMLVL'][r_exp2, r_imp, m_idx]
                carrying_capacity = demand_step[r_imp]
                
                # Euler approach
                delta_d12 = sub_rate * d1_lvls * d2_lvls / carrying_capacity * (fij_d - fji_d) * dt
                delta_d21 = -delta_d12
                
                # Store in variable
                d_trade_competition[r_exp1, r_exp2] = delta_d12
                d_trade_competition[r_exp2, r_exp1] = delta_d21
                
        # Estimate change in supply flows due to market growth
        d_market_growth = delta_demand_step[r_imp] * (data_dt['NH3SMSHAR'][:, r_imp, m_idx])
        
        # Total change in bilateral flows
        d_total = d_market_growth + d_trade_competition.sum(axis=1)
        
        # Checks
        # Sum across all elements of the d_trade_competition matrix should equal zero
        if not np.isclose(d_trade_competition.sum().sum(), 0.0):
            
            raise ValueError("Competition matrix does not sum up to 0 in {}. Please check!".format(titles['RTI'][r_imp]))
            
        # The total growth should equal to demand growth
        if not np.isclose(d_market_growth.sum(), delta_demand_step[r_imp]):
            
            raise ValueError("Total growth does not equal to demand growth in {}. Please check!\n\tMarket Growth: {}, Demand: {}".format(titles['RTI'][r_imp], d_market_growth.sum(), demand_step.sum()))
            
        # New supply map
        if delta_demand_step[r_imp] > 0.0:
            data['NH3SMLVL'][:, r_imp, m_idx] = data_dt['NH3SMLVL'][:, r_imp, m_idx] + d_total
        else:
            data['NH3SMLVL'][:, r_imp, m_idx] = data_dt['NH3SMLVL'][:, r_imp, m_idx] * ((data_dt['NH3SMLVL'][:, r_imp, m_idx].sum() +d_market_growth.sum())/ data_dt['NH3SMLVL'][:, r_imp, m_idx].sum())
            
        if np.any(data['NH3SMLVL'][:, r_imp, m_idx] < 0.0):
            
            raise ValueError("Negative values found in {}".format(titles['RTI'][r_imp]))
            
        # Estimate supply map in shares
        if data['NH3SMLVL'][:, r_imp, m_idx].sum() > 0.0:
            data['NH3SMSHAR'][:, r_imp, m_idx] = data['NH3SMLVL'][:, r_imp, m_idx] /  data['NH3SMLVL'][:, r_imp, m_idx].sum()
            
        # Nudge the system so that previously non-existing trade relations can
        # now appear
        if (np.any(np.isclose(data['NH3SMSHAR'][:, r_imp, m_idx], 0.0))
            and np.sum(data['NH3SMSHAR'][:, r_imp, m_idx]) > 0.0):
            
            # How many countries will need to be nudged?
            no_of_countries = np.sum(np.isclose(data['NH3SMSHAR'][:, r_imp, m_idx], 0.0))
            # choose a nudge factor
            # base value is divided by the number of iterations and by the number
            # of countries that need to be nudged.
            nudge_factor = 1e-3 / float(noit) / float(no_of_countries)
            # Replace zero with nudge factor
            data['NH3SMSHAR'][:, r_imp, m_idx] = np.where(np.isclose(data['NH3SMSHAR'][:, r_imp, m_idx], 0.0),
                                                          nudge_factor,
                                                          data['NH3SMSHAR'][:, r_imp, m_idx])
            
            # Rescale the supply map in shares so that it adds to 1
            data['NH3SMSHAR'][:, r_imp, m_idx] = data['NH3SMSHAR'][:, r_imp, m_idx] / np.sum(data['NH3SMSHAR'][:, r_imp, m_idx])
            
            # Re-estimate the supply map in levels which now includes the nudge
            data['NH3SMLVL'][:, r_imp, m_idx] = data['NH3SMSHAR'][:, r_imp, m_idx] * demand_step[r_imp]
                        
        
    # %% End of r_imp loop
    
    # Final accounting
    data['NH3PROD'][:, m_idx, 0] = data['NH3SMLVL'][:, :, m_idx].sum(axis=1)
    data['NH3IMP'][:, m_idx, 0] = (data['NH3SMLVL'][:, :, m_idx] * (1.0-np.eye(len(titles['RTI'])))).sum(axis=0)
    data['NH3EXP'][:, m_idx, 0] = (data['NH3SMLVL'][:, :, m_idx] * (1.0-np.eye(len(titles['RTI'])))).sum(axis=1)
    
    # Final check on demand
    if not np.any(np.isclose(data['NH3SMLVL'][:, :, m_idx].sum(axis=0), data_dt['NH3DEM'][:, m_idx,0])):
        
        print("Demand as following from the accounting does not reproduce exogenous demand!")
            
    return data
        
            
            
                                        
                    
                    

                    
        
        
    
    

    return 
