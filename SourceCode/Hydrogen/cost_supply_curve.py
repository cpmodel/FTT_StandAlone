# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 03:02:18 2025

@author: pv
"""

# Standard library imports
from math import sqrt
import copy
import warnings

# Third party imports
import numpy as np
import scipy.stats as stats
from scipy.stats import lognorm
import pandas as pd

# Local library imports
from SourceCode.support.divide import divide
from SourceCode.Hydrogen.ftt_h2_lcoh import get_lcoh as get_lcoh2
from SourceCode.Hydrogen.ftt_h2_csc import get_csc
from SourceCode.Hydrogen.ftt_h2_pooledtrade import pooled_trade
from SourceCode.core_functions.substitution_frequencies import sub_freq



def calc_csc(lc, lc_sd, demand, capacity, max_capacity_factor, transport_cost, titles, market, year, bins=500):
    
    # Categories for the cost matrix (BHTC)
    # c7ti = {category: index for index, category in enumerate(titles['C7TI'])}
    # jti = {category: index for index, category in enumerate(titles['JTI'])}
    # hyti = {category: index for index, category in enumerate(titles['HYTI'])}
    
    # Initialise a 3D variable
    lc_erf = np.zeros([bins, len(titles['RTI']), len(titles['HYTI'])])
    
    capacity_online = capacity[:, :, 0] * max_capacity_factor
    
    tot_reg_cap = capacity_online.sum(axis=1)
    
    # Estimate the weigthed transportation costs
    # From the perspective of the exporter
    demand_weight = demand[:, 0, 0] / demand.sum()
    weighted_tc_exp = np.sum(transport_cost[:, :, 0] * demand_weight[:, None], axis=0)
    
    # From the perspective of the importer
    capacity_weight = tot_reg_cap / tot_reg_cap.sum()
    weighted_tc_imp = np.sum(transport_cost[:, :, 0] * capacity_weight[None, :], axis=1)

    
    # Min and max lcoh2
    lc_min = np.min(lc[:, :, 0] *0.9 + weighted_tc_exp[:, None])
    lc_max = np.max(lc[:, :, 0] *1.1 + weighted_tc_exp[:, None]) 
    
    # Cost spacing of the bins
    cost_space = np.linspace(lc_min, lc_max, bins)
    
    # Regional CSC
    for r in range(len(titles['RTI'])):
        
        for i in range(len(titles['HYTI'])):
        
            mu = lc[r, i, 0]
            sigma = lc_sd[r, i, 0]
            if sigma > 0.1 * mu:
                sigma = 0.1 * mu
            cap = capacity_online[r, i]
            
            # Transform mu and sigma (assumed normally distributed) to log-normal params
            shape = np.sqrt(np.log(1 + (sigma / mu) ** 2))
            scale = mu / np.sqrt(1 + (sigma / mu) ** 2)
            
            lc_erf[:, r, i] = cap * lognorm.cdf(cost_space, shape, scale=scale)
            
            # x = stats.norm(loc=mu, scale=sigma).cdf(cost_space)*cap
            
    # Collapse the regional CSC into one global CSC
    glo_csc = lc_erf.sum(axis=1).sum(axis=1)
    
    # Check is there is sufficient supply
    if glo_csc[-2] > demand.sum():
        
        # Find the index of the minimum value in absolute terms after 
        # subtracting global demand
        idx = np.argmin(np.abs(glo_csc - demand.sum()))
        
        if glo_csc[idx] < demand.sum() and idx != bins-1:
            
            idx += 1
        
        # Import price 
        hy_price =  cost_space[idx]
        
        # Now use the found price to see which technologies and regions are exporters
        production = lc_erf[idx, :, :]
        
        # Production will always be higher than demand, so rescale
        production *= demand.sum() / production.sum()
        
    else:
        
        # Take the last bin
        idx = bins-1
        
        # Import price 
        hy_price =  cost_space[idx]
        
        # Now use the found price to see which technologies and regions are exporters
        production = lc_erf[idx, :, :]
        
        # If we reach the last cost bin, then use total capacity for scaling
        production *= capacity_online.sum() / production.sum()
        
    # Add upper and lower limit of prices
    if hy_price < lc_min:
        hy_price = lc_min
    if hy_price > lc_max:
        hy_price = lc_max
        
    
    capacity_factor_new = divide(production, capacity[:, :, 0])
    
    # Calculate the average of average transportation costs
    glo_avg_tc = np.sum(weighted_tc_exp * production.sum(axis=1) / production.sum())
        
    # To estimate regionalised prices, remove global average transportation
    # costs from the global price and add transportation costs as seen by 
    # exporters to the weighted demand region, and importers from the weighted
    # producing capacity regions.
    export_price = hy_price + weighted_tc_exp - glo_avg_tc
    import_price = hy_price + weighted_tc_imp - glo_avg_tc
    
    # To check the CSC development, print out the CSC values
    # csc_out = pd.DataFrame(0.0, index=np.arange(1, bins+1), columns=['Price', 'Quantity', 'DequalsQ'])
    # csc_out.Price = cost_space.copy()
    # csc_out.Quantity = glo_csc.copy()
    # csc_out.loc[idx,'DequalsQ'] = 1
    # csc_out.to_csv('CostSupplyCurve/CSC_{}_{}.csv'.format(market, year))
    
    

    
    return production, hy_price, capacity_factor_new, export_price, import_price
    
    
    
        
        
    
    
    