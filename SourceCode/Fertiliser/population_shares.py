# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:17:54 2023

@author: adh
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.stats import norm
from statistics import NormalDist



def green_population_share(data, time_lags, titles, year):

    # Set ammonia prices
    h2_content_in_nh3 = 0.176  # kg H2 / kg NH3
    h2_benchmarkprice = 1.5  # Euro/kg H2
    nh3_benchmarkprice = 0.45 # Euro/kg NH3
    
    # Should be about 60%. This is effectively the elasticity
    h2_share_in_nh3_price = h2_content_in_nh3 * h2_benchmarkprice / nh3_benchmarkprice
    
    if year > 2023: 
        # Actual price to benchmark price ratio
        green_act_to_benchmarkprice = time_lags['WCPR'][:,0,0]/h2_benchmarkprice
        grey_act_to_benchmarkprice = time_lags['WCPR'][:,1,0]/h2_benchmarkprice
        
        # Ammonia prices
        green_nh3_price = green_act_to_benchmarkprice * h2_share_in_nh3_price * nh3_benchmarkprice
        grey_nh3_price = grey_act_to_benchmarkprice * h2_share_in_nh3_price * nh3_benchmarkprice
    
        # Assume that fertiliser prices are 10% on top of ammonia prices
        # Also convert to Euro/t
        grey_fert_price = grey_nh3_price * 1.1 * 1000
        green_fert_price = green_nh3_price * 1.1 * 1000
        
        # Prevent zero's
        grey_fert_price[np.isclose(grey_fert_price, 0.0)] = np.mean(grey_fert_price[~np.isclose(grey_fert_price,0.0)])
        green_fert_price[np.isclose(green_fert_price, 0.0)] = np.mean(green_fert_price[~np.isclose(green_fert_price,0.0)])

    else:
        
        grey_fert_price = np.ones(len(titles['RTI'])) * nh3_benchmarkprice * 1000
        green_fert_price = np.ones(len(titles['RTI'])) * nh3_benchmarkprice * 1000 * 2.5 
        
    # Print to PFRA
    data['PFRA'][:, 0, 0] = green_fert_price
    data['PFRA'][:, 1, 0] = grey_fert_price
    
    # Assumed standard deviation
    green_fert_std = green_fert_price * 0.1
    grey_fert_std = grey_fert_price * 0.2

    # Calculate the potential population
    # Assume that 2.5% of the population are innovators
    innovators = 0.025
    for reg, country in enumerate(titles['RTI']):
        # Get regional prices
        reg_green_price = green_fert_price[reg]
        reg_grey_price = grey_fert_price[reg]
        reg_green_std = green_fert_std[reg]
        reg_grey_std = grey_fert_std[reg]
        # Assume that fertiliser prices follow normal distribution
        # Calculate the share of popoulation that sees green fertiliser more beneficial
        if reg_green_price > reg_grey_price:
            # In this case the potential green fertiliser adopters are the intersect of the two cost distributions
            green_pop_share = NormalDist(mu = reg_green_price, sigma = reg_green_std).overlap(NormalDist(mu = reg_grey_price, sigma = reg_grey_std))
        else:
            # If green fertiliser is cheaper than it is the non-overlapping part
            green_pop_share = 1 - NormalDist(mu = reg_green_price, sigma = 70).overlap(NormalDist(mu = reg_grey_price, sigma = 50))

        # Set minimum population to the share of innovators
        green_pop_share = np.maximum(innovators, green_pop_share)
        green_pop_share = np.minimum(1, green_pop_share)
        data['FERTS'][reg, :, :] = green_pop_share


    return data
