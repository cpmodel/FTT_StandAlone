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



def green_population_share(data, time_lags, titles):

    # Set ammonia prices
    green_nh_price = data['PFRA'][:, titles['TFTI'].index('Green fertiliser'), 0]
    grey_nh_price = data['PFRA'][:, titles['TFTI'].index('Grey fertiliser'), 0]

    # Assume that ammonia gives the half of grey fertiliser costs
    grey_fert_price = grey_nh_price * 2
    green_fert_price = green_nh_price + grey_nh_price
    green_fert_std = grey_fert_price * data['BFTC'][:, titles['CFTI'].index('1 Price std'), 0]
    grey_fert_std = data['PFRA'][:, titles['TFTI'].index('Grey fertiliser'), 0]

    # Calculate the potential population
    # Assume that 2.5% of the population are innovators
    innovators = 0.025
    for reg, nuts3 in enumerate(titles['RTI']):
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
