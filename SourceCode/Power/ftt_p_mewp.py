# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:41:13 2023

@author: Femke Nijsse
"""
import numpy as np

def get_marginal_fuel_prices_mewp(data, titles):
    """Compute marginal fuel prices MEWP based on development within
    ftt:power"""
    
    #data["MPRI"] = 1
    
    # for r in range(len(titles['RTI'])):
    #     if data["MPRI"] == 1:
    #         weight_new = 0.0
    #         if np.sum(data["MEWK"][r, :, 0]) > 0.0:
    #             weight_new = np.sum(data["MEWI"][r, :, 0]) / np.sum(data["MEWK"][r, :, 0])
    #         if weight_new > 1.0:
    #             weight_new = 1.0
    #         weight_old = 1.0 - weight_new

    #         shares_new = data["MEWI"][r, :, 0] / np.sum(data["MEWI"][r, :, 0])
    #         shares_new = shares_new / np.sum(shares_new)

    #         shares_old = (data["MEWK"][r, :, 0] - data["MEWI"][r, :, 0]) / np.sum(data["MEWK"][r, :, 0])
    #         shares_old[shares_old < 0.0] = 0.0
    #         shares_old = shares_old / np.sum(shares_old)

    #         weighted_lcoe_new = np.sum(shares_new * data["MEWL"][r, :, 0] * data["MECC"][r, :, 0]) / np.sum(shares_new * data["MEWL"][r, :, 0])
    #         weighted_lcoe_old = np.sum(shares_old * data["MEWL"][r, :, 0] * data["MEWC"][r, :, 0]) / np.sum(shares_old * data["MEWL"][r, :, 0])

    #         data["MEWP"][r, 7, 0] = weight_new * weighted_lcoe_new + weight_old * weighted_lcoe_old
    
    return data