# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:41:13 2023

@author: Femke Nijsse
"""
import numpy as np
from SourceCode.support.divide import divide

def get_marginal_fuel_prices_mewp(data, titles, Svar, glb3):
    """Compute marginal fuel prices MEWP based on development within
    ftt:power"""

    data["MPRI"][:] = 1


    non_vre_lb_weight = [0] * 5  # Initialize array with 5 elements

    non_vre_lb_weight[4] = 80.0 / 8766.0
    non_vre_lb_weight[3] = (700.0 - 80.0) / 8766.0
    non_vre_lb_weight[2] = (2200.0 - 700.0) / 8766.0
    non_vre_lb_weight[1] = (4400.0 - 2200.0) / 8766.0
    non_vre_lb_weight[0] = (8766.0 - 4400.0) / 8766.0
    

    data["MEWP"][:, 0, 0] = data["MERC"][:, 2, 0]   # Hard coal
    data["MEWP"][:, 1, 0] = data["MERC"][:, 2, 0]   # Soft coal
    data["MEWP"][:, 2, 0] = data["MERC"][:, 1, 0]   # Crude oil
    data["MEWP"][:, 3, 0] = data["MERC"][:, 3, 0]   # Natural gas
    data["MEWP"][:, 10, 0] = data["MERC"][:, 4, 0]  # Biomass



    # For each region r
    for r in range(len(titles['RTI'])):
        
        # If MPRI == 1 --> use the weighted average LCOE
        if data["MPRI"][r] == 1:
            weight_new = 0.0
            if np.sum(data["MEWK"][r, :, 0]) > 0.0:
                weight_new = np.sum(data["MEWI"][r, :, 0]) / np.sum(data["MEWK"][r, :, 0])
            weight_new = min(weight_new, 1.0) 
            weight_old = 1.0 - weight_new
            
            if np.sum(data["MEWI"][r, :, 0]) > 0:
                shares_new = data["MEWI"][r, :, 0] / np.sum(data["MEWI"][r, :, 0])
                shares_new = shares_new / np.sum(shares_new)
            else:
                shares_new = np.zeros(data["MEWI"][r, :, 0].shape)

            shares_old = (data["MEWK"][r, :, 0] - data["MEWI"][r, :, 0]) / np.sum(data["MEWK"][r, :, 0])
            shares_old[shares_old < 0.0] = 0.0
            shares_old = shares_old / np.sum(shares_old)

            
            weighted_lcoe_new = divide(
                        np.sum(shares_new * data["MEWL"][r, :, 0] * data["MECC"][r, :, 0]),
                        np.sum(shares_new * data["MEWL"][r, :, 0]) )
              
            weighted_lcoe_old = np.divide(
                        np.sum(shares_old * data["MEWL"][r, :, 0] * data["MEWC"][r, :, 0]),
                        np.sum(shares_old * data["MEWL"][r, :, 0]) )

            data["MEWP"][r, 7, 0] = weight_new * weighted_lcoe_new + weight_old * weighted_lcoe_old

        # If MPRI == 2 --> estimate a costs based on a merit order
        elif data["MPRI"][r] == 2:

            glb_dict = {0: data["MWG1"][r, :, 0], 1: data["MWG2"][r, :, 0], 2: data["MWG3"][r, :, 0],
                        3: data["MWG4"][r, :, 0], 4: data["MWG5"][r, :, 0], 5: data["MWG6"][r, :, 0]}
            n_loadbands = len(glb3[0])

            # We loop over our load bands
            for LB in range(n_loadbands):
                mc_tech_by_lb = np.zeros_like(data["MWMC"][r, :, 0])  # Initialize marginal costs array
                

                # Only select technologies with non-zero generation in each load band
                # Remove carbon tax, because they are added in the PJR routine # TODO: change this for Standalone
                where_condition = glb_dict[LB] > 0.0
                mc_tech_by_lb[where_condition] = data["MWMC"][r, :, 0][where_condition] \
                                                - data["BCET"][r, :, 0][where_condition]
                
                # Taking the maximum marginal cost has proven to cause massive fluctuations
                # Instead, take weighted average of the marginal cost to stabilise the fluctuations
                # (It's mainly oil that is causing issues)
                if np.sum(glb_dict[LB]) > 0.0:
                    data["MLBP"][r, LB, 0] = np.sum(mc_tech_by_lb * glb_dict[LB]) / np.sum(glb_dict[LB])
                else:
                    data["MLBP"][r, LB, 0] = np.max(data["MWMC"][r, :, 0] * Svar[r, :])

                   
            data["MLBP"][r, 4, 0] *= 1.25 # To reflect increased costs due to start-up and switch off
            data["MLBP"][r, 3, 0] *= 1.1  # Same as above but there is less intermittency
            data["MLBP"][r, 2, 0] *= 1.05
            data["MLBP"][r, 5, 0] *= 1.3  # Reflecting higher transmission costs for VRE

            vre_weight = np.zeros((len(titles['RTI'])))
            non_vre_price =  np.zeros((len(titles['RTI'])))
            
            # Estimate prices as a weighted average of marginal costs
            # Above 25% VRE penetration, we assume that at certain moments
            # electricity prices are completely determined by VRE technologies.
            
            if np.sum(data["MEWG"][r, :, 0] * Svar[r, :]) / np.sum(data["MEWG"][r, :]) > 0.25:
                vre_weight[r] = \
                    (1.0 / 0.75) * np.sum(data["MEWG"][r, :, 0] * Svar[r, :]) \
                    / np.sum(data["MEWG"][r, :, 0]) - (1.0 / 3.0)
    
            if np.sum(np.array([glb_dict[LB] for LB in range(n_loadbands-1)])) > 0.0:
                non_vre_price[r] = np.sum(data["MLBP"][r, :n_loadbands-1, 0] * non_vre_lb_weight)
                
            data["MEWP"][r, 7, 0] = vre_weight[r] * data["MLBP"][r, 5, 0] + \
                                    (1.0 - vre_weight[r]) * non_vre_price[r] 
            
            
    return data
