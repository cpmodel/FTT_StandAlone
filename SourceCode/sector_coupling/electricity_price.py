# -*- coding: utf-8 -*-
"""
Electricity price feedback from power costs and time-of-use tariffs

This module:
1. Updates electricity costs using MEWP growth (FTT-Power coupling).
2. Simulates endogenous smart meter diffusion.
3. Simulates TOU tariff uptake adoption driven by VRE and smart meters.
4. Applies TOU tariff uptake effect directly to existing electricity cost variables.

"""

import os
import pandas as pd
import numpy as np

from SourceCode.support.divide import divide


# =========================================================
# 1. Electricity price feedback (supply side)
# =========================================================
def electricity_price_feedback(data, time_lag):
    """
    Updates electricity cost variables using MEWP-driven growth.

    This function propagates cost changes through all sectoral electricity
    cost channels defined in the mapping file.
    """

    mewp_growth = divide(data["MEWP"], time_lag["MEWP"])
    elec_mewp_growth = mewp_growth[:, 7, 0][np.newaxis].T

    elec_map = pd.read_csv(
        os.path.join("Utilities", "mappings", "Electricity_cost_mapping.csv"),
        index_col=0
    )

    for model in elec_map.index:
        elec_index = [
            int(x) for x in elec_map.loc[model, "Electricity_index"].split(",")
        ]
        cost_var = elec_map.loc[model, "Cost_var"]
        cost_index = elec_map.loc[model, "Cost index"]

        lag_cost = time_lag[cost_var][:, elec_index, cost_index]

        data[cost_var][:, elec_index, cost_index] = lag_cost * elec_mewp_growth

    return data


# =========================================================
# 2. Smart meter diffusion
# =========================================================
def smart_meter_uptake(data, time_lag, rollout_rate = 0.05):
    """
    Logistic diffusion of smart meter uptake.

    Represents rollout of enabling infrastructure required for TOU tariff uptake pricing.
    """
    sm = time_lag["Smart meter uptake"]

    dsm = rollout_rate

    data["Smart meter uptake"] = np.clip(sm + dsm, 0.0, 1.0)
    
    return data


# =========================================================
# 3. VRE share helper
# =========================================================
def _compute_vre_share(data, time_lag):
    """
    Computes VRE share (solar + wind) from lagged generation data.
    """

    n_regions = data["MEWG"].shape[0]
    vre_share = np.zeros(n_regions)

    for r in range(n_regions):
        total_gen = np.sum(time_lag["MEWG"][r])

        if total_gen > 0:
            solar = time_lag["MEWG"][r, 18, 0]
            wind = np.sum(time_lag["MEWG"][r, 16:18, 0])
            vre_share[r] = (solar + wind) / total_gen

    return vre_share


# # =========================================================
# # 4. Time-of-use tariff diffusion
# # =========================================================
# def TOU_uptake_feedback(data, time_lag,
#                         alpha_TOU=0.01,
#                         beta_VRE=0.05,
#                         p=0.001):
#     """
#     Diffusion of TOU tariff uptake tariffs driven by:
#     - VRE penetration (need for encouraging TOU)
#     - Smart meter penetration (enabling infrastructure)

#     Logistic bounded diffusion process.
#     """

#     vre_share = _compute_vre_share(data, time_lag)
#     sm = data["Smart meter uptake"]

#     u = time_lag["TOU tariff uptake"]

#     du = (p + beta_VRE * vre_share[:, None, None]) * (sm - u) + alpha_TOU * u * (sm - u)

#     data["TOU tariff uptake"] = np.clip(u + du, 0.0, 1.0)
    
#     print(f'TOU taiff uptake Be is: {data["TOU tariff uptake"][[0, 41], 0, 0]}')
    
#     return data




# def TOU_uptake_feedback(data, time_lag,
#                         r0=0.006,
#                         r_vre=0.25,
#                         k=5.0):
#     """
#     TOU diffusion constrained by smart meter penetration.

#     u_{t+1} = u_t + r(V) * u_t * (1 - u_t / S_t)
#     """

#     vre_share = _compute_vre_share(data, time_lag)

#     sm = np.clip(data["Smart meter uptake"], 1e-6, 1.0)
#     u = time_lag["TOU tariff uptake"]

#     # VRE-driven adoption speed (nonlinear sensitivity)
#     vre_effect = 1.0 / (1.0 + np.exp(-k * (vre_share - 0.3)))

#     r = r0 + r_vre * vre_effect

#     du = r[:, None, None] * u * (1.0 - u / sm)

#     data["TOU tariff uptake"] = np.clip(u + du, 0.0, sm)
    
#     print(f'TOU taiff uptake BE/IN is: {data["TOU tariff uptake"][[0, 41], 0, 0]}')

#     return data



def TOU_uptake_feedback(data, time_lag):
    """
    Minimal logistic TOU diffusion model.

    No calibration parameters:
    - smart meters = carrying capacity
    - VRE = speed modifier
    - lambda fixed by diffusion timescale
    """

    vre = _compute_vre_share(data, time_lag)

    sm = np.clip(data["Smart meter uptake"], 1e-6, 1.0)
    u = time_lag["TOU tariff uptake"]

    # fixed diffusion timescale (doubling ~12 years)
    lam = np.log(2) / 12.0

    # VRE effect (bounded multiplier, no extra parameters)
    r_eff = lam * vre

    du = r_eff[:, None, None] * u * (1.0 - u / sm)

    data["TOU tariff uptake"] = np.clip(u + du, 0.0, sm)
    
    print(f'TOU tariff uptake BE/IN is: {data["TOU tariff uptake"][[0, 41], 0, 0]}')


    return data




# =========================================================
# 5. TOU tariff uptake price transmission (NO new variables)
# =========================================================
def TOU_price_feedback(data, time_lag):
    """
    Applies TOU tariff uptake adoption to existing electricity cost variables.

    This modifies already-computed electricity cost arrays inside:
    - electricity_price_feedback()

    No new cost variables are created. The adjustment is applied
    multiplicatively to all electricity cost channels listed in
    the mapping file.
    """
    
    
    data = smart_meter_uptake(data, time_lag)
    data = TOU_uptake_feedback(data, time_lag)
    dTOU = data['TOU tariff uptake'] - time_lag['TOU tariff uptake']
 
    data['Elec price volatility'][:, 0, 0] = (
        np.clip((data["MLBP"][:, 3, 0] - data["MLBP"][:, 0, 0]) / data["MLBP"][:, 3, 0], 0, 1.0)
        )
    
    # Max discount under full TOU tariff uptake adoption. We assume that utilities
    # will pass half of discounts on to consumers
    max_discount = data['Elec price volatility'] * 0.5

    dprice_factor = max_discount * dTOU

    elec_map = pd.read_csv(
        os.path.join("Utilities", "mappings", "Electricity_cost_mapping.csv"),
        index_col=0
    )

    # Apply to all electricity cost variables already in the system
    for model in elec_map.index:
        elec_index = [
            int(x) for x in elec_map.loc[model, "Electricity_index"].split(",")
        ]
        cost_var = elec_map.loc[model, "Cost_var"]
        cost_index = elec_map.loc[model, "Cost index"]

        # apply TOU tariff uptake discount to lagged-updated cost
        data[cost_var][:, elec_index, cost_index] *= (1 - dprice_factor[:, :, 0])

    return data