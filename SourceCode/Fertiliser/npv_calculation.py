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



def npv_calculation(data, titles, subsidy, lump_sum, period):

    # Set main parameters
    yearly_cons = data['consumption'][:, :, :, :] * 12
    cons_size = [1.5, 1, 0.75]
    pl_idx = titles['pv_data'].index('elec_price_low')
    price_low = data['pv_specs'][pl_idx, 0, 0, 0]
    ph_idx = titles['pv_data'].index('elec_price_high')
    price_high =  data['pv_specs'][ph_idx, 0, 0, 0]
    plim_idx = titles['pv_data'].index('elec_price_limit')
    price_limit = data['pv_specs'][plim_idx, 0, 0, 0]
    # price_limit = 2000
    pgr_idx = titles['pv_data'].index('elec_price_growth')
    price_gr = data['pv_specs'][pgr_idx, 0, 0, 0]
    disc_idx = titles['pv_data'].index('discount_rate')
    discount_rate = data['pv_specs'][disc_idx, 0, 0, 0]
    pvc_idx = titles['pv_data'].index('pv_cost')
    pv_cost = data['pv_specs'][pvc_idx, 0, 0, 0]
    lc_idx = titles['pv_data'].index('labour_cost')
    labour_cost = data['pv_specs'][lc_idx, 0, 0, 0]
    lt_idx = titles['pv_data'].index('lifetime')
    lifetime = data['pv_specs'][lt_idx, 0, 0, 0]
    sc_idx = titles['pv_data'].index('self_consumption')
    self_consumption = data['pv_specs'][sc_idx, 0, 0, 0]
    fit_idx = titles['pv_data'].index('feed_in_tariff')
    feed_in_tariff = data['pv_specs'][fit_idx, 0, 0, 0]
    pv_price_chng = data['pv_price'][1, 0, 0, period]
    pv_cost = pv_cost * pv_price_chng

    # Adjust profiles with consumption
    yearly_cons_size = np.repeat(yearly_cons, len(titles['cons_size']), axis = 1)
    yearly_cons_size = yearly_cons_size * np.expand_dims(cons_size, axis = (0, 2, 3))
    profile_sum = data['profiles'][0, 0, :, :].sum(axis = 1).sum(axis = 0)

    for i, cty in enumerate(titles['RTI']):
        # Adjust profile with consumption profiles
        adj_profile = np.repeat(data['profiles'], len(titles['cons_size']), axis = 0)
        adj_profile = adj_profile * np.expand_dims(yearly_cons_size[i, :, :, :], axis = (1)) / profile_sum

        # Adjust solar profile to meet annual consumption
        pv_sum = data['pv_gen'][i, :, :, :].sum(axis = 1).sum(axis = 1)
        pv_size = yearly_cons_size[i, :, 0, 0] / pv_sum
        adj_pv_gen = data['pv_gen'][i, :, :, :] * np.expand_dims(pv_size, axis = (1, 2))
        # Extend PV array with NUTS3 dimension
        adj_pv_gen = np.repeat(np.expand_dims(adj_pv_gen, axis = (0)), len(titles['profile_type']), axis = 0)
        # Calculate PV overproduction
        overprod = adj_pv_gen - adj_profile
        overprod[overprod < 0] = 0
        # Calculate PV self-consumption
        grid_cons = adj_profile - adj_pv_gen
        grid_cons[grid_cons < 0] = 0
        self_cons = adj_profile - grid_cons
        # data['pv_overprod'] = overprod

        # Calculate actual price
        price_low_cons = yearly_cons_size[i, :, :, :].copy()
        price_low_cons[price_low_cons > price_limit] = price_limit
        price_high_cons = yearly_cons_size[i, :, :, :].copy() - price_low_cons
        price_high_cons[price_high_cons < 0] = 0

        price = (price_low_cons * price_low + price_high_cons * price_high) / yearly_cons_size[i, :, :, :]
        price = np.repeat(np.expand_dims(price[:, 0, 0], axis = (1)), len(titles['profile_type']), axis = 1)
        # Annuity factor for NPV
        annuity_factor = (1 - ((1 + price_gr) / (1 + discount_rate))**lifetime)
        # Total benefits from battery
        npv_benefit = (self_cons.sum(axis = 2).sum(axis = 2) * price + overprod.sum(axis = 2).sum(axis = 2) * feed_in_tariff) / (discount_rate - price_gr) * annuity_factor
        data['pv_benefit'][i, :, :, period] = npv_benefit

        # Adjustment of labour cost with nuts3 income
        inc_ratio = data['income'][i, :, 0, 0] / data['income'][:, :, 0, 0].mean()
        inv = (pv_cost + labour_cost * inc_ratio) * pv_size * (1 - subsidy) - lump_sum

        # Calculae subsidy
        # Assume that realtive subsidy provides subsidy until NPV = 0
        # Therefore covers the subsidy % of the benefits
        subs = npv_benefit / (1 - subsidy) * subsidy + lump_sum
        # subs = (battery_cost + labour_cost * inc_ratio) * battery_size * subsidy + lump_sum
        # Extend investment array with profile_type dimension
        inv_2d = np.repeat(np.expand_dims(inv, axis = (0)), len(titles['profile_type']), axis = 0)
        data['pv_investment'][i, :, :, period] = inv_2d
        # subs_2d = np.repeat(np.expand_dims(subs, axis = (0)), len(titles['profile_type']), axis = 0)


        # NPV
        npv = np.subtract(npv_benefit, inv_2d)
        data['pv_npv'][i, :, :, period] = npv
        data['pv_subsidy'][i, :, :, period] = subs

    return data


def potential_population(data, titles, period):


    # Assume that 2.5% of the population are innovators
    innovators = 0.025

    b_rel_std_idx = titles['battery_data'].index('battery_cost_std')
    battery_cost_rel_std = data['battery_specs'][b_rel_std_idx, 0, 0, 0]

    # Gather relevant variables
    inv_3d = data['battery_investment'][:, :, :, period]
    cost_std = inv_3d * battery_cost_rel_std
    benefits = data['battery_benefit'][:, :, :, period]

    # Calculate the potential population
    # Use the cumulative distribution function of normal distribution
    # To find the probability of finding a battery with investment cost
    # where NPV is positive
    # + add the share of innovators to the potential population
    pot_population_share = norm.cdf(benefits, inv_3d, cost_std)
    for reg, nuts3 in enumerate(titles['nuts3']):
        reg_pop_share = pot_population_share[reg, :, :]
        reg_pop_share[reg_pop_share < innovators[reg]] = innovators[reg]
        reg_pop_share[reg_pop_share > 1] = 1
        data['battery_potential_pop_share'][reg, :, :, period] = reg_pop_share
    # pot_population_share[pot_population_share < innovators] = innovators
    pot_population_share[pot_population_share > 1] = 1
    data['battery_potential_pop_share'][:, :, :, period] = pot_population_share


    # Calculate number of households by profile_type
    # Extend house nurmber array with profile_type dimension
    nr_houses_3d = np.repeat(data['nr_houses'][:, :, :, 0], len(titles['profile_type']), axis = 1)
    nr_houses_3d = np.repeat(data['nr_houses'][:, :, :, 0], len(titles['cons_size']), axis = 2)
    nr_houses_profile = data['profile_shares'][:, :, :, 0] * nr_houses_3d
    nr_houses_profile_size = cons_size_w * nr_houses_profile

    data['nr_houses_profile'][:, :, :, period] = nr_houses_profile_size

    # Potential population where NPV is positive, so they might buy battery
    pot_population = pot_population_share * nr_houses_profile_size
    data['battery_potential_pop'][:, :, :, period] = pot_population

    return data
