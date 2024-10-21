# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:58:37 2024

@author: adh
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def battery_use(data, d6_data, titles):
    # Set main parameters
    yearly_cons = data['consumption'][:, :, :, :] * 12
    cons_size = [1.5, 1, 0.75]
    battery_size = np.multiply(cons_size, 4)
    eff_idx = titles['battery_data'].index('efficiency')
    battery_eff = data['battery_specs'][eff_idx, 0, 0, 0]
    # Get DoD
    dod_idx = titles['battery_data'].index('depth_of_discharge')
    dod = data['battery_specs'][dod_idx, 0, 0, 0]

    # Adjust profiles with consumption
    yearly_cons_size = np.repeat(yearly_cons, len(titles['cons_size']), axis = 1)
    yearly_cons_size = yearly_cons_size * np.expand_dims(cons_size, axis = (0, 2, 3))
    profile_sum = data['profiles'][0, 0, :, :].sum(axis = 1).sum(axis = 0)

    if 'charge' not in d6_data.keys():
        d6_data['charge'] = np.zeros((len(titles['nuts3']), len(titles['profile_type']),
                                           len(titles['cons_size']), len(titles['date']),
                                           len(titles['hour']), 1))

    if 'charge_level' not in d6_data.keys():
        d6_data['charge_level'] = np.zeros((len(titles['nuts3']), len(titles['profile_type']),
                                           len(titles['cons_size']), len(titles['date']),
                                           len(titles['hour']), 1))

    if 'discharge' not in d6_data.keys():
        d6_data['discharge'] = np.zeros((len(titles['nuts3']), len(titles['profile_type']),
                                           len(titles['cons_size']), len(titles['date']),
                                           len(titles['hour']), 1))

    for i, nuts3 in enumerate(titles['nuts3']):
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

        # Calculate residual load
        residual_demand = adj_profile - adj_pv_gen
        residual_demand[residual_demand < 0] = 0

        charge = np.zeros_like(residual_demand)
        charge_level = np.zeros_like(residual_demand)
        discharge = np.zeros_like(residual_demand)


        for d, day in enumerate(titles['date']):

            for h, hour in enumerate(titles['hour']):
                # Add charge to battery and adjust for efficiency
                if h > 0:
                    prev_charge_level = charge_level[:, :, d, h - 1]
                    charge_level[:, :, d, h]  = prev_charge_level + overprod[:, :, d, h] * battery_eff
                    # Get hourly charge
                    charge[:, :, d, h] = charge_level[:, :, d, h] - prev_charge_level

                elif h == 0 and d > 0:
                    last_h = max(list(titles['hour_short']))
                    prev_charge_level = charge_level[:, :, d - 1, last_h]
                    charge_level[:, :, d, h]  = prev_charge_level + overprod[:, :, d, h] * battery_eff
                    # Get hourly charge
                    charge[:, :, d, h] = charge_level[:, :, d, h] - prev_charge_level
                else:
                    charge_level[:, :, d, h]  = overprod[:, :, d, h] * battery_eff
                    # Get hourly charge
                    charge[:, :, d, h] = charge_level[:, :, d, h]



                # Create holder for discharge potential
                discharge_potential = np.zeros_like(discharge[:, :, d, h])
                # Cap charge with battery size
                for s, size in enumerate(titles['cons_size']):
                    charge_level[:, s, d, h][charge_level[:, s, d, h] > battery_size[s]] = battery_size[s]
                    # Do not allow battery to go below 20%
                    # Calculate discharge potential
                    discharge_potential[:, s] = np.maximum(0, charge_level[:, s, d, h] - (1 - dod) * battery_size[s])

                # Calculate hourly charge
                if h > 0:
                    prev_charge_level = charge_level[:, :, d, h - 1]
                    charge[:, :, d, h] = (charge_level[:, :, d, h] - prev_charge_level) / battery_eff

                elif h == 0 and d > 0:
                    last_h = max(list(titles['hour_short']))
                    prev_charge_level = charge_level[:, :, d - 1, last_h]
                    charge[:, :, d, h] = (charge_level[:, :, d, h] - prev_charge_level) / battery_eff
                else:
                    charge[:, :, d, h] = (charge_level[:, :, d, h]) / battery_eff


                # Calculate discharge
                discharge[:, :, d, h] = np.minimum(discharge_potential[:, :], residual_demand[:, :, d, h])

                # Remove discharge
                charge_level[:, :, d, h] = charge_level[:, :, d, h] - discharge[:, :, d, h]

        d6_data['charge'][i, :, :, :, :, 0] = charge
        d6_data['charge_level'][i, :, :, :, :, 0] = charge_level
        d6_data['discharge'][i, :, :, :, :, 0] = discharge



    return d6_data


def total_battery_use(data, d6_data, titles, timeline, period):

    if 'charge_total' not in d6_data.keys():
        d6_data['charge_total'] = np.zeros((len(titles['nuts3']), len(titles['profile_type']),
                                           len(titles['cons_size']), len(titles['date']),
                                           len(titles['hour']), len(timeline)))
    if 'discharge_total' not in d6_data.keys():
        d6_data['discharge_total'] = np.zeros((len(titles['nuts3']), len(titles['profile_type']),
                                           len(titles['cons_size']), len(titles['date']),
                                           len(titles['hour']), len(timeline)))

    d6_data['charge_total'][:, :, :, :, :, period] = d6_data['charge'][:, :, :, :, :, 0] * np.expand_dims(data['battery_cum'][:, :, :, period], axis = (3, 4))
    d6_data['discharge_total'][:, :, :, :, :, period] = d6_data['discharge'][:, :, :, :, :, 0] * np.expand_dims(data['battery_cum'][:, :, :, period], axis = (3, 4))

    return d6_data
