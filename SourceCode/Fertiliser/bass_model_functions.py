# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:12:08 2024

@author: adh
"""


import pandas as pd
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt


def Bass_param_estimation(data, titles, histend, train_var, pop_var, model_start, p_max = 0.5, q_max = 0.1, steps = 100):

    timeline = np.arange(model_start, histend[train_var])
    # Create dataframe for history
    train = pd.DataFrame(data[train_var], index = titles['regions'], columns = timeline)
    # Total potential population
    population = pd.DataFrame(data[pop_var], index = titles['regions'], columns = timeline)


    # Create grid for parameter tuning
    # Getting all permutations of list_1
    # with length of list_2
    p_list = [i / steps for i in list(range(1, int(p_max * steps + 1), 1))]
    q_list = [i / steps for i in list(range(1, int(q_max * steps + 1), 1))]
    c = list(itertools.product(q_list, p_list))

    p_list = []
    q_list = []

    for i, reg in train.iterrows():
        # Change t0 depending on the n in first year
        first_year = min(timeline)
        non_zero = [i for i, r in enumerate(reg) if r == 0]

        if len(non_zero) > 0:
            t0 = min(max(non_zero), 20)
        else:
            t0 = 0
        m = population.loc[i]

        df = pd.DataFrame(0, columns = timeline, index = c)


        for ix, ind in enumerate(df.index):
            # p is the coefficient of innovation
            # q is the coefficient of imitation
            q = ind[0]
            p = ind[1]

            pred = []
            for t in timeline:
                t_idx= int(t) - first_year + 1 + t0
                # Estimate diffusion with p, q combination
                pred_t = (m[t] * (1 - np.exp(-(p+q)*(t_idx-t0)))/(1+q/p*np.exp(-(p+q)*(t_idx-t0))))
                pred = pred + [pred_t]

            df.iloc[ix, :] = pred

        # Find paa
        min_ind = np.argmin((df.subtract(reg, axis = 1) ** 2).sum(axis = 1))
        min_values = df.index[min_ind]
        q = min_values[0]
        p = min_values[1]
        q_list = q_list + [q]
        p_list = p_list + [p]

        print('    Innovation and imitation parameters for', i)
        print('      p:', p)
        print('      q:', q)

        pred = []
        for t in timeline:
            t_idx= int(t) - first_year + 1 + t0
            pred_t = (m[t] * (1 - np.exp(-(p+q)*(t_idx-t0)))/(1+q/p*np.exp(-(p+q)*(t_idx-t0))))
            pred = pred + [pred_t]

        test = pd.Series(pred, index = range(first_year, 2022), name = 'Projection')
        fig = pd.concat([reg, test], axis = 1)
        fig.plot()
        plt.savefig('CLEAFS/figures/diffusion_diag_' + i + '.jpg')


        # pred = []
        # for t in range(2008, 2051):
        #     t = t - 2007 + t0
        #     pred_t = (m*(1 - np.exp(-(p+q)*(t-t0)))/(1+q/p*np.exp(-(p+q)*(t-t0))))
        #     pred = pred + [pred_t]

        # test = pd.Series(pred, index = range(2008, 2051))
        # test.plot()
        # reg.plot()

    pd.Series(p_list, index = train.index, name = 'p').to_csv('p.csv')
    pd.Series(q_list, index = train.index, name = 'q').to_csv('q.csv')

    data['p'][:, 0, 0, 0] = p_list
    data['q'][:, 0, 0, 0] = q_list

    return data


def simulate_Bass_diffusion(data, titles, simulation_start, year, period):

    battery_cum_lag = data['battery_cum'][:, :, :, period - 1]
    t0 = simulation_start
    t_current = year - simulation_start


    for i, nuts3 in enumerate(titles['nuts3']):
        p = data['p'][i, 0, 0, 0]
        q = data['q'][i, 0, 0, 0]

        m = data['battery_potential_pop'][i, :, :, period]
        nuts3_bat_lag = battery_cum_lag[i, :, :].sum()
        # Find diffusion year
        t_sim = np.array(range(0, 500)) / 10
        m_total = m.sum()


        pred_sim = (m_total * (1 - np.exp(-(p+q)*(t_sim)))/(1+q/p*np.exp(-(p+q)*(t_sim))))

        # Find closest value
        t_prev = np.argmin(abs(pred_sim - nuts3_bat_lag))
        t = t_prev / 10 + 1

        # if nuts3 == 'Pest':
            # print('    Potential population:', nuts3, ':', m_total)
            # print('    Diffusion year', nuts3, ':', t)

        # Estimate diffusion
        pred_t = (m * (1 - np.exp(-(p+q)*(t)))/(1+q/p*np.exp(-(p+q)*(t))))
        # if t > t0:
        #     pred_lag = (m * (1 - np.exp(-(p+q)*(t_lag-t0)))/(1+q/p*np.exp(-(p+q)*(t_lag-t0))))
        # else:
        #     pred_lag = np.zeros(len(titles['profile_type']))
        # Calculate new batteries
        data['battery_new'][i, :, :, period] = pred_t - battery_cum_lag[i, :, :]

    # Remove batteries over their lifetime
    lt_idx = titles['battery_data'].index('lifetime')
    lifetime = data['battery_specs'][lt_idx, 0, 0, 0]
    scrap_year = int(period - lifetime)
    battery_new = data['battery_new'][:, :, :, period]
    if scrap_year > 0:
        battery_scrap = data['battery_new'][:, :, :, scrap_year]
        data['battery_scrap'][:, :, :, period] = battery_scrap
        data['battery_cum'][:, :, :, period] = battery_cum_lag + battery_new #- battery_scrap
    else:
        data['battery_cum'][:, :, :, period] = battery_cum_lag + battery_new
    # Calculate share of battery owners
    data['battery_share'][:, :, :, period] = data['battery_cum'][:, :, :, period] / data['nr_houses'][:, :, :, 0]
    return data


def simulate_fertiliser_demand_diffusion(data, titles, simulation_start, year, period):


    fertiliser_demand_cum_lag = data['fertiliser_demand_cum'][:, :, :, period - 1]
    t0 = simulation_start
    t_current = year - simulation_start


    if year < 2023:
        hist_idx = titles['hist_year'].index(str(year))
        for i, nuts3 in enumerate(titles['nuts3']):
            # Distribute fertiliser_demand based on potential population
            fertiliser_demand_cum = data['fertiliser_demand_nr'][i, hist_idx, :, :]
            m = data['fertiliser_demand_potential_pop'][i, :, :, period]
            m_shares = m / m.sum()
            data['fertiliser_demand_cum'][i, :, :, period] = fertiliser_demand_cum * m_shares
            data['fertiliser_demand_new'][i, :, :, period] = fertiliser_demand_cum - fertiliser_demand_cum_lag[i, :, :]

    else:
        for i, nuts3 in enumerate(titles['nuts3']):
            p = data['p'][i, 0, 0, 0]
            q = data['q'][i, 0, 0, 0]

            m = data['fertiliser_demand_potential_pop'][i, :, :, period]
            nuts3_fertiliser_demand_lag = fertiliser_demand_cum_lag[i, :, :].sum()
            # Find diffusion year
            t_sim = np.array(range(0, 500)) / 10
            m_total = m.sum()


            pred_sim = (m_total * (1 - np.exp(-(p+q)*(t_sim)))/(1+q/p*np.exp(-(p+q)*(t_sim))))

            # Find closest value
            t_prev = np.argmin(abs(pred_sim - nuts3_fertiliser_demand_lag))
            t = t_prev / 10 + 1

            # if nuts3 == 'Pest':
                # print('    Potential population:', nuts3, ':', m_total)
                # print('    Diffusion year', nuts3, ':', t)

            # Estimate diffusion
            pred_t = (m * (1 - np.exp(-(p+q)*(t)))/(1+q/p*np.exp(-(p+q)*(t))))
            # if t > t0:
            #     pred_lag = (m * (1 - np.exp(-(p+q)*(t_lag-t0)))/(1+q/p*np.exp(-(p+q)*(t_lag-t0))))
            # else:
            #     pred_lag = np.zeros(len(titles['profile_type']))
            # Calculate new batteries
            data['fertiliser_demand_new'][i, :, :, period] = pred_t - fertiliser_demand_cum_lag[i, :, :]

        # Remove batteries over their lifetime
        lt_idx = titles['fertiliser_demand_data'].index('lifetime')
        lifetime = data['fertiliser_demand_specs'][lt_idx, 0, 0, 0]
        scrap_year = int(period - lifetime)
        fertiliser_demand_new = data['fertiliser_demand_new'][:, :, :, period]
        if scrap_year > 0:
            fertiliser_demand_scrap = data['fertiliser_demand_new'][:, :, :, scrap_year]
            data['fertiliser_demand_scrap'][:, :, :, period] = fertiliser_demand_scrap
            data['fertiliser_demand_cum'][:, :, :, period] = fertiliser_demand_cum_lag + fertiliser_demand_new #- battery_scrap
        else:
            data['fertiliser_demand_cum'][:, :, :, period] = fertiliser_demand_cum_lag + fertiliser_demand_new
    # Calculate share of battery owners
    data['fertiliser_demand_share'][:, :, :, period] = data['fertiliser_demand_cum'][:, :, :, period] / data['nr_houses'][:, :, :, 0]
    return data
