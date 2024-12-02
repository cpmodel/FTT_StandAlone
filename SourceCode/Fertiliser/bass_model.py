# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:11:06 2023

@author: adh
"""


import pandas as pd
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt

# os.chdir("C:/Users/adh/OneDrive - Cambridge Econometrics/ADH CE/Phd/ÚNKP_2023/data")


def Bass_param_estimation(data, titles):

    # hist_years = [int(year) for year in titles['hist_year']]
    hist_years = list(e3me_total_fertiliser.columns)[:-1]
    # Create dataframe for fertiliser demand
    fertiliser_demand = e3me_total_fertiliser
    # fertiliser_demand = pd.DataFrame(data['fertiliser_demand'][:, :, 0, 0], index = titles['regions'], columns = hist_years)
    # Annual fertiliser demand demand
    fertiliser_demand_diff = fertiliser_demand.diff()
    # Total potential population
    max_fertiliser = e3me_max_fertiliser


    # Create grid for parameter tuning
    # Getting all permutations of list_1
    # with length of list_2
    p_list = [i / 100000 + 49 / 100000 for i in list(range(1, 1000, 50))]
    q_list = [i / 10000 + 49 / 10000 for i in list(range(1, 4000, 50))]
    c = list(itertools.product(q_list, p_list))

    p_list = []
    q_list = []

    for i, reg in fertiliser_demand.iterrows():
        # Change t0 depending on the n in first year
        first_year = min(hist_years)
        non_zero = [i for i, r in enumerate(reg) if r == 0]

        if len(non_zero) > 0:
            t0 = min(max(non_zero), 20)
        else:
            t0 = 0
        m = max_fertiliser.loc[i]

        df = pd.DataFrame(0, columns = hist_years, index = c)


        for ix, ind in enumerate(df.index):
            # p is the coefficient of innovation
            # q is the coefficient of imitation
            q = ind[0]
            p = ind[1]

            pred = []
            for t in hist_years:
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
        for t in hist_years:
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

    pd.Series(p_list, index = fertiliser_demand.index, name = 'p').to_csv('p.csv')
    pd.Series(q_list, index = fertiliser_demand.index, name = 'q').to_csv('q.csv')

    data['p'][:, 0, 0, 0] = p_list
    data['q'][:, 0, 0, 0] = q_list

    return data


def simulate_bass_diffusion(data, time_lags, titles, histend, tech, sim_var, population):

    tech_idx = titles['TFTI'].index(tech)
    diffusion_lag = time_lags[sim_var][:, tech_idx, :]
    p_array = data['BFTC'][:, tech_idx, titles['CFTI'].index('3 Innovation parameter')]
    q_array = data['BFTC'][:, tech_idx, titles['CFTI'].index('4 Imitation parameter')]
    pop_shares = data['FERTS'][:, 0, 0]

    for i, cty in enumerate(titles['RTI']):
        p = p_array[i]
        q = q_array[i]

        m = population[i] * pop_shares[i]
        diff_lag = diffusion_lag[i]
        # Find diffusion year
        t_sim = np.array(range(0, 500)) / 10
        m_total = m.sum()

        # Bass model equation
        pred_sim = (m_total * (1 - np.exp(-(p+q)*(t_sim)))/(1+q/p*np.exp(-(p+q)*(t_sim))))

        # Find closest value to previous year
        t_prev = np.argmin(abs(pred_sim - diff_lag))
        t = t_prev / 10 + 1

        # Estimate diffusion
        pred_t = (m * (1 - np.exp(-(p+q)*(t)))/(1+q/p*np.exp(-(p+q)*(t))))

        # Calculate tech demand
        if data[sim_var][i, tech_idx, 0] == 0:
            data[sim_var][i, tech_idx, 0] = pred_t

    return data


