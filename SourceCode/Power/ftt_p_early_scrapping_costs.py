# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:07:31 2025

@author: Femke
"""

import numpy as np

def early_scrapping_costs(data, data_dt, c2ti, titles):
    earlysc = np.zeros((len(titles['RTI']), len(titles['T2TI'])))
    lifetsc = np.zeros_like(earlysc)

    for r in range(len(titles['RTI'])):
        for tech in range(len(titles['T2TI'])):

            if data['MEWK'][r, tech, 0] - data_dt['MEWK'][r, tech, 0] >= 0.0:
                earlysc[r, tech] = 0.0
                lifetsc[r, tech] = data_dt['BCET'][r, tech, c2ti['9 Lifetime (years)']]
                data['MESC'][r, tech, 0] = 0.0
            else:
                earlysc[r, tech] = data['MEWK'][r, tech, 0] - data_dt['MEWK'][r, tech, 0]
                lifetsc[r, tech] = ((1.0 - data['MEWK'][r, tech, 0]/np.sum(data['MEWK'][r, :, 0])) 
                                    * (data['MEWK'][r, tech, 0] / earlysc[r, tech]) * 5)

            if (lifetsc[r, tech] - data_dt['BCET'][r, tech, c2ti['9 Lifetime (years)']]) < 0.0:
                data['MESC'][r, tech, 0] = -earlysc[r, tech] * (
                    (data_dt['BCET'][r, tech, c2ti['9 Lifetime (years)']] - lifetsc[r, tech]) /
                    data_dt['BCET'][r, tech, c2ti['9 Lifetime (years)']] *
                    data_dt['BCET'][r, tech, c2ti['3 Investment ($/kW)']])
                data['MELF'][r, tech, 0] = lifetsc[r, tech]
            else:
                data['MESC'][r, tech, 0] = 0.0
                data['MELF'][r, tech, 0] = data_dt['BCET'][r, tech, c2ti['9 Lifetime (years)']]

    return data['MESC'][:, :, 0], data['MELF'][:, :, 0]


def early_scrapping_costs_vectorised(data, data_dt, c2ti):
    rdim, tdim = data['MEWK'].shape[:2]
    earlysc = np.zeros((rdim, tdim))
    lifetsc = np.zeros_like(earlysc)

    lifetime_idx = c2ti['9 Lifetime (years)']
    investment_idx = c2ti['3 Investment ($/kW)']

    delta_mewk = data['MEWK'][:, :, 0] - data_dt['MEWK'][:, :, 0]
    
    # If there is capacity growth
    mask_pos = delta_mewk >= 0.0
    earlysc[mask_pos] = 1e-10       # Small value to avoid dividing by zero
    lifetsc[mask_pos] = data_dt['BCET'][:, :, lifetime_idx][mask_pos]
    data['MESC'][:, :, 0][mask_pos] = 0.0
    
    # If there is capacity loss
    mask_neg = ~mask_pos
    earlysc[mask_neg] = delta_mewk[mask_neg]
    total_mewk = np.sum(data['MEWK'][:, :, 0], axis=1, keepdims=True)
    
    # Early scrapping lifetimes from the rate of decline (smaller than planned lifetimes e.g. <40 years for coal)
    # This is calculated assuming a logistic trajectory, hence the (1-f)f/(df/dt).
    # Factor 5 comes from time scaling, i.e. 5 time constants in logistic diffusion from 50% to 1%, consistent with Aij matrix
    lifetsc[mask_neg] = ((1.0 - data['MEWK'][:, :, 0] / total_mewk)
                         * (data['MEWK'][:, :, 0] / earlysc)
                         * 5.0)[mask_neg]
    
    # Compare theoretical lifetime with achieved lifetime
    bcet_lifetime = data_dt['BCET'][:, :, lifetime_idx]
    lifetime_shortfall = lifetsc - bcet_lifetime
    mask_short = lifetime_shortfall < 0.0
    
    # Compute costs of early scrappage using investment costs
    data['MESC'][:, :, 0][mask_short] = -earlysc[mask_short] * (
        -lifetime_shortfall[mask_short] / bcet_lifetime[mask_short] *
        data_dt['BCET'][:, :, investment_idx][mask_short]
    )
    data['MELF'][:, :, 0][mask_short] = lifetsc[mask_short]

    data['MESC'][:, :, 0][~mask_short] = 0.0
    data['MELF'][:, :, 0][~mask_short] = bcet_lifetime[~mask_short]

    return data['MESC'][:, :, 0], data['MELF'][:, :, 0]