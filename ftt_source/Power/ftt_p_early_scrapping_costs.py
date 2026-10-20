# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:07:31 2025

@author: Femke
"""

import numpy as np

def early_scrapping_costs(data, lagged_data, c2ti):
    ''' Computes the cost of early scrappage. The values are nonzero for the
    baseline scenario, so take these values with a big grain of salt'''
    
    required = ('MEWK', 'MESC', 'MELF')
    if any(data[name].ndim != 2 for name in required) or lagged_data['MEWK'].ndim != 2:
        raise ValueError("early_scrapping_costs expects 2D region-by-technology arrays")

    rdim, tdim = data['MEWK'].shape[:2]
    earlysc = np.zeros((rdim, tdim))
    lifetsc = np.zeros_like(earlysc)

    lifetime_idx = c2ti['9 Lifetime (years)']
    investment_idx = c2ti['3 Investment ($/kW)']

    delta_mewk = data['MEWK'] - lagged_data['MEWK']
    
    # If there is capacity growth
    mask_pos = delta_mewk >= 0.0
    earlysc[mask_pos] = 1e-10       # Small value to avoid dividing by zero
    lifetsc[mask_pos] = lagged_data['BCET'][:, :, lifetime_idx][mask_pos]
    data['MESC'][mask_pos] = 0.0
    
    # If there is capacity loss
    mask_neg = ~mask_pos
    earlysc[mask_neg] = delta_mewk[mask_neg]
    total_mewk = np.sum(data['MEWK'], axis=1, keepdims=True)
    
    # Early scrapping lifetimes from the rate of decline 
    # that is, smaller than planned lifetimes (e.g. <40 years for coal)
    # This is calculated assuming a logistic trajectory, hence the (1-f)f/(df/dt).
    # Factor 5 comes from time scaling, i.e. 5 time constants in logistic diffusion from 50% to 1%, consistent with Aij matrix
    lifetsc[mask_neg] = ((1.0 - data['MEWK'] / total_mewk)
                         * (data['MEWK'] / earlysc)
                         * 5.0)[mask_neg]
    
    # Compare theoretical lifetime with achieved lifetime
    bcet_lifetime = lagged_data['BCET'][:, :, lifetime_idx]
    lifetime_shortfall = lifetsc - bcet_lifetime
    mask_short = lifetime_shortfall < 0.0
    
    # Compute costs of early scrappage using investment costs
    data['MESC'][mask_short] = -earlysc[mask_short] * (
        -lifetime_shortfall[mask_short] / bcet_lifetime[mask_short] *
        lagged_data['BCET'][:, :, investment_idx][mask_short]
    )
    data['MELF'][mask_short] = lifetsc[mask_short]

    data['MESC'][~mask_short] = 0.0
    data['MELF'][~mask_short] = bcet_lifetime[~mask_short]

    return data['MESC'], data['MELF']


def early_scrapping_costs_old(data, lagged_data, c2ti):
    """Legacy 3D wrapper for :func:`early_scrapping_costs`."""
    names = ('MEWK', 'MESC', 'MELF')
    shapes = {name: data[name].shape for name in names}
    lag_shape = lagged_data['MEWK'].shape
    for name in names:
        if data[name].ndim != 3 or data[name].shape[2] != 1:
            raise ValueError("early_scrapping_costs_old expects singleton 3D storage arrays")
        data[name] = data[name][:, :, 0]
    if lagged_data['MEWK'].ndim != 3 or lagged_data['MEWK'].shape[2] != 1:
        raise ValueError("early_scrapping_costs_old expects singleton 3D storage arrays")
    lagged_data['MEWK'] = lagged_data['MEWK'][:, :, 0]
    try:
        mesc, melf = early_scrapping_costs(data, lagged_data, c2ti)
        return mesc.reshape(shapes['MESC']), melf.reshape(shapes['MELF'])
    finally:
        for name, shape in shapes.items():
            data[name] = data[name].reshape(shape)
        lagged_data['MEWK'] = lagged_data['MEWK'].reshape(lag_shape)
