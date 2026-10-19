# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:07:31 2025

@author: Femke
"""

import numpy as np

def early_scrapping_costs(data, lagged_data, c2ti):
    ''' Computes the cost of early scrappage. The values are nonzero for the
    baseline scenario, so take these values with a big grain of salt'''
    
    def as_region_technology(array):
        if array.ndim == 2:
            return array
        if array.ndim == 3 and array.shape[2] == 1:
            return array[:, :, 0]
        if array.ndim == 3 and array.shape[1] == 1:
            return array[:, 0, :]
        raise ValueError(f"Expected a region-by-technology array, got {array.shape}.")

    def restore_storage_shape(array, original_shape):
        if len(original_shape) == 2:
            return array
        if original_shape[2] == 1:
            return array[:, :, None]
        return array[:, None, :]

    original_data_shapes = {
        variable: data[variable].shape for variable in ('MEWK', 'MESC', 'MELF')
    }
    original_lagged_shape = lagged_data['MEWK'].shape
    for variable in original_data_shapes:
        data[variable] = as_region_technology(data[variable])
    lagged_data['MEWK'] = as_region_technology(lagged_data['MEWK'])

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

    mesc, melf = data['MESC'], data['MELF']
    for variable, original_shape in original_data_shapes.items():
        data[variable] = restore_storage_shape(data[variable], original_shape)
    lagged_data['MEWK'] = restore_storage_shape(
        lagged_data['MEWK'], original_lagged_shape
    )
    return mesc, melf
