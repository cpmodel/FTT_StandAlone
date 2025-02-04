# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:55:48 2025

@author: pv
"""

import numpy as np


def calc_capacity_growthrate(sys_cf, sys_lt, midpoint=0.6, max_growth=0.25):
    """
    

    Parameters
    ----------
    sys_cf : 1D NumPy Array
        System-average capacity factors for each country are used to determine
        how much capacity will be added in the next year. High capacity factors
        indicate superior economic performance.
    sys_lt : NumPy Array
        System-average lifetime of capacity. This is used to determine the
        decline rate if capacity factors are below the midpoint.
    midpoint : float, optional
        Describes the transition from expansion to reduction on the basis of the
        system-average capacity factor. The default is 0.6.
    max_growth : float, optional
        DESCRIPTION. The default is 0.25.

    Returns
    -------
    None.

    """
    
    grow = 0.5+0.5*np.tanh(1.25*(sys_cf - np.mean([1.0, midpoint]))/(0.25*(1.0-midpoint)))
    decline = 0.5+0.5*np.tanh(1.25*(sys_cf - np.mean([0.0, midpoint]))/(0.3*np.mean([0.0, midpoint]))) -1.0
    max_grow_rate = 0.25
    max_decline_rate = 1.0/sys_lt
    
    growh_rate = np.where(sys_cf > midpoint,
                          grow * max_grow_rate,
                          decline * max_decline_rate)
    
    
    return growh_rate

# %% Test
if __name__ == '__main__':
    import pandas as pd
    
    y = np.zeros((100))
    x = np.linspace(0, 1, num=100)
    
    for i in range(x.shape[0]):
        y[i] = calc_capacity_growthrate(x[i], 25, midpoint=0.6, max_growth=0.35)
        
    df = pd.DataFrame(y, index=x, columns=['Growth rate'])
    ax = df.plot(grid=True, legend=False, color='black')
    ax.set(xlabel='System-average capacity factor', ylabel='Growth rate')

    