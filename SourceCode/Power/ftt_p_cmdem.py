# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:25:47 2024

@author: wb591318
"""

import numpy as np


def cmdem(data, titles):
    

    data['CMDEM'] = data['CMIN'] * np.broadcast_to(data['MEWI'], data['CMIN'].shape)
    
    return data

