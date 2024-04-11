# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:25:47 2024

@author: wb591318
"""
import copy
import numpy as np


def cmdem(data, titles):
    
    cmdem_copy = copy.deepcopy(data['CMIN'])
    # for i in range(len(data['CMIN'][:, :, :])):
    #     cmdem_copy[:, :, i] =  cmdem_copy[:, :, i] * data['MEWI'][:, :, 0]

    cmdem_copy = cmdem_copy * data['MEWI']

    data['CMDEM'] = cmdem_copy
    
    return data

