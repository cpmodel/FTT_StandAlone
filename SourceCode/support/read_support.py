# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:43:30 2021

@author: lb
"""

import numpy as np
import pandas as pd
import os


def read_support():
    
    path = 'utilities\\titles'
    
    lagged_sales = pd.read_excel(os.path.join(path,'Lagged_sales.xlsx'), index_col = 0, header=0, sheet_name = None)
    
    converter = pd.read_excel(os.path.join(path,'Converters.xlsx'), index_col = 0, header=0, sheet_name = None)
    
    return converter, lagged_sales

