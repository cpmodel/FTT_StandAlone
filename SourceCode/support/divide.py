# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:16:41 2021

@author: pv
"""

import numpy as np

def divide(a, b):
    """Return element-wise a / b with zeroes in place of divide-by-zeroes."""
    return np.divide(
        a, b,
        # Use `zeros()`, not `zeros_like()`, to ensure type is `float`
        out=np.zeros(a.shape),
        where=~np.isclose(b, 0),
        casting='unsafe')