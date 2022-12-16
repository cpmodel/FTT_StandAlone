# -*- coding: utf-8 -*-
"""
=========================================
divide.py
=========================================
Bespoke divide

Functions included:
    - divide
        Return element-wise a / b with zeroes in place of divide-by-zeroes.


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
