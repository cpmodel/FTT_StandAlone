# -*- coding: utf-8 -*-
"""
=========================================
ftt_h2_csc.py
=========================================
Hydrogen cost-supply curves FTT module.
####################################

This file models the regional hydrogen cost-supply curves based on
hylcs and their respective standard deviations. The function assumes
normal distribution of the hylcs for each technology and constructs an aggregated
aggregated cost-supply curve.

Functions included:
    - get_csc
        Calculate cost-supply curves

variables:
hylc = levelised cost of hydrogen
hyld = standard deviation of LCOH
hywk = hydrogen production capacities
hycsc = cost supply curve of hydrogen supply

"""

# Third party imports
import numpy as np
from scipy.stats import norm
import pandas as pd
# Local library imports
from SourceCode.support.divide import divide

# %% CSC
# -------------------------------------------------------------------------
# -------------------------- CSC function ---------------------------------
# -------------------------------------------------------------------------

def get_csc(hylc, hyld, hywk, hycsc, titles):
    """
    Calculate cost-supply curves.

    The function calculates the regional hydrogen cost-supply curves based on the hylcs of the
    production technolgies and their standard deviation.
    """

    # Categories for the cost matrix (BCHY)
    c7ti = {category: index for index, category in enumerate(titles['C7TI'])}
    hyti = {category: index for index, category in enumerate(titles['HYTI'])}

    # Replace nans
    hylc = np.nan_to_num(hylc, 0)
    hyld = np.nan_to_num(hyld, 0)
    hywk = np.nan_to_num(hywk, 0)


    # Find the 5th and the 95th percentile of the hylc distributions
    hylc_05 = hylc - 2 * hyld
    hylc_95 = hylc + 2 * hyld

    # Find min and max
    hylc_min = np.min(hylc_05)
    hylc_max = np.max(hylc_95)

    # Calculate cost bins
    bins_nr = len(titles['bins'])
    # Get the width of the bins
    bins_width = (hylc_max - hylc_min) / (bins_nr - 1)
    # Create a list with the bin limits
    bins = list(range(int(hylc_min + bins_width), int(hylc_max + bins_width), int(bins_width)))

    # Calculate the proportion of capacities for each cost bin
    bin_proportions = np.zeros_like(hycsc)
    # Calculate the probabilites for the first bin
    for b, limit in enumerate(bins):
        cdf = norm.cdf(limit, hylc, hyld)
        bin_proportions[:, :, b] = cdf

    # By-product adjustment
    bin_proportions = np.nan_to_num(bin_proportions, nan = 1)
    # Calculate capacities by bin
    hycsc = bin_proportions * hywk[:, :, np.newaxis]
    hycsc = np.nan_to_num(hycsc, 0)

    # pd.Series(hycsc.sum(axis = 0).sum(axis = 0), index = titles['bins']).plot()


    return hycsc, bins