# -*- coding: utf-8 -*-
"""
=========================================
ftt_h2_pooledtrade.py
=========================================
Hydrogen pooled trade FTT module.
####################################

This file models the pooled hydrogen trade based on aggregated regional
cost supply curves. The method applies a fixed export share to the regional
demand volumes and assumes that it is supplied from the regional pool.

Functions included:
    - pooled_trade
        Calculate pooled trade

variables:
hycsc = cost supply curve of hydrogen supply


"""

# Third party imports
import numpy as np
# Local library imports
from SourceCode.support.divide import divide

# %% Trade
# -------------------------------------------------------------------------
# -------------------------- Trade function ---------------------------------
# -------------------------------------------------------------------------


#%% Find price given a demand level
def ProductionFromLowestBins(demand, csc, bins, lcoe):
    """
    The function passes the value of demand to the Cost-Supply Curve (CSC) variable.
    It removes the demanded volumes from the least-cost bins to supply demand.
    The price of hydrogen it determined by the average price of hydrogen supplied.

    Parameters
    -----------
    demand: float64
        Volume of demand to be satisfied
    csc: DataFrame
        Cost supply curve for hydrogen production

    Returns
    ----------
    res_demand: float64
        Residual demand that is not satisfied
    res_csc: numpy array
        Updated cost supply curve after satisfying demand
    avg_loce: numpy array
        Average LCOE of production

    """

    # Satisfy demand by using hydrogen capacities from the lowest bins
    # Aggregate CSC by technologies to see total supply by cost bin
    supply_by_bin = csc.sum(axis = 0).sum(axis = 0)
    # Assess total capacity
    total_cap = supply_by_bin[-1]

    # Check if supply can satisfy the total demand
    if total_cap >= demand:
        # If demand can be met by the supply find the last bin used
        res_bin = supply_by_bin - demand
        last_bin = min([i for i, x in enumerate(res_bin) if x > 0])
        # Capacities used
        last_bin_used = csc[:, :, last_bin]
        # Find the previous bin from which all capacities are used
        previous_bin_used = csc[:, :, (last_bin - 1)]
        # Get
        last_bin_demand = demand - previous_bin_used.sum()
        last_bin_supply = last_bin_used.sum() - previous_bin_used.sum()
        cap_used = previous_bin_used + (last_bin_demand / last_bin_supply * (last_bin_used - previous_bin_used))
        # Calcucate residual CSC
        res_csc = np.maximum(csc - cap_used[:, :, np.newaxis], 0)
        # Average LCOE
        avg_lcoe = np.divide((cap_used * lcoe).sum(axis = 1), cap_used.sum(axis = 1), out=np.zeros_like((cap_used * lcoe).sum(axis = 1)), where=cap_used.sum(axis = 1)!=0)
        # Residual demand
        res_demand = 0

    else:
        # If available supply is not enough to meet the total demand use all capacities
        cap_used = csc[:, :, -1]
        res_csc = np.zeros_like(csc)
        # Average LCOE
        avg_lcoe = (cap_used * lcoe).sum(axis = 1) / cap_used.sum(axis = 1)
        # Calculate residual demand
        res_demand = demand - csc[:, :, -1].sum()


    return cap_used, res_demand, res_csc, avg_lcoe

#%%

def pooled_trade(hyd1, fix_hyexpsh, flex_hyexpsh, hylc, lag_hyg1, hywk, hycsc, hyrcsc, bins, titles):
    """
    Calculate pooled trade.

    The function calculates regional trade volumes using regionally pooled cost supply curves.
    """


    # Categories
    c7ti = {category: index for index, category in enumerate(titles['C7TI'])}
    hyti = {category: index for index, category in enumerate(titles['HYTI'])}
    rti = {category: index for index, category in enumerate(titles['RTI_short'])}
    arti = {category: index for index, category in enumerate(titles['ARTI'])}


    # Calculate domestic and export production
    dom_share = 1 - fix_hyexpsh - flex_hyexpsh
    dom_capacity = hywk * dom_share[:, np.newaxis]
    dom_demand = hyd1 * dom_share
    # A fixed share of last year's export is assumed to take place
    fix_exp_prod = lag_hyg1.sum(axis = 1) * fix_hyexpsh
    flex_exp_prod = hyd1.sum() - dom_demand.sum() - fix_exp_prod.sum()
    # Calculate import demand
    imp_demand = hyd1 - dom_demand
    # Split CSCs
    dom_csc = hycsc * dom_share[:, np.newaxis, np.newaxis]
    exp_csc = hycsc * (1 - dom_share[:, np.newaxis, np.newaxis])
    fix_exp_csc = hycsc * fix_hyexpsh[:, np.newaxis, np.newaxis]
    flex_exp_csc = hycsc * flex_hyexpsh[:, np.newaxis, np.newaxis]

    prod = np.zeros_like(hywk)
    # Domestic production
    for r, reg in enumerate(titles['RTI_short']):
        # Get regional values
        r_demand = dom_demand[r]
        r_csc = dom_csc[r, np.newaxis, :, :]
        r_lcoe = hylc[r, np.newaxis, :]
        # Calculate capacities used to supply domestic demand and average LCOE
        cap_used, res_demand, res_csc, avg_lcoe = ProductionFromLowestBins(r_demand, r_csc, bins, r_lcoe)
        flex_exp_prod += res_demand
        # Add remaining capacity to the export CSC
        exp_csc[r, :, :] += res_csc[0, :, :]
        prod[r, :] += cap_used[0, :]

    # Fixed export
    flex_exp_csc = np.zeros_like(exp_csc)
    for r, reg in enumerate(titles['RTI_short']):
        # Get regional values
        r_export = fix_exp_prod[r]
        r_csc = exp_csc[r, np.newaxis, :, :]
        r_lcoe = hylc[r, np.newaxis, :]
        # Calculate capacities used to supply domestic demand and average LCOE
        cap_used, res_demand, res_csc, avg_lcoe = ProductionFromLowestBins(r_export, r_csc, bins, r_lcoe)
        flex_exp_prod += res_demand
        flex_exp_csc[r, :, :] = res_csc[0, :, :]
        prod[r, :] += cap_used[0, :]

    # Flexible global export
    global_lcoe = hylc[:, :]
    # Calculate capacities used to supply domestic demand and average LCOE
    cap_used, res_demand, res_csc, avg_lcoe = ProductionFromLowestBins(flex_exp_prod, flex_exp_csc, bins, global_lcoe)
    prod[:, :] += cap_used[:, :]

    hyg1 = prod.copy()
    capacity_factor = np.divide(hyg1, hywk, out=np.zeros_like(hyg1), where=hywk!=0)



    # reg_conv = dict({'BE': 'Europe',
    #     'DK': 'Europe',
    #     'DE': 'Europe',
    #     'EL': 'Europe',
    #     'ES': 'Europe',
    #     'FR': 'Europe',
    #     'IE': 'Europe',
    #     'IT': 'Europe',
    #     'LX': 'Europe',
    #     'NL': 'Europe',
    #     'AT': 'Europe',
    #     'PT': 'Europe',
    #     'FI': 'Europe',
    #     'SW': 'Europe',
    #     'UK': 'Europe',
    #     'CZ': 'Europe',
    #     'EN': 'Europe',
    #     'CY': 'Europe',
    #     'LV': 'Europe',
    #     'LT': 'Europe',
    #     'HU': 'Europe',
    #     'MT': 'Europe',
    #     'PL': 'Europe',
    #     'SI': 'Europe',
    #     'SK': 'Europe',
    #     'BG': 'Europe',
    #     'RO': 'Europe',
    #     'NO': 'Europe',
    #     'CH': 'Europe',
    #     'IS': 'Europe',
    #     'HR': 'Europe',
    #     'TR': 'Europe',
    #     'MK': 'Europe',
    #     'US': 'North America',
    #     'JA': 'East Asia and Pacific',
    #     'CA': 'North America',
    #     'AU': 'East Asia and Pacific',
    #     'NZ': 'East Asia and Pacific',
    #     'RS': 'Russia',
    #     'RA': 'Central Asia',
    #     'CN': 'East Asia and Pacific',
    #     'IN': 'South Asia',
    #     'MX': 'Latin America',
    #     'BR': 'Latin America',
    #     'AR': 'Latin America',
    #     'CO': 'Latin America',
    #     'LA': 'Latin America',
    #     'KR': 'East Asia and Pacific',
    #     'TW': 'East Asia and Pacific',
    #     'ID': 'East Asia and Pacific',
    #     'AS': 'East Asia and Pacific',
    #     'OP': 'Middle East and North Africa',
    #     'RW': 'Rest of the World',
    #     'UA': 'Central Asia',
    #     'SD': 'Middle East and North Africa',
    #     'NG': 'Africa',
    #     'SA': 'Africa',
    #     'ON': 'Middle East and North Africa',
    #     'OC': 'Africa',
    #     'MY': 'East Asia and Pacific',
    #     'KZ': 'Central Asia',
    #     'AN': 'Africa',
    #     'AC': 'Africa',
    #     'AW': 'Africa',
    #     'AE': 'Africa',
    #     'ZA': 'Africa',
    #     'EG': 'Middle East and North Africa',
    #     'DC': 'Africa',
    #     'KE': 'Africa',
    #     'UE': 'Middle East and North Africa',
    #     'PK': 'South Asia'})


    # # Populate regional cost-supply curves and demand
    # for r, reg in enumerate(titles['RTI_short']):
    #     agg_reg = reg_conv[reg]
    #     agg_reg_idx = titles['ARTI'].index(agg_reg)
    #     hyrcsc[agg_reg_idx, :, :] += exp_csc[r, :, :]

    # # Regional trade
    # for r, reg in enumerate(titles['ARTI_short']):
    #     # Get regional values
    #     r_demand = dom_hyd1[r]
    #     r_csc = hycsc[r, np.newaxis, :, :]
    #     r_lcoe = hylc[r, np.newaxis, :]
    #     # Calculate capacities used to supply domestic demand and average LCOE
    #     cap_used, res_demand, res_csc, avg_lcoe = ProductionFromLowestBins(r_demand, r_csc, bins, r_lcoe)
    #     exp_csc[r, :, :] = r_csc[0, :, :]
    #     imp_hyd1[r] += res_demand
    #     prod[r, :] = cap_used[0, :]

    return hyg1, capacity_factor