# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_lcoe.py


Power LCOE FTT module.

Functions included:
    - get_lcoe
        Calculate levelized costs

"""

import numpy as np

def get_lcoe2(data, titles):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of electricity in $2013/MWh. It includes
    intangible costs (gamma values) and determines the investor preferences.

    Parameters
    -----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) for all
        variables. Variable names are keys and the values are 3D NumPy arrays.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.

    Returns
    ----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.
    """

    # Categories for the cost matrix (BCET)
    c2ti = {category: index for index, category in enumerate(titles['C2TI'])}
    num_regions = len(titles['RTI'])
    num_technologies = data['BCET'].shape[1]

    # Cost matrix
    bcet = data['BCET']

    # Plant lifetime
    lt = bcet[:, :, c2ti['9 Lifetime (years)']]
    bt = bcet[:, :, c2ti['10 Lead Time (years)']]
    max_lt = int(np.max(bt + lt))

    # Define masks to turn off cost components before or after construction 
    full_lt_mat = np.arange(max_lt)
    bt_mask = full_lt_mat <= (bt[..., None] - 1)
    lt_mask = full_lt_mat <= (bt[..., None] + lt[..., None] - 1)
    bt_mask_out = ~bt_mask
    lt_mask = lt_mask & bt_mask_out

    # Capacity factor of marginal unit (for decision-making)
    cf_mu = bcet[:, :, c2ti['11 Decision Load Factor']].copy()
    cf_mu[cf_mu < 0.000001] = 0.000001
    conv_mu = 1 / (bt[..., None] * cf_mu[..., None] * 8766 * 1000)

    # Average capacity factor (for electricity price)
    cf_av = data['MEWL'][:, :, 0].copy()
    cf_av[cf_av < 0.000001] = 0.000001
    conv_av = 1 / (bt[..., None] * cf_av[..., None] * 8766 * 1000)

    # Discount rate
    dr = bcet[:, :, c2ti['17 Discount Rate (%)'], None]

    # Initialize the levelised cost components
    def calculate_cost_matrix(base_cost, conversion_factor, mask):
        cost = np.ones((num_regions, num_technologies, max_lt)) * base_cost[..., None] * conversion_factor
        return np.where(mask, cost, 0)

    it_mu = calculate_cost_matrix(bcet[:, :, c2ti['3 Investment ($/kW)']], conv_mu, bt_mask)
    it_av = calculate_cost_matrix(bcet[:, :, c2ti['3 Investment ($/kW)']], conv_av, bt_mask)
    dit_mu = calculate_cost_matrix(bcet[:, :, c2ti['4 std ($/MWh)']], conv_mu, bt_mask)
    dit_av = calculate_cost_matrix(bcet[:, :, c2ti['4 std ($/MWh)']], conv_av, bt_mask)
    st = calculate_cost_matrix(bcet[:, :, c2ti['3 Investment ($/kW)']] * data['MEWT'], conv_mu, bt_mask)
    ft = calculate_cost_matrix(bcet[:, :, c2ti['5 Fuel ($/MWh)']], 1, lt_mask)
    dft = calculate_cost_matrix(bcet[:, :, c2ti['6 std ($/MWh)']], 1, lt_mask)
    fft = calculate_cost_matrix(bcet[:, :, c2ti['5 Fuel ($/MWh)']] * data['MTFT'][:, :, 0], 1, lt_mask)
    omt = calculate_cost_matrix(bcet[:, :, c2ti['7 O&M ($/MWh)']], 1, lt_mask)
    domt = calculate_cost_matrix(bcet[:, :, c2ti['8 std ($/MWh)']], 1, lt_mask)
    ct = calculate_cost_matrix(bcet[:, :, c2ti['1 Carbon Costs ($/MWh)']], 1, lt_mask)
    
    # Storage costs and marginal costs (lifetime only)
    stor_cost = np.ones((num_regions, num_technologies, max_lt))
    marg_stor_cost = np.ones((num_regions, num_technologies, max_lt))

    msal_rounded = np.rint(data['MSAL'][:, 0, 0])

    stor_cost = np.where(msal_rounded[:, None, None] == 2,
                         (data['MSSP'] + data['MLSP']) / 1000,
                         np.where(msal_rounded[:, None, None] >= 3,
                                  (data['MSSP'] + data['MLSP']) / 1000,
                                  0))

    marg_stor_cost = np.where(msal_rounded[:, None, None] >= 3,
                              (data['MSSM'] + data['MLSM']) / 1000,
                              0)
    
    stor_cost = np.where(lt_mask, stor_cost, 0)
    marg_stor_cost = np.where(lt_mask, marg_stor_cost, 0)

    # Net present value calculations
    # Discount rate
    denominator = (1 + dr) ** full_lt_mat

    # 1-Expenses
    npv_expenses1 = (it_av + ft + omt + stor_cost + marg_stor_cost) / denominator
    npv_expenses2 = (it_av + fft + st + ft + ct + omt + stor_cost + marg_stor_cost) / denominator
    npv_expenses3 = (it_mu + ft + ct + omt + stor_cost + marg_stor_cost) / denominator

    # 2-Utility
    npv_utility = np.where(lt_mask, 1, 0) / denominator

    # 3-Standard deviation (propagation of error)
    npv_std = np.sqrt(dit_mu ** 2 + dft ** 2 + domt ** 2) / denominator

    # 1-levelised cost variants in $/pkm
    lcoe = np.sum(npv_expenses1, axis=2) / np.sum(npv_utility, axis=2)
    tlcoe = np.sum(npv_expenses2, axis=2) / np.sum(npv_utility, axis=2) - data['MEFI'][:, :, 0]
    lcoeco2 = np.sum(npv_expenses3, axis=2) / np.sum(npv_utility, axis=2)
    dlcoe = np.sum(npv_std, axis=2) / np.sum(npv_utility, axis=2)

    # LCOE augmented with gamma values
    tlcoeg = tlcoe + data['MGAM'][:, :, 0]

    # Pass to variables that are stored outside.
    data['MEWC'][:, :, 0] = lcoe     # The real bare LCOE without taxes
    data['MECW'][:, :, 0] = lcoeco2  # The real bare LCOE with taxes
    data['METC'][:, :, 0] = tlcoeg   # As seen by consumer (generalised cost)
    data['MTCD'][:, :, 0] = dlcoe    # Variation on the LCOE distribution

    data['MWIC'][:, :, 0] = bcet[:, :, 2]
    data['MWFC'][:, :, 0] = bcet[:, :, 4]
    data['MCOC'][:, :, 0] = bcet[:, :, 0]

    mwmc_condition = np.rint(data['MSAL'][:, 0, 0]) > 1
    data['MWMC'][:, :, 0] = np.where(mwmc_condition[:, None], 
                                     bcet[:, :, 0] + bcet[:, :, 4] + bcet[:, :, 6] + (data['MSSP'][:, :, 0] + data['MLSP'][:, :, 0]) / 1000,
                                     bcet[:, :, 0] + bcet[:, :, 4] + bcet[:, :, 6])

    data['MMCD'][:, :, 0] = np.sqrt(bcet[:, :, 1] ** 2 + bcet[:, :, 5] ** 2 + bcet[:, :, 7] ** 2)

    return data
