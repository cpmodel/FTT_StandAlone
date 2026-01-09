# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_lcoe.py
=========================================
Power LCOE FTT module.


Functions included:
    - set carbon_tax
        Determine carbon tax from REPP
    - get_lcoe
        Calculate levelized costs

"""

# Third party imports
import numpy as np



def set_carbon_tax(data, c2ti, year):
    '''
    Convert the carbon price in REPP from euro / tC to $2013 dollars. 
    Apply the carbon price to power sector technologies based on their efficiencies

    The function changes data.
    
    REX --> EU local currency per euros rate (33 is US$)
    PRSC --> price index (local currency) consumer expenditure
    EX --> EU local currency per euro, 2005 == 1
    The 13 part of the variable mean denotes 2013. 
    '''

    carbon_costs = (
                    data['BCET'][:, :, c2ti['15 Emissions (tCO2/GWh)']]   # Emission per GWh
                    * data["REPPX"][:, :, 0]                              # Carbon price in euro / tC
                    * data["REX13"][33, 0, 0] / ( data["PRSCX"][:, :, 0] * data["EX13"][:, :, 0] / (data["PRSC13"][:, :, 0]  * data["EXX"][:, :, 0]) )
                    / 1000 / 3.666                                        # Conversion from GWh to MWh and from C to CO2
                    )
    
    
    if np.isnan(carbon_costs).any():
        print(f"Carbon price is nan in year {year}")
        print(f"The arguments of the nans are {np.argwhere(np.isnan(carbon_costs))}")
        print( ('Conversion factor:'
              f'{data["REX13"][33, 0, 0] / ( data["PRSCX"][:, :, 0] * data["EX13"][:, :, 0] / (data["PRSC13"][:, :, 0]  * data["EXX"][:, :, 0]) )}') )
        print(f"Emissions intensity {data['BCET'][:, :, c2ti['15 Emissions (tCO2/GWh)']]}")
        
        raise ValueError

    return carbon_costs

# %% lcoe
# -----------------------------------------------------------------------------
# --------------------------- LCOE function -----------------------------------
# -----------------------------------------------------------------------------
  
def get_lcoe(data, titles):
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
    cf_mu = bcet[:, :, c2ti['11 Decision Load Factor']]
    cf_mu[cf_mu < 0.000001] = 0.000001
    conv_mu = 1 / (bt[..., None] * cf_mu[..., None] * 8766) * 1000

    # Average capacity factor (for electricity price)
    cf_av = data['MEWL'][:, :, 0]
    cf_av[cf_av < 0.000001] = 0.000001
    conv_av = 1 / (bt[..., None] * cf_av[..., None] * 8766) * 1000
        
    def get_cost_component(base_cost, conversion_factor, mask):
        '''Mask costs during build or life time, and apply
        conversion to generation where appropriate'''
        cost = np.multiply(base_cost[..., None], conversion_factor)
        return np.multiply(cost, mask)

    it_mu = get_cost_component(bcet[:, :, c2ti['3 Investment ($/kW)']], conv_mu, bt_mask)
    it_av = get_cost_component(bcet[:, :, c2ti['3 Investment ($/kW)']], conv_av, bt_mask)
    dit_mu = get_cost_component(bcet[:, :, c2ti['4 std ($/MWh)']], conv_mu, bt_mask)
    st = get_cost_component(bcet[:, :, c2ti['3 Investment ($/kW)']] * data['MEWT'][:, :, 0], conv_mu, bt_mask)
    ft = get_cost_component(bcet[:, :, c2ti['5 Fuel ($/MWh)']], 1, lt_mask)
    dft = get_cost_component(bcet[:, :, c2ti['6 std ($/MWh)']], 1, lt_mask)
    fft = get_cost_component(bcet[:, :, c2ti['5 Fuel ($/MWh)']] * data['MTFT'][:, :, 0], 1, lt_mask)
    omt = get_cost_component(bcet[:, :, c2ti['7 O&M ($/MWh)']], 1, lt_mask)
    domt = get_cost_component(bcet[:, :, c2ti['8 std ($/MWh)']], 1, lt_mask)
    ct = get_cost_component(bcet[:, :, c2ti['1 Carbon Costs ($/MWh)']], 1, lt_mask)
    
    # Who pays for storage, and total storage costs
    msal_rounded = np.rint(data['MSAL'][:, 0, 0])
    storage_sum = (data['MSSP'] + data['MLSP']) / 1000
    marg_storage_sum = (data['MSSM'] + data['MLSM']) / 1000
    
    # Storage cost
    stor_mask = (msal_rounded >= 2)
    stor_cost = np.where(stor_mask, storage_sum, 0)
    stor_cost = get_cost_component(stor_cost[:, :, 0], 1, lt_mask)
    dstor_cost = 0.2 * stor_cost  # Assume standard deviation of 20%

    # Battery-only storage cost (short-term only, for MECW battery only variant)
    battery_cost = np.where(stor_mask, data['MSSP'] / 1000, 0)
    battery_cost = get_cost_component(battery_cost[:, :, 0], 1, lt_mask)

    # Marginal storage cost
    marg_mask = (msal_rounded >= 3)
    marg_stor_cost = np.where(marg_mask, marg_storage_sum, 0)
    marg_stor_cost = get_cost_component(marg_stor_cost[:, :, 0], 1, lt_mask)
    
    
    # Net present value calculations
    # Discount rate
    dr = bcet[:, :, c2ti['17 Discount Rate (%)'], None]
    denominator = (1 + dr) ** full_lt_mat

    # 1a – Expenses – marginal units (for investor decisions)
    # Note: marg_stor_cost is NOT in bare LCOE (matches cascading branch)
    npv_expenses_mu_bare = (it_mu + ft + omt + stor_cost) / denominator
    npv_expenses_mu_co2 = npv_expenses_mu_bare + ct / denominator
    # Marginal storage cost added only in policy variant (with subsidies)
    npv_expenses_mu_policy = npv_expenses_mu_co2 + (fft + st + marg_stor_cost) / denominator
    # Battery-only variant (short-term storage only)
    npv_expenses_mu_battery_only = (it_mu + ft + omt + battery_cost) / denominator

    # 1b – Expenses – average units (for electricity pricing)
    npv_expenses_av_no_policy = (it_av + ft + omt + stor_cost) / denominator
    npv_expenses_av_only_co2 = npv_expenses_av_no_policy + ct / denominator
    npv_expenses_av_all = npv_expenses_av_no_policy + (st + ct) / denominator
    npv_expenses_av_all_but_co2 = npv_expenses_av_no_policy + st / denominator  # Policies without CO2
    npv_expenses_av_all_policy = (it_av + ft + ct + omt + stor_cost + fft + st) / denominator

    # 1c - Operation costs (for MLCO output)
    npv_operation = (ft + omt + stor_cost + marg_stor_cost) / denominator

    # 2 – Utility
    npv_utility = np.where(lt_mask, 1, 0) / denominator

    # 3 – Standard deviation (propagation of error, matches cascading)
    # Include carbon cost std (dct) and 15% discount factor adjustment
    dct = get_cost_component(bcet[:, :, c2ti['2 std ($/MWh)']], 1, lt_mask)
    variance_terms = dit_mu**2 + dft**2 + domt**2 + dct**2 + dstor_cost**2
    summed_variance = np.sum(variance_terms / (denominator**2), axis=2)
    # Add 15% of total expenses as additional variance (discount rate uncertainty)
    variance_plus_dcf = summed_variance + (np.sum(npv_expenses_av_all, axis=2) * 0.15)**2

    utility_sum = np.sum(npv_utility, axis=2)
    
    
    # 1-levelised cost variants in $/MWh
    # Marginal unit costs (for investor decisions)
    lcoe_bare = np.sum(npv_expenses_mu_bare, axis=2) / utility_sum
    lcoe_mu_all_policies = np.sum(npv_expenses_mu_policy, axis=2) / utility_sum - data['MEFI'][:, :, 0]
    lcoeco2 = np.sum(npv_expenses_mu_co2, axis=2) / utility_sum
    lcoe_battery_only = np.sum(npv_expenses_mu_battery_only, axis=2) / utility_sum
    dlcoe = np.sqrt(variance_plus_dcf) / utility_sum  # Updated std calculation
    lcoo = np.sum(npv_operation, axis=2) / utility_sum  # Levelised cost of operation

    # Average unit costs (for electricity pricing via MEWP)
    lcoe_av = np.sum(npv_expenses_av_all_policy, axis=2) / utility_sum
    lcoe_only_co2 = np.sum(npv_expenses_av_only_co2, axis=2) / utility_sum - data['MEFI'][:, :, 0]
    lcoe_incl_co2 = np.sum(npv_expenses_av_all, axis=2) / utility_sum - data['MEFI'][:, :, 0]
    lcoe_all_but_co2 = np.sum(npv_expenses_av_all_but_co2, axis=2) / utility_sum - data['MEFI'][:, :, 0]

    # LCOE augmented with gamma values AND value factor (matches cascading branch)
    # Formula: METC = lcoe_mu_all_policies * (1 + Gamma) / ValueFactor
    # This properly accounts for VRE intermittency (value factor < 1 for solar/wind)
    gamma = bcet[:, :, c2ti['22 Gamma']]
    value_factor = bcet[:, :, c2ti['23 Value factor']]
    # Guard against division by zero (value_factor should never be 0, but be safe)
    value_factor = np.where(value_factor < 0.01, 1.0, value_factor)
    lcoe_mu_gamma = lcoe_mu_all_policies * (1 + gamma) / value_factor

    # Pass to variables that are stored outside.
    data['MEWC'][:, :, 0] = lcoe_bare       # The real bare LCOE without taxes
    data['MECW'][:, :, 0] = lcoeco2         # The real bare LCOE with CO2 costs
    data['MECC'][:, :, 0] = lcoe_all_but_co2  # LCOE with policy, without CO2 (matches cascading)
    data['METC'][:, :, 0] = lcoe_mu_gamma   # As seen by consumer (generalised cost with value factor)
    data['MTCD'][:, :, 0] = dlcoe           # Variation on the LCOE distribution
    data['MECW battery only'][:, :, 0] = lcoe_battery_only  # LCOE with only short-term storage

    # Additional LCOE variants for MEWP electricity pricing
    data['MECC only CO2'][:, :, 0] = lcoe_only_co2   # Bare LCOE + CO2 costs only (for old capacity pricing)
    data['MECC incl CO2'][:, :, 0] = lcoe_incl_co2   # LCOE with subsidies + CO2 (for new capacity pricing)
    data['MLCO'][:, :, 0] = lcoo                     # Levelised cost of operation

    data['MWIC'][:, :, 0] = bcet[:, :, 2]
    data['MWFC'][:, :, 0] = bcet[:, :, 4]
    data['MCOC'][:, :, 0] = bcet[:, :, 0]
    data['MCFC'][:, :, 0] = bcet[:, :, c2ti['11 Decision Load Factor']]  # Marginal capacity factor

    mwmc_condition = np.rint(data['MSAL'][:, 0, 0]) > 1
    data['MWMC'][:, :, 0] = np.where(mwmc_condition[:, None], 
                                 bcet[:, :, 0] + bcet[:, :, 4] + bcet[:, :, 6] + (data['MSSP'][:, :, 0] + data['MLSP'][:, :, 0]) / 1000,
                                 bcet[:, :, 0] + bcet[:, :, 4] + bcet[:, :, 6])

    data['MMCD'][:, :, 0] = np.sqrt(bcet[:, :, 1] ** 2 + bcet[:, :, 5] ** 2 + bcet[:, :, 7] ** 2)

      
    # Check if METC is nan
    if np.isnan(data['METC']).any():
        nan_indices_metc = np.where(np.isnan(data['METC']))
        raise ValueError(f"NaN values detected in lcoe ('metc') at indices: {nan_indices_metc}")
              

    return data
