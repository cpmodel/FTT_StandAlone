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
import time

# Local library imports
from SourceCode.ftt_core.ftt_get_levelised_costs import get_levelised_costs_with_build


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

def get_lcoe(data, titles, year):
    """
    Calculate levelized costs using generic CRF-based function and compare with original.
    """
    # Start timing for the new implementation
    start = time.perf_counter()

    # Categories for the cost matrix (BCET)
    c2ti = {category: index for index, category in enumerate(titles['C2TI'])}
    
    # Cost matrix
    bcet = data['BCET']
    
    # Plant parameters
    lt = bcet[:, :, c2ti['9 Lifetime (years)']]
    bt = bcet[:, :, c2ti['10 Lead Time (years)']]
    dr = bcet[:, :, c2ti['17 Discount Rate (%)']]
    
    # Testing with some policies:
    data['MTFT'][:, :, 0] = 0.2
    data['MEWT'][:, :, 0] = 0.2
    #data['MEFI'][:, :, 0] = 20
    
    
    
    # Capacity factors
    cf_mu = bcet[:, :, c2ti['11 Decision Load Factor']]  # Marginal unit (for decision-making)
    cf_mu[cf_mu < 0.000001] = 0.000001
    cf_av = data['MEWL'][:, :, 0]  # Average capacity factor (for electricity price)
    cf_av[cf_av < 0.000001] = 0.000001
    
    # Convert kW to MWh
    hours_per_year = 8766
    service_marg = cf_mu * hours_per_year / 1000  # MWh per kW
    service_av = cf_av * hours_per_year / 1000
    
    # Upfront costs (in $/kW)
    upfront_inv = bcet[:, :, c2ti['3 Investment ($/kW)']]
    upfront_inv_sd = bcet[:, :, c2ti['4 std ($/MWh)']]            
    upfront_subsidy = upfront_inv * data['MEWT'][:, :, 0]
    

    # Storage costs (annual in $/MWh)
    msal_rounded = np.rint(data['MSAL'][:, 0, 0])
    storage_sum = (data['MSSP'] + data['MLSP']) / 1000
    marg_storage_sum = (data['MSSM'] + data['MLSM']) / 1000
    
    stor_mask = (msal_rounded >= 2)
    variable_storage = np.where(stor_mask, storage_sum, 0)[:, :, 0]
    variable_storage_sd = 0.2 * variable_storage
    
    marg_mask = (msal_rounded >= 3)
    variable_marg_storage = np.where(marg_mask, marg_storage_sum, 0)[:, :, 0]
    
    # Variable costs (in $/MWh)
    variable = (bcet[:, :, c2ti['5 Fuel ($/MWh)']]
                + bcet[:, :, c2ti['7 O&M ($/MWh)']]
                + variable_storage)
    variable_pol = (bcet[:, :, c2ti['5 Fuel ($/MWh)']] * data['MTFT'][:, :, 0] # Fuel tax
                    + bcet[:, :, c2ti['1 Carbon Costs ($/MWh)']])
    variable_sd = np.sqrt(bcet[:, :, c2ti['6 std ($/MWh)']] ** 2
                          + bcet[:, :, c2ti['8 std ($/MWh)']] ** 2
                          + variable_storage_sd ** 2)
    
    # Annual costs for new (marginal) units -> lower capacity factor + more anticipated storage
    annual_marg = (variable + variable_marg_storage) * service_marg
    annual_pol_marg = variable_pol * service_marg
    annual_sd_marg = variable_sd * service_marg
    annual_carbon_marg = bcet[:, :, c2ti['1 Carbon Costs ($/MWh)']] * service_marg
    
    # Annual costs for average units, used for LCOE calculations
    annual_av = (variable) * service_av
    annual_pol_av = variable_pol * service_av
    annual_sd_av = variable_sd * service_av
  
    
    
    # Standard deviations
    service_sd = 0.0  # Assuming no uncertainty in capacity factor for now
    
    # Calculate LCOE for marginal unit (for decision-making)
    # 1. Bare and with all policies (carbon + subsidies + taxes)
    lcoe_bare, lcoe_policy, lcoe_sd = get_levelised_costs_with_build(
        upfront=upfront_inv,
        upfront_policies=upfront_subsidy,
        upfront_sd=upfront_inv_sd,
        annual=annual_marg,
        annual_policies=annual_pol_marg,
        annual_sd=annual_sd_marg,
        service_delivered=service_marg,
        service_sd=service_sd,
        lifetimes=lt,
        leadtimes=bt,
        r=dr
    )
    
    # Subtract feed-in tariff
    lcoe_policy = lcoe_policy - data['MEFI'][:, :, 0]
    

    # 2. With carbon costs only
    _, lcoe_co2, _ = get_levelised_costs_with_build(
        upfront=upfront_inv,
        upfront_policies=0.0,
        upfront_sd=upfront_inv_sd,
        annual=annual_marg,
        annual_policies=annual_carbon_marg,
        annual_sd=annual_sd_marg,
        service_delivered=service_marg,
        service_sd=service_sd,
        lifetimes=lt,
        leadtimes=bt,
        r=dr
    )
       
  
    # 3. Average LCOE (for electricity price, using average capacity factor)
    _, lcoe_av, _ = get_levelised_costs_with_build(
        upfront=upfront_inv,
        upfront_policies=upfront_subsidy,
        upfront_sd=upfront_inv_sd,
        annual=annual_av,
        annual_policies=annual_pol_av,
        annual_sd=annual_sd_av,
        service_delivered=service_av,
        service_sd=service_sd,
        lifetimes=lt,
        leadtimes=bt,
        r=dr
    )
    
    # LCOE augmented with gamma values
    lcoe_mu_gamma = lcoe_policy + bcet[:, :, c2ti['22 Gamma']]
    
    # Store results
    data['MEWC'][:, :, 0] = lcoe_bare       # The real bare LCOE without taxes
    data['MECW'][:, :, 0] = lcoe_co2        # The real LCOE with carbon costs
    data['MECC'][:, :, 0] = lcoe_av         # The average LCOE costs including policies
    data['METC'][:, :, 0] = lcoe_mu_gamma   # As seen by consumer (generalised cost)
    data['MTCD'][:, :, 0] = lcoe_sd         # Variation on the LCOE distribution
    
    data['MWIC'][:, :, 0] = bcet[:, :, 2]
    data['MWFC'][:, :, 0] = bcet[:, :, 4]
    data['MCOC'][:, :, 0] = bcet[:, :, 0]
    
    mwmc_condition = np.rint(data['MSAL'][:, 0, 0]) > 1
    data['MWMC'][:, :, 0] = np.where(mwmc_condition[:, None], 
                                 bcet[:, :, 0] + bcet[:, :, 4] + bcet[:, :, 6] + (data['MSSP'][:, :, 0] + data['MLSP'][:, :, 0]) / 1000,
                                 bcet[:, :, 0] + bcet[:, :, 4] + bcet[:, :, 6])
    
    data['MMCD'][:, :, 0] = np.sqrt(bcet[:, :, 1] ** 2 + bcet[:, :, 5] ** 2 + bcet[:, :, 7] ** 2)

    # Elapsed time for new implementation
    elapsed = time.perf_counter() - start

    # --- Testing block: compare with original implementation ---
    start2 = time.perf_counter()
    lcoe_bare_old, lcoe_co2_old, lcoe_av_old, lcoe_mu_gamma_old, lcoe_sd_old = get_lcoe_original(data, titles)
    elapsed2 = time.perf_counter() - start2

    print(f"Runtime: {elapsed2 / elapsed:.2f}x faster (new is {elapsed:.4f}s, old is {elapsed2:.4f}s)")

    def msre(a, b, eps=1e-12):
        denom = np.where(np.abs(b) < eps, eps, b)
        return np.average(((a - b) / denom) ** 2)

    print('Difference between new and old (mean squared relative error):')
    print(f'MEWC (bare LCOE): {msre(data["MEWC"][:, :, 0], lcoe_bare_old):.6e}')
    print(f'MECW (LCOE w/ CO2): {msre(data["MECW"][:, :, 0], lcoe_co2_old):.6e}')
    print(f'MECC (avg LCOE): {msre(data["MECC"][:, :, 0], lcoe_av_old):.6e}')
    print(f'METC (w/ gamma): {msre(data["METC"][:, :, 0], lcoe_mu_gamma_old):.6e}')
    print(f'MTCD (std dev): {msre(data["MTCD"][:, :, 0], lcoe_sd_old):.6e}')

    return data


def get_lcoe_original(data, titles):
    """
    Original LCOE implementation (NPV over lifetime) that returns arrays.
    Does not mutate the data container. Used for validation/testing.
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
    
    # Marginal storage cost
    marg_mask = (msal_rounded >= 3)
    marg_stor_cost = np.where(marg_mask, marg_storage_sum, 0)
    marg_stor_cost = get_cost_component(marg_stor_cost[:, :, 0], 1, lt_mask)
    
    # Net present value calculations
    dr = bcet[:, :, c2ti['17 Discount Rate (%)'], None]
    denominator = (1 + dr) ** full_lt_mat
    
    # 1 – Expenses
    npv_expenses_mu_bare = (it_mu + ft + omt + stor_cost + marg_stor_cost) / denominator
    npv_expenses_mu_co2 = npv_expenses_mu_bare + ct / denominator
    npv_expenses_mu_policy = npv_expenses_mu_co2 + (fft + st) / denominator
    npv_expenses_av_all_policy = (it_av + ft + ct + omt + stor_cost + fft + st) / denominator
    
    # 2 – Utility
    npv_utility = np.where(lt_mask, 1, 0) / denominator
    
    # 3 – Standard deviation (propagation of error)
    # npv_std = np.sqrt(dit_mu ** 2 + dft ** 2 + domt ** 2 + dstor_cost**2) / denominator
    npv_var = (dit_mu ** 2 + dft ** 2 + domt ** 2 + dstor_cost**2) / denominator**2
    
    utility_sum = np.sum(npv_utility, axis=2)
    
    # 1-levelised cost variants in $/MWh
    lcoe_bare = np.sum(npv_expenses_mu_bare, axis=2) / utility_sum
    lcoe_mu_all_policies = np.sum(npv_expenses_mu_policy, axis=2) / utility_sum - data['MEFI'][:, :, 0]
    lcoeco2 = np.sum(npv_expenses_mu_co2, axis=2) / utility_sum
    lcoe_av = np.sum(npv_expenses_av_all_policy, axis=2) / utility_sum
    dlcoe = np.sqrt(np.sum(npv_var, axis=2)) / utility_sum
    
    # LCOE augmented with gamma values
    lcoe_mu_gamma = lcoe_mu_all_policies + bcet[:, :, c2ti['22 Gamma']]

    return lcoe_bare, lcoeco2, lcoe_av, lcoe_mu_gamma, dlcoe
