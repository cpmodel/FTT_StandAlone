# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_lcoe.py
=========================================
Power LCOE FTT module.


Functions included:
    - set carbon_tax
        Determine carbon tax from the regional CO2 tax input
    - get_lcoe
        Calculate levelized costs

"""

# Third party imports
import numpy as np

# Local library imports
from ftt_source.ftt_core.ftt_get_levelised_costs import get_levelised_costs_with_build


def set_carbon_tax(data, c2ti, year):
    '''
    Apply the carbon price to power sector technologies based on their efficiencies
    
    '''

    co2_tax = data["CO2taxP"][:, None]
    carbon_costs = (
                    data['BCET'][:, :, c2ti['15 Emissions (tCO2/GWh)']]    # Emission per GWh
                    * co2_tax                                               # Carbon price in USD/tCO2
                    / 1000                                                 # Conversion from GWh to MWh
                    )
    
    
    if np.isnan(carbon_costs).any():
        print(f"Carbon price is nan in year {year}")
        print(f"The arguments of the nans are {np.argwhere(np.isnan(carbon_costs))}")
        print(f"Emissions intensity {data['BCET'][:, :, c2ti['15 Emissions (tCO2/GWh)']]}")
        
        raise ValueError

    return carbon_costs

# %% lcoe
# -----------------------------------------------------------------------------
# --------------------------- LCOE function -----------------------------------
# -----------------------------------------------------------------------------
  
def get_lcoe(data, titles, gamma_mode='multiplicative'):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of electricity in $2013/MWh. It includes
    intangible costs (gamma values) and determines the investor preferences.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the current year
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.
    gamma_mode: string
        To be removed later, whether gamma values are additive (old) or multiplicative (new)

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the current year
        The values are updated and returned to the main routine.
    """

    # Categories for the cost matrix (BCET)
    c2ti = {category: index for index, category in enumerate(titles['C2TI'])}

    # Cost matrix
    bcet = data['BCET']

    # Plant parameters
    lt = bcet[:, :, c2ti['9 Lifetime (years)']]
    bt = bcet[:, :, c2ti['10 Lead Time (years)']]
    dr = bcet[:, :, c2ti['17 Discount Rate (%)']]

    # Capacity factor of marginal unit (for decision-making)
    cf_mu = bcet[:, :, c2ti['11 Decision Load Factor']]
    cf_mu[cf_mu < 0.000001] = 0.000001

    # Average capacity factor (for electricity price)
    cf_av = data['MEWL']
    cf_av[cf_av < 0.000001] = 0.000001

    # Annual service delivered per kW installed, in MWh/kW
    hours_per_year = 8766
    service_mu = cf_mu * hours_per_year / 1000
    service_av = cf_av * hours_per_year / 1000

    # Upfront costs ($/kW), their standard deviation and the subsidy or tax on them.
    # The subsidy is always converted with the marginal capacity factor.
    investment = bcet[:, :, c2ti['3 Investment ($/kW)']]
    investment_sd = bcet[:, :, c2ti['4 std ($/MWh)']]
    investment_policy = investment * data['MEWT']
    investment_policy_av = investment_policy * service_av / service_mu

    # Per-MWh costs. Fuel costs are scaled by FPIX when provided.
    fpix = data['FPIX'] if 'FPIX' in data else 1.0
    fuel = bcet[:, :, c2ti['5 Fuel ($/MWh)']] * fpix
    fuel_tax = fuel * data['MTFT']
    fuel_sd = bcet[:, :, c2ti['6 std ($/MWh)']]
    om = bcet[:, :, c2ti['7 O&M ($/MWh)']]
    om_sd = bcet[:, :, c2ti['8 std ($/MWh)']]
    carbon = bcet[:, :, c2ti['1 Carbon Costs ($/MWh)']]
    carbon_sd = bcet[:, :, c2ti['2 std ($/MWh)']]

    # Who pays for storage, and total storage costs
    msal_rounded = np.rint(data['MSAL'])
    storage_sum = (data['MSSP'] + data['MLSP']) / 1000
    marg_storage_sum = (data['MSSM'] + data['MLSM']) / 1000

    stor_mask = (msal_rounded >= 2)[:, None]
    stor_cost = np.where(stor_mask, storage_sum, 0)
    stor_sd = 0.2 * stor_cost  # Assume standard deviation of 20%

    # Battery-only storage cost (short-term only)
    battery_cost = np.where(stor_mask, data['MSSP'] / 1000, 0)

    # Marginal storage cost
    marg_mask = (msal_rounded >= 3)[:, None]
    marg_stor_cost = np.where(marg_mask, marg_storage_sum, 0)

    # Annual costs per kW installed, in $/kW/year
    annual_mu = (fuel + om + stor_cost) * service_mu
    annual_battery_only = (fuel + om + battery_cost) * service_mu
    annual_policy_mu = (carbon + fuel_tax + marg_stor_cost) * service_mu
    annual_carbon_mu = carbon * service_mu
    annual_sd_mu = np.sqrt(fuel_sd**2 + om_sd**2 + carbon_sd**2 + stor_sd**2) * service_mu

    annual_av = (fuel + om + stor_cost) * service_av
    annual_carbon_av = carbon * service_av

    def levelised_cost(annual, annual_policies, upfront_policies, service,
                       annual_sd=0.0):
        '''Levelised cost per MWh without and with policies, and its standard deviation'''
        return get_levelised_costs_with_build(
            upfront=investment, upfront_policies=upfront_policies,
            upfront_sd=investment_sd,
            annual=annual, annual_policies=annual_policies, annual_sd=annual_sd,
            service_delivered=service, service_sd=0.0,
            lifetimes=lt, leadtimes=bt, r=dr)

    # Marginal unit costs (for investor decisions)
    lcoe_bare, lcoe_policy, _ = levelised_cost(
        annual_mu, annual_policy_mu, investment_policy, service_mu)
    lcoe_mu_all_policies = lcoe_policy - data['MEFI']
    _, lcoeco2, _ = levelised_cost(annual_mu, annual_carbon_mu, 0.0, service_mu)
    lcoe_battery_only, _, _ = levelised_cost(annual_battery_only, 0.0, 0.0, service_mu)
    _, _, lcoe_sd = levelised_cost(annual_mu, 0.0, 0.0, service_mu, annual_sd_mu)

    # Levelised cost of operation: constant over the lifetime, so no discounting needed
    lcoo = fuel + om + stor_cost + marg_stor_cost

    # Average unit costs (for electricity pricing via MEWP)
    lcoe_av_bare, lcoe_av_incl_co2_gross, _ = levelised_cost(
        annual_av, annual_carbon_av, investment_policy_av, service_av)
    _, lcoe_av_only_co2_gross, _ = levelised_cost(annual_av, annual_carbon_av, 0.0, service_av)
    lcoe_only_co2 = lcoe_av_only_co2_gross - data['MEFI']
    lcoe_incl_co2 = lcoe_av_incl_co2_gross - data['MEFI']
    # Subsidies without the CO2 cost: removing the CO2 effect, which is additive
    lcoe_all_but_co2 = (lcoe_av_incl_co2_gross
                        - (lcoe_av_only_co2_gross - lcoe_av_bare)
                        - data['MEFI'])

    # Add 15% of the total expenses as additional uncertainty (discount rate uncertainty)
    dlcoe = np.sqrt(lcoe_sd**2 + (0.15 * lcoe_av_incl_co2_gross)**2)

    # LCOE augmented with gamma values (and optionally value factor)
    gamma = bcet[:, :, c2ti['22 Gamma']]
    value_factor = bcet[:, :, c2ti['23 Value factor']]
    value_factor = np.where(value_factor < 0.01, 1.0, value_factor)
    if gamma_mode == 'additive':
        lcoe_mu_gamma = (lcoe_mu_all_policies + gamma) / value_factor
    else:
        lcoe_mu_gamma = lcoe_mu_all_policies * (1 + gamma) / value_factor

    # Pass to variables that are stored outside.
    data['MEWC'] = lcoe_bare       # The real bare LCOE without taxes
    data['MECW'] = lcoeco2         # The real bare LCOE with CO2 costs
    data['MECC'] = lcoe_all_but_co2  # LCOE with policy, without CO2 (matches cascading)
    data['METC'] = lcoe_mu_gamma   # As seen by consumer (generalised cost with value factor)
    data['MTCD'] = dlcoe           # Variation on the LCOE distribution
    data['MECW battery only'] = lcoe_battery_only  # LCOE with only short-term storage

    # Additional LCOE variants for MEWP electricity pricing
    data['MECC only CO2'] = lcoe_only_co2   # Bare LCOE + CO2 costs only (for old capacity pricing)
    data['MECC incl CO2'] = lcoe_incl_co2   # LCOE with subsidies + CO2 (for new capacity pricing)
    data['MLCO'] = lcoo                     # Levelised cost of operation

    data['MWIC'] = bcet[:, :, 2]
    data['MWFC'] = bcet[:, :, 4]
    data['MCOC'] = bcet[:, :, 0]
    data['MCFC'] = bcet[:, :, c2ti['11 Decision Load Factor']]  # Marginal capacity factor

    # Marginal costs
    data['MWMC'] = bcet[:, :, 0] + bcet[:, :, 4] + bcet[:, :, 6]

    data['MMCD'] = np.sqrt(bcet[:, :, 1] ** 2 + bcet[:, :, 5] ** 2 + bcet[:, :, 7] ** 2)

      
    # Check if METC is nan
    if np.isnan(data['METC']).any():
        nan_indices_metc = np.where(np.isnan(data['METC']))
        raise ValueError(f"NaN values detected in lcoe ('metc') at indices: {nan_indices_metc}")
              

    return data
