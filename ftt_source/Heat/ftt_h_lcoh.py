# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:30 2024

@author: Alex Edwards and Femke Nijsse

=========================================
ftt_h_lcoh.py
=========================================
Domestic Heat FTT module.


This is the main file for FTT: Heat, which models technological diffusion of
domestic heat technologies due to consumer decision making. 
Consumers compare the **levelised cost of heat**, which leads to changes in the
market shares of different technologies.

The outputs of this module include levelised cost of heat technologies

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Element-wise divide which replaces divide-by-zeros with zeros


Functions included:
    - get_lcoh
        Calculate levelised cost of heating
        
variables: 
cf = capacity factor
ce = conversion efficiency


"""

# Third party imports
import numpy as np

# Local library imports
from ftt_source.support.divide import divide
from ftt_source.ftt_core.ftt_get_levelised_costs import get_levelised_costs


def set_carbon_tax(data, c4ti):
    '''
    Convert the carbon price from USD / tCO2 to 2020 euros / kWhUD.
    Apply the carbon price to heat sector technologies based on their emission factors

    Returns:
        Carbon costs per country and technology (2D)
    '''
 
    co2_tax_h = data["CO2taxH"]

    carbon_costs = (co2_tax_h[:, np.newaxis]    # Carbon price in USD/tCO2
                    * data['BHTC'][:, :, c4ti['15 Emission factor']]     # kg CO2 / MWh 
                     / 1000 / 1000              # Conversion from C to CO2 and MWh to kWh, kg to tonne 
                     * data['REXX'][33]  # USD to EUR exchange rate
                     / 1.18                     # Hardcoded inflation from 2023 to 2020
                    )
    
    
    if np.isnan(carbon_costs).any():
        #print(f"Carbon price is nan in year {year}")
        print(f"The arguments of the nans are {np.argwhere(np.isnan(carbon_costs))}")
        print(f"Emissions intensity {data['BHTC'][:, :, c4ti['15 Emission factor']]}")
        
        raise ValueError

    return carbon_costs


# %% LCOH
# --------------------------------------------------------------------------
# -------------------------- LCOH function ---------------------------------
# --------------------------------------------------------------------------

def get_lcoh(data, titles, carbon_costs):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of heat in 2020 Euros/kWh per
    boiler type. It includes intangible costs (gamma values) and together
    determines the investor preferences.
    """
    # Categories for the cost matrix (BHTC)
    c4ti = {category: index for index, category in enumerate(titles['C4TI'])}

    # Cost matrix
    bhtc = data['BHTC']

    # Heating device lifetimes
    lt = bhtc[:, :, c4ti['5 Lifetime']]

    # Pay-back thresholds
    pb = bhtc[:, :, c4ti['16 Payback time, mean']]
    dpb = bhtc[:, :, c4ti['17 Payback time, SD']]

    # Capacity factor
    cf = bhtc[:, :, c4ti['13 Capacity factor mean']]
    dcf = bhtc[:, :, c4ti['14 Capacity factor SD']]
    cf[cf < 0.0001] = 0.0001
    conv_cf = 1 / (cf * 1000)

    # Conversion efficiency
    ce = bhtc[:, :, c4ti['9 Conversion efficiency']]
    ce[ce < 0.0001] = 0.0001
    conv_ce = 1 / ce

    # Upfront costs, policies and standard deviations
    upfront = bhtc[:, :, c4ti['1 Inv cost mean (EUR/kW)']] * conv_cf
    upfront_pol = upfront * data['HTVS']
    upfront_sd = bhtc[:, :, c4ti['2 Inv Cost SD']] * conv_cf

    # Annual costs, policies and standard deviations
    fuel_costs = bhtc[:, :, c4ti['10 Fuel cost  (EUR/kWh)']] * data['HEWP'] * conv_ce
    annual = bhtc[:, :, c4ti['3 O&M mean (EUR/kW)']] * conv_cf + fuel_costs
    annual_pol_no_carbon = (data['HTRT'] * conv_ce    # Subsidy or tax on fuel use
                            - data['HEFI'])           # Feed-in tariffs
    annual_pol = annual_pol_no_carbon + carbon_costs
    annual_sd = np.sqrt((bhtc[:, :, c4ti['4 O&M SD']] * conv_cf)**2
                        + (bhtc[:, :, c4ti['11 Fuel cost SD']] * fuel_costs)**2)

    # The upfront and annual costs have already been converted to the service
    # provided, so the service delivered is 1
    lcoh, lcoh_pol, lcoh_sd = get_levelised_costs(
        upfront=upfront,
        upfront_policies=upfront_pol,
        upfront_sd=upfront_sd,
        annual=annual,
        annual_policies=annual_pol,
        annual_sd=annual_sd,
        service_delivered=1,
        service_sd=dcf / cf,
        lifetimes=lt,
        r=bhtc[:, :, c4ti['8 Discount rate']])

    # Marginal costs of existing units, and pay-back costs of potential alternatives
    marginal_costs = annual + annual_pol_no_carbon
    payback_costs = marginal_costs + (upfront + upfront_pol) / pb
    payback_sd = np.sqrt(annual_sd**2
                         + divide(upfront_sd**2, pb**2)
                         + divide(upfront**2, pb**4) * dpb**2)

    # Costs augmented with non-pecuniary costs
    gamma = bhtc[:, :, c4ti['12 Gamma value']]
    lcoh_pol_gam = lcoh_pol * (1 + gamma)
    marginal_costs_gam = marginal_costs * (1 + gamma)
    payback_costs_gam = payback_costs * (1 + gamma)

    # Pass to variables that are stored outside.
    data['HEWC'] = lcoh                  # The real bare LCOH without taxes
    data['HETC'] = lcoh_pol              # The real bare LCOH with taxes
    data['HGC1'] = lcoh_pol_gam          # As seen by consumer (generalised cost)
    data['HWCD'] = lcoh_sd               # Variation on the LCOH distribution
    data['HGC2'] = marginal_costs_gam    # Total marginal costs
    data['HGD2'] = annual_sd             # SD of Total marginal costs
    data['HGC3'] = payback_costs_gam     # Total payback costs
    data['HGD3'] = payback_sd            # SD of Total payback costs

    return data
