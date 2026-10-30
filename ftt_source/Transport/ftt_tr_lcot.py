# -*- coding: utf-8 -*-
"""
=========================================
ftt_tr_lcot.py
=========================================

Get the levelised cost of transport, in lognormal space


Local library imports:

Functions included:
    - solve
        Main solution function for the module
    - get_lcot
        Calculate levelised cost of transport

variables:
cf = capacity factor
ff = filling factors
ns = number of seats


"""
import numpy as np

from ftt_source.ftt_core.ftt_get_levelised_costs import get_levelised_costs


def set_carbon_tax(data, c3ti, year):
    '''
    Convert the carbon price from USD / tCO2 to 2022$/pkm
    Apply the carbon price to transport sector technologies based on their emission factors

    Returns:
        Carbon costs per country and technology (2D)
    '''

    # Number of seats
    ns = data['BTTC'][:, :, c3ti['15 Seats/Veh']]

    # Occupancy rates
    ff = data['BTTC'][:, :, c3ti['11 occupancy rate p/sea']]


    carbon_costs = (data["CO2taxTr"][:, None]                                    # Carbon price in USD/tCO2
                    * data['BTTC'][:, :, c3ti['14 CO2Emissions']]             # g CO2 / km
                    / ns / ff                                                 # Conversion from per km to per pkm
                    / 10**6                                                  # Conversion from grams to tonnes.
                    )


    if np.isnan(carbon_costs).any():
        print(f"Carbon price is nan in year {year}")
        print(f"The arguments of the nans are {np.argwhere(np.isnan(carbon_costs))}")
        print(f"Emissions intensity {data['BTTC'][:, :, c3ti['14 CO2Emissions']]}")

        raise ValueError

    return carbon_costs


# %% lcot
# -----------------------------------------------------------------------------
# --------------------------- LCOT function -----------------------------------
# -----------------------------------------------------------------------------
def get_lcot(data, titles, carbon_costs, year):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of transport in 2012$/p-km per
    vehicle type. It includes intangible costs (gamma values) and together
    determines the investor preferences.
    """

    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}

    # Taxable categories for fuel tax, CNG, EVs and H2 exempt
    taxable_fuels = np.ones([len(titles['RTI']), len(titles['VTTI'])])
    taxable_fuels[:, 12:15] = 0   # CNG
    taxable_fuels[:, 18:21] = 0   # EVs
    taxable_fuels[:, 24:27] = 0   # Hydrogen

    # Taxable categories for carbon tax: only EVs and H2 exempt
    tf_carbon = np.ones((len(titles['RTI']), len(titles['VTTI'])))
    tf_carbon[:, 18:21] = 0   # EVs
    tf_carbon[:, 24:27] = 0   # Hydrogen

    bttc = data['BTTC']

    # Vehicle and usage parameters
    lt = bttc[:, :, c3ti['8 lifetime']]
    cf = bttc[:, :, c3ti['12 Cap_F (Mpkm/kseats-y)']]
    ff = bttc[:, :, c3ti['11 occupancy rate p/sea']]
    ns = bttc[:, :, c3ti['15 Seats/Veh']]
    en = bttc[:, :, c3ti['9 energy use (MJ/km)']]

    conv_full = 1 / ns / ff / cf / 1000
    conv_pkm = 1 / ns / ff

    # Upfront cost (base, policy, standard deviation)
    upfront = bttc[:, :, c3ti['1 Prices cars (USD/veh)']] * conv_full
    upfront_pol = (upfront * data['Base registration rate']      # Tax as share of the price
                   + conv_full * (data['TTVT']                   # Purchase tax
                                  + data['RTCO'][:, None] * bttc[:, :, c3ti['14 CO2Emissions']]))  # CO2-dependent tax
    upfront_sd = bttc[:, :, c3ti['2 Std of price']] * conv_full

    # Annual variable cost (base, policy, standard deviation)
    annual = (bttc[:, :, c3ti['3 fuel cost (USD/km)']]
              + bttc[:, :, c3ti['5 O&M costs (USD/km)']]) * conv_pkm
    annual_pol = (carbon_costs * tf_carbon                             # Carbon tax
                  + data['RTFT'][:, None] * en * conv_pkm * taxable_fuels  # Fuel tax
                  + data['TTRT'] * conv_full)                          # Yearly road tax
    annual_sd = np.sqrt(bttc[:, :, c3ti['6 std O&M']]**2
                        + bttc[:, :, c3ti['4 std fuel cost']]**2) * conv_pkm

    # The upfront and annual costs have already been converted to the service
    # provided, so the service delivered is 1
    lcot, lcot_pol, _, lcot_sd = get_levelised_costs(
        upfront=upfront,
        upfront_policies=upfront_pol,
        upfront_sd=upfront_sd,
        annual=annual,
        annual_policies=annual_pol,
        annual_sd=annual_sd,
        service_delivered=1,
        service_sd=0.1,    # Assume a 10% variation in load factors
        lifetimes=lt,
        r=bttc[:, :, c3ti['7 Discount rate']])

    # Generalised cost, and transformation into lognormal space
    gamma = bttc[:, :, c3ti['13 Gamma']]
    lcot_pol_gam = lcot_pol * (1 + gamma)
    log_lcot_pol = np.log(lcot_pol**2 / np.sqrt(lcot_sd**2 + lcot_pol**2)) + gamma
    log_lcot_pol_sd = np.sqrt(np.log(1.0 + lcot_sd**2 / lcot_pol**2))

    # Pass to variables that are stored outside.
    data['TEWC'] = lcot             # The real bare LCOT without taxes
    data['TETC'] = lcot_pol         # The real bare LCOT with taxes
    data['TEGC'] = lcot_pol_gam     # As seen by consumer (generalised cost)
    data['TELC'] = log_lcot_pol     # In lognormal space
    data['TECD'] = lcot_sd          # Variation on the LCOT distribution
    data['TLCD'] = log_lcot_pol_sd  # Log variation on the LCOT distribution

    return data
