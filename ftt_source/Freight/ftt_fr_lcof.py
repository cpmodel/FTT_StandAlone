# -*- coding: utf-8 -*-
"""
=========================================
ftt_fr_lcof.py
=========================================
Freight LCOF FTT module with optional feebate system.
########################

Functions included:
    - set_carbon_tax: Calculate carbon tax costs
    - verify_revenue_neutrality: Check feebate balance
    - get_lcof: Calculate levelized costs with optional feebate system
"""

import numpy as np

from ftt_source.ftt_core.ftt_get_levelised_costs import get_levelised_costs


def set_carbon_tax(data, c6ti):
    '''
    Apply the carbon price to freight sector technologies based on their emission factors

    Returns:
        Carbon costs per country and technology (2D)
    '''
    carbon_costs = (data["CO2taxFr"][:, None]                             # Carbon price in USD/tCO2
                    * data['BZTC'][:, :, c6ti['12 CO2 emissions (gCO2/km)']]    # g CO2 / km
                    / 10**6                                                     # Conversion from g to tonne
                    )
    
    if np.isnan(carbon_costs).any():
        print(f"The arguments of the nans are {np.argwhere(np.isnan(carbon_costs))}")
        print(f"Emissions intensity {data['BZTC'][:, :, c6ti['12 CO2 emissions (gCO2/km)']]}")
        raise ValueError
                       
    return carbon_costs

def verify_revenue_neutrality(data, titles, c6ti):
    """
    Verify revenue neutrality of the feebate system for countries with active subsidies.
    Only checks regions where BEV subsidies are implemented.
    """
    n_veh_classes = 5
    results = {}
    
    for r in range(len(titles['RTI'])):
        region = titles['RTI'][r]
        
        # Check if this region has any BEV subsidies
        bev_indices = [31, 32, 33]  # BEV indices
        ztvt = data['ZTVT']
        zewi = data['ZEWI']
        has_subsidies = np.any(ztvt[r, bev_indices] < 0)
        
        if not has_subsidies:
            continue
            
        results[region] = {'subsidy_costs': [], 'tax_revenues': []}
        print(f"\nChecking revenue neutrality for {region}")
        
        for veh_class in range(n_veh_classes):
            bev_indices_class = [idx for idx in [31, 32, 33] if idx % n_veh_classes == veh_class]
            ice_indices_class = [idx for idx in range(25) if idx % n_veh_classes == veh_class]
            
            # Calculate total subsidy being paid out
            bev_costs = data['BZTC'][r, bev_indices_class, c6ti['1 Purchase cost (USD/veh)']]
            bev_numbers = zewi[r, bev_indices_class]
            bev_subsidy_rates = ztvt[r, bev_indices_class]
            total_subsidy = np.sum(bev_costs * bev_numbers * abs(bev_subsidy_rates))
            
            if total_subsidy > 0:
                # Calculate total tax being collected
                ice_costs = data['BZTC'][r, ice_indices_class, c6ti['1 Purchase cost (USD/veh)']]
                ice_numbers = zewi[r, ice_indices_class]
                ice_tax_rates = ztvt[r, ice_indices_class]
                total_tax = np.sum(ice_costs * ice_numbers * ice_tax_rates)
                
                results[region]['subsidy_costs'].append(total_subsidy)
                results[region]['tax_revenues'].append(total_tax)
                
                print(f"\nVehicle Class {veh_class}")
                print(f"Total BEV subsidy cost: ${total_subsidy:,.2f}")
                print(f"Total ICE tax revenue: ${total_tax:,.2f}")
                print(f"Difference: ${total_tax - total_subsidy:,.2f}")
                print(f"Revenue neutral? {abs(total_tax - total_subsidy) < 1e-6}")
        
        if results[region]['subsidy_costs']:
            total_subsidies = sum(results[region]['subsidy_costs'])
            total_taxes = sum(results[region]['tax_revenues'])
            print(f"\nTOTALS FOR {region}:")
            print(f"Total subsidies: ${total_subsidies:,.2f}")
            print(f"Total tax revenue: ${total_taxes:,.2f}")
            print(f"Overall difference: ${total_taxes - total_subsidies:,.2f}")
            print("-" * 50)
    
    return results

def calculate_feebate_rates(data, titles, c6ti, n_veh_classes):
    """Calculate feebate rates for each vehicle class to maintain revenue neutrality"""
    for r in range(len(titles['RTI'])):
        for veh_class in range(n_veh_classes):
            # Get indices for this vehicle class
            bev_indices_class = [idx for idx in [31, 32, 33] if idx % n_veh_classes == veh_class]
            ice_indices_class = [idx for idx in range(25) if idx % n_veh_classes == veh_class]
            
            # Calculate BEV subsidies for this class
            bev_costs = data['BZTC'][r, bev_indices_class, c6ti['1 Purchase cost (USD/veh)']]
            zewi = data['ZEWI']
            ztvt = data['ZTVT']
            bev_numbers = zewi[r, bev_indices_class]
            total_subsidy = np.sum(
                bev_costs * bev_numbers * abs(ztvt[r, bev_indices_class])
            )
            
            # Calculate ICE tax rate for this class
            ice_numbers = zewi[r, ice_indices_class]
            ice_costs = data['BZTC'][r, ice_indices_class, c6ti['1 Purchase cost (USD/veh)']]
            denominator = np.sum(ice_costs * ice_numbers)
            
            if denominator > 0:
                ice_tax_rate = total_subsidy / denominator
                ice_tax_rate = min(ice_tax_rate, 1.0)  # Cap at 100%
                ztvt[r, ice_indices_class] = ice_tax_rate

def get_lcof(data, titles, carbon_costs, year):
    """
    Calculate levelized costs.

    Calculates the levelised cost of freight transport in 2012$/t-km per
    vehicle type. These costs are then converted into 2010 Euros/t-km per vehicle type.
    It includes intangible costs (gamma values) and together
    determines the investor preferences.

    Parameters
    -----------
    data: dictionary
        Model variables for the current year
    titles: dictionary
        Titles contains all permissible dimension titles of the model.

    Returns
    ----------
    data: dictionary
        Data is a dictionary with the data of the current year for all variables.
        The values are updated and returned to the main routine.

    Notes
    ---------
    Calculate levelized costs with optional feebate system.
    Feebate is activated if Feebate_active flag in data is set to 1.
    """
    
    # Categories for the cost matrix (BZTC)
    c6ti = {category: index for index, category in enumerate(titles['C6TI'])}
    n_veh_classes = 5

    # Check if feebate system is active
    if data.get('Feebate active', 0) == 1:
        if data.get('iteration', 0) == 0:
            print("\nFeebate system is active")
            calculate_feebate_rates(data, titles, c6ti, n_veh_classes)
            print("\nVerifying revenue neutrality...")
            verify_revenue_neutrality(data, titles, c6ti)

    
    taxable_fuels = np.ones([len(titles['RTI']), len(titles['FTTI'])])
    taxable_fuels[:, 20:45] = 0   # CNG, PHEV, BEV, bio-ethanol, FCEV exempt
    
    bztc = data['BZTC']
    ztvt = data['ZTVT']
    ztrt = data['ZTRT']

    # Vehicle and usage parameters
    lt = bztc[:, :, c6ti['8 Lifetime (y)']]
    mileage = bztc[:, :, c6ti['15 Average mileage (km/y)']]
    load = bztc[:, :, c6ti['10 Loads (t or passengers/veh)']]
    energy_use = bztc[:, :, c6ti['9 Energy use (MJ/vkm)']]

    # Costs are converted from per vehicle-km to per tonne-km (or passenger-km)
    purchase_cost_per_km = bztc[:, :, c6ti['1 Purchase cost (USD/veh)']] / mileage

    # Upfront cost (base, policy, standard deviation)
    upfront = purchase_cost_per_km / load
    upfront_pol = (bztc[:, :, c6ti['12 CO2 emissions (gCO2/km)']] * data['RZCO'][:, None]  # CO2-dependent tax
                   + purchase_cost_per_km * ztvt                                          # Registration tax or subsidy
                   ) / load
    upfront_sd = bztc[:, :, c6ti['2 Std of purchase cost']] / mileage / load

    # Annual variable cost (base, policy, standard deviation)
    annual = (bztc[:, :, c6ti['3 fuel cost (USD/km)']]
              + bztc[:, :, c6ti['5 O&M costs (USD/km)']]) / load
    annual_pol = (carbon_costs                                  # Carbon tax
                  + data['RZFT'][:, None] * energy_use * taxable_fuels   # Fuel tax or subsidy
                  + ztrt                                        # Yearly road tax
                  ) / load
    annual_sd = np.sqrt(bztc[:, :, c6ti['4 std fuel cost']]**2
                        + bztc[:, :, c6ti['6 std O&M']]**2) / load

    # The upfront and annual costs have already been converted to the service
    # provided, so the service delivered is 1
    LCOF, TLCOF, dLCOF, dTLCOF = get_levelised_costs(
        upfront=upfront,
        upfront_policies=upfront_pol,
        upfront_sd=upfront_sd,
        annual=annual,
        annual_policies=annual_pol,
        annual_sd=annual_sd,
        service_delivered=1,
        service_sd=0.15,    # Assume a 15% variation in load factors
        lifetimes=lt,
        r=bztc[:, :, c6ti['7 Discount rate']])

    # Introduce Gamma Values
    TLCOFG = TLCOF * (1 + bztc[:, :, c6ti['11 Gamma']])

    # Compute levelised cost in freight in $/tkm
    data['ZTLC'] = LCOF        # LCOF without policy
    data['ZTLD'] = dLCOF       # LCOF without policy SD
    data['ZTTC'] = TLCOF       # LCOF with policy
    data['ZTTD'] = dTLCOF      # LCOF with policy SD
    data['ZEGC'] = TLCOFG      # LCOF with policy and gamma

    
    # Vehicle price components for front end ($/veh)
    data["ZWIC"] = (
        bztc[:, :, c6ti['1 Purchase cost (USD/veh)']] * (1 + ztvt)
        + bztc[:, :, c6ti["12 CO2 emissions (gCO2/km)"]]
        * data["RZCO"][:, None]
    )
    
    # Vehicle fuel price components for front end ($/km)
    data["ZWFC"] = (
        bztc[:, :, c6ti["3 fuel cost (USD/km)"]]
        + data['RZFT'][:, None]
        * bztc[:, :, c6ti["9 Energy use (MJ/vkm)"]] * taxable_fuels
    )
    
    return data