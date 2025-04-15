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

def set_carbon_tax(data, c6ti):
    '''
    Convert the carbon price in REPP from euro / tC to 2012$/km 
    Apply the carbon price to freight sector technologies based on their emission factors

    Returns:
        Carbon costs per country and technology (2D)
    '''
    carbon_costs = (data["REPP3X"][:, :, 0]                          # Carbon price in euro / tC
                    * data['BZTC'][:, :, c6ti['12 CO2 emissions (gCO2/km)']]     # g CO2 / km
                    / 3.666 / 10**6                                   # Conversion from C to CO2, and g to tonne
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
        has_subsidies = np.any(data['ZTVT'][r, bev_indices, 0] < 0)
        
        if not has_subsidies:
            continue
            
        results[region] = {'subsidy_costs': [], 'tax_revenues': []}
        print(f"\nChecking revenue neutrality for {region}")
        
        for veh_class in range(n_veh_classes):
            bev_indices_class = [idx for idx in [31, 32, 33] if idx % n_veh_classes == veh_class]
            ice_indices_class = [idx for idx in range(25) if idx % n_veh_classes == veh_class]
            
            # Calculate total subsidy being paid out
            bev_costs = data['BZTC'][r, bev_indices_class, c6ti['1 Purchase cost (USD/veh)']]
            bev_numbers = data['ZEWI'][r, bev_indices_class, 0]
            bev_subsidy_rates = data['ZTVT'][r, bev_indices_class, 0]
            total_subsidy = np.sum(bev_costs * bev_numbers * abs(bev_subsidy_rates))
            
            if total_subsidy > 0:
                # Calculate total tax being collected
                ice_costs = data['BZTC'][r, ice_indices_class, c6ti['1 Purchase cost (USD/veh)']]
                ice_numbers = data['ZEWI'][r, ice_indices_class, 0]
                ice_tax_rates = data['ZTVT'][r, ice_indices_class, 0]
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
            bev_numbers = data['ZEWI'][r, bev_indices_class, 0]
            total_subsidy = np.sum(bev_costs * bev_numbers * abs(data['ZTVT'][r, bev_indices_class, 0]))
            
            # Calculate ICE tax rate for this class
            ice_numbers = data['ZEWI'][r, ice_indices_class, 0]
            ice_costs = data['BZTC'][r, ice_indices_class, c6ti['1 Purchase cost (USD/veh)']]
            denominator = np.sum(ice_costs * ice_numbers)
            
            if denominator > 0:
                ice_tax_rate = total_subsidy / denominator
                ice_tax_rate = min(ice_tax_rate, 1.0)  # Cap at 100%
                data['ZTVT'][r, ice_indices_class, 0] = ice_tax_rate

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
        Dictionary with the data of the current year for all variables.
        variables. Variable names are keys and the values are 3D NumPy arrays.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.

    Returns
    ----------
    data: dictionary
        Data is a dictionary with the data of the current year for all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.

    Notes
    ---------
    Calculate levelized costs with optional feebate system.
    Feebate is activated if Feebate_active flag in data is set to 1.
    """
    
    # Categories for the cost matrix (BZTC)
    c6ti = {category: index for index, category in enumerate(titles['C6TI'])}
    n_veh_classes = 5

    # Check if feebate system is active
    if data.get('Feebate active', np.zeros((1,1,1)))[0,0,0] == 1:
        if data.get('iteration', 0) == 0:
            print("\nFeebate system is active")
            calculate_feebate_rates(data, titles, c6ti, n_veh_classes)
            print("\nVerifying revenue neutrality...")
            verify_revenue_neutrality(data, titles, c6ti)

    
    tf = np.ones([len(titles['FTTI']), 1])
    tf[20:45] = 0   # CNG, PHEV, BEV, bio-ethanol, FCEV exempt
    taxable_fuels = np.zeros([len(titles['RTI']), len(titles['FTTI']), 1])
    
    bztc = data['BZTC']
    
    # Heating device lifetimes and build year
    lt = bztc[:, :, c6ti['8 Lifetime (y)']]
    max_lt = int(np.max(lt))
    full_lt_mat = np.arange(max_lt)
    lt_mask = full_lt_mat <= (lt[..., None] - 1)        # Life time mask
    bt_mask = full_lt_mat < np.ones_like(lt[..., None]) # Build time mask
    
    def get_cost_elem(base_cost, conversion_factor, mask):
        '''Mask costs during build or life time, and apply
        conversion to generation where appropriate'''
        cost = np.multiply(base_cost[..., None], conversion_factor)
        return np.multiply(cost, mask)
    
    It = get_cost_elem(bztc[:, :, c6ti['1 Purchase cost (USD/veh)']] / bztc[:, :, c6ti['15 Average mileage (km/y)']], 1, bt_mask)
    dIt = get_cost_elem(bztc[:, :, c6ti['2 Std of purchase cost']] / bztc[:, :, c6ti['15 Average mileage (km/y)']], 1, bt_mask)
    # Reg tax based on carbon price, RTCOt = ($/tCO2/km)/(tCO2/km)
    RZCOt = get_cost_elem(bztc[:, :, c6ti['12 CO2 emissions (gCO2/km)']] * data['RZCO'][:, 0], 1, bt_mask)
    # Registration taxes, ZTVT is vehicle tax
    ItVT = get_cost_elem(bztc[:, :, c6ti['1 Purchase cost (USD/veh)']] * data['ZTVT'][:, :, 0], 1, bt_mask)
    Ft = get_cost_elem(bztc[:, :, c6ti['3 fuel cost (USD/km)']], 1, lt_mask)
    dFt = get_cost_elem(bztc[:, :, c6ti['4 std fuel cost']], 1, lt_mask)
    ct = get_cost_elem(carbon_costs, 1, lt_mask)
    # fuel tax/subsidies
    fft = get_cost_elem(data['RZFT'][:, 0] * bztc[:, :, c6ti["9 Energy use (MJ/vkm)"]] * taxable_fuels[:, :, 0], 1, lt_mask)
    OMt = get_cost_elem(bztc[:, :, c6ti['5 O&M costs (USD/km)']], 1, lt_mask)
    dOMt = get_cost_elem(bztc[:, :, c6ti['6 std O&M']], 1, lt_mask)
    RT = get_cost_elem(data['ZTRT'][:, :, 0], 1, lt_mask)
    
    Lfactor = bztc[:, :, c6ti['10 Loads (t or passengers/veh)'], np.newaxis]
    
    # Discount rate
    dr = bztc[:,:,c6ti['7 Discount rate'], np.newaxis]
    denominator = (1+dr)**full_lt_mat
    
    
    # Calculate LCOF without policy, and find standard deviation
    npv_expenses_bare = (It + Ft + OMt) / Lfactor
    npv_expenses_bare = npv_expenses_bare / denominator
    
    # Calculate LCOF with policy, and find standard deviation
    npv_expenses_policy = (It + ct + ItVT + Ft + fft + OMt + RT) / Lfactor
    npv_expenses2 = npv_expenses_policy / denominator
    
    # 3 – Utility
    npv_utility = np.where(lt_mask, 1, 0) / denominator
    utility_sum = np.sum(npv_utility, axis=2)
    
    
    # Standard deviation calculation
    # First sum the variances
    variance_terms = dIt**2 + dFt**2 + dOMt**2
    # Scale variances by load factor
    scaled_variance = variance_terms/(Lfactor**2)
    # Apply proper discounting (squared) and sum
    summed_variance = np.sum(scaled_variance/(denominator**2), axis=2)
    # TODO, softcode. Hardcoded uncertainty  With uncertainty in load factors ()
    variance_plus_dcf = summed_variance + (np.sum(npv_expenses_bare, axis=2) * 0.15)**2
    # Calculate final standard deviation
    
    LCOF = np.sum(npv_expenses_bare, axis=2) / utility_sum
    dLCOF = np.sqrt(variance_plus_dcf) / utility_sum
    
    TLCOF = np.sum(npv_expenses_policy, axis=2) / utility_sum
    
    # Introduce Gamma Values
    TLCOFG = TLCOF * (1 + bztc[:, :, c6ti['11 Gamma']])

    data['ZTLC'][:, :, 0] = LCOF        # LCOF without policy
    data['ZTLD'][:, :, 0] = dLCOF       # LCOF without policy SD
    data['ZTTC'][:, :, 0] = TLCOF       # LCOF with policy
    data['ZTTD'][:, :, 0] = dLCOF       # LCOF with policy SD
    data['ZEGC'][:, :, 0] = TLCOFG      # LCOF with policy and gamma

    
    # Vehicle price components for front end ($/veh)
    data["ZWIC"][:, :, 0] = bztc[:, :, c6ti['1 Purchase cost (USD/veh)']] \
                            * (1 + data["ZTVT"][:, :, 0]) \
                            + bztc[:, :, c6ti["12 CO2 emissions (gCO2/km)"]] \
                            * data["RZCO"][:, 0]
    
    # Vehicle fuel price components for front end ($/km)
    data["ZWFC"][:, :, 0] = bztc[:, :, c6ti["3 fuel cost (USD/km)"]] \
                            + data['RZFT'][:, 0] \
                            * bztc[:, :, c6ti["9 Energy use (MJ/vkm)"]] \
                            * taxable_fuels[: , :, 0]
    
    return data