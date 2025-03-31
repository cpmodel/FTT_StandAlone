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

    The function calculates the levelised cost of freight transport in 2012$/t-km per
    vehicle type. These costs are then converted into 2010 Euros/t-km per vehicle type.
    It includes intangible costs (gamma values) and together
    determines the investor preferences.

    Parameters
    -----------
    data: dictionary
        Data is a dictionary with the data of the current year for all variables.
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
    Additional notes if required.
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

    for r in range(len(titles['RTI'])):
        # Defining and Initialising Variables
        #Cost matrix
        BZTC = data['BZTC'][r, :, :]
        carbon_c = carbon_costs[r]

        # Lifetime calculations
        LF = BZTC[:, c6ti['8 Lifetime (y)']]
        max_LF = int(np.max(LF))
        LF_mat = np.linspace(np.zeros(len(titles['FTTI'])), max_LF-1,
                             num=max_LF, axis=1, endpoint=True)
        LF_max_mat = np.concatenate(int(max_LF) * [LF[:, np.newaxis]], axis=1)
        mask = LF_mat < LF_max_mat
	
        # Taxable fuels
        taxable_fuels[r,:] = tf[:]
	
        # Discount rate
        rM = BZTC[:,c6ti['7 Discount rate'], np.newaxis]
        # For NPV calculations
        denominator = (1+rM)**LF_mat

        # Costs of trucks, paid once in a lifetime
        It = np.ones([len(titles['FTTI']), int(max_LF)])
        It = It * BZTC[:, c6ti['1 Purchase cost (USD/veh)'], np.newaxis]
        It = It / BZTC[:, c6ti['15 Average mileage (km/y)'], np.newaxis]
        It[:,1:] = 0

        # Standard deviation of costs of trucks
        dIt = np.ones([len(titles['FTTI']), int(max_LF)])
        dIt = dIt * BZTC[:, c6ti['2 Std of purchase cost'], np.newaxis]
        dIt = dIt / BZTC[:, c6ti['15 Average mileage (km/y)'], np.newaxis]
        dIt[:,1:] = 0

        # Reg tax based on carbon price, RTCOt = ($/tCO2/km)/(tCO2/km)
        RZCOt = np.ones([len(titles['FTTI']), int(max_LF)])
        RZCOt = (RZCOt * BZTC[:, c6ti['12 CO2 emissions (gCO2/km)'], np.newaxis]
              * data['RZCO'][r,0,0])
        RZCOt[:,1:] = 0

        # Registration taxes, ZTVT is vehicle tax
        ItVT = It * (data['ZTVT'][r, :, 0, np.newaxis])
        
        # Fuel Cost
        FT = np.ones([len(titles['FTTI']), int(max_LF)])
        FT = FT * BZTC[:, c6ti['3 fuel cost (USD/km)'], np.newaxis]
        FT = np.where(mask, FT, 0)

        # Standard deviation of fuel costs
        dFT = np.ones([len(titles['FTTI']), int(max_LF)])
        dFT = dFT * BZTC[:, c6ti['4 std fuel cost'], np.newaxis]
        dFT = np.where(mask, dFT, 0)
        
        # Average carbon costs
        ct = np.ones([len(titles['FTTI']), int(max_LF)])
        ct = ct * carbon_c[:, np.newaxis]
        ct = np.where(mask, ct, 0)

        # fuel tax/subsidies
        fft = np.ones([len(titles['FTTI']), int(max_LF)])
        fft = fft * data['RZFT'][r, :, 0, np.newaxis] \
              * BZTC[:, c6ti["9 Energy use (MJ/vkm)"], np.newaxis] \
              * taxable_fuels[r, :]
        fft = np.where(mask, fft, 0)

        # O&M costs
        OMt = np.ones([len(titles['FTTI']), int(max_LF)])
        OMt = OMt * BZTC[:, c6ti['5 O&M costs (USD/km)'], np.newaxis]
        OMt = np.where(mask, OMt, 0)

        # Standard deviation of O&M costs
        dOMt = np.ones([len(titles['FTTI']), int(max_LF)])
        dOMt = dOMt * BZTC[:, c6ti['6 std O&M'], np.newaxis]
        dOMt = np.where(mask, dOMt, 0)

        # Capacity factors
        Lfactor = np.ones([len(titles['FTTI']), int(max_LF)])
        Lfactor = Lfactor * BZTC[:, c6ti['10 Loads (t or passengers/veh)'], np.newaxis]

        # Road Tax
        RT = np.ones([len(titles['FTTI']), int(max_LF)])
        RT = RT * data['ZTRT'][r, :, 0, np.newaxis]
        RT = np.where(mask, RT, 0)
        
        # Calculate LCOF without policy, and find standard deviation
        npv_expenses1 = (It+FT+OMt)/Lfactor
        npv_expenses1 = (npv_expenses1/denominator)
        npv_utility = 1/denominator
        
        # Remove 1s for tech with small lifetime than max
        npv_utility[npv_utility==1] = 0
        npv_utility[:,0] = 1
        LCOF = np.sum(npv_expenses1, axis=1)/np.sum(npv_utility, axis=1)

        # MODIFIED: Corrected standard deviation calculation
        # First sum the variances
        variance_terms = dIt**2 + dFT**2 + dOMt**2
        # Scale variances by load factor
        scaled_variance = variance_terms/(Lfactor**2)
        # Apply proper discounting (squared) and sum
        summed_variance = np.sum(scaled_variance/(denominator**2), axis=1)
        # TODO, softcode. Hardcoded uncertainty  With uncertainty in load factors ()
        variance_plus_dcf = summed_variance + (np.sum(npv_expenses1, axis=1) * 0.15)**2
        # Calculate final standard deviation
        dLCOF = np.sqrt(variance_plus_dcf)/np.sum(npv_utility, axis=1)

        # Calculate LCOF with policy, and find standard deviation
        npv_expenses2 = (It + ct + ItVT + FT + fft + OMt + RT) / Lfactor
        npv_expenses2 = npv_expenses2/denominator
        TLCOF = np.sum(npv_expenses2, axis=1)/np.sum(npv_utility, axis=1)
        dTLCOF = dLCOF 

        # Introduce Gamma Values
        TLCOFG = TLCOF * (1 + BZTC[:, c6ti['11 Gamma']])

        # Convert costs into logarithmic space - applying a log-normal distribution
        LTLCOF = np.log10((TLCOF**2)/np.sqrt((dTLCOF**2)+(TLCOF**2))) + BZTC[:, c6ti['11 Gamma']]
        dLTLCOF = np.sqrt(np.log10(1+(dTLCOF**2)/(TLCOF**2)))

        data['ZTLC'][r, :, 0] = LCOF        # LCOF without policy
        data['ZTLD'][r, :, 0] = dLCOF       # LCOF without policy SD
        data['ZTTC'][r, :, 0] = TLCOF       # LCOF with policy
        data['ZTTD'][r, :, 0] = dTLCOF      # LCOF with policy SD
        data['ZEGC'][r, :, 0] = TLCOFG      # LCOF with policy and gamma
        data['ZTLL'][r, :, 0] = LTLCOF      # LCOF log space with policy and gamma
        data['ZTDD'][r, :, 0] = dLTLCOF     # LCOF log space with policy SD
        
        # Vehicle price components for front end ($/veh)
        data["ZWIC"][r, :, 0] = BZTC[:, c6ti['1 Purchase cost (USD/veh)']] \
                                * (1 + data["ZTVT"][r, :, 0]) \
                                + BZTC[:, c6ti["12 CO2 emissions (gCO2/km)"]] \
                                * data["RZCO"][r, 0, 0]
        
        # Vehicle fuel price components for front end ($/km)
        data["ZWFC"][r, :, 0] = BZTC[:, c6ti["3 fuel cost (USD/km)"]] \
                                + data['RZFT'][r, 0, 0] \
                                * BZTC[:, c6ti["9 Energy use (MJ/vkm)"]] \
                                * taxable_fuels[r, :, 0]
    return data