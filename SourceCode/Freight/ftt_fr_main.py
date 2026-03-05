# -*- coding: utf-8 -*-
"""
============================================================
ftt_fr_main.py
============================================================
Freight transport FTT module.

This is the main file for FTT: Freight, which models technological
diffusion of freight vehicle types due to simulated consumer decision making.
Consumers compare the **levelised cost of freight**, which leads to changes 
in the market shares of different technologies.

The outputs of this module include market shares, fuel use, and emissions.

Local library imports:

    FTT: Freight functions:
    - `get_lcof <ftt_fr_lcof.html>`__
        Levelised cost calculation

Functions included:
    - solve
        Main solution function for the module
    - get_lcof
        Calculate levelised cost of freight

"""


# Third party imports
import numpy as np

# Local library imports
from SourceCode.ftt_core.ftt_sales_or_investments import get_sales
from SourceCode.ftt_core.ftt_shares import shares_change
from SourceCode.ftt_core.ftt_mandate import implement_seeding, implement_mandate


from SourceCode.support.divide import divide
from SourceCode.support.check_market_shares import check_market_shares


from SourceCode.Freight.ftt_fr_lcof import get_lcof, set_carbon_tax
from SourceCode.Freight.ftt_fr_regulatory_policies import implement_shares_policies
from SourceCode.Freight.ftt_fr_emissions_regulation import implement_emissions_regulation
from SourceCode.Freight.ftt_fr_local_learning import get_start_local_capacity, add_local_capacity, get_local_learning

from SourceCode.sector_coupling.battery_lbd import battery_costs, get_start_cap

GREEN_INDICES_EV = [6]

# %% main function
# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, titles, histend, year, domain):
    """
    Main solution function for the module.

    This function simulates investor decision making in the freight sector.
    Levelised costs (from the get_lcof function) are taken and market shares
    for each vehicle type are simulated to ensure demand is met.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for given year of solution
    time_lag: type
        Model variables from the previous year
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of histrorical data by variable
    year: int
        Current year

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    """

    # Categories for the cost matrix
    c6ti = {category: index for index, category in enumerate(titles['C6TI'])}

    sector = 'freight'
    num_regions = len(titles['RTI'])
    num_techs = len(titles['FTTI'])       # Number of technologies
    num_fuels = len(titles['JTI'])
    n_veh_classes = len(titles['FSTI'])   # Number of classes

    # Factor used to create intermediate data from annual figures
    no_it = int(data['noit'][0, 0, 0])
    dt = 1 / no_it

    # Creating variables
    # Technology to fuel user conversion matrix
    zjet = data['ZJET'][0, :, :]
    # Initialise the emission correction factor
    emis_corr = np.zeros([num_regions, num_techs])
    
    def sum_over_classes(var):
        
        output = np.empty((var.shape[0], n_veh_classes))
        for veh_class in range(n_veh_classes):
            output[:, veh_class] = var[:, veh_class::n_veh_classes, 0].sum(axis=1)
        output = output[:, :, None]
       
        return output
    
    def get_class(var, veh_class):
        '''Select all values corresponding to specific vehicle class'''
        return var[:, veh_class::n_veh_classes]

    # Initialise up to the last year of historical data
    if year <= histend["RFLZ"]:
        
        # Normalise market shares if they do not add up to 1
        class_totals = sum_over_classes(data['ZEWS'])
        for r in range(num_regions):
            for veh_class in range(n_veh_classes):
                idx = slice(veh_class, None, n_veh_classes)
        
                if (not np.isclose(class_totals[r, veh_class, 0], 1.0, atol=1e-9)
                    and class_totals[r, veh_class, 0] > 0.0):
        
                    data['ZEWS'][r, idx, 0] /= class_totals[r, veh_class, 0]
            
        
        # Calculate number of vehicles per technology. Repeat demand 'rflz' across columns to match 'ZEWS' shape
        rflz_reshaped = np.tile(data['RFLZ'], (1, data['ZEWS'].shape[1] // data['RFLZ'].shape[1], 1))
        data['ZEWK'] = data['ZEWS'] * rflz_reshaped
        
        # Find total service area in Mvkm, first by tech, then by vehicle class
        data['ZEVV'] = data['ZEWK'] * data['BZTC'][:, :, c6ti['15 Average mileage (km/y)'], np.newaxis] / 1e6
        data['ZESG'] = sum_over_classes(data['ZEVV'])
        
        # Calculate demand in million ton vehicle-km OR million passenger vehicle km, per vehicle class
        data['ZEST'] = data['ZEVV'] * data['BZTC'][:, :, c6ti['10 Loads (t or passengers/veh)'], np.newaxis]
        data['RVKZ'] = sum_over_classes(data['ZEST'])
        
        # Emissions 
        data['ZEWE'] = (data['ZEVV']
                        * data['BZTC'][:, :, c6ti['12 CO2 emissions (gCO2/km)'], np.newaxis]
                        * (1 - data['ZBFM']) / (1e6) )
        
        for r in range(num_regions):
        
            for veh in range(num_techs):
                for fuel in range(num_fuels):
                    if titles['JTI'][fuel] == '11 Biofuels' and zjet[veh, fuel] == 1:
                        # No biofuel blending mandate in the historical period
                        zjet[veh, fuel] = 0
                        
            # Find fuel use
            data['ZJNJ'][r, :, 0] = (np.matmul(np.transpose(zjet), data['ZEVV'][r, :, 0] * \
                                    data['BZTC'][r, :, c6ti['9 Energy use (MJ/vkm)']])) / 41.868
        
        carbon_costs = set_carbon_tax(data, c6ti)
        data = get_lcof(data, titles, carbon_costs, year)

        
        if year == histend["RFLZ"]:
            # Calculate levelised cost
                        
            data["BZTC initial"] = np.copy(data["BZTC"])
            
            # # Battery starting capacity
            data["Cumulative total batcap"] = get_start_cap(data, titles)
            
            # Get local learning
            data["Freight local capacity"] = get_start_local_capacity(data, year)


    "Model Dynamics"

    # Endogenous calculation starts here
    if year > histend['RFLZ']:
        data_dt = {}
        
        for var in time_lag.keys():
            if var.startswith(("R", "Z", "B")) or var in ["Freight local capacity"]:
                data_dt[var] = np.copy(time_lag[var])
                

        # Find if there is a regulation and if it is exceeded
        relative_excess = divide((time_lag['ZEWK'][:, :, 0] - data['ZREG'][:, :, 0]),
                           data['ZREG'][:, :, 0])       # 0 when dividing by 0
        reg_constr = 0.5 + 0.5 * np.tanh(1.5 + 10 * relative_excess)
        reg_constr[data['ZREG'][:, :, 0] == 0.0] = 1.0
        reg_constr[data['ZREG'][:, :, 0] == -1.0] = 0.0

        
        for t in range(1, no_it + 1):
            
            # Interpolations to avoid staircase profile
            D = time_lag['RVKZ'] + (data['RVKZ'] - time_lag['RVKZ']) * t * dt
            Utot = time_lag['RFLZ'] + (data['RFLZ'] - time_lag['RFLZ']) * t * dt
            Utot_dt = time_lag['RFLZ'] + (data['RFLZ'] - time_lag['RFLZ']) * (t - 1) * dt
            Utot_reshaped = np.tile(Utot, (1, data['ZEWS'].shape[1] // Utot.shape[1], 1))[:, :, 0] # Reshape to 71 x #tech (duplicate info)
           
            # The core FTT equations, taking into account old shares, costs and regulations
            change_in_shares = shares_change(
                dt=dt,                          
                regions=np.arange(num_regions),
                shares_dt=data_dt["ZEWS"],      # Shares at previous t
                costs=data_dt['ZEGC'],          # Costs
                costs_sd=data_dt['ZTTD'],       # Standard deviation of costs
                subst=data['ZEWA'] * data['BZTC'][:, :, c6ti['14 Turnover rate (1/y)'], None],  # Substitution turnover rate
                reg_constr=reg_constr,          # Constraint due to regulation
                num_regions=num_regions,        # Number of regions
                num_techs=num_techs             # Number of techs
            )
            
            # Calculate endogenous market shares
            endo_shares = data_dt['ZEWS'][:, :, 0] + change_in_shares
            endo_capacity = endo_shares * Utot_reshaped
            
            # Shares after exogenous sales and regulation correction taken into account
            data['ZEWS'] = implement_shares_policies(
                endo_capacity, endo_shares, 
                titles, data['ZWSA'], data['ZREG'], reg_constr,
                sum_over_classes, n_veh_classes, Utot, no_it)
                        
            
            check_market_shares(data['ZEWS'], titles, sector, year)

                    
            # Copy over costs that don't change
            data['BZTC'][:, :, 1:20] = data_dt['BZTC'][:, :, 1:20]
            
            # Number of trucks by technology
            data['ZEWK'] = data['ZEWS'] * Utot_reshaped[:, :, None]
           
            # Investment (sales) = new capacity created
            # zewi_t is new additions at current timestep/iteration
            data["ZEWI"], zewi_t = get_sales(
                    cap=data["ZEWK"],
                    cap_dt=data_dt["ZEWK"], 
                    cap_lag=time_lag["ZEWK"],
                    sales_or_investment_in=data["ZEWI"],
                    timescales=data['BZTC'][:, :, c6ti['8 Lifetime (y)']],
                    dt=dt,
                 
                    )
            
            # Policy levers are MUTUALLY EXCLUSIVE: mandate/kickstarter OR emissions regulation
            # Check which policies are active
            mandate_active = not np.all(data["EV truck mandate"][:, 2, 0] == 0)
            emissions_reg_active = "emissions regulation" in data and not np.all(data["emissions regulation"][:, 0, 0] == 0)

            
            for v_class in range(n_veh_classes):
                
                # Indices corresponding to class
                idx = slice(v_class, None, n_veh_classes)
                
                # Regions with very low EV numbers see some diffusion from other regions
                data['ZEWI'][:, idx], zewi_t[:, idx], data['ZEWK'][:, idx] = implement_seeding(
                            data['ZEWK'][:, idx],
                            data['ZEWI'][:, idx],
                            zewi_t[:, idx],
                            year, GREEN_INDICES_EV, histend['RFLZ']
                        )

                if mandate_active:
                    # Adjust max mandate for HDTs (2/3 of stated maximum)
                    truck_mandate = np.copy(data['EV truck mandate'])
                    # if v_class == 3:  # HDTs
                    #     truck_mandate[:, 2, 0] = 0.67 * truck_mandate[:, 2, 0]
                    # A policy of a minimum sales share
                    data['ZEWI'][:, idx], zewi_t[:, idx], data['ZEWK'][:, idx] = implement_mandate(
                                data['ZEWK'][:, idx],
                                data['ZEWI'][:, idx],
                                zewi_t[:, idx],
                                year, GREEN_INDICES_EV,
                                truck_mandate
                            )
                

            if emissions_reg_active and not mandate_active:
                data['ZEWI'], zewi_t, data['ZEWK'] = implement_emissions_regulation(
                                data['ZEWK'], data["emissions regulation"], data['ZEWI'], zewi_t,
                                n_veh_classes, year,
                                data['BZTC'][:, :, c6ti['12 CO2 emissions (gCO2/km)']])
            
            # Recalculate zews per class
            for r in range(num_regions):
                for veh_class in range(n_veh_classes):
                    denominator = np.sum(data['ZEWK'][r, veh_class::n_veh_classes])
                    if denominator > 0:
                        data['ZEWS'][r, veh_class::n_veh_classes, 0] = ( 
                                data['ZEWK'][r, veh_class::n_veh_classes, 0]
                                / denominator )

            
            # Find total service area and demand, first by tech, then by vehicle class     
            data['ZEVV'] = data['ZEWK'] * data['BZTC'][:, :, c6ti['15 Average mileage (km/y)'], np.newaxis] / 1e6
            data['ZEST'] = data['ZEVV'] * data['BZTC'][:, :, c6ti['10 Loads (t or passengers/veh)'], np.newaxis]
            data['ZESG'] = sum_over_classes(data['ZEVV'])
            data['RVKZ'] = sum_over_classes(data['ZEST'])
                                    
            # Emissions
            data['ZEWE'] = ( data['ZEVV']
                            * data['BZTC'][:, :, c6ti['12 CO2 emissions (gCO2/km)'], None]
                            * (1 - data['ZBFM']) / 1e6 )
            
            
            # Reopen country loop
            for r in range(num_regions):

                if np.sum(D[r]) == 0.0:
                    continue
                                                        
                for veh in range(num_techs):
                    for fuel in range(num_fuels):
                        #  Middle distillates
                        if titles['JTI'][fuel] == '5 Middle distillates' and data['ZJET'][0, veh, fuel]  == 1:  

                            #  Mix with biofuels if there's a biofuel mandate
                            zjet[veh, fuel] = zjet[veh, fuel] * (1.0 - data['ZBFM'][r, 0, 0])

                            # Emission correction factor
                            emis_corr[r, veh] = 1.0 - data['ZBFM'][r, 0, 0]

                        elif titles['JTI'][fuel] == '11 Biofuels' and data['ZJET'][0, veh, fuel] == 1:

                            zjet[veh, fuel] = data['ZJET'][0, veh, fuel] * data['ZBFM'][r, 0, 0]

                # Fuel use by fuel type - Convert TJ (BZTC * ZEVV) to ktoe, so divide by 41.868
                data['ZJNJ'][r, :, 0] = (np.matmul(np.transpose(zjet), data['ZEVV'][r, :, 0] * \
                                    data['BZTC'][r, :, c6ti['9 Energy use (MJ/vkm)']])) / 41.868
            

            
            # Cumulative investment, not in region loop as it is global
            bi = np.matmul(zewi_t[:, :, 0], data['ZEWB'][0, :, :])
            dw = np.sum(bi, axis=0)
            
            data['ZEWW'][0, :, 0] = data_dt['ZEWW'][0, :, 0] + dw
            
            # Update local capacity based on new sales
            data = add_local_capacity(data, data_dt, zewi_t, year)
            
            # Learning-by-doing based on global battery learning
            
            added_battery_cap = zewi_t[:, :, 0] * data["BZTC"][:, :, c6ti['16 Battery capacity (kWh)']]
            added_battery_cap = added_battery_cap / 1e6  # Convert kWh to GWh.
            summed_added_battery_cap = np.sum(added_battery_cap)  # Summing across sectors
            
            data["Battery cap additions"][2, t - 1, 0] = summed_added_battery_cap
                
            # Copy over the technology cost categories that do not change 
            data["BZTC initial"] = np.copy(data_dt['BZTC initial'])
            
            # Battery learning
            data = battery_costs(data, time_lag, year, t, titles, histend)
            
            # Save battery cost
            nonbat_cost = np.zeros([num_regions,num_techs, 1])
            nonbat_cost_dt = np.zeros([num_regions,num_techs, 1])
            
            if year > histend["BZTC"]:
            
                # Learning-by-doing effects on investment
                for tech in range(num_techs):
    
                    if data['ZEWW'][0, tech, 0] > 0.1:
                        
                        # Wright's law approximation for discrete time
                        # Apply only a fraction of learning as global learning
                        # e.g., 70% of the learning effect is global; the rest is local
                        global_learning_share = 0.7
                        global_learning_factor = (
                            1.0 + data["BZTC"][:, tech, c6ti['13 Learning exponent']]
                            * dw[tech] / data['ZEWW'][0, tech, 0]
                        )
                        local_learning_factor = get_local_learning(data, zewi_t, titles, tech)
                        
                        learning_factor = 1.0 + global_learning_share * (global_learning_factor - 1.0) + (1.0 - global_learning_share) * (local_learning_factor - 1.0)
                        
                        # For PHEV, BEV, and FCEVs, add the battery costs to the non-battery costs
                        if tech in range(25, 35) or tech in range(40, 45):
                            
                            nonbat_cost_dt[:, tech, 0] = (
                                    data_dt['BZTC'][:, tech, c6ti['1 Purchase cost (USD/veh)']] 
                                    - data_dt['Battery price'][:, 0, 0]  
                                    * data_dt["BZTC"][:, tech, c6ti['16 Battery capacity (kWh)']]
                                    )
                            
                            nonbat_cost[:, tech, 0] = nonbat_cost_dt[:, tech, 0] * learning_factor
                                                    

                            data['BZTC'][:, tech, c6ti['1 Purchase cost (USD/veh)']] = (
                                    nonbat_cost[:, tech, 0] 
                                    + (data['Battery price'][:, 0, 0]
                                    * data["BZTC"][:, tech, c6ti['16 Battery capacity (kWh)']])
                                    )
                            # Introducing LBD on O&M costs, assuming the same learning rate. #TODO: think about this again
                            # Just do this for EVs for now, as it's more likely that learning effects would reduce maintenance costs for EVs than for ICE vehicles
                            data['BZTC'][:, tech, c6ti['5 O&M costs (USD/km)']] =  (
                                    data_dt['BZTC'][:, tech, c6ti['5 O&M costs (USD/km)']] 
                                    * learning_factor )
                            
                            # Learning-by-doing for the standard deviations, scaling with above
                            data['BZTC'][:, tech, c6ti['2 Std of purchase cost']] = (
                                data_dt['BZTC'][:, tech, c6ti['2 Std of purchase cost']] * learning_factor )
                            data['BZTC'][:, tech, c6ti['6 std O&M']] = (
                                data_dt['BZTC'][:, tech, c6ti['6 std O&M']] * learning_factor)
                        
                        # For non-EVs, add only the non-battery costs (and still use global learning factor)
                        else:
                            data['BZTC'][:, tech, c6ti['1 Purchase cost (USD/veh)']] = (
                                    data_dt['BZTC'][:, tech, c6ti['1 Purchase cost (USD/veh)']] 
                                                    * global_learning_factor )                            
                        
                            # Learning-by-doing for the standard deviations, scaling with above
                            data['BZTC'][:, tech, c6ti['2 Std of purchase cost']] = (
                                data_dt['BZTC'][:, tech, c6ti['2 Std of purchase cost']] * global_learning_factor )
                        
            # Calculate levelised cost
            carbon_costs = set_carbon_tax(data, c6ti)
            data = get_lcof(data, titles, carbon_costs, year)
            
            # Save non bat costs
            data["non battery truck cost"] = nonbat_cost

            # Set up data_dt for next timestep
            for var in time_lag.keys():
                if var.startswith(("R", "Z", "B")) or var in ["Freight local capacity"]:
                    data_dt[var] = np.copy(data[var])
        
        # Calculate total investment by technology in terms of truck purchases
        data['ZWIY'] = data['ZEWI'] * data["BZTC"][:, :, c6ti['1 Purchase cost (USD/veh)'], None]
        
        if year == 2050:
            print(f'Number of electric trucks globally: {np.sum(get_class(data['ZEWK'], 3)[:, 6])/1e6:.1f}M')

    return data
