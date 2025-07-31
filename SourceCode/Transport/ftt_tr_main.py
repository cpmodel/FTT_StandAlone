# -*- coding: utf-8 -*-
"""
=========================================
ftt_tr_main.py
=========================================
Passenger road transport FTT module.


This is the main file for FTT: Transport, which models technological
diffusion of passenger vehicle types due to simulated consumer decision making.
Consumers compare the log of the **levelised costs**, which leads to changes in the
market shares of different technologies.

The outputs of this module include sales, fuel use, and emissions.

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Element-wise divide which replaces divide-by-zeros with zeros

Functions included:
    - solve
        Main solution function for the module
    - get_lcot
        Calculate levelised cost of transport
    - get_sales
        Calculate new sales/additions in FTT-Transport 
        
"""

# Third party imports
import numpy as np
import time

# Local library imports
from SourceCode.ftt_core.ftt_sales_or_investments import get_sales
from SourceCode.ftt_core.ftt_shares import shares_change
from SourceCode.ftt_core.ftt_regulatory_policies import implement_regulatory_policies

from SourceCode.support.divide import divide
from SourceCode.support.check_market_shares import check_market_shares

from SourceCode.Transport.ftt_tr_shares import shares_transport
from SourceCode.Transport.ftt_tr_lcot import get_lcot
from SourceCode.Transport.ftt_tr_emission_corrections import co2_corr, biofuel_corr, compute_emissions_and_fuel_use
from SourceCode.Transport.ftt_tr_survival import survival_function, add_new_cars_age_matrix


# %% Fleet size - under development
# -----------------------------------------------------------------------------
# ----------------- Gompertz equation for fleet size --------------------------
# -----------------------------------------------------------------------------
# def fleet_size(data, titles):
#
#     return print("Hello")


# %% main function
# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, iter_lag, titles, histend, year, specs):
    """
    Main solution function for the module.

    Add an extended description in the future.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution
    time_lag: type
        Model variables in previous year
    iter_lag: type
        Description
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of histrorical data by variable
    year: int
        Curernt/active year of solution
    specs: dictionary of NumPy arrays
        Function specifications for each region and module

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    Notes
    ---------
    This function should be broken up into more elements in development.
    """
    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}
    jti = {category: index for index, category in enumerate(titles['JTI'])}

    fuelvars = ['FR_1', 'FR_2', 'FR_3', 'FR_4', 'FR_5', 'FR_6',
                'FR_7', 'FR_8', 'FR_9', 'FR_10', 'FR_11', 'FR_12']

    sector = "tr_road_pass"
    sector_index = 0
    sector_index = 15  #titles['FUTI'].index('16 Road Transport')

    # Store fuel prices and convert to $2013/toe
    # It's actually in current$/toe
    # TODO: Temporary deflator values
    data['TE3P'][:, jti["5 Middle distillates"], 0] = iter_lag['PFRM'][:, sector_index, 0] / 1.33
    data['TE3P'][:, jti["7 Natural gas"], 0] = iter_lag['PFRG'][:, sector_index, 0] / 1.33
    data['TE3P'][:, jti["8 Electricity"], 0] = iter_lag['PFRE'][:, sector_index, 0] / 1.33
    data['TE3P'][:, jti["11 Biofuels"], 0] = iter_lag['PFRB'][:, sector_index, 0] / 1.33
#    data['TE3P'][:, "12 Hydrogen", 0] = data['PFRE'][:, sector_index, 0] * 2.0

    if year == 2012:
        start_i_cost = np.zeros([len(titles['RTI']), len(titles['VTTI']), 1])
        for veh in range(len(titles['VTTI'])):
            if 17 < veh < 24:
                # Starting EV/PHEV cost (without battery)
                start_i_cost[:, veh, 0] = (data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']]
                                           / data['BTTC'][:, veh, c3ti['20 Markup factor']]
                                           - data['BTTC'][:, veh, c3ti['18 Battery cap (kWh)']]
                                           * data['BTTC'][:, veh, c3ti['19 Battery cost ($/kWh)']]
                                           - data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] * 0.15)
            else:
                start_i_cost[:, veh, 0] = 0
        # TEVC is 'Start_ICost' in the Fortran model
        data["TEVC"] = start_i_cost

        # Define starting battery capacity, this does not change
        data["TWWB"] = np.copy(data["TEWW"])
        for veh in range(len(titles['VTTI'])):
            if (veh < 18) or (veh > 23):
                # Set starting cumulative battery capacities to 0 for ICE vehicles
                data["TWWB"][0, veh, 0] = 0

        # Save initial cost matrix (BTCLi from FORTRAN model)
        data["BTCI"] = np.copy(data["BTTC"])
    elif year > 2012:
        # Copy over TEVC and TWWB values
        data['TEVC'] = np.copy(time_lag['TEVC'])
        data['TWWB'] = np.copy(time_lag['TWWB'])
        data["BTCI"] = np.copy(time_lag['BTCI'])

    for r in range(len(titles['RTI'])):
        # %% Initialise
        # Up to the last year of historical market share data
        if year <= data["TDA1"][r, 0, 0]:

            # Correction to market shares
            # Sometimes historical market shares do not add up to 1.0
            share_sum = np.sum(data['TEWS'][r, :, 0])
            if (abs(share_sum - 1.0) > 1e-9) and (share_sum > 0.0):
                data['TEWS'][r, :, 0] /= share_sum

            # Computes initial values for the capacity factor, numbers of
            # vehicles by technology and distance driven

            # "Capacities", defined as 1000 vehicles
            data['TEWK'][:, :, 0] = data['TEWS'][:, :, 0] * \
                data['RFLT'][:, 0, 0, np.newaxis]

            # "Generation", defined as total mln km driven
            data['TEWG'][:, :, 0] = data['TEWK'][:, :, 0] * \
                data['RVKM'][:, 0, 0, np.newaxis] * 1e-3

            # "Emissions", Factor 1.2 approx fleet efficiency factor, corrected later with CO2_corr
            data['TEWE'][:, :, 0] = data['TEWG'][:, :, 0] * \
                data['BTTC'][:, :, c3ti['14 CO2Emissions']]/1e6*1.2

        if year == data["TDA1"][r, 0, 0]: 
            # Define starting battery capacity
            start_bat_cap = np.copy(data["TEWW"])
            for veh in range(len(titles['VTTI'])):
                if (veh < 18) or (veh > 23):
                    # Set starting cumulative battery capacities to 0 for ICE vehicles
                    start_bat_cap[0,veh,0] = 0
            # TWWB is 'StartBatCap' in the FORTRAN model
            data["TWWB"] = start_bat_cap

            # Save initial cost matrix (BTCLi from FORTRAN model)
            data["BTCI"] = np.copy(data["BTTC"])

    # Fuel use and emissions
    
    regions = np.where(year <= data['TDA1'][:, 0, 0])[0]
    # First: correct for age effects, with older vehicles emitting more CO2
    co2_corrct, has_fleet = co2_corr(data, titles, regions)
    # Then, correct for the biofuel mandate, reducing emissions
    biofuel_corrct, fuel_converter = biofuel_corr(data, titles, has_fleet, regions)
    # Finally, compute TEWE (emissions) and TJEF (fuel use) (TO BE FIXED to avoid double counting)
    compute_emissions_and_fuel_use(data, titles, co2_corrct, biofuel_corrct, fuel_converter, c3ti, regions)

    
    # Call the survival function routine, updating scrappage and age matrix:
    if year <= np.max(data["TDA1"][:, 0, 0]):
        data = survival_function(data, time_lag, histend, year, titles)

    # Calculate the LCOT for each vehicle type.
    data = get_lcot(data, titles, year)

    # %% Simulation of stock and energy specs

    if year > np.min(data["TDA1"][:, 0, 0]):
        # Regions have different start dates (TDA1); regions with historical data are later skipped

        # Create a local dictionary for timeloop variables
        # It contains values between timeloop interations in the FTT core
        data_dt = {}

        # First, fill the time loop variables with the their lagged equivalents
        for var in time_lag.keys():

            data_dt[var] = np.copy(time_lag[var])


        # Create the regulation variable
        division = divide((time_lag['TEWK'][:, :, 0] - data['TREG']
                          [:, :, 0]), data['TREG'][:, :, 0])  # 0 when dividing by 0
        reg_constr = 0.5 + 0.5*np.tanh(1.5 + 10 * division)
        reg_constr[data['TREG'][:, :, 0] == 0.0] = 1.0
        reg_constr[data['TREG'][:, :, 0] == -1.0] = 0.0

        # Factor used to create quarterly data from annual figures
        no_it = int(data['noit'][0, 0, 0])
        dt = 1 / float(no_it)

        ############## Computing new shares ##################

        # Start the computation of shares
        for t in range(1, no_it + 1):

            # Both rvkm and RFLT are exogenous at the moment
            # Interpolate to prevent staircase profile.
            rvkmt = time_lag['RVKM'][:, 0, 0] + \
                (data['RVKM'][:, 0, 0] - time_lag['RVKM'][:, 0, 0]) * t * dt
            rfllt = time_lag['RFLT'][:, 0, 0] + \
                (data['RFLT'][:, 0, 0] - time_lag['RFLT'][:, 0, 0]) * (t-1) * dt
            rfltt = time_lag['RFLT'][:, 0, 0] + \
                (data['RFLT'][:, 0, 0] - time_lag['RFLT'][:, 0, 0]) * t * dt

            # Skip regions for which more recent data is available or with zero demand            
            regions = np.where((rfltt > 0) & (data['TDA1'][:, 0, 0] < year))[0]

            # Speed comparison between new jitted shares and original shares_transport
            if year in [2020, 2050] and t==no_it:  # Test every 10 years to avoid too much output
                import time as timing_module
                
                # Test original shares_transport function
                start_time = timing_module.time()
                endo_shares_old, endo_capacity_old = shares_transport(data_dt, data, year, rfltt, reg_constr, titles, c3ti, dt)
                time_old = timing_module.time() - start_time
                                
                
                # Test jitted shares function
                start_time = timing_module.time()
                change_in_shares = shares_change(
                    dt, regions, data_dt["TEWS"], data_dt["TELC"], data_dt["TLCD"],
                    data['TEWA'] * data['BTTC'][:, :, c3ti['17 Turnover rate'], None],
                    reg_constr, len(titles['RTI']), len(titles['VTTI'])
                )
                endo_shares_jit = np.zeros((len(titles['RTI']), len(titles['VTTI'])))
                endo_shares_jit[regions] = data_dt['TEWS'][regions, :, 0] + change_in_shares[regions]
                endo_capacity = endo_shares_jit * rfltt[:, np.newaxis]
                time_jit = timing_module.time() - start_time
                
                
                # Calculate speedup and check accuracy
                speedup_jit = time_old / time_jit if time_jit > 0 else float('inf')
                accuracy_jit = np.allclose(endo_shares_old, endo_shares_jit, atol=1e-10)
                
                print(f"\nYear {year}: TRANSPORT SHARES - old={time_old*1000:.1f}ms, new={time_jit*1000:.1f}ms, speedup={speedup_jit:.1f}x, accurate={accuracy_jit}")
                
            else:
                change_in_shares = shares_change(
                    dt, regions, data_dt["TEWS"], data_dt["TELC"], data_dt["TLCD"],
                    data['TEWA'] * data['BTTC'][:, :, c3ti['17 Turnover rate'], None],
                    reg_constr, len(titles['RTI']), len(titles['VTTI'])
                )
                endo_shares = np.zeros((len(titles['RTI']), len(titles['VTTI'])))
                endo_shares[regions] = data_dt['TEWS'][regions, :, 0] + change_in_shares[regions]
                endo_capacity = endo_shares * rfltt[:, np.newaxis]

            # Test vectorized regulatory policies function against old implementation
            if year % 10 == 0 and t==no_it:  # Test every 10 years
                import time as timing_module
                
                # Store old TEWS for comparison
                data_TEWS_before_reg = np.copy(data['TEWS'])
                
                # Test old loop-based implementation (for comparison)
                start_time = timing_module.time()
                data_TEWS_old_method = np.copy(data_TEWS_before_reg)
                for r in regions:
                    if r < len(titles['RTI']):  # Safety check
                        dUkTK = np.zeros([len(titles['VTTI'])])
                        dUkREG = np.zeros([len(titles['VTTI'])])
                        TWSA_scalar = 1.0

                        if (data['TWSA'][r, :, 0].sum() > 0.8 * rfltt[r] / 13):
                            TWSA_scalar = data['TWSA'][r, :, 0].sum() / (0.8 * rfltt[r] / 13)
                        
                        reg_vs_exog = ((data['TWSA'][r, :, 0]/TWSA_scalar/no_it + endo_capacity[r])
                                       > data['TREG'][r, :, 0]) & (data['TREG'][r, :, 0] >= 0.0)

                        dUkTK = np.where(reg_vs_exog, 0.0, data['TWSA'][r, :, 0] / TWSA_scalar / no_it)
                        dUkREG = -(endo_capacity[r] - endo_shares[r] * rfltt[r, np.newaxis]) * reg_constr[r, :].reshape([len(titles['VTTI'])])

                        dUk = dUkTK + dUkREG
                        dUtot = np.sum(dUk)

                        data_TEWS_old_method[r, :, 0] = (endo_capacity[r] + dUk) / (np.sum(endo_capacity[r]) + dUtot)
                
                time_old_reg = timing_module.time() - start_time
                
                # Test new vectorized implementation
                start_time = timing_module.time()
                data_TEWS_new_method = implement_regulatory_policies(endo_shares, endo_capacity, regions,
                                              data_TEWS_before_reg, data['TWSA'], data['TREG'], reg_constr,
                                              rfltt, rfllt, no_it, data['BTTC'][:, :, c3ti['8 lifetime']])
                time_new_reg = timing_module.time() - start_time
                
                # Compare results
                reg_speedup = time_old_reg / time_new_reg if time_new_reg > 0 else float('inf')
                reg_accuracy = np.allclose(data_TEWS_old_method, data_TEWS_new_method, atol=1e-10)
                
                #print(f"Year {year}: TRANSPORT REGULATORY POLICIES - old={time_old_reg*1000:.1f}ms, new={time_new_reg*1000:.1f}ms, speedup={reg_speedup:.1f}x, accurate={reg_accuracy}")
                
                # Use the new method result
                data['TEWS'] = data_TEWS_new_method
            else:
                # Normal operation - just use the vectorized regulatory policies
                data['TEWS'] = implement_regulatory_policies(endo_shares, endo_capacity, regions,
                                          data['TEWS'], data['TWSA'], data['TREG'], reg_constr,
                                          rfltt, rfllt, no_it, data['BTTC'][:, :, c3ti['8 lifetime']])


            # Raise error if there are negative values 
            # or regional market shares do not add up to one
            check_market_shares(data['TEWS'], titles, sector, year)
                

            ############## Update variables ##################

            # Update demand for driving (in km/ veh/ y) - exogenous atm
            # data['TEWL'][:, :, 0] = rvkmt[:, 0, 0]
            # Vehicle composition
            data['TEWK'][:, :, 0] = data['TEWS'][:, :, 0] * rfltt[:, np.newaxis]
            # Total distance driven per vehicle type
            data['TEWG'][:, :, 0] = data['TEWK'][:, :, 0] * \
                rvkmt[:, np.newaxis] * 1e-3

            # New additions (TEWI)
            data["TEWI"], tewi_t = get_sales(
                cap=data["TEWK"],
                cap_dt=data_dt["TEWK"], 
                cap_lag=time_lag["TEWK"],
                sales_or_investment_in=data["TEWI"],
                timescales=data['BTTC'][:, :, c3ti['8 lifetime']],
                dt=dt
                )
            
           
            # Fuel use and emissions
            # First: correct for age effects, with older vehicles emitting more CO2
            co2_corrct, region_has_fleet = co2_corr(data, titles)
            # Then, adjust fuel use and emissions factors for biofuel mandate
            biofuel_corrct, fuel_converter = biofuel_corr(data, titles, region_has_fleet)
            # Finally, compute TEWE (emissions) and TJEF (fuel use)
            compute_emissions_and_fuel_use(data, titles, co2_corrct, biofuel_corrct, fuel_converter, c3ti)
            
            
            ############## Learning-by-doing ##################

            # Cumulative global learning
            # Using a technological spill-over matrix (TEWB) together with capacity
            # additions (TEWI) we can estimate total global spillover of similar
            # vehicles
            # bi = np.matmul(data['TEWI'][:, :, 0], data['TEWB'][0, :, :])
            # dw = np.sum(bi, axis=0)*dt

            # Calculate learning only after TEWW histend
            if year > histend["TEWW"]:
                # New battery additions (MWh) = new sales (1000 vehicles) * average battery capacity (KWh)
                new_bat = np.zeros(
                    [len(titles['RTI']), len(titles['VTTI']), 1])
                new_bat[:, :, 0] = tewi_t[:, :, 0] * \
                    data["BTTC"][:, :, c3ti["18 Battery cap (kWh)"]]

                # Cumulative investment for learning cost reductions
                bi = np.zeros((len(titles['RTI']), len(titles['VTTI'])))
                for r in range(len(titles['RTI'])):
                    # Investment spillover
                    bi[r, :] = np.matmul(data['TEWB'][0, :, :], tewi_t[r, :, 0])

                # Total new investment
                dw = np.sum(bi, axis=0)
                # Total new battery investment (in MWh)
                dwev = np.sum(new_bat, axis=0)

                # Cumulative capacity for batteries first
                data['TEWW'][0, :, 0] = data_dt['TEWW'][0, :, 0] + dwev[:, 0]
                bat_cap = np.copy(data["TEWW"])

                # Cumulative capacity for ICE vehicles
                for veh in range(len(titles['VTTI'])):
                    if (veh < 18) or (veh > 23):
                        data['TEWW'][0, veh, 0] = data_dt['TEWW'][0,
                                                                  veh, 0] + dw[veh]
                        # Make sure bat_cap for ICE vehicles is 0
                        bat_cap[0, veh, 0] = 0

                # Copy over the technology cost categories that do not change
                # (all except prices which are updated through learning-by-doing below)
                data['BTTC'] = np.copy(data_dt['BTTC'])

                # Battery learning
                for veh in range(len(titles['VTTI'])):
                    if 17 < veh < 24:
                        # Battery cost as a result of learning
                        # Battery cost = energy density over time*rare metal price trend over time
                        data["BTTC"][:, veh, c3ti['19 Battery cost ($/kWh)']] = (
                            (data["BTTC"][:, veh, c3ti['22 Energy density']]
                             ** (year - 2022))
                            * (data["BTTC"][:, veh, c3ti['21 Rare metal price']] ** (year - 2022))
                            * data['BTCI'][:, veh, c3ti['19 Battery cost ($/kWh)']]
                            * (np.sum(bat_cap, axis=1) / np.sum(data["TWWB"], axis=1))
                            ** data["BTTC"][:, veh, c3ti['16 Learning exponent']])

                # Save battery cost
                data["TEBC"] = np.zeros(
                    [len(titles['RTI']), len(titles['VTTI']), 1])
                data["TEBC"][:, :, 0] = data["BTTC"][:,
                                                     :, c3ti['19 Battery cost ($/kWh)']]

                # Initialise variable for indirect EV/PHEV costs
                id_cost = np.zeros(
                    [len(titles['RTI']), len(titles['VTTI']), 1])
                # Initialise variable for cost of EV/PHEV - battery
                i_cost = np.zeros([len(titles['RTI']), len(titles['VTTI']), 1])

                # Learning-by-doing effects on investment
                for veh in range(len(titles['VTTI'])):
                    if data['TEWW'][0, veh, 0] > 0.1:
                        # Learning on indirect costs (only relevant for EVs and PHEVs)
                        id_cost[:, veh, 0] = (data['BTCI'][:, veh, c3ti['1 Prices cars (USD/veh)']]
                                              * 0.15 * 0.993**(year - 2022))
                        # Learning on the EV/PHEV (seperate from battery)
                        i_cost[:, veh, 0] = data["TEVC"][:, veh, 0] * ((np.sum(bat_cap, axis=1) / np.sum(data["TWWB"], axis=1))
                                                                       ** (data["BTTC"][:, veh, c3ti['16 Learning exponent']]/2))


                        # Calculate new costs (seperate treatments for ICE vehicles and EVs/PHEVs)
                        if 17 < veh < 24:
                            # EVs and PHEVs HERE
                            # Cost (excluding the cost of battery) + Updated cost of battery
                            # + updated indirect cost) * Markup factor
                            data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] =  \
                                ((i_cost[:, veh, 0] + (data["BTTC"][:, veh, c3ti['19 Battery cost ($/kWh)']]
                                                       * data["BTTC"][:, veh, c3ti['18 Battery cap (kWh)']]) + id_cost[:, veh, 0])
                                 * data['BTTC'][:, veh, c3ti['20 Markup factor']])
                        else:
                            # ICE HERE
                            data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] = \
                                data_dt['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] \
                                * (1.0 + data['BTTC'][:, veh, c3ti['16 Learning exponent']]
                                   * dw[veh] / data['TEWW'][0, veh, 0])


            # Save battery costs for front end
            data["TEBC"] = np.zeros(
                [len(titles['RTI']), len(titles['VTTI']), 1])
            data["TEBC"][:, :, 0] = data["BTTC"][:,
                                                 :, c3ti['19 Battery cost ($/kWh)']]

            # =================================================================
            # Update the time-loop variables
            # =================================================================

            # Calculate levelised cost again
            data = get_lcot(data, titles, year)

            # Update time loop variables:
            for var in data_dt.keys():

                data_dt[var] = np.copy(data[var])

        # Investment in terms of car purchases
        data['TWIY'][:, :, 0] = (data['TEWI'][:, :, 0] 
                                 * data['BTTC'][:, :, c3ti['1 Prices cars (USD/veh)']] / 1.33
                                 )

        # Call the survival function routine
        data = survival_function(data, time_lag, histend, year, titles)

    return data


