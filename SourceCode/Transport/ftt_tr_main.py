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

# Local library imports
from SourceCode.ftt_core.ftt_sales_or_investments import get_sales
from SourceCode.ftt_core.ftt_shares import shares_change
from SourceCode.sector_coupling.battery_lbd import battery_costs

from SourceCode.support.divide import divide
from SourceCode.support.check_market_shares import check_market_shares
from SourceCode.support.get_vars_to_copy import get_loop_vars_to_copy, get_domain_vars_to_copy

from SourceCode.Transport.ftt_tr_lcot import get_lcot, set_carbon_tax
from SourceCode.Transport.ftt_tr_emission_corrections import co2_corr, biofuel_corr, compute_emissions_and_fuel_use
from SourceCode.Transport.ftt_tr_survival import survival_function, add_new_cars_age_matrix
from SourceCode.ftt_core.ftt_mandate import implement_seeding, implement_mandate
from SourceCode.Transport.ftt_tr_kickstarter import implement_kickstarter
from SourceCode.Transport.ftt_tr_emissions_regulation import implement_emissions_regulation

# Green technology indices for Transport (EVs and PHEVs)
GREEN_INDICES_EV = [18, 19, 20]


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
def solve(data, time_lag, iter_lag, titles, histend, year, domain):
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
    domain: dictionary of lists
        Pairs variables to domains

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

    # Use data-driven timing based on TDA2 (last year of cost data)
    if year == np.min(data["TDA2"][:, 0, 0]):
        start_i_cost = np.zeros([len(titles['RTI']), len(titles['VTTI']), 1])
        for veh in range(len(titles['VTTI'])):
            if 17 < veh < 24:
                # Starting EV/PHEV cost (without battery)
                # Uses global Battery price instead of per-vehicle battery cost
                start_i_cost[:, veh, 0] = (data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']]
                                           / data['BTTC'][:, veh, c3ti['19 Markup factor']]
                                           - data['BTTC'][:, veh, c3ti['18 Battery cap (kWh)']]
                                           * data['Battery price'][:, 0, 0]
                                           - data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] * 0.15)
            else:
                start_i_cost[:, veh, 0] = 0
        # TEVC is 'Start_ICost' in the Fortran model
        data["TEVC"] = start_i_cost
        data["BTCI"] = np.copy(data["BTTC"])  # Save start cost matrix
        time_lag['BTCI'] = np.copy(data["BTTC"])  # Save to lagged data too

        # Define starting battery capacity, this does not change
        data["TWWB"] = np.copy(data["TEWW"])
        for veh in range(len(titles['VTTI'])):
            if (veh < 18) or (veh > 23):
                # Set starting cumulative battery capacities to 0 for ICE vehicles
                data["TWWB"][0, veh, 0] = 0

    if year > np.min(data["TDA2"][:, 0, 0]):
        data["BTTC"] = np.copy(time_lag["BTTC"])  # The cost matrix
        data["BTCI"] = np.copy(time_lag["BTCI"])  # The starting cost matrix
        data['TEVC'] = np.copy(time_lag['TEVC'])  # The cost without batteries
        data['TWWB'] = np.copy(time_lag['TWWB'])

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
    carbon_costs = set_carbon_tax(data, c3ti, year)
    data = get_lcot(data, titles, carbon_costs, year)

    # %% Simulation of stock and energy specs

    if year > np.min(data["TDA1"][:, 0, 0]):
        # Regions have different start dates (TDA1); regions with historical data are later skipped

        # Create a local dictionary for timeloop variables
        # It contains values between timeloop interations in the FTT core
        data_dt = {}

        # First, fill the time loop variables with the their lagged equivalents
        vars_to_copy = get_domain_vars_to_copy(time_lag, domain, 'FTT-Tr')
        for var in vars_to_copy:
            data_dt[var] = np.copy(time_lag[var])

        # Initialize TWIY accumulator for this year (will accumulate within timesteps)
        data_dt['TWIY'] = np.zeros([len(titles['RTI']), len(titles['VTTI']), 1])

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
            
            # The core FTT equations, taking into account old shares, costs and regulations
            change_in_shares = shares_change(
                dt=dt,                          
                regions=regions,
                shares_dt=data_dt["TEWS"],      # Shares at previous t
                costs=data_dt["TELC"],          # Logarithm of costs
                costs_sd=data_dt["TLCD"],       # Standard deviation of log(costs)
                subst=data['TEWA'] * data['BTTC'][:, :, c3ti['17 Turnover rate'], None],  # Substitution turnover rate
                reg_constr=reg_constr,          # Constraint due to regulation
                num_regions=len(titles['RTI']), # Number of regions
                num_techs=len(titles['VTTI'])   # Number of techs
            )
            
            endo_shares = np.zeros((len(titles['RTI']), len(titles['VTTI'])))
            endo_shares[regions] = data_dt['TEWS'][regions, :, 0] + change_in_shares[regions]
            endo_capacity = endo_shares * rfltt[:, np.newaxis]

            
            # Implement exogenous sales and correct for stretching
            for r in regions:
                if r < len(titles['RTI']):  # Safety check
                    dUkTK = np.zeros([len(titles['VTTI'])])
                    dUkREG = np.zeros([len(titles['VTTI'])])
                    TWSA_scalar = 1.0
                    
                    # Check that exogenous sales additions aren't too large
                    # As a proxy it can't be greater than 80% of the fleet size
                    # divided by 13 (the average lifetime of vehicles)
                    if (data['TWSA'][r, :, 0].sum() > 0.8 * rfltt[r] / 13):
                        TWSA_scalar = data['TWSA'][r, :, 0].sum() / (0.8 * rfltt[r] / 13)
                
                    # Check endogenous capacity plus additions for a single time step does not exceed regulated capacity.
                    reg_vs_exog = ((data['TWSA'][r, :, 0]/TWSA_scalar/no_it + endo_capacity[r])
                                   > data['TREG'][r, :, 0]) & (data['TREG'][r, :, 0] >= 0.0)
                    
                    # TWSA is yearly capacity additions. We need to split it up based on the number of time steps, and also scale it if necessary.
                    dUkTK = np.where(reg_vs_exog, 0.0, data['TWSA'][r, :, 0] / TWSA_scalar / no_it)
                    
                    # Correct for regulations due to the stretching effect. This is the difference in capacity due only to rflt increasing.
                    # This is the difference between capacity based on the endogenous capacity, and what the endogenous capacity would have been
                    # if rflt (i.e. total demand) had not grown.
                    dUkREG = -(endo_capacity[r] - endo_shares[r] * rfllt[r, np.newaxis]) * reg_constr[r, :].reshape([len(titles['VTTI'])])
                    
                    # Sum effect of exogenous sales additions (if any) with effect of regulations.
                    dUk = dUkTK + dUkREG
                    dUtot = np.sum(dUk)
                    
                    # Calculate changes to endogenous capacity, and use to find new market shares
                    # Zero capacity will result in zero shares
                    # All other capacities will be streched

                    data['TEWS'][r, :, 0] = (endo_capacity[r] + dUk) / (np.sum(endo_capacity[r]) + dUtot)
            

            # Raise error if any values are negative or market shares do not sum to 1
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

            # Apply EV seeding (2025-2030) - small boost for low-adoption regions
            data["TEWI"], tewi_t, data["TEWK"] = implement_seeding(
                data['TEWK'], data['TEWI'], tewi_t, year, GREEN_INDICES_EV
            )

            # Policy levers are MUTUALLY EXCLUSIVE: mandate OR kickstarter OR emissions regulation
            # Check which policies are active
            mandate_active = not np.all(data["EV mandate"][:, 0, 0] == 0)
            kickstarter_active = not np.all(data["EV kickstarter"][:, 0, 0] == 0)
            emissions_reg_active = ("emissions regulation" in data and
                                    not np.all(data["emissions regulation"][:, 0, 0] == 0))

            if mandate_active:
                # Full mandate - only runs if EV mandate != 0 (disabled in S0)
                data["TEWI"], tewi_t, data["TEWK"] = implement_mandate(
                    data['TEWK'], data['TEWI'], tewi_t, year, GREEN_INDICES_EV, data["EV mandate"]
                )
            elif kickstarter_active:
                # Kickstarter policy - only runs if EV kickstarter != 0 (disabled in S0)
                data["TEWI"], tewi_t, data["TEWK"] = implement_kickstarter(
                    data['TEWK'], data["EV kickstarter"], data['TEWI'], tewi_t, year
                )
            elif emissions_reg_active:
                # Emissions regulation - segment-specific targets with proportional redistribution
                # Baseline emissions are cached in the module, not in data dictionary
                data["TEWI"], tewi_t, data["TEWK"] = implement_emissions_regulation(
                    data['TEWK'],
                    data["emissions regulation"],
                    data['TEWI'],
                    tewi_t,
                    year,
                    data['BTTC'][:, :, c3ti['14 CO2Emissions']]
                )

            # Recalculate TEWS/TEWG after seeding
            for r in regions:
                if np.sum(data['TEWK'][r, :, 0]) > 0:
                    data['TEWS'][r, :, 0] = data['TEWK'][r, :, 0] / np.sum(data['TEWK'][r, :, 0])
            data['TEWG'][:, :, 0] = data['TEWK'][:, :, 0] * rvkmt[:, np.newaxis] * 1e-3

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

            # New battery additions (MWh) = new sales (1000 vehicles) * average battery capacity (kWh)
            new_bat = tewi_t[:, :, 0] * data["BTTC"][:, :, c3ti["18 Battery cap (kWh)"]]
            # Track battery additions for sector coupling (Transport is sector index 1)
            data["Battery cap additions"][1, t-1, 0] = np.sum(new_bat) / 1000  # In GWh

            # Cumulative investment for learning cost reductions
            bi = np.zeros((len(titles['RTI']), len(titles['VTTI'])))
            for r in range(len(titles['RTI'])):
                # Investment spillover
                bi[r, :] = np.matmul(data['TEWB'][0, :, :], tewi_t[r, :, 0])

            # Total new investment
            dw = np.sum(bi, axis=0)

            # Cumulative capacity
            data['TEWW'][0, :, 0] = data_dt['TEWW'][0, :, 0] + dw

            # Copy over the technology cost categories
            # Prices are updated through learning-by-doing below
            data['BTTC'] = np.copy(data_dt['BTTC'])
            # Copy over the initial cost matrix
            data["BTCI"] = np.copy(data_dt['BTCI'])

            # Global battery learning via sector coupling
            data = battery_costs(data, data_dt, time_lag, year, t, titles, histend)

            # Initialise variable for indirect EV/PHEV costs
            id_cost = np.zeros([len(titles['RTI']), len(titles['VTTI']), 1])

            # Copy prices during historical period
            rs_to_copy = year <= data["TDA2"][:, 0, 0]
            rs = year > data['TDA2'][:, 0, 0]
            data['BTTC'][rs_to_copy] = time_lag['BTTC'][rs_to_copy]

            # Learning-by-doing effects on investment
            for veh in range(len(titles['VTTI'])):
                if data['TEWW'][0, veh, 0] > 0.1:
                    # Calculate new costs (separate treatments for ICE vehicles and EVs/PHEVs)
                    if 17 < veh < 24:
                        # Learning on indirect costs (only relevant for EVs and PHEVs)
                        id_cost[rs, veh, 0] = (data_dt['BTCI'][rs, veh, c3ti['1 Prices cars (USD/veh)']]
                                              * 0.15 * 0.993**(year - 2022))

                        # Learning on the EV/PHEV (separate from battery)
                        data['TEVC'][rs, veh, 0] = (
                            data_dt['TEVC'][rs, veh, 0]
                            * (1.0 + data['BTTC'][rs, veh, c3ti['16 Learning exponent']] / 2
                               * dw[veh] / data['TEWW'][0, veh, 0])
                        )

                        # EVs and PHEVs: Cost = (base cost + battery cost + indirect cost) * markup
                        # Uses global Battery price from sector coupling
                        data['BTTC'][rs, veh, c3ti['1 Prices cars (USD/veh)']] = (
                            (data['TEVC'][rs, veh, 0]  # Costs without batteries
                             + (data["Battery price"][rs, 0, 0] * data["BTTC"][rs, veh, c3ti['18 Battery cap (kWh)']])
                             + id_cost[rs, veh, 0])  # Indirect costs
                            * data['BTTC'][rs, veh, c3ti['19 Markup factor']]
                        )
                    else:
                        # ICE vehicles
                        data['BTTC'][rs, veh, c3ti['1 Prices cars (USD/veh)']] = (
                            data_dt['BTTC'][rs, veh, c3ti['1 Prices cars (USD/veh)']]
                            * (1.0 + data['BTTC'][rs, veh, c3ti['16 Learning exponent']]
                               * dw[veh] / data['TEWW'][0, veh, 0])
                        )

            # Investment in terms of car purchases (cumulative within year):
            data['TWIY'][:, :, 0] = (data_dt['TWIY'][:, :, 0]
                                     + data['TEWI'][:, :, 0] * dt
                                     * data['BTTC'][:, :, c3ti['1 Prices cars (USD/veh)']] / 1.33)

            # Calculate levelised cost again
            carbon_costs = set_carbon_tax(data, c3ti, year)
            data = get_lcot(data, titles, carbon_costs, year)
            
            # =================================================================
            # Update the time-loop variables
            # =================================================================
            
            # Copy transport variables that have changed in data_dt
            vars_to_copy = get_loop_vars_to_copy(data, data_dt, domain, 'FTT-Tr')
            for var in vars_to_copy:
                data_dt[var] = np.copy(data[var])

            # Update TWIY in data_dt for next iteration accumulation
            data_dt['TWIY'] = np.copy(data['TWIY'])

        # Call the survival function routine
        data = survival_function(data, time_lag, histend, year, titles)

        
        if year==2050:
            print(f"Total electric cars in 2050: {np.sum(data['TEWK'][:, 18, 0] + data['TEWK'][:, 19, 0] + data['TEWK'][:, 20, 0])/10e3:.0f} M cars")


    return data


