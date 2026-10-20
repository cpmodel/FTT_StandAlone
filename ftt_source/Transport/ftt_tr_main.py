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
from ftt_source.ftt_core.ftt_sales_or_investments import get_sales
from ftt_source.ftt_core.ftt_shares import shares_change
from ftt_source.ftt_core.ftt_mandate import implement_seeding, implement_mandate
from ftt_source.sector_coupling.battery_lbd import battery_costs
from ftt_source.ftt_core.ftt_exogenous_sales import exogenous_sales
from ftt_source.ftt_core.ftt_exogenous_capacity import regulation_correction

from ftt_source.support.divide import divide
from ftt_source.support.check_market_shares import check_market_shares
from ftt_source.support.get_vars_to_copy import get_domain_vars_to_copy
from ftt_source.support.singleton_dimension import (
    restore_all_singleton_axes,
    squeeze_all_singleton_axes,
    squeeze_singleton_third_axis,
    unsqueeze_singleton_third_axis,
)

from ftt_source.Transport.ftt_tr_lcot import get_lcot, set_carbon_tax
from ftt_source.Transport.ftt_tr_emission_corrections import co2_corr, biofuel_corr, compute_emissions_and_fuel_use
from ftt_source.Transport.ftt_tr_survival import survival_function, add_new_cars_age_matrix
from ftt_source.ftt_core.ftt_emissions_regulation import implement_emissions_regulation

# Green technology indices for Transport (EVs)
GREEN_INDICES_EV = [18, 19, 20]


# %% Fleet size - under development
# -----------------------------------------------------------------------------
# ----------------- Gompertz equation for fleet size --------------------------
# -----------------------------------------------------------------------------
# def fleet_size(data, titles):
#
#     return print("Hello")


# =============================================================================
# Main solver
# =============================================================================
def solve(data, time_lag, titles, histend, year, domain):
    """
    Main solution function for the module.

    Add an extended description in the future.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution
    time_lag: dict
        Model variables in previous year
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of historical data by variable
    year: int
        Current year
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
    # Temporarily squeeze singleton third axes for Transport-internal calculations.
    # Arrays whose third axis represents costs, ages, fuel, or substitutions remain 3D.
    shared_vars = {
        'PFRM', 'PFRG', 'PFRE', 'PFRB', 'Battery price',
        'Cumulative total batcap', 'Battery cap additions', 'noit',
        'TEWK', 'ZEWK',
    }
    squeeze_candidates = {var for var, dom in domain.items() if dom == 'FTT-Tr'} | shared_vars
    squeeze_candidates -= {'TESF', 'TESH', 'TETH', 'MSSC'}
    squeezed_data_keys = squeeze_singleton_third_axis(data, squeeze_candidates)
    squeezed_time_lag_keys = squeeze_singleton_third_axis(time_lag, squeeze_candidates)
    vector_variables = {
        'Battery price', 'CO2taxTr', 'RBFM',
        'RFLT', 'RTCO', 'RVKM', 'TDA1', 'TDA2', 'TEWW', 'TWWB', 'noit',
    }
    data_vector_shapes = squeeze_all_singleton_axes(data, vector_variables)
    time_lag_vector_shapes = squeeze_all_singleton_axes(
        time_lag, vector_variables
    )

    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}
    jti = {category: index for index, category in enumerate(titles['JTI'])}

    sector = "tr_road_pass"
    sector_index = 0
    sector_index = 15  #titles['FUTI'].index('16 Road Transport')

    num_regions = len(titles['RTI'])
    num_techs = len(titles['VTTI'])

    # Store fuel prices and convert to $2013/toe
    # It's actually in current$/toe
    # TODO: Temporary deflator values
    data['TE3P'][:, jti["5 Middle distillates"]] = time_lag['PFRM'][:, sector_index] / 1.33
    data['TE3P'][:, jti["7 Natural gas"]] = time_lag['PFRG'][:, sector_index] / 1.33
    data['TE3P'][:, jti["8 Electricity"]] = time_lag['PFRE'][:, sector_index] / 1.33
    data['TE3P'][:, jti["11 Biofuels"]] = time_lag['PFRB'][:, sector_index] / 1.33
#    data['TE3P'][:, "12 Hydrogen", 0] = data['PFRE'][:, sector_index, 0] * 2.0

    # Use data-driven timing based on TDA2 (last year of cost data)
    if year == np.min(data["TDA2"]):
        start_i_cost = np.zeros([num_regions, num_techs])
        for veh in range(num_techs):
            if 17 < veh < 24:
                # Starting EV/PHEV cost (without battery)
                # Uses global Battery price instead of per-vehicle battery cost
                start_i_cost[:, veh] = (data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']]
                                           / data['BTTC'][:, veh, c3ti['19 Markup factor']]
                                           - data['BTTC'][:, veh, c3ti['18 Battery cap (kWh)']]
                                           * data['Battery price']
                                           - data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] * 0.15)
            else:
                start_i_cost[:, veh] = 0
        # TEVC is 'Start_ICost' in the Fortran model
        data["TEVC"] = start_i_cost
        data["BTCI"] = np.copy(data["BTTC"])  # Save start cost matrix
        time_lag['BTCI'] = np.copy(data["BTTC"])  # Save to lagged data too

        # Define starting battery capacity, this does not change
        data["TWWB"] = np.copy(data["TEWW"])
        for veh in range(num_techs):
            if (veh < 18) or (veh > 23):
                # Set starting cumulative battery capacities to 0 for ICE vehicles
                data["TWWB"][veh] = 0

    if year > np.min(data["TDA2"]):
        data["BTTC"] = np.copy(time_lag["BTTC"])  # The cost matrix
        data["BTCI"] = np.copy(time_lag["BTCI"])  # The starting cost matrix
        data['TEVC'] = np.copy(time_lag['TEVC'])  # The cost without batteries
        data['TWWB'] = np.copy(time_lag['TWWB'])

    for r in range(num_regions):
        # %% Initialise
        # Up to the last year of historical market share data
        if year <= data["TDA1"][r]:

            # Correction to market shares
            # Sometimes historical market shares do not add up to 1.0
            share_sum = np.sum(data['TEWS'][r])
            if (abs(share_sum - 1.0) > 1e-9) and (share_sum > 0.0):
                data['TEWS'][r] /= share_sum

            # Computes initial values for the capacity factor, numbers of
            # vehicles by technology and distance driven

            # "Capacities", defined as 1000 vehicles
            data['TEWK'] = data['TEWS'] * \
                data['RFLT'][:, None]

            # "Generation", defined as total mln km driven
            data['TEWG'] = data['TEWK'] * \
                data['RVKM'][:, None] * 1e-3

            # "Emissions", Factor 1.2 approx fleet efficiency factor, corrected later with CO2_corr
            data['TEWE'] = data['TEWG'] * \
                data['BTTC'][:, :, c3ti['14 CO2Emissions']]/1e6*1.2

        if year == data["TDA1"][r]:
            # Define starting battery capacity
            start_bat_cap = np.copy(data["TEWW"])
            for veh in range(num_techs):
                if (veh < 18) or (veh > 23):
                    # Set starting cumulative battery capacities to 0 for ICE vehicles
                    start_bat_cap[veh] = 0
            # TWWB is 'StartBatCap' in the FORTRAN model
            data["TWWB"] = start_bat_cap

            # Save initial cost matrix (BTCLi from FORTRAN model)
            data["BTCI"] = np.copy(data["BTTC"])

    # Fuel use and emissions

    regions = np.where(year <= data['TDA1'])[0]
    # First: correct for age effects, with older vehicles emitting more CO2
    co2_corrct, has_fleet = co2_corr(data, titles, regions)
    # Then, correct for the biofuel mandate, reducing emissions
    biofuel_corrct, fuel_converter = biofuel_corr(data, titles, has_fleet, regions)
    # Finally, compute TEWE (emissions) and TJEF (fuel use) (TO BE FIXED to avoid double counting)
    compute_emissions_and_fuel_use(data, titles, co2_corrct, biofuel_corrct, fuel_converter, c3ti, regions)


    # Call the survival function routine, updating scrappage and age matrix:
    if year <= np.max(data["TDA1"]):
        data = survival_function(data, time_lag, histend, year, titles)

    # Calculate the LCOT for each vehicle type.
    carbon_costs = set_carbon_tax(data, c3ti, year)
    data = get_lcot(data, titles, carbon_costs, year)

    # %% Simulation of stock and energy specs

    if year > np.min(data["TDA1"]):
        # Regions have different start dates (TDA1); regions with historical data are later skipped

        # Create a local dictionary for timeloop variables
        # It contains values between timeloop interations in the FTT core
        data_dt = {}

        # First, fill the time loop variables with the their lagged equivalents
        vars_to_copy = get_domain_vars_to_copy(time_lag, domain, 'FTT-Tr')
        for var in vars_to_copy:
            data_dt[var] = np.copy(time_lag[var])


        # Create the regulation variable
        relative_excess = divide((time_lag['TEWK'] - data['TREG']), data['TREG'])  # 0 when dividing by 0
        reg_constr = 0.5 + 0.5*np.tanh(1.5 + 10 * relative_excess)
        reg_constr[data['TREG'] == 0.0] = 1.0
        reg_constr[data['TREG'] == -1.0] = 0.0

        # Factor used to create subannual data from annual figures
        no_it = int(data['noit'])
        dt = 1 / float(no_it)

        ############## Computing new shares ##################

        # Start the computation of shares
        for t in range(1, no_it + 1):

            # Both rvkm and RFLT are exogenous at the moment
            # Interpolate to prevent staircase profile.
            rvkmt = time_lag['RVKM'] + (data['RVKM'] - time_lag['RVKM']) * t * dt
            rfllt = time_lag['RFLT'] + (data['RFLT'] - time_lag['RFLT']) * (t - 1) * dt
            rfltt = time_lag['RFLT'] + (data['RFLT'] - time_lag['RFLT']) * t * dt

            # Skip regions for which more recent data is available or with zero demand
            regions = np.where((rfltt > 0) & (data['TDA1'] < year))[0]

            # The core FTT equations, taking into account old shares, costs and regulations
            change_in_shares = shares_change(
                dt=dt,
                regions=regions,
                shares_dt=data_dt["TEWS"],  # Shares at previous t
                costs=data_dt["TELC"],      # Logarithm of costs
                costs_sd=data_dt["TLCD"],   # Standard deviation of log(costs)
                subst=data['TEWA'] * data['BTTC'][:, :, c3ti['17 Turnover rate'], None],  # Substitution turnover rate
                reg_constr=reg_constr,          # Constraint due to regulation
                num_regions=num_regions, # Number of regions
                num_techs=num_techs   # Number of techs
            )

            endo_shares = np.zeros((num_regions, num_techs))
            endo_shares[regions] = data_dt['TEWS'][regions] + change_in_shares[regions]
            endo_capacity = endo_shares * rfltt[:, np.newaxis]

            # Change in capacity from exogenous sales, capped at maximum sales
            dcap_exog_sales = exogenous_sales(
                data['TWSA'][regions], rfltt[regions], endo_capacity[regions], data['TREG'][regions],
                no_it, data['BTTC'][regions, :, c3ti['8 lifetime']]
            )

            # Correction for regulation (phase-out) when demand is growing; main effect in shares equation
            dcap_reg_corr = regulation_correction(
                endo_capacity[regions], endo_shares[regions], rfltt[regions, None], reg_constr[regions])

            # New capacity and shares
            new_capacity = endo_capacity[regions] + dcap_exog_sales + dcap_reg_corr
            total_capacity = np.sum(new_capacity, axis=1)
            data['TEWS'][regions] = divide(new_capacity, total_capacity[:, None])

            # Raise error if there are negative values
            # or regional market shares do not add up to one
            check_market_shares(data['TEWS'], titles, sector, year)


            ############## Update variables ##################

            # Update demand for driving (in km/ veh/ y) - exogenous atm
            # data['TEWL'][:, :, 0] = rvkmt[:, 0, 0]
            # Vehicle composition
            data['TEWK'] = data['TEWS'] * rfltt[:, np.newaxis]
            # Total distance driven per vehicle type
            data['TEWG'] = data['TEWK'] * \
                rvkmt[:, np.newaxis] * 1e-3

            # New additions (TEWI)
            tewi, tewi_t = get_sales(
                cap=data["TEWK"],
                cap_dt=data_dt["TEWK"],
                cap_lag=time_lag["TEWK"],
                sales_or_investment_in=data["TEWI"],
                timescales=data['BTTC'][:, :, c3ti['8 lifetime']],
                dt=dt
            )
            data["TEWI"] = tewi

            # Apply EV seeding (first 5 years) - small boost for low-adoption regions
            tewi, tewi_t, tewk = implement_seeding(
                data['TEWK'], tewi, tewi_t, year, GREEN_INDICES_EV,
                np.min(data["TDA1"]),
            )
            data["TEWI"] = tewi
            data["TEWK"] = tewk


            # Policy levers are MUTUALLY EXCLUSIVE: mandate/kickstarter OR emissions regulation
            # Check which policies are active
            mandate_active = not np.all(data["EV mandate"][:, 2] == 0)
            emissions_reg_active = ("EV CO2 regulation" in data and
                                    not np.all(data["EV CO2 regulation"][:, 2] == 0))

            if mandate_active:
                tewi, tewi_t_2d, tewk = implement_mandate(
                    data['TEWK'],
                    data['TEWI'],
                    tewi_t,
                    year,
                    GREEN_INDICES_EV,
                    data['EV mandate'],
                )
                data["TEWI"] = tewi
                tewi_t = tewi_t_2d
                data["TEWK"] = tewk

            elif emissions_reg_active:
                # Do not hard-code a calendar threshold here: regulation timing
                # should be driven by the policy input and handled inside the
                # regulation implementation for inactive/pre-start years.
                tewi, tewi_t, tewk = implement_emissions_regulation(
                    data['TEWK'],
                    data['TEWI'],
                    tewi_t, year,
                    data['BTTC'][:, :, c3ti['14 CO2Emissions']],
                    data['EV CO2 regulation'], sector='transport',
                )
                data["TEWI"] = tewi
                data["TEWK"] = tewk

            # Recalculate TEWS/TEWG after seeding. Do not touch regions with pseudoshares (>51)
            total_capacity = data['TEWK'].sum(axis=1, keepdims=True)
            mask = total_capacity[:, 0] > 0
            data['TEWS'][mask] = data['TEWK'][mask] / total_capacity[mask]
            data['TEWG'] = data['TEWK'] * rvkmt[:, np.newaxis] * 1e-3

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
            new_bat = tewi_t * data["BTTC"][:, :, c3ti["18 Battery cap (kWh)"]]
            # Track battery additions for sector coupling (Transport is sector index 1)
            data["Battery cap additions"][1, t-1] = np.sum(new_bat) / 1000  # In GWh

            # Cumulative investment for learning cost reductions
            bi = np.zeros((num_regions, num_techs))
            for r in range(num_regions):
                # Investment spillover
                bi[r, :] = np.matmul(data['TEWB'][0, :, :], tewi_t[r])

            # Total new investment
            dw = np.sum(bi, axis=0)

            # Cumulative capacity
            data['TEWW'] = data_dt['TEWW'] + dw

            # Copy over the technology cost categories
            # Prices are updated through learning-by-doing below
            data['BTTC'] = np.copy(data_dt['BTTC'])
            # Copy over the initial cost matrix
            data["BTCI"] = np.copy(data_dt['BTCI'])

            # Global battery learning via sector coupling
            data = battery_costs(data, time_lag, year, t, titles, histend)

            # Initialise variable for indirect EV/PHEV costs
            id_cost = np.zeros([num_regions, num_techs])

            # Copy prices during historical period
            rs_to_copy = year <= data["TDA2"]
            rs = year > data['TDA2']
            data['BTTC'][rs_to_copy] = time_lag['BTTC'][rs_to_copy]

            # Learning-by-doing effects on investment
            for veh in range(num_techs):
                if data['TEWW'][veh] > 0.1:
                    # Calculate new costs (separate treatments for ICE vehicles and EVs/PHEVs)
                    if 17 < veh < 24:
                        # Learning on indirect costs (only relevant for EVs and PHEVs)
                        id_cost[rs, veh] = (data_dt['BTCI'][rs, veh, c3ti['1 Prices cars (USD/veh)']]
                                              * 0.15 * 0.993**(year - 2022))

                        # Learning on the EV/PHEV (separate from battery)
                        data['TEVC'][rs, veh] = (
                            data_dt['TEVC'][rs, veh]
                            * (1.0 + data['BTTC'][rs, veh, c3ti['16 Learning exponent']] / 2
                               * dw[veh] / data['TEWW'][veh])
                        )

                        # EVs and PHEVs: Cost = (base cost + battery cost + indirect cost) * markup
                        # Uses global Battery price from sector coupling
                        data['BTTC'][rs, veh, c3ti['1 Prices cars (USD/veh)']] = (
                            (data['TEVC'][rs, veh]  # Costs without batteries
                             + (data["Battery price"][rs] * data["BTTC"][rs, veh, c3ti['18 Battery cap (kWh)']])
                             + id_cost[rs, veh])  # Indirect costs
                            * data['BTTC'][rs, veh, c3ti['19 Markup factor']]
                        )
                    else:
                        # ICE vehicles
                        data['BTTC'][rs, veh, c3ti['1 Prices cars (USD/veh)']] = (
                            data_dt['BTTC'][rs, veh, c3ti['1 Prices cars (USD/veh)']]
                            * (1.0 + data['BTTC'][rs, veh, c3ti['16 Learning exponent']]
                               * dw[veh] / data['TEWW'][veh])
                        )



            # Calculate levelised cost again
            carbon_costs = set_carbon_tax(data, c3ti, year)
            data = get_lcot(data, titles, carbon_costs, year)

            # =================================================================
            # Update the time-loop variables
            # =================================================================

            # Update time loop variables:
            for var in vars_to_copy:

                data_dt[var] = np.copy(data[var])

        # Investment in terms of car purchases (cumulative within year):
        data['TWIY'] = (data['TEWI']
                                 * data['BTTC'][:, :, c3ti['1 Prices cars (USD/veh)']] / 1.33)

        # Call the survival function routine
        data = survival_function(data, time_lag, histend, year, titles)

    # Restore singleton third axes for compatibility with the rest of the model.
    restore_all_singleton_axes(data, data_vector_shapes)
    restore_all_singleton_axes(time_lag, time_lag_vector_shapes)
    unsqueeze_singleton_third_axis(data, squeezed_data_keys)
    unsqueeze_singleton_third_axis(time_lag, squeezed_time_lag_keys)

    return data
