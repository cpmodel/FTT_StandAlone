# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:30 2024

@author: AE & CL

=========================================
ftt_h_main.py
=========================================
Domestic Heat FTT module.
####################################

This is the main file for FTT: Heat, which models technological
diffusion of residential heating technologies due to simulated consumer decision making.
Consumers compare the **levelised cost of heat**, which leads to changes in the
market shares of different technologies.

The outputs of this module include changes in final energy demand and boiler sales.

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Element-wise divide which replaces divide-by-zeros with zeros

Functions included:
    - solve
        Main solution function for the module
    - get_lcoh
        Calculate levelised cost of residential heating

"""
# Third party imports
import numpy as np


# Local library imports
from ftt_source.ftt_core.ftt_shares import shares_change, shares_change_premature
from ftt_source.ftt_core.ftt_mandate import implement_mandate, implement_seeding
from ftt_source.ftt_core.ftt_sales_or_investments import get_sales, get_sales_yearly
from ftt_source.ftt_core.ftt_exogenous_sales import exogenous_sales
from ftt_source.ftt_core.ftt_exogenous_capacity import regulation_correction


# Green technology indices for Heat (heat pumps: ground source, air-water, air-air)
GREEN_INDICES_HP = [9, 10, 11]

from ftt_source.support.get_vars_to_copy import get_domain_vars_to_copy
from ftt_source.support.divide import divide
from ftt_source.support.check_market_shares import check_market_shares
from ftt_source.support.singleton_dimension import (
    restore_all_singleton_axes,
    squeeze_singleton_third_axis,
    squeeze_all_singleton_axes,
    unsqueeze_singleton_third_axis,
)

from ftt_source.Heat.ftt_h_lcoh import get_lcoh, set_carbon_tax


# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, titles, histend, year, domain):
    """
    Main solution function for the module.

    Add an extended description in the future.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution
    time_lag: type
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

    """

    # Categories for the cost matrix (BHTC)
    c4ti = {category: index for index, category in enumerate(titles['C4TI'])}

    sector = 'residential'
    num_regions = len(titles['RTI'])
    num_techs = len(titles['HTTI'])

    data['PRSC14'] = np.copy(time_lag['PRSC14'])
    if year == 2014:
        data['PRSC14'] = np.copy(data['PRSCX'])

    # Temporarily squeeze singleton 3rd axis for Heat-internal calculations.
    # Shapes are restored before returning from solve.
    shared_vars = {'HFFC', 'HJFC', 'RHUD', 'PRSC14', 'PRSCX', 'CO2taxH', 'noit'}
    squeeze_candidates = {var for var, dom in domain.items() if dom == 'FTT-H'} | shared_vars
    singleton_third_axis_candidates = squeeze_candidates - {'HEWW', 'noit'}
    squeezed_data_keys = squeeze_singleton_third_axis(
        data, singleton_third_axis_candidates
    )
    squeezed_time_lag_keys = squeeze_singleton_third_axis(
        time_lag, singleton_third_axis_candidates
    )
    global_variables = {
        'HEWA', 'HEWB', 'HEWW', 'HJET',
        'RHUD', 'PRSC14', 'CO2taxH', 'noit',
    }
    data_global_shapes = squeeze_all_singleton_axes(data, global_variables)
    time_lag_global_shapes = squeeze_all_singleton_axes(
        time_lag, global_variables
    )

    # Calculate the LCOH for each heating technology
    carbon_costs = set_carbon_tax(data, c4ti)
    data = get_lcoh(data, titles, carbon_costs)

    # =============================================================================
    # Historical initialisation
    # =============================================================================
    # Up to the last year of historical useful energy demand by technology
    # Historical data ends in 2020, so we need to initialise data when it is
    # 2021 to make sure the model runs.
    # The historical data contains final energy demand.
    if year <= histend['HEWF']:
        # Useful energy demand by technology
        # The historical data contains final energy demand
        data['HEWG'] = data['HEWF'] * data['BHTC'][:, :, c4ti["9 Conversion efficiency"]]

        for r in range(num_regions):
            # Total useful heat demand; this is the demand driver for FTT:Heat.
            if data['RHUD'][r] > 0.0:
                # Market shares (based on useful energy demand).
                data['HEWS'][r] = data['HEWG'][r] / data['RHUD'][r]

        # Correction to market shares: historical shares do not always add up
        # to 1.0.
        region_sums = data['HEWS'].sum(axis=1)
        needs_correction = (np.abs(region_sums - 1.0) > 1e-9) & (region_sums > 0.0)
        data['HEWS'][needs_correction] /= region_sums[needs_correction, None]

        # Normalise HEWG to RHUD.
        data['HEWG'] = data['HEWS'] * data['RHUD'][:, None]

        # Recalculate HEWF based on RHUD
        data['HEWF'] = data['HEWG'] / data['BHTC'][:, :, c4ti["9 Conversion efficiency"]]

        # Capacity by technology.
        # Capacity (GW): capacity factors are in MWh/kW (= GWh/MW), therefore /1000.
        data['HEWK'] = divide(
            data['HEWG'], data['BHTC'][:, :, c4ti["13 Capacity factor mean"]]
        ) / 1000

        # Emissions
        data['HEWE'] = data['HEWF'] * data['BHTC'][:, :, c4ti["15 Emission factor"]] / 1e6

        # Final energy demand by energy carrier.
        for r in range(num_regions):
            for fuel in range(len(titles['JTI'])):
                # Fuel use for heating.
                data['HJHF'][r, fuel] = np.sum(
                    data['HEWF'][r] * data['HJET'][:, fuel]
                )
                if data['HJFC'][r, fuel] > 0.0:
                    # Fuel use for the total residential sector. 0.0859 is the
                    # conversion factor from GWh to ktoe.
                    data['HJEF'][r, fuel] = data['HJHF'][r, fuel] / data['HJFC'][r, fuel] * 0.08598

        # Investment (capacity additions) by technology in GW/y.
        if year > 2014:
            data['HEWI'] = get_sales_yearly(
                data['HEWK'],
                time_lag['HEWK'],
                data['HEWI'],
                time_lag['BHTC'][:, :, c4ti['6 Replacetime']],
            )

            bi = np.zeros((num_regions, num_techs))
            for r in range(num_regions):
                bi[r] = np.matmul(data['HEWB'], data['HEWI'][r])
            dw = np.sum(bi, axis=0)
            data['HEWW'] = time_lag['HEWW'] + dw

    if year == histend['HEWF']:
        # Historical data ends in 2020, so initialise data when it is 2021.
        for r in range(num_regions):
            # Final energy demand by energy carrier.
            for fuel in range(len(titles['JTI'])):
                # Fuel use for heating
                data['HJHF'][r, fuel] = np.sum(
                    data['HEWF'][r] * data['HJET'][:, fuel]
                )
                # Fuel use for total residential sector #HFUX is missing
                if data['HJFC'][r, fuel] > 0.0:
                    data['HJEF'][r, fuel] = data['HJHF'][r, fuel] / data['HJFC'][r, fuel]

        carbon_costs = set_carbon_tax(data, c4ti)
        data = get_lcoh(data, titles, carbon_costs)

    data["FU14A"] = np.copy(data['HJHF'])
    data['FU14B'] = data["HJEF"] * data["HJFC"]

    # =============================================================================
    # Endogenous calculation
    # =============================================================================
    if year > histend['HEWF']:
        # Create a local dictionary for time-loop variables.
        data_dt = {}
        
        # First, fill the time loop variables with the their lagged equivalents
        vars_to_copy = get_domain_vars_to_copy(time_lag, domain, 'FTT-H')
        for var in vars_to_copy:
            data_dt[var] = np.copy(time_lag[var])
            
        # Preserve baseline fuel demand from previous timestep (for output tracking)
        data["FU14A"] = time_lag["FU14A"]
        data["FU14B"] = time_lag["FU14B"]

        relative_excess = divide(time_lag['HEWS'] - data['HREG'], data['HREG'])
        reg_constr = 0.5 + 0.5 * np.tanh(1.5 + 10 * relative_excess)

        reg_constr[data['HREG'] == 0.0] = 1.0
        reg_constr[data['HREG'] == -1.0] = 0.0
        
        # Factor used to create subannual data from annual figures
        no_it = int(data['noit'])
        dt = 1 / float(no_it)

        for t in range(1, no_it + 1):
            # Interpolate demand to prevent a staircase profile.
            demand_change = data['RHUD'] - time_lag['RHUD']
            rhudt = time_lag['RHUD'] + demand_change * t * dt
            rhudlt = time_lag['RHUD'] + demand_change * (t - 1) * dt
            
            # Get regions with non-zero heat demand.
            regions = np.flatnonzero(rhudt > 0.0)

            # ---------------------------------------------------------------------
            # Market-share dynamics
            # ---------------------------------------------------------------------
            # The core FTT equations account for old shares, costs and regulations.
            change_in_shares = shares_change(
                dt=dt,
                regions=regions,
                shares_dt=data_dt["HEWS"],          # Shares at previous t
                costs=data_dt["HGC1"],              # Costs
                costs_sd=data_dt["HWCD"],           # Standard deviations costs
                subst=data["HEWA"] * data["HETR"][:, :, np.newaxis],  # Substitution turnover rates
                reg_constr=reg_constr,              # Constraint due to regulation
                num_regions=num_regions,            # Number of regions
                num_techs=num_techs,                # Number of technologies
            )
            
            payback_times = data['BHTC'][:, :, c4ti["16 Payback time, mean"]]
            # Calculate scrappage rate for all regions
            SR_all = np.maximum(
                divide(np.ones_like(payback_times), payback_times) - data['HETR'],
                0.0,
            )
            
            # Premature replacements, use scrappage rate time scales and amended costs
            changes_in_shares_prem_repl = shares_change_premature(
                dt=dt,
                regions=regions,
                shares_dt=data_dt["HEWS"],          # Shares at previous t
                costs_marg=data_dt["HGC2"],         # Marginal costs (HGC2)
                costs_marg_sd=data_dt["HGD2"],      # SD Marginal costs (HGD2)
                costs_payb=data_dt["HGC3"],         # Payback costs (HGC3)
                costs_payb_sd=data_dt["HGD3"],      # SD Payback costs (HGD3)
                subst=data["HEWA"] * SR_all[:, :, np.newaxis],  # Substitution turnover rates
                reg_constr=reg_constr,              # Regulation constraint
                num_regions=num_regions,            # Number of regions
                num_techs=num_techs,                # Number of technologies
            )

            endo_shares = data_dt['HEWS'] + change_in_shares + changes_in_shares_prem_repl
            endo_gen = endo_shares * rhudt[:, None]

            # ---------------------------------------------------------------------
            # Regulatory policies
            # ---------------------------------------------------------------------
            # Change in generation from exogenous sales, capped at maximum sales.
            dgen_exog_sales = exogenous_sales(
                data['HWSA'][regions] * rhudt[regions, None],
                rhudt[regions],
                endo_gen[regions],
                data['HREG'][regions] * rhudt[regions, None],
                no_it,
                data['BHTC'][regions, :, c4ti['5 Lifetime']]
            )

            dgen_reg_corr = regulation_correction(
                endo_gen[regions],
                endo_shares[regions],
                rhudlt[regions],
                reg_constr[regions],
            )

            # Correct for regulation when demand is growing; main effect in
            # shares equation.
            # New generation and shares.
            new_generation = endo_gen[regions] + dgen_exog_sales + dgen_reg_corr
            total_generation = np.sum(new_generation, axis=1)
            data['HEWS'][regions] = divide(new_generation, total_generation[:, None])

            check_market_shares(data['HEWS'], titles, sector, year)

            # ---------------------------------------------------------------------
            # Update variables
            # ---------------------------------------------------------------------
            
            data['HEWG'] = data['HEWS'] * rhudt[:, None]
            # Capacity (GW) (capacity factors: (MWh/kW=GWh/MW, therefore /1000)
            data['HEWK'] = divide(
                data['HEWG'], data['BHTC'][:, :, c4ti["13 Capacity factor mean"]]
            ) / 1000
            
            # New additions (HEWI)
            data['HEWI'], hewi_t = get_sales(
                data['HEWK'],
                data_dt['HEWK'],
                time_lag['HEWK'],
                data['HEWI'],
                data_dt['BHTC'][:, :, c4ti['6 Replacetime']],
                dt,
            )
            # Seed heat pumps in regions with low adoption (first five years
            # of the simulation).
            data['HEWI'], hewi_t, data['HEWK'] = implement_seeding(
                data['HEWK'],
                data['HEWI'],
                hewi_t,
                year,
                GREEN_INDICES_HP,
                histend['HEWF'],
            )

            # Change capacity and sales after mandate (only runs if the heat
            # pump mandate is not zero).
            data['HEWI'], hewi_t, data['HEWK'] = implement_mandate(
                data['HEWK'],
                data['HEWI'],
                hewi_t,
                year,
                GREEN_INDICES_HP,
                data["HP mandate"],
            )

            # Calculate HEWG, HEWS and HEWF after mandates.
            # Useful heat by technology.
            data['HEWG'] = data['HEWK'] * data['BHTC'][:, :, c4ti["13 Capacity factor mean"]] * 1000
            data['HEWS'] = data['HEWG'] / np.sum(data['HEWG'], axis=1)[:, None]
            # Final energy by technology.
            data['HEWF'] = divide(
                data['HEWG'], data['BHTC'][:, :, c4ti["9 Conversion efficiency"]]
            )
            # Emissions.
            data['HEWE'] = data['HEWF'] * data['BHTC'][:, :, c4ti["15 Emission factor"]] / 1e6

            data['HEWP'][:, 0] = data['HFFC'][:, 4]   # Oil boiler -> heavy fuel oil
            data['HEWP'][:, 1] = data['HFFC'][:, 4]   # Oil technology -> heavy fuel oil
            data['HEWP'][:, 2] = data['HFFC'][:, 6]   # Gas boiler -> other gas
            data['HEWP'][:, 3] = data['HFFC'][:, 6]    # Gas condensing -> other gas
            data['HEWP'][:, 4] = data['HFFC'][:, 10] # Wood stove -> biomass
            data['HEWP'][:, 5] = data['HFFC'][:, 10]   # Wood boiler -> biomass
            data['HEWP'][:, 6] = data['HFFC'][:, 0]  # Coal -> hard coal
            data['HEWP'][:, 7] = data['HFFC'][:, 8]   # District heat -> electricity (proxy)
            data['HEWP'][:, 8] = data['HFFC'][:, 8]   # Electric heating -> electricity
            data['HEWP'][:, 9] = data['HFFC'][:, 8]   # Heat pump ground -> electricity
            data['HEWP'][:, 10] = data['HFFC'][:, 8]  # Heat pump air-water -> electricity
            data['HEWP'][:, 11] = data['HFFC'][:, 8]  # Heat pump air-air -> electricity

            # Map fuel prices to heating technologies.
            # HFFC indices: 0=hard coal, 4=heavy fuel oil, 6=other gas,
            # 7=natural gas, 8=electricity, 10=combustible waste/biomass.
            # Final energy demand for heating purposes.
            data['HJHF'] = np.matmul(data['HEWF'], data['HJET'])

            # ---------------------------------------------------------------------
            # Learning-by-doing
            # ---------------------------------------------------------------------
            # Cumulative global learning.
            # Using the technological spill-over matrix together with capacity
            # additions, estimate total global spill-over of similar technologies.
            bi = np.zeros((num_regions, num_techs))
            for r in range(num_regions):
                bi[r] = np.matmul(data['HEWB'], hewi_t[r])
            dw = np.sum(bi, axis=0)

            
            data['HEWW'] = data_dt['HEWW'] + dw
            data['BHTC'] = np.copy(data_dt['BHTC'])
            
            # Learning-by-doing effects on investment and efficiency
            for b in range(num_techs):
                if data['HEWW'][b] > 0.0001:
                    data['BHTC'][:, b, c4ti['1 Inv cost mean (EUR/kW)']] = (
                        data_dt['BHTC'][:, b, c4ti['1 Inv cost mean (EUR/kW)']]
                        * (1.0 + data['BHTC'][:, b, c4ti['7 Investment LR']] * dw[b] / data['HEWW'][b]))
                    data['BHTC'][:, b, c4ti['2 Inv Cost SD']] = (
                        data_dt['BHTC'][:, b, c4ti['2 Inv Cost SD']]
                        * (1.0 + data['BHTC'][:, b, c4ti['7 Investment LR']] * dw[b] / data['HEWW'][b]))
                    data['BHTC'][:, b, c4ti['9 Conversion efficiency']] = (
                        data_dt['BHTC'][:, b, c4ti['9 Conversion efficiency']]
                        * 1.0 / (1.0 + data['BHTC'][:, b, c4ti['20 Efficiency LR']] * dw[b] / data['HEWW'][b]))

            data["HWIC"] = data["BHTC"][:, :, c4ti['1 Inv cost mean (EUR/kW)']]
            data["HEFF"] = data["BHTC"][:, :, c4ti['9 Conversion efficiency']]
            
            # Calculate levelised cost again
            carbon_costs = set_carbon_tax(data, c4ti)
            data = get_lcoh(data, titles, carbon_costs)
            
            # Store heat variables that have changed in data_dt
            for var in vars_to_copy:
                data_dt[var] = np.copy(data[var])

        price_levels = data['PRSC14']
        data['HWIY'] = (
            data['HEWI'] * data['BHTC'][:, :, c4ti['1 Inv cost mean (EUR/kW)']]
            / price_levels[:, None]
        )

        if year == 2050 and t == no_it:
            print(f"Total heat pumps in 2050 is: {np.sum(data['HEWG'][:, 9:12]) / 10**6:.3f} M GWh")

    # Restore singleton 3rd axis for compatibility with the rest of the model.
    restore_all_singleton_axes(data, data_global_shapes)
    restore_all_singleton_axes(time_lag, time_lag_global_shapes)
    unsqueeze_singleton_third_axis(data, squeezed_data_keys)
    unsqueeze_singleton_third_axis(time_lag, squeezed_time_lag_keys)

    return data
