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
    shared_vars = {'HFFC', 'HJFC', 'RHUD', 'PRSC14', 'PRSCX', 'noit'}
    squeeze_candidates = {var for var, dom in domain.items() if dom == 'FTT-H'} | shared_vars
    singleton_third_axis_candidates = squeeze_candidates - {'HEWW', 'noit'}
    squeezed_data_keys = squeeze_singleton_third_axis(
        data, singleton_third_axis_candidates
    )
    squeezed_time_lag_keys = squeeze_singleton_third_axis(
        time_lag, singleton_third_axis_candidates
    )
    data_global_shapes = squeeze_all_singleton_axes(data, {'HEWW', 'noit'})
    time_lag_global_shapes = squeeze_all_singleton_axes(
        time_lag, {'HEWW', 'noit'}
    )

    # Calculate the LCOH for each heating technology.
    carbon_costs = set_carbon_tax(data, c4ti)
    data = get_lcoh(data, titles, carbon_costs)

    # =============================================================================
    # Historical initialisation
    # =============================================================================
    # Historical data provides final energy demand. Convert it to useful heat,
    # then derive the corresponding technology shares and capacity.
    if year <= histend['HEWF']:
        data['HEWG'] = data['HEWF'] * data['BHTC'][:, :, c4ti["9 Conversion efficiency"]]

        for r in range(num_regions):
            if data['RHUD'][r, 0] > 0.0:
                data['HEWS'][r] = data['HEWG'][r] / data['RHUD'][r, 0]

        # Historical market shares occasionally do not add up to one.
        region_sums = data['HEWS'].sum(axis=1)
        needs_correction = (np.abs(region_sums - 1.0) > 1e-9) & (region_sums > 0.0)
        data['HEWS'][needs_correction] /= region_sums[needs_correction, None]

        # Align useful heat demand with the regional demand driver.
        data['HEWG'] = data['HEWS'] * data['RHUD']

        # Recalculate HEWF based on RHUD
        data['HEWF'] = data['HEWG'] / data['BHTC'][:, :, c4ti["9 Conversion efficiency"]]

        # Capacity by boiler (GWh/MW conversion).
        data['HEWK'] = divide(
            data['HEWG'], data['BHTC'][:, :, c4ti["13 Capacity factor mean"]]
        ) / 1000

        # Emissions
        data['HEWE'] = data['HEWF'] * data['BHTC'][:, :, c4ti["15 Emission factor"]] / 1e6

        for r in range(num_regions):
            for fuel in range(len(titles['JTI'])):
                data['HJHF'][r, fuel] = np.sum(data['HEWF'][r] * data['HJET'][0, :, fuel])
                if data['HJFC'][r, fuel] > 0.0:
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
                bi[r] = np.matmul(data['HEWB'][0], data['HEWI'][r])
            dw = np.sum(bi, axis=0)
            data['HEWW'] = time_lag['HEWW'] + dw

    if year == histend['HEWF']:
        for r in range(num_regions):
            for fuel in range(len(titles['JTI'])):
                data['HJHF'][r, fuel] = np.sum(data['HEWF'][r] * data['HJET'][0, :, fuel])
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
        vars_to_copy = get_domain_vars_to_copy(time_lag, domain, 'FTT-H')
        for var in vars_to_copy:
            data_dt[var] = np.copy(time_lag[var])

        data["FU14A"] = time_lag["FU14A"]
        data["FU14B"] = time_lag["FU14B"]

        relative_excess = divide(time_lag['HEWS'] - data['HREG'], data['HREG'])
        reg_constr = 0.5 + 0.5 * np.tanh(1.5 + 10 * relative_excess)
        reg_constr[data['HREG'] == 0.0] = 1.0
        reg_constr[data['HREG'] == -1.0] = 0.0

        no_it = int(data['noit'])
        dt = 1 / float(no_it)

        for t in range(1, no_it + 1):
            # Interpolate demand to prevent a staircase profile.
            demand_change = data['RHUD'] - time_lag['RHUD']
            rhudt = time_lag['RHUD'] + demand_change * t * dt
            rhudlt = time_lag['RHUD'] + demand_change * (t - 1) * dt
            regions = np.flatnonzero(rhudt[:, 0] > 0.0)

            # ---------------------------------------------------------------------
            # Market-share dynamics
            # ---------------------------------------------------------------------
            change_in_shares = shares_change(
                dt=dt,
                regions=regions,
                shares_dt=data_dt["HEWS"],
                costs=data_dt["HGC1"],
                costs_sd=data_dt["HWCD"],
                subst=data["HEWA"] * data["HETR"][:, :, np.newaxis],
                reg_constr=reg_constr,
                num_regions=num_regions,
                num_techs=num_techs,
            )

            payback_times = data['BHTC'][:, :, c4ti["16 Payback time, mean"]]
            SR_all = np.maximum(
                divide(np.ones_like(payback_times), payback_times) - data['HETR'],
                0.0,
            )

            changes_in_shares_prem_repl = shares_change_premature(
                dt=dt,
                regions=regions,
                shares_dt=data_dt["HEWS"],
                costs_marg=data_dt["HGC2"],
                costs_marg_sd=data_dt["HGD2"],
                costs_payb=data_dt["HGC3"],
                costs_payb_sd=data_dt["HGD3"],
                subst=data["HEWA"] * SR_all[:, :, np.newaxis],
                reg_constr=reg_constr,
                num_regions=num_regions,
                num_techs=num_techs,
            )

            endo_shares = data_dt['HEWS'] + change_in_shares + changes_in_shares_prem_repl
            endo_gen = endo_shares * rhudt

            # ---------------------------------------------------------------------
            # Regulatory policies
            # ---------------------------------------------------------------------
            dgen_exog_sales = exogenous_sales(
                data['HWSA'][regions] * rhudt[regions],
                rhudt[regions, 0],
                endo_gen[regions],
                data['HREG'][regions] * rhudt[regions],
                no_it,
                data['BHTC'][regions, :, c4ti['5 Lifetime']]
            )

            dgen_reg_corr = regulation_correction(
                endo_gen[regions],
                endo_shares[regions],
                rhudlt[regions],
                reg_constr[regions],
            )

            new_generation = endo_gen[regions] + dgen_exog_sales + dgen_reg_corr
            total_generation = np.sum(new_generation, axis=1)
            data['HEWS'][regions] = divide(new_generation, total_generation[:, None])

            check_market_shares(data['HEWS'], titles, sector, year)

            # ---------------------------------------------------------------------
            # Update variables
            # ---------------------------------------------------------------------
            data['HEWG'] = data['HEWS'] * rhudt
            data['HEWK'] = divide(
                data['HEWG'], data['BHTC'][:, :, c4ti["13 Capacity factor mean"]]
            ) / 1000

            data['HEWI'], hewi_t = get_sales(
                data['HEWK'],
                data_dt['HEWK'],
                time_lag['HEWK'],
                data['HEWI'],
                data_dt['BHTC'][:, :, c4ti['6 Replacetime']],
                dt,
            )
            data['HEWI'], hewi_t, data['HEWK'] = implement_seeding(
                data['HEWK'],
                data['HEWI'],
                hewi_t,
                year,
                GREEN_INDICES_HP,
                histend['HEWF'],
            )

            data['HEWI'], hewi_t, data['HEWK'] = implement_mandate(
                data['HEWK'],
                data['HEWI'],
                hewi_t,
                year,
                GREEN_INDICES_HP,
                data["HP mandate"],
            )

            # Recalculate energy, shares, fuel demand, and emissions after policy.
            data['HEWG'] = data['HEWK'] * data['BHTC'][:, :, c4ti["13 Capacity factor mean"]] * 1000
            data['HEWS'] = data['HEWG'] / np.sum(data['HEWG'], axis=1)[:, None]
            data['HEWF'] = divide(
                data['HEWG'], data['BHTC'][:, :, c4ti["9 Conversion efficiency"]]
            )
            data['HEWE'] = data['HEWF'] * data['BHTC'][:, :, c4ti["15 Emission factor"]] / 1e6

            data['HEWP'][:, 0] = data['HFFC'][:, 4]
            data['HEWP'][:, 1] = data['HFFC'][:, 4]
            data['HEWP'][:, 2] = data['HFFC'][:, 6]
            data['HEWP'][:, 3] = data['HFFC'][:, 6]
            data['HEWP'][:, 4] = data['HFFC'][:, 10]
            data['HEWP'][:, 5] = data['HFFC'][:, 10]
            data['HEWP'][:, 6] = data['HFFC'][:, 0]
            data['HEWP'][:, 7] = data['HFFC'][:, 8]
            data['HEWP'][:, 8] = data['HFFC'][:, 8]
            data['HEWP'][:, 9] = data['HFFC'][:, 8]
            data['HEWP'][:, 10] = data['HFFC'][:, 8]
            data['HEWP'][:, 11] = data['HFFC'][:, 8]

            # Map technology fuel demand to energy carriers.
            data['HJHF'] = np.matmul(data['HEWF'], data['HJET'][0])

            bi = np.zeros((num_regions, num_techs))
            for r in range(num_regions):
                bi[r] = np.matmul(data['HEWB'][0], hewi_t[r])
            dw = np.sum(bi, axis=0)

            # ---------------------------------------------------------------------
            # Learning-by-doing
            # ---------------------------------------------------------------------
            data['HEWW'] = data_dt['HEWW'] + dw
            data['BHTC'] = np.copy(data_dt['BHTC'])

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

            carbon_costs = set_carbon_tax(data, c4ti)
            data = get_lcoh(data, titles, carbon_costs)

            for var in vars_to_copy:
                data_dt[var] = np.copy(data[var])

        price_levels = data['PRSC14'][:, 0]
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
