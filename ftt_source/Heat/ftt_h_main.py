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
    squeezed_keys = []
    for var in squeeze_candidates:
        if var in data and hasattr(data[var], 'ndim') and data[var].ndim == 3 and data[var].shape[2] == 1:
            data[var] = data[var][:, :, 0]
            squeezed_keys.append(var)

    # Calculate the LCOH for each heating technology.
    carbon_costs = set_carbon_tax(data, c4ti)
    data = get_lcoh(data, titles, carbon_costs)

    # Up to the last year of historical useful energy demand by boiler
    if year <= histend['HEWF']:
        data['HEWG'][:, :] = data['HEWF'][:, :] * data['BHTC'][:, :, c4ti["9 Conversion efficiency"]]

        for r in range(num_regions):
            if data['RHUD'][r, 0] > 0.0:
                data['HEWS'][r, :] = data['HEWG'][r, :] / data['RHUD'][r, 0]

        # CORRECTION TO MARKET SHARES
        region_sums = data['HEWS'].sum(axis=1)
        needs_correction = (np.abs(region_sums - 1.0) > 1e-9) & (region_sums > 0.0)
        data['HEWS'][needs_correction, :] /= region_sums[needs_correction, np.newaxis]

        # Normalise HEWG to RHUD
        data['HEWG'][:, :] = data['HEWS'][:, :] * data['RHUD'][:, :]

        # Recalculate HEWF based on RHUD
        data['HEWF'][:, :] = data['HEWG'][:, :] / data['BHTC'][:, :, c4ti["9 Conversion efficiency"]]

        # Capacity by boiler
        data['HEWK'][:, :] = divide(data['HEWG'][:, :],
                                    data['BHTC'][:, :, c4ti["13 Capacity factor mean"]]) / 1000

        # Emissions
        data['HEWE'][:, :] = data['HEWF'][:, :] * data['BHTC'][:, :, c4ti["15 Emission factor"]] / 1e6

        for r in range(num_regions):
            for fuel in range(len(titles['JTI'])):
                data['HJHF'][r, fuel] = np.sum(data['HEWF'][r, :] * data['HJET'][0, :, fuel])
                if data['HJFC'][r, fuel] > 0.0:
                    data['HJEF'][r, fuel] = data['HJHF'][r, fuel] / data['HJFC'][r, fuel] * 0.08598

        # Investment (= capacity additions) by technology (in GW/y)
        if year > 2014:
            hewi_3d = get_sales_yearly(
                data['HEWK'][:, :, np.newaxis],
                time_lag['HEWK'],
                data['HEWI'][:, :, np.newaxis],
                time_lag['BHTC'][:, :, c4ti['6 Replacetime']]
            )
            data['HEWI'][:, :] = hewi_3d[:, :, 0]

            bi = np.zeros((num_regions, num_techs))
            for r in range(num_regions):
                bi[r, :] = np.matmul(data['HEWB'][0, :, :], data['HEWI'][r, :])
            dw = np.sum(bi, axis=0)
            data['HEWW'][0, :] = time_lag['HEWW'][0, :, 0] + dw

    if year == histend['HEWF']:
        for r in range(num_regions):
            for fuel in range(len(titles['JTI'])):
                data['HJHF'][r, fuel] = np.sum(data['HEWF'][r, :] * data['HJET'][0, :, fuel])
                if data['HJFC'][r, fuel] > 0.0:
                    data['HJEF'][r, fuel] = data['HJHF'][r, fuel] / data['HJFC'][r, fuel]

        carbon_costs = set_carbon_tax(data, c4ti)
        data = get_lcoh(data, titles, carbon_costs)

    data["FU14A"] = np.copy(data['HJHF'])
    data['FU14B'] = data["HJEF"] * data["HJFC"]

    # Endogenous calculation takes over from here
    if year > histend['HEWF']:
        data_dt = {}
        vars_to_copy = get_domain_vars_to_copy(time_lag, domain, 'FTT-H')
        for var in vars_to_copy:
            if var in squeezed_keys and hasattr(time_lag[var], 'ndim') and time_lag[var].ndim == 3 and time_lag[var].shape[2] == 1:
                data_dt[var] = np.copy(time_lag[var][:, :, 0])
            else:
                data_dt[var] = np.copy(time_lag[var])

        data["FU14A"] = time_lag["FU14A"][:, :, 0]
        data["FU14B"] = time_lag["FU14B"][:, :, 0]

        relative_excess = divide((time_lag['HEWS'][:, :, 0] - data['HREG'][:, :]), data['HREG'][:, :])
        reg_constr = 0.5 + 0.5 * np.tanh(1.5 + 10 * relative_excess)
        reg_constr[data['HREG'][:, :] == 0.0] = 1.0
        reg_constr[data['HREG'][:, :] == -1.0] = 0.0

        no_it = int(data['noit'][0, 0])
        dt = 1 / float(no_it)

        for t in range(1, no_it + 1):
            rhudt = time_lag['RHUD'][:, :, :] + (data['RHUD'][:, :, np.newaxis] - time_lag['RHUD'][:, :, :]) * t * dt
            rhudlt = time_lag['RHUD'][:, :, :] + (data['RHUD'][:, :, np.newaxis] - time_lag['RHUD'][:, :, :]) * (t - 1) * dt

            regions = np.where(rhudt[:, 0, 0] > 0.0)[0]

            change_in_shares = shares_change(
                dt=dt,
                regions=regions,
                shares_dt=data_dt["HEWS"][:, :, np.newaxis],
                costs=data_dt["HGC1"][:, :, np.newaxis],
                costs_sd=data_dt["HWCD"][:, :, np.newaxis],
                subst=data["HEWA"] * data["HETR"][:, :, np.newaxis],
                reg_constr=reg_constr,
                num_regions=num_regions,
                num_techs=num_techs,
            )

            SR_all = np.zeros((num_regions, num_techs))
            for r in range(num_regions):
                SR = divide(np.ones(num_techs),
                            data['BHTC'][r, :, c4ti["16 Payback time, mean"]]) - data['HETR'][r, :]
                SR_all[r, :] = np.where(SR < 0.0, 0.0, SR)

            changes_in_shares_prem_repl = shares_change_premature(
                dt=dt,
                regions=regions,
                shares_dt=data_dt["HEWS"][:, :, np.newaxis],
                costs_marg=data_dt["HGC2"][:, :, np.newaxis],
                costs_marg_sd=data_dt["HGD2"][:, :, np.newaxis],
                costs_payb=data_dt["HGC3"][:, :, np.newaxis],
                costs_payb_sd=data_dt["HGD3"][:, :, np.newaxis],
                subst=data["HEWA"] * SR_all[:, :, np.newaxis],
                reg_constr=reg_constr,
                num_regions=num_regions,
                num_techs=num_techs,
            )

            endo_shares = data_dt['HEWS'] + change_in_shares + changes_in_shares_prem_repl
            endo_gen = endo_shares * rhudt[:, 0, 0][:, np.newaxis]

            dgen_exog_sales = exogenous_sales(
                data['HWSA'][regions, :] * rhudt[regions, 0, 0][:, None],
                rhudt[regions, 0, 0],
                endo_gen[regions],
                data['HREG'][regions, :] * rhudt[regions, 0, 0][:, None],
                no_it,
                data['BHTC'][regions, :, c4ti['5 Lifetime']]
            )

            dgen_reg_corr = regulation_correction(
                endo_gen[regions], endo_shares[regions], rhudlt[regions, 0], reg_constr[regions])

            new_generation = endo_gen[regions] + dgen_exog_sales + dgen_reg_corr
            total_generation = np.sum(new_generation, axis=1)
            data['HEWS'][regions, :] = divide(new_generation, total_generation[:, None])

            check_market_shares(data['HEWS'][:, :, np.newaxis], titles, sector, year)

            data['HEWG'][:, :] = data['HEWS'][:, :] * rhudt[:, 0, 0][:, np.newaxis]
            data['HEWK'][:, :] = divide(data['HEWG'][:, :],
                                        data['BHTC'][:, :, c4ti["13 Capacity factor mean"]]) / 1000

            hewi_3d, hewi_t = get_sales(
                data['HEWK'][:, :, np.newaxis],
                data_dt['HEWK'][:, :, np.newaxis],
                time_lag['HEWK'],
                data['HEWI'][:, :, np.newaxis],
                data_dt['BHTC'][:, :, c4ti['6 Replacetime']],
                dt
            )
            data['HEWI'][:, :] = hewi_3d[:, :, 0]

            hewi_3d, hewi_t, hewk_3d = implement_seeding(
                data['HEWK'][:, :, np.newaxis],
                data['HEWI'][:, :, np.newaxis],
                hewi_t,
                year,
                GREEN_INDICES_HP,
                histend['HEWF']
            )
            data['HEWI'][:, :] = hewi_3d[:, :, 0]
            data['HEWK'][:, :] = hewk_3d[:, :, 0]

            hewi_3d, hewi_t, hewk_3d = implement_mandate(
                data['HEWK'][:, :, np.newaxis],
                data['HEWI'][:, :, np.newaxis],
                hewi_t,
                year,
                GREEN_INDICES_HP,
                data["HP mandate"]
            )
            data['HEWI'][:, :] = hewi_3d[:, :, 0]
            data['HEWK'][:, :] = hewk_3d[:, :, 0]

            data['HEWG'][:, :] = data['HEWK'][:, :] * data['BHTC'][:, :, c4ti["13 Capacity factor mean"]] * 1000
            data['HEWS'][:, :] = data['HEWG'][:, :] / np.sum(data['HEWG'][:, :], axis=1)[:, None]
            data['HEWF'][:, :] = divide(data['HEWG'][:, :],
                                        data['BHTC'][:, :, c4ti["9 Conversion efficiency"]])
            data['HEWE'][:, :] = data['HEWF'][:, :] * data['BHTC'][:, :, c4ti["15 Emission factor"]] / 1e6

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

            data['HJHF'][:, :] = np.matmul(data['HEWF'][:, :], data['HJET'][0, :, :])

            bi = np.zeros((num_regions, num_techs))
            for r in range(num_regions):
                bi[r, :] = np.matmul(data['HEWB'][0, :, :], hewi_t[r, :, 0])
            dw = np.sum(bi, axis=0)

            data['HEWW'][0, :] = data_dt['HEWW'][0, :] + dw
            data['BHTC'] = np.copy(data_dt['BHTC'])

            for b in range(num_techs):
                if data['HEWW'][0, b] > 0.0001:
                    data['BHTC'][:, b, c4ti['1 Inv cost mean (EUR/kW)']] = (
                        data_dt['BHTC'][:, b, c4ti['1 Inv cost mean (EUR/kW)']]
                        * (1.0 + data['BHTC'][:, b, c4ti['7 Investment LR']] * dw[b] / data['HEWW'][0, b]))
                    data['BHTC'][:, b, c4ti['2 Inv Cost SD']] = (
                        data_dt['BHTC'][:, b, c4ti['2 Inv Cost SD']]
                        * (1.0 + data['BHTC'][:, b, c4ti['7 Investment LR']] * dw[b] / data['HEWW'][0, b]))
                    data['BHTC'][:, b, c4ti['9 Conversion efficiency']] = (
                        data_dt['BHTC'][:, b, c4ti['9 Conversion efficiency']]
                        * 1.0 / (1.0 + data['BHTC'][:, b, c4ti['20 Efficiency LR']] * dw[b] / data['HEWW'][0, b]))

            data["HWIC"][:, :] = data["BHTC"][:, :, c4ti['1 Inv cost mean (EUR/kW)']]
            data["HEFF"][:, :] = data["BHTC"][:, :, c4ti['9 Conversion efficiency']]

            carbon_costs = set_carbon_tax(data, c4ti)
            data = get_lcoh(data, titles, carbon_costs)

            for var in vars_to_copy:
                if var in squeezed_keys:
                    data_dt[var] = np.copy(data[var])
                else:
                    data_dt[var] = np.copy(data[var])

        data['HWIY'][:, :] = (data['HEWI'][:, :] * data['BHTC'][:, :, c4ti['1 Inv cost mean (EUR/kW)']]
                              / data['PRSC14'][:, 0][:, np.newaxis])

        if year == 2050 and t == no_it:
            print(f"Total heat pumps in 2050 is: {np.sum(data['HEWG'][:, 9:12]) / 10**6:.3f} M GWh")

    # Restore singleton 3rd axis for compatibility with the rest of the model.
    for var in squeezed_keys:
        if hasattr(data[var], 'ndim') and data[var].ndim == 2:
            data[var] = data[var][:, :, np.newaxis]

    return data
