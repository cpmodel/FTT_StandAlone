# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_main.py
=========================================
Power generation FTT module.



This is the main file for the power module, FTT: Power. The power
module models technological replacement of electricity generation technologies due
to simulated investor decision making. Investors compare the **levelised cost of
electricity**, which leads to changes in the market shares of different technologies.

After market shares are determined, the rldc function is called, which calculates
**residual load duration curves**. This function estimates how much power needs to be
supplied by flexible or baseload technologies to meet electricity demand at all times.
This function also returns load band heights, curtailment, and storage information,
including storage costs and marginal costs for wind and solar.

FTT: Power also includes **dispatchers decisions**; dispatchers decide when different technologies
supply the power grid. Investor decisions and dispatcher decisions are matched up, which is an
example of a stable marriage problem.

Costs in the model change due to endogenous learning curves, costs for electricity
storage, as well as increasing marginal costs of resources calculated using cost-supply
curves. **Cost-supply curves** are recalculated at the end of the routine.

Local library imports:

    FTT: Core functions:
    - `get_sales <get_sales_or_investment.htlm>
        Generic investment function (new plus end-of-life replacement)
        
    FTT: Power functions:

    - `rldc <ftt_p_rldc.html>`__
        Residual load duration curves
    - `dspch <ftt_p_dspch.html>`__
        Dispatch of capcity
    - `get_lcoe <ftt_p_lcoe.html>`__
        Levelised cost calculation
    - `survival_function <ftt_p_surv.html>`__
        Calculation of scrappage, sales, tracking of age, and average efficiency.
    - `shares <ftt_p_shares.html>`__
        Market shares simulation (core of the model)
    - `cost_curves <ftt_p_costc.html>`__
        Calculates increasing marginal costs of resources

    Support functions:

    - `divide <divide.html>`__
        Element-wise divide which replaces divide-by-zeros with zeros

Functions included:
    - solve
        Main solution function for the module
"""

# Third party imports
import csv
import numpy as np

# Local library imports
from SourceCode.support.divide import divide
from SourceCode.ftt_core.ftt_sales_or_investments import get_sales, get_sales_yearly
from SourceCode.ftt_core.ftt_shares import shares_change


from SourceCode.Power.ftt_p_rldc import rldc
from SourceCode.Power.ftt_p_dspch import dspch
from SourceCode.Power.ftt_p_lcoe import get_lcoe, set_carbon_tax
from SourceCode.Power.ftt_p_surv import survival_function
from SourceCode.Power.ftt_p_shares import shares
from SourceCode.Power.ftt_p_costc import cost_curves
from SourceCode.Power.ftt_p_mewp import get_marginal_fuel_prices_mewp

from SourceCode.Power.ftt_p_phase_out import set_linear_coal_phase_out

from SourceCode.sector_coupling.transport_batteries_to_power import second_hand_batteries
from SourceCode.sector_coupling.battery_lbd import quarterly_bat_add_power

from SourceCode.Power.io_debug_export import export_io_summary
from SourceCode.Power.io_debug_export import export_rooftop_trace


_SMALLSCALE_GAMMA = None
_SMALLSCALE_MAX_ROOFTOP_SHARE = None


def _simulate_household_share(initial_share, roof_cost, grid_cost, roof_std,
                              grid_std, subst_row, dt, no_it,
                              max_rooftop_share=1.0):
    """Run the two-option household share equation for one region."""
    shares_local = np.zeros((1, 2, 1))
    shares_local[0, :, 0] = initial_share
    costs_local = np.zeros((1, 2, 1))
    costs_local[0, :, 0] = [roof_cost, grid_cost]
    std_local = np.zeros((1, 2, 1))
    std_local[0, :, 0] = [roof_std, grid_std]
    subst_local = np.zeros((1, 2, 2))
    subst_local[0, :, :] = subst_row
    upper_limit = np.ones((1, 2, 1))
    lower_limit = np.zeros((1, 2, 1))
    upper_limit[0, 0, 0] = max_rooftop_share

    for _ in range(no_it):
        change = shares_change(
            dt=dt,
            regions=np.array([0]),
            shares_dt=shares_local,
            costs=costs_local,
            costs_sd=std_local,
            subst=subst_local,
            reg_constr=np.zeros((1, 2)),
            num_regions=1,
            num_techs=2,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            limits_active=True
        )
        shares_local[:, :, 0] = np.clip(shares_local[:, :, 0] + change, 0.0, 1.0)
        shares_local[0, 0, 0] = min(shares_local[0, 0, 0], max_rooftop_share)
        shares_local[0, 1, 0] = 1.0 - shares_local[0, 0, 0]
        row_sum = shares_local[0, :, 0].sum()
        if row_sum > 0.0:
            shares_local[0, :, 0] /= row_sum

    return shares_local[0, 0, 0]


def _calibrate_rooftop_gamma(initial_shares, target_rooftop_share, costs,
                             costs_std, subst, dt, no_it, valid_target,
                             max_rooftop_share):
    """Calibrate rooftop perceived-cost offsets to match the first observed step."""
    gamma = np.zeros(initial_shares.shape[0])

    for r in np.where(valid_target)[0]:
        initial_share = initial_shares[r, :, 0]
        if initial_share.sum() <= 0.0 or not np.all(np.isfinite(initial_share)):
            continue

        target = float(np.clip(target_rooftop_share[r], 0.0, max_rooftop_share[r]))
        low, high = -1.0e5, 1.0e5
        low_share = _simulate_household_share(
            initial_share, costs[r, 0, 0] + low, costs[r, 1, 0],
            costs_std[r, 0, 0], costs_std[r, 1, 0], subst[r], dt, no_it,
            max_rooftop_share[r])
        high_share = _simulate_household_share(
            initial_share, costs[r, 0, 0] + high, costs[r, 1, 0],
            costs_std[r, 0, 0], costs_std[r, 1, 0], subst[r], dt, no_it,
            max_rooftop_share[r])

        if not (high_share <= target <= low_share):
            continue

        for _ in range(60):
            mid = 0.5 * (low + high)
            mid_share = _simulate_household_share(
                initial_share, costs[r, 0, 0] + mid, costs[r, 1, 0],
                costs_std[r, 0, 0], costs_std[r, 1, 0], subst[r], dt, no_it,
                max_rooftop_share[r])
            if mid_share > target:
                low = mid
            else:
                high = mid

        gamma[r] = 0.5 * (low + high)

    return gamma


def _estimate_max_rooftop_share(initial_rooftop_share, observed_rooftop_share,
                                valid_target):
    """Estimate a country-specific rooftop ceiling until roof-space data exists."""
    reference_share = np.maximum(initial_rooftop_share, observed_rooftop_share * valid_target)
    max_share = 0.25 + 1.2 * reference_share
    return np.clip(max_share, 0.15, 0.85)


def _load_max_rooftop_share(titles, initial_rooftop_share, observed_rooftop_share,
                            valid_target):
    """Load country rooftop ceilings, with observed-data consistency checks."""
    fallback = _estimate_max_rooftop_share(
        initial_rooftop_share, observed_rooftop_share, valid_target)
    ceiling = fallback.copy()
    path = "Inputs/_Assumptions/RSMX.csv"

    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            values = {
                row["region"].strip(): float(row["max_rooftop_share"])
                for row in reader
                if row.get("region") and row.get("max_rooftop_share")
            }
    except FileNotFoundError:
        print(f"{path} not found; using inferred rooftop ceilings.")
        values = {}

    for r, region_code in enumerate(titles["RTI_short"]):
        if region_code in values:
            ceiling[r] = values[region_code]

    min_required = np.maximum(initial_rooftop_share, observed_rooftop_share * valid_target)
    ceiling = np.maximum(ceiling, np.minimum(min_required + 0.05, 0.95))
    return np.clip(ceiling, 0.05, 0.95)




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
        Model variables for the current year
    time_lag: type
        Model variables in previous year
    iter_lag: type
        Description
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of historical data by variable
    year: int
        Current/active year of solution
    Domain: dictionary of lists
        Pairs variables to domains

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    Notes
    ---------
        survival_function is currently unused.
    """

    global _SMALLSCALE_GAMMA, _SMALLSCALE_MAX_ROOFTOP_SHARE
    
    sector = 'power'
    # Categories for the cost matrix (BCET)
    c2ti = {category: index for index, category in enumerate(titles['C2TI'])}
    t2ti = {category: index for index, category in enumerate(titles['T2TI'])}
    num_regions = len(titles['RTI'])
    num_techs = len(titles['T2TI'])

    # Conditional vector concerning technology properties
    # (same for all regions)
    Svar = data['BCET'][:, :, c2ti['18 Variable (0 or 1)']]


    # TODO: This is a generic survival function
    HalfLife = data['BCET'][:, :, c2ti['9 Lifetime (years)']]/2
    dLifeT = HalfLife/10

    for age in range(len(titles['TYTI'])):

        age_matrix = np.ones_like(data['MSRV'][:, :, age]) * age

        data['MSRV'][:, :, age] = 1.0 - 0.5*(1+np.tanh(1.25*(HalfLife-age_matrix)/dLifeT))

   

    # Copy over PRSC/EX values

    data['PRSC13'] = np.copy(time_lag['PRSC13'] )
    data['EX13'] = np.copy(time_lag['EX13'] )
    data['PRSC15'] = np.copy(time_lag['PRSC15'] )
    data["REX13"] = np.copy(time_lag["REX13"])
    # %% First initialise if necessary

    T_Scal = 10      # Time scaling factor used in the share dynamics

    # Initialisation, which corresponds to lines 389 to 556 in fortran
    if year == 2013:
        data['PRSC13'] = np.copy(data['PRSCX'])
        data['EX13'] = np.copy(data['EXX'])
        data['REX13'] = np.copy(data['REXX'])

        data['MEWL'][:, :, 0] = data["MWLO"][:, :, 0]
        data['MEWK'][:, :, 0] = np.divide(data['MEWG'][:, :, 0], data['MEWL'][:, :, 0],
                              where=data['MEWL'][:, :, 0] > 0.0) / 8766
        data['MEWS'][:, :, 0] = np.divide(data['MEWK'][:,:,0], data['MEWK'][:,:,0].sum(axis=1)[:,np.newaxis],
                                          where=data['MEWK'][:, :, 0].sum(axis=1)[:,np.newaxis] > 0.0)

        bcet, bcsc, mewl, mepd, merc, rery, mred, mres = cost_curves(
                data['BCET'], data['MCSC'], data['MEWDX'], data['MEWG'], data['MEWL'], data['MEPD'],
                data['MERC'], time_lag['MERC'], data['RERY'], data['MPTR'], data['MRED'], data['MRES'],
                titles['RTI'], titles['T2TI'], titles['ERTI'], year, 1.0, data['MWLO']
                )

        data['BCET'] = bcet
        data['MCSC'] = bcsc
        data['MEWL'] = mewl
        data['MEPD'] = mepd
        data['MERC'] = merc
        data['RERY'] = rery
        data['MRED'] = mred
        data['MRES'] = mres

        data = get_lcoe(data, titles, year)
        data = rldc(data, data["MEWDX"][:, 7, 0], time_lag, iter_lag, year, 1, titles, histend)
        mslb, mllb, mes1, mes2 = dspch(data['MWDD'], data['MEWS'], data['MKLB'], data['MCRT'],
                                   data['MEWL'], data['MWMC'], data['MMCD'],
                                   len(titles['RTI']), len(titles['T2TI']), len(titles['LBTI']))
        data['MSLB'] = mslb
        data['MLLB'] = mllb
        data['MES1'] = mes1
        data['MES2'] = mes2
        
        # Total electricity demand
        tot_elec_dem = data['MEWDX'][:,7,0] * 1000/3.6

        earlysc = np.zeros([len(titles['RTI']), len(titles['T2TI'])])
        lifetsc = np.zeros([len(titles['RTI']), len(titles['T2TI'])])

        for r in range(len(titles['RTI'])):

            # Generation by tech x load band is share of total electricity demand
            glb3 = data['MSLB'][r,:,:] * data['MLLB'][r,:,:] * tot_elec_dem[r]
            # Capacity by tech x load band
            klb3 = glb3 / data['MLLB'][r,:,:]
            
            # Load factors
            data['MEWL'][r, :, 0] = np.zeros(len(titles['T2TI']))

            nonzero_cap = np.sum(klb3, axis=1)>0
            data['MEWL'][r, nonzero_cap, 0] =  np.sum(glb3[nonzero_cap, :], axis=1) / np.sum(klb3[nonzero_cap,:], axis=1)
                                                            

            # Generation by load band
            data['MWG1'][r, :, 0] = glb3[:, 0]
            data['MWG2'][r, :, 0] = glb3[:, 1]
            data['MWG3'][r, :, 0] = glb3[:, 2]
            data['MWG4'][r, :, 0] = glb3[:, 3]
            data['MWG5'][r, :, 0] = glb3[:, 4]
            data['MWG6'][r, :, 0] = glb3[:, 5]
            # To avoid division by 0 if 0 shares
            zero_lf = data['MEWL'][r,:,0]==0
            data['MEWL'][r, zero_lf, 0] = data['MWLO'][r, zero_lf, 0]
            
            

            # Capacities
            data['MEWK'][r, :, 0] = divide(data['MEWG'][r, :, 0], data['MEWL'][r, :, 0]) / 8766
            
            # Update market shares
            data["MEWS"][r, :, 0] = data['MEWK'][r, :, 0] / data['MEWK'][r, :, 0].sum()
                        
            
            cap_diff = data['MEWK'][r, :, 0] - time_lag['MEWK'][r, :, 0]
            cap_drpctn = time_lag['MEWK'][r, :, 0] / time_lag['BCET'][r, :, c2ti['9 Lifetime (years)']]
            data['MEWI'][r, :, 0] = np.where(cap_diff > 0.0,
                                             cap_diff + cap_drpctn,
                                             cap_drpctn)
            

       
        data['MEWL'][:, :, 0] = data['MWLO'][:, :, 0].copy()
        data['MCFC'][:, :, 0] = data['MWLO'][:, :, 0].copy()
        data['BCET'][:, :, c2ti['11 Decision Load Factor']] = data['MCFC'][:, :, 0].copy()
        
        data = get_lcoe(data, titles, year)                                  # Get the levelised costs
        data = get_marginal_fuel_prices_mewp(data, titles, Svar, glb3)       # Get the marginal fuel prices



    #%%
    # Up to the last year of historical market share data
    elif year <= histend['MEWG']:
        if year == 2015: 
            data['PRSC15'] = np.copy(data['PRSCX'])

        # Set starting values for MERC
        data['MERC'][:, 0, 0] = 0.255
        data['MERC'][:, 1, 0] = 5.689
        data['MERC'][:, 2, 0] = 0.4246
        data['MERC'][:, 3, 0] = 3.374
        data['MERC'][:, 4, 0] = 0.001
        data['MERC'][:, 7, 0] = 0.001

        # Initialise load factors (last year's, or exogenous if first year)
#        loadfac = data['MEWLX'][:, :, 0]
#        if not loadfac.any():
#            loadfac = data['MWLO'][:, :, 0]
#        data['MEWL'][:, :, 0] = np.copy(loadfac)

        if year > 2013: 
            data['MEWL'][:, :, 0] = time_lag['MEWL'][:, :, 0].copy()

        cond = np.logical_and(data['MEWL'][:, :, 0] < 0.01, data['MWLO'][:, :, 0] > 0.0)
        data['MEWL'][:, :, 0] = np.where(cond,
                                 data['MWLO'][:, :, 0],
                                 data['MEWL'][:, :, 0])


        # Initialise starting capacities
        if year <= 2012:
            data['MEWK'][:, :, 0] = divide(data['MEWG'][:, :, 0], data['MWLO'][:, :, 0]) / 8766
        else:
            data['MEWK'][:, :, 0] = divide(data['MEWG'][:, :, 0], data['MEWL'][:, :, 0]) / 8766
        

        data['MEWS'][:, :, 0] = np.divide(data['MEWK'][:,:,0], data['MEWK'][:,:,0].sum(axis=1)[:, np.newaxis])
        
        
        
        # If first year, get initial MC, dMC for DSPCH ( TODO FORTRAN??)
        if not time_lag['MMCD'][:, :, 0].any():
            time_lag = get_lcoe(data, titles, year)
        # Call RLDC function for capacity and load factor by LB, and storage costs
        
        if year >= 2013: # Still in historical period

            # 1 and 2 -- Estimate RLDC and storage parameters
            data = rldc(data, data["MEWDX"][:, 7, 0], time_lag, iter_lag, year, 1, titles, histend)

            # 3--- Call dispatch routine to connect market shares to load bands
            # Call DSPCH function to dispatch flexible capacity based on MC
            if year == 2013:
                mslb, mllb, mes1, mes2 = dspch(data['MWDD'], data['MEWS'], data['MKLB'], data['MCRT'],
                                        data['MEWL'], data['MWMC'], data['MMCD'],
                                        len(titles['RTI']), len(titles['T2TI']), len(titles['LBTI']))
            else:
                mslb, mllb, mes1, mes2 = dspch(data['MWDD'], data['MEWS'], data['MKLB'], data['MCRT'],
                                        data['MEWL'], time_lag['MWMC'], time_lag['MMCD'],
                                        len(titles['RTI']), len(titles['T2TI']), len(titles['LBTI']))
            data['MSLB'] = mslb
            data['MLLB'] = mllb
            data['MES1'] = mes1
            data['MES2'] = mes2
            
            # Change currency from EUR2015 to USD2013
            if year >= 2015:

                data['MSSP'][:, :, 0] = data['MSSP'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis]/data['PRSC15'][:, 0, 0, np.newaxis])/ data['EX13'][33, 0, 0]
                data['MLSP'][:, :, 0] = data['MLSP'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis]/data['PRSC15'][:, 0, 0, np.newaxis])/ data['EX13'][33, 0, 0]
                data['MSSM'][:, :, 0] = data['MSSM'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis]/data['PRSC15'][:, 0, 0, np.newaxis])/ data['EX13'][33, 0, 0]
                data['MLSM'][:, :, 0] = data['MLSM'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis]/data['PRSC15'][:, 0, 0, np.newaxis])/ data['EX13'][33, 0, 0]

            # TODO: This is not per se correct but it's how it is in E3ME
            else:

                data['MSSP'][:, :, 0] = 0.0
                data['MLSP'][:, :, 0] = 0.0
                data['MSSM'][:, :, 0] = 0.0
                data['MLSM'][:, :, 0] = 0.0

            # Total electricity demand
            tot_elec_dem = data['MEWDX'][:, 7, 0] * 1000/3.6

            earlysc = np.zeros([len(titles['RTI']), len(titles['T2TI'])])
            lifetsc = np.zeros([len(titles['RTI']), len(titles['T2TI'])])
            
            # 4--- Calculate average capacity factors according to load bands
            for r in range(len(titles['RTI'])):

                # Generation by tech x load band is share of total electricity demand
                glb3 = data['MSLB'][r,:,:] * data['MLLB'][r,:,:] * tot_elec_dem[r]
                # Capacity by tech x load band
                klb3 = glb3/data['MLLB'][r, :, :]
                # Load factors

                data['MEWL'][r, :, 0] = np.zeros(len(titles['T2TI']))

                nonzero_cap = np.sum(klb3, axis=1)>0
                data['MEWL'][r, nonzero_cap, 0] = np.sum(glb3[nonzero_cap,:], axis=1) / np.sum(klb3[nonzero_cap,:], axis=1)


                # Generation by load band
                data['MWG1'][r, :, 0] = glb3[:, 0]
                data['MWG2'][r, :, 0] = glb3[:, 1]
                data['MWG3'][r, :, 0] = glb3[:, 2]
                data['MWG4'][r, :, 0] = glb3[:, 3]
                data['MWG5'][r, :, 0] = glb3[:, 4]
                data['MWG6'][r, :, 0] = glb3[:, 5]
                # To avoid division by 0 if 0 shares
                zero_lf = data['MEWL'][r, :, 0] == 0
                data['MEWL'][r, zero_lf, 0] = data['MWLO'][r, zero_lf, 0]
                
                # Adjust capacity factors for VRE due to curtailment, and to cover efficiency losses during
                # Gross Curtailed electricity
                data['MCGA'][r, 0, 0] = data['MCRT'][r,0,0] * np.sum(Svar[r, :] * data['MEWG'][r,:,0])

                # Net curtailed generation
                # Remove long-term storage demand and assume that at least 45% of gross curtailment is retained.
                # On average 45% of curtailed electricity can be reused for long-term storage:
                # Source: https://www.frontiersin.org/articles/10.3389/fenrg.2020.527910/full
                data['MCNA'][r, 0, 0] = np.maximum(data['MCGA'][r, 0, 0] - 0.45*2*data['MLSG'][r,0,0], 0.55*data['MCGA'][r,0,0])
                # Impact of net curtailment on load factors for VRE technologies
                # Scale down the curtailment rate by taking into account the electricity that is actually used for long-term storage
                if (data['MCGA'][r, 0, 0] == 0):
                    data['MCTN'][r, :, 0] = 0
                else:
                    data['MCTN'][r, :, 0] = data['MCTG'][r, :, 0] * data['MCNA'][r, 0, 0] / data['MCGA'][r, 0, 0]
                                
                # Total additional electricity that needs to be generated
                data['MADG'][r,0,0] = data['MCGA'][r,0,0] - data['MCNA'][r, 0, 0] + data['MSSG'][r,0,0]
                
                data['MEWK'][:, :, 0] = divide(data['MEWG'][:, :, 0], data['MEWL'][:, :, 0]) / 8766
                
                data['MEWS'][:, :, 0] = np.divide(data['MEWK'][:,:,0], data['MEWK'][:,:,0].sum(axis=1)[:,np.newaxis],
                                                  where=data['MEWK'][:, :, 0].sum(axis=1)[:,np.newaxis] > 0.0)

                # C02 emissions for carbon costs (MtC02)
                data['MEWE'][r, :, 0] = data['MEWG'][r, :, 0] * data['BCET'][r, :, c2ti['15 Emissions (tCO2/GWh)']]/1e6

                # TODO: Clean this up - for now just an ugly loop
                for t in range(len(titles['T2TI'])):

                    if data['MEWK'][r, t, 0] - time_lag['MEWK'][r, t, 0] >= 0.0:

                        earlysc[r, t] = 0.0
                        lifetsc[r, t] = time_lag['BCET'][r, t, c2ti['9 Lifetime (years)']]
                        data['MESC'][r,t,0] = 0.0

                    else:

                        earlysc[r, t] = data['MEWK'][r, t, 0] - time_lag['MEWK'][r, t, 0]
                        lifetsc[r, t] = (1.0 - data['MEWK'][r, t, 0]/np.sum(data['MEWK'][r, :, 0])) * data['MEWK'][r, t, 0] / earlysc[r, t]*5

                    if (lifetsc[r, t]-time_lag['BCET'][r, t, c2ti['9 Lifetime (years)']]) < 0.0:

                        data['MESC'][r,t,0] = -earlysc[r, t] * (
                                time_lag['BCET'][r, t, c2ti['9 Lifetime (years)']] -
                                lifetsc[r, t]/data['BCET'][r, t, c2ti['9 Lifetime (years)']]*
                                time_lag['BCET'][r, t, c2ti['3 Investment ($/kW)']])

                        data['MELF'][r,t,0] = lifetsc[r, t]

                    else:

                        data['MESC'][r,t,0] = 0.0
                        data['MELF'][r,t,0] = time_lag['BCET'][r, t, c2ti['9 Lifetime (years)']]

            data["MEWI"] = get_sales_yearly(
                            data["MEWK"], time_lag["MEWK"], data["MEWI"],
                            data['BCET'][:, :, c2ti["9 Lifetime (years)"]], year)

            # =============================================================
            # Learning-by-doing
            # =============================================================

            # Cumulative global learning
            # Using a technological spill-over matrix (MEWB) together with capacity
            # additions (MEWI) we can estimate total global spillover of similar techs

            mewi0 = np.sum(data['MEWI'][:, :, 0], axis=0)
            dw = np.zeros(len(titles["T2TI"]))
            
            for i in range(len(titles["T2TI"])):
                dw_temp = np.copy(mewi0)
                dw_temp[dw_temp > dw_temp[i]] = dw_temp[i]
                dw[i] = np.dot(dw_temp, data['MEWB'][0, i, :])


            # Cumulative capacity incl. learning spill-over effects
            data["MEWW"][0, :, 0] = time_lag['MEWW'][0, :, 0] + dw

            # Copy over the technology cost categories that do not change (all except prices which are updated through learning-by-doing below)
            data['BCET'][:, :, 1:22] = time_lag['BCET'][:, :, 1:22].copy()

            # Add in carbon costs due to EU ETS
            data['BCET'][:, :, c2ti['1 Carbon Costs ($/MWh)']]  = set_carbon_tax(data, c2ti, year)
            
            # For dispatchable techs, set decision load factor at MEWL
            data['BCET'][Svar==0, c2ti['11 Decision Load Factor']] = data["MEWL"][Svar==0, 0]

            
             
            # Learning-by-doing effects on investment
            if year > histend['BCET']:
                for tech in range(len(titles['T2TI'])):
                    if data['MEWW'][0, tech, 0] > 0.1:
                        data['BCET'][:, tech, c2ti['3 Investment ($/kW)']] = time_lag['BCET'][:, tech, c2ti['3 Investment ($/kW)']] * \
                                                                                (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0])
                        data['BCET'][:, tech, c2ti['4 std ($/MWh)']] = time_lag['BCET'][:, tech, c2ti['4 std ($/MWh)']] * \
                                                                            (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0])
                        data['BCET'][:, tech, c2ti['7 O&M ($/MWh)']] = time_lag['BCET'][:, tech, c2ti['7 O&M ($/MWh)']] * \
                                                                            (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0])
                        data['BCET'][:, tech, c2ti['8 std ($/MWh)']] = time_lag['BCET'][:, tech, c2ti['8 std ($/MWh)']] * \
                                                                            (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0])
            
            # Investment in terms of power technologies:
            for r in range(len(titles['RTI'])):
                data['MWIY'][r, :, 0] = time_lag['MWIY'][r, :, 0] + data['MEWI'][r, :, 0]*data['BCET'][r, :, c2ti['3 Investment ($/kW)']] / 1.33

            # =====================================================================
            # Cost-supply curve
            # =====================================================================

            bcet, bcsc, mewl, mepd, merc, rery, mred, mres = cost_curves(
                data['BCET'], data['MCSC'], data['MEWDX'], data['MEWG'], data['MEWL'], data['MEPD'],
                data['MERC'], time_lag['MERC'], data['RERY'], data['MPTR'], data['MRED'], data['MRES'],
                titles['RTI'], titles['T2TI'], titles['ERTI'], year, 1.0, data['MWLO']
                )

            data['BCET'] = bcet
            data['MCSC'] = bcsc
            data['MEWL'] = mewl
            data['MEPD'] = mepd
            data['MERC'] = merc
            data['RERY'] = rery
            data['MRED'] = mred
            data['MRES'] = mres
            
            # Take into account curtailment again:
            data["MEWL"] = data["MEWL"] * (1 - data["MCTN"])
            data['BCET'][:, :, c2ti['11 Decision Load Factor']]  *= (1 - data["MCTN"][:, :, 0])
            
            # =====================================================================
            # Initialise the LCOE variables
            # =====================================================================
            data = get_lcoe(data, titles, year)
            data = get_marginal_fuel_prices_mewp(data, titles, Svar, glb3) 
            # Historical differences between demand and supply.
            # This variable covers transmission losses and net exports
            # Hereafter, the lagged variable will have these values stored
            # We assume that the values do not change throughout simulation.
    #        data['MELO'][:, 0, 0] = data['MEWG'][:,:,0].sum(axis=1) - tot_elec_dem
            data["MWDL"] = time_lag["MEWDX"]        # Save so that you can access twice lagged demand
            

# %% Simulation of stock and energy specs
    
    # Stock based solutions first
    elif year > histend['MEWG']:
        # TODO: Implement survival function to get a more accurate depiction of
        # technologies being phased out and to be able to track the age of the fleet.
        # This means that a new variable will need to be implemented which is
        # basically PG_VFLT with a third dimension (techicle age in years- up to 23y)
        # Reduced efficiences can then be tracked properly as well.

        # =====================================================================
        # Start of simulation
        # =====================================================================
        
        data_dt = {}

        # First, fill the time loop variables with the their lagged equivalents
        for var in time_lag.keys():

            if domain[var] == 'FTT-P':

                data_dt[var] = np.copy(time_lag[var])
                
        # START WRITING HERE !!!!!!!!!!!!!!!!!!!!!!!!!! #
        """ Create a function that calculates the marketshare between rooftop solar and utilities from data_dt (t-1)
        # Run the rooftop solar model first
        # Make a new share files looking at the transport model (simpler) -> main file of the transport model, line 251 (initialise... market share dynamics)
        # Actually copied from freight the shares file and renamed it to ftt_p_shares_rftsolar
        # in the shares function I will need to create new variables
        # copy MEWA and use a similar format to calculate with dumb data starting with one -> we will need the classifications for MEWA
        # Initially put this in harcode to make it simpler
        # that is because initially turnover rate will be calculated using dummy data
        # set all values of isreg to zero
        # utot is basically the household demand variable I created
        # For the markershares, I need a new variable with the household demand -> Make a new tab (variable) -> a bit like variable RBFM in the input file of transport model
        # Import the new demand data and run the model to check whether there are any problems
        # I will need a new variable which is the split between the rooftop solar and utilities
        
        
        # Create a variable that represents the total price of utilities, which is a weighted average of the LCOE with the shares of each technology
        # This variable above is MEWP (Pay attention it is in USD/GJ) -> ftt_p_mewp.py
        # METC ($/MWh) is the LCOE for every technology -> I will need to use it for rooftop solar (It does not have the gamma values)
        # Assume (metc_dt) a width of the distribution of 30% -> Because you will need a std for the price of utilities. Just hardcode price x 0.3
        # Updating the gamma values will be the final step so don't worry about it now. There are scripts that can help."""
        
        # THE INFORMATION ON MEWDH HAS BEEN IMPORTED CORRECTLY
        #print(data['MEWDH']) 
        # UNIT IS thousand toe SO NEEDS CONVERSION -> GWh
        # MAYBE I WILL NEED TO ALSO CONVERT IT TO PJ BECAUSE THAT IS THE UNIT OF MEWD (If I ever need to compare both)
        
        # ELECTRICITY PRICE INFORMATION SENT BY CORMAC HAS BEEN IMPORTED CORRECTY
        # This information will be in the marketshares calculation when compared to prices of rooftop solar panels
        # Add it to the main input file
        
        #Standard deviation = add 30% 
        
        # USE THIS PART HERE
        # REMEMBER TO SET num_techs = 2
        # PASS THE SUBSTITUTION MATRIX AS 1 IN ALL [[]] - the matrix should be a 2 x 2 matrix
        # LOOK AT THE INPUT IN THE DEBUG MODE TO CHECK THE NUMBER OF DIMENSIONS - Maybe a third empty dimension is required
        #  reg_constr -> PASS AN ARRAY OF ZEROS
        # DONT NEED TO PASS ANY INFORMATION ON UPPER AND LOWER LIMITS -> THE DEFAULT IS FALSE
        
        # CARLOS -> creates an array with every region
        valid_regions = np.arange(num_regions)

        # Create the regulation variable
        division = np.zeros_like(data_dt['MEWR'][:, :, 0])
        np.divide((data_dt['MEWK'][:, :, 0] - data['MEWR'][:, :, 0]), data['MEWR'][:, :, 0],
                  out=division, where=data['MEWR'][:, :, 0] > 0)
        isReg = 0.5 + 0.5 * np.tanh(1.5 + 10 * division)
       

        isReg[data['MEWR'][:, :, 0] == 0.0] = 1.0
        isReg[data['MEWR'][:, :, 0] == -1.0] = 0.0

        # Call the survival function routine.
#        data = survival_function(data, time_lag, histend, year, titles)

        # Total number of scrapped techicles:
#        tot_eol = np.sum(data['MEOL'][:, :, 0], axis=1)

        # Total capacity additions
#        data['MWIA'][:, 0, 0] = (data['PG_TTC'][:, 0, 0]
#                                   - time_lag['PG_TTC'][:, 0, 0]
#                                   + tot_eol)
#        data['MWIA'][:, 0, 0][data['MEWKA'][:, 0, 0] < 0.0] = 0.0

        # Number of timesteps no_it and timestep size dt
        no_it = int(data['noit'][0, 0, 0])
        dt = 1 / float(no_it)

        data["MWDL"] = time_lag["MEWDX"]             # Save so that you can access twice lagged demand
        growth_rate = 1 + (time_lag["MEWDX"][:, 7, 0] - time_lag["MWDL"][:, 7, 0])/time_lag["MWDL"][:, 7, 0]
        
        # =====================================================================
        # Start of the quarterly time-loop
        # =====================================================================

        # Start the computation of shares

        # Initialise household_shares per region when they are invalid/uninitialised.
        # A global check can leave some regions at [0, 0], which then creates
        # region-specific random drops near the simulation boundary.
        rooftop_idx = t2ti['23 Rooftop Solar']
        observed_rooftop_gen = np.copy(data['MEWG'][:, rooftop_idx, 0])
        observed_household_demand = data['MEWDH'][:, 0, 0] * 11.63
        observed_rooftop_share = np.zeros(num_regions)
        np.divide(
            observed_rooftop_gen,
            observed_household_demand,
            out=observed_rooftop_share,
            where=observed_household_demand > 0.0
        )
        observed_rooftop_share = np.clip(observed_rooftop_share, 0.0, 1.0)
        near_term_calibration_year = year <= histend['MEWG'] + 2
        valid_rooftop_target = (
            near_term_calibration_year &
            (observed_rooftop_gen > 0.0) &
            (observed_household_demand > 0.0)
        )

        hh_prev = np.copy(data_dt['household_shares'][:, :, 0])
        hh_sum = np.sum(hh_prev, axis=1)
        invalid_hh = np.any(~np.isfinite(hh_prev), axis=1) | (hh_sum <= 0.0)

        demand_gwh = data_dt['MEWDH'][:, 0, 0] * 11.63
        rooftop_share_init = np.zeros(num_regions)
        np.divide(
            data_dt['MEWG'][:, rooftop_idx, 0],
            demand_gwh,
            out=rooftop_share_init,
            where=demand_gwh > 0.0
        )
        rooftop_share_init = np.clip(rooftop_share_init, 0.0, 1.0)

        if np.any(invalid_hh):
            hh_prev[invalid_hh, 0] = rooftop_share_init[invalid_hh]
            hh_prev[invalid_hh, 1] = 1.0 - rooftop_share_init[invalid_hh]

        # Keep all regions numerically well-formed: [0, 1] and row sum = 1.
        hh_prev = np.clip(hh_prev, 0.0, 1.0)
        hh_sum = np.sum(hh_prev, axis=1)
        valid_hh = hh_sum > 0.0
        hh_prev[valid_hh, :] = hh_prev[valid_hh, :] / hh_sum[valid_hh, np.newaxis]
        hh_prev[~valid_hh, 0] = rooftop_share_init[~valid_hh]
        hh_prev[~valid_hh, 1] = 1.0 - rooftop_share_init[~valid_hh]

        data_dt['household_shares'][:, :, 0] = hh_prev
        if (year == histend['MEWG'] + 1 or
                _SMALLSCALE_MAX_ROOFTOP_SHARE is None or
                len(_SMALLSCALE_MAX_ROOFTOP_SHARE) != num_regions):
            _SMALLSCALE_MAX_ROOFTOP_SHARE = _load_max_rooftop_share(
                titles,
                data_dt['household_shares'][:, 0, 0],
                observed_rooftop_share,
                valid_rooftop_target
        )
        max_rooftop_share = _SMALLSCALE_MAX_ROOFTOP_SHARE

        # Near-term rooftop observations are used as calibration anchors, but
        # rooftop PV should not be forced to decline at the simulation boundary.
        # If the 2023-2024 input value is lower than the previous simulated year,
        # hold the previous generation level and let FTT dynamics resume after 2024.
        calibration_target_gen = np.maximum(
            observed_rooftop_gen,
            time_lag['MEWG'][:, rooftop_idx, 0]
        )
        calibration_target_share = np.zeros(num_regions)
        np.divide(
            calibration_target_gen,
            observed_household_demand,
            out=calibration_target_share,
            where=observed_household_demand > 0.0
        )
        calibration_target_share = np.minimum(calibration_target_share, max_rooftop_share)

        if (_SMALLSCALE_GAMMA is None or len(_SMALLSCALE_GAMMA) != num_regions):
            _SMALLSCALE_GAMMA = np.zeros(num_regions)
        smallscale_gamma = _SMALLSCALE_GAMMA
        household_start_share = np.copy(data_dt['household_shares'][:, 0, 0])
        for t in range(1, no_it + 1):

            # Household electricity price from the grid: USD/MWh
            price_grid = data_dt['PRICH'][:, 0, 0] / 11.63 / data_dt['EXX'][:, 0, 0]

            # Self-consumption: %
            # Start with an exogenous/fixed value; later make it dynamic with EV uptake.
            self_consumption_rate = data_dt['RSSC'][:, 0, 0]

            # Levelized export price / feed-in tariff: convert USD/kWh to USD/MWh.
            price_export = data_dt['RSFT'][:, 0, 0] * 1000.0

            # Annual benefit for rooftop solar (USD/MWh)
            annual_benefit = (self_consumption_rate * price_grid) + ((1-self_consumption_rate) * price_export)

            # Investment cost for rooftop solar (USD/kW)
            investment_per_kw = data_dt['BCET'][:, t2ti['23 Rooftop Solar'], c2ti['3 Investment ($/kW)']]

            # Capacity needed to generate 1 MWh/year
            # 1 kW produces load_factor * 8.766 MWh/year
            pv_size_kw = 1 / (data['BCET'][:, t2ti['23 Rooftop Solar'], c2ti['11 Decision Load Factor']] * 8.766)
            inv_cost_pv = investment_per_kw * pv_size_kw

            # Discounted benefit
            discount_rate = 0.1
            elec_price_growth = 0.02

            lifetime_pv = data_dt['BCET'][:, t2ti['23 Rooftop Solar'], c2ti['9 Lifetime (years)']]
            disc_factor_benefit = ((1.0 - ((1.0 + elec_price_growth) / (1.0 + discount_rate)) ** lifetime_pv)/ (discount_rate - elec_price_growth))
            disc_benefit_pv = annual_benefit * disc_factor_benefit

            # Net Present Cost of rooftop solar (USD/MWh)
            npc_pv = inv_cost_pv - disc_benefit_pv

            # Build household cost matrix
            data_dt['costs_household'] = np.zeros((num_regions, 2, 1))

            # Rooftop PV: NPC per useful self-consumed MWh
            data_dt['costs_household'][:, 0, 0] = npc_pv

            # Grid baseline is zero because avoided grid purchases are included in the PV benefit.
            data_dt['costs_household'][:, 1, 0] = 0.0
            
            # Std deviation for costs households
            data_dt['costs_household_std'] = np.zeros((num_regions, 2, 1))
            data_dt['costs_household_std'][:, 0] = data_dt['MTCD'][:, t2ti['23 Rooftop Solar']]
            data_dt['costs_household_std'][:, 1] = 0.3*data_dt['costs_household'][:, 1]

            # Keep diagnostics in the main data container for year-end tracing.
            data['costs_household'] = np.copy(data_dt['costs_household'])
            data['costs_household_std'] = np.copy(data_dt['costs_household_std'])
            
            # Assuming 4 years for the decision of people going grid -> solar and 30 years for solar -> grid
            subst_households = np.zeros((num_regions, 2, 2))
            subst_households[:, 0, 1] = 1/4   # above diagonal
            subst_households[:, 1, 0] = 1/15  # below diagonal

            if t == 1 and np.any(valid_rooftop_target):
                _SMALLSCALE_GAMMA = _calibrate_rooftop_gamma(
                    data_dt['household_shares'],
                    calibration_target_share,
                    data_dt['costs_household'],
                    data_dt['costs_household_std'],
                    subst_households,
                    dt,
                    no_it,
                    valid_rooftop_target,
                    max_rooftop_share
                )
                smallscale_gamma = _SMALLSCALE_GAMMA

            data_dt['costs_household'][:, 0, 0] += smallscale_gamma

            upper_limit_households = np.ones((num_regions, 2, 1))
            upper_limit_households[:, 0, 0] = max_rooftop_share
            lower_limit_households = np.zeros((num_regions, 2, 1))

            change_in_shares = shares_change(
                     dt=dt,
                     regions=valid_regions,
                     shares_dt=data_dt['household_shares'],        # Shares at previous t
                     costs=data_dt['costs_household'],             # Costs
                     costs_sd=data_dt['costs_household_std'],      # Standard deviation costs
                     subst=subst_households,                       # Substitution turnover rates
                     reg_constr=np.zeros((num_regions, 2)),     # Constraint due to regulation
                     num_regions=num_regions,                      # Number of regions
                     num_techs=2,
                     upper_limit=upper_limit_households,
                     lower_limit=lower_limit_households,
                     limits_active=True)

            data['household_shares'][:, :, 0] = data_dt['household_shares'][:, :, 0] + change_in_shares
            data['household_shares'][:, 0, 0] = np.minimum(
                data['household_shares'][:, 0, 0], max_rooftop_share)
            data['household_shares'][:, 0, 0] = np.maximum(data['household_shares'][:, 0, 0], 0.0)
            data['household_shares'][:, 1, 0] = 1.0 - data['household_shares'][:, 0, 0]

            if np.any(valid_rooftop_target):
                near_term_path = (
                    household_start_share
                    + (calibration_target_share - household_start_share) * t / no_it
                )
                data['household_shares'][valid_rooftop_target, 0, 0] = near_term_path[valid_rooftop_target]
                data['household_shares'][valid_rooftop_target, 1, 0] = (
                    1.0 - data['household_shares'][valid_rooftop_target, 0, 0]
                )
                                                                
            # Get the power load factor for rooftop solar
            data['MEWL'][:, -1, 0] = data['BCET'][:, t2ti['23 Rooftop Solar'], c2ti['11 Decision Load Factor']]

            # Interpolate household demand within the year, analogous to grid demand.
            MEWDHt = time_lag['MEWDH'][:, 0, 0] + (data['MEWDH'][:, 0, 0] - time_lag['MEWDH'][:, 0, 0]) * t/no_it
               
            # Recalculating the generation with the new share information in GWh - 1 toe = 11.63 MWh -> 1 ktoe = 11.63 GWh
            # Opposite of above: there we went from generation to shares,
            # here we go from shares to generation with household demand
            data['MEWG'][:, -1] = data['household_shares'][:, 0] * (MEWDHt[:, np.newaxis]*11.63)
               
            # Recalculating the capacity with the new generation calculated
            # Capacity = MEWK, 8766 = number of hours
            data['MEWK'][:, -1] = data['MEWG'][:, -1] / data['MEWL'][:, -1] / 8766                                                     

            # Electricity demand is exogenous at the moment
            # TODO: Replace, using price elasticities and feedback from other
            # FTT modules
            
            # Like in FORTRAN, we estimate the growth of demand from extrapolating last year's demand. 
            # MEWDt = time_lag['MEWDX'][:,7,0] + (time_lag['MEWDX'][:, 7, 0] * growth_rate - time_lag['MEWDX'][:, 7, 0]) * t/no_it
            
            # Given that we know the demand at the end of the year, we can alternatively cheat for additional accuracy
            MEWDt = time_lag['MEWDX'][:, 7, 0] + (data['MEWDX'][:, 7, 0] - time_lag['MEWDX'][:, 7, 0]) * t/no_it
            
            MEWDt += data_dt['MADG'][:,0,0] * 0.0036
            e_demand = MEWDt * 1000/3.6

            
            # For checking
            if t == no_it:
                data["MEWD"] = np.copy(data['MEWDX'])
            
            data["MWKA"] = set_linear_coal_phase_out(data["coal phaseout"],
                                                     data["MWKA"], time_lag["MWKA"], time_lag["MEWK"], year)
            
            
            # Redifining MEWS to MEWS utilities (num regions, num techs, empty dimension necessary for the code)
            # Remember MEWS is market shares capacity
            # data_dt['MEWS_utilities'] = np.zeros((num_regions, num_techs-1, 1))
            # data_dt['MEWS_utilities'] = data_dt['MEWS'][:, :-1] / data_dt['MEWS'][:, -1].sum()
            # time_lag['MEWS_utilities'] = time_lag['MEWS'][:, :-1] / time_lag['MEWS'][:, -1].sum()
            data_dt['MEWS_utilities'] = data_dt['MEWK'][:, :-1] / data_dt['MEWK'][:, :-1].sum(axis=1, keepdims=True) 
            time_lag['MEWS_utilities'] = time_lag['MEWK'][:, :-1] / time_lag['MEWK'][:, :-1].sum(axis=1, keepdims=True)

            # =================================================================
            # Shares equation
            # =================================================================
            mews, mewl, mewg, mewk = shares(dt, t, T_Scal, MEWDt,
                                            data_dt['MEWS_utilities'], data_dt['METC'][:, :-1],
                                            data_dt['MTCD'][:, :-1], data['MWKA'][:, :-1],
                                            data_dt['MES1'][:, :-1], data_dt['MES2'][:, :-1],
                                            data['MEWA'][:, :-1, :-1], isReg[:, :-1], data_dt['MEWK'][:, :-1],
                                            time_lag['MEWK'][:, :-1], data['MEWR'][:, :-1],
                                            data_dt['MEWL'][:, :-1], time_lag['MEWS_utilities'],
                                            data['MWLO'][:, :-1],
                                            len(titles['RTI']), len(titles['T2TI']) - 1,  no_it, year)
            m = mews
            data['MEWL'][:, :-1] = mewl
            data['MEWG'][:, :-1] = mewg
            data['MEWK'][:, :-1] = mewk

            # Updated market shares for technologies together with rooftop solar
            # data['MEWS'][:, -1] = data['MEWK'][:, -1] / data['MEWK'][:, -1].sum()
            # data['MEWS'] = data['MEWK'] / data['MEWK'].sum()
            data['MEWS'] = data['MEWK'] / data['MEWK'].sum(axis=1, keepdims=True)  
            
            export_io_summary(data_dt, t2ti, year, tech_key="23 Rooftop Solar", outfile="io_check.txt")

            # VERIFY FROM HERE
            
            if np.any(np.isnan(data['MEWS'])):
                print(f"NaNs found in MEWS in {year}")
                raise ValueError
            if ~np.any(np.isclose(data['MEWS'][:,:,0].sum(axis=1), 1.0)):
                print(f"Sum of MEWS does not add up to 1 in {year}")
            if np.any(data['MEWS'][:,:,0]< 0.0):
                print(f"Negative MEWS found in {year}")
                r_err, t_err = np.unravel_index(np.nanargmin(data['MEWS'][:,:,0]), data['MEWS'][:,:,0].shape)
                
                print(data['MEWS'][r_err,t_err,0], titles['RTI'][r_err], titles["T2TI"][t_err])
            
            # =================================================================
            # Second-hand batteries. Only run at first timestep
            # =================================================================
            if t == 1: 
                data = second_hand_batteries(data, time_lag, iter_lag, year, titles)
            
            # =================================================================
            # Residual load-duration curve
            # =================================================================
            # Call RLDC function for capacity and load factor by LB, and storage costs
            data = rldc(data, MEWDt, time_lag, data_dt, year, t, titles, histend)
            
            # Change currency from EUR2015 to USD2013 (This is wrong, but in terms of logic and by misstating currency year for storage)
            data['MSSP'][:, :, 0] = data['MSSP'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis]/data['PRSC15'][:, 0, 0, np.newaxis]) / data['EX13'][33, 0, 0]
            data['MLSP'][:, :, 0] = data['MLSP'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis]/data['PRSC15'][:, 0, 0, np.newaxis]) / data['EX13'][33, 0, 0]
            data['MSSM'][:, :, 0] = data['MSSM'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis]/data['PRSC15'][:, 0, 0, np.newaxis]) / data['EX13'][33, 0, 0]
            data['MLSM'][:, :, 0] = data['MLSM'][:, :, 0] * (data['PRSC13'][:, 0, 0, np.newaxis]/data['PRSC15'][:, 0, 0, np.newaxis]) / data['EX13'][33, 0, 0]


            # =================================================================
            # Dispatch routine
            # =================================================================
            # Call DSPCH function to dispatch flexible capacity based on marginal costs (MC)
            
            test = 1
            mslb, mllb, mes1, mes2 = dspch(data['MWDD'], data['MEWS'], data['MKLB'], data['MCRT'],
                                           data['MEWL'], data_dt['MWMC'], data_dt['MMCD'],
                                           len(titles['RTI']), len(titles['T2TI']), len(titles['LBTI']))
            data['MSLB'] = mslb
            data['MLLB'] = mllb
            data['MES1'] = mes1
            data['MES2'] = mes2

            # Total electricity demand
            tot_elec_dem = data['MEWDX'][:,7,0] * 1000/3.6

            for r in range(len(titles['RTI'])):
                # Generation by tech x load band is share of total electricity demand
                glb3 = data['MSLB'][r,:,:] * data['MLLB'][r,:,:] * tot_elec_dem[r]
                # Capacity by tech x load band
                klb3 = glb3 / data['MLLB'][r,:,:]
                # Load factors
                data['MEWL'][r, :, 0] = np.zeros(len(titles['T2TI']))
                nonzero_cap = np.sum(klb3, axis=1) > 0
                data['MEWL'][r, nonzero_cap, 0] = np.sum(glb3[nonzero_cap,:], axis=1) / np.sum(klb3[nonzero_cap,:], axis=1)
                # Generation by load band
                data['MWG1'][r, :, 0] = glb3[:, 0]
                data['MWG2'][r, :, 0] = glb3[:, 1]
                data['MWG3'][r, :, 0] = glb3[:, 2]
                data['MWG4'][r, :, 0] = glb3[:, 3]
                data['MWG5'][r, :, 0] = glb3[:, 4]
                data['MWG6'][r, :, 0] = glb3[:, 5]
                # To avoid division by 0 or near 0, if very low shares
                zero_lf = data['MEWL'][r,:,0] <= 0.0001
                data['MEWL'][r, zero_lf, 0] = data["MWLO"][r, zero_lf, 0]

                # Re-calculate capacities
#                data['MEWK'][r, :, 0] = divide(data['MEWG'][r, :, 0], data['MEWL'][r, :, 0])/8766

            # =============================================================
            #  Update variables wrt curtailment
            # =============================================================
         
            # Adjust capacity factors for VRE due to curtailment, and to cover efficiency losses during
            # Gross Curtailed electricity
            data['MCGA'][:,0,0] = data['MCRT'][:,0,0] * np.sum(Svar * data['MEWG'][:,:,0], axis=1)

            # Net curtailed generation
            # Remove long-term storage demand and assume that at least 45% of gross curtailment is retained.
            # On average 45% of curtailed electricity can be reused for long-term storage:
            # Source: https://www.frontiersin.org/articles/10.3389/fenrg.2020.527910/full
            data['MCNA'][:, 0, 0] = np.maximum(data['MCGA'][:, 0, 0] - 0.45*2*data['MLSG'][:,0,0], 0.55*data['MCGA'][:,0,0])
            # Impact of net curtailment on load factors for VRE technologies
            # Scale down the curtailment rate by taking into account the electricity that is actually used for long-term storage
            data['MCTN'][:,:,0] = data['MCTG'][:,:,0] * divide(data['MCNA'][:, :, 0],
                                                               data['MCGA'][:, :, 0])
            
            
            # Total additional electricity that needs to be generated
            data['MADG'][:,0,0] = data['MCGA'][:,0,0] - data['MCNA'][:, 0, 0] + data['MSSG'][:,0,0]
            
            # Allocate utility generation on residual demand after household rooftop
            # production is accounted for.
            apply_rooftop_residual_demand = True

            updated_e_sup = e_demand[:] + data['MADG'][:, 0, 0] - data_dt['MADG'][:, 0, 0]

            household_rooftop_gen = data['household_shares'][:, 0, 0] * (MEWDHt * 11.63)

            if near_term_calibration_year:
                household_rooftop_gen = np.maximum(
                    household_rooftop_gen,
                    time_lag['MEWG'][:, rooftop_idx, 0]
                )
                rooftop_share_from_gen = np.zeros(num_regions)
                np.divide(
                    household_rooftop_gen,
                    MEWDHt * 11.63,
                    out=rooftop_share_from_gen,
                    where=MEWDHt > 0.0
                )
                data['household_shares'][:, 0, 0] = np.minimum(
                    rooftop_share_from_gen,
                    max_rooftop_share
                )
                data['household_shares'][:, 1, 0] = 1.0 - data['household_shares'][:, 0, 0]
                household_rooftop_gen = data['household_shares'][:, 0, 0] * (MEWDHt * 11.63)

            if apply_rooftop_residual_demand:
                utility_mews = data['MEWS'][:, :-1, 0]
                utility_mewl = data['MEWL'][:, :-1, 0]
                utility_denominator = np.sum(utility_mews * utility_mewl, axis=1)[:, np.newaxis]
                residual_e_sup = np.maximum(updated_e_sup - household_rooftop_gen, 0.0)

                data['MEWG'][:, :-1, 0] = divide(
                    utility_mews * residual_e_sup[:, np.newaxis] * utility_mewl,
                    utility_denominator
                )
                data['MEWG'][:, -1, 0] = household_rooftop_gen
            else:
                denominator = np.sum(data['MEWS'][:, :, 0] * data['MEWL'][:, :, 0], axis=1)[:, np.newaxis]
                data['MEWG'][:, :, 0] = divide(
                    data['MEWS'][:, :, 0] * updated_e_sup[:, np.newaxis] * data['MEWL'][:, :, 0],
                    denominator
                )
                data['MEWG'][:, -1, 0] = household_rooftop_gen

            # Keep rooftop-solar accounting aligned with the household model.
            data['MEWL'][:, -1, 0] = data['BCET'][:, t2ti['23 Rooftop Solar'], c2ti['11 Decision Load Factor']]

            # Update capacities
            data['MEWK'] = divide(data['MEWG'], data['MEWL']) / 8766
            data['MEWS'] = divide(data['MEWK'], np.sum(data['MEWK'], axis=1, keepdims=True))
            # Update emissions
            data['MEWE'][:, :, 0] = data['MEWG'][:, :, 0] * data['BCET'][:, :, c2ti['15 Emissions (tCO2/GWh)']] / 1e6
            
            # Update investment. Note that sum(mewi_t) not exactly mewi
            _, mewi_t = get_sales(
                data["MEWK"], data_dt["MEWK"], time_lag["MEWK"], data["MEWI"],
                data['BCET'][:, :, c2ti["9 Lifetime (years)"]], dt)
            
            data["MEWI"] = get_sales_yearly(
                data["MEWK"], time_lag["MEWK"], data["MEWI"],
                data['BCET'][:, :, c2ti["9 Lifetime (years)"]], year)
            

            earlysc = np.zeros([len(titles['RTI']), len(titles['T2TI'])])
            lifetsc = np.zeros([len(titles['RTI']), len(titles['T2TI'])])

            for r in range(len(titles['RTI'])):
            # TODO: Clean this up - for now just an ugly loop
                for tech in range(len(titles['T2TI'])):

                    if data['MEWK'][r, tech, 0] - data_dt['MEWK'][r, tech, 0] >= 0.0:

                        earlysc[r, tech] = 0.0
                        lifetsc[r, tech] = data_dt['BCET'][r, tech, c2ti['9 Lifetime (years)']]
                        data['MESC'][r, tech, 0] = 0.0

                    else:

                        earlysc[r, tech] = data['MEWK'][r, tech, 0] - data_dt['MEWK'][r, tech, 0]
                        lifetsc[r, tech] = ((1.0 - data['MEWK'][r, tech, 0]/np.sum(data['MEWK'][r, :, 0])) 
                                            * (data['MEWK'][r, tech, 0] / earlysc[r, tech]) * 5 )

                    if (lifetsc[r, tech] - data_dt['BCET'][r, tech, c2ti['9 Lifetime (years)']]) < 0.0:

                        data['MESC'][r, tech, 0] = -earlysc[r, tech] * (
                                (data_dt['BCET'][r, tech, c2ti['9 Lifetime (years)']] - lifetsc[r, tech]) /
                                data_dt['BCET'][r, tech, c2ti['9 Lifetime (years)']] *
                                data_dt['BCET'][r, tech, c2ti['3 Investment ($/kW)']])

                        data['MELF'][r, tech, 0] = lifetsc[r, tech]

                    else:

                        data['MESC'][r, tech, 0] = 0.0
                        data['MELF'][r, tech, 0] = data_dt['BCET'][r, tech, c2ti['9 Lifetime (years)']]

            # =============================================================
            # Learning-by-doing
            # =============================================================

            # Cumulative global learning
            # Using a technological spill-over matrix (PG_SPILL) together with capacity
            # additions (PG_CA) we can estimate total global spillover of similar techs
            mewi0 = np.sum(mewi_t[:, :, 0], axis=0)
            dw = np.zeros(len(titles["T2TI"]))
            
            
            for i in range(len(titles["T2TI"])):
                dw_temp = np.copy(mewi0)
                dw_temp[dw_temp > dw_temp[i]] = dw_temp[i]
                dw[i] = np.dot(dw_temp, data['MEWB'][0, i, :])


            # Cumulative capacity incl. learning spill-over effects
            data["MEWW"][0, :, 0] = data_dt['MEWW'][0, :, 0] + dw
           

            # Copy over the technology cost categories. We update the investment and capacity factors below
            data['BCET'][:, :, 1:22] = time_lag['BCET'][:, :, 1:22].copy()

            # Add in carbon costs
            data['BCET'][:, :, c2ti['1 Carbon Costs ($/MWh)']] = set_carbon_tax(data, c2ti, year)
            
            # For dispatchable techs, set decision load factor at MEWL
            data['BCET'][Svar==0, c2ti['11 Decision Load Factor']] = data["MEWL"][Svar==0, 0]
            
            # %%
            data["Battery cap additions"][0, t-1, 0] = quarterly_bat_add_power(no_it, data, data_dt, titles)

            # Learning-by-doing effects on investment
            for tech in range(len(titles['T2TI'])):

                if data['MEWW'][0, tech, 0] > 0.001:

                    data['BCET'][:, tech, c2ti['3 Investment ($/kW)']] = (
                            data_dt['BCET'][:, tech, c2ti['3 Investment ($/kW)']] 
                            * (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0]))
                    data['BCET'][:, tech, c2ti['4 std ($/MWh)']] = (
                            data_dt['BCET'][:, tech, c2ti['4 std ($/MWh)']] 
                             *  (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0]))
                    data['BCET'][:, tech, c2ti['7 O&M ($/MWh)']] = (
                            data_dt['BCET'][:, tech, c2ti['7 O&M ($/MWh)']] 
                             * (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0]))
                    data['BCET'][:, tech, c2ti['8 std ($/MWh)']] = (
                            data_dt['BCET'][:, tech, c2ti['8 std ($/MWh)']] 
                            * (1.0 + data['BCET'][:, tech, c2ti['16 Learning exp']] * dw[tech]/data['MEWW'][0, tech, 0]))
            
            
            # Investment (1.33 an exchange rate factor, code differs from FORTRAN)
            data['MWIY'][:, :, 0] = (data_dt['MWIY'][:, :, 0]
                                     + (data['MEWI'][:, :, 0] * dt * data['BCET'][:, :, c2ti['3 Investment ($/kW)']] / 1.33))
            

            # =================================================================
            # Cost-Supply curves
            # =================================================================  
           
            bcet, bcsc, mewl, mepd, merc, rery, mred, mres = cost_curves(
                data['BCET'], data['MCSC'], data['MEWDX'], data['MEWG'], data['MEWL'], data['MEPD'],
                data['MERC'], time_lag['MERC'], data['RERY'], data['MPTR'], data['MRED'], data['MRES'],
                titles['RTI'], titles['T2TI'], titles['ERTI'], year, dt, data['MWLO']
                )

            data['BCET'] = bcet
            data['MCSC'] = bcsc
            data['MEWL'] = mewl
            data['MEPD'] = mepd
            data['MERC'] = merc
            data['RERY'] = rery
            data['MRED'] = mred
            data['MRES'] = mres
            
            
            # Take into account curtailment, computed above:
            data["MEWL"] = data["MEWL"] * (1 - data["MCTN"])
            data['BCET'][:, :, c2ti['11 Decision Load Factor']]  *= (1 - data["MCTN"][:, :, 0])
            
            
            # =================================================================
            # Update LCOE
            # =================================================================
            data = get_lcoe(data, titles, year)
            data = get_marginal_fuel_prices_mewp(data, titles, Svar, glb3)

            # =================================================================
            # Update the time-loop variables data_dt
            # =================================================================

            # Update time loop variables:

            for var in data_dt.keys():

                if domain[var] == 'FTT-P':

                    data_dt[var] = np.copy(data[var])
            
            # export_io_summary(data_dt, t2ti, year, tech_key="23 Rooftop Solar", outfile="io_check.txt")

        if year >= 2022:
            export_rooftop_trace(data, data_dt, titles, year,
                                 first_year=(histend['MEWG'] + 1),
                                 region_key="34 USA (US)",
                                 tech_key="23 Rooftop Solar",
                                 outfile="rooftop_trace_usa.txt")
            if "37 Australia (AU)" in titles['RTI']:
                export_rooftop_trace(data, data_dt, titles, year,
                                     first_year=(histend['MEWG'] + 1),
                                     region_key="37 Australia (AU)",
                                     tech_key="23 Rooftop Solar",
                                     outfile="rooftop_trace_au.txt")

        if year == 2050 and t == no_it:
            print(f"Total solar in 2050 is: {np.sum(data['MEWG'][:, 18, 0])/10**6:.1f} M GWh, "
                  f"solar&wind is {np.sum(data['MEWG'][:, 16:19, 0])/10**6:.1f} M GWh")

    return data
