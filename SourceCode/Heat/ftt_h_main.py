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
from SourceCode.ftt_core.ftt_shares import shares_change, shares_change_premature
from SourceCode.ftt_core.ftt_mandate import implement_seeding, implement_mandate
from SourceCode.ftt_core.ftt_sales_or_investments import get_sales, get_sales_yearly

# Green technology indices for Heat (heat pumps: ground source, air-water, air-air)
GREEN_INDICES_HP = [9, 10, 11]

from SourceCode.support.get_vars_to_copy import get_loop_vars_to_copy, get_domain_vars_to_copy
from SourceCode.support.divide import divide
from SourceCode.support.check_market_shares import check_market_shares

from SourceCode.Heat.ftt_h_lcoh import get_lcoh, set_carbon_tax


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

    data['PRSC14'] = np.copy(time_lag['PRSC14'] )
    if year == 2014:
        data['PRSC14'] = np.copy(data['PRSCX'])

    # Calculate the LCOH for each heating technology
    carbon_costs = set_carbon_tax(data, c4ti)
    data = get_lcoh(data, titles, carbon_costs)


    # %% First initialise if necessary
    # Initialise in case of stock solution specification
    #if np.any(specs[sector]) < 5:

    # Up to the last year of historical useful energy demand by boiler
    # Historical data ends in 2020, so we need to initialise data
    # when it's 2021 to make sure the model runs.
    if year <= histend['HEWF']:
        # Useful energy demand by boilers
        # The historical data contains final energy demand
        data['HEWG'][:, :, 0] = data['HEWF'][:, :, 0] * data['BHTC'][:, :, c4ti["9 Conversion efficiency"]]

        for r in range(len(titles['RTI'])):

            # Total useful heat demand
            # This is the demand driver for FTT:Heat
            #data['RHUD'][r, 0, 0] = np.sum(data['HEWG'][r, :, 0])

            if data['RHUD'][r, 0, 0] > 0.0:

                # Market shares (based on useful energy demand)
                data['HEWS'][r, :, 0] = data['HEWG'][r, :, 0] / data['RHUD'][r, 0, 0]
                # Shares of final energy demand (without electricity)
                #data['HESR'][:, :, 0] = data['HEWF'][:, :, 0]
                #data['HESR'][r, :, 0] = data['HEWF'][r, :, 0] * data['BHTC'][r, :, c4ti["19 RES calc"]] / np.sum(data['HEWF'] * data['BHTC'][r, :, c4ti["19 RES calc"]])

        # CORRECTION TO MARKET SHARES
        # Sometimes historical market shares do not add up to 1.0
        hews = data['HEWS'][:, :, 0]
        region_sums = hews.sum(axis=1)
        
        needs_correction = (np.abs(region_sums - 1.0) > 1e-9) & (region_sums > 0.0)
        data['HEWS'][needs_correction, :, 0] /= region_sums[needs_correction, np.newaxis]
                    
        # Normalise HEWG to RHUD
        data['HEWG'][:, :, 0] = data['HEWS'][:, :, 0] * data['RHUD'][:, :, 0]
        
        # Recalculate HEWF based on RHUD
        data['HEWF'][:, :, 0] = data['HEWG'][:, :, 0] / data['BHTC'][:, :, c4ti["9 Conversion efficiency"]]

        # Capacity by boiler
        # Capacity (GW) (13th are capacity factors (MWh/kW=GWh/MW, therefore /1000)
        data['HEWK'][:, :, 0] = divide(data['HEWG'][:, :, 0],
                                data['BHTC'][:, :, c4ti["13 Capacity factor mean"]])/1000
        
        # Emissions
        data['HEWE'][:, :, 0] = data['HEWF'][:, :, 0] * data['BHTC'][:, :, c4ti["15 Emission factor"]] / 1e6

        for r in range(len(titles['RTI'])):
            # Final energy demand by energy carrier
            for fuel in range(len(titles['JTI'])):
                # Fuel use for heating
                data['HJHF'][r, fuel, 0] = np.sum(data['HEWF'][r, :, 0] * data['HJET'][0, :, fuel])
                # Fuel use for total residential sector
                # 0.0859 is the conversion factor from GWh to th toe
                if data['HJFC'][r, fuel, 0] > 0.0:
                    data['HJEF'][r, fuel, 0] = data['HJHF'][r, fuel, 0] / data['HJFC'][r, fuel, 0] * 0.08598

        # Investment (= capacity additions) by technology (in GW/y)
        if year > 2014:
            data["HEWI"] = get_sales_yearly(data["HEWK"], time_lag["HEWK"],
                              data["HEWI"], time_lag['BHTC'][:, :, c4ti['6 Replacetime']],
                              year)
            
            bi = np.zeros((len(titles['RTI']), len(titles['HTTI'])))
            for r in range(len(titles['RTI'])):
                bi[r,:] = np.matmul(data['HEWB'][0, :, :], data['HEWI'][r, :, 0])
            dw = np.sum(bi, axis=0)
            data['HEWW'][0, :, 0] = time_lag['HEWW'][0, :, 0] + dw

    if year == histend['HEWF']:
        # Historical data ends in 2020, so we need to initialise data
        # when it's 2021 to make sure the model runs.

        # If switch is set to 1, then an exogenous price rate is used
        # Otherwise, the price rates are set to endogenous

        #data['HFPR'][:, :, 0] = data['HFFC'][:, :, 0]

        # Now transform price rates by fuel to price rates by boiler
        #data['HEWP'][:, :, 0] = np.matmul(data['HFFC'][:, :, 0], data['HJET'][0, :, :].T)

        for r in range(len(titles['RTI'])):

            # Final energy demand by energy carrier
            for fuel in range(len(titles['JTI'])):

                # Fuel use for heating
                data['HJHF'][r, fuel, 0] = np.sum(data['HEWF'][r, :, 0] * data['HJET'][0, :, fuel])

                # Fuel use for total residential sector #HFUX is missing
                if data['HJFC'][r, fuel, 0] > 0.0:
                    data['HJEF'][r, fuel, 0] = data['HJHF'][r, fuel, 0] / data['HJFC'][r, fuel, 0]

        # Calculate the LCOH for each heating technology.
        carbon_costs = set_carbon_tax(data, c4ti)
        data = get_lcoh(data, titles, carbon_costs)

# %% Simulation of stock and energy specs
#    t0 = time.time()
    # Stock based solutions first
#    if np.any(specs[sector] < 5):
    # TODO: change this to start year FTT:H (also in update??)
    data["FU14A"] = np.copy(data['HJHF'])
    data['FU14B'] = data["HJEF"] * data["HJFC"]
    
    # Endogenous calculation takes over from here
    if year > histend['HEWF']:

        # Create a local dictionary for timeloop variables
        # It contains values between timeloop interations in the FTT core
        data_dt = {}

        # First, fill the time loop variables with the their lagged equivalents
        vars_to_copy = get_domain_vars_to_copy(time_lag, domain, 'FTT-H')
        for var in vars_to_copy:
            data_dt[var] = np.copy(time_lag[var])

        # Initialize HWIY accumulator for this year (will accumulate within timesteps)
        data_dt['HWIY'] = np.zeros([len(titles['RTI']), len(titles['HTTI']), 1])

        # Preserve baseline fuel demand from previous timestep (for output tracking)
        data["FU14A"] = time_lag["FU14A"]
        data["FU14B"] = time_lag["FU14B"]

        division = divide((time_lag['HEWS'][:, :, 0] - data['HREG'][:, :, 0]),
                           data['HREG'][:, :, 0]) # 0 if dividing by 0
        reg_constr = 0.5 + 0.5 * np.tanh(1.5 + 10 * division)
        reg_constr[data['HREG'][:, :, 0] == 0.0] = 1.0
        reg_constr[data['HREG'][:, :, 0] == -1.0] = 0.0
    
        # Factor used to create quarterly data from annual figures
        no_it = int(data['noit'][0, 0, 0])
        dt = 1 / float(no_it)
        

        ############## Computing new shares ##################

        # Start the computation of shares
        for t in range(1, no_it+1):

            # Interpolate to prevent staircase profile.
            rhudt = time_lag['RHUD'][:, :, :] + (data['RHUD'][:, :, :] - time_lag['RHUD'][:, :, :]) * t * dt
            rhudlt = time_lag['RHUD'][:, :, :] + (data['RHUD'][:, :, :] - time_lag['RHUD'][:, :, :]) * (t-1) * dt

            # Get regions with non-zero heat demand
            regions = np.where(rhudt[:, 0, 0] > 0.0)[0]
                
            # The core FTT equations, taking into account old shares, costs and regulationss
            change_in_shares = shares_change(
                dt=dt,
                regions=regions,
                shares_dt=data_dt["HEWS"],         # Shares at previous t
                costs=data_dt["HGC1"],             # Costs
                costs_sd=data_dt["HWCD"],          # Standard deviations costs
                subst=data["HEWA"] * data["HETR"], # Substitution turnover rates
                reg_constr=reg_constr,             # Constraint due to regulation
                num_regions = len(titles['RTI']),  # Number of regions
                num_techs = len(titles['HTTI']),   # Number of technologies
            )

            # Calculate scrappage rate for all regions
            SR_all = np.zeros((len(titles['RTI']), len(titles['HTTI'])))
            for r in range(len(titles['RTI'])):
                SR = divide(np.ones(len(titles['HTTI'])),
                            data['BHTC'][r, :, c4ti["16 Payback time, mean"]]) - data['HETR'][r, :, 0]
                SR_all[r, :] = np.where(SR<0.0, 0.0, SR)
            
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
                num_regions = len(titles['RTI']),   # Number of regions
                num_techs = len(titles['HTTI']),    # Number of technologies
            )

            # Calculate endogenous market shares from both changes
            endo_shares = data_dt['HEWS'][:, :, 0] + change_in_shares + changes_in_shares_prem_repl
            
            
            #################### Regulatory policies #################

            for r in range(len(titles['RTI'])):

                if rhudt[r] == 0.0:
                    continue
                
                endo_gen = endo_shares[r] * rhudt[r, np.newaxis]


                # -----------------------------------------------------
                # Step 3: Exogenous sales additions
                # -----------------------------------------------------
                # Add in exogenous sales figures. These are blended with
                # endogenous result! Note that it's different from the
                # ExogSales specification!
                Utot = rhudt[r]
                dUk = np.zeros([len(titles['HTTI'])])
                dUkTK = np.zeros([len(titles['HTTI'])])
                dUkREG = np.zeros([len(titles['HTTI'])])

                # Note, as in FTT: H shares are shares of generation, corrections MUST be done in terms of generation.
                # Otherwise, the corrections won't line up with the market shares.

                
                dUkTK = data['HWSA'][r, :, 0]*Utot/no_it
                # Check endogenous shares plus additions for a single time step does not exceed regulated shares
                reg_vs_exog = ((data['HWSA'][r, :, 0]*Utot/no_it + endo_gen) > data['HREG'][r, :, 0]*Utot) & (data['HREG'][r, :, 0] >= 0.0)
                # Filter capacity additions based on regulated shares
                dUkTK = np.where(reg_vs_exog, 0.0, dUkTK)


                # Correct for regulations due to the stretching effect. This is the difference in generation due only to demand increasing.
                # This will be the difference between generation based on the endogenous generation, and what the endogenous generation would have been
                # if total demand had not grown.

                dUkREG = -(endo_gen - endo_shares[r] * rhudlt[r,np.newaxis]) * reg_constr[r, :].reshape([len(titles['HTTI'])])
                     

                # Sum effect of exogenous sales additions (if any) with
                # effect of regulations
                dUk = dUkREG + dUkTK
                dUtot = np.sum(dUk)

  
                # Calaculate changes to endogenous generation, and use to find new market shares
                # Zero generation will result in zero shares
                # All other capacities will be streched

                if (np.sum(endo_gen) + dUtot) > 0.0:
                    data['HEWS'][r, :, 0] = (endo_gen + dUk)/(np.sum(endo_gen)+dUtot)


            # Raise error if any values are negative or market shares do not sum to 1
            check_market_shares(data['HEWS'], titles, sector, year)

            ############## Update variables ##################
            
            data['HEWG'][:, :, 0] = data['HEWS'][:, :, 0] * rhudt[:, 0, 0, np.newaxis]
            
            #Capacity (GW) (13th are capacity factors (MWh/kW=GWh/MW, therefore /1000)
            data['HEWK'][:, :, 0] = divide(data['HEWG'][:, :, 0],
                                    data['BHTC'][:, :, c4ti["13 Capacity factor mean"]])/1000

            # New additions (HEWI)
            data['HEWI'], hewi_t = get_sales(
                  data["HEWK"], data_dt["HEWK"], time_lag["HEWK"],
                  data["HEWI"], data_dt['BHTC'][:, :, c4ti['6 Replacetime']],
                  dt
                  )

            # Seed heat pumps in regions with low adoption (2025-2030)
            data['HEWI'], hewi_t, data["HEWK"] = implement_seeding(
                data['HEWK'], data['HEWI'], hewi_t, year, GREEN_INDICES_HP)

            # Change capacity and sales after mandate (only runs if hp mandate != 0)
            data['HEWI'], hewi_t, data["HEWK"] = implement_mandate(
                data['HEWK'], data['HEWI'], hewi_t, year, GREEN_INDICES_HP, data["hp mandate"])
            
            # Calculate HEWG, HEWS and HEWF after mandates  
            
            # Useful heat by boiler
            data['HEWG'][:, :, 0] = data['HEWK'][:, :, 0] * data['BHTC'][:, :, c4ti["13 Capacity factor mean"]] * 1000
            
            data['HEWS'][:, :, 0] = data['HEWG'][:, :, 0] / np.sum(data['HEWG'][:, :, 0], axis=1)[:, None]

            # Final energy by boiler
            data['HEWF'][:, :, 0] = divide(data['HEWG'][:, :, 0],
                                             data['BHTC'][:, :, c4ti["9 Conversion efficiency"]])
            
            # Emissions
            data['HEWE'][:, :, 0] = data['HEWF'][:, :, 0] * data['BHTC'][:, :, c4ti["15 Emission factor"]]/1e6
            
            
            # Map fuel prices to heating technologies
            # HFFC indices: 0=hard coal, 4=heavy fuel oil, 6=other gas, 7=natural gas,
            #               8=electricity, 10=combustible waste/biomass
            data['HEWP'][:, 0, 0] = data['HFFC'][:, 4, 0]   # Oil boiler -> heavy fuel oil
            data['HEWP'][:, 1, 0] = data['HFFC'][:, 4, 0]   # Oil condensing -> heavy fuel oil
            data['HEWP'][:, 2, 0] = data['HFFC'][:, 6, 0]   # Gas boiler -> other gas
            data['HEWP'][:, 3, 0] = data['HFFC'][:, 6, 0]   # Gas condensing -> other gas
            data['HEWP'][:, 4, 0] = data['HFFC'][:, 10, 0]  # Wood stove -> biomass
            data['HEWP'][:, 5, 0] = data['HFFC'][:, 10, 0]  # Wood boiler -> biomass
            data['HEWP'][:, 6, 0] = data['HFFC'][:, 0, 0]   # Coal -> hard coal
            data['HEWP'][:, 7, 0] = data['HFFC'][:, 8, 0]   # District heat -> electricity (proxy)
            data['HEWP'][:, 8, 0] = data['HFFC'][:, 8, 0]   # Electric heating -> electricity
            data['HEWP'][:, 9, 0] = data['HFFC'][:, 8, 0]   # Heat pump ground -> electricity
            data['HEWP'][:, 10, 0] = data['HFFC'][:, 8, 0]  # Heat pump air-water -> electricity
            data['HEWP'][:, 11, 0] = data['HFFC'][:, 8, 0]  # Heat pump air-air -> electricity

            # Final energy demand for heating purposes
            data['HJHF'][:, :, 0] = np.matmul(data['HEWF'][:, :, 0], data['HJET'][0, :, :])


            ############## Learning-by-doing ##################

            # Cumulative global learning
            # Using a technological spill-over matrix (HEWB) together with capacity
            # additions (HEWI) we can estimate total global spillover of similar
            # technologies
            bi = np.zeros((len(titles['RTI']),len(titles['HTTI'])))
            for r in range(len(titles['RTI'])):
                bi[r,:] = np.matmul(data['HEWB'][0, :, :],hewi_t[r, :, 0])
            dw = np.sum(bi, axis=0)

            # Cumulative capacity incl. learning spill-over effects
            data['HEWW'][0, :, 0] = data_dt['HEWW'][0, :, 0] + dw

            # Copy over the technology cost categories that do not change (all except prices which are updated through learning-by-doing below)
            data['BHTC'] = np.copy(data_dt['BHTC'])

            # Learning-by-doing effects on investment and efficiency
            for b in range(len(titles['HTTI'])):

                if data['HEWW'][0, b, 0] > 0.0001:

                    data['BHTC'][:, b, c4ti['1 Inv cost mean (EUR/kW)']] = (
                            data_dt['BHTC'][:, b, c4ti['1 Inv cost mean (EUR/kW)']]  
                            * (1.0 + data['BHTC'][:, b, c4ti['7 Investment LR']] * dw[b] / data['HEWW'][0, b, 0]))
                    data['BHTC'][:, b, c4ti['2 Inv Cost SD']] = (
                            data_dt['BHTC'][:, b, c4ti['2 Inv Cost SD']] 
                            * (1.0 + data['BHTC'][:, b, c4ti['7 Investment LR']] * dw[b] / data['HEWW'][0, b, 0]))
                    data['BHTC'][:, b, c4ti['9 Conversion efficiency']] = (
                            data_dt['BHTC'][:, b, c4ti['9 Conversion efficiency']]
                            * 1.0 / (1.0 + data['BHTC'][:, b, c4ti['20 Efficiency LR']] * dw[b]/data['HEWW'][0, b, 0]))


            # Total investment in new capacity (cumulative within year, m 2014 euros):
            # HEWI is the continuous time amount of new capacity built per unit time dI/dt (GW/y)
            # BHTC are the investment costs (2014Euro/kW)
            data['HWIY'][:, :, 0] = (data_dt['HWIY'][:, :, 0]
                                     + data['HEWI'][:, :, 0] * dt * data['BHTC'][:, :, c4ti['1 Inv cost mean (EUR/kW)']]
                                     / data['PRSC14'][:, 0, 0, np.newaxis])

            # Save investment cost for front end
            data["HWIC"][:, :, 0] = data["BHTC"][:, :, c4ti['1 Inv cost mean (EUR/kW)']]
            # Save efficiency for front end
            data["HEFF"][:, :, 0] = data["BHTC"][:, :, c4ti['9 Conversion efficiency']]

            # Calculate levelised cost again
            carbon_costs = set_carbon_tax(data, c4ti)
            data = get_lcoh(data, titles, carbon_costs)


            # Store heat variables that have changed in data_dt
            vars_to_copy = get_loop_vars_to_copy(data, data_dt, domain, 'FTT-H')
            for var in vars_to_copy:
                data_dt[var] = np.copy(data[var])

            # Update HWIY in data_dt for next iteration accumulation
            data_dt['HWIY'] = np.copy(data['HWIY'])

        if year == 2050 and t == no_it:
            print(f"Total heat pumps in 2050 is: {np.sum(data['HEWG'][:, 9:12, 0])/10**6:.3f} M GWh")
            
    return data
