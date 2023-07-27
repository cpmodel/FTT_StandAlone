# -*- coding: utf-8 -*-
"""
=========================================
ftt_tr_main.py
=========================================
Passenger road transport FTT module.
####################################

This is the main file for FTT: Transport, which models technological
diffusion of passenger vehicle types due to simulated consumer decision making.
Consumers compare the **levelised cost of heat**, which leads to changes in the
market shares of different technologies.

The outputs of this module include sales, fuel use, and emissions.

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros
    - `estimation <econometrics_functions.html>`__
        Predict future values according to the estimated coefficients.

Functions included:
    - solve
        Main solution function for the module
    - get_lcot
        Calculate levelised cost of transport
"""

# Standard library imports
from math import sqrt
import os
import copy
import sys
import warnings

# Third party imports
import pandas as pd
import numpy as np

# Local library imports
from support.divide import divide
from support.econometrics_functions import estimation


# %% lcot
# -----------------------------------------------------------------------------
# --------------------------- LCOT function -----------------------------------
# -----------------------------------------------------------------------------
def get_lcot(data, titles):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of transport in 2012$/p-km per
    vehicle type. It includes intangible costs (gamma values) and together
    determines the investor preferences.
    """

    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}

    tf = np.ones([len(titles['VTTI']), 1])
    tf[12:15] = 0
    tf[18:21] = 0

    for r in range(len(titles['RTI'])):

        # Cost matrix
        bttc = data['BTTC'][r, :, :]

        # Vehicle lifetime
        lt = bttc[:, c3ti['8 lifetime']]
        max_lt = int(np.max(lt))
        lt_mat = np.linspace(np.zeros(len(titles['VTTI'])), max_lt-1,
                             num=max_lt, axis=1, endpoint=True)
        lt_max_mat = np.concatenate(int(max_lt)*[lt[:, np.newaxis]], axis=1)
        mask = lt_mat < lt_max_mat
        lt_mat = np.where(mask, lt_mat, 0)

        # Capacity factor used in decisions (constant), not actual capacity factor
        cf = bttc[:, c3ti['12 Cap_F (Mpkm/kseats-y)'], np.newaxis]

        # Discount rate
        # dr = bttc[6]
        dr = bttc[:, c3ti['7 Discount rate'], np.newaxis]

        # Occupancy rates
        ff = bttc[:, c3ti['11 occupancy rate p/sea'], np.newaxis]

        # Number of seats
        ns = bttc[:, c3ti['15 Seats/Veh'], np.newaxis]

        # Energy use
        en = bttc[:, c3ti['9 energy use (MJ/km)'], np.newaxis]

        # Initialse the levelised cost components
        # Average investment cost
        it = np.zeros([len(titles['VTTI']), int(max_lt)])
        it[:, 0, np.newaxis] = bttc[:, c3ti['1 Prices cars (USD/veh)'], np.newaxis]/ns/ff/cf/1000

        # Standard deviation of investment cost
        dit = np.zeros([len(titles['VTTI']), int(max_lt)])
        dit[:, 0, np.newaxis] = bttc[:, c3ti['2 Std of price'], np.newaxis]/ns/ff/cf/1000

        # Vehicle tax at purchase
        vtt = np.zeros([len(titles['VTTI']), int(max_lt)])
        vtt[:, 0, np.newaxis] = (data['TTVT'][r, :, 0, np.newaxis]+data['RTCO'][r, 0, 0]*bttc[:,c3ti['14 CO2Emissions'], np.newaxis])/ns/ff/cf/1000

        # Average fuel costs
        ft = np.ones([len(titles['VTTI']), int(max_lt)])
        ft = ft * bttc[:,c3ti['3 fuel cost (USD/km)'], np.newaxis]/ns/ff
        ft = np.where(mask, ft, 0)

        # Stadard deviation of fuel costs
        dft = np.ones([len(titles['VTTI']), int(max_lt)])
        dft = dft * bttc[:, c3ti['4 std fuel cost'], np.newaxis]
        dft = np.where(mask, dft, 0)

        # Fuel tax costs
        fft = np.ones([len(titles['VTTI']), int(max_lt)])
        fft = fft * data['RTFT'][r, 0, 0]*en/ns/ff*tf
        fft = np.where(mask, fft, 0)

        # Average operation & maintenance cost
        omt = np.ones([len(titles['VTTI']), int(max_lt)])
        omt = omt * bttc[:, c3ti['5 O&M costs (USD/km)'], np.newaxis]/ns/ff
        omt = np.where(mask, omt, 0)

        # Standard deviation of operation & maintenance cost
        domt = np.ones([len(titles['VTTI']), int(max_lt)])
        domt = domt * bttc[:, c3ti['6 std O&M'], np.newaxis]/ns
        domt = np.where(mask, domt, 0)

        # Road tax cost
        rtt = np.ones([len(titles['VTTI']), int(max_lt)])
        rtt = rtt * data['TTRT'][r, :, 0, np.newaxis]/cf/ns/ff/1000
        rtt = np.where(mask, rtt, 0)

        # Net present value calculations
        # Discount rate
        denominator = (1+dr)**lt_mat

        # 1-Expenses
        # 1.1-Without policy costs
        npv_expenses1 = (it+ft+omt)/denominator
        # 1.2-With policy costs
        npv_expenses2 = (it+vtt+ft+fft+omt+rtt)/denominator
        # 1.3-Only policy costs
        npv_expenses3 = (vtt+fft+rtt)/denominator
        # 2-Utility
        npv_utility = 1/denominator
        #Remove 1s for tech with small lifetime than max
        npv_utility[npv_utility==1] = 0
        npv_utility[:,0] = 1
        # 3-Standard deviation (propagation of error)
        npv_std = np.sqrt(dit**2 + dft**2 + domt**2)/denominator

        # 1-levelised cost variants in $/pkm
        # 1.1-Bare LCOT
        lcot = np.sum(npv_expenses1, axis=1)/np.sum(npv_utility, axis=1)
        # 1.2-LCOT including policy costs
        tlcot = np.sum(npv_expenses2, axis=1)/np.sum(npv_utility, axis=1)
        # 1.3-LCOT of policy costs
        lcot_pol = np.sum(npv_expenses3, axis=1)/np.sum(npv_utility, axis=1)
        # Standard deviation of LCOT
        dlcot = np.sum(npv_std, axis=1)/np.sum(npv_utility, axis=1)

        # LCOT augmented with non-pecuniary costs
        tlcotg = tlcot*(1+data['TGAM'][r, :, 0])

        # Transform into lognormal space
        logtlcot = np.log(tlcot*tlcot/np.sqrt(dlcot*dlcot + tlcot*tlcot)) + data['TGAM'][r, :, 0]
        dlogtlcot = np.sqrt(np.log(1.0 + dlcot*dlcot/(tlcot*tlcot)))

        # Pass to variables that are stored outside.
        data['TEWC'][r, :, 0] = lcot            # The real bare LCOT without taxes
        data['TETC'][r, :, 0] = tlcot           # The real bare LCOE with taxes
        data['TEGC'][r, :, 0] = tlcotg         # As seen by consumer (generalised cost)
        data['TELC'][r, :, 0] = logtlcot      # In lognormal space
        data['TECD'][r, :, 0] = dlcot          # Variation on the LCOT distribution
        data['TLCD'][r, :, 0] = dlogtlcot     # Log variation on the LCOT distribution

    return data
# %% Fleet size - under development
# -----------------------------------------------------------------------------
# ----------------- Gompertz equation for fleet size --------------------------
# -----------------------------------------------------------------------------
# def fleet_size(data, titles):
#
#     return print("Hello")


# %% survival function
# -----------------------------------------------------------------------------
# -------------------------- Survival calcultion ------------------------------
# -----------------------------------------------------------------------------
def survival_function(data, time_lag, histend, year, titles):
    # In this function we calculate scrappage, sales, tracking of age, and
    # average efficiency.
    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}


    # Create a generic matrix of fleet-stock by age
    # Assume uniform distribution, but only do so when we still have historical
    # market share data. Afterwards it becomes endogeous
    if year < histend['TEWS']:

        # TODO: This needs to be replaced with actual data
        correction = np.linspace(1/(3*len(titles['VYTI'])), 3/len(titles['VYTI']), len(titles['VYTI'])) * 0.6

        for age in range(len(titles['VYTI'])):

            data['RLTA'][:, :, age] = correction[age] *  data['TEWK'][:, :, 0]

    else:
        # Once we start to calculate the market shares and total fleet sizes
        # endogenously, we can update the vehicle stock by age matrix and
        # calculate scrappage, sales, average age, and average efficiency.
        for r in range(len(titles['RTI'])):

            for veh in range(len(titles['VTTI'])):

                # Move all vehicles one year up:
                # New sales will get added to the age-tracking matrix in the main
                # routine.
                data['RLTA'][r, veh, :-1] = copy.deepcopy(time_lag['RLTA'][r, veh, 1:])

                # Current age-tracking matrix:
                # Only retain the fleet that survives
                data['RLTA'][r, veh, :] = data['RLTA'][r, veh, :] * data['TESF'][r, 0, :]

                # Total amount of vehicles that survive:
                survival = np.sum(data['RLTA'][r, veh, :])

                # EoL scrappage: previous year's stock minus what survived
                if time_lag['TEWK'][r, veh, 0] > survival:

                    data['REVS'][r, veh, 0] = time_lag['TEWK'][r, veh, 0] - survival

                elif time_lag['TEWK'][r, veh, 0] < survival:
                    if year > 2016:
                        msg = """
                        Erronous outcome!
                        Check year {}, region {} - {}, vehicle {} - {}
                        Vehicles that survived are greater than what was in the fleet before:
                        {} versus {}
                        """.format(year, r, titles['RTI'][r], veh,
                                   titles['VTTI'][veh], time_lag['TEWK'][r, veh, 0], survival)
#                        print(msg)

    # calculate fleet size
    return data

# %% main function
# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, iter_lag, titles, histend, year, specs, scenario):
    """
    Main solution function for the module.

    Add an extended description in the future.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution
    time_lag: type
        Description
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
    sector_index = 15 #titles['FUTI'].index('16 Road Transport')

    # Store fuel prices and convert to $2013/toe
    # It's actually in current$/toe
    # TODO: Temporary deflator values
    data['TE3P'][:, jti["5 Middle distillates"], 0] = iter_lag['PFRM'][:, sector_index, 0] / 1.33
    data['TE3P'][:, jti["7 Natural gas"], 0] = iter_lag['PFRG'][:, sector_index, 0] / 1.33
    data['TE3P'][:, jti["8 Electricity"], 0] = iter_lag['PFRE'][:, sector_index, 0] / 1.33
    data['TE3P'][:, jti["11 Biofuels"], 0] = iter_lag['PFRB'][:, sector_index, 0] / 1.33
#    data['TE3P'][:, "12 Hydrogen", 0] = data['PFRE'][:, sector_index, 0] * 2.0
    # %% First initialise if necessary


    # Up to the last year of historical market share data
    if year <= histend['TEWS']:

        for r in range(len(titles['RTI'])):

            # CORRECTION TO MARKET SHARES
            # Sometimes historical market shares do not add up to 1.0
            if (~np.isclose(np.sum(data['TEWS'][r, :, 0]), 0.0, atol=1e-9)
                    and np.sum(data['TEWS'][r, :, 0]) > 0.0 ):
                data['TEWS'][r, :, 0] = np.divide(data['TEWS'][r, :, 0],
                                                   np.sum(data['TEWS'][r, :, 0]))

        # Computes initial values for the capacity factor, numbers of
        # vehicles by technology and distance driven

        # "Capacities", defined as 1000 vehicles
        data['TEWK'][:, :, 0] = data['TEWS'][:, :, 0] * data['RFLT'][:, 0, 0, np.newaxis]

        # "Generation", defined as total km driven
        data['TEWG'][:, :, 0] = data['TEWK'][:, :, 0] * data['RVKM'][:, 0, 0, np.newaxis] * 1e-3

        # "Emissions", Factor 1.2 approx fleet efficiency factor, corrected later with CO2Corr
        data['TEWE'][:, :, 0] = data['TEWG'][:, :, 0] * data['BTTC'][:, :, c3ti['14 CO2Emissions']]/1e6*1.2

        # Call the survival function routine.
        #data = survival_function(data, time_lag, histend, year, titles)

        if year == histend['TEWS']:

            # Calculate scrapping
            #data['REVS'][:, 0, 0] = sum(data['TESH'][:, :, 0]*data['TSFD'][:, :, 0], axis=1)

            #This corrects for higher emissions/fuel use at older age depending how fast the fleet has grown
            CO2Corr = np.ones(len(titles['RTI']))

            # Fuel use
            # Compute fuel use as distance driven times energy use, corrected by the biofuel mandate.
            emis_corr = np.zeros([len(titles['RTI']), len(titles['VTTI'])])
            fuel_converter = np.zeros([len(titles['VTTI']), len(titles['JTI'])])
            #fuel_converter = copy.deepcopy(data['TJET'][0, :, :])

            for r in range(len(titles['RTI'])):
                if data['RFLT'][r, 0, 0] > 0.0:
                    CO2Corr[r] = (data['TESH'][r, :, 0]*data['TESF'][r, :, 0]* \
                             data['TETH'][r, :, 0]).sum()/(data['TESH'][r, :, 0]*data['TESF'][r, :, 0]).sum()

                for fuel in range(len(titles['JTI'])):

                    for veh in range(len(titles['VTTI'])):

                        if titles['JTI'][fuel] == '5 Middle distillates' and data['TJET'][0, veh, fuel] ==1:  # Middle distillates

                            # Mix with biofuels/hydrogen if there's a biofuel mandate or hydrogen mandate
                            fuel_converter[veh, fuel] = data['TJET'][0, veh, fuel] * (1.0 - data['RBFM'][r, 0, 0])

                            # Emission correction factor
                            emis_corr[r, veh] = 1.0 - data['RBFM'][r, 0, 0]

                        elif titles['JTI'][fuel] == '11 Biofuels'  and data['TJET'][0, veh, fuel] == 1:

                            fuel_converter[veh, fuel] = data['TJET'][0, veh, fuel] * data['RBFM'][r, 0, 0]

                # Calculate fuel use - passenger car only! Therefore this will
                # differ from E3ME results
                # TEWG:                          km driven
                # Convert energy unit (1/41.868) ktoe/MJ
                # Energy demand (BBTC)           MJ/km

                data['TJEF'][r, :, 0] = (np.matmul(np.transpose(fuel_converter), data['TEWG'][r, :, 0]*\
                                        data['BTTC'][r, :, c3ti['9 energy use (MJ/km)']]*CO2Corr[r]/41.868))


                data['TVFP'][r, :, 0] = (np.matmul(fuel_converter/fuel_converter.sum(axis=0), data['TE3P'][r, :, 0]))*\
                                            data['BTTC'][r, :, c3ti['9 energy use (MJ/km)']] * \
                                                    CO2Corr[r]/ 41868

                # "Emissions"
                data['TEWE'][r, :, 0] = data['TEWG'][r, :, 0] * data['BTTC'][r, :, c3ti['14 CO2Emissions']]*CO2Corr[r]*emis_corr[r,:]/1e6




        # Calculate the LCOT for each vehicle type.
        # Call the function
        data = get_lcot(data, titles)

    # %% Simulation of stock and energy specs

    if year > histend['TEWS']:
        # TODO: Implement survival function to get a more accurate depiction of
        # vehicles being phased out and to be able to track the age of the fleet.
        # This means that a new variable will need to be implemented which is
        # basically TP_VFLT with a third dimension (vehicle age in years- up to 23y)
        # Reduced efficiences can then be tracked properly as well.

        # Create a local dictionary for timeloop variables
        # It contains values between timeloop interations in the FTT core
        data_dt = {}

        # First, fill the time loop variables with the their lagged equivalents
        for var in time_lag.keys():

            data_dt[var] = copy.deepcopy(time_lag[var])

        data_dt['TWIY'] = np.zeros([len(titles['RTI']), len(titles['VTTI']), 1])

        #Create the regulation variable
        isReg = np.zeros([len(titles['RTI']), len(titles['VTTI'])])
        isReg = np.where(data['TREG'][:, :, 0] > 0.0,
                          (np.tanh(1 +
                              (data_dt['TEWK'][:, :, 0] - data['TREG'][:, :, 0]) 
                                  / data['TREG'][:, :, 0])),
                          0.0)

        isReg[data['TREG'][:, :, 0] == -1.0] = 0.0
        isReg[data['TREG'][:, :, 0] == 0.0] = 1.0

        # Call the survival function routine.
        #data = survival_function(data, time_lag, histend, year, titles)

        # Total number of scrapped vehicles: #TP_TEOL changed to RVTS #Is this really needed?
        #data['RVTS'][:, 0, 0] = np.sum(data['REVS'][:, :, 0], axis=1)

        # Factor used to create quarterly data from annual figures
        no_it = int(data['noit'][0,0,0])
        dt = 1 / float(no_it)

        ############## Computing new shares ##################

        #Start the computation of shares
        for t in range(1, no_it+1):

            # Both rvkm and RFLT are exogenous at the moment
            # Interpolate to prevent staircase profile.
            rvkmt = time_lag['RVKM'][:, 0, 0] + (data['RVKM'][:, 0, 0] - time_lag['RVKM'][:, 0, 0]) * t * dt

            rfllt = time_lag['RFLT'][:, 0, 0] + (data['RFLT'][:, 0, 0] - time_lag['RFLT'][:, 0, 0]) * (t-1) * dt
            rfltt = time_lag['RFLT'][:, 0, 0] + (data['RFLT'][:, 0, 0] - time_lag['RFLT'][:, 0, 0]) * t * dt

            for r in range(len(titles['RTI'])):

                if rfltt[r] == 0.0:
                    continue

                ############################ FTT ##################################
                # Initialise variables related to market share dynamics
                # DSiK contains the change in shares
                dSik = np.zeros([len(titles['VTTI']), len(titles['VTTI'])])

                # F contains the preferences
                F = np.ones([len(titles['VTTI']), len(titles['VTTI'])])*0.5

#                    if int(data['TDA1'][r]) < year:

                # TODO: Check Specs dimensions
                #if np.any(specs[sector][r, :] == 1):  # FTT Specification

                for v1 in range(len(titles['VTTI'])):

                    if not (data_dt['TEWS'][r, v1, 0] > 0.0 and
                            data_dt['TELC'][r, v1, 0] != 0.0 and
                            data_dt['TLCD'][r, v1, 0] != 0.0):
                        continue

                    S_veh_i = data_dt['TEWS'][r, v1, 0]


                    for v2 in range(v1):

                        if not (data_dt['TEWS'][r, v2, 0] > 0.0 and
                                data_dt['TELC'][r, v2, 0] != 0.0 and
                                data_dt['TLCD'][r, v2, 0] != 0.0):
                            continue

                        S_veh_k = data_dt['TEWS'][r, v2, 0]
                        Aik = data['TEWA'][0,v1 , v2]/data['BTTC'][r, v1, c3ti['17 Turnover rate']]
                        Aki = data['TEWA'][0,v2 , v1]/data['BTTC'][r, v2, c3ti['17 Turnover rate']]

                        # Propagating width of variations in perceived costs
                        dFik = sqrt(2) * sqrt((data_dt['TLCD'][r, v1, 0]*data_dt['TLCD'][r, v1, 0] + data_dt['TLCD'][r, v2, 0]*data_dt['TLCD'][r, v2, 0]))

                        # Consumer preference incl. uncertainty
                        Fik = 0.5*(1+np.tanh(1.25*(data_dt['TELC'][r, v2, 0]-data_dt['TELC'][r, v1, 0])/dFik))

                        # Preferences are then adjusted for regulations
                        F[v1, v2] = Fik*(1.0-isReg[r, v1])* (1.0 - isReg[r, v2]) + isReg[r, v2]*(1.0-isReg[r, v1]) + 0.5*(isReg[r, v1]*isReg[r, v2]) 
                        F[v2, v1] = (1.0-Fik)*(1.0-isReg[r, v2]) * (1.0 - isReg[r, v1]) + isReg[r, v1]*(1.0-isReg[r, v2]) + 0.5*(isReg[r, v2]*isReg[r, v1]) 

                        if scenario == 'S0':

                            # Market share dynamics
                            dSik[v1, v2] = S_veh_i*S_veh_k* (Aik*F[v1,v2] - Aki*F[v2,v1])*dt
                            
                            dSik[v2, v1] = -dSik[v1, v2]

                        else:

                        #Runge-Kutta market share dynamiccs
                            k_1 = S_veh_i*S_veh_k* (Aik*F[v1,v2] - Aki*F[v2,v1])
                            k_2 = (S_veh_i+dt*k_1/2)*(S_veh_k-dt*k_1/2)* (Aik*F[v1,v2] - Aki*F[v2,v1])
                            k_3 = (S_veh_i+dt*k_2/2)*(S_veh_k-dt*k_2/2) * (Aik*F[v1,v2] - Aki*F[v2,v1])
                            k_4 = (S_veh_i+dt*k_3)*(S_veh_k-dt*k_3) * (Aik*F[v1,v2] - Aki*F[v2,v1])

                            dSik[v1, v2] = dt*(k_1+2*k_2+2*k_3+k_4)/6

                            dSik[v2, v1] = -dSik[v1, v2]

                #calculate temportary market shares and temporary capacity from endogenous results
                endo_shares = data_dt['TEWS'][r, :, 0] + np.sum(dSik, axis=1) 
                endo_capacity = endo_shares * rfltt[r, np.newaxis]

                Utot = rfltt[r]
                dSk = np.zeros([len(titles['VTTI'])])
                dUk = np.zeros([len(titles['VTTI'])])
                dUkTK = np.zeros([len(titles['VTTI'])])
                dUkREG = np.zeros([len(titles['VTTI'])])
                TWSA_scalar = 1.0

                # Check that exogenous sales additions aren't too large
                # As a proxy it can't be greater than 80% of the fleet size
                # divided by 13 (the average lifetime of vehicles)
                if (data['TWSA'][r, :, 0].sum() > 0.8 * rfltt[r] / 13):

                    TWSA_scalar = data['TWSA'][r, :, 0].sum() / (0.8 * rfltt[r] / 13)
                #Check endogenous capacity plus additions for a single time step does not exceed regulated capacity.
                reg_vs_exog = ((data['TWSA'][r, :, 0]*TWSA_scalar/no_it + endo_capacity) > data['TREG'][r, :, 0]) & (data['TREG'][r, :, 0] >= 0.0)
                data['TWSA'][r, :, 0] = np.where(reg_vs_exog, 0.0, data['TWSA'][r, :, 0])

                #TWSA is yearly capacity additions. We need to split it up based on the number of time steps, and also scale it if necessary.
                dUkTK =  data['TWSA'][r, :, 0]*TWSA_scalar/no_it

                # Correct for regulations due to the stretching effect. This is the difference in capacity due only to rflt increasing.
                # This will be the difference between capacity based on the endogenous capacity, and what the endogenous capacity would have been
                # if rflt (i.e. total demand) had not grown.

                dUkREG = -(endo_capacity - endo_shares*rfllt[r,np.newaxis])*isReg[r, :].reshape([len(titles['VTTI'])])
                           

                # Sum effect of exogenous sales additions (if any) with effect of regulations. 
                dUk = dUkTK + dUkREG
                dUtot = np.sum(dUk)

                # Convert to market shares 
                #Although capacity additions and regulations are in levels, we model in shares, 
                #so we need to calculate the change in shares, and then recalculate capacity.
                #This makes sure we still match total demand, and do not add/take away

                #Converting the changes in capacity to changes in shares will redistribute the shares based on the new capacity additions/substractions.
                #This is essentially a renormalisation of shares based on the additions.

                # dSk = dUk/Utot - Uk dUtot/Utot^2  (Chain derivative)
                dSk = np.divide(dUk, Utot) - endo_capacity*np.divide(dUtot, (Utot*Utot))

                #Correct endogenous market shares based on capacity additions and regulations              
                data['TEWS'][r, :, 0] = endo_shares + dSk



                if ~np.isclose(np.sum(data['TEWS'][r, :, 0]), 1.0, atol=1e-5):
                    msg = """Sector: {} - Region: {} - Year: {}
                    Sum of market shares do not add to 1.0 (instead: {})
                    """.format(sector, titles['RTI'][r], year, np.sum(data['TEWS'][r, :, 0]))
                    warnings.warn(msg)

                if np.any(data['TEWS'][r, :, 0] < 0.0):
                    msg = """Sector: {} - Region: {} - Year: {}
                    Negative market shares detected! Critical error!
                    """.format(sector, titles['RTI'][r], year)
                    warnings.warn(msg)



            ############## Update variables ##################

            # Update demand for driving (in km/ veh/ y) - exogenous atm
            #data['TEWL'][:, :, 0] = rvkmt[:, 0, 0]
            # Vehicle composition
            data['TEWK'][:, :, 0] = data['TEWS'][:, :, 0] * rfltt[:, np.newaxis]
            # Total distance driven per vehicle type
            data['TEWG'][:, :, 0] = data['TEWK'][:, :, 0] * rvkmt[:, np.newaxis] * 1e-3

            # Sales are the difference between fleet sizes and the addition of scrapped vehicles #TODO check this
            #data['TEWI'][:, :, 0] = (data['TEWK'][:, :, 0] - data_dt['TEWK'][:, :, 0])/dt


            #to match e3me:
            #data['TEWI'][:, :, 0] = (data['TEWK'][:, :, 0] - data_dt['TEWK'][:, :, 0])/dt  
            #copying ftt power
            data['TEWI'][:, :, 0] = (data['TEWK'][:, :, 0] - time_lag['TEWK'][:, :, 0])
            
            data['TEWI'][:, :, 0] = np.where(data['TEWI'][:, :, 0] < 0.0,
                                               np.where(data['BTTC'][:, :, c3ti['8 lifetime']] !=0.0,
                                                            divide(data_dt['TEWK'][:, :, 0],
                                                            data['BTTC'][:, :, c3ti['8 lifetime']]),0.0),
                                               data['TEWI'][:, :, 0]+ np.where(data['BTTC'][:, :, c3ti['8 lifetime']] !=0.0,
                                                            divide(data_dt['TEWK'][:, :, 0],
                                                            data['BTTC'][:, :, c3ti['8 lifetime']]),0.0))
            



            #This corrects for higher emissions/fuel use at older age depending how fast the fleet has grown
            CO2Corr = np.ones(len(titles['RTI']))
            # Fuel use
            # Compute fuel use as distance driven times energy use, corrected by the biofuel mandate.
            emis_corr = np.zeros([len(titles['RTI']), len(titles['VTTI'])])
            fuel_converter = np.zeros([len(titles['VTTI']), len(titles['JTI'])])
            #fuel_converter = copy.deepcopy(data['TJET'][0, :, :])

            for r in range(len(titles['RTI'])):
                if data['RFLT'][r, 0, 0] > 0.0:
                    CO2Corr[r] = (data['TESH'][r, :, 0]*data['TESF'][r, :, 0]* \
                             data['TETH'][r, :, 0]).sum()/(data['TESH'][r, :, 0]*data['TESF'][r, :, 0]).sum()

                for fuel in range(len(titles['JTI'])):

                    for veh in range(len(titles['VTTI'])):

                        if titles['JTI'][fuel] == '5 Middle distillates' and data['TJET'][0, veh, fuel] ==1:  # Middle distillates

                            # Mix with biofuels/hydrogen if there's a biofuel mandate or hydrogen mandate
                            fuel_converter[veh, fuel] = data['TJET'][0, veh, fuel] * (1.0 - data['RBFM'][r, 0, 0])

                            # Emission correction factor
                            emis_corr[r, veh] = 1.0 - data['RBFM'][r, 0, 0]

                        elif titles['JTI'][fuel] == '11 Biofuels'  and data['TJET'][0, veh, fuel] == 1:

                            fuel_converter[veh, fuel] = data['TJET'][0, veh, fuel] * data['RBFM'][r, 0, 0]

                # Calculate fuel use - passenger car only! Therefore this will
                # differ from E3ME results
                # TEWG:                          km driven
                # Convert energy unit (1/41.868) ktoe/MJ
                # Energy demand (BBTC)           MJ/km

                data['TJEF'][r, :, 0] = (np.matmul(np.transpose(fuel_converter), data['TEWG'][r, :, 0]*\
                                        data['BTTC'][r, :, c3ti['9 energy use (MJ/km)']]*CO2Corr[r]/41.868))


                data['TVFP'][r, :, 0] = (np.matmul(fuel_converter/fuel_converter.sum(axis=0), data['TE3P'][r, :, 0]))*\
                                            data['BTTC'][r, :, c3ti['9 energy use (MJ/km)']] * \
                                                    CO2Corr[r]/ 41868
                # "Emissions"
                data['TEWE'][r, :, 0] = data['TEWG'][r, :, 0] * data['BTTC'][r, :, c3ti['14 CO2Emissions']]*CO2Corr[r]*emis_corr[r,:]/1e6

       ############## Learning-by-doing ##################

            # Cumulative global learning
            # Using a technological spill-over matrix (TEWB) together with capacity
            # additions (TEWI) we can estimate total global spillover of similar
            # vehicals
#            bi = np.matmul(data['TEWI'][:, :, 0], data['TEWB'][0, :, :])
#            dw = np.sum(bi, axis=0)*dt

            bi = np.zeros((len(titles['RTI']),len(titles['VTTI'])))
            for r in range(len(titles['RTI'])):
                bi[r,:] = np.matmul(data['TEWB'][0, :, :],data['TEWI'][r, :, 0])
            dw = np.sum(bi, axis=0)*dt

            # Cumulative capacity incl. learning spill-over effects
            data['TEWW'][0, :, 0] = data_dt['TEWW'][0, :, 0] + dw

            # Copy over the technology cost categories that do not change (all except prices which are updated through learning-by-doing below)
            data['BTTC'] = copy.deepcopy(data_dt['BTTC'])

            # Learning-by-doing effects on investment
            for veh in range(len(titles['VTTI'])):

                if data['TEWW'][0, veh, 0] > 0.1:

                    data['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] = data_dt['BTTC'][:, veh, c3ti['1 Prices cars (USD/veh)']] * \
                                                                             (1.0 + data['BTTC'][:, veh, c3ti['16 Learning exponent']] * dw[veh]/data['TEWW'][0, veh, 0])

            # Investment in terms of car purchases:
            for r in range(len(titles['RTI'])):

                data['TWIY'][r, :, 0] = data_dt['TWIY'][r, :, 0] + data['TEWI'][r, :, 0]*dt*data['BTTC'][r, :, c3ti['1 Prices cars (USD/veh)']]/1.33


            # =================================================================
            # Update the time-loop variables
            # =================================================================

            #Calculate levelised cost again
            data = get_lcot(data, titles)

            #Update time loop variables:
            for var in data_dt.keys():

                data_dt[var] = copy.deepcopy(data[var])

    return data
