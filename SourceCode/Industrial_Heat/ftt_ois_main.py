# -*- coding: utf-8 -*-
"""
=========================================
ftt_ois_main.py
=========================================
Industrial other sectors FTT module.
###################################################


This is the main file for FTT: Industrial Heat - OIS, which models technological
diffusion of industrial heat processes within the other sectors due
to simulated investor decision making. Investors compare the **levelised cost of
industrial heat**, which leads to changes in the market shares of different technologies.

The outputs of this module include changes in final energy demand and emissions due
chemical heat processes for the EU28.

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros

Functions included:

    - solve
        Main solution function for the module
    - get_lcoih
        Calculates the levelised cost of industrial heat

"""
# Standard library imports
from math import sqrt
import os
import copy
import sys
import warnings
import time

# Third party imports
import pandas as pd
import numpy as np

# Local library imports
from SourceCode.support.divide import divide
# %% lcoh
# -----------------------------------------------------------------------------
# --------------------------- LCOH function -----------------------------------
# -----------------------------------------------------------------------------
def get_lcoih(data, titles, year):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of industrial heat in 2019 Euros
    It includes intangible costs (gamma values) and together
    determines the investor preferences.

    Parameters
    -----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) for all
        variables. Variable names are keys and the values are 3D NumPy arrays.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.

    Returns
    ----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.

    Notes
    ---------
    Additional notes if required.
    """

    # Categories for the cost matrix (BIC5)
    ctti = {category: index for index, category in enumerate(titles['CTTI'])}

    for r in range(len(titles['RTI'])):
        if data['IUD5'][r, :, 0].sum(axis=0)==0:
            continue

        # Cost matrix
        #BIC5 = data['BIC5'][r, :, :]

        lt = data['BIC5'][r,:, ctti['5 Lifetime (years)']]

        max_lt = int(np.max(lt))
        lt_mat = np.linspace(np.zeros(len(titles['ITTI'])), max_lt-1,
                             num=max_lt, axis=1, endpoint=True)
        lt_max_mat = np.concatenate(int(max_lt)*[lt[:, np.newaxis]], axis=1)
        mask = lt_mat < lt_max_mat
        lt_mat = np.where(mask, lt_mat, 0)


        # Capacity factor used in decisions (constant), not actual capacity factor #TODO ask about this
        cf = data['BIC5'][r,:, ctti['13 Capacity factor mean'], np.newaxis]

        #conversion efficiency
        ce = data['BIC5'][r,:, ctti['9 Conversion efficiency'], np.newaxis]

        # Trap for very low CF
        cf[cf<0.000001] = 0.000001

        # Factor to transfer cost components in terms of capacity to generation
#        ones = np.ones([len(titles['ITTI']), 1])
        conv = 1/(cf)/8766 #number of hours in a year

        # Discount rate
        # dr = data['BIC5'][r,6]
        dr = data['BIC5'][r,:, ctti['8 Discount rate'], np.newaxis]

        # Initialse the levelised cost components
        # Average investment cost
        it = np.zeros([len(titles['ITTI']), int(max_lt)])
        it[:, 0, np.newaxis] =  data['BIC5'][r,:, ctti['1 Investment cost mean (MEuro per MW)'], np.newaxis] * conv*(1*10^6)


        # Standard deviation of investment cost
        dit = np.zeros([len(titles['ITTI']), int(max_lt)])
        dit[:, 0, np.newaxis] =  data['BIC5'][r,:, ctti['2 Investment cost SD'], np.newaxis] * conv*(1*10^6)


        # Subsidies as a percentage of investment cost
        st = np.zeros([len(titles['ITTI']), int(max_lt)])
        st[:, 0, np.newaxis] = (data['BIC5'][r,:, ctti['1 Investment cost mean (MEuro per MW)'], np.newaxis]
             * data['ISB5'][r, :, 0,np.newaxis] * conv)*(1*10^6)


        # Average fuel costs 2010Euros/toe to euros/MWh 1 toe = 11.63 MWh
        ft = np.ones([len(titles['ITTI']), int(max_lt)])
        ft = ft * data['BIC5'][r,:, ctti['10 Fuel cost mean'], np.newaxis]/11.63/ce
        ft = np.where(mask, ft, 0)

        # Standard deviation of fuel costs
        dft = np.ones([len(titles['ITTI']), int(max_lt)])
        dft = dft * data['BIC5'][r,:, ctti['11 Fuel cost SD'], np.newaxis]/11.63/ce
        dft = np.where(mask, dft, 0)

        #fuel tax/subsidies
        ftt = np.ones([len(titles['ITTI']), int(max_lt)])
        ftt = ftt * data['IFT5'][r,:, 0, np.newaxis]/ce
        ftt = np.where(mask, ft, 0)

        # Fixed operation & maintenance cost - variable O&M available but not included
        omt = np.ones([len(titles['ITTI']), int(max_lt)])
        omt = omt * data['BIC5'][r,:, ctti['3 O&M cost mean (Euros/MJ/s/year)'], np.newaxis]*conv #(euros per MW) in a year
        omt = np.where(mask, omt, 0)

        # Standard deviation of operation & maintenance cost
        domt = np.ones([len(titles['ITTI']), int(max_lt)])
        domt = domt * data['BIC5'][r,:, ctti['4 O&M cost SD'], np.newaxis]*conv
        domt = np.where(mask, domt, 0)



        # Net present value calculations
        # Discount rate
        denominator = (1+dr)**lt_mat

        # 1-Expenses
        # 1.1-Without policy costs
        npv_expenses1 = (it+ft+omt)/denominator
        # 1.2-With policy costs
        npv_expenses2 = (it+st+ft+ftt+omt)/denominator
        # 1.3-Only policy costs
        #npv_expenses3 = (st+fft-fit)/denominator
        # 2-Utility
        npv_utility = 1/denominator
        #Remove 1s for tech with small lifetime than max
        npv_utility[npv_utility==1] = 0
        npv_utility[:,0] = 1
        # 3-Standard deviation (propagation of error)
        npv_std = np.sqrt(dit**2 + dft**2 + domt**2)/denominator

        # 1-levelised cost variants in $/pkm
        # 1.1-Bare LCOT

        lcoe = np.sum(npv_expenses1, axis=1)/np.sum(npv_utility, axis=1)

        # 1.2-LCOT including policy costs
        tlcoe = np.sum(npv_expenses2, axis=1)/np.sum(npv_utility, axis=1)#+data['IEFI'][r, :, 0]
        # 1.3 LCOE excluding policy, including co2 price
        #lcoeco2 = np.sum(npv_expenses3, axis=1)/np.sum(npv_utility, axis=1)
        # 1.3-LCOT of policy costs
        # lcoe_pol = np.sum(npv_expenses3, axis=1)/np.sum(npv_utility, axis=1)+data['MEFI'][r, :, 0]
        # Standard deviation of LCOT
        dlcoe = np.sum(npv_std, axis=1)/np.sum(npv_utility, axis=1)

        # LCOE augmented with gamma values, no gamma values yet
        tlcoeg = tlcoe+data['IAM5'][r, :, 0]

        # Pass to variables that are stored outside.
        data['ILC5'][r, :, 0] = lcoe            # The real bare LCOT without taxes (euros/mwh)
        #data['IHLT'][r, :, 0] = tlcoe           # The real bare LCOE with taxes
        data['ILG5'][r, :, 0] = tlcoeg         # As seen by consumer (generalised cost)
        data['ILD5'][r, :, 0] = dlcoe          # Variation on the LCOT distribution



    return data

#Final energy demand has to match IEA

# %% main function
# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, iter_lag, titles, histend, year, domain):#, #specs, converter, coefficients):
    """

    Main solution function for the module.

    Simulates investor decision making.

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


    """

    # Categories for the cost matrix (BIC5)
    ctti = {category: index for index, category in enumerate(titles['CTTI'])}

    sector = 'Metals, transport and machinery equipment'

    #Get fuel prices from E3ME and add them to the data for this code
    #Initialise everything #TODO

    #Calculate or read in FED
    #Calculate historical emissions
    data = get_lcoih(data, titles, year)

    # Endogenous calculation takes over from here
    if year > histend['IUD5']:

        # Create a local dictionary for timeloop variables
        # It contains values between timeloop interations in the FTT core
        data_dt = {}

        # First, fill the time loop variables with the their lagged equivalents
        for var in time_lag.keys():

            data_dt[var] = copy.deepcopy(time_lag[var])

        # Create the regulation variable #Regulate capacity #no regulations yet, isReg full of zeros
        division = divide((data_dt['IWK5'][:, :, 0] - data['IRG5'][:, :, 0]), data_dt['IRG5'][:, :, 0]) # 0 when dividing by 0
        isReg = 0.5 + 0.5*np.tanh(1.5+10*division)
        isReg[data['IRG5'][:, :, 0] == 0.0] = 1.0
        isReg[data['IRG5'][:, :, 0] == -1.0] = 0.0


        # Factor used to create quarterly data from annual figures
        no_it = 4
        dt = 1 / no_it
        kappa = 10 #tech substitution constant

        ############## Computing new shares ##################
        IUD5tot = data['IUD5'][:, :, 0].sum(axis=1)
        #Start the computation of shares
        for t in range(1, no_it+1):

            # Interpolate to prevent staircase profile.
            #Time lagged UED plus change in UED * (no of iterations) * dt

            IUD5t = time_lag['IUD5'][:, :, 0].sum(axis=1) + (IUD5tot - time_lag['IUD5'][:, :, 0].sum(axis=1)) * t * dt
            IUD5lt = time_lag['IUD5'][:, :, 0].sum(axis=1) + (IUD5tot - time_lag['IUD5'][:, :, 0].sum(axis=1)) * (t-1) * dt

            for r in range(len(titles['RTI'])):

                if IUD5t[r] == 0.0:
                    continue



            ############################ FTT ##################################

                # DSiK contains the change in shares
                dSik = np.zeros([len(titles['ITTI']), len(titles['ITTI'])])

                # F contains the preferences
                F = np.ones([len(titles['ITTI']), len(titles['ITTI'])])*0.5

                # Market share constraints
                Gijmax = np.ones(len(titles['ITTI']))
                #Gijmin = np.ones((t2ti))

                # -----------------------------------------------------
                # Step 1: Endogenous EOL replacements
                # -----------------------------------------------------
                for b1 in range(len(titles['ITTI'])):

                    if  not (data_dt['IWS5'][r, b1, 0] > 0.0 and
                             data_dt['ILG5'][r, b1, 0] != 0.0 and
                             data_dt['ILD5'][r, b1, 0] != 0.0):
                        continue

                    #TODO: create market share constraints
                    Gijmax[b1] = np.tanh(1.25*(data_dt['ISC5'][r, b1, 0] - data_dt['IWS5'][r, b1, 0])/0.1)
                    #Gijmin[b1] = np.tanh(1.25*(-mes2_dt[r, b1, 0] + mews_dt[r, b1, 0])/0.1)



                    S_i = data_dt['IWS5'][r, b1, 0]


                    for b2 in range(b1):

                        if  not (data_dt['IWS5'][r, b2, 0] > 0.0 and
                                 data_dt['ILG5'][r, b2, 0] != 0.0 and
                                 data_dt['ILD5'][r, b2, 0] != 0.0):
                            continue

                        S_k = data_dt['IWS5'][r,b2, 0]
                        Aik = data['IWA5'][0,b1 , b2]*kappa
                        Aki = data['IWA5'][0,b2, b1]*kappa

                        # Propagating width of variations in perceived costs
                        dFik = sqrt(2) * sqrt((data_dt['ILD5'][r, b1, 0]*data_dt['ILD5'][r, b1, 0] + data_dt['ILD5'][r, b2, 0]*data_dt['ILD5'][r, b2, 0]))

                        # Consumer preference incl. uncertainty
                        Fik = 0.5*(1+np.tanh(1.25*(data_dt['ILG5'][r, b2, 0]-data_dt['ILG5'][r, b1, 0])/dFik))

                        # Preferences are then adjusted for regulations
                        F[b1, b2] = Fik*(1.0-isReg[r, b1]) * (1.0 - isReg[r, b2]) + isReg[r, b2]*(1.0-isReg[r, b1]) + 0.5*(isReg[r, b1]*isReg[r, b2])
                        F[b2, b1] = (1.0-Fik)*(1.0-isReg[r, b2]) * (1.0 - isReg[r, b1]) + isReg[r, b1]*(1.0-isReg[r, b2]) + 0.5*(isReg[r, b2]*isReg[r, b1])


                        #Runge-Kutta market share dynamiccs
                        k_1 = S_i*S_k * (Aik*F[b1, b2]*Gijmax[b1] - Aki*F[b2, b1]*Gijmax[b2])
                        k_2 = (S_i+dt*k_1/2)*(S_k-dt*k_1/2)* (Aik*F[b1, b2]*Gijmax[b1] - Aki*F[b2, b1]*Gijmax[b2])
                        k_3 = (S_i+dt*k_2/2)*(S_k-dt*k_2/2) * (Aik*F[b1, b2]*Gijmax[b1] - Aki*F[b2, b1]*Gijmax[b2])
                        k_4 = (S_i+dt*k_3)*(S_k-dt*k_3) * (Aik*F[b1, b2]*Gijmax[b1] - Aki*F[b2, b1]*Gijmax[b2])

                        #This method currently applies RK4 to the shares, but all other components of the equation are calculated for the overall time step
                        #We must assume the the LCOE does not change significantly in a time step dt, so we can focus on the shares.

                        dSik[b1, b2] = dt*(k_1+2*k_2+2*k_3+k_4)/6
                        dSik[b2, b1] = -dSik[b1, b2]

                        #dSik[b1, b2] = S_i*S_k* (Aik*F[b1,b2]*Gijmax[b1] - Aki*F[b2,b1]*Gijmax[b2])*dt
                        #dSik[b2, b1] = -dSik[b1, b2]

                # calculate temporary market shares and temporary capacity from endogenous results
                endo_shares = data_dt['IWS5'][r, :, 0] + np.sum(dSik, axis=1) 
                endo_ued = endo_shares * IUD5t[r, np.newaxis]


                # -----------------------------------------------------
                # Step 3: Exogenous sales additions
                # -----------------------------------------------------
                # Add in exogenous sales figures. These are blended with endogenous result!


                # Add in exogenous sales figures. These are blended with
                # endogenous result! Note that it's different from the
                # ExogSales specification!
                Utot = IUD5t[r]
                
                dSk = np.zeros((len(titles['ITTI'])))
                dUk = np.zeros((len(titles['ITTI'])))
                dUkTK = np.zeros((len(titles['ITTI'])))
                dUkREG = np.zeros((len(titles['ITTI'])))

                # Convert exogenous share changes to capacity/useful energy demand. They do not need to sum to one.
                dUkTK = data['IXS5'][r, :, 0]*Utot/no_it

                #Check endogenous capacity plus additions for a single time step does not exceed regulated capacity.
                #Convert ued to capcity for this check
                reg_vs_exog = ((dUkTK + endo_ued)/data['BIC5'][r, :, ctti["13 Capacity factor mean"]]/8766 > data['IRG5'][r, :, 0]) & (data['IRG5'][r, :, 0] >= 0.0)
                dUkTK = np.where(reg_vs_exog, 0.0, dUkTK)


                # Correct for regulations due to the stretching effect. This is the difference in ued due only to demand increasing.
                # This will be the difference between ued based on the endogenous ued, and what the endogenous ued would have been
                # if total demand had not grown.

                dUkREG = -(endo_ued - endo_shares*IUD5lt[r,np.newaxis])*isReg[r, :].reshape([len(titles['ITTI'])])


                # Sum effect of exogenous sales additions (if any) with
                # effect of regulations
                #Note that the share of indirect heating vs direct must be preserved
                indirect_cut_off = 7
                dUk = dUkREG + dUkTK
                dUtot_indirect = np.sum(dUk[:indirect_cut_off])
                dUtot_direct = np.sum(dUk[indirect_cut_off:])

                indirect_shares = divide((endo_ued[:indirect_cut_off] + dUk[:indirect_cut_off]),(np.sum(endo_ued[:indirect_cut_off])+dUtot_indirect))
                direct_shares = divide((endo_ued[indirect_cut_off:] + dUk[indirect_cut_off:]),(np.sum(endo_ued[indirect_cut_off:])+dUtot_direct))
                indirect_weighting = np.sum(endo_shares[:indirect_cut_off])
                

                # Calaculate changes to endogenous ued, and use to find new market shares
                # Zero ued will result in zero shares
                # All other ueds will be streched



                data['IWS5'][r, :indirect_cut_off, 0] = indirect_shares*indirect_weighting
                data['IWS5'][r, indirect_cut_off:, 0] = direct_shares*(1-indirect_weighting)


                if ~np.isclose(np.sum(data['IWS5'][r, :, 0]), 1.0, atol=1e-5):
                    msg = """Sector: {} - Region: {} - Year: {}
                    Sum of market shares do not add to 1.0 (instead: {})
                    """.format(sector, titles['RTI'][r], year, np.sum(data['IWS5'][r, :, 0]))
                    warnings.warn(msg)

                if np.any(data['IWS5'][r, :, 0] < 0.0):
                    msg = """Sector: {} - Region: {} - Year: {}
                    Negative market shares detected! Critical error!
                    """.format(sector, titles['RTI'][r], year)
                    warnings.warn(msg)




            # =============================================================
            #  Update variables
            # =============================================================

            #TODO: what else needs to go here? TODO calculate new capacity and new yearly capacity change

            #Useful heat by technology, calculate based on new market shares #Regional totals
            data['IUD5'][:, :, 0] = data['IWS5'][:, :, 0]* IUD5t[:, np.newaxis]

            # Capacity by technology
            data['IWK5'][:, :, 0] = divide(data['IUD5'][:, :, 0],
                                              data['BIC5'][:, :, ctti["13 Capacity factor mean"]]*8766)
            #add number of devices replaced due to breakdowns = IWK4_lagged/lifetime to yearly capacity additions
            #note some values of IWI4 negative
            data["IWI1"][:, :, 0] = 0
            for r in range(len(titles['RTI'])):
                for tech in range(len(titles['ITTI'])):
                    if(data['IWK5'][r, tech, 0]-time_lag['IWK5'][r, tech, 0]) > 0:
                        data['IWI5'][r, tech, 0] = (data['IWK5'][r, tech, 0]-time_lag['IWK5'][r, tech, 0])
            data['IWI5'][:, :, 0] = data['IWI5'][:, :, 0] + np.where(data['BIC5'][:, :, ctti['5 Lifetime (years)']] !=0.0,
                                                                        divide(time_lag['IWK5'][:, :, 0],data['BIC5'][:, :, ctti['5 Lifetime (years)']]),
                                                                        0.0)

            #Update emissions
            #IHW4 is the global average emissions per unit of UED (GWh). IHW4 has units of kt of CO2/GWh
            for r in range(len(titles['RTI'])):
                data['IWE5'][r, :, 0] = data['IUD5'][r, :, 0] * data['IHW5'][0, :, 0]


            #Final energy by technology
            data['IFD5'][:, :, 0] = np.where(data['BIC5'][:, :, ctti["9 Conversion efficiency"]] !=0.0,
                                             divide(data['IUD5'][:, :, 0],
                                                    data['BIC5'][:, :, ctti["9 Conversion efficiency"]]),
                                            0.0)



            # =============================================================
            # Learning-by-doing
            # =============================================================

            # Cumulative global learning
            # Using a technological spill-over matrix (IEWB spillover matrix) together with capacity
            # additions (IWI4 Capacity additions) we can estimate total global spillover of similar
            # techicals





            bi = np.zeros((len(titles['RTI']),len(titles['ITTI'])))
            for r in range(len(titles['RTI'])):
                bi[r,:] = np.matmul(data['IWB5'][0, :, :],data['IWI5'][r, :, 0])
            dw = np.sum(bi, axis=0)*dt

            # # Cumulative capacity incl. learning spill-over effects
            data['IWW5'][0, :, 0] = data_dt['IWW5'][0, :, 0] + dw
            #
            # # Copy over the technology cost categories that do not change (all except prices which are updated through learning-by-doing below)
            data['BIC5'] = copy.deepcopy(data_dt['BIC5'])
            #
            # # Learning-by-doing effects on investment
            for tech in range(len(titles['ITTI'])):

                if data['IWW5'][0, tech, 0] > 0.1:

                    data['BIC5'][:, tech, ctti['1 Investment cost mean (MEuro per MW)']] = data_dt['BIC5'][:, tech, ctti['1 Investment cost mean (MEuro per MW)']] * \
                                                                           (1.0 + data['BIC5'][:, tech, ctti['15 Learning exponent']] * dw[tech]/data['IWW5'][0, tech, 0])

            # =================================================================
            # Update the time-loop variables
            # =================================================================

            #Calculate levelised cost again
            data = get_lcoih(data, titles, year)

            #Update time loop variables:
            for var in data_dt.keys():

                data_dt[var] = copy.deepcopy(data[var])


    return data
