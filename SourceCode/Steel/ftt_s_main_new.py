# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:30 2024

@author: WRI India, Femke, Cormac

=========================================
ftt_s_main.py
=========================================
Steel production module.
####################################

This is the main file for FTT: Steel, which models technological
diffusion of residential heating technologies due to simulated consumer decision making.
Consumers compare the **levelised cost of heat**, which leads to changes in the
market shares of different technologies.

The outputs of this module include changes in final energy demand and boiler sales.

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros
    - `estimation <econometrics_functions.html>`__
        Predict future values according to the estimated coefficients.

Functions included:
    - solve
        Main solution function for the module
    - get_lcoh
        Calculate levelised cost of residential heating

"""
# Standard library imports
import csv
from math import sqrt
import copy
import os
import warnings
import pandas as pd

# Third party imports
import numpy as np

# Local library imports
from SourceCode.support.divide import divide
from SourceCode.ftt_core.ftt_s_sales_or_investments import get_sales
from SourceCode.Steel.ftt_s_lcos import get_lcos
from SourceCode.Steel.ftt_s_scrap import scrap_calc
from SourceCode.Steel.ftt_s_rawmaterials import raw_material_distr
from SourceCode.Steel.ftt_s_fuel_consumption import ftt_s_fuel_consumption

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
        Description
    iter_lag: tytpe
        Description
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of historical data by variable
    year: int
        Current/active year of solution
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

    # Categories for the cost matrix (BSTC)
    c5ti = {category: index for index, category in enumerate(titles['C5TI'])}
    stti = {category: index for index, category in enumerate(titles['STTI'])}
    # Fuels
    jti = {category: index for index, category in enumerate(titles['JTI'])}

    sector = 'steel'
    no_it = data["noit"][0, 0, 0]
    iteration = 1
    data = scrap_calc(data, time_lag, titles, year)
    data['BSTC'] = copy.deepcopy(time_lag['BSTC'])
    # Historical data currently ends in 2019, so we need to initialise data
    # Simulation period starts in 2020   # Calculate capacities (SEWK)
    # Convert strings to lists of values

    

    if year <= histend['SEWG']:   ## histend['SEWG'] = 2019

    # Connect historical data to future projections (only for DATE == 2014)

        for r in range(len(titles['RTI'])):
            # data['SEWG'] is in historic sheets, so no need to calculate that.
            
            # Capacity (kton) (11th are capacity factors) 
            data['SEWK'][r, :, 0] = 0.0
            data['SEWK'][r, :, 0] = data['SEWG'][r, :, 0] / data['BSTC'][r, :, c5ti["CF"]]
            data['SPSA'][r, 0, 0] = sum(data['SEWG'][r, :, 0])  

            # 'Historical' employment in th FTE 
            data['SEMS'][r, :, 0] = data['SEWK'][r, :, 0] * data['BSTC'][r, :, c5ti["Employment"]] * 1.1
        
            # In this preliminary model SPSP is historical production while SPSA is exogenous future production (based on E3ME baseline)
            # Total steel production by region (kton/y) = demand
            # SPSP[j] = sum(data['SEWG'][:, j])

            # Market capacity shares of steelmaking technologies: 
            data['SEWS'][r, :, 0] = 0
            data['SEWS'] [r, :, 0] = np.divide (data['SEWK'][r, :, 0], np.sum(data['SEWK'][r, :, 0]),
                                                where =(data['SPSP'][r, 0, 0] > 0.0) and (data['SEWK'][r, :, 0] > 0.0)) 
            
            # Emissions (MtCO2/y) (13th is emissions factors tCO2/tcs)
            # A crude backwards calculation of emissions using simple emission factors keep
            data['SEWE'][r, :, 0] = data['SEWG'][r, :, 0] * data['BSTC'][r, :, c5ti["EF"]] / 1000

            # Regional average energy intensity (GJ/tcs) 
            data['STEI'][r, :, 0] = data['BSTC'][r, :, c5ti["Energy Intensity"]]
            data['SEIA'][r, 0, 0] = sum(data['STEI'][r, :, 0] * data['SEWS'][r, :, 0])

    if year == 2019:   ## change from fortran
        data['SPSA'] = data['SPSP']
        data = raw_material_distr(data, titles, year, 1)
    
        # Calculate fuel use (SJEF)
        #Set
        sewg_sum = np.sum(data["SEWG"], axis=1)
        og_base = np.zeros_like(sewg_sum)

        ## For technologies from 1 to 7
        og_base[sewg_sum > 0.0] = np.sum(data["SEWG"][:, 0:7], axis=1)[sewg_sum > 0.0] / sewg_sum[sewg_sum > 0.0] 
        og_sim = og_base
        #ccs_share = 0.0

        ### start 
        ccs_share = np.zeros(len(titles['RTI']))
        SJEF = np.zeros(len(titles['RTI']))
        SJCO = np.zeros(len(titles['RTI']))

        for r in range(len(titles['RTI'])):
            # Regional average energy intensity (GJ/tcs)
            data['SEIA'][r] = np.sum(data['STEI'][r,:] * data['SEWS'][r,:])
            
            if (data['SPSA'][r]).any() > 0.0:
                if np.sum(data['SEWG'][r,:]) > 0.0:
                    # Calculate og_sim
                    og_sim[r] = np.sum(data['SEWG'][r, 0:19]) / np.sum(data['SEWG'] [r,:])
                    
                    # Calculate ccs_share
                    ccs_share[r] = (
                        np.sum(data['SEWG'][r,3:7]) + np.sum(data['SEWG'][r, 9:11]) + 
                        np.sum(data['SEWG'][r, 13:15]) + np.sum(data['SEWG'][r, 17:19]) + 
                        np.sum(data['SEWG'][r, 21:23])
                    ) / np.sum(data['SEWG'][r, :])
        ### end

        data['SJEF'], data['SJCO']= ftt_s_fuel_consumption(data['BSTC'], data['SEWG'], data['SMED'], len(titles['RTI']), len(titles['STTI']), c5ti, len(titles['JTI']))
    
        data['SJFR'] = data['SJEF']

        #Call the LCOS function within the year==2019 conditional statement
        data = get_lcos(data, titles) 
            
        # Calculate cumulative capacities (SEWW)

        
        bi = np.zeros((len(titles['RTI']), len(titles['STTI'])))

        ## change : loop can use r from the outer loop
        for r in range(len(titles['RTI'])):
            bi[r,:] = np.matmul(data['SEWB'][0, :, :], data['SEWK'][r, :, 0])   ## matmul : matrix multiplication
            data['SEWW'] = np.sum (bi, axis = 0)
        
        data['SEWW'] = data['SEWW'][None, :, None]

        #This needs to be in the year==2019 conditional statement
        #These 2 loops happen twice. Let's just make it a separate function and call it here and in year > histend
        #Can be modified to make shorter/more efficient
        for t1 in range(len(titles['STTI'])): 
            for t2 in range(len(titles['SSTI'])):
                if data['STIM'][0, t1, t2] == 1:
                    if (t2 < 8): 
                        data['SICA'][:, t2, 0] = data['SICA'][:, t2, 0] + 1.1 * data['SEWW'][0, t1, 0] * np.sum(data['BSTC'][:, t1, 25+t2])/np.count_nonzero(data['SPSA'][:, :, 0])
                
                    elif (t2 > 7 and t2 < 21):
                            data['SICA'][:, t2, 0] = data['SICA'][:, t2, 0] + 1.1 * data['SEWW'][: , t1 , 0] 
                    # Estimate installed capacities of steelmaking plants
                    elif (t2 > 20 and t2 < 27): 
                        data['SICA'][:, t2, 0] = data['SICA'][:, t2, 0] + data['SEWW'][:, t1, 0] 
                    # Estimate installed capacities of finishing plants. 
                    # Note that after this step it's not crude steel anymore. Therefore it is divided by 1.14  
                    elif (t2 == 27):
                        data['SICA'][:, t2, 0] = data['SICA'][:, t2, 0] + data['SEWW'][:, t1, 0] /1.14 
        
        #Check this statement. It's using BSTC (lifetime)
        data['SEWI'][: , : , 0] = np.where((data['BSTC'][: , : , 5] > 0.0) , (data['SEWI'][: , : , 0] + (data['SWKL'][: , :, 0]/data['BSTC'][: , : , 5])) , np.max((data['SEWK'][: , : , 0] - data['SWKL'][: , : , 0]), 0))
    
                                             
    elif year > histend["SEWG"]:  

        no_it = int(data['noit'][0, 0, 0]) 
        dt = 1 / float(no_it)
        invdt = no_it
        tScaling = 10.0
        isReg = np.zeros((len(titles['RTI']), len(titles['STTI']),1))
        # Apply the first condition where SEWR > 0.0
        isReg = np.where(data['SEWR'] > 0.0, 1.0 + np.tanh(1.5 + 10 * (data['SWKL'] - data['SEWR']) / data['SEWR']), isReg)

        # Apply the second condition where SEWR == 0.0
        isReg = np.where(data['SEWR'] == 0.0, 1.0, isReg)
        data['SPSA'] = data['SPSP']   ## intilialization of SPSA

        for t in range(1, no_it):
            
            # Create a local dictionary for timeloop variables
            # It contains values between timeloop interations in the FTT core
            data_dt = {}

            # First, fill the time loop variables with the their lagged equivalents
            for var in time_lag.keys():
                data_dt[var] = copy.copy(time_lag[var])    ## change : increasing time complexity and space complexity
            #data = raw_material_distr(data, titles, year, t)
            # Market share calculation : What Shruti has done
                    
        # print(data_dt.keys())
        # --------------------------------------------------------------------------
        # !--------------FTT SIMULATION-----------------------------------------------
        # !---------------------------------------------------------------------------
        # !------TIME LOOP!!: we calculate quarterly: t=1 means the end of the first quarter   
        if (iteration==1):  ## change 
                ## change : loop, for t in loop of range(no_it)
                for t in range(0, (invdt - 1)):
                    spsa_dt = 0.0
                    spsa_dtl = 0.0         

                    # Time-step of steel demand
                    spsa_dt = data['SPSL'][:, 0, 0] + data['SPSA'][:, 0, 0] - data['SPSL'][:, 0, 0] * t/invdt    
                    spsa_dtl = data['SPSL'][:, 0, 0] + data['SPSA'][:, 0, 0] - data['SPSL'][:, 0, 0] * (t-1)/invdt    ## shape (71,)


                    #If there's not enough scrap to supply the scrap route or there's not enough iron supply to meet the gap of scrap supply, then regulate scrap. 
                    #This is more of a weak regulation. 
                    
                    # Calculate the condition for the first WHERE statement
                    condition1 = (~(data['SEWR'][:, 25,0] > data_dt['SEWK'][:, 25,0])) & (data['SEWR'][:, 25,0] > -1.0)

                    # Apply the first WHERE statement
                    isReg[:, 25,0] = np.where(condition1, 1.0 - np.tanh(2 * 1.25 * data_dt['BSTC'][:, stti['Scrap - EAF'], c5ti['Scrap']] - 0.5) / 0.5, isReg[:, 25,0])

                    # Calculate the condition for the second WHERE statement
                    condition2 = (isReg[:, 25] > 1.0)

                    # Apply the second WHERE statement
                    isReg[:, 25] = np.where(condition2, 1.0, isReg[:, 25])
                                                            
                        
            ############################ FTT ##################################
            #                        t3 = time.time()
            #                        print("Solving {}".format(titles["RTI"][r]))
            # Initialise variables related to market share dynamics
            # DSij contains the change in shares           
                    for r in range(len(titles['RTI'])):    
                        ## change
                        dSij = np.zeros([len(titles['STTI']), len(titles['STTI'])])   
                        F = np.ones([len(titles['STTI']), len(titles['STTI'])]) * 0.5
                        FE = np.ones([len(titles['STTI']), len(titles['STTI'])]) * 0.5
                                    
                        for b1 in range(len(titles['STTI'])): 
                            if  not (data['SEWS'][r, b1, 0] > 0.0 and
                                data['SGC1'][r, b1, 0] != 0.0 and
                                data['SEWC'][r, b1, 0] != 0.0):
                                continue
                                    
                            S_i = data['SEWS'][r, b1, 0]
                                    
                            for b2 in range(b1):                 
                                if not (data['SEWS'][r, b2, 0] > 0.0 and
                                    data['SGC1'][r, b2, 0] != 0.0 and 
                                    data['SEWC'][r, b2, 0] != 0.0): 
                                    continue                      
                                        
                                S_k = data['SEWS'][r, b2, 0]
                                    
                                # Propagating width of variations in perceived costs
                                dFij = 1.414 * sqrt((data_dt['SEWC'][r, b1, 0] * data_dt['SEWC'][r, b1, 0] 
                                                    + data_dt['SEWC'][r, b2, 0] * data_dt['SEWC'][r, b2, 0]))
                                        
                                # Investor preference incl. uncertainty
                                Fij = 0.5 * (1 + np.tanh(1.25 * (data_dt['SGC1'][r, b2, 0]
                                            - data_dt['SGC1'][r, b1, 0])) / dFij)
                                        
                                # Preferences are then adjusted for regulations ## change
                                F[b1, b2] = Fij * (1.0 - isReg[r, b1]) * (1.0 - isReg[r,b1]) + isReg[r, b2] \
                                                            * (1.0 - isReg[r, b1]) + 0.5 * (isReg[r, b1] * isReg[r, b2])
                                F[b2, b1] = (1.0 - Fij) * (1.0 - isReg[r, b2]) * (1.0 - isReg[r, b1]) + isReg[r, b1] \
                                                            * (1.0 - isReg[r, b2]) + 0.5 * (isReg[r, b2] * isReg[r, b1])
                                        
                                #Runge-Kutta market share dynamiccs
                                k_1 = S_i * S_k * (data['SEWA'][0, b1, b2] * F[b1,b2] - data['SEWA'][0, b2, b1]* F[b2,b1])
                                k_2 = (S_i + dt * k_1/2) * (S_k - dt * k_1/2) * (data['SEWA'][0, b1, b2] * F[b1,b2] - data['SEWA'][0, b2, b1] * F[b2,b1])
                                k_3 = (S_i + dt * k_2/2) * (S_k - dt * k_2/2) * (data['SEWA'][0, b1, b2] * F[b1,b2]- data['SEWA'][0, b2, b1]  * F[b2,b1])
                                k_4 = (S_i + dt * k_3) * (S_k - dt * k_3) * (data['SEWA'][0, b1, b2] * F[b1,b2] - data['SEWA'][0, b2, b1] * F[b2,b1])
                                        
                                dSij[b1, b2] = dt * (k_1+2*k_2+2*k_3+k_4)/6/tScaling   ## change
                                dSij[b2, b1] = -dSij[b1, b2]
                
                        # # -----------------------------------------------------
                        # # Step 2: Endogenous premature replacements
                        # # -----------------------------------------------------
                        # # Initialise variables related to market share dynamics
                        # #  DSiK contains the change in shares
                        # # !This is due to more plants being eligible for scrapping. However, the investor preference should be much lower than the one from above.
                        # SR = np.zeros(len(titles['RTI']))
                        # SR = 1/data['BSTC'][r, :, c5ti['Payback period']] - 1/data['BSTC'][r, :, c5ti['Lifetime']]
                            
                        # # Constant used to reduce the magnitude of premature scrapping 
                        # SR_C = 2.5
                        # #!Only calculate for non-zero shares (SWSLt>0), only if scrapping decision rate > 0
                        dSEij = np.zeros([len(titles['STTI']), len(titles['STTI'])])
                
                        
                    #  !------End of RK4 Alogrithm!------------------------------------------------------------------------!                                             
                    # Calulate endogenous shares!
                        
                        endo_shares = data_dt['SWSL'][r, :, 0] + np.sum(dSij, axis=1) + np.sum(dSEij, axis=1)  ## shape (26,)
                        endo_eol =  np.sum(dSij, axis=1) 
                        endo_capacity = data_dt['SPSA'][r, :, 0]/np.sum(endo_shares * data['BSTC'][r, :, c5ti['CF']]) * endo_shares
                        # Note: for steel, shares are shares of generation
                        endo_gen = endo_capacity * data['BSTC'][r, :, c5ti['CF']]                                       
                        demand_weight = np.sum(endo_shares * (data['BSTC'][r, :, c5ti['CF']]))  ## value is 0.0

                        ## change            
                        Utot = np.sum(endo_shares * demand_weight)
                        dUk = np.zeros([len(titles['STTI'])])
                        dUkSK = np.zeros([len(titles['STTI'])])
                        dUkREG = np.zeros([len(titles['STTI'])])  # shape : (26,)
                        #dUkKST = np.zeros([len(titles['STTI'])])
                                                    
                                    
                        # Kickstart in terms of SPSAt
                        dUkKST = np.where(data['SWKA'][r, :, 0] < 0,
                                            data['SKST'][r, :, 0] * Utot * dt, 0)
                                                    
                        # Regulations have priority over kickstarts
                        dUkKST = np.where(((dUkKST / data['BSTC'][r, :, c5ti['CF']]) + endo_capacity > data['SEWR'][r, :, 0]) & (data['SEWR'][r, :, 0] >= 0.0),
                                            0,
                                            dUkKST)
                        # Shares are shares of demand, divided by average capacity factor 
                        # Regulation is done in terms of shares of raw demand (no weighting)
                        # Correct for regulations using difference between endogenous demand and demand from last time step with endo shares


                        # ## change : shape to be 26
                        dUkREG = np.where((endo_capacity * demand_weight - (endo_shares * spsa_dtl[r])) > 0, 
                                            - ((endo_capacity * demand_weight) - endo_shares * spsa_dtl[r]) * isReg[r, :],
                                            0)                                               
                        # Calculate demand subtractions based on exogenous capacity after regulations and kickstart, to prevent subtractions being too large and causing negatve shares.
                        # Convert to direct shares of SPSAt - no weighting!
                        dUkSK = np.where((data['SWKA'][r, :, 0] > endo_capacity) & (data['SWKA'][r, :, 0] > data ['SWKL'][r, :, 0]),
                                            ((data['SWKA'][r, :, 0] - endo_capacity) * demand_weight - dUkREG - dUkKST) * (t/invdt),
                                            0)                                    
                        # If SWKA is a target and is larger than the previous year's capacity, treat as a kick-start based on previous year's capacity. Small additions will help the target be met. 
                        dUkSK = np.where((data['SWKA'][r, :, 0] > endo_capacity) & (data['SWKA'][r, :, 0] > data ['SWKL'][r, :, 0]),
                                        (data['SWKA'][r, :, 0] - endo_capacity) * demand_weight * (t/invdt),
                                            dUkSK)                                                                        
                        # Regulations have priority over exogenous capacity
                        dUkSK = np.where((data['SWKA'][r, :, 0] < 0) | (data['SEWR'][r, :, 0] >= 0) & (data['SWKA'][r, :, 0] > data['SEWR'][r, :, 0]),
                                        0, dUkSK)
                
                                                                        
                        dUk = dUkREG + dUkSK + dUkKST
                        dUtot  = np.sum(dUk)
                                                
                                                    
                        #Use modified shares of demand and total modified demand to recalulate market shares
                        #This method will mean any capacities set to zero will result in zero shares
                        #It avoids negative shares
                        #All other capacities will be stretched, depending on the magnitude of dUtot and how much of a change this makes to total capacity/demand
                        #If dUtot is small and implemented in a way which will not under or over estimate capacity greatly, SWKA is fairly accurate
                    
                        #Market share changes due to exogenous settings and regulations
                        if np.sum((endo_capacity[r] * demand_weight[r]) + dUk[r]) > 0:
                            data['SWSA'][r, :, 0] = dUk[r]/np.sum((endo_capacity[r] * demand_weight[r]) + dUk[r])
                                        
                        #New market shares
                        if np.sum((endo_capacity[r] * demand_weight[r]) + dUk[r]) > 0:
                            data['SEWS'][r, :, 0] = ((endo_capacity[r] * demand_weight[r]) + dUk[r])/(np.sum(endo_capacity[r] * demand_weight[r]) + dUk[r])
                                                    
                        #Changes due to end-of-lifetime replacements
                        data['SEOL'][r, :, 0] = np.sum(dSij, axis = 1)
                        #Changes due to premature scrapping
                        data['SBEL'][r, :, 0] = np.sum(dSEij,axis = 1)

                        print(f"SEWK before update: {data['SEWK'][r, :, 0]}")
                        data['SEWK'][r, :, 0] = spsa_dt[r]/np.sum(data['SEWS'][r, :, 0] * data['BSTC'][r, :, c5ti['CF']]) * data['SEWS'][r, :, 0]   ## check
                        print(f"SEWK after update: {data['SEWK'][r, :, 0]}")

                        #Actual steel production per technology (kton) (capacity factors column 12)
                        data['SEWG'][r, :, 0] = (data['SEWK'][r, :, 0] * data['BSTC'][r, :, c5ti['CF']])
                        #Emissions (MtCO2/y) (14th is emissions factors tCO2/tcs)
                        data['SEWE'][r, :, 0] = (data['SEWG'][r, :, 0] * data ['STEF'][r, :, 0])/1e3
            
                          
                        
    #--Main variables once we have new shares:--
    #Steel production capacity per technology (kton)
    # for r in range(len(data['RTI'])):
            ## change  : shape SEWK : 71,26
            ## change: why multiple twice with 'SEWS'
             
                        #EOL replacements based on shares growth
                        eol_replacements_t = np.zeros([len(titles['RTI'])])
                        eol_replacements = np.zeros([len(titles['RTI'])])
                        
                        if (t==0):  
                            for t in range(0, (invdt - 1)):
                                data['SEWI'][r, :, 0] = 0.0
                                data_dt['SEWI'][r, :, 0] = 0.0
                        
                        ## change : why eol_replacements_t overwritten?                                     
                        eol_replacements_t = np.where((endo_eol >= 0 & (data['BSTC'][r, :, c5ti['Lifetime']] > 0)), 
                                                        (time_lag['SEWK'][r, :, 0] * dt/data['BSTC'][r, :, c5ti['Lifetime']]), 0)
                        ## wrong parenthesis
                        ## This varubale should be eol_replacements
                        eol_replacements_t = np.where((time_lag['SEWS'][r, :, 0] * (dt / data['BSTC'][r, :, c5ti['Lifetime']]) < endo_eol) & (endo_eol < 0) & (data['BSTC'][r, :, c5ti['Lifetime']] > 0), 
                                                        (data['SEWS'][r, :, 0] - time_lag['SEWS'][r, :, 0] + time_lag['SEWS'][r, :, 0] * dt/data['BSTC'][r, :, c5ti['Lifetime']]) * time_lag['SEWK'][r, :, 0], eol_replacements_t) 
                        
                        # Capacity growth
                        ## change : (data['SEWK'][r, :, 0] - data['SWKL'][r, :, 0]) is an array 
                        var_sewk = data['SEWK'][r, :, 0] - data['SWKL'][r, :, 0]
                        if var_sewk[b1]> 0 : ## iterate for 26 times
                                data_dt['SEWI'][r, :, 0] = data['SEWK'][r, :, 0] - data['SWKL'][r, :, 0] + eol_replacements_t
                        else:
                                data_dt['SEWI'][r, :, 0] = eol_replacements_t
                        
                        # Capacity growth, add each time step to get total at end of loop
                        data['SEWI'][r, :, 0] = data['SEWI'][r, :, 0] + data_dt['SEWI'][r, :, 0]             
                                    
            # Check what investment and learning thing comes here from line 800 to 844 in Fortran
            # Cumulative investment for learning cost reductions
            # (Learning knowledge is global Therefore we sum over regions)
            # BI = MATMUL(SEWB,SEWIt)          Investment spillover: spillover matrix B
            # dW = SUM(BI,dim=2)         Total new investment dW (see after eq 3 Mercure EP48)

                    sewi0= np.sum(data_dt['SEWI'][:, :, 0] , axis=0)
                    bi = np.zeros((len(titles['RTI']),len(titles['STTI'])))
                    dw = np.zeros((len(titles["STTI"]),1))   ## changed

                    for path in range(len(titles['STTI'])):  
                        dw_temp = sewi0         ## change why do we need this
                        dw[path] = np.matmul(dw_temp, data['SEWB'][0, path, :])
                

                    #Cumulative capacity for learning
                    data['SEWW'] = data_dt['SEWW'] + dw

                    #Update technology costs for both the carbon price and for learning
                    #Some costs do not change

                    data['BSTC'][: , : , 0:21] = data_dt['BSTC'][: , : , 0:21]
                    data['SCMM']= data_dt['SCMM']

                    # for j in range(len(titles['RTI'])):
                    #     #Switch: Do governments feedback xx% of their carbon tax revenue as energy efficiency investments?
                    #     #Government income due to carbon tax in mln$(2008) 13/3/23 RSH: is this 2008 or 2013 as the conversion suggests?
                    # Lines 820 to 844 relate to E3ME variables
                    # New additions (SEWI)
                    sales_or_investments, sewi_t = get_sales(data['SEWK'], data_dt['SEWK'], time_lag['SEWK'], data['SEWS'], data_dt['SEWS'], data['SEWI'], data['BSTC'][:, :, c5ti['Lifetime']], dt)
                ############## Learning-by-doing ##################
                

                    # Learning-by-doing effects on investment and efficiency
                    for b in range(len(titles['STTI'])): 

                        if data['SEWW'][0, b, 0] > 0.0001:

                            data['BSTC'][:, b, c5ti['IC']] = (data_dt['BSTC'][:, b, c5ti['IC']]  \
                                                                                        *(1.0 + data['BSTC'][:, b, c5ti['Learning rate (IC)']] * dw[b]/data['SEWW'][0, b, 0]))
                            data['BSTC'][:, b, c5ti['dIC']] = (data_dt['BSTC'][:, b, c5ti['dIC']]  \
                                                                                        *(1.0 + data['BSTC'][:, b, c5ti['Learning rate (IC)']] * dw[b]/data['SEWW'][0, b, 0]))

                    #Total investment in new capacity in a year (m 2014 euros):
                        #SEWI is the continuous time amount of new capacity built per unit time dI/dt (GW/y)
                        #BHTC(:,:,1) are the investment costs (2014Euro/kW)
                    data['SWIY'][:,:,0] = data['SWIY'][:,:,0] + data['SEWI'][:,:,0]*dt*data['BSTC'][:,:,0]/data['PRSC14'][:,0,0,np.newaxis]
                    # Save investment cost for front end
                    data["SWIC"][:, :, 0] = data["BSTC"][:, :, c5ti['IC']]
                    
                    # Add lines 884 to 905 from Fortran (LBD)
                    sica_lr = -0.015
                    #Calculate learning in terms of energy/material consumption
                    # for mat in range(len(titles['SMTI'])):
                    #      for plant in range(len(titles['SSTI'])):
                    #           if(data['SICA'][0, plant, 0] > 0.0):
                    #                data['SCMM'][0, mat, plant] = (data_dt['SCMM'][0, mat, plant] - data['SEEM'][0, mat, plant]) * (1.0 + (sica_lr[0,plant]) * ((data['SICA'][0, plant, 0]-data_dt['SICL'][0, plant, 0])/data_dt['SICL'][0, plant, 0])) + data['SEEM'][0, mat, plant]

                    #Update material and cost input output matrix
                    data['SLCI'][:, 4, 10] = data['SCMM'][:, 0, 6]
                    
        if (iteration > 1): 
            #Redistribute materials
            #if (iter < 10 and t == invdt):
            data = raw_material_distr(data, titles, year, t)

            #Regional average energy intensity (G/tcs)
            # data['SEIA'] = np.sum((data['STEI'] * data['SEWS']),axis = 1)
            data['SEIA'][r] = np.sum((data['STEI'][r,:] * data['SEWS'][r,:])) ## changed

            #Calculate bottom-up employment growth rates
            data['SEMS'] = data['SEWK'] * data['BSTC'][ :, : , 4].reshape(71, 26, 1) * 1.1   ## changed 
            data['SEMR'] = np.where((np.sum(data_dt['SMPL']) > 0.0), (np.sum(data['SEMS'])/np.sum(data['SMPL'])), 0.0)

            # Call the capacity function
            # Calculate levelised cost again
            data = get_lcos(data, titles)
            data['SJEF'], data['SJCO']= ftt_s_fuel_consumption(data['BSTC'], data['SEWG'], data['SMED'], len(titles['RTI']), len(titles['STTI']), c5ti, len(titles['JTI']))
            
        
        #  =================================================================
        # Update the time-loop variables
        # =================================================================

        #Update time loop variables:
        for var in data_dt.keys():

            data_dt[var] = copy.deepcopy(data[var])


    return data
