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
from ..support.divide import divide
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

    data_dt = {}
    for var in time_lag.keys():
        data_dt[var] = np.copy(time_lag[var]) 

    
    sector = 'steel'
    no_it = data["noit"][0, 0, 0]

    dt = 1.0/no_it
    invdt = no_it
    tScaling = 10.0
    data = scrap_calc(data, time_lag, titles, year)
    var_iter = 1

    if year<2020:  ## year<=2019
        for r in range(len(titles['RTI'])):
            data['SEWK'][r,:] = 0.0
            data['SEWK'][r,:, 0] = data['SEWG'][r,:,0]/data['BSTC'][r,:,c5ti['CF']]

            data['SEMS'][r,:,0] = data['SEWK'][r,:,0] * data['BSTC'][r,:,c5ti["Employment"]]*11

            data['SEWS'][r,:] = 0
            if data['SPSP'][r] > 0.0:  
                valid_indices = data['SEWK'][r,:] > 0.0  
                if np.any(valid_indices): 
                    data['SEWS'][r,valid_indices] = data['SEWK'][r,valid_indices] / np.sum(data['SEWK'][r,valid_indices])

            data['SEWE'][r,:,0] = data['BSTC'][r,:,c5ti["EF"]]/1000

            data['STEI'][r,:,0] = data['BSTC'][r,:,c5ti['Energy Intensity']]
            data['SEIA'][r] = np.sum(data['STEI'][r,:]*data['SEWS'][r,:])

            ## fuel consumption function
            data['SJEF'], data['SJCO']= ftt_s_fuel_consumption(data['BSTC'], data['SEWG'], data['SMED'], len(titles['RTI']), len(titles['STTI']), c5ti, len(titles['JTI']))


        if year==2019:
            data['SPSA'] = data['SPSP']
                    
            ## Raw material distribution
            data = raw_material_distr(data, titles, year, 1,data['SPSA'])

            og_base = np.ones((71), dtype=np.float64)
            if np.sum(data['SEWG'], axis=1).all() > 0.0:  
                og_base = np.sum(data['SEWG'][:, 0:7], axis=1) / np.sum(data['SEWG'], axis=1)
            og_sim = og_base

            # ccs_share = 0.0
            data['SJEF'] = 0.0
            data['SJCO'] = 0.0

            for r in range(len(titles['RTI'])):
                data['SEIA'][r] = np.sum(data['STEI'][r,:]*data['SEWS'][r,:])

                if data['SPSA'][r] > 0.0:
                    if np.sum(data['SEWG'][r,:]) > 0.0:
                        og_sim[r] = np.sum(data['SEWG'][r,0:19])/np.sum(data['SEWG'][r,:])

                        # ccs_share[r] = (np.sum(data['SEWG'][r, 3:7]) + np.sum(data['SEWG'][r, 9:11]) + np.sum(data['SEWG'][r, 13:15]) + np.sum(data['SEWG'][r, 17:19]) + np.sum(data['SEWG'][r, 21:23]))/ np.sum(data['SEWG'][r, :])

                    data['SJEF'], data['SJCO']= ftt_s_fuel_consumption(data['BSTC'], data['SEWG'], data['SMED'], len(titles['RTI']), len(titles['STTI']), c5ti, len(titles['JTI']))

                data['SJFR'] = data['SJEF']

                
                data_dt['SMPL'] = data['SMPL']
                if np.sum(data['SMPL'], axis=1).all() > 0.0:
                    data['SEMR'] = np.sum(data['SEMS'], axis=1) / np.sum(data['SMPL'], axis=1)
                
                data = get_lcos(data, titles,data['SPSA'])

                bi = np.zeros((len(titles['RTI']), len(titles['STTI'])))
                bi[r,:] = np.matmul(data['SEWB'][0,:,:],data['SEWK'][r,:,0])  ## sewb = [0,26,26]; sewk = [71,26,0]
                data['SEWW'] = np.sum(bi, axis=0)  ## seww = [0,26,0]
                data['SEWW'] = data['SEWW'][None, :, None]
            
                data['SICA'] = np.zeros((1,29,1), dtype=np.float64)
                
                for t1 in range(len(titles['STTI'])):
                    for t2 in range(len(titles['SSTI'])):
                        if data['STIM'][0, t1, t2] == 1:
                            if t2 < 8:
                                # Ensure there are non-zero entries in 'SPSA'
                                non_zero_count = np.count_nonzero(data['SPSA'][:, 0, 0])
                                if non_zero_count > 0:
                                    # Safely assign values to 'SICA'
                                    data['SICA'][0, t2, 0] += (
                                        1.1 * data['SEWW'][0, t1, 0] *
                                        np.sum(data['BSTC'][:, t1, 25 + t2]) /
                                        non_zero_count
                                    )                        
                            elif t2 > 7 and t2 < 21:
                                data['SICA'][0, t2, 0] = data['SICA'][0, t2, 0] + 1.1 * data['SEWW'][0,t1,0]

                            elif t2 > 20 and t2 < 27:
                                data['SICA'][0, t2, 0] = data['SICA'][0, t2, 0]+ data['SEWW'][0,t1,0]
                            
                            elif t2 == 27:
                                data['SICA'][0, t2, 0] = data['SICA'][0, t2, 0] + data['SEWW'][0,t1,0] / 1.14
                
                
                data['SEWI'] = data['SEWK'] - data['SWKL']
                data['SEWI'] = np.where(data['SEWI'] < 0.0, 0.0, data['SEWI'])

                data['SEWI'][: , : , 0] = np.where((data['BSTC'][: , : , 5] > 0.0) , (data['SEWI'][: , : , 0] + (data['SWKL'][: , :, 0]/data['BSTC'][: , : , 5])) , np.max((data['SEWK'][: , : , 0] - data['SWKL'][: , : , 0]), 0))

    for var in data.keys():
        data_dt[var] = copy.deepcopy(data[var])

    isReg = np.zeros((len(titles['RTI']), len(titles['STTI'])))

    isReg = np.where(data['SEWR'] > 0.0, 1.0 + np.tanh(1.5 + 10 * (data['SWKL'] - data['SEWR'])) / data['SEWR'],0.0)

    isReg = np.where(data['SEWR'] == 0.0, 1.0,0.0)

    data['SPSA'] = np.zeros((len(titles['RTI']),1,1))
    endo_eol = np.zeros((len(titles['RTI']), len(titles['STTI'])))
    endo_gen = np.zeros((len(titles['RTI']), len(titles['STTI'])))

    shape_rs = (len(titles['RTI']), len(titles['STTI']),1)
    dUk = np.zeros(shape_rs)
    dUkSk = np.zeros(shape_rs)
    dUkREG = np.zeros(shape_rs)
    dUkKST = np.zeros(shape_rs)
    demand_weight = np.zeros(len(titles['RTI']))
    endo_shares = np.zeros((len(titles['RTI']), len(titles['STTI'])))
    endo_capacity = np.zeros((len(titles['RTI']), len(titles['STTI'])))

    if year > 2019:
        for var in data.keys():
            data_dt[var] = copy.deepcopy(data[var])
        # if var_iter == 1:
        for t in range(int(invdt)):
            spsa_dt = np.zeros((71))
            spsa_dt = data['SPSL'] + (data['SPSA'] + data['SPSL']) * t/invdt
            spsa_dtl = data['SPSL'] + (data['SPSA'] + data['SPSL']) * (t-1)/invdt

            # Negation of the condition
            condition1 = (~(data['SEWR'][:, 25,0] > data_dt['SEWK'][:, 25,0])) & (data['SEWR'][:, 25,0] > -1.0)

            # Apply the first WHERE statement
            isReg[:, 25,0] = np.where(condition1, 1.0 - np.tanh(2 * 1.25 * data_dt['BSTC'][:, stti['Scrap - EAF'], c5ti['Scrap']] - 0.5) / 0.5, isReg[:, 25,0])

            # Second condition: if isReg > 1.0, set it to 1.0
            isReg[:, 25] = np.where(isReg[:, 25] > 1.0, 1.0, isReg[:, 25]) 

            for r in range(len(titles['RTI'])):
                dSij = np.zeros([len(titles['STTI']), len(titles['STTI'])])   
                F = np.ones([len(titles['STTI']), len(titles['STTI'])]) * 0.5
                FE = np.ones([len(titles['STTI']), len(titles['STTI'])]) * 0.5
                k_1 = np.zeros([len(titles['STTI']), len(titles['STTI'])])
                k_2 = np.zeros([len(titles['STTI']), len(titles['STTI'])])
                k_3 = np.zeros([len(titles['STTI']), len(titles['STTI'])])
                k_4 = np.zeros([len(titles['STTI']), len(titles['STTI'])])
                

                if np.any(spsa_dt[r] > 0.0):
                    for i in range(len(titles['STTI'])): 
                        if  not (data['SEWS'][r, i, 0] > 0.0 and
                            data['SGC1'][r, i, 0] != 0.0 and
                            data['SEWC'][r, i, 0] != 0.0):
                            continue
                                
                        S_i = data['SEWS'][r, i, 0]
                        if S_i[r,i] > 0.0:
                            dSij[i,i] = 0

                            for k in range(i-1):                 
                                if not (data['SEWS'][r, k, 0] > 0.0 and
                                    data['SGC1'][r, k, 0] != 0.0 and 
                                    data['SEWC'][r, k, 0] != 0.0): 
                                    continue                      
                                        
                                S_k = data['SEWS'][r, k, 0]

                                if S_k[r,k] > 0.0:
                                    
                                    dFij = 1.414 * sqrt((data['SGD1'][r, i, 0] * data['SGD1'][r, i, 0] 
                                                        + data['SGD1'][r, k, 0] * data['SGD1'][r, k, 0]))
                                            
                                    # Investor preference incl. uncertainty
                                    Fij = 0.5 * (1 + np.tanh(1.25 * (data_dt['SGC1'][r, k, 0]
                                                - data_dt['SGC1'][r, i, 0])) / dFij)

                                    F[i, k] = Fij * (1.0 - isReg[r, i]) * (1.0 - isReg[r,i]) + isReg[r, k] \
                                                                * (1.0 - isReg[r, i]) + 0.5 * (isReg[r, i] * isReg[r, k])
                                    F[k, i] = (1.0 - Fij) * (1.0 - isReg[r, k]) * (1.0 - isReg[r, i]) + isReg[r, i] \
                                                                * (1.0 - isReg[r, k]) + 0.5 * (isReg[r, k] * isReg[r, i])

                                    k_1[i,k] = S_i * S_k * (data['SEWA'][0, i, k] * F[i,k] - data['SEWA'][0, k, i]* F[k,i])
                                    k_1[k,i] = -k_1[i,k]
                                    k_2[i,k] = (S_i + dt * k_1[i,k]/2) * (S_k - dt * k_1[k,i]/2) * (data['SEWA'][0, i, k] * F[i,k] - data['SEWA'][0, k, i] * F[k,i])
                                    k_2[k,i] = -k_2[i,k]
                                    k_3[i,k] = (S_i + dt * k_2[i,k]/2) * (S_k - dt * k_2[k,i]/2) * (data['SEWA'][0, i, k] * F[i,k]- data['SEWA'][0, k, i]  * F[k,i])
                                    k_3[k,i] = -k_3[i,k]
                                    k_4[i,k] = (S_i + dt * k_3[i,k]) * (S_k - dt * k_3[k,i]) * (data['SEWA'][0, i, k] * F[i,k] - data['SEWA'][0, k, i] * F[k,i])
                                    k_4[k,i] = -k_4[i,k]

                                    dSij[i, k] = dt * (k_1[i,k]+2*k_2[i,k]+2*k_3[i,k]+k_4[i,k])/6/tScaling   ## change
                                    dSij[k, i] = -dSij[i, k]

                    SR = np.zeros(len(titles['RTI']))
                    SR = 1/data['BSTC'][r, :, c5ti['Payback period']] - 1/data['BSTC'][r, :, c5ti['Lifetime']]
                        
                    # Constant used to reduce the magnitude of premature scrapping 
                    SR_C = 2.5 

                    kE_1 = np.zeros([len(titles['STTI']), len(titles['STTI'])])
                    kE_2 = np.zeros([len(titles['STTI']), len(titles['STTI'])])
                    kE_3 = np.zeros([len(titles['STTI']), len(titles['STTI'])])
                    kE_4 = np.zeros([len(titles['STTI']), len(titles['STTI'])])
                            
                    for i in range(len(titles['STTI'])):
                        if  not (data['SEWS'][r, i, 0] > 0.0 and
                            data['SGC1'][r, i, 0] != 0.0 and
                            data['SEWC'][r, i, 0] != 0.0):
                            continue
                                
                        SE_i = data['SEWS'][r, i, 0]
                        dSEij = np.zeros([len(titles['STTI']), len(titles['STTI'])])

                        if SE_i[r,i] > 0.0:
                            dSEij[i,i] = 0

                            for k in range(i-1):                 
                                if not (data['SEWS'][r, k, 0] > 0.0 and
                                    data['SGC1'][r, k, 0] != 0.0 and 
                                    data['SEWC'][r, k, 0] != 0.0): 
                                    continue                      
                                        
                                SE_k = data['SEWS'][r, k, 0]

                                if SE_k[r,k] > 0.0:
                                    dFEij = 1.414 * sqrt((data['SGD2'][r, k, 0] ** 2 + data['SGD3'][r, i, 0] ** 2))
                                    dFEji = 1.414 * sqrt((data['SGD2'][r, i, 0] ** 2 + data['SGD3'][r, k, 0] ** 2))

                                    FEij = 0.5 * (1 + np.tanh(1.25 *(data['SGC2'][r,k] - data['SGC3'][r,i]))/dFEij)
                                    FEji = 0.5 * (1 + np.tanh(1.25 *(data['SGC2'][r,i] - data['SGC3'][r,k]))/dFEji)

                                    FE[i,k] = FEij * (1.0 - isReg[r,i])
                                    FE[k,i] = FEji * (1.0 - isReg[r,k])

                                    # dSEij(I,K) = SWSLt(I,J)*SWSLt(K,J)*(SWAP(I,K)*FE(I,K)*SR(K,J)/SR_C - SWAP(K,I)*FE(K,I)*SR(I,J)/SR_C)*dt
                                    
                                    kE_1[i,k] = SE_i[r,i] * SE_k[r,k] * (data['SWAP'][0, i, k] * FE[i,k] * SR[r,k]/SR_C - data['SWAP'][0, k, i] * FE[k, i] * SR[r,i]/SR_C)
                                    kE_1[k,i] = -kE_1[i,k]
                                    kE_2[i,k] = (SE_i[r,i] + dt * kE_1[i,k]/2) * (SE_k[r,k] + dt * kE_1[k,i]/2) * (data['SWAP'][0, i,k] * FE[i,k] * SR[r,k]/SR_C - data['SWAP'][0, k,i] * FE[k,i]*SR[r,i]/SR_C)
                                    kE_2[k,i] = -kE_2[i,k]
                                    kE_3[i,k] = (SE_i[r,i] + dt * kE_2[i,k]/2) * (SE_k[r,k] + dt * kE_2[k,i]/2) * (data['SWAP'][0, i,k] * FE[i,k] * SR[r,k]/SR_C - data['SWAP'][0, k,i] * FE[k,i]*SR[r,i]/SR_C)
                                    kE_3[k,i] = -kE_3[i,k]
                                    kE_4[i,k] = (SE_i[r,i] + dt * kE_3[i,k]) * (SE_k[r,k] + dt * kE_3[k,i]) * (data['SWAP'][0, i,k] * FE[i,k] * SR[r,k]/SR_C - data['SWAP'][0, k,i] * FE[k,i]*SR[r,i]/SR_C)
                                    kE_4[k,i] = -kE_4[k,i]     

                                    dSEij[i,k] = dt * (kE_1[i,k]+2*kE_2[i,k]+2*kE_3[i,k]+kE_4[i,k])/6/tScaling
                                
                    endo_shares[r,:] = data['SEWS'][r, :,0] + np.sum(dSij,axis = 1) + np.sum(dSEij, axis = 1)
                    endo_eol[r,:] = np.sum(dSij,axis=1) ## this is getting (26,)
                    endo_capacity[r,:] = spsa_dt[r, 0,0]/np.sum(endo_shares[r,:] * data['BSTC'][r, :, c5ti['CF']]) * endo_shares[r,:]

                    endo_gen[r,:] = endo_capacity[r,:] * (data['BSTC'][r, :, c5ti['CF']])

                    # demand_weight = np.zeros(len(titles['RTI']))

                    demand_weight[r] =  np.sum(endo_shares[r,:] *(data['BSTC'][r, :, c5ti['CF']]))

                    Utot = np.sum(endo_shares[r,:] * demand_weight[r])

                    shape_rs = (len(titles['RTI']), len(titles['STTI']))
                    dUk = np.zeros(shape_rs)
                    dUkSk = np.zeros(shape_rs)
                    dUkREG = np.zeros(shape_rs)
                    dUkKST = np.zeros(shape_rs)

                    dUkKST[r,:] = np.where(data['SWKA'][r, :, 0] < 0,
                                                data['SKST'][r, :, 0] * Utot * dt, 0)

                    dUkKST[r,:] = np.where(((dUkKST[r,:] / data['BSTC'][r, :, c5ti['CF']]) + endo_capacity[r,:] > data['SEWR'][r, :, 0]) & (data['SEWR'][r, :, 0] >= 0.0), 0, dUkKST[r,:])

                    dUkREG[r,:] = np.where((endo_capacity[r,:] * demand_weight[r] - (endo_shares[r,:] * spsa_dtl[r])) > 0, - ((endo_capacity[r,:] * demand_weight[r]) - endo_shares[r,:] * spsa_dtl[r]) * isReg[r, :, 0],0)

                    dUkSk[r,:] = np.where((data['SWKA'][r, :, 0] > endo_capacity[r,:]) & (data['SWKA'][r, :, 0] > data ['SWKL'][r, :, 0]),
                                                ((data['SWKA'][r, :, 0] - endo_capacity[r,:]) * demand_weight[r] - dUkREG[r,:] - dUkKST[r,:]) * (t/invdt),
                                                0)                                    
                
                    dUkSk[r,:] = np.where((data['SWKA'][r, :, 0] > endo_capacity[r,:]) & (data['SWKA'][r, :, 0] > data ['SWKL'][r, :, 0]),
                                    (data['SWKA'][r, :, 0] - endo_capacity[r,:]) * demand_weight[r] * (t/invdt),
                                        dUkSk[r,:])   
                                                                                            
                    dUkSk[r,:] = np.where((data['SWKA'][r, :, 0] < 0) | (data['SEWR'][r, :, 0] >= 0) & (data['SWKA'][r, :, 0] > data['SEWR'][r, :, 0]),
                                    0, dUkSk[r,:])
                            
                    dUk[r,:] = dUkREG[r,:] + dUkSk[r,:] + dUkKST[r,:]
                    dUtot  = np.sum(dUk[r,:])
                    
                    if np.sum(endo_capacity[r,:]*demand_weight[r] + dUk[r,:]) > 0:
                        data['SWSA'][r,:] = dUk[r,:] / np.sum(endo_capacity[r,:]*demand_weight[r] + dUk[r,:])

                        data['SEWS'][r,:] = (endo_capacity[r,:] * demand_weight[r] + dUk[r,:])/np.sum(endo_capacity[r,:]*demand_weight[r] + dUk[r,:])
                    
                    data['SEOL'][r,:] = np.sum(dSij, axis=1)
                    data['SBEL'][r,:] = np.sum(dSEij, axis=1)

                    data['SEWK'][r,:] = spsa_dt[r]/np.sum(data['SEWS'][r,:]*data['BSTC'][r,:,11]) * data['SEWS'][r,:]
                    data['SEWG'][r,:] = data['SEWK'][r,:] * data['BSTC'][r,:,12]

                    data['SEWE'][r,:] = data['SEWG'][r,:] * data['STEF'][r,:]/1e3

                eol_replacements_t = np.zeros((len(titles['RTI']), len(titles['STTI']),1))
                eol_replacements = np.zeros((len(titles['RTI']), len(titles['STTI']),1))

                if t==1:
                    data['SEWI'][r, :, 0] = 0.0
                    data_dt['SEWI'][r, :, 0] = 0.0
            
                if (endo_eol[r,:] >= 0).all() and (data['BSTC'][r,:,5] > 0).all():
                    eol_replacements_t[r,:,0] = data['SWKL'][r,:,0] * dt /data['BSTC'][r,:,5]

                condition1 = (data_dt['SWSL'] * dt / data['BSTC'][r, :, 5]) < endo_eol[r, :]
                condition2 = endo_eol[r, :] < 0
                condition3 = data['BSTC'][r, :, 5] > 0

                if (condition1 & condition2).all() and condition3.all():
                    eol_replacements_t[r,:,0] = (data['SEWS'][r,:] - data_dt['SWSL'][r,:] + data_dt['SWSL'] * dt/data['BSTC'][r,:,5]) * data['SWKL'][r,:,0]
            
                if ((data['SEWK'][r,:] - data['SWKL'][r,:]) > 0).all():
                    data_dt['SEWI'][r,:,0] = data['SEWK'][r,:] - data_dt['SWKL'][r,:] + eol_replacements_t[r,:,0]
                
                else:
                    data_dt['SEWI'][r,:,0] = eol_replacements_t[r,:,0]

                data['SEWI'][r, :, 0] += data_dt['SEWI'][r, :, 0]

                sewi0 = np.sum(data_dt['SEWI'],axis=0)
                dW = np.zeros((26))

                for i in range(len(titles['STTI'])):
                    dW_temp = np.squeeze(sewi0)
                    dW[i] = np.dot(dW_temp, data['SEWB'][0,i,:])   

                data['SEWW'][0,:,0] = data_dt['SEWW'][0,:,0] + dW
                data['BSTC'][: , : , 0:21] = data_dt['BSTC'][: , : , 0:21]
                data['SCMM'] = data_dt['SCML'] 
                sales_or_investments, sewi_t = get_sales(data['SEWK'], data_dt['SEWK'], time_lag['SEWK'], data['SEWS'], data_dt['SEWS'], data['SEWI'], data['BSTC'][:, :, c5ti['Lifetime']], dt)    ## line 820 to 883

                for b in range(len(titles['STTI'])): ## line 846 to 853
                    if data['SEWW'][0, b, 0] > 0.0001:

                        data['BSTC'][:, b, c5ti['IC']] = (data_dt['BSTC'][:, b, c5ti['IC']]  \
                                                                                    *(1.0 + data['BSTC'][:, b, c5ti['Learning rate (IC)']] * dW[b]/data['SEWW'][0, b, 0]))
                        data['BSTC'][:, b, c5ti['dIC']] = (data_dt['BSTC'][:, b, c5ti['dIC']]  \
                                                                                        *(1.0 + data['BSTC'][:, b, c5ti['Learning rate (IC)']] * dW[b]/data['SEWW'][0, b, 0]))

                data['SWIY'][:,:,0] = data['SWIY'][:,:,0] + data['SEWI'][:,:,0]*dt*data['BSTC'][:,:,0]/data['PRSC14'][:,0,0,np.newaxis]
                
                # Save investment cost for front end
                data["SWIC"][:, :, 0] = data["BSTC"][:, :, c5ti['IC']]
                
                data['SLCI'][:, 4, 10] = data['SCMM'][:, 0, 6]

                # if (var_iter < 10) and (t == invdt):
                data = data = raw_material_distr(data, titles, year, t,spsa_dt)

                data['SEIA'][r] = np.sum((data['STEI'][r,:] * data['SEWS'][r,:]),axis=0) 
                
                data['SEMS'] = data['SEWK'] * data['BSTC'][:,:,4].reshape(71, 26, 1) * 1.1

                data['SEMR'] = np.where((np.sum(data_dt['SMPL']) > 0.0), (np.sum(data['SEMS'])/np.sum(data['SMPL'])),0)

                data = get_lcos(data, titles,spsa_dt)
                data['SJEF'], data['SJCO']= ftt_s_fuel_consumption(data['BSTC'], data['SEWG'], data['SMED'], len(titles['RTI']), len(titles['STTI']), c5ti, len(titles['JTI']))
        
        ### Commented in the original python code
        # if var_iter > 1:
        for r in range(len(titles['RTI'])):
            if data['SPSA'][r] > 0.0 :
                data['SEWK'][r,:] = data['SPSA'][r]/np.sum(data['SEWS'][r,:] * data['BSTC'][r,:,11] * data['SEWS'][r,:])
                data['SEWG'][r,:] = data['SEWK'][r,:] * data['BSTC'][r,:,11]
                data['SEWE'][r,:] = data['SEWG'][r,:] * data['STEF'][r,:] / 1e3

                eol_replacements_t[r,:] = 0.0
                eol_replacements[r,:] = 0.0

                if (data['SEWS'][r,:] - data_dt['SWSL'][r,:]) > 0 and data['BSTC'][r,:,5] > 0.0:
                    eol_replacements[r,:] = data['SWKL'][r,:]/data['BSTC'][r,:,5]
                
                if ((-data['SWSL'][r,:]/data['BSTC'][r,:,5] < data['SEWS'][r,:] - data['SWSL'][r,:] < 0)) and (data['BSTC'][r,:,5] > 0.0):
                    eol_replacements[r,:] = (data['SEWS'][r,:] - data['SWSL'][r,:] + data['SWSL'][r,:]/data['BSTC'][r,:,5]) * data['SWKL'][r,:]
                
                if (data['SEWK'][r,:] - data['SWKL'][r,:]) > 0:
                    data['SEWI'][r,:] = (data['SEWK'][r,:] - data['SWKL'][r,:]) + eol_replacements[r,:]

                else:
                    data['SEWI'][r,:] = eol_replacements[r,:]
                
                sewi0 = np.sum(data_dt['SEWI'],axis=0)

                for i in range(len(titles['SSTI'])):
                    dW_temp = sewi0
                    dW[i] = np.dot(dW_temp, data['SEWB'][0,i,:])   

                data['SEWW'][0,:,0] = data_dt['SEWW'][0,:,0] + dW

                # if var_iter < 10:
                data = data = raw_material_distr(data, titles, year, t,spsa_dt)

                data['SEIA'][r] = np.sum((data['STEI'][r,:] * data['SEWS'][r,:]),axis=0) 

                data['SEMS'] = data['SEWK'] * data['BSTC'][:,:,4] * 1.1

                data['SEMR'] = np.where((np.sum(data_dt['SMPL']) > 0.0), (np.sum(data['SEMS'])/np.sum(data['SMPL'])))

                data = get_lcos(data, titles, spsa_dt)
                data['SJEF'], data['SJCO']= ftt_s_fuel_consumption(data['BSTC'], data['SEWG'], data['SMED'], len(titles['RTI']), len(titles['STTI']), c5ti, len(titles['JTI']))


    return data
