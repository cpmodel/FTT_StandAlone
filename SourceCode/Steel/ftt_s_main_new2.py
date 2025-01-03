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
import sys
sys.path.append('C:/Users/swara/Documents/VS code/FTT_StandAlone-Steel_Swarali')

import csv
from math import sqrt
import copy
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
    shape_rs = (len(titles['RTI']), len(titles['STTI']),1)
    shape_st = (len(titles['STTI']), len(titles['STTI']),1)

    dUk = np.empty(shape=shape_rs)
    dUkSk = np.empty(shape=shape_rs)
    dUkREG = np.empty(shape=shape_rs)
    dUkKST = np.empty(shape=shape_rs)
    Utot = 0.0
    demand_weight = np.empty(shape=(len(titles['RTI']),1,1))
    S_i = np.empty(shape=shape_rs)
    S_k = np.empty(shape=shape_rs)
    SE_i = np.empty(shape=shape_rs)
    SE_k = np.empty(shape=shape_rs)
    k_1 = np.empty(shape=shape_st)
    k_2 = np.empty(shape=shape_st)
    k_3 = np.empty(shape=shape_st)
    k_4 = np.empty(shape=shape_st)
    kE_1 = np.empty(shape=shape_st)
    kE_2 = np.empty(shape=shape_st)
    kE_3 = np.empty(shape=shape_st)
    kE_4 = np.empty(shape=shape_st)

    SR = np.empty(shape=shape_rs,dtype=np.float64)
    dSij = np.empty(shape=shape_st)
    dSEij = np.empty(shape=shape_st) 
    F = np.ones(shape_st) * 0.5
    FE = np.ones(shape_st) * 0.5

    data['SPSA'] = np.empty(shape=(len(titles['RTI']),1,1))

    spsa_dt = np.empty(shape=(len(titles['RTI']),1,1))

    growthRate1 =  np.ones((len(titles['RTI']),1,1))
    growthRate2 =  np.ones((len(titles['RTI']),1,1))

    endo_eol = np.empty(shape=shape_rs)
    endo_gen = np.empty(shape=shape_rs)

    endo_shares = np.empty(shape=shape_rs)
    endo_capacity = np.empty(shape=shape_rs)

    og_base = np.ones((71,1,1), dtype=np.float64)

    ccs_share = np.empty(shape=(len(titles['RTI']),1,1))
    data['SJEF'] = np.empty(shape=(len(titles['RTI']), len(titles['JTI']),1))
    data['SJCO'] = np.empty(shape=(len(titles['RTI']), len(titles['JTI']),1))

    bi = np.empty(shape=shape_rs)
    data['SICA'] = np.empty(shape=(1,29,1), dtype=np.float64)

    isReg = np.empty(shape=shape_rs)

    data['SWIY'] = np.empty(shape=shape_rs)
    data['SWIG'] = np.empty(shape=shape_rs)

    eol_replacements_t = np.empty(shape=shape_rs)
    eol_replacements = np.empty(shape=shape_rs)

    dW = np.empty(shape=(26,1,1))

    data['SICA'] = np.empty(shape=(1, len(titles['SSTI']), 1))
  
    # Categories for the cost matrix (BSTC)
    c5ti = {category: index for index, category in enumerate(titles['C5TI'])}
    stti = {category: index for index, category in enumerate(titles['STTI'])}


    # sector = 'steel'
    no_it = data["noit"][0, 0, 0]
    dt = 1.0/no_it
    invdt = no_it
    tScaling = 10.0
    var_iter = 1


    data_dt = {}
    for var in time_lag.keys():
        data_dt[var] = copy.deepcopy(time_lag[var])

    # if var_iter == 1:
    data = scrap_calc(data, time_lag, titles, year)

    data_dt['SWSL'] = data['SEWS']
    data_dt['SWKL'] = data['SEWK']
    data_dt['SG1L'] = data['SGC1']
    data_dt['SD1L'] = data['SGD1']
    data_dt['SG2L'] = data['SGC2']
    data_dt['SD2L'] = data['SGD2']
    data_dt['SG3L'] = data['SGC3']
    data_dt['SD3L'] = data['SGD3']
    data_dt['BSTL'] = data['BSTC']
    data_dt['SWWL'] = data['SEWW']
    data_dt['SWYL'] = data['SWIY']
    data_dt['SWIL'] = data['SWII']
    data_dt['SWGL'] = data['SWIG']
    data_dt['SCML'] = data['SCMM']
    data_dt['SPCL'] = data['SPRC']
    data_dt['SICL'] = data['SICA']
    data_dt['SMPL'] = data['SEMS']

    if year > 2019:
        data_dt['SPRL'] = data['SPRC']

    
    if year < 2020:  
        for r in range(len(titles['RTI'])):
            data['SEWK'][r,:] = 0.0
            data['SEWK'][r,:, 0] = data['SEWG'][r,:,0]/data['BSTC'][r,:,c5ti['CF']]
            data['SPSA'][r, 0, 0] = np.sum(data['SEWG'][r, :, 0]) 

            data['SEMS'][r,:,0] = data['SEWK'][r,:,0] * data['BSTC'][r,:,c5ti["Employment"]]*11

            # data['SEWS'][r,:] = 0
            data['SEWS'] [r, :, 0] = np.divide(data['SEWK'][r, :, 0], np.sum(data['SEWK'][r, :, 0]),
                                                where =(data['SPSP'][r, 0, 0] > 0.0) and (data['SEWK'][r, :, 0] > 0.0)) 

            data['SEWE'][r,:,0] = (data['SEWG'][r,:,0] * data['BSTC'][r,:,c5ti["EF"]])/1000

            data['STEI'][r,:,0] = data['BSTC'][r,:,c5ti['Energy Intensity']]
            data['SEIA'][r] = np.sum(data['STEI'][r,:]*data['SEWS'][r,:])

            # if year > 2016:
            #     bi = np.matmul(data['SEWB'], data['SEWK'])  ## sewb = [0,26,26]; sewk = [71,26,0]
            #     data['SEWW'] = np.sum(bi, axis=0)
            #     data['SEWW'] = data['SEWW'][None, :]

            data['SJEF'], data['SJCO']= ftt_s_fuel_consumption(data['BSTC'], data['SEWG'], data['SMED'], len(titles['RTI']), len(titles['STTI']), c5ti, len(titles['JTI']))

        if year == 2019:
            
            data['SPSA'] = data['SPSP']
            for t in range(1,int(invdt) + 1):
                ## Raw material distribution
                data = raw_material_distr(data, titles, year, t,data['SPSA'])

            if np.sum(data['SEWG'], axis=1).all() > 0.0:  
                og_base = np.sum(data['SEWG'][:, :7], axis=1) / np.sum(data['SEWG'][:,:], axis=1)
            og_sim = og_base


            for r in range(len(titles['RTI'])):
                data['SEIA'][r] = np.sum(data['STEI'][r,:]*data['SEWS'][r,:])

                if (data['SPSA'][r]).all() > 0.0:
                    if np.sum(data['SEWG'][r,:]) > 0.0:
                        og_sim[r] = np.sum(data['SEWG'][r,:19])/np.sum(data['SEWG'][r,:])

                        ccs_share[r] = (np.sum(data['SEWG'][r, 3:7]) + np.sum(data['SEWG'][r, 9:11]) + np.sum(data['SEWG'][r, 13:15]) + np.sum(data['SEWG'][r, 17:19]) + np.sum(data['SEWG'][r, 21:23]))/ np.sum(data['SEWG'][r, :])

                    data['SJEF'], data['SJCO']= ftt_s_fuel_consumption(data['BSTC'], data['SEWG'], data['SMED'], len(titles['RTI']), len(titles['STTI']), c5ti, len(titles['JTI']))

                data['SPMT'][r,:11] = data['SPMA'][r,:11]
                data['SPMT'][r,11] = data['SMED'][0,11,0]
                data['SPMT'][r,12] = data['SMED'][0,12,0]
                data['SPMT'][r,15] = data['SPMA'][r,15]
                data['SPMT'][r,16] = data['SPMA'][r,16]
                data['SPMT'][r,17] = data['SPMA'][r,17]
                data['SPMT'][r,18] = data['SPMA'][r,18]
                data['SPMT'][r,19] = data['SPMA'][r,19]

            data['SJFR'] = data['SJEF']

            data_dt['SMPL'] = data['SMPL']
            data['SEMR'] = np.where(np.sum(data['SMPL']) > 0.0, np.sum(data['SEMS'],axis=1,keepdims=True)/np.sum(data['SMPL'],axis=1,keepdims=True),data['SEMR'])
                                            
            data = get_lcos(data,titles,data['SPSA'])

            for r in range(71):
                # Perform the matrix multiplication
                bi[r, :] = np.expand_dims(np.matmul(data['SEWB'][0, :, :], data['SEWK'][r, :, 0]), axis=1)
                data['SEWW'] = np.sum(bi[r],axis=1)

            data['SEWW'] = data['SEWW'][None, :, None]
            
            for t1 in range(len(titles['STTI'])):
                for t2 in range(len(titles['SSTI'])):
                    if data['STIM'][0, t1, t2] == 1:
                        if t2 < 7:
                            # Ensure there are non-zero entries in 'SPSA'
                            non_zero_count = np.count_nonzero(data['SPSA'][:, 0, 0])
                            if non_zero_count > 0:
                                # Safely assign values to 'SICA'
                                data['SICA'][0, t2, 0] += (
                                    1.1 * data['SEWW'][0,t1, 0] *
                                    np.sum(data['BSTC'][:, t1, 25 + t2]) /
                                    non_zero_count
                                )                        
                        elif t2 > 6 and t2 < 20:
                            data['SICA'][0, t2, 0] = data['SICA'][0, t2, 0] + 1.1 * data['SEWW'][0,t1,0]

                        elif t2 > 19 and t2 < 26:
                            data['SICA'][0, t2, 0] = data['SICA'][0, t2, 0]+ data['SEWW'][0,t1,0]
                        
                        elif t2 == 26:
                            data['SICA'][0, t2, 0] = data['SICA'][0, t2, 0] + data['SEWW'][0,t1,0] / 1.14
                    
            data['SEWI'] = data['SEWK'] - data_dt['SEWK']
            data['SEWI'] = np.where(data['SEWI'] < 0.0, 0.0, data['SEWI'])

            data['SEWI'][: , : , 0] = np.where((data['BSTC'][: , : , 5] > 0.0) , (data['SEWI'][: , : , 0] + (data_dt['SEWK'][: , :, 0]/data['BSTC'][: , : , 5])) , np.max((data['SEWK'][: , : , 0] - data_dt['SEWK'][: , : , 0]), 0))
    
    data_dt['SWSL'] = data['SEWS']
    data_dt['SWKL'] = data['SEWK']
    data_dt['SG1L'] = data['SGC1']
    data_dt['SD1L'] = data['SGD1']
    data_dt['SG2L'] = data['SGC2']
    data_dt['SD2L'] = data['SGD2']
    data_dt['SG3L'] = data['SGC3']
    data_dt['SD3L'] = data['SGD3']
    data_dt['BSTL'] = data['BSTC']
    data_dt['SWWL'] = data['SEWW']
    data_dt['SWYL'] = data['SWIY']
    data_dt['SWIL'] = data['SWII']
    data_dt['SWGL'] = data['SWIG']
    data_dt['SCML'] = data['SCMM']
    data_dt['SPCL'] = data['SPRC']
    data_dt['SICL'] = data['SICA']
    data_dt['SMPL'] = data['SEMS']

    isReg = np.where(data['SEWR'] > 0.0, 1.0 + np.tanh(1.5 + 10 * (data_dt['SWKL'] - data['SEWR'])) / data['SEWR'],isReg)

    isReg = np.where(data['SEWR'] == 0.0, 1.0,isReg)

    data_dt['SWSLt'] = data_dt['SWSL'] 
    data_dt['SWKLt'] = data_dt['SWKL'] 
    data_dt['SG1Lt'] = data_dt['SG1L'] 
    data_dt['SD1Lt'] = data_dt['SD1L'] 
    data_dt['SG2Lt'] = data_dt['SG2L'] 
    data_dt['SD2Lt'] = data_dt['SD2L'] 
    data_dt['SG3Lt'] = data_dt['SG3L'] 
    data_dt['SD3Lt'] = data_dt['SD3L'] 
    data_dt['BSTLt'] = data_dt['BSTL'] 
    data_dt['SWWLt'] = data_dt['SWWL'] 
    data_dt['SWYLt'] = data_dt['SWYL'] 
    data_dt['SWILt'] = data_dt['SWIL'] 
    data_dt['SWGLt'] = data_dt['SWGL'] 
    data_dt['SCMLt'] = data_dt['SCML'] 
    data_dt['SPCLt'] = data_dt['SPCL'] 
    data_dt['SICLt'] = data_dt['SICL'] 
    data_dt['SMPLt'] = data_dt['SMPL'] 
    data['SPSA'][:,0,0] = data['SPSP'][:,0,0] * growthRate1[:,0,0]

    if year > 2019:
        
        data = get_lcos(data,titles,data['SPSA'])

        for var in data.keys():
            data_dt[var] = copy.deepcopy(data[var])

        # if var_iter == 1 :
        data['SPSL'] = data['SPSP'] * growthRate2

        # if var_iter == 1:

        for t in range(1,int(invdt) + 1):
            spsa_dt[:,:] = 0.0
            spsa_dt = data['SPSL'] + (data['SPSA'] + data['SPSL']) * t/invdt
            spsa_dtl = data['SPSL'] + (data['SPSA'] + data['SPSL']) * (t-1)/invdt

            # Negation of the condition
            condition1 = (~(data['SEWR'][:, 25,0] > data_dt['SEWK'][:, 25,0])) & (data['SEWR'][:, 25,0] > -1.0)

            # Apply the first WHERE statement
            isReg[:, 25,0] = np.where(condition1, 1.0 - np.tanh(2 * 1.25 * data_dt['BSTC'][:, stti['Scrap - EAF'], c5ti['Scrap']] - 0.5) / 0.5, isReg[:, 25,0])

            # Second condition: if isReg > 1.0, set it to 1.0
            isReg[:, 25] = np.where(isReg[:, 25] > 1.0, 1.0, isReg[:, 25]) 

            for r in range(len(titles['RTI'])):
                            
                if np.any(spsa_dt[r] > 0.0):
                    for i in range(len(titles['STTI'])): 
                        if  not (data['SEWS'][r, i, 0] > 0.0 and
                            data['SGC1'][r, i, 0] != 0.0 and
                            data['SEWC'][r, i, 0] != 0.0):
                            continue
                                
                        S_i[r,i] = data['SEWS'][r, i, 0]

                        if S_i[r,i] > 0.0:
                            dSij[i,i] = 0

                            for k in range(i-1):                 
                                if not (data['SEWS'][r, k, 0] > 0.0 and
                                    data['SGC1'][r, k, 0] != 0.0 and 
                                    data['SEWC'][r, k, 0] != 0.0): 
                                    continue                      
                                        
                                S_k[r,k] = data['SEWS'][r, k, 0]

                                # if S_k[r,k] > 0.0:
                                
                                dFij = 1.414 * sqrt((data_dt['SEWC'][r, i, 0] * data_dt['SEWC'][r, i, 0] 
                                            + data_dt['SEWC'][r, k, 0] * data_dt['SEWC'][r, k, 0]))
                                
                                # Investor preference incl. uncertainty
                                Fij = 0.5 * (1 + np.tanh(1.25 * (data_dt['SGC1'][r, k, 0]
                                            - data_dt['SGC1'][r, i, 0])) / dFij)

                                F[i, k] = Fij * (1.0 - isReg[r, i]) * (1.0 - isReg[r,i]) + isReg[r, k] \
                                                            * (1.0 - isReg[r, i]) + 0.5 * (isReg[r, i] * isReg[r, k])
                                F[k, i] = (1.0 - Fij) * (1.0 - isReg[r, k]) * (1.0 - isReg[r, i]) + isReg[r, i] \
                                                            * (1.0 - isReg[r, k]) + 0.5 * (isReg[r, k] * isReg[r, i])

                                k_1[i,k] = S_i[r,i] * S_k[r,k] * (data['SEWA'][0, i, k] * F[i,k] - data['SEWA'][0, k, i]* F[k,i])
                                k_1[k,i] = -k_1[i,k]
                                k_2[i,k] = (S_i[r,i] + dt * k_1[i,k]/2) * (S_k[r,k] - dt * k_1[k,i]/2) * (data['SEWA'][0, i, k] * F[i,k] - data['SEWA'][0, k, i] * F[k,i])
                                k_2[k,i] = -k_2[i,k]
                                k_3[i,k] = (S_i[r,i] + dt * k_2[i,k]/2) * (S_k[r,k] - dt * k_2[k,i]/2) * (data['SEWA'][0, i, k] * F[i,k]- data['SEWA'][0, k, i]  * F[k,i])
                                k_3[k,i] = -k_3[i,k]
                                k_4[i,k] = (S_i[r,i] + dt * k_3[i,k]) * (S_k[r,k] - dt * k_3[k,i]) * (data['SEWA'][0, i, k] * F[i,k] - data['SEWA'][0, k, i] * F[k,i])
                                k_4[k,i] = -k_4[i,k]

                                dSij[i, k] = dt * (k_1[i,k]+2*k_2[i,k]+2*k_3[i,k]+k_4[i,k])/6/tScaling   ## change
                                dSij[k, i] = -dSij[i, k]

                    SR[r,:,0] = 1/data['BSTC'][r, :, c5ti['Payback period']] - 1/data['BSTC'][r, :, c5ti['Lifetime']]
                    # Constant used to reduce the magnitude of premature scrapping 
                    SR_C = 2.5 

                    for i in range(len(titles['STTI'])):
                        if  not (data['SEWS'][r, i, 0] > 0.0 and
                            data['SGC1'][r, i, 0] != 0.0 and
                            data['SEWC'][r, i, 0] != 0.0):
                            continue
                                
                        SE_i[r,i] = data['SEWS'][r, i, 0]

                        if SE_i[r,i] > 0.0:
                            dSEij[i,i] = 0

                            for k in range(i-1):                 
                                if not (data['SEWS'][r, k, 0] > 0.0 and
                                    data['SGC1'][r, k, 0] != 0.0 and 
                                    data['SEWC'][r, k, 0] != 0.0): 
                                    continue                      
                                        
                                SE_k[r,k] = data['SEWS'][r, k, 0]

                                # if SE_k[r,k] > 0.0:
                                dFEij = 1.414 * sqrt((data_dt['SGD3'][r, i, 0] * data_dt['SGD3'][r, i, 0] + data_dt['SGD2'][r, k, 0] * data_dt['SGD2'][r, k, 0]))
                                dFEji = 1.414 * sqrt((data_dt['SGD2'][r, i, 0] * data_dt['SGD3'][r, i, 0] + data_dt['SGD3'][r, k, 0] * data_dt['SGD3'][r, k, 0]))
                                    
                                # Preferences based on cost differences by technology pairs (asymmetric!)
                                FEij = 0.5*(1+np.tanh(1.25*(data_dt['SGC2'][r, k, 0] - data_dt['SGC3'][r, i, 0])/dFEij))
                                FEji = 0.5*(1+np.tanh(1.25*(data_dt['SGC2'][r, i, 0] - data_dt['SGC3'][r, k, 0])/dFEji))

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
                                dSEij[k,i] = - dSEij[i,k]            
                
                    for i in range(len(titles['STTI'])):
                        endo_shares[r,i] = 0.0
                        endo_capacity[r,i] = 0.0

                    endo_shares[r,:] = data_dt['SEWS'][r, :] + np.sum(dSij,axis = 1) + np.sum(dSEij, axis = 1)
                    endo_eol[r,:] = np.sum(dSij,axis=1) ## this is getting (26,)
                    endo_capacity[r,:] = spsa_dt[r, 0]/np.sum(endo_shares[r,:] * data['BSTC'][r, :, c5ti['CF']]) * endo_shares[r,:]

                    endo_gen[r,:] = endo_capacity[r,:] * (data['BSTC'][r, :, c5ti['CF']]).reshape(26,1)

                    demand_weight[r] =  np.sum(endo_shares[r,:] *(data['BSTC'][r, :, c5ti['CF']]))

                    Utot = np.sum(endo_shares[r,:] * demand_weight[r])

                    dUkKST[r,:,0] = np.where(data['SWKA'][r, :, 0] < 0,
                                                data['SKST'][r, :, 0] * Utot * dt, 0.0)
                    
                    bstc_reshaped = data['BSTC'][r, :, c5ti['CF']].reshape(26,1)
                    dUkKST[r,:] = np.where(((dUkKST[r,:] / bstc_reshaped) + endo_capacity[r,:] > data['SEWR'][r, :]) & (data['SEWR'][r, :] >= 0.0), 0.0, dUkKST[r,:])

                    dUkREG[r,:] = np.where((endo_capacity[r,:] * demand_weight[r] - (endo_shares[r,:] * spsa_dtl[r])) > 0, - ((endo_capacity[r,:] * demand_weight[r]) - endo_shares[r,:] * spsa_dtl[r]) * isReg[r, :],0.0)

                    dUkSk[r,:] = np.where((data['SWKA'][r, :] < endo_capacity[r,:]),
                                                ((data['SWKA'][r, :] - endo_capacity[r,:]) * demand_weight[r] - dUkREG[r,:] - dUkKST[r,:]) * (t/invdt),
                                                0.0)                                    
                
                    dUkSk[r,:] = np.where((data['SWKA'][r, :] > endo_capacity[r,:]) & (data['SWKA'][r, :] > data_dt['SWKL'][r, :]),
                                    (data['SWKA'][r, :] - endo_capacity[r,:]) * demand_weight[r] * (t/invdt),
                                        dUkSk[r,:])   
                                                                                            
                    dUkSk[r, :] = np.where(np.logical_or(data['SWKA'][r, :] < 0.0, np.logical_and(data['SEWR'][r, :] >= 0.0, data['SWKA'][r, :] > data['SEWR'][r, :])), 
                                            0, dUkSk[r, :])
                    
                    dUk[r,:] = dUkREG[r,:] + dUkSk[r,:] + dUkKST[r,:]
                    dUtot  = np.sum(dUk[r,:])
            
                    if np.sum(endo_capacity[r,:]*demand_weight[r] + dUk[r,:]) > 0:
                        data['SWSA'][r,:] = dUk[r,:] / np.sum(endo_capacity[r,:]*demand_weight[r] + dUk[r,:])

                        data['SEWS'][r,:] = (endo_capacity[r,:] * demand_weight[r] + dUk[r,:])/np.sum(endo_capacity[r,:]*demand_weight[r] + dUk[r,:])
                    
                    data['SEOL'][r,:] = np.sum(dSij, axis=1)
                    data['SBEL'][r,:] = np.sum(dSEij, axis=1)

                    data['SEWK'][r,:] = spsa_dt[r]/np.sum(data['SEWS'][r,:]*data['BSTC'][r,:,11]) * data['SEWS'][r,:]
                    data['SEWG'][r,:] = data['SEWK'][r,:] * data['BSTC'][r,:,c5ti['CF']].reshape(26,1)  ## For SEWG
                    data['SEWE'][r,:] = data['SEWG'][r,:] * data['STEF'][r,:]/1e3

    ####################################################################
                if t==1:
                    data['SEWI'][r, :] = 0.0
                    data_dt['SEWI'][r, :] = 0.0
            
                if (endo_eol[r,:] >= 0).all() and (data['BSTC'][r,:,5] > 0).all():
                    eol_replacements_t[r,:] = data['SWKL'][r,:] * dt /data['BSTC'][r,:,5].reshape(26,1)

                condition1 = (data_dt['SWSL'] * dt / data['BSTC'][r, :, 5].reshape(26,1)) < endo_eol[r, :]
                condition2 = endo_eol[r, :] < 0
                condition3 = data['BSTC'][r, :, 5] > 0

                if (condition1 & condition2).all() and condition3.all():
                    eol_replacements_t[r,:] = (data['SEWS'][r,:] - data_dt['SWSL'][r,:] + data_dt['SWSL'] * dt/data['BSTC'][r,:,5]) * data['SWKL'][r,:]
            
                if ((data['SEWK'][r,:] - data_dt['SWKL'][r,:]) > 0).all():
                    data_dt['SEWI'][r,:] = data['SEWK'][r,:] - data_dt['SWKL'][r,:] + eol_replacements_t[r,:]
                
                else:
                    data_dt['SEWI'][r,:] = eol_replacements_t[r,:]

                data['SEWI'][r, :] += data_dt['SEWI'][r, :]

                data['SPMT'][r,:11] = data['SPMA'][r,:11]

                if r < 33:
                    data['SPMT'][r,15] = data['SPMA'][r,15]
                    data['SPMT'][r,16] = data['SPMA'][r,16]
                else:
                    data['SPMT'][r,15] = data['SPMA'][r,15]
                    data['SPMT'][r,16] = data['SPMA'][r,16]
                
                data['SPMT'][r,17] = data['SPMA'][r,17]
                data['SPMT'][r,18] = data['SPMA'][r,18]
                data['SPMT'][r,19] = data['SPMA'][r,19]

            sewi0 = np.sum(data_dt['SEWI'],axis=0)

            for i in range(len(titles['STTI'])):
                dW_temp = sewi0[None,:]
                dW[i] = np.matmul(dW_temp[0,:,0], data['SEWB'][0,i,:])   

            data['SEWW'][0,:,0] = data_dt['SWWL'][0,:,0] + dW[:,0,0]

            data['BSTC'][: , : , :22] = data_dt['BSTC'][: , : , :22]
            data['SCMM'] = data_dt['SCML'] 
                
            # data, hewi_t = get_sales(data, data_dt, time_lag, titles, dt, t, endo_eol)    ## line 820 to 883

            for b in range(len(titles['STTI'])): ## line 846 to 853
                if data['SEWW'][0, b, 0] > 0.0001:

                    data['BSTC'][:, b, c5ti['IC']] = (data_dt['BSTC'][:, b, c5ti['IC']]  \
                                                                                *(1.0 + data['BSTC'][:, b, c5ti['Learning rate (IC)']] * dW[b]/data['SEWW'][0, b, 0]))
                    data['BSTC'][:, b, c5ti['dIC']] = (data_dt['BSTC'][:, b, c5ti['dIC']]  \
                                                                                    *(1.0 + data['BSTC'][:, b, c5ti['Learning rate (IC)']] * dW[b]/data['SEWW'][0, b, 0]))

            data['SCIN'] = data['BSTC'][:,:,0].reshape(71,26,1)
            
            for t1 in range(len(titles['STTI'])):
                for t2 in range(len(titles['SSTI'])):
                    if data['STIM'][0, t1, t2] == 1:
                        if t2 < 7:
                            # Ensure there are non-zero entries in 'SPSA'
                            non_zero_count = np.count_nonzero(spsa_dt[:, 0, 0])
                            if non_zero_count > 0:
                                # Safely assign values to 'SICA'
                                data['SICA'][0, t2, 0] += (
                                    1.1 * data['SEWW'][0,t1, 0] *
                                    np.sum(data['BSTC'][:, t1, 25 + t2]) /
                                    non_zero_count
                                )                        
                        elif t2 > 6 and t2 < 20:
                            data['SICA'][0, t2, 0] = data['SICA'][0, t2, 0] + 1.1 * data['SEWW'][0,t1,0]

                        elif t2 > 19 and t2 < 26:
                            data['SICA'][0, t2, 0] = data['SICA'][0, t2, 0]+ data['SEWW'][0,t1,0]
                        
                        elif t2 == 26:
                            data['SICA'][0, t2, 0] = data['SICA'][0, t2, 0] + data['SEWW'][0,t1,0] / 1.14

            sica_lr = np.ones((len(titles['SSTI']),1,1)) *-0.015
            # sica_lr = -0.015

            for mat in range(len(titles['SMTI'])):
                for t2 in range(len(titles['SSTI'])):
                    if data['SICA'][0,t2,0] - data_dt['SICL'][0,t2,0] > 0.0:
                        data['SCMM'][0,mat,t2] = (data_dt['SCML'][0,mat,t2] - data['SEEM'][0,mat,t2]) * (1.0 + sica_lr[t2] * (data['SICA'][0,t2,0] - data_dt['SICL'][0,t2,0]) / data_dt['SICL'][0,t2,0]) + data['SEEM'][0,mat,t2]
            
            data['SLCI'][0,:,4:11] = data['SCMM'][0,:, :7]

            # if (var_iter < 10) and (t == invdt):
            data = raw_material_distr(data, titles, year, t,spsa_dt)

            data['SEIA'][:,0,0]  = np.sum(data['STEI'][:,:,0] * data['SEWS'][:,:,0] ,axis=1) 

            data['SEMS'][:,:,0] = (data['SEWK'][:,:,0] * data['BSTC'][:,:,4] )* 1.1

            # data_dt['SMPL'] = data['SMPL']

            if np.sum(data['SMPL']).all() > 0.0:
                data['SEMR'] = np.sum(data['SEMS'],axis=1)/np.sum(data['SMPL'],axis=1)
            
            data = get_lcos(data,titles,spsa_dt)
            data['SJEF'], data['SJCO']= ftt_s_fuel_consumption(data['BSTC'], data['SEWG'], data['SMED'], len(titles['RTI']), len(titles['STTI']), c5ti, len(titles['JTI']))

            data_dt['SWSLt'] = data['SEWS'] 
            data_dt['SWKLt'] = data['SWKL'] 
            data_dt['SG1Lt'] = data['SGC1'] 
            data_dt['SD1Lt'] = data['SGD1'] 
            data_dt['SG2Lt'] = data['SGC2'] 
            data_dt['SD2Lt'] = data['SGD2'] 
            data_dt['SG3Lt'] = data['SGC3'] 
            data_dt['SD3Lt'] = data['SGD3'] 
            data_dt['BSTLt'] = data['BSTC'] 

            data_dt['SWWLt'] = data_dt['SWWL'] 
            # data_dt['SWYLt'] = data_dt['SWYL'] 
            data_dt['SWILt'] = data_dt['SWII'] 
            data_dt['SWGLt'] = data_dt['SWIG'] 
            data_dt['SCMLt'] = data_dt['SCMM'] 
            data_dt['SPCLt'] = data_dt['SPRL'] 
            data_dt['SICLt'] = data_dt['SICA'] 
            data_dt['SMPLt'] = data_dt['SEMS'] 
        
            
        ### Commented in the original python code
        # if var_iter > 1:
        data['SPSA'] = data['SPSP'] * growthRate1

        for r in range(len(titles['RTI'])):
            if data['SPSA'][r] > 0.0 :
                data['SEWK'][r,:] = data['SPSA'][r]/np.sum(data['SEWS'][r,:] * data['BSTC'][r,:,11]) * data['SEWS'][r,:]
                data['SEWG'][r,:] = data['SEWK'][r,:] * data['BSTC'][r,:,c5ti['CF']].reshape(26,1) # For SEWG
                data['SEWE'][r,:] = data['SEWG'][r,:] * data['STEF'][r,:] / 1e3

                eol_replacements_t[r,:] = 0.0
                eol_replacements[r,:] = 0.0

                if (data['SEWS'][r,:] - data_dt['SWSL'][r,:]).all() > 0 and data['BSTC'][r,:,c5ti['Lifetime']] > 0.0:
                    eol_replacements[r,:] = data_dt['SWKL'][r,:]/data['BSTC'][r,:,c5ti['Lifetime']]
                
                if ((-data_dt['SWSL'][r,:]/data['BSTC'][r,:,c5ti['Lifetime']] < data['SEWS'][r,:] - data_dt['SWSL'][r,:]).all() < 0) and (data['BSTC'][r,:,c5ti['Lifetime']] > 0.0):
                    eol_replacements[r,:] = (data['SEWS'][r,:] - data_dt['SWSL'][r,:] + data_dt['SWSL'][r,:]/data['BSTC'][r,:,c5ti['Lifetime']]) * data_dt['SWKL'][r,:]
                
                if (data['SEWK'][r,:] - data_dt['SWKL'][r,:]).all() > 0:
                    data['SEWI'][r,:] = (data['SEWK'][r,:] - data_dt['SWKL'][r,:]) + eol_replacements[r,:]

                else:
                    data['SEWI'][r,:] = eol_replacements[r,:]
                
                data['SPMT'][r,:11] = data['SPMA'][r,:11]
                data['SPMT'][r,11] = data['SMED'][0,11,0]
                data['SPMT'][r,12] = data['SMED'][0,12,0]
                if r < 33:
                    data['SPMT'][r,15] = data['SPMA'][r,15]
                    data['SPMT'][r,16] = data['SPMA'][r,16]
                else:
                    data['SPMT'][r,15] = data['SPMA'][r,15]
                    data['SPMT'][r,16] = data['SPMA'][r,16]
                
                data['SPMT'][r,17] = data['SPMA'][r,17]
                data['SPMT'][r,18] = data['SPMA'][r,18]
                data['SPMT'][r,19] = data['SPMA'][r,19]

            sewi0 = np.sum(data['SEWI'],axis=0)

            for i in range(len(titles['STTI'])):
                dW_temp = sewi0[None,:]
                dW[i] = np.dot(dW_temp[0,:,0], data['SEWB'][0,i,:])   

            data['SEWW'][0,:,0] = data_dt['SWWL'][0,:,0] + dW[0,:,0]

            # if var_iter < 10 :
            data = raw_material_distr(data, titles, year, t,spsa_dt)

            data['SEIA'][r] = np.sum((data['STEI'][r,:] * data['SEWS'][r,:]),axis=0) 

            data['SEMS'][:,:,0] = data['SEWK'][:,:,0] * data['BSTC'][:,:,4] * 1.1

            data['SEMR'] = np.where(np.sum(data_dt['SMPL']) > 0.0, np.sum(data['SEMS'],axis=1,keepdims=True)/np.sum(data['SMPL'],axis=1,keepdims=True),data['SEMR'])

            data = get_lcos(data,titles,spsa_dt)
            data['SJEF'], data['SJCO']= ftt_s_fuel_consumption(data['BSTC'], data['SEWG'], data['SMED'], len(titles['RTI']), len(titles['STTI']), c5ti, len(titles['JTI']))
            
    return data
