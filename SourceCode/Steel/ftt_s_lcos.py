# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:3np.zeros(len(titles['STTI'])) 2024

@author: Varun

=========================================
ftt_s_lcos.py
=========================================
Domestic Heat FTT module.
####################################

This is the LOOS file for FTT: Steel, which calculates levelized cost of steel production

The outputs of this module include levelised cost of steel production technologies

Local library imports:

Support functions:

- `divide <divide.html>`__
    Bespoke element-wise divide which replaces divide-by-zeros with zeros
- `estimation <econometrics_functions.html>`__
    Predict future values according to the estimated coefficients.

Parameters
----------
- get_lcos(data, titles)
- data: dictionary
    Data is a container that holds all cross-sectional (of time) for all
    variables. Variable names are keys and the values are 3D NumPy arrays.
- titles: dictionary
    Titles is a container of all permissible dimension titles of the model.

Returns
----------
data: dictionary
    Data is a container that holds all cross-sectional (of time) data for
    all variables.
    Variable names are keys and the values are 3D NumPy arrays.
    The values inside the container are updated and returned to the main
    routine.

"""

# Third party imports
import numpy as np
import math

# Local library imports
from SourceCode.support.divide import divide

# %% LCOS
# --------------------------------------------------------------------------
# -------------------------- LCOS function ---------------------------------
# --------------------------------------------------------------------------

def get_lcos(data, titles,spsa_dt):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of heat in 2014 Euros/kWh per
    boiler type. It includes intangible costs (gamma values) and together
    determines the investor preferences.
    """
    TotCost_I = np.zeros(len(titles['STTI'])) #Total cost due to material/fuel consumption
    TaxTotCost_I = np.zeros(len(titles['STTI']))          #Total cost due to material/fuel consumption inc. additional tax/subsidies on materials
    IC_I = np.zeros(len(titles['STTI']))                  #Investment costs 
    OM_I = np.zeros(len(titles['STTI']))                  #O&M costs
    EF_I = np.zeros(len(titles['STTI']))                  #Emission factors - total
    EF_I_Fossil = np.zeros(len(titles['STTI']))           #Emission factors - fossils only
    EF_I_Biobased = np.zeros(len(titles['STTI']))         #Emission factors - biobased only
    EI_I = np.zeros(len(titles['STTI']))                  #Energy intensity
    CF_I = np.zeros(len(titles['STTI']))                  #Capacity factors
    r_I = np.zeros(len(titles['STTI']))                   #Discount rate
    L_I = np.zeros(len(titles['STTI']))                   #Lifetime
    B_I = np.zeros(len(titles['STTI']))                   #Leadtime
    Sub_I = np.zeros(len(titles['STTI']))                 #Subsidies/tax on investment costs
    It = np.zeros(len(titles['STTI'])) #investment cost ($(2008)/tcs)), mean
    dIt = np.zeros(len(titles['STTI'])) #investment cost ($(2008)/tcs)), SD
    St = np.zeros(len(titles['STTI']))  #upfront subsidy/tax on investment cost ($(2008)/tcs))
    Ft = np.zeros(len(titles['STTI'])) #fuel cost ($(2008)/tcs)), mean
    dFt = np.zeros(len(titles['STTI'])) #fuel cost ($(2008)/tcs)), SD
    FTt = np.zeros(len(titles['STTI'])) #fuel tax/subsidy (($(2008)/tcs)))
    OMt = np.zeros(len(titles['STTI'])) #O&M cost (($(2008)/tcs))), mean
    dOMt = np.zeros(len(titles['STTI'])) #O&M cost ($(2008)/tcs)), SD
    CO2t = np.zeros(len(titles['STTI'])) #CO2 cost ($(2008)/tcs)), mean
    dCO2t = np.zeros(len(titles['STTI'])) #CO2 cost ($(2008)/tcs)), SD
    Pt = np.zeros(len(titles['STTI'])) #Production of steel (np.zeros(len(titles['STTI'])) during leadtime, 1 during lifetime)
    
    TMC = np.zeros(len(titles['STTI']))
    dTMC = np.zeros(len(titles['STTI']))
    PB = np.zeros(len(titles['STTI']))
    dPB = np.zeros(len(titles['STTI']))
    TPB = np.zeros(len(titles['STTI']))
    dTPB = np.zeros(len(titles['STTI']))
    local_average_price = np.zeros(len(titles['RTI']))  
    iron_demand = np.zeros(len(titles['RTI'])) 
    iron_supply = np.zeros(len(titles['RTI']))
    TaxTotCost = np.zeros(len(titles['STTI']))
    inputprices_taxes = np.zeros((len(titles['SMTI'])))
    inv_mat_I = np.zeros ((len(titles['SMTI'])))
 
    
    #Calculate cost of Iron making, price of Iron production, inventory, emission factors

    for r in range(len(titles['RTI'])):
        if (spsa_dt[r]) > 0:  
            inputprices_taxes = 0.0
            inputprices_taxes = (1 + data['STRT'][r,:] * data['SPMT'][r,:])  
            mkt_shares = data['SEWS'][r,:,0]  
            #LCOI Calculation starts here         
            TotCost_I[:] = 0.0
            TaxTotCost_I[:] = 0.0
            IC_I[:] = 0.0
            OM_I[:] = 0.0
            EF_I[:] = 0.0
            EF_I_Fossil[:] = 0.0
            EF_I_Biobased[:] = 0.0
            EI_I[:] = 0.0
            CF_I[:] = 0.0
            data['SITC'][r,:] = 0.0
            r_I[:] = 0.0
            L_I[:] = 0.0
            B_I[:] = 0.0
            Sub_I[:] = 0.0
            data['SWGI'][r,:] = 0.0

            for path in range(len(titles['STTI'])):
                for plant in range (len(titles['SSTI'])):
                    #Not all integrated steelmaking routes have an Iron making step and the condition below skips those routes.
                    #Also, only a specific range of the NSS classification is required
                    ## Steel making process
                    if (data['STIM'][0, path, plant] == 1) and (8 <= plant <= 20):
                        #First, add IronmakingP
                        inv_mat_I[:] = 0.0
                        inv_mat_I[:] = data['SCMM'][0, :, plant]
                    
                        #Second, add sinter and pellet
                        inv_mat_I[:] = inv_mat_I [:] + inv_mat_I [6] * data['SCMM'] [0,:,2] + inv_mat_I [7] * data['SCMM'] [0,:,3] + inv_mat_I [8] * data['SCMM'] [0,:,4] + inv_mat_I [9] * data['SCMM'] [0,:,5]
                        #Third, add coke
                        inv_mat_I[:] = inv_mat_I [:] + inv_mat_I [4] * data['SCMM'] [0,:,0] + inv_mat_I [5] * data['SCMM'] [0,:,1]
                        #Fourth, add oxygen
                        inv_mat_I[:] = inv_mat_I [:] + inv_mat_I [10] * data['SCMM'] [0,:,6] 
                        
                        #Increase fuel consumption if CCS
                        if data['BSTC'] [r,path,21] == 1:

                            inv_mat_I [11:15] *= 1.1   ## changed
                            inv_mat_I [17:20] *= 1.1   ## changed

                        ## Saving all SCMM for each plant in one variable
                        IC_I[path] = data['SCMM'][0, 20, plant]
                        OM_I[path] = data['SCMM'][0, 21, plant]

                        ## Finding emission factor using SMEF for different fuel types
                        EF_I[path] = np.sum (inv_mat_I * data['SMEF'][0, :, 0])
                        EF_I_Fossil[path] = np.sum (inv_mat_I[:15] * data['SMEF'][0, :15, 0])
                        EF_I_Biobased[path] = np.sum (inv_mat_I[15:24] * data['SMEF'][0, 15:24, 0])
    
                        ## Calculating Energy intensity (iron production) using material specific emission factor
                        data['SIEI'][r, path, 0] = np.sum (inv_mat_I * data['SMED'][0, :, 0])
                        CF_I[path] = 0.9    

                        ## Calculating total cost and tax 
                        TotCost_I[path] = np.sum(inv_mat_I * data['SPMT'][r, :, 0])   
                        TaxTotCost_I[path] = np.sum(inv_mat_I * inputprices_taxes) 


                        ## Initiliazing cost matrix for all the regions and path 
                        r_I[path] = data['BSTC'][r,path,9]
                        L_I[path] = data['BSTC'][r,path,5]
                        B_I[path] = data['BSTC'][r,path,6]

                        ## Subsidies/tax on investment cost
                        Sub_I[path] = data['SEWT'][r,path,0]    
                        data['SWGI'][r, path,0] = data['SEWG'][r, path, 0] * CF_I[path]
                        data['SIEF'][r,path] = 0.0
                        
                        #IEmission factor if CCS
                        if data['BSTC'][r, path, 21] == 1:
                            data['SIEF'][r, path, 0] = 0.1 * EF_I[path] - EF_I_Biobased[path]
                        else:
                            data['SIEF'][r, path, 0] = EF_I_Fossil[path]
                        
                        #Calculation of levelised cost starts here
                        if spsa_dt[r, 0, 0] > 0.0:
                            
                            It_I = 0.0
                            St_I = 0.0
                            Ft_I = 0.0
                            FTt_I = 0.0
                            OMt_I = 0.0
                            NPV1p_I = 0.0
                            NPV2p_I = 0.0

                            for t in range (0, int(B_I[path] + L_I[path])):  
                                if t < B_I[path]:
                                   #Investment costs are divided over each building year
                                   It_I = IC_I[path] / (CF_I[path] * B_I[path])
                                   #Tech-specific subsidy or tax
                                   St_I = Sub_I[path] * It_I
                                   #No material cost, tax/subsidy, O&M costs, CO2 tax during construction
                                   Ft_I = 0.0
                                   FTt_I = 0.0
                                   OMt_I = 0.0
                                   data['SCOI'][r,path] = 0.0
                                else:
                                   #No Investment costs or subsidy or tax after construction
                                   It_I = 0.0
                                   St_I = 0.0
                                   #Material costs
                                   Ft_I = TotCost_I[path]
                                   #Subsidy or tax on materials
                                   FTt_I = TaxTotCost_I[path]
                                   #Operation and maintenance costs
                                   OMt_I = OM_I[path]
                                   #Costs due to CO2 tax 
                                   #NEW CALCULATION OF SCOT FOR PI!!!!!
                                   data['SCOI'][r,path] = EF_I[path] #*(data['REPP'][r]*data['FETS'][r,3] + data['RTCA'][r,year-2000]*data['FEDS'][r,3])  * data['REX13'](33) / (data['PRSC'][r]*data['EX13'][r]/(data['PRSC13'][r]*data['EX'][r])) /3.66  
                                
                                NPV1p_I = NPV1p_I + (It_I + St_I + FTt_I + OMt_I + data['SCOI'][r,path] ) / (1 + r_I[path]) ** t ## 8
                                NPV2p_I = NPV2p_I + 1 / (1 + r_I[path]) ** t 
                            
                            data['SITC'][r, path, 0] = NPV1p_I / NPV2p_I #Levelised cost of ironmaking, this is used to determine the cost of pig iron which may be used in the Scrap-EAF route
                            
                        else:
                            data ['SITC'][r, path, 0] = 0.0
            
            #The average price of pig iron is the levelised costs of ironmaking times the share of each ironmaking technology
            data['SIPR'][r] = 0.0
            
            #Total production of intermediate iron product
            data['STGI'][r] = np.sum(data['SWGI'][r, :])
            if data['STGI'][r] > 0.0:
                data['SIPR'][r] = np.sum(data['SITC'][r, :] * data['SWGI'][r, :]) / np.sum(data['SWGI'][r,:])
            
    #Global average price:
    mean_price_I = np.sum(data['SIPR'] * data ['STGI'] / np.sum(data['STGI']))  
    local_average_price[:] = 0.0
    iron_demand[:] = 0.0
    iron_supply[:] = 0.0            
    
    #Global average EF for intermediate iron (to connect to imported iron by Scrap - EAF route)
    mean_EF_I = np.sum(data['SIEF'] * data['SWGI']) / np.sum (data['STGI']) 

    for r in range(len(titles['RTI'])):
        if (spsa_dt[r] > 0.0):  
            inputprices_taxes = (1 + data['STRT'][r,:]) * data['SPMT'][r,:]   ## changed
            iron_demand[r] = data['BSTC'][r, 25, 24] * data['SEWG'][r, 25]
            
            EF_sec_route = 0.0
            if data['STGI'][r] > np.sum (data['BSTC'][r, :, 23] * data['SEWG'] [r, :]):
                iron_supply[r] = data['STGI'][r , 0, 0] - np.sum (data['BSTC'][r, :, 23] * data['SEWG'][r,:])
            
            if iron_demand[r] > 0.0:
                #The price of intermediate iron can only be used directly if there's enough iron production 
                if data['SIPR'][r] > 0.0 and data['SIPR'][r] < 3 * data['SPMT'][r, 2] and iron_supply[r] >= iron_demand[r]:
                    data['SPMT'][r, 2] = data['SIPR'][r]
                #If there isn't enough domestic supply of iron, then a part of the price has to be calculated from global price and global EF of iron production resp.  
                elif data['SIPR'][r] > 0.0 and data['SIPR'][r] < 3 * data['SPMT'][r, 2] and iron_supply[r] > 0.0 and iron_demand[r] > iron_supply[r]:
                    local_average_price[r] = ((iron_demand[r] - iron_supply[r]) * mean_price_I + iron_supply[r] * data ['SIPR'][r]) / iron_demand[r]
                    data['SPMT'][r, 2] = local_average_price[r]
                #If there's no iron production (only MOE or Scrap-EAF present) or local price is ridiculously high, then take global average.
                else:
                    data['SPMT'][r, 2] =  mean_price_I
            
                #Emissions related to iron production that is used in the Scrap-EAF route need to be allocated to this route
                #There's enough domestic supply to cover the domestic demand
                if iron_supply[r] >= iron_demand[r]:
                    EF_sec_route = np.sum (data['SIEF'][r, :] * data['SWGI'][r, :] / data['STGI'][r])
                #If there's some iron supply, but not enough demand,partly use the global mean    
                elif iron_supply[r] < iron_demand[r] and iron_supply[r] > 0.0:         
                    EF_sec_route = ((iron_demand[r] - iron_supply[r]) * mean_EF_I + np.sum (data['SIEF'][r, :] * data['SWGI'] [r, :] / data['STGI'][r]) * iron_supply[r]) / iron_demand[r]
                #If there's no iron supply at all, use global mean completely.
                else:
                    EF_sec_route = mean_EF_I
                    
            #LCOS Calculation starts here
            for path in range(len(titles['STTI'])):
                if (spsa_dt[r] > 0.0):
                    #Initialize CostMatrix variables
                    IC = data ['BSTC'] [r, path, 0]
                    dIC = data ['BSTC'] [r, path, 1]
                    OM = data ['BSTC'] [r, path, 2]
                    dOM = data ['BSTC'] [r, path, 3]
                    L = data ['BSTC'] [r, path, 5]
                    B = data ['BSTC'] [r, path, 6]
                    dr = data ['BSTC'] [r, path, 9]
                    Gam = data ['BSTC'] [r, path, 10]
                    CF = data ['BSTC'] [r, path, 11]
                    dCF = data ['BSTC'] [r, path, 12]
                    
                    #Adjust emission factor for Scrap-EAF due to emissions.
                    if path == 25:
                        data['STEF'][r, path] = data['BSTC'][r, path, 13] + data['BSTC'][r, path, 24] * EF_sec_route
                    
                    EF = data['STEF'][r, path]
                    dEF = data['BSTC'][r, path, 14]
                    TotCost = 0.0
                    dTotCost = 0.0 
                    TaxTotCost[:] = 0.0
                    
                    for materialindex in range(len(titles['SMTI']) - 4):
                        TotCost = TotCost + data['BSTC'][r, path, 21 + materialindex] * data['SPMT'][r, materialindex]
                        TaxTotCost[path] = TaxTotCost[path] + data['BSTC'][r, path, 21 + materialindex] * inputprices_taxes[materialindex]
                        dTotCost = dTotCost + 0.1 * data['BSTC'][r, path, 21 + materialindex] * data['SPMT'][r, materialindex]
                    
                    It[:] = 0.0
                    dIt[path] = 0.0
                    St[:] = 0.0
                    Ft[:] = 0.0
                    dFt[path] = 0.0
                    FTt[:] = 0.0
                    OMt[:] = 0.0
                    dOMt[path] = 0.0
                    CO2t[:] = 0.0
                    dCO2t[:] = 0.0
                    NPV1 = 0 #Discounted costs for the LCOS calculation
                    NPV2 = 0 #Denominator for the LCOS calculation 
                    NPV1o = 0
                    NPV1p = 0 #Discounted costs for the TLCOS/LTLCOS calculation
                    NPV2p = 0 #Denominator for the TLCOS/LTLCOS calculation
                    dNPV =  0 #SD for the TLCOS/LTLCOS calculation
                    dNPVp = 0 #SD for the TLCOS/LTLCOS calculation
                    ICC = 0 #Investment cost component of LTLCOS
                    FCC = 0 #Fuel/Material cost component of LTLCOS
                    OMC = 0 #O&M cost component of LTLCOS
                    CO2C =0  #CO2 tax cost component of LTLCOS
                    Pt[:] = 0.0

                    for t in range (int(B+L) - 1):
                        if t < B:
                            #Investment costs are divided over each building year
                            It[path] = IC / (CF * B)
                            dIt[path] = dIC / (CF * B)
                            #Tech-specific subsidy or tax
                            St[path] = data['SEWT'][r, path] * It[path]
                            #No material cost, tax/subsidy, O&M costs, CO2 tax during construction
                            Ft[path] = 0.0
                            dFt[path] = 0.0
                            FTt[path] = 0.0
                            OMt[path] = 0.0
                            dOMt[path] = 0.0
                            CO2t[path] = 0.0
                            dCO2t[path] = 0.0
                            Pt[path] = 1.0 #No production
                        else:
                            #No Investment costs or subsidy or tax after construction
                            It[path] = 0
                            dIt[path] = 0
                            St[path] = 0
                            #Material costs
                            Ft[path] = TotCost
                            dFt[path] = dTotCost
                            #Material subsidy or tax
                            FTt[path] = TaxTotCost[path]
                            #Operation and maintenance
                            OMt[path] = OM
                            dOMt[path] = dOM
                            #CO2 tax
                            # NOTE: We use the emission factors stored in BSTC rather than the ones stored in STEF. 
                            #EFs in STEF are altered to take into account the emissions due to iron production that is used in the scrap-EAF route.
                            #These emissions do not take place directly in the Scrap-EAF route and therefore this technology does not bare the costs.
                            data['SCOT'][r, path] = data['BSTC'][r, path, 13] #* (data['REPP'][r]*data['FETS'][r,3] + data['RTCA'][r,year-2000]*data['FEDS'][r,3])  * data['REX13'](33) / (data['PRSC'][r]*data['EX13'][r]/(data['PRSC13'][r]*data['EX'][r])) /3.66
                            dCO2t[path] = 0.1 * data['SCOT'][r, path] 
                            Pt[path] = 1.0   

                        NPV1 = NPV1 + (It[path] + Ft[path] + OMt[path]) / (1+dr) ** t
                        NPV1p = NPV1p + (It[path] + St[path] + FTt[path] + OMt[path] + data['SCOI'][r,path]) / (1+dr) ** t #CO2 tax not added yet. SCOT(I,J) ## 1
                        NPV1o = NPV1o + (It[path] + St[path] + FTt[path] + OMt[path]) / (1+dr) ** t
                   
                        dNPV = dNPV + math.sqrt(dIt[path] ** 2 + dFt[path] ** 2 + dOMt[path] ** 2) / (1+dr) ** t
                        dNPVp = dNPVp + math.sqrt(dIt[path] ** 2 + dFt[path] ** 2 + dOMt[path] ** 2 + dCO2t[path] ** 2)/ (1+dr) ** t # CO2 tax not added yet. dCO2t(I)**2 ## 2
                    
                        ICC = ICC + (It[path] + St[path]) / (1+dr) ** t
                        FCC = FCC + FTt[path] / (1+dr) ** t
                        OMC = OMC + OMt[path] / (1+dr) ** t
                        CO2C = CO2C + (data['SCOT'][r,path])/(1+r)**t ## 3
                    
                        NPV2 = NPV2 + Pt[path] / (1+dr) ** t
                        NPV2p = NPV2p + Pt[path] / (1+dr) ** t
                        
                    
                    LCOS = NPV1/NPV2
                    dLCOS = dNPV/NPV2
                    LCOSprice = NPV1o/NPV2
                    TLCOS = NPV1p/NPV2p
                    dTLCOS = dNPVp/NPV2p                                  
                    LTLCOS = TLCOS + Gam  
                    
                    #Variables for endogenous scrapping
                    PB[path] = data['BSTC'][r, path, 19] #payback threshold
                    dPB[path] = 0.3* PB[path]
                    It[path] = data['BSTC'][r, path, 0] / CF 
                    dIt[path] = data['BSTC'][r, path, 1] / CF
                    St[path] = data['SEWT'][r, path] * It[path]
                    FTt[path] = TaxTotCost[path]
                    dFt[path] = dTotCost
                    OMt[path] = OM
                    dCO2t[path] = 0.1 * data['SCOT'][r,path] ## 4
            
                    #Marginal cost calculation (for endogenous scrapping) (fuel cost+OM cost+fuel and CO2 tax policies)   
                    TMC[path] = Ft[path] + OMt[path] + FTt[path] + data['SCOT'][r,path]  ## 5
                    dTMC[path] = math.sqrt(dFt[path] ** 2 + dOMt[path] ** 2)
            
                    #Payback cost calculation (for endogenous scrapping) (TMC+(investment cost+investment subsidy)/payback threshold)   
                    TPB[path] = TMC[path] + (It[path] + St[path]) / PB[path]
                    dTPB[path] = math.sqrt(dFt[path] ** 2 + dOMt[path] ** 2 + dIt[path] ** 2 / PB[path] ** 2 + (It[path] ** 2 / PB[path] ** 4) * dPB[path] ** 2)
                        
                    data['SGC2'][r, path] = TMC[path] + Gam
                    data['SGC3'][r, path] = TPB[path] + Gam
                    data['SGD2'][r, path] = dTMC[path]
                    data['SGD3'][r, path] = dTPB[path]
                    
                    #For the cost components we exclude the gamma value as we assume the gamma value is accordingly distributed over the cost components
                    data['SWIC'][r, path] = ICC/NPV2p  # Investment cost component of TLCOS
                    data['SWFC'][r, path] = FCC/NPV2p  # Material cost component of TLCOS
                    data['SOMC'][r, path] = OMC/NPV2p  # O&M cost component of TLCOS
                    data['SCOC'][r, path] = CO2C/NPV2p # CO2 cost component of TLCOS ## 6
                                    
                    data['SEWC'][r, path] = LCOS
                    data['SETC'][r, path] = LCOSprice
                    data['SGC1'][r, path] = LTLCOS

                    data['SDWC'][r, path] = dLCOS
                    data['SGD1'][r, path] = dTLCOS
                    
                else:
                    LCOS = 0
                    dLCOS = 0
                    TLCOS = 0
                    dTLCOS = 0
                    LTLCOS = 0                
                   
                    data['SWIC'][r, path] = 0
                    data['SWFC'][r, path] = 0
                    data['SOMC'][r, path] = 0
                    data['SCOC'][r, path] = 0  # CO2 cost component of TLCOS ## 7
                
                    data['SEWC'][r, path] = 0
                    data['SETC'][r, path] = 0
                    data['SGC1'][r, path] = 0
                    data['SGC2'] [r, path] = 0
                    data['SGC3'] [r, path] = 0

                    data['SDWC'][r, path] = 0
                    data['SGD1'][r, path] = 0
                    data['SGD2'] [r, path] = 0
                    data['SGD3'] [r, path] = 0
            
            ## Exogenous projected steel demand is there then calculate 
            if (spsa_dt[r] > 0.0):
               PLCOS = np.where(data['SGC1'][r, :] != 0.0, data['SGC1'][r, :], 0.0)
               data['SPRI'][r] = np.sum(PLCOS * mkt_shares) / np.sum(mkt_shares)
               PLCOS = np.where(data['SETC'][r, :] != 0.0, data['SETC'][r, :], 0.0)
               data['SPRC'][r] = np.sum(PLCOS * mkt_shares) / np.sum(mkt_shares)

    return data


               
               
               
                   
            
              
             
               
                    
                        
                        
                    
                    
                   
                        
                 
                  
    
                            
                            
    """ 
    #FTT heat code 
        # Boiler lifetime
        lt = data['BHTC'][r,:, c4ti['5 Lifetime']]
        max_lt = int(np.max(lt))
        lt_mat = np.linspace(np.zeros(len(titles['HTTI'])), max_lt-1,
                             num=max_lt, axis=1, endpoint=True)
        lt_max_mat = np.concatenate(int(max_lt)*[lt[:, np.newaxis]], axis=1)
        mask = lt_mat < lt_max_mat
       
        lt_mat = np.where(mask, lt_mat, 0)

        # Capacity factor
        cf = data['BHTC'][r,:, c4ti['13 Capacity factor mean'], np.newaxis]

        # Conversion efficiency
        ce = data['BHTC'][r,:, c4ti['9 Conversion efficiency'], np.newaxis]
        #print("ce:", ce)

        # Discount rate
        dr = data['BHTC'][r,:, c4ti['8 Discount rate'], np.newaxis]

        # Initialse the levelised cost components
        # Average investment cost
        it = np.zeros([len(titles['HTTI']), int(max_lt)])
        # print(it.shape)
        # print(data['BHTC'][r,:, c4ti['1 Investment cost mean']].shape)
        it[:, 0,np.newaxis] = data['BHTC'][r,:, c4ti['1 Inv cost mean (EUR/Kw)'],np.newaxis]/(cf*1000)


        # Standard deviation of investment cost
        dit = np.zeros([len(titles['HTTI']), int(max_lt)])
        dit[:, 0, np.newaxis] = divide(data['BHTC'][r,:, c4ti['2 Inv Cost SD'], np.newaxis], (cf*1000))

        # Upfront subsidy/tax at purchase time
        st = np.zeros([len(titles['HTTI']), int(max_lt)])
        st[:, 0, np.newaxis] = it[:, 0, np.newaxis] * data['HTVS'][r, :, 0, np.newaxis]

        # Average fuel costs
        ft = np.ones([len(titles['HTTI']), int(max_lt)])
        ft = ft * divide(data['BHTC'][r,:, c4ti['1np.zeros(len(titles['STTI'])) Fuel cost  (EUR/kWh)'], np.newaxis]*data['HEWP'][r, :, 0, np.newaxis], ce)
        ft = np.where(mask, ft, 0)

        # Standard deviation of fuel costs
        dft = np.ones([len(titles['HTTI']), int(max_lt)])
        dft = dft * ft * data['BHTC'][r,:, c4ti['11 Fuel cost SD'], np.newaxis] 
        dft = np.where(mask, dft, 0)
        
        # Fuel tax costs
        fft = np.ones([len(titles['HTTI']), int(max_lt)])
        fft = fft* divide(data['HTRT'][r, :, 0, np.newaxis], ce)
        fft = np.where(mask, fft, 0)
        #print("fft:", fft)
        #print(fft.shape)

        # Average operation & maintenance cost
        omt = np.ones([len(titles['HTTI']), int(max_lt)])
        omt = omt * divide(data['BHTC'][r,:, c4ti['3 O&M mean (EUR/kW)'], np.newaxis], (cf*1000))
        omt = np.where(mask, omt, 0)

        # Standard deviation of operation & maintenance cost
        domt = np.ones([len(titles['HTTI']), int(max_lt)])
        domt = domt * divide(data['BHTC'][r,:, c4ti['4 O&M SD'], np.newaxis], (cf*1000))
        domt = np.where(mask, domt, 0)

        # Feed-in-Tariffs
        fit = np.ones([len(titles['HTTI']), int(max_lt)])
        fit = fit * data['HEFI'][r, :, 0, np.newaxis]
        fit = np.where(mask, fit, 0)

        # Net present value calculations
        # Discount rate
        denominator = (1+dr)**lt_mat

        # 1-Expenses
        # 1.1-Without policy costs
        npv_expenses1 = (it+ft+omt)/denominator
        # 1.2-With policy costs
        npv_expenses2 = (it+st+ft+fft+omt-fit)/denominator
        # 1.3-Only policy costs
        npv_expenses3 = (st+fft-fit)/denominator
        # 2-Utility
        npv_utility = 1/denominator
        #Remove 1s for tech with small lifetime than max
        npv_utility[npv_utility==1] = 0
        npv_utility[:,0] = 1
        # 3-Standard deviation (propagation of error)
        npv_std = np.sqrt(dit**2 + dft**2 + domt**2)/denominator

        # 1-levelised cost variants in $/pkm
        # 1.1-Bare LCOH
        lcoh = np.sum(npv_expenses1, axis=1)/np.sum(npv_utility, axis=1)
        # 1.2-LCOH including policy costs
        tlcoh = np.sum(npv_expenses2, axis=1)/np.sum(npv_utility, axis=1)
        # 1.3-LCOH of policy costs
        lcoh_pol = np.sum(npv_expenses3, axis=1)/np.sum(npv_utility, axis=1)
        # Standard deviation of LCOH
        dlcoh = np.sum(npv_std, axis=1)/np.sum(npv_utility, axis=1)

        # LCOH augmented with non-pecuniary costs
        tlcohg = tlcoh + data['BHTC'][r, :, c4ti['12 Gamma value']]

        # Pay-back thresholds
        pb = data['BHTC'][r,:, c4ti['16 Payback time, mean']]
        dpb = data['BHTC'][r,:, c4ti['17 Payback time, SD']]

        # Marginal costs of existing units
        tmc = ft[:, 0] + omt[:, 0] + fft[:, 0] - fit[:, 0]
        dtmc = np.sqrt(dft[:, 0]**2 + domt[:, 0]**2)

        # Total pay-back costs of potential alternatives
        tpb = tmc + (it[:, 0] + st[:, 0])/pb
        dtpb = np.sqrt(dft[:, 0]**2 + domt[:, 0]**2 +
                       divide(dit[:, 0]**2, pb**2) +
                       divide(it[:, 0]**2, pb**4)*dpb**2)

        # Add gamma values
        tmc = tmc + data['BHTC'][r, :, c4ti['12 Gamma value']]
        tpb = tpb + data['BHTC'][r, :, c4ti['12 Gamma value']]

        # Pass to variables that are stored outside.
        data['HEWC'][r, :, 0] = lcoh            # The real bare LCOH without taxes
        data['HETC'][r, :, 0] = tlcoh           # The real bare LCOH with taxes
        data['HGC1'][r, :, 0] = tlcohg         # As seen by consumer (generalised cost)
        data['HWCD'][r, :, 0] = dlcoh          # Variation on the LCOH distribution
        data['HGC2'][r, :, 0] = tmc             # Total marginal costs
        data['HGD2'][r, :, 0] = dtmc          # SD of Total marginal costs
        data['HGC3'][r, :, 0] = tpb             # Total payback costs
        data['HGD3'][r, :, 0] = dtpb          # SD of Total payback costs

    return data
"""