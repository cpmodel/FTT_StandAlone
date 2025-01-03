# # -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:47:32 2024

@author: Arpan.Golechha
=========================================
ftt_s_rawmaterials.py
=========================================
Steel production module.
####################################

This function performs a detailed computation related to the distribution of raw materials and their impact on steel production. 
Finding scrap intensity, emission intensity, energy intensity, employment factor and capital investment costs 
using scrap imports and exports, cost matrix and scrap avaliability.

Parameters
----------
- raw_material_distr(data, titles, year, t)
- data: dictionary
    Data is a container that holds all cross-sectional (of time) for all
    variables. Variable names are keys and the values are 3D NumPy arrays.
- titles: dictionary
    Titles is a container of all permissible dimension titles of the model.
- year: int
    Curernt/active year of solution
- t: int
    The iteration number for the current calculation cycle.

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

def raw_material_distr(data, titles, year, t,spsa_dt):
    

    ## Initialization of variables
    metalinput = np.zeros((len(titles['STTI']), 3))   
    maxscrapdemand = np.zeros((len(titles['RTI']),1,1))   
    maxscrapdemand_p = np.zeros_like(maxscrapdemand)  
    scraplimittrade = np.zeros_like(maxscrapdemand)   
    scrapcost = np.zeros((len(titles['RTI']),len(titles['STTI']),1))     

    # pig-iron (or DRI) to crude steel ratio. Rationale: pig iron/DRI
    pitcsr = 1.1
    # has higher carbon content which is removed in steel making
    # process (to about 0.0001 - 0.005%wt)
    data['SXSF'] = np.zeros((71,1,1))

    # Simple treatment of scrap trade. Flows are only a function of scrap shortages
    # for the secondary steelmaking route.
    scrapcost[:,:,0] = data['BSTC'][:, :, 16]
    maxscrapdemand[:,0,0] = np.sum(scrapcost[:, :,0] * data['SEWG'][:, : , 0],axis=1)  ## axis = 1 means summing across column for each row

    maxscrapdemand_p[:,0,0] = np.sum(scrapcost[: , :24,0] * data['SEWG'][:, :24, 0],axis=1)
    scraplimittrade[:,0,0] = scrapcost[:, 25,0] * data['SEWG'][ :, 25, 0]   ## Adding maxscrap and MOE in 1990
    
    ## t is no. of iterations per year
    if t == 1: 
        scrapshortage = np.zeros((len(titles['RTI']),1,1))    ## change
        scrapabundance = np.zeros((len(titles['RTI']),1,1))    ## shape : (71,26)
        data['SXEX'] = np.zeros_like(scrapabundance)     ## shape : (71,26)
        data['SXIM'] = np.zeros((71,1,1))

        ## If Exogenous Scrap availability is less then the calculated limits
        ## Calculating shortage and abundance
        scrapshortage = np.where(data['SXSC'] < scraplimittrade, scraplimittrade - data['SXSC'], 0.0)
        scrapabundance = np.where(data['SXSC'] > maxscrapdemand, data['SXSC'] - maxscrapdemand, 0.0)

       # If there's global abundance of scrap then the shortages can simply be met through imports
        if np.sum(scrapshortage) > np.sum(scrapabundance) and np.sum(scrapshortage) > 0.0:
            # mask = scrapshortage > 0.0
            # data['SXIM'][:,:,0] = np.zeros((len(titles['RTI']), len(titles['STTI'])))
            data['SXIM'] = np.where(scrapshortage > 0.0, scrapshortage, 0.0)

        # If the supply of scrap is insufficient to meet global demand then weight import according to the ratio of abundance and shortage.
        if np.sum(scrapshortage) > np.sum(scrapabundance) and np.sum(scrapshortage) > 0.0:
            # mask = scrapshortage > 0.0
            # data['SXIM'] = np.zeros_like(scrapshortage)
            data['SXIM'] = np.where(scrapshortage > 0.0, scrapshortage * (np.sum(scrapabundance) / np.sum(scrapshortage)), 0.0)

        scrapabundance = np.squeeze(scrapabundance)
        # Countries export scrap according to their weights in the global abundance
        data['SXEX'][:,0,0] = np.where(np.squeeze(data['SXSC'] > maxscrapdemand), np.sum(data['SXIM'][:,0,0]) * (scrapabundance / np.sum(scrapabundance)), 0.0)  # Shape (71,)
          
    
    data['SXSR']= data['SXSC'] + data['SXIM'] - data['SXEX']      
                   
    for r in range(len(titles['RTI'])):
        if (spsa_dt[r]).all() > 0.0: 
            for path in range(len(titles['STTI']) - 2):
                # There's enough scrap to meet the maximum scrap demand
                if (np.any(data['SXSR'][r, 0, 0] >= maxscrapdemand)):
                    metalinput[path,0] = (1.0 - 0.09 - scrapcost[r, path]) * pitcsr 
                    metalinput[path,1] = 0.0
                    metalinput[path,2] = scrapcost[r, path] +0.09   
                    metalinput[25,0] = 0.0
                    metalinput[25,1] = 0.0
                    metalinput[25,2] = scrapcost[r, 25] +0.09
            #There's not enough scrap to feed into all the technologies, but there's 
            #enough scrap to feed into the Scrap-EAF route.             
                elif ((data['SXSR'][r, 0, 0] < maxscrapdemand[r]) and (data['SXSR'][r, 0, 0] >= scraplimittrade[r])):  
                    metalinput[path, 1] = 0.0
            
                    if (np.sum(data ['SEWG'][r, :24] * scrapcost[r, :24]) > 0.0):  ## changed
                        metalinput[path,2] = 0.09 + (data['SXSR'][r,0,0]-scraplimittrade[r])/maxscrapdemand_p[r] * scrapcost[r,path] ## change scrapcost defined shape : (71,26)
                        
                    else:
                        metalinput[path,2] = scrapcost[r,path]/2 +0.09  
                    metalinput [path,0] = (1.0 - metalinput[path,2]) * pitcsr

                    metalinput[25,0] = 0.0
                    metalinput[25,1] = 0.0
                    metalinput[25,2] = scrapcost[r, 25] +0.09

            #There's not enough scrap available to meet the demand, so all available
            #scrap will be fed into the Scrap-EAF route.    
                elif ((data['SXSR'][r,0,0] < maxscrapdemand [r]) and (data['SXSR'][r,0,0] < data['SEWG'][r,25,0]*(1-0.09))):    ## data['SEWG'] change [r,25,0]
                ##((data['SXSR'][r,0,0] < maxscrapdemand [path]) and (data['SXSR'][r,0,0] < data['SEWG'][25,r,0]*(1-0.09))):  ## changed
                    metalinput[path,0] = pitcsr * (1.0 - 0.09)
                    metalinput[path,1] = 0.0
                    metalinput[path,2] = 0.09

                    metalinput[25,0] = 0.0
                    metalinput[25,1] = (1 - 0.09 - data['SXSR'][r,0,0] / data['SEWG'][r, 25, 0])*pitcsr
                    metalinput[25,2] = 0.09 + data['SXSR'][r, 0, 0] / data['SEWG'][r, 25, 0]  

        ii = np.eye((len(titles['SMTI'])))
        fd = np.zeros((len(titles['SMTI'])))
        fd[0] = 1.0

        inventory = np.zeros((len(titles['STTI']),len(titles['SMTI'])))

        for a in range(len(titles['STTI'])):
            for b in range(len(titles['SSTI'])):
                if ((data['STIM'][0,a,b]==1) and b>6 and b<20):   
                    if (a<24):
                        data['SLCI'][0,:,1]= data['SCMM'][0,:,b]   
                    else:
                        data['SLCI'][0,:,1]=0.0
                if(data['STIM'][0,a,b]==1 and b>19 and b<26):
                    #Now select the correct steelmaking technology.
                    data['SLCI'][0,:,0] = data['SCMM'][0,:,b]
                    data['SLCI'][0,1,0] = metalinput[a,0]
                    data['SLCI'][0,2,0] = metalinput[a,1]
                    data['SLCI'][0,3,0] = metalinput[a,2]
        
            #The inventory matrix (SLCI) is now completely filled. Now calculate the inventory vector per technology.
            #First, we add the steelmaking step to the inventory vector. 
            inventory[a,:] = data['SLCI'][0, :,0]
            #Add ironmaking step scaled to the need of PIP
            inventory[a,:] += data['SLCI'][0,1,0]*data['SLCI'][0,:,1]
            #Add Sinter and Pellet inventory
            inventory[a,:] += inventory[a,6] * data['SCMM'][0,:,2] + inventory[a,7] * data['SCMM'][0,:,3] + inventory[a,8] * data['SCMM'][0,:,4] + inventory[a,9] * data['SCMM'][0, :,5]
            #Add coke inventory
            inventory[a,:] += inventory[a,4] * data['SCMM'][0,:,0] + inventory[a,5] * data['SCMM'][0,:,1]
            #Add oxygen inventory
            inventory[a,:] += inventory[a,10] * data['SCMM'][0,:,6]
            #Add finishing step
            inventory[a,:] += (data['SCMM'][0,:,26])/1.14
        
            #Material based emission intensity
            ef = np.sum(inventory[a,:]*data['SMEF'][0,:,0])   
            ef_fossil = np.sum(inventory[a,:15]*data['SMEF'][0,:15,0])
            ef_biobased = np.sum(inventory[a,15:24] * data['SMEF'][0,15:24,0])

            #From here on: Put the results of the inventory calculation into the CostMatrix
            #Material based Energy Intensity
            data['BSTC'][r,a,15] = np.sum(inventory[a,:] * data['SMED'][0, :, 0])
            data['STEI'][r,a,0]= data['BSTC'][r,a,15]
            data['STSC'][r,a,0]= inventory[a,4]

            ## change : why are we using this loop
            for mat in range(len(titles['SMTI'])-4):
                data['BSTC'][r,a,21+mat]= inventory[a,mat] 

        
            #Select CCS technologies
            if (data['BSTC'][r,a,21]==1):
                #Increase fuel consumption
                inventory[a, 11:15] *= 1.1
                inventory[a, 17:20] *= 1.1
                #capital investment costs
                if year == 2017:
                    data['SCIN'][r, a, 0] = inventory[a,20]+50 * 0.9 * ef
            
                #Adjust O&M Costs due to CO2 transport and storage
                data['BSTC'][r,a,2]= inventory[a,21]+ 5.0*0.9*ef
                data['BSTC'][r,a,3]= data['BSTC'][r,a,2]*0.3  

                #Adjust EF for CCS utilising technologies
                data['BSTC'][r,a,13] = 0.1 * ef - ef_biobased
                data['STEF'][r,a,0] = 0.1 * ef - ef_biobased  
                data['BSTC'][r,a,14] = data['BSTC'][r,a,13]*0.1

                #Adjust total Employment for complete steelmaking route (increase by 5% due to CCS)
                data['BSTC'][r,a,4] = 1.05 * inventory[a, 23]
                data['SEPF'][r,a,0] = 1.05 * inventory[a, 23]  

                #Adjust electricity consumption due to CCS use
                data['BSTC'][r,a,36] *= 1.1   
        
            else:
                #capital investment costs
                if year == 2017:
                    data['SCIN'][r, a, 0] = inventory[a,20]   ## shape (SCIN) : (71,26)
            
                #OM and dOM
                data['BSTC'][r,a,2]= inventory[a, 21]
                data['BSTC'][r,a,3]= data['BSTC'][r,a,2]*0.3

                #EF and dEF
                data['BSTC'][r,a,13]= ef-ef_biobased
                data['STEF'][r,a,0]= ef-ef_biobased
                data['BSTC'][r,a,14]= data['BSTC'][r,a,13]*0.1

                #Employment
                data['BSTC'][r,a,4] = inventory[a, 23]
                data['SEPF'][r,a,0] = inventory[a, 23]
    

    return data

'''
print(maxscrapdemand.shape)
print(maxscrapdemand_p.shape)
print(scraplimittrade.shape)
print(scrapcost.shape)
(71, 26)
(71, 26)
(71, 26)
(71, 26)

second
(26,)
(24,)
(71,)
(71, 26)

# print(inventory.shape[1] - data['SLCI'].shape[1])  ## anwser is 2
# inventory[:,:].shape ==  data['SLCI'][0,:,1].shape   ## problem
## where is "SLCI" coming from ?
# print("second")
# print(data['SEWG'].shape)
'''