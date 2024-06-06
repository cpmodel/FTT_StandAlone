# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:47:32 2024

@author: Arpan.Golechha
"""
# Standard library imports
from math import sqrt
import copy
import warnings

# Third party imports
import numpy as np

def raw_material_distr(data, titles, year, t):
    metalinput = 0.0
    maxscrapdemand = np.zeroes
    maxscrapdemand_p = np.zeroes
    scraplimittrade = np.zeroes
    scrapcost = np.zeroes
    # pig-iron (or DRI) to crude steel ratio. Rationale: pig iron/DRI
    pitcsr = 1.1
    # has higher carbon content which is removed in steel making
    # process (to about 0.0001 - 0.005%wt)
    sxsf = np.zeroes
    
    # Simple treatment of scrap trade. Flows are only a function of scrap shortages
    # for the secondary steelmaking route.
    
    scrapcost = data['BSTC'][:, :, 16]
    maxscrapdemand = np.sum(scrapcost[:, :] * data['SEWG'][:, :], axis=1)
    maxscrapdemand_p = np.sum(scrapcost[0:23, :] * data['SEWG'][0:23, :], axis=1)
    scraplimittrade = scrapcost[25, :, :] * data['SEWG'][25, :, :]
    
    if t == 1:
        scrapshortage = np.zeros()
        scrapabundance = np.zeros()
        sxim = np.zeros()
        sxex = np.zeros()
        sxsr = np.zeros()
        
        if np.any(data['SXSC'][:, 0, 0] < scraplimittrade):
            scrapshortage = scraplimittrade - data['SXSC'][:, :, 0]
        
        if np.any(data['SXSC'][:, :, 0] > maxscrapdemand):
            scrapabundance = data['SXSC'][:, :, 0] - maxscrapdemand
            
            # If there's global abundance of scrap then the shortages can simply be met through imports
            if np.sum(scrapshortage) > np.sum(scrapabundance) and np.sum(scrapshortage) > 0.0:
                mask = scrapshortage > 0.0
                data['SXIM'][:,:,0] = np.zeros_like(scrapshortage)
                data['SXIM'][mask] = scrapshortage[mask]
            
            # If the supply of scrap is insufficient to meet global demand then weight import according to the ratio of abundance and shortage.
            if np.sum(scrapshortage) > np.sum(scrapabundance) and np.sum(scrapshortage) > 0.0:
                mask = scrapshortage > 0.0
                data['SXIM'] = np.zeros_like(scrapshortage)
                data['SXIM'][mask] = scrapshortage[mask] * (np.sum(scrapabundance) / np.sum(scrapshortage))

        sxsc = data['SXSC'][:, :, 0]
        mask = sxsc > maxscrapdemand
        sxex = np.zeros_like(sxsc)
        sxex[mask] = np.sum(data['SXIM']) * (scrapabundance[mask] / np.sum(scrapabundance))

    data['SXSR']= data['SXSC'] + data['SXIM'] - data['SXEX']
    
    for r in range(len(titles['RTI'])):
        if data['SPSA'][r, 0 , 0] > 0.0:
                for path in range(len(titles['STTI'])-2):
                    # There's enough scrap to meet the maximum scrap demand
                    if data ['SXSR'][r, 0, 0] >= maxscrapdemand[r, 0, 0]:
                        metalinput[path,0] = (1.0 - 0.09 - scrapcost[path,r]) * pitcsr 
                        metalinput[path,1] = 0.0
                        metalinput[path,2] = scrapcost[path,r] +0.09
                        metalinput[25,0] = 0.0
                        metalinput[25,1] = 0.0
                        metalinput[25,2] = scrapcost[25,r] +0.09
                #There's not enough scrap to feed into all the technologies, but there's 
                #enough scrap to feed into the Scrap-EAF route.             
                    elif ((data['SXSR'][r, 0, 0] < maxscrapdemand[r, 0, 0]) and (data['SXSR'][r, 0, 0] >= scraplimittrade[r, 0, 0])): 
                        metalinput[path, 1] = 0.0
                   
                        if (sum (data ['SEWG'][1:24,r] * scrapcost[1:24,r]) > 0.0):
                            metalinput[path,2] = 0.09 + (data['SXSR'][r,0,0]-scraplimittrade[r, 0, 0])/maxscrapdemand_p[r, 0, 0] * scrapcost[path,r]
                        else:
                            metalinput[path,2] = scrapcost[path,r]/2 +0.09
                    
                        metalinput [path,0] = (1.0 - metalinput[path,2]) * pitcsr
                        metalinput[25,0] = 0.0
                        metalinput[25,1] = 0.0
                        metalinput[25,2] = scrapcost[26,r] +0.09
     
                #There's not enough scrap available to meet the demand, so all available
                #scrap will be fed into the Scrap-EAF route.    
                    elif ((data['SXSR'][r,0,0] < maxscrapdemand [r,0,0]) and (data['SXSR'][r,0,0] < data['SEWG'][25,r,0]*(1-0.09))):
                        metalinput[path,0] = pitcsr * (1.0 - 0.09)
                        metalinput[path,1] = 0.0
                        metalinput[path,2] = 0.09
     
                        metalinput[25,0] = 0.0
                        metalinput[25,1] = (1 - 0.09 - data['SXSR'][r,0,0] / data['SEWG'][25,r])*pitcsr
                        metalinput[25,2] = 0.09 + data['SXSR'][r, 0, 0] / data['SEWG'][25,r]  
    
        inventory = np.zeroes
        #Select the correct ironmaking technology. Not applicable for techs 25 and 26
        for a in range(len(titles['STTI'])):
            for b in range(len(titles['SSTI'])):
                if ((data['STIM'][0,a,b]==1) and b>7 and b<21):
                    if (a<25):
                        data['SLCI'][0,:,2]= data['SCMM'][0,:,b]
                    else:
                        data['SLCI'][0,:,2]=0.0
                if(data['STIM'][0,a,b]==1 and b>20 and b<27):
                    #Now select the correct steelmaking technology.
                    data['SLCI'][0,:,0] = data['SCMM'][0,:,b]
                    data['SLCI'][0,1,0] = metalinput[0,a,0]
                    data['SLCI'][0,2,0] = metalinput[0,a,1]
                    data['SLCI'][0,3,0] = metalinput[0,a,2]
            
            #The inventory matrix (SLCI) is now completely filled. Now calculate the inventory vector per technology.
            #First, we add the steelmaking step to the inventory vector. 
            inventory[0,:,a] = data['SLCI'][0,:,1]
            #Add ironmaking step scaled to the need of PIP
            inventory[0,:,a] += data['SLCI'][0,2,1]*data['SLCI'][0,:,2]
            #Add Sinter and Pellet inventory
            inventory[0,:,a] += inventory[0,7,a] * data['SCMM'][0,:,3] + inventory[0,8,a] * data['SCMM'][0,:,4] + inventory[0,9,a] * data['SCMM'][0,:,5] + inventory[0,10,a] * data['SCMM'][0,:,6]
            #Add coke inventory
            inventory[0,:,a] += inventory[0,5,a] * data['SCMM'][0,:,1] + inventory[0,6,a] * data['SCMM'][0,:,2]
            #Add oxygen inventory
            inventory[0,:,a] += inventory[0,11,a] * data['SCMM'][0,:,7]
            #Add finishing step
            inventory[0,:,a] += (data['SCMM'][0,:,27])/1.14
            
            #Material based emission intensity
            ef = np.sum(Inventory[0,:,a]*data['SMEF'][0,:,0])
            ef_fossil = np.sum(inventory[0,1:15,a]*data['SMEF'][0,1:15,0])
            ef_biobased = np.sum(inventory(0,16:24,a)*data['SMEF'][0,16:24,0])

            #From here on: Put the results of the inventory calculation into the CostMatrix
            #Material based Energy Intensity
            data['BSTC'][r,a,15]= np.sum(inventory[0,:,a]*data['SMED'][0;:,0])
            data['STEI'][r,a,0]= data['BSTC'][r,a,15]
            data['STSC'][r,a,0]= inventory[0,4,a]

            for mat in range(len['SMTI'])
                data['BSTC'][r,a,21]= inventory[0,21,a]
            
            #Select CCS technologies
            if (data['BSTC'][r,a,22]==1):
                #Increase fuel consumption
                inventory[0, 12:15, a] *= 1.1
                inventory[0, 18:20, a] *= 1.1
                #capital investment costs
                if (year == 2017)
                    data['SCIN'][r, a, 0] = inventory[0,20,a]
                
                #Adjust O&M Costs due to CO2 transport and storage
                data['BSTC'][r,a,2]= inventory[0,21,a]+ 5.0*0.9*ef
                data['BSTC'][r,a,3]= data[r,a,2]*0.3

                #Adjust EF for CCS utilising technologies
                data['BSTC'][r,a,13] = 0.1 * ef - ef_biobased
                data['STEF'][r,a,,0] = 0.1 * ef - ef_biobased
                data['BSTC'][r,a,14] = data['BSTC'][r,a,13]*0.1

                #Adjust total Employment for complete steelmaking route (increase by 5% due to CCS)
                data['BSTC'][r,a,4] = 1.05 * inventory[0,23,a]
                data['SEPF'][r,a,0] = 1.05 * inventory[0,23,a]

                #Adjust electricity consumption due to CCS use
                data['BSTC'][r,a,36] *= 1.1
            
            else
                #capital investment costs
                if (year == 2017)
                    data['SCIN'][r, a, 0] = inventory[0,20,a]
                
                #OM and dOM
                data['BSTC'][r,a,2]= inventory[0,21,a]
                data['BSTC'][r,a,3]= data['BSTC'][r,a,3]*0.3

                #EF and dEF
                data['BSTC'][r,a,13]= ef-ef_biobased
                data['STEF'][r,a,0]= ef-ef_biobased
                data['BSTC'][r,a,14]= data['BSTC'][r,a,13]*0.1

                #Employment
                data['BSTC'][r,a,4] = inventory[0.23.a]
                data['SEPF'][r,a,0] = inventory[0,23,a]
        
        ef2= data['BSTC'][r,:,13]
        om= data['BSTC'][r,:,2]
        empl= inventory[0,23,:]




