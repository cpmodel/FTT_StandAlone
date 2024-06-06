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

def solve(data, time_lag, iter_lag, titles, histend, year, specs):
    metalinput = 0.0
    maxscrapdemand = 0.0
    maxscrapdemand_p = 0.0
    scraplimittrade = 0.0
    scrapcost = 0.0
    # pig-iron (or DRI) to crude steel ratio. Rationale: pig iron/DRI
    pitcsr = 1.1
    # has higher carbon content which is removed in steel making
    # process (to about 0.0001 - 0.005%wt)
    sxsf = 0.0
    
    # Simple treatment of scrap trade. Flows are only a function of scrap shortages
    # for the secondary steelmaking route.
    
    scrapcost = data['BSTC'][:, :, 16]
    maxscrapdemand = np.sum(scrapcost[:, :] * data['SEWG'][:, :], axis=1)
    maxscrapdemand_p = np.sum(scrapcost[0:23, :] * data['SEWG'][0:23, :], axis=1)
    scraplimittrade = scrapcost[25, :, :] * data['SEWG'][25, :, :]
    
    if t == 1:
        scrapshortage = 0.0
        scrapabundance = 0.0
        sxim = 0.0
        sxex = 0.0
        sxsr = 0.0
        
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
                    if data ['SXSR'][r, 0, 0] >= MaxScrapDemand[r, 0, 0]:
                        MetalInput[path,0] = (1.0 - 0.09 - MSSPP[path,r]) * PITCSR 
                        MetalInput[path,1] = 0.0
                        MetalInput[path,2] = MSSPP[path,r] +0.09
                        MetalInput[25,0] = 0.0
                        MetalInput[25,1] = 0.0
                        MetalInput[25,2] = MSSPP[25,r] +0.09
                #There's not enough scrap to feed into all the technologies, but there's 
                #enough scrap to feed into the Scrap-EAF route.             
                    elif ((data['SXSR'][r, 0, 0] < MaxScrapDemand[r, 0, 0]) and (data['SXSR'][r, 0, 0] >= ScraplimitTrade[r, 0, 0])): 
                        MetalInput[path, 1] = 0.0
                   
                        if (sum (data ['SEWG'][1:24,r] * MSSPP[1:24,r]) > 0.0):
                            MetalInput[path,2] = 0.09 + (data['SXSR'][r,0,0]-ScraplimitTrade[r, 0, 0])/MaxScrapDemandP[r, 0, 0] * MSSPP[path,r]
                        else:
                            MetalInput[path,2] = MSSPP[path,r]/2 +0.09
                    
                        MetalInput [path,0] = (1.0 - MetalInput[path,2]) * PITCSR
                        MetalInput[25,0] = 0.0
                        MetalInput[25,1] = 0.0
                        MetalInput[25,2] = MSSPP[26,r] +0.09
     
                #There's not enough scrap available to meet the demand, so all available
                #scrap will be fed into the Scrap-EAF route.    
                    elif ((data['SXSR'][r,0,0] < MaxScrapDemand [r,0,0]) and (data[SXSR][r,0,0] < data['SEWG'][25,r,0]*(1-0.09))):
                        MetalInput[path,0] = PITCSR * (1.0 - 0.09)
                        MetalInput[path,1] = 0.0
                        MetalInput[path,2] = 0.09
     
                        MetalInput[25,0] = 0.0
                        MetalInput[25,1] = (1 - 0.09 - data['SXSR'][r,0,0] / data['SEWG'][25,r])*PITCSR
                        MetalInput[25,2] = 0.09 + data['SXSR'][r, 0, 0] / SEWG[25,r]  