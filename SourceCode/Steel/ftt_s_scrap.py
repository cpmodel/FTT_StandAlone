# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:09:26 2024

@author: AE

=========================================
ftt_s_scrap.py
=========================================
####################################

    Calculate scrap availability in FTT-Steel.

    This function calculates the avaliability of scrap
    This is based on:

    Parameters
    -----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) for all
        variables. Variable names are keys and the values are 3D NumPy arrays.
    data_dt: dictionary
        Data_dt is a container that holds all cross-sectional (of time) data
        for all variables of the previous iteration.
    time_lag: dictionary
        Time_lag is a container that holds all cross-sectional (of time) data
        for all variables of the previous year.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.
    dt: integer
        Dt is an integer - 1 / number of iterations.

    Returns
    ----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.
"""
# _______________________________________________________________________
# ============FTT:Steel model ========== Created by JF Mercure, F Knobloch, L. van Duuren, and P. Vercoulen
# Follows general equations of the FTT model (e.g. Mercure Energy Policy 2012)
#  Adapted to the iron and steel industry.
#  
# FTT determines the secnology mix
# ________________________________________jm801@cam.ac.uk________________

# ---------------Scrap availability calculation function-----------------

def scrap_calc(data, time_lag, titles, year):
     data['SSSR'] = data['SXSS'].copy()
     for r in range(len(titles['RTI'])): 
        for sec in range(len(titles['XPTI'])):
            
            Retrievedate = int(year - (data['SXLT'][r, sec, 0]))
            data['SXSC'][r, 0, 0] = 0.0
            if (year > 2017):
                data['SHS2'][r, year - 2000, 0] = data['SPSA'][r, 0, 0]
       # Calcultate Scrap availability
       # SXSCtot = SXSCpg + HSP(T- LTpg) * RRpg * SSpg * (1 - LRpg) for all pg
        if (Retrievedate < 1918):
             #If data has to be retrieved from a date at which is before the period included on the databank, then make an assumption what the steelproduction would have been
             #Assumed growth rate of 3.5%
             data['SXSC'][r, :, 0] = data['SXSC'][r, 0, 0] + data['SHS1'][r, 1, 0]*0.965**(1918 - Retrievedate) * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0]*(1 - (data['SXLR'][r, sec, 0]))
        elif (Retrievedate < 2017):
            data['SXSC'][r, :, 0] = data['SXSC'][r, 0, 0] + data['SHS1'][r, Retrievedate - 1917, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.2
        if (Retrievedate+1 < 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS1'][r, Retrievedate+0 - 1917, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.2
        if (Retrievedate+2 < 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS1'][r, Retrievedate+1 - 1917, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.2
        if (Retrievedate+3 < 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS1'][r, Retrievedate+2 - 1917, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.15
        if (Retrievedate+4 < 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS1'][r, Retrievedate+3 - 1917, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.1
        if (Retrievedate+5 < 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS1'][r, Retrievedate+4 - 1917, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.1
        if (Retrievedate+6 < 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS1'][r, Retrievedate+5 - 1917, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.05          
        if (Retrievedate+1 >= 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+0 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.2
        if (Retrievedate+2 >= 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+1 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.2
        if (Retrievedate+3 >= 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+2 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.15
        if (Retrievedate+4 >= 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+3 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.1
        if (Retrievedate+5 >= 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+4 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.1
        if (Retrievedate+6 >= 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+5 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.05
        #If endogenous sector split calculation is going to be included then it should be fed in here to replace SXSS
        elif (Retrievedate >= 2017):
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.2
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+0 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.2
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+1 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.2
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+2 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.15
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+3 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.1
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+4 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.1
            data['SXSC'][r,: , 0] = data['SXSC'][r, 0, 0] + data['SHS2'][r, Retrievedate+5 - 2000, 0] * data['SXRR'][r, sec, 0] * data['SSSR'][r, sec, 0] * (1 - data['SXLR'][r, sec, 0]) * 0.05

     return data