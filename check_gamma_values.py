# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:18:52 2024

@author: pv
"""
from collections import OrderedDict
import configparser
import csv
import datetime
import json
import os
from pathlib import Path
import pickle
import sys
import time
from threading import Timer, Thread
import webbrowser

from bottle import (route, run, request, response, static_file)
import numpy as np
import pandas as pd

from SourceCode.support import titles_functions


if __name__ == '__main__':

    dirmain = os.path.dirname(os.path.realpath(__file__))
    diroutput = os.path.join(dirmain, "Output")
    
    # Read titles to eventually create 2D dataframes
    titles = titles_functions.load_titles()
    
    # time horizon
    tl = np.arange(2010, 2027+1)
    
    # Read pickle file
    with open('Output\Gamma.pickle', 'rb') as f:
        output = pickle.load(f)
        
    # Scenarios
    scens = list(output.keys())
        
    # FTT:P gamma values
    mgam = dict()
    mgamx = dict()
    bcet = dict()
    mewg =  dict()
    
    for scen in scens:
        
        mgam[scen] = dict()
        bcet[scen] = dict()
        mewg[scen] = dict()
        
        for r, reg in enumerate(titles["RTI"]):
            
            bcet[scen][reg] = pd.DataFrame(output[scen]["BCET"][r, :, :, 0],
                                           index=titles["T2TI"],
                                           columns=titles["C2TI"])
            
            mgam[scen][reg] = pd.DataFrame(output[scen]["MGAM"][r, :, 0, :], 
                                           index=titles["T2TI"], 
                                           columns=tl)
        
            # mgamx[scen][reg] = pd.DataFrame(output[scen]["MGAMX"][r, :, 0, :], 
            #                                index=titles["T2TI"], 
            #                                columns=tl)    
    
            mewg[scen][reg] = pd.DataFrame(output[scen]["MEWG"][r, :, 0, :], 
                                           index=titles["T2TI"], 
                                           columns=tl)    

#%% Output to input csv folder

scenario = 'S0' # Edit as desired
file_path = f'Inputs/{scenario}/FTT-P/MGAM_{country}.csv' # Final save location

for r, reg in enumerate(titles["RTI"]):

    country = titles['RTI_short'][r]
    
    df = pd.DataFrame(columns=range(2001,2101), index = mgam['Gamma'][reg].index)
    
    # Merge the blank DataFrame with the DataFrame from gamma
    merged_df = df.combine_first(mgam['Gamma'][reg])
    merged_df.fillna(0, inplace = True)    
    
    # Save the merged DataFrame to a CSV file
    file_path = f'Inputs/{scenario}/FTT-P/MGAM_{country}.csv'
    merged_df.to_csv(file_path, index=True)