# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:33:54 2022

@author: Femke
"""
# -*- coding: utf-8 -*-

import os
from celib import DB1, MRE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pylab as pylab
from pathlib import Path
import matplotlib as mpl

from E3MEpackage import country_to_E3ME_region, E3ME_regions_names
from preprocessing import get_df


from matplotlib.ticker import MaxNLocator

# This helps with the bug that seaborn seems to override mpl rcParams
import importlib
importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)

params = {'legend.fontsize': 16,
          'figure.figsize': (7, 5),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize':16,
         'ytick.labelsize':16,
         'text.usetex': False,
         "svg.fonttype": 'none'}
pylab.rcParams.update(params)

#%%
if __name__ == '__main__':
    do_plot = True
    figures_directory = "C://Users\Femke\OneDrive - University of Exeter (1)\Documents\Postdoc\E3ME\BNEFdata\Figures//"
    # This script can be saved in post-processing which is on the same level as Master
    dirp_graph = os.path.dirname(os.path.realpath(__file__))
    dirp = Path(dirp_graph).parents[1]
    dir_db = os.path.join(dirp, 'databank')
    dirp_out = os.path.join(dirp, 'Output')
    
    udbp = os.path.join(dir_db, 'U.db1')
    with DB1(udbp) as db1:
        rti = db1['RSHORTTI']
        stti = db1['STTI']
        t2ti = db1['T2TI']
        lbti = db1['LBTI']
        jti = db1['JTI']
        yti = db1['YTI']
    tl_2070 = list(range(2010, 2060+1))
#    tl_2050 = list(range(2005, 2050+1))
    lbti = [lb[2:] for lb in lbti]

    scenarios = dict(("DAN_UC"+str(ind), "DAN_UC"+str(ind)+".mre") for ind in range(10, 12))
    # scenarios = {
    #             'Baseline': 'Dan_ba.mre',
    #             'Scenario A': 'Dan_ba5.mre',
    #             'Baseline_new': 'Dan_ba25.mre',
    #             }

    colour_list = ['cyan', 'lightsalmon', 'black', 'darkgrey', 'purple',
                   'violet', 'forestgreen', 'lime', 'darkblue', 'deepskyblue',
                   'gold', 'orange', 'brown', 'lightcoral']
    #colour_map_tech = dict(zip(new_tech_agg, colour_list))
    ce_cols = ['#0B1F2C', '#909090', '#C5446E', '#49C9C5', '#AAB71D', '#009FE3']
    colour_map_lbs = dict(zip(lbti, ce_cols))

    regs_to_print = {
                     'EU-27': [x for x in range(33) if (x < 27 or x == 32) and x != 14],
                     'USA': [33],
                     'India': [41],
                     'China': [40],
                     'Japan': [34],
                     'Africa': [x for x in range(58, 70) if x not in [60, 61]],
                     'Global': list(range(len(rti)))
                     }
  
    # EU countries
    regs_to_print2 = {"Germany": [2],
                     "France": [5],
                     "United Kingdom": [14],
                     "Spain": [4],
                     "Italy": [7]}
    my_map = E3ME_regions_names()
    
    #regs_to_print = dict((v, [k-1]) for k, v in my_map.items())
    scenarios_to_print = ["DAN_UC"+str(ind) for ind in range(10, 12)]

    #%%
    df, df_shares, df_loadband = get_df(scenarios, scenarios_to_print, dirp_out, regs_to_print, 
                           print_temperature=True)
        
    
    colours = {'Nuclear': 'C0', "Coal": 'grey', "Oil": 'darkgrey', "Gas": 'lightgrey',
               "Bioenergy":'C2', 'Other': 'C8','Large Hydro': 'paleturquoise',
               'Onshore': 'cornflowerblue', "Offshore": 'royalblue',
               'Solar PV': 'salmon', 'CSP': 'gold'}
    df_shares_global = df_shares[df_shares['Region']=='Global']
    final_shares = df_shares_global[df_shares_global["Year"]==2060]
    
    sns.histplot(final_shares, x="Solar PV", y="Coal")
    
    

           
    # Plotting share of energy
   

    


        
        
    
