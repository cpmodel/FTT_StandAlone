# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 17:10:10 2019

This script extracts all the data from FTT excel sheets in the
"/In/FTTAssumptions/[model]" folders and stores it on the databank of your
choosing (default C databank).

The user can select one or more scenarios to upload from the excel sheet.

@author: MM
"""

import pandas as pd
import numpy as np
import os
from celib import DB1
from pathlib import Path
import SourceCode.support.dimensions_functions as dims_f
import SourceCode.support.titles_functions as titles_f
import get_e3me as e3me
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import matplotlib.colors as mcolors
import matplotlib.pylab as pylab
from matplotlib.ticker import MaxNLocator
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FormatStrFormatter
params = {'legend.fontsize': 9,
          'figure.figsize': (5.5, 9),
          'axes.labelsize': 9,
          'axes.titlesize': 9,
          'xtick.labelsize':8,
          'ytick.labelsize':8}
pylab.rcParams.update(params)
# pyplot.locator_params(nbins=4)
# %matplotlib qt


#%%
if __name__ == '__main__':
### ----------------------------------------------------------------------- ###
### -------------------------- VARIABLE SETUP ----------------------------- ###
### ----------------------------------------------------------------------- ###
    # Define paths, directories and subfolders
    dirp = os.path.dirname(os.path.realpath(__file__))
    dirp_up = Path(dirp).parents[0]
    dirp_in = os.path.join(dirp, 'Inputs')
    dirp_master = os.path.join(dirp_in, "_MasterFiles")
    dirp_e3me = "C:\E3ME\Master"
            
    # Get classifications
    titles = titles_f.load_titles()
    
    # Get variable dimensions
    dims, histend, domain, forstart = dims_f.load_dims()

    # Get titles
    rti = e3me.get_dim_titles(dirp_e3me, "RSHORTTI")
    rti_long = e3me.get_dim_titles(dirp_e3me, "RTI")
    rti_dict = dict(zip(rti, rti_long))
    t2ti = e3me.get_dim_titles(dirp_e3me, "T2TI")
    yti = e3me.get_dim_titles(dirp_e3me, "YTI")
    yrti = e3me.get_dim_titles(dirp_e3me, "YRTI")
    cti = e3me.get_dim_titles(dirp_e3me, "CTI")
    crti = e3me.get_dim_titles(dirp_e3me, "CRTI")
    gti = e3me.get_dim_titles(dirp_e3me, "GTI")
    futi = e3me.get_dim_titles(dirp_e3me, "FUTI")
    jti = e3me.get_dim_titles(dirp_e3me, "JTI")
    
    scenario = "Dan1.mre"
    
    misc = e3me.get_mre_data(dirp_e3me, scenario, ["FR0", "PRSC", "REX"], regs_short=True)
    prsc = misc["PRSC"]
    
    retrieve_vars = ["PFRC", "PFRG", "PFRM", "PFRE", "PFRB"]    
    e3me_data = e3me.get_mre_data(dirp_e3me, scenario, retrieve_vars, sectors=['6 Chemicals'], regs_short=True)
    e3me_data_processed = {var: pd.DataFrame(0.0, index=rti, columns=e3me_data["PFRE"]['AT'].columns)for var in e3me_data.keys()}
    
    for var in e3me_data_processed.keys():
        
        for r, reg in enumerate(rti):
        
            e3me_data_processed[var].loc[reg, :] = e3me_data[var][reg].iloc[0,:]
            
        e3me_data_processed[var] /= prsc
        
        # Divide to rates compared to 2015 values
        e3me_data_processed[var] = e3me_data_processed[var].divide(e3me_data_processed[var].loc[:, 2015], axis=0)
    
    # %%
    # Reshuffle
    ener_price_rates = dict()
    ener_price_for_graph = dict()
    mapping = dict(zip(titles['ITTI'], ["PFRC", "PFRM", "PFRG", "PFRB", "PFRE", "PFRG", "PFRE", "PFRC", "PFRM", "PFRG", "PFRB", "PFRE", "PFRG"]))
    mapping_fuel = dict(zip(["Coal", "Gas", "Oil", "Electricity", "Biomass"], ["PFRC", "PFRG", "PFRM", "PFRE", "PFRB"]))
    eu_regs = [reg for r, reg in enumerate(titles['RTI_short']) if r < 27 or r == 30]
    dir_out = os.path.join(dirp_in, 'S0', "FTT-IH-CHI")
    for reg in rti:
        ener_price_rates[reg] = pd.DataFrame(1.0, index=titles["ITTI"], columns=prsc.columns)
        
        if reg in eu_regs:
            ener_price_for_graph[reg] = pd.DataFrame(1.0, index=mapping_fuel.keys(), columns=prsc.columns)
            for en1, en2 in mapping_fuel.items():
                ener_price_for_graph[reg].loc[en1, :] = e3me_data_processed[en2].loc[reg, :]
            
        
        for tech in titles["ITTI"]:
            ener_price_rates[reg].loc[tech, :] = e3me_data_processed[mapping[tech]].loc[reg, :]
            
        output_file = os.path.join(dir_out, "{}_{}.csv".format("IRFT", reg))
        ener_price_rates[reg].to_csv(output_file)
        
    # %% Create energy price graph per EU reg
    
    # Figure size
    figsize = (6.75, 11)
    # Create subplot
    fig, axes = plt.subplots(nrows=7, ncols=4,
                             figsize=figsize,
                             sharex=True, sharey='row')   
    
    
    tl_out = np.arange(2020, 2050+1)
    
    axes_flat = axes.flatten()
    
    cols = ["#0B1F2C", "#909090", "#C5446E", "#49C9C5", "#AAB71D"]
        
    for r, reg in enumerate(eu_regs):
        
        for f, fuel in enumerate(ener_price_for_graph[reg].index):
            axes_flat[r].plot(np.asarray(tl_out),
                              ener_price_for_graph[reg].loc[fuel, tl_out],
                              label=fuel,
                              color=cols[f],
                              linewidth=1.5)

                
            axes_flat[r].set_xlim([tl_out[0], tl_out[-1]]);
            axes_flat[r].grid(alpha=0.4, color="#E3E3E3");
            axes_flat[r].tick_params('x', labelrotation=60)
            # axes_flat[r].label_outer()
            axes_flat[r].set_xticks([2020, 2030, 2040, 2050])
            axes_flat[r].set_yticks([1.0, 3.0, 5.0, 7.0])
            axes_flat[r].set_title(rti_dict[reg])
            
        if reg == rti_dict["FI"]: 
            axes_flat[r].set_ylabel("Energy price index (2015=1)")

    h1, l1 = axes_flat[r].get_legend_handles_labels()

    fig.legend(handles=h1[::-1],
               labels=l1[::-1],
               loc="lower center",
               bbox_to_anchor=(0.5, 0.05),
               frameon=False,
               borderaxespad=0.,
               ncol=3,
               title="Energy carriers",
               fontsize=9)

    fig.subplots_adjust(hspace=0.35, wspace=0.15, bottom=0.175)

    # fig.savefig(fp)
    plt.show()    
            
        
            
        
            
    
    

    
    
    