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
import pickle
params = {'legend.fontsize': 9,
          'figure.figsize': (5.5, 9),
          'axes.labelsize': 9,
          'axes.titlesize': 9,
          'xtick.labelsize':8,
          'ytick.labelsize':8}
pylab.rcParams.update(params)
# pyplot.locator_params(nbins=4)
# %matplotlib qt
# 

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
    dirp_out = os.path.join(dirp, "Output")
    dirp_graphs = os.path.join(dirp, "Graphs")
            
    # Get classifications
    titles = titles_f.load_titles()
    eu_regs = [reg for r, reg in enumerate(titles['RTI_short']) if r < 27 or r == 30]
    
    # Get variable dimensions
    dims, histend, domain, forstart = dims_f.load_dims()

    # Get titles
    # rti = e3me.get_dim_titles(dirp_e3me, "RSHORTTI")
    # rti_long = e3me.get_dim_titles(dirp_e3me, "RTI")
    # rti_dict = dict(zip(rti, rti_long))
    # t2ti = e3me.get_dim_titles(dirp_e3me, "T2TI")
    # yti = e3me.get_dim_titles(dirp_e3me, "YTI")
    # yrti = e3me.get_dim_titles(dirp_e3me, "YRTI")
    # cti = e3me.get_dim_titles(dirp_e3me, "CTI")
    # crti = e3me.get_dim_titles(dirp_e3me, "CRTI")
    # gti = e3me.get_dim_titles(dirp_e3me, "GTI")
    # futi = e3me.get_dim_titles(dirp_e3me, "FUTI")
    # jti = e3me.get_dim_titles(dirp_e3me, "JTI")
    # eu_regs = [reg for r, reg in enumerate(titles['RTI_short']) if r < 27 or r == 30]
 
    
    # scenario = "Dan1.mre"
    
    # misc = e3me.get_mre_data(dirp_e3me, scenario, ["FR0", "PRSC", "REX"], regs_short=True)
    # prsc = misc["PRSC"]
    
    # retrieve_vars = ["PFRC", "PFRG", "PFRM", "PFRE", "PFRB"]    
    # e3me_data = e3me.get_mre_data(dirp_e3me, scenario, retrieve_vars, sectors=['6 Chemicals'], regs_short=True)
    # e3me_data_processed = {var: pd.DataFrame(0.0, index=rti, columns=e3me_data["PFRE"]['AT'].columns)for var in e3me_data.keys()}
    
    # for var in e3me_data_processed.keys():
        
    #     for r, reg in enumerate(rti):
        
    #         e3me_data_processed[var].loc[reg, :] = e3me_data[var][reg].iloc[0,:]
            
    #     e3me_data_processed[var] /= prsc
        
    #     # Divide to rates compared to 2015 values
    #     e3me_data_processed[var] = e3me_data_processed[var].divide(e3me_data_processed[var].loc[:, 2015], axis=0)
        
    # %% Read pickle data
    
    pickle_fp = os.path.join(dirp_out, "Results.pickle")
    with open(pickle_fp, 'rb') as f:
        output = pickle.load(f)
        
    # Scenarios
    scenarios = {"Reference": "S0",
                 "Subsidies": "subs",
                 "Subsidies + CarbonTax": "subs_ct",
                 "Subsidies + CarbonTax + Regulations": "subs_ct_reg"}
    
    # Sectors
    sectors = {"Chemical industry": "CHI",
               "Food, beverages, and tobacco": "FBT",
               "Non-Ferrous Metals, Machinery, and Transport Equipment": "MTM",
               "Non-Metallic Minerals": "NMM",
               "Other Industrial Sectors": "OIS"}
    
    vars_fed = ["IFD1", "IFD2", "IFD3", "IFD4", "IFD5"]
    vars_ued = ["IUD1", "IUD2", "IUD3", "IUD4", "IUD5"]
    vars_lcoih = ["ILG1", "ILG2", "ILG3", "ILG4", "ILG5"]
    vars_gamma = ["IAM1", "IAM2", "IAM3", "IAM4", "IAM5"]
    vars_lcoih_sd = ["ILD1", "ILD2", "ILD3", "ILD4", "ILD5"]
    vars_emis = ["IWE1", "IWE2", "IWE3", "IWE4", "IWE5"]
    
    vars_all = vars_fed + vars_ued + vars_emis
    
    output_df = {}
    output_df_sector_tot = {}
    
    tl_pickle = list(range(2015, 2050+1))
    
    # Create dataframes and EU27+UK sums
    for scen_full, scen_short in scenarios.items():
        
        output_df[scen_full] = {}
        output_df_sector_tot[scen_full] = {}
        
        output_df_sector_tot[scen_full]["UED"] = {}
        output_df_sector_tot[scen_full]["FED"] = {}
        output_df_sector_tot[scen_full]["EMIS"] = {}
        
        output_df_sector_tot[scen_full]["UED"]['EU27+UK'] = pd.DataFrame(0.0, index=titles['ITTI'], columns=tl_pickle)
        output_df_sector_tot[scen_full]["FED"]['EU27+UK'] = pd.DataFrame(0.0, index=titles['ITTI'], columns=tl_pickle)
        output_df_sector_tot[scen_full]["EMIS"]['EU27+UK'] = pd.DataFrame(0.0, index=titles['ITTI'], columns=tl_pickle)

        
        for var in vars_all:
            
            output_df[scen_full][var] = {}
            output_df[scen_full][var]['EU27+UK'] = pd.DataFrame(0.0, index=titles['ITTI'], columns=tl_pickle)
            
            for r, reg in enumerate(titles["RTI_short"]):
                
                if reg in eu_regs:
                
                    output_df_sector_tot[scen_full]["UED"][reg] = pd.DataFrame(0.0, 
                                                                          index=titles['ITTI'], 
                                                                          columns=tl_pickle)
                    output_df_sector_tot[scen_full]["FED"][reg] = pd.DataFrame(0.0, 
                                                                          index=titles['ITTI'], 
                                                                          columns=tl_pickle)
                    output_df_sector_tot[scen_full]["EMIS"][reg] = pd.DataFrame(0.0, 
                                                                          index=titles['ITTI'], 
                                                                          columns=tl_pickle)
                    output_df[scen_full][var][reg] = pd.DataFrame(output[scen_short][var][r, :, 0, :], 
                                                                   index=titles['ITTI'], 
                                                                   columns=tl_pickle)
                    output_df[scen_full][var]['EU27+UK'] += output_df[scen_full][var][reg]
                    
                    if var in vars_fed:
                        
                        output_df_sector_tot[scen_full]["FED"][reg] += output_df[scen_full][var][reg]
                        output_df_sector_tot[scen_full]["FED"]["EU27+UK"] += output_df[scen_full][var][reg]
                        
                    if var in vars_ued:
                        
                        output_df_sector_tot[scen_full]["UED"][reg] += output_df[scen_full][var][reg]
                        output_df_sector_tot[scen_full]["UED"]["EU27+UK"] += output_df[scen_full][var][reg]
    
                    if var in vars_emis:
                        
                        output_df_sector_tot[scen_full]["EMIS"][reg] += output_df[scen_full][var][reg]                    
                        output_df_sector_tot[scen_full]["EMIS"]["EU27+UK"] += output_df[scen_full][var][reg]
    # %% Get weighted averages of LCOIH


    # for scen_full, scen_short in scenarios.items():

    #     for v, var in enumerate(vars_lcoih):  
            
    #         vgam = "IAM{}".format(v+1)
    #         vlcoih = "ILG{}".format(v+1)
    #         vlcoihsd = "ILD{}".format(v+1)
    #         vued = "IUD{}".format(v+1)
    
    #         for r, reg in enumerate(titles["RTI_short"]):
                
    #             if reg in eu_regs:
                    
                    
                    
    # %% Colour scheme
    
    colmap = {"Indirect Heating Coal": "black",
              "Indirect Heating Oil": "sienna",
              "Indirect Heating Gas": "gray",
              "Indirect Heating Biomass": "green",
              "Indirect Heating Electric": "goldenrod",
              "Indirect Heating Steam Distributed": "crimson",
              "Heat Pumps (Electricity)": "aqua",
              "Direct Heating Coal": "dimgrey",
              "Direct Heating Oil": "chocolate",
              "Direct Heating Gas": "lightgrey",
              "Direct Heating Biomass": "limegreen",
              "Direct Heating Electric": "gold",
              "Direct Heating Steam Distributed": "palevioletred"}
    
    techs_indirect = [tech for tech in titles["ITTI"] if "Direct" not in tech]
    colmap_indirect = {tech: colmap[tech] for tech in techs_indirect}
    techs_direct = [tech for tech in titles["ITTI"] if "Direct" not in tech]
    colmap_direct = {tech: colmap[tech] for tech in techs_direct}
    
    selected_regs = ["EU27+UK"]
    # selected_regs = list(output_df["S0"]["IUD1"].keys())
    
    # %% UED chart
    
    
    
    # eu_regs = ["EU27+UK"]
    for r, reg in enumerate(selected_regs):
        
        # Figure size
        figsize = (8, 10)
        # Create subplot
        fig, axes = plt.subplots(nrows=5, ncols=4,
                                 figsize=figsize,
                                 sharex=True, sharey='row')   
        
        
        tl_out = np.arange(2020, 2050+1)
        
        fig.suptitle("Process heat by technology\n{}".format(reg))
        
        for col, scen_full in enumerate(scenarios.keys()):
            
            scen_short = scenarios[scen_full]
            
            for row, sector in enumerate(sectors.values()):
                
                var_to_extract = "IUD{}".format(row+1)
                data = output_df[scen_full][var_to_extract][reg].loc[:, tl_out]*1e-3
                
                # First, the indirect technologies
                axes[row, col].stackplot(np.asarray(tl_out),
                                         data,
                                         labels=list(colmap.keys()),
                                         colors=list(colmap.values()))   
                
                # # Second, the direct technologies
                # axes[row, col].stackplot(np.asarray(tl_out),
                #                          data.loc[techs_direct, :].add(data.loc[techs_indirect, :].sum(axis=0), axis=1),
                #                          labels=list(colmap_direct.keys()),
                #                          colors=list(colmap_direct.values()),
                #                          hatch="x")
                    
                axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
                axes[row, col].grid(alpha=0.4, color="#E3E3E3");
                axes[row, col].tick_params('x', labelrotation=60)
                # axes_flat[r].label_outer()
                axes[row, col].set_xticks([2020, 2030, 2040, 2050])
                if row == 0: 
                    if scen_short == "S0":
                        axes[row, col].set_title("ref")
                    else:
                        axes[row, col].set_title(scen_short)
                if col == 0: axes[row, col].set_ylabel("UED in {}\nTWh/y".format(sector))
    
        h1, l1 = axes[0, 0].get_legend_handles_labels()
    
        fig.legend(handles=h1[::-1],
                   labels=l1[::-1],
                   loc="lower center",
                   bbox_to_anchor=(0.5, 0.05),
                   frameon=False,
                   borderaxespad=0.,
                   ncol=3,
                   title="Technologies",
                   fontsize=9)
    
        fig.subplots_adjust(hspace=0.35, wspace=0.15, bottom=0.25)
        
        fp = os.path.join(dirp_graphs, "UED_{}.svg".format(reg))
        fig.savefig(fp)
        # plt.show()  
        
    # %% UED all sectors
    for r, reg in enumerate(selected_regs):
        
        # Figure size
        figsize = (7, 5)
        # Create subplot
        fig, axes = plt.subplots(nrows=2, ncols=2,
                                 figsize=figsize,
                                 sharex=True, sharey='row')  
        axes_flat = axes.flatten()
        
        for s, scen_full in enumerate(scenarios.keys()):
            
            scen_short = scenarios[scen_full]       
            
            data = output_df_sector_tot[scen_full]["UED"][reg].loc[:, tl_out]*1e-3
            
            # First, the indirect technologies
            axes_flat[s].stackplot(np.asarray(tl_out),
                                     data,
                                     labels=list(colmap.keys()),
                                     colors=list(colmap.values()))             

            axes_flat[s].set_xlim([tl_out[0], tl_out[-1]]);
            axes_flat[s].grid(alpha=0.4, color="#E3E3E3");
            axes_flat[s].tick_params('x', labelrotation=60)
            # axes_flat[r].label_outer()
            axes_flat[s].set_xticks([2020, 2030, 2040, 2050])
            if scen_short == "S0": 
                axes_flat[s].set_title("ref")
            else:
                axes_flat[s].set_title(scen_short)
            if scen_short in ["S0", "subs_ct"]: axes_flat[s].set_ylabel("UED total\nTWh/y")
    
        h1, l1 = axes_flat[0].get_legend_handles_labels()
        
        fig.legend(handles=h1[::-1],
                   labels=l1[::-1],
                   loc="upper right",
                   bbox_to_anchor=(0.9, 0.85),
                   frameon=False,
                   borderaxespad=0.,
                   ncol=1,
                   title="Technologies",
                   fontsize=9)
    
        fig.subplots_adjust(hspace=0.2, wspace=0.15, bottom=0.25, right=0.5)
        
        fp = os.path.join(dirp_graphs, "UED_total_{}.svg".format(reg))
        fig.savefig(fp)
        plt.show()        
        
    
    # %% FED chart
    
    # eu_regs = ["EU27+UK"]
    for r, reg in enumerate(selected_regs):
        
        # Figure size
        figsize = (8, 10)
        # Create subplot
        fig, axes = plt.subplots(nrows=5, ncols=4,
                                 figsize=figsize,
                                 sharex=True, sharey='row')   
        
        
        tl_out = np.arange(2020, 2050+1)
        
        fig.suptitle("Final energy demand by technology\n{}".format(reg))
        
        for col, scen_full in enumerate(scenarios.keys()):
            
            scen_short = scenarios[scen_full]
            
            for row, sector in enumerate(sectors.values()):
                
                var_to_extract = "IFD{}".format(row+1)
                data = output_df[scen_full][var_to_extract][reg].loc[:, tl_out]*1e-3
                
                # First, the indirect technologies
                axes[row, col].stackplot(np.asarray(tl_out),
                                         data,
                                         labels=list(colmap.keys()),
                                         colors=list(colmap.values()))   
                
                # # Second, the direct technologies
                # axes[row, col].stackplot(np.asarray(tl_out),
                #                          data.loc[techs_direct, :].add(data.loc[techs_indirect, :].sum(axis=0), axis=1),
                #                          labels=list(colmap_direct.keys()),
                #                          colors=list(colmap_direct.values()),
                #                          hatch="x")
                    
                axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
                axes[row, col].grid(alpha=0.4, color="#E3E3E3");
                axes[row, col].tick_params('x', labelrotation=60)
                # axes_flat[r].label_outer()
                axes[row, col].set_xticks([2020, 2030, 2040, 2050])
                if row == 0: 
                    if scen_short == "S0":
                        axes[row, col].set_title("ref")
                    else:
                        axes[row, col].set_title(scen_short)
                if col == 0: axes[row, col].set_ylabel("FED in {}\nTWh/y".format(sector))
    
        h1, l1 = axes[0, 0].get_legend_handles_labels()
    
        fig.legend(handles=h1[::-1],
                   labels=l1[::-1],
                   loc="lower center",
                   bbox_to_anchor=(0.5, 0.05),
                   frameon=False,
                   borderaxespad=0.,
                   ncol=3,
                   title="Technologies",
                   fontsize=9)
    
        fig.subplots_adjust(hspace=0.35, wspace=0.15, bottom=0.25)
        
        fp = os.path.join(dirp_graphs, "FED_{}.svg".format(reg))
        fig.savefig(fp)
        # plt.show()       
        
    # %% UED all sectors
    for r, reg in enumerate(selected_regs):
        
        # Figure size
        figsize = (7, 5)
        # Create subplot
        fig, axes = plt.subplots(nrows=2, ncols=2,
                                 figsize=figsize,
                                 sharex=True, sharey='row')  
        axes_flat = axes.flatten()
        
        for s, scen_full in enumerate(scenarios.keys()):
            
            scen_short = scenarios[scen_full]       
            
            data = output_df_sector_tot[scen_full]["FED"][reg].loc[:, tl_out]*1e-3
            
            # First, the indirect technologies
            axes_flat[s].stackplot(np.asarray(tl_out),
                                     data,
                                     labels=list(colmap.keys()),
                                     colors=list(colmap.values()))             

            axes_flat[s].set_xlim([tl_out[0], tl_out[-1]]);
            axes_flat[s].grid(alpha=0.4, color="#E3E3E3");
            axes_flat[s].tick_params('x', labelrotation=60)
            # axes_flat[r].label_outer()
            axes_flat[s].set_xticks([2020, 2030, 2040, 2050])
            if scen_short == "S0": 
                axes_flat[s].set_title("ref")
            else:
                axes_flat[s].set_title(scen_short)
            if scen_short in ["S0", "subs_ct"]: axes_flat[s].set_ylabel("FED total\nTWh/y")
    
        h1, l1 = axes_flat[0].get_legend_handles_labels()
    
        fig.legend(handles=h1[::-1],
                   labels=l1[::-1],
                   loc="upper right",
                   bbox_to_anchor=(0.9, 0.85),
                   frameon=False,
                   borderaxespad=0.,
                   ncol=1,
                   title="Technologies",
                   fontsize=9)
    
        fig.subplots_adjust(hspace=0.2, wspace=0.15, bottom=0.25, right=0.5)
        
        fp = os.path.join(dirp_graphs, "FED_total_{}.svg".format(reg))
        fig.savefig(fp)
        plt.show()     
    # %% Emissions chart
    
    # eu_regs = ["EU27+UK"]
    for r, reg in enumerate(selected_regs):
        
        # Figure size
        figsize = (8, 10)
        # Create subplot
        fig, axes = plt.subplots(nrows=5, ncols=4,
                                 figsize=figsize,
                                 sharex=True, sharey='row')   
        
        
        tl_out = np.arange(2020, 2050+1)
        
        fig.suptitle("Direct emissions by technology\n{}".format(reg))
        
        for col, scen_full in enumerate(scenarios.keys()):
            
            scen_short = scenarios[scen_full]
            
            for row, sector in enumerate(sectors.values()):
                
                var_to_extract = "IWE{}".format(row+1)
                data = output_df[scen_full][var_to_extract][reg].loc[:, tl_out]*1e-3
                
                # First, the indirect technologies
                axes[row, col].stackplot(np.asarray(tl_out),
                                         data,
                                         labels=list(colmap.keys()),
                                         colors=list(colmap.values()))   
                
                # # Second, the direct technologies
                # axes[row, col].stackplot(np.asarray(tl_out),
                #                          data.loc[techs_direct, :].add(data.loc[techs_indirect, :].sum(axis=0), axis=1),
                #                          labels=list(colmap_direct.keys()),
                #                          colors=list(colmap_direct.values()),
                #                          hatch="x")
                    
                axes[row, col].set_xlim([tl_out[0], tl_out[-1]]);
                axes[row, col].grid(alpha=0.4, color="#E3E3E3");
                axes[row, col].tick_params('x', labelrotation=60)
                # axes_flat[r].label_outer()
                axes[row, col].set_xticks([2020, 2030, 2040, 2050])
                if row == 0: 
                    if scen_short == "S0":
                        axes[row, col].set_title("ref")
                    else:
                        axes[row, col].set_title(scen_short)
                if col == 0: axes[row, col].set_ylabel("Emissions in {}\nmtCO2/y".format(sector))
    
        h1, l1 = axes[0, 0].get_legend_handles_labels()
    
        fig.legend(handles=h1[::-1],
                   labels=l1[::-1],
                   loc="lower center",
                   bbox_to_anchor=(0.5, 0.05),
                   frameon=False,
                   borderaxespad=0.,
                   ncol=3,
                   title="Technologies",
                   fontsize=9)
    
        fig.subplots_adjust(hspace=0.35, wspace=0.15, bottom=0.25)
        
        fp = os.path.join(dirp_graphs, "EMIS_{}.svg".format(reg))
        fig.savefig(fp)
        # plt.show()        

    # %% EMIS all sectors
    for r, reg in enumerate(selected_regs):
        
        # Figure size
        figsize = (7, 5)
        # Create subplot
        fig, axes = plt.subplots(nrows=2, ncols=2,
                                 figsize=figsize,
                                 sharex=True, sharey=True)  
        axes_flat = axes.flatten()
        
        for s, scen_full in enumerate(scenarios.keys()):
            
            scen_short = scenarios[scen_full]       
            
            data = output_df_sector_tot[scen_full]["EMIS"][reg].loc[:, tl_out]*1e-3
            print(data.sum(axis=0))
            # First, the indirect technologies
            axes_flat[s].stackplot(np.asarray(tl_out),
                                     data,
                                     labels=list(colmap.keys()),
                                     colors=list(colmap.values()))             

            axes_flat[s].set_xlim([tl_out[0], tl_out[-1]]);
            axes_flat[s].grid(alpha=0.4, color="#E3E3E3");
            axes_flat[s].tick_params('x', labelrotation=60)
            # axes_flat[r].label_outer()
            axes_flat[s].set_xticks([2020, 2030, 2040, 2050])
            if scen_short == "S0": 
                axes_flat[s].set_title("ref")
            else:
                axes_flat[s].set_title(scen_short)
            if scen_short in ["S0", "subs_ct"]: axes_flat[s].set_ylabel("Emissions\nmtCO2/y")
    
        h1, l1 = axes_flat[0].get_legend_handles_labels()
    
        fig.legend(handles=h1[::-1],
                   labels=l1[::-1],
                   loc="upper right",
                   bbox_to_anchor=(0.9, 0.85),
                   frameon=False,
                   borderaxespad=0.,
                   ncol=1,
                   title="Technologies",
                   fontsize=9)
    
        fig.subplots_adjust(hspace=0.2, wspace=0.15, bottom=0.25, right=0.5)
        
        fp = os.path.join(dirp_graphs, "EMIS_total_{}.svg".format(reg))
        fig.savefig(fp)
        plt.show()               
        
            
        
            
    
    

    
    
    