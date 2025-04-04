# -*- coding: utf-8 -*-
"""
=========================================
run_file.py
=========================================
Run file for FTT Stand alone.
#############################


Programme calls the FTT stand-alone model run class, and executes model run.
Call this file from the command line (or terminal) to run FTT Stand Alone.

Local library imports:

    Model Class:

    - `ModelRun <model_class.html>`__
        Creates a new instance of the ModelRun class


"""


# Local library imports
from SourceCode.model_class import ModelRun
import numpy as np
import pandas as pd

# Instantiate the run
model = ModelRun()
# model.scenarios = ['S{}'.format(i) for i in [0,3]]

# Fetch ModelRun attributes, for examination
# Titles of the model
titles = model.titles
# Dimensions of model variables
dims = model.dims
# Model inputs
inputs = model.input
# Metadata for inputs of the model
histend = model.histend
# Domains to which variables belong
domain = model.domain
tl = model.timeline
scens = model.scenarios

scen_dict = dict(zip(model.scenarios, ['REF', 'CP', 'MD', 'CP+MD']))
# %%
# Call the 'run' method of the ModelRun class to solve the model
model.run()

# Fetch ModelRun attributes, for examination
# Output of the model
output_all = model.output

#

# %% Graph init
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
from SourceCode.support.divide import divide
params = {'legend.fontsize': 9,
          'figure.figsize': (5.5, 9),
          'axes.labelsize': 8,
          'axes.titlesize': 9,
          'xtick.labelsize':7,
          'ytick.labelsize':7}
pylab.rcParams.update(params)

# ce_cols = 
techs_to_show = [tech for t, tech in enumerate(titles['HYTI']) if t in [0,8, 9,10]]

tl_out = np.arange(2025, 2051)
colors = ["#AAB71D", "#49C9C5", "#C5446E", "#009FE3"]

y_2024 = tl.tolist().index(2024)
inflator = output_all['S0']['PRSC'][:, 0, 0, y_2024] / output_all['S0']['EX'][:, 0, 0, y_2024]
# Set inflator to average EU price levels (~Eurozone)
inflator = inflator* 0 + 1.3

version = 3

# %% Setup converters and colour maps

dem_vectors =['NH3 for fertiliser', 'NH3 for chemicals', 'MeOH for chemicals', ' H2 for oil refining']
dem_colmap = dict(zip(dem_vectors, colors))

agg_techs2 = ['FF', 'FF+CCS', 'ELEC-Grid', 'ELEC-VRE', 'Other']
conv2 = pd.DataFrame(0.0, index=titles['HYTI'], columns=agg_techs2)
conv2.iloc[[0,2], 0] = 1
conv2.iloc[[1,3, 4], 1] = 1
conv2.iloc[[5,6,7],2] = 1
conv2.iloc[[8,9,10],3] = 1
conv2.iloc[4, 4] = 0

agg_techs = ['FF', 'FF+CCS', 'ELEC', 'Other']
conv = pd.DataFrame(0.0, index=titles['HYTI'], columns=agg_techs)
conv.iloc[[0,2], 0] = 1
conv.iloc[[1,3], 1] = 1
conv.iloc[[5,6,7,8,9,10],2] = 1
conv.iloc[4, 3] = 1

ls_scens = ['solid', 'dashed', 'dotted', 'dashdot']
tech_colors = ['black', "#49C9C5", "#AAB71D", "#C5446E", "#AAB71D", "#C5446E", 'purple']
tech_colors2 = ['black', "#49C9C5", "darkgreen", "#AAB71D", "#AAB71D", "#C5446E", 'purple']

var_weight = 'HYG1'
var_lcoh = 'HYCC'
var_prod = 'HYG1'

# %% Graph 1 - Global average LCOH2 movements

# fp = os.path.join('Graphs', 'LCOH_polbrief_v{}.svg'.format(version))
# # Figure size
# figsize = (7.5, 3.75)
# # Create subplot    
# fig, axes = plt.subplots(nrows=1,
#                          ncols=4,
#                          figsize=figsize,
#                          sharey=True,
#                          sharex=True)

# axes_flat = axes.flatten()

# var_weight = 'HYG1'
# var_lcoh = 'HYCC'
# reg_idx = 3

# for t, tech in enumerate(agg_techs2[:-1]):
    
#     tech_titles = conv2.index[conv2[tech] == 1].tolist()
#     idx = [conv2.index.get_loc(i) for i in tech_titles]
    
#     for s, scen in enumerate(scen_dict.keys()):
        
#         # Global weighted average by individual technology
#         lcoh_weight = divide(output_all[scen][var_weight][:, idx, 0, :],
#                              output_all[scen][var_weight][:, idx, 0, :].sum(axis=0).sum(axis=0)[None, None, :])
        
#         lcoh_avg = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] * lcoh_weight, axis=0).sum(axis=0)
#         lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :], axis=(0,1))

#         df_lcoh_avg = pd.Series(lcoh_avg2, index=tl)
        
#         axes_flat[t].plot(np.asarray(tl_out),
#                         df_lcoh_avg[tl_out].values,
#                         label=scen_dict[scen],
#                         color=tech_colors2[t],
#                         linestyle=ls_scens[s]) 
        
#         axes_flat[t].set_title(tech)
        
#     axes_flat[t].set_xlim([tl_out[0], tl_out[-1]]);
#     axes_flat[t].grid(alpha=0.4, color="#E3E3E3");
#     axes_flat[t].tick_params('x', labelrotation=60)
#     axes_flat[t].set_ylabel('Euro/kg H2')  
#     axes_flat[t].label_outer()
#     # axes_flat[t].set_xticks([2025, 2030, 2040, 2050])   
      

# h1, l1 = axes_flat[0].get_legend_handles_labels()

# fig.legend(handles=h1,
#            labels=l1,
#            loc="lower center",
#            bbox_to_anchor=(0.5,0.05),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=4,
#            title="Scenarios",
#            fontsize=8)

# fig.subplots_adjust(hspace=0.2, wspace=0.2, right=0.97, bottom=0.27, left=0.05, top=0.95)

# fig.savefig(fp)
# plt.show()    
# %% Graph 1 - Global average LCOH2 movements + production

# fp = os.path.join('Graphs', 'LCOH_Production_polbrief_v{}.svg'.format(version))
# # Figure size
# figsize = (7.5,5)
# # Create subplot    
# fig, axes = plt.subplots(nrows=2,
#                          ncols=4,
#                          figsize=figsize,
#                          sharey='row',
#                          sharex=True)

# # axes = axes.flatten()

# var_weight = 'HYG1'
# var_lcoh = 'HYCC'
# var_prod = 'HYG1'
# reg_idx = 3

# for t, tech in enumerate(agg_techs2[:-1]):
    
#     tech_titles = conv2.index[conv2[tech] == 1].tolist()
#     idx = [conv2.index.get_loc(i) for i in tech_titles]
    
#     for s, scen in enumerate(scen_dict.keys()):
        
#         # Global weighted average by individual technology
#         lcoh_weight = divide(output_all[scen][var_weight][:, idx, 0, :],
#                              output_all[scen][var_weight][:, idx, 0, :].sum(axis=0).sum(axis=0)[None, None, :])
        
#         lcoh_avg = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] * inflator[:, None, None] * lcoh_weight, axis=0).sum(axis=0)
#         lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :] * inflator[:, None, None] , axis=(0,1))

#         # lcoh_avg = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] * lcoh_weight, axis=0).sum(axis=0)
#         # lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :] , axis=(0,1))

#         df_lcoh_avg = pd.Series(lcoh_avg2, index=tl)
        
#         axes[0,t].plot(np.asarray(tl_out),
#                         df_lcoh_avg[tl_out].values,
#                         label=scen_dict[scen],
#                         color=tech_colors2[t],
#                         linestyle=ls_scens[s]) 
        
#         axes[0,t].set_title(tech)
        
#     axes[0,t].set_xlim([tl_out[0], tl_out[-1]]);
#     axes[0,t].set_ylim([0, 7])
#     axes[0,t].grid(alpha=0.4, color="#E3E3E3");
#     axes[0,t].tick_params('x', labelrotation=60)
#     axes[0,t].set_ylabel('Levelised cost\nEuro(2024)/kg H$_2$')  
#     axes[0,t].label_outer()
#     # axes_flat[t].set_xticks([2025, 2030, 2040, 2050]) 
    
#     for s, scen in enumerate(scen_dict.keys()):
        
#         prod = output_all[scen][var_prod][:, idx, 0, :].sum(axis=0).sum(axis=0) * 1e-3
#         df_prod = pd.Series(prod, index=tl)
        
#         axes[1,t].plot(np.asarray(tl_out),
#                         df_prod[tl_out].values,
#                         label=scen_dict[scen],
#                         color=tech_colors2[t],
#                         linestyle=ls_scens[s]) 
        
        
#     axes[1,t].set_xlim([tl_out[0], tl_out[-1]]);
#     axes[1,t].grid(alpha=0.4, color="#E3E3E3");
#     axes[1,t].tick_params('x', labelrotation=60)
    
#     # axes_flat[t].set_xticks([2023, 2030, 2040, 2050])   
#     axes[1,t].set_ylabel('Production\nMt H$_2$')   
#     axes[1,t].label_outer()
      

# h1, l1 = axes[0,0].get_legend_handles_labels()

# fig.legend(handles=h1,
#            labels=l1,
#            loc="lower center",
#            bbox_to_anchor=(0.5,0.05),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=4,
#            title="Scenarios",
#            fontsize=8)

# fig.subplots_adjust(hspace=0.2, wspace=0.1, right=0.97, bottom=0.27, left=0.1, top=0.95)

# fig.savefig(fp)
# plt.show()   

# %% Graph 1 - Global average LCOH2 movements + production by scen

# fp = os.path.join('Graphs', 'LCOH_Production_polbrief_by_scen_v{}.svg'.format(version))
# # Figure size
# figsize = (7.5,5)
# # Create subplot    
# fig, axes = plt.subplots(nrows=2,
#                          ncols=4,
#                          figsize=figsize,
#                          sharey='row',
#                          sharex=True)

# # axes = axes.flatten()

# var_weight = 'HYG1'
# var_lcoh = 'HYCC'
# var_prod = 'HYG1'
# reg_idx = 3

# for s, scen in enumerate(scen_dict.keys()):
    
#     for t, tech in enumerate(agg_techs2[:-1]):
    
#         tech_titles = conv2.index[conv2[tech] == 1].tolist()
#         idx = [conv2.index.get_loc(i) for i in tech_titles]
    

        
#         # Global weighted average by individual technology
#         lcoh_weight = divide(output_all[scen][var_weight][:, idx, 0, :],
#                              output_all[scen][var_weight][:, idx, 0, :].sum(axis=0).sum(axis=0)[None, None, :])
        
#         lcoh_avg = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] * inflator[:, None, None] * lcoh_weight, axis=0).sum(axis=0)
#         lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :] * inflator[:, None, None] , axis=(0,1))

#         # lcoh_avg = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] * lcoh_weight, axis=0).sum(axis=0)
#         # lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :] , axis=(0,1))

#         df_lcoh_avg = pd.Series(lcoh_avg2, index=tl)
        
#         axes[0,s].plot(np.asarray(tl_out),
#                         df_lcoh_avg[tl_out].values,
#                         label=tech,
#                         color=tech_colors2[t]) 
        
#         axes[0,s].set_title(scen_dict[scen])
        
#     axes[0,s].set_xlim([tl_out[0], tl_out[-1]]);
#     axes[0,s].set_ylim([0, 7])
#     axes[0,s].grid(alpha=0.4, color="#E3E3E3");
#     axes[0,s].tick_params('x', labelrotation=60)
#     axes[0,s].set_ylabel('Levelised cost\nEuro(2024)/kg H$_2$')  
#     axes[0,s].label_outer()
#     # axes[0,s].set_xticks([2025, 2030, 2040, 2050]) 
    
#     # for s, scen in enumerate(scen_dict.keys()):
        
#     prod = output_all[scen][var_prod][:, idx, 0, :].sum(axis=0).sum(axis=0) * 1e-3
#     prod = np.matmul(conv2.iloc[:, :-1].T, output_all[scen][var_prod].sum(axis=0)[:, 0, :]) * 1e-3
#     df_prod = pd.DataFrame(prod.values, index=agg_techs2[:-1], columns=tl)
    
#     axes[1,s].stackplot(np.asarray(tl_out),
#                     df_prod[tl_out].values,
#                     labels=agg_techs2[:-1],
#                     colors=tech_colors2) 
        
        
#     axes[1,s].set_xlim([tl_out[0], tl_out[-1]]);
#     axes[1,s].grid(alpha=0.4, color="#E3E3E3");
#     axes[1,s].tick_params('x', labelrotation=60)
    
#     # axes[1,s].set_xticks([2025, 2030, 2040, 2050])   
#     axes[1,s].set_ylabel('Production\nMt H$_2$')   
#     axes[1,s].label_outer()
      

# h1, l1 = axes[1,0].get_legend_handles_labels()

# fig.legend(handles=h1,
#            labels=l1,
#            loc="lower center",
#            bbox_to_anchor=(0.5,0.05),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=4,
#            title="Technologies",
#            fontsize=8)

# fig.subplots_adjust(hspace=0.2, wspace=0.1, right=0.97, bottom=0.27, left=0.1, top=0.95)

# fig.savefig(fp)
# plt.show()     
        
# %% Graph 2 - Demand projections
scen='S0'

fp = os.path.join('Graphs', 'Demand_v{}.svg'.format(version))
# Figure size
figsize = (5, 5.5)
# Create subplot    
fig, axes = plt.subplots(nrows=1,
                          ncols=1,
                          figsize=figsize)

df_demand_vectors = pd.DataFrame(0, index=dem_vectors, columns=tl)
df_demand_vectors.iloc[0, :] = output_all[scen]['HYD1'][:,0,0,:].sum(axis=0)
df_demand_vectors.iloc[1, :] = output_all[scen]['HYD2'][:,0,0,:].sum(axis=0)
df_demand_vectors.iloc[2, :] = output_all[scen]['HYD3'][:,0,0,:].sum(axis=0)
df_demand_vectors.iloc[3, :] = output_all[scen]['HYD4'][:,0,0,:].sum(axis=0)

green_demand = pd.Series(output_all[scen]['WGRM'][:,0,0,:].sum(axis=0), index=tl)

axes.stackplot(np.asarray(tl_out),
                df_demand_vectors.loc[:, tl_out].values*1e-3,
                labels=dem_colmap.keys(),
                colors=dem_colmap.values())

# axes.plot(np.asarray(tl_out),
#           green_demand.loc[tl_out].values*1e-3,
#           label='Green fertiliser demand',
#           color='black',
#           linestyle=':')

axes.set_xlim([tl_out[0], tl_out[-1]]);
axes.grid(alpha=0.4, color="#E3E3E3");
axes.tick_params('x', labelrotation=60)
# axes[s, 0].label_outer()
# axes.set_xticks([2023, 2030, 2040, 2050])   
axes.set_ylabel('Mt H$_2$-eq.')

axes.set_title('Vectors of Hydrogen demand')


h1, l1 = axes.get_legend_handles_labels()
l1_adj = ['NH$_3$ for fertiliser', 
          'NH$_3$ for chemicals', 
          'MeOH for chemicals',
          'H$_2$ for oil refining']

fig.legend(handles=h1,
            labels=l1_adj,
            loc="lower center",
            bbox_to_anchor=(0.5,0.04),
            frameon=False,
            borderaxespad=0.,
            ncol=2,
            title="Demand vectors",
            fontsize=8)

fig.subplots_adjust(hspace=0.2, wspace=0.2, right=0.97, bottom=0.25, left=0.15, top=0.95)

fig.savefig(fp)
plt.show()
 

# %% Graph 3 - Technology composition

# fp = os.path.join('Graphs', 'Tech_comp_v{}.svg'.format(version))
# # Figure size
# figsize = (7.5, 3.75)
# # Create subplot    
# fig, axes = plt.subplots(nrows=1,
#                          ncols=3,
#                          figsize=figsize,
#                          sharey=True,
#                          sharex=True)

# axes_flat = axes.flatten()

# var_prod = 'HYG1'

# for t, tech in enumerate(agg_techs[:-1]):
    
#     tech_titles = conv.index[conv[tech] == 1].tolist()
#     idx = [conv.index.get_loc(i) for i in tech_titles]
    
#     for s, scen in enumerate(scen_dict.keys()):
        
#         prod = output_all[scen][var_prod][:, idx, 0, :].sum(axis=0).sum(axis=0) * 1e-3
#         df_prod = pd.Series(prod, index=tl)
        
#         axes_flat[t].plot(np.asarray(tl_out),
#                         df_prod[tl_out].values,
#                         label=scen_dict[scen],
#                         color=colors[t],
#                         linestyle=ls_scens[s]) 
        
#         axes_flat[t].set_title(tech)
        
#     axes_flat[t].set_xlim([tl_out[0], tl_out[-1]]);
#     axes_flat[t].grid(alpha=0.4, color="#E3E3E3");
#     axes_flat[t].tick_params('x', labelrotation=60)
#     axes_flat[t].label_outer()
#     # axes_flat[t].set_xticks([2023, 2030, 2040, 2050])   
#     axes_flat[t].set_ylabel('Mt H2')    

# h1, l1 = axes_flat[0].get_legend_handles_labels()

# fig.legend(handles=h1,
#            labels=l1,
#            loc="lower center",
#            bbox_to_anchor=(0.5,0.04),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=4,
#            title="Scenarios",
#            fontsize=8)

# fig.subplots_adjust(hspace=0.2, wspace=0.2, right=0.97, bottom=0.35, left=0.15, top=0.95)

# fig.savefig(fp)
# plt.show()   

# %% Graph 4 -  comparing Brazil vs World

# Technology composition in 2050
# Price of Hydrogen on 2nd axis





# %% Table 1 -  cumulative investment 

# FF H2
# FF + CCS H2 
# ELEC H2
# Dedicated power capacity

years = [2050]
cols = ["{} - {}".format(year, sn) for year in years for sn in scen_dict.values()]
rows = ['Cumulative investment in FF capacity (bEuro(2024))',
        'Cumulative investment in FF+CCS capacity (bEuro(2024))',
        'Cumulative investment in ELEC-grid capacity (bEuro(2024))',
        'Cumulative investment in ELEC-VRE capacity (bEuro(2024))',
        'Cumulative investment in dedicated VRE capacity (bEuro(2024))',
        'Energy (incl. feedstock) consumption by FF technologies (ktoe/y)',
        'Energy (incl. feedstock) consumption by FF+CCS technologies (ktoe/y)',
        'Energy consumption by ELEC-grid technologies (ktoe/y)',
        'Energy consumption by ELEC-VRE technologies (ktoe/y)',
        'Dedicated electricity generation at ELEC-VRE sites (ktoe/y)',
        'Annual emissions (Gt CO$_2$/y)',
        'Cumulative emissions (Gt CO$_2$)']
table_out = pd.DataFrame(0.0, index=rows, columns=cols)

scen_dict_reverse = dict(zip(scen_dict.values(), scen_dict.keys()))

year_idx_ini = tl.tolist().index(2025)

for col in cols:
    
    year = int(col.split(' - ')[0])
    scen_name = col.split(' - ')[1]
    
    year_idx = tl.tolist().index(year)
    scen = scen_dict_reverse[scen_name]
    
    r = 0
    
    for t, tech in enumerate(agg_techs2[:-1]):
    
        tech_titles = conv2.index[conv2[tech] == 1].tolist()
        idx = [conv2.index.get_loc(i) for i in tech_titles] 
        
        table_out.loc[rows[r], col] = np.sum(output_all[scen]['HYIY'][:, idx, 0, year_idx_ini:year_idx]* 
                                             inflator[:, None, None])
        

        r += 1
        if tech  ==  'ELEC-VRE':
            
            table_out.loc[rows[r], col]= (np.sum(output_all[scen]['HYIT'][:, idx, 0, year_idx_ini:year_idx]* 
                                                 inflator[:, None, None]) -
                                          np.sum(output_all[scen]['HYIY'][:, idx, 0, year_idx_ini:year_idx]* 
                                                 inflator[:, None, None]))
            r += 1
            
            
    # Energy use
    for t, tech in enumerate(agg_techs2[:-1]):
    
        tech_titles = conv2.index[conv2[tech] == 1].tolist()
        idx = [conv2.index.get_loc(i) for i in tech_titles] 
        
        # Feedstock
        feedstock = np.sum(output_all[scen]['BCHY'][:, idx, 8, 0] *
                          output_all[scen]['HYG1'][:, idx, 0, year_idx])
        # Process heat
        heat = np.sum(output_all[scen]['BCHY'][:, idx, 10, 0] *
                          output_all[scen]['HYG1'][:, idx, 0, year_idx])
        # Electricity
        elec = np.sum(output_all[scen]['BCHY'][:, idx, 12, 0] *
                          output_all[scen]['HYG1'][:, idx, 0, year_idx])
        
        table_out.loc[rows[r], col] = (feedstock + heat + elec) * 1e3 / 11630
        
        r +=1
        if tech  ==  'ELEC-VRE':
            
            vre_generation = (output_all[scen]['BCHY'][:, -6:-3, 12, 0] *
                              output_all[scen]['HYG1'][:, -3:, 0, year_idx]) * 1e3 # in kWh
            vre_generation /= 11630 # in ktoe
            table_out.loc[rows[r], col] = vre_generation.sum()
            r += 1
        
    # table_out.loc[rows[4], col] = output_all[scen]['HYJF'][:, 6, 0, year_idx].sum()
    # table_out.loc[rows[5], col] = output_all[scen]['HYJF'][:, 0, 0, year_idx].sum()
    # table_out.loc[rows[6], col] = output_all[scen]['HYJF'][:, 7, 0, year_idx].sum()
    
    # vre_generation = (output_all[scen]['BCHY'][:, -6:-3, 12, 0] *
    #                   output_all[scen]['HYG1'][:, -3:, 0, year_idx]) * 1e3 # in kWh
    # vre_generation /= 11630 # in ktoe
    # table_out.loc[rows[7], col] = vre_generation.sum()
    
    # vre_capex_factor = (output_all[scen]['WSSH'][:, 0, 0, :]*output_all[scen]['WSIC'][:, 0, 0, :]+
    #                     output_all[scen]['WOSH'][:, 0, 0, :]*output_all[scen]['WOIC'][:, 0, 0, :]+
    #                     output_all[scen]['WWSH'][:, 0, 0, :]*output_all[scen]['WWIC'][:, 0, 0, :])
    
    # table_out.loc[rows[7], col] = np.sum(output_all[scen]['HYG1'][:, -3:, 0, year_idx]
    #                                      * vre_capex_factor[:, None, year_idx]
    #                                      )
    table_out.loc[rows[-2], col] = output_all[scen]['HYWE'][:, :, 0, year_idx].sum()
    table_out.loc[rows[-1], col] = output_all[scen]['HYWE'][:, :, 0, year_idx_ini:year_idx].sum()
table_out *= 1e-3
table_out.to_csv('summary_results_v{}.csv'.format(version))

# %% Table 2 -  cumulative investment (Brazil)

# FF H2
# FF + CCS H2 
# ELEC H2
# Dedicated power capacity

years = [2050]
cols = ["{} - {}".format(year, sn) for year in years for sn in scen_dict.values()]
rows = ['Cumulative investment in FF capacity (bEuro(2024))',
        'Cumulative investment in FF+CCS capacity (bEuro(2024))',
        'Cumulative investment in ELEC-grid capacity (bEuro(2024))',
        'Cumulative investment in ELEC-VRE capacity (bEuro(2024))',
        'Cumulative investment in dedicated VRE capacity (bEuro(2024))',
        'Energy (incl. feedstock) consumption by FF technologies (ktoe/y)',
        'Energy (incl. feedstock) consumption by FF+CCS technologies (ktoe/y)',
        'Energy consumption by ELEC-grid technologies (ktoe/y)',
        'Energy consumption by ELEC-VRE technologies (ktoe/y)',
        'Dedicated electricity generation at ELEC-VRE sites (ktoe/y)',
        'Annual emissions (Gt CO$_2$/y)',
        'Cumulative emissions (Gt CO$_2$)']
table_out = pd.DataFrame(0.0, index=rows, columns=cols)

scen_dict_reverse = dict(zip(scen_dict.values(), scen_dict.keys()))

year_idx_ini = tl.tolist().index(2025)

for col in cols:
    
    year = int(col.split(' - ')[0])
    scen_name = col.split(' - ')[1]
    
    year_idx = tl.tolist().index(year)
    scen = scen_dict_reverse[scen_name]
    
    r = 0
    
    for t, tech in enumerate(agg_techs2[:-1]):
    
        tech_titles = conv2.index[conv2[tech] == 1].tolist()
        idx = [conv2.index.get_loc(i) for i in tech_titles] 
        
        table_out.loc[rows[r], col] = np.sum(output_all[scen]['HYIY'][43, idx, 0, year_idx_ini:year_idx]* 
                                             inflator[43, None, None])
        

        r += 1
        if tech  ==  'ELEC-VRE':
            
            table_out.loc[rows[r], col]= (np.sum(output_all[scen]['HYIT'][43, idx, 0, year_idx_ini:year_idx]* 
                                                 inflator[43, None, None]) -
                                          np.sum(output_all[scen]['HYIY'][43, idx, 0, year_idx_ini:year_idx]* 
                                                 inflator[43, None, None]))
            r += 1
            
            
    # Energy use
    for t, tech in enumerate(agg_techs2[:-1]):
    
        tech_titles = conv2.index[conv2[tech] == 1].tolist()
        idx = [conv2.index.get_loc(i) for i in tech_titles] 
        
        # Feedstock
        feedstock = np.sum(output_all[scen]['BCHY'][43, idx, 8, 0] *
                          output_all[scen]['HYG1'][43, idx, 0, year_idx])
        # Process heat
        heat = np.sum(output_all[scen]['BCHY'][43, idx, 10, 0] *
                          output_all[scen]['HYG1'][43, idx, 0, year_idx])
        # Electricity
        elec = np.sum(output_all[scen]['BCHY'][43, idx, 12, 0] *
                          output_all[scen]['HYG1'][43, idx, 0, year_idx])
        
        table_out.loc[rows[r], col] = (feedstock + heat + elec) * 1e3 / 11630
        
        r +=1
        if tech  ==  'ELEC-VRE':
            
            vre_generation = (output_all[scen]['BCHY'][43, -6:-3, 12, 0] *
                              output_all[scen]['HYG1'][43, -3:, 0, year_idx]) * 1e3 # in kWh
            vre_generation /= 11630 # in ktoe
            table_out.loc[rows[r], col] = vre_generation.sum()
            r += 1
        
    # table_out.loc[rows[4], col] = output_all[scen]['HYJF'][:, 6, 0, year_idx].sum()
    # table_out.loc[rows[5], col] = output_all[scen]['HYJF'][:, 0, 0, year_idx].sum()
    # table_out.loc[rows[6], col] = output_all[scen]['HYJF'][:, 7, 0, year_idx].sum()
    
    # vre_generation = (output_all[scen]['BCHY'][:, -6:-3, 12, 0] *
    #                   output_all[scen]['HYG1'][:, -3:, 0, year_idx]) * 1e3 # in kWh
    # vre_generation /= 11630 # in ktoe
    # table_out.loc[rows[7], col] = vre_generation.sum()
    
    # vre_capex_factor = (output_all[scen]['WSSH'][:, 0, 0, :]*output_all[scen]['WSIC'][:, 0, 0, :]+
    #                     output_all[scen]['WOSH'][:, 0, 0, :]*output_all[scen]['WOIC'][:, 0, 0, :]+
    #                     output_all[scen]['WWSH'][:, 0, 0, :]*output_all[scen]['WWIC'][:, 0, 0, :])
    
    # table_out.loc[rows[7], col] = np.sum(output_all[scen]['HYG1'][:, -3:, 0, year_idx]
    #                                      * vre_capex_factor[:, None, year_idx]
    #                                      )
    table_out.loc[rows[-2], col] = output_all[scen]['HYWE'][43, :, 0, year_idx].sum()
    table_out.loc[rows[-1], col] = output_all[scen]['HYWE'][43, :, 0, year_idx_ini:year_idx].sum()
table_out *= 1e-3
table_out.to_csv('summary_results_brazil_v{}.csv'.format(version))
    
# %% Net trade Brazil

cols = ['Production', 'Demand', 'Net exports']
rows = scen_dict.keys()
trade_table = pd.DataFrame(0., index=rows, columns=cols)

for r, row in enumerate(rows):
    
    trade_table.iloc[r, 0] = output_all[row]['HYG1'][43, :, 0, year_idx].sum()
    trade_table.iloc[r, 1] = output_all[row]['HYDT'][43, :, 0, year_idx].sum()
    trade_table.iloc[r, 2] = trade_table.iloc[r, 0] - trade_table.iloc[r, 1]

trade_table *= 1e-3
             
# %% Graph 1 - Alternative

# fp = os.path.join('Graphs', 'LCOH_polbrief_full_detail_v{}.svg'.format(version))
# # Figure size
# figsize = (6.5, 6)
# # Create subplot    
# fig, axes = plt.subplots(nrows=4,
#                          ncols=4,
#                          figsize=figsize,
#                          sharey=True,
#                          sharex=True)

# # axes_flat = axes.flatten()

# var_weight = 'HYG1'
# var_lcoh = 'HYCC'
# reg_idx = 3

# for s, scen in enumerate(scen_dict.keys()):

#     for t, tech in enumerate(agg_techs2[:-1]):
        
#         tech_titles = conv2.index[conv2[tech] == 1].tolist()
#         idx = [conv2.index.get_loc(i) for i in tech_titles]
        
#         for it in idx:
            
#             for r, reg in enumerate(titles['RTI']):
                
#                 lcoh_ind = output_all[scen][var_lcoh][r, it, 0, :]
                
#                 df_lcoh = pd.Series(lcoh_ind, index=tl)
                
#                 if r == 43 and it == 9:
#                     axes[t,s].plot(np.asarray(tl_out),
#                                     df_lcoh[tl_out].values,
#                                     color='#AAB71D',
#                                     label='Green alkaline electrolysis in Brazil')                     
#                 elif r == 51 and it==1:
#                     axes[t,s].plot(np.asarray(tl_out),
#                                     df_lcoh[tl_out].values,
#                                     color='#009FE3',
#                                     label='SMR+CCS in OPEC')                     
#                 else:
#                     axes[t,s].plot(np.asarray(tl_out),
#                                     df_lcoh[tl_out].values,
#                                     color='gray',
#                                     alpha=0.1,
#                                     linewidth=0.8 ) 
                
#         # Weighted average across all options
#         weights = divide(output_all[scen][var_weight][:, idx, 0, :],
#                              output_all[scen][var_weight][:, idx, 0, :].sum(axis=0).sum(axis=0)[None, None, :])        
#         lcoh_avg_weighted = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] *inflator[:, None, None] * weights, axis=0).sum(axis=0)
#         df_lcoh_avg_weighted = pd.Series(lcoh_avg_weighted, index=tl)

#         axes[t,s].plot(np.asarray(tl_out),
#                         df_lcoh_avg_weighted[tl_out].values,
#                         color="#49C9C5",
#                         label='Weighted average',
#                         linestyle='dashed') 

#         # Unweighted average
#         lcoh_avg = np.mean(output_all[scen][var_lcoh][:, idx, 0, :], axis=(0,1))
#         df_lcoh_avg = pd.Series(lcoh_avg, index=tl)

        
#         axes[t,s].plot(np.asarray(tl_out),
#                         df_lcoh_avg[tl_out].values,
#                         color= "#C5446E",
#                         label='Unweighted average',
#                         linestyle='dashed') 
                
#         if t == 0:
        
#             axes[t,s].set_title(scen_dict[scen])
            
#         if s == 0:
            
#             axes[t,s].set_ylabel('{}\nEuro(2024)/kg H2'.format(tech))
            
#         axes[t,s].set_xlim([tl_out[0], tl_out[-1]]);
#         axes[t,s].grid(alpha=0.4, color="#E3E3E3");
#         axes[t,s].tick_params('x', labelrotation=60)        
      

# h1, l1 = axes[0,0].get_legend_handles_labels()
# h2, l2 = axes[1,0].get_legend_handles_labels()
# h3, l3 = axes[2,0].get_legend_handles_labels()

# fig.legend(handles=h1,
#            labels=l1,
#            loc="lower center",
#            bbox_to_anchor=(0.25,0.03),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=1,
#            title="Global LCOH$_2$",
#            fontsize=8)

# fig.legend(handles=[h2[0], h3[0]],
#            labels=[l2[0], l3[0]],
#            loc="lower center",
#            bbox_to_anchor=(0.7,0.03),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=1,
#            title="Highlighted LCOH$_2$",
#            fontsize=8)

# fig.subplots_adjust(hspace=0.0, wspace=0.0, right=0.97, bottom=0.2, left=0.1, top=0.95)

# fig.savefig(fp)
# plt.show()           
        
# %% Graph - display individual green hydrogen technologies

# agg_tech = 'ELEC-VRE'
# tech_titles = conv2.index[conv2[agg_tech] == 1].tolist()
# idx = [conv2.index.get_loc(i) for i in tech_titles]
# idx = [8,9,10]

# fp = os.path.join('Graphs', 'LCOH_range_v{}.svg'.format(version))
# # Figure size
# figsize = (6.5, 6)
# # Create subplot    
# fig, axes = plt.subplots(nrows=3,
#                          ncols=4,
#                          figsize=figsize,
#                          sharey=True,
#                          sharex=True)

# reg_colors = ['#AAB71D', '#49C9C5', '#C5446E', '#009FE3', 'purple', 'red']

# for i, t in enumerate(idx):

#     for s, scen in enumerate(scen_dict.keys()):
        
#         # Only select instances with positive production
#         cond = output_all[scen]['HYG1'][:, t, 0, :]>0.0
        
#         # Global Average        
#         lcoh_avg = np.mean(output_all[scen][var_lcoh][:, t, 0, :]*
#                                inflator[:, None], 
#                            axis=0, 
#                            where=cond)
#         df_lcoh_avg = pd.Series(lcoh_avg, index=tl)    
        
#         axes[i,s].plot(np.asarray(tl_out),
#                         df_lcoh_avg[tl_out].values,
#                         color= "black",
#                         label='Global average',
#                         linestyle='dashed')
        
#         # Maximum value
#         lcoh_max = np.max(output_all[scen][var_lcoh][:, t, 0, :]*
#                               inflator[:, None], 
#                           axis=0)
#         df_lcoh_max = pd.Series(lcoh_max, index=tl)    

#         # Minimum value
#         lcoh_min = np.min(output_all[scen][var_lcoh][:, t, 0, :]*
#                               inflator[:, None],
#                           axis=0)
#         df_lcoh_min = pd.Series(lcoh_min, index=tl)    

#         # Shaded area representing variability
#         axes[i,s].fill_between(np.asarray(tl_out),
#                                df_lcoh_min[tl_out].values,
#                                df_lcoh_max[tl_out].values,
#                                color='gray',
#                                alpha=0.5)
        
#         # Special cases here
#         # Brazil, Australia, China, Germany, US, Japan
#         for j, r in enumerate([43, 36, 40, 2, 33, 34]):
            
#             reg = ' '.join(titles['RTI'][r].split(' ')[1:-1])
#             df_lcoh_r = pd.Series(output_all[scen][var_lcoh][r, t, 0, :]*inflator[r, None], index=tl)
            
#             axes[i,s].plot(np.asarray(tl_out),
#                             df_lcoh_r[tl_out].values,
#                             color= reg_colors[j],
#                             label=reg)            
        
#         if i == 0:
        
#             axes[i,s].set_title(scen_dict[scen])
            
#         if s == 0:
            
#             axes[i,s].set_ylabel('{}\nEuro(2024)/kg H$_2$'.format(titles['HYTI'][t].split(' ')[1]))
            
#         axes[i,s].set_xlim([tl_out[0], tl_out[-1]]);
#         axes[i,s].grid(alpha=0.4, color="#E3E3E3");
#         axes[i,s].tick_params('x', labelrotation=60)          
        
# h1, l1 = axes[0,0].get_legend_handles_labels()
# # h2, l2 = axes[1,0].get_legend_handles_labels()
# # h3, l3 = axes[2,0].get_legend_handles_labels()

# fig.legend(handles=h1,
#            labels=l1,
#            loc="lower center",
#            bbox_to_anchor=(0.5,0.03),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=4,
#            title="LCOH$_2$",
#            fontsize=8)    

# fig.subplots_adjust(hspace=0.0, wspace=0.0, right=0.97, bottom=0.2, left=0.1, top=0.95)

# fig.savefig(fp)
# plt.show()       

# %%
tl_out = np.arange(2023, 2050+1)
agg_tech = 'ELEC'
tech_titles = conv.index[conv[agg_tech] == 1].tolist()
idx = [conv.index.get_loc(i) for i in tech_titles]
# idx = [8,9,10]

fp = os.path.join('Graphs', 'LCOH_range_single_scen_v{}.svg'.format(version))
# Figure size
figsize = (5, 6)
# Create subplot    
fig, axes = plt.subplots(nrows=3,
                         ncols=2,
                         figsize=figsize,
                         sharey=True,
                         sharex=True)

scen = 'S3'

axes_flat = axes.T.flatten()

reg_colors = ['#AAB71D', '#49C9C5', '#C5446E', '#009FE3', 'purple', 'red']

for i, t in enumerate(idx):

    # Only select instances with positive production
    cond = output_all[scen]['HYG1'][:, t, 0, :]>0.0
        
    # Global Average        
    lcoh_avg = np.mean(output_all[scen][var_lcoh][:, t, 0, :]*
                           inflator[:, None]
                       , axis=0, 
                       where=cond)
    df_lcoh_avg = pd.Series(lcoh_avg, index=tl)    
    
    axes_flat[i].plot(np.asarray(tl_out),
                    df_lcoh_avg[tl_out].values,
                    color= "black",
                    label='Global average electrolytic',
                    linestyle='dashed')
    
    # Global Average        
    lcoh_avg_smr = np.mean(output_all[scen][var_lcoh][:, 0, 0, :]*
                           inflator[:, None]
                       , axis=0, 
                       where=cond)
    df_lcoh_avg_smr = pd.Series(lcoh_avg_smr, index=tl)    
    
    axes_flat[i].plot(np.asarray(tl_out),
                    df_lcoh_avg_smr[tl_out].values,
                    color= "black",
                    label='Global average SMR',
                    linestyle='dotted')
    
    # Maximum value
    lcoh_max = np.max(output_all[scen][var_lcoh][:, t, 0, :]*
                          inflator[:, None]
                      , axis=0)
    df_lcoh_max = pd.Series(lcoh_max, index=tl)    

    # Minimum value
    lcoh_min = np.min(output_all[scen][var_lcoh][:, t, 0, :]*
                          inflator[:, None]
                      , axis=0)
    df_lcoh_min = pd.Series(lcoh_min, index=tl)    

    # Shaded area representing variability
    axes_flat[i].fill_between(np.asarray(tl_out),
                           df_lcoh_min[tl_out].values,
                           df_lcoh_max[tl_out].values,
                           color='gray',
                           alpha=0.5)
    
    # Special cases here
    # Brazil, Australia, China, Germany, US, Japan
    for j, r in enumerate([43, 36, 40, 2, 33, 47]):
        
        reg = ' '.join(titles['RTI'][r].split(' ')[1:-1])
        df_lcoh_r = pd.Series(output_all[scen][var_lcoh][r, t, 0, :]*inflator[r, None], index=tl)
        
        axes_flat[i].plot(np.asarray(tl_out),
                        df_lcoh_r[tl_out].values,
                        color= reg_colors[j],
                        label=reg)            
    
    if 'green' in titles['HYTI'][t]:
        axes_flat[i].set_title(titles['HYTI'][t].split(' ')[1].split('-')[0] + ('-VRE'))
    else:
        axes_flat[i].set_title(titles['HYTI'][t].split(' ')[1])
        
        
    axes_flat[i].set_ylabel('Euro(2024)/kg H$_2$')
        
    axes_flat[i].set_xlim([tl_out[0], tl_out[-1]]);
    axes_flat[i].grid(alpha=0.4, color="#E3E3E3");
    axes_flat[i].tick_params('x', labelrotation=60)
    axes_flat[i].label_outer()          
        
h1, l1 = axes_flat[0].get_legend_handles_labels()
# h2, l2 = axes[1,0].get_legend_handles_labels()
# h3, l3 = axes[2,0].get_legend_handles_labels()

fig.legend(handles=h1,
           labels=l1,
           loc="lower center",
           bbox_to_anchor=(0.5,0.03),
           frameon=False,
           borderaxespad=0.,
           ncol=4,
           title="Regional levelised cost\nof hydrogen production",
           fontsize=8)    

fig.subplots_adjust(hspace=0.2, wspace=0.1, right=0.97, bottom=0.22, left=0.1, top=0.95)

fig.savefig(fp)
plt.show()

# %% Column chart of green hydrogen demand

fp = os.path.join('Graphs', 'Green_H2_demand_v{}.svg'.format(version))
# Figure size
figsize = (5, 3)
# Create subplot    
fig, axes = plt.subplots(nrows=1,
                         ncols=2,
                         figsize=figsize,
                         sharey='row',
                         sharex=True)

years = [2035, 2050]

for y, year in enumerate(years):
    
    idx = tl.tolist().index(year)
    
    green_h2_demand = []
    
    for scen in scen_dict.keys():
        
        green_h2_demand.append(output_all[scen]['WGRM'][:, 0, 0, idx].sum()*1e-3)
        
    green_h2_demand.append(output_all[scen]['HYDT'][:, 0, 0, idx].sum()*1e-3)
    
    bars = axes[y].bar(list(scen_dict.values())+['Total'], green_h2_demand, color='#AAB71D')
    bars[-1].set_color('black')
    
    for b, bar in enumerate(bars):
        
        if b != len(bars)-1:
            yval = np.round(bar.get_height(), decimals=1)
            axes[y].text(bar.get_x() + bar.get_width()/2, yval , f'{yval}', ha='center', va='bottom', fontsize=9)
        else:
            yval = np.round(bar.get_height(), decimals=1)
            axes[y].text(bar.get_x()-0.5, yval * 0.9, f'{yval}', ha='center', va='bottom', fontsize=9)
            
    # Set labels and title
    # axes[y].set_xlabel('Scenario')
    axes[y].set_ylabel('Green hydrogen market\nMt H$_2$')
    axes[y].set_title(year) 
    axes[y].label_outer() 
    
fig.subplots_adjust(hspace=0.2, wspace=0., right=0.97, bottom=0.2, left=0.13, top=0.9)

fig.savefig(fp)
plt.show()      

# %% Brazil vs global bar chart

year = 2050
y = tl.tolist().index(year)

fp = os.path.join('Graphs', 'Brazil_vs_world_v{}.svg'.format(version))
# Figure size
figsize = (5, 6)
# Create subplot    
fig, axes = plt.subplots(nrows=4,
                         ncols=2,
                         figsize=figsize,
                         sharey=True,
                         sharex=True)

# Secondary axis for LCOH 
# axes2 = axes.twinx()

axes[0, 0].set_title('Brazil')
axes[0, 1].set_title('Global')

for s, scen in enumerate(scen_dict.keys()):
    
    prod_ls_brazil = []
    prod_ls_glo = []
    
    lcoh_ls_brazil = []
    lcoh_ls_glo = []
    
    for t, tech in enumerate(agg_techs2[:-1]):
    
        tech_titles = conv2.index[conv2[tech] == 1].tolist()
        idx = [conv2.index.get_loc(i) for i in tech_titles]

        prod_brazil = output_all[scen][var_prod][43, idx, 0, y].sum()
        prod_brazil = prod_brazil / (output_all[scen][var_prod][43, :, 0, y].sum())
        prod_ls_brazil.append(prod_brazil*100)
        
        prod_glo = output_all[scen][var_prod][:, idx, 0, y].sum()
        prod_glo = prod_glo / (output_all[scen][var_prod][:, :, 0, y].sum())
        prod_ls_glo.append(prod_glo*100)   
        
        if tech == 'ELEC':
            idx = [8, 9, 10]
            
        lcoh_brazil = np.mean(output_all[scen]['HYCC'][43, idx, 0, y] * inflator[43, None, None],
                                 where=output_all[scen][var_prod][43, idx, 0, y]>0.0)
        lcoh_ls_brazil.append(lcoh_brazil)
        
        lcoh_glo = np.mean(output_all[scen]['HYCC'][:, idx, 0, y] * inflator[:, None, None],
                                 where=output_all[scen][var_prod][:, idx, 0, y]>0.0)
        lcoh_ls_glo.append(lcoh_glo)
        
    bars_brazil = axes[s, 0].bar(agg_techs2[:-1], prod_ls_brazil, color='#AAB71D')
    bars_glo = axes[s, 1].bar(agg_techs2[:-1], prod_ls_glo, color='#AAB71D')
    
    ax2_brazil = axes[s, 0].twinx()
    ax2_glo = axes[s, 1].twinx()
    
    ax2_brazil.scatter(np.arange(len(agg_techs2[:-1])),
                        lcoh_ls_brazil,
                        color='purple')
    ax2_glo.scatter(np.arange(len(agg_techs2[:-1])),
                        lcoh_ls_glo,
                        color='purple')
    
    ax2_brazil.tick_params(axis='y', labelcolor='purple')
    ax2_glo.tick_params(axis='y', labelcolor='purple')
    
    for b in range(len(bars_brazil)):
        
        bars_brazil[b].set_color(tech_colors2[b])
        bars_glo[b].set_color(tech_colors2[b])
        
    # Formatting
    axes[s, 0].set_ylabel('{}\n% of prod.'.format(scen_dict[scen]))
    ax2_glo.set_ylabel('Euro(2024)/kg H$_2$', rotation=270,labelpad=20, color='purple')
    ax2_brazil.set_ylim(0, 8.5)
    ax2_glo.set_ylim(0, 8.5)
    axes[s, 0].label_outer() 
    axes[s, 1].label_outer()
    axes[s, 0].set_xticklabels(agg_techs2[:-1], rotation=90)
    axes[s, 1].set_xticklabels(agg_techs2[:-1], rotation=90)
    
    ax2_brazil.tick_params(right=False)
    ax2_brazil.set_yticks([])
        
fig.subplots_adjust(hspace=0.2, wspace=0.1, right=0.85, bottom=0.1, left=0.13, top=0.9)



fig.savefig(fp)
plt.show()     
        
        
# %% Capacity growth rates

# scen = 'S3'
# tech_agg = 'ELEC-VRE'
# tl_out = np.arange(2030, 2050+1)

# tech_titles = conv2.index[conv2[tech_agg] == 1].tolist()
# idx = [conv2.index.get_loc(i) for i in tech_titles]

# fp = os.path.join('Graphs', 'Growth_rates_v{}.svg'.format(version))
# # Figure size
# figsize = (4, 3.5)
# # Create subplot    
# fig, axes = plt.subplots(nrows=1,
#                          ncols=1,
#                          figsize=figsize,
#                          sharey=True,
#                          sharex=True)

# for r, reg in enumerate(titles['RTI']):
    
#     for i, t in enumerate(idx): 
        
#         prod = output_all[scen][var_prod][r, t, 0, :]
#         df_prod = pd.Series(prod, index=tl)
#         df_prod_yoy = df_prod.pct_change() * 100
#         df_prod_yoy[df_prod_yoy<-40] = -40
        
#         if r != 43:
            
#             axes.plot(np.asarray(tl_out),
#                       df_prod_yoy[tl_out],
#                       color='gray',
#                       alpha=0.2)
        
#         else:
            
#             axes.plot(np.asarray(tl_out),
#                       df_prod_yoy[tl_out],
#                       color=colors[i],
#                       label=tech_titles[i]) 
            
#     # Set labels and title
#     # axes[y].set_xlabel('Scenario')
#     axes.set_ylabel('Year-on-year growth rate\n%')
#     # axes[y].set_title() 
#     # axes.label_outer() 

# axes.fill_between(np.asarray(tl_out),
#                        20,
#                        40,
#                        color='goldenrod',
#                        alpha=0.3,
#                        label='Solar PV (range)')

# axes.set_xlim([tl_out[0], tl_out[-1]]);
# axes.grid(alpha=0.4, color="#E3E3E3");
# axes.tick_params('x', labelrotation=60)
# axes.set_xticks([2030, 2035, 2040, 2045, 2050])
    
# fig.subplots_adjust(hspace=0.2, wspace=0., right=0.97, bottom=0.3, left=0.2, top=0.9)
# h1, l1 = axes.get_legend_handles_labels()
# # h2, l2 = axes[1,0].get_legend_handles_labels()
# # h3, l3 = axes[2,0].get_legend_handles_labels()

# fig.legend(handles=h1,
#            labels=l1,
#            loc="lower center",
#            bbox_to_anchor=(0.5,0.03),
#            frameon=False,
#            borderaxespad=0.,
#            ncol=2,
#            title="Brazil",
#            fontsize=8)

# fig.savefig(fp)
# plt.show()               

# %%
tl_out = np.arange(2025, 2050+1)
fp = os.path.join('Graphs', 'Production_polbrief_by_scen_v{}.svg'.format(version))
# Figure size
figsize = (7.5,3.3)
# Create subplot    
fig, axes = plt.subplots(nrows=1,
                         ncols=4,
                         figsize=figsize,
                         sharey='row',
                         sharex=True)

# axes = axes.flatten()

var_weight = 'HYG1'
var_lcoh = 'HYCC'
var_prod = 'HYG1'
reg_idx = 3

for s, scen in enumerate(scen_dict.keys()):
    
    # for t, tech in enumerate(agg_techs2[:-1]):
    
    #     tech_titles = conv2.index[conv2[tech] == 1].tolist()
    #     idx = [conv2.index.get_loc(i) for i in tech_titles]
    

        
    #     # Global weighted average by individual technology
    #     lcoh_weight = divide(output_all[scen][var_weight][:, idx, 0, :],
    #                          output_all[scen][var_weight][:, idx, 0, :].sum(axis=0).sum(axis=0)[None, None, :])
        
    #     lcoh_avg = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] * inflator[:, None, None] * lcoh_weight, axis=0).sum(axis=0)
    #     lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :] * inflator[:, None, None] , axis=(0,1))

    #     # lcoh_avg = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] * lcoh_weight, axis=0).sum(axis=0)
    #     # lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :] , axis=(0,1))

    #     df_lcoh_avg = pd.Series(lcoh_avg2, index=tl)
        
    #     axes[0,s].plot(np.asarray(tl_out),
    #                     df_lcoh_avg[tl_out].values,
    #                     label=tech,
    #                     color=tech_colors2[t]) 
        
    #     axes[0,s].set_title(scen_dict[scen])
        
    # axes[0,s].set_xlim([tl_out[0], tl_out[-1]]);
    # axes[0,s].set_ylim([0, 7])
    # axes[0,s].grid(alpha=0.4, color="#E3E3E3");
    # axes[0,s].tick_params('x', labelrotation=60)
    # axes[0,s].set_ylabel('Levelised cost\nEuro(2024)/kg H$_2$')  
    # axes[0,s].label_outer()
    # # axes[0,s].set_xticks([2025, 2030, 2040, 2050]) 
    
    # # for s, scen in enumerate(scen_dict.keys()):
        
    prod = output_all[scen][var_prod][:, idx, 0, :].sum(axis=0).sum(axis=0) * 1e-3
    prod = np.matmul(conv2.iloc[:, :-1].T, output_all[scen][var_prod].sum(axis=0)[:, 0, :]) * 1e-3
    df_prod = pd.DataFrame(prod.values, index=agg_techs2[:-1], columns=tl)
    
    axes[s].stackplot(np.asarray(tl_out),
                    df_prod[tl_out].values,
                    labels=agg_techs2[:-1],
                    colors=tech_colors2) 
        
        
    axes[s].set_xlim([tl_out[0], tl_out[-1]]);
    axes[s].grid(alpha=0.4, color="#E3E3E3");
    axes[s].tick_params('x', labelrotation=60)
    
    # axes[1,s].set_xticks([2025, 2030, 2040, 2050])   
    axes[s].set_ylabel('Production\nMt H$_2$')   
    axes[s].label_outer()
    axes[s].set_title(scen_dict[scen])
      

h1, l1 = axes[0].get_legend_handles_labels()

fig.legend(handles=h1,
           labels=l1,
           loc="lower center",
           bbox_to_anchor=(0.5,0.05),
           frameon=False,
           borderaxespad=0.,
           ncol=4,
           title="Technologies",
           fontsize=8)

fig.subplots_adjust(hspace=0.2, wspace=0.1, right=0.97, bottom=0.3, left=0.1, top=0.85)

fig.savefig(fp)
plt.show()
            
# %% LCOH
tl_out = np.arange(2025, 2050+1)
fp = os.path.join('Graphs', 'LCOH_polbrief_by_scen_v{}.svg'.format(version))
# Figure size
figsize = (7.5,3.3)
# Create subplot    
fig, axes = plt.subplots(nrows=1,
                         ncols=4,
                         figsize=figsize,
                         sharey='row',
                         sharex=True)

# axes = axes.flatten()

var_weight = 'HYG1'
var_lcoh = 'HYCC'
var_prod = 'HYG1'
reg_idx = 3

for s, scen in enumerate(scen_dict.keys()):
    
     for t, tech in enumerate(agg_techs2[:-1]):
    
         tech_titles = conv2.index[conv2[tech] == 1].tolist()
         idx = [conv2.index.get_loc(i) for i in tech_titles]
    

        
         # Global weighted average by individual technology
         lcoh_weight = divide(output_all[scen][var_weight][:, idx, 0, :],
                              output_all[scen][var_weight][:, idx, 0, :].sum(axis=0).sum(axis=0)[None, None, :])
        
         lcoh_avg = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] * inflator[:, None, None] * lcoh_weight, axis=0).sum(axis=0)
         lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :] * inflator[:, None, None] , axis=(0,1))

         # lcoh_avg = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] * lcoh_weight, axis=0).sum(axis=0)
         # lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :] , axis=(0,1))

         df_lcoh_avg = pd.Series(lcoh_avg2, index=tl)
        
         axes[s].plot(np.asarray(tl_out),
                         df_lcoh_avg[tl_out].values,
                         label=tech,
                         color=tech_colors2[t]) 
        
         axes[s].set_title(scen_dict[scen])
        
     axes[s].set_xlim([tl_out[0], tl_out[-1]])
     axes[s].set_ylim([0, 7])
     axes[s].grid(alpha=0.4, color="#E3E3E3");
     axes[s].tick_params('x', labelrotation=60)
     axes[s].set_ylabel('Levelised cost\nEuro(2024)/kg H$_2$')  
     axes[s].label_outer()

axes[s].set_xlim([tl_out[0], tl_out[-1]]);
axes[s].grid(alpha=0.4, color="#E3E3E3");
axes[s].tick_params('x', labelrotation=60)
    
# axes[1,s].set_xticks([2025, 2030, 2040, 2050])   
axes[s].set_ylabel('Production\nMt H$_2$')   
axes[s].label_outer()
axes[s].set_title(scen_dict[scen])
      

h1, l1 = axes[0].get_legend_handles_labels()

fig.legend(handles=h1,
           labels=l1,
           loc="lower center",
           bbox_to_anchor=(0.5,0.05),
           frameon=False,
           borderaxespad=0.,
           ncol=4,
           title="Technologies",
           fontsize=8)

fig.subplots_adjust(hspace=0.2, wspace=0.1, right=0.97, bottom=0.3, left=0.1, top=0.85)

fig.savefig(fp)
plt.show()     
     

        



    
    

