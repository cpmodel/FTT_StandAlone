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
model.scenarios = ['S{}'.format(i) for i in [0,13]]

scen_dict = dict(zip(model.scenarios, ['REF', 'HighMan_HighCP']))

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
# %%
# Call the 'run' method of the ModelRun class to solve the model
model.run()

# Fetch ModelRun attributes, for examination
# Output of the model
output_all = model.output

#

# %% Check outputs

hywk = {}
hyg1 = {}
hywk_glo = pd.DataFrame(0.0, index=titles['HYTI'], columns=tl)
hyg1_glo = pd.DataFrame(0.0, index=titles['HYTI'], columns=tl)


for r, reg in enumerate(titles['RTI']):
    
    hywk[reg] = pd.DataFrame(output_all['S0']['HYWK'][r, :, 0, :], index=titles['HYTI'], columns=tl)
    hyg1[reg] = pd.DataFrame(output_all['S0']['HYG1'][r, :, 0, :], index=titles['HYTI'], columns=tl)
    
    hywk_glo += hywk[reg]
    hyg1_glo += hyg1[reg]
    
hykf_glo = pd.DataFrame(output_all['S0']['HYKF'][:, 0, 0, :], index=titles['RTI'], columns=tl)


aaa = pd.DataFrame(output_all['S0']['FERTD'].sum(axis=0)[:, 0, :], index=titles['TFTI'], columns=tl)
bbb = pd.DataFrame(output_all['S0']['FERTS'].sum(axis=0)[:, 0, :], index=titles['TFTI'], columns=tl)    
ccc = aaa.div(aaa.sum(axis=0),axis=1) 

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

tl_out = np.arange(2021, 2051)
colors = ["#AAB71D", "#49C9C5", "#C5446E", "#009FE3"]

# %% LCOH - global avg

for scen, name in scen_dict.items():
    
    fp = os.path.join('Graphs', 'LCOH_{}.svg'.format(name))
    # Figure size
    figsize = (5, 5.5)
    # Create subplot    
    fig, axes = plt.subplots(nrows=1,
                             ncols=1,
                             figsize=figsize)
    
    lcoh = np.sum(output_all[scen]['HYCC'] * output_all[scen]['HYG1'], axis=0)
    lcoh = divide(lcoh, output_all[scen]['HYG1'].sum(axis=0))
    
    df_lcoh = pd.DataFrame(lcoh[:,0,:], index=titles['HYTI'], columns=tl)
    
    for t, tech in enumerate(techs_to_show):
        axes.plot(np.asarray(tl_out),
                        df_lcoh.loc[tech, tl_out].values,
                        label=tech,
                        color=colors[t])

    axes.set_xlim([tl_out[0], tl_out[-1]]);
    axes.grid(alpha=0.4, color="#E3E3E3");
    axes.tick_params('x', labelrotation=60)
    # axes[s, 0].label_outer()
    axes.set_xticks([2023, 2030, 2040, 2050])   
    axes.set_ylabel('Euro/kg H2')

    axes.set_title(name)
    
    
    h1, l1 = axes.get_legend_handles_labels()

    
    fig.legend(handles=h1,
               labels=l1,
               loc="lower center",
               bbox_to_anchor=(0.5,0.04),
               frameon=False,
               borderaxespad=0.,
               ncol=2,
               title="Technologies",
               fontsize=8)

    fig.subplots_adjust(hspace=0.2, wspace=0.2, right=0.97, bottom=0.25, left=0.15, top=0.95)

    fig.savefig(fp)
    plt.show()


# %% Demand vectors

vectors =['NH3 for fertiliser', 'NH3 for chemicals', 'MeOH for chemicals', ' H2 for oil refining']
colmap = dict(zip(vectors, colors))


for scen, name in scen_dict.items():
    
    fp = os.path.join('Graphs', 'Demand_{}.svg'.format(name))
    # Figure size
    figsize = (5, 5.5)
    # Create subplot    
    fig, axes = plt.subplots(nrows=1,
                             ncols=1,
                             figsize=figsize)
    
    df_demand_vectors = pd.DataFrame(0, index=vectors, columns=tl)
    df_demand_vectors.iloc[0, :] = output_all[scen]['HYD1'][:,0,0,:].sum(axis=0)
    df_demand_vectors.iloc[1, :] = output_all[scen]['HYD2'][:,0,0,:].sum(axis=0)
    df_demand_vectors.iloc[2, :] = output_all[scen]['HYD3'][:,0,0,:].sum(axis=0)
    df_demand_vectors.iloc[3, :] = output_all[scen]['HYD4'][:,0,0,:].sum(axis=0)
    
    green_demand = pd.Series(output_all[scen]['WGRM'][:,0,0,:].sum(axis=0), index=tl)
    
    axes.stackplot(np.asarray(tl_out),
                    df_demand_vectors.loc[:, tl_out].values*1e-3,
                    labels=colmap.keys(),
                    colors=colmap.values())
    
    axes.plot(np.asarray(tl_out),
              green_demand.loc[tl_out].values*1e-3,
              label='Green fertiliser demand',
              color='black',
              linestyle=':')
    
    axes.set_xlim([tl_out[0], tl_out[-1]]);
    axes.grid(alpha=0.4, color="#E3E3E3");
    axes.tick_params('x', labelrotation=60)
    # axes[s, 0].label_outer()
    axes.set_xticks([2023, 2030, 2040, 2050])   
    axes.set_ylabel('Mt H2')

    axes.set_title(name)
    
    
    h1, l1 = axes.get_legend_handles_labels()

    
    fig.legend(handles=h1,
               labels=l1,
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

# %% Production 

agg_techs = ['SMR', 'Gasification', 'Pyrolysis', 'PEM', 'ALK', 'SOEC']
conv = pd.DataFrame(0.0, index=titles['HYTI'], columns=agg_techs)
conv.iloc[[0,1], 0] = 1
conv.iloc[[2,3], 1] = 1
conv.iloc[4,2] = 1
conv.iloc[[5,8], 3] = 1
conv.iloc[[6,9], 4] = 1
conv.iloc[[7,10], 5] = 1

colmap = dict(zip(agg_techs, ['grey', 'black', 'turquoise',"#AAB71D", "#49C9C5", "#C5446E"]))


for scen, name in scen_dict.items():
    
    fp = os.path.join('Graphs', 'Production_{}.svg'.format(name))
    # Figure size
    figsize = (5, 5.5)
    # Create subplot    
    fig, axes = plt.subplots(nrows=1,
                             ncols=1,
                             figsize=figsize)
    
    df_production = pd.DataFrame(output_all[scen]['HYG1'][:,:,0,:].sum(axis=0), index=titles['HYTI'], columns=tl)
    df_production_conv = conv.T.dot(df_production)
    
    green_demand = pd.Series(output_all[scen]['WGRM'][:,0,0,:].sum(axis=0), index=tl)
    
    axes.stackplot(np.asarray(tl_out),
                    df_production_conv.loc[:, tl_out].values*1e-3,
                    labels=colmap.keys(),
                    colors=colmap.values())
    
    axes.plot(np.asarray(tl_out),
              green_demand.loc[tl_out].values*1e-3,
              label='Green fertiliser demand',
              color='black',
              linestyle=':')
    
    axes.set_xlim([tl_out[0], tl_out[-1]]);
    axes.grid(alpha=0.4, color="#E3E3E3");
    axes.tick_params('x', labelrotation=60)
    # axes[s, 0].label_outer()
    axes.set_xticks([2023, 2030, 2040, 2050])   
    axes.set_ylabel('Mt H2')

    axes.set_title(name)
    
    
    h1, l1 = axes.get_legend_handles_labels()

    
    fig.legend(handles=h1,
               labels=l1,
               loc="lower center",
               bbox_to_anchor=(0.5,0.04),
               frameon=False,
               borderaxespad=0.,
               ncol=2,
               title="technologies",
               fontsize=8)

    fig.subplots_adjust(hspace=0.2, wspace=0.2, right=0.97, bottom=0.25, left=0.15, top=0.95)

    fig.savefig(fp)
    plt.show()


# %% Emissions
    
    

