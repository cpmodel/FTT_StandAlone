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
from SourceCode.support.divide import divide
import numpy as np
import pandas as pd
import os


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
        

# %% Setup converters and colour maps

y_2024 = tl.tolist().index(2024)
inflator = output_all['S0']['PRSC'][:, 0, 0, y_2024] / output_all['S0']['EX'][:, 0, 0, y_2024]
# Set inflator to average EU price levels (~Eurozone)
inflator = inflator* 0 + 1.3
version = 10

dem_vectors =['NH3 for fertiliser', 'NH3 for chemicals', 'MeOH for chemicals', ' H2 for oil refining']

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

var_lcoh = 'HYCC'
var_prod = 'HYG1'
var_weight = 'HYG1'


# %% Graph 2 - Demand projections

scen='S0'
tl_out = np.arange(2025, 2051)


df_demand_vectors = pd.DataFrame(0, index=dem_vectors, columns=tl)
df_demand_vectors.iloc[0, :] = output_all[scen]['HYD1'][:,0,0,:].sum(axis=0)
df_demand_vectors.iloc[1, :] = output_all[scen]['HYD2'][:,0,0,:].sum(axis=0)
df_demand_vectors.iloc[2, :] = output_all[scen]['HYD3'][:,0,0,:].sum(axis=0)
df_demand_vectors.iloc[3, :] = output_all[scen]['HYD4'][:,0,0,:].sum(axis=0)


df_demand_wide = df_demand_vectors.loc[:, tl_out]*1e-3
df_demand_long = df_demand_wide.reset_index().melt(id_vars='index', var_name='Year', value_name='Value')# tolong
df_demand_long.rename(columns={'index': 'Demand factor'}, inplace=True)
df_demand_long['Unit'] = 'Mt H2-equivalent'

df_demand_long.to_csv(os.path.join('Graphs', 'Demand_v{}.csv'.format(version)), index=False)
        
# %% LCOH_range_single_scen

tl_out = np.arange(2023, 2050+1)
agg_tech = 'ELEC'
tech_titles = conv.index[conv[agg_tech] == 1].tolist()
idx = [conv.index.get_loc(i) for i in tech_titles]

scen = 'S3'
#scen = 'S0'

df_lcoh_long = pd.DataFrame(columns = ['Technology', 'Regional aggregation', 'Year', 'Value', 'Unit'])

for i, t in enumerate(idx):

    if 'green' in titles['HYTI'][t]:
        techlbl = titles['HYTI'][t].split(' ')[1].split('-')[0] + ('-VRE')
    else:
        techlbl = titles['HYTI'][t].split(' ')[1]

    # Only select instances with positive production
    cond = output_all[scen]['HYG1'][:, t, 0, :]>0.0
        
    # Global Average electrolytic
    lcoh_avg = np.mean(output_all[scen][var_lcoh][:, t, 0, :]*
                           inflator[:, None]
                       , axis=0)
    
    df_lcoh_avg = pd.Series(lcoh_avg, index=tl).fillna(0)[tl_out]
    
    for year, val in df_lcoh_avg.items():
        df_lcoh_long = pd.concat([df_lcoh_long, 
                                 pd.DataFrame({'Technology': ['Global average electrolytic '+techlbl], 
                                               'Regional aggregation': ['Global'],
                                               'Year': [year], 'Value': [val], 'Unit': ['Euro(2024)/kg H2']})], ignore_index = True)
    # Global Average SMR
    lcoh_avg_smr = np.mean(output_all[scen][var_lcoh][:, [0,2], 0, :]*
                           inflator[:, None, None]
                       , axis=(0,1))
    df_lcoh_avg_smr = pd.Series(lcoh_avg_smr, index=tl)[tl_out]
    for year, val in df_lcoh_avg_smr.items():
        df_lcoh_long = pd.concat([df_lcoh_long,
                                  pd.DataFrame({'Technology': ['Global average FF'], 
                                                'Regional aggregation': ['Global'],
                                                'Year': [year], 'Value': [val], 'Unit': ['Euro(2024)/kg H2']})], ignore_index = True)

    # Global maximum value
    lcoh_max = np.max(output_all[scen][var_lcoh][:, t, 0, :]*
                          inflator[:, None]
                      , axis=0)
    df_lcoh_max = pd.Series(lcoh_max, index=tl)[tl_out]
    for year, val in df_lcoh_max.items():
        df_lcoh_long = pd.concat([df_lcoh_long,
                                  pd.DataFrame({'Technology': [techlbl + ' maximum LCOH'], 
                                                'Regional aggregation': ['Global'],
                                                'Year': [year], 'Value': [val], 'Unit': ['Euro(2024)/kg H2']})], ignore_index = True)

    # Global minimum value
    lcoh_min = np.min(output_all[scen][var_lcoh][:, t, 0, :]*
                          inflator[:, None]
                      , axis=0)
    df_lcoh_min = pd.Series(lcoh_min, index=tl)[tl_out]
    for year, val in df_lcoh_min.items():
        df_lcoh_long = pd.concat([df_lcoh_long,
                                  pd.DataFrame({'Technology': [techlbl + ' minimum LCOH'], 
                                            'Regional aggregation': 'Global',
                                                'Year': [year], 'Value': [val], 'Unit': ['Euro(2024)/kg H2']})], ignore_index = True)

    # Brazil, Australia, China, Germany, US, Japan
    for j, r in enumerate([43, 36, 40, 2, 33, 47]):
        reg = ' '.join(titles['RTI'][r].split(' ')[1:-1])
        df_lcoh_r = pd.Series(output_all[scen][var_lcoh][r, t, 0, :]*inflator[r, None], index=tl)[tl_out]
        for year, val in df_lcoh_r.items():
            df_lcoh_long = pd.concat([df_lcoh_long,
                                      pd.DataFrame({'Technology': [techlbl], 
                                                    'Regional aggregation': [reg],
                                                    'Year': [year], 'Value': [val], 'Unit': ['Euro(2024)/kg H2']})], ignore_index = True)



df_lcoh_long.to_csv(os.path.join('Graphs', 'LCOH_range_single_scen_v{}.csv'.format(version)), index=False)


# %% Column chart of green hydrogen demand

years = [2035, 2050]

df_green_h2_demand_long = pd.DataFrame(columns = ['Scenario', 'Year', 'Value', 'Unit'])

for y, year in enumerate(years):
    
    idx = tl.tolist().index(year)
    
    green_h2_demand = []
    
    for scen in scen_dict.keys():
        df_green_h2_demand_long = pd.concat([df_green_h2_demand_long,
                                            pd.DataFrame({'Scenario': [scen_dict[scen]],
                                                         'Year': [year], 
                                                         'Value': [output_all[scen]['WGRM'][:, 0, 0, idx].sum()*1e-3],
                                                         'Unit': ['Mt H2']})], ignore_index = True)
#        green_h2_demand.append(output_all[scen]['WGRM'][:, 0, 0, idx].sum()*1e-3)
    df_green_h2_demand_long = pd.concat([df_green_h2_demand_long,
                                        pd.DataFrame({'Scenario': ['Total'],
                                                      'Year': [year], 
                                                      'Value': [output_all[scen]['HYDT'][:, 0, 0, idx].sum()*1e-3],
                                                      'Unit': ['Mt H2']})], ignore_index = True)
#       green_h2_demand.append(output_all[scen]['HYDT'][:, 0, 0, idx].sum()*1e-3)


df_green_h2_demand_long.to_csv(os.path.join('Graphs', 'Green_H2_demand_v{}.csv'.format(version)), index=False)


# %% Brazil vs global bar chart

year = 2050
y = tl.tolist().index(year)

df_brazil_vs_world_long = pd.DataFrame(columns = ['Technology group', 'Regional aggregation', 
                                                  'Scenario', 'Indicator', 'Value', 'Unit'])

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
        df_brazil_vs_world_long = pd.concat([df_brazil_vs_world_long,
                                            pd.DataFrame({'Technology group': [agg_techs2[t]], 'Regional aggregation': ['Brazil'],
                                                         'Scenario': [scen_dict[scen]], 'Indicator': ['Percentage of production'], 
                                                         'Value': [prod_brazil*100],
                                                         'Unit': ['% of production']})], ignore_index = True)
        
        prod_glo = output_all[scen][var_prod][:, idx, 0, y].sum()
        prod_glo = prod_glo / (output_all[scen][var_prod][:, :, 0, y].sum())
        prod_ls_glo.append(prod_glo*100)   
        df_brazil_vs_world_long = pd.concat([df_brazil_vs_world_long,
                                            pd.DataFrame({'Technology group': [agg_techs2[t]], 'Regional aggregation': ['World'],
                                                         'Scenario': [scen_dict[scen]], 'Indicator': ['Percentage of production'], 
                                                         'Value': [prod_glo*100],
                                                         'Unit': ['% of production']})], ignore_index = True)

        if tech == 'ELEC':
            idx = [8, 9, 10]
            
        lcoh_brazil = np.mean(output_all[scen]['HYCC'][43, idx, 0, y] * inflator[43, None, None],
                                 where=output_all[scen][var_prod][43, idx, 0, y]>0.0)
        lcoh_ls_brazil.append(lcoh_brazil)
        df_brazil_vs_world_long = pd.concat([df_brazil_vs_world_long,
                                            pd.DataFrame({'Technology group': [agg_techs2[t]], 'Regional aggregation': ['Brazil'],
                                                         'Scenario': [scen_dict[scen]], 'Indicator': ['Levelised cost of Hydrogen'], 
                                                         'Value': [lcoh_brazil],
                                                         'Unit': ['Euro(2024)/kg H2']})], ignore_index = True)

        lcoh_glo = np.mean(output_all[scen]['HYCC'][:, idx, 0, y] * inflator[:, None, None],
                                 where=output_all[scen][var_prod][:, idx, 0, y]>0.0)
        lcoh_ls_glo.append(lcoh_glo)
        df_brazil_vs_world_long = pd.concat([df_brazil_vs_world_long,
                                            pd.DataFrame({'Technology group': [agg_techs2[t]], 'Regional aggregation': ['World'],
                                                         'Scenario': [scen_dict[scen]], 'Indicator': ['Levelised cost of Hydrogen'], 
                                                         'Value': [lcoh_glo],
                                                         'Unit': ['Euro(2024)/kg H2']})], ignore_index = True)


df_brazil_vs_world_long.to_csv(os.path.join('Graphs', 'Brazil_vs_world_v{}.csv'.format(version)), index=False)


# %% Production polbrief

tl_out = np.arange(2025, 2050+1)
df_production_long = pd.DataFrame(columns=['Technology group', 'Scenario',
                                           'Year', 'Value'])

for s, scen in enumerate(scen_dict.keys()):
    prod = output_all[scen][var_prod][:, idx, 0, :].sum(axis=0).sum(axis=0) * 1e-3
    prod = np.matmul(conv2.iloc[:, :-1].T, output_all[scen][var_prod].sum(axis=0)[:, 0, :]) * 1e-3
    df_prod = pd.DataFrame(prod.values, index=agg_techs2[:-1], columns=tl)

    df_prod_long = df_prod[tl_out].reset_index().melt(id_vars='index', var_name='Year', value_name='Value')# tolong
    df_prod_long.rename(columns={'index': 'Technology group'}, inplace=True)
    df_prod_long.insert(0, 'Scenario', scen_dict[scen])

    df_production_long = pd.concat([df_production_long, df_prod_long])

df_production_long['Unit'] = 'Mt H2'

df_production_long.to_csv(os.path.join('Graphs', 'Production_polbrief_by_scen_v{}.csv'.format(version)), index=False)


# %% LCOH
tl_out = np.arange(2025, 2050+1)

df_lcoh_polbrief_long = pd.DataFrame(columns=['Technology group', 'Scenario',
                                           'Year', 'Value', 'Unit'])


for s, scen in enumerate(scen_dict.keys()):
    
     for t, tech in enumerate(agg_techs2[:-1]):
    
         tech_titles = conv2.index[conv2[tech] == 1].tolist()
         idx = [conv2.index.get_loc(i) for i in tech_titles]
    

        
         # Global weighted average by individual technology
         lcoh_weight = divide(output_all[scen][var_weight][:, idx, 0, :],
                              output_all[scen][var_weight][:, idx, 0, :].sum(axis=0).sum(axis=0)[None, None, :])
        
         lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :] * inflator[:, None, None] , axis=(0,1))

         # lcoh_avg = np.sum(output_all[scen][var_lcoh][:, idx, 0, :] * lcoh_weight, axis=0).sum(axis=0)
         # lcoh_avg2 = np.mean(output_all[scen][var_lcoh][:, idx, 0, :] , axis=(0,1))

         df_lcoh_avg = pd.Series(lcoh_avg2, index=tl)[tl_out]
         
         for year, value in df_lcoh_avg.items():
             df_lcoh_polbrief_long = pd.concat([df_lcoh_polbrief_long,
                                                pd.DataFrame({'Technology group': [agg_techs2[t]], 
                                                              'Scenario': [scen_dict[scen]],
                                                              'Year': [year], 'Value': [value],
                                                              'Unit': ['Euro(2024)/kg H2']})])
         

df_lcoh_polbrief_long.to_csv(os.path.join('Graphs', 'LCOH_polbrief_by_scen_v{}.csv'.format(version)), index=False)


# %%
