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
from copy import deepcopy as dc

"""
Structure of dict:
Model name: [Model abbreviated name, scenarios to read in, excel file]
Scenario 0 = Baseline
Scenario 1 = 2-degree scenario (default)
Scenario 2 = 1.5-degree scenario (default)
ENTER SCENARIO NUMBERS HERE! This will dictate which sheets are read in.
"""
# models = {'FTT-Tr': [[0], 'FTT-Tr_25x70_2021'],
#           'FTT-P': [[0], 'FTT-P-24x71_2022'],
#           'FTT-H': [[0], 'FTT-H-13x70_2021'],
#           'FTT-S': [[0], 'FTT-S-26x70_2021']}

models = {'FTT-IH-CHI': [[0], 'FTT-IH-CHI-13x70_2022'],
          'FTT-IH-FBT': [[0], 'FTT-IH-FBT-13x70_2022'],
          'FTT-IH-MTM': [[0], 'FTT-IH-MTM-13x70_2022'],
          'FTT-IH-NMM': [[0], 'FTT-IH-NMM-13x70_2022'],
          'FTT-IH-OIS': [[0], 'FTT-IH-OIS-13x70_2022'],
}

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

    # Time horizons
    time_horizon_df = pd.read_excel(os.path.join(dirp_master, 'FTT_variables.xlsx'),
                                    sheet_name='Time_Horizons')
    tl_dict = {}
    for i, var in enumerate(time_horizon_df['Variable name']):
        if time_horizon_df.loc[i, 'Time horizon'] == 'tl_1990':
            tl_dict[var] = list(range(1990, 2100+1))
        elif time_horizon_df.loc[i, 'Time horizon'] == 'tl_2000':
            tl_dict[var] = list(range(2000, 2100+1))
        elif time_horizon_df.loc[i, 'Time horizon'] == 'tl_2001':
            tl_dict[var] = list(range(2001, 2100+1))
        elif time_horizon_df.loc[i, 'Time horizon'] == 'tl_1918':
            tl_dict[var] = list(range(1918, 2017+1))
        elif time_horizon_df.loc[i, 'Time horizon'] == 'tl_1995':
            tl_dict[var] = list(range(1995, 2100+1))
        elif time_horizon_df.loc[i, 'Time horizon'] == 'tl_2018':
            tl_dict[var] = list(range(2018, 2100+1))
            
    # Get classifications
    titles = titles_f.load_titles()
    
    # Get variable dimensions
    dims, histend, domain, forstart = dims_f.load_dims()

    # timeline
    tl = list(range(2000, 2060+1))
    
    # %% Exog market share additions
    output_ixs = {}
    
    eu_regs = [reg for r, reg in enumerate(titles['RTI_short']) if r < 27 or r == 30]
    
    
    
    for m, model in enumerate(models.keys()):
        
        var_to_extract = "IXS{}".format(m+1)
        output_ixs[model] = {}
        dir_out = os.path.join(dirp_in, 'S0', model)
        
        scen_nos = models[model][0]
        for scen_no in scen_nos:
            master_file_name = models[model][1]
            xlsx_path = os.path.join(dirp_master, model, "{}_S{}.xlsx".format(master_file_name, scen_no))
            
            raw_data = pd.read_excel(xlsx_path, sheet_name=var_to_extract)
            raw_titles = pd.read_excel(xlsx_path, sheet_name="Titles")
            itti_raw = list(raw_titles['ITTI'])[:20]
            
            row_start = 4
            col_start = 2
            col_end = col_start + len(list(range(2000, 2060+1)))
            
            for r, reg in enumerate(titles["RTI_short"]):
                
                if reg in eu_regs:
                    row_end = row_start+len(titles["ITTI"])
                    
                    output_ixs[model][reg] = pd.DataFrame(raw_data.iloc[row_start:row_end, col_start:col_end].values,
                                     index=titles["ITTI"],
                                     columns = list(range(2000, 2060+1)))
                        
                    dir_out_fn = os.path.join(dir_out, "{}_{}.csv".format(var_to_extract, reg))
                    output_ixs[model][reg].to_csv(dir_out_fn) 
                
                row_start += len(itti_raw)+1
                
    # %% Emission factors
    
    output_ihw = {}
    
    for m, model in enumerate(models.keys()):
        
        var_to_extract = "IHW{}".format(m+1)
        output_ixs[model] = {}
        dir_out = os.path.join(dirp_in, 'S0', model)
        
        scen_nos = models[model][0]
        for scen_no in scen_nos:
            master_file_name = models[model][1]
            xlsx_path = os.path.join(dirp_master, model, "{}_S{}.xlsx".format(master_file_name, scen_no))
            
            raw_data = pd.read_excel(xlsx_path, sheet_name=var_to_extract)
            raw_titles = pd.read_excel(xlsx_path, sheet_name="Titles")
            itti_raw = list(raw_titles['ITTI'])[:20]
            
            row_start = 4
            col_start = 2
            col_end = col_start + len(list(range(2000, 2060+1)))
            row_end = row_start + len(titles["ITTI"])
                    
            output_ixs[model] = pd.DataFrame(raw_data.iloc[row_start:row_end, col_start:col_end].values,
                             index=titles["ITTI"],
                             columns = list(range(2000, 2060+1)))
            
            # Correction to direct biomass
            output_ixs[model].loc['Indirect Heating Biomass', :] = 0.0
                
            dir_out_fn = os.path.join(dir_out, "{}.csv".format(var_to_extract))
            output_ixs[model].to_csv(dir_out_fn) 
                
                
    # %% write stragic deployment
    

    
    
    # %% create new policy inputs
    
    techs_lowcarb = ['Indirect Heating Biomass', 'Indirect Heating Electric', 
                     'Heat Pumps (Electricity)', 'Direct Heating Biomass', 
                     'Direct Heating Electric']
    
    techs_highcarb = ['Indirect Heating Coal', 'Indirect Heating Oil',
                      'Indirect Heating Gas', 'Direct Heating Coal',
                      'Direct Heating Oil', 'Direct Heating Gas',]
    
    output_isb = {}
    output_regs = {}
    output_strat = {}
    
    scens = ["subs", "subs_ct", "subs_ct_reg"]
    
    tl_policies = list(range(2025, 2050+1))
    tl_policies_strat = list(range(2025, 2030+1))
    
    for s, scen in enumerate(scens):
        
        output_isb[scen] = {}
        output_regs[scen] = {}
        output_strat[scen] = {}
    
        for m, model in enumerate(models.keys()):
            
            var_isb = "ISB{}".format(m+1)
            output_isb[scen][model] = {}
            
            var_regs = "IRG{}".format(m+1)
            output_regs[scen][model] = {}
            
            var_strat = "IXS{}".format(m+1)
            output_strat[scen][model] = {}
            
            dir_out = os.path.join(dirp_in, scen, model)
            
            for reg in eu_regs:
                
                if "subs" in scen:
                    output_isb[scen][model][reg] = pd.DataFrame(0.0, index=titles["ITTI"], columns=tl)
                    output_isb[scen][model][reg].loc[techs_lowcarb, tl_policies] = -0.5
                    output_isb[scen][model][reg].to_csv(os.path.join(dir_out, "{}_{}.csv".format(var_isb, reg)))
                    
                if "reg" in scen:
                    output_regs[scen][model][reg] = pd.DataFrame(-1, index=titles["ITTI"], columns=tl)
                    output_regs[scen][model][reg].loc[techs_highcarb, tl_policies] = 0.0
                    output_regs[scen][model][reg].to_csv(os.path.join(dir_out, "{}_{}.csv".format(var_regs, reg)))
                    
                output_strat[scen][model][reg] = dc(output_ixs[model][reg])
                output_strat[scen][model][reg].loc[techs_lowcarb, tl_policies_strat] = 0.0005
                output_strat[scen][model][reg].to_csv(os.path.join(dir_out, "{}_{}.csv".format(var_strat, reg)))
    
    # %% write phase-out regulations
    
                

                
                
                
                
                
            
            
            
        
        
        











#     # Dict to collect errors
#     errors = {}

#     # Dict to collect all data
#     vardict = {}

#     # Which variables are uploaded?
#     vars_to_upload = {}

#     # Loop over all FTT models of interest
#     for model in models.keys():
#         scenarios = models[model][0]
#         errors[model] = []
#         vardict[model] = {}

#         # Get variable dimensions/attributes
#         variables_df = pd.read_excel(os.path.join(dirp, 'FTT_variables.xlsx'),
#                                      sheet_name=model,
#                                      true_values='y',
#                                      false_values='n',
#                                      na_values='-')

#         for i, var in enumerate(variables_df['Variable name']):
#             vardict[model][var] = {}
#             vardict[model][var]['Code'] = int(variables_df.loc[i, 'Code'])
#             vardict[model][var]['Desc'] = variables_df.loc[i, 'Description']
#             vardict[model][var]['Dims'] = [variables_df.loc[i, attr]
#                                     for attr in ['RowDim', 'ColDim', '3DDim']
#                                     if variables_df.loc[i, attr] not in [0, np.nan]]
#             vardict[model][var]['Read in?'] = variables_df.loc[i, 'Read in?']
# #            vardict[var]['Scen'] = variables_df.loc[i, 'Scenarios']
#             vardict[model][var]['Data'] = {}

#         # Get model classifications
#         dims = list(pd.concat([variables_df['RowDim'], variables_df['ColDim'], variables_df['3DDim']]))
#         dims = list(set([dim for dim in dims if dim not in ['TIME', np.nan, 0]]))
#         dims = {dim: None for dim in dims}
#         with DB1(os.path.join(dirpdb, 'U.db1')) as db1:
#             for dim in dims:
#                 dims[dim] = db1[dim]
# #%%
# ### ----------------------------------------------------------------------- ###
# ### ---------------------------- EXTRACT DATA ----------------------------- ###
# ### ----------------------------------------------------------------------- ###
#         # Define which sheets to load
#         vars_to_upload[model] = [var for var in vardict[model] if vardict[model][var]['Read in?']]
# #        scen_vars = [var for var in vardict if vardict[var]['Scen']]

#         sheets = ['Titles'] + vars_to_upload[model]

#         for scen in scenarios:

#             out_dir = os.path.join(dirp_up, 'S{}'.format(scen), model)
#             if not os.path.exists(out_dir):
#                 os.makedirs(out_dir)


#             # Check whether the excel files exist
#             raw_f = models[model][1]
#             raw_p = os.path.join(dirp, model, '{}_S{}.xlsx'.format(raw_f, scen))
#             if not os.path.exists(raw_p):
#                 msg = "{} does not exists. {} variables will not be uploaded".format(raw_f, model)
#                 print(msg)
#                 continue
#             # Tell the user that the file is being read in.
#             msg = "Extracting {} variables of scenario {} from the excelsheets".format(model, scen)
#             print(msg)

#             # Load sheets
#             raw_data = pd.read_excel(raw_p, sheet_name=sheets, header=None)

#             # Get titles from the Titles sheet in the excel file
#             raw_titles = raw_data['Titles']
#             ftt_titles = {}
#             for col in range(1, raw_titles.shape[1], 2):
#                 var = raw_titles.iloc[0, col]
#                 titles = list(raw_titles.iloc[:, col].dropna())[1:]
#                 ftt_titles[var] = titles

#             # Extract data & save to relevant variable
#             row_start = 5
#             regs = dims['RSHORTTI']

#             ## Read in sheet by sheet
#             ci = 2
#             for i, var in enumerate(vars_to_upload[model]):
#                 ndims = len(vardict[model][var]['Dims'])
#                 rdim = len(dims[vardict[model][var]['Dims'][0]])
#                 r_ttle = dims[vardict[model][var]['Dims'][0]]
#                 if len(vardict[model][var]['Dims']) == 1:
#                     cdim = 1
#                     c_ttle = ['NA']
#                 elif vardict[model][var]['Dims'][1] != 'TIME':
#                     cdim = len(dims[vardict[model][var]['Dims'][1]])
#                     c_ttle = dims[vardict[model][var]['Dims'][1]]
#                 else:
#                     cdim = len(tl_dict[var])
#                     c_ttle = tl_dict[var]
#                 excel_dim = len(ftt_titles[vardict[model][var]['Dims'][0]])
#                 cf = ci + cdim
#                 sep = 1 + excel_dim - rdim
#                 vardict[model][var]['Data'][scen] = {}
#                 sheet_name = var
#                 if ndims == 3:
#                     vardict[model][var]['Data'][scen] = {}
#                     for i, reg in enumerate(regs):
#                         ri = row_start + i*(rdim + sep)
#                         rf = ri + rdim
#                         data = raw_data[sheet_name].iloc[ri:rf, ci:cf]
#                         vardict[model][var]['Data'][scen][reg] = np.array(data.astype(np.float32))

#                         out_fn = os.path.join(out_dir, "{}_{}.csv".format(var, reg))
#                         df = pd.DataFrame(data.values, index=r_ttle, columns=c_ttle)
#                         df.to_csv(out_fn)

#                 elif ndims==2:
#                     ri = row_start
#                     rf = ri + rdim
#                     data = raw_data[sheet_name].iloc[ri:rf, ci:cf]
#                     vardict[model][var]['Data'][scen] = np.array(data.astype(np.float32))

#                     out_fn = os.path.join(out_dir, "{}.csv".format(var))
#                     df = pd.DataFrame(data.values, index=r_ttle, columns=c_ttle)
#                     df.to_csv(out_fn)

#                 elif ndims==1:
#                     ri = row_start
#                     rf = ri + rdim
#                     data = raw_data[sheet_name].iloc[ri:rf, ci:cf]
#                     vardict[model][var]['Data'][scen] = np.array(data.astype(np.float32))

#                     out_fn = os.path.join(out_dir, "{}.csv".format(var))
#                     df = pd.DataFrame(data.values, index=r_ttle, columns=c_ttle)
#                     df.to_csv(out_fn)

#                 msg = "Data for {} saved to CSV. Model: {}".format(var, model)
#                 print(msg)