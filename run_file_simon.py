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

    Support functions:

    - `paths_append <paths_append.html>`__
        Appends file path to sys path to enable import
    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros

"""
print("test")

# Standard library imports
import copy
import os
import sys

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local library imports
import SourceCode.support.paths_append
from SourceCode.model_class import ModelRun
from SourceCode.support.divide import divide

# Instantiate the run
model = ModelRun()

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


# Call the 'run' method of the ModelRun class to solve the model
model.run()

# Fetch ModelRun attributes, for examination
# Output of the model
output_all = model.output


output_all['S0']['MEWDX'][0, :, 0, :]
# list all datasets in the output_all dictionary
output_all['S0'].keys()
output_all['S0']['MEWDX'][0, :, 0, :]
# export output_all to csv
"""
# export output_all numpy array to csv with a new sheet for each key
import xlsxwriter
workbook = xlsxwriter.Workbook('output_all.xlsx', {'nan_inf_to_errors': True})
for key in output_all['S0'].keys():
    worksheet = workbook.add_worksheet(key)
    df = pd.DataFrame(output_all['S0'][key][0, :, 0, :])
    for r, row in enumerate(df.iloc[:, 0]):  # Use iloc to select the first column
        print(row)
        worksheet.write(r, 0, row)  # Write the row to the worksheet
workbook.close()
"""

import xlsxwriter

# Create the workbook and the 'results' sheet
workbook = xlsxwriter.Workbook('output_all.xlsx', {'nan_inf_to_errors': True})
worksheet = workbook.add_worksheet('results')

# Write the headers to the 'results' sheet
col_idx = 0
for key in output_all['S0'].keys():
    worksheet.write(0, col_idx, key)
    col_idx += 1

# Iterate over the output data and write each column to the 'results' sheet
col_idx = 0
for key in output_all['S0'].keys():
    df = pd.DataFrame(output_all['S0'][key][0, :, 0, :])
    for r, row in enumerate(df.iloc[:, 0]):
        worksheet.write(r+1, col_idx, row)
    col_idx += 1

# Close the workbook
workbook.close()


# convert output_all to pandas dataframe
output_all_df = pd.DataFrame(output_all['S0']['MEWDX'])


# %% Plot MEWDX
# mewdx_in = {}
# mewdx_out = {}
#
# for r, reg in enumerate(titles['RTI_short']):
#     mewdx_in[reg] = pd.DataFrame(inputs['S0']['MEWDX'][r, :, 0, :], titles[dims['MEWDX'][1]], columns=titles[dims['MEWDX'][-1]])
#     mewdx_out[reg] = pd.DataFrame(output_all['S0']['MEWDX'][r, :, 0, :], titles[dims['MEWDX'][1]], columns=titles[dims['MEWDX'][-1]])
#
# # %% Plot MEWDX
# mewdx_in = {}
# mewdx_out = {}
#
# for r, reg in enumerate(titles['RTI_short']):
#     mewdx_in[reg] = pd.DataFrame(inputs['S0']['MEWDX'][r, :, 0, :], titles[dims['MEWDX'][1]], columns=titles[dims['MEWDX'][-1]])
#     mewdx_out[reg] = pd.DataFrame(output_all['S0']['MEWDX'][r, :, 0, :], titles[dims['MEWDX'][1]], columns=titles[dims['MEWDX'][-1]])
#
#
# # %% Plot MEWDX
# mewdx_in = {}
# mewdx_out = {}
#
# for r, reg in enumerate(titles['RTI_short']):
#     mewdx_in[reg] = pd.DataFrame(inputs['S0']['MEWDX'][r, :, 0, :], titles[dims['MEWDX'][1]], columns=titles[dims['MEWDX'][-1]])
#     mewdx_out[reg] = pd.DataFrame(output_all['S0']['MEWDX'][r, :, 0, :], titles[dims['MEWDX'][1]], columns=titles[dims['

# # %% Check MEWDX
# mewdx_in = {}
#
# mewdx_out = {}
#
# for r, reg in enumerate(titles['RTI_short']):
#     mewdx_in[reg] = pd.DataFrame(inputs['S0']['MEWDX'][r, :, 0, :], titles[dims['MEWDX'][1]], columns=titles[dims['MEWDX'][-1]])
#     mewdx_out[reg] = pd.DataFrame(output_all['S0']['MEWDX'][r, :, 0, :], titles[dims['MEWDX'][1]], columns=titles[dims['MEWDX'][-1]])
#
# # %% Compare results to E3ME outcomes
#
# var_endo = ['MEWK', 'MEWS', 'MEWG', 'METC', 'MSSP', 'MLSP', 'MSSM', 'MLSM', 'MERC', 'MEWL', 'MWMC', 'MWMD']
# var_exog = [var+'X' for var in var_endo]
#
# comp_ratio = {}
# comp_diff = {}
# comp_pct_diff = {}
# comp_close = {}
#
# scen = 'S0'
#
# endo = {}
# exog = {}
#
#
# for var in var_endo:
#
#     num_dims = [dim for dim in dims[var] if dim != 'NA']
#
#     endo[var] = {}
#     exog[var] = {}
#     comp_ratio[var] = {}
#     comp_diff[var] = {}
#     comp_pct_diff[var] = {}
#     comp_close[var] = {}
#
#     if num_dims == 2:
#
#         endo[var][0] = pd.DataFrame(output_all[scen][var][:, 0, 0, :], index=titles[num_dims[0]], columns=titles[num_dims[-1]])
#         exog[var][0] = pd.DataFrame(output_all[scen][var+'X'][:, 0, 0, :], index=titles[num_dims[0]], columns=titles[num_dims[-1]])
#
#         comp_ratio[var][0] = endo[var][0].div(exog[var][0])
#         comp_diff[var][0] = endo[var][0].subtract(exog[var][0])
#         comp_pct_diff[var][0] = comp_diff[var][0].div(exog[var][0])
#         comp_close[var][0] = np.isclose(endo[var][0], exog[var][0])
#
#     else:
#
#         # MSSP and MLSP are 3D in the standalone, but not E3ME
#         # All values are the same for all techs, so just take index 18 (solar pv)
#         if var in ['MSSP', 'MLSP']:
#
#             endo[var][0] = pd.DataFrame(output_all[scen][var][:, 18, 0, :], index=titles[num_dims[0]], columns=titles[num_dims[-1]])
#             exog[var][0] = pd.DataFrame(output_all[scen][var+'X'][:, 0, 0, :], index=titles[num_dims[0]], columns=titles[num_dims[-1]])
#
#             comp_ratio[var][0] = endo[var][0].div(exog[var][0])
#             comp_diff[var][0] = endo[var][0].subtract(exog[var][0])
#             comp_pct_diff[var][0] = comp_diff[var][0].div(exog[var][0])
#             comp_close[var][0] = np.isclose(endo[var][0], exog[var][0])
#
#         else:
#
#             for r, reg in enumerate(titles['RTI_short']):
#
#                 endo[var][reg] = pd.DataFrame(output_all[scen][var][r, :, 0, :], index=titles[num_dims[1]], columns=titles[num_dims[-1]])
#
#                 if var != 'MEWS':
#
#                     exog[var][reg] = pd.DataFrame(output_all[scen][var+'X'][r, :, 0, :], index=titles[num_dims[1]], columns=titles[num_dims[-1]])
#
#                 else:
#
#                     mewsx = np.divide(output_all[scen]['MEWKX'][r, :, 0, :], output_all[scen]['MEWKX'][r, :, 0, :].sum(axis=0))
#                     exog[var][reg] = pd.DataFrame(mewsx, index=titles[num_dims[1]], columns=titles[num_dims[-1]])
#
#
#                 comp_ratio[var][reg] = endo[var][reg].div(exog[var][reg])
#                 comp_diff[var][reg] = endo[var][reg].subtract(exog[var][reg])
#                 comp_pct_diff[var][reg] = comp_diff[var][reg].div(exog[var][reg])
#                 comp_close[var][reg] = np.isclose(endo[var][reg], exog[var][reg])
#

