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

# Standard library imports
import copy
import os
import sys

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local library imports
import SourceCode.paths_append
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
tl = model.timeline
scens = model.scenarios

# Call the 'run' method of the ModelRun class to solve the model
model.run()

# Fetch ModelRun attributes, for examination
# Output of the model
output_all = model.output

# %%
# mwka = {}
# mewk = {}
# mewg = {}
# mews = {}
# mewsx = {}
# mewd = {}
# mlsp = {}
# mssp = {}
# mlsm = {}
# mssm = {}
# mewr = {}
# metc = {}
# mewl = {}
# mwfc = {}
# mwic = {}
# mgam = {}
# mewl = {}
# mwmc = {}
# mklb = {}
# mred = {}
# mres = {}
# mepd = {}
# mtcd = {}
# mewc = {}
# mcfc = {}
# mcfcx = {}
# mewlx = {}
# for r, reg in enumerate(titles["RTI"]):
#     mwka[reg] = pd.DataFrame(inputs[scens]['MWKA'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mewr[reg] = pd.DataFrame(inputs[scens]['MEWR'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mewk[reg] = pd.DataFrame(output_all[scens]['MEWK'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mewk[reg].loc["TOT", :] = mewk[reg].sum(axis=0)
#     mewg[reg] = pd.DataFrame(output_all[scens]['MEWG'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mewg[reg].loc["TOT", :] = mewg[reg].sum(axis=0)    
#     mews[reg] = pd.DataFrame(output_all[scens]['MEWS'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mews[reg].loc["TOT", :] = mews[reg].sum(axis=0)
#     mewsx[reg] = pd.DataFrame(output_all[scens]['MEWSX'][r, :, 0, :], index=titles['T2TI'], columns=tl)

#     mewd[reg] = pd.DataFrame(output_all[scens]['MEWDX'][r, :, 0, :], index=titles['JTI'], columns=tl)
#     mlsp[reg] = pd.DataFrame(output_all[scens]['MLSP'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mssp[reg] = pd.DataFrame(output_all[scens]['MSSP'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mlsm[reg] = pd.DataFrame(output_all[scens]['MLSM'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mssm[reg] = pd.DataFrame(output_all[scens]['MSSM'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     msal = pd.DataFrame(output_all[scens]['MSAL'][:, 0, 0, :], index=titles['RTI'], columns=tl)
#     metc[reg] = pd.DataFrame(output_all[scens]['METC'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mewc[reg] = pd.DataFrame(output_all[scens]['MEWC'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mtcd[reg] = pd.DataFrame(output_all[scens]['MTCD'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mwfc[reg] = pd.DataFrame(output_all[scens]['MWFC'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mwic[reg] = pd.DataFrame(output_all[scens]['MWIC'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mgam[reg] = pd.DataFrame(output_all[scens]['MGAM'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mewl[reg] = pd.DataFrame(output_all[scens]['MEWL'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mewlx[reg] = pd.DataFrame(output_all[scens]['MEWLX'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mcfc[reg] = pd.DataFrame(output_all[scens]['MCFC'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mcfcx[reg] = pd.DataFrame(output_all[scens]['MCFCX'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mwmc[reg] = pd.DataFrame(output_all[scens]['MWMC'][r, :, 0, :], index=titles['T2TI'], columns=tl)
#     mklb[reg] = pd.DataFrame(output_all[scens]['MKLB'][r, :, 0, :], index=titles['LBTI'], columns=tl)
#     mred[reg] = pd.DataFrame(output_all[scens]['MRED'][r, :, 0, :], index=titles['ERTI'], columns=tl)
#     mres[reg] = pd.DataFrame(output_all[scens]['MRES'][r, :, 0, :], index=titles['ERTI'], columns=tl)
#     mepd[reg] = pd.DataFrame(output_all[scens]['MEPD'][r, :, 0, :], index=titles['ERTI'], columns=tl)

# mssr = pd.DataFrame(output_all[scens]['MSSR'][:, 0, 0, :], index=titles['RTI'], columns=tl)
# mlsr = pd.DataFrame(output_all[scens]['MLSR'][:, 0, 0, :], index=titles['RTI'], columns=tl)

# # %% Pick a country
# r_select = 2
# reg_select = titles["RTI"][r_select]

# excel_fn = "EEIST_vs_SA_comparison_for_{}.xlsx".format(titles["RTI"][r_select][-3:-1])
# with pd.ExcelWriter("./Output/{}".format(excel_fn)) as writer:
    
#     mews[reg_select].loc[titles["T2TI"], 2013:].to_excel(writer, sheet_name="Result SA")
#     mewsx[reg_select].loc[titles["T2TI"], 2013:].to_excel(writer, sheet_name="Result EEIST")
#     diff = mews[reg_select].subtract(mewsx[reg_select])    
#     diff.loc[titles["T2TI"], 2013:].to_excel(writer, sheet_name="Diff")
#     pct_diff = diff.div(mewsx[reg_select])
#     pct_diff.loc[titles["T2TI"], 2013:].to_excel(writer, sheet_name="Pct diff")    


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
