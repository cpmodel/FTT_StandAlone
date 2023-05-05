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
import csv

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


#output_all['S0']['MWMC'][0, :, 0, :]
#list all datasets in the output_all dictionary
#output_all['S0'].keys()

pd.DataFrame(np.sum(output_all['S0']['MEWK'], axis = 0).reshape(-1,41)).to_csv('./outputs/S0_MEWK.csv')
pd.DataFrame(np.sum(output_all['S0']['MEWG'], axis = 0).reshape(-1,41)).to_csv('./outputs/S0_MEWG.csv')
pd.DataFrame(np.sum(output_all['S0']['MEWC'], axis = 0).reshape(-1,41)/71).to_csv('./outputs/S0_MEWC.csv')
pd.DataFrame(np.sum(output_all['S0']['MECW'], axis = 0).reshape(-1,41)/71).to_csv('./outputs/S0_MECW.csv')

pd.DataFrame(np.sum(output_all['S1']['MEWK'], axis = 0).reshape(-1,41)).to_csv('./outputs/S1_MEWK.csv')
pd.DataFrame(np.sum(output_all['S1']['MEWG'], axis = 0).reshape(-1,41)).to_csv('./outputs/S1_MEWG.csv')
pd.DataFrame(np.sum(output_all['S1']['MEWC'], axis = 0).reshape(-1,41)/71).to_csv('./outputs/S1_MEWC.csv')
pd.DataFrame(np.sum(output_all['S1']['MECW'], axis = 0).reshape(-1,41)/71).to_csv('./outputs/S1_MECW.csv')

pd.DataFrame(np.sum(output_all['S2']['MEWK'], axis = 0).reshape(-1,41)).to_csv('./outputs/S2_MEWK.csv')
pd.DataFrame(np.sum(output_all['S2']['MEWG'], axis = 0).reshape(-1,41)).to_csv('./outputs/S2_MEWG.csv')
pd.DataFrame(np.sum(output_all['S2']['MEWC'], axis = 0).reshape(-1,41)/71).to_csv('./outputs/S2_MEWC.csv')
pd.DataFrame(np.sum(output_all['S2']['MECW'], axis = 0).reshape(-1,41)/71).to_csv('./outputs/S2_MECW.csv')