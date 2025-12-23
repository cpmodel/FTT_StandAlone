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

SAVE_GRAPHS = False
FORMAT = 'svg' # png, jpeg


# %% Setup converters and colour maps

# Regions
# Global, Russia, USA, China, India, Indonesia, Saudi Arabia, LATAM, EU, MENA, Japan & Korea


# %% Graph 1 - Production by scenario and region (group) 2025 and 2050

# %% Graph 2 - Demand by scenario and region (group) 2025 and 2050

# %% Graph 3 - Sankey Diagrams of trade flows 2023, 2035, 2050 each scenario

# %% Graph 4 - Delivery costs from top 5 exporters to top 5 imports

# %% Graph 5 - LCOH by technology and region

# %% Graph 6 - Investment

# %% Graph 7 - Sensitivity around costs

# %% Graph 8 - Sensitivity around market share of production 

# %% Table 1 - Emissions (total, direct, indirect, transport)

# %% Annex graph x - Grid emission factors

# %% Annex graph x 



    
    

