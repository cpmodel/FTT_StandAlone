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
import pandas as pd

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
    
    
    
    

