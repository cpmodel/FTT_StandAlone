# -*- coding: utf-8 -*-
"""
=========================================
run_file.py
=========================================
Run file for FTT Stand alone.


Programme calls the FTT stand-alone model run class, and executes model run.
Run this script from Spyder or VS Code directly, or call it from the command line (or terminal) to run FTT Stand Alone.

Local library imports:

    Model Class:

    - `ModelRun <model_class.html>`__
        Creates a new instance of the ModelRun class


"""

# External library imports
import pickle

# Local library imports
from SourceCode.model_class import ModelRun
from Emulation.code.emulation_code.em_output_save_4 import save_core

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

# Save core output for emulation
save_core(output_all = output_all)