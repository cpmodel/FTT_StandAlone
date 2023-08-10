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

#
