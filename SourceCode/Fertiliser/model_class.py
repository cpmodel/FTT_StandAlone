# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:12:42 2023

@author: adh
"""



# Standard library imports
import configparser
import copy
import os
import sys
import time
import pickle
from tqdm import tqdm

# Third party imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import math

# Support modules
import SourceCode.support.input_functions as in_f
import SourceCode.support.titles_functions as titles_f
import SourceCode.support.dimensions_functions as dims_f
from SourceCode.support.cross_section import cross_section as cs
from SourceCode.initialise_csv_files import initialise_csv_files
import npv_calculation as npv_calc
import bass_model as bm


class ModelRun:
    """
    Class to run the CLEAFS model.

    Class object is a single run of the model.

    Local library imports:


    Attributes
    -----------
    name: str
        Name of model run, and specification file read.
    hist_start: int
        Starting year of the historical data.
    model_start: int
        First year of model timeline.
    current: int
        Current/active year of solution.
    days: int
        Number of days in the model timeline.
    timeline: list of int
        Years of the model timeline.
    titles: dict of {str: list}
        Dictionary containing all title classifications.
    dims: dict of {str: tuple (str, str, str, str)}
        Variable classifications by dimension.
    histend: dict of {str: int}
        Final year of historical data by variable.
    input: dict of {str: numpy.ndarray}
        Dictionary containing all model input variables.
    results_list: list of str
        List of variables to print in results.
    variables: dict of {str: numpy.ndarray}
        Dictionary containing all model variables for a given year of solution.
    converter: dict of {str: DataFrame}
        Model converters.
    time_lags: dict of {int: dict of {str: numpy.ndarray}}
        Lagged values (in previous years of solution).
    output: dict of {str: numpy.ndarray}
        Dictionary containing all model variables for output.

    Methods
    -----------
    run
        Solve model run and save results.
    solve_all
        Solve model for each year of the simulation period.
    solve_year
        Solve model for a specific year.
    update
        Update model variables for a new year of solution.


    """

    def __init__(self):
        """ Instantiate model run object. """
        # Attributes given in settings.ini file
        config = configparser.ConfigParser()
        config.read('settings.ini')
        self.name = config.get('settings', 'name')
        self.model_start = int(config.get('settings', 'model_start'))
        self.simulation_start = int(config.get('settings', 'simulation_start'))
        self.model_end = int(config.get('settings', 'model_end'))
        # self.run_bass_model = str(config.get('settings', 'run_bass_model'))
        self.subsidy = float(config.get('settings', 'subsidy'))
        self.lump_sum = float(config.get('settings', 'lump_sum'))


        # self.model_start = datetime.strptime(config.get('settings', 'model_start'), "%Y-%m-%d")
        # self.model_end = datetime.strptime(config.get('settings', 'model_end'), "%Y-%m-%d")

        # Further defintion of attributes
        # Define timeline of the whole model, including history
        self.years = (self.model_start, self.model_end)
        self.timeline = list(range(self.model_start, self.model_end+1))

        # Load classification titles
        self.titles = titles_f.load_titles()

        # Load variable dimensions
        self.dims, self.histend = dims_f.load_dims()

#        # Load historical data, exogenous assumptions
        self.data = in_f.load_data(self.titles, self.dims, self.timeline)

        # Initiate Bass model
        if self.run_bass_model == 'yes':
            print('Initiate Bass model')
            self.data = bm.Bass_param_estimation(self.data, self.titles)


        # Read converters
        self.converter = in_f.load_converters()

        # Initialize remaining attributes
        self.variables = {}
        self.time_lags = {}
        self.output = {}

    #   self.lag_sales = {}  #Erase after checking

        print("Initiated")

    def run(self):
        """ Solve model run and save results. """
        # Run the solve all method (self.input contains all results)
        self.solve()


    def solve(self):
        """ Solve model for each year. """
        # Clear any previous instances of the progress bar
        try:
            tqdm._instances.clear()
        except AttributeError:
            pass

        # Create progress bar:
        with tqdm(list(self.timeline)) as pbar:
            # Call solve_day method for each year of the observation period
            for period, year in enumerate(self.timeline):
                # Set the description to be the current year
                s1 = 'Model run {}'.format(self.name)
#                s2 = 'Specification: {}'.format(spec)
                s3 = 'Solving year: {}'.format(year)
                pbar.set_description('{} ; {}'.format(s1, s3))
                # NPV of rooftop solar PVs
                self.data = npv_calc.npv_calculation(self.data, self.titles, self.subsidy, self.lump_sum, period)
                self.data = npv_calc.potential_population(self.data, self.titles, period)


                # Increment the progress bar by one step
                pbar.update(1)
#
            # Populate output container
            # self.output = self.data

        # Set the progress bar to say it's complete
        pbar.set_description("Model run {} finished".format(self.name))





