# -*- coding: utf-8 -*-
"""
model_class.py
=========================================
ModelRun class: main class for operation of model.

"""

# Standard library imports
import configparser
import copy
import os
import sys

# Third party imports
import pandas as pd
import numpy as np
from tqdm import tqdm

# Local library imports
# Separate FTT modules
import ftt_p_main as ftt_p
#import ftt_tr_main as ftt_tr
#import ftt_h_main as ftt_h
#import ftt_s_main as ftt_s
#import ftt_agri_main as ftt_agri
#import ftt_freight_main as ftt_fr
#import ftt_flex_main as ftt_flex
#import ftt_chemicals_main as ftt_chem


# Support modules
import support.input_functions as in_f
#import support.specification_functions as specs_f
import support.titles_functions as titles_f
import support.dimensions_functions as dims_f
from support.cross_section import cross_section as cs


class ModelRun:
    """
    Class to run the NEMF model.

    Class object is a single run of the model.

    Attributes
    -----------
    name: str
        Name of model run, and specification file read
    hist_start: int
        Starting year of the historical data
    model_start: int
        First year of model timeline
    model_end: int
        Final year of model timeline
    current: int
        Curernt/active year of solution
    years: tuple of (int, int)
        Bookend years of model_timeline
    fd_sectors: list of str
        Final demand sectors to solve endogenously (otherwise spec set to
        exogenous)
    st_sectors: list of str
        Supply and transformation sectors to solve endogenously (otherwise spec
        set to exogenous)
    timeline: list of int
        Years of the model timeline
    time_sim_index: list of int
        Index values of simulation years from model timeline
    titles: dictionary of lists
        Dictionary containing all title classifications
    dims: dict of tuples (str, str, str, str)
        Variable classifications by dimension
    histend: dict of integers
        Final year of histrorical data by variable
    specs: dictionary of NumPy arrays
        Function specifications for each region and module
    input: dictionary of NumPy arrays
        Dictionary containing all model input variables
    results_list: list of str
        List of variables to print in results
    variables: dictionary of NumPy arrays
        Dictionary containing all model variables for a given year of solution
    lags: dictionary of NumPy arrays
        Dictionary containing lag variables
    output: dictionary of NumPy arrays
        Dictionary containing all model variables for output

    Methods
    -----------
    run
        Solve model run and save results
    solve_all
        Solve model for each year of the simulation period
    solve_year
        Solve model for a specific year
    update
        Update model variables for a new year of solution
    """

    def __init__(self):
        """ Instantiate model run object """

        # Attributes given in settings.ini file
        config = configparser.ConfigParser()
        config.read('settings.ini')
        self.name = config.get('settings', 'name')
        self.model_start = int(config.get('settings', 'model_start'))
        self.model_end = int(config.get('settings', 'model_end'))
        self.simulation_start = int(config.get('settings', 'simulation_start'))
        self.simulation_end = int(config.get('settings', 'simulation_end'))
        self.current = self.model_start
        self.years = np.arange(self.model_start, self.model_end+1)
        self.timeline = np.arange(self.simulation_start, self.simulation_end+1)
        self.ftt_modules = config.get('settings', 'enable_modules')
        self.scenarios = config.get('settings', 'scenarios')

        # Load classification titles
        self.titles = titles_f.load_titles()

        # Load variable dimensions
        self.dims, self.histend, self.domain, self.forstart = dims_f.load_dims()

        # Retrieve inputs
        self.input = in_f.load_data(self.titles, self.dims, self.timeline,
                                    self.scenarios, self.ftt_modules,
                                    self.forstart)


        # Initialize remaining attributes
        self.variables = {}
        self.lags = {}
        self.output = {}

    def run(self):
        """ Solve model run and save results """

        # Run the solve all method (self.input contains all results)
        self.solve_all()

    def solve_all(self):
        """ Solve model for each year of the simulation period """

        # Define output container
        self.output = {scen: {var: np.full_like(self.input[scen][var], 0) for var in self.input[scen]} for scen in self.input}
        # self.output = copy.deepcopy(self.input)

        # Clear any previous instances of the progress bar
        try:
            tqdm._instances.clear()
        except AttributeError:
            pass
        for scen in self.input:

            # Create progress bar:
            with tqdm(self.timeline) as pbar:

            # Call solve_year method for each year of the simulation period
#                for year_index, year in enumerate(self.timeline):
                for y, year in enumerate(self.timeline):
                    # Set the description to be the current year
                    pbar.set_description('Running Scenario: {} - Solving year: {}'.format(scen, year))

                    self.variables, self.lags = self.solve_year(year, y, scen)

                    # Increment the progress bar by one step
                    pbar.update(1)

                    # Populate output container
                    for var in self.variables:
                        if 'TIME' in self.dims[var]:
                            try:
                                self.output[scen][var][:, :, :, y] = self.variables[var]
                            except ValueError:
                                print(var)
                                print(self.variables[var])
                                raise
                        else:
                            self.output[scen][var][:, :, :, 0] = self.variables[var]

            # Set the progress bar to say it's complete
            pbar.set_description("Model run {} finished".format(self.name))

    def solve_year(self, year, y, scenario, max_iter=1):
        """ Solve model for a specific year """

        # Need to add a convergence check here in the future

        # Run update
        variables, time_lags = self.update(year, y, scenario)
        iter_lags = copy.deepcopy(time_lags)

        # Define whole period
        tl = self.timeline


        # Iteration loop here
        for iter in range(0, max_iter):

            if "FTT-P" in self.ftt_modules:
                variables = ftt_p.solve(variables, time_lags, iter_lags,
                                        self.titles, self.histend, tl[y],
                                        self.domain)
            elif "FTT-Tr" in self.ftt_modules:
                print("Module needs to be created")
            elif "FTT-H" in self.ftt_modules:
                print("Module needs to be created")
            elif "FTT-S" in self.ftt_modules:
                print("Module needs to be created")
            else:
                print("Incorrect selection of modules. Check settings.ini")

            # Third, solve energy supply
            # Overwrite iter_lags to be used in the next iteration round
            iter_lags = copy.deepcopy(variables)
#        # Print any diagnstics
#
        return variables, time_lags

    def update(self, year, y, scenario):
        """ Update model variables for a new year of solution """

        # Update the current year attribute
        self.current = year

        # Set any required variables as equal to the previous year
        # This is how E3ME solves a number of variables, return to this

        # Read required variables from the cross section
        # This is how E3ME solves a number of variables, return to this
        data_to_model = cs(self.input, self.dims, year, y, scenario)

        # LB TODO: to improve the treatment of lags to include also historical data
        if y == 0:
            lags = cs(self.input, self.dims, year, y, scenario)   #If year is the first year, lags equal variables in starting year
        else:
            #lags = cs(self.variables, self.dims, year-1)
            lags = self.variables

        return data_to_model, lags

