# -*- coding: utf-8 -*-
"""
This script is meant to generate a large set of scenarios

The files that are changes are:
    * What types of sector couplings are turned on
    * General variation in uncertain parameters

It also allows copying the MSAL files around. Note there are types
@author: Femke
"""

import shutil, os
import os.path
import numpy as np

from scipy.stats import poisson, binom, uniform, norm, randint
np.random.seed(123)

MSAL = binom(1, 0.5)                                 # Which MSAL switch to use
WACC_range = uniform()                               # Scale between unequal and equal access to finance
learning_rate_solar = norm(-0.303, 0.047)            # Learning rate solar
learning_rate_wind = norm(-0.158, 0.045)             # Learning rate wind
lifetime = randint(25, 35)                           # Lifetime of solar panel
gamma = norm(loc=1, scale=0.2)                       # Scaling factor of gamma
fuel_costs = norm(loc=1, scale=0.2)                  # Scaling factor of gamma
grid_expansion_duration = poisson(0.6)               # The lead time of solar

Nsample = 1

MC_samples = np.vstack([MSAL.rvs(Nsample),     
           WACC_range.rvs(Nsample),                             # BCET
           learning_rate_solar.rvs(Nsample),                    # BCET
           learning_rate_wind.rvs(Nsample),                     # BCET
           lifetime.rvs(Nsample),                               # BCET & MEWA
           gamma.rvs(Nsample),                                  # MGAM
           fuel_costs.rvs(Nsample),                             # BCET
           grid_expansion_duration.rvs(Nsample)+1]).transpose() # BCET & MEWA

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the parent directory
top_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

#%% Copying the MSAL files
source_dirA = os.path.join(top_dir, "S0", "FTT-P", "MSAL.csv")
source_dirB = os.path.join(top_dir, "_Masterfiles", "MSAL.csv")
MSAL = MC_samples[:, 0]

for i, dirp_out in enumerate([f'S0_{i:03d}' for i in range(Nsample)]):
    desti_dir = os.path.join(top_dir, dirp_out, "FTT-P") 
    if MSAL[i] == 0:
        shutil.copy(source_dirA, desti_dir)
    elif MSAL[i] == 1:
        shutil.copy(source_dirB, desti_dir)


