# -*- coding: utf-8 -*-
"""
This script copies all the files from the S0 folder to the subfolders
in case the files do not exist yet.

It also allows copying the MSAL files around. Note there are types
@author: Femke
"""

import shutil, os, glob
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

Nsample = 200

MC_samples = np.vstack([MSAL.rvs(Nsample),     
           WACC_range.rvs(Nsample),                             # BCET
           learning_rate_solar.rvs(Nsample),                    # BCET
           learning_rate_wind.rvs(Nsample),                     # BCET
           lifetime.rvs(Nsample),                               # BCET & MEWA
           gamma.rvs(Nsample),                                  # MGAM
           fuel_costs.rvs(Nsample),                             # BCET
           grid_expansion_duration.rvs(Nsample)+1]).transpose() # BCET & MEWA

top_dir = r"C://Users\Femke\Documents\E3ME_versions\FTT_Stand_Alone_Flex\Inputs"


#%% Copying the General files
source_dir = os.path.join(top_dir, "S0/General")
for dirp_out in [f'S0_{i:03d}' for i in range(Nsample)]:
    desti_dir = os.path.join(top_dir, dirp_out, "General") 
    shutil.copytree(source_dir, desti_dir)
    
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


#%% Copying the rest of the FTT:Power files
source_dir = os.path.join(top_dir, "S0", "FTT-P")

GLOB_PARMS = "*" #maybe "*.pdf" ?
for dirp_out in [f'S0_{i:03d}' for i in range(Nsample)]:
    desti_dir = os.path.join(top_dir, dirp_out, "FTT-P") 
    for file in glob.glob(os.path.join(source_dir, GLOB_PARMS)):
        if file not in glob.glob(os.path.join(desti_dir, GLOB_PARMS)):
            shutil.copy(file,desti_dir)
        # else:
        #     print("{} exists in {}".format(
        #         file,os.path.join(os.path.split(desti_dir)[-2:])))
            # This is just a print command that outputs to console that the
            # file was already in directory