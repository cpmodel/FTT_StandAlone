# -*- coding: utf-8 -*-
"""
Created on Mon May 19 09:43:43 2025

@author: Test Profile
"""

import os
import pickle
from preprocessing import define_path

project_root_absolute_path = define_path()

large_files = ['Results_sequencing2.pickle', "Results_sxp.pickle"]

for large_file in large_files:
    pickle_path = os.path.join(project_root_absolute_path, 'Output', large_file) 
    
    
    # Load original file
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Define variables to keep
    keep_vars = {'MEWS', 'HEWS', 'ZEWS', 'TEWS', 'MLCO', 'MECW battery only',
                 'TEWC', 'HEWC', 'ZTLC', 'MEWG', 'TEWK', 'HEWG', 'ZEWK', 'ZEWI'}
    
    # Filter data
    filtered_data = {
        scenario: {k: v for k, v in variables.items() if k in keep_vars}
        for scenario, variables in data.items()
    }
    
    # Save filtered data
    with open('Analysis/Input/' + large_file, 'wb') as f:
        pickle.dump(filtered_data, f)