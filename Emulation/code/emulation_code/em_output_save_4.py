# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:16:19 2023

Output save functions to be performed after model runs

@author: ib400
"""
import pickle

#%% Save Core Output

def save_core(output_all, variables_core = ['MEWW', 'MEWS', 'MEWK', 'MEWI', 
                  'MEWG', 'MEWE', 'MEWD', 'METC', 'MRES', 
                  'REPPX', 'MEWC', 'BCET', 
                  'MEWT', 'MEWR', 'MEFI', 'MEWC', 
                 'MECW', 'MEWP' ]):    
    output_all = output_all
    variables_core = variables_core
    
    # Loop through scenarios and rip out core vars
    for scen in output_all.keys():
        
        scenario = output_all[scen]
        scen_core = {}
        for var in scenario.keys():
            if var in variables_core:
                scen_core[var] = scenario[var]
            else:
                pass
        
        with open(f'Output\Results_{scen}_core.pickle', 'wb') as f:
            pickle.dump(scen_core, f)
    
        print(scen, f' saved to Output/Results_{scen}_core.pickle')
