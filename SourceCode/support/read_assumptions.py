# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:00:24 2024

@author: Femke Nijsse
"""


# Third-party libraries
import pandas as pd

def read_sc_assumptions():

    # Path to the CSV file
    file_path = 'Inputs/S0/Assumptions/sector_coupling_assumptions.csv'
    
    # Reading the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # # Display the first few rows of the DataFrame
    # print(df.head())
    
    return df