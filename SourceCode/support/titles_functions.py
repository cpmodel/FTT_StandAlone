# -*- coding: utf-8 -*-
"""
=========================================
titles_functions.py
=========================================
Functions to load classification titles.

Functions included in the file:
    - load_titles
        Load model classifications and titles
"""

# Standard library imports
import os


# Third party imports
import pandas as pd
from pathlib import Path

from SourceCode.paths import get_utilities_path


def load_titles():
    """Load model classifications and titles from new wide-format CSV."""

    titles_file = 'classification_titles.csv'
    titles_path = get_utilities_path() / 'titles' / titles_file

    if not titles_path.is_file():
        raise FileNotFoundError(f"Classification titles file not found at: {titles_path}")

    df = pd.read_csv(titles_path, header=None, keep_default_na=False, dtype=str)

    # Expected structure: classification, description, name type, value1, value2, ...
    titles_dict = {}
<<<<<<< HEAD
    for sheet in sheet_names:
        active = titles_wb[sheet]
        for column_values in active.iter_cols(min_row=1, values_only=True):
            # Assigning the full names (e.g. "1 Petrol Econ")
            if column_values[0] == 'Full name':  # First row
                titles_dict[f'{sheet}'] = column_values[1:]
            # Assigning the short names (e.g. "1")
            if column_values[0] == 'Short name': # First row
                titles_dict[f'{sheet}_short'] = column_values[1:]
                
            # Loading extra gamma automation titles
            if column_values[0] == 'shares_var': 
                titles_dict[f'{sheet}_shares_var'] = column_values[1:]
            if column_values[0] == 'Cost_var': 
                titles_dict['Cost_var'] = column_values[1:]
            if column_values[0] == 'Gamma_ind': 
                titles_dict['Gamma_ind'] = column_values[1:]
            if column_values[0] == 'histend_var': 
                titles_dict[f'{sheet}_histend_var'] = column_values[1:]
            if column_values[0] == 'Gamma_Value': 
                titles_dict['Gamma_Value'] = column_values[1:]
            if column_values[0] == 'tech_var': 
                titles_dict['tech_var'] = column_values[1:] 
=======
>>>>>>> origin/main

    for _, row in df.iterrows():
        classification = row[0]
        name_type = row[4]
        values = [v for v in row.iloc[5:] if v != '' and pd.notna(v)]

        # Vectorised int conversion (faster than loop)
        cleaned_vals = [int(v) if v.isdigit() else v for v in values]

        if name_type == 'Full name':
            titles_dict[classification] = tuple(cleaned_vals)
        elif name_type == 'Short name':
            titles_dict[f"{classification}_short"] = tuple(cleaned_vals)

    return titles_dict
