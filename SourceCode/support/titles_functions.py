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


def load_titles():
    """Load model classifications and titles from new wide-format CSV."""

    dir_file = os.path.dirname(os.path.realpath(__file__))
    dir_root = Path(dir_file).parents[1]

    titles_file = 'classification_titles.csv'
    titles_path = os.path.join(dir_root, 'Utilities', 'titles', titles_file)
    
    if not os.path.isfile(titles_path):
        print('Classification titles file not found.')

    df = pd.read_csv(titles_path, header=None, keep_default_na=False, dtype=str)

    # Expected structure: classification, description, name type, value1, value2, ...
    titles_dict = {}

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
