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
from openpyxl import load_workbook


def load_titles():
    """ Load model classifications and titles. """

    # Declare file name
    titles_file = 'classification_titles.xlsx'

    # Check that classification titles workbook exists
    # Note this function is being called from 'run_NEMF.py'
    titles_path = os.path.join('Utilities', 'titles', titles_file)
    if not os.path.isfile(titles_path):
        print('Classification titles file not found.')

    titles_wb = load_workbook(titles_path)
    sn = titles_wb.sheetnames
    sn.remove('Cover')

    # Iterate through worksheets and add to titles dictionary
    titles_dict = {}
    for sheet in sn:
        active = titles_wb[sheet]
        for value in active.iter_cols(min_row=1, values_only=True):
            if value[0] == 'Full name':
                titles_dict['{}'.format(sheet)] = value[1:]
            if value[0] == 'Short name':
                titles_dict['{}_short'.format(sheet)] = value[1:]

    # Return titles dictionary
    return titles_dict
