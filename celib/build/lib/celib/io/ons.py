# -*- coding: utf-8 -*-
"""
ons
===
Helper function to read ONS CSV-format datasets to `pandas` DataFrames.

"""

from io import StringIO
import re

import requests


def load_ons_db(filepath_or_buffer, freq='A'):
    """Read ONS CSV data for a selected frequency (annual, quarterly, monthly).

    For an example of the ONS CSV format, see:
      https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/labourmarketstatistics/current/previous/v22/lms.csv

    Parameters
    ----------
    filepath_or_buffer : str (filepath or URL) or file-like
    freq : str {'A', 'Q', 'M'}, default 'A'
        The frequency of the data to be selected: (A)nnual, (Q)uarterly or
        (M)onthly

    Returns
    -------
    A DataFrame with the ONS Series ID codes as column names. The format of the
    index depends on the selected frequency:
     - annual: integers e.g. 2000, 2001, 2002
     - quarterly: strings e.g. '2000Q1', '2000Q2', '2000Q3'
     - monthly: strings e.g. '2000 JAN', '2000 FEB', '2000 MAR'

    Notes
    -----
    This function does not automatically convert the DataFrame's index to a
    `pandas` `PeriodIndex`. While `PeriodIndex` objects do include additional
    features useful for time-series work, few of them are applicable to common
    data-analysis tasks here at CE. Moreover, these objects are more
    complicated to use. For common tasks, these complexities outweigh the
    benefits of using `PeriodIndex` objects by default.

    However, the index is constructed in a way that makes it straightforward to
    convert to a `PeriodIndex` if necessary:

    >>> from pandas import PeriodIndex

    >>> annual_data = load_ons_db('path/to/data.csv')
    >>> annual_data.index = PeriodIndex(annual_data.index, freq='A')

    >>> quarterly_data = load_ons_db('path/to/data.csv', freq='Q')
    >>> quarterly_data.index = PeriodIndex(quarterly_data.index, freq='Q')

    >>> monthly_data = load_ons_db('path/to/data.csv', freq='M')
    >>> monthly_data.index = PeriodIndex(monthly_data.index, freq='M')

    """
    import pandas as pd

    if freq not in ['A', 'Q', 'M']:
        raise ValueError("`freq` argument must be one of 'A', 'Q', 'M'")

    if (filepath_or_buffer.startswith('http://') or
        filepath_or_buffer.startswith('https://')):
        f = requests.get(filepath_or_buffer)
        df = pd.read_csv(StringIO(f.text),
                         index_col=0,
                         header=1,
                         low_memory=False)
    else:
        df = pd.read_csv(filepath_or_buffer,
                         index_col=0,
                         header=1,
                         low_memory=False)

    # Filter data by selected frequency
    if freq == 'A':
        # Annual data: elements are all digits - convert to integers
        df = df.loc[[j.isdigit() for j in df.index]]
        df.index = map(int, df.index)
    elif freq == 'Q':
        # Quarterly data: second-to-last character is 'Q' - leave as string but
        # remove the space in the middle i.e. '2000 Q1' -> '2000Q1'
        selection = [x for x in df.index if x[-2] == 'Q']
        df = df.loc[selection, :]
        df.index = list(map(lambda x: x.replace(' ', ''), df.index))
    elif freq == 'M':
        # Monthly data: year (digits) followed by three-letter month - leave as
        # string
        pattern = re.compile(r'^[0-9]{4}\s[A-Z]{3}$')
        selection = list(filter(lambda x: pattern.match(x) is not None,
                                df.index))
        df = df.loc[selection, :]

    # Convert all elements to floats and delete index name
    df = df.astype(float, copy=False)
    df.index.name = None

    return df
