# -*- coding: utf-8 -*-
"""
test_ons
==========
Tests for the `load_ons_db` function, which loads ONS csv time series datasets
into a format that is easier to work with in Python.

Implements a test class, `TestONS`, to run tests when loading `lms.csv` file.
"""

import os
import string
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal


from celib import load_ons_db

# The test data are in a folder called 'data', which is in the same folder as
# this script
data_dir = os.path.join(os.path.split(__file__)[0], 'data')


class TestONS(unittest.TestCase):
    """Test class implemented to check `load_ons_db` function.

    Defines one constant:
      - input_file  :  path to read in CSV file to test

    """

    input_file = os.path.join(data_dir, 'lms.csv')

    def test_frequencies(self):
        # Test that the function only accepts `A`, `Q` and `M` as arguments for
        # `freq=`. Loops through and tests all letters of the alphabet.
        letters_not_freq = [letter for letter
                            in list(string.ascii_uppercase)
                            if letter not in ['A', 'Q', 'M']]
        for letter in letters_not_freq:
            self.assertRaises(ValueError,
                              load_ons_db,
                              self.input_file, letter)

    def test_values_float(self):
        # Tests that the DataFrame values are set to `float`. `pd.Series.all()`
        # function is used to ensure that ALL series have `dtypes` equal to
        # `float`
        self.assertEqual(load_ons_db(self.input_file).dtypes.all(), float)

    def test_load_ons_annual(self):
        expected_file = os.path.join(data_dir, 'lms_out_a.csv')
        expected = pd.read_csv(expected_file,
                               header=0,
                               index_col=0)
        expected = expected.astype(float)
        results = load_ons_db(self.input_file, freq='A')
        assert_frame_equal(results, expected)

    def test_load_ons_quarterly(self):
        expected_file = os.path.join(data_dir, 'lms_out_q.csv')
        expected = pd.read_csv(expected_file,
                               header=0,
                               index_col=0)
        expected = expected.astype(float)

        results = load_ons_db(self.input_file, freq='Q')
        assert_frame_equal(results, expected)

    def test_load_ons_monthly(self):
        expected_file = os.path.join(data_dir, 'lms_out_m.csv')
        expected = pd.read_csv(expected_file,
                               header=0,
                               index_col=0)
        expected = expected.astype(float)
        results = load_ons_db(self.input_file, freq='M')
        assert_frame_equal(results, expected)


if __name__ == '__main__':
    unittest.main()
