# -*- coding: utf-8 -*-
"""
test_tabls
==========
Tests for the `Tabls` class, to process an IDIOM model's tabls.dat file,
categorising by {variable,title} and providing functions to more easily search
in those two groups.

Implements a test class, `TestE3ME`, to run tests on the E3ME tabls.dat
file. The MDM-E3 equivalent, `TestMDME3` subclasses `TestE3ME`, over-riding
E3ME-specific data with MDM-E3 data (as well as ignoring duplicate title codes,
which occur in the E3ME file).

"""

from collections import Counter
import csv
import os
import unittest

from celib import Tabls
from celib.exceptions import TablsError


# The test data are in a folder called 'data', which is in the same folder as
# this script
data_dir = os.path.join(os.path.split(__file__)[0], 'data')


class TestE3ME(unittest.TestCase):
    """Test class implemented for E3ME to check `Tabls` class functionality.

    Defines three constants:
     - input_file : path to tabls.dat file to test
     - expected_variables_file : path to CSV file of expected variable
                                 information (a subset of the full tabls file)
     - expected_titles_file : path to CSV file of expected titles information

    Derived classes should over-ride the above to test alternative tabls files.

    """

    input_file = os.path.join(data_dir, 'tabls_e3me.dat')
    expected_variables_file = os.path.join(data_dir, 'e3me_variables.csv')
    expected_titles_file = os.path.join(data_dir, 'e3me_titles.csv')

    def setUp(self):

        def parse(entry):
            """`eval()` selected dict values to convert to int/str etc."""
            for key in ['dimnames', 'dimcodes', 'code', 'units']:
                try:
                    entry[key] = eval(entry[key])
                except KeyError:
                    pass

            return entry

        def read_csv(path):
            """Return the CSV data (including header) in `path` as a list of dicts."""
            with open(path) as f:
                reader = csv.DictReader(f)
                return list(map(parse, reader))

        # Tabls object to test
        self.tabls = Tabls(self.input_file)

        # Expected results to test against
        self.expected_variables = read_csv(self.expected_variables_file)
        self.expected_titles = read_csv(self.expected_titles_file)


    def test_get_titles_by_code_unique(self):
        # Loop through the classifications by code, using the code to get the
        # corresponding entry from `self.tabls`. Test for equality between the
        # actual and expected titles entries (coerced from OrderedDict to dict
        # as necessary, to avoid ordering mismatches).
        # This is the unique-case version, which only tests code numbers that
        # are unique to a particular set of titles e.g. regions ('RTI').

        duplicated = set()

        for classification in self.expected_titles:
            code = classification['code']

            # If this code has already been encountered, continue
            if code in duplicated:
                continue
            # Store the code to check against future duplicates
            duplicated.add(code)

            tabls_entry = dict(self.tabls.get_titles_by_code(code))

            assert tabls_entry == dict(classification), (
                print('Found: {}\nExpected: {}'.format(
                    tabls_entry, dict(classification))))

    def test_get_titles_by_code_duplicate(self):
        # Duplicate-case version of `test_get_titles_by_code_unique()`.
        # Test code numbers which apply to multiple sets of classifications
        # e.g. industries ('YTI'), which shares a code (2) with 'STI', 'XTI',
        # and 'QTI'.
        # The expected behaviour is to return the first entry of that code
        # only. (In the example above, the entry for 'YTI'.)

        # Count the number of instances of each titles code
        counts = Counter(x['code'] for x in self.expected_titles)

        # Get duplicated codes, which occur more than once (have counts >1)
        duplicate_codes = [k for k, v in counts.items() if v > 1]

        # Store for the entry of the first occurrence of a titles code
        duplicate_entries = {}

        for classification in self.expected_titles:
            code = classification['code']

            # Continue if code is not a duplicate
            if code not in duplicate_codes:
                continue

            # Store the entry for the first occurrence of a code
            if code not in duplicate_entries:
                duplicate_entries[code] = classification

            tabls_entry = dict(self.tabls.get_titles_by_code(code))

            assert tabls_entry == dict(duplicate_entries[code]), (
                print('Found: {}\nExpected: {}'.format(
                    tabls_entry, dict(duplicate_entries[code]))))

    def test_get_titles_by_code_not_found(self):
        with self.assertRaises(KeyError):
            self.tabls.get_titles_by_code(0)

    def test_get_titles_by_name(self):
        # Loop through the classifications, using the name to get the
        # corresponding entry from `self.tabls`. Test for equality between the
        # dictionaries (coerced from OrderedDict to dict as necessary, to avoid
        # ordering mismatches).
        for classification in self.expected_titles:
            name = classification['name']
            tabls_entry = dict(self.tabls.get_titles_by_name(name))

            assert tabls_entry == dict(classification), (
                print('Found: {}\nExpected: {}'.format(
                    tabls_entry, dict(classification))))

    def test_get_titles_by_name_not_found(self):
        with self.assertRaises(KeyError):
            self.tabls.get_titles_by_name('ABCD')

    def test_get_var(self):
        # Loop through the expected variable list and check that the entries
        # match those in the `Tabls` object. Entries coerced to dict (from
        # OrderedDict) as necessary.
        for entry in self.expected_variables:
            name = entry['name']

            assert dict(self.tabls.get_var(name)) == dict(entry), (
                print('Found: {}\nExpected: {}'.format(
                    dict(self.tabls.get_var(name)), dict(entry))))

    def test_get_var_not_found(self):
        with self.assertRaises(KeyError):
            self.tabls.get_var('ABCD')

    def test_variables_list(self):
        # Extract the variable names from the expected variables data, trim the
        # length of the full data in `self.tabls` and test for equality.
        expected_list = [x['name'] for x in self.expected_variables]
        tabls_list = self.tabls.variables[:len(expected_list)]

        assert tabls_list == expected_list, (
            print('Found: {}\nExpected: {}'.format(tabls_list, expected_list)))

    def test_titles_list(self):
        # As for `test_variables_list()`, but using the titles rather than the
        # variables
        expected_list = [x['name'] for x in self.expected_titles]
        tabls_list = self.tabls.titles[:len(expected_list)]

        assert tabls_list == expected_list, (
            print('Found: {}\nExpected: {}'.format(tabls_list, expected_list)))

    def test_undefined_classification(self):
        # E3ME only: Some variables have undefined classifications. Should
        #            throw a `TablsError` if accessed naively
        with self.assertRaises(TablsError):
            self.tabls.get_var('SR')

    def test_undefined_classification_caught(self):
        # E3ME only: `Tabls` still allows access for variables with undefined
        #            classifications, but the user has to do this explicitly
        try:
            self.tabls.get_var('SR')
        except TablsError as e:
            assert dict(e.entry) == {
                'name': 'SR',
                'desc': 'stockbuilding by sector of destination (disabled HP 27/06/08)',
                'units': 1,
                'dimcodes': (7, 1), 'dimnames': (None, 'RTI')}


class TestMDME3(TestE3ME):

    input_file = os.path.join(data_dir, 'tabls_mdm-e3.dat')
    expected_variables_file = os.path.join(data_dir, 'mdm-e3_variables.csv')
    expected_titles_file = os.path.join(data_dir, 'mdm-e3_titles.csv')

    def test_get_titles_by_code_duplicate(self):
        # Over-ride the base-class function just to check that `Tabls` finds
        # *no* duplicates in the MDM-E3 version of tabls.dat.
        codes = [x['code'] for x in self.expected_titles]

        assert len(codes) == len(set(codes)), (
            print('Found duplicates in: {}'.format(codes)))

    def test_undefined_classification(self):
        # Over-ride base-class test because all MDM-E3 variables have defined
        # classifications
        pass

    def test_undefined_classification_caught(self):
        # Over-ride base-class test because all MDM-E3 variables have defined
        # classifications
        pass


if __name__ == '__main__':
    unittest.main()
