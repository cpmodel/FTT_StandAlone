# -*- coding: utf-8 -*-
"""
test_db1
========
Unit tests for the DB1 class.
"""

import os
import pickle
import unittest
import warnings
import zipfile
import shutil
import numpy as np
import pandas as pd

from celib import DB1


current_dir = os.path.split(__file__)[0]


class SetupDB1:
    """Utility methods for tests that manipulate DB1 databanks."""

    TEST_PATH = 'test.db1'

    def _delete_test_databank(self):
        if os.path.exists(self.TEST_PATH):
            os.remove(self.TEST_PATH)

    def setUp(self):
        self._delete_test_databank()

        with DB1(user='CE') as db1:
            db1.create(self.TEST_PATH)

    def tearDown(self):
        self._delete_test_databank()


class test_written_data(unittest.TestCase):

    def setUp(self):
        # Open databanks for reading
        self.dbu = DB1(os.path.join(current_dir, 'data', 'U.db1'))
        self.dbt = DB1(os.path.join(current_dir, 'data', 'T.db1'))

        # Create a new databank to test creation and for reading/writing
        self.new_test_databank_path = os.path.join(current_dir, 'data', 'new_test_databank.db1')

        db1_test = DB1()
        db1_test.user = 'CE'
        db1_test.create(self.new_test_databank_path)
        db1_test.close()

    def tearDown(self):
        self.dbu.close()
        self.dbt.close()
        # Delete the temporary databank
        if os.path.exists(self.new_test_databank_path):
            os.remove(self.new_test_databank_path)

    def test_rti(self):
        rti = ('1 London, 2 South East, 3 East of England, 4 South West, ' +
               '5 West Midlands, 6 East Midlands, 7 Yorkshire & the Humbe, ' +
               '8 North West, 9 North East, 10 Wales, 11 Scotland, ' +
               '12 Northern Ireland, 13 Extra Regio').split(', ')

        rti_db = self.dbu.get('RTI')
        for j, r in enumerate(rti):
            # not the same as self.assertCountEqual(rti, rti_db)
            # which only does unordered checks
            self.assertEqual(r, rti_db[j])

    def test_arrays(self):
        """
        # We pickled the whole databank into a dictionary
        # of code -> value, let's check it against that.
        # (only lists were pickled)
            import pickle
            from CEStd.db1 import DB1

            udb = DB1('data/U.db1')

            dt = dict()
            for c in udb.codes:
                if udb.vindex[c]['vtype'] != 4: continue
                dt[c] = udb.get(c)

            with open('data/U.pickle', 'wb') as f:
                pickle.dump(dt, f)

            udb.close()


        """
        with open(os.path.join(current_dir, 'data', 'U.pickle'), 'rb') as f:
            dt = pickle.load(f)

        for k, pcd in dt.items():
            dbdt = self.dbu.get(k)

            self.assertEqual(len(dbdt), len(pcd))
            for v1, v2 in zip(dbdt, pcd):
                self.assertEqual(v1, v2)

    def test_all_numerical(self):
        """
        We exported the whole of T.db1 into a zip of CSVs,
        with the index in _index.csv
        Let's check the live db1 data against it.
            from CEStd.db1 import DB1

            fn = 'data/T.db1'
            tdb = DB1(fn)

            for c in tdb.codes:
                tdb.get_df(c).to_csv('tmp/%d.csv' % c)

            tdb.close()
        """
        zf = zipfile.ZipFile(os.path.join(current_dir, 'data', 'T_db.zip'))  # this has the whole databank
        for fn in zf.filelist:
            if fn.filename == '_index.csv':
                continue
            with zf.open(fn) as f:
                df = pd.read_csv(f, index_col=0)
            dbv = self.dbt.get(int(fn.filename[:-4]))  # .flatten()
            self.assertTrue(np.allclose(df.values, dbv))
        zf.close()

    def test_index(self):
        zf = zipfile.ZipFile(os.path.join(current_dir, 'data', 'T_db.zip'))  # this has the whole databank
        ind = pd.read_csv(zf.open('_index.csv'), index_col=0)
        ind_db = self.dbt.index
        # tests of unordered data
        self.assertCountEqual(ind.index, ind_db.index)
        self.assertCountEqual(ind.name, ind_db.name)

        zf.close()

    def test_write_list_of_str(self):
        # Test that a list of strings can be written and then read back in
        test_list = ["a", "b", "c"]

        # Open the databank for writing and add the list
        with DB1(self.new_test_databank_path, 'write') as db1:
            db1.user = 'CE'
            db1.write_list('test_strings', 101, test_list, "Test list of strings")

        # Open the databank for reading and check the contents match what we
        # put to the databank
        with DB1(self.new_test_databank_path) as db1:
            self.assertEqual(test_list, db1.get('test_strings'))

    def test_write_list_of_int(self):
        # Test that a list of integers can be written and then read back in
        test_list = [1, 2, 3]

        # Open the databank for writing and add the list
        with DB1(self.new_test_databank_path, 'write') as db1:
            db1.user = 'CE'
            db1.write_list('test_integers', 102, test_list, "Test list of integers")

        # Open the databank for reading and check the contents match what we
        # put to the databank
        with DB1(self.new_test_databank_path) as db1:
            self.assertEqual(test_list, db1.get('test_integers'))

    def test_write_list_of_float(self):
        # Test that a list of floats can be written and then read back in
        test_list = [1.0, 2.0, 3.0]

        # Open the databank for writing and add the list
        with DB1(self.new_test_databank_path, 'write') as db1:
            db1.user = 'CE'
            db1.write_list('test_floats', 103, test_list, "Test list of floats")

        # Open the databank for reading and check the contents match what we
        # put to the databank
        with DB1(self.new_test_databank_path) as db1:
            self.assertEqual(test_list, db1.get('test_floats'))

    def test_write_list_error_mixed_types(self):
        # DB1.write_list() should raise a TypeError if the list has a mix of
        # types e.g. str, int
        with DB1(self.new_test_databank_path, 'write') as db1:
            db1.user = 'CE'

            with self.assertRaises(TypeError):
                test_list = ["a", "b", 1]
                db1.write_list('test_list', 104, test_list, "Test list of mixed types")


class TestProperties(SetupDB1, unittest.TestCase):

    def setUp(self):
        # Slightly modified setup method to test alternative user-setting
        # mechanism
        self._delete_test_databank()

        db1 = DB1(user='CE')
        db1.create(self.TEST_PATH)
        db1.close()

    def test_start_year(self):
        expected = np.arange(12, dtype=float).reshape((3, -1))

        # Write two matrices with different start years
        with DB1(self.TEST_PATH, 'write', user='CE') as db1:
            self.assertEqual(db1.start_year, 0)

            # Set a start year using the old method (suppress warnings
            # accordingly)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                db1.set_start_year(2019)

            # Write the data
            db1.write_matrix('test', 1, expected)

            # Change the start year using the new method and write
            db1.start_year = 1985
            db1.write_matrix('test', 2, expected)

        # Check the results and the start years
        with DB1(self.TEST_PATH, 'read') as db1:
            self.assertEqual(db1.get_start_year(1), 2019)

            result = db1[1]
            self.assertEqual(result.shape, expected.shape)
            self.assertTrue(np.allclose(result, expected))

            self.assertEqual(db1.get_start_year(2), 1985)

            result = db1[2]
            self.assertEqual(result.shape, expected.shape)
            self.assertTrue(np.allclose(result, expected))

    def test_gsyear(self):
        # Test deprecated `gsyear` property
        expected = np.arange(12, dtype=float).reshape((3, -1))

        # Write two matrices with different start years
        with DB1(self.TEST_PATH, 'write', user='CE') as db1:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                self.assertEqual(db1.gsyear, 0)

                db1.gsyear = 2019
                db1.write_matrix('test', 1, expected)

                db1.gsyear = 1985
                db1.write_matrix('test', 2, expected)

        # Check the results and the start years
        with DB1(self.TEST_PATH, 'read') as db1:
            self.assertEqual(db1.get_start_year(1), 2019)

            result = db1[1]
            self.assertEqual(result.shape, expected.shape)
            self.assertTrue(np.allclose(result, expected))

            self.assertEqual(db1.get_start_year(2), 1985)

            result = db1[2]
            self.assertEqual(result.shape, expected.shape)
            self.assertTrue(np.allclose(result, expected))

    def test_header(self):
        # Check databank header is accessible but not modifiable
        with DB1(self.TEST_PATH, 'read') as db1:
            # User-facing and internal header variables match
            header = db1.header
            self.assertEqual(header, db1._header)

            # Changing a copy of the header leaves the databank's version of
            # the header unchanged
            header['user_modified'] = 'XY'

            self.assertEqual(header.keys(), db1._header.keys())

            for k in header:
                if k == 'user_modified':
                    self.assertNotEqual(header[k], db1._header[k])
                else:
                    self.assertEqual(header[k], db1._header[k])

            # User-facing databank header is read-only
            # TODO: See if there's a way to implement some kind of
            #       warning/error for this case: (how) does pandas do it?
            db1.header['user_modified'] = 'XY'
            self.assertNotEqual(db1.header['user_modified'], 'XY')
            self.assertEqual(db1.header, db1._header)


class TestCreate(SetupDB1, unittest.TestCase):

    def setUp(self):
        # Modified setup that doesn't create an initial databank
        self._delete_test_databank()

    def test_create_and_write(self):
        db1 = DB1()

        # Set the username the old way but suppress the expected
        # `DeprecationWarning`
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            db1.set_user('CE')

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))

        # Create a new databank
        db1.create(self.TEST_PATH)

        # Write a matrix to the databank
        expected = np.arange(12, dtype=float).reshape((3, -1))
        db1.write_matrix('test', 1, expected)

        # Close the databank
        db1.close()

        # Reopen the databank and retrieve the data
        with DB1(self.TEST_PATH) as db1:
            result = db1[1]

        # Check stored results matched the original
        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(np.allclose(result, expected))

    def test_no_username_error(self):
        db1 = DB1()

        # No username specified: should throw an exception
        with self.assertRaises(RuntimeError):
            db1.create(self.TEST_PATH)


class TestIndex(SetupDB1, unittest.TestCase):

    def test_empty_databank_has_empty_index(self):
        # Should be possible to get the index of a databank as a DataFrame,
        # even if the databank is empty: should return an empty DataFrame
        with DB1(self.TEST_PATH) as db1:
            self.assertEqual(len(db1.index.index), 0)


class TestErrors(unittest.TestCase):

    def test_init_invalid_mode(self):
        # DB1 should only support 'read' and 'write' modes
        with self.assertRaises(ValueError):
            db1 = DB1(mode='invalid mode')

    def test_init_invalid_username(self):
        # Should fail if username not a string
        with self.assertRaises(TypeError):
            db1 = DB1(user=0)

        # Should fail if username is an empty string
        with self.assertRaises(ValueError):
            db1 = DB1(user='')

        # Should fail if username is too long
        with self.assertRaises(ValueError):
            db1 = DB1(user='123456789')

    def test_invalid_username_change(self):
        db1 = DB1()

        # Check username is still `None`, but using the old-style getter method
        # (suppress the accompanying warning to avoid alarming the developer
        # when running unit tests)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertIsNone(db1.get_user())

        # Should fail if username not a string
        with self.assertRaises(TypeError):
            db1.user = 0

        # Should fail if username is an empty string
        with self.assertRaises(ValueError):
            db1.user = ''

        # Should fail if username is too long
        with self.assertRaises(ValueError):
            db1.user = '123456789'

    def test_invalid_start_year_type(self):
        # Start year must be an integer
        db1 = DB1()

        with self.assertRaises(TypeError):
            db1.start_year = '2019'


class TestCustomCaller(SetupDB1, unittest.TestCase):

    def test_custom_caller(self):
        # Check that the user can override the default caller information

        # Write custom caller information to the databank
        with DB1(self.TEST_PATH, 'write', user='CE') as db1:
            db1.write_matrix('test_matrix', 1000, np.arange(12).reshape((3, -1)), 'Test matrix', caller='Custom caller for matrix')
            db1.write_list('test_list', 2000, list(range(10)), 'Test list', caller='Custom caller for list')

        # Open the databank for reading and check that the caller values are as
        # expected
        with DB1(self.TEST_PATH) as db1:
            index = db1.index

            self.assertEqual(index.loc[1000, 'caller'], 'Custom caller for matrix')
            self.assertEqual(index.loc[2000, 'caller'], 'Custom caller for list')


class TestCondense(unittest.TestCase):
    def test_condense(self):
        #Create copy of T.db1 to use as test case for condense
        temp_path = os.path.join(current_dir, 'data', 'T_temp.db1')
        shutil.copy(os.path.join(current_dir, 'data', 'T.db1'), temp_path)

        db1_t = DB1(os.path.join(current_dir, 'data', 'T.db1'))

        with DB1(temp_path,"write") as db1:
            db1.user = 'CE'
            db1.condense()

        with DB1(temp_path) as db1:
            #Assert that same number of unscratched variable codes are available
            self.assertEqual(db1.codes,db1_t.codes)
            #Assert date and user created and modified are same but date condensed is different
            self.assertEqual(db1._header['time_modified'], db1_t._header['time_modified'])
            self.assertEqual(db1._header['time_created'], db1_t._header['time_created'])

            self.assertEqual(db1._header['user_modified'], db1_t._header['user_modified'])
            self.assertEqual(db1._header['user_created'], db1_t._header['user_created'])

            # CT: Better/clearer to use `self.assertNotEqual()`?
            #     See: https://docs.python.org/3.6/library/unittest.html#unittest.TestCase.debug
            self.assertTrue(db1._header['time_condensed'] != db1_t._header['time_condensed'])
            self.assertTrue(db1._header['user_condensed'] != db1_t._header['user_condensed'])

            # Check number of variables reported
            self.assertEqual(db1._header['number_of_live_variables'],db1_t._header['number_of_live_variables'])

            #Confirm no scratched variables
            self.assertEqual(db1._header['number_of_live_variables'],db1._header['number_of_variables'])

            #Check index for user and date written is the same
            self.assertEqual(db1.index['user'].tolist(),db1_t.index['user'].tolist())
            self.assertEqual(db1.index['date'].tolist(),db1_t.index['date'].tolist())

        if os.path.exists(temp_path):
            os.remove(temp_path)

        db1_t.close()


if __name__ == '__main__':
    unittest.main()
