# -*- coding: utf-8 -*-
"""
test_mre
========
Test suite for the MRE class.
"""

import os
import unittest
import warnings
import zipfile

import numpy as np

from celib import MRE
from celib.exceptions import DimensionError


current_dir = os.path.split(__file__)[0]


class TestRead(unittest.TestCase):

    def setUp(self):
        self.mf = MRE(os.path.join(current_dir, 'data', 'DA.mre'))

    def tearDown(self):
        pass

    def test_all_numerical(self):
        zf = zipfile.ZipFile(os.path.join(current_dir, 'data', 'DA.zip'))  # this has the whole databank
        for fn in zf.filelist:
            with zf.open(fn) as f:
                mt = np.loadtxt(f)
            vn, dim = fn.filename[:-4].split('_')

            # there are two basic access functions
            mremt = self.mf.get_var(vn, int(dim))
            mremt2 = self.mf.get_full_var(vn)[int(dim)]

            self.assertTrue(np.allclose(mremt, mt))
            self.assertTrue(np.allclose(mremt, mremt2))
        zf.close()

    def test_fused(self):
        """
            Testing a file where numbers are so large (or have that)
            many decimal points, that they 'fuse' together.
        """
        # mrep = read_mre(os.path.join(current_dir, 'data', 'fused.mre'), 'test', 0)['RGDP'][0]
        r = MRE(os.path.join(current_dir, 'data', 'fused.mre'))
        mrer = r.get_var('RGDP', 0)

        mrec = np.loadtxt(os.path.join(current_dir, 'data', 'fused.txt')) # canonical values

        #self.assertTrue(np.allclose(mrec, mrep))
        self.assertTrue(np.allclose(mrec, mrer))

    def test_asterisks(self):
        """Test MREs with asterisks in them"""
        pass # TODO

    # def test_index(self):
    #     ind = pd.read_csv('data/T_index.csv', index_col=0)
    #     ind_db = self.dbt.dfindex()
    #     # tests of unordered data
    #     self.assertCountEqual(ind.index, ind_db.index)
    #     self.assertCountEqual(ind.name, ind_db.name)

class TestWrite(unittest.TestCase):

    def test_write_data(self):
        """Test that the write method can reproduce an MRE file with the correct metadata (dimensions, variable names and description).

        Notes
        -----
        This test reads the contents of an existing MRE file
        (data/mre_write_example.mre) and writes them back out to another MRE
        file (data/mre_write_example2.mre). The contents of the two files
        should be identical.

        Because `get_full_var()` fills in missing matrices in three-dimensional
        variables with `None`, these are filtered out in this test.

        Output without filtering (`None` used in place of missing matrices):
          [Y1, Y2, None, None, Y5, Y6]

        Output with filtering (`None`s excluded when reading, writing and
        comparing):
          [Y1, Y2, Y5, Y6]  <- this is what gets written back as a test
        """
        # ---------------------------------------------------------------------
        # 1. Read the contents of the existing MRE file
        f = MRE(os.path.join(current_dir, 'data', 'mre_write_example.mre'))
        # the index of original mre: gives the variable names and details
        mred_index = f.index
        # Get the variables names and store them to a list
        varname = list(mred_index.keys())
        data = dict()
        comm = dict()
        # Extract data and comments
        for i in varname:
            # `get_full_var()` fills missing matrices in three-dimensional
            # variables with `None`: exclude these
            data[i] = [t for t in f.get_full_var(i) if t is not None]
            comm[i] = []
            for j in list(f.index[i].keys()):
                comm[i].append(mred_index[i][j]['comment'])

        # ---------------------------------------------------------------------
        # 2. Write the contents to a new MRE file
        output_filepath = os.path.join(current_dir, 'data', 'mre_write_example2.mre')
        if os.path.isfile(output_filepath):
            os.remove(output_filepath)

        mre2 = MRE()
        mre2.write(data, comm, output_filepath)

        # ---------------------------------------------------------------------
        # 3. Read the contents back in and compare with the original data
        f2 = MRE(output_filepath)
        mred2_index = f2.index
        varname2 = list(mred2_index.keys())
        data2 = dict()
        # Read in new data
        for i in varname2:
            # `get_full_var()` fills missing matrices in three-dimensional
            # variables with `None`: exclude these
            data2[i] = [t for t in f2.get_full_var(i) if t is not None]

        for i, j in [(i, j) for i in varname for j, k in enumerate(data[i])]:
            self.assertTrue(np.allclose(data[i][j], data2[i][j]))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            mre2.close()

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertEqual(str(w[0].message),
                             'As of version 0.4.1 of `celib`, `MRE.close()` is deprecated. '
                             'You can safely delete these calls from your script.')

    def test_2d_data(self):
        # Check that a single 2D array is written correctly
        output_file = os.path.join(current_dir, 'data', 'test_2d.mre')

        if os.path.isfile(output_file):
            os.remove(output_file)

        mre = MRE()
        mre.write({'XYZ': np.arange(12).reshape((3, 4))},
                  {'XYZ': 'Test matrix'},
                  output_file)

        expected = '''\
3  4  XYZ  Test matrix
         0.0000         1.0000         2.0000         3.0000
         4.0000         5.0000         6.0000         7.0000
         8.0000         9.0000        10.0000        11.0000
'''
        with open(output_file) as f:
            self.assertEqual(f.read(), expected)

    def test_3d_data(self):
        # Check that three-dimensional variables (lists of arrays) are
        # flattened correctly when written to an MRE file, with indexes
        # automatically generated and incremented
        output_file = os.path.join(current_dir, 'data', 'test_3d.mre')

        if os.path.isfile(output_file):
            os.remove(output_file)

        # ABC is a four-element list of 2D NumPy arrays, representing a 3D
        # variable of an IDIOM model
        data = {'ABC': [np.full((2, 3), x, dtype=float) for x in range(4)]}

        # The accompanying description is just a single string: the MRE class
        # should be able to expand this to: '? 01 Dummy data',
        # '? 02 Dummy data', '? 03 Dummy data', '? 04 Dummy data'
        comments = {'ABC': 'Dummy data'}

        # Write data
        mre = MRE()
        mre.write(data, comments, output_file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            mre.close()

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertEqual(str(w[0].message),
                             'As of version 0.4.1 of `celib`, `MRE.close()` is deprecated. '
                             'You can safely delete these calls from your script.')

        # Read data and check
        expected = '''\
2  3  ABC  ? 01 Dummy data
         0.0000         0.0000         0.0000
         0.0000         0.0000         0.0000
2  3  ABC  ? 02 Dummy data
         1.0000         1.0000         1.0000
         1.0000         1.0000         1.0000
2  3  ABC  ? 03 Dummy data
         2.0000         2.0000         2.0000
         2.0000         2.0000         2.0000
2  3  ABC  ? 04 Dummy data
         3.0000         3.0000         3.0000
         3.0000         3.0000         3.0000
'''
        with open(output_file) as f:
            self.assertEqual(f.read(), expected)

    def test_3d_data_missing_error(self):
        # Check that, in the current version of the MRE class (we may want to
        # change this in the future), a three-dimensional variable supplied as
        # a list of arrays, but with missing elements (one or more `None`
        # placeholders), raises a `ValueError` to signal incomplete data

        # Expected behaviour as of version 0.4.1: The class should fail to
        # write the specified output file in the event of a missing element.
        # This contrasts with the pre-0.4.1 treatment, which kept writing the
        # matrices as long as they were available, until it encountered a
        # missing element, stopping and then leaving an incomplete output file.

        # First delete any instances of the (intended) output file
        output_file = os.path.join(current_dir, 'data', 'test_3d_missing.mre')
        if os.path.isfile(output_file):
            os.remove(output_file)

        # Create the invalid input
        data = {'ABC': [np.full((2, 3), x, dtype=float) for x in range(4)]}
        data['ABC'][1] = None  # A missing matrix
        comments = {'ABC': 'Dummy data'}

        # Try to write the data, expecting an error
        mre = MRE()

        with self.assertRaises(ValueError):
            mre.write(data, comments, output_file)

        # Because of the error above, the intended output file shouldn't exist
        self.assertFalse(os.path.isfile(output_file))

    def test_basic_type_errors(self):
        # Check that argument types are correctly checked
        mre = MRE()

        # `data` (should be a dict)
        with self.assertRaises(TypeError):
            mre.write(None, {}, 'debug.mre')

        # `comm` (should be a dict)
        with self.assertRaises(TypeError):
            mre.write({}, None, 'debug.mre')

        # `data` *and* `comm` (should have identical keys)
        with self.assertRaises(ValueError):
            mre.write({'A': None}, {'B': None}, 'debug.mre')

        # `outfile` (should be a str)
        with self.assertRaises(TypeError):
            mre.write({'A': None}, {'A': None}, None)

        # If this test runs through, delete the debug output
        if os.path.isfile('debug.mre'):
            os.remove('debug.mre')

    def test_nonempty_object_error(self):
        # Check that attempting to write a file with a non-empty MRE object
        # fails

        # Delete any instances of the (intended) output file
        output_file = os.path.join(current_dir, 'data', 'test_nonempty_object_error.mre')
        if os.path.isfile(output_file):
            os.remove(output_file)

        # Create a new MRE object, reading in the contents of an existing file
        mre = MRE(os.path.join(current_dir, 'data', 'mre_write_example.mre'))

        # Try to write some data, expecting an error (because the object
        # already contains data)
        with self.assertRaises(RuntimeError):
            mre.write({'A': np.zeros((2, 3))}, {'A': 'Test data'}, output_file)

        # Because of the error above, the intended output file shouldn't exist
        self.assertFalse(os.path.isfile(output_file))

    def test_wrong_data_type(self):
        # Check that invalid data types (e.g. None) are caught
        mre = MRE()

        with self.assertRaises(TypeError):
            mre.write({'A': None}, {'A': 'Test data'}, 'debug.mre')

        # If this test runs through, delete the debug output
        if os.path.isfile('debug.mre'):
            os.remove('debug.mre')

    def test_ndarray_wrong_shape(self):
        # Check for a `DimensionError` when trying to write a non-2D array
        mre = MRE()

        with self.assertRaises(DimensionError):
            mre.write({'A': np.zeros((2, 3, 4))}, {'A': 'Test data'}, 'debug.mre')

        # If this test runs through, delete the debug output
        if os.path.isfile('debug.mre'):
            os.remove('debug.mre')

    def test_ndarray_wrong_comment_type(self):
        # Check that `write()` correctly catches non-string and non-list
        # variable types for arrays
        mre = MRE()

        with self.assertRaises(TypeError):
            mre.write({'A': np.zeros((2, 3))}, {'A': None}, 'debug.mre')

        # If this test runs through, delete the debug output
        if os.path.isfile('debug.mre'):
            os.remove('debug.mre')

    def test_ndarray_wrong_comment_type_list(self):
        # Check that `write()` correctly catches invalid list comments for
        # arrays
        mre = MRE()

        # Check lists of comments are of the wrong length
        with self.assertRaises(ValueError):
            mre.write({'A': np.zeros((2, 3))}, {'A': []}, 'debug.mre')

        with self.assertRaises(ValueError):
            mre.write({'A': np.zeros((2, 3))}, {'A': ['', '']}, 'debug.mre')

        # Check lists of comments are of the right length (1) but the contents
        # are of the wrong type
        with self.assertRaises(ValueError):
            mre.write({'A': np.zeros((2, 3))}, {'A': [None]}, 'debug.mre')

        # If this test runs through, delete the debug output
        if os.path.isfile('debug.mre'):
            os.remove('debug.mre')

    def test_3d_wrong_types(self):
        # Check that `write` raises exceptions for lists of data
        # (three-dimensional variables) that don't contain exclusively 2D
        # arrays
        mre = MRE()

        # Catch `None` as a ValueError (because it usually denotes a missing
        # variable element from a read operation)
        with self.assertRaises(ValueError):
            mre.write({'A': [np.zeros((2, 3)), np.ones((2, 3)), None]},
                      {'A': 'Test data'}, 'debug.mre')

        # Catch non-array elements
        with self.assertRaises(TypeError):
            mre.write({'A': [np.zeros((2, 3)), np.ones((2, 3)), 'Missing']},
                      {'A': 'Test data'}, 'debug.mre')

        # Catch non-2D array elements
        with self.assertRaises(DimensionError):
            mre.write({'A': [np.zeros((2, 3, 4)), np.ones((2, 3)), 'Missing']},
                      {'A': 'Test data'}, 'debug.mre')

        # If this test runs through, delete the debug output
        if os.path.isfile('debug.mre'):
            os.remove('debug.mre')

    def test_3d_wrong_comment_type(self):
        # Check that `write()` correctly catches non-string and non-list
        # variable types for three-dimensional variables
        mre = MRE()

        with self.assertRaises(TypeError):
            mre.write({'A': [np.zeros((2, 3)), np.ones((2, 3))]},
                      {'A': None}, 'debug.mre')

        # If this test runs through, delete the debug output
        if os.path.isfile('debug.mre'):
            os.remove('debug.mre')

    def test_3d_wrong_comment_type_list(self):
        # Check that `write()` correctly catches invalid list comments for
        # three-dimensional variables
        mre = MRE()

        # Check lists of comments are of the wrong length
        with self.assertRaises(ValueError):
            mre.write({'A': [np.zeros((2, 3)), np.ones((2, 3))]},
                      {'A': []}, 'debug.mre')

        with self.assertRaises(ValueError):
            mre.write({'A': [np.zeros((2, 3)), np.ones((2, 3))]},
                      {'A': ['', '', '']}, 'debug.mre')

        # Check lists of comments are of the right length (1 or 2) but the
        # contents are of the wrong type
        with self.assertRaises(TypeError):
            mre.write({'A': [np.zeros((2, 3)), np.ones((2, 3))]},
                      {'A': [None]}, 'debug.mre')

        with self.assertRaises(TypeError):
            mre.write({'A': [np.zeros((2, 3)), np.ones((2, 3))]},
                      {'A': ['', None]}, 'debug.mre')

        # If this test runs through, delete the debug output
        if os.path.isfile('debug.mre'):
            os.remove('debug.mre')


class TestOpenClose(unittest.TestCase):

    def test_empty_open_close(self):
        # Check that opening and closing an MRE file with no further action
        # raises no errors, but does produce a deprecation warning on calling
        # close(), because it's now redundant
        mre = MRE()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            mre.close()

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertEqual(str(w[0].message),
                             'As of version 0.4.1 of `celib`, `MRE.close()` is deprecated. '
                             'You can safely delete these calls from your script.')

    def test_empty_context_manager(self):
        # Check that a context manager with no action raises no errors or
        # warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            with MRE() as mre:
                pass

            self.assertEqual(len(w), 0)


if __name__ == '__main__':
    unittest.main()
