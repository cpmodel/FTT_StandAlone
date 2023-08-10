# -*- coding: utf-8 -*-
"""
test_fillmat
============
Tests for functions in the fillmat collection.
"""

import unittest
import warnings

import numpy as np
from pandas import DataFrame, Series

import celib.fillmat as fm


class TestFillMat(unittest.TestCase):

    def setUp(self):
        """Define common variables needed for the tests"""
        self.X = np.array([[1, np.nan, 3], [2, np.nan, np.nan], [2, np.nan, 4],
                           [3, np.nan, np.nan], [4, np.nan, np.nan]])

        self.Y = np.array([[1, 2, 3], [2, 4, 6], [2, 4, 4], [3, 6, 7], [4, 5, 5]])
        self.Z = np.array([7, 8, 9, 10, 11])

        self.A = np.vstack((self.Y.copy(), self.Y.copy()+5, self.Y.copy()+11))

        self.B = np.array([[1, 2, np.nan], [2, np.nan, np.nan], [np.nan, np.nan, 4],
                           [3, np.nan, np.nan], [np.nan, 5, np.nan]])
        self.B = np.vstack((self.B.copy(), self.B.copy()+4.5, self.B.copy()+10))

        self.C = np.array([[1, 2, 3.1135], [2, 2.5149, 3.529], [2.5755, 3.1623, 4],
                           [3.3166, 3.9764, 4.6508], [4.271, 5, 5.4076],
                           [5.5, 6.5, 6.2875], [6.5, 7.348, 7.3105],
                           [7.4137, 8.3065, 8.5], [8.4558, 9.3902, 9.3921],
                           [9.6443, 10.615, 10.378], [11, 12, 11.467],
                           [12, 12.688, 12.67], [12.49, 13.416, 14],
                           [13, 14.186, 15.868], [15.835, 15, 17.986]])

        self.D = np.array([1, 2, 2, 3, 3.8297, 5.5, 6.5, 6.5312, 7.5, 8.3432,
                           11, 12, 12.036, 13, np.nan])

        self.E = self.C.copy()
        self.E[6:, :] = np.nan

        self.F = self.C.copy()
        self.F[:9, :] = np.nan

        self.G = self.C.copy()
        self.G[5:11, :] = np.nan

    def test_convert_to_array(self):
        """Test convert_to_np_arrays() returns np array as the data type"""

        array = self.X.copy()
        series = Series(self.D.copy())
        frame = DataFrame(self.X.copy())

        self.assertTrue(isinstance(fm.convert_to_np_arrays(array), np.ndarray))
        self.assertTrue(isinstance(fm.convert_to_np_arrays(series), np.ndarray))
        self.assertTrue(isinstance(fm.convert_to_np_arrays(frame), np.ndarray))

    def test_restore_type(self):
        """Test restore_original_type() returns the correct data type"""

        array = self.X.copy()
        series = Series(self.D.copy())
        frame = DataFrame(self.X.copy())

        self.assertTrue(isinstance(fm.restore_original_type(array, array), np.ndarray))
        self.assertTrue(isinstance(fm.restore_original_type(series.values, series), Series))
        self.assertTrue(isinstance(fm.restore_original_type(frame.values, frame), DataFrame))

    def test_share(self):
        """Test fill_with_share() returns the same results as the Ox equivalent"""

        value = fm.fill_with_shares(self.X, self.Z, self.Y)

        expected = np.array([[1, 3, 3], [2, np.nan, np.nan], [2, np.nan, 4],
                             [3, np.nan, np.nan], [4, np.nan, np.nan]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected) | np.isnan(value)))

    def test_share_no_change(self):
        """Test that a complete variable is left unchanged."""
        value = fm.fill_with_shares(self.Y, self.Z, self.Y)

        self.assertTrue(np.array_equal(value, self.Y))

    def test_growth(self):
        """Test fill_with_growth_rates() returns the same results as the Ox equivalent"""

        fillvar = np.array([[1, 2, np.nan], [2, np.nan, np.nan], [np.nan, np.nan, 4],
                            [3, np.nan, np.nan], [np.nan, 5, np.nan]])

        value = fm.fill_with_growth_rates(fillvar, self.Y)

        expected = np.array([[1, 2, 3], [2, np.nan, 6], [np.nan, np.nan, 4],
                             [3, np.nan, 7], [4, 5, 5]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected) | np.isnan(value)))

    def test_extrapolate(self):
        """Test extrapolate() returns the same results as the Ox equivalent"""

        fillvar = np.array([[1, np.nan, np.nan], [2, np.nan, np.nan], [np.nan, 2, 0],
                            [3, np.nan, np.nan], [np.nan, 5, np.nan]])

        value = fm.extrapolate(fillvar)

        expected = np.array([[1, 0.8, 0], [2, 1.2649, 0], [np.nan, 2, 0],
                             [3, np.nan, 0], [4.3267, 5, 0]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_extrapolate_zero(self):
        """Test extrapolate() returns the expected values where the first year
        (when filling backward) or the last year (when filling forward) is zero
        (should be filled with zeros)"""

        fillvar = np.array([[1, np.nan, np.nan], [2, np.nan, np.nan], [np.nan, 2, 0],
                            [0, np.nan, np.nan], [np.nan, 5, 1]])

        value = fm.extrapolate(fillvar)

        # Modify the expected values where there should be zeros
        expected = np.array([[1, 0.8, 0], [2, 1.2649, 0], [np.nan, 2, 0],
                             [0, np.nan, np.nan], [0, 5, 1]])

        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_extrapolate_zerobase(self):
        """Test interpolate() returns the expected values where the first year
        (when filling forward) or the last year (when filling backward) is zero
        (so growth rates cannot be used)"""

        fillvar = np.array([[0, np.nan, np.nan], [2, np.nan, np.nan], [np.nan, 2, 0],
                            [3, 1, np.nan], [np.nan, 0, np.nan]])

        value = fm.extrapolate(fillvar)

        # Modify the expected values where there should be zeros
        expected = np.array([[0, 8, 0], [2, 4, 0], [np.nan, 2, 0],
                             [3, 1, 0], [3.6742, 0, 0]])

        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_interpolate(self):
        """Test interpolate(), without the optional growth argument, returns the same results as the Ox equivalent"""
        value = fm.interpolate(self.B)
        expected = np.array([[1, 2, np.nan], [2, 2.5149, np.nan], [2.4495, 3.1623, 4],
                             [3, 3.9764, 4.6508], [4.062, 5, 5.4076], [5.5, 6.5, 6.2875],
                             [6.5, 7.1469, 7.3105], [6.9821, 7.8581, 8.5],
                             [7.5, 8.6401, 9.3921], [9.083, 9.5, 10.378],
                             [11, 12, 11.467], [12, 12.688, 12.67], [12.49, 13.416, 14],
                             [13, 14.186, np.nan], [np.nan, 15, np.nan]])


        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_interpolate_with_growth_option(self):
        """Test interpolate(), with the optional growth argument, returns the same results as the Ox equivalent"""
        value = fm.interpolate(self.B, self.A)
        expected = np.array([[1, 2, np.nan], [2, 4, np.nan], [2, 4, 4],
                             [3, 6, 6.9204], [3.8297, 5, 4.887], [5.5, 6.5, 7.7303],
                             [6.5, 8.4049, 10.508], [6.5312, 8.453, 8.5],
                             [7.5, 10.391, 11.307], [8.3432, 9.5, 9.3998],
                             [11, 12, 13.129], [12, 13.9, 15.904], [12.036, 13.954, 14],
                             [13, 15.876, np.nan], [np.nan, 15, np.nan]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_interpolate_linear(self):
        """Test interpolate(), without the optional growth argument, returns the expected values where the first or last year of data is zero (so growth rates cannot be used)"""
        fillvar = self.B.copy()
        fillvar[2, 2] = 0

        value = fm.interpolate(fillvar)
        expected = np.array([[1, 2, np.nan], [2, 2.5149, np.nan], [2.4495, 3.1623, 0],
                             [3, 3.9764, 1.7], [4.062, 5, 3.4], [5.5, 6.5, 5.1],
                             [6.5, 7.1469, 6.8], [6.9821, 7.8581, 8.5],
                             [7.5, 8.6401, 9.3921], [9.083, 9.5, 10.378],
                             [11, 12, 11.467], [12, 12.688, 12.67], [12.49, 13.416, 14],
                             [13, 14.186, np.nan], [np.nan, 15, np.nan]])

        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_interpolate_linear_with_growth_option(self):
        """Test interpolate(), with the optional growth argument, returns the expected values where the first or last year of data is zero (so growth rates cannot be used)"""
        fillvar = self.B.copy()
        fillvar[2, 2] = 0

        value = fm.interpolate(fillvar, self.A)
        expected = np.array([[1, 2, np.nan], [2, 4, np.nan], [2, 4, 0],
                             [3, 6, 1.7], [3.8297, 5, 3.4], [5.5, 6.5, 5.1],
                             [6.5, 8.4049, 6.8], [6.5312, 8.453, 8.5],
                             [7.5, 10.391, 11.307], [8.3432, 9.5, 9.3998],
                             [11, 12, 13.129], [12, 13.9, 15.904], [12.036, 13.954, 14],
                             [13, 15.876, np.nan], [np.nan, 15, np.nan]])

        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_fill(self):
        """Test that fill_all_gaps(), without the optional argument, returns the same results as the Ox equivalent"""
        fillvar = self.B.copy()
        fillvar[[3, 8, 9], :] = np.nan

        # Fill with and without optional variable
        value = fm.fill_all_gaps(fillvar)
        expected = np.array([[1, 2, 3.1135], [2, 2.5149, 3.529], [2.5755, 3.1623, 4],
                             [3.3166, 3.9764, 4.6508], [4.271, 5, 5.4076],
                             [5.5, 6.5, 6.2875], [6.5, 7.348, 7.3105],
                             [7.4137, 8.3065, 8.5], [8.4558, 9.3902, 9.3921],
                             [9.6443, 10.615, 10.378], [11, 12, 11.467],
                             [12, 12.688, 12.67], [12.49, 13.416, 14],
                             [13, 14.186, 15.868], [15.835, 15, 17.986]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_fill_with_option(self):
        """Test that fill_all_gaps(), with the optional argument, returns the same results as the Ox equivalent"""
        fillvar = self.B.copy()
        fillvar[[3, 8, 9], :] = np.nan

        value = fm.fill_all_gaps(fillvar, self.A, self.A[:, 2]*2)
        expected = np.array([[1, 2, 3], [2, 4, 6], [1.957, 4, 4],
                             [2.8723, 6, 6.9204], [3.7473, 5, 4.887],
                             [5.5, 6.5, 7.7303], [6.5, 8.3472, 10.508],
                             [6.4791, 8.3373, 8.5], [7.3808, 10.178, 11.307],
                             [8.2767, 9.2417, 9.3998], [11, 12, 13.129],
                             [12, 13.9, 15.904], [12.036, 13.954, 14],
                             [13, 15.876, 16.8], [13.929, 15, 14.933]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_restforw_missing_consecutive_years(self):
        """Test restricted_fill_forward(), with missing data in consecutive years, returns the same results as the Ox equivalent"""

        # First test for when data are missing for consecutive years
        value = fm.restricted_fill_forward(self.E, self.D, self.C)

        expected = np.array([[1, 2, 3.1135], [2, 2.5149, 3.529], [2.5755, 3.1623, 4],
                             [3.3166, 3.9764, 4.6508], [4.271, 5, 5.4076],
                             [5.5, 6.5, 6.2875], [6.6226, 7.5192, 7.4707],
                             [6.7045, 7.4057, 7.6059], [7.7937, 8.5508, 8.593],
                             [8.7988, 9.5575, 9.3849], [11.653, 12.712, 12.21],
                             [12.79, 13.534, 13.576], [12.51, 13.412, 14.097],
                             [13.033, 14.197, 15.995], [np.nan, np.nan, np.nan]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_restforw_missing_nonconsecutive_years(self):
        """Test restricted_fill_forward(), with missing data in non-consecutive years, returns the same results as the Ox equivalent"""
        fillvar = self.C.copy()
        fillvar[[0, 1, 8, 11, 12, 13, 14], :] = np.nan

        value = fm.restricted_fill_forward(fillvar, self.D, self.C)
        expected = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
                             [2.57551, 3.16228, 4], [3.31662, 3.97635, 4.65084],
                             [4.271, 5, 5.40759], [5.5, 6.5, 6.28746],
                             [6.5, 7.34796, 7.3105], [7.41367, 8.30655, 8.5],
                             [np.nan, np.nan, np.nan], [9.64435, 10.6152, 10.3777],
                             [11, 12, 11.4669], [12.0737, 12.7762, 12.7504],
                             [11.8086, 12.6603, 13.2431], [12.3042, 13.402, 15.0273],
                             [np.nan, np.nan, np.nan]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_restback_missing_consecutive_years(self):
        """Test restricted_fill_backward(), with missing data in consecutive years, returns the same results as the Ox equivalent"""
        value = fm.restricted_fill_backward(self.F, self.D, self.C)
        expected = np.array([[0.58149, 1.283, 1.8076], [1.8461, 2.253, 3.2451],
                             [2.0144, 2.3505, 2.9794], [3.0795, 3.6054, 4.3314],
                             [4.1061, 4.7142, 5.2428], [6.0226, 7.1843, 6.9898],
                             [7.2596, 8.299, 8.3101], [7.3466, 8.2226, 8.414],
                             [8.5457, 9.4991, 9.4961], [9.6443, 10.615, 10.378],
                             [11, 12, 11.467], [12, 12.688, 12.67], [12.49, 13.416, 14],
                             [13, 14.186, 15.868], [15.835, 15, 17.986]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_restback_missing_nonconsecutive_years(self):
        """Test restricted_fill_backward(), with missing data in non-consecutive years, returns the same results as the Ox equivalent"""
        fillvar = self.C.copy()
        fillvar[[1, 2, 3, 4, 8, 12, 13, 14], :] = np.nan
        fillvar[0, 1] = np.nan

        value = fm.restricted_fill_backward(fillvar, self.D, self.C)
        expected = np.array([[1, 0, 3.11348], [1.68366, 2.04016, 2.92616],
                             [1.83485, 2.12774, 2.6874], [2.80911, 3.26369, 3.90218],
                             [3.74587, 4.26656, 4.72133], [5.5, 6.5, 6.28746],
                             [6.5, 7.34796, 7.3105], [7.41367, 8.30655, 8.5],
                             [np.nan, np.nan, np.nan], [9.64435, 10.6152, 10.3777],
                             [11, 12, 11.4669], [12, 12.6885, 12.6703],
                             [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
                             [np.nan, np.nan, np.nan]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_restinter_weighted(self):
        """Test restricted_interpolation() with weighted method returns the same results as the Ox equivalent"""
        # Expect the following warnings:
        # 'No data are missing.' (x16)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            value = fm.restricted_interpolation(self.G, self.D, self.C)

            self.assertEqual(len(w), 16)
            for warning in w:
                self.assertTrue(issubclass(warning.category, UserWarning))
                self.assertEqual(str(warning.message), 'No data are missing.')

        expected = np.array([[1, 2, 3.1135], [2, 2.5149, 3.529], [2.5755, 3.1623, 4],
                             [3.3166, 3.9764, 4.6508], [4.271, 5, 5.4076],
                             [6.0506, 7.2596, 7.2049], [7.102, 8.1584, 8.3165],
                             [7.015, 7.8251, 8.1785], [7.9382, 8.7742, 8.9498],
                             [8.7148, 9.5137, 9.4612], [11.247, 12.307, 11.822],
                             [12, 12.688, 12.67], [12.49, 13.416, 14],
                             [13, 14.186, 15.868], [15.835, 15, 17.986]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_restinter_unweighted(self):
        """Test restricted_interpolation() with unweighted/simple method returns the same results as the Ox equivalent"""
        # Expect the following warnings:
        # 'No data are missing.' (x16)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            value = fm.restricted_interpolation(self.G, self.D, self.C, weight=False)

            self.assertEqual(len(w), 16)
            for warning in w:
                self.assertTrue(issubclass(warning.category, UserWarning))
                self.assertEqual(str(warning.message), 'No data are missing.')

        expected = np.array([[1, 2, 3.1135], [2, 2.5149, 3.529], [2.5755, 3.1623, 4],
                             [3.3166, 3.9764, 4.6508], [4.271, 5, 5.4076],
                             [5.6698, 6.7502, 6.6816], [6.829, 7.8024, 7.9433],
                             [6.9209, 7.7112, 8.0508], [8.0471, 8.9056, 9.095],
                             [9.0853, 9.955, 9.9359], [12.037, 13.264, 12.902],
                             [12, 12.688, 12.67], [12.49, 13.416, 14],
                             [13, 14.186, 15.868], [15.835, 15, 17.986]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_restinter_random_weighted(self):
        """Test restricted_interpolation() with random gaps and the weighted method returns the same results as the Ox equivalent"""
        fillvar = self.C.copy()
        fillvar[[0, 12, 14], :] = np.nan
        fillvar[5:11, :] = np.nan

        # Expect the following warnings:
        # 'First year to be filled is greater than last year of total. Variable cannot be filled.' (x12)
        # 'Total contains missing values in first year to be filled. Variable cannot be filled.' (x1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            value = fm.restricted_interpolation(fillvar, self.D, self.C)

            self.assertEqual(len(w), 13)

            for warning in w:
                self.assertTrue(issubclass(warning.category, UserWarning))

            for i in range(12):
                self.assertEqual(str(w[i].message),
                                 'First year to be filled is greater than last year of total. Variable cannot be filled.')

            self.assertEqual(str(w[-1].message),
                             'Total contains missing values in first year to be filled. Variable cannot be filled.')

        expected = np.array([[np.nan, np.nan, np.nan], [2, 2.51487, 3.52901],
                             [2.57551, 3.16228, 4], [3.31662, 3.97635, 4.65084],
                             [4.271, 5, 5.40759], [6.05058, 7.2596, 7.20494],
                             [7.10204, 8.15838, 8.31654], [7.01505, 7.82509, 8.17849],
                             [7.93819, 8.7742, 8.9498], [8.71481, 9.51373, 9.46117],
                             [11.2474, 12.3065, 11.8223], [12, 12.6885, 12.6703],
                             [12.1069, 12.9877, 13.5706], [13, 14.1861, 15.8685],
                             [np.nan, np.nan, np.nan]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_restinter_random_unweighted(self):
        """Test restricted_interpolation() with random gaps and the unweighted/simple method returns the same results as the Ox equivalent"""
        fillvar = self.C.copy()
        fillvar[[0, 12, 14], :] = np.nan
        fillvar[5:11, :] = np.nan

        # Expect the following warnings:
        # 'First year to be filled is greater than last year of total. Variable cannot be filled.' (x12)
        # 'Total contains missing values in first year to be filled. Variable cannot be filled.' (x1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            value = fm.restricted_interpolation(fillvar, self.D, self.C, weight=False)

            self.assertEqual(len(w), 13)

            for warning in w:
                self.assertTrue(issubclass(warning.category, UserWarning))

            for i in range(12):
                self.assertEqual(str(w[i].message),
                                 'First year to be filled is greater than last year of total. Variable cannot be filled.')

            self.assertEqual(str(w[-1].message),
                             'Total contains missing values in first year to be filled. Variable cannot be filled.')

        expected = np.array([[np.nan, np.nan, np.nan], [2, 2.51487, 3.52901],
                             [2.57551, 3.16228, 4], [3.31662, 3.97635, 4.65084],
                             [4.271, 5, 5.40759], [5.66977, 6.75024, 6.68162],
                             [6.82895, 7.80237, 7.94333], [6.92088, 7.71121, 8.05084],
                             [8.04712, 8.90555, 9.09501], [9.08531, 9.95505, 9.93586],
                             [12.0372, 13.2643, 12.9018], [12, 12.6885, 12.6703],
                             [12.1069, 12.9877, 13.5706], [13, 14.1861, 15.8685],
                             [np.nan, np.nan, np.nan]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_targetforw(self):
        """Test target_forward() returns the same results as the Ox equivalent"""

        value = fm.target_forward(self.E, 5)[0]

        expected = np.array([[1, 2, 3.1135], [2, 2.5149, 3.529], [2.5755, 3.1623, 4],
                             [3.3166, 3.9764, 4.6508], [4.271, 5, 5.4076],
                             [5.5, 6.5, 6.2875], [7.7346, 8.2279, 7.2364],
                             [10.877, 10.415, 8.3285], [15.296, 13.184, 9.5855],
                             [21.511, 16.689, 11.032], [30.25, 21.125, 12.697],
                             [42.54, 26.741, 14.613], [59.823, 33.849, 16.819],
                             [84.128, 42.848, 19.357], [118.31, 54.238, 22.279]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_targetback(self):
        """Test target_backward() returns the same results as the Ox equivalent"""
        value = fm.target_backward(self.F, 5)[0]

        expected = np.array([[3.9503, 5.6968, 3.8565], [4.3621, 6.1047, 4.3049],
                             [4.8169, 6.5418, 4.8054], [5.3192, 7.0102, 5.3641],
                             [5.8737, 7.5121, 5.9877], [6.4861, 8.05, 6.6839],
                             [7.1624, 8.6264, 7.461], [7.9091, 9.244, 8.3285],
                             [8.7338, 9.9059, 9.2968], [9.6443, 10.615, 10.378],
                             [11, 12, 11.467], [12, 12.688, 12.67], [12.49, 13.416, 14],
                             [13, 14.186, 15.868], [15.835, 15, 17.986]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))

    def test_targetinter(self):
        """Test target_interpolation() returns the same results as the Ox equivalent"""
        value = fm.target_interpolation(self.G, 3)[0]

        expected = np.array([[1, 2, 3.11348], [2, 2.51487, 3.52901],
                             [2.57551, 3.16228, 4], [3.31662, 3.97635, 4.65084],
                             [4.271, 5, 5.40759], [6.7142, 7.68314, 6.32895],
                             [7.77653, 8.75285, 7.20703], [9.07534, 10.0458, 8.20743],
                             [10.7203, 11.6166, 9.34808], [12.727, 13.5334, 10.639],
                             [15.2088, 15.8821, 12.1313], [12, 12.6885, 12.6703],
                             [12.49, 13.4164, 14], [13, 14.1861, 15.8685],
                             [15.8355, 15, 17.9863]])

        self.assertEqual(value.shape, expected.shape)
        self.assertTrue((np.isnan(value) == np.isnan(expected)).all())
        self.assertTrue(np.all(np.isclose(value, expected, rtol=1e-04) | np.isnan(value)))


if __name__ == '__main__':
    unittest.main()
