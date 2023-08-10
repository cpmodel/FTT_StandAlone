# -*- coding: utf-8 -*-
"""
test_ras
========
Tests of the Python RAS procedure to update a matrix according to new row and
column totals.
"""

import unittest

import numpy as np

from celib.matrix import ras
from celib.exceptions import NonConvergenceError


class TestRAS(unittest.TestCase):

    def test_ras_double_2d(self):
        # Target row and column sums are just double that of the initial
        # matrix; use two-dimensional vectors
        initial_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        row_sums = np.array([[6.0], [14.0], [22.0]])
        column_sums = np.array([[18.0, 24.0]])

        expected = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])

        result, converged, iterations = ras(initial_matrix, row_sums, column_sums)

        self.assertTrue(np.allclose(result, expected))
        self.assertTrue(converged)
        self.assertEqual(iterations, 1)  # Simple case should complete in 1 iteration

    def test_ras_double_1d(self):
        # Target row and column sums are just double that of the initial
        # matrix; use one-dimensional vectors
        initial_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        row_sums = np.array([6.0, 14.0, 22.0])
        column_sums = np.array([18.0, 24.0])

        expected = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])

        result, converged, iterations = ras(initial_matrix, row_sums, column_sums)

        self.assertTrue(np.allclose(result, expected))
        self.assertTrue(converged)
        self.assertEqual(iterations, 1)  # Simple case should complete in 1 iteration

    def test_ras_iterative_floats(self):
        # Reproduce results from Ox RAS with the same inputs (as floats)
        initial_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        row_sums = np.array([[2.0], [8.0], [15.0]])
        column_sums = np.array([[10.0, 15.0]])

        # Copied from Ox RAS results
        expected = np.array([[0.601376758734, 1.398623764073],
                             [3.136678430666, 4.863321735967],
                             [6.261944810499, 8.738054499859]])

        result, converged, iterations = ras(initial_matrix, row_sums, column_sums)

        self.assertTrue(np.allclose(result, expected))
        self.assertTrue(converged)

    def test_ras_iterative_integers(self):
        # Reproduce results from Ox RAS with the same inputs (as integers)
        initial_matrix = np.array([[1, 2], [3, 4], [5, 6]])
        row_sums = np.array([[2], [8], [15]])
        column_sums = np.array([[10, 15]])

        # Copied from Ox RAS results
        expected = np.array([[0.601376758734, 1.398623764073],
                             [3.136678430666, 4.863321735967],
                             [6.261944810499, 8.738054499859]])

        result, converged, iterations = ras(initial_matrix, row_sums, column_sums)

        self.assertTrue(np.allclose(result, expected))
        self.assertTrue(converged)

    def test_ras_failed_convergence_error(self):
        # Ox RAS with these inputs fails to converge after 100 iterations;
        # check Python version raises an exception
        initial_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 6.0]])
        row_sums = np.array([[2.0], [8.0], [15.0]])
        column_sums = np.array([[10.0, 15.0]])

        with self.assertRaises(NonConvergenceError):
            ras(initial_matrix, row_sums, column_sums)

    def test_ras_suppress_failed_convergence(self):
        # Ox RAS with these inputs fails to converge after 100 iterations;
        # check Python version continues after failed convergence if
        # `must_converge` is `False`
        initial_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 6.0]])
        row_sums = np.array([[2.0], [8.0], [15.0]])
        column_sums = np.array([[10.0, 15.0]])

        result, converged, iterations = ras(initial_matrix, row_sums, column_sums,
                                            must_converge=False)

        self.assertFalse(converged)

    def test_ras_integer_inputs(self):
        # Pass integer arrays to `ras()` and check that NumPy coerces the types
        # (without throwing an exception) and gives the correct result
        initial_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
        row_sums = np.array([6, 14, 22], dtype=int)
        column_sums = np.array([18, 24], dtype=int)

        result, converged, iterations = ras(initial_matrix, row_sums, column_sums)

        self.assertTrue(np.allclose(result, initial_matrix * 2))
        self.assertTrue(converged)

    def test_ras_zeroes_in_rows(self):
        initial_matrix = np.array([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]])
        row_sums = np.array([6.0, 0.0, 22.0])
        column_sums = np.array([12.0, 16.0])

        result, converged, iterations = ras(initial_matrix, row_sums, column_sums)

        self.assertTrue(np.allclose(result, initial_matrix * 2))
        self.assertTrue(converged)

    def test_ras_zeroes_in_columns(self):
        initial_matrix = np.array([[1.0, 0.0, 5.0], [2.0, 0.0, 6.0]])
        row_sums = np.array([12.0, 16.0])
        column_sums = np.array([6.0, 0.0, 22.0])

        result, converged, iterations = ras(initial_matrix, row_sums, column_sums)

        self.assertTrue(np.allclose(result, initial_matrix * 2))
        self.assertTrue(converged)


if __name__ == '__main__':
    unittest.main()
