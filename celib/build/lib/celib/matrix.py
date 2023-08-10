# -*- coding: utf-8 -*-
"""
matrix
======
CE-specific matrix operations (typically functions) not in NumPy.
"""

import numpy as np

from celib.exceptions import NonConvergenceError


def ras(matrix, row_sums, column_sums, *, max_iter=100, tol=1e-10, must_converge=True):
    """RAS procedure for matrix updating (iterative proportional fitting).

    Iteratively scale the elements of `matrix` to conform to the new totals in
    `row_sums` and `column_sums`.

    Return a 3-tuple of the updated matrix, convergence status and iterations
    taken (see below).

    **`must_converge=False` disables the convergence check and silently
      continues in the event of non-convergence. *Use this at your own risk*.**

    Parameters
    ----------
    matrix : (m x n) NumPy array
        Two-dimensional array to be updated
    row_sums : m-length or (m x 1)/(1 x m) NumPy array
        Vector of row sums for updated matrix
    column_sums : n-length or (1 x n)/(n x 1) NumPy array
        Vector of column sums for updated matrix
    max_iter : int, default 100
        Maximum number of iterations
    tol : float, default 1e-10
        Threshold for convergence
    must_converge : bool, default `True`
        In the event of non-convergence, if `must_converge` is:
         - `True` : raise `NonConvergenceError`
         - `False` : return the results from the final (non-converging)
                     iteration

    Returns
    -------
    As a 3-tuple:

    updated_matrix : (m x n) NumPy array
        The updated array
    converged : bool
        Whether updated_matrix has converged
    iteration : int
        The number of iterations that occurred

    Raises
    ------
    NonConvergenceError
        If `matrix` cannot be scaled to both `row_sums` and `column_sums`
        within the maximum number of iterations

        Possible causes of non-convergence include:
        (a) zeroes in certain elements of `row_sums` or `column_sums` where the
            corresponding row/column in `matrix` is non-zero
        (b) the reverse of (a) where rows/columns in `matrix` sum to zero but
            the corresponding element in `row_sums` or `column_sums` is non-zero
        (c) if the sum of `row_sums` differs from the sum of `column_sums`
        (d) if some elements of the inputs are negative in value

    `must_converge=False` disables the convergence check/error and returns the
    results from the final iteration. In this case, the array returned **will
    not** conform to both the row and column totals.

    Notes
    -----
    Allows for either 1d and 2d arrays for `row_sums` and `column_sums`

    Examples
    --------
    >>> import numpy as np
    >>> from celib.matrix import ras

    >>> initial_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> row_sums = np.array([[6.0], [14.0], [22.0]])
    >>> column_sums = np.array([[18.0, 24.0]])

    >>> updated_matrix, converged, iterations = ras(initial_matrix, row_sums, column_sums)

    >>> updated_matrix
    array([[  2.,   4.],
           [  6.,   8.],
           [ 10.,  12.]])

    >>> converged
    True

    >>> iterations
    1
    """
    def divide(a, b):
        """Return element-wise a / b with zeroes in place of divide-by-zeroes."""
        return np.divide(
            a, b,
            # Use `zeros()`, not `zeros_like()`, to ensure type is `float`
            out=np.zeros(a.shape),
            where=~np.isclose(b, 0),
            casting='unsafe')

    # Flatten row_sums and column_sums to simplify scaling operations in loop
    # below
    row_controls = row_sums.flatten()
    column_controls = column_sums.flatten()

    # Form tuples of dimension sizes for broadcasting in matrix multiplications
    # in loop below
    row_dims = (matrix.shape[0], 1)
    column_dims = (1, matrix.shape[1])

    updated_matrix = matrix.copy()
    converged = False

    for iteration in range(1, max_iter+1):
        # Calculate row scaling factors
        r = divide(row_controls, updated_matrix.sum(axis=1))
        updated_matrix = updated_matrix * r.reshape(row_dims)

        # Calculate column scaling factors
        s = divide(column_controls, updated_matrix.sum(axis=0))
        updated_matrix = updated_matrix * s.reshape(column_dims)

        # Convergence check tests that row sums match controls after column
        # scaling
        squared_differences = (row_controls - updated_matrix.sum(axis=1)) ** 2
        if np.all(squared_differences < tol):
            converged = True
            break

    if must_converge and not converged:
        raise NonConvergenceError(
            'RAS procedure failed to converge after {} iteration(s)'.format(iteration))

    return updated_matrix, converged, iteration
