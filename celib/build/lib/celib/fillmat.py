# -*- coding: utf-8 -*-
"""
fillmat
=======

Functions to fill a DataFrame/Series/Numpy array that contains missing data.

Filling functions are:
    - fill_with_shares - fill where all years are missing by sharing out
                         a total based on supplied shares
    - fill_with_growth_rates - fill gaps at the two ends of a series
                               using growth rates from another variable
    - extrapolate - fill gaps at the two ends of a series by extrapolation
    - interpolate - fill gaps in the middle of a series by interpolation
                    for up to 6 blocks of missing data
    - restricted_fill_forward - fill gaps at the end of a series by extrapolating
                                forward subject to a constraint (target growth
                                rates eg previous version of the variable)
    - restricted_fill_backward - fill gaps at the beginning of a series by
                                 extrapolating backward subject to a constraint
    - restricted_interpolation - fill gaps in the middle of a series by
                                interpolating subject to a constraint
    - fill_all_gaps - fill gaps anywhere in the series by sequentially
                      trying each of the four methods above

The rest of the functions are called by the filling functions:
    - convert_to_np_arrays and restore_original_type - convert between data
      types to accommodate inputs that are not Numpy arrays
    - findgap - find the first and last missing years in a timeseries
    - target_forward, target_backward and target_interpolation - used to generate
      the reference variable (3rd input) in filling functions with restrictions

"""

import warnings

import numpy as np
from pandas import DataFrame, Series


def convert_to_np_arrays(invar):
    """ Convert DataFrames or Series to np arrays."""
    if isinstance(invar, DataFrame) or isinstance(invar, Series):
        outvar = invar.copy().values
    else:
        outvar = invar.copy()

    if len(outvar.shape) == 1:
        outvar = np.expand_dims(outvar.copy(), axis=1)

    return outvar


def restore_original_type(newvar, originalvar):
    """ Convert output from the filling functions to the same type as
    the original input variable to be filled."""
    if isinstance(originalvar, DataFrame):
        output = DataFrame(newvar, index=originalvar.index,
                           columns=originalvar.columns)
    elif isinstance(originalvar, Series):
        output = Series(newvar.flatten(), index=originalvar.index)
    else:
        output = newvar

    return output


def fill_all_gaps(var_to_fill, var_to_get_shares=None, total=None):
    """
    Apply filling methods sequentially to fill the variable with missing values.

    Parameters
    ----------
    var_to_fill : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable to be filled, missing values must be supplied as NaNs

    var_to_get_shares : (t x n) DataFrame/Numpy array or (t x 1) Series
        (optional) Variable holding shares of the total to be split when data
        are missing for all years or growth rates to be applied when
        data are missing at the start or end (time x sectors), must have the
        same dimensions as var_to_fill and must not contain missing values

    total : (t x 1) Series or Numpy array
        (optional) Total to be split, must not contain missing values

    Filling procedure
    ----------
    Calling other functions in the following order:
        1. fill_with_shares()
        2. fill_with_growth_rates()
        3. extrapolate()
        4. interpolate()

    Returns
    ----------
    filled : (t x n) DataFrame/Numpy array or (t x 1) Series
       Filled variable (which may still contain missing data) in the original data structure

    """
    original_var = var_to_fill.copy()
    filled = var_to_fill.copy()

    # Share out total using second variable if both are supplied
    if var_to_get_shares is not None and total is not None:
        filled = fill_with_shares(var_to_fill, total, var_to_get_shares)

    # Fill gaps at the two ends using growth in second variable if supplied
    if var_to_get_shares is not None:
        filled = fill_with_growth_rates(filled, var_to_get_shares)
        # Apply extrapolation if no filling variable is supplied
    else:
        filled = extrapolate(filled)

    # Fill gaps in the middle using linear inteprolation
    # (and using second variable if supplied)
    filled = interpolate(filled, var_to_get_shares)

    if np.isnan(filled).any().any():
        warnings.warn('Some missing values remain unfilled.')

    return restore_original_type(filled, original_var)


def fill_with_shares(var_to_fill, total, var_to_get_shares):
    """
    Fill first input variable by sharing out a total using shares from
    second input variable.

    Only fill elements in the non-time dimension when data are missing for all
    years and assume the sum of non-missing sectors in the variable to be
    filled does not exceed the supplied total.

    Parameters
    ----------
    var_to_fill : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable to be filled, missing values must be supplied as NaNs

    var_to_get_shares : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable holding shares of the total to be split, must have the same
        dimensions as var_to_fill and must not contain missing values

    total : (t x 1) Series or Numpy array
        Total to be split to fill missing values, must not contain missing values

    Filling procedure
    ----------
        1. Check variable dimensions and values
        2. Find sectors that are missing for all years and sectors that have some data
        3. Find the first and last years of data for non-missing data (if only
           one year is available, first and last years will be the same) - this
           is the range for filling var_to_fill
        4. Fill missing sectors in the filling range using var_to_get_shares and total
        5. If not all sectors are missing, scale estimates for missing sectors
           so that the total is unchanged

    Returns
    ----------
    filled : (t x n) DataFrame/Numpy array or (t x 1) Series
       Filled variable in the original data structure

    """
    original_var = var_to_fill.copy()

    # Make sure all inputs are np arrays
    var_to_fill = convert_to_np_arrays(var_to_fill)
    total = convert_to_np_arrays(total)
    var_to_get_shares = convert_to_np_arrays(var_to_get_shares)

    # Check the dimensions of input variables
    # var_to_fill and var_to_get_shares should be the same size
    if var_to_fill.shape != var_to_get_shares.shape:
        msg = 'Variable to fill and variable to take shares have different sizes.'
        raise ValueError(msg)

    # var_to_fill and total should be the same size
    if var_to_fill.shape[0] != total.shape[0]:
        msg = 'Variable to fill and total have different sizes.'
        raise ValueError(msg)

    # var_to_get_shares and total should not have missing values
    if np.isnan(var_to_get_shares).any():
        raise ValueError('Variable to take share contains missing values.')
    if np.isnan(total).any():
        raise ValueError('Total contains missing values.')

    # Return original variable if there's no sector where data are missing
    # for all years
    missing = np.argwhere(np.isnan(var_to_fill).all(axis=0)).flatten()
    if len(missing) == 0:
        return original_var

    # Find non-missing sectors
    notmissing = [i for i in range(var_to_fill.shape[1]) if i not in missing]

    # Find the first and last years where data are available for
    # all sectors that are not deemed missing for all years
    nonmissingyears = np.argwhere(np.isfinite(var_to_fill[:, notmissing]).all(axis=1))

    # Do nothing if there is no year with data for all other sectors
    if len(nonmissingyears) == 0:
        return original_var

    # Otherwise, find the first year of data
    start = nonmissingyears.flatten()[0]

    # If there is only one year, set that as the end year
    if len(nonmissingyears) == 1:
        end = start
    else:
        if np.isfinite(var_to_fill[start+1:, notmissing]).all(axis=1).all():
            end = -1
        else:
            find_end = var_to_fill[np.ix_(list(range(start+1, var_to_fill.shape[0])), notmissing)]
            end = np.argwhere(np.isfinite(find_end).all(axis=1)).flatten()[0] + start - 1

    years_to_fill = list(range(start, end+1))

    # Make estimates for misisng sectors using total and share variable
    shares = var_to_get_shares / np.expand_dims(var_to_get_shares.sum(axis=1), axis=1)
    filled = var_to_fill.copy()

    values_to_fill = np.ix_(years_to_fill, missing)
    filled[values_to_fill] = shares[values_to_fill] * total[years_to_fill]

    # If not all sectors are missing, scale estimates for missing sectors
    # so that total is unchanged
    if len(notmissing) > 0:
        sumfilled = filled[values_to_fill].copy().sum(axis=1)
        filled[values_to_fill] /= np.expand_dims(sumfilled, axis=0).T

        notmissingtotal = np.nansum(var_to_fill[years_to_fill, :], axis=1)
        filled[values_to_fill] *= (total[years_to_fill]
                                   - np.expand_dims(notmissingtotal, axis=0).T)

    return restore_original_type(filled, original_var)


def fill_with_growth_rates(var_to_fill, var_to_get_growth):
    """
    Fill gaps at the two ends of the first variable using
    growth rates from the second variable.

    Series with missing data that only contains zeros are filled with zeros.

    Series with all years missing are returned unfilled.

    Parameters
    ----------
    var_to_fill : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable to be filled, missing values must be supplied as NaNs

    var_to_get_growth : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable holding growth rates, must have the same dimensions as
        var_to_fill and must not contain missing values

    Filling procedure
    ----------
        1. Check variable dimensions and values
        2. Loop backward once, find the filling range for sectors that have some data
            a. If only zeros found, missing values are filled with zeros
            b. Otherwise, growth rates from reference variable are applied
        3. Loop forward once, repeat the same procedure as step 2 (missing
           values that were filled in step 2 will now be treated as real).

    Returns
    ----------
    filled : (t x n) DataFrame/Numpy array or (t x 1) Series
       Filled variable in the original data structure

    """
    original_var = var_to_fill.copy()

    # Make sure all inputs are np arrays
    var_to_fill = convert_to_np_arrays(var_to_fill)
    var_to_get_growth = convert_to_np_arrays(var_to_get_growth)

    # Check the dimensions of input variables
    # var_to_fill and var_to_get_growth should be the same size
    if var_to_fill.shape != var_to_get_growth.shape:
        msg = 'Variable to fill and variable to take growth rates have different sizes.'
        raise ValueError(msg)

    # var_to_get_growth should not have missing values
    if np.isnan(var_to_get_growth).any():
        raise ValueError('Variable to take growth rates contains missing values.')

    # Direction loop: 1 - backward fill, 2 - forward fill
    filled = var_to_fill.copy()
    for i in range(1, 3):
        # Sectoral/regional loop
        for j in range(var_to_fill.shape[1]):
            # Do not proceed if data are missing for all years or
            # row does not need filling (ie looping backwards and first year
            # not missing or looping forwards and last year not missing)
            if np.isnan(var_to_fill[:, j]).all() or np.isfinite(var_to_fill[1-i, j]):
                continue

            start, end = findgap(i, var_to_fill[:, j])

            # If only zeros in series, set missing values to zeros
            if np.nansum(var_to_fill[:, j]) == 0:
                filled[start:end+1, j] = 0
                continue

            if i == 1:
                index = var_to_get_growth[start:end+1, j] / var_to_get_growth[end+1, j]
                filled[start:end+1, j] = index * filled[end+1, j]
            if i == 2:
                index = var_to_get_growth[start:end+1, j] / var_to_get_growth[start-1, j]
                filled[start:end+1, j] = index * filled[start-1, j]

    return restore_original_type(filled, original_var)


def extrapolate(var_to_fill):
    """
    Fill gaps at the two ends by extrapolating backward and forward.

    Series with missing data that only contains zeros are filled with zeros.
    Series with only one year of data are returned unfilled.

    Parameters
    ----------
    var_to_fill : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable to be filled, missing values must be supplied as NaNs

    Filling procedure
    ----------
        1. Check variable dimensions and values
        2. Loop backward once:
            a. For sectors that have some data, find all missing and non-missing years
                - If only zeros found, missing values are filled with zeros
                - Return the variable unfilled if only one year is found
            b. Find first and last years of available data (can have missing
               values in the middle)
            c. Fill:
                - If the first year is zero, fill missing values with zeros
                - If the last year is zero, find the next non-zero value (exit
                  and return the variable unfilled if none is found)
                - If both years are non-zero, use average growth rates from
                  those years to fill backward
        3. Loop forward once, repeat the same procedure as step 2 (missing
           values that were filled in step 2 will now be treated as real).
           - If the last year is zero, missing values are filled with zeros
           - If the first year is zero, find the next non-zero value

    Returns
    ----------
    filled : (t x n) DataFrame/Numpy array or (t x 1) Series
       Filled variable in the original data structure

    """
    original_var = var_to_fill.copy()

    # Make sure all inputs are np arrays
    var_to_fill = convert_to_np_arrays(var_to_fill)

    # Direction loop: 1 - backward fill, 2 - forward fill
    filled = var_to_fill.copy()
    for i in range(1, 3):
        # Sectoral/regional loop
        for j in range(var_to_fill.shape[1]):
            # Do not proceed if data are missing for all years or
            # row does not need filling (ie looping backwards and first year
            # not missing or looping forwards and last year not missing)
            if np.isnan(var_to_fill[:, j]).all() or not np.isnan(var_to_fill[1-i, j]):
                continue

            start, end = findgap(i, var_to_fill[:, j])
            missingyears = np.argwhere(np.isnan(var_to_fill[:, j]))
            notmissingyrs = np.argwhere(np.isfinite(var_to_fill[:, j]))

            # If only zeros in series, set missing values to zeros
            if np.nansum(var_to_fill[:, j]) == 0:
                filled[missingyears, j] = 0
                continue

            # Do notthing if there is only one year of data (not in Ox ver)
            if len(notmissingyrs) == 1:
                continue

            # Find first and last years of available data
            first = var_to_fill[notmissingyrs[0], j]
            last = var_to_fill[notmissingyrs[-1], j]

            # If filling backwards and first year is zero or filling forwards
            # and last year is zero, fill with zeros
            if (i == 1 and first == 0) or (i == 2 and last == 0):
                filled[start:end+1, j] = 0
                continue

            # If filling backwards and the last year is zero or if filling forwards
            # and the first year is zero, look for the next non-zero value
            if i == 1 and last == 0:
                growth = None
                for k in reversed(list(notmissingyrs)[1:-1]):
                    # If another non-zero value other than the first year exists
                    # Use that value to calculate the growth rate
                    if var_to_fill[k, j] != 0:
                        growth = ((first / var_to_fill[k, j]) ** (1 / (k - notmissingyrs[0])))
                        break

                # Do nothing if the first year is the only non-zero value
                if growth is not None:
                    for k in reversed(range(start, end+1)):
                        filled[k, j] = filled[k+1, j] * growth

                continue

            if i == 2 and first == 0:
                growth = None
                for k in list(notmissingyrs)[1:-1]:
                    # If another non-zero value other than the last year exists
                    # Use that value to calculate the growth rate
                    if var_to_fill[k, j] != 0:
                        growth = ((last / var_to_fill[k, j]) ** (1 / (notmissingyrs[-1] - k)))
                        break

                # Do nothing if the first year is the only non-zero value
                if growth is not None:
                    for k in range(start, end+1):
                        filled[k, j] = filled[k-1, j] * growth

                continue

            # General case where both the first and last values are non-zero
            if i == 1:
                growth = ((first / last) ** (1 / (notmissingyrs[-1] - notmissingyrs[0])))
                for k in reversed(range(start, end+1)):
                    filled[k, j] = filled[k+1, j] * growth

            if i == 2:
                growth = ((last / first) ** (1 / (notmissingyrs[-1] - notmissingyrs[0])))
                for k in range(start, end+1):
                    filled[k, j] = filled[k-1, j] * growth

    return restore_original_type(filled, original_var)


def interpolate(var_to_fill, var_to_get_growth=None):
    """
    Fill gaps in the middle of the series by interpolating between
    years of available data. Interpolation uses compound annual growth rates.
    Contains optional argument to scale growth rates to growth rates of another.

    If either the first year or the last year of data is zero, linear
    interpolation is used instead.

    Parameters
    ----------
    var_to_fill : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable to be filled, missing values must be supplied as NaNs

    var_to_get_growth : (t x n) DataFrame/Numpy array or (t x 1) Series
        (optional) variable holding growth rates to match after interpolation,
        must have the same dimensions as var_to_fill and must not contain missing values

    Filling procedure
    ----------
        1. Check variable dimensions and values
        2. Loop through sectors:
            a. Skip if data are missing for all years
            b. Otherwise, find the first block of missing data (first and last
               years in the block are not missing, missing values in the middle)
                - Skip if data are only missing at the two ends or if only one
                  year of data is found
                - If both the first and last years are non-zero, use growth rates
                - If either is zero, use absolute changes
            c. Scale filled values to reference growth rates if supplied
                - If filling with growth rates, only scale if the reference
                  growth rates are valid (neither the numerator nor the denominator is zero)
                - If filling with linear interpolation, no scaling is done
            d. Continue to the next block (up to 5 more) and repeat b and c.

    Returns
    ----------
    filled : (t x n) DataFrame/Numpy array or (t x 1) Series
       Filled variable in the original data structure

    """
    original_var = var_to_fill.copy()

    # Make sure all inputs are np arrays
    var_to_fill = convert_to_np_arrays(var_to_fill)
    if var_to_get_growth is not None:
        var_to_get_growth = convert_to_np_arrays(var_to_get_growth)

        # Check the dimensions of input variables
        # var_to_fill and var_to_get_shares should be the same size
        if var_to_fill.shape != var_to_get_growth.shape:
            msg = 'Variable to fill and variable to take growth rates have different sizes.'
            raise ValueError(msg)

        # var_to_get_growth should not have missing values
        if np.isnan(var_to_get_growth).any():
            raise ValueError('Variable to take growth rates contains missing values.')

    # Sectoral/regional loop
    filled = var_to_fill.copy()
    for i in range(filled.shape[1]):
        # Do nothing if data are missing for all years
        if np.isnan(filled[:, i]).all():
            continue

        # Apply interpolation for up to 6 blocks of missing data
        for j in range(7):
            # Find years with data
            notmissingyrs = np.argwhere(np.isfinite(filled[:, i]))
            first, last = int(notmissingyrs[0]), int(notmissingyrs[-1])

            # Do nothing if there is only one year of data
            if len(notmissingyrs) == 1:
                continue
            # Do nothing if data are only missing at the end
            if np.isfinite(filled[first:last, i]).all():
                continue

            # Find first year of missing data in the middle of series
            start = first + int(np.argwhere(np.isnan(filled[first:, i]))[0])

            # Do nothing if data are only missing at the beginning
            if np.isfinite(filled[start:, i]).all():
                continue

            # Find last year of missing data in the middle of series
            end = start + int(np.argwhere(np.isfinite(filled[start+1:, i]))[0])

            # Calculate alternative growth rates if both start and end
            # values are non-zeros, otherwise raise a warning and do not use
            if var_to_get_growth is not None:
                if var_to_get_growth[end+1, i] != 0 and var_to_get_growth[start-1, i] != 0:
                    ref_growth = ((var_to_get_growth[end+1, i] / var_to_get_growth[start-1, i])
                                   ** (1 / (end - start + 2)))
                else:
                    var_to_get_growth = None
                    warnings.warn('Alternative growth rates unusable, results will not be scaled.')

            # Interpolate according to the start and end years' values
            # If both values are non-zero, use compound growth rates
            if filled[end+1, i] != 0 and filled[start-1, i] != 0:
                # Calculate growth rate over interpolation period
                growth = ((filled[end+1, i] / filled[start-1, i])
                           ** (1 / (end - start + 2)))

                # Apply growth rates to fill missing data
                for k in range(start, end+1):
                    filled[k, i] = filled[k-1, i] * growth
                    # Scale to alternative growth rates if supplied
                    if var_to_get_growth is not None:
                        filled[k, i] *= (var_to_get_growth[k, i]
                                         / var_to_get_growth[k-1, i] / ref_growth)

            # If either is zero, use linear interpolation
            else:
                # Calculate absolute annual changes
                change = ((filled[end+1, i] - filled[start-1, i])/(end - start + 2))

                # Apply absolute changes to fill missing data
                for k in range(start, end+1):
                    filled[k, i] = filled[k-1, i] + change

    return restore_original_type(filled, original_var)


def findgap(direction, data):
    """
    Find the first and last years of missing data in the specified
    direction.

    Parameters
    ----------
    direction : integer (1 - backward or 2 - forward)
        Direction to inspect data
    data : (t x 1) Series or Numpy array
        Data to be inspected

    Returns
    ----------
    start : integer
        Index of the first year of missing data
    end : integer
        Index of the last year of missing data

    """
    # Find the indices of all available data points
    notmissing = np.argwhere(np.isfinite(data))

    if len(notmissing) == 0:
        start = -1
        end = -1
    else:
        if direction == 1:
            start = 0
            end = int(notmissing[0]-1)
        else:
            start = int(notmissing[-1]+1)
            end = len(data)-1

    return start, end


def restricted_fill_forward(var_to_fill, total, referencevar, pdr=1):
    """
    Extrapolate forwards subject to a constraint.
    The rows in variable to fill are extrapoated so that the sum of
    all sectors grow as total and keeping as close as possible to
    the growth of the sectors in reference variable (eg past growth rates).

    If all years are missing, the variable is returned unfilled.

    Only fill gaps at the end of the series.

    Paramters
    ----------
    var_to_fill : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable to be filled, missing values must be supplied as NaNs

    total : (t x 1) Series or Numpy array
        Growth contraint on the sum of all sectors, if containing missing values
        var_to_fill will not be filled for the missing years

    referencevar : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable holding growth rates to match, must have the same dimensions
        as var_to_fill and must not contain missing values from the first year
        of missing data in var_to_fill

    Filling procedure
    ----------
        1. Check variable dimensions and values
        2. Skip if no data are missing or if all data are missing
        3. Otherwise, find the first year of missing data (first year after the
           last year of data available for at least some sectors) - this is the
           first year for filling
        4. Loop forward in the total and find the first year of missing data
           - this is the last year for filling (must be later than the first
           year for filling, otherwise exit function)

        5.Loop forward through the filling range to find missing and non-missing sectors
        6. Calculate annual growth rates in the total, reference variable and fill variable
            a. If previous year was zero, assume 100% growth
            b. If this year is zero, assume -100% growth

        7. Calculate sectoral weights (shares of the sum)
           - allocate equal weights if all sectors are zeros
        8. Fill missing values so that the sum of sectors grows with the total
           and each sector grows as close as possible with the reference variable

    Returns
    ----------
    filled : (t x n) DataFrame/Numpy array or (t x 1) Series
       Filled variable in the original data structure

    """
    original_var = var_to_fill.copy()

    # Make sure all inputs are np arrays
    var_to_fill = convert_to_np_arrays(var_to_fill)
    total = convert_to_np_arrays(total)
    referencevar = convert_to_np_arrays(referencevar)

    # Total should only have one column
    if total.shape[1] != 1:
        raise ValueError('Dimension of total matrix is not one.')

    # var_to_fill and referencevar should be the same size
    if var_to_fill.shape != referencevar.shape:
        msg = 'Variable to fill and reference variable have different sizes.'
        raise ValueError(msg)

    # var_to_fill and total should be the same size
    if var_to_fill.shape[0] != total.shape[0]:
        msg = 'Variable to fill and total have different sizes.'
        raise ValueError(msg)

    # referencevar should not have missing values
    for i in range(var_to_fill.shape[1]):
        start, end = findgap(2, var_to_fill[:, i])
        if start > 0:
            if np.isnan(referencevar[start-1:, i]).any():
                raise ValueError('Reference variable contains missing values.')

    # Check for cases where there is no missing data or all sectors
    # are missing and exit the function
    missingyears = np.argwhere(np.isnan(var_to_fill).all(axis=1))

    if len(missingyears) == var_to_fill.shape[0]:
        warnings.warn('All data are missing. Variable cannot be filled.')
        return original_var
    else:
        notmissing = np.argwhere(np.isfinite(var_to_fill).all(axis=1))
        if len(notmissing) == var_to_fill.shape[0]:
            warnings.warn('No data are missing.')
            return original_var
        elif len(notmissing) == 0:
            warnings.warn('No year with complete data found. Variable cannot be filled.')
            return original_var
        else:
            start = int(notmissing[-1]) + 1

    # total should not have missing values from the first missing year
    if start >= total.shape[0]:
        warnings.warn('First year to be filled is greater than last year of total. Variable cannot be filled.')
        return original_var
    if np.isnan(total[start]):
        warnings.warn('Total contains missing values in first year to be filled. Variable cannot be filled.')
        return original_var

    # If any sector in the reference variable contains zeros
    # set the whole series to the total
    for i in range(referencevar.shape[1]):
        if np.count_nonzero(referencevar[:, i]) < referencevar.shape[0]:
            referencevar[:, i] = total.flatten()

    # Find first year of missing data in total
    end, _ = findgap(2, total.flatten())

    filled = var_to_fill.copy()

    # Start iterating from first year of missing data in main variable
    # to last year of data in total
    for i in range(start, end):
        iteration = 1
        missing = np.isnan(filled[i, :])
        notmissing = np.isfinite(filled[i, :])

        total_growth = total[i] / total[i-1]
        sector_growth = filled[i, :] / filled[i-1, :]
        target_growth = referencevar[i, :] / referencevar[i-1, :]

        # If previous year was zero, replace with no growth
        target_growth[referencevar[i-1, :] == 0] = 1
        sector_growth[filled[i-1, :] == 0] = 1
        total_growth[total[i-1] == 0] = 1  # *** Set to 0 in Ox version (why?)

        # If this year is zero, make target growth -100% (zero)
        # and make sectoral growth -100% and filled value zero
        target_growth[referencevar[i, :] == 0] = 0
        sector_growth[filled[i, :] == 0] = 0
        filled[i, sector_growth == 0] = 0

        # Alfa measures sectoral weight, if all zeros weights are equal
        alfa = filled[i-1, :] / filled[i-1, :].sum()
        if filled[i-1, :].sum() == 0:
            alfa[:] = 1 / len(filled[i-1, :])

        while iteration == 1:
            # Iterate only once by default
            iteration = 0
            alfb = np.full_like(alfa, 0)

            alfb[missing] = (alfa[missing]**2) ** (1/(4*pdr-2))

            growth = target_growth.copy()
            growth[notmissing] = sector_growth[notmissing]

            graux_create = (growth + np.divide(total_growth - np.dot(growth, alfa),
                                       np.dot(alfb, alfa),
                                       out=np.zeros_like(growth),
                                       where=np.dot(alfb, alfa)!=0) * alfb)

            graux = graux_create.copy()
            graux[notmissing] = growth[notmissing]
            mask = np.dot(alfb, alfa) == 0
            if mask:
                graux = growth.copy()

            filled[i, :] = filled[i-1, :] * graux

            """If any of the results are negative the value is fixed to zero.
            This is equvalent to a minimisation of the distance to
            target with a positiveness restriction. Those values that
            result negative in the first pass are fixed to zero and
            they don't participate in the optimisation anymore. When the
            condition below is true, the progaram is asked to iterate once
            more solving the problem with the positiveness restrictions on."""
            negative = filled[i, :] < 0
            if np.any(negative):
                iteration = 1
                sector_growth[negative] = 0
                target_growth[negative] = 0
                alfa[negative] = 0
                # Negative values in filled variable fixed to zero
                # and the rest reset to original values
                filled[i, negative] = 0
                filled[i, ~negative] = var_to_fill[i, ~negative].copy()

    return restore_original_type(filled, original_var)


def restricted_fill_backward(var_to_fill, total, referencevar, pdr=1):
    """
    Extrapolate backward subject to a constraint, using the
    restricted_fill_forward() function by reverting the time order.

    If all years are missing, the variable is returned unfilled.

    Only fill gaps at the beginning of the series.
    """
    original_var = var_to_fill.copy()

    # Make sure all inputs are np arrays and in reversed order
    var_to_fill = np.flip(convert_to_np_arrays(var_to_fill), axis=0).copy()
    total = np.flip(convert_to_np_arrays(total), axis=0).copy()
    referencevar = np.flip(convert_to_np_arrays(referencevar), axis=0).copy()

    filled_reversed = restricted_fill_forward(var_to_fill, total, referencevar)
    filled = np.flip(filled_reversed, axis=0)

    return restore_original_type(filled, original_var)


def restricted_interpolation(var_to_fill, total, referencevar, weight=True, pdr=1):
    """
    Generate a target matrix using the average of restricted_fill_forward() and
    restricted_fill_backward().

    Only fill gaps in the middle of the series.

    Parameters
    ----------
    var_to_fill : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable to be filled, missing values must be supplied as NaNs

    total : (t x 1) Series or Numpy array
        Growth contraint on the sum of all sectors, must not contain missing values

    referencevar : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable holding growth rates to match, must have the same dimensions
        as var_to_fill and must not contain missing values from the first year
        of missing data in var_to_fill

    weight : boolean
        (optional) Default to True, the function implents a simpler
        interpolation, otherwise it implements a reweighted one.

    Filling procedure
    ----------
        1. Simple method:
            a. Call restricted_fill_forward() to get a first estimate
            b. Call restricted_fill_forward() again with all variables reversed
               in the time dimension, get a second estimate (equivalent to
               using restricted_fill_backward)
            c. Take the simple average of the two estimates as filled values

        2. Weighted method:
            a. Repeat the first two steps in the simple method
            b. For each year, calculate the weights for the forward and backward estimates
                - Weights are based on how close the estimates are from the
                  reference variables
                - More weights on the forward estimates if they are closer from
                  the right (later years)
                - More weights on the backward estimates if they are closer
                  from the left (earlier years)
            c. Take a weighted average of the two estimates as filled values

    Returns
    ----------
    filled : (t x n) DataFrame/Numpy array or (t x 1) Series
       Filled variable in the original data structure

    """
    original_var = var_to_fill.copy()

    # Make sure all inputs are np arrays
    var_to_fill = convert_to_np_arrays(var_to_fill)
    total = convert_to_np_arrays(total)
    referencevar = convert_to_np_arrays(referencevar)

    var_to_fill_reversed = np.flip(var_to_fill, axis=0).copy()
    total_reversed = np.flip(total, axis=0).copy()
    refvar_reversed = np.flip(referencevar, axis=0).copy()
    filled = var_to_fill.copy()

    # The simple method weights equally whether the actual data is close
    # or far from the estimated data
    if weight is False:
        for i in range(1, filled.shape[0]):
            aux = restricted_fill_forward(filled[:i+1, :], total[:i+1, :],
                                          referencevar[:i+1, :])
            filled[:i+1, :] = aux

            auxr = restricted_fill_forward(var_to_fill_reversed[:i+1, :],
                                           total_reversed[:i+1, :],
                                           refvar_reversed[:i+1, :])
            var_to_fill_reversed[:i+1, :] = auxr

        # Take the simple average of the two estimations
        # If either is NaN, the returned value will be NaN
        filled = ((filled + np.flip(var_to_fill_reversed, axis=0)) / 2)

    # The "Other" method put more weight on the restricted_fill_forward estimation if
    # actual data is close from the left and weights more the restricted_fill_backward
    # estimation if data is close from the right
    else:
        index = np.full_like(var_to_fill, 0)
        index_reversed = np.full_like(var_to_fill, 0)

        for i in range(1, filled.shape[0]):
            # Calculate weights for each estimation
            missing = np.isnan(var_to_fill[i, :])
            index[i, missing] = index[i-1, missing] + 1
            missing_reversed = np.isnan(var_to_fill_reversed[i, :])
            index_reversed[i, missing_reversed] = index_reversed[i-1, missing_reversed] + 1

            # Carry our estimations as with the simple method
            aux = restricted_fill_forward(filled[:i+1, :], total[:i+1, :],
                                          referencevar[:i+1, :])
            filled[:i+1, :] = aux

            auxr = restricted_fill_forward(var_to_fill_reversed[:i+1, :],
                                           total_reversed[:i+1, :],
                                           refvar_reversed[:i+1, :])
            var_to_fill_reversed[:i+1, :] = auxr

        index[index != 0] = 1 / index[index != 0]
        index_reversed[index_reversed != 0] = 1 / index_reversed[index_reversed != 0]

        # Flip reversed variables back to the correct order
        index_reversed = np.flip(index_reversed, axis=0)
        var_to_fill_reversed = np.flip(var_to_fill_reversed, axis=0)

        # Calculate the weighted average of the two estimations
        # If either is NaN, the returned value will be NaN
        filled[index != 0] = ((np.multiply(filled[index != 0], index[index != 0])
                              + np.multiply(var_to_fill_reversed[index != 0],
                                            index_reversed[index != 0]))
                              / (index[index != 0] + index_reversed[index != 0]))

    return restore_original_type(filled, original_var)


def target_forward(var_to_fill, window):
    """
    To be used in conjunction with restricted_fill_forward() to create values for the
    third argument in that function.

    Create a set of target growth rates from a matrix with missing values ahead.

    It maintains the past growth rates, looking a number of years back,
    for the missing observations that follow.

    If the supplied window is longer than the data available before the first
    missing year, growth rates are calculated from the available data.

    Paramters
    ----------
    var_to_fill : (t x n) DataFrame/Numpy array or (t x 1) Series
        Variable to be filled, missing values must be supplied as NaNs
    window : integer
        Number of years for growth rate calculations

    Returns
    ----------
    filled : (t x n) DataFrame/Numpy array or (t x 1) Series
        Filled variable in the original data structure
    unusable : list
        Indices of elements in the n dimension that cannot be filled because
        all data are missing.

    """
    original_var = var_to_fill.copy()

    # Make sure all inputs are np arrays
    var_to_fill = convert_to_np_arrays(var_to_fill)
    filled = var_to_fill.copy()
    unusable = []

    for i in range(filled.shape[1]):
        # start is first missing value found from end of series
        # end is last observation
        start, end = findgap(2, var_to_fill[:, i])

        # Ensures that there are enough observations (at least two) behind
        # start to create a target; modifies years accordingly
        years = window
        if 0 < start <= end:
            for j in reversed(range(1, years+1)):
                if start-1-j < 0:
                    years = years-1

        # Checks if findgap found a completely missing series and that
        # there are at least two contiguous non-missing observations.
        if years == 1:
            warnings.warn('Target-growth cannot be created in sector {}; too many missing values'.format(i))
            unusable.append(i)
        else:
            if start <= end:
                for j in range(start, end+1):
                    rate = var_to_fill[start-1, i] / var_to_fill[start-1-years, i]
                    filled[j, i] = filled[j-1, i] * (rate ** (1/years))
                    if years < window:
                        years += 1

    return restore_original_type(filled, original_var), unusable


def target_backward(var_to_fill, window):
    """
    To be use jointly with restricted_fill_backward() to create values for the third
    argument in that function.

    Create a set of target growth rates from a matrix with missing values at the beginning.
    Use target_forward() with the time-reverse of the matrix.
    """
    original_var = var_to_fill.copy()

    # Make sure all inputs are np arrays
    var_to_fill = np.flip(convert_to_np_arrays(var_to_fill), axis=0).copy()
    filled_reversed, unusable = target_forward(var_to_fill, window)
    filled = np.flip(filled_reversed, axis=0)

    return restore_original_type(filled, original_var), unusable


def target_interpolation(var_to_fill, window):
    """
    Generates a target matrix using the average of target_forward() and
    target_backward()
    """
    original_var = var_to_fill.copy()

    # Make sure all inputs are np arrays
    var_to_fill = convert_to_np_arrays(var_to_fill)
    filled = var_to_fill.copy()
    var_to_fill_reversed = np.flip(var_to_fill, axis=0)
    filled_r = var_to_fill_reversed.copy()
    unusable, unusable_r = [], []

    for i in range(1, var_to_fill.shape[0]):
        filled_tar_for, unusable_tar_for = target_forward(filled[:i+1, :], window)
        filled[:i+1, :] = filled_tar_for[:i+1, :]
        unusable.extend(unusable_tar_for)

        filled_tar_back, unusable_tar_back = target_forward(filled_r[:i+1, :], window)
        filled_r[:i+1, :] = filled_tar_back[:i+1, :]
        unusable_r.extend(unusable_tar_back)

    filled_r = np.flip(filled_r, axis=0)

    # Take the simple average of the two estimations
    # If either is NaN, the returned value will be NaN
    # This step is done twice in Ox version - why?
    filled = (filled + filled_r) / 2
    unusable = list(set(unusable + unusable_r))

    return restore_original_type(filled, original_var), unusable
