# -*- coding: utf-8 -*-
"""
mre
===
Defines the MRE class, to read and write MRE-format ('machine-readable') data
files e.g. as produced by E3ME.
"""

from collections import defaultdict
import math
import os
from typing import Dict
import warnings

import numpy as np

from celib.exceptions import DimensionError


class MRE:
    """Class to handle MRE-format ('machine-readable') data files e.g. as produced by E3ME.

    Examples
    --------
    >>> import numpy as np
    >>> from celib import MRE

    # Read data from an MRE file
    >>> with MRE('path/to/some/file.mre') as mre:
    ...     # Return value is a one-element list containing a 2D NumPy array of
    ...     # real GDP (regions x time) - use `[0]` to get the array
    ...     real_gdp = mre['RGDP'][0]

    ...     # YRE is a three-dimensional variable (regions x sector x time)
    ...     # Returns a sorted (by region) list of 2D NumPy arrays. Any missing
    ...     # arrays in the sequence are filled with `None`
    ...     employment = mre['YRE']

    # Write data to an MRE file:
    #  - x: a 2D NumPy array e.g. as RGDP would be (above)
    #  - y: a list of 2D NumPy arrays e.g. as YRE would be (above)
    >>> data = {'x': np.zeros((3, 4)), 'y': [np.ones((2, 3))] * 4}
    >>> comments = {'x': 'Variable X (2D)', 'y': 'Variable Y (3D)'}

    >>> with MRE() as mre:
    ...     mre.write(data, comments, 'path/to/some/file.mre')
    """

    def __init__(self, filename=None):
        self.fn = filename

        self.index = None
        if self.fn is not None:
            self.index = self._read_structure(self.fn)

    def close(self):
        """Deprecated as of version 0.4.1 of `celib`: calls to this method can be safely deleted."""
        warnings.warn('As of version 0.4.1 of `celib`, `MRE.close()` is deprecated. '
                      'You can safely delete these calls from your script.',
                      DeprecationWarning)

    @staticmethod
    def _read_structure(filepath: str) -> Dict:
        """Read the structure of an MRE file, returning the index as a dictionary."""
        # Initialise variable to store final index
        # TODO: Figure out why it needs to be a `defaultdict`, rather than a
        #       regular empty dictionary, `{}`
        index = defaultdict(str)

        lines_read = 0  # Counter of number of lines read so far (for debugging)
        j = 0  # To count the number of lines read for the current variable

        with open(filepath) as infile:
            pos = 0  # position in the file, for seek()
            ln = None  # length of a number in this file
            # need to use iter to disable buffer
            for s in iter(infile.readline, ''):
                lines_read += 1
                if j == 0:  # header

                    parts = s.lstrip().split()
                    rows = int(parts[0])
                    cols = int(parts[1])

                    varname = parts[2]

                    secdim = 1  # default in case no second dimension

                    if parts[3] == '?':
                        secdim = int(parts[4])  # second dimension
                        desc = ' '.join(parts[3:])
                    else:
                        desc = ' '.join(parts[3:])
                    j += 1
                    ln = None
                    pos = infile.tell()  # save position
                else:  # data
                    if ln is None:  # calculate length of number
                        if s[0] == '*':
                            raise ValueError(
                                'Cannot determine the number format of %s, because of a severe float overflow.' % varname)

                        dt = s.find('.')  # find first decimal point
                        if dt == -1:
                            raise ValueError(
                                'No decimal points in %s, failed to determine number format.' % varname)

                        sp = s[dt:].find(' ')  # and first space after that
                        # but what if there's a decimal before the first space
                        # - it means the subsequent number has overflown into
                        # this one
                        if s[dt:].find('.') < sp:
                            sp = -1

                        if sp == -1:
                            # try getting the number of decimal places from the
                            # last number
                            sp = s[::-1].find('.')

                            # but we may still fail to get it:
                            if s[-1] == '*' or sp == -1:
                                raise ValueError(
                                    'Numerical overflow in %s, failed to determine number format.' % varname)

                        ln = dt + sp
                        valperline = round(len(s) / float(ln))

                        # how many rows does a matrix row span?
                        nperl = int(math.ceil(cols / valperline))
                        jmax = rows * nperl

                        index[varname] = index[varname] if varname in index else dict()
                        index[varname][secdim - 1] = (
                            {'position': pos, 'lines': jmax, 'rows': rows, 'cols': cols, 'nperl': nperl, 'comment': desc})

                    if j == jmax:
                        j = 0   # All lines read: reset counter
                    else:
                        j += 1  # Still reading the current variable: increment counter

        return index

    def get_full_var(self, varname, out='list'):
        """
        Loads all dimensions of a variable into a list of numpy arrays.

        The loading is fairly slow, because it opens the source file when
        reading each matrix. This is due to wrapping get_var(). Maybe include
        this functionality in get_var() itself (e.g. when ommiting 'secdim')
        """
        if varname not in self.index:
            raise ValueError(
                'Trying to load a non-existent variable: %s' % varname)
        if out not in ['list', 'df']:
            raise ValueError('Output of get_full_var can only be list or df')

        inds = self.index[varname]
        ind_codes = list(inds.keys())

        res = [None] * (ind_codes[-1]+1)

        for j in ind_codes:
            res[j] = self.get_var(varname, j)

        if out == 'list':
            return res
        else:
            import pandas as pd
            resdf = None
            for j, mt in enumerate(res):
                if mt is None:
                    pass
                else:
                    df = pd.DataFrame(mt)
                    df.insert(0, 'sector', range(mt.shape[0]))
                    df.insert(0, 'dimension', j)
                    resdf = df if resdf is None else pd.concat([resdf, df])

        return resdf.set_index(['dimension', 'sector']).T

    def get_full_vars(self, varnames):
        """
        Get a fully annotated dataframe with multiple variables.
        """
        import pandas as pd
        if type(varnames) != list:
            raise ValueError('When reading multiple variables in get_full_vars\
            , you need to supply a list of names')

        ret = None
        for vn in varnames:
            df = self.get_full_var(vn, out='df').T.reset_index()
            # TODO: or did I use 'varname' elsewhere?
            df.insert(0, 'variable', vn)
            ret = df if ret is None else pd.concat([ret, df])

        return ret.set_index(['variable', 'dimension', 'sector']).T

    def get_var(self, varname, secdim):
        """Load a specific variable from an MRE file. This is a part of the
           index passed through _read_structure().
        """
        if varname not in self.index:
            raise ValueError('Trying to load a non-existent variable')

        index = self.index[varname][secdim]
        # current variable
        cvar = np.ndarray(shape=(index['rows'], index['cols']), dtype=float)
        csec = []  # current sector of current variable
        ln = None
        # default value to put in when stars are present (overflow)
        dvl = -99999999

        with open(self.fn) as infile:
            infile.seek(index['position'])  # get to the right place
            j = 0  # row in the textual representation in the MRE file
            crow = 0  # current row in the matrix
            for s in infile:  # range(1, index['lines']+1):
                if j == index['lines']:
                    return cvar
                else:
                    j += 1

                if ln is None:  # calculate length of number
                    if s[0] == '*':
                        raise ValueError(
                            'Cannot determinte the number format of %s, because of a severe float overflow.' % varname)

                    dt = s.find('.')  # find first decimal point
                    if dt == -1:
                        raise ValueError(
                            'No decimal points in %s, failed to determine number format.' % varname)
                    sp = s[dt:].find(' ')  # and first space after that
                    # but what if there's a decimal before the first space - it
                    # means the subsequent number has overflown into this one
                    if s[dt:].find('.') < sp:
                        sp = -1

                    if sp == -1:
                        # try getting the number of decimal places from the
                        # last number
                        sp = s[::-1].find('.')

                        # but we may still fail to get it:
                        if s[-1] == '*' or sp == -1:
                            raise ValueError(
                                'Numerical overflow in %s, failed to determine number format.' % varname)

                    ln = dt + sp
                    valperline = round(len(s) / float(ln))
                ret = [str.strip(s[ln * m:ln * (m + 1)])
                       for m in range(int(valperline))]

                if '*' in s:  # replace overflows
                    warnings.warn('Found asterisks when parsing %s' % varname)
                    ret = [str(dvl) if '*' in ll else ll for ll in ret]

                ret = [float(x) for x in ret if len(x) > 0]

                csec += ret

                # we have finished a sector, append to variable
                if j > 0 and j % index['nperl'] == 0:
                    cvar[crow] = csec
                    csec = []
                    crow += 1

        return cvar

    def write(self, data, comm, outfile):
        """Write `data` and accompanying metadata (in `comm`) to `outfile`. **Can only be called from an empty MRE object** i.e. an object that has not been previously used to read data.

        Parameters
        ----------
        data =  A dictionary of one or more variables. Elements of the
                dictionary must be either numpy arrays or lists of numpy
                arrays. The numpy arrays contained must be two dimensional.

        comm =  A dictionary of variable descriptions to be printed as part of
                the header. Elements of the dictionary must be either strings
                or lists of strings.

        outfile = Name and filepath of the mre file to write to.

        Notes
        -----
        This function allows the user to write one or multiple variables to a
        new/empty mre file that'll be created within the function.  The
        function writes out one variable at a time using a loop within the
        function.

        Regardless of the number of variables stored, the data must be a
        dictionary of numpy arrays where the key(s) correspond to the assigned
        variable name(s).

        Comments must also be provided as a dictionary of strings where the
        key(s) correspond to the assigned variable name(s).

        In the case of writing variables with an additional dimension
        (e.g. industry variables in E3ME), each entry in the data
        dictionary should be a list of numpy arrays. Each entry in the comment
        dictionary can either be a list of strings or a string, in which case
        the additional dimensions and counters are deduced.

        For example, for a variable in `data` with key (name) 'YRE':
         - if the data is an X-length list of 2D NumPy arrays
         - and the accompanying comment is either a single string ('industry
           employment') or a one-element list containing a string (['industry
           employment'])

        The method writes the individual 2D arrays in turn, with header
        comments as follows:
         - '? 01 industry employment'
         - '? 02 industry employment'
         - '? 03 industry employment'
         ...
         - '?  X industry employment'

        For each variable in `data` and accompanying description(s) in `comm`,
        the treatment is as follows, depending on their respective dimensions:

        | # elements | # comments | Result                                    |
        |   (arrays) |            |                                           |
        |------------+------------+-------------------------------------------|
        |          1 |          1 | Write the variable and its accompanying   |
        |            |            | comment                                   |
        |            |            |                                           |
        |          N |          1 | Write the variables with a common         |
        |            |            | comment, but with an incrementing         |
        |            |            | counter e.g. 'industry employment' ->     |
        |            |            | '? 01 industry employment',               |
        |            |            | '? 02 industry employment',               |
        |            |            | '? 03 industry employment', ...           |
        |            |            |                                           |
        |          N |          N | Write each variable with its accompanying |
        |            |            | comment                                   |

        IMPORTANT NOTE: If there are missing elements in the additional
        dimension (e.g. missing countries), the user should provide a list of
        comments, indicating the dimension number (e.g. ['? 01 actuals for
        PYR', '? 04 actuals for PYR', '? 06 actuals for PYR', ...]).

        This method can only be called using an empty MRE object; that is, one
        that has not already been used to read data:

        # Fine
        >>> with MRE() as mre:
        ...     mre.write(...)

        # Raises a `RuntimeError`
        >>> with MRE('path/to/somefile.mre') as mre:
        ...     mre.write(...)

        While the latter is not necessarily problematic, we choose to guard
        against it because:
        1. at best, not doing so risks confusion because the contents written
           by this method will bear no relation to the contents of the MRE
           object. This is the consequence of an earlier design decision that
           we have chosen to retain, to preserve backward compatibility.
        2. as a worst case, if the user overwrites the same file they read
           from, there is a risk that the object's contents will no longer
           correctly access the contents of that file. We have opted to head
           off this possibility at the earliest opportunity, rather than
           surprise (or confound) the user at a later stage in their script.
            * This situation is still possible if the user has another
              (i.e. separate) MRE object that points to the same file but this
              is not easily dealt with unless we further complicate things
              (e.g. by introducing code to lock a file).
            * In any case, this is a situation that we can reasonably expect a
              responsible user to avoid, whereas the risk we guard against is
              in part a product of the underlying implementation (that we would
              otherwise need the user to understand).
        """
        # Initial check of arguments
        if type(data) != dict:
            raise TypeError('`data` must be a dictionary of data, either NumPy arrays or lists of arrays')
        if type(comm) != dict:
            raise TypeError('`comm` must be a dictionary of strings (comments) to accompany `data`')
        if comm.keys() != data.keys():
            raise ValueError('The keys of the arguments comm and data should be the same')
        if type(outfile) != str:
            raise TypeError('Output filename must be a string')

        # Raise an exception if the user tries to write a file using a
        # non-empty MRE object (i.e. one that has already been used to read
        # data)
        if self.index is not None:
            raise RuntimeError(
                'Cannot write files using a non-empty MRE object (i.e. one previously used to read data): '
                'Always create a new, empty MRE object to write data.')

        # Check that the contents of `data` are all of the right type(s) and
        # that the accompanying comment(s) (in `comm`) are of the right
        # length(s)
        for name, variable in data.items():
            comment = comm[name]

            # NumPy array: Check for two dimensions and that there is only one
            # accompanying comment
            if isinstance(variable, np.ndarray):
                # Check that the array is 2D
                if variable.ndim != 2:
                    raise DimensionError(
                        "`data['{}']` is not a 2D NumPy array".format(name))

                # Check comment: Must be either a string or a one-item list
                # containing a string
                if isinstance(comment, str):
                    # No check for length: zero-length strings are permitted
                    pass

                elif isinstance(comment, list):
                    if len(comment) != 1 or not isinstance(comment[0], str):
                        raise ValueError(
                            "`data['{}']` is a 2D NumPy array: "
                            "the accompanying comment (in `comm`) must be "
                            "either a string or a one-element list containing "
                            "a string".format(name))

                else:
                    raise TypeError(
                        "Invalid variable type for comment with key '{}': {} "
                        "- must be either a string or a one-element list containing a string"
                        .format(name, type(comment)))

            # List of NumPy arrays: Check data for 2D NumPy arrays and that the
            # length of the comments variable conforms (is either 1 or N)
            elif isinstance(variable, list):

                # Check for 2D NumPy arrays
                for i, item in enumerate(variable):

                    # Check specifically for `None` and raise an exception:
                    # this is the placeholder for missing data in the read
                    # methods
                    if item is None:
                        raise ValueError(
                            "`data['{}'][{}] is `None` (typically indicating a missing element). "
                            "`None` cannot be written to an MRE file. "
                            "Remove before passing to `MRE.write()`, "
                            "along with individual comments and index numbers for the arrays".format(name, i))

                    # Catch other type mismatches
                    if not isinstance(item, np.ndarray):
                        raise TypeError(
                            "`data['{}'][{}]` is not a 2D NumPy array".format(name, i))

                    # Check that the array is 2D
                    if item.ndim != 2:
                        raise DimensionError(
                            "`data['{}'][{}]` is not a 2D NumPy array".format(name, i))

                # Check comment(s)
                if isinstance(comment, str):
                    # No check for length: zero-length strings are permitted
                    pass

                elif isinstance(comment, list):
                    # List: Check for 1 or N comment(s)
                    if len(comment) not in (1, len(variable)):
                        raise ValueError(
                            "`data['{}']` is a list of {} NumPy arrays. "
                            "Accompanying comment(s) entry, `comm['{}']`, has length {}: must be either "
                            "1 (for automatic comment expansion) "
                            "or {} (one comment per array)".format(
                                name, len(variable),
                                name, len(comment),
                                len(variable)))

                    # Check all comments are strings
                    for i, x in enumerate(comment):
                        if not isinstance(x, str):
                            raise TypeError(
                                "`comm['{}'][{}]` is not a string: "
                                "`comm['{}'] must be either string or a list of strings".format(
                                    name, i, name))
                else:
                    raise TypeError(
                        "Invalid variable type for comment with key '{}': {} "
                        "- must be either a string "
                        "or a list of strings "
                        "(of length 1 or {}, to match the number of matrices in `data['{}']`)"
                        .format(name, type(comment),
                                len(variable), name))

            # Neither a NumPy array or a list of NumPy arrays: TypeError
            else:
                raise TypeError(
                    "Invalid variable type for `data['{}']` "
                    "(must be either a NumPy array or a list of NumPy arrays): {}".format(
                        name, type(variable)))

        # Having passed all the checks, construct and write the data to disk,
        # one variable at a time
        with open(outfile, 'w') as f:
            for name in data.keys():
                variable = data[name]
                comment = comm[name]

                # Cast to lists as needed, to simplify the process that follows
                if not isinstance(variable, list):
                    variable = [variable]
                if not isinstance(comment, list):
                    comment = [comment]

                # Special handling of N arrays but only one comment: construct
                # variable descriptions with an incrementing counter
                if len(variable) != len(comment):
                    comment = ['? {:02d} {}'.format(i + 1, comment[0])
                               for i in range(len(variable))]

                # Write header(s) and data
                for values, description in zip(variable, comment):
                    f.write('{}  {}  {}  {}\n'.format(*values.shape, name, description))
                    np.savetxt(f, values, fmt='%15.4f', delimiter='')


    def __contains__(self, item):
        return item in self.index

    def __getitem__(self, key):
        return self.get_full_var(key)

    def __setitem__(self, key, value):
        raise NotImplementedError('MRE modification not implemented yet')

    def __len__(self):
        return len(self.index)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
