# -*- coding: utf-8 -*-
"""
db1
===
Defines the `DB1` class and supporting functions to handle CE DB1-format
databanks.
"""

from collections import OrderedDict
from enum import IntEnum
import inspect
import os
import shutil
import struct
import time
from typing import List, NamedTuple, Optional, Union
import warnings

import numpy as np

from celib.exceptions import DimensionError


class VariableType(IntEnum):
    """Mapping of variable types to DB1 type identifiers (integer codes).

    Copied 03/09/2019 from:
        M:/Ox/CELib/Source/Classes/ClassDB1.ox

    (See the documentation next to `decl m_mTypes ...`.)
    """
    # Standard types in Ox
    CHARACTER = STRING = 0
    INTEGER = 1
    DOUBLE = FLOAT = 2
    MATRIX = 3
    ARRAY = 4

    # CE classes
    MODEL = 5
    MODEL_FUNCTION = 6
    SPECIFICATION = 7

    VARIABLE = 8
    VARIABLES = 9

    EQUATION = 10
    EQUATIONS = 11

    MDV = 12


class IndexEntry(NamedTuple):
    """Container for metadata about a single DB1 variable."""
    name: str
    code: int
    comment: str

    frequency: int
    start_year: int
    start_period: int

    type_: VariableType
    nrows: int
    ncols: int

    user: str
    date: str
    caller: str  # Name of calling job (version 3.1 onwards)

    address: int  # Byte start position
    next_address: int  # Start position of next variable


class DB1:
    """CE binary database format e.g. for IDIOM models.

    Notes
    -----
    Currently only supports, for 'get' operations:
     - strings
     - integers
     - floats (Ox doubles)
     - 2D NumPy arrays (Ox matrices)
     - lists (Ox arrays), recursively as needed
     - CE MDVs (but as lists, rather than MDV objects)

    for 'put' (write) operations:
     - strings
     - integers
     - floats (Ox doubles)
     - 2D NumPy arrays (Ox matrices)
     - lists (Ox arrays), recursively as needed

    So full support for the Python equivalents of base Ox data
    types. Extraction of the data inside CE MDVs is also possible.

    Example
    -------
    >>> databank = DB1('path/to/some/file.db1')  # Mode is 'read' by default
    >>> print(databank.index)                    # The `index` property is a `pandas` DataFrame

    >>> x = databank.get('variable_name')        # By variable name (str)
    >>> y = databank.get(12345678)               # By variable code (int)

    # Implements some container methods
    >>> a = databank['ABC']
    >>> b = databank[123]

    >>> 'xyz' in databank

    >>> print(databank.varnames)                 # List of str
    >>> print(databank.codes)                    # List of int

    >>> databank.close()
    """

    VERSION: float = 3.1
    USER_STR_LENGTH: int = 8

    def __init__(self, filename=None, mode='read', *, user=None):
        self._filepath = filename

        modes = ['read', 'write']
        if mode not in modes:
            raise ValueError(
                'Unrecognised `mode` argument: {} (must be one of {})'.format(
                    mode, ', '.join(modes)))

        self._mode = mode

        self._header = {'version': self.VERSION,
                        'user_str_length': self.USER_STR_LENGTH}
        self._header_byte_positions = {}

        self._index_by_code = OrderedDict()
        self._index_by_name = OrderedDict()

        self._start_year = 0
        self._frequency = 0
        self._start_period = 0

        # List of variable codes
        self.codes = []

        # List of (unique) variable names
        self.varnames = []

        if user is None:
            self._user = user  # Set attribute directly to permit `None`
        else:
            self.user = user  # Otherwise, use standard property interface

        # Attribute to store the file object for the databank
        self._file = None

        if self._filepath is not None:
            self.open(self._filepath)

    def open(self, filename: str):
        """Open the databank at `filename` (leave it open) and read its header."""
        modes = {
            'read': 'rb',
            'write': 'rb+',
        }
        mode = modes[self._mode]

        # Need to explicitly close with `self.close()`
        self._file = open(filename, mode)

        self._header['version'] = self._read_double()

        # Read timestamps as integers and convert from seconds to date strings
        timestamps = ['time_created', 'time_modified', 'time_condensed']
        for x in timestamps:
            # Record byte position
            self._header_byte_positions[x] = self._file.tell()
            # Get timestamp
            self._header[x] = time.ctime(self._read_integer())

        # Read integer information on string lengths and numbers of variables
        header_info = ['user_str_length', 'number_of_variables', 'number_of_live_variables']
        for x in header_info:
            # Record byte position
            self._header_byte_positions[x] = self._file.tell()
            # Get info
            self._header[x] = self._read_integer()

        # Read usernames for various actions
        user_info = ['user_created', 'user_modified', 'user_condensed']
        for x in user_info:
            # Record byte position
            self._header_byte_positions[x] = self._file.tell()
            # Get username
            self._header[x] = self._read_string(self._header['user_str_length'])

        # Store byte position of first item of data and then build the index
        self._header_byte_positions['variable_start_position'] = self._file.tell()
        self._build_index()

    def create(self, fn: str) -> None:
        """Create a new databank file at `fn`."""

        # Check that a valid username has been set
        if self.user is None:
            raise RuntimeError(
                "Username must be set before creating a new databank: "
                "use, for example, `db1.user = 'XY'`")

        self._file = open(fn, 'wb')

        # Write header
        self._write_double(self._header['version'])

        # Current time x3 (created, modified, condensed)
        current_time = int(time.time())
        for _ in range(3):
            self._write_integer(current_time)

        self._write_integer(self._header['user_str_length'])

        # Number of variables (and number that are live)
        self._write_integer(0)
        self._write_integer(0)

        # Created/modified/condensed by
        user_name = self.user.ljust(self._header['user_str_length'])
        for _ in range(3):
            self._write_string(user_name)

        # Update the mode
        self._mode = 'write'

        # Close and then reopen the databank to generate a new, complete index
        # TODO: This is a workaround because there isn't a generalised
        #       treatment for DB1 databank headers. Rather than duplicate all
        #       that code here, the simplest thing to do is to close the
        #       databank and reopen it as if it were an existing databank
        self._file.close()
        self.open(fn)

    def _build_index(self) -> None:
        # Go to the start of the variable data in the databank (by byte
        # position)
        self._file.seek(self._header_byte_positions['variable_start_position'])

        # Initialise index variables
        self._index_by_code = OrderedDict()
        self._index_by_name = OrderedDict()
        self.codes = []
        self.varnames = []

        # Read all variable information (including scratched variables)
        for _ in range(self._header['number_of_variables']):
            name_str_length = self._read_integer()
            comment_str_length = self._read_integer()

            # From DB1 version 3.1 onwards, variable metadata include the path
            # of the calling job that wrote the variable to the databank
            caller_str_length = None
            if self._header['version'] > 3.0:
                caller_str_length = self._read_integer()

            entry = IndexEntry(
                name=self._read_string(name_str_length),
                comment=self._read_string(comment_str_length),

                # Only read the calling job name if DB1 version 3.1 or later
                # (`caller_str_length` is not `None`)
                caller=(self._read_string(caller_str_length)
                        if caller_str_length is not None
                        else ''),

                user=self._read_string(self._header['user_str_length']),
                date=time.ctime(self._read_integer()),
                code=self._read_integer(),

                address=self._read_integer(),
                next_address=self._read_integer(),

                type_=self._read_integer(),

                frequency=self._read_integer(),
                start_year=self._read_integer(),
                start_period=self._read_integer(),

                nrows=self._read_integer(),
                ncols=self._read_integer(),
            )

            # Set file pointer to jump to the next variable
            self._file.seek(entry.next_address)

            # Continue if variable was scratched (only store live variables to
            # the remaining attributes)
            if entry.code < 0:
                continue

            # Store variable information
            # TODO: In time (but no time soon), review how to store the
            #       `IndexEntry` object directly, rather than recasting to a
            #       dictionary for compatibility with the existing codebase
            self._index_by_code[entry.code] = entry._asdict()
            self._index_by_name[entry.name] = entry._asdict()

            self.codes.append(entry.code)
            self.varnames.append(entry.name)

        # Keep unique variable names only
        # TODO: Decide whether the most recent name should be at the first or
        #       last position
        self.varnames = list(set(self.varnames))

    @property
    def header(self):
        """Read-only dictionary of databank header information (metadata).

        Note
        ----
        As an object property, the contents of the header can be viewed but not
        modified: there's no accompanying property setter. This ensures the
        attribute is read-only such that code like the below will throw an
        exception:

        >>> db1.header = None
        AttributeError: can't set attribute

        However, while code like the below won't modify the databank header,
        there doesn't seem to be an easy way to tell the user that this has
        failed. Nothing happens, and silently:

        # No warning message, but `db1.header['user_modified']` remains
        # unchanged
        >>> db1.header['user_modified'] = 'XY'

        TODO: See if there's a way to implement some kind of warning/error for
              the second case: (how) does pandas do it?
        """
        return self._header.copy()

    @property
    def user(self) -> str:
        """The current user (typically their initials)."""
        return self._user

    @user.setter
    def user(self, user: str) -> None:
        if not isinstance(user, str):
            raise TypeError('`user` argument must be of type `str`')

        if len(user) == 0:
            raise ValueError('`user` argument cannot be a zero-length string')

        if len(user) > self._header['user_str_length']:
            raise ValueError(
                "`user` argument '{}' is too long: maximum length is {} character(s)".format(
                    user, self._header['user_str_length']))

        self._user = user

    def set_user(self, user: str) -> None:
        """Set the current user. Deprecated: prefer `db1.user = 'XY'`."""
        warnings.warn(
            "`DB1.set_user()` will continue to work as expected "
            "but, from `celib` version 0.4.0 onwards, "
            "prefer, for example: `db1.user = 'XY'`",
            DeprecationWarning)
        self.user = user

    def get_user(self) -> str:
        """Get the current user. Deprecated: prefer `user = db1.user`."""
        warnings.warn(
            '`DB1.get_user()` will continue to work as expected '
            'but, from `celib` version 0.4.0 onwards, '
            'prefer, for example: `user = db1.user`',
            DeprecationWarning)
        return self.user

    @property
    def start_year(self) -> int:
        """The start year to apply when writing new data."""
        return self._start_year

    @start_year.setter
    def start_year(self, year: int) -> None:
        if not isinstance(year, int):
            raise TypeError('`year` argument must be of type `int`')
        self._start_year = year

    def set_start_year(self, syear: int) -> None:
        """Set the start year. Deprecated: prefer `db1.start_year = 2019`."""
        warnings.warn(
            '`DB1.set_start_year()` will continue to work as expected '
            'but, from `celib` version 0.4.0 onwards, '
            'prefer, for example: `db1.start_year = 2019`',
            DeprecationWarning)
        self.start_year = syear

    # Retain the `gsyear` ('global start year'?) attribute as a property, for
    # backward compatibility
    @property
    def gsyear(self) -> int:
        """The start year to apply when writing new data. Deprecated: prefer `db1.start_year`."""
        warnings.warn(
            'The `DB1.gsyear` attribute will continue to work as expected '
            'but, from `celib` version 0.4.0 onwards, '
            'prefer, for example: '
            '`start_year = db1.start_year` or `db1.start_year = 2019`',
            DeprecationWarning)
        return self.start_year

    @gsyear.setter
    def gsyear(self, year: int) -> None:
        """Set the start year. Deprecated: prefer `db1.start_year = 2019`."""
        warnings.warn(
            'The `DB1.gsyear` attribute will continue to work as expected '
            'but, from `celib` version 0.4.0 onwards, '
            'prefer, for example: `db1.start_year = 2019`',
            DeprecationWarning)
        self.start_year = year

    @property
    def frequency(self) -> int:
        """The frequency to apply when writing new data."""
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: int) -> None:
        if not isinstance(frequency, int):
            raise TypeError('`frequency` argument must be of type `int`')
        self._frequency = frequency

    @property
    def start_period(self) -> int:
        """The start period to apply when writing new data."""
        return self._start_period

    @start_period.setter
    def start_period(self, start_period: int) -> None:
        if not isinstance(start_period, int):
            raise TypeError('`start_period` argument must be of type `int`')
        self._start_period = start_period

    @property
    def index(self):
        """Databank index as a `pandas` DataFrame."""
        from pandas import DataFrame

        if len(self._index_by_code):
            index_as_df = (DataFrame(self._index_by_code).T
                           .reindex(columns=IndexEntry._fields)
            )
        else:  # For empty databanks
            index_as_df = DataFrame(columns=IndexEntry._fields)

        return index_as_df.set_index('code')

    def close(self):
        """Close connection to file on disk."""
        self._file.close()

    def _read_double(self) -> float:
        return struct.unpack('d', self._file.read(8))[0]

    def _read_doubles(self, n: int) -> List[float]:
        return [self._read_double() for _ in range(n)]

    def _read_integer(self) -> int:
        return struct.unpack('i', self._file.read(4))[0]

    def _read_integers(self, n: int) -> List[int]:
        return [self._read_integer() for _ in range(n)]

    def _read_string(self, length: int, trim: bool = True, *,
                     encoding: str = 'ascii', errors: str = 'ignore') -> str:
        """Return `length` characters as a string; if `trim`,
        remove leading and trailing whitespace."""
        sequence = struct.unpack('{}s'.format(length),
                                 self._file.read(length))[0]
        string = sequence.decode(encoding, errors)

        if trim:
            string = string.strip()

        return string

    def get_start_year(self, code_name: Union[int, str]) -> int:
        """Return the start year for a variable, whether specified by code
        (integer) or name (string)."""
        corresponding_code = self._get_code(code_name)
        return self._index_by_code[corresponding_code]['start_year']

    def _get_code(self, code_or_name: Union[int, str]) -> int:
        """Return the databank code that corresponds to `code_or_name`.

        Notes
        -----
        If `code_or_name` is already a valid integer code, just return
        it. Otherwise, find the code that corresponds to the variable name
        string.
        """
        # User supplied an integer (variable code): check it exists in the
        # index and return
        if isinstance(code_or_name, int):
            if code_or_name not in self.codes:
                raise KeyError(
                    'Variable with code {} not found in databank index'.format(
                        code_or_name))
            return code_or_name

        # User supplied a string (variable name): check it has a corresponding
        # code in the index and return
        elif isinstance(code_or_name, str):
            try:
                return self._index_by_name[code_or_name]['code']
            except KeyError:
                raise KeyError(
                    'Variable with name {} not found in databank index'.format(
                        code_or_name))

        # Error on any other variable type
        else:
            raise TypeError(
                'Unrecognised type: {} (must be either int or str)'.format(
                    code_or_name))


    def get(self, vind: Union[int, str]):
        """Return a variable from the databank either by code (int) or name (str)."""
        variable_code = self._get_code(vind)
        return self._read_data(variable_code)

    def read_df(self, vind: Union[int, str]):
        """Return the variable with code (int) or name (str) `vind` as a DataFrame."""
        from pandas import DataFrame

        data = self.get(vind)

        start_year = self.get_start_year(vind)
        span = range(start_year, start_year + data.shape[1])

        return DataFrame(data, columns=span)

    def _read_matrix(self, nrows: int, ncols: int):
        """Read and return an (`nrows` x `ncols`) NumPy array."""
        return np.array([self._read_doubles(ncols) for i in range(nrows)])

    def _read_array(self, n: int):
        """Return an array (list) of length `n`, filling recursively as needed."""
        array = []

        for _ in range(n):
            type_, nrows, ncols = self._read_integers(3)

            if type_ == VariableType.STRING:
                array.append(self._read_string(nrows))

            elif type_ == VariableType.INTEGER:
                array.append(self._read_integer())

            elif type_ == VariableType.DOUBLE:
                array.append(self._read_double())

            elif type_ == VariableType.MATRIX:
                array.append(self._read_matrix(nrows, ncols))

            elif type_ == VariableType.ARRAY:  # Fill recursively
                array.append(self._read_array(nrows))

            else:
                raise ValueError('Unrecognised type identifier: {}'.format(type_))

        return array

    def _read_data(self, vcode):
        if vcode not in self._index_by_code:
            raise KeyError(
                'Variable with code {} not found in databank index'.format(vcode))

        # Extract variable metadata and seek to the byte position of the data
        entry = self._index_by_code[vcode]
        self._file.seek(entry['address'])
        # Read data according to its type
        if entry['type_'] == VariableType.CHARACTER:
            raise NotImplementedError

        elif entry['type_'] == VariableType.INTEGER:
            raise NotImplementedError

        elif entry['type_'] == VariableType.DOUBLE:
            raise NotImplementedError

        elif entry['type_'] == VariableType.MATRIX:
            return self._read_matrix(entry['nrows'], entry['ncols'])

        elif entry['type_'] in [VariableType.ARRAY, VariableType.MDV]:
            nrows = entry['nrows']
            ncols = entry['ncols']

            if ncols != 0:
                raise ValueError(
                    'Variable with code {} is an array '
                    'but has non-zero columns in its metadata: '
                    'this is undefined behaviour'.format(vcode))

            return self._read_array(nrows)

        else:
            raise ValueError(
                'Unsupported (or unrecognised) variable type identifier: ')

    def _write_integer(self, value: int):
        self._file.write(struct.pack('i', value))

    def _write_double(self, value: float):
        self._file.write(struct.pack('d', value))

    def _write_string(self, value: str):
        for c in bytes(value, 'ascii'):
            self._file.write(struct.pack('c', bytes([c])))


    def _check_write_arguments(self, vname: str, vcode: int, comment: str, caller: Optional[str]):
        """Run common checks of databank state and arguments before writing.

        Parameters
        ----------
        vname : str
            Variable name to write: must be a string of non-zero length
        vcode : int
            Variable code to write: must be int and not already present in the index
        comment : str
            Comment to accompany the variable: must be a string
        caller : str or `None`
            Name/path of calling script/job
        """
        # Check databank state
        if self._mode != 'write':
            raise RuntimeError("Databank `mode` must be set to 'write' before adding new variables")
        if self.user is None:
            raise RuntimeError('Must set a username before writing')

        # `vname` should be a string of non-zero length
        if not isinstance(vname, str):
            raise TypeError('Variable name argument `vname` must be a string')
        if len(vname) == 0:
            raise ValueError('Variable name argument `vname` cannot be a zero-length string')

        # `vcode` must be an integer with a value not already in use by the
        # databank
        if not isinstance(vcode, int):
            raise TypeError('Variable code argument `vcode` must be an integer')
        if vcode in self.codes:
            raise ValueError('There is already a variable in the databank with code {}; '
                             'either use a different code for `vcode` or scratch first'
                             .format(vcode))

        # `comment` must be a string
        if not isinstance(comment, str):
            raise TypeError('Variable comment argument `comment` must be a string')

        # `caller` must be either a string or `None`
        if not isinstance(caller, str) and caller is not None:
            raise TypeError('Variable caller argument `caller` must be a string or `None`')

    def write_matrix(self, vname: str, vcode: int, vcont: np.ndarray,
                     comment: str = '', caller: Optional[str] = None, *,
                     build_index: bool = True, timecreated: Optional[int] = None
                     , user: Optional[str] = None):
        """Write a 2D NumPy array to the databank.

        Parameters
        ----------
        vname : str
            label for variable being written to the databank
        vcode : int
            unique code identifier for variable entry in databank index
        vcont : (m x n) numpy array
            2 dimensional numpy array representing a matrix (or vector if m or n =1)
            of data to write to the databank
        comment : str
            Description of the variable to be added
        caller : str, default `None`
            The name/path of the calling script/job e.g. `__file__`.
            If `None` (the default), tries to infer the caller from the call
            stack. Depending on the calling environment, this may be inaccurate.
        build_index : bool, default True
            If True, index is rebuilt after variable is added to databank
        timecreated: int, default None
            timestamp original variable on databank was created (used by condense)
            if none, uses the current time
        user: str, default None
            user that wrote original variable on databank (used by condense)
            if none, uses current set user
        Returns : None

        Raises
        ------
            ValueError:
                (a) Databank mode is not set to write
                (b) Databank user has not been set
                (c) Function arguements are not defined as describe in parameters
        """
        # Check write arguments
        self._check_write_arguments(vname, vcode, comment, caller)

        # `vcont` must be a 2D NumPy array
        if not isinstance(vcont, np.ndarray):
            raise TypeError('Data (`vcont` argument) must be a 2D NumPy array')

        if vcont.ndim != 2:
            raise DimensionError('Data (`vcont` argument) must be a 2D NumPy array')

        self._file.seek(0, 2)  # end of file

        # write header
        self._write_integer(len(vname))
        self._write_integer(len(comment))

        # If no `caller` passed, try to infer from the call stack
        # TODO: See if there's a more flexible/accurate method that works
        #       across multiple calling environments
        if caller is None:
            caller = os.path.abspath(inspect.stack()[-1][1])

        if self._header['version'] > 3.0:
            self._write_integer(len(caller))

        self._write_string(vname)
        self._write_string(comment)

        if self._header['version'] > 3.0:
            self._write_string(caller)

        if user is None:
            # pad username with spaces to fill necessary quota
            self._write_string(self.user.ljust(self._header['user_str_length']))
        else:
            #Keep username of copied variable
            self._write_string(user.ljust(self._header['user_str_length']))

        if timecreated is None:
            self._write_integer(int(time.time()))  # current date and time
        else:
            self._write_integer(timecreated) # date and time of copied variable
        self._write_integer(vcode)

        pos = self._file.tell()
        # data beginning - current + 28 bytes. 8 int properties (7x4)
        self._write_integer(pos + 32)
        # next var - current + 28 bytes + our contents (rows x columns x 8)
        self._write_integer(pos + 32 + len(vcont.flat) * 8)

        self._write_integer(VariableType.MATRIX)

        self._write_integer(self._frequency)

        self._write_integer(self._start_year)

        self._write_integer(self._start_period)

        nrows, ncols = vcont.shape
        self._write_integer(nrows)
        self._write_integer(ncols)

        # Write the individual matrix elements
        for element in vcont.flat:
            self._write_double(element)

        # change global header
        self.codes += [vcode]

        self._header['time_modified'] = time.ctime(time.time())
        self._header['number_of_variables'] += 1
        self._header['number_of_live_variables'] += 1

        self._file.seek(self._header_byte_positions['time_modified'])
        self._write_integer(int(time.time()))
        self._file.seek(self._header_byte_positions['number_of_variables'])
        self._write_integer(self._header['number_of_variables'])
        self._file.seek(self._header_byte_positions['number_of_live_variables'])
        self._write_integer(self._header['number_of_live_variables'])

        # rebuild index
        # TODO: make it so that you don't have to call build_index after adding
        #       a variable... just add it to the relevant places
        if build_index:
            self._build_index()

    def write_list(self, vname: str, vcode: int, vcont: List, comment: str = ''
                   , caller: Optional[str] = None, *, build_index: bool = True,
                   timecreated: Optional[int] = None, user: Optional[str] = None):
        """Writes a list object to the databank object

        List must of contain items of single data type. The types currently supported are
        strings, integers and doubles. Lists of lists are not supported.

        Parameters
        ----------
        vname : str
            label for variable being written to the databank
        vcode : int
            unique code identifier for variable entry in databank index
        vcont : list (1 dimensional)
            list object where all items are the same data types. Supported types
            strings, integers and floats
        comment : str
            Description of the variable to be added
        caller : str, default `None`
            The name/path of the calling script/job e.g. `__file__`.
            If `None` (the default), tries to infer the caller from the call
            stack. Depending on the calling environment, this may be inaccurate.
        build_index : bool, default True
            If True, index is rebuilt after variable is added to databank
        timecreated: int, default None
            timestamp original variable on databank was created (used by condense)
            if none, uses the current time
        user: str, default None
            user that wrote original variable on databank (used by condense)
            if none, uses current set user
        Returns : None

        Raises
        ------
            TypeError:
                (a) If all items in the list are not the same data type
                (b) If the unique data type is not string, integer or float
            ValueError:
                (a) Databank mode is not set to write
                (b) Databank user has not been set
                (c) Function arguements are not defined as describe in parameters
        """
        # Check write arguments
        self._check_write_arguments(vname, vcode, comment, caller)

        if not isinstance(vcont, list):
            raise TypeError('Data (`vcont` argument) must be a list')

        # Identify variable types in the list (should only be one for now)
        variable_types = list(set([type(x) for x in vcont]))  # Unique list of variable types

        if len(variable_types) != 1:
            raise TypeError(
                'List is not a single data type (strings, floats or integers '
                + 'function does not support mixed data types of nested list')

        # Check that the common variable type is permitted (assumes a common
        # variable type, as above)
        if variable_types[0] not in (str, int, float):
            raise TypeError(
                'List is not a single data type (strings, floats or integers '
                + 'function does not support mixed data types of nested list')

        self._file.seek(0, 2)  # end of file

        # write header
        self._write_integer(len(vname))
        self._write_integer(len(comment))

        # If no `caller` passed, try to infer from the call stack
        # TODO: See if there's a more flexible/accurate method that works
        #       across multiple calling environments
        if caller is None:
            caller = os.path.abspath(inspect.stack()[-1][1])

        if self._header['version'] > 3.0:
            self._write_integer(len(caller))

        self._write_string(vname)
        self._write_string(comment)

        if self._header['version'] > 3.0:
            self._write_string(caller)

        if user is None:
            # pad username with spaces to fill necessary quota
            self._write_string(self.user.ljust(self._header['user_str_length']))
        else:
            #Keep username of copied variable
            self._write_string(user.ljust(self._header['user_str_length']))

        if timecreated is None:
            self._write_integer(int(time.time()))  # current date and time
        else:
            self._write_integer(timecreated) # date and time of copied variable

        self._write_integer(vcode)

        pos = self._file.tell()
        # data beginning - current + 28 bytes. 8 int properties (7x4)
        self._write_integer(pos + 32)
        # next var - current + 28 bytes + our contents (rows x columns x 8)
        self._write_integer(pos + 32 + self._get_list_bytesize(vcont))

        self._write_integer(VariableType.ARRAY)
        self._write_integer(0)  # freq - invalid for lists

        self._write_integer(0) # start year invalid for lists

        self._write_integer(0)  # period - invalid for lists

        rows, columns = (len(vcont), 0)  # rows and columns of our matrix
        self._write_integer(rows)
        self._write_integer(columns)

        # Write data
        for element in vcont:
            if isinstance(element, str):
                self._write_integer(VariableType.STRING.value)
                self._write_integer(len(element))  # Length of string
                self._write_integer(0)  # No relevant second (columns) dimension
                self._write_string(element)
            elif isinstance(element, int):
                self._write_integer(VariableType.INTEGER.value)
                self._write_integer(0)  # No relevant first (rows) dimension
                self._write_integer(0)  # No relevant second (columns) dimension
                self._write_integer(element)
            elif isinstance(element, float):
                self._write_integer(VariableType.DOUBLE.value)
                self._write_integer(0)  # No relevant first (rows) dimension
                self._write_integer(0)  # No relevant second (columns) dimension
                self._write_double(element)

        # Append code to header
        self.codes += [vcode]

        self._header['time_modified'] = time.ctime(time.time())
        self._header['number_of_variables'] += 1
        self._header['number_of_live_variables'] += 1

        self._file.seek(self._header_byte_positions['time_modified'])
        self._write_integer(int(time.time()))
        self._file.seek(self._header_byte_positions['number_of_variables'])
        self._write_integer(self._header['number_of_variables'])
        self._file.seek(self._header_byte_positions['number_of_live_variables'])
        self._write_integer(self._header['number_of_live_variables'])

        # rebuild index
        # TODO: make it so that you don't have to call build_index after adding
        #       a variable... just add it to the relevant places
        if build_index:
            self._build_index()

    def _get_list_bytesize(self, list_to_size):
        """Returns the byte size required for the list object passed"""
        if not isinstance(list_to_size, list):
            raise TypeError("item to size is not a list")
        bytesize = 0
        for item in list_to_size:
            # 3*4 for header int bytes
            bytesize += 12
            # 1 byte per string character
            bytesize += self._get_item_bytesize(item)
        return bytesize

    def _get_item_bytesize(self, item_to_size):
        """Returns the byte size of the passed item. Items supports are
        string, integer or float"""
        bytesize = 0
        # 1 byte per string character
        if isinstance(item_to_size, str):
            bytesize = len(item_to_size)
        # 4 bytes per integer
        elif isinstance(item_to_size, int):
            bytesize = 4
        # 8 bytes per float
        elif isinstance(item_to_size, float):
            bytesize = 8
        else:
            raise TypeError("Item is not a string, integer or float")
        return bytesize

    def build_index(self):
        """If you added variables in bulk and skipped building the index, you can
        do so separately, after all has been written."""
        self._build_index()

    def scratch(self, code, *, build_index: bool = True):
        if not isinstance(code, int):
            raise ValueError(
                'Can only scratch by code, not by name, as that is not necessarily unique')  # TODO?
        if code not in self.codes:
            raise ValueError('This variable does not exist!')
        if self._mode != 'write':
            raise ValueError('You need to have set the mode to \'write\'')

        bpr = self._index_by_code[code]  # variable properties
        # get to the start of data minus 36 bytes (where the code value is
        # stored)
        self._file.seek(bpr['address'] - 36)
        self._write_integer(-code)  # rewrite the code as negative

        self._header['number_of_live_variables'] -= 1

        if build_index:
            # TODO: OPTIM: expensive - maybe just manually remove it from given
            # places?
            self._build_index()

    def scratch_if_exists(self, code, *, build_index: bool = True):
        if code in self.codes:
            self.scratch(code, build_index=build_index)

    def condense(self):
        """
        Condenses the databank by copying all unscratched variables to a new
        temporary databank whilst preserving all necessary metadata from the
        orginal. The orginal uncondensed databank is then overwritten with the
        temporary condensed databank

        TO DO:
            copy across additional variable metadata on frequency
        """
        # No action if all variables are live
        if self._header["number_of_live_variables"] == self._header['number_of_variables']:
            return

        if self._mode != 'write':
            raise ValueError(
                "Cannot condense a databank unless you have mode set to 'write'")

        if self.user is None:
            raise ValueError('Need to set the user to condense')

        for code in self._index_by_code:
            variable_type = self._index_by_code[code]['type_']

            if variable_type not in (VariableType.MATRIX, VariableType.ARRAY):
                raise ValueError(
                    'Cannot condense databanks with data that is neither a matrix or a list of single type (string, float, integer)')

        temp_db1 = DB1()

        temp_db1.user = self.user

        temp_filepath = self._filepath.split(".")[0] + "__." + self._filepath.split(".")[1]
        temp_db1.create(temp_filepath)

        for code in self.codes:
            data = self.get(code)
            var_metadata = self._index_by_code[code]

            # Start year
            temp_db1.start_year = (var_metadata['start_year'])
            temp_db1.frequency = (var_metadata['frequency'])
            temp_db1.start_period = (var_metadata['start_period'])

            if self._header['version'] > 3.0:
                caller_var = var_metadata['caller']
            else:
                caller_var=None
            #Convert string date time to seconds
            date = int(time.mktime(time.strptime(var_metadata['date'])))

            if isinstance(data, np.ndarray):
                temp_db1.write_matrix(var_metadata['name'], code, data,
                                       var_metadata['comment'],caller=caller_var,
                               user=var_metadata['user'], timecreated=date)
            elif isinstance(data, list):
                temp_db1.write_list(var_metadata['name'], code, data, var_metadata['comment'],
                             caller=caller_var,user=var_metadata['user'], timecreated=date)
            else:
                raise TypeError('Unsupported variable type for writing: {}'.format(type(data)))

        # Update header information
        temp_db1._header['time_condensed'] = time.ctime(time.time())
        # Convert existing date modified information
        time_modified = int(time.mktime(time.strptime(self._header['time_modified'])))
        time_created = int(time.mktime(time.strptime(self._header['time_created'])))
        temp_db1._header['time_modified'] = time.ctime(time_modified)
        temp_db1._header['time_created'] = time.ctime(time_created)
        temp_db1._header['user_created'] = self._header['user_created']
        temp_db1._header['user_modified'] = self._header['user_modified']

        # Write header information
        temp_db1._file.seek(temp_db1._header_byte_positions['time_condensed'])
        temp_db1._write_integer(int(time.time()))
        temp_db1._file.seek(temp_db1._header_byte_positions['time_modified'])
        temp_db1._write_integer(time_modified)
        temp_db1._file.seek(temp_db1._header_byte_positions['time_created'])
        temp_db1._write_integer(time_created)
        temp_db1._file.seek(temp_db1._header_byte_positions['user_created'])
        temp_db1._write_string(self._header['user_created'])
        temp_db1._file.seek(temp_db1._header_byte_positions['user_modified'])
        temp_db1._write_string(self._header['user_modified'])

        # Close both databanks
        self.close()  # Current databank
        temp_db1.close()  # Condensed databank (in temporary file)

        # Overwrite the current databank with the condensed version
        shutil.move(temp_filepath, self._filepath)

        # Open the newly condensed databank
        self.open(self._filepath)

    def __contains__(self, item):
        return item in self.codes or item in self.varnames

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        raise NotImplementedError('Databank modification not implemented yet')

    def __len__(self):
        return len(self.codes)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
