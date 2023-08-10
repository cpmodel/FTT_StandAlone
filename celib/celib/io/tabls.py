# -*- coding: utf-8 -*-
"""Tools for reading metadata from IDIOM tabls.dat files."""

from collections import OrderedDict
from celib.exceptions import TablsError


class Tabls(object):
    """Reader for IDIOM tabls.dat files, separating variables and titles.

    Example usage (using E3ME as an example):

    >>> from celib import Tabls
    >>> tb = Tabls('Tabls.dat')

    >>> tb.get_titles_by_code(1)
    {'code': 1, 'desc': 'titles of standard regions for E3ME4', 'name': 'RTI'}

    >>> tb.get_titles_by_name('RTI')  # Identical output to code 1 above
    {'code': 1, 'desc': 'titles of standard regions for E3ME4', 'name': 'RTI'}

    >>> tb.get_var('QR')
    {'desc': 'output of products at basic prices',
     'dimcodes': (2, 1),
     'dimnames': ('YTI', 'RTI'),  # Dimension name & code automatically matched
     'name': 'QR',
     'units': 1}
    """

    def __init__(self, input_file):
        """Read `input_file` and separate variables and titles.

        Creates the following (ordered) dictionaries:
         - variables_dict : model variables with names as keys
         - titles_dict_by_code : model titles with codes as keys
         - titles_dict_by_name : model titles with names as keys

        and lists:
         - variables : keys of `variables_dict`
         - titles : keys of `titles_dict_by_name`
        """
        self.variables_dict = OrderedDict()
        self.titles_dict_by_code = OrderedDict()
        self.titles_dict_by_name = OrderedDict()

        # Read the tabls file, separating into a dict of variable entries and a
        # dict of titles entries
        with open(input_file, 'r') as f:
            for line in f:
                # Only lines that begin with alphanumeric characters contain
                # variable entries. Skip all others
                if not line[0].isalnum():
                    continue

                name = line[:4].strip()
                dim1 = int(line[4:7].strip())
                dim2 = int(line[8:10].strip())
                units = int(line[11:12].strip())
                desc = line[18:].strip()

                # Ignore 'titles' variables that are actually placeholders
                # (listed as 'dummy scalar' with dim1=999)
                if dim1 == 999:
                    continue

                # Initialise dictionary entry with values common to variables
                # and titles
                entry = {'name': name, 'desc': desc}

                # Expand and store depending on whether the entry is a title
                # (units=7) or a variable (everything else)
                if units == 7:
                    entry['code'] = dim1
                    self.titles_dict_by_name[name] = entry
                else:
                    entry['units'] = units

                    # If dim2=0, dimcodes (and the variable) is one-dimensional
                    if dim2 == 0:
                        dimcodes = (dim1, )
                    else:
                        dimcodes = (dim1, dim2)
                    entry['dimcodes'] = dimcodes

                    self.variables_dict[name] = entry

        # Create a version of the titles dictionary keyed by code.
        # In the event of duplicates, only store the first set of titles for
        # each code.
        for k, v in self.titles_dict_by_name.items():
            code = v['code']
            if code not in self.titles_dict_by_code:
                self.titles_dict_by_code[code] = v

        # Add dimension names to accompany dimension codes in variables
        # dictionary
        for k in self.variables_dict.keys():
            dimnames = []

            dimcodes = self.variables_dict[k]['dimcodes']
            for code in dimcodes:
                try:
                    name = self.titles_dict_by_code[code]['name']
                except KeyError:
                    name = None
                dimnames.append(name)

            self.variables_dict[k]['dimnames'] = tuple(dimnames)

        # Store title and variable names to lists
        self.titles = list(self.titles_dict_by_name.keys())
        self.variables = list(self.variables_dict.keys())

    def get_titles_by_code(self, code):
        """Return the titles corresponding to `code` (an integer).

        Example usage:
            tb = Tabls('Tabls.dat')
            tb.get_titles_by_code(1)
        Example output:
            {'desc': 'titles of standard regions for E3ME4',
            'name': 'RTI', 'code': 1}

        """
        return self.titles_dict_by_code[code]

    def get_titles_by_name(self, name):
        """Return the titles corresponding to `name` (a string).

        Example usage:
            tb = Tabls('Tabls.dat')
            tb.get_titles_by_name('RTI')
        Example output:
            {'desc': 'titles of standard regions for E3ME4',
            'name': 'RTI', 'code': 1}

        """
        return self.titles_dict_by_name[name]

    def get_var(self, name):
        """Return the metadata for variable `name` (a string).

        Example usage:
            tb = Tabls('Tabls.dat')
            tb.get_var('QR')
        Example output:
            {'name': 'QR', 'desc': 'output of products at basic prices',
            'units': 1, 'dimcodes': (2, 1), 'dimnames': ('YTI', 'RTI')}

        Note:
            Some variables have dimension codes with no corresponding
            classification defined.

            For example, in E3ME, variable 'SR' has dimensions (7, 1) but there
            is no classification with code 7. To avoid this error
            (accidentally) propagating through a script, an attempt to access
            such variables will throw a `TablsError`.

            Should the user genuinely want to access this variable, they should
            do so explicitly, by catching the exception and accessing its
            `entry` attribute. For example:

                from celib import Tabls
                from celib.exceptions import TablsError

                tb = Tabls('Tabls.dat')

                try:
                    sr = tb.get_var('SR')
                except TablsError as e:
                    sr = e.entry

        """
        entry = self.variables_dict[name]

        if None in entry['dimnames']:
            raise TablsError(
                'Variable {} has undefined titles: {} -> {}'.format(
                    name, entry['dimcodes'], entry['dimnames']), entry)

        return entry

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
