# -*- coding: utf-8 -*-
"""
celib
=====
Setup script for CE Python library.

(C) 2015-20 Cambridge Econometrics

You may not supply this script or the contents of the `celib` package in
uncompiled form to colleagues outside of CE.

"""

import os
import warnings

from setuptools import setup


# Load version information from the library itself
exec(open(os.path.join('celib', '__init__.py')).read())

setup(name='celib',
      version=VERSION,
      description='CE Python library',
      author='Cambridge Econometrics',
      packages=['celib',
                'celib.econometrics',
                'celib.io',
                'celib.io.tests',
                'celib.styles',
                'celib.styles.styles',
                'celib.styles.tests',
                'celib.tests', ],
      package_data={
          'celib.io.tests': [os.path.join('data', pattern)
                             for pattern in ['*.zip', '*.dat', '*.csv',
                                             '*.DB1', '*.mre',
                                             '*.pickle', '*.txt']],
          'celib.styles': [os.path.join('styles', '*.mplstyle'), ], }, )
