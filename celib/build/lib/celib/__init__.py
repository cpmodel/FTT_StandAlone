# -*- coding: utf-8 -*-
"""
celib
=====
Cambridge Econometrics' core Python library.

(C) 2015-20 Cambridge Econometrics

You may not supply the contents of this package in uncompiled form to
colleagues outside of CE.

"""

import warnings


MAJOR = 0
MINOR = 4
PATCH = 1
DEV = False

VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH,
                              '.dev' if DEV else '')
__version__ = VERSION


if DEV:
    warnings.warn('''
This is a development version of the CE Python library ({})
It is not ready for deployment yet'''.format(VERSION))


from celib.api import *
