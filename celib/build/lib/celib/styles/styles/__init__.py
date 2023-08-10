# -*- coding: utf-8 -*-
"""
styles
======
CE chart styles. Exports `available`, a tuple of style names.

Usage (to inspect the list of available styles):

    import celib
    print(celib.styles.styles.available)

`_style_files` is a dict that stores the filepaths of the style-definition
files with the style names (matching the filenames) as keys.

Other than this __init__ file, this sub-package stores CE styles in matplotlib
'.mplstyle'-format files. See ce-default.mplstyle for an example.

"""

import glob
import os


__all__ = ['available']


_style_files = {}

for filepath in glob.glob(os.path.join(os.path.dirname(__file__), '*.mplstyle')):
    name = os.path.splitext(os.path.split(filepath)[1])[0]
    _style_files[name] = filepath

available = tuple(_style_files.keys())
