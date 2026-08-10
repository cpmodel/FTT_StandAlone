# -*- coding: utf-8 -*-
"""
=========================================
paths_append.py
=========================================
Append __file__ path to sys.path to enable import.

.. code-block:: python

   sys.path.append(os.path.dirname(__file__))
   

"""

import sys
import os

sys.path.append(os.path.dirname(__file__))
