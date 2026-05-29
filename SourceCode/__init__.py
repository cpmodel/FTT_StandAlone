# -*- coding: utf-8 -*-
"""
Future Technology Transformation – SourceCode package
=====================================
Internal package containing all FTT model source code.

For external use, prefer importing from the ``future_technology_transformation``
top-level package which provides the stable public API::

    from future_technology_transformation import RunFTT
"""

from SourceCode.model_class import RunFTT
from SourceCode.paths import set_paths, get_inputs_path, get_utilities_path, reset_paths

__all__ = [
    "RunFTT",
    "set_paths",
    "get_inputs_path",
    "get_utilities_path",
    "reset_paths",
]

