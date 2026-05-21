# -*- coding: utf-8 -*-
"""
FTT Stand-alone – ftt_source package
=====================================
Internal package containing all FTT model source code.

For external use, prefer importing from the ``future_technology_transformation``
top-level package which provides the stable public API::

    from future_technology_transformation import RunFTT
"""

from ftt_source.model_class import RunFTT
from ftt_source.paths import set_paths, get_inputs_path, get_utilities_path, reset_paths

__all__ = [
    "RunFTT",
    "set_paths",
    "get_inputs_path",
    "get_utilities_path",
    "reset_paths",
]

