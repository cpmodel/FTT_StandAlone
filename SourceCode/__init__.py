# -*- coding: utf-8 -*-
"""
FTT Stand-alone – SourceCode package
=====================================
Internal package containing all FTT model source code.

For external use, prefer importing from the ``ftt_standalone`` top-level
package which provides the stable public API::

    from ftt_standalone import RunFTT
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

