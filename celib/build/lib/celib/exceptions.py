# -*- coding: utf-8 -*-
"""
exceptions
==========
Errors and exceptions specific to `celib`.
"""

class CELibError(Exception):
    """Base `celib` exception."""
    pass


class DimensionError(CELibError):
    """Unexpected/invalid NumPy array dimensions."""
    pass

class NonConvergenceError(CELibError):
    """Numerical procedure failed to converge."""
    pass

class TablsError(CELibError):
    """Undefined tabls classification (information accessible by the `entry` attribute)."""

    def __init__(self, message, entry):
        self.message = message
        self.entry = entry
