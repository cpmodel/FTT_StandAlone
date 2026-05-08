# -*- coding: utf-8 -*-
"""
=========================================
paths.py
=========================================
Centralized path configuration for FTT Stand-alone.

This module provides a single place to configure the data directories used by
the model (Inputs and Utilities). Override the defaults before instantiating
:class:`~SourceCode.model_class.RunFTT` when using FTT Stand-alone as an
imported package with local data.

Example usage
-------------
Default behaviour (uses the data bundled with the repository)::

    from future_technology_transformation import RunFTT
    model = RunFTT()

Using custom data directories::

    from future_technology_transformation import RunFTT, set_paths
    set_paths(
        inputs_path="/path/to/my/Inputs",
        utilities_path="/path/to/my/Utilities",
    )
    model = RunFTT()

Or equivalently via :class:`RunFTT` keyword arguments::

    from future_technology_transformation import RunFTT
    model = RunFTT(
        inputs_path="/path/to/my/Inputs",
        utilities_path="/path/to/my/Utilities",
    )
"""

from pathlib import Path

# Root of the installed / development package.
# This is the directory that *contains* SourceCode/ (i.e. the repo root when
# running from source, or the package root when pip-installed).
_PACKAGE_ROOT: Path = Path(__file__).parents[1]

# User-configurable overrides.  None means "use the defaults".
_inputs_path: Path | None = None
_utilities_path: Path | None = None


def set_paths(
    inputs_path: "str | Path | None" = None,
    utilities_path: "str | Path | None" = None,
) -> None:
    """Set the data directories used by the model.

    Parameters
    ----------
    inputs_path:
        Path to the *Inputs* directory.  When ``None`` the value is left
        unchanged (defaults are kept).
    utilities_path:
        Path to the *Utilities* directory.  When ``None`` the value is left
        unchanged.
    """
    global _inputs_path, _utilities_path
    if inputs_path is not None:
        _inputs_path = Path(inputs_path)
    if utilities_path is not None:
        _utilities_path = Path(utilities_path)


def get_inputs_path() -> Path:
    """Return the active *Inputs* directory as a :class:`~pathlib.Path`.

    Resolution order:

    1. A path set explicitly via :func:`set_paths` or the ``inputs_path``
       argument of :class:`~SourceCode.model_class.RunFTT`.
    2. An ``Inputs/`` directory located next to the package root (works both
       for editable installs and for the source tree).
    3. ``Inputs/`` relative to the current working directory (legacy
       behaviour).
    """
    if _inputs_path is not None:
        return _inputs_path
    pkg_inputs = _PACKAGE_ROOT / "Inputs"
    if pkg_inputs.exists():
        return pkg_inputs
    return Path.cwd() / "Inputs"


def get_utilities_path() -> Path:
    """Return the active *Utilities* directory as a :class:`~pathlib.Path`.

    Resolution order mirrors :func:`get_inputs_path`.
    """
    if _utilities_path is not None:
        return _utilities_path
    pkg_utilities = _PACKAGE_ROOT / "Utilities"
    if pkg_utilities.exists():
        return pkg_utilities
    return Path.cwd() / "Utilities"


def reset_paths() -> None:
    """Reset both path overrides to their defaults."""
    global _inputs_path, _utilities_path
    _inputs_path = None
    _utilities_path = None
