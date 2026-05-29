# -*- coding: utf-8 -*-
"""
future_technology_transformation
================================
Public API for the Future Technology transformation model.

Importing this package gives access to the main :class:`RunFTT` class and the
path-configuration helpers without needing to know the internal ``SourceCode``
package layout.

Quick start
-----------
Run the model with the built-in data::

    from future_technology_transformation import RunFTT
    model = RunFTT()
    model.run()

Run with custom Inputs / Utilities folders::

    from future_technology_transformation import RunFTT
    model = RunFTT(
        inputs_path="/path/to/my/Inputs",
        utilities_path="/path/to/my/Utilities",
    )
    model.run()

Override individual settings programmatically::

    from future_technology_transformation import RunFTT
    model = RunFTT(
        inputs_path="/path/to/my/Inputs",
        settings={
            "scenarios": "S0, S1",
            "simulation_start": "2020",
            "simulation_end": "2040",
        },
    )
    model.run()

Outputs are available on ``model.output`` (dict of scenario → variable → array).
"""

from SourceCode.model_class import RunFTT
from SourceCode.paths import (
    set_paths,
    get_inputs_path,
    get_utilities_path,
    reset_paths,
)

__all__ = [
    "RunFTT",
    "set_paths",
    "get_inputs_path",
    "get_utilities_path",
    "reset_paths",
    "run",
    "run_cli",
]


def run(
    inputs_path=None,
    utilities_path=None,
    settings_path=None,
    settings=None,
) -> "RunFTT":
    """Create a :class:`RunFTT`, solve it, and return the instance.

    Parameters
    ----------
    inputs_path:
        Optional override for the *Inputs* data directory.
    utilities_path:
        Optional override for the *Utilities* data directory.
    settings_path:
        Optional path to a ``settings.ini`` file.  Defaults to the one
        bundled with the package.
    settings:
        Optional dict of settings to override values in the ini file, e.g.
        ``{"scenarios": "S0, S1", "simulation_end": "2040"}``.

    Returns
    -------
    RunFTT
        The solved model instance (``model.output`` contains results).
    """
    model = RunFTT(
        inputs_path=inputs_path,
        utilities_path=utilities_path,
        settings_path=settings_path,
        settings=settings,
    )
    model.run()
    return model


def run_cli() -> None:
    """Entry point for the ``ftt-run`` command-line script."""
    import argparse
    import pickle
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Run the FTT model from the command line."
    )
    parser.add_argument(
        "--inputs", metavar="DIR", default=None,
        help="Path to a custom Inputs directory.",
    )
    parser.add_argument(
        "--utilities", metavar="DIR", default=None,
        help="Path to a custom Utilities directory.",
    )
    parser.add_argument(
        "--settings", metavar="FILE", default=None,
        help="Path to a custom settings.ini file.",
    )
    parser.add_argument(
        "--output", metavar="FILE", default="Output/Results.pickle",
        help="Path for the output pickle file (default: Output/Results.pickle).",
    )
    args = parser.parse_args()

    model = run(
        inputs_path=args.inputs,
        utilities_path=args.utilities,
        settings_path=args.settings,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model.output, f)
    print(f"Results saved to {output_path}")
