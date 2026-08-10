# -*- coding: utf-8 -*-
"""
Integration checks for FTT Stand-alone.

Runs FTT-Tr, FTT-Fr, FTT-H, and FTT-P from 2010 to 2030 with noit=4
(instead of the standard 20).  The model is solved once per session and
the result is shared across all three checks:

  * No NaN values in any output array
  * No infinite values (±inf) in any output array
  * Wall-clock time does not exceed the threshold in tests/runtime_baseline.json
"""

import configparser
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add the workspace root to sys.path so ftt_source is importable when pytest
# is invoked from a different directory (e.g. the repository root in CI).
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = Path(__file__).parent / "runtime_baseline.json"

sys.path.insert(0, str(WORKSPACE_ROOT))

from ftt_source.model_class import RunFTT 


@pytest.fixture(scope="session")
def model_run_results():
    """Run the model once for the whole test session.

    Settings overridden for CI speed:
    * ``simulation_end`` → 2030
    * ``enable_modules`` → FTT-Tr, FTT-P, FTT-H, FTT-Fr
    * ``scenarios``      → S0
    * ``noit``           → 4  (instead of the standard 20)

    Returns
    -------
    tuple of (output dict, elapsed seconds)
    """
    # The model resolves settings.ini and Inputs/ relative to cwd.
    original_cwd = Path.cwd()
    os.chdir(WORKSPACE_ROOT)

    try:
        original_read = configparser.ConfigParser.read

        def patched_read(self, filenames, encoding=None):
            result = original_read(self, filenames, encoding=encoding)
            if "settings" in self:
                self.set("settings", "simulation_end", "2030")
                self.set("settings", "enable_modules", "FTT-Tr, FTT-P, FTT-H, FTT-Fr")
                self.set("settings", "scenarios", "S0")
            return result

        with patch.object(configparser.ConfigParser, "read", patched_read):
            model = RunFTT()

        for scen in model.input:
            model.input[scen]["noit"][:] = 4

        t0 = time.perf_counter()
        model.run()
        elapsed = time.perf_counter() - t0

        yield model.output, elapsed
    finally:
        os.chdir(original_cwd)


def test_no_nans_in_output(model_run_results):
    """Fail if any output array contains a NaN."""
    output, _ = model_run_results

    nan_report = []
    for scen, variables in output.items():
        for var, arr in variables.items():
            if np.any(np.isnan(arr)):
                n_nans = int(np.sum(np.isnan(arr)))
                nan_report.append(f"  {scen}/{var}: {n_nans} NaN value(s)")

    if nan_report:
        pytest.fail("NaN values found in model output:\n" + "\n".join(nan_report))


def test_no_infs_in_output(model_run_results):
    """Fail if any output array contains a +inf or -inf value."""
    output, _ = model_run_results

    inf_report = []
    for scen, variables in output.items():
        for var, arr in variables.items():
            if np.any(np.isinf(arr)):
                n_infs = int(np.sum(np.isinf(arr)))
                inf_report.append(f"  {scen}/{var}: {n_infs} infinite value(s)")

    if inf_report:
        pytest.fail("Infinite values found in model output:\n" + "\n".join(inf_report))


def test_runtime_regression(model_run_results):
    """Fail if the model run exceeds the threshold in tests/runtime_baseline.json.

    To update the baseline after a legitimate performance change, measure the
    runtime locally (pytest tests/test_integration.py -v -s) and set
    ``max_seconds`` in runtime_baseline.json to roughly 2x that value.
    """
    _, elapsed = model_run_results

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    max_seconds = baseline["max_seconds"]

    assert elapsed < max_seconds, (
        f"Model run took {elapsed:.1f}s, exceeding the {max_seconds}s threshold "
        f"in tests/runtime_baseline.json. If this slowdown is intentional, "
        f"update max_seconds to roughly 2x your measured local runtime."
    )
