"""
Load variable dimension names and related metadata.

Functions included:
    - load_dims
        Load model dimensions
"""

import pandas as pd

from ftt_source.paths import get_utilities_path


def load_dims():
    """Load variable dimensions and related metadata."""
    dims_path = get_utilities_path() / "titles" / "VariableListing.csv"
    if not dims_path.is_file():
        raise FileNotFoundError(f"Dimensions name file not found: {dims_path}")

    dims_data = pd.read_csv(dims_path, keep_default_na=False)
    dims = {}
    histend = {}
    forstart = {}
    domain = {}
    unit = {}

    # Each row defines one model variable.
    for _, row in dims_data.iterrows():
        variable = row["Variable name"]
        dims[variable] = [
            row[dimension]
            for dimension in ("Dim1", "Dim2", "Dim3", "Dim4")
            if row[dimension] not in ("NA", "")
        ]
        domain[variable] = row["Domain"]
        histend[variable] = (
            int(row["History end"])
            if row["History end"] not in ("-", "")
            else 0
        )
        forstart[variable] = row["Forecast start"]
        unit[variable] = row["Unit"]

    return dims, histend, domain, forstart, unit
