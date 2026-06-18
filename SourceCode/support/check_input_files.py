"""
=========================================
check_input_files.py
=========================================
Functions to check that input files are formatted correctly. Appropriate
errors are raised if any issues are found.

Functions included in the file:
    - check_var_dims
        Checks that the dimensions of the input file are expected.
    - check_vars_exist
        Checks that the required variables exist in the input folder.
"""

# Global imports
import polars as pl

# Local imports
from SourceCode.paths import get_inputs_path, get_utilities_path


def _dim_len(titles, dim):
    """Return the expected length of a classification dimension."""
    val = titles.get(dim)
    if val is None:
        return None
    return len(val) if isinstance(val, (list, tuple)) else int(val)


def check_var_dims(var_name, df, titles, dimensions):
    """
    Check that a CSV DataFrame's shape matches the expected dimensions for
    ``var_name``.  Must be called with the raw polars DataFrame (before the
    data are loaded into the numpy array) so that extra rows or columns added
    to the file are detectable.

    Parameters
    ----------
    var_name : str
        Name of the variable.
    df : polars.DataFrame
        DataFrame read directly from the variable's CSV file.
    titles : dict
        Pre-loaded titles dict mapping classification name → sequence of
        labels (or integer count).
    dimensions : dict
        Pre-loaded dimensions dict mapping variable name → list of four
        dimension name strings, e.g. ``['RTI', 'VTTI', 'NA', 'TIME']``.

    Raises
    ------
    ValueError
        If the number of value columns or the number of rows in the CSV does
        not match the expected classification lengths.
    """
    if var_name not in dimensions:
        return

    var_dims = dimensions[var_name]

    # Determine the wide axis (mirrors _wide_column_axis in data_loading).
    if var_dims[3] == 'TIME':
        wide_axis = 3
    elif var_dims[2] not in ('NA', 'TIME') and var_dims[3] == 'NA':
        wide_axis = 2
    elif (var_dims[1] not in ('NA', 'TIME')
          and var_dims[2] == 'NA'
          and var_dims[3] == 'NA'):
        wide_axis = 1
    else:
        wide_axis = None  # scalar

    # Coordinate axes: non-NA, non-TIME, and not the wide axis.
    coord_axes = [
        (ax, var_dims[ax])
        for ax in range(4)
        if var_dims[ax] not in ('NA', 'TIME') and ax != wide_axis
    ]
    n_coord = len(coord_axes)

    errors = []

    # ------------------------------------------------------------------ #
    # Detect label/value row format (no coord axes, first col == wide     #
    # dim name). e.g. SCA,Mean — rows enumerate the classification.      #
    # In this case check row count, not column count.                     #
    # ------------------------------------------------------------------ #
    label_value_format = (
        n_coord == 0
        and wide_axis is not None
        and var_dims[wide_axis] != 'TIME'
        and len(df.columns) >= 1
        and df.columns[0] == var_dims[wide_axis]
    )

    if label_value_format:
        wide_dim = var_dims[wide_axis]
        expected_len = _dim_len(titles, wide_dim)
        if expected_len is not None and len(df) != expected_len:
            errors.append(
                f"  row count: expected {expected_len} rows for '{wide_dim}', "
                f"got {len(df)}"
            )
        if errors:
            raise ValueError(
                f"Variable '{var_name}' CSV has unexpected dimensions:\n"
                + "\n".join(errors)
            )
        return

    # ------------------------------------------------------------------ #
    # Column count check (skip TIME — timeline length is run-dependent)  #
    # ------------------------------------------------------------------ #
    if wide_axis is not None and var_dims[wide_axis] != 'TIME':
        wide_dim = var_dims[wide_axis]
        expected_cols = _dim_len(titles, wide_dim)
        if expected_cols is not None:
            actual_cols = len(df.columns) - n_coord
            if actual_cols != expected_cols:
                errors.append(
                    f"  Dim {wide_axis} ('{wide_dim}'): "
                    f"expected {expected_cols} value columns, got {actual_cols}"
                )

    # ------------------------------------------------------------------ #
    # Row count check = product of all coordinate dimension lengths       #
    # ------------------------------------------------------------------ #
    if coord_axes:
        expected_rows = 1
        for ax, dim in coord_axes:
            dim_len = _dim_len(titles, dim)
            if dim_len is None:
                expected_rows = None
                break
            expected_rows *= dim_len

        if expected_rows is not None and len(df) != expected_rows:
            msg = f"  row count: expected {expected_rows}, got {len(df)}"

            errors.append(msg)

    if errors:
        raise ValueError(
            f"Variable '{var_name}' CSV has unexpected dimensions:\n"
            + "\n".join(errors)
        )


def check_vars_exist(model):
    """
    Check that the required S0 input files exist for a given model.

    Parameters
    ----------
    model : str
        Model/domain name (e.g. 'FTT-Tr', 'FTT-P').  Must match the
        ``Domain`` column in VariableListing.csv and the corresponding
        sub-folder under ``Inputs/S0/``.

    Raises
    ------
    ValueError
        If any required input variable file is missing from
        ``Inputs/S0/<model>/``.
    """
    var_list_path = get_utilities_path() / 'titles' / 'VariableListing.csv'
    if not var_list_path.is_file():
        raise FileNotFoundError(f"VariableListing.csv not found: {var_list_path}")

    var_listing_df = pl.read_csv(
        str(var_list_path), infer_schema_length=10000, null_values=['', 'nan']
    )
    # Filter for this model's required input variables
    required_vars = (
        var_listing_df
        .filter(pl.col('Domain') == model)
        .filter(pl.col('Is input variable') == 'Y')
        .select('Variable name')
        .to_series()
        .to_list()
    )

    if not required_vars:
        return  # Nothing listed as required for this model.

    s0_model_dir = get_inputs_path() / 'S0' / model
    missing_vars = [
        var for var in required_vars
        if not (s0_model_dir / f'{var}.csv').is_file()
    ]

    if missing_vars:
        raise ValueError(
            f"The following required input files are missing from\n "
            f"'{s0_model_dir}':\n  " + ", ".join(missing_vars)
        )