# -*- coding: utf-8 -*-
"""
=========================================
input_functions.py
=========================================
Loader for per-variable CSV inputs stored in Inputs/.

Functions included:
    - load_data
        Load all model data; uses polars-based reader for selected
        modules and falls back to the legacy loader for all others.

CSV schema
-------------------------------------------------------
TIME variables (dim[3] == 'TIME'):
    Coordinate columns (e.g. RTI, VTTI) followed by year columns (1990..2100).

Non-TIME wide variables — Dim3 as columns (dim[2] != 'NA', dim[3] == 'NA'):
    Coordinate columns followed by Dim3 label columns (e.g. C3TI labels).

Non-TIME wide variables — Dim2 as columns (dim[1] != 'NA', dim[2] == 'NA', dim[3] == 'NA'):
    Coordinate column (e.g. RTI) followed by Dim2 label columns (e.g. VYTI labels).

Scalar / 1-D variables (all trailing dims == 'NA'):
    Single coordinate column (e.g. RTI) followed by a 'value' column.
"""

# Standard library imports
import os

# Third party imports
import numpy as np
import polars as pl


def load_data(titles, dimensions, timeline, scenarios, ftt_modules, forstart,
              progress_callback=None, log_callback=None):
    """
    Load all model data for all variables and all years.

    Behaviour
    ---------
    Calls the legacy loader first for non-transport modules so that those
    variables are populated.
    When 'FTT-Tr' appears in *ftt_modules*, transport variables are read
    from the CSVs in
    ``Inputs_new/<scenario>/FTT-Tr/`` using polars.

    Parameters
    ----------
    titles : dict of lists/tuples
        Classification titles produced by ``load_titles()``.
    dimensions : dict of lists
        Per-variable dimension names, e.g. ``{'TEWS': ['RTI','VTTI','NA','TIME'], ...}``.
    timeline : list of int
        All years covered by the run.
    scenarios : str
        Comma-separated scenario names.
    ftt_modules : str
        Comma-separated module names (e.g. 'FTT-Tr,FTT-P').
    forstart : dict
        Forecast start year per variable.
    progress_callback : callable, optional
        ``progress_callback(current, total)`` for GUI progress bars.
    log_callback : callable, optional
        ``log_callback(message)`` for GUI log output.

    Returns
    -------
    data : dict
        ``data[scenario][variable]`` → ``np.ndarray`` with shape
        ``(len(Dim1), len(Dim2), len(Dim3), len(Dim4))``.
    """
    titles['TIME'] = timeline

    models_enabled = [m.strip() for m in ftt_modules.split(',') if m.strip()]
    supported_models = {'FTT-Tr', 'FTT-P', 'FTT-H', 'FTT-Fr'}
    models_to_load = [m for m in models_enabled if m in supported_models]
    models_to_load.append('General')  # Always load General
    

    scenario_list = [s.strip() for s in scenarios.split(',')]
    scenario_list = ['S0'] + [s for s in scenario_list if s != 'S0']
    data = {
        scen: {
            var: np.zeros([
                len(titles[dimensions[var][0]]),
                len(titles[dimensions[var][1]]),
                len(titles[dimensions[var][2]]),
                len(titles[dimensions[var][3]]),
            ]) for var in dimensions
        } for scen in scenario_list
    }

    # Overlay variables (FTT-Tr, General, etc.).
    scenario_list = [s.strip() for s in scenarios.split(',')]
    scenario_list = ['S0'] + [s for s in scenario_list if s != 'S0']

    # Pre-build a timeline-index lookup once.
    tl_idx = {year: i for i, year in enumerate(timeline)}

    def _read_and_fill_module_folder(scen, model, directory):
        """Read all csv files in a model folder and fill scenario arrays."""
        loaded_vars = set()
        if not os.path.isdir(directory):
            return loaded_vars

        for filename in os.listdir(directory):
            if not filename.endswith('.csv'):
                continue

            var = filename[:-4]  # strip '.csv'
            if var not in dimensions or var not in data[scen]:
                continue

            file_path = os.path.join(directory, filename)
            try:
                df = pl.read_csv(
                    file_path,
                    infer_schema_length=10000,
                    null_values=['', 'NA', 'nan'],
                )
            except Exception as exc:
                import warnings
                warnings.warn(
                    f'Could not read {file_path}: {exc}'
                )
                continue

            _fill_from_input_df(
                df, var, dimensions, titles, timeline, tl_idx, forstart,
                data[scen][var],
            )
            loaded_vars.add(var)

        return loaded_vars

    for module in models_to_load:
        s0_directory = os.path.join('Inputs', 'S0', module)
        module_vars = _read_and_fill_module_folder('S0', module, s0_directory)

        for scen in scenario_list:
            if scen == 'S0':
                continue

            # Scenario fallback: start from S0 values for variables in this module.
            for var in module_vars:
                data[scen][var] = data['S0'][var].copy()

            # Overlay scenario-specific files where present (partial scenario folders).
            scen_directory = os.path.join('Inputs', scen, module)
            _read_and_fill_module_folder(scen, module, scen_directory)

    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wide_column_axis(variable_dims):
    """Return the array axis (1, 2, or 3) used as the wide column axis.

    Returns ``None`` for scalar variables that use a 'value' column.
    Returns ``3`` for TIME variables.
    Returns ``2`` when Dim3 is the wide axis (non-TIME, Dim3 != 'NA').
    Returns ``1`` when Dim2 is the wide axis (non-TIME, Dim3 == 'NA', Dim2 != 'NA').
    """
    if variable_dims[3] == 'TIME':
        return 3
    if variable_dims[2] not in ('NA', 'TIME') and variable_dims[3] == 'NA':
        return 2
    if (variable_dims[1] not in ('NA', 'TIME')
            and variable_dims[2] == 'NA'
            and variable_dims[3] == 'NA'):
        return 1
    return None  # scalar / 'value' column


def _resolve_dim_index(raw_label, dim_name, idx_map, idx_map_casefold):
    """Resolve a CSV label to a dimension index with alias/casefold fallback."""
    label = str(raw_label)

    idx = idx_map.get(label)
    if idx is not None:
        return idx

    return idx_map_casefold.get(label.strip().casefold())


def _fill_from_input_df(df, var, dims, titles, timeline, tl_idx, forstart, target):
    """Fill *target* ndarray in-place from a polars DataFrame.

    Parameters
    ----------
    df : polars.DataFrame
        Data read from the CSV file.
    var : str
        Variable name (used to look up dims and forstart).
    dims : dict
        ``dims[var]`` → list of 4 dimension name strings.
    titles : dict
        Classification titles dict.
    timeline : list of int
        All simulation years.
    tl_idx : dict
        Pre-built ``{year: timeline_index}`` lookup.
    forstart : dict
        Forecast start year per variable.
    target : np.ndarray
        Array to fill in-place, shape ``(D0, D1, D2, D3)``.
    """
    variable_dims = dims[var]  # list of 4 strings
    is_time = variable_dims[3] == 'TIME'
    wide_axis = _wide_column_axis(variable_dims)

    # Coordinate axes: all non-NA, non-TIME axes that are not the wide axis.
    coord_axes_info = [
        (axis, variable_dims[axis])
        for axis in range(4)
        if variable_dims[axis] not in ('NA', 'TIME') and axis != wide_axis
    ]

    columns = df.columns
    n_coord = len(coord_axes_info)
    
    # For scalar variables with no coordinate columns, 'value' is the only column
    if n_coord == 0:
        if wide_axis is not None:
            # 1D non-time variable stored as label/value rows, e.g. SCA,Mean.
            wide_dim = variable_dims[wide_axis]
            if columns and columns[0] == wide_dim and len(columns) >= 2:
                idx_map = {
                    str(label): i for i, label in enumerate(titles[wide_dim])
                }
                labels = df.select(pl.col(columns[0]).cast(pl.Utf8)).to_numpy().reshape(-1)
                values = (
                    df.select(pl.col(columns[1]).cast(pl.Float64))
                    .fill_null(0.0)
                    .to_numpy()
                    .reshape(-1)
                )

                for label, value in zip(labels, values):
                    idx = idx_map.get(label)
                    if idx is None:
                        continue
                    coords = [0, 0, 0, 0]
                    coords[wide_axis] = idx
                    target[tuple(coords)] = value
                return

        # Scalar: just fill target[0,0,0,0] with the value
        try:
            value = float(df.select(pl.col('value')).to_numpy()[0, 0])
            target[0, 0, 0, 0] = value
        except Exception:
            # If value column doesn't exist or can't convert, leave as is
            pass
        return
    
    coord_col_headers = columns[:n_coord]
    value_col_headers = columns[n_coord:]

    if not value_col_headers:
        return  # nothing to fill

    # --- Build index maps: str(label) → array index for each coord dimension ---
    idx_maps = {}
    idx_maps_casefold = {}
    for _, dim_name in coord_axes_info:
        if dim_name not in idx_maps:
            idx_maps[dim_name] = {
                str(label): i for i, label in enumerate(titles[dim_name])
            }
            idx_maps_casefold[dim_name] = {
                str(label).strip().casefold(): i
                for i, label in enumerate(titles[dim_name])
            }

    # --- Build column → array-index map for the value columns ---
    if is_time:
        try:
            var_start = int(forstart[var])
        except (KeyError, ValueError, TypeError):
            var_start = timeline[0]
        expected_years = set(range(var_start, timeline[-1] + 1))
        col_to_idx = {
            col: tl_idx[int(col)]
            for col in value_col_headers
            if col.lstrip('-').isdigit() and int(col) in tl_idx and int(col) in expected_years
        }
    elif wide_axis is not None:
        wide_dim = variable_dims[wide_axis]
        if wide_dim not in idx_maps:
            idx_maps[wide_dim] = {
                str(label): i for i, label in enumerate(titles[wide_dim])
            }
        col_to_idx = {
            col: idx_maps[wide_dim][col]
            for col in value_col_headers
            if col in idx_maps[wide_dim]
        }

        # Fallback for legacy-exported wide tables that use numeric
        # positional headers (e.g. 0..N-1) instead of title labels.
        if len(col_to_idx) < len(value_col_headers):
            parsed_ints = []
            all_numeric = True
            for col in value_col_headers:
                text = str(col).strip()
                if text.lstrip('-').isdigit():
                    parsed_ints.append(int(text))
                else:
                    all_numeric = False
                    break

            if all_numeric and len(set(parsed_ints)) == len(parsed_ints):
                if parsed_ints == list(range(len(parsed_ints))):
                    col_to_idx = {
                        col: idx for col, idx in zip(value_col_headers, parsed_ints)
                        if 0 <= idx < len(titles[wide_dim])
                    }
                elif parsed_ints == list(range(1, len(parsed_ints) + 1)):
                    col_to_idx = {
                        col: idx - 1 for col, idx in zip(value_col_headers, parsed_ints)
                        if 1 <= idx <= len(titles[wide_dim])
                    }
    else:
        # Scalar: single 'value' column — no index mapping needed.
        col_to_idx = {}

    # --- Extract data as numpy arrays for fast iteration ---
    coord_np = (
        df.select([pl.col(c).cast(pl.Utf8) for c in coord_col_headers])
        .to_numpy()
    )
    value_np = (
        df.select([pl.col(c).cast(pl.Float64) for c in value_col_headers])
        .fill_null(0.0)
        .to_numpy()
    )

    # Precompute per-column wide indices to avoid repeated dict look-ups in loop.
    if wide_axis is not None and not is_time:
        col_indices = [col_to_idx.get(col) for col in value_col_headers]
    elif is_time:
        col_indices = [col_to_idx.get(col) for col in value_col_headers]
    else:
        col_indices = None  # scalar

    # --- Fill target array row by row ---
    for row_i in range(coord_np.shape[0]):
        # Resolve coordinate labels → array indices.
        base_coords = [0, 0, 0, 0]
        valid = True
        for j, (axis, dim_name) in enumerate(coord_axes_info):
            label = coord_np[row_i, j]
            idx = _resolve_dim_index(
                label,
                dim_name,
                idx_maps[dim_name],
                idx_maps_casefold[dim_name],
            )
            if idx is None:
                valid = False
                break
            base_coords[axis] = idx

        if not valid:
            continue

        if wide_axis is None:
            # Scalar variable: assign the single 'value' column.
            target[tuple(base_coords)] = value_np[row_i, 0]
        else:
            # Wide variable: iterate over value columns.
            row_vals = value_np[row_i]
            for col_j, w_idx in enumerate(col_indices):
                if w_idx is None:
                    continue
                coords = base_coords.copy()
                coords[wide_axis] = w_idx
                target[tuple(coords)] = row_vals[col_j]
