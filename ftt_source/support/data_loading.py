"""
Load per-variable CSV inputs stored in ``Inputs/``.

CSV schema
----------
Time-dependent variables use coordinate columns followed by year columns.
Other table variables use coordinate columns followed by the final dimension's
labels. Scalars use a single ``value`` column.
"""

import csv
import os
import warnings

import numpy as np
import polars as pl

from ftt_source.paths import get_inputs_path, get_utilities_path
from ftt_source.support.check_input_files import check_var_dims, check_vars_exist


def load_data(
    titles,
    dimensions,
    timeline,
    scenarios,
    ftt_modules,
    forstart,
    progress_callback=None,
    log_callback=None,
):
    """
    Load all model data for all scenarios.

    Scenario input folders inherit values from their matching S0 module folder
    before applying their own input files.
    """
    titles["TIME"] = timeline
    enabled = [module.strip() for module in ftt_modules.split(",") if module.strip()]
    supported = get_valid_ftt_models()
    unsupported = [module for module in enabled if module not in supported]
    if unsupported:
        raise ValueError(
            f"Unsupported FTT module(s) specified: {', '.join(unsupported)}. "
            f"Supported modules are: {', '.join(supported)}."
        )

    # General inputs are required for every model configuration.
    modules = enabled + ["General"]
    scenario_list = ["S0"] + [
        scenario.strip()
        for scenario in scenarios.split(",")
        if scenario.strip() and scenario.strip() != "S0"
    ]
    # Allocate each variable from the classifications it actually uses.
    data = {
        scenario: {
            variable: np.zeros(
                tuple(len(titles[dimension]) for dimension in dimensions[variable])
            )
            for variable in dimensions
        }
        for scenario in scenario_list
    }
    timeline_indices = {year: index for index, year in enumerate(timeline)}
    title_indices = {
        dimension: {
            str(value).strip().casefold(): index
            for index, value in enumerate(values)
        }
        for dimension, values in titles.items()
    }
    inputs_root = str(get_inputs_path())
    region_codes = {str(region) for region in titles.get("RTI_short", [])}

    empty_scenarios = []
    for scenario in scenario_list:
        if scenario == "S0":
            continue
        has_files = any(
            os.path.isdir(os.path.join(inputs_root, scenario, module))
            and any(
                filename.endswith(".csv")
                for filename in os.listdir(
                    os.path.join(inputs_root, scenario, module)
                )
            )
            for module in modules
        )
        if not has_files:
            empty_scenarios.append(scenario)

    if empty_scenarios:
        _emit_warning(
            f"No input files found for scenario(s): {', '.join(empty_scenarios)}.\n"
            "These scenarios will not be run.",
            log_callback,
        )
        for scenario in empty_scenarios:
            del data[scenario]
        scenario_list = [
            scenario for scenario in scenario_list if scenario not in empty_scenarios
        ]

    def read_folder(scenario, module, directory):
        """Read all CSV files in one scenario/module input folder."""
        loaded = set()
        legacy_variables = set()
        if not os.path.isdir(directory):
            return loaded
        for filename in os.listdir(directory):
            if not filename.endswith(".csv"):
                continue
            variable = filename[:-4]
            base, separator, suffix = variable.rpartition("_")
            if (
                separator
                and base in dimensions
                and suffix in region_codes
            ):
                legacy_variables.add(base)
                continue
            if variable not in dimensions:
                continue
            path = os.path.join(directory, filename)
            try:
                frame = pl.read_csv(
                    path,
                    infer_schema_length=10000,
                    null_values=["", "NA", "nan"],
                )
            except Exception as exc:
                _emit_warning(f"Could not read {path}: {exc}", log_callback)
                continue
            if scenario == "S0":
                check_var_dims(variable, frame, titles, dimensions)
            _fill_from_input_df(
                frame,
                variable,
                dimensions,
                titles,
                title_indices,
                timeline,
                timeline_indices,
                forstart,
                data[scenario][variable],
                log_callback=log_callback,
            )
            loaded.add(variable)
        if legacy_variables:
            _emit_warning(
                _legacy_files_message(
                    os.path.join(scenario, module),
                    legacy_variables,
                    sorted(region_codes)[0] if region_codes else "XX",
                ),
                log_callback,
            )
        return loaded

    for module in modules:
        # Load S0 first so scenario folders can override individual variables.
        check_vars_exist(module)
        module_variables = read_folder(
            "S0", module, os.path.join(inputs_root, "S0", module)
        )
        for scenario in scenario_list:
            if scenario == "S0":
                continue
            for variable in module_variables:
                data[scenario][variable] = data["S0"][variable].copy()
            read_folder(
                scenario,
                module,
                os.path.join(inputs_root, scenario, module),
            )

    return data


def _wide_axis(variable_dims):
    """Return the dimension represented by the value columns."""
    if not variable_dims:
        return None
    if len(variable_dims) == 1 and variable_dims[0] == "RTI":
        return None
    if "TIME" in variable_dims:
        return variable_dims.index("TIME")
    return len(variable_dims) - 1


def _resolve_indices(values, dimension, title_indices):
    """Resolve input labels to classification indices, using -1 for unknown labels."""
    return np.fromiter(
        (
            title_indices[dimension].get(str(value).strip().casefold(), -1)
            for value in values
        ),
        dtype=int,
        count=len(values),
    )


def _fill_from_input_df(
    frame,
    variable,
    dimensions,
    titles,
    title_indices,
    timeline,
    timeline_indices,
    forstart,
    target,
    log_callback=None,
):
    """Fill one variable array from its input table."""
    variable_dims = dimensions[variable]
    wide_axis = _wide_axis(variable_dims)
    is_time = wide_axis is not None and variable_dims[wide_axis] == "TIME"
    coordinate_dims = [
        (axis, dimension)
        for axis, dimension in enumerate(variable_dims)
        if axis != wide_axis
    ]
    columns = frame.columns

    # One-dimensional classification tables use label/value rows, for
    # example SectorCouplingAssumps.csv.
    if (
        len(variable_dims) == 1
        and wide_axis == 0
        and columns
        and columns[0] == variable_dims[0]
        and len(columns) >= 2
    ):
        labels = frame.select(pl.col(columns[0]).cast(pl.Utf8)).to_numpy().reshape(-1)
        values = (
            frame.select(pl.col(columns[1]).cast(pl.Float64))
            .fill_null(0.0)
            .to_numpy()
            .reshape(-1)
        )
        indices = _resolve_indices(labels, variable_dims[0], title_indices)
        valid = indices >= 0
        target[indices[valid]] = values[valid]
        skipped_rows = [
            (row_index + 2, variable_dims[0], label)
            for row_index, (label, is_valid) in enumerate(zip(labels, valid))
            if not is_valid
        ]
        if skipped_rows:
            _emit_warning(
                _row_mismatch_message(
                    variable,
                    len(labels),
                    skipped_rows,
                    [variable_dims[0]],
                    [columns[0]],
                ),
                log_callback,
            )
        return

    coordinate_columns = columns[: len(coordinate_dims)]
    value_columns = columns[len(coordinate_dims):]

    if not value_columns:
        return

    # Scalar variables contain only a single value column.
    if not variable_dims:
        target[...] = (
            frame.select(pl.col(value_columns[0]).cast(pl.Float64))
            .to_numpy()[0, 0]
        )
        return

    # Map each value column to its position in the target array.
    if is_time:
        start = forstart.get(variable)
        try:
            start = int(start)
        except (TypeError, ValueError):
            start = timeline[0]
        allowed = set(range(start, timeline[-1] + 1))
        column_indices = {
            column: timeline_indices[int(column)]
            for column in value_columns
            if str(column).lstrip("-").isdigit()
            and int(column) in timeline_indices
            and int(column) in allowed
        }
    elif wide_axis is not None:
        wide_dimension = variable_dims[wide_axis]
        labels = {
            str(value): index for index, value in enumerate(titles[wide_dimension])
        }
        column_indices = {
            column: labels.get(
                str(column),
                title_indices[wide_dimension].get(
                    str(column).strip().casefold()
                ),
            )
            for column in value_columns
            if (
                str(column) in labels
                or str(column).strip().casefold() in title_indices[wide_dimension]
            )
        }
        # Some exported inputs use positional headers rather than titles.
        if len(column_indices) != len(value_columns):
            numeric = [str(column).strip() for column in value_columns]
            if all(value.lstrip("-").isdigit() for value in numeric):
                positions = [int(value) for value in numeric]
                if positions == list(range(len(positions))):
                    column_indices = {
                        column: position
                        for column, position in zip(value_columns, positions)
                        if position < len(titles[wide_dimension])
                    }
                elif positions == list(range(1, len(positions) + 1)):
                    column_indices = dict(
                        (column, position - 1)
                        for column, position in zip(value_columns, positions)
                        if position <= len(titles[wide_dimension])
                    )
    else:
        column_indices = {}

    if coordinate_columns:
        coordinate_values = frame.select(
            [pl.col(column).cast(pl.Utf8) for column in coordinate_columns]
        ).to_numpy()
    else:
        coordinate_values = np.empty((frame.height, 0), dtype=object)
    values = (
        frame.select([pl.col(column).cast(pl.Float64) for column in value_columns])
        .fill_null(0.0)
        .to_numpy()
    )

    coordinate_indices = {}
    valid_rows = np.ones(len(coordinate_values), dtype=bool)
    skipped_rows = []
    skipped_row_indices = set()
    for coordinate_index, (axis, dimension) in enumerate(coordinate_dims):
        indices = _resolve_indices(
            coordinate_values[:, coordinate_index], dimension, title_indices
        )
        for row_index in np.flatnonzero(indices < 0):
            row_index = int(row_index)
            if row_index not in skipped_row_indices:
                skipped_rows.append(
                    (
                        row_index + 2,
                        dimension,
                        coordinate_values[row_index, coordinate_index],
                    )
                )
                skipped_row_indices.add(row_index)
        valid_rows &= indices >= 0
        coordinate_indices[axis] = indices

    if skipped_rows:
        _emit_warning(
            _row_mismatch_message(
                variable,
                len(coordinate_values),
                skipped_rows,
                [dimension for _, dimension in coordinate_dims],
                list(coordinate_columns),
            ),
            log_callback,
        )

    if wide_axis is not None and not is_time:
        matched_columns = sum(column in column_indices for column in value_columns)
        if matched_columns < len(value_columns):
            _emit_warning(
                _column_mismatch_message(
                    variable,
                    variable_dims[wide_axis],
                    matched_columns,
                    list(value_columns),
                ),
                log_callback,
            )

    if wide_axis is None:
        target[tuple(
            coordinate_indices[axis][valid_rows]
            for axis in range(len(variable_dims))
        )] = values[valid_rows, 0]
        return

    value_indices = [
        (value_index, column_indices[column])
        for value_index, column in enumerate(value_columns)
        if column in column_indices
    ]
    if not value_indices:
        return

    value_positions, wide_indices = zip(*value_indices)
    row_count = np.count_nonzero(valid_rows)
    array_indices = []
    for axis in range(len(variable_dims)):
        if axis == wide_axis:
            array_indices.append(np.tile(wide_indices, row_count))
        else:
            array_indices.append(
                np.repeat(coordinate_indices[axis][valid_rows], len(wide_indices))
            )
    target[tuple(array_indices)] = values[valid_rows][:, value_positions].reshape(-1)


def _emit_warning(message, log_callback=None):
    """Send warnings to the GUI log when available, or to the CLI otherwise."""
    if log_callback is not None:
        log_callback(f"WARNING: {message}")
    else:
        warnings.warn(message)


def _row_mismatch_message(
    variable, row_count, skipped_rows, expected_headers, actual_headers, max_listed=5
):
    skipped_count = len(skipped_rows)
    matched_count = row_count - skipped_count
    quantifier = "all" if matched_count == 0 else str(skipped_count)
    row_word = "input row" if skipped_count == 1 else "input rows"
    lines = [
        f"{variable}: {quantifier} {row_word} skipped due to classification mismatch "
        f"against classification_titles.csv for {expected_headers} "
        f"({matched_count}/{row_count} rows matched)."
    ]
    listed = "; ".join(
        f"line {line}: {dimension}='{label}'"
        for line, dimension, label in skipped_rows[:max_listed]
    )
    if skipped_count > max_listed:
        listed += f"; ... and {skipped_count - max_listed} more"
    lines.append(f"Unmatched labels ({listed}).")
    if actual_headers != expected_headers:
        lines.append(
            f"Column headers {actual_headers} do not match expected headers "
            f"{expected_headers}."
        )
    if matched_count == 0:
        lines.append(
            "The file is likely in the wrong format. "
            "All values for this variable will remain zero."
        )
    return "\n".join(lines)


def _column_mismatch_message(variable, wide_dimension, matched_count, headers):
    message = (
        f"{variable}: {matched_count}/{len(headers)} value column header(s) "
        f"{headers} matched the expected labels for '{wide_dimension}'."
    )
    if matched_count == 0:
        message += (
            "\nThe file is likely in the wrong format. "
            "All values for this variable will remain zero."
        )
    return message


def _legacy_files_message(folder, legacy_variables, example_region):
    names = sorted(legacy_variables)
    return (
        f"{folder}: found old per-country input files "
        f"(e.g. {names[0]}_{example_region}.csv) for {len(names)} variable(s): "
        f"{', '.join(names)}.\n"
        "These files are ignored. Inputs now use a single file per variable "
        "(e.g. BZTC.csv).\n"
        "Update or delete the old files and use the new format."
    )


def get_valid_ftt_models():
    """Return model names listed in ``classification_titles.csv``."""
    path = get_utilities_path() / "titles" / "classification_titles.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Classification titles file not found at: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if row[0].strip() == "Models" and row[4].strip() == "Short name":
                return [value.strip() for value in row[5:] if value.strip()]
    return []
