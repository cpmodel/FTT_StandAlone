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

    def read_folder(scenario, module, directory):
        """Read all CSV files in one scenario/module input folder."""
        loaded = set()
        if not os.path.isdir(directory):
            return loaded
        for filename in os.listdir(directory):
            if not filename.endswith(".csv"):
                continue
            variable = filename[:-4]
            if variable not in dimensions:
                continue
            path = os.path.join(directory, filename)
            try:
                frame = pl.read_csv(
                    path,
                    infer_schema_length=10000,
                    null_values=["", "NA", "nan"],
                )
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
                )
            except Exception as exc:
                warnings.warn(f"Could not read {path}: {exc}")
                continue
            loaded.add(variable)
        return loaded

    for module in modules:
        # Load S0 first so scenario folders can override individual variables.
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


def _resolve_index(label, dimension, title_indices):
    """Resolve a CSV label against its classification titles."""
    return title_indices[dimension].get(str(label).strip().casefold())


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
        for label, value in zip(labels, values):
            index = _resolve_index(label, variable_dims[0], title_indices)
            if index is not None:
                target[index] = value
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
            column: labels[column]
            for column in value_columns
            if str(column) in labels
        }
        # Some exported inputs use positional headers rather than titles.
        if len(column_indices) != len(value_columns):
            numeric = [str(column).strip() for column in value_columns]
            if all(value.lstrip("-").isdigit() for value in numeric):
                positions = [int(value) for value in numeric]
                if positions == list(range(len(positions))):
                    column_indices = dict(zip(value_columns, positions))
                elif positions == list(range(1, len(positions) + 1)):
                    column_indices = dict(
                        zip(value_columns, [position - 1 for position in positions])
                    )
    else:
        column_indices = {}

    coordinate_values = frame.select(
        [pl.col(column).cast(pl.Utf8) for column in coordinate_columns]
    ).to_numpy()
    values = (
        frame.select([pl.col(column).cast(pl.Float64) for column in value_columns])
        .fill_null(0.0)
        .to_numpy()
    )

    # Resolve coordinates row by row, then assign each value column.
    for row_index in range(coordinate_values.shape[0]):
        coordinates = [0] * len(variable_dims)
        valid = True
        for coordinate_index, (axis, dimension) in enumerate(coordinate_dims):
            index = _resolve_index(
                coordinate_values[row_index, coordinate_index],
                dimension,
                title_indices,
            )
            if index is None:
                valid = False
                break
            coordinates[axis] = index
        if not valid:
            continue

        if wide_axis is None:
            target[tuple(coordinates)] = values[row_index, 0]
            continue
        for value_index, column in enumerate(value_columns):
            index = column_indices.get(column)
            if index is None:
                continue
            coordinates[wide_axis] = index
            target[tuple(coordinates)] = values[row_index, value_index]


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
