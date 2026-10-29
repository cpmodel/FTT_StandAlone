# -*- coding: utf-8 -*-
"""
=========================================
convert_inputs_to_new_format.py
=========================================
Converter for masterfile, single region csv input data to new format.

This script reads legacy masterfiles and existing CSVs from Inputs_existing/,
validates the source structure, and exports all data in new format to Inputs/.

The new format schema:
    - non-time variables (e.g., SectorCouplingAssumps): one row with a 'value' column
    - non-time 3D matrices (e.g, RTKM): first dimensions as row coordinates, third dimension as columns
    - time variables (e.g., MEWG): one row per non-time coordinate with year columns

Usage:
    python convert_inputs_to_new_format.py [--models MODEL1,MODEL2] [--scenarios S0,S1] [--overwrite]

Examples:
    python convert_inputs_to_new_format.py                          
    python convert_inputs_to_new_format.py --models FTT-P,FTT-Tr   # Convert specific models
    python convert_inputs_to_new_format.py --overwrite             # Force overwrite existing files
"""

import argparse
import configparser
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / 'SourceCode').exists() else SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SourceCode.support.convert_masterfiles_to_csv import (
    extract_data,
    get_model_classification,
    get_sheets_to_convert,
    read_data,
    set_up_cols,
    set_up_rows,
    variable_setup,
)
from SourceCode.support.dimensions_functions import load_dims
from SourceCode.support.titles_functions import load_titles

DIMENSION_NAME_MAP = {
    'RSHORTTI': 'RTI',
}


def resolve_project_path(path_value):
    """Resolve a path relative to project root when not absolute."""
    path_obj = Path(path_value)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert legacy input data to new format.'
    )
    parser.add_argument(
        '--models',
        help='Comma-separated model names. Defaults to all available models.',
    )
    parser.add_argument(
        '--scenarios',
        help='Comma-separated scenario names like S0,S1. Defaults to S0.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files in Inputs/',
    )
    return parser.parse_args()

def get_defaults():
    """Return default models and scenarios."""
    return {
        'models': ['FTT-P', 'FTT-Fr', 'FTT-Tr', 'FTT-H'],
        'scenarios': ['S0'],
    }


def generate_model_list_from_source(source_root, ftt_modules, scenarios):
    """Build model mapping from masterfiles under source_root/_MasterFiles."""
    scenarios = [item for item in scenarios if item != 'Gamma']
    models = {}

    for module in ftt_modules:
        module_scenarios = []
        file_root = None
        module_master_dir = Path(source_root) / '_MasterFiles' / module

        for scenario in scenarios:
            pattern = f"{module}*_{scenario}.xlsx"
            matches = list(module_master_dir.glob(pattern)) if module_master_dir.exists() else []

            if not matches:
                continue

            if len(matches) > 1:
                print(
                    f"Warning: Multiple files matched for module {module} and scenario {scenario}; "
                    f"using {matches[0].name}."
                )

            base_name = matches[0].name
            end_index = base_name.index(f'_{scenario}.xlsx')
            file_root = base_name[:end_index]
            module_scenarios.append(int(scenario[1:]))

        if module_scenarios and file_root:
            models[module] = [module_scenarios, file_root]

    return models


def validate_source_folder(source_root='Inputs_existing'):
    """
    Validate that source folder exists and contains masterfiles.
    
    Parameters
    ----------
    source_root : str
        Path to legacy inputs folder.
        
    Returns
    -------
    bool
        True if valid source structure is found.
        
    Raises
    ------
    FileNotFoundError
        If source folder does not exist or contains no masterfiles.
    """
    source_root_path = resolve_project_path(source_root)

    if not source_root_path.is_dir():
        raise FileNotFoundError(
            f"\nLegacy inputs folder '{source_root_path}' not found.\n"
            f"To convert data, please rename your old 'Inputs' folder to 'Inputs_existing':\n"
            f"  mv Inputs Inputs_existing\n"
            f"Then run this script again."
        )
    
    # Check for masterfiles in FTT-P, FTT-Tr, etc.
    masterfiles_dir = source_root_path / '_MasterFiles'
    if not masterfiles_dir.is_dir():
        raise FileNotFoundError(
            f"\nNo '_MasterFiles' folder found in '{source_root_path}'.\n"
            f"Data might have already been converted. Use --overwrite to force re-conversion."
        )
    
    masterfiles = [f for f in os.listdir(masterfiles_dir) if f.endswith('.xlsx')]
    if not masterfiles:
        raise FileNotFoundError(
            f"\nNo Excel masterfiles (.xlsx) found in '{masterfiles_dir}'.\n"
            f"Cannot proceed with data conversion."
        )
    
    return True


def normalise_dimension_name(dim_name):
    """Map legacy dimension names to new format dimension names."""
    return DIMENSION_NAME_MAP.get(dim_name, dim_name)


def export_labels(dim_name, titles):
    """Export dimension labels from titles dict."""
    return list(titles[dim_name])


def create_export_array(variable_dims, titles, time_labels=None):
    """Create appropriately-sized numpy array for variable export."""
    shape = []
    for dim_name in variable_dims:
        if dim_name == 'NA':
            shape.append(1)
        elif dim_name == 'TIME':
            shape.append(len(time_labels or []))
        else:
            shape.append(len(titles[dim_name]))
    return np.zeros(shape, dtype=np.float32)


def fill_array_from_frame(target_array, variable_dims, row_dim_name, data_frame,
                          col_dim_name=None, block_dim_name=None, block_index=0,
                          row_axis_override=None, col_axis_override=None,
                          block_axis_override=None):
    """Fill target array from DataFrame using dimension mappings."""
    coordinates = [0, 0, 0, 0]

    if block_dim_name is not None:
        if block_axis_override is not None:
            block_axis = block_axis_override
        else:
            block_axis = variable_dims.index(normalise_dimension_name(block_dim_name))
        coordinates[block_axis] = block_index

    if row_axis_override is not None:
        row_axis = row_axis_override
    else:
        row_axis = variable_dims.index(normalise_dimension_name(row_dim_name))

    col_axis = None
    if col_dim_name is not None:
        if col_axis_override is not None:
            col_axis = col_axis_override
        elif col_dim_name == 'TIME':
            col_axis = 3
        else:
            col_axis = variable_dims.index(normalise_dimension_name(col_dim_name))

    values = data_frame.to_numpy(dtype=np.float32)

    if col_axis is None:
        values = values.reshape(-1)
        for row_index, value in enumerate(values):
            current = coordinates.copy()
            current[row_axis] = row_index
            target_array[tuple(current)] = value
        return

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            current = coordinates.copy()
            current[row_axis] = row_index
            current[col_axis] = col_index
            target_array[tuple(current)] = values[row_index, col_index]


def dataframe_from_array(variable_dims, array, titles, time_labels=None):
    """Convert numpy array to DataFrame in new format."""
    coord_axes = [
        (axis, dim_name) for axis, dim_name in enumerate(variable_dims)
        if dim_name not in ('NA', 'TIME')
    ]

    records = []

    if variable_dims[3] == 'TIME':
        coord_ranges = [range(array.shape[axis]) for axis, _ in coord_axes]
        for coord_values in itertools.product(*coord_ranges):
            record = {}
            coordinates = [0, 0, 0, 0]
            for value, (axis, dim_name) in zip(coord_values, coord_axes):
                coordinates[axis] = value
                record[dim_name] = export_labels(dim_name, titles)[value]

            for time_index, year in enumerate(time_labels or []):
                coordinates[3] = time_index
                record[str(year)] = float(array[tuple(coordinates)])

            records.append(record)

        return pd.DataFrame(records)

    coord_ranges = [range(array.shape[axis]) for axis, _ in coord_axes]
    for coord_values in itertools.product(*coord_ranges):
        record = {}
        coordinates = [0, 0, 0, 0]
        for value, (axis, dim_name) in zip(coord_values, coord_axes):
            coordinates[axis] = value
            record[dim_name] = export_labels(dim_name, titles)[value]

        record['value'] = float(array[tuple(coordinates)])
        records.append(record)

    return pd.DataFrame(records)


def dataframe_from_wide_non_time(variable_dims, array, titles, column_axis):
    """Convert numpy array to wide (non-time) DataFrame format."""
    row_axes = [
        (axis, dim_name)
        for axis, dim_name in enumerate(variable_dims)
        if dim_name not in ('NA', 'TIME') and axis != column_axis
    ]
    column_dim_name = variable_dims[column_axis]
    column_labels = export_labels(column_dim_name, titles)

    records = []
    row_ranges = [range(array.shape[axis]) for axis, _ in row_axes]

    for row_values in itertools.product(*row_ranges):
        record = {}
        coordinates = [0, 0, 0, 0]

        for value, (axis, dim_name) in zip(row_values, row_axes):
            coordinates[axis] = value
            record[dim_name] = export_labels(dim_name, titles)[value]

        for column_index, column_label in enumerate(column_labels):
            coordinates[column_axis] = column_index
            record[column_label] = float(array[tuple(coordinates)])

        records.append(record)

    return pd.DataFrame(records)


def wide_column_axis(variable_dims):
    """Determine which axis is used as columns in wide non-time format."""
    if variable_dims[3] == 'TIME':
        return None
    if variable_dims[2] not in ('NA', 'TIME') and variable_dims[3] == 'NA':
        return 2
    if variable_dims[1] not in ('NA', 'TIME') and variable_dims[2] == 'NA' and variable_dims[3] == 'NA':
        return 1
    return None


def parse_titles_sheet(raw_titles_sheet):
    """Parse titles from Excel sheet."""
    titles = {}
    for column_index in range(1, raw_titles_sheet.shape[1], 2):
        title_name = raw_titles_sheet.iloc[0, column_index]
        title_values = list(raw_titles_sheet.iloc[:, column_index].dropna())[1:]
        titles[title_name] = title_values
    return titles


def find_missing_model_dimensions(model_variables_df, titles):
    """Identify dimension keys in masterfile that don't exist in titles."""
    missing = set()
    for col_name in ['RowDim', 'ColDim', '3DDim']:
        for dim in model_variables_df[col_name].dropna().tolist():
            if dim in (0, 'TIME'):
                continue
            normalised_dim = normalise_dimension_name(str(dim))
            if normalised_dim not in titles:
                missing.add(str(dim))
    return sorted(missing)


def write_output_file(file_path, dataframe, overwrite):
    """Write DataFrame to CSV file."""
    if os.path.exists(file_path) and not overwrite:
        return False
    
    dataframe.to_csv(file_path, index=False)
    return True


def _map_rti_label(label, rti_short_to_long, titles):
    """Map region label from legacy format to full name."""
    label = str(label)
    if label in titles['RTI']:
        return label

    legacy_aliases = {
        'XX': titles['RTI'][-1],
        '71 Dummy Region': titles['RTI'][-1],
    }
    if label in legacy_aliases:
        return legacy_aliases[label]

    return rti_short_to_long.get(label, label)


def _normalise_rti_label(label, titles, rti_short_to_long, rti_aliases):
    """Normalize region label with alias support."""
    label = str(label)
    if label in titles['RTI']:
        return label
    if label in ('XX', '71 Dummy Region'):
        return titles['RTI'][-1]
    mapped_code = rti_aliases.get(label, label)
    return rti_short_to_long.get(mapped_code, label)


def _sort_by_rti(frame, titles):
    """Sort DataFrame rows by canonical RTI ordering."""
    if 'RTI' not in frame.columns:
        return frame
    frame = frame.copy()
    rti_short_to_long = {
        code: name for code, name in zip(titles['RTI_short'], titles['RTI'])
    }
    rti_aliases = {'XX': titles['RTI_short'][-1]}
    frame['RTI'] = frame['RTI'].map(
        lambda label: _normalise_rti_label(label, titles, rti_short_to_long, rti_aliases)
    )
    frame['RTI'] = pd.Categorical(frame['RTI'], categories=titles['RTI'], ordered=True)
    frame = frame.sort_values('RTI', kind='stable').reset_index(drop=True)
    frame['RTI'] = frame['RTI'].astype(str)
    return frame


def _build_standalone_supplement_frame(file_path, variable_name, variable_dims, titles):
    """Convert a legacy standalone CSV into new format."""
    data_frame = pd.read_csv(file_path, index_col=0)

    if variable_dims == ['NA', 'NA', 'NA', 'NA']:
        value = float(data_frame.iloc[0, 0]) if not data_frame.empty else 0.0
        return pd.DataFrame({'value': [value]})

    if variable_dims[0] == 'RTI' and variable_dims[1] == 'NA' and variable_dims[3] == 'TIME':
        export_frame = data_frame.reset_index()
        export_frame.rename(columns={'index': 'RTI'}, inplace=True)
        rti_short_to_long = {
            short: full for short, full in zip(titles['RTI_short'], titles['RTI'])
        }
        export_frame['RTI'] = export_frame['RTI'].map(
            lambda label: _map_rti_label(label, rti_short_to_long, titles)
        )
        return export_frame

    if variable_dims[0] == 'NA' and variable_dims[1] != 'NA' and variable_dims[3] == 'TIME':
        export_frame = data_frame.reset_index()
        export_frame.rename(columns={'index': variable_dims[1]}, inplace=True)
        return export_frame

    if variable_dims[0] == 'RTI' and variable_dims[1] == 'NA' and variable_dims[3] == 'NA':
        export_frame = data_frame.reset_index()
        export_frame.rename(columns={'index': 'RTI'}, inplace=True)
        rti_short_to_long = {
            short: full for short, full in zip(titles['RTI_short'], titles['RTI'])
        }
        export_frame['RTI'] = export_frame['RTI'].map(
            lambda label: _map_rti_label(label, rti_short_to_long, titles)
        )
        value_columns = [col for col in export_frame.columns if col != 'RTI']
        if len(value_columns) == 1:
            export_frame.rename(columns={value_columns[0]: 'value'}, inplace=True)
        return export_frame

    return None


def _build_region_split_supplement_frame(grouped_files, variable_name, variable_dims, titles):
    """Convert legacy region-split CSVs like MEWD_BE.csv into new format."""
    if not (
        variable_dims[0] == 'RTI'
        and variable_dims[1] != 'NA'
        and variable_dims[2] == 'NA'
        and variable_dims[3] == 'TIME'
    ):
        return None

    rti_short_to_long = {
        short: full for short, full in zip(titles['RTI_short'], titles['RTI'])
    }
    rti_order = {short: i for i, short in enumerate(titles['RTI_short'])}
    rti_aliases = {
        'XX': titles['RTI_short'][-1],
    }
    second_dim = variable_dims[1]
    frames_by_region = {}
    year_columns = None

    sorted_region_items = sorted(
        grouped_files.items(),
        key=lambda item: rti_order.get(item[0], 10**9),
    )

    for region_code, file_path in sorted_region_items:
        region_code = rti_aliases.get(region_code, region_code)
        if region_code not in rti_short_to_long:
            continue

        data_frame = pd.read_csv(file_path, index_col=0)
        export_frame = data_frame.reset_index()
        export_frame.rename(columns={'index': second_dim}, inplace=True)
        export_frame.insert(0, 'RTI', rti_short_to_long[region_code])
        frames_by_region[region_code] = export_frame
        if year_columns is None:
            year_columns = [col for col in export_frame.columns if col not in ['RTI', second_dim]]

    if not frames_by_region:
        return None

    all_frames = []
    for region_code in titles['RTI_short']:
        region_name = rti_short_to_long[region_code]
        if region_code in frames_by_region:
            all_frames.append(frames_by_region[region_code])
            continue

        # Fill missing regions with zeros to preserve canonical RTI ordering.
        if year_columns is None:
            year_columns = []
        fill_rows = pd.DataFrame({
            'RTI': [region_name] * len(titles[second_dim]),
            second_dim: list(titles[second_dim]),
        })
        for col in year_columns:
            fill_rows[col] = 0.0
        all_frames.append(fill_rows)

    return pd.concat(all_frames, ignore_index=True)


def export_existing_csv_supplements(legacy_dir, out_dir, titles, variable_dims_map, overwrite=False):
    """Export legacy-only CSV inputs that were not written from masterfiles."""
    if not os.path.isdir(legacy_dir):
        return

    csv_files = [name for name in os.listdir(legacy_dir) if name.endswith('.csv')]
    region_suffixes = set(titles['RTI_short'])
    grouped_region_files = {}
    standalone_files = {}

    for filename in csv_files:
        stem = filename[:-4]
        parts = stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in region_suffixes:
            grouped_region_files.setdefault(parts[0], {})[parts[1]] = os.path.join(legacy_dir, filename)
        else:
            standalone_files[stem] = os.path.join(legacy_dir, filename)

    for variable_name, file_path in sorted(standalone_files.items()):
        output_path = os.path.join(out_dir, f'{variable_name}.csv')
        if os.path.exists(output_path) and not overwrite:
            continue
        if variable_name not in variable_dims_map:
            continue

        export_frame = _build_standalone_supplement_frame(
            file_path, variable_name, variable_dims_map[variable_name], titles
        )
        if export_frame is None:
            continue
        write_output_file(output_path, export_frame, overwrite=overwrite)

    for variable_name, grouped_files in sorted(grouped_region_files.items()):
        output_path = os.path.join(out_dir, f'{variable_name}.csv')
        if os.path.exists(output_path) and not overwrite:
            continue
        if variable_name not in variable_dims_map:
            continue

        export_frame = _build_region_split_supplement_frame(
            grouped_files, variable_name, variable_dims_map[variable_name], titles
        )
        if export_frame is None:
            continue
        write_output_file(output_path, export_frame, overwrite=overwrite)


def convert_masterfiles_to_new_format(models, overwrite=False, output_root='Inputs', source_root='Inputs_existing'):
    """
    Convert masterfile data to new format CSVs.
    
    Parameters
    ----------
    models : list of str
        Model names to convert.
    overwrite : bool
        Whether to overwrite existing output files.
    output_root : str
        Root directory for output (e.g., 'Inputs' to write to Inputs/S0/FTT-P/).
    """
    source_root_path = resolve_project_path(source_root)
    output_root_path = resolve_project_path(output_root)

    dir_inputs = str(source_root_path)
    dir_masterfiles = str(source_root_path / '_MasterFiles')
    
    variables_df_dict, var_dict, _, scenarios, timeline_dict = variable_setup(dir_masterfiles, models)
    titles = load_titles()
    variable_dims_map, _, _, _, _ = load_dims()

    failed_models = []

    for model in models:
        model_variables_df = variables_df_dict[model]
        missing_dims = find_missing_model_dimensions(model_variables_df, titles)
        if missing_dims:
            print(
                f'Skipping model {model}: missing title dimensions {missing_dims}. '
                'Update titles mapping before exporting this model.'
            )
            failed_models.append(model)
            continue

        model_classifications = get_model_classification(model_variables_df)

        for scenario in scenarios:
            vars_to_convert, sheets = get_sheets_to_convert(var_dict, model, scenario)
            out_dir = os.path.join(str(output_root_path), f'S{scenario}', model)
            os.makedirs(out_dir, exist_ok=True)

            raw_data = read_data(models, model, dir_masterfiles, scenario, sheets)
            if raw_data is not None:
                titles_sheet = parse_titles_sheet(raw_data['Titles'])
                row_start = 5
                column_start = 2
                regions = model_classifications['RSHORTTI']

                for variable_name in vars_to_convert:
                    try:
                        if variable_name not in variable_dims_map:
                            continue

                        master_dims = var_dict[model][variable_name]['Dims']
                        variable_dims = variable_dims_map[variable_name]
                        row_dim_name = normalise_dimension_name(master_dims[0])
                        row_count, _ = set_up_rows(model, variable_name, var_dict, model_classifications)
                        col_count, col_titles = set_up_cols(model, variable_name, var_dict, model_classifications, timeline_dict)
                        excel_dim = len(titles_sheet[master_dims[0]])
                        column_finish = column_start + col_count
                        separator = 1 + excel_dim - row_count
                        conversion = var_dict[model][variable_name]['Conversion?']

                        time_labels = col_titles if len(master_dims) > 1 and master_dims[1] == 'TIME' else None
                        if len(master_dims) == 1 and conversion == 'TIME':
                            time_labels = timeline_dict[variable_name]

                        export_array = create_export_array(variable_dims, titles, time_labels)
                        if len(master_dims) == 3:
                            row_axis_override = None
                            col_axis_override = None
                            normalised_row_dim = normalise_dimension_name(row_dim_name)
                            normalised_col_dim = normalise_dimension_name(master_dims[1])
                            if normalised_row_dim == normalised_col_dim:
                                matching_axes = [
                                    axis for axis, dim_name in enumerate(variable_dims)
                                    if dim_name == normalised_row_dim
                                ]
                                if len(matching_axes) >= 2:
                                    row_axis_override = matching_axes[0]
                                    col_axis_override = matching_axes[1]

                            for region_index, _region_name in enumerate(regions):
                                row_index = row_start + region_index * (row_count + separator)
                                data_frame = extract_data(raw_data, variable_name, row_index, row_count, column_start, column_finish)
                                fill_array_from_frame(
                                    export_array,
                                    variable_dims,
                                    row_dim_name=row_dim_name,
                                    data_frame=data_frame,
                                    col_dim_name=master_dims[1],
                                    block_dim_name=master_dims[2],
                                    block_index=region_index,
                                    row_axis_override=row_axis_override,
                                    col_axis_override=col_axis_override,
                                )

                            export_wide_axis = wide_column_axis(variable_dims)
                            if export_wide_axis is not None:
                                export_frame = dataframe_from_wide_non_time(variable_dims, export_array, titles, export_wide_axis)
                            else:
                                export_frame = dataframe_from_array(variable_dims, export_array, titles, time_labels)
                            output_path = os.path.join(out_dir, f'{variable_name}.csv')
                            write_output_file(output_path, export_frame, overwrite)
                            continue

                        data_frame = extract_data(raw_data, variable_name, row_start, row_count, column_start, column_finish)

                        if len(master_dims) == 2:
                            needs_transposing = model_variables_df.loc[
                                model_variables_df['Variable name'] == variable_name,
                                'ColDim'
                            ].iloc[0] == 'RSHORTTI'
                            export_row_dim_name = row_dim_name
                            export_col_dim_name = master_dims[1]
                            if needs_transposing:
                                data_frame = data_frame.T
                                export_row_dim_name = normalise_dimension_name(master_dims[1])
                                export_col_dim_name = master_dims[0]

                            row_axis_override = None
                            col_axis_override = None
                            normalised_row_dim = normalise_dimension_name(export_row_dim_name)
                            normalised_col_dim = normalise_dimension_name(export_col_dim_name)
                            if normalised_row_dim == normalised_col_dim:
                                matching_axes = [
                                    axis for axis, dim_name in enumerate(variable_dims)
                                    if dim_name == normalised_row_dim
                                ]
                                if len(matching_axes) >= 2:
                                    row_axis_override = matching_axes[0]
                                    col_axis_override = matching_axes[1]

                            fill_array_from_frame(
                                export_array,
                                variable_dims,
                                row_dim_name=export_row_dim_name,
                                data_frame=data_frame,
                                col_dim_name=export_col_dim_name,
                                row_axis_override=row_axis_override,
                                col_axis_override=col_axis_override,
                            )

                        elif len(master_dims) == 1:
                            if conversion == 'TIME':
                                tiled = np.tile(data_frame.to_numpy(dtype=np.float32), (1, len(time_labels)))
                                tiled_frame = pd.DataFrame(tiled, index=data_frame.index, columns=time_labels)
                                fill_array_from_frame(
                                    export_array,
                                    variable_dims,
                                    row_dim_name=row_dim_name,
                                    data_frame=tiled_frame,
                                    col_dim_name='TIME',
                                )
                            else:
                                fill_array_from_frame(
                                    export_array,
                                    variable_dims,
                                    row_dim_name=row_dim_name,
                                    data_frame=data_frame,
                                )

                        export_wide_axis = wide_column_axis(variable_dims)
                        if export_wide_axis is not None:
                            export_frame = dataframe_from_wide_non_time(variable_dims, export_array, titles, export_wide_axis)
                        else:
                            export_frame = dataframe_from_array(variable_dims, export_array, titles, time_labels)
                        output_path = os.path.join(out_dir, f'{variable_name}.csv')
                        write_output_file(output_path, export_frame, overwrite)
                    except Exception as exc:
                        raise RuntimeError(
                            f'Failed exporting {variable_name} for {model} S{scenario}. '
                            f'Master dims: {master_dims}; new format dims: {variable_dims}'
                        ) from exc

            legacy_dir = os.path.join(dir_inputs, f'S{scenario}', model)
            export_existing_csv_supplements(legacy_dir, out_dir, titles, variable_dims_map, overwrite=overwrite)

    if failed_models:
        print(f'Export completed with skipped models: {sorted(set(failed_models))}')


def convert_general_to_new_format(output_root='Inputs', overwrite=False):
    """
    Convert legacy General CSVs to new format.
    
    Handles both standalone and region-keyed files.
    Validates RTI ordering and ensures consistent dimension handling.
    
    Parameters
    ----------
    output_root : str
        Root directory for output (e.g., 'Inputs' to write to Inputs/S0/General/).
    overwrite : bool
        Whether to overwrite existing output files.
    """
    titles = load_titles()
    dims, _, _, _, _ = load_dims()

    # Region mapping
    rti_short_to_long = {code: name for code, name in zip(titles['RTI_short'], titles['RTI'])}
    rti_order = {code: i for i, code in enumerate(titles['RTI_short'])}
    rti_aliases = {'XX': titles['RTI_short'][-1]}

    input_dir = resolve_project_path('Inputs_existing/S0/General')
    output_dir = resolve_project_path(output_root) / 'S0/General'
    os.makedirs(output_dir, exist_ok=True)

    if not input_dir.is_dir():
        print(f"Skipping General conversion: {input_dir} not found")
        return

    print(f"Converting General CSVs from {input_dir} to {output_dir}")

    # Collect all files
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"Found {len(all_files)} CSV files in {input_dir}")

    # Separate region-keyed files from standalone files
    region_keyed = {}
    standalone = []

    for filename in all_files:
        var_base = filename[:-4]

        if '_' in var_base:
            parts = var_base.split('_')
            if len(parts) == 2 and len(parts[1]) == 2:
                region_code = parts[1]
                var_name = parts[0]
                if var_name not in region_keyed:
                    region_keyed[var_name] = {}
                region_keyed[var_name][region_code] = os.path.join(str(input_dir), filename)
                continue

            standalone.append((var_base, os.path.join(str(input_dir), filename)))


    # === Convert standalone files ===
    for var_name, filepath in standalone:
        if var_name not in dims:
            print(f"  SKIP {var_name}: not in dimensions")
            continue

        var_dims = tuple(dims[var_name]) if isinstance(dims[var_name], list) else dims[var_name]

        df = pd.read_csv(filepath, index_col=0)

        if var_dims == ('NA', 'NA', 'NA', 'NA'):
            value = float(df.iloc[0, 0]) if df.shape[0] > 0 and df.shape[1] > 0 else 0.0
            canonical_df = pd.DataFrame({'value': [value]})

        elif var_dims == ('RTI', 'NA', 'NA', 'TIME'):
            canonical_df = df.reset_index()
            canonical_df.rename(columns={'index': 'RTI'}, inplace=True)
            canonical_df['RTI'] = canonical_df['RTI'].map(
                lambda label: _normalise_rti_label(label, titles, rti_short_to_long, rti_aliases)
            )
            year_cols = [col for col in canonical_df.columns if col not in ['RTI']]
            canonical_df = canonical_df[['RTI'] + year_cols]
            for col in year_cols:
                try:
                    canonical_df[col] = pd.to_numeric(canonical_df[col], errors='coerce').fillna(0.0)
                except:
                    canonical_df[col] = 0.0

        elif var_dims == ('NA', 'MTI', 'NA', 'TIME'):
            canonical_df = df.reset_index()
            canonical_df.rename(columns={'index': 'MTI'}, inplace=True)
            year_cols = [col for col in canonical_df.columns if col not in ['MTI']]
            canonical_df = canonical_df[['MTI'] + year_cols]
            for col in year_cols:
                try:
                    canonical_df[col] = pd.to_numeric(canonical_df[col], errors='coerce').fillna(0.0)
                except:
                    canonical_df[col] = 0.0

        elif var_dims == ('NA', 'SCA', 'NA', 'NA'):
            canonical_df = df.reset_index()
            canonical_df.rename(columns={'index': 'SCA'}, inplace=True)
            for col in canonical_df.columns:
                if col != 'SCA':
                    try:
                        canonical_df[col] = pd.to_numeric(canonical_df[col], errors='coerce').fillna(0.0)
                    except:
                        canonical_df[col] = 0.0

        else:
            print(f"  UNSUPPORTED dims {var_dims}, skipping")
            continue

        canonical_df = _sort_by_rti(canonical_df, titles)
        output_path = os.path.join(str(output_dir), f'{var_name}.csv')
        if not write_output_file(output_path, canonical_df, overwrite):
            print(f"  (skipped existing {var_name}.csv)")

    # === Convert region-keyed files ===
    for var_name, region_files in region_keyed.items():
        if var_name not in dims:
            print(f"  SKIP {var_name}: not in dimensions")
            continue

        var_dims = tuple(dims[var_name]) if isinstance(dims[var_name], list) else dims[var_name]

        all_data = []
        seen_regions = set()
        value_columns_template = None

        sorted_region_items = sorted(
            region_files.items(),
            key=lambda item: rti_order.get(item[0], 10**9)
        )

        for region_code, filepath in sorted_region_items:
            region_code = rti_aliases.get(region_code, region_code)
            if region_code not in rti_short_to_long:
                print(f"  WARN: Unknown region code {region_code}")
                continue

            rti_long = rti_short_to_long[region_code]
            seen_regions.add(region_code)

            df = pd.read_csv(filepath, index_col=0)
            if value_columns_template is None:
                value_columns_template = list(df.columns)

            if var_dims == ('RTI', 'NA', 'NA', 'TIME'):
                row_data = {'RTI': rti_long}
                for col in df.columns:
                    try:
                        row_data[col] = float(df.iloc[0, df.columns.get_loc(col)])
                    except:
                        row_data[col] = 0.0
                all_data.append(row_data)

            elif var_dims == ('RTI', 'NA', 'NA', 'NA'):
                row_data = {'RTI': rti_long, 'value': float(df.iloc[0, 0]) if df.shape[0] > 0 and df.shape[1] > 0 else 0.0}
                all_data.append(row_data)

            elif var_dims[1] != 'NA':
                dim2_name = var_dims[1]
                for idx in df.index:
                    row_data = {'RTI': rti_long, dim2_name: str(idx)}
                    for col in df.columns:
                        try:
                            row_data[col] = float(df.loc[idx, col])
                        except:
                            row_data[col] = 0.0
                    all_data.append(row_data)

            else:
                row_data = {'RTI': rti_long}
                for col in df.columns:
                    try:
                        row_data[col] = float(df.iloc[0, df.columns.get_loc(col)])
                    except:
                        row_data[col] = 0.0
                all_data.append(row_data)

        # Fill missing regions with zero rows
        missing_region_codes = [code for code in titles['RTI_short'] if code not in seen_regions]
        for region_code in missing_region_codes:
            rti_long = rti_short_to_long[region_code]
            if var_dims == ('RTI', 'NA', 'NA', 'TIME'):
                row_data = {'RTI': rti_long}
                for col in (value_columns_template or []):
                    row_data[col] = 0.0
                all_data.append(row_data)
            elif var_dims == ('RTI', 'NA', 'NA', 'NA'):
                all_data.append({'RTI': rti_long, 'value': 0.0})
            elif var_dims[1] != 'NA':
                dim2_name = var_dims[1]
                for idx_label in titles[dim2_name]:
                    row_data = {'RTI': rti_long, dim2_name: str(idx_label)}
                    for col in (value_columns_template or []):
                        row_data[col] = 0.0
                    all_data.append(row_data)

        if not all_data:
            print(f"  SKIP {var_name}: no valid regional data")
            continue

        canonical_df = pd.DataFrame(all_data)
        coord_cols = [c for c in canonical_df.columns if c in ['RTI', 'DPTI', 'FUTI', 'ERTI', 'CSCTI']]
        value_cols = [c for c in canonical_df.columns if c not in coord_cols]
        canonical_df = canonical_df[coord_cols + value_cols]

        for col in value_cols:
            try:
                canonical_df[col] = pd.to_numeric(canonical_df[col], errors='coerce').fillna(0.0)
            except:
                canonical_df[col] = 0.0

        canonical_df = _sort_by_rti(canonical_df, titles)
        output_path = os.path.join(str(output_dir), f'{var_name}.csv')
        if write_output_file(output_path, canonical_df, overwrite):
            print(f"  ✓ Created {var_name}.csv shape={canonical_df.shape}")
        else:
            print(f"  (skipped existing {var_name}.csv)")


def main():
    """Main entry point."""
    args = parse_args()
    defaults = get_defaults()
    model_names = [item.strip() for item in (args.models or ','.join(defaults['models'])).split(',') if item.strip()]
    scenario_names = [item.strip() for item in (args.scenarios or ','.join(defaults['scenarios'])).split(',') if item.strip()]

    print("\n" + "=" * 60)
    print("FTT Input Data Conversion to New Format")
    print("=" * 60)

    # Validate source folder
    try:
        validate_source_folder('Inputs_existing')
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    print(f"Converting models: {model_names}")
    print(f"Converting scenarios: {scenario_names}")

    # Generate model list from Inputs_existing masterfiles
    source_root = resolve_project_path('Inputs_existing')
    output_root = resolve_project_path('Inputs')
    models = generate_model_list_from_source(source_root, model_names, scenario_names)

    if not models:
        print("✗ No matching masterfiles found in Inputs_existing/_MasterFiles for requested models and scenarios")
        sys.exit(1)

    # Convert masterfiles to new format
    print(f"\nExporting {len(models)} model+scenario combinations from masterfiles...")
    try:
        convert_masterfiles_to_new_format(
            models,
            overwrite=args.overwrite,
            output_root=output_root,
            source_root=source_root,
        )
    except Exception as e:
        print(f"✗ Export failed: {e}")
        sys.exit(1)

    # Convert General CSVs to new format
    try:
        convert_general_to_new_format(output_root=output_root, overwrite=args.overwrite)
    except Exception as e:
        print(f"General conversion failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All conversions completed successfully")
    print("Input data has been converted to new format in 'Inputs/'")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
