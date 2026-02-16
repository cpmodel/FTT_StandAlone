# -*- coding: utf-8 -*-
"""
Results analysis and plotting engine for FTT results visualization.

Handles:
- Loading pickle files
- Parsing variable metadata
- Extracting and aggregating data
- Generating plotly figures
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from plotly import graph_objects as go
import itertools


class ResultsEngine:
    """Engine for loading and analyzing FTT model results."""
    
    def __init__(self):
        """Initialize the engine with metadata from CSV files."""
        # Load metadata from CSV files
        self.var_listing_df = pd.read_csv('Utilities/titles/VariableListing.csv')
        self.classification_df = pd.read_csv('Utilities/titles/classification_titles.csv', 
                                             header=None, keep_default_na=False, dtype=str)
        
        # Parse classification data into dictionary
        self.classifications = self._parse_classifications()
        
        # Filter variables to only those with TIME dimension
        self.time_variables = self.var_listing_df[self.var_listing_df['Dim4'] == 'TIME'].copy()
        
        # Data containers
        self.loaded_pickles = {}  # {scenario_name: {var_name: np_array}}
        
    def _parse_classifications(self):
        """Parse classification titles CSV into a dictionary."""
        classifications = {}
        for _, row in self.classification_df.iterrows():
            class_code = row[0]
            if class_code and class_code not in ['Models', 'CSCTI']:
                # Get full names (row with "Full name" at index 4)
                if len(row) > 4 and row[4] == 'Full name':
                    values = [v.strip() for v in row[5:] if v and isinstance(v, str) and v.strip()]
                    if values:
                        classifications[class_code] = values
        
        return classifications
    
    def get_available_pickle_files(self):
        """Get list of available pickle files in Output directory."""
        output_dir = Path('Output')
        return sorted([f.name for f in output_dir.glob('*.pickle')]) if output_dir.exists() else []
    
    def load_pickle_files(self, filenames):
        """
        Load multiple pickle files with optional suffix for scenario names.
        
        Args:
            filenames: List of pickle filenames to load
        
        Returns:
            Dictionary of {scenario_name: {var_name: np_array}}
        """
        self.loaded_pickles = {}
        output_dir = Path('Output')
        
        for filename in filenames:
            filepath = output_dir / filename
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
                # Add filename suffix to scenario names if multiple files loaded
                if len(filenames) > 1:
                    suffix = f"_{filename.replace('.pickle', '')}"
                    data = {f"{scen}{suffix}": vars_dict for scen, vars_dict in data.items()}
                
                self.loaded_pickles.update(data)
        
        return self.loaded_pickles
    
    def get_scenario_names(self):
        """Get list of loaded scenario names."""
        return list(self.loaded_pickles.keys())
    
    def get_variable_options(self):
        """Get dictionary of {var_code: var_description}."""
        return {row['Variable name']: f"{row['Variable name']} - {row['Variable description']}" 
                for _, row in self.time_variables.iterrows()}
    
    def get_variable_dimensions(self, variable_name):
        """
        Get dimension info for a variable.
        
        Args:
            variable_name: Name of variable (e.g., 'MWSY')
        
        Returns:
            List of dimension codes: [Dim1, Dim2, Dim3, Dim4]
        """
        matching = self.time_variables[self.time_variables['Variable name'] == variable_name]
        if matching.empty:
            return ['NA', 'NA', 'NA', 'TIME']
        var_row = matching.iloc[0]
        return [var_row['Dim1'], var_row['Dim2'], var_row['Dim3'], var_row['Dim4']]
    
    def get_dimension_values(self, dim_code):
        """
        Get list of available values for a dimension.
        
        Args:
            dim_code: Dimension code (e.g., 'RTI', 'T2TI')
        
        Returns:
            List of dimension values
        """
        return self.classifications.get(dim_code, [])
    
    def extract_and_aggregate(self, variable_name, scenarios, dim_selections, dim_aggregates, 
                             year_range=None, dark_mode=False, result_type="levels", baseline_scenario=None):
        """
        Extract data from loaded pickles and generate plotly figure.
        
        Args:
            variable_name: Name of variable to plot
            scenarios: List of scenario names to include
            dim_selections: Dict mapping dim_index -> selected values
                          e.g., {0: ['Belgium', 'France'], 1: ['Tech1']}
            dim_aggregates: List of booleans indicating which dims to sum over
            year_range: Tuple of (start_year, end_year) or None for all
            dark_mode: Boolean for plotly template
            result_type: "levels" (default), "absolute_diff", or "relative_diff"
            baseline_scenario: Name of baseline scenario (required if result_type != "levels")
        
        Returns:
            plotly Figure object
        """
        if not variable_name or not scenarios or not self.loaded_pickles:
            return go.Figure()
        
        # Get dimension info
        dims = self.get_variable_dimensions(variable_name)
        var_row = self.time_variables[self.time_variables['Variable name'] == variable_name].iloc[0]
        
        # Build figure
        fig = go.Figure()
        
        # Extract baseline data if needed
        baseline_data = None
        if result_type != "levels" and baseline_scenario and baseline_scenario in self.loaded_pickles:
            baseline_data = self._extract_scenario_data(baseline_scenario, variable_name, dims, 
                                                        dim_selections, dim_aggregates)
        
        for idx_scenario, scenario in enumerate(scenarios):
            # Skip baseline scenario when plotting differences (it would show all zeros)
            if result_type != "levels" and scenario == baseline_scenario:
                continue
                
            if scenario not in self.loaded_pickles:
                continue
            
            # Extract scenario data
            scenario_data = self.loaded_pickles[scenario]
            if variable_name not in scenario_data:
                continue
            
            var_data = scenario_data[variable_name]  # Shape: (region, tech, cat, time)
            
            # Ensure var_data has 4 dimensions by inserting size-1 dims where dims[i] == 'NA'
            # Build the target 4D shape based on which dimensions are active
            shape_4d = []
            data_idx = 0
            for dim_name in dims:
                if dim_name == 'NA':
                    shape_4d.append(1)
                else:
                    if data_idx < var_data.ndim:
                        shape_4d.append(var_data.shape[data_idx])
                        data_idx += 1
                    else:
                        shape_4d.append(1)
            var_data = var_data.reshape(shape_4d)
            
            # Get selected indices for each dimension
            indices = []
            for i, dim_name in enumerate(dims):
                if dim_name == 'NA':
                    indices.append([0])
                elif i == 3:
                    # TIME dimension - always use ALL time points (it's the x-axis)
                    max_time = var_data.shape[3] if var_data.ndim > 3 else 1
                    indices.append(list(range(max_time)))
                else:
                    selected_vals = dim_selections.get(i, [])
                    if not selected_vals:
                        # Default to first value
                        dim_values = self.get_dimension_values(dim_name)
                        indices.append([0])
                    else:
                        dim_values = self.get_dimension_values(dim_name)
                        indices.append([dim_values.index(v) for v in selected_vals if v in dim_values])
            
            # Generate combinations for plotting
            combinations = self._generate_combinations(indices, dim_aggregates)
            
            for combo in combinations:
                # Extract and aggregate data
                try:
                    data_slice = var_data.copy()
                    
                    # Apply slicing/aggregation for dims 0-2
                    for dim_idx in range(min(3, len(combo))):
                        # Select specific indices for this dimension first
                        if isinstance(combo[dim_idx], list):
                            idx_list = combo[dim_idx]
                        else:
                            idx_list = [combo[dim_idx]]

                        if dim_idx < data_slice.ndim:
                            data_slice = np.take(data_slice, idx_list, axis=dim_idx)

                            # If aggregating, sum only across the selected indices
                            if dim_idx < len(dim_aggregates) and dim_aggregates[dim_idx]:
                                data_slice = np.sum(data_slice, axis=dim_idx, keepdims=True)
                    
                    # Extract time series (dim 3) - take all time indices or selected ones
                    if data_slice.ndim > 3:
                        time_indices = indices[3] if len(indices) > 3 else range(data_slice.shape[3])
                        data_slice = np.take(data_slice, time_indices, axis=3)
                    
                    # Flatten to get 1D array
                    y_values = data_slice.flatten()
                    
                    # Apply result type transformations
                    if result_type == "absolute_diff" and baseline_data is not None:
                        y_values = self._calculate_absolute_diff(y_values, baseline_data, idx_scenario, combo)
                    elif result_type == "relative_diff" and baseline_data is not None:
                        y_values = self._calculate_relative_diff(y_values, baseline_data, idx_scenario, combo)
                    
                    # Generate label
                    label_parts = [scenario]
                    for i in range(3):
                        if not dim_aggregates[i]:
                            dim_name = dims[i]
                            if dim_name != 'NA':
                                val_idx = combo[i][0] if isinstance(combo[i], list) else combo[i]
                                dim_vals = self.get_dimension_values(dim_name)
                                if val_idx < len(dim_vals):
                                    label_parts.append(dim_vals[val_idx][:15])
                    
                    trace_label = ' '.join(label_parts)
                    
                    # Generate x-axis (years)
                    # Default: start at 2010
                    x_values = list(range(2010, 2010 + len(y_values)))
                    
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='lines',
                        name=trace_label
                    ))
                except Exception as e:
                    print(f"ERROR processing combo {combo}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Update layout with appropriate y-axis title
        template = 'plotly_dark' if dark_mode else 'simple_white'
        
        if result_type == "absolute_diff":
            y_title = f"Abs. diff from baseline ({var_row['Unit']})"
        elif result_type == "relative_diff":
            y_title = "Rel. diff from baseline (%)"
        else:
            y_title = f"{var_row['Unit']}"
        
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=40),
            template=template,
            showlegend=True,
            xaxis_title='Year',
            yaxis_title=y_title,
            yaxis=dict(title=dict(text=y_title, standoff=10, font=dict(size=14))),
            title=dict(text=f"{variable_name} - {var_row['Variable description']}", font=dict(size=18))
        )
        
        # Add hover template with 3 decimal places
        # For relative differences, append % sign to the value
        for trace in fig.data:
            if result_type == "relative_diff":
                trace.hovertemplate = '<b>%{x}</b><br><b>%{fullData.name}</b><br>Value: %{y:.3f}%<extra></extra>'
            else:
                trace.hovertemplate = '<b>%{x}</b><br><b>%{fullData.name}</b><br>Value: %{y:.3f}<extra></extra>'
    
        return fig
    
    def _generate_combinations(self, indices_list, aggregate_flags):
        """
        Generate all combinations of dimension indices for plotting.
        
        For aggregated dimensions, all indices are included (to be summed).
        For non-aggregated dimensions, each index is separate (one line per combo).
        """
        combos = []
        for i, (idx_list, agg) in enumerate(zip(indices_list[:3], aggregate_flags[:3])):
            if agg:
                # Single combo that includes all selected indices for aggregation
                combos.append([idx_list])
            else:
                combos.append([[idx] for idx in idx_list])  # Individual values
        
        # Generate all combinations
        results = []
        for combo in itertools.product(*combos):
            results.append(combo)
        return results
    
    def _extract_scenario_data(self, scenario, variable_name, dims, dim_selections, dim_aggregates):
        """
        Extract all data combinations for a scenario to serve as baseline.
        
        Returns a dictionary mapping combo -> y_values for efficient lookup.
        """
        scenario_data = self.loaded_pickles[scenario]
        if variable_name not in scenario_data:
            return {}
        
        var_data = scenario_data[variable_name]
        
        # Ensure var_data has 4 dimensions by inserting size-1 dims where dims[i] == 'NA'
        # Build the target 4D shape based on which dimensions are active
        shape_4d = []
        data_idx = 0
        for dim_name in dims:
            if dim_name == 'NA':
                shape_4d.append(1)
            else:
                if data_idx < var_data.ndim:
                    shape_4d.append(var_data.shape[data_idx])
                    data_idx += 1
                else:
                    shape_4d.append(1)
        var_data = var_data.reshape(shape_4d)
        
        # Get selected indices for each dimension
        indices = []
        for i, dim_name in enumerate(dims):
            if dim_name == 'NA':
                indices.append([0])
            elif i == 3:
                max_time = var_data.shape[3] if var_data.ndim > 3 else 1
                indices.append(list(range(max_time)))
            else:
                selected_vals = dim_selections.get(i, [])
                if not selected_vals:
                    dim_values = self.get_dimension_values(dim_name)
                    indices.append([0])
                else:
                    dim_values = self.get_dimension_values(dim_name)
                    indices.append([dim_values.index(v) for v in selected_vals if v in dim_values])
        
        # Generate combinations and extract data
        combinations = self._generate_combinations(indices, dim_aggregates)
        baseline_dict = {}
        
        for combo in combinations:
            try:
                data_slice = var_data.copy()
                
                # Apply slicing/aggregation for dims 0-2
                for dim_idx in range(min(3, len(combo))):
                    if isinstance(combo[dim_idx], list):
                        idx_list = combo[dim_idx]
                    else:
                        idx_list = [combo[dim_idx]]

                    if dim_idx < data_slice.ndim:
                        data_slice = np.take(data_slice, idx_list, axis=dim_idx)

                        if dim_idx < len(dim_aggregates) and dim_aggregates[dim_idx]:
                            data_slice = np.sum(data_slice, axis=dim_idx, keepdims=True)
                
                # Extract time series
                if data_slice.ndim > 3:
                    time_indices = indices[3] if len(indices) > 3 else range(data_slice.shape[3])
                    data_slice = np.take(data_slice, time_indices, axis=3)
                
                y_values = data_slice.flatten()
                baseline_dict[str(combo)] = y_values
            except Exception as e:
                print(f"ERROR extracting baseline combo {combo}: {str(e)}")
                continue
        
        return baseline_dict
    
    def _calculate_absolute_diff(self, scenario_values, baseline_dict, scenario_idx, combo):
        """
        Calculate absolute difference (scenario - baseline).
        
        Args:
            scenario_values: 1D array of scenario data
            baseline_dict: Dictionary of baseline data from _extract_scenario_data
            scenario_idx: Index of current scenario (used to identify which combo)
            combo: Current combo tuple
        
        Returns:
            1D array of absolute differences
        """
        combo_key = str(combo)
        if combo_key in baseline_dict:
            baseline_values = baseline_dict[combo_key]
            if len(scenario_values) == len(baseline_values):
                return scenario_values - baseline_values
        
        return scenario_values
    
    def _calculate_relative_diff(self, scenario_values, baseline_dict, scenario_idx, combo):
        """
        Calculate relative difference as percentage: ((scenario - baseline) / |baseline|) * 100
        
        Handles division by zero by returning 0 when baseline is 0.
        
        Args:
            scenario_values: 1D array of scenario data
            baseline_dict: Dictionary of baseline data from _extract_scenario_data
            scenario_idx: Index of current scenario (used to identify which combo)
            combo: Current combo tuple
        
        Returns:
            1D array of relative differences (in percentage)
        """
        combo_key = str(combo)
        if combo_key in baseline_dict:
            baseline_values = baseline_dict[combo_key]
            if len(scenario_values) == len(baseline_values):
                # Handle division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_diff = ((scenario_values - baseline_values) / np.abs(baseline_values)) * 100
                    # Replace NaN and Inf with 0 where baseline is 0
                    rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
                return rel_diff
        
        return scenario_values
