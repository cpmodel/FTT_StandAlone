from nicegui import ui
from pathlib import Path

from .shared import shared_layout
from .state import state
from .results_engine import ResultsEngine

def render_results_page():
        # Lazy import plotly only when results page is actually rendered
        from plotly import graph_objects as go
        import plotly.io as pio
        
        shared_layout()
        
        # Initialize results engine
        engine = ResultsEngine()

        with ui.column().classes('w-full h-[calc(100vh-8rem)] flex no-wrap items-start'):
            
            # SECTION 1. Figure
            with ui.card().classes('w-full h-1/2 shadow-sm border p-0 border-gray-200'):
                plt_tem = 'plotly_dark' if state.dark_mode else 'simple_white'
                fig = go.Figure()
                fig.update_layout(
                    margin=dict(l=40, r=20, t=40, b=40),
                    template=plt_tem,
                    showlegend=True,
                    xaxis_title='Year',
                    yaxis_title='Value'
                )
                plot = ui.plotly(fig).classes('w-full h-full p-0')

            # SECTION 2. Data selector
            with ui.card().classes('w-full h-1/2 p-3 overflow-y-auto shadow-sm border border-gray-200'):
                with ui.row().classes('w-full h-full'):
                    # set tabs
                    if state.dark_mode:
                        tab_style = 'vertical indicator-color=blue-400 active-color=white active-bg-color=gray-700'
                    else:
                        tab_style = 'vertical indicator-color=blue-400 active-color=black active-bg-color=gray-200'
                    with ui.tabs().props(tab_style).classes('border-r') as tabs:
                        t1 = ui.tab('Load Files').classes('h-1/2').props('icon=folder_open')
                        t2 = ui.tab('Analysis').classes('h-1/2').props('icon=bar_chart')

                    with ui.tab_panels(tabs, value=t1).classes('flex-grow h-full'):
                
                        # TAB 1: FILE PICKER
                        with ui.tab_panel(t1).classes('w-full h-full p-3 overflow-auto'):
                            with ui.row().classes('w-full h-full gap-4 items-stretch'):
                                # Left side: Pickle file section
                                with ui.column().classes('flex-1 items-start'):
                                    ui.label('Select pickle files to load').classes('text-sm font-semibold')
                                    
                                    # Get available pickle files from engine
                                    pickle_files = engine.get_available_pickle_files()
                                    
                                    with ui.row().classes('w-full'):
                                        
                                        file_picker = ui.select(
                                            options=pickle_files,
                                            label='Available Files',
                                            multiple=True,
                                            with_input=True
                                        ).classes('w-2/3')
                                        
                                        def load_pickles():
                                            if file_picker.value:
                                                engine.load_pickle_files(file_picker.value)
                                                # Update scenario selector options
                                                scenarios = engine.get_scenario_names()
                                                scenario_selector.options = scenarios
                                                # Auto-select all scenarios
                                                scenario_selector.value = scenarios
                                                scenario_selector.update()
                                                
                                                # Update models display
                                                models_info = engine.get_models_run()
                                                models_container.clear()
                                                
                                                # Define colors for each model
                                                model_colors = {
                                                    'FTT-P': 'blue',
                                                    'FTT-H': 'orange',
                                                    'FTT-Tr': 'green',
                                                    'FTT-Fr': 'purple'
                                                }
                                                
                                                with models_container:
                                                    # Group scenarios by models to show more compactly
                                                    models_by_scenario = {}
                                                    for scenario, models in sorted(models_info.items()):
                                                        models_key = tuple(sorted(models)) if models else ()
                                                        if models_key not in models_by_scenario:
                                                            models_by_scenario[models_key] = []
                                                        models_by_scenario[models_key].append(scenario)
                                                    # Display each unique model combination
                                                    with ui.column().classes('gap-1 mb-1'):
                                                        ui.label('Model results loaded').classes('text-sm font-semibold gap-2')
                                                    for models_tuple, scen_list in sorted(models_by_scenario.items(), 
                                                                                        key=lambda x: (-len(x[0]), x[0])):
                                                        # Model badges
                                                        if models_tuple:
                                                            with ui.row().classes('w-full'):

                                                                for model in models_tuple:
                                                                    color = model_colors.get(model, 'gray')
                                                                    ui.badge(model).props(f'color={color}').classes('text-body1')
                                                                ui.label(scen_list)
                                                        else:
                                                            ui.badge('No models').props('color=red').classes('text-body1')
                                                        
                                                        # Scenario names
                                                        # ui.label(', '.join(scen_list)).classes('text-xs text-gray-600')
                                                
                                                ui.notify(f'Loaded {len(scenarios)} scenarios', type='positive')
                                        
                                        ui.button('Load Selected', on_click=load_pickles).classes('w-1/4 h-10 text-sm px-2')
                                    
                                    # Models run display
                                    models_container = ui.row().classes('w-full gap-2')

                                # Right side: Scenario section
                                with ui.column().classes('flex-1 items-start'):
                                    ui.label('Select scenarios to plot').classes('text-sm font-semibold')
                                    
                                    scenario_selector = ui.select(
                                        options=[],
                                        label='Scenarios',
                                        multiple=True,
                                        with_input=True
                                    ).classes('w-full').bind_value(state, 'selected_scenarios')
                                    
                                    baseline_selector = ui.select(
                                        options=[],
                                        label='Baseline (optional)',
                                        with_input=True,
                                        clearable=True
                                    ).classes('w-full').bind_value(state, 'selected_baseline')
                                    
                                    # Update baseline options when scenarios change
                                    def update_baseline_options():
                                        baseline_selector.options = state.selected_scenarios
                                        baseline_selector.update()
                                        update_result_type_availability()
                                    
                                    scenario_selector.on_value_change(update_baseline_options)
                                    
                                    # Update availability and plot when baseline changes
                                    def on_baseline_change(e):
                                        update_result_type_availability()
                                        update_plot()
                                    
                                    baseline_selector.on_value_change(on_baseline_change)

                        # Define update_result_type_availability function that will be used in Analysis tab
                        def update_result_type_availability():
                            """
                            Enable/disable result type options based on available scenarios and baseline.
                            Differences from baseline require:
                            - At least 2 scenarios loaded
                            - A baseline scenario selected (not None)
                            """
                            has_baseline = state.selected_baseline is not None and state.selected_baseline != ''
                            sufficient_scenarios = len(engine.get_scenario_names()) >= 2
                            
                            can_show_diffs = has_baseline and sufficient_scenarios
                            
                            # If differences aren't available but user selected them, revert to 'Levels'
                            if not can_show_diffs and state.result_type != 'levels':
                                state.result_type = 'levels'
                                # Update selector value if it exists
                                if result_type_selector is not None:
                                    result_type_selector.value = 'Levels'
                            
                        with ui.tab_panel(t2).classes('w-full items-start p-2'):
                            with ui.column().classes('w-full items-start no-scroll gap-3'):
                                # Variable selector row with download button
                                ui.label('Select variable').classes('text-xs p-0 justify-left font-semibold').props('dense')
                                with ui.row().classes('w-full gap-5 items-center'):
                                    var_options = engine.get_variable_options()
                                    variable_selector = ui.select(
                                        options=var_options,
                                        label='Variable',
                                        with_input=True
                                    ).classes('flex-grow gap-1').props('dense')
                                    
                                    def on_variable_change(e):
                                        state.selected_variable = e.value
                                        update_dimensions()
                                        update_plot()
                                    
                                    variable_selector.on_value_change(on_variable_change)
                                    
                                    # Download CSV button
                                    def download_csv():
                                        import io
                                        import pandas as pd
                                        from datetime import datetime
                                        
                                        if not plot.figure or not plot.figure.data:
                                            ui.notify('No data to download', type='warning')
                                            return
                                        
                                        # Extract data from plotly figure
                                        data_dict = {'Year': []}
                                        for trace in plot.figure.data:
                                            data_dict[trace.name] = []
                                        
                                        # Get x values (years) from first trace
                                        if len(plot.figure.data) > 0:
                                            x_values = list(plot.figure.data[0].x)
                                            data_dict['Year'] = x_values
                                            
                                            # Get y values for each trace
                                            for trace in plot.figure.data:
                                                data_dict[trace.name] = list(trace.y)
                                        
                                        # Create DataFrame and CSV
                                        df = pd.DataFrame(data_dict)
                                        # convert to longform
                                        df_long = df.melt(id_vars=['Year'], var_name='Scenario', value_name='Value')
                                        csv_buffer = io.StringIO()
                                        df_long.to_csv(csv_buffer, index=False)
                                        csv_content = csv_buffer.getvalue()
                                        
                                        # Generate filename
                                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                        var_name = state.selected_variable if state.selected_variable else 'data'
                                        filename = f"{var_name}_{timestamp}.csv"
                                        
                                        # Trigger download
                                        ui.download(csv_content.encode(), filename)
                                        ui.notify(f'Downloaded {filename}', type='positive')
                                    
                                    ui.button('Download data', on_click=download_csv, icon='download').classes('h-full text-xs')

                                # Dimension selectors (horizontal layout)
                                dimension_container = ui.row().classes('w-[calc(100vw-14rem)] h-full gap-3 overflow-y-auto flex-nowrap')
                        
                        # Track result_type_selector reference (will be recreated in update_dimensions)
                        result_type_selector = None
                        
                        # Define result type options and handler outside of update_dimensions
                        # so they persist across dimension rebuilds
                        result_type_options = {
                            'Levels': 'levels',
                            'Absolute difference from baseline': 'absolute_diff',
                            'Relative difference from baseline': 'relative_diff'
                        }
                        
                        def on_result_type_change(e):
                            # Convert display label to internal value
                            state.result_type = result_type_options.get(e.value, 'levels')
                            # Reset to 'levels' if conditions aren't met
                            update_result_type_availability()
                            update_plot()
                            
        # Function to update dimension selectors when variable changes
        def update_dimensions():
            nonlocal result_type_selector
            dimension_container.clear()

            # If no variable selected yet, show empty selectors (disabled)
            if not state.selected_variable:
                dims = ['Dim1', 'Dim2', 'Dim3']
                for i, dim_name in enumerate(dims):
                    with dimension_container:
                        with ui.column().classes('w-full h-full gap-1'):
                            ui.label(f'{dim_name}').classes('text-xs font-semibold')

                            dim_select = ui.select(
                                options=[],
                                label=f'Select {dim_name}',
                                multiple=True,
                                with_input=True,
                                clearable=True
                            ).classes('w-full h-full overflow-auto').props('dense use-chips options-dense').disable()
                            with ui.row().classes('w-full gap-2'):
                                ui.checkbox('All').props('dense').disable()
                                ui.checkbox('Sum').props('dense').disable()
                
                # Add result type selector to the right of dimension selectors
                with dimension_container:
                    with ui.column().classes('w-full h-full no-scroll gap-1 border-l border-gray-300 pl-4'):
                        ui.label('Transformation').classes('text-xs font-semibold')
                        result_type_selector = ui.select(
                            options=list(result_type_options.keys()),
                            value='Levels',
                            label='Select Transformation',
                            with_input=False
                        ).classes('w-full h-full overflow-auto').props('dense').disable()
                return

            # Get dimension info for selected variable
            dims = engine.get_variable_dimensions(state.selected_variable)

            for i, dim_name in enumerate(dims):
                # Skip NA and TIME (dim 4) selectors
                if dim_name == 'NA' or i == 3 or isinstance(dim_name, float):
                    continue

                # Get available values for this dimension
                dim_values = engine.get_dimension_values(dim_name)

                with dimension_container:
                    with ui.column().classes('w-full h-full gap-1'):
                        dim_labels = ['Region', 'Technology', 'Category', 'Time']
                        ui.label(f'{dim_name} - {dim_labels[i]}').classes('text-xs font-semibold')
                        # ui.label(f'{dim_name}').classes('text-xs font-semibold')

                        # Multi-select for dimension values
                        dim_select = ui.select(
                            options=dim_values,
                            label=f'Select {dim_name}',
                            multiple=True,
                            with_input=True,
                            clearable=True
                        ).classes('w-full h-full overflow-auto').props('dense use-chips options-dense')

                        # Initialize with first value if not set
                        if dim_values and f'dim{i}_values' not in state.dim_selections[i]:
                            dim_select.value = [dim_values[0]]
                        elif f'dim{i}_values' in state.dim_selections[i]:
                            dim_select.value = state.dim_selections[i][f'dim{i}_values']

                        def make_select_handler(idx):
                            def handler(e):
                                state.dim_selections[idx][f'dim{idx}_values'] = e.sender.value
                                update_plot()
                            return handler

                        dim_select.on_value_change(make_select_handler(i))

                        # Checkboxes (Select All and Sum)
                        with ui.row().classes('w-full gap-4'):
                            # Select All checkbox
                            select_all = ui.checkbox('All').props('dense')
                            def make_select_all_handler(idx, selector, vals):
                                def handler(e):
                                    if e.value:
                                        selector.value = vals
                                    else:
                                        selector.value = []
                                return handler

                            select_all.on_value_change(make_select_all_handler(i, dim_select, dim_values))

                            # Aggregate checkbox
                            agg_check = ui.checkbox('Sum').props('dense')
                            agg_check.value = state.dim_aggregate[i]
                            def make_agg_handler(idx):
                                def handler(e):
                                    state.dim_aggregate[idx] = e.value
                                    update_plot()
                                return handler
                            agg_check.on_value_change(make_agg_handler(i))
            
            # Add result type selector to the right of dimension selectors
            with dimension_container:
                with ui.column().classes('w-full h-full no-scroll gap-1 border-l border-gray-300 pl-4'):
                    ui.label('Transformation').classes('text-xs font-semibold')
                    # Get current value from state to preserve selection
                    current_value = 'Levels'
                    for label, val in result_type_options.items():
                        if val == state.result_type:
                            current_value = label
                            break
                    
                    result_type_selector = ui.select(
                        options=list(result_type_options.keys()),
                        value=current_value,
                        label='Select Transformation',
                        with_input=False
                    ).classes('w-full h-full overflow-none').props('dense')
                    result_type_selector.on_value_change(on_result_type_change)
        
        # Function to update plot
        def update_plot():
            if not state.selected_variable or not state.selected_scenarios or not engine.get_scenario_names():
                return

            # Prepare dimension selections dictionary
            dim_selections = {}
            for i in range(4):
                if f'dim{i}_values' in state.dim_selections[i]:
                    dim_selections[i] = state.dim_selections[i][f'dim{i}_values']

            # Generate figure using engine
            fig = engine.extract_and_aggregate(
                variable_name=state.selected_variable,
                scenarios=state.selected_scenarios,
                dim_selections=dim_selections,
                dim_aggregates=state.dim_aggregate,
                dark_mode=state.dark_mode,
                result_type=state.result_type,
                baseline_scenario=state.selected_baseline
            )

            plot.update_figure(fig)
        
        # Initial render of dimension selectors
        update_dimensions()
        
        # Initialize result type availability
        update_result_type_availability()

        # Bind scenario selector update
        scenario_selector.on_value_change(update_plot)