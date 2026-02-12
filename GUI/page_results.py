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
                    with ui.tabs().props(tab_style).classes('p-1 h-full border-r') as tabs:
                        t1 = ui.tab('Load Files').classes('h-1/2').props('icon=folder_open')
                        t2 = ui.tab('Analysis').classes('h-1/2').props('icon=bar_chart')

                    with ui.tab_panels(tabs, value=t1).classes('flex-grow h-full'):
                
                        # TAB 1: FILE PICKER
                        with ui.tab_panel(t1).classes('w-full h-full p-3 overflow-auto'):
                            with ui.row().classes('w-full h-full gap-4 items-center'):
                                # Left side: Pickle file section
                                with ui.column().classes('flex-1 items-start'):
                                    ui.label('Select pickle files to load').classes('text-sm font-semibold')
                                    
                                    # Get available pickle files from engine
                                    pickle_files = engine.get_available_pickle_files()
                                    
                                    file_picker = ui.select(
                                        options=pickle_files,
                                        label='Available Files',
                                        multiple=True,
                                        with_input=True
                                    ).classes('w-full')
                                    
                                    def load_pickles():
                                        if file_picker.value:
                                            engine.load_pickle_files(file_picker.value)
                                            # Update scenario selector options
                                            scenarios = engine.get_scenario_names()
                                            scenario_selector.options = scenarios
                                            # Auto-select all scenarios
                                            scenario_selector.value = scenarios
                                            scenario_selector.update()
                                            ui.notify(f'Loaded {len(scenarios)} scenarios', type='positive')
                                    
                                    ui.button('Load Selected', on_click=load_pickles).classes('w-50 h-10 items-center')

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
                                    ).classes('w-full')
                                    
                                    # Update baseline options when scenarios change
                                    def update_baseline_options():
                                        baseline_selector.options = state.selected_scenarios
                                        baseline_selector.update()
                                    
                                    scenario_selector.on_value_change(update_baseline_options)

                        # TAB 2: VARIABLE & DIMENSIONS EXPLORER
                        with ui.tab_panel(t2).classes('w-full items-start overflow-auto p-2'):
                            with ui.column().classes('w-full items-start no-scroll gap-3'):
                                # Variable selector
                                ui.label('Select variable').classes('text-xs p-0 justify-left font-semibold').props('dense')
                                var_options = engine.get_variable_options()
                                variable_selector = ui.select(
                                    options=var_options,
                                    label='Variable',
                                    with_input=True
                                ).classes('w-full gap-1').props('dense')
                                
                                def on_variable_change(e):
                                    state.selected_variable = e.value
                                    update_dimensions()
                                    update_plot()
                                
                                variable_selector.on_value_change(on_variable_change)

                                # Dimension selectors (horizontal layout)
                                dimension_container = ui.row().classes('w-[calc(75vw-12rem)] h-full gap-3 overflow-y-auto flex-nowrap')
                            
        # Function to update dimension selectors when variable changes
        def update_dimensions():
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
                return

            # Get dimension info for selected variable
            dims = engine.get_variable_dimensions(state.selected_variable)

            for i, dim_name in enumerate(dims):
                # Skip NA and TIME (dim 4) selectors
                if dim_name == 'NA' or i == 3:
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
                dark_mode=state.dark_mode
            )

            plot.update_figure(fig)
        
        # Initial render of dimension selectors
        update_dimensions()

        # Bind scenario selector update
        scenario_selector.on_value_change(update_plot)