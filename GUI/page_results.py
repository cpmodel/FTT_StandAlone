from nicegui import ui
from plotly import graph_objects as go
import plotly.io as pio

from .shared import shared_layout
from .state import state

def render_results_page():
        shared_layout()

        with ui.column().classes('w-full no-wrap gap-3 items-start'):
            
            # SECTION 1. Figure
            with ui.card().classes('w-full p-4 shadow-sm border border-gray-200'):
                # determine whether dark mode enabled for plotting template
                plt_tem = 'plotly_dark' if state.dark_mode else 'simple_white'
                fig = go.Figure(go.Scatter(name='1 Belgium', x=[2020, 2030, 2040, 2050], y=[1, 2, 3, 2.5]))
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), template = plt_tem, showlegend=True)
                plot = ui.plotly(fig).classes('w-full h-[250px]')

                print(list(pio.templates))

            # SECTION 2. Data selector
            with ui.card().classes('w-full h-[240px] p-4 shadow-sm border border-gray-200'):
                with ui.row().classes('w-full h-full'):
                    # set tabs
                    with ui.tabs().props('vertical stretch=true indicator-color=blue-400 dense=true').classes('h-full justify-center p-1 border-r') as tabs:
                        t1 = ui.tab('Load Files')
                        t2 = ui.tab('Scenarios')
                        t3 = ui.tab('Variables')
                        t4 = ui.tab('Chart Settings')


                    with ui.tab_panels(tabs, value=t1).classes('flex-grow h-full'):
                
                        # TAB 1: FILE PICKER
                        with ui.tab_panel(t1).classes('w-full h-full items-center justify-center'):
                            with ui.row().classes('items-center'):
                                ui.select(options=["a","b"], label = "Select Pickle files").classes('w-64')
                                ui.button('Load')

                        # TAB 2: SCENARIO PICKER
                        with ui.tab_panel(t2).classes('w-full h-full items-center justify-center'):
                            with ui.row().classes('items-center'):
                                ui.select(options=["a","b"], label = "Select Scenarios").classes('w-64')
                                ui.select(options=["a","b"], label = "Select Baseline").classes('w-64')

                        # TAB 3: VARIABLE EXPOLORER
                        with ui.tab_panel(t3).classes('w-full h-full items-center justify-center'):
                            with ui.row().classes('items-center'):
                                ui.select(options=["a","b"], label = "Variable").classes('w-50')
                                ui.select(options=["a","b"], label = "Dim 1").classes('w-50')
                                ui.select(options=["a","b"], label = "Dim 2").classes('w-50')
                                ui.select(options=["a","b"], label = "Dim 3").classes('w-50')

                        with ui.tab_panel(t4):
                            with ui.row():
                                grouped_options = [
                                    {
                                        'label': 'Fruits', # This is the header for the group
                                        'children': [
                                            {'label': 'Apple', 'value': 'apple'},
                                            {'label': 'Banana', 'value': 'banana'},
                                        ]
                                    },
                                    {
                                        'label': 'Vegetables',
                                        'children': [
                                            {'label': 'Carrot', 'value': 'carrot'},
                                            {'label': 'Potato', 'value': 'potato'},
                                        ]
                                    }
                                ]

                                ui.select(
                                    options=grouped_options, 
                                    with_input=True, 
                                    label='Search Categories'
                                ).classes('w-64').props('emit-value map-options')