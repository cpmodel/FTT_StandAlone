from nicegui import ui
from plotly import graph_objects as go

from .shared import shared_layout
from .state import state

def render_results_page():
        shared_layout()

        with ui.column().classes('w-full no-wrap gap-3 items-start'):
            
            # SECTION 1. Figure
            with ui.card().classes('w-full p-4 shadow-sm border border-gray-200'):
                # determine whether dark mode enabled for plotting template
                plt_tem = 'plotly_dark' if state.dark_mode else 'plotly'
                fig = go.Figure(go.Scatter(x=[1, 2, 3, 4], y=[1, 2, 3, 2.5]))
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), template = plt_tem)
                plot = ui.plotly(fig).classes('w-full h-[300px]')

            # SECTION 2. Data selector
            with ui.card().classes('w-full h-[190px] p-4 shadow-sm border border-gray-200'):
                with ui.row().classes('w-full h-full'):
                    # set tabs
                    with ui.tabs().props('vertical stretch=true indicator-color=blue-400').classes('h-full justify-center p-2 border-r') as tabs:
                        t1 = ui.tab('1. Load Files')
                        t2 = ui.tab('2. Scenarios')
                        t3 = ui.tab('3. Variables')

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