from nicegui import ui

from .shared import shared_layout

def render_figures_page():
        shared_layout()

        with ui.row().classes('w-full no-wrap gap-6 items-start'):

            with ui.card().classes('w-full p-6 shadow-sm border border-gray-200'):
                
                ui.label('Generate Figures').classes('text-h6 mb-2')
                ui.label('Coming soon...').classes('text-body1')