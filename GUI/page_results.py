from nicegui import ui

from .shared import shared_layout


def render_results_page():
        shared_layout()

        with ui.column().classes('w-full q-pa-lg gap-4'):
            with ui.row().classes('w-full no-wrap gap-6 items-start'):

                with ui.card().classes('w-full p-6 shadow-sm border border-gray-200'):
                    ui.label('Results Explorer').classes('text-h6 mb-2')