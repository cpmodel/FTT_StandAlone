from nicegui import ui

from .shared import shared_layout, save_gui_settings
from .state import state

def render_settings_page():
    shared_layout()

    with ui.column().classes('w-full mx-auto gap-6'):

        with ui.card().classes('w-full p-6 shadow-sm border border-gray-200'):
            with ui.row().classes('items-center gap-2 mb-4'):
                ui.label('Appearance').classes('text-h6')

            with ui.row().classes('items-center'):
                with ui.column().classes('gap-0'):
                    ui.label('Dark mode').classes('text-sm font-medium')
                ui.switch('').bind_value(state, 'dark_mode').on_value_change(lambda _: save_gui_settings())

        with ui.card().classes('w-full p-6 shadow-sm border border-gray-200'):
            with ui.row().classes('items-center gap-2 mb-4'):
                ui.label('Plot settings').classes('text-h6')

            with ui.row().classes('items-center gap-1 mb-3'):
                ui.label('X-axis year range').classes('text-sm font-medium')
                with ui.icon('info_outline').classes('text-gray-400 text-base cursor-default'):
                    ui.tooltip('Controls the visible year range on results charts.')

            with ui.row().classes('gap-4 items-center'):
                ui.number(
                    label='Start year',
                    format='%d',
                    step=1,
                    precision=0,
                ).props('dense outlined').classes('w-36').bind_value(state, 'plot_start_year') \
                    .on_value_change(lambda _: save_gui_settings())
                ui.icon('arrow_forward').classes('text-gray-400')
                ui.number(
                    label='End year',
                    format='%d',
                    step=1,
                    precision=0,
                ).props('dense outlined').classes('w-36').bind_value(state, 'plot_end_year') \
                    .on_value_change(lambda _: save_gui_settings())
