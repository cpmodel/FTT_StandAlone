from nicegui import ui
from .state import state

def shared_layout():
    """
    Function for rendering the heading and footer part of the ui that remains
    the same between pages.
    """

    ui.dark_mode().bind_value(state, 'dark_mode')
    
    current_path = ui.context.client.page.path
    with ui.header().classes('bg-slate-800 items-center justify-between'):
        with ui.row().classes('items-center'):
            ui.image('GUI/images/ftt_icon_white.png').classes('w-18 h-12 ml-4')
            # Use a column to stack the Title and Subtitle
            with ui.column().classes('gap-0 ml-2'):
                ui.label('Future Technology Transformations').classes('font-bold text-2xl ml-2')
                ui.label('Model frontend').classes('text-xs text-slate-400 tracking-wider ml-2')
            
        with ui.row().classes('gap-1 mr-4'):
            # If adding a new page, the button name, route, and icon 
            # (if desired) should be added here
            for label, path, icon in [
                ('Run', '/', 'play_circle'),
                ('Results', '/results', 'insights'),
                ('Figures', '/figures', 'bar_chart'),
                ('Settings', '/settings', 'settings')
                ]:
                color = 'blue-300' if current_path == path else 'white'
                ui.button(label, icon=icon, 
                on_click=lambda p=path: ui.navigate.to(p)).props(f'flat color={color}') \
                    .bind_enabled_from(state, 'is_running', backward=lambda r: not r)
    
    with ui.footer(fixed=False).classes('bg-slate-800 p-4 items-center justify-between'):
        ui.label('Copyright © 2026 Climate Policy Assessment Community of Models').classes('text-gray-400 text-xs')

        with ui.row().classes('ml-4'):
            for label, url in[
                ('Documentation', 'https://ftt-standalone.readthedocs.io/en/latest/'),
                ('Exeter Climate Policy', 'https://exeterclimatepolicy.com/'),
                ('GitHub', 'https://github.com/cpmodel/FTT_StandAlone')
                ]:
                color = 'blue-300' if current_path == path else 'white'
                ui.link(label, url, new_tab=True).classes('text-gray-400 text-xs')
                # format with seperator but ignore on last url entry (should find more elegant solution)
                if label != 'GitHub':
                    ui.label(' | ').classes('text-gray-400 text-xs')
