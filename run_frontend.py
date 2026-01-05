from nicegui import ui

from GUI.page_run import render_run_page
from GUI.page_results import render_results_page
from GUI.page_figures import render_figures_page
from GUI.page_settings import render_settings_page

# Define the routes
@ui.page('/', title='Run | FTT')
def run_page():
    render_run_page()

@ui.page('/results', title='Results | FTT')
def results_page():
    render_results_page()

@ui.page('/figures', title='Figures | FTT')
def figures_page():
    render_figures_page()

@ui.page('/settings', title='Settings | FTT')
def settings_page():
    render_settings_page()

ui.run(title="FTT", port=8080, favicon='GUI/images/ftt_favicon.png')