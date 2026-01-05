from nicegui import ui, run
import pickle
import configparser
from pathlib import Path

from .shared import shared_layout
from .state import state
from SourceCode.model_class import ModelRun

def render_run_page():
    # Add shared layout (header, footer, etc.)
    shared_layout()

    # Add row for the two columns
    with ui.row().classes('items-stretch w-full no-wrap gap-6'):
        # Left col (2/3 width)
        with ui.column().classes('w-2/3'):
            with ui.card().classes('w-full h-full p-6 shadow-sm border border-gray-200'):
                ui.label('Model Configuration').classes('text-h6 mb-2')
                
                # Layout grid for inputs inside the card to look cleaner
                with ui.grid(columns=2).classes('w-full gap-4'):
                    available_models = ('FTT-P,FTT-Tr,FTT-H,FTT-Fr').split(',')
                    model_select = ui.select(available_models, label='Model(s) to run', multiple=True)
                    model_select.value = [available_models[0]]

                    available_scenarios = ('S0,S1,S2').split(',')
                    scenario_select = ui.select(available_scenarios, label='Scenario(s) to run', multiple=True)
                    scenario_select.value = [available_scenarios[0]]

                    default_horizon = 2050
                    horizon = ui.number(label = 'End year', value=default_horizon,
                                        validation = {'Enter a valid year between 2020 and 2050.': 
                                                    lambda value: int(value) >= 2020 and
                                                        int(value) <= 2050 and
                                                        len(str(value)) > 0})

                    output_name = ui.input(label='Output name', value='Results',
                                            validation = {'Enter a valid name.':
                                                            lambda value: len(str(value)) > 0})

                ui.separator().classes('my-4')
                with ui.row().classes('w-full justify-center'):
                    run_btn = ui.button('Run', on_click=lambda: start_run()).classes('w-30 h-14 text-lg')

        # Right col (1/3 width)
        with ui.column().classes('w-1/3'):
            with ui.card().classes('w-full p-6 shadow-sm border border-gray-200'):
                ui.label('Run Progress').classes('text-h6 mb-2')

                progress_bar = ui.linear_progress(value=0).classes('w-full mb-4').props('track-color=gray-600')

                log_console = ui.log(max_lines=15).classes('w-full h-64 bg-slate-900 text-green-400 text-xs font-mono p-2 rounded')

    # --- Run Logic ---
    async def start_run():
        # Disable run button and set running state (disables navigation buttons)
        run_btn.disable()
        state.is_running = True
        progress_bar.set_value(0.1)
        
        try:
            # Capture values
            models_value = model_select.value
            end_year_value = int(horizon.value)
            scenarios_value = scenario_select.value
            output_value = output_name.value

            # Execute model (IO bound thread)
            log_console.push(f"Running models: {', '.join(models_value)}")
            await run.io_bound(execute_model, models_value, end_year_value,
                                scenarios_value, output_value)
            
            progress_bar.set_value(1.0)
            log_console.push("SUCCESS: Results written to pickle.")
            ui.notify('Run Complete', type='positive')
            
        except Exception as e:
            log_console.push(f"CRITICAL ERROR: {str(e)}")
            ui.notify('Error', type='negative')
        finally:
            run_btn.enable()
            state.is_running = False

def execute_model(models, end_year, scenarios, output_name):
    """
    Function to update settings.ini with front end selections,
    call model run, and write output to pickle file.

    """
    # First, set settings.ini with inputted values
    config = configparser.ConfigParser()
    config.read('settings.ini')
    config.set('settings', 'enable_modules',", ".join(models))
    config.set('settings', 'simulation_end', str(end_year))
    config.set('settings', 'model_end', str(end_year))
    config.set('settings', "scenarios",", ".join(scenarios))
    with open('settings.ini', 'w') as configfile:
        config.write(configfile)

    # This remains the bridge to your existing Model class
    model = ModelRun()
    model.run()
    # Save output for all scenarios to pickle
    results =  model.output

    # Create Output folder if it doesn't exist
    (Path('.') / 'Output').mkdir(parents=True, exist_ok=True)

    if len(output_name) < 1:
        output_name = 'results'

    with open(Path('.') / 'Output' / f'{output_name}.pickle', 'wb') as f:
        pickle.dump(results, f)