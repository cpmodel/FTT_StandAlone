from nicegui import ui, run
import pickle
import configparser
from pathlib import Path
import asyncio
from queue import Queue

from .shared import shared_layout
from .state import state

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
                    available_models = get_available_models()
                    model_select = ui.select(available_models, value='FTT-P', label='Model(s) to run', multiple=True)

                    available_scenarios = get_available_scenarios()
                    scenario_select = ui.select(available_scenarios, value='S0', label='Scenario(s) to run', multiple=True)

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

                progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full h-5 mb-4').props('track-color=gray-600')

                log_console = ui.log(max_lines=50).classes('w-full h-64 bg-slate-900 text-green-400 text-xs font-mono p-2 rounded')

    # --- Run Logic ---
    # Create queues for progress and log updates from backend thread
    progress_queue = Queue()
    log_queue = Queue()
    update_timer = None
    
    async def update_from_queues():
        """Pull updates from queues and update UI"""
        # Process progress updates
        while not progress_queue.empty():
            current, total = progress_queue.get()
            progress_value = current / total if total > 0 else 0
            progress_bar.set_value(progress_value)
            progress_bar.props(f'instant-feedback')
        
        # Process log updates
        while not log_queue.empty():
            message = log_queue.get()
            log_console.push(message)
    
    async def start_run():
        nonlocal update_timer
        
        # Disable run button and set running state (disables navigation buttons)
        run_btn.disable()
        state.is_running = True
        progress_bar.set_value(0)
        log_console.clear()
        
        # Start periodic timer to check queues
        update_timer = ui.timer(0.1, update_from_queues)
        
        try:
            # Capture values
            models_value = model_select.value
            end_year_value = int(horizon.value)
            scenarios_value = scenario_select.value
            output_value = output_name.value

            # Execute model (IO bound thread)
            log_console.push(f"Initializing model run...")
            log_console.push(f"Models: {', '.join(models_value)}")
            log_console.push(f"Scenarios: {', '.join(scenarios_value)}")
            log_console.push(f"End year: {end_year_value}")
            log_console.push("-" * 40)
            
            await run.io_bound(execute_model, models_value, end_year_value,
                                scenarios_value, output_value, progress_queue, log_queue)
            
            progress_bar.set_value(1.0)
            log_console.push("-" * 40)
            log_console.push("SUCCESS: Results written to pickle.")
            ui.notify('Run Complete', type='positive')
            
        except Exception as e:
            log_console.push("-" * 40)
            log_console.push(f"CRITICAL ERROR: {str(e)}")
            ui.notify('Error', type='negative')
        finally:
            # Stop the update timer
            if update_timer:
                update_timer.cancel()
            # Final update to clear queues
            await update_from_queues()
            run_btn.enable()
            state.is_running = False

def execute_model(models, end_year, scenarios, output_name, progress_queue, log_queue):
    """
    Function to update settings.ini with front end selections,
    call model run, and write output to pickle file.

    Args:
        models: List of models to run
        end_year: End year for simulation
        scenarios: List of scenarios to run
        output_name: Name for output pickle file
        progress_queue: Queue for progress updates (current, total)
        log_queue: Queue for log messages
    """
    # Define callbacks to push updates to queues
    def progress_callback(current, total):
        """Called by model to report progress"""
        progress_queue.put((current, total))
    
    def log_callback(message):
        """Called by model to report log messages"""
        log_queue.put(message)
    
    config = configparser.ConfigParser()
    config.read('settings.ini')
    config.set('settings', 'enable_modules',", ".join(models))
    config.set('settings', 'simulation_end', str(end_year))
    config.set('settings', 'model_end', str(end_year))
    config.set('settings', "scenarios",", ".join(scenarios))
    with open('settings.ini', 'w') as configfile:
        config.write(configfile)
    
    # Import ModelRun only when needed (lazy loading for faster GUI startup)
    from SourceCode.model_class import ModelRun
    
    # Create model with callbacks
    model = ModelRun(progress_callback=progress_callback, log_callback=log_callback)
    
    log_queue.put("Running model...")
    model.run()
    
    # Save output for all scenarios to pickle
    results =  model.output

    # Create Output folder if it doesn't exist
    (Path('.') / 'Output').mkdir(parents=True, exist_ok=True)

    if len(output_name) < 1:
        output_name = 'results'

    with open(Path('.') / 'Output' / f'{output_name}.pickle', 'wb') as f:
        pickle.dump(results, f)
    
    log_queue.put(f"Results saved to Output/{output_name}.pickle")
    
def get_available_scenarios():
    """
    Returns a list of available scenario names based on the folders located in
    inputs folder (exclues folders with _prefix)

    Returns:
        scenario_list: List of available scenario names
    """
    input_path = Path('inputs')
    scenario_list = [folder.name for folder in input_path.iterdir() if folder.is_dir() and not folder.name.startswith('_')]
    return scenario_list

def get_available_models():
    """
    Returns a list of available model names based on the folders located in 
    Inputs/_MasterFiles

    Returns:
        model_list: List of available model names
    """
    source_path = Path('Inputs/_MasterFiles')
    model_list = [folder.name for folder in source_path.iterdir() if folder.is_dir()]
    return model_list