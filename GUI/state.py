import csv
from pathlib import Path


def _load_gui_settings():
    """Load display settings from GUI/gui_settings.csv, returning a dict with defaults."""
    defaults = {'dark_mode': False, 'plot_start_year': 2010, 'plot_end_year': 2050}
    settings_path = Path('GUI/gui_settings.csv')
    if not settings_path.exists():
        return defaults
    try:
        with open(settings_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get('SETTING', '').strip()
                val = row.get('VALUE', '').strip()
                if key == 'dark_mode':
                    defaults[key] = val in ('1', 'true', 'True', 'yes')
                elif key in ('plot_start_year', 'plot_end_year'):
                    try:
                        defaults[key] = int(val)
                    except ValueError:
                        pass
    except Exception:
        pass
    return defaults


class GUIState:
    """
    Class for maintaining the state of the GUI application.
    """

    def __init__(self):
        self.is_running = False
        _settings = _load_gui_settings()
        self.dark_mode = _settings['dark_mode']
        self.plot_start_year = _settings['plot_start_year']
        self.plot_end_year = _settings['plot_end_year']
        # Results page state
        self.selected_result_files = []
        self.selected_scenarios = []
        self.selected_baseline = None
        self.selected_variable = None
        self.result_type = "levels"  # Options: "levels", "absolute_diff", "relative_diff"
        self.dim_selections = [{} for _ in range(4)]
        self.dim_aggregate = [False, False, False, False]
        self.dim_selection_cache = {}
        self.dim_aggregate_cache = {}

state = GUIState()
