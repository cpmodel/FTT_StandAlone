class GUIState:
    """
    Class for maintaining the state of the GUI application.
    """

    def __init__(self):
        self.is_running = False
        self.dark_mode = False
        # Results page state
        self.selected_scenarios = []
        self.selected_variable = None
        self.dim_selections = [{} for _ in range(4)]
        self.dim_aggregate = [False, False, False, False]

state = GUIState()