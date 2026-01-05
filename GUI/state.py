class GUIState:
    """
    Class for maintaining the state of the GUI application.
    """

    def __init__(self):
        self.is_running = False
        self.dark_mode = False

state = GUIState()