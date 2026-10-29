import asyncio
import socket
import sys
import os

# Ensure we're in the repo root directory
repo_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(repo_root)


def _is_interactive() -> bool:
    return 'spyder_kernels' in sys.modules or 'ipykernel' in sys.modules


def _enable_interactive_asyncio_patch() -> None:
    import nest_asyncio

    nest_asyncio.apply()
    patched_asyncio_run = asyncio.run

    def asyncio_run_compat(main, *, debug=None, loop_factory=None):
        return patched_asyncio_run(main, debug=debug)

    asyncio.run = asyncio_run_compat


def _select_port(preferred: int = 8080, max_tries: int = 10) -> int:
    """Return a free TCP port.

    Tries a preferred contiguous range first, then falls back to an OS-assigned
    ephemeral port so launch does not fail just because a small range is busy.
    """
    for port in range(preferred, preferred + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port

    # Fallback: ask OS for any available port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


if _is_interactive():
    _enable_interactive_asyncio_patch()

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

# Select an available port and start the server
selected_port = _select_port(8080)
if selected_port != 8080:
    print(f'Port 8080 is unavailable, starting frontend on port {selected_port} instead.')

ui.run(title="FTT", port=selected_port, favicon='GUI/images/ftt_favicon.png', reload=False)