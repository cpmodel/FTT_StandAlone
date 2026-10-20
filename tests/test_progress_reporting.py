"""Check run progress without loading inputs or running sector solvers."""

from unittest.mock import Mock

import numpy as np
import pytest

from ftt_source.model_class import RunCancelledError, RunFTT


def make_model():
    """Create a minimal model whose solver can be inspected without input files."""
    model = RunFTT.__new__(RunFTT)
    model.timeline = [2020, 2021]
    model.input = {scen: {'value': np.zeros((1, 1, 1, 2))}
                   for scen in ['S0', 'S3', 'S2']}
    model.dims = {'value': ['TIME']}
    model.progress_callback = Mock()
    model.log_callback = Mock()
    model.stop_callback = None
    model.solve_year = Mock(return_value=({'value': np.ones((1, 1, 1))}, {}))
    return model


def test_progress_counts_years_across_scenarios(capsys):
    """Report cumulative year progress, one line per scenario, and total time."""
    model = make_model()
    model.run()

    assert [call.args for call in model.progress_callback.call_args_list] == [
        (completed, 6) for completed in range(7)
    ]
    messages = [call.args[0] for call in model.log_callback.call_args_list]
    assert messages[:-1] == [
        f'Running scenario {scenario}' for scenario in model.input
    ]
    assert messages[-1].startswith('Total elapsed time is ')
    for scenario in model.input:
        assert np.all(model.output[scenario]['value'] == 1)
    terminal = capsys.readouterr().out
    assert all(message in terminal for message in messages)


def test_failed_year_does_not_report_completion():
    """Do not advance progress or report total time when a year fails."""
    model = make_model()
    model.solve_year.side_effect = RuntimeError('Solver failed')

    with pytest.raises(RuntimeError, match='Solver failed'):
        model.run()

    model.progress_callback.assert_called_once_with(0, 6)
    model.log_callback.assert_called_once_with('Running scenario S0')


def test_callbacks_are_optional():
    """Complete every scenario and year when reporting callbacks are omitted."""
    model = make_model()
    model.progress_callback = None
    model.log_callback = None
    model.run()
    assert model.solve_year.call_count == 6


def test_stop_request_cancels_before_next_year():
    """Stop at the next year boundary without reporting extra progress."""
    model = make_model()
    model.stop_callback = Mock(side_effect=[False, False, True])

    with pytest.raises(RunCancelledError, match='Run cancelled by user'):
        model.run()

    model.solve_year.assert_called_once_with(2020, 0, 'S0')
    assert [call.args for call in model.progress_callback.call_args_list] == [
        (0, 6), (1, 6)
    ]
