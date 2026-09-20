"""Check run progress without loading inputs or running sector solvers."""

from unittest.mock import Mock

import numpy as np
import pytest

from ftt_source.model_class import RunFTT


def make_model():
    model = RunFTT.__new__(RunFTT)
    model.timeline = [2020, 2021]
    model.input = {scen: {'value': np.zeros((1, 1, 1, 2))}
                   for scen in ['S0', 'S3', 'S2']}
    model.dims = {'value': ['TIME']}
    model.progress_callback = Mock()
    model.log_callback = Mock()
    model.solve_year = Mock(return_value=({'value': np.ones((1, 1, 1))}, {}))
    return model


def test_progress_counts_years_across_scenarios(capsys):
    model = make_model()
    model.run()

    assert [call.args for call in model.progress_callback.call_args_list] == [
        (completed, 6) for completed in range(7)
    ]
    messages = [call.args[0] for call in model.log_callback.call_args_list]
    assert len(messages) == 6
    for index, scenario in enumerate(model.input):
        assert messages[index * 2] == f'Starting scenario {scenario}'
        assert messages[index * 2 + 1].startswith(f'Finished scenario {scenario}.')
        assert np.all(model.output[scenario]['value'] == 1)
    terminal = capsys.readouterr().out
    assert all(message in terminal for message in messages)


def test_failed_year_does_not_report_completion():
    model = make_model()
    model.solve_year.side_effect = RuntimeError('Solver failed')

    with pytest.raises(RuntimeError, match='Solver failed'):
        model.run()

    model.progress_callback.assert_called_once_with(0, 6)
    model.log_callback.assert_called_once_with('Starting scenario S0')


def test_callbacks_are_optional():
    model = make_model()
    model.progress_callback = None
    model.log_callback = None
    model.run()
    assert model.solve_year.call_count == 6
