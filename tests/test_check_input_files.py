import pytest
import polars as pl

from ftt_source.support import check_input_files


def test_check_var_dims_accepts_compact_dimensions():
    titles = {'RTI': ['A', 'B'], 'TIME': [2020, 2021]}
    dimensions = {'VAR': ['RTI', 'TIME']}
    frame = pl.DataFrame(
        {'RTI': ['A', 'B'], '2020': [1, 2], '2021': [3, 4]}
    )

    check_input_files.check_var_dims('VAR', frame, titles, dimensions)


def test_check_var_dims_reports_compact_dimension_mismatch():
    titles = {'RTI': ['A', 'B'], 'TIME': [2020, 2021]}
    dimensions = {'VAR': ['RTI', 'TIME']}
    frame = pl.DataFrame({'RTI': ['A'], '2020': [1], '2021': [2]})

    with pytest.raises(ValueError, match='expected 2, got 1'):
        check_input_files.check_var_dims('VAR', frame, titles, dimensions)


def test_check_vars_exist_explains_previous_ev_regulation_filename(
    tmp_path, monkeypatch
):
    utilities_path = tmp_path / 'Utilities'
    titles_path = utilities_path / 'titles'
    titles_path.mkdir(parents=True)
    (titles_path / 'VariableListing.csv').write_text(
        'Variable name,Domain,Is input variable\n'
        'EV CO2 regulation,FTT-Tr,Y\n'
    )

    inputs_path = tmp_path / 'Inputs'
    model_path = inputs_path / 'S0' / 'FTT-Tr'
    model_path.mkdir(parents=True)
    (model_path / 'EV regulation.csv').touch()

    monkeypatch.setattr(
        check_input_files,
        'get_utility_path',
        lambda relative_path: utilities_path / relative_path,
    )
    monkeypatch.setattr(check_input_files, 'get_inputs_path', lambda: inputs_path)

    with pytest.raises(ValueError, match="rename it to 'EV CO2 regulation.csv'"):
        check_input_files.check_vars_exist('FTT-Tr')
