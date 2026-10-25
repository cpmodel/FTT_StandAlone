import pytest

from ftt_source.support import check_input_files


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

    monkeypatch.setattr(check_input_files, 'get_utilities_path', lambda: utilities_path)
    monkeypatch.setattr(check_input_files, 'get_inputs_path', lambda: inputs_path)

    with pytest.raises(ValueError, match="rename it to 'EV CO2 regulation.csv'"):
        check_input_files.check_vars_exist('FTT-Tr')
