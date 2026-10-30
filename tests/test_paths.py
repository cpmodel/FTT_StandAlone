from ftt_source import paths


def test_utility_file_override_falls_back_to_bundled_defaults(tmp_path, monkeypatch):
    override_path = tmp_path / "Utilities"
    titles_path = override_path / "titles"
    titles_path.mkdir(parents=True)
    classification_titles = titles_path / "classification_titles.csv"
    classification_titles.write_text("custom titles")

    monkeypatch.setattr(paths, "_utilities_path", override_path)

    assert paths.get_utility_path(
        "titles/classification_titles.csv"
    ) == classification_titles
    assert paths.get_utility_path(
        "titles/VariableListing.csv"
    ) == paths._PACKAGE_ROOT / "Utilities" / "titles" / "VariableListing.csv"
