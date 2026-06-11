import configparser

from SourceCode.Power.power_settings import PowerSettings, load_power_settings
from SourceCode.support.titles_functions import load_titles


def test_load_power_settings_defaults():
    config = configparser.ConfigParser()
    config.add_section("settings")

    settings = load_power_settings(load_titles(), config)

    assert isinstance(settings, PowerSettings)
    assert settings.prsc_base_year == 2013
    assert settings.ex_base_year == 2013
    assert settings.prsc_var == "PRSC13"
    assert settings.ex_var == "EX13"
    assert settings.rex_var == "REX13"
    assert settings.wind_solar_indices["wind"] == (16, 17)
    assert settings.wind_solar_indices["solar"] == 18


def test_load_power_settings_overrides():
    config = configparser.ConfigParser()
    config.add_section("settings")
    config.set("settings", "prsc_base_year", "2015")
    config.set("settings", "ex_base_year", "2015")
    config.set("settings", "usd_exchange_region", "35 Japan (JA)")
    config.set("settings", "gamma_mode", "additive")
    config.set("settings", "sector_coupling", "false")
    config.set("settings", "mset_coupling", "true")
    config.set("settings", "rldc_start_year", "2016")
    config.set("settings", "bcet_copy_range_end", "20")

    settings = load_power_settings(load_titles(), config)

    assert settings.prsc_base_year == 2015
    assert settings.ex_base_year == 2015
    assert settings.prsc_var == "PRSC15"
    assert settings.ex_var == "EX15"
    assert settings.rex_var == "REX15"
    assert settings.gamma_mode == "additive"
    assert settings.sector_coupling is False
    assert settings.mset_coupling is True
    assert settings.rldc_start_year == 2016
    assert settings.bcet_copy_range_end == 20
    assert settings.usd_idx == 34
