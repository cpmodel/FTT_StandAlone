# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_initialisation.py
=========================================
Builds the FTT:Power settings dict computed once per run.

This separates run-level configuration (read from settings.ini and the
CSV-driven mappings) from the per-year solve loop in ftt_p_main.py.
"""

from ftt_source.Power.ftt_p_costc import (get_tech_to_resource, get_resource_to_fuel_map,
                                          get_cf_multipliers, get_gen_tech_indices)
from ftt_source.Power.ftt_p_rldc import get_wind_solar_indices
from ftt_source.Power.ftt_p_fuel_price import get_fuel_price_indices


def build_power_settings(titles, config):
    """Build the dict of FTT:Power constants computed once per run.

    Parameters
    ----------
    titles : dict of lists
        Dimension titles loaded by RunFTT.
    config : configparser.ConfigParser
        Already-read settings, e.g. from RunFTT.__init__.
    """
    # Year the model initialises its first-year state (can be later than the simulation_start year of 2010)
    power_init_year    = int(config.get('settings', 'power_init_year',    fallback='2013'))
    return {
        'tech_to_resource':           get_tech_to_resource(titles),
        'resource_to_fuel_map':       get_resource_to_fuel_map(titles),  # ERTI resource idx -> [JTI fuel idx, ...]
        'cf_multipliers':             get_cf_multipliers(titles),
        'wind_solar_indices':         get_wind_solar_indices(titles),
        'fuel_price_indices':         get_fuel_price_indices(titles),
        'gen_tech_indices':           get_gen_tech_indices(titles),
        'power_init_year':            power_init_year,
        'gamma_mode':                 config.get('settings', 'gamma_mode',                     fallback='multiplicative'),
        'sector_coupling':            config.getboolean('settings', 'sector_coupling',         fallback=True),
        'mset_coupling':              config.getboolean('settings', 'mset_coupling',           fallback=False),
        'elec_idx':   list(titles['JTI']).index('8 Electricity'),
        'nuclear_idx': list(titles['T2TI']).index('1 Nuclear'),
    }
