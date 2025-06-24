# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:20:24 2025

@author: Femke
"""

import numpy as np

def co2_corr(data, titles, regions=None):
    """
    Compute regional CO2 correction factors based on vehicle age and survival.

    Parameters:
    - data: Dictionary of input arrays with keys 'TESH', 'TESF', 'TETH', 'RFLT'.
    - titles: Dictionary containing 'RTI' (region indices).
    - regions: Optional list/array of region indices to process. Defaults to all regions.

    Returns:
    - CO2_corr: 1D array of CO2 correction factors for specified regions.
    - region_has_fleet: Boolean array indicating regions with cars on the road.
    """
    if regions is None:
        regions = np.arange(len(titles['RTI']))

    TESH = data['TESH'][regions, :, 0]  # Vehicle sales history
    TESF = data['TESF'][regions, :, 0]  # Survival function
    TETH = data['TETH'][regions, :, 0]  # Efficiency correction by age
    RFLT = data['RFLT'][regions, 0, 0]  # Fleet stocks in use

    numer = (TESH * TESF * TETH).sum(axis=1)
    total_fleet = (TESH * TESF).sum(axis=1)

    CO2_corr = np.ones(len(regions))
    region_has_fleet = RFLT > 0.0
    CO2_corr[region_has_fleet] = numer[region_has_fleet] / total_fleet[region_has_fleet]
    
    # The CO2 correction is slightly underestimated for historical data, and may overestimate it for future fleets
    return CO2_corr, region_has_fleet


def biofuel_corr(data, titles, region_has_fleet, regions=None):
    """
    Apply biofuel-related emission corrections and adjust fuel shares accordingly.

    Parameters:
    - data: Dictionary of input arrays including 'TJET' and 'RBFM'.
    - titles: Dictionary containing 'RTI', 'VTTI', 'JTI'.
    - region_has_fleet: Boolean array indicating regions with vehicles.
    - regions: Optional list/array of region indices. Defaults to all regions.

    Returns:
    - biofuel_corr: 2D array of emission corrections [region x vehicle type].
    - fuel_converter: 3D array of adjusted fuel shares [region x vehicle x fuel].
    """
    if regions is None:
        regions = np.arange(len(titles['RTI']))

    n_regions = len(regions)
    n_vehicles = len(titles['VTTI'])

    biofuel_corr = np.ones((n_regions, n_vehicles))
    fuel_converter = np.tile(data['TJET'][0, :, :], (n_regions, 1, 1))

    JTI = titles['JTI']
    biofuel_ind = JTI.index('11 Biofuels')
    middle_dist_ind = JTI.index('5 Middle distillates')

    biofuel_mand = data['RBFM'][regions, 0, 0]
    TJET_middle = data['TJET'][0, :, middle_dist_ind] != 0

    biofuel_corr[region_has_fleet, :] = 1.0 - biofuel_mand[region_has_fleet, None] * TJET_middle[None, :]
    fuel_converter[:, :, middle_dist_ind] *= (1.0 - biofuel_mand[:, None]) * TJET_middle
    fuel_converter[:, :, biofuel_ind] *= biofuel_mand[:, None] * TJET_middle

    return biofuel_corr, fuel_converter


def compute_emissions_and_fuel_use(data, titles, CO2_corr, biofuel_corr, fuel_converter, c3ti, regions=None):
    """
    Compute total emissions and fuel use for each region and vehicle type.

    Parameters:
    - data: Dictionary with 'TEWG', 'BTTC', 'TJEF', 'TEWE'.
    - titles: Dictionary containing 'RTI', 'VTTI'.
    - CO2_corr: 1D array of CO2 correction factors.
    - biofuel_corr: 2D array of emission corrections [region x vehicle type].
    - fuel_converter: 3D array of adjusted fuel shares [region x vehicle x fuel].
    - c3ti: Dictionary of column indices for BTTC variables.
    - regions: Optional list/array of region indices. Defaults to all regions.
    """
    if regions is None:
        regions = np.arange(len(titles['RTI']))

    energy_use_per_tech = (data['TEWG'][regions, :, 0]
                          * data['BTTC'][regions, :, c3ti['9 energy use (MJ/km)']]
                          * CO2_corr[:, None] / 41.868)
    
    data['TJEF'][regions, :, 0] = np.einsum('v f r, r v -> r f',
                                           fuel_converter.transpose(1, 2, 0),
                                           energy_use_per_tech)

    # TODO: are we double counting electric emissions? They should likely be zero in this model.
    data['TEWE'][regions, :, 0] = (data['TEWG'][regions, :, 0]
                                   * data['BTTC'][regions, :, c3ti['14 CO2Emissions']]
                                   * CO2_corr[:, None] * biofuel_corr / 1e6)
