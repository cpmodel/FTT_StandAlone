# -*- coding: utf-8 -*-
"""
Integration costs for variable renewable energy (VRE).

Adds balancing and grid integration costs to electricity prices as VRE
share increases. Based on cascading's ftt_p_mewp.py but simplified for
main's architecture (no load band generation required).

Reference: https://www.sciencedirect.com/science/article/pii/S0360544213009390
"""

import numpy as np


def get_vre_shares(data, r):
    """Compute share of solar PV and share of onshore + offshore wind."""
    total_gen = np.sum(data["MEWG"][r, :, 0])
    if total_gen == 0:
        return 0.0, 0.0

    solar_share = data["MEWG"][r, 18, 0] / total_gen  # Solar PV is index 18
    wind_share = np.sum(data["MEWG"][r, 16:18, 0]) / total_gen  # Wind is 16-17

    return solar_share, wind_share


def get_balancing_costs(solar_share, wind_share):
    """
    Add balancing costs. These go from 2 to 4 EUR/MWh (2013 values).
    After 30% of generation, it is set at the maximum.
    """
    min_costs = 2 * 1.36  # Convert 2013 EUR to 2023 USD
    max_costs = 4 * 1.36

    solar_share_scaled = np.clip((solar_share - 0.05) / (0.30 - 0.05), 0, 1)
    wind_share_scaled = np.clip((wind_share - 0.05) / (0.30 - 0.05), 0, 1)

    balancing_costs_solar = min_costs + (max_costs - min_costs) * solar_share_scaled
    balancing_costs_wind = min_costs + (max_costs - min_costs) * wind_share_scaled

    balancing_costs = balancing_costs_solar * solar_share + balancing_costs_wind * wind_share

    return balancing_costs


def get_grid_integration_costs(solar_share, wind_share):
    """
    Add grid integration costs. These go linearly to 7.5 EUR/MWh (2013 values).
    After 40% of generation, it is set at the maximum.
    """
    max_costs = 7.5 * 1.36  # Convert 2013 EUR to 2023 USD

    solar_share_scaled = np.clip(solar_share / 0.4, 0, 1)
    wind_share_scaled = np.clip(wind_share / 0.4, 0, 1)

    grid_costs_solar = max_costs * solar_share_scaled
    grid_costs_wind = max_costs * wind_share_scaled

    grid_integration_costs = grid_costs_solar * solar_share + grid_costs_wind * wind_share

    return grid_integration_costs


def add_vre_integration_costs(data, titles):
    """
    Add VRE integration costs to the generalized cost (METC).

    This function adds balancing and grid integration costs to solar (18)
    and wind (16, 17) technologies based on their market share.
    Costs are added to METC which feeds directly into the shares calculation.

    Parameters
    ----------
    data : dict
        Model variables
    titles : dict
        Title classifications

    Returns
    -------
    data : dict
        Updated with integration costs added to METC (generalized costs)
    """
    # Technology indices for VRE
    solar_idx = 18
    wind_onshore_idx = 16
    wind_offshore_idx = 17

    for r in range(len(titles['RTI'])):
        # Get current VRE shares
        solar_share, wind_share = get_vre_shares(data, r)

        # Calculate integration costs ($/MWh)
        balancing = get_balancing_costs(solar_share, wind_share)
        grid_costs = get_grid_integration_costs(solar_share, wind_share)
        total_integration_cost = balancing + grid_costs

        # Add integration costs to METC (generalized costs used in shares)
        # This makes solar/wind more expensive as their share grows
        # METC is in log space, so we add to the underlying cost
        if solar_share > 0 and data['METC'][r, solar_idx, 0] != 0:
            # Convert from log, add cost, convert back
            current_cost = np.exp(data['METC'][r, solar_idx, 0])
            data['METC'][r, solar_idx, 0] = np.log(current_cost + total_integration_cost)

        if wind_share > 0:
            if data['METC'][r, wind_onshore_idx, 0] != 0:
                current_cost = np.exp(data['METC'][r, wind_onshore_idx, 0])
                data['METC'][r, wind_onshore_idx, 0] = np.log(current_cost + total_integration_cost)
            if data['METC'][r, wind_offshore_idx, 0] != 0:
                current_cost = np.exp(data['METC'][r, wind_offshore_idx, 0])
                data['METC'][r, wind_offshore_idx, 0] = np.log(current_cost + total_integration_cost)

    return data
