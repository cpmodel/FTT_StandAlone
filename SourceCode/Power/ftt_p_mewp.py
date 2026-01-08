# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_mewp.py
=========================================
Marginal electricity pricing module for FTT:Power.

Computes marginal fuel prices (MEWP) with VRE integration costs.
Based on: https://www.sciencedirect.com/science/article/pii/S0360544213009390

Functions included:
    - get_gen_share: Compute solar and wind generation shares
    - add_balancing_costs: Add VRE balancing costs (2-4 EUR/MWh)
    - add_grid_integration_costs: Add grid integration costs (up to 7.5 EUR/MWh)
    - get_marginal_fuel_prices_mewp: Main function to compute MEWP

@author: Femke Nijsse
Created: Mon Nov 6 2023
"""
import numpy as np
from SourceCode.support.divide import divide


def get_gen_share(data, r):
    """Compute share of solar PV and share of onshore + offshore wind.

    Parameters
    ----------
    data : dict
        Model variables dictionary
    r : int
        Region index

    Returns
    -------
    tuple
        (solar_share, wind_share) as fractions of total generation
    """
    total_gen = np.sum(data["MEWG"][r])
    if total_gen <= 0:
        return 0.0, 0.0

    solar_share = data["MEWG"][r, 18] / total_gen
    wind_share = np.sum(data["MEWG"][r, 16:18]) / total_gen

    return solar_share, wind_share


def add_balancing_costs(solar_share, wind_share, r):
    """Add balancing costs for VRE integration.

    Balancing costs range from 2 to 4 EUR/MWh (2013 values).
    After 30% of generation, costs are at maximum.

    Reference: https://www.sciencedirect.com/science/article/pii/S0360544213009390

    Parameters
    ----------
    solar_share : float
        Solar generation as fraction of total
    wind_share : float
        Wind generation as fraction of total
    r : int
        Region index (unused, kept for consistency)

    Returns
    -------
    float
        Balancing costs in $/MWh
    """
    min_costs = 2 * 1.36  # Conversion 2013 EUR to 2013 USD
    max_costs = 4 * 1.36

    solar_share_scaled = np.clip((solar_share - 0.05) / (0.3 - 0.05), 0, 1)
    wind_share_scaled = np.clip((wind_share - 0.05) / (0.3 - 0.05), 0, 1)

    balancing_costs_solar = min_costs + (max_costs - min_costs) * solar_share_scaled
    balancing_costs_wind = min_costs + (max_costs - min_costs) * wind_share_scaled

    balancing_costs = balancing_costs_solar * solar_share + balancing_costs_wind * wind_share

    return balancing_costs


def add_grid_integration_costs(solar_share, wind_share, r):
    """Add grid integration costs for VRE.

    Grid integration costs increase linearly to 7.5 EUR/MWh (2013 values).
    After 40% of generation, costs are at maximum.

    Reference: https://www.sciencedirect.com/science/article/pii/S0360544213009390

    Parameters
    ----------
    solar_share : float
        Solar generation as fraction of total
    wind_share : float
        Wind generation as fraction of total
    r : int
        Region index (unused, kept for consistency)

    Returns
    -------
    float
        Grid integration costs in $/MWh
    """
    solar_share_scaled = np.clip(solar_share / 0.4, 0, 1)
    wind_share_scaled = np.clip(wind_share / 0.4, 0, 1)

    max_costs = 7.5 * 1.36  # Conversion 2013 EUR to 2013 USD

    grid_costs_solar = max_costs * solar_share_scaled
    grid_costs_wind = max_costs * wind_share_scaled

    grid_integration_costs = grid_costs_solar * solar_share + grid_costs_wind * wind_share

    return grid_integration_costs


def get_marginal_fuel_prices_mewp(data, titles, Svar):
    """Compute marginal fuel prices MEWP based on development within FTT:Power.

    This function calculates electricity prices (MEWP index 7) using either:
    - MPRI == 1: Weighted average LCOE approach (default)
    - MPRI == 2: Merit order approach

    For MPRI == 1:
    - New capacity uses LCOE including all policies (MECC incl CO2)
    - Old capacity uses LCOE with CO2 costs only (MECC only CO2)
    - Adds balancing and grid integration costs based on VRE share

    Parameters
    ----------
    data : dict
        Model variables dictionary containing:
        - MEWP: Output - marginal fuel prices
        - MPRI: Pricing mode switch (1 or 2)
        - MEWK: Installed capacity
        - MEWI: New investment
        - MEWL: Load factors
        - MEWG: Generation
        - MECC incl CO2: LCOE with all policies
        - MECC only CO2: LCOE with CO2 only
        - MERC: Fuel prices
        - MWG1-6: Generation by load band
        - MWMC: Marginal costs
        - MLBP: Load band prices (output for MPRI==2)
    titles : dict
        Dimension titles, requires 'RTI' for regions
    Svar : ndarray
        Variable technology indicator (1 for VRE, 0 for dispatchable)

    Returns
    -------
    data : dict
        Updated with MEWP values
    """
    # Set pricing mode to 1 (weighted LCOE) for all regions
    data["MPRI"][:] = 1

    # Load band weights for merit order pricing (if MPRI == 2)
    n_loadbands = 6
    non_vre_lb_weight = [0] * 5
    non_vre_lb_weight[4] = 80.0 / 8766.0
    non_vre_lb_weight[3] = (700.0 - 80.0) / 8766.0
    non_vre_lb_weight[2] = (2200.0 - 700.0) / 8766.0
    non_vre_lb_weight[1] = (4400.0 - 2200.0) / 8766.0
    non_vre_lb_weight[0] = (8766.0 - 4400.0) / 8766.0

    # Set fuel prices for specific fuels
    data["MEWP"][:, 0, 0] = data["MERC"][:, 2, 0]   # Hard coal
    data["MEWP"][:, 1, 0] = data["MERC"][:, 2, 0]   # Soft coal
    data["MEWP"][:, 2, 0] = data["MERC"][:, 1, 0]   # Crude oil
    data["MEWP"][:, 3, 0] = data["MERC"][:, 3, 0]   # Natural gas
    data["MEWP"][:, 10, 0] = data["MERC"][:, 4, 0]  # Biomass

    # For each region
    for r in range(len(titles['RTI'])):

        # MPRI == 1: Use the weighted average LCOE approach
        if data["MPRI"][r] == 1:
            # Calculate weight for new vs old capacity
            weight_new = 0.0
            total_capacity = np.sum(data["MEWK"][r, :, 0])
            if total_capacity > 0.0:
                weight_new = np.sum(data["MEWI"][r, :, 0]) / total_capacity
            weight_new = min(weight_new, 1.0)
            weight_old = 1.0 - weight_new

            # Shares of new capacity by technology
            total_new = np.sum(data["MEWI"][r, :, 0])
            if total_new > 0:
                shares_new = data["MEWI"][r, :, 0] / total_new
            else:
                shares_new = np.zeros(data["MEWI"][r, :, 0].shape)

            # Shares of old capacity by technology
            shares_old = (data["MEWK"][r, :, 0] - data["MEWI"][r, :, 0]) / max(total_capacity, 1e-10)
            shares_old[shares_old < 0.0] = 0.0
            shares_old_sum = np.sum(shares_old)
            if shares_old_sum > 0:
                shares_old = shares_old / shares_old_sum

            # Weighted LCOE for new capacity (includes all policies)
            weighted_lcoe_new = divide(
                np.sum(shares_new * data["MEWL"][r, :, 0] * data["MECC incl CO2"][r, :, 0]),
                np.sum(shares_new * data["MEWL"][r, :, 0])
            )

            # Weighted LCOE for old capacity (CO2 costs only)
            denom_old = np.sum(shares_old * data["MEWL"][r, :, 0])
            if denom_old > 0:
                weighted_lcoe_old = np.sum(shares_old * data["MEWL"][r, :, 0] * data["MECC only CO2"][r, :, 0]) / denom_old
            else:
                weighted_lcoe_old = 0.0

            # Combined electricity price
            data["MEWP"][r, 7, 0] = weight_new * weighted_lcoe_new + weight_old * weighted_lcoe_old

            # Add grid and balancing costs based on VRE share
            solar_share, wind_share = get_gen_share(data, r)
            data["MEWP"][r, 7, 0] += add_balancing_costs(solar_share, wind_share, r)
            data["MEWP"][r, 7, 0] += add_grid_integration_costs(solar_share, wind_share, r)


        # MPRI == 2: Merit order approach (not used by default)
        elif data["MPRI"][r] == 2:
            # Generation in each load band
            glb_dict = {
                0: data["MWG1"][r, :, 0],
                1: data["MWG2"][r, :, 0],
                2: data["MWG3"][r, :, 0],
                3: data["MWG4"][r, :, 0],
                4: data["MWG5"][r, :, 0],
                5: data["MWG6"][r, :, 0]
            }

            # Loop over load bands
            for LB in range(n_loadbands):
                mc_tech_by_lb = np.zeros_like(data["MWMC"][r, :, 0])

                # Only select technologies with non-zero generation
                where_condition = glb_dict[LB] > 0.0
                mc_tech_by_lb[where_condition] = (
                    data["MWMC"][r, :, 0][where_condition]
                    - data["BCET"][r, :, 0][where_condition]
                )

                # Weighted average marginal cost
                if np.sum(glb_dict[LB]) > 0.0:
                    data["MLBP"][r, LB, 0] = np.sum(mc_tech_by_lb * glb_dict[LB]) / np.sum(glb_dict[LB])
                else:
                    data["MLBP"][r, LB, 0] = np.max(data["MWMC"][r, :, 0] * Svar[r, :])

            # Adjust load band prices for start-up costs and VRE transmission
            data["MLBP"][r, 4, 0] *= 1.25  # Peak load - highest start-up costs
            data["MLBP"][r, 3, 0] *= 1.1
            data["MLBP"][r, 2, 0] *= 1.05
            data["MLBP"][r, 5, 0] *= 1.3   # VRE - higher transmission costs

            vre_weight = 0.0
            non_vre_price = 0.0

            # VRE weight increases above 40% penetration
            share_VRE = np.sum(data["MEWG"][r, :, 0] * Svar[r, :]) / max(np.sum(data["MEWG"][r, :]), 1e-10)
            if share_VRE > 0.40:
                vre_weight = (1.0 / 0.6) * share_VRE - 2.0 / 3.0

            if np.sum(np.array([glb_dict[LB] for LB in range(n_loadbands-1)])) > 0.0:
                non_vre_price = np.sum(data["MLBP"][r, :n_loadbands-1, 0] * non_vre_lb_weight)

            data["MEWP"][r, 7, 0] = vre_weight * data["MLBP"][r, 5, 0] + (1.0 - vre_weight) * non_vre_price


    return data
