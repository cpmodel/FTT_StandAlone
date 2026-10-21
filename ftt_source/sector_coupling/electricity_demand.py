"""
Calculate electricity-demand feedback from the demand sectors to Power.

This feedback was developed for the REFEREE and EEIST II projects.
"""

import pandas as pd

from ftt_source.paths import get_utilities_path


def electricity_demand_feedback(data, data_baseline, y, titles, units):
    """Add changes in sector electricity demand to ``MEWDX``."""
    # Energy carrier classification
    electricity_index = titles["JTI"].index("8 Electricity")
    # Electricity mapping for each model
    mappings = pd.read_csv(
        get_utilities_path() / "mappings" / "Electricity_demand_mapping.csv",
        index_col=0,
    )
    conversions = pd.read_csv(
        get_utilities_path() / "mappings" / "Energy_unit_conversions.csv",
        index_col=0,
    )
    demand_unit = units["MEWD"]

    for model in mappings.index:
        demand_variable = mappings.loc[model, "fuel_var"]
        # Convert each sector demand variable to the MEWD unit (PJ).
        factor = conversions.loc[units[demand_variable], demand_unit]
        # Compute the change from the S0 baseline.
        baseline = data_baseline[demand_variable][:, electricity_index, y]
        current = data[demand_variable][:, electricity_index]
        data["MEWDX"][:, electricity_index] += (current - baseline) * factor

    return data
