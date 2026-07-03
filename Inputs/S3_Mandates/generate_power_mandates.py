"""
Generate MEWR files for S3_Mandates scenario.
Phases out coal and oil from 2025.
"""
import os
import csv

# Years from 2001 to 2100
years = list(range(2001, 2101))

# Technology names (22 technologies in FTT-Power)
technologies = [
    "1 Nuclear",
    "2 Oil",
    "3 Coal",
    "4 Coal + CCS",
    "5 Waste",
    "6 Waste + CCS",
    "7 CCGT",
    "8 CCGT + CCS",
    "9 OCGT",
    "10 OCGT + CCS",
    "11 Biomass",
    "12 Biomass + CCS",
    "13 Large Hydro",
    "14 Pumped Hydro",
    "15 Geothermal",
    "16 Marine",
    "17 Onshore",
    "18 Offshore",
    "19 Solar PV",
    "20 CSP",
    "21 Fuel cells / Turbine",
    "22 Lithium-ion"
]

# Countries
countries = [
    "AC", "AE", "AN", "AR", "AS", "AT", "AU", "AW", "BE", "BG",
    "BR", "CA", "CH", "CN", "CO", "CY", "CZ", "DC", "DE", "DK",
    "EG", "EL", "EN", "ES", "FI", "FR", "HR", "HU", "ID", "IE",
    "IN", "IS", "IT", "JA", "KE", "KR", "KZ", "LA", "LT", "LV",
    "LX", "MK", "MT", "MX", "MY", "NG", "NL", "NO", "NZ", "OC",
    "ON", "OP", "PK", "PL", "PT", "RA", "RO", "RS", "RW", "SA",
    "SD", "SI", "SK", "SW", "TR", "TW", "UA", "UE", "UK", "US", "ZA"
]

# Phase-out year
PHASE_OUT_YEAR = 2025

# Technologies to phase out (0-indexed): Oil=1, Coal=2
PHASE_OUT_TECHS = [1, 2]  # Oil and Coal

def get_mewr_value(tech_idx, year):
    """
    Get MEWR value for a technology in a given year.

    Returns:
        -1: No regulation (free market)
        0: Phase-out/ban
    """
    # Phase out Oil and Coal from 2025
    if tech_idx in PHASE_OUT_TECHS:
        if year >= PHASE_OUT_YEAR:
            return 0.0  # Banned
        else:
            return -1.0  # Free before phase-out

    # All other technologies: no regulation
    return -1.0

def generate_mewr_file(country, output_dir):
    """Generate MEWR file for a country."""
    filepath = os.path.join(output_dir, f"MEWR_{country}.csv")

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header row with years
        header = [''] + [str(y) for y in years]
        writer.writerow(header)

        # Data rows for each technology
        for tech_idx, tech_name in enumerate(technologies):
            row = [tech_name]
            for year in years:
                value = get_mewr_value(tech_idx, year)
                row.append(value)
            writer.writerow(row)

def main():
    output_dir = "FTT-P"
    os.makedirs(output_dir, exist_ok=True)

    for country in countries:
        generate_mewr_file(country, output_dir)
        print(f"Generated MEWR_{country}.csv")

    print(f"\nGenerated {len(countries)} MEWR files in {output_dir}/")
    print("Coal and Oil phased out from 2025")

if __name__ == "__main__":
    main()
