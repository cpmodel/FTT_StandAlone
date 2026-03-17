"""
A script that changes conditions for battery electric trucks in FTT Freight to consider
good and bad conditions for their adoption. Paramneters are set in extremes_params.csv.

Script takes two optional command-line arguments:
1. Scenario name (default: "freight_extremes")
2. Extreme type (positive or negatvie) (default: "positive")

@author: CL
"""

from pathlib import Path
import sys
import polars as pl


DEFAULT_SCENARIO_NAME = "freight_extremes"
DEFAULT_EXTREME_TYPE = "positive"
PARAMS_FILE = "extremes_params.csv"
MASTERFILE_DIR = "_MasterFiles/FTT-Fr/"
MASTERFILE_NAME = "FTT-Fr-45x71_2024_S0.xlsx"
BEV_TRUCKS = ["BEV MDT", "BEV HDT"]
DICT_BZTC = {
    "turnover_rate": 14,
    "payload": 10,
    "mileage": 15,
    "truck_lr": 13,
    "inv_cost": 1,
    "inv_cost_std": 2,
    }


def parse_args() -> tuple[str, int | str]:
    """Return the scenario name and extreme type from command-line arguments."""
    args = sys.argv[1:]

    scen_name = args[0] if len(args) >= 1 else DEFAULT_SCENARIO_NAME
    extreme_type = args[1] if len(args) >= 2 else DEFAULT_EXTREME_TYPE
    if extreme_type not in ["positive", "negative"]:
        raise ValueError("Invalid extreme type. Must be 'positive' or 'negative'.")

    return scen_name, extreme_type


def generate_freight_extremes() -> None:
    """
    Generate the freight extremes scenario structure. Battery learning rate
    is updated automatically. Cost matrix parameters are saved 
    in a new csv ready to be copied into the new masterfile. User must manually
    copy csv data into masterfile.
    """
    params_df = pl.read_csv(PARAMS_FILE)
    scen_name, extreme_type = parse_args()

    print(f"Generating freight extremes for scenario: {scen_name}...")

    scenario_path = Path(scen_name)
    scenario_path.mkdir(parents=True, exist_ok=True)
    
    # create a copy of s0 freight masterfile with scenario name as suffix
    masterfile_path = Path(MASTERFILE_DIR) / MASTERFILE_NAME
    new_masterfile_path = (Path(MASTERFILE_DIR) /
                           f"{MASTERFILE_NAME.replace('S0.xlsx', f'{scen_name}.xlsx')}")
    new_masterfile_path.write_bytes(masterfile_path.read_bytes())
    
    # Stage 1 -- update battery learning rate
    par_batt_lr = params_df.filter(
        pl.col("parameter") == "battery_lr").select(pl.col(extreme_type)).item()
    
    # read existing battery learning rate from SectorCouplingAssumps file
    sec_coup_path = Path("S0/General/SectorCouplingAssumps.csv")
    sector_coupling_df = pl.read_csv(sec_coup_path)
    old_batt_lr = sector_coupling_df.filter(
        pl.nth(0) == "Battery learning exponent").select(pl.nth(1)).item()
    new_batt_lr = round(par_batt_lr * old_batt_lr, 3)
    
    # Make copy of SectorCouplingAssumps with new battery learning rate
    new_sec_coup_path = Path(scen_name) / "General" / "SectorCouplingAssumps.csv"
    new_sec_coup_path.parent.mkdir(parents=True, exist_ok=True)
    sector_coupling_df = sector_coupling_df.with_columns(
        Mean=pl.when(pl.nth(0)=='Battery learning exponent')
                .then(pl.lit(new_batt_lr))
                .otherwise(pl.col('Mean')))
    
    sector_coupling_df.write_csv(new_sec_coup_path)
    print(f"Updated battery learning rate from {old_batt_lr} to {new_batt_lr}")
    
    # Stage 2 -- update cost matrix values
    # read new masterfile and update cost matrix values
    masterfile_df = pl.read_excel(
        new_masterfile_path,
        sheet_name="BZTC",
        read_options={"skip_rows": 4}
        )
    
    masterfile_df = masterfile_df.drop(masterfile_df.columns[0])
        
    # for bev trucks, update the relevant parameters by multiplying with the parameter from extremes_params.csv.
    # column indexes are stored in dict_bztc
    
    for param, col_idx in DICT_BZTC.items():
        col_name = masterfile_df.columns[col_idx]
        par_value = params_df.filter(
            pl.col("parameter") == param).select(pl.col(extreme_type)).item()
        # truck_lr and inv_cost are only applied to bev trucks, while the other parameters are applied to all trucks.
        if param == "truck_lr" or param == "inv_cost" or param == "inv_cost_std":
            masterfile_df = masterfile_df.with_columns(
                pl.when(pl.nth(0).is_in(BEV_TRUCKS))
                    .then(pl.nth(col_idx).cast(pl.Float64, strict=False) * par_value)
                    .otherwise(pl.nth(col_idx).cast(pl.Float64, strict=False))
                    .alias(col_name)
            )
        else:
            masterfile_df = masterfile_df.with_columns(
                pl.nth(col_idx).cast(pl.Float64, strict=False) * par_value
            )
    
    masterfile_df.write_csv('_MasterFiles/FTT-Fr/' + f"bztc_{scen_name}.csv")
    print(f"Updated parameters for {scen_name}.")
    
    
if __name__ == "__main__":
    generate_freight_extremes()
