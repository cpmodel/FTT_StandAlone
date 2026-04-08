"""
Script to overwrite certain BZTC values with constant values from
constant_params.csv. This also applies a consistent payload penalty to all BEV 
trucks.

@author: CL
"""
from pathlib import Path
import polars as pl

DICT_BZTC = {
    "discount_rate": 7,
    "lifetime": 8,
    "payload": 10,
    "mileage": 15,
}
ROWS_PER_REGION = 45
HEADER_ROWS_BETWEEN_REGIONS = 1
ROWS_PER_VEHICLE_TYPE = 5
TRUCKS_PER_REGION = 9
BEV_TRUCKS = ["BEV MDT", "BEV HDT"]



def build_truck_indices(total_rows: int, start_idx: int) -> list[int]:
    """Build row indices for a truck type across all regional BZTC blocks."""
    indices = []
    region_stride = ROWS_PER_REGION + HEADER_ROWS_BETWEEN_REGIONS

    for region_start in range(0, total_rows, region_stride):
        for truck_idx in range(TRUCKS_PER_REGION):
            row_idx = region_start + start_idx + truck_idx * ROWS_PER_VEHICLE_TYPE
            if row_idx < total_rows:
                indices.append(row_idx)

    return indices


def set_constant_params() -> None:
    """Overwrite BZTC parameters with constant values from constant_params.csv."""
    params_df = pl.read_csv("constant_params.csv")
    params_df = params_df.select(["parameter", "truck_type", "value"])
    
    masterfile_path = Path("_MasterFiles/FTT-Fr/FTT-Fr-45x71_2024_S0.xlsx") 
    masterfile_df = pl.read_excel(
        masterfile_path,
        sheet_name="BZTC",
        read_options={"skip_rows": 4}
        )
    masterfile_df = masterfile_df.drop(masterfile_df.columns[0])
    mdt_indices = build_truck_indices(masterfile_df.height, start_idx=2)
    hdt_indices = build_truck_indices(masterfile_df.height, start_idx=3)
    
    # do mdt first 
    mdt_params = params_df.filter(pl.col("truck_type") == "MDT")
    for param, col_idx in DICT_BZTC.items():
        col_name = masterfile_df.columns[col_idx]
        par_value = mdt_params.filter(
            pl.col("parameter") == param).select(pl.col("value")).item()
        print(f"Updating MDT {param} to {par_value}")
        masterfile_df = masterfile_df.with_columns(
            pl.when(pl.int_range(0, pl.len()).is_in(mdt_indices))
                    .then(pl.lit(par_value).cast(pl.Float64, strict=False))
                    .otherwise(pl.nth(col_idx))
                    .alias(col_name)
        )
    
    # hdt
    hdt_params = params_df.filter(pl.col("truck_type") == "HDT")
    for param, col_idx in DICT_BZTC.items():
        col_name = masterfile_df.columns[col_idx]
        par_value = hdt_params.filter(
            pl.col("parameter") == param).select(pl.col("value")).item()
        print(f"Updating HDT {param} to {par_value}")
        masterfile_df = masterfile_df.with_columns(
            pl.when(pl.int_range(0, pl.len()).is_in(hdt_indices))
                    .then(pl.lit(par_value).cast(pl.Float64, strict=False))
                    .otherwise(pl.nth(col_idx))
                    .alias(col_name)
        )
        
    # payload penalty for BEV trucks (penalty is recorded in percentage decrease)
    payload_penalty = params_df.filter(
        pl.col("parameter") == "payload_penalty"
        ).select(pl.col("value")).item()
    masterfile_df = masterfile_df.with_columns(
        pl.when(pl.nth(0).is_in(BEV_TRUCKS))
                .then(pl.nth(DICT_BZTC["payload"]).cast(pl.Float64, strict=False) 
                      * (1 - payload_penalty))
                .otherwise(pl.nth(DICT_BZTC["payload"]))
                .alias(masterfile_df.columns[DICT_BZTC["payload"]])
    )
    

    masterfile_df.write_csv('_MasterFiles/FTT-Fr/' + "bztc_altered.csv")
    print("Updated BZTC parameters with constant values from constant_params.csv.")
    
    
if __name__ == "__main__":
    set_constant_params()