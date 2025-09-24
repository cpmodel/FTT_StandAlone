# io_debug_export.py
import numpy as np
from datetime import datetime
from pathlib import Path

def export_io_summary(
    data_dt,
    t2ti,
    tech_key="23 Rooftop Solar",
    outfile="io_check.txt",
    *,
    max_items=20,      # max number of array elements to print per line
    edge_items=5       # how many items to show from start/end when truncated
):
    """
    Create a TXT file summarizing selected inputs and outputs per iteration.

    Inputs per iteration i:
      - MEWG[i, t2ti['23 Rooftop Solar']]  -> rooftop solar install capacity (scalar)
      - MEWDH[i, :, 0]                     -> household demand (vector)
      - METC[i, t2ti['23 Rooftop Solar']]  -> rooftop solar std (scalar)
      - PRICH[i, :, 0]                     -> household electricity price (vector)

    Outputs per iteration i:
      - household_shares[i, 0] -> share rooftop solar
      - household_shares[i, 1] -> share grid
      - costs_household[i, 0]  -> cost rooftop solar
      - costs_household[i, 1]  -> cost grid
      - costs_household_std[i, 0] -> std rooftop solar
      - costs_household_std[i, 1] -> std grid
    """
    if tech_key not in t2ti:
        raise KeyError(f"'{tech_key}' not found in t2ti")

    tech_idx = int(t2ti[tech_key])
    n_iter = int(data_dt["household_shares"].shape[0])

    # Pretty formatter: scalars vs arrays (with truncation)
    def fmt_val(val):
        arr = np.asarray(val)
        if arr.size == 1:
            return f"{arr.item():,.6g}"
        flat = arr.ravel()
        n = flat.size
        if n <= max_items:
            shown = flat
        else:
            # Take head and tail slices
            head = flat[:edge_items]
            tail = flat[-edge_items:]
            shown = np.concatenate([head, tail])
        # Build string with potential ellipsis and shape/len info
        body = np.array2string(
            shown,
            precision=6,
            separator=", ",
            suppress_small=False,
            max_line_width=120,
            formatter={"float_kind": lambda x: f"{x:,.6g}"}
        )
        if flat.size > max_items:
            return f"{body[:-1]}, ... ]  (shape={arr.shape}, total={n})"
        else:
            return f"{body}  (shape={arr.shape}, total={n})"

    outpath = Path(outfile)
    with outpath.open("w", encoding="utf-8") as f:
        f.write("# IO CHECK SUMMARY\n")
        f.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# Tech key: '{tech_key}' (index {tech_idx})\n")
        f.write(f"# Iterations: {n_iter}\n")
        f.write(f"# Print limits: max_items={max_items}, edge_items={edge_items}\n")
        f.write("#" + "-"*78 + "\n\n")

        for i in range(n_iter):
            # Inputs
            cap_rooftop = data_dt["MEWG"][i, tech_idx]
            demand_vec  = data_dt["MEWDH"][i, :, 0]
            std_rooftop = data_dt["METC"][i, tech_idx]
            price_vec   = data_dt["PRICH"][i, :, 0]

            # Outputs
            share_roof, share_grid = data_dt["household_shares"][i, 0], data_dt["household_shares"][i, 1]
            cost_roof, cost_grid   = data_dt["costs_household"][i, 0], data_dt["costs_household"][i, 1]
            std_roof_o, std_grid_o = data_dt["costs_household_std"][i, 0], data_dt["costs_household_std"][i, 1]

            f.write(f"=== Iteration {i+1} ===\n")
            f.write("INPUTS\n")
            f.write(f"  Rooftop Solar Capacity: {fmt_val(cap_rooftop)}\n")
            f.write(f"  Household Demand: {fmt_val(demand_vec)}\n")
            f.write(f"  Rooftop Solar Std: {fmt_val(std_rooftop)}\n")
            f.write(f"  Household Electricity Price: {fmt_val(price_vec)}\n")

            f.write("OUTPUTS\n")
            f.write(f"  Shares: rooftop={fmt_val(share_roof)}, grid={fmt_val(share_grid)}\n")
            f.write(f"  Costs:  rooftop={fmt_val(cost_roof)}, grid={fmt_val(cost_grid)}\n")
            f.write(f"  Cost Std: rooftop={fmt_val(std_roof_o)}, grid={fmt_val(std_grid_o)}\n")
            f.write("\n")

            print(f"Wrote iteration {i+1}/{n_iter}")

        f.write("# End of report\n")

if __name__ == "__main__":
    # Example: call with custom truncation if desired
    try:
        export_io_summary(
            data_dt, t2ti,
            tech_key="23 Rooftop Solar",
            outfile="io_check.txt",
            max_items=20,
            edge_items=5
        )
        print("Summary written to io_check.txt")
    except NameError:
        print("Define `data_dt` and `t2ti` before running this script, or import them here.")