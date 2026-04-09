# io_debug_export.py
import numpy as np
from datetime import datetime
from pathlib import Path

def export_io_summary(
    data_dt,
    t2ti,
    year,
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

    # outpath = Path(outfile)
    output_dir = Path("IO_Output")
    output_dir.mkdir(exist_ok=True)

    # Create year-specific filename
    year_outfile = f"{Path(outfile).stem}_{year}.txt"

    outpath = output_dir / year_outfile
    
    with outpath.open("w", encoding="utf-8") as f:
        f.write(f"# IO CHECK SUMMARY YEAR: {year}\n")
        f.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# Tech key: '{tech_key}' (index {tech_idx})\n")
        # f.write(f"# Year: {year}\n")
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

            # print(f"Wrote iteration {i+1}/{n_iter}")

        f.write("# End of report\n")


def export_rooftop_trace(
    data,
    data_dt,
    titles,
    year,
    first_year=None,
    region_key="34 USA (US)",
    tech_key="23 Rooftop Solar",
    outfile="rooftop_trace.txt",
):
    """Write a one-line year-end trace for rooftop solar in a single region."""

    if region_key not in titles["RTI"]:
        raise KeyError(f"'{region_key}' not found in titles['RTI']")
    if tech_key not in titles["T2TI"]:
        raise KeyError(f"'{tech_key}' not found in titles['T2TI']")

    region_idx = int(titles["RTI"].index(region_key))
    tech_idx = int(titles["T2TI"].index(tech_key))

    output_dir = Path("IO_Output")
    output_dir.mkdir(exist_ok=True)
    outpath = output_dir / Path(outfile).name

    mode = "w" if (first_year is not None and year == first_year) else "a"

    with outpath.open(mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# Rooftop trace generated {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"Year {year} | Region {region_key} | Tech {tech_key}\n")
        f.write(
            f"  MEWDH={data_dt['MEWDH'][region_idx, 0, 0]:,.6g} | "
            f"PRICH={data_dt['PRICH'][region_idx, 0, 0]:,.6g} | "
            f"METC={data_dt['METC'][region_idx, tech_idx, 0]:,.6g} | "
            f"MTCD={data_dt['MTCD'][region_idx, tech_idx, 0]:,.6g}\n"
        )
        cost_source = data if ('costs_household' in data and 'costs_household_std' in data) else data_dt
        if 'costs_household' in cost_source and 'costs_household_std' in cost_source:
            f.write(
                f"  cost_roof={cost_source['costs_household'][region_idx, 0, 0]:,.6g} | "
                f"cost_grid={cost_source['costs_household'][region_idx, 1, 0]:,.6g} | "
                f"std_roof={cost_source['costs_household_std'][region_idx, 0, 0]:,.6g} | "
                f"std_grid={cost_source['costs_household_std'][region_idx, 1, 0]:,.6g}\n"
            )
        f.write(
            f"  share={data['household_shares'][region_idx, 0, 0]:,.6g} | "
            f"MEWG={data['MEWG'][region_idx, tech_idx, 0]:,.6g} | "
            f"MEWL={data['MEWL'][region_idx, tech_idx, 0]:,.6g} | "
            f"MEWK={data['MEWK'][region_idx, tech_idx, 0]:,.6g} | "
            f"MEWS={data['MEWS'][region_idx, tech_idx, 0]:,.6g}\n"
        )

        rooftop_cap = float(data['MEWK'][region_idx, tech_idx, 0])
        total_cap = float(np.sum(data['MEWK'][region_idx, :, 0]))
        non_rooftop_cap = total_cap - rooftop_cap

        f.write(
            f"  total_cap={total_cap:,.6g} | "
            f"non_rooftop_cap={non_rooftop_cap:,.6g} | "
            f"rooftop_cap_share={rooftop_cap / total_cap if total_cap > 0 else 0.0:,.6g}\n"
        )

        top_n = min(3, len(titles['T2TI']))
        top_idx = np.argsort(data['MEWK'][region_idx, :, 0])[::-1][:top_n]
        top_caps = ", ".join(
            [f"{titles['T2TI'][int(i)]}={data['MEWK'][region_idx, int(i), 0]:,.6g}" for i in top_idx]
        )
        f.write(f"  top_capacity_techs: {top_caps}\n")
        f.write("\n")

#if __name__ == "__main__":
    # Example: call with custom truncation if desired
#    try:
#        export_io_summary(
#            data_dt, t2ti, year,
#            tech_key="23 Rooftop Solar",
#            outfile="io_check.txt",
#            max_items=20,
#            edge_items=5
#        )
#        print("Summary written to io_check.txt")
#    except NameError:
#        print("Define `data_dt` and `t2ti` before running this script, or import them here.")