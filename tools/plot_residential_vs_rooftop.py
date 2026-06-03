import csv
from pathlib import Path
import math
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OLD_CSV = ROOT / "IO_Output" / "sensitivity_analysis" / "all_runs_timeseries.csv"
NEW_CSV = ROOT / "IO_Output" / "rooftop_trace_au_series.csv"
OUT_DIR = ROOT / "IO_Output" / "sensitivity_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_old(path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("run") != "base":
                continue
            y = int(r["year"])
            rows[y] = {
                "MEWDH": float(r.get("MEWDH", math.nan)),
                "MEWG": float(r.get("MEWG", math.nan)),
                "MEWK": float(r.get("MEWK", math.nan)),
                "rooftop_cap_share": float(r.get("rooftop_cap_share", math.nan)),
                "non_rooftop_cap": float(r.get("non_rooftop_cap", math.nan)),
                "total_cap": float(r.get("total_cap", math.nan)),
            }
    return rows


def read_new(path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            y = int(r["Year"])
            rows[y] = {
                "MEWDH": float(r.get("PRICH", math.nan)),
                # Note: rooftop extract file uses different columns; MEWG in that file is rooftop generation
                "MEWG": float(r.get("MEWG", math.nan)),
                "MEWK": float(r.get("MEWK", math.nan)),
                "rooftop_cap_share": float(r.get("rooftop_cap_share", math.nan)),
            }
    return rows


def main():
    old = read_old(OLD_CSV)
    new = read_new(NEW_CSV)

    years = sorted(set(old.keys()) & set(new.keys()))

    out_rows = []
    years_list = []
    res_demand = []
    rooftop_gen = []
    est_total_gen = []

    for y in years:
        # Residential demand: MEWDH in the traces is in ktoe-like units; convert by *11.63 to GWh (as used in model)
        # The rooftop extract used MEWDH separately; here we use old MEWDH when available
        mewdh = old[y]["MEWDH"]
        res_gwh = mewdh * 11.63
        rg = new[y]["MEWG"] if not math.isnan(new[y]["MEWG"]) else old[y]["MEWG"]
        cap_share = new[y].get("rooftop_cap_share", old[y].get("rooftop_cap_share", math.nan))
        # Estimate total generation by dividing rooftop generation by rooftop cap share (approx)
        if cap_share and cap_share > 0:
            total_gen_est = rg / cap_share
        else:
            total_gen_est = math.nan

        years_list.append(y)
        res_demand.append(res_gwh)
        rooftop_gen.append(rg)
        est_total_gen.append(total_gen_est)

        out_rows.append({
            "year": y,
            "residential_demand_gwh": res_gwh,
            "rooftop_generation_gwh": rg,
            "est_total_generation_gwh": total_gen_est,
        })

    # Write CSV
    csv_out = OUT_DIR / "residential_vs_rooftop_timeseries.csv"
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["year", "residential_demand_gwh", "rooftop_generation_gwh", "est_total_generation_gwh"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(years_list, res_demand, label="Residential demand (GWh)", linewidth=2)
    plt.plot(years_list, rooftop_gen, label="Rooftop generation (GWh)", linewidth=2)
    plt.plot(years_list, est_total_gen, label="Estimated total generation (GWh)", linewidth=2, linestyle='--')
    plt.xlabel("Year")
    plt.ylabel("GWh")
    plt.title("Australia: Residential demand vs Rooftop generation vs Estimated total generation")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    png_out = OUT_DIR / "residential_vs_rooftop_timeseries.png"
    plt.savefig(png_out, dpi=180)
    plt.close()

    print(f"Wrote {csv_out}")
    print(f"Wrote {png_out}")


if __name__ == '__main__':
    main()
