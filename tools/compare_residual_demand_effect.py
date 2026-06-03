import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OLD_BASE_CSV = ROOT / "IO_Output" / "sensitivity_analysis" / "all_runs_timeseries.csv"
NEW_BASE_CSV = ROOT / "IO_Output" / "rooftop_trace_au_series.csv"
OUT_DIR = ROOT / "IO_Output" / "sensitivity_analysis"


def read_old_base(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("run") != "base":
                continue
            year = int(r["year"])
            rows[year] = {
                "share": float(r["share"]),
                "rooftop_cap_share": float(r["rooftop_cap_share"]),
                "MEWK": float(r["MEWK"]),
                "total_cap": float(r["total_cap"]),
                "non_rooftop_cap": float(r["non_rooftop_cap"]),
            }
    return rows


def read_new_base(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            year = int(r["Year"])
            mewk = float(r["MEWK"])
            cap_share = float(r["rooftop_cap_share"])
            total_cap = mewk / cap_share if cap_share > 0 else 0.0
            rows[year] = {
                "share": float(r["share"]),
                "rooftop_cap_share": cap_share,
                "MEWK": mewk,
                "total_cap": total_cap,
                "non_rooftop_cap": total_cap - mewk,
            }
    return rows


def write_comparison_csv(old_rows, new_rows):
    years = sorted(set(old_rows) & set(new_rows))
    out_path = OUT_DIR / "residual_demand_impact_comparison.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "year",
            "old_rooftop_cap_share",
            "new_rooftop_cap_share",
            "delta_rooftop_cap_share",
            "old_share",
            "new_share",
            "old_total_cap",
            "new_total_cap",
            "delta_total_cap",
            "old_non_rooftop_cap",
            "new_non_rooftop_cap",
            "delta_non_rooftop_cap",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for y in years:
            old = old_rows[y]
            new = new_rows[y]
            writer.writerow(
                {
                    "year": y,
                    "old_rooftop_cap_share": old["rooftop_cap_share"],
                    "new_rooftop_cap_share": new["rooftop_cap_share"],
                    "delta_rooftop_cap_share": new["rooftop_cap_share"] - old["rooftop_cap_share"],
                    "old_share": old["share"],
                    "new_share": new["share"],
                    "old_total_cap": old["total_cap"],
                    "new_total_cap": new["total_cap"],
                    "delta_total_cap": new["total_cap"] - old["total_cap"],
                    "old_non_rooftop_cap": old["non_rooftop_cap"],
                    "new_non_rooftop_cap": new["non_rooftop_cap"],
                    "delta_non_rooftop_cap": new["non_rooftop_cap"] - old["non_rooftop_cap"],
                }
            )
    return out_path


def plot_comparison(old_rows, new_rows):
    years = sorted(set(old_rows) & set(new_rows))
    old_cap = [old_rows[y]["rooftop_cap_share"] for y in years]
    new_cap = [new_rows[y]["rooftop_cap_share"] for y in years]

    plt.figure(figsize=(10, 6))
    plt.plot(years, old_cap, label="old base", linewidth=2.0)
    plt.plot(years, new_cap, label="residual-demand base", linewidth=2.0)
    plt.title("Australia Rooftop Capacity Share: Before vs Residual-Demand Fix")
    plt.xlabel("Year")
    plt.ylabel("rooftop_cap_share")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path = OUT_DIR / "residual_demand_impact_plot.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def write_text_summary(old_rows, new_rows):
    y0 = 2030
    y1 = 2050
    old_2030 = old_rows[y0]["rooftop_cap_share"]
    old_2050 = old_rows[y1]["rooftop_cap_share"]
    new_2030 = new_rows[y0]["rooftop_cap_share"]
    new_2050 = new_rows[y1]["rooftop_cap_share"]

    old_drop = old_2050 - old_2030
    new_drop = new_2050 - new_2030

    lines = []
    lines.append("Residual-Demand Allocation Impact (Australia Rooftop)")
    lines.append("===================================================")
    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append("The new run allocates utility generation to residual demand after household rooftop")
    lines.append("generation is deducted. This reduces denominator inflation in total system capacity.")
    lines.append("")
    lines.append("Key numbers")
    lines.append("-----------")
    lines.append(f"Old base rooftop_cap_share 2030: {old_2030:.6f}")
    lines.append(f"Old base rooftop_cap_share 2050: {old_2050:.6f}")
    lines.append(f"Old base change 2030->2050: {old_drop:.6f}")
    lines.append("")
    lines.append(f"New base rooftop_cap_share 2030: {new_2030:.6f}")
    lines.append(f"New base rooftop_cap_share 2050: {new_2050:.6f}")
    lines.append(f"New base change 2030->2050: {new_drop:.6f}")
    lines.append("")
    lines.append(f"2050 improvement in rooftop_cap_share: {new_2050 - old_2050:.6f}")
    lines.append(f"Reduction in decline magnitude: {abs(old_drop) - abs(new_drop):.6f}")
    lines.append("")
    lines.append("Conclusion")
    lines.append("----------")
    lines.append("If the decline becomes visibly less steep in the new run, that supports the hypothesis")
    lines.append("that the previous drop was dominated by system denominator growth rather than only")
    lines.append("household rooftop saturation.")

    out_path = OUT_DIR / "residual_demand_impact.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    old_rows = read_old_base(OLD_BASE_CSV)
    new_rows = read_new_base(NEW_BASE_CSV)

    csv_out = write_comparison_csv(old_rows, new_rows)
    plot_out = plot_comparison(old_rows, new_rows)
    txt_out = write_text_summary(old_rows, new_rows)

    print(f"Wrote {csv_out}")
    print(f"Wrote {plot_out}")
    print(f"Wrote {txt_out}")


if __name__ == "__main__":
    main()
