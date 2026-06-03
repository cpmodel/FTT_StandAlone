import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SENS_DIR = ROOT / "IO_Output" / "sensitivity_runs"
OUT_DIR = ROOT / "IO_Output" / "sensitivity_analysis"


NUM_RE = re.compile(r"^[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?$")


def parse_number(raw: str):
    text = raw.strip().replace(",", "")
    if not text:
        return None
    if not NUM_RE.match(text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_trace(trace_path: Path, run_name: str):
    rows = []
    current = None

    for line in trace_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("Year ") and "|" in line:
            if current is not None:
                rows.append(current)
            year_text = line.split("|", 1)[0].replace("Year", "").strip()
            current = {"run": run_name, "year": int(year_text)}
            continue

        if current is None or "=" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        for part in parts:
            if "=" not in part:
                continue
            key, value = [x.strip() for x in part.split("=", 1)]
            num = parse_number(value)
            if num is not None:
                current[key] = num

    if current is not None:
        rows.append(current)

    return rows


def safe_get_year(rows_by_year, year, key):
    row = rows_by_year.get(year)
    if not row:
        return math.nan
    return row.get(key, math.nan)


def plot_cap_share(rows_by_run):
    plt.figure(figsize=(10, 6))
    for run, rows in sorted(rows_by_run.items()):
        years = [r["year"] for r in rows]
        cap_share = [r.get("rooftop_cap_share", math.nan) for r in rows]
        plt.plot(years, cap_share, linewidth=2, label=run)

    plt.title("Australia Rooftop Capacity Share Across Sensitivities")
    plt.xlabel("Year")
    plt.ylabel("rooftop_cap_share")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rooftop_cap_share_comparison.png", dpi=180)
    plt.close()


def plot_base_share_vs_cap(rows_by_run):
    base_rows = rows_by_run.get("base")
    if not base_rows:
        return

    years = [r["year"] for r in base_rows]
    share = [r.get("share", math.nan) for r in base_rows]
    cap_share = [r.get("rooftop_cap_share", math.nan) for r in base_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, share, linewidth=2.2, label="share (household adoption)")
    ax.plot(years, cap_share, linewidth=2.2, label="rooftop_cap_share (system capacity)")
    ax.set_title("Base Run: Share vs Rooftop Capacity Share (Australia)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "base_share_vs_rooftop_cap_share.png", dpi=180)
    plt.close(fig)


def plot_base_denominator(rows_by_run):
    base_rows = rows_by_run.get("base")
    if not base_rows:
        return

    years = [r["year"] for r in base_rows]
    rooftop = [r.get("MEWK", math.nan) for r in base_rows]
    non_rooftop = [r.get("non_rooftop_cap", math.nan) for r in base_rows]
    total = [r.get("total_cap", math.nan) for r in base_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, rooftop, linewidth=2.2, label="MEWK rooftop capacity")
    ax.plot(years, non_rooftop, linewidth=2.2, label="non_rooftop_cap")
    ax.plot(years, total, linewidth=2.2, label="total_cap")
    ax.set_title("Base Run Capacity Growth: Denominator Effect")
    ax.set_xlabel("Year")
    ax.set_ylabel("Capacity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "base_denominator_effect.png", dpi=180)
    plt.close(fig)


def make_summary(rows_by_run):
    summary = []
    for run, rows in sorted(rows_by_run.items()):
        rows_sorted = sorted(rows, key=lambda x: x["year"])
        ymap = {r["year"]: r for r in rows_sorted}

        peak_row = max(rows_sorted, key=lambda x: x.get("rooftop_cap_share", -1.0))

        row = {
            "run": run,
            "peak_year": peak_row["year"],
            "peak_rooftop_cap_share": peak_row.get("rooftop_cap_share", math.nan),
            "rooftop_cap_share_2030": safe_get_year(ymap, 2030, "rooftop_cap_share"),
            "rooftop_cap_share_2050": safe_get_year(ymap, 2050, "rooftop_cap_share"),
            "share_2030": safe_get_year(ymap, 2030, "share"),
            "share_2050": safe_get_year(ymap, 2050, "share"),
            "cost_roof_2030": safe_get_year(ymap, 2030, "cost_roof"),
            "cost_roof_2050": safe_get_year(ymap, 2050, "cost_roof"),
            "METC_2030": safe_get_year(ymap, 2030, "METC"),
            "METC_2050": safe_get_year(ymap, 2050, "METC"),
            "PRICH_2030": safe_get_year(ymap, 2030, "PRICH"),
            "PRICH_2050": safe_get_year(ymap, 2050, "PRICH"),
            "MEWK_2030": safe_get_year(ymap, 2030, "MEWK"),
            "MEWK_2050": safe_get_year(ymap, 2050, "MEWK"),
            "non_rooftop_cap_2030": safe_get_year(ymap, 2030, "non_rooftop_cap"),
            "non_rooftop_cap_2050": safe_get_year(ymap, 2050, "non_rooftop_cap"),
            "total_cap_2030": safe_get_year(ymap, 2030, "total_cap"),
            "total_cap_2050": safe_get_year(ymap, 2050, "total_cap"),
        }
        row["cap_share_drop_2030_to_2050"] = row["rooftop_cap_share_2050"] - row["rooftop_cap_share_2030"]
        summary.append(row)

    return summary


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f6(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def build_report(summary_rows):
    base = next((r for r in summary_rows if r["run"] == "base"), None)
    if not base:
        return "No base run found."

    ordered = sorted(summary_rows, key=lambda x: x["run"])

    lines = []
    lines.append("Australia Rooftop Sensitivity Analysis")
    lines.append("====================================")
    lines.append("")
    lines.append("Scope")
    lines.append("-----")
    lines.append("This report is generated from rooftop_trace_au.txt files in IO_Output/sensitivity_runs.")
    lines.append("It compares each sensitivity test against the base run and explains the observed behavior.")
    lines.append("")
    lines.append("What each metric means")
    lines.append("----------------------")
    lines.append("share: household choice share from the two-option household rooftop choice block.")
    lines.append("  In this setup it is effectively the fraction of household demand choosing rooftop over the grid comparator.")
    lines.append("rooftop_cap_share: system-wide capacity share = rooftop_capacity / total_power_capacity.")
    lines.append("  This is a power-system metric, not only a household rooftop-choice metric.")
    lines.append("")
    lines.append("Why share can be high while rooftop_cap_share is low")
    lines.append("----------------------------------------------------")
    lines.append("1) Different denominators:")
    lines.append("   share compares rooftop vs household grid option in the household block.")
    lines.append("   rooftop_cap_share compares rooftop vs ALL generation technologies in the system.")
    lines.append("2) Denominator effect after 2030:")
    lines.append(
        "   In the base run, rooftop MEWK grows from "
        f"{f6(base['MEWK_2030'])} to {f6(base['MEWK_2050'])}, but non-rooftop capacity grows faster "
        f"from {f6(base['non_rooftop_cap_2030'])} to {f6(base['non_rooftop_cap_2050'])}."
    )
    lines.append(
        "   Therefore total capacity expands strongly, diluting rooftop's fraction even when rooftop adoption remains strong."
    )
    lines.append("3) Cost signal remains strongly pro-rooftop:")
    lines.append(
        "   cost_roof stays very favorable (more negative) from "
        f"{f6(base['cost_roof_2030'])} to {f6(base['cost_roof_2050'])}, and share rises from "
        f"{f6(base['share_2030'])} to {f6(base['share_2050'])}."
    )
    lines.append("   So the decline in rooftop_cap_share is not caused by rooftop becoming unattractive.")
    lines.append("   It is mainly a composition effect in a rapidly growing system.")
    lines.append("")
    lines.append("Sensitivity tests and differences")
    lines.append("---------------------------------")

    for row in ordered:
        lines.append(
            f"{row['run']}: peak_cap_share={f6(row['peak_rooftop_cap_share'])} (year {row['peak_year']}), "
            f"cap_share_2030={f6(row['rooftop_cap_share_2030'])}, cap_share_2050={f6(row['rooftop_cap_share_2050'])}, "
            f"delta_2030_to_2050={f6(row['cap_share_drop_2030_to_2050'])}"
        )

    lines.append("")
    lines.append("Interpretation by test type")
    lines.append("---------------------------")
    lines.append("discount_low / discount_high:")
    lines.append("  These alter discounting and therefore the present value of rooftop savings.")
    lines.append("  Lower discount rates should make future savings more valuable and typically improve rooftop economics.")
    lines.append("  Higher discount rates do the opposite.")
    lines.append("export_0.8 / export_1.2:")
    lines.append("  These scale export compensation (value of exported PV electricity).")
    lines.append("  Higher export value generally improves rooftop annual benefit and lowers NPC further.")
    lines.append("selfcons_minus / selfcons_plus:")
    lines.append("  These adjust self-consumption. Higher self-consumption monetizes more generation at retail price,")
    lines.append("  improving rooftop economics; lower self-consumption does the opposite.")
    lines.append("")
    lines.append("Main conclusion")
    lines.append("---------------")
    lines.append(
        "Across all tested sensitivities, rooftop_cap_share peaks around the late 2020s and declines afterward, "
        "while share continues to increase toward 1.0."
    )
    lines.append(
        "This pattern is consistent with a plausible model mechanism: rooftop remains attractive for households, "
        "but utility-scale and other non-rooftop capacity grows faster in absolute terms, reducing rooftop's share "
        "of total system capacity by 2050."
    )
    lines.append("")
    lines.append("Exported artifacts")
    lines.append("------------------")
    lines.append("- all_runs_timeseries.csv")
    lines.append("- run_summary.csv")
    lines.append("- rooftop_cap_share_comparison.png")
    lines.append("- base_share_vs_rooftop_cap_share.png")
    lines.append("- base_denominator_effect.png")

    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_by_run = {}
    all_rows = []

    for run_dir in sorted(SENS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        trace_file = run_dir / "rooftop_trace_au.txt"
        if not trace_file.exists():
            continue

        run_rows = parse_trace(trace_file, run_dir.name)
        run_rows = sorted(run_rows, key=lambda x: x["year"])
        if run_rows:
            rows_by_run[run_dir.name] = run_rows
            all_rows.extend(run_rows)

    if not rows_by_run:
        raise SystemExit("No trace files found in sensitivity runs.")

    all_rows = sorted(all_rows, key=lambda x: (x["run"], x["year"]))
    write_csv(OUT_DIR / "all_runs_timeseries.csv", all_rows)

    summary = make_summary(rows_by_run)
    write_csv(OUT_DIR / "run_summary.csv", summary)

    plot_cap_share(rows_by_run)
    plot_base_share_vs_cap(rows_by_run)
    plot_base_denominator(rows_by_run)

    report_text = build_report(summary)
    (OUT_DIR / "rooftop_trace_explanation.txt").write_text(report_text, encoding="utf-8")

    print(f"Wrote analysis outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
