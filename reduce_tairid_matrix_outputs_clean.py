#!/usr/bin/env python3
"""
Clean reducer for TAIRID combined matrix scan.

Reads all case_summary.json files from downloaded GitHub Actions artifacts,
compares each case to the CDM baseline, and ranks the candidates.

Boundary:
Internal CLASS proxy matrix scan only.
Not a final likelihood.
Not proof of TAIRID cosmology.
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ELL_TT_MIN = 30
ELL_TT_MAX = 2000
ELL_LENSING_MIN = 40
ELL_LENSING_MAX = 1000


def read_csv_dicts(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_numeric_csv(path):
    rows = read_csv_dicts(path)
    if not rows:
        return {}

    columns = {}
    for key in rows[0].keys():
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except Exception:
                values.append(float("nan"))
        columns[key] = np.asarray(values, dtype=float)

    return columns


def interp_log_x(x, y, x_target):
    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if np.sum(good) < 2:
        return float("nan")
    return float(np.interp(math.log(x_target), np.log(x[good]), y[good]))


def curve_drift(base_x, base_y, case_x, case_y, x_min, x_max):
    if not np.array_equal(base_x, case_x):
        case_y = np.interp(base_x, case_x, case_y)

    mask = (
        (base_x >= x_min)
        & (base_x <= x_max)
        & np.isfinite(base_y)
        & np.isfinite(case_y)
        & (base_y > 0)
    )

    if np.sum(mask) < 2:
        return float("nan"), float("nan"), float("nan")

    ratio = case_y[mask] / base_y[mask]
    drift = ratio - 1.0

    rms = float(np.sqrt(np.nanmean(drift * drift)))
    max_abs = float(np.nanmax(np.abs(drift)))
    mean_ratio = float(np.nanmean(ratio))

    return mean_ratio, rms, max_abs


def bao_max_drift(base_rows, case_rows):
    base_by_z = {float(row["z"]): row for row in base_rows}

    max_drift = 0.0

    for row in case_rows:
        z = float(row["z"])
        if z not in base_by_z:
            continue

        base = base_by_z[z]

        for key in ["D_M_over_rd", "D_H_over_rd", "D_V_over_rd"]:
            ratio = float(row[key]) / float(base[key])
            max_drift = max(max_drift, abs(ratio - 1.0))

    return float(max_drift)


def classify(row):
    if row["case"] == "cdm_baseline":
        return "baseline"

    peaks_ok = (
        row["ell_peak1"] == 221
        and row["ell_peak2"] == 538
        and row["ell_peak3"] == 815
    )

    if not peaks_ok:
        return "fail_peak_shift"

    if (
        row["S8"] <= 0.833
        and row["growth_delta_chi2_vs_cdm"] <= 0
        and row["pk_ratio_k10"] >= 0.80
        and row["lensing_max_abs_drift"] <= 0.05
        and row["tt_max_abs_drift"] <= 0.01
        and row["bao_max_abs_drift"] <= 0.0025
    ):
        return "best_balanced_frontier"

    if (
        row["S8"] <= 0.835
        and row["growth_delta_chi2_vs_cdm"] <= 0
        and row["pk_ratio_k10"] >= 0.75
        and row["lensing_max_abs_drift"] <= 0.06
        and row["tt_max_abs_drift"] <= 0.015
        and row["bao_max_abs_drift"] <= 0.005
    ):
        return "balanced_candidate"

    if row["lensing_max_abs_drift"] > 0.08:
        return "lensing_too_suppressed"

    if row["pk_ratio_k10"] < 0.65:
        return "high_k_too_suppressed"

    if row["growth_delta_chi2_vs_cdm"] > 0:
        return "growth_not_improved"

    return "mixed"


def score(row):
    if row["case"] == "cdm_baseline":
        return 999.0

    target_s8 = 0.830
    value = 0.0

    value += 80.0 * abs(row["S8"] - target_s8)
    value += 8.0 * max(0.0, row["lensing_max_abs_drift"] - 0.05)
    value += 4.0 * max(0.0, 0.80 - row["pk_ratio_k10"])
    value += 5.0 * max(0.0, row["growth_delta_chi2_vs_cdm"])
    value += 20.0 * max(0.0, row["tt_max_abs_drift"] - 0.01)
    value += 20.0 * max(0.0, row["bao_max_abs_drift"] - 0.0025)

    return float(value)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: reduce_tairid_matrix_outputs_clean.py <artifact_root>")

    artifact_root = Path(sys.argv[1])
    outdir = Path("combined_matrix_clean_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in artifact_root.rglob("case_summary.json"):
        data = json.loads(path.read_text())
        data["_folder"] = str(path.parent)
        summaries.append(data)

    if not summaries:
        raise RuntimeError("No case_summary.json files found.")

    by_name = {item["name"]: item for item in summaries}

    if "cdm_baseline" not in by_name:
        raise RuntimeError("Missing cdm_baseline case.")

    baseline = by_name["cdm_baseline"]
    baseline_folder = Path(baseline["_folder"])

    base_pk = read_numeric_csv(baseline_folder / "pk_z0.csv")
    base_tt = read_numeric_csv(baseline_folder / "cmb_tt_spectrum.csv")
    base_lensing = read_numeric_csv(baseline_folder / "cmb_lensing_curve.csv")
    base_bao = read_csv_dicts(baseline_folder / "bao_distances.csv")

    baseline_s8 = float(baseline["S8"])
    baseline_growth_chi2 = float(baseline["growth_chi2"])

    rows = []

    for item in summaries:
        if item.get("status") != "success":
            rows.append(
                {
                    "case": item["name"],
                    "status": "failed",
                    "diagnostic_flag": "failed",
                    "error": item.get("error", ""),
                }
            )
            continue

        folder = Path(item["_folder"])
        name = item["name"]
        peaks = item.get("peak_ell", [0, 0, 0])

        case_pk = read_numeric_csv(folder / "pk_z0.csv")
        pk_ratio = case_pk["Pk_z0"] / base_pk["Pk_z0"]

        case_tt = read_numeric_csv(folder / "cmb_tt_spectrum.csv")
        _, tt_rms, tt_max = curve_drift(
            base_tt["ell"],
            base_tt["D_ell_TT"],
            case_tt["ell"],
            case_tt["D_ell_TT"],
            ELL_TT_MIN,
            ELL_TT_MAX,
        )

        case_lensing = read_numeric_csv(folder / "cmb_lensing_curve.csv")
        lens_mean, lens_rms, lens_max = curve_drift(
            base_lensing["ell"],
            base_lensing["lensing_scaled"],
            case_lensing["ell"],
            case_lensing["lensing_scaled"],
            ELL_LENSING_MIN,
            ELL_LENSING_MAX,
        )

        case_bao = read_csv_dicts(folder / "bao_distances.csv")
        bao_drift = bao_max_drift(base_bao, case_bao)

        row = {
            "case": name,
            "status": "success",
            "warm_fraction": float(item["warm_fraction"]),
            "m_ncdm_eV": item["m_ncdm_eV"],
            "S8": float(item["S8"]),
            "growth_chi2": float(item["growth_chi2"]),
            "growth_delta_chi2_vs_cdm": float(item["growth_chi2"] - baseline_growth_chi2),
            "pk_ratio_k1": interp_log_x(base_pk["k_1_per_Mpc"], pk_ratio, 1.0),
            "pk_ratio_k5": interp_log_x(base_pk["k_1_per_Mpc"], pk_ratio, 5.0),
            "pk_ratio_k10": interp_log_x(base_pk["k_1_per_Mpc"], pk_ratio, 10.0),
            "lensing_mean_ratio": lens_mean,
            "lensing_rms_drift": lens_rms,
            "lensing_max_abs_drift": lens_max,
            "tt_rms_drift": tt_rms,
            "tt_max_abs_drift": tt_max,
            "bao_max_abs_drift": bao_drift,
            "ell_peak1": int(peaks[0]),
            "ell_peak2": int(peaks[1]),
            "ell_peak3": int(peaks[2]),
            "error": "",
        }

        row["diagnostic_flag"] = classify(row)
        row["combined_score"] = score(row)

        rows.append(row)

    rows = sorted(
        rows,
        key=lambda row: (
            row.get("diagnostic_flag") not in ["best_balanced_frontier", "balanced_candidate"],
            row.get("combined_score", 9999.0),
            row.get("S8", 999.0),
        ),
    )

    output = {
        "boundary": "Internal CLASS proxy matrix scan only. Not proof of TAIRID cosmology.",
        "baseline_S8": baseline_s8,
        "baseline_growth_chi2": baseline_growth_chi2,
        "case_count": len(rows),
        "best_case": rows[0],
        "rows": rows,
    }

    (outdir / "combined_matrix_clean_summary.json").write_text(json.dumps(output, indent=2))

    header = [
        "rank",
        "case",
        "status",
        "diagnostic_flag",
        "combined_score",
        "warm_fraction",
        "m_ncdm_eV",
        "S8",
        "growth_chi2",
        "growth_delta_chi2_vs_cdm",
        "pk_ratio_k1",
        "pk_ratio_k5",
        "pk_ratio_k10",
        "lensing_mean_ratio",
        "lensing_max_abs_drift",
        "tt_max_abs_drift",
        "bao_max_abs_drift",
        "ell_peak1",
        "ell_peak2",
        "ell_peak3",
        "error",
    ]

    with open(outdir / "combined_matrix_clean_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rank, row in enumerate(rows, start=1):
            writer.writerow([rank] + [row.get(key, "") for key in header[1:]])

    candidates = [
        row for row in rows
        if row.get("diagnostic_flag") in ["best_balanced_frontier", "balanced_candidate"]
    ]

    with open(outdir / "combined_matrix_clean_candidates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rank, row in enumerate(candidates, start=1):
            writer.writerow([rank] + [row.get(key, "") for key in header[1:]])

    success_rows = [
        row for row in rows
        if row.get("status") == "success" and row.get("case") != "cdm_baseline"
    ]

    if success_rows:
        labels = [row["case"] for row in success_rows]
        s8_values = [row["S8"] for row in success_rows]
        lensing_values = [row["lensing_max_abs_drift"] for row in success_rows]
        pk10_values = [row["pk_ratio_k10"] for row in success_rows]
        scores = [row["combined_score"] for row in success_rows]

        plt.figure(figsize=(10, 6))
        plt.scatter(lensing_values, s8_values)
        for label, x, y in zip(labels, lensing_values, s8_values):
            if x <= 0.06 and y <= 0.835:
                plt.annotate(label, (x, y), fontsize=7)
        plt.axhline(0.833, linewidth=1)
        plt.axvline(0.05, linewidth=1)
        plt.axvline(0.06, linewidth=1)
        plt.xlabel("max CMB lensing drift versus CDM")
        plt.ylabel("S8")
        plt.title("Clean matrix: S8 versus CMB lensing drift")
        plt.tight_layout()
        plt.savefig(outdir / "clean_matrix_s8_vs_lensing.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(pk10_values, s8_values)
        for label, x, y in zip(labels, pk10_values, s8_values):
            if x >= 0.75 and y <= 0.835:
                plt.annotate(label, (x, y), fontsize=7)
        plt.axhline(0.833, linewidth=1)
        plt.axvline(0.80, linewidth=1)
        plt.axvline(0.75, linewidth=1)
        plt.xlabel("P(k=10)/P_CDM")
        plt.ylabel("S8")
        plt.title("Clean matrix: S8 versus high-k survival")
        plt.tight_layout()
        plt.savefig(outdir / "clean_matrix_s8_vs_pk10.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(scores, s8_values)
        for label, x, y in zip(labels, scores, s8_values):
            if x <= 1.5:
                plt.annotate(label, (x, y), fontsize=7)
        plt.xlabel("combined score, lower is better")
        plt.ylabel("S8")
        plt.title("Clean matrix combined score")
        plt.tight_layout()
        plt.savefig(outdir / "clean_matrix_score_plot.png", dpi=160)
        plt.close()

    print("")
    print("Clean combined matrix reduction complete.")
    print("Created:")
    print("  combined_matrix_clean_outputs/combined_matrix_clean_summary.json")
    print("  combined_matrix_clean_outputs/combined_matrix_clean_summary.csv")
    print("  combined_matrix_clean_outputs/combined_matrix_clean_candidates.csv")
    print("  combined_matrix_clean_outputs/clean_matrix_s8_vs_lensing.png")
    print("  combined_matrix_clean_outputs/clean_matrix_s8_vs_pk10.png")
    print("  combined_matrix_clean_outputs/clean_matrix_score_plot.png")
    print("")
    print("Best case:")
    print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
