#!/usr/bin/env python3
"""
Aggregate TAIRID combined matrix scan outputs.

Purpose:
Combine all per-case artifacts into one ranked matrix summary.

The reducer compares every case to the CDM baseline across:
- S8
- observed f_sigma8 growth chi-square
- high-k P(k) survival
- CMB TT shape drift
- CMB lensing drift
- BAO distance-ratio drift
- CMB peak-position stability

Boundary:
This is an internal proxy audit.
It is not a final cosmology likelihood.
It does not prove TAIRID cosmology.
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
K_REPORT = [1.0, 5.0, 10.0]


def read_csv_dicts(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_numeric_csv(path):
    rows = read_csv_dicts(path)
    columns = {}

    if not rows:
        return columns

    for key in rows[0].keys():
        values = []
        for row in rows:
            value = row[key]
            if value == "" or value is None:
                values.append(float("nan"))
            else:
                try:
                    values.append(float(value))
                except Exception:
                    values.append(float("nan"))
        columns[key] = np.asarray(values, dtype=float)

    return columns


def interp_log_x(x, y, x_target):
    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if np.sum(good) < 2:
        return float("nan")
    return float(np.interp(math.log(x_target), np.log(x[good]), y[good]))


def curve_ratio_metrics(base_x, base_y, case_x, case_y, x_min, x_max):
    if len(base_x) == 0 or len(case_x) == 0:
        return {
            "mean_ratio": float("nan"),
            "rms_drift": float("nan"),
            "max_abs_drift": float("nan"),
            "mean_abs_drift": float("nan"),
        }

    if not np.array_equal(base_x, case_x):
        case_interp = np.interp(base_x, case_x, case_y)
    else:
        case_interp = case_y

    mask = (
        (base_x >= x_min)
        & (base_x <= x_max)
        & np.isfinite(base_y)
        & np.isfinite(case_interp)
        & (base_y > 0)
    )

    ratio = np.full_like(base_y, np.nan, dtype=float)
    ratio[mask] = case_interp[mask] / base_y[mask]

    diff = ratio[mask] - 1.0
    abs_diff = np.abs(diff)

    if len(diff) == 0:
        return {
            "mean_ratio": float("nan"),
            "rms_drift": float("nan"),
            "max_abs_drift": float("nan"),
            "mean_abs_drift": float("nan"),
        }

    return {
        "mean_ratio": float(np.nanmean(ratio[mask])),
        "rms_drift": float(np.sqrt(np.nanmean(diff * diff))),
        "max_abs_drift": float(np.nanmax(abs_diff)),
        "mean_abs_drift": float(np.nanmean(abs_diff)),
    }


def bao_drift_metrics(base_rows, case_rows):
    if not base_rows or not case_rows:
        return {
            "bao_max_abs_drift": float("nan"),
            "bao_DM_max_abs_drift": float("nan"),
            "bao_DH_max_abs_drift": float("nan"),
            "bao_DV_max_abs_drift": float("nan"),
        }

    base_by_z = {float(row["z"]): row for row in base_rows}

    max_dm = 0.0
    max_dh = 0.0
    max_dv = 0.0

    for row in case_rows:
        z = float(row["z"])
        if z not in base_by_z:
            continue

        base = base_by_z[z]

        dm_ratio = float(row["D_M_over_rd"]) / float(base["D_M_over_rd"])
        dh_ratio = float(row["D_H_over_rd"]) / float(base["D_H_over_rd"])
        dv_ratio = float(row["D_V_over_rd"]) / float(base["D_V_over_rd"])

        max_dm = max(max_dm, abs(dm_ratio - 1.0))
        max_dh = max(max_dh, abs(dh_ratio - 1.0))
        max_dv = max(max_dv, abs(dv_ratio - 1.0))

    return {
        "bao_max_abs_drift": float(max(max_dm, max_dh, max_dv)),
        "bao_DM_max_abs_drift": float(max_dm),
        "bao_DH_max_abs_drift": float(max_dh),
        "bao_DV_max_abs_drift": float(max_dv),
    }


def classify(row):
    if row["case"] == "cdm_baseline":
        return "baseline"

    if row["status"] != "success":
        return "failed"

    peak_ok = (
        row["ell_peak1"] == 221
        and row["ell_peak2"] == 538
        and row["ell_peak3"] == 815
    )

    if not peak_ok:
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


def combined_score(row, baseline_s8):
    if row["case"] == "cdm_baseline":
        return 999.0

    if row["status"] != "success":
        return 9999.0

    s8_target = 0.830
    score = 0.0

    score += 80.0 * abs(row["S8"] - s8_target)
    score += 8.0 * max(0.0, row["lensing_max_abs_drift"] - 0.05)
    score += 4.0 * max(0.0, 0.80 - row["pk_ratio_k10"])
    score += 5.0 * max(0.0, row["growth_delta_chi2_vs_cdm"])
    score += 20.0 * max(0.0, row["tt_max_abs_drift"] - 0.01)
    score += 20.0 * max(0.0, row["bao_max_abs_drift"] - 0.0025)

    s8_improvement = max(0.0, baseline_s8 - row["S8"])
    score -= 0.5 * s8_improvement

    return float(score)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: aggregate_tairid_combined_matrix_scan.py <artifact_root>")

    artifact_root = Path(sys.argv[1])
    outdir = Path("combined_matrix_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for summary_path in artifact_root.rglob("case_summary.json"):
        data = json.loads(summary_path.read_text())
        data["_folder"] = str(summary_path.parent)
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

    baseline_growth_chi2 = float(baseline["growth_chi2"])
    baseline_s8 = float(baseline["S8"])

    combined_rows = []

    for item in summaries:
        name = item["name"]
        folder = Path(item["_folder"])

        row = {
            "case": name,
            "status": item.get("status"),
            "warm_fraction": float(item.get("warm_fraction", float("nan"))),
            "m_ncdm_eV": item.get("m_ncdm_eV"),
            "Omega_m_total": float(item.get("Omega_m_total", float("nan"))) if item.get("status") == "success" else float("nan"),
            "sigma8_z0": float(item.get("sigma8_z0", float("nan"))) if item.get("status") == "success" else float("nan"),
            "S8": float(item.get("S8", float("nan"))) if item.get("status") == "success" else float("nan"),
            "growth_chi2": float(item.get("growth_chi2", float("nan"))) if item.get("status") == "success" else float("nan"),
            "growth_reduced_chi2": float(item.get("growth_reduced_chi2", float("nan"))) if item.get("status") == "success" else float("nan"),
            "growth_delta_chi2_vs_cdm": float("nan"),
            "ell_peak1": None,
            "ell_peak2": None,
            "ell_peak3": None,
            "pk_ratio_k1": float("nan"),
            "pk_ratio_k5": float("nan"),
            "pk_ratio_k10": float("nan"),
            "tt_rms_drift": float("nan"),
            "tt_max_abs_drift": float("nan"),
            "tt_mean_abs_drift": float("nan"),
            "lensing_mean_ratio": float("nan"),
            "lensing_rms_drift": float("nan"),
            "lensing_max_abs_drift": float("nan"),
            "lensing_mean_abs_drift": float("nan"),
            "bao_max_abs_drift": float("nan"),
            "bao_DM_max_abs_drift": float("nan"),
            "bao_DH_max_abs_drift": float("nan"),
            "bao_DV_max_abs_drift": float("nan"),
            "diagnostic_flag": "",
            "combined_score": float("nan"),
            "error": item.get("error", ""),
        }

        if item.get("status") != "success":
            row["diagnostic_flag"] = "failed"
            combined_rows.append(row)
            continue

        peaks = item.get("peak_ell", [None, None, None])
        row["ell_peak1"] = peaks[0]
        row["ell_peak2"] = peaks[1]
        row["ell_peak3"] = peaks[2]
        row["growth_delta_chi2_vs_cdm"] = float(row["growth_chi2"] - baseline_growth_chi2)

        case_pk = read_numeric_csv(folder / "pk_z0.csv")
        pk_ratio = case_pk["Pk_z0"] / base_pk["Pk_z0"]

        row["pk_ratio_k1"] = interp_log_x(base_pk["k_1_per_Mpc"], pk_ratio, 1.0)
        row["pk_ratio_k5"] = interp_log_x(base_pk["k_1_per_Mpc"], pk_ratio, 5.0)
        row["pk_ratio_k10"] = interp_log_x(base_pk["k_1_per_Mpc"], pk_ratio, 10.0)

        case_tt = read_numeric_csv(folder / "cmb_tt_spectrum.csv")
        tt_metrics = curve_ratio_metrics(
            base_tt["ell"],
            base_tt["D_ell_TT"],
            case_tt["ell"],
            case_tt["D_ell_TT"],
            ELL_TT_MIN,
            ELL_TT_MAX,
        )
        row["tt_rms_drift"] = tt_metrics["rms_drift"]
        row["tt_max_abs_drift"] = tt_metrics["max_abs_drift"]
        row["tt_mean_abs_drift"] = tt_metrics["mean_abs_drift"]

        case_lensing = read_numeric_csv(folder / "cmb_lensing_curve.csv")
        lensing_metrics = curve_ratio_metrics(
            base_lensing["ell"],
            base_lensing["lensing_scaled"],
            case_lensing["ell"],
            case_lensing["lensing_scaled"],
            ELL_LENSING_MIN,
            ELL_LENSING_MAX,
        )
        row["lensing_mean_ratio"] = lensing_metrics["mean_ratio"]
        row["lensing_rms_drift"] = lensing_metrics["rms_drift"]
        row["lensing_max_abs_drift"] = lensing_metrics["max_abs_drift"]
        row["lensing_mean_abs_drift"] = lensing_metrics["mean_abs_drift"]

        case_bao = read_csv_dicts(folder / "bao_distances.csv")
        bao_metrics = bao_drift_metrics(base_bao, case_bao)
        row.update(bao_metrics)

        row["diagnostic_flag"] = classify(row)
        row["combined_score"] = combined_score(row, baseline_s8)

        combined_rows.append(row)

    combined_rows = sorted(
        combined_rows,
        key=lambda row: (
            row["diagnostic_flag"] not in ["best_balanced_frontier", "balanced_candidate"],
            row["combined_score"],
            row["S8"] if np.isfinite(row["S8"]) else 999.0,
        ),
    )

    summary_json = {
        "boundary": "Internal CLASS proxy matrix scan only. Not a final likelihood and not proof of TAIRID cosmology.",
        "baseline_S8": baseline_s8,
        "baseline_growth_chi2": baseline_growth_chi2,
        "case_count": len(combined_rows),
        "best_case": combined_rows[0],
        "rows": combined_rows,
    }

    (outdir / "combined_matrix_summary.json").write_text(json.dumps(summary_json, indent=2))

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

    with open(outdir / "combined_matrix_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rank, row in enumerate(combined_rows, start=1):
            writer.writerow(
                [
                    rank,
                    row["case"],
                    row["status"],
                    row["diagnostic_flag"],
                    row["combined_score"],
                    row["warm_fraction"],
                    row["m_ncdm_eV"],
                    row["S8"],
                    row["growth_chi2"],
                    row["growth_delta_chi2_vs_cdm"],
                    row["pk_ratio_k1"],
                    row["pk_ratio_k5"],
                    row["pk_ratio_k10"],
                    row["lensing_mean_ratio"],
                    row["lensing_max_abs_drift"],
                    row["tt_max_abs_drift"],
                    row["bao_max_abs_drift"],
                    row["ell_peak1"],
                    row["ell_peak2"],
                    row["ell_peak3"],
                    row["error"],
                ]
            )

    ranked_candidates = [
        row for row in combined_rows
        if row["diagnostic_flag"] in ["best_balanced_frontier", "balanced_candidate"]
    ]

    with open(outdir / "combined_matrix_ranked_candidates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rank, row in enumerate(ranked_candidates, start=1):
            writer.writerow(
                [
                    rank,
                    row["case"],
                    row["status"],
                    row["diagnostic_flag"],
                    row["combined_score"],
                    row["warm_fraction"],
                    row["m_ncdm_eV"],
                    row["S8"],
                    row["growth_chi2"],
                    row["growth_delta_chi2_vs_cdm"],
                    row["pk_ratio_k1"],
                    row["pk_ratio_k5"],
                    row["pk_ratio_k10"],
                    row["lensing_mean_ratio"],
                    row["lensing_max_abs_drift"],
                    row["tt_max_abs_drift"],
                    row["bao_max_abs_drift"],
                    row["ell_peak1"],
                    row["ell_peak2"],
                    row["ell_peak3"],
                    row["error"],
                ]
            )

    success_rows = [
        row for row in combined_rows
        if row["status"] == "success" and row["case"] != "cdm_baseline"
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
        plt.title("Combined matrix: S8 versus CMB lensing drift")
        plt.tight_layout()
        plt.savefig(outdir / "combined_matrix_s8_vs_lensing.png", dpi=160)
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
        plt.title("Combined matrix: S8 versus high-k survival")
        plt.tight_layout()
        plt.savefig(outdir / "combined_matrix_s8_vs_pk10.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(scores, s8_values)

        for label, x, y in zip(labels, scores, s8_values):
            if x <= 1.5:
                plt.annotate(label, (x, y), fontsize=7)

        plt.xlabel("combined score, lower is better")
        plt.ylabel("S8")
        plt.title("Combined matrix score")
        plt.tight_layout()
        plt.savefig(outdir / "combined_matrix_score_plot.png", dpi=160)
        plt.close()

    print("")
    print("TAIRID combined matrix aggregation complete.")
    print("Created:")
    print("  combined_matrix_outputs/combined_matrix_summary.json")
    print("  combined_matrix_outputs/combined_matrix_summary.csv")
    print("  combined_matrix_outputs/combined_matrix_ranked_candidates.csv")
    print("  combined_matrix_outputs/combined_matrix_s8_vs_lensing.png")
    print("  combined_matrix_outputs/combined_matrix_s8_vs_pk10.png")
    print("  combined_matrix_outputs/combined_matrix_score_plot.png")
    print("")
    print("Best case:")
    print(json.dumps(combined_rows[0], indent=2))


if __name__ == "__main__":
    main()
