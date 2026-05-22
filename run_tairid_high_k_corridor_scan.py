#!/usr/bin/env python3
"""
TAIRID high-k corridor scan v0.1

Purpose:
The current TAIRID cosmology proxy corridor survived simplified CMB peak-position,
S8, and growth-history checks. The unresolved danger is whether the same
delayed / non-cold neutral fraction suppresses small-scale matter power too much.

This scan tests nearby corridor points around:

    f_warm ≈ 7.5%
    m_proxy ≈ 25–30 eV

Boundary:
This is a CLASS proxy test. It is not a custom TAIRID perturbation equation.
The phrase neutral substrate remains an interpretation layer.

Outputs:
- high_k_corridor_scan_summary.csv
- high_k_corridor_scan_summary.json
- high_k_pk_ratios.csv
- high_k_corridor_scan_pk_ratio_plot.png
- high_k_corridor_scan_s8_plot.png
"""

import csv
import json
import math
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from classy import Class


OMEGA_NEUTRAL_PHYSICAL = 0.1200000000
H = 0.66893180

K_VALUES = np.logspace(-4, 1.2, 260)
K_REPORT = [0.1, 0.5, 1.0, 5.0, 10.0]

PEAK_WINDOWS = [
    (100, 350),
    (350, 650),
    (650, 1000),
]

BASE = {
    "output": "tCl,pCl,lCl,mPk",
    "lensing": "yes",
    "h": H,
    "omega_b": 0.0223700000,
    "omega_cdm": OMEGA_NEUTRAL_PHYSICAL,
    "N_ur": 3.046,
    "Omega_k": 0.0,
    "Omega_Lambda": 0.6817397872,
    "n_s": 0.9649,
    "A_s": 2.100549e-9,
    "tau_reio": 0.0544,
    "YHe": 0.245,
    "T_cmb": 2.7255,
    "l_max_scalars": 2500,
    "P_k_max_1/Mpc": 50.0,
    "z_max_pk": 3.0,
    "z_pk": "0",
}

CASES = [
    {"name": "cdm_baseline", "warm_fraction": 0.000, "m_ncdm_eV": None},

    {"name": "test_2p5pct_20eV", "warm_fraction": 0.025, "m_ncdm_eV": 20.0},
    {"name": "test_2p5pct_25eV", "warm_fraction": 0.025, "m_ncdm_eV": 25.0},
    {"name": "test_2p5pct_30eV", "warm_fraction": 0.025, "m_ncdm_eV": 30.0},

    {"name": "test_5pct_20eV", "warm_fraction": 0.050, "m_ncdm_eV": 20.0},
    {"name": "test_5pct_25eV", "warm_fraction": 0.050, "m_ncdm_eV": 25.0},
    {"name": "test_5pct_30eV", "warm_fraction": 0.050, "m_ncdm_eV": 30.0},
    {"name": "test_5pct_35eV", "warm_fraction": 0.050, "m_ncdm_eV": 35.0},

    {"name": "current_7p5pct_25eV", "warm_fraction": 0.075, "m_ncdm_eV": 25.0},
    {"name": "current_7p5pct_30eV", "warm_fraction": 0.075, "m_ncdm_eV": 30.0},
    {"name": "test_7p5pct_35eV", "warm_fraction": 0.075, "m_ncdm_eV": 35.0},
    {"name": "test_7p5pct_40eV", "warm_fraction": 0.075, "m_ncdm_eV": 40.0},

    {"name": "old_10pct_30eV", "warm_fraction": 0.100, "m_ncdm_eV": 30.0},
]


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def make_params(case):
    params = dict(BASE)

    f = float(case["warm_fraction"])
    if f <= 0:
        return params

    omega_warm = OMEGA_NEUTRAL_PHYSICAL * f
    omega_cold = OMEGA_NEUTRAL_PHYSICAL * (1.0 - f)

    params["omega_cdm"] = omega_cold
    params["N_ncdm"] = 1
    params["m_ncdm"] = float(case["m_ncdm_eV"])
    params["Omega_ncdm"] = omega_warm / (H * H)

    return params


def omega_total_matter_proxy(params):
    h = float(params["h"])
    h2 = h * h

    omega_b = float(params.get("omega_b", 0.0))
    omega_cdm = float(params.get("omega_cdm", 0.0))
    omega_ncdm = float(params.get("Omega_ncdm", 0.0)) * h2

    omega_m = omega_b + omega_cdm + omega_ncdm
    Omega_m = omega_m / h2

    return omega_m, Omega_m


def top_hat_window(x):
    x = np.asarray(x)
    w = np.ones_like(x)

    small = np.abs(x) < 1.0e-4
    large = ~small

    xl = x[large]
    w[large] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / (xl ** 3)

    xs = x[small]
    w[small] = 1.0 - (xs ** 2) / 10.0

    return w


def sigma_R_from_pk(k_values, pk_values, R_mpc):
    k = np.asarray(k_values)
    pk = np.asarray(pk_values)

    good = np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0)
    k = k[good]
    pk = pk[good]

    if len(k) < 10:
        return float("nan")

    x = k * R_mpc
    w = top_hat_window(x)

    integrand = (k ** 3) * pk * (w ** 2) / (2.0 * math.pi * math.pi)
    ln_k = np.log(k)

    sigma2 = integrate_trapezoid(integrand, ln_k)

    if sigma2 <= 0 or not np.isfinite(sigma2):
        return float("nan")

    return float(math.sqrt(sigma2))


def s8_from_sigma8(sigma8, Omega_m):
    return float(sigma8 * math.sqrt(Omega_m / 0.3))


def peak_by_windows(ell, dl):
    peak_ells = []
    peak_heights = []

    for lo, hi in PEAK_WINDOWS:
        mask = (ell >= lo) & (ell <= hi)

        if not np.any(mask):
            peak_ells.append(None)
            peak_heights.append(None)
            continue

        ell_window = ell[mask]
        dl_window = dl[mask]
        idx = int(np.argmax(dl_window))

        peak_ells.append(int(ell_window[idx]))
        peak_heights.append(float(dl_window[idx]))

    return peak_ells, peak_heights


def interp_ratio(k_grid, ratio, k_target):
    return float(np.interp(math.log(k_target), np.log(k_grid), ratio))


def run_class_case(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    class_sigma8_z0 = float(cosmo.sigma8())

    cl = cosmo.lensed_cl(2500)
    ell = np.asarray(cl["ell"])
    tt = np.asarray(cl["tt"])
    dl = ell * (ell + 1) * tt / (2.0 * math.pi)

    pk_z0 = []

    for k in K_VALUES:
        try:
            pk_z0.append(float(cosmo.pk(float(k), 0.0)))
        except Exception:
            pk_z0.append(float("nan"))

    pk_z0 = np.asarray(pk_z0)

    h = float(params["h"])
    R8_mpc = 8.0 / h
    sigma8_integral = sigma_R_from_pk(K_VALUES, pk_z0, R8_mpc)

    cosmo.struct_cleanup()
    cosmo.empty()

    return class_sigma8_z0, sigma8_integral, ell, dl, pk_z0


results = {}
pk_tables = {}

for case in CASES:
    name = case["name"]
    params = make_params(case)
    omega_m, Omega_m = omega_total_matter_proxy(params)

    print("")
    print("Running case:", name)
    print("  warm_fraction:", case["warm_fraction"])
    print("  m_ncdm_eV:", case["m_ncdm_eV"])
    print("  omega_cdm:", params.get("omega_cdm"))
    print("  Omega_ncdm:", params.get("Omega_ncdm"))
    print("  Omega_m:", Omega_m)

    try:
        class_sigma8_z0, sigma8_integral, ell, dl, pk_z0 = run_class_case(params)
        peaks_ell, peaks_height = peak_by_windows(ell, dl)
        S8 = s8_from_sigma8(class_sigma8_z0, Omega_m)

        pk_file = f"{name}_pk_z0.txt"
        np.savetxt(
            pk_file,
            np.column_stack([K_VALUES, pk_z0]),
            header="k_1_per_Mpc Pk_z0",
        )

        results[name] = {
            "status": "success",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "omega_cdm": float(params.get("omega_cdm", float("nan"))),
            "Omega_ncdm": None if "Omega_ncdm" not in params else float(params["Omega_ncdm"]),
            "Omega_m_total": float(Omega_m),
            "class_sigma8_z0": class_sigma8_z0,
            "sigma8_integral_z0": sigma8_integral,
            "S8_from_class_sigma8": S8,
            "peak_ell": peaks_ell,
            "peak_height": peaks_height,
            "pk_file": pk_file,
        }

        pk_tables[name] = pk_z0

        print("  success")
        print("  S8:", S8)
        print("  sigma8:", class_sigma8_z0)
        print("  peaks:", peaks_ell)

    except Exception as exc:
        print("  FAILED:", exc)

        results[name] = {
            "status": "failed",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "params": params,
        }


# Add P(k) ratios against CDM baseline.
baseline_name = "cdm_baseline"

if baseline_name in pk_tables:
    baseline_pk = pk_tables[baseline_name]

    for name, pk in pk_tables.items():
        good = np.isfinite(pk) & np.isfinite(baseline_pk) & (baseline_pk > 0)
        ratio = np.full_like(pk, np.nan, dtype=float)
        ratio[good] = pk[good] / baseline_pk[good]

        ratio_file = f"{name}_pk_ratio_to_cdm.txt"
        np.savetxt(
            ratio_file,
            np.column_stack([K_VALUES, ratio]),
            header="k_1_per_Mpc Pk_ratio_to_cdm",
        )

        results[name]["pk_ratio_file"] = ratio_file

        for k_target in K_REPORT:
            results[name][f"pk_ratio_k{k_target:g}"] = interp_ratio(K_VALUES, ratio, k_target)

    base = results[baseline_name]

    for name, data in results.items():
        if data["status"] != "success":
            continue

        data["delta_vs_cdm_baseline"] = {
            "delta_S8": float(data["S8_from_class_sigma8"] - base["S8_from_class_sigma8"]),
            "delta_sigma8": float(data["class_sigma8_z0"] - base["class_sigma8_z0"]),
            "ratio_S8": float(data["S8_from_class_sigma8"] / base["S8_from_class_sigma8"]),
            "ratio_sigma8": float(data["class_sigma8_z0"] / base["class_sigma8_z0"]),
        }


# Assign simple diagnostic flags.
for name, data in results.items():
    if data["status"] != "success":
        continue

    if name == baseline_name:
        data["diagnostic_flag"] = "baseline"
        continue

    k10 = data.get("pk_ratio_k10", float("nan"))
    s8 = data.get("S8_from_class_sigma8", float("nan"))

    if not np.isfinite(k10):
        data["diagnostic_flag"] = "warning_missing_high_k_ratio"
    elif k10 < 0.50 and s8 <= 0.81:
        data["diagnostic_flag"] = "warning_s8_help_but_high_k_suppression_large"
    elif k10 >= 0.50 and s8 <= 0.82:
        data["diagnostic_flag"] = "best_nearby_candidate"
    elif k10 >= 0.60:
        data["diagnostic_flag"] = "high_k_gentler_but_check_s8"
    else:
        data["diagnostic_flag"] = "mixed"


# Save JSON summary.
Path("high_k_corridor_scan_summary.json").write_text(json.dumps(results, indent=2))


# Save summary CSV.
summary_columns = [
    "case",
    "status",
    "diagnostic_flag",
    "warm_fraction",
    "m_ncdm_eV",
    "Omega_m_total",
    "class_sigma8_z0",
    "S8",
    "ell_peak1",
    "ell_peak2",
    "ell_peak3",
    "pk_ratio_k0.1",
    "pk_ratio_k0.5",
    "pk_ratio_k1",
    "pk_ratio_k5",
    "pk_ratio_k10",
    "delta_S8_vs_cdm",
    "ratio_S8_vs_cdm",
    "error",
]

with open("high_k_corridor_scan_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(summary_columns)

    for name, data in results.items():
        if data["status"] == "success":
            delta = data.get("delta_vs_cdm_baseline", {})
            peaks = data.get("peak_ell", [None, None, None])

            writer.writerow(
                [
                    name,
                    data["status"],
                    data.get("diagnostic_flag"),
                    data["warm_fraction"],
                    data["m_ncdm_eV"],
                    data["Omega_m_total"],
                    data["class_sigma8_z0"],
                    data["S8_from_class_sigma8"],
                    peaks[0],
                    peaks[1],
                    peaks[2],
                    data.get("pk_ratio_k0.1"),
                    data.get("pk_ratio_k0.5"),
                    data.get("pk_ratio_k1"),
                    data.get("pk_ratio_k5"),
                    data.get("pk_ratio_k10"),
                    delta.get("delta_S8"),
                    delta.get("ratio_S8"),
                    "",
                ]
            )
        else:
            writer.writerow(
                [
                    name,
                    data["status"],
                    "",
                    data["warm_fraction"],
                    data["m_ncdm_eV"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    data.get("error", ""),
                ]
            )


# Save long-form ratio table.
with open("high_k_pk_ratios.csv", "w", newline="") as f:
    writer = csv.writer(f)

    header = ["k_1_per_Mpc"]
    for name in pk_tables:
        header.append(name + "_Pk_ratio_to_cdm")
    writer.writerow(header)

    baseline_pk = pk_tables.get(baseline_name)

    for i, k in enumerate(K_VALUES):
        row = [float(k)]

        for name, pk in pk_tables.items():
            if baseline_pk is None or not np.isfinite(baseline_pk[i]) or baseline_pk[i] <= 0:
                row.append("")
            else:
                row.append(float(pk[i] / baseline_pk[i]))

        writer.writerow(row)


# Plot P(k) ratios.
success_names = [
    name for name, data in results.items()
    if data["status"] == "success" and name != baseline_name
]

if baseline_name in pk_tables and success_names:
    baseline_pk = pk_tables[baseline_name]

    plt.figure(figsize=(10, 6))

    for name in success_names:
        pk = pk_tables[name]
        good = np.isfinite(pk) & np.isfinite(baseline_pk) & (baseline_pk > 0)
        ratio = np.full_like(pk, np.nan, dtype=float)
        ratio[good] = pk[good] / baseline_pk[good]

        plt.plot(K_VALUES, ratio, label=name, linewidth=1.2)

    plt.axhline(1.0, linewidth=1)
    plt.axhline(0.5, linewidth=1)
    plt.xscale("log")
    plt.xlabel("k [1/Mpc]")
    plt.ylabel("P(k) / P_CDM")
    plt.title("TAIRID proxy high-k suppression scan")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("high_k_corridor_scan_pk_ratio_plot.png", dpi=160)
    plt.close()


# Plot S8 by case.
success_all = [
    name for name, data in results.items()
    if data["status"] == "success"
]

if success_all:
    plt.figure(figsize=(11, 5))

    labels = success_all
    s8_vals = [results[name]["S8_from_class_sigma8"] for name in labels]

    x = np.arange(len(labels))
    plt.bar(x, s8_vals)
    plt.axhline(0.80, linewidth=1)
    plt.xticks(x, labels, rotation=55, ha="right", fontsize=8)
    plt.ylabel("S8")
    plt.title("S8 by high-k corridor case")
    plt.tight_layout()
    plt.savefig("high_k_corridor_scan_s8_plot.png", dpi=160)
    plt.close()


print("")
print("TAIRID high-k corridor scan complete.")
print("Created:")
print("  high_k_corridor_scan_summary.json")
print("  high_k_corridor_scan_summary.csv")
print("  high_k_pk_ratios.csv")
print("  high_k_corridor_scan_pk_ratio_plot.png")
print("  high_k_corridor_scan_s8_plot.png")
print("")
print("Read boundary:")
print("  This is a proxy scan only. A pass does not prove TAIRID cosmology.")
print("  A fail means the current warm-neutral corridor is too destructive at small scales.")
