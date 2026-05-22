#!/usr/bin/env python3
"""
TAIRID lensing-safe corridor scan.

Purpose:
The 5%, 20 eV lane survived several proxy tests, but the CMB lensing pressure
test showed noticeable lensing suppression.

This scan searches gentler lanes near:

    f_warm = 1% to 5%
    m_proxy = 10 to 40 eV

Goal:
Find whether a lower delayed / non-cold neutral fraction can preserve more
CMB lensing while still lowering S8 and keeping high-k suppression moderate.

Boundary:
This is not a Planck lensing likelihood.
This is not a Lyman-alpha likelihood.
This is not proof of TAIRID cosmology.
It is an internal CLASS proxy scan.
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

ELL_MIN = 40
ELL_MAX = 1000

K_VALUES = np.logspace(-4, 1.2, 220)
K_REPORT = [1.0, 5.0, 10.0]

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

    {"name": "test_1pct_10eV", "warm_fraction": 0.010, "m_ncdm_eV": 10.0},
    {"name": "test_1pct_20eV", "warm_fraction": 0.010, "m_ncdm_eV": 20.0},
    {"name": "test_1pct_30eV", "warm_fraction": 0.010, "m_ncdm_eV": 30.0},

    {"name": "test_2p5pct_10eV", "warm_fraction": 0.025, "m_ncdm_eV": 10.0},
    {"name": "test_2p5pct_20eV", "warm_fraction": 0.025, "m_ncdm_eV": 20.0},
    {"name": "test_2p5pct_25eV", "warm_fraction": 0.025, "m_ncdm_eV": 25.0},
    {"name": "test_2p5pct_30eV", "warm_fraction": 0.025, "m_ncdm_eV": 30.0},
    {"name": "test_2p5pct_40eV", "warm_fraction": 0.025, "m_ncdm_eV": 40.0},

    {"name": "test_3p5pct_20eV", "warm_fraction": 0.035, "m_ncdm_eV": 20.0},
    {"name": "test_3p5pct_25eV", "warm_fraction": 0.035, "m_ncdm_eV": 25.0},
    {"name": "test_3p5pct_30eV", "warm_fraction": 0.035, "m_ncdm_eV": 30.0},

    {"name": "previous_best_5pct_20eV", "warm_fraction": 0.050, "m_ncdm_eV": 20.0},
    {"name": "previous_backup_5pct_25eV", "warm_fraction": 0.050, "m_ncdm_eV": 25.0},
]


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


def s8_from_sigma8(sigma8, Omega_m):
    return float(sigma8 * math.sqrt(Omega_m / 0.3))


def get_lensing_pp(cosmo, lmax):
    try:
        cl = cosmo.lensed_cl(lmax)
        if "pp" in cl:
            return np.asarray(cl["ell"]), np.asarray(cl["pp"])
    except Exception:
        pass

    try:
        cl = cosmo.raw_cl(lmax)
        if "pp" in cl:
            return np.asarray(cl["ell"]), np.asarray(cl["pp"])
    except Exception:
        pass

    raise RuntimeError("Could not find pp lensing spectrum in CLASS output.")


def interp_ratio(k_grid, ratio, k_target):
    return float(np.interp(math.log(k_target), np.log(k_grid), ratio))


def run_class_case(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    sigma8 = float(cosmo.sigma8())

    ell, pp = get_lensing_pp(cosmo, 2500)

    lensing_scaled = np.zeros_like(pp, dtype=float)
    good = ell > 0
    lensing_scaled[good] = (ell[good] ** 4) * pp[good] / (2.0 * math.pi)

    pk_z0 = []

    for k in K_VALUES:
        try:
            pk_z0.append(float(cosmo.pk(float(k), 0.0)))
        except Exception:
            pk_z0.append(float("nan"))

    pk_z0 = np.asarray(pk_z0)

    cosmo.struct_cleanup()
    cosmo.empty()

    return sigma8, ell, lensing_scaled, pk_z0


results = {}
lensing_curves = {}
pk_curves = {}

for case in CASES:
    name = case["name"]
    params = make_params(case)
    omega_m, Omega_m = omega_total_matter_proxy(params)

    print("")
    print("Running case:", name)
    print("  warm_fraction:", case["warm_fraction"])
    print("  m_ncdm_eV:", case["m_ncdm_eV"])

    try:
        sigma8, ell, lensing_scaled, pk_z0 = run_class_case(params)
        S8 = s8_from_sigma8(sigma8, Omega_m)

        lensing_file = f"{name}_lensing_curve.txt"
        np.savetxt(
            lensing_file,
            np.column_stack([ell, lensing_scaled]),
            header="ell ell4_C_ell_pp_over_2pi",
        )

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
            "Omega_m_total": float(Omega_m),
            "sigma8": sigma8,
            "S8": S8,
            "lensing_file": lensing_file,
            "pk_file": pk_file,
        }

        lensing_curves[name] = {
            "ell": ell,
            "lensing_scaled": lensing_scaled,
        }

        pk_curves[name] = pk_z0

        print("  success")
        print("  S8:", S8)

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


baseline_name = "cdm_baseline"

if baseline_name in lensing_curves and baseline_name in pk_curves:
    base_ell = lensing_curves[baseline_name]["ell"]
    base_lensing = lensing_curves[baseline_name]["lensing_scaled"]
    base_pk = pk_curves[baseline_name]

    compare_mask = (
        (base_ell >= ELL_MIN)
        & (base_ell <= ELL_MAX)
        & np.isfinite(base_lensing)
        & (base_lensing > 0)
    )

    base_result = results[baseline_name]

    for name, curve in lensing_curves.items():
        ell = curve["ell"]
        lensing = curve["lensing_scaled"]

        if not np.array_equal(ell, base_ell):
            lensing_interp = np.interp(base_ell, ell, lensing)
        else:
            lensing_interp = lensing

        lensing_ratio = np.full_like(base_lensing, np.nan, dtype=float)
        good_lensing = np.isfinite(lensing_interp) & np.isfinite(base_lensing) & (base_lensing > 0)
        lensing_ratio[good_lensing] = lensing_interp[good_lensing] / base_lensing[good_lensing]

        diff = lensing_ratio[compare_mask] - 1.0
        abs_diff = np.abs(diff)

        results[name]["lensing_ratio_mean_ell40_1000"] = float(np.nanmean(lensing_ratio[compare_mask]))
        results[name]["lensing_ratio_rms_ell40_1000"] = float(np.sqrt(np.nanmean(diff * diff)))
        results[name]["lensing_ratio_max_abs_ell40_1000"] = float(np.nanmax(abs_diff))
        results[name]["lensing_ratio_mean_abs_ell40_1000"] = float(np.nanmean(abs_diff))

        lensing_ratio_file = f"{name}_lensing_ratio_to_cdm.txt"
        np.savetxt(
            lensing_ratio_file,
            np.column_stack([base_ell, lensing_ratio]),
            header="ell lensing_ratio_to_cdm",
        )
        results[name]["lensing_ratio_file"] = lensing_ratio_file

        pk = pk_curves[name]
        pk_ratio = np.full_like(pk, np.nan, dtype=float)
        good_pk = np.isfinite(pk) & np.isfinite(base_pk) & (base_pk > 0)
        pk_ratio[good_pk] = pk[good_pk] / base_pk[good_pk]

        pk_ratio_file = f"{name}_pk_ratio_to_cdm.txt"
        np.savetxt(
            pk_ratio_file,
            np.column_stack([K_VALUES, pk_ratio]),
            header="k_1_per_Mpc Pk_ratio_to_cdm",
        )
        results[name]["pk_ratio_file"] = pk_ratio_file

        for k_target in K_REPORT:
            results[name][f"pk_ratio_k{k_target:g}"] = interp_ratio(K_VALUES, pk_ratio, k_target)

        if name == baseline_name:
            results[name]["delta_S8_vs_cdm"] = 0.0
            results[name]["diagnostic_flag"] = "baseline"
            continue

        results[name]["delta_S8_vs_cdm"] = float(results[name]["S8"] - base_result["S8"])

        max_lensing = results[name]["lensing_ratio_max_abs_ell40_1000"]
        pk10 = results[name]["pk_ratio_k10"]
        s8 = results[name]["S8"]

        if max_lensing <= 0.03 and pk10 >= 0.75 and s8 <= 0.835:
            results[name]["diagnostic_flag"] = "best_lensing_safe_candidate"
        elif max_lensing <= 0.05 and pk10 >= 0.65 and s8 <= 0.830:
            results[name]["diagnostic_flag"] = "balanced_candidate"
        elif max_lensing <= 0.08 and pk10 >= 0.55 and s8 <= 0.825:
            results[name]["diagnostic_flag"] = "warning_but_interesting"
        elif max_lensing > 0.08:
            results[name]["diagnostic_flag"] = "lensing_too_suppressed"
        elif pk10 < 0.55:
            results[name]["diagnostic_flag"] = "high_k_too_suppressed"
        else:
            results[name]["diagnostic_flag"] = "mixed"


Path("lensing_safe_corridor_summary.json").write_text(json.dumps(results, indent=2))


with open("lensing_safe_corridor_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(
        [
            "case",
            "status",
            "diagnostic_flag",
            "warm_fraction",
            "m_ncdm_eV",
            "Omega_m_total",
            "sigma8",
            "S8",
            "delta_S8_vs_cdm",
            "pk_ratio_k1",
            "pk_ratio_k5",
            "pk_ratio_k10",
            "lensing_ratio_mean_ell40_1000",
            "lensing_ratio_rms_ell40_1000",
            "lensing_ratio_max_abs_ell40_1000",
            "lensing_ratio_mean_abs_ell40_1000",
            "error",
        ]
    )

    for name, data in results.items():
        if data["status"] == "success":
            writer.writerow(
                [
                    name,
                    data["status"],
                    data.get("diagnostic_flag"),
                    data["warm_fraction"],
                    data["m_ncdm_eV"],
                    data["Omega_m_total"],
                    data["sigma8"],
                    data["S8"],
                    data.get("delta_S8_vs_cdm"),
                    data.get("pk_ratio_k1"),
                    data.get("pk_ratio_k5"),
                    data.get("pk_ratio_k10"),
                    data.get("lensing_ratio_mean_ell40_1000"),
                    data.get("lensing_ratio_rms_ell40_1000"),
                    data.get("lensing_ratio_max_abs_ell40_1000"),
                    data.get("lensing_ratio_mean_abs_ell40_1000"),
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
                    data.get("error", ""),
                ]
            )


with open("lensing_safe_corridor_pk_ratios.csv", "w", newline="") as f:
    writer = csv.writer(f)

    header = ["k_1_per_Mpc"]
    for name in pk_curves:
        header.append(name + "_Pk_ratio_to_cdm")
    writer.writerow(header)

    base_pk = pk_curves.get(baseline_name)

    for i, k in enumerate(K_VALUES):
        row = [float(k)]

        for name, pk in pk_curves.items():
            if base_pk is None or not np.isfinite(base_pk[i]) or base_pk[i] <= 0:
                row.append("")
            else:
                row.append(float(pk[i] / base_pk[i]))

        writer.writerow(row)


if baseline_name in lensing_curves:
    base_ell = lensing_curves[baseline_name]["ell"]
    base_lensing = lensing_curves[baseline_name]["lensing_scaled"]

    with open("lensing_safe_corridor_lensing_ratios.csv", "w", newline="") as f:
        writer = csv.writer(f)

        header = ["ell"]
        for name in lensing_curves:
            header.append(name + "_lensing_ratio_to_cdm")
        writer.writerow(header)

        for i, ell_value in enumerate(base_ell):
            row = [int(ell_value)]

            for name, curve in lensing_curves.items():
                ell = curve["ell"]
                lensing = curve["lensing_scaled"]

                if not np.array_equal(ell, base_ell):
                    lensing_interp = np.interp(base_ell, ell, lensing)
                else:
                    lensing_interp = lensing

                if base_lensing[i] > 0 and np.isfinite(base_lensing[i]) and np.isfinite(lensing_interp[i]):
                    row.append(float(lensing_interp[i] / base_lensing[i]))
                else:
                    row.append("")

            writer.writerow(row)


success_names = [
    name for name, data in results.items()
    if data["status"] == "success"
]

if baseline_name in lensing_curves and success_names:
    base_ell = lensing_curves[baseline_name]["ell"]
    base_lensing = lensing_curves[baseline_name]["lensing_scaled"]

    plt.figure(figsize=(10, 6))

    for name in success_names:
        if name == baseline_name:
            continue

        curve = lensing_curves[name]
        ell = curve["ell"]
        lensing = curve["lensing_scaled"]

        if not np.array_equal(ell, base_ell):
            lensing_interp = np.interp(base_ell, ell, lensing)
        else:
            lensing_interp = lensing

        ratio = lensing_interp / base_lensing

        mask = (
            (base_ell >= ELL_MIN)
            & (base_ell <= ELL_MAX)
            & np.isfinite(ratio)
        )

        plt.plot(base_ell[mask], ratio[mask], label=name, linewidth=1.1)

    plt.axhline(1.0, linewidth=1)
    plt.xlabel("ell")
    plt.ylabel("CMB lensing ratio to CDM baseline")
    plt.title("TAIRID lensing-safe corridor scan")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("lensing_safe_corridor_lensing_ratio_plot.png", dpi=160)
    plt.close()


if success_names:
    labels = success_names
    s8_values = [results[name]["S8"] for name in labels]
    lensing_max = [results[name].get("lensing_ratio_max_abs_ell40_1000", np.nan) for name in labels]
    pk10_values = [results[name].get("pk_ratio_k10", np.nan) for name in labels]

    plt.figure(figsize=(10, 6))
    plt.scatter(lensing_max, s8_values)

    for label, x, y in zip(labels, lensing_max, s8_values):
        plt.annotate(label, (x, y), fontsize=7)

    plt.axhline(0.83, linewidth=1)
    plt.axvline(0.03, linewidth=1)
    plt.axvline(0.05, linewidth=1)
    plt.xlabel("max CMB lensing drift versus CDM")
    plt.ylabel("S8")
    plt.title("S8 versus CMB lensing drift")
    plt.tight_layout()
    plt.savefig("lensing_safe_corridor_s8_vs_lensing.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(pk10_values, s8_values)

    for label, x, y in zip(labels, pk10_values, s8_values):
        plt.annotate(label, (x, y), fontsize=7)

    plt.axhline(0.83, linewidth=1)
    plt.axvline(0.75, linewidth=1)
    plt.axvline(0.65, linewidth=1)
    plt.xlabel("P(k=10)/P_CDM")
    plt.ylabel("S8")
    plt.title("S8 versus high-k survival")
    plt.tight_layout()
    plt.savefig("lensing_safe_corridor_s8_vs_pk10.png", dpi=160)
    plt.close()


print("")
print("TAIRID lensing-safe corridor scan complete.")
print("Created:")
print("  lensing_safe_corridor_summary.json")
print("  lensing_safe_corridor_summary.csv")
print("  lensing_safe_corridor_pk_ratios.csv")
print("  lensing_safe_corridor_lensing_ratios.csv")
print("  lensing_safe_corridor_lensing_ratio_plot.png")
print("  lensing_safe_corridor_s8_vs_lensing.png")
print("  lensing_safe_corridor_s8_vs_pk10.png")
print("")
print("Read boundary:")
print("  This is an internal proxy scan only.")
print("  It searches for a gentler lane after the CMB lensing warning.")
