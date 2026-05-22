#!/usr/bin/env python3
"""
TAIRID CMB lensing pressure test.

Purpose:
The current best TAIRID proxy lane is:

    f_warm = 5%
    m_proxy = 20 eV

It has survived:
- S8 reduction
- high-k pressure better than the 7.5% warning lane
- observed growth proxy chi-square
- CMB TT peak-position and shape pressure
- BAO distance-ratio pressure

Now we test whether it distorts the CMB lensing potential spectrum too much
relative to the CDM proxy baseline.

Boundary:
This is not a Planck lensing likelihood.
It does not include full covariance, reconstruction noise, foregrounds,
polarization likelihoods, or nuisance parameters.
It is an internal CLASS lensing-shape pressure test against the CDM proxy baseline.
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
    {"name": "cdm_baseline", "warm_fraction": 0.0, "m_ncdm_eV": None},
    {"name": "best_current_5pct_20eV", "warm_fraction": 0.05, "m_ncdm_eV": 20.0},
    {"name": "backup_5pct_25eV", "warm_fraction": 0.05, "m_ncdm_eV": 25.0},
    {"name": "s8_warning_7p5pct_25eV", "warm_fraction": 0.075, "m_ncdm_eV": 25.0},
    {"name": "s8_warning_7p5pct_30eV", "warm_fraction": 0.075, "m_ncdm_eV": 30.0},
    {"name": "destructive_old_10pct_30eV", "warm_fraction": 0.10, "m_ncdm_eV": 30.0},
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


def run_class_case(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    sigma8 = float(cosmo.sigma8())

    ell, pp = get_lensing_pp(cosmo, 2500)

    lensing_scaled = np.zeros_like(pp, dtype=float)
    good = ell > 0
    lensing_scaled[good] = (ell[good] ** 4) * pp[good] / (2.0 * math.pi)

    cosmo.struct_cleanup()
    cosmo.empty()

    return sigma8, ell, pp, lensing_scaled


results = {}
spectra = {}

for case in CASES:
    name = case["name"]
    params = make_params(case)
    omega_m, Omega_m = omega_total_matter_proxy(params)

    print("")
    print("Running case:", name)

    try:
        sigma8, ell, pp, lensing_scaled = run_class_case(params)
        S8 = s8_from_sigma8(sigma8, Omega_m)

        spectrum_file = f"{name}_cmb_lensing_spectrum.txt"
        np.savetxt(
            spectrum_file,
            np.column_stack([ell, pp, lensing_scaled]),
            header="ell C_ell_pp ell4_C_ell_pp_over_2pi",
        )

        results[name] = {
            "status": "success",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "Omega_m_total": float(Omega_m),
            "sigma8": sigma8,
            "S8": S8,
            "spectrum_file": spectrum_file,
        }

        spectra[name] = {
            "ell": ell,
            "pp": pp,
            "lensing_scaled": lensing_scaled,
        }

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

if baseline_name in spectra:
    base_ell = spectra[baseline_name]["ell"]
    base_curve = spectra[baseline_name]["lensing_scaled"]

    compare_mask = (
        (base_ell >= ELL_MIN)
        & (base_ell <= ELL_MAX)
        & np.isfinite(base_curve)
        & (base_curve > 0)
    )

    for name, spec in spectra.items():
        ell = spec["ell"]
        curve = spec["lensing_scaled"]

        if not np.array_equal(ell, base_ell):
            curve_interp = np.interp(base_ell, ell, curve)
        else:
            curve_interp = curve

        ratio = np.full_like(base_curve, np.nan, dtype=float)
        good = np.isfinite(curve_interp) & np.isfinite(base_curve) & (base_curve > 0)
        ratio[good] = curve_interp[good] / base_curve[good]

        ratio_file = f"{name}_cmb_lensing_ratio_to_cdm.txt"
        np.savetxt(
            ratio_file,
            np.column_stack([base_ell, ratio]),
            header="ell lensing_scaled_ratio_to_CDM",
        )

        results[name]["ratio_file"] = ratio_file

        diff = ratio[compare_mask] - 1.0
        abs_diff = np.abs(diff)

        if len(diff) > 0:
            results[name]["lensing_ratio_rms_ell40_1000"] = float(np.sqrt(np.nanmean(diff * diff)))
            results[name]["lensing_ratio_max_abs_ell40_1000"] = float(np.nanmax(abs_diff))
            results[name]["lensing_ratio_mean_abs_ell40_1000"] = float(np.nanmean(abs_diff))
            results[name]["lensing_ratio_mean_ell40_1000"] = float(np.nanmean(ratio[compare_mask]))
        else:
            results[name]["lensing_ratio_rms_ell40_1000"] = None
            results[name]["lensing_ratio_max_abs_ell40_1000"] = None
            results[name]["lensing_ratio_mean_abs_ell40_1000"] = None
            results[name]["lensing_ratio_mean_ell40_1000"] = None

        if name != baseline_name and results[name]["status"] == "success":
            base_result = results[baseline_name]

            results[name]["delta_vs_cdm_baseline"] = {
                "delta_S8": float(results[name]["S8"] - base_result["S8"]),
                "delta_sigma8": float(results[name]["sigma8"] - base_result["sigma8"]),
            }


for name, data in results.items():
    if data["status"] != "success":
        continue

    if name == baseline_name:
        data["diagnostic_flag"] = "baseline"
        continue

    rms = data.get("lensing_ratio_rms_ell40_1000")
    max_abs = data.get("lensing_ratio_max_abs_ell40_1000")
    mean_ratio = data.get("lensing_ratio_mean_ell40_1000")

    if rms is None or max_abs is None or mean_ratio is None:
        data["diagnostic_flag"] = "warning_missing_lensing_metrics"
    elif max_abs <= 0.03:
        data["diagnostic_flag"] = "lensing_close_to_baseline"
    elif max_abs <= 0.08:
        data["diagnostic_flag"] = "lensing_warning_but_survives_proxy"
    else:
        data["diagnostic_flag"] = "lensing_distortion_warning"


Path("cmb_lensing_pressure_summary.json").write_text(json.dumps(results, indent=2))


with open("cmb_lensing_pressure_summary.csv", "w", newline="") as f:
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
            "lensing_ratio_mean_ell40_1000",
            "lensing_ratio_rms_ell40_1000",
            "lensing_ratio_max_abs_ell40_1000",
            "lensing_ratio_mean_abs_ell40_1000",
            "delta_S8_vs_cdm",
            "error",
        ]
    )

    for name, data in results.items():
        if data["status"] == "success":
            delta = data.get("delta_vs_cdm_baseline", {})

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
                    data.get("lensing_ratio_mean_ell40_1000"),
                    data.get("lensing_ratio_rms_ell40_1000"),
                    data.get("lensing_ratio_max_abs_ell40_1000"),
                    data.get("lensing_ratio_mean_abs_ell40_1000"),
                    delta.get("delta_S8"),
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
                    data.get("error", ""),
                ]
            )


if baseline_name in spectra:
    base_ell = spectra[baseline_name]["ell"]
    base_curve = spectra[baseline_name]["lensing_scaled"]

    with open("cmb_lensing_ratio_curves.csv", "w", newline="") as f:
        writer = csv.writer(f)

        header = ["ell"]
        for name in spectra:
            header.append(name + "_lensing_ratio_to_CDM")
        writer.writerow(header)

        for i, ell_value in enumerate(base_ell):
            row = [int(ell_value)]

            for name, spec in spectra.items():
                ell = spec["ell"]
                curve = spec["lensing_scaled"]

                if not np.array_equal(ell, base_ell):
                    curve_interp = np.interp(base_ell, ell, curve)
                else:
                    curve_interp = curve

                if base_curve[i] > 0 and np.isfinite(base_curve[i]) and np.isfinite(curve_interp[i]):
                    row.append(float(curve_interp[i] / base_curve[i]))
                else:
                    row.append("")

            writer.writerow(row)


success_names = [
    name for name, data in results.items()
    if data["status"] == "success"
]

if baseline_name in spectra and success_names:
    base_ell = spectra[baseline_name]["ell"]
    base_curve = spectra[baseline_name]["lensing_scaled"]

    plt.figure(figsize=(10, 6))

    for name in success_names:
        if name == baseline_name:
            continue

        ell = spectra[name]["ell"]
        curve = spectra[name]["lensing_scaled"]

        if not np.array_equal(ell, base_ell):
            curve_interp = np.interp(base_ell, ell, curve)
        else:
            curve_interp = curve

        ratio = curve_interp / base_curve

        mask = (
            (base_ell >= ELL_MIN)
            & (base_ell <= ELL_MAX)
            & np.isfinite(ratio)
        )

        plt.plot(base_ell[mask], ratio[mask], label=name, linewidth=1.1)

    plt.axhline(1.0, linewidth=1)
    plt.xlabel("ell")
    plt.ylabel("CMB lensing ratio to CDM baseline")
    plt.title("CMB lensing pressure test")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("cmb_lensing_pressure_ratio_plot.png", dpi=160)
    plt.close()

    labels = success_names
    s8_values = [results[name]["S8"] for name in labels]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    plt.bar(x, s8_values)
    plt.axhline(0.80, linewidth=1)
    plt.xticks(x, labels, rotation=55, ha="right", fontsize=8)
    plt.ylabel("S8")
    plt.title("S8 by CMB lensing-pressure case")
    plt.tight_layout()
    plt.savefig("cmb_lensing_pressure_s8_plot.png", dpi=160)
    plt.close()


print("")
print("TAIRID CMB lensing pressure test complete.")
print("Created:")
print("  cmb_lensing_pressure_summary.json")
print("  cmb_lensing_pressure_summary.csv")
print("  cmb_lensing_ratio_curves.csv")
print("  cmb_lensing_pressure_ratio_plot.png")
print("  cmb_lensing_pressure_s8_plot.png")
print("")
print("Read boundary:")
print("  This is not a Planck lensing likelihood.")
print("  It only checks internal CMB lensing-spectrum drift relative to the CDM proxy baseline.")
