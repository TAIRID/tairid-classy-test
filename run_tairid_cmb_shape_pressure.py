#!/usr/bin/env python3
"""
TAIRID CMB peak-height / spectrum-shape pressure test.

Purpose:
Earlier proxy tests showed that the candidate lanes preserve simplified
CMB acoustic peak positions near:

    ell = 221, 538, 815

But peak positions are not enough.

This test checks whether the candidate lanes distort the CMB TT spectrum
shape and peak heights too much relative to the CDM proxy baseline.

Boundary:
This is still a proxy pressure test.
It is not a Planck likelihood.
It does not test polarization, lensing likelihoods, nuisance parameters,
foregrounds, or full covariance.
It only compares internal CLASS TT spectrum shape against the baseline.
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

PEAK_WINDOWS = [
    (100, 350),
    (350, 650),
    (650, 1000),
]

ELL_COMPARE_MIN = 30
ELL_COMPARE_MAX = 2000

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
    {"name": "best_combined_5pct_20eV", "warm_fraction": 0.05, "m_ncdm_eV": 20.0},
    {"name": "secondary_5pct_25eV", "warm_fraction": 0.05, "m_ncdm_eV": 25.0},
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


def run_class_case(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    sigma8 = float(cosmo.sigma8())

    cl = cosmo.lensed_cl(2500)
    ell = np.asarray(cl["ell"])
    tt = np.asarray(cl["tt"])

    dl_tt = ell * (ell + 1) * tt / (2.0 * math.pi)

    cosmo.struct_cleanup()
    cosmo.empty()

    return sigma8, ell, dl_tt


results = {}
spectra = {}

for case in CASES:
    name = case["name"]
    params = make_params(case)
    omega_m, Omega_m = omega_total_matter_proxy(params)

    print("")
    print("Running case:", name)
    print("  warm_fraction:", case["warm_fraction"])
    print("  m_ncdm_eV:", case["m_ncdm_eV"])

    try:
        sigma8, ell, dl_tt = run_class_case(params)
        S8 = s8_from_sigma8(sigma8, Omega_m)
        peak_ell, peak_height = peak_by_windows(ell, dl_tt)

        spectrum_file = f"{name}_cmb_tt_spectrum.txt"
        np.savetxt(
            spectrum_file,
            np.column_stack([ell, dl_tt]),
            header="ell D_ell_TT_internal_units",
        )

        results[name] = {
            "status": "success",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "Omega_m_total": float(Omega_m),
            "sigma8": sigma8,
            "S8": S8,
            "peak_ell": peak_ell,
            "peak_height": peak_height,
            "spectrum_file": spectrum_file,
        }

        spectra[name] = {
            "ell": ell,
            "dl_tt": dl_tt,
        }

        print("  success")
        print("  S8:", S8)
        print("  peaks:", peak_ell)

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
    base_dl = spectra[baseline_name]["dl_tt"]

    compare_mask = (
        (base_ell >= ELL_COMPARE_MIN)
        & (base_ell <= ELL_COMPARE_MAX)
        & np.isfinite(base_dl)
        & (base_dl > 0)
    )

    for name, spec in spectra.items():
        ell = spec["ell"]
        dl = spec["dl_tt"]

        if not np.array_equal(ell, base_ell):
            dl_interp = np.interp(base_ell, ell, dl)
        else:
            dl_interp = dl

        ratio = np.full_like(base_dl, np.nan, dtype=float)
        good = np.isfinite(dl_interp) & np.isfinite(base_dl) & (base_dl > 0)
        ratio[good] = dl_interp[good] / base_dl[good]

        ratio_file = f"{name}_cmb_tt_ratio_to_cdm.txt"
        np.savetxt(
            ratio_file,
            np.column_stack([base_ell, ratio]),
            header="ell D_ell_TT_ratio_to_CDM",
        )

        results[name]["ratio_file"] = ratio_file

        diff = ratio[compare_mask] - 1.0
        abs_diff = np.abs(diff)

        if len(diff) > 0:
            results[name]["tt_ratio_rms_ell30_2000"] = float(np.sqrt(np.nanmean(diff * diff)))
            results[name]["tt_ratio_max_abs_ell30_2000"] = float(np.nanmax(abs_diff))
            results[name]["tt_ratio_mean_abs_ell30_2000"] = float(np.nanmean(abs_diff))
        else:
            results[name]["tt_ratio_rms_ell30_2000"] = None
            results[name]["tt_ratio_max_abs_ell30_2000"] = None
            results[name]["tt_ratio_mean_abs_ell30_2000"] = None

        if name != baseline_name and results[name]["status"] == "success":
            base_result = results[baseline_name]

            results[name]["delta_vs_cdm_baseline"] = {
                "delta_S8": float(results[name]["S8"] - base_result["S8"]),
                "delta_sigma8": float(results[name]["sigma8"] - base_result["sigma8"]),
                "peak1_shift": int(results[name]["peak_ell"][0] - base_result["peak_ell"][0]),
                "peak2_shift": int(results[name]["peak_ell"][1] - base_result["peak_ell"][1]),
                "peak3_shift": int(results[name]["peak_ell"][2] - base_result["peak_ell"][2]),
                "peak1_height_ratio": float(results[name]["peak_height"][0] / base_result["peak_height"][0]),
                "peak2_height_ratio": float(results[name]["peak_height"][1] / base_result["peak_height"][1]),
                "peak3_height_ratio": float(results[name]["peak_height"][2] / base_result["peak_height"][2]),
            }


for name, data in results.items():
    if data["status"] != "success":
        continue

    if name == baseline_name:
        data["diagnostic_flag"] = "baseline"
        continue

    rms = data.get("tt_ratio_rms_ell30_2000")
    max_abs = data.get("tt_ratio_max_abs_ell30_2000")
    delta = data.get("delta_vs_cdm_baseline", {})

    peak_shifts = [
        abs(delta.get("peak1_shift", 999)),
        abs(delta.get("peak2_shift", 999)),
        abs(delta.get("peak3_shift", 999)),
    ]

    if rms is None or max_abs is None:
        data["diagnostic_flag"] = "warning_missing_shape_metrics"
    elif max(peak_shifts) > 3:
        data["diagnostic_flag"] = "fail_peak_position_shift"
    elif rms <= 0.01 and max_abs <= 0.03:
        data["diagnostic_flag"] = "shape_close_to_baseline"
    elif rms <= 0.03 and max_abs <= 0.08:
        data["diagnostic_flag"] = "shape_warning_but_survives_proxy"
    else:
        data["diagnostic_flag"] = "shape_distortion_warning"


Path("cmb_shape_pressure_summary.json").write_text(json.dumps(results, indent=2))


with open("cmb_shape_pressure_summary.csv", "w", newline="") as f:
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
            "ell_peak1",
            "ell_peak2",
            "ell_peak3",
            "peak1_height_ratio_vs_cdm",
            "peak2_height_ratio_vs_cdm",
            "peak3_height_ratio_vs_cdm",
            "tt_ratio_rms_ell30_2000",
            "tt_ratio_max_abs_ell30_2000",
            "tt_ratio_mean_abs_ell30_2000",
            "delta_S8_vs_cdm",
            "error",
        ]
    )

    for name, data in results.items():
        if data["status"] == "success":
            peaks = data["peak_ell"]
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
                    peaks[0],
                    peaks[1],
                    peaks[2],
                    delta.get("peak1_height_ratio"),
                    delta.get("peak2_height_ratio"),
                    delta.get("peak3_height_ratio"),
                    data.get("tt_ratio_rms_ell30_2000"),
                    data.get("tt_ratio_max_abs_ell30_2000"),
                    data.get("tt_ratio_mean_abs_ell30_2000"),
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
    base_dl = spectra[baseline_name]["dl_tt"]

    with open("cmb_tt_ratio_curves.csv", "w", newline="") as f:
        writer = csv.writer(f)

        header = ["ell"]
        for name in spectra:
            header.append(name + "_TT_ratio_to_CDM")
        writer.writerow(header)

        for i, ell_value in enumerate(base_ell):
            row = [int(ell_value)]

            for name, spec in spectra.items():
                dl = spec["dl_tt"]
                ell = spec["ell"]

                if not np.array_equal(ell, base_ell):
                    dl_interp = np.interp(base_ell, ell, dl)
                else:
                    dl_interp = dl

                if base_dl[i] > 0 and np.isfinite(base_dl[i]) and np.isfinite(dl_interp[i]):
                    row.append(float(dl_interp[i] / base_dl[i]))
                else:
                    row.append("")

            writer.writerow(row)


success_names = [
    name for name, data in results.items()
    if data["status"] == "success"
]

if baseline_name in spectra and success_names:
    plt.figure(figsize=(10, 6))

    base_ell = spectra[baseline_name]["ell"]
    base_dl = spectra[baseline_name]["dl_tt"]

    for name in success_names:
        if name == baseline_name:
            continue

        ell = spectra[name]["ell"]
        dl = spectra[name]["dl_tt"]

        if not np.array_equal(ell, base_ell):
            dl_interp = np.interp(base_ell, ell, dl)
        else:
            dl_interp = dl

        ratio = dl_interp / base_dl

        mask = (
            (base_ell >= ELL_COMPARE_MIN)
            & (base_ell <= ELL_COMPARE_MAX)
            & np.isfinite(ratio)
        )

        plt.plot(base_ell[mask], ratio[mask], label=name, linewidth=1.1)

    plt.axhline(1.0, linewidth=1)
    plt.xlabel("ell")
    plt.ylabel("TT ratio to CDM baseline")
    plt.title("CMB TT spectrum-shape pressure test")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("cmb_shape_pressure_tt_ratio_plot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))

    labels = success_names
    s8_values = [results[name]["S8"] for name in labels]

    x = np.arange(len(labels))
    plt.bar(x, s8_values)
    plt.axhline(0.80, linewidth=1)
    plt.xticks(x, labels, rotation=55, ha="right", fontsize=8)
    plt.ylabel("S8")
    plt.title("S8 by CMB shape-pressure case")
    plt.tight_layout()
    plt.savefig("cmb_shape_pressure_s8_plot.png", dpi=160)
    plt.close()


print("")
print("TAIRID CMB shape pressure test complete.")
print("Created:")
print("  cmb_shape_pressure_summary.json")
print("  cmb_shape_pressure_summary.csv")
print("  cmb_tt_ratio_curves.csv")
print("  cmb_shape_pressure_tt_ratio_plot.png")
print("  cmb_shape_pressure_s8_plot.png")
print("")
print("Read boundary:")
print("  This is not a Planck likelihood.")
print("  It only checks internal TT shape drift relative to the CDM proxy baseline.")
