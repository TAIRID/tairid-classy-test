#!/usr/bin/env python3
"""
TAIRID warm-neutral sigma8 / S8 scan.

Purpose:
The previous warm-neutral proxy scan showed that small warm/non-cold neutral
fractions can preserve CMB peak locations while suppressing small-scale matter
power P(k).

This test asks the next pressure question:

    What sigma8 and S8 does each warm-neutral case produce?

Boundary:
This is still a CLASS proxy test, not final TAIRID.
The non-cold piece is implemented using CLASS ncdm as a proxy.
"""

import csv
import json
import math
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from classy import Class


BASE = {
    "output": "tCl,pCl,lCl,mPk",
    "lensing": "yes",
    "h": 0.66893180,
    "omega_b": 0.0223700000,
    "omega_cdm": 0.1200000000,
    "N_ur": 3.046,
    "Omega_k": 0.0,
    "Omega_Lambda": 0.6817397872,
    "n_s": 0.9649,
    "A_s": 2.100549e-9,
    "tau_reio": 0.0544,
    "YHe": 0.245,
    "T_cmb": 2.7255,
    "l_max_scalars": 2500,
    "P_k_max_1/Mpc": 10.0,
    "z_pk": "0",
}

OMEGA_NEUTRAL_PHYSICAL = 0.1200000000
H = 0.66893180

PEAK_WINDOWS = [
    (100, 350),
    (350, 650),
    (650, 1000),
]

CASES = [
    {"name": "cdm_baseline", "warm_fraction": 0.0, "m_ncdm_eV": None},
    {"name": "warm_1pct_1000eV", "warm_fraction": 0.01, "m_ncdm_eV": 1000.0},
    {"name": "warm_5pct_1000eV", "warm_fraction": 0.05, "m_ncdm_eV": 1000.0},
    {"name": "warm_10pct_1000eV", "warm_fraction": 0.10, "m_ncdm_eV": 1000.0},
    {"name": "warm_1pct_100eV", "warm_fraction": 0.01, "m_ncdm_eV": 100.0},
    {"name": "warm_5pct_100eV", "warm_fraction": 0.05, "m_ncdm_eV": 100.0},
    {"name": "warm_10pct_100eV", "warm_fraction": 0.10, "m_ncdm_eV": 100.0},
]


SURVEY_S8 = [
    {"name": "Planck_2018_reference", "S8": 0.834, "sigma": 0.016},
    {"name": "DES_Y6_NLA", "S8": 0.798, "sigma": 0.015},
    {"name": "DES_Y6_TATT", "S8": 0.783, "sigma": 0.017},
    {"name": "DES_Y3_3x2pt", "S8": 0.776, "sigma": 0.017},
    {"name": "KiDS_Legacy", "S8": 0.815, "sigma": 0.019},
    {"name": "HSC_Y3", "S8": 0.776, "sigma": 0.033},
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


def run_class(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    sigma8 = float(cosmo.sigma8())

    cl = cosmo.lensed_cl(2500)
    ell = np.asarray(cl["ell"])
    tt = np.asarray(cl["tt"])

    k_values = np.logspace(-3, 0, 80)
    pk_values = []
    for k in k_values:
        try:
            pk_values.append(float(cosmo.pk(float(k), 0.0)))
        except Exception:
            pk_values.append(np.nan)

    cosmo.struct_cleanup()
    cosmo.empty()

    return sigma8, ell, tt, k_values, np.asarray(pk_values)


def dl_from_cl(ell, cl):
    return ell * (ell + 1) * cl / (2.0 * np.pi)


def peak_by_windows(ell, dl):
    peak_ells = []
    peak_heights = []

    for lo, hi in PEAK_WINDOWS:
        mask = (ell >= lo) & (ell <= hi)
        ell_window = ell[mask]
        dl_window = dl[mask]
        idx = int(np.argmax(dl_window))
        peak_ells.append(int(ell_window[idx]))
        peak_heights.append(float(dl_window[idx]))

    return peak_ells, peak_heights


def s8_from_sigma8(sigma8, Omega_m):
    return float(sigma8 * math.sqrt(Omega_m / 0.3))


results = {}
spectra = {}
pk_tables = {}

for case in CASES:
    name = case["name"]
    params = make_params(case)
    omega_m, Omega_m = omega_total_matter_proxy(params)

    print("\nRunning case:", name)
    print("  warm fraction:", case["warm_fraction"])
    print("  m_ncdm_eV:", case["m_ncdm_eV"])
    print("  omega_cdm:", params.get("omega_cdm"))
    print("  Omega_ncdm:", params.get("Omega_ncdm"))
    print("  omega_m total:", omega_m)
    print("  Omega_m total:", Omega_m)

    try:
        sigma8, ell, tt, k_values, pk_values = run_class(params)
        S8 = s8_from_sigma8(sigma8, Omega_m)
        dl = dl_from_cl(ell, tt)
        peaks_ell, peaks_height = peak_by_windows(ell, dl)

        cl_file = f"{name}_cl_tt.txt"
        pk_file = f"{name}_pk_z0.txt"

        np.savetxt(
            cl_file,
            np.column_stack([ell, tt, dl]),
            header="ell C_l_TT_raw D_l_TT_raw",
        )

        np.savetxt(
            pk_file,
            np.column_stack([k_values, pk_values]),
            header="k_raw_1_over_Mpc Pk_z0_raw",
        )

        survey_tensions = {}
        for obs in SURVEY_S8:
            delta = S8 - obs["S8"]
            survey_tensions[obs["name"]] = {
                "observed_S8": obs["S8"],
                "sigma": obs["sigma"],
                "delta_model_minus_observed": float(delta),
                "z_score_abs": float(abs(delta) / obs["sigma"]),
            }

        results[name] = {
            "status": "success",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "omega_cdm": float(params.get("omega_cdm", np.nan)),
            "Omega_ncdm": None if "Omega_ncdm" not in params else float(params["Omega_ncdm"]),
            "omega_m_total": float(omega_m),
            "Omega_m_total": float(Omega_m),
            "sigma8": float(sigma8),
            "S8": float(S8),
            "peak_ell": peaks_ell,
            "survey_tensions": survey_tensions,
            "cl_file": cl_file,
            "pk_file": pk_file,
        }

        spectra[name] = {"ell": ell, "dl": dl}
        pk_tables[name] = {"k": k_values, "pk": pk_values}

        print("  success")
        print("  sigma8:", sigma8)
        print("  S8:", S8)
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


if "cdm_baseline" in results and results["cdm_baseline"]["status"] == "success":
    base = results["cdm_baseline"]

    for name, data in results.items():
        if data["status"] != "success":
            continue

        data["delta_vs_cdm_baseline"] = {
            "sigma8_delta": float(data["sigma8"] - base["sigma8"]),
            "sigma8_fractional_delta": float((data["sigma8"] - base["sigma8"]) / base["sigma8"]),
            "S8_delta": float(data["S8"] - base["S8"]),
            "S8_fractional_delta": float((data["S8"] - base["S8"]) / base["S8"]),
            "peak_ell_delta": [
                int(data["peak_ell"][i] - base["peak_ell"][i])
                for i in range(3)
            ],
        }

        if name in pk_tables:
            k = pk_tables[name]["k"]
            pk = pk_tables[name]["pk"]
            base_pk = pk_tables["cdm_baseline"]["pk"]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.divide(pk, base_pk, out=np.full_like(pk, np.nan), where=base_pk != 0)

            data["pk_ratio_at_k"] = {
                "k_0p01": float(np.interp(0.01, k, ratio)),
                "k_0p1": float(np.interp(0.1, k, ratio)),
                "k_1p0": float(np.interp(1.0, k, ratio)),
            }


Path("warm_neutral_s8_scan_summary.json").write_text(json.dumps(results, indent=2))


with open("warm_neutral_s8_scan_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "case",
        "status",
        "warm_fraction",
        "m_ncdm_eV",
        "omega_cdm",
        "Omega_ncdm",
        "omega_m_total",
        "Omega_m_total",
        "sigma8",
        "S8",
        "delta_sigma8_vs_cdm",
        "frac_delta_sigma8_vs_cdm",
        "delta_S8_vs_cdm",
        "frac_delta_S8_vs_cdm",
        "ell_peak1",
        "ell_peak2",
        "ell_peak3",
        "pk_ratio_k0p01",
        "pk_ratio_k0p1",
        "pk_ratio_k1p0",
        "DES_Y6_NLA_z",
        "DES_Y6_TATT_z",
        "KiDS_Legacy_z",
        "Planck_2018_z",
        "error",
    ])

    for name, data in results.items():
        if data["status"] == "success":
            d = data.get("delta_vs_cdm_baseline", {})
            pkd = data.get("pk_ratio_at_k", {})
            tensions = data.get("survey_tensions", {})

            writer.writerow([
                name,
                data["status"],
                data["warm_fraction"],
                data["m_ncdm_eV"],
                data["omega_cdm"],
                data["Omega_ncdm"],
                data["omega_m_total"],
                data["Omega_m_total"],
                data["sigma8"],
                data["S8"],
                d.get("sigma8_delta"),
                d.get("sigma8_fractional_delta"),
                d.get("S8_delta"),
                d.get("S8_fractional_delta"),
                *data["peak_ell"],
                pkd.get("k_0p01"),
                pkd.get("k_0p1"),
                pkd.get("k_1p0"),
                tensions.get("DES_Y6_NLA", {}).get("z_score_abs"),
                tensions.get("DES_Y6_TATT", {}).get("z_score_abs"),
                tensions.get("KiDS_Legacy", {}).get("z_score_abs"),
                tensions.get("Planck_2018_reference", {}).get("z_score_abs"),
                "",
            ])
        else:
            writer.writerow([
                name,
                data["status"],
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
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                data.get("error", ""),
            ])


success_names = [name for name, data in results.items() if data["status"] == "success"]

if success_names:
    labels = success_names
    s8_values = [results[name]["S8"] for name in labels]
    sigma8_values = [results[name]["sigma8"] for name in labels]

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 5.5))
    plt.bar(x, s8_values)
    for obs in SURVEY_S8:
        if obs["name"] in ["DES_Y6_NLA", "DES_Y6_TATT", "KiDS_Legacy", "Planck_2018_reference"]:
            plt.axhline(obs["S8"], linewidth=1, label=obs["name"])
    plt.xticks(x, labels, rotation=55, ha="right", fontsize=8)
    plt.ylabel("S8")
    plt.title("Warm neutral proxy scan: S8 by case")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("warm_neutral_s8_values.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5.5))
    plt.bar(x, sigma8_values)
    plt.xticks(x, labels, rotation=55, ha="right", fontsize=8)
    plt.ylabel("sigma8")
    plt.title("Warm neutral proxy scan: sigma8 by case")
    plt.tight_layout()
    plt.savefig("warm_neutral_sigma8_values.png", dpi=160)
    plt.close()

if "cdm_baseline" in pk_tables:
    k = pk_tables["cdm_baseline"]["k"]
    base_pk = pk_tables["cdm_baseline"]["pk"]

    plt.figure(figsize=(10, 5))
    for name, table in pk_tables.items():
        if name == "cdm_baseline":
            continue
        ratio = np.divide(table["pk"], base_pk, out=np.full_like(base_pk, np.nan), where=base_pk != 0)
        plt.plot(k, ratio, label=name, linewidth=1.1)
    plt.axhline(1.0, linewidth=1)
    plt.xscale("log")
    plt.xlabel("k raw 1/Mpc")
    plt.ylabel("P(k) ratio to CDM baseline")
    plt.title("Warm neutral proxy scan: matter power ratio at z=0")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("warm_neutral_s8_pk_ratio.png", dpi=160)
    plt.close()


print("\nWarm neutral S8 scan complete.")
print("Created:")
print("  warm_neutral_s8_scan_summary.json")
print("  warm_neutral_s8_scan_summary.csv")
print("  warm_neutral_s8_values.png")
print("  warm_neutral_sigma8_values.png")
print("  warm_neutral_s8_pk_ratio.png")
