#!/usr/bin/env python3
"""
TAIRID warm / non-cold neutral-substrate proxy scan.

Purpose:
The previous tests showed that TAIRID's gate-derived neutral substrate fits the
CMB acoustic peaks when it is treated as CDM-like with:

    omega_neutral = 0.1200000000

This test asks whether a small part of that neutral substrate can behave
non-cold / free-streaming without breaking the CMB peaks.

Boundary:
This is still a CLASS proxy test, not final TAIRID.
It does not add a true TAIRID perturbation equation or a real c_N^2 field.
It uses CLASS's standard ncdm component as a proxy for non-cold behavior.

Important:
If CLASS rejects any ncdm parameter combination, the script records the error
and continues. A failed case will not kill the workflow.
"""

import csv
import json
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


def run_class(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

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

    return ell, tt, k_values, np.asarray(pk_values)


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


def peak_ratios(heights):
    return float(heights[1] / heights[0]), float(heights[2] / heights[0])


results = {}
spectra = {}
pk_tables = {}

for case in CASES:
    name = case["name"]
    params = make_params(case)

    print("\nRunning case:", name)
    print("  warm fraction:", case["warm_fraction"])
    print("  m_ncdm_eV:", case["m_ncdm_eV"])
    print("  omega_cdm:", params.get("omega_cdm"))
    print("  Omega_ncdm:", params.get("Omega_ncdm"))

    try:
        ell, tt, k_values, pk_values = run_class(params)
        dl = dl_from_cl(ell, tt)
        peaks_ell, peaks_height = peak_by_windows(ell, dl)
        ratio21, ratio31 = peak_ratios(peaks_height)

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

        results[name] = {
            "status": "success",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "omega_cdm": float(params.get("omega_cdm", np.nan)),
            "Omega_ncdm": None if "Omega_ncdm" not in params else float(params["Omega_ncdm"]),
            "peak_ell": peaks_ell,
            "peak_Dl_raw": peaks_height,
            "ratio_peak2_to_peak1": ratio21,
            "ratio_peak3_to_peak1": ratio31,
            "cl_file": cl_file,
            "pk_file": pk_file,
        }

        spectra[name] = {"ell": ell, "dl": dl}
        pk_tables[name] = {"k": k_values, "pk": pk_values}

        print("  success")
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
            "peak_ell_delta": [
                int(data["peak_ell"][i] - base["peak_ell"][i])
                for i in range(3)
            ],
            "peak_height_fractional_delta": [
                float(
                    (data["peak_Dl_raw"][i] - base["peak_Dl_raw"][i])
                    / base["peak_Dl_raw"][i]
                )
                for i in range(3)
            ],
            "ratio21_delta": float(
                data["ratio_peak2_to_peak1"] - base["ratio_peak2_to_peak1"]
            ),
            "ratio31_delta": float(
                data["ratio_peak3_to_peak1"] - base["ratio_peak3_to_peak1"]
            ),
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


Path("warm_neutral_proxy_scan_summary.json").write_text(json.dumps(results, indent=2))


with open("warm_neutral_proxy_scan_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "case",
        "status",
        "warm_fraction",
        "m_ncdm_eV",
        "omega_cdm",
        "Omega_ncdm",
        "ell_peak1",
        "ell_peak2",
        "ell_peak3",
        "ratio21",
        "ratio31",
        "delta_ell1",
        "delta_ell2",
        "delta_ell3",
        "frac_delta_height1",
        "frac_delta_height2",
        "frac_delta_height3",
        "pk_ratio_k0p01",
        "pk_ratio_k0p1",
        "pk_ratio_k1p0",
        "error",
    ])

    for name, data in results.items():
        if data["status"] == "success":
            d = data.get("delta_vs_cdm_baseline", {})
            pkd = data.get("pk_ratio_at_k", {})
            writer.writerow([
                name,
                data["status"],
                data["warm_fraction"],
                data["m_ncdm_eV"],
                data["omega_cdm"],
                data["Omega_ncdm"],
                *data["peak_ell"],
                data["ratio_peak2_to_peak1"],
                data["ratio_peak3_to_peak1"],
                *(d.get("peak_ell_delta", [None, None, None])),
                *(d.get("peak_height_fractional_delta", [None, None, None])),
                pkd.get("k_0p01"),
                pkd.get("k_0p1"),
                pkd.get("k_1p0"),
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
                data.get("error", ""),
            ])


if "cdm_baseline" in spectra:
    base_ell = spectra["cdm_baseline"]["ell"]
    base_dl = spectra["cdm_baseline"]["dl"]

    plt.figure(figsize=(12, 6))
    for name, spec in spectra.items():
        plt.plot(spec["ell"], spec["dl"], label=name, linewidth=1.1)
    plt.xlabel("multipole ell")
    plt.ylabel("raw D_ell_TT")
    plt.title("Warm neutral proxy scan: TT spectra")
    plt.xlim(0, 1600)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("warm_neutral_proxy_tt_spectra.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    for name, spec in spectra.items():
        if name == "cdm_baseline":
            continue
        interp = np.interp(base_ell, spec["ell"], spec["dl"])
        ratio = np.divide(interp, base_dl, out=np.full_like(base_dl, np.nan), where=base_dl != 0)
        plt.plot(base_ell, ratio, label=name, linewidth=1.1)
    plt.axhline(1.0, linewidth=1)
    plt.xlabel("multipole ell")
    plt.ylabel("D_ell ratio to CDM baseline")
    plt.title("Warm neutral proxy scan: TT ratio")
    plt.xlim(2, 1600)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("warm_neutral_proxy_tt_ratio.png", dpi=160)
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
    plt.savefig("warm_neutral_proxy_pk_ratio.png", dpi=160)
    plt.close()


print("\nWarm neutral proxy scan complete.")
print("Created:")
print("  warm_neutral_proxy_scan_summary.json")
print("  warm_neutral_proxy_scan_summary.csv")
print("  warm_neutral_proxy_tt_spectra.png")
print("  warm_neutral_proxy_tt_ratio.png")
print("  warm_neutral_proxy_pk_ratio.png")
