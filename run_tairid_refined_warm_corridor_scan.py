#!/usr/bin/env python3
"""
TAIRID refined warm-neutral corridor scan.

Purpose:
The small-scale sanity scan identified the current best candidate corridor near:

    warm fraction ~ 10 percent
    proxy mass ~ 30 eV

This workflow refines that region:

    warm_fraction = 5%, 7.5%, 10%, 12.5%, 15%
    m_ncdm_eV    = 20, 25, 30, 35, 40, 50

It asks whether there is a gentler point that still lowers S8 toward the lensing
range without suppressing small-scale matter power too severely.

Boundary:
This is still a CLASS proxy test, not final TAIRID.
The non-cold neutral part is represented by CLASS ncdm as a stand-in for delayed
consolidation / non-cold neutral behavior.
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
    "P_k_max_1/Mpc": 50.0,
    "z_pk": "0",
}

OMEGA_NEUTRAL_PHYSICAL = 0.1200000000
H = 0.66893180
TARGET_S8 = 0.800

PEAK_WINDOWS = [
    (100, 350),
    (350, 650),
    (650, 1000),
]

WARM_FRACTIONS = [
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
]

M_NCDM_EV_VALUES = [
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    50.0,
]

K_PROBES = [
    0.1,
    0.3,
    0.5,
    1.0,
    3.0,
    5.0,
    10.0,
]


def build_cases():
    cases = [{"name": "cdm_baseline", "warm_fraction": 0.0, "m_ncdm_eV": None}]

    for fraction in WARM_FRACTIONS:
        for mass in M_NCDM_EV_VALUES:
            fraction_label = str(fraction * 100).replace(".", "p")
            if fraction_label.endswith("0"):
                fraction_label = fraction_label[:-1]
            mass_label = int(round(mass))
            cases.append(
                {
                    "name": f"warm_{fraction_label}pct_{mass_label}eV",
                    "warm_fraction": fraction,
                    "m_ncdm_eV": mass,
                }
            )

    return cases


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

    k_values = np.logspace(-3, 1, 150)
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


def s8_from_sigma8(sigma8, Omega_m):
    return float(sigma8 * math.sqrt(Omega_m / 0.3))


def safe_interp(x, xp, fp):
    good = np.isfinite(fp)
    if np.sum(good) < 2:
        return float("nan")
    return float(np.interp(x, xp[good], fp[good]))


def corridor_score(S8, pk1, pk10, peaks_locked):
    """
    Lower is better.
    We want:
    - S8 close to 0.800
    - P(k=1) not too suppressed
    - P(k=10) not erased
    - peaks locked
    """
    if not peaks_locked:
        return 999.0
    if not np.isfinite(pk1) or not np.isfinite(pk10):
        return 999.0

    s8_penalty = abs(S8 - TARGET_S8) / 0.01
    pk1_penalty = max(0.0, 0.50 - pk1) / 0.10
    pk10_penalty = max(0.0, 0.25 - pk10) / 0.10

    return float(s8_penalty + pk1_penalty + 0.5 * pk10_penalty)


def sanity_label(S8, pk1, pk10, peaks_locked):
    if not peaks_locked:
        return "peak drift"

    if abs(S8 - TARGET_S8) <= 0.015 and pk1 >= 0.50 and pk10 >= 0.25:
        return "best corridor candidate"

    if abs(S8 - TARGET_S8) <= 0.020 and pk1 >= 0.50:
        return "candidate with small-scale warning"

    if abs(S8 - TARGET_S8) <= 0.020:
        return "near S8 target but too much small-scale suppression"

    if pk1 >= 0.75:
        return "small-scale safe but S8 high"

    if pk1 >= 0.50:
        return "moderate suppression"

    return "strong suppression"


results = {}
spectra = {}
pk_tables = {}

for case in build_cases():
    name = case["name"]
    params = make_params(case)
    omega_m, Omega_m = omega_total_matter_proxy(params)

    print("\nRunning case:", name)
    print("  warm fraction:", case["warm_fraction"])
    print("  m_ncdm_eV:", case["m_ncdm_eV"])
    print("  omega_cdm:", params.get("omega_cdm"))
    print("  Omega_ncdm:", params.get("Omega_ncdm"))
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
            "S8_distance_to_target_0p800": float(abs(S8 - TARGET_S8)),
            "peak_ell": peaks_ell,
            "peak_Dl_raw": peaks_height,
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

        peak_delta = [
            None if data["peak_ell"][i] is None or base["peak_ell"][i] is None
            else int(data["peak_ell"][i] - base["peak_ell"][i])
            for i in range(3)
        ]
        peaks_locked = peak_delta == [0, 0, 0]

        data["delta_vs_cdm_baseline"] = {
            "sigma8_delta": float(data["sigma8"] - base["sigma8"]),
            "sigma8_fractional_delta": float((data["sigma8"] - base["sigma8"]) / base["sigma8"]),
            "S8_delta": float(data["S8"] - base["S8"]),
            "S8_fractional_delta": float((data["S8"] - base["S8"]) / base["S8"]),
            "peak_ell_delta": peak_delta,
        }

        data["peaks_locked"] = bool(peaks_locked)

        if name in pk_tables:
            k = pk_tables[name]["k"]
            pk = pk_tables[name]["pk"]
            base_pk = pk_tables["cdm_baseline"]["pk"]

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.divide(pk, base_pk, out=np.full_like(pk, np.nan), where=base_pk != 0)

            pk_ratios = {}
            for probe in K_PROBES:
                key = "k_" + str(probe).replace(".", "p")
                pk_ratios[key] = safe_interp(probe, k, ratio)

            data["pk_ratio_at_k"] = pk_ratios

            pk1 = pk_ratios["k_1p0"]
            pk10 = pk_ratios["k_10p0"]

            data["corridor_score"] = corridor_score(data["S8"], pk1, pk10, peaks_locked)
            data["corridor_label"] = sanity_label(data["S8"], pk1, pk10, peaks_locked)


success_items = [
    (name, data)
    for name, data in results.items()
    if data["status"] == "success"
]

candidate_items = [
    (name, data)
    for name, data in success_items
    if name != "cdm_baseline"
]

candidate_items_sorted = sorted(
    candidate_items,
    key=lambda item: item[1].get("corridor_score", 999.0),
)

best_summary = {
    name: {
        "warm_fraction": data["warm_fraction"],
        "m_ncdm_eV": data["m_ncdm_eV"],
        "S8": data["S8"],
        "sigma8": data["sigma8"],
        "peak_ell": data["peak_ell"],
        "peaks_locked": data.get("peaks_locked"),
        "pk_ratio_at_k": data.get("pk_ratio_at_k", {}),
        "corridor_score": data.get("corridor_score"),
        "corridor_label": data.get("corridor_label"),
    }
    for name, data in candidate_items_sorted[:12]
}

Path("refined_warm_corridor_best_cases.json").write_text(json.dumps(best_summary, indent=2))
Path("refined_warm_corridor_scan_summary.json").write_text(json.dumps(results, indent=2))


with open("refined_warm_corridor_scan_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "case",
        "status",
        "warm_fraction",
        "m_ncdm_eV",
        "Omega_m_total",
        "sigma8",
        "S8",
        "distance_to_S8_0p800",
        "ell_peak1",
        "ell_peak2",
        "ell_peak3",
        "peaks_locked",
        "P_ratio_k0p1",
        "P_ratio_k0p3",
        "P_ratio_k0p5",
        "P_ratio_k1p0",
        "P_ratio_k3p0",
        "P_ratio_k5p0",
        "P_ratio_k10p0",
        "corridor_score",
        "corridor_label",
        "error",
    ])

    for name, data in results.items():
        if data["status"] == "success":
            pkd = data.get("pk_ratio_at_k", {})
            writer.writerow([
                name,
                data["status"],
                data["warm_fraction"],
                data["m_ncdm_eV"],
                data["Omega_m_total"],
                data["sigma8"],
                data["S8"],
                data["S8_distance_to_target_0p800"],
                *data["peak_ell"],
                data.get("peaks_locked"),
                pkd.get("k_0p1"),
                pkd.get("k_0p3"),
                pkd.get("k_0p5"),
                pkd.get("k_1p0"),
                pkd.get("k_3p0"),
                pkd.get("k_5p0"),
                pkd.get("k_10p0"),
                data.get("corridor_score"),
                data.get("corridor_label"),
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
                data.get("error", ""),
            ])


# Plots
grid_items = [
    data
    for name, data in candidate_items
    if data.get("pk_ratio_at_k")
]

if grid_items:
    masses = np.array([d["m_ncdm_eV"] for d in grid_items], dtype=float)
    fractions = np.array([d["warm_fraction"] for d in grid_items], dtype=float) * 100.0
    s8_vals = np.array([d["S8"] for d in grid_items], dtype=float)
    pk1_vals = np.array([d["pk_ratio_at_k"]["k_1p0"] for d in grid_items], dtype=float)
    pk10_vals = np.array([d["pk_ratio_at_k"]["k_10p0"] for d in grid_items], dtype=float)
    scores = np.array([d.get("corridor_score", np.nan) for d in grid_items], dtype=float)

    plt.figure(figsize=(8.5, 5.8))
    sc = plt.scatter(masses, fractions, c=s8_vals, s=140)
    plt.xscale("log")
    plt.xlabel("m_ncdm eV")
    plt.ylabel("warm fraction percent")
    plt.title("Refined corridor scan: S8")
    cbar = plt.colorbar(sc)
    cbar.set_label("S8")
    plt.tight_layout()
    plt.savefig("refined_warm_corridor_S8_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8.5, 5.8))
    sc = plt.scatter(masses, fractions, c=pk1_vals, s=140)
    plt.xscale("log")
    plt.xlabel("m_ncdm eV")
    plt.ylabel("warm fraction percent")
    plt.title("Refined corridor scan: P(k=1) ratio")
    cbar = plt.colorbar(sc)
    cbar.set_label("P(k=1)/P_CDM")
    plt.tight_layout()
    plt.savefig("refined_warm_corridor_pk1_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8.5, 5.8))
    sc = plt.scatter(masses, fractions, c=scores, s=140)
    plt.xscale("log")
    plt.xlabel("m_ncdm eV")
    plt.ylabel("warm fraction percent")
    plt.title("Refined corridor scan: score lower is better")
    cbar = plt.colorbar(sc)
    cbar.set_label("corridor score")
    plt.tight_layout()
    plt.savefig("refined_warm_corridor_score_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8.5, 6))
    plt.scatter(pk1_vals, s8_vals, c=fractions, s=140)
    plt.axhline(TARGET_S8, linewidth=1, label="target S8 0.800")
    plt.axvline(0.50, linewidth=1, label="P(k=1) ratio 0.50")
    plt.xlabel("P(k=1)/P_CDM")
    plt.ylabel("S8")
    plt.title("Refined corridor scan: S8 versus small-scale suppression")
    plt.legend(fontsize=8)
    cbar = plt.colorbar()
    cbar.set_label("warm fraction percent")
    plt.tight_layout()
    plt.savefig("refined_warm_corridor_S8_vs_pk1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8.5, 6))
    plt.scatter(pk10_vals, s8_vals, c=fractions, s=140)
    plt.axhline(TARGET_S8, linewidth=1, label="target S8 0.800")
    plt.axvline(0.25, linewidth=1, label="P(k=10) ratio 0.25")
    plt.xlabel("P(k=10)/P_CDM")
    plt.ylabel("S8")
    plt.title("Refined corridor scan: S8 versus high-k suppression")
    plt.legend(fontsize=8)
    cbar = plt.colorbar()
    cbar.set_label("warm fraction percent")
    plt.tight_layout()
    plt.savefig("refined_warm_corridor_S8_vs_pk10.png", dpi=160)
    plt.close()


if "cdm_baseline" in pk_tables:
    k = pk_tables["cdm_baseline"]["k"]
    base_pk = pk_tables["cdm_baseline"]["pk"]

    plt.figure(figsize=(11, 6))

    for name, data in candidate_items_sorted[:10]:
        if name not in pk_tables:
            continue
        pk = pk_tables[name]["pk"]
        ratio = np.divide(pk, base_pk, out=np.full_like(base_pk, np.nan), where=base_pk != 0)
        plt.plot(k, ratio, label=name, linewidth=1.1)

    plt.axhline(1.0, linewidth=1)
    plt.axhline(0.5, linewidth=1)
    plt.axhline(0.25, linewidth=1)
    plt.xscale("log")
    plt.xlabel("k raw 1/Mpc")
    plt.ylabel("P(k)/P_CDM")
    plt.title("Top refined corridor candidates: matter power ratio")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig("refined_warm_corridor_top_pk_ratios.png", dpi=160)
    plt.close()


print("\nRefined warm corridor scan complete.")
print("Created:")
print("  refined_warm_corridor_scan_summary.json")
print("  refined_warm_corridor_scan_summary.csv")
print("  refined_warm_corridor_best_cases.json")
print("  refined_warm_corridor_S8_scatter.png")
print("  refined_warm_corridor_pk1_scatter.png")
print("  refined_warm_corridor_score_scatter.png")
print("  refined_warm_corridor_S8_vs_pk1.png")
print("  refined_warm_corridor_S8_vs_pk10.png")
print("  refined_warm_corridor_top_pk_ratios.png")
