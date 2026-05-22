#!/usr/bin/env python3
"""
TAIRID small-scale structure sanity scan.

Purpose:
The broad warm-neutral S8 scan found candidate lanes that can lower S8 while
keeping CMB peak locations locked. The pressure point is whether those lanes
suppress small-scale linear matter power too much.

This script tests targeted candidate cases and reports:

- sigma8 and S8
- first three TT peak locations
- P(k)/P_CDM at k = 0.1, 0.3, 0.5, 1, 3, 5, 10 1/Mpc
- approximate half-mode scale where P(k)/P_CDM falls below 0.25
  which corresponds to transfer amplitude sqrt(P/P_CDM) below 0.5

Boundary:
This is still a CLASS proxy test, not final TAIRID.
The non-cold neutral part is represented by CLASS ncdm as a stand-in for
delayed consolidation / non-cold neutral behavior.
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

K_PROBES = [
    0.1,
    0.3,
    0.5,
    1.0,
    3.0,
    5.0,
    10.0,
]

CASES = [
    {"name": "cdm_baseline", "warm_fraction": 0.0, "m_ncdm_eV": None},

    # Best S8 corridor cases found in prior scan
    {"name": "candidate_10pct_30eV", "warm_fraction": 0.10, "m_ncdm_eV": 30.0},
    {"name": "candidate_20pct_50eV", "warm_fraction": 0.20, "m_ncdm_eV": 50.0},
    {"name": "candidate_40pct_100eV", "warm_fraction": 0.40, "m_ncdm_eV": 100.0},
    {"name": "candidate_50pct_100eV", "warm_fraction": 0.50, "m_ncdm_eV": 100.0},

    # Less aggressive comparison lanes
    {"name": "soft_10pct_50eV", "warm_fraction": 0.10, "m_ncdm_eV": 50.0},
    {"name": "soft_10pct_100eV", "warm_fraction": 0.10, "m_ncdm_eV": 100.0},
    {"name": "soft_20pct_100eV", "warm_fraction": 0.20, "m_ncdm_eV": 100.0},
    {"name": "soft_30pct_100eV", "warm_fraction": 0.30, "m_ncdm_eV": 100.0},

    # Extreme pressure checks
    {"name": "extreme_30pct_30eV", "warm_fraction": 0.30, "m_ncdm_eV": 30.0},
    {"name": "extreme_50pct_30eV", "warm_fraction": 0.50, "m_ncdm_eV": 30.0},
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

    k_values = np.logspace(-3, 1, 140)
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


def half_mode_k(k, pk_ratio):
    """
    Return approximate k where P/P_CDM first drops below 0.25.
    That is transfer amplitude sqrt(P/P_CDM)=0.5.
    If never crosses, return None.
    """
    good = np.isfinite(k) & np.isfinite(pk_ratio)
    kk = k[good]
    rr = pk_ratio[good]

    if len(kk) < 2:
        return None

    below = rr <= 0.25
    if not np.any(below):
        return None

    idx = int(np.argmax(below))
    if idx == 0:
        return float(kk[0])

    k1, k2 = kk[idx - 1], kk[idx]
    r1, r2 = rr[idx - 1], rr[idx]

    if r1 == r2:
        return float(k2)

    # Linear interpolation in log k versus ratio.
    x1 = math.log10(k1)
    x2 = math.log10(k2)
    frac = (0.25 - r1) / (r2 - r1)
    x = x1 + frac * (x2 - x1)
    return float(10 ** x)


def sanity_label(pk_ratio_k1, pk_ratio_k10, S8):
    """
    Simple diagnostic labels, not official cosmological constraints.
    """
    if not np.isfinite(pk_ratio_k1):
        return "unknown"

    if pk_ratio_k1 < 0.25:
        return "very strong small-scale suppression"

    if pk_ratio_k1 < 0.50:
        return "strong small-scale suppression"

    if pk_ratio_k1 < 0.75:
        if abs(S8 - TARGET_S8) < 0.015:
            return "candidate corridor: lowers S8 with moderate/strong suppression"
        return "moderate suppression"

    if pk_ratio_k1 < 0.90:
        return "mild suppression"

    return "CDM-like small-scale power"


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

        data["delta_vs_cdm_baseline"] = {
            "sigma8_delta": float(data["sigma8"] - base["sigma8"]),
            "sigma8_fractional_delta": float((data["sigma8"] - base["sigma8"]) / base["sigma8"]),
            "S8_delta": float(data["S8"] - base["S8"]),
            "S8_fractional_delta": float((data["S8"] - base["S8"]) / base["S8"]),
            "peak_ell_delta": [
                None if data["peak_ell"][i] is None or base["peak_ell"][i] is None
                else int(data["peak_ell"][i] - base["peak_ell"][i])
                for i in range(3)
            ],
        }

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

            k_half = half_mode_k(k, ratio)
            pk_ratio_k1 = pk_ratios["k_1p0"]
            pk_ratio_k10 = pk_ratios["k_10p0"]

            data["pk_ratio_at_k"] = pk_ratios
            data["half_mode_k_for_P_ratio_0p25"] = k_half
            data["small_scale_sanity_label"] = sanity_label(pk_ratio_k1, pk_ratio_k10, data["S8"])


Path("small_scale_sanity_summary.json").write_text(json.dumps(results, indent=2))


with open("small_scale_sanity_summary.csv", "w", newline="") as f:
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
        "delta_ell1",
        "delta_ell2",
        "delta_ell3",
        "P_ratio_k0p1",
        "P_ratio_k0p3",
        "P_ratio_k0p5",
        "P_ratio_k1p0",
        "P_ratio_k3p0",
        "P_ratio_k5p0",
        "P_ratio_k10p0",
        "half_mode_k_P_ratio_0p25",
        "small_scale_sanity_label",
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
                data["Omega_m_total"],
                data["sigma8"],
                data["S8"],
                data["S8_distance_to_target_0p800"],
                *data["peak_ell"],
                *(d.get("peak_ell_delta", [None, None, None])),
                pkd.get("k_0p1"),
                pkd.get("k_0p3"),
                pkd.get("k_0p5"),
                pkd.get("k_1p0"),
                pkd.get("k_3p0"),
                pkd.get("k_5p0"),
                pkd.get("k_10p0"),
                data.get("half_mode_k_for_P_ratio_0p25"),
                data.get("small_scale_sanity_label"),
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
                data.get("error", ""),
            ])


success_names = [name for name, data in results.items() if data["status"] == "success"]

# Plot P(k) ratio for all non-baseline cases.
if "cdm_baseline" in pk_tables:
    k = pk_tables["cdm_baseline"]["k"]
    base_pk = pk_tables["cdm_baseline"]["pk"]

    plt.figure(figsize=(11, 6))
    for name, table in pk_tables.items():
        if name == "cdm_baseline":
            continue
        ratio = np.divide(table["pk"], base_pk, out=np.full_like(base_pk, np.nan), where=base_pk != 0)
        plt.plot(k, ratio, label=name, linewidth=1.0)
    plt.axhline(1.0, linewidth=1)
    plt.axhline(0.5, linewidth=1)
    plt.axhline(0.25, linewidth=1)
    plt.xscale("log")
    plt.xlabel("k raw 1/Mpc")
    plt.ylabel("P(k) ratio to CDM baseline")
    plt.title("Small-scale sanity scan: matter power ratio at z=0")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig("small_scale_sanity_pk_ratio.png", dpi=160)
    plt.close()


# S8 versus P(k=1) plot.
plot_points = []
for name, data in results.items():
    if data["status"] != "success" or name == "cdm_baseline":
        continue
    pk1 = data.get("pk_ratio_at_k", {}).get("k_1p0", np.nan)
    plot_points.append((name, data["S8"], pk1, data["warm_fraction"], data["m_ncdm_eV"]))

if plot_points:
    labels = [p[0] for p in plot_points]
    s8 = np.array([p[1] for p in plot_points])
    pk1 = np.array([p[2] for p in plot_points])
    warm = np.array([p[3] for p in plot_points]) * 100.0
    mass = np.array([p[4] for p in plot_points])

    plt.figure(figsize=(8.5, 6))
    sc = plt.scatter(pk1, s8, c=warm, s=120)
    plt.axhline(TARGET_S8, linewidth=1, label="target S8 0.800")
    plt.axvline(0.5, linewidth=1, label="P(k=1) ratio 0.5")
    plt.xlabel("P(k=1)/P_CDM")
    plt.ylabel("S8")
    plt.title("Small-scale sanity: S8 versus P(k=1) suppression")
    plt.legend(fontsize=8)
    cbar = plt.colorbar(sc)
    cbar.set_label("warm fraction percent")
    plt.tight_layout()
    plt.savefig("small_scale_sanity_S8_vs_pk1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8.5, 6))
    sc = plt.scatter(mass, warm, c=s8, s=120)
    plt.xscale("log")
    plt.xlabel("m_ncdm eV")
    plt.ylabel("warm fraction percent")
    plt.title("Small-scale sanity: S8 across candidate grid")
    cbar = plt.colorbar(sc)
    cbar.set_label("S8")
    plt.tight_layout()
    plt.savefig("small_scale_sanity_S8_grid.png", dpi=160)
    plt.close()


# Peak lock plot.
if success_names:
    labels = success_names
    x = np.arange(len(labels))

    peak1 = [results[name]["peak_ell"][0] for name in labels]
    peak2 = [results[name]["peak_ell"][1] for name in labels]
    peak3 = [results[name]["peak_ell"][2] for name in labels]

    plt.figure(figsize=(12, 5))
    plt.plot(x, peak1, marker="o", label="peak 1")
    plt.plot(x, peak2, marker="o", label="peak 2")
    plt.plot(x, peak3, marker="o", label="peak 3")
    plt.xticks(x, labels, rotation=65, ha="right", fontsize=7)
    plt.ylabel("peak ell")
    plt.title("Small-scale sanity scan: TT peak locations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("small_scale_sanity_peak_locations.png", dpi=160)
    plt.close()


print("\nSmall-scale sanity scan complete.")
print("Created:")
print("  small_scale_sanity_summary.json")
print("  small_scale_sanity_summary.csv")
print("  small_scale_sanity_pk_ratio.png")
print("  small_scale_sanity_S8_vs_pk1.png")
print("  small_scale_sanity_S8_grid.png")
print("  small_scale_sanity_peak_locations.png")
