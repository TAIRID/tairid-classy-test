#!/usr/bin/env python3
"""
TAIRID high-k / Lyman-alpha-style small-scale suppression proxy v1.

Purpose:
The growth observation chi-square test showed the warm-neutral corridor can
slightly improve a simple f_sigma8(z) comparison. The next pressure point is
whether the same candidates suppress high-k small-scale matter power too much.

This script compares candidate P(k,z) curves against the CDM-like baseline at:
    z = 0, 2, 3, 4, 5
and probes:
    k = 0.5, 1, 2, 3, 5, 10, 20 1/Mpc

Boundary:
This is NOT a real Lyman-alpha likelihood.
It is a diagnostic high-k survival proxy. It does not include hydrodynamics,
IGM thermal history, nuisance parameters, covariance matrices, or survey likelihoods.

Interpretation:
- If a candidate erases high-k power too strongly, it is risky.
- If it lowers S8 while retaining reasonable high-k power, it survives this proxy.
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

Z_VALUES = np.array([0.0, 2.0, 3.0, 4.0, 5.0])
K_VALUES = np.logspace(-3, 1.5, 180)  # 0.001 to about 31.6 1/Mpc
K_PROBES = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

PEAK_WINDOWS = [
    (100, 350),
    (350, 650),
    (650, 1000),
]

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
    "z_max_pk": 5.5,
    "z_pk": ",".join(str(float(z)) for z in Z_VALUES),
}

CASES = [
    {"name": "cdm_baseline", "warm_fraction": 0.0, "m_ncdm_eV": None},

    # Current best corridor from refined scan / growth checks.
    {"name": "best_S8_7p5pct_25eV", "warm_fraction": 0.075, "m_ncdm_eV": 25.0},
    {"name": "best_growth_7p5pct_30eV", "warm_fraction": 0.075, "m_ncdm_eV": 30.0},

    # Other nearby corridor candidates.
    {"name": "candidate_10pct_35eV", "warm_fraction": 0.10, "m_ncdm_eV": 35.0},
    {"name": "candidate_12p5pct_40eV", "warm_fraction": 0.125, "m_ncdm_eV": 40.0},
    {"name": "old_10pct_30eV", "warm_fraction": 0.10, "m_ncdm_eV": 30.0},
    {"name": "safe_5pct_20eV", "warm_fraction": 0.05, "m_ncdm_eV": 20.0},

    # Known aggressive cases as negative controls.
    {"name": "aggressive_20pct_50eV", "warm_fraction": 0.20, "m_ncdm_eV": 50.0},
    {"name": "aggressive_40pct_100eV", "warm_fraction": 0.40, "m_ncdm_eV": 100.0},
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
    dl = ell * (ell + 1) * tt / (2.0 * math.pi)

    pk_by_z = {}

    for z in Z_VALUES:
        pk_values = []
        for k in K_VALUES:
            try:
                pk_values.append(float(cosmo.pk(float(k), float(z))))
            except Exception:
                pk_values.append(float("nan"))
        pk_by_z[float(z)] = np.asarray(pk_values)

    cosmo.struct_cleanup()
    cosmo.empty()

    return sigma8, ell, dl, pk_by_z


def safe_interp(x, xp, fp):
    good = np.isfinite(xp) & np.isfinite(fp)
    if np.sum(good) < 2:
        return float("nan")
    return float(np.interp(x, xp[good], fp[good]))


def first_crossing_k(k_values, ratio_values, threshold):
    """
    Return approximate first k where ratio <= threshold.
    Uses log-k interpolation. Returns None if no crossing.
    """
    k = np.asarray(k_values)
    r = np.asarray(ratio_values)

    good = np.isfinite(k) & np.isfinite(r) & (k > 0)
    k = k[good]
    r = r[good]

    if len(k) < 2:
        return None

    below = r <= threshold
    if not np.any(below):
        return None

    idx = int(np.argmax(below))
    if idx == 0:
        return float(k[0])

    k1, k2 = k[idx - 1], k[idx]
    r1, r2 = r[idx - 1], r[idx]

    if r1 == r2:
        return float(k2)

    x1 = math.log10(k1)
    x2 = math.log10(k2)
    frac = (threshold - r1) / (r2 - r1)
    x = x1 + frac * (x2 - x1)
    return float(10 ** x)


def high_k_proxy_score(ratios_z3):
    """
    Lower is better. This is a diagnostic proxy, not a real likelihood.

    Desired behavior:
    - keep P(k=1) fairly intact
    - avoid early half-mode suppression
    - keep some power at k=5 and k=10
    """
    pk1 = ratios_z3.get("k_1p0", float("nan"))
    pk3 = ratios_z3.get("k_3p0", float("nan"))
    pk5 = ratios_z3.get("k_5p0", float("nan"))
    pk10 = ratios_z3.get("k_10p0", float("nan"))

    if not all(np.isfinite([pk1, pk3, pk5, pk10])):
        return 999.0

    penalty = 0.0
    penalty += max(0.0, 0.80 - pk1) / 0.10
    penalty += max(0.0, 0.60 - pk3) / 0.10
    penalty += max(0.0, 0.50 - pk5) / 0.10
    penalty += max(0.0, 0.30 - pk10) / 0.10

    return float(penalty)


def high_k_label(ratios_z3, k_half_025):
    pk1 = ratios_z3.get("k_1p0", float("nan"))
    pk5 = ratios_z3.get("k_5p0", float("nan"))
    pk10 = ratios_z3.get("k_10p0", float("nan"))

    if not np.isfinite(pk1):
        return "unknown"

    if pk1 >= 0.80 and pk5 >= 0.50 and pk10 >= 0.30:
        return "high-k proxy survival"

    if pk1 >= 0.70 and pk5 >= 0.35 and pk10 >= 0.20:
        return "moderate high-k risk"

    if k_half_025 is not None and k_half_025 < 5.0:
        return "strong high-k suppression"

    return "high-k warning"


results = {}
pk_tables = {}
spectra = {}

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
        sigma8, ell, dl, pk_by_z = run_class_case(params)
        S8 = s8_from_sigma8(sigma8, Omega_m)
        peaks_ell, peaks_height = peak_by_windows(ell, dl)

        cl_file = f"{name}_cl_dl.txt"
        np.savetxt(cl_file, np.column_stack([ell, dl]), header="ell D_l_TT_raw")

        for z, pk_values in pk_by_z.items():
            pk_file = f"{name}_pk_z{str(z).replace('.', 'p')}.txt"
            np.savetxt(pk_file, np.column_stack([K_VALUES, pk_values]), header="k_1_over_Mpc Pk_raw")

        results[name] = {
            "status": "success",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "Omega_m_total": float(Omega_m),
            "sigma8": float(sigma8),
            "S8": float(S8),
            "peak_ell": peaks_ell,
            "peak_Dl_raw": peaks_height,
            "cl_file": cl_file,
        }

        pk_tables[name] = pk_by_z
        spectra[name] = {"ell": ell, "dl": dl}

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


ratio_curve_rows = []
probe_rows = []

if "cdm_baseline" in results and results["cdm_baseline"]["status"] == "success":
    base = results["cdm_baseline"]

    for name, data in results.items():
        if data["status"] != "success":
            continue

        peak_delta = []
        for i in range(3):
            if data["peak_ell"][i] is None or base["peak_ell"][i] is None:
                peak_delta.append(None)
            else:
                peak_delta.append(int(data["peak_ell"][i] - base["peak_ell"][i]))

        data["delta_vs_cdm_baseline"] = {
            "S8_delta": float(data["S8"] - base["S8"]),
            "sigma8_delta": float(data["sigma8"] - base["sigma8"]),
            "peak_ell_delta": peak_delta,
        }

        data["pk_ratio_by_z"] = {}

        for z in Z_VALUES:
            base_pk = pk_tables["cdm_baseline"][float(z)]
            model_pk = pk_tables[name][float(z)]

            ratio = np.divide(
                model_pk,
                base_pk,
                out=np.full_like(model_pk, np.nan),
                where=base_pk != 0,
            )

            z_key = "z_" + str(float(z)).replace(".", "p")
            ratios_at_probes = {}

            for probe in K_PROBES:
                key = "k_" + str(probe).replace(".", "p")
                ratios_at_probes[key] = safe_interp(probe, K_VALUES, ratio)

            k_half_025 = first_crossing_k(K_VALUES, ratio, 0.25)
            k_half_050 = first_crossing_k(K_VALUES, ratio, 0.50)

            data["pk_ratio_by_z"][z_key] = {
                "ratios_at_probes": ratios_at_probes,
                "k_cross_P_ratio_0p25": k_half_025,
                "k_cross_P_ratio_0p50": k_half_050,
            }

            if float(z) == 3.0:
                data["high_k_proxy_score_z3"] = high_k_proxy_score(ratios_at_probes)
                data["high_k_proxy_label_z3"] = high_k_label(ratios_at_probes, k_half_025)

            for k_val, ratio_val in zip(K_VALUES, ratio):
                ratio_curve_rows.append(
                    {
                        "case": name,
                        "z": float(z),
                        "k": float(k_val),
                        "P_ratio_to_CDM": float(ratio_val),
                    }
                )

            row = {
                "case": name,
                "z": float(z),
                "k_cross_P_ratio_0p25": k_half_025,
                "k_cross_P_ratio_0p50": k_half_050,
            }
            row.update(ratios_at_probes)
            probe_rows.append(row)


Path("high_k_suppression_proxy_v1_summary.json").write_text(json.dumps(results, indent=2))


with open("high_k_suppression_proxy_v1_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "case",
            "status",
            "warm_fraction",
            "m_ncdm_eV",
            "Omega_m_total",
            "sigma8",
            "S8",
            "ell_peak1",
            "ell_peak2",
            "ell_peak3",
            "delta_ell1",
            "delta_ell2",
            "delta_ell3",
            "P_ratio_z3_k0p5",
            "P_ratio_z3_k1p0",
            "P_ratio_z3_k2p0",
            "P_ratio_z3_k3p0",
            "P_ratio_z3_k5p0",
            "P_ratio_z3_k10p0",
            "P_ratio_z3_k20p0",
            "k_cross_z3_P_ratio_0p25",
            "k_cross_z3_P_ratio_0p50",
            "high_k_proxy_score_z3",
            "high_k_proxy_label_z3",
            "error",
        ]
    )

    for name, data in results.items():
        if data["status"] == "success":
            d = data.get("delta_vs_cdm_baseline", {})
            z3 = data.get("pk_ratio_by_z", {}).get("z_3p0", {})
            ratios = z3.get("ratios_at_probes", {})

            writer.writerow(
                [
                    name,
                    data["status"],
                    data["warm_fraction"],
                    data["m_ncdm_eV"],
                    data["Omega_m_total"],
                    data["sigma8"],
                    data["S8"],
                    *data["peak_ell"],
                    *(d.get("peak_ell_delta", [None, None, None])),
                    ratios.get("k_0p5"),
                    ratios.get("k_1p0"),
                    ratios.get("k_2p0"),
                    ratios.get("k_3p0"),
                    ratios.get("k_5p0"),
                    ratios.get("k_10p0"),
                    ratios.get("k_20p0"),
                    z3.get("k_cross_P_ratio_0p25"),
                    z3.get("k_cross_P_ratio_0p50"),
                    data.get("high_k_proxy_score_z3"),
                    data.get("high_k_proxy_label_z3"),
                    "",
                ]
            )
        else:
            writer.writerow(
                [
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
                    "",
                    data.get("error", ""),
                ]
            )


with open("high_k_probe_ratios_v1.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "case",
            "z",
            "k_cross_P_ratio_0p25",
            "k_cross_P_ratio_0p50",
            "k_0p5",
            "k_1p0",
            "k_2p0",
            "k_3p0",
            "k_5p0",
            "k_10p0",
            "k_20p0",
        ]
    )

    for row in probe_rows:
        writer.writerow(
            [
                row["case"],
                row["z"],
                row["k_cross_P_ratio_0p25"],
                row["k_cross_P_ratio_0p50"],
                row.get("k_0p5"),
                row.get("k_1p0"),
                row.get("k_2p0"),
                row.get("k_3p0"),
                row.get("k_5p0"),
                row.get("k_10p0"),
                row.get("k_20p0"),
            ]
        )


with open("high_k_ratio_curves_v1.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case", "z", "k", "P_ratio_to_CDM"])

    for row in ratio_curve_rows:
        writer.writerow([row["case"], row["z"], row["k"], row["P_ratio_to_CDM"]])


# Plots.
success_names = [
    name for name, data in results.items()
    if data["status"] == "success" and name != "cdm_baseline"
]

if success_names:
    # z=3 ratio curves
    plt.figure(figsize=(11, 6))
    for name in success_names:
        base_pk = pk_tables["cdm_baseline"][3.0]
        model_pk = pk_tables[name][3.0]
        ratio = np.divide(model_pk, base_pk, out=np.full_like(model_pk, np.nan), where=base_pk != 0)
        plt.plot(K_VALUES, ratio, label=name, linewidth=1.15)

    plt.axhline(1.0, linewidth=1)
    plt.axhline(0.5, linewidth=1)
    plt.axhline(0.25, linewidth=1)
    plt.xscale("log")
    plt.xlabel("k raw 1/Mpc")
    plt.ylabel("P(k,z=3) / P_CDM(k,z=3)")
    plt.title("High-k suppression proxy: z=3 matter power ratios")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig("high_k_suppression_z3_ratios_v1.png", dpi=160)
    plt.close()

    # bar score
    labels = success_names
    scores = [results[name].get("high_k_proxy_score_z3", np.nan) for name in labels]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    plt.bar(x, scores)
    plt.xticks(x, labels, rotation=55, ha="right", fontsize=8)
    plt.ylabel("high-k proxy score at z=3, lower is better")
    plt.title("High-k proxy score")
    plt.tight_layout()
    plt.savefig("high_k_proxy_score_v1.png", dpi=160)
    plt.close()

    # S8 vs k=10 ratio
    pk10 = []
    s8 = []
    color_frac = []
    for name in success_names:
        z3 = results[name]["pk_ratio_by_z"]["z_3p0"]["ratios_at_probes"]
        pk10.append(z3.get("k_10p0"))
        s8.append(results[name]["S8"])
        color_frac.append(results[name]["warm_fraction"] * 100.0)

    plt.figure(figsize=(8.5, 6))
    sc = plt.scatter(pk10, s8, c=color_frac, s=130)
    for name, xval, yval in zip(success_names, pk10, s8):
        plt.annotate(name.replace("candidate_", "").replace("best_", "").replace("safe_", "").replace("old_", "").replace("aggressive_", ""),
                     (xval, yval), fontsize=7, xytext=(4, 3), textcoords="offset points")
    plt.axhline(0.800, linewidth=1, label="S8 target 0.800")
    plt.axvline(0.30, linewidth=1, label="P(k=10) ratio 0.30")
    plt.xlabel("P(k=10,z=3)/P_CDM")
    plt.ylabel("S8")
    plt.title("S8 versus high-k survival proxy")
    plt.legend(fontsize=8)
    cbar = plt.colorbar(sc)
    cbar.set_label("warm fraction percent")
    plt.tight_layout()
    plt.savefig("high_k_S8_vs_k10_ratio_v1.png", dpi=160)
    plt.close()

    # z comparison for best cases
    top_cases = [
        "best_S8_7p5pct_25eV",
        "best_growth_7p5pct_30eV",
        "candidate_10pct_35eV",
        "old_10pct_30eV",
    ]

    plt.figure(figsize=(11, 6))
    for name in top_cases:
        if name not in results or results[name]["status"] != "success":
            continue
        for z in [2.0, 3.0, 5.0]:
            base_pk = pk_tables["cdm_baseline"][z]
            model_pk = pk_tables[name][z]
            ratio = np.divide(model_pk, base_pk, out=np.full_like(model_pk, np.nan), where=base_pk != 0)
            plt.plot(K_VALUES, ratio, label=f"{name} z={z}", linewidth=1.0)

    plt.axhline(1.0, linewidth=1)
    plt.axhline(0.5, linewidth=1)
    plt.axhline(0.25, linewidth=1)
    plt.xscale("log")
    plt.xlabel("k raw 1/Mpc")
    plt.ylabel("P(k,z)/P_CDM(k,z)")
    plt.title("Top corridor cases across high redshift")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig("high_k_top_cases_redshift_comparison_v1.png", dpi=160)
    plt.close()


print("")
print("High-k suppression proxy v1 complete.")
print("Created:")
print("  high_k_suppression_proxy_v1_summary.json")
print("  high_k_suppression_proxy_v1_summary.csv")
print("  high_k_probe_ratios_v1.csv")
print("  high_k_ratio_curves_v1.csv")
print("  high_k_suppression_z3_ratios_v1.png")
print("  high_k_proxy_score_v1.png")
print("  high_k_S8_vs_k10_ratio_v1.png")
print("  high_k_top_cases_redshift_comparison_v1.png")
