#!/usr/bin/env python3
"""
TAIRID growth history scan v3.

Fix from v2:
NumPy on GitHub no longer has np.trapz, so this version uses np.trapezoid.

Purpose:
Test whether the refined warm-neutral corridor has a reasonable growth history.

It computes:
- sigma8(z), using a top-hat integral over the CLASS matter power spectrum
- f(z) = d ln sigma8 / d ln a
- f_sigma8(z) = f(z) * sigma8(z)
- ratios against the CDM-like baseline

Boundary:
This is still a CLASS proxy test, not final TAIRID.
The non-cold neutral fraction is represented by CLASS ncdm.
"""

import csv
import json
import math
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from classy import Class


Z_VALUES = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
K_VALUES = np.logspace(-4, 1.2, 220)

OMEGA_NEUTRAL_PHYSICAL = 0.1200000000
H = 0.66893180

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
    "z_max_pk": 3.0,
    "z_pk": "0,0.1,0.2,0.3,0.5,0.7,1,1.5,2",
}

CASES = [
    {"name": "cdm_baseline", "warm_fraction": 0.0, "m_ncdm_eV": None},
    {"name": "best_7p5pct_25eV", "warm_fraction": 0.075, "m_ncdm_eV": 25.0},
    {"name": "old_10pct_30eV", "warm_fraction": 0.10, "m_ncdm_eV": 30.0},
    {"name": "candidate_10pct_35eV", "warm_fraction": 0.10, "m_ncdm_eV": 35.0},
    {"name": "candidate_12p5pct_40eV", "warm_fraction": 0.125, "m_ncdm_eV": 40.0},
    {"name": "safe_5pct_20eV", "warm_fraction": 0.05, "m_ncdm_eV": 20.0},
    {"name": "safe_7p5pct_30eV", "warm_fraction": 0.075, "m_ncdm_eV": 30.0},
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


def compute_growth_quantities(sigma8_values):
    z = Z_VALUES
    sigma = np.asarray(sigma8_values)

    ln_sigma = np.log(sigma)
    dlnsigma_dz = np.gradient(ln_sigma, z)

    f_growth = -(1.0 + z) * dlnsigma_dz
    f_sigma8 = f_growth * sigma

    return f_growth, f_sigma8


def s8_from_sigma8(sigma8, Omega_m):
    return float(sigma8 * math.sqrt(Omega_m / 0.3))


def run_class_case(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    class_sigma8_z0 = float(cosmo.sigma8())

    cl = cosmo.lensed_cl(2500)
    ell = np.asarray(cl["ell"])
    tt = np.asarray(cl["tt"])
    dl = ell * (ell + 1) * tt / (2.0 * math.pi)

    h = float(params["h"])
    R8_mpc = 8.0 / h

    sigma8_values = []

    for z in Z_VALUES:
        pk_values = []

        for k in K_VALUES:
            try:
                pk_values.append(float(cosmo.pk(float(k), float(z))))
            except Exception:
                pk_values.append(float("nan"))

        pk_values = np.asarray(pk_values)
        sigma8_values.append(sigma_R_from_pk(K_VALUES, pk_values, R8_mpc))

    sigma8_values = np.asarray(sigma8_values)

    cosmo.struct_cleanup()
    cosmo.empty()

    return class_sigma8_z0, ell, dl, sigma8_values


results = {}
growth_tables = {}

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
        class_sigma8_z0, ell, dl, sigma8_values = run_class_case(params)

        f_growth, f_sigma8 = compute_growth_quantities(sigma8_values)
        peaks_ell, peaks_height = peak_by_windows(ell, dl)

        S8_integral = s8_from_sigma8(float(sigma8_values[0]), Omega_m)
        S8_class = s8_from_sigma8(class_sigma8_z0, Omega_m)

        cl_file = f"{name}_cl_dl.txt"
        growth_file = f"{name}_growth_history.txt"

        np.savetxt(
            cl_file,
            np.column_stack([ell, dl]),
            header="ell D_l_TT_raw",
        )

        np.savetxt(
            growth_file,
            np.column_stack([Z_VALUES, sigma8_values, f_growth, f_sigma8]),
            header="z sigma8_integral f_growth f_sigma8",
        )

        results[name] = {
            "status": "success",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "omega_cdm": float(params.get("omega_cdm", float("nan"))),
            "Omega_ncdm": None if "Omega_ncdm" not in params else float(params["Omega_ncdm"]),
            "omega_m_total": float(omega_m),
            "Omega_m_total": float(Omega_m),
            "class_sigma8_z0": class_sigma8_z0,
            "integral_sigma8_z0": float(sigma8_values[0]),
            "S8_from_class_sigma8": S8_class,
            "S8_from_integral_sigma8": S8_integral,
            "peak_ell": peaks_ell,
            "peak_Dl_raw": peaks_height,
            "cl_file": cl_file,
            "growth_file": growth_file,
        }

        growth_tables[name] = {
            "z": Z_VALUES,
            "sigma8": sigma8_values,
            "f_growth": f_growth,
            "f_sigma8": f_sigma8,
        }

        print("  success")
        print("  class sigma8:", class_sigma8_z0)
        print("  integral sigma8:", sigma8_values[0])
        print("  S8:", S8_integral)
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
    base_growth = growth_tables["cdm_baseline"]

    for name, data in results.items():
        if data["status"] != "success":
            continue

        gt = growth_tables[name]

        peak_delta = []
        for i in range(3):
            if data["peak_ell"][i] is None or base["peak_ell"][i] is None:
                peak_delta.append(None)
            else:
                peak_delta.append(int(data["peak_ell"][i] - base["peak_ell"][i]))

        fs8_ratio = np.divide(
            gt["f_sigma8"],
            base_growth["f_sigma8"],
            out=np.full_like(gt["f_sigma8"], np.nan),
            where=base_growth["f_sigma8"] != 0,
        )

        sigma_ratio = np.divide(
            gt["sigma8"],
            base_growth["sigma8"],
            out=np.full_like(gt["sigma8"], np.nan),
            where=base_growth["sigma8"] != 0,
        )

        data["delta_vs_cdm_baseline"] = {
            "S8_integral_delta": float(data["S8_from_integral_sigma8"] - base["S8_from_integral_sigma8"]),
            "S8_integral_fractional_delta": float(
                (data["S8_from_integral_sigma8"] - base["S8_from_integral_sigma8"])
                / base["S8_from_integral_sigma8"]
            ),
            "class_sigma8_delta": float(data["class_sigma8_z0"] - base["class_sigma8_z0"]),
            "peak_ell_delta": peak_delta,
        }

        data["growth_ratio_to_cdm"] = {
            "z_values": [float(x) for x in Z_VALUES],
            "sigma8_ratio": [float(x) for x in sigma_ratio],
            "f_sigma8_ratio": [float(x) for x in fs8_ratio],
        }


Path("growth_history_scan_v3_summary.json").write_text(json.dumps(results, indent=2))


with open("growth_history_scan_v3_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "case",
        "status",
        "warm_fraction",
        "m_ncdm_eV",
        "Omega_m_total",
        "class_sigma8_z0",
        "integral_sigma8_z0",
        "S8_from_class_sigma8",
        "S8_from_integral_sigma8",
        "ell_peak1",
        "ell_peak2",
        "ell_peak3",
        "delta_ell1",
        "delta_ell2",
        "delta_ell3",
        "f_sigma8_z0",
        "f_sigma8_z0p5",
        "f_sigma8_z1",
        "f_sigma8_z2",
        "f_sigma8_ratio_z0",
        "f_sigma8_ratio_z0p5",
        "f_sigma8_ratio_z1",
        "f_sigma8_ratio_z2",
        "error",
    ])

    z_list = list(Z_VALUES)

    def value_at(arr, ztarget):
        idx = z_list.index(ztarget)
        return float(arr[idx])

    for name, data in results.items():
        if data["status"] == "success":
            gt = growth_tables[name]
            d = data.get("delta_vs_cdm_baseline", {})
            ratios = data.get("growth_ratio_to_cdm", {})
            fs8_ratio = ratios.get("f_sigma8_ratio", [None] * len(Z_VALUES))

            writer.writerow([
                name,
                data["status"],
                data["warm_fraction"],
                data["m_ncdm_eV"],
                data["Omega_m_total"],
                data["class_sigma8_z0"],
                data["integral_sigma8_z0"],
                data["S8_from_class_sigma8"],
                data["S8_from_integral_sigma8"],
                *data["peak_ell"],
                *(d.get("peak_ell_delta", [None, None, None])),
                value_at(gt["f_sigma8"], 0.0),
                value_at(gt["f_sigma8"], 0.5),
                value_at(gt["f_sigma8"], 1.0),
                value_at(gt["f_sigma8"], 2.0),
                fs8_ratio[z_list.index(0.0)] if fs8_ratio[0] is not None else None,
                fs8_ratio[z_list.index(0.5)] if fs8_ratio[0] is not None else None,
                fs8_ratio[z_list.index(1.0)] if fs8_ratio[0] is not None else None,
                fs8_ratio[z_list.index(2.0)] if fs8_ratio[0] is not None else None,
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


if growth_tables:
    plt.figure(figsize=(9, 5.5))
    for name, gt in growth_tables.items():
        plt.plot(gt["z"], gt["sigma8"], marker="o", label=name)
    plt.xlabel("z")
    plt.ylabel("sigma8(z), integral proxy")
    plt.title("Growth history scan v3: sigma8(z)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("growth_history_v3_sigma8_z.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for name, gt in growth_tables.items():
        plt.plot(gt["z"], gt["f_sigma8"], marker="o", label=name)
    plt.xlabel("z")
    plt.ylabel("f sigma8(z), derivative proxy")
    plt.title("Growth history scan v3: f sigma8(z)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("growth_history_v3_fsigma8_z.png", dpi=160)
    plt.close()

if "cdm_baseline" in growth_tables:
    base_growth = growth_tables["cdm_baseline"]

    plt.figure(figsize=(9, 5.5))
    for name, gt in growth_tables.items():
        if name == "cdm_baseline":
            continue

        ratio = np.divide(
            gt["f_sigma8"],
            base_growth["f_sigma8"],
            out=np.full_like(gt["f_sigma8"], np.nan),
            where=base_growth["f_sigma8"] != 0,
        )

        plt.plot(gt["z"], ratio, marker="o", label=name)

    plt.axhline(1.0, linewidth=1)
    plt.xlabel("z")
    plt.ylabel("f sigma8 / CDM baseline")
    plt.title("Growth history scan v3: f sigma8 ratio")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("growth_history_v3_fsigma8_ratio.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for name, gt in growth_tables.items():
        if name == "cdm_baseline":
            continue

        ratio = np.divide(
            gt["sigma8"],
            base_growth["sigma8"],
            out=np.full_like(gt["sigma8"], np.nan),
            where=base_growth["sigma8"] != 0,
        )

        plt.plot(gt["z"], ratio, marker="o", label=name)

    plt.axhline(1.0, linewidth=1)
    plt.xlabel("z")
    plt.ylabel("sigma8(z) / CDM baseline")
    plt.title("Growth history scan v3: sigma8 ratio")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("growth_history_v3_sigma8_ratio.png", dpi=160)
    plt.close()


print("")
print("Growth history scan v3 complete.")
print("Created:")
print("  growth_history_scan_v3_summary.json")
print("  growth_history_scan_v3_summary.csv")
print("  growth_history_v3_sigma8_z.png")
print("  growth_history_v3_fsigma8_z.png")
print("  growth_history_v3_fsigma8_ratio.png")
print("  growth_history_v3_sigma8_ratio.png")
