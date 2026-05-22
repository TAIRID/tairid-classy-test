#!/usr/bin/env python3
"""
TAIRID observed growth survivor chi-square test.

Purpose:
Test the high-k survivor candidates against a simple observed f_sigma8(z) table.

Boundary:
This is a proxy test only.
It is not a final survey likelihood.
It ignores covariance.
It does not prove TAIRID cosmology.
"""

import csv
import json
import math
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from classy import Class


OBS_FSIGMA8 = [
    {"label": "6dFGS", "z": 0.067, "fs8": 0.423, "sigma": 0.055},
    {"label": "SDSS_MGS", "z": 0.150, "fs8": 0.530, "sigma": 0.160},
    {"label": "GAMA", "z": 0.180, "fs8": 0.360, "sigma": 0.090},
    {"label": "BOSS_LOWZ", "z": 0.320, "fs8": 0.384, "sigma": 0.095},
    {"label": "BOSS_DR12_z038", "z": 0.380, "fs8": 0.497, "sigma": 0.045},
    {"label": "WiggleZ_z044", "z": 0.440, "fs8": 0.413, "sigma": 0.080},
    {"label": "BOSS_DR12_z051", "z": 0.510, "fs8": 0.458, "sigma": 0.038},
    {"label": "WiggleZ_z060", "z": 0.600, "fs8": 0.390, "sigma": 0.063},
    {"label": "BOSS_DR12_z061", "z": 0.610, "fs8": 0.436, "sigma": 0.034},
    {"label": "eBOSS_LRG", "z": 0.698, "fs8": 0.473, "sigma": 0.044},
    {"label": "WiggleZ_z073", "z": 0.730, "fs8": 0.437, "sigma": 0.072},
    {"label": "VIPERS", "z": 0.800, "fs8": 0.470, "sigma": 0.080},
    {"label": "eBOSS_ELG", "z": 0.850, "fs8": 0.315, "sigma": 0.095},
    {"label": "eBOSS_QSO", "z": 1.480, "fs8": 0.462, "sigma": 0.045},
]

OBS_Z = np.array([x["z"] for x in OBS_FSIGMA8], dtype=float)

Z_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 2.0, 61),
            OBS_Z,
            np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00]),
        ]
    )
)
Z_GRID.sort()

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
    "z_pk": ",".join(str(float(z)) for z in Z_GRID),
}

CASES = [
    {"name": "cdm_baseline", "warm_fraction": 0.0, "m_ncdm_eV": None},
    {"name": "highk_survivor_5pct_20eV", "warm_fraction": 0.05, "m_ncdm_eV": 20.0},
    {"name": "highk_survivor_5pct_25eV", "warm_fraction": 0.05, "m_ncdm_eV": 25.0},
    {"name": "s8_warning_7p5pct_25eV", "warm_fraction": 0.075, "m_ncdm_eV": 25.0},
    {"name": "s8_warning_7p5pct_30eV", "warm_fraction": 0.075, "m_ncdm_eV": 30.0},
    {"name": "destructive_old_10pct_30eV", "warm_fraction": 0.10, "m_ncdm_eV": 30.0},
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

    for lo, hi in PEAK_WINDOWS:
        mask = (ell >= lo) & (ell <= hi)

        if not np.any(mask):
            peak_ells.append(None)
            continue

        ell_window = ell[mask]
        dl_window = dl[mask]
        idx = int(np.argmax(dl_window))

        peak_ells.append(int(ell_window[idx]))

    return peak_ells


def compute_growth_quantities(sigma8_values):
    z = Z_GRID
    sigma = np.asarray(sigma8_values)

    sigma = np.where(sigma > 1.0e-30, sigma, np.nan)

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

    for z in Z_GRID:
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


def interp_model_at_obs(y_grid):
    return np.interp(OBS_Z, Z_GRID, y_grid)


def chi_square_to_observations(model_fs8_obs):
    rows = []
    chi2 = 0.0

    for obs, pred in zip(OBS_FSIGMA8, model_fs8_obs):
        residual = pred - obs["fs8"]
        pull = residual / obs["sigma"]
        contribution = pull * pull
        chi2 += contribution

        rows.append(
            {
                "label": obs["label"],
                "z": obs["z"],
                "observed_fs8": obs["fs8"],
                "sigma": obs["sigma"],
                "model_fs8": float(pred),
                "residual_model_minus_obs": float(residual),
                "pull_sigma": float(pull),
                "chi2_contribution": float(contribution),
            }
        )

    return float(chi2), rows


results = {}
growth_tables = {}
residual_rows_all = []

for case in CASES:
    name = case["name"]
    params = make_params(case)
    omega_m, Omega_m = omega_total_matter_proxy(params)

    print("")
    print("Running case:", name)

    try:
        class_sigma8_z0, ell, dl, sigma8_values = run_class_case(params)

        f_growth, f_sigma8 = compute_growth_quantities(sigma8_values)
        peaks_ell = peak_by_windows(ell, dl)
        S8 = s8_from_sigma8(class_sigma8_z0, Omega_m)

        model_fs8_obs = interp_model_at_obs(f_sigma8)
        chi2, residual_rows = chi_square_to_observations(model_fs8_obs)

        dof = len(OBS_FSIGMA8)
        reduced_chi2 = chi2 / dof

        growth_file = f"{name}_growth_curve.txt"
        np.savetxt(
            growth_file,
            np.column_stack([Z_GRID, sigma8_values, f_growth, f_sigma8]),
            header="z sigma8_integral f_growth f_sigma8",
        )

        for row in residual_rows:
            row2 = dict(row)
            row2["case"] = name
            residual_rows_all.append(row2)

        results[name] = {
            "status": "success",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "Omega_m_total": float(Omega_m),
            "class_sigma8_z0": class_sigma8_z0,
            "S8_from_class_sigma8": S8,
            "peak_ell": peaks_ell,
            "chi2_growth_observed": chi2,
            "dof_growth_observed": dof,
            "reduced_chi2_growth_observed": reduced_chi2,
            "growth_file": growth_file,
        }

        growth_tables[name] = {
            "z": Z_GRID,
            "f_sigma8": f_sigma8,
        }

        print("  success")
        print("  S8:", S8)
        print("  chi2:", chi2)
        print("  reduced chi2:", reduced_chi2)
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
            "delta_chi2": float(data["chi2_growth_observed"] - base["chi2_growth_observed"]),
            "delta_reduced_chi2": float(
                data["reduced_chi2_growth_observed"] - base["reduced_chi2_growth_observed"]
            ),
            "delta_S8": float(data["S8_from_class_sigma8"] - base["S8_from_class_sigma8"]),
        }


Path("growth_survivor_chisq_summary.json").write_text(json.dumps(results, indent=2))


with open("growth_survivor_chisq_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(
        [
            "case",
            "status",
            "warm_fraction",
            "m_ncdm_eV",
            "Omega_m_total",
            "class_sigma8_z0",
            "S8",
            "ell_peak1",
            "ell_peak2",
            "ell_peak3",
            "chi2",
            "dof",
            "reduced_chi2",
            "delta_chi2_vs_cdm",
            "delta_reduced_chi2_vs_cdm",
            "delta_S8_vs_cdm",
            "error",
        ]
    )

    for name, data in results.items():
        if data["status"] == "success":
            delta = data.get("delta_vs_cdm_baseline", {})
            peaks = data["peak_ell"]

            writer.writerow(
                [
                    name,
                    data["status"],
                    data["warm_fraction"],
                    data["m_ncdm_eV"],
                    data["Omega_m_total"],
                    data["class_sigma8_z0"],
                    data["S8_from_class_sigma8"],
                    peaks[0],
                    peaks[1],
                    peaks[2],
                    data["chi2_growth_observed"],
                    data["dof_growth_observed"],
                    data["reduced_chi2_growth_observed"],
                    delta.get("delta_chi2"),
                    delta.get("delta_reduced_chi2"),
                    delta.get("delta_S8"),
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
                    data.get("error", ""),
                ]
            )


with open("growth_survivor_observation_residuals.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(
        [
            "case",
            "label",
            "z",
            "observed_fs8",
            "sigma",
            "model_fs8",
            "residual_model_minus_obs",
            "pull_sigma",
            "chi2_contribution",
        ]
    )

    for row in residual_rows_all:
        writer.writerow(
            [
                row["case"],
                row["label"],
                row["z"],
                row["observed_fs8"],
                row["sigma"],
                row["model_fs8"],
                row["residual_model_minus_obs"],
                row["pull_sigma"],
                row["chi2_contribution"],
            ]
        )


with open("growth_survivor_model_curves.csv", "w", newline="") as f:
    writer = csv.writer(f)

    header = ["z"]
    for name in growth_tables:
        header.append(name + "_f_sigma8")
    writer.writerow(header)

    for i, z in enumerate(Z_GRID):
        row = [float(z)]
        for name in growth_tables:
            row.append(float(growth_tables[name]["f_sigma8"][i]))
        writer.writerow(row)


success_names = [
    name for name, data in results.items()
    if data["status"] == "success"
]

if success_names:
    labels = success_names
    chi2_vals = [results[name]["chi2_growth_observed"] for name in labels]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    plt.bar(x, chi2_vals)
    plt.xticks(x, labels, rotation=55, ha="right", fontsize=8)
    plt.ylabel("chi-square against observed f_sigma8 proxy")
    plt.title("Growth survivor chi-square by case")
    plt.tight_layout()
    plt.savefig("growth_survivor_chi2_by_case.png", dpi=160)
    plt.close()

    obs_z = np.array([x["z"] for x in OBS_FSIGMA8])
    obs_y = np.array([x["fs8"] for x in OBS_FSIGMA8])
    obs_err = np.array([x["sigma"] for x in OBS_FSIGMA8])

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        obs_z,
        obs_y,
        yerr=obs_err,
        fmt="o",
        markersize=4,
        capsize=2,
        label="observed proxy table",
    )

    for name in success_names:
        gt = growth_tables[name]
        plt.plot(gt["z"], gt["f_sigma8"], label=name, linewidth=1.2)

    plt.xlabel("z")
    plt.ylabel("f_sigma8(z)")
    plt.title("Growth survivor curves against observed proxy table")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("growth_survivor_fsigma8_curves.png", dpi=160)
    plt.close()


print("")
print("TAIRID growth survivor chi-square test complete.")
print("Created:")
print("  growth_survivor_chisq_summary.json")
print("  growth_survivor_chisq_summary.csv")
print("  growth_survivor_observation_residuals.csv")
print("  growth_survivor_model_curves.csv")
print("  growth_survivor_chi2_by_case.png")
print("  growth_survivor_fsigma8_curves.png")
