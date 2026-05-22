#!/usr/bin/env python3
"""
TAIRID frontier observed growth chi-square test.

Purpose:
The lensing/S8 frontier refinement found better compromise candidates after
the 5%, 20 eV lane showed too much CMB lensing suppression.

This test checks whether the new frontier candidates still improve or preserve
the observed f_sigma8(z) proxy fit.

Boundary:
This is a diagnostic proxy test only.
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
    {
        "name": "cdm_baseline",
        "warm_fraction": 0.0,
        "m_ncdm_eV": None,
        "prior_pk10": 1.0,
        "prior_lensing_max": 0.0,
    },
    {
        "name": "strict_lensing_safe_1pct_15eV",
        "warm_fraction": 0.010,
        "m_ncdm_eV": 15.0,
        "prior_pk10": 0.900706,
        "prior_lensing_max": 0.029206,
    },
    {
        "name": "conservative_1pct_10eV",
        "warm_fraction": 0.010,
        "m_ncdm_eV": 10.0,
        "prior_pk10": 0.898958,
        "prior_lensing_max": 0.036391,
    },
    {
        "name": "frontier_1p25pct_10eV",
        "warm_fraction": 0.0125,
        "m_ncdm_eV": 10.0,
        "prior_pk10": 0.875259,
        "prior_lensing_max": 0.045303,
    },
    {
        "name": "best_strict_frontier_1p5pct_12p5eV",
        "warm_fraction": 0.015,
        "m_ncdm_eV": 12.5,
        "prior_pk10": 0.853394,
        "prior_lensing_max": 0.048263,
    },
    {
        "name": "near_frontier_1p75pct_15eV",
        "warm_fraction": 0.0175,
        "m_ncdm_eV": 15.0,
        "prior_pk10": 0.832451,
        "prior_lensing_max": 0.050614,
    },
    {
        "name": "near_frontier_2pct_15eV",
        "warm_fraction": 0.020,
        "m_ncdm_eV": 15.0,
        "prior_pk10": 0.810813,
        "prior_lensing_max": 0.057657,
    },
    {
        "name": "near_frontier_2p5pct_20eV",
        "warm_fraction": 0.025,
        "m_ncdm_eV": 20.0,
        "prior_pk10": 0.772864,
        "prior_lensing_max": 0.059700,
    },
    {
        "name": "old_best_5pct_20eV",
        "warm_fraction": 0.050,
        "m_ncdm_eV": 20.0,
        "prior_pk10": 0.595221,
        "prior_lensing_max": 0.116234,
    },
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

    return class_sigma8_z0, sigma8_values


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
        class_sigma8_z0, sigma8_values = run_class_case(params)

        f_growth, f_sigma8 = compute_growth_quantities(sigma8_values)
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
            "prior_pk10": float(case["prior_pk10"]),
            "prior_lensing_max": float(case["prior_lensing_max"]),
            "Omega_m_total": float(Omega_m),
            "class_sigma8_z0": class_sigma8_z0,
            "S8_from_class_sigma8": S8,
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

    except Exception as exc:
        print("  FAILED:", exc)

        results[name] = {
            "status": "failed",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "prior_pk10": float(case["prior_pk10"]),
            "prior_lensing_max": float(case["prior_lensing_max"]),
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

        if name == "cdm_baseline":
            data["diagnostic_flag"] = "baseline"
        elif (
            data["prior_lensing_max"] <= 0.05
            and data["prior_pk10"] >= 0.80
            and data["S8_from_class_sigma8"] <= 0.833
            and data["delta_vs_cdm_baseline"]["delta_chi2"] <= 0
        ):
            data["diagnostic_flag"] = "best_current_frontier_candidate"
        elif (
            data["prior_lensing_max"] <= 0.06
            and data["prior_pk10"] >= 0.75
            and data["S8_from_class_sigma8"] <= 0.829
            and data["delta_vs_cdm_baseline"]["delta_chi2"] <= 0
        ):
            data["diagnostic_flag"] = "stronger_s8_but_lensing_warning"
        elif data["delta_vs_cdm_baseline"]["delta_chi2"] <= 0:
            data["diagnostic_flag"] = "growth_survives_but_other_warning"
        else:
            data["diagnostic_flag"] = "growth_not_improved"


Path("frontier_growth_chisq_summary.json").write_text(json.dumps(results, indent=2))


with open("frontier_growth_chisq_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(
        [
            "case",
            "status",
            "diagnostic_flag",
            "warm_fraction",
            "m_ncdm_eV",
            "prior_pk10",
            "prior_lensing_max",
            "Omega_m_total",
            "class_sigma8_z0",
            "S8",
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

            writer.writerow(
                [
                    name,
                    data["status"],
                    data.get("diagnostic_flag"),
                    data["warm_fraction"],
                    data["m_ncdm_eV"],
                    data["prior_pk10"],
                    data["prior_lensing_max"],
                    data["Omega_m_total"],
                    data["class_sigma8_z0"],
                    data["S8_from_class_sigma8"],
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
                    "",
                    data["warm_fraction"],
                    data["m_ncdm_eV"],
                    data["prior_pk10"],
                    data["prior_lensing_max"],
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


with open("frontier_growth_observation_residuals.csv", "w", newline="") as f:
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


with open("frontier_growth_model_curves.csv", "w", newline="") as f:
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
    plt.title("Frontier growth chi-square by case")
    plt.tight_layout()
    plt.savefig("frontier_growth_chi2_by_case.png", dpi=160)
    plt.close()

    s8_values = [results[name]["S8_from_class_sigma8"] for name in labels]
    lensing_values = [results[name]["prior_lensing_max"] for name in labels]

    plt.figure(figsize=(10, 6))
    plt.scatter(lensing_values, s8_values)

    for label, x, y in zip(labels, lensing_values, s8_values):
        plt.annotate(label, (x, y), fontsize=7)

    plt.axhline(0.833, linewidth=1)
    plt.axvline(0.05, linewidth=1)
    plt.axvline(0.06, linewidth=1)
    plt.xlabel("prior max CMB lensing drift")
    plt.ylabel("S8")
    plt.title("Frontier candidates: S8 versus lensing drift")
    plt.tight_layout()
    plt.savefig("frontier_growth_s8_vs_lensing.png", dpi=160)
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
        plt.plot(gt["z"], gt["f_sigma8"], label=name, linewidth=1.1)

    plt.xlabel("z")
    plt.ylabel("f_sigma8(z)")
    plt.title("Frontier growth curves against observed proxy table")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("frontier_growth_fsigma8_curves.png", dpi=160)
    plt.close()


print("")
print("TAIRID frontier growth chi-square test complete.")
print("Created:")
print("  frontier_growth_chisq_summary.json")
print("  frontier_growth_chisq_summary.csv")
print("  frontier_growth_observation_residuals.csv")
print("  frontier_growth_model_curves.csv")
print("  frontier_growth_chi2_by_case.png")
print("  frontier_growth_s8_vs_lensing.png")
print("  frontier_growth_fsigma8_curves.png")
print("")
print("Read boundary:")
print("  This is a proxy growth test only.")
print("  It does not prove TAIRID cosmology.")
