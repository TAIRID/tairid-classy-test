#!/usr/bin/env python3
"""
TAIRID BAO distance-ratio pressure test.

Purpose:
The current best TAIRID proxy lane is:

    f_warm = 5%
    m_proxy = 20 eV

It has survived:
- S8 reduction
- high-k pressure better than the 7.5% lane
- observed growth proxy chi-square
- CMB TT peak-position and shape pressure

Now we test whether it distorts BAO-style background distance ratios
relative to the CDM proxy baseline.

Boundary:
This is not a full BAO likelihood.
It does not use DESI/BOSS/eBOSS covariance.
It does not prove TAIRID cosmology.
It is an internal distance-ratio pressure test against the CDM proxy baseline.
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

BAO_Z = [
    0.106,
    0.150,
    0.380,
    0.510,
    0.610,
    0.700,
    0.850,
    1.480,
    2.330,
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


def get_rs_drag(cosmo):
    try:
        return float(cosmo.rs_drag())
    except Exception:
        pass

    try:
        derived = cosmo.get_current_derived_parameters(["rs_drag"])
        return float(derived["rs_drag"])
    except Exception:
        pass

    return float("nan")


def run_class_case(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    sigma8 = float(cosmo.sigma8())
    rd = get_rs_drag(cosmo)

    rows = []

    for z in BAO_Z:
        da = float(cosmo.angular_distance(z))
        dm = da * (1.0 + z)

        hubble = float(cosmo.Hubble(z))
        dh = 1.0 / hubble

        dv = (z * dm * dm * dh) ** (1.0 / 3.0)

        rows.append(
            {
                "z": float(z),
                "D_A_Mpc": da,
                "D_M_Mpc": dm,
                "D_H_Mpc": dh,
                "D_V_Mpc": dv,
                "r_d_Mpc": rd,
                "D_M_over_rd": dm / rd,
                "D_H_over_rd": dh / rd,
                "D_V_over_rd": dv / rd,
            }
        )

    cosmo.struct_cleanup()
    cosmo.empty()

    return sigma8, rd, rows


def s8_from_sigma8(sigma8, Omega_m):
    return float(sigma8 * math.sqrt(Omega_m / 0.3))


results = {}
bao_tables = {}

for case in CASES:
    name = case["name"]
    params = make_params(case)
    omega_m, Omega_m = omega_total_matter_proxy(params)

    print("")
    print("Running case:", name)

    try:
        sigma8, rd, rows = run_class_case(params)
        S8 = s8_from_sigma8(sigma8, Omega_m)

        results[name] = {
            "status": "success",
            "warm_fraction": float(case["warm_fraction"]),
            "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
            "Omega_m_total": float(Omega_m),
            "sigma8": sigma8,
            "S8": S8,
            "r_d_Mpc": rd,
        }

        bao_tables[name] = rows

        case_file = f"{name}_bao_distances.txt"
        with open(case_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "z",
                    "D_A_Mpc",
                    "D_M_Mpc",
                    "D_H_Mpc",
                    "D_V_Mpc",
                    "r_d_Mpc",
                    "D_M_over_rd",
                    "D_H_over_rd",
                    "D_V_over_rd",
                ]
            )

            for row in rows:
                writer.writerow(
                    [
                        row["z"],
                        row["D_A_Mpc"],
                        row["D_M_Mpc"],
                        row["D_H_Mpc"],
                        row["D_V_Mpc"],
                        row["r_d_Mpc"],
                        row["D_M_over_rd"],
                        row["D_H_over_rd"],
                        row["D_V_over_rd"],
                    ]
                )

        results[name]["bao_distance_file"] = case_file

        print("  success")
        print("  S8:", S8)
        print("  r_d:", rd)

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

if baseline_name in bao_tables:
    base_rows = bao_tables[baseline_name]
    base_by_z = {row["z"]: row for row in base_rows}

    for name, rows in bao_tables.items():
        if name == baseline_name:
            results[name]["diagnostic_flag"] = "baseline"
            results[name]["max_abs_DM_ratio_drift"] = 0.0
            results[name]["max_abs_DH_ratio_drift"] = 0.0
            results[name]["max_abs_DV_ratio_drift"] = 0.0
            continue

        ratio_rows = []

        max_dm = 0.0
        max_dh = 0.0
        max_dv = 0.0

        for row in rows:
            z = row["z"]
            base = base_by_z[z]

            dm_ratio = row["D_M_over_rd"] / base["D_M_over_rd"]
            dh_ratio = row["D_H_over_rd"] / base["D_H_over_rd"]
            dv_ratio = row["D_V_over_rd"] / base["D_V_over_rd"]

            dm_drift = dm_ratio - 1.0
            dh_drift = dh_ratio - 1.0
            dv_drift = dv_ratio - 1.0

            max_dm = max(max_dm, abs(dm_drift))
            max_dh = max(max_dh, abs(dh_drift))
            max_dv = max(max_dv, abs(dv_drift))

            ratio_rows.append(
                {
                    "case": name,
                    "z": z,
                    "DM_over_rd_ratio_to_cdm": dm_ratio,
                    "DH_over_rd_ratio_to_cdm": dh_ratio,
                    "DV_over_rd_ratio_to_cdm": dv_ratio,
                    "DM_drift": dm_drift,
                    "DH_drift": dh_drift,
                    "DV_drift": dv_drift,
                }
            )

        ratio_file = f"{name}_bao_ratio_to_cdm.txt"
        with open(ratio_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "case",
                    "z",
                    "DM_over_rd_ratio_to_cdm",
                    "DH_over_rd_ratio_to_cdm",
                    "DV_over_rd_ratio_to_cdm",
                    "DM_drift",
                    "DH_drift",
                    "DV_drift",
                ]
            )

            for row in ratio_rows:
                writer.writerow(
                    [
                        row["case"],
                        row["z"],
                        row["DM_over_rd_ratio_to_cdm"],
                        row["DH_over_rd_ratio_to_cdm"],
                        row["DV_over_rd_ratio_to_cdm"],
                        row["DM_drift"],
                        row["DH_drift"],
                        row["DV_drift"],
                    ]
                )

        results[name]["bao_ratio_file"] = ratio_file
        results[name]["max_abs_DM_ratio_drift"] = float(max_dm)
        results[name]["max_abs_DH_ratio_drift"] = float(max_dh)
        results[name]["max_abs_DV_ratio_drift"] = float(max_dv)

        worst = max(max_dm, max_dh, max_dv)

        if worst <= 0.0025:
            results[name]["diagnostic_flag"] = "bao_shape_close_to_baseline"
        elif worst <= 0.01:
            results[name]["diagnostic_flag"] = "bao_warning_but_survives_proxy"
        else:
            results[name]["diagnostic_flag"] = "bao_distance_distortion_warning"


Path("bao_distance_pressure_summary.json").write_text(json.dumps(results, indent=2))


with open("bao_distance_pressure_summary.csv", "w", newline="") as f:
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
            "r_d_Mpc",
            "max_abs_DM_ratio_drift",
            "max_abs_DH_ratio_drift",
            "max_abs_DV_ratio_drift",
            "error",
        ]
    )

    for name, data in results.items():
        if data["status"] == "success":
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
                    data["r_d_Mpc"],
                    data.get("max_abs_DM_ratio_drift"),
                    data.get("max_abs_DH_ratio_drift"),
                    data.get("max_abs_DV_ratio_drift"),
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
                    data.get("error", ""),
                ]
            )


with open("bao_distance_ratios_to_cdm.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(
        [
            "case",
            "z",
            "DM_over_rd_ratio_to_cdm",
            "DH_over_rd_ratio_to_cdm",
            "DV_over_rd_ratio_to_cdm",
        ]
    )

    if baseline_name in bao_tables:
        base_by_z = {row["z"]: row for row in bao_tables[baseline_name]}

        for name, rows in bao_tables.items():
            for row in rows:
                z = row["z"]
                base = base_by_z[z]

                writer.writerow(
                    [
                        name,
                        z,
                        row["D_M_over_rd"] / base["D_M_over_rd"],
                        row["D_H_over_rd"] / base["D_H_over_rd"],
                        row["D_V_over_rd"] / base["D_V_over_rd"],
                    ]
                )


success_names = [
    name for name, data in results.items()
    if data["status"] == "success" and name != baseline_name
]

if success_names and baseline_name in bao_tables:
    base_by_z = {row["z"]: row for row in bao_tables[baseline_name]}

    plt.figure(figsize=(10, 6))

    for name in success_names:
        rows = bao_tables[name]

        z_values = []
        dv_ratios = []

        for row in rows:
            z = row["z"]
            base = base_by_z[z]
            z_values.append(z)
            dv_ratios.append(row["D_V_over_rd"] / base["D_V_over_rd"])

        plt.plot(z_values, dv_ratios, marker="o", label=name, linewidth=1.2)

    plt.axhline(1.0, linewidth=1)
    plt.xlabel("z")
    plt.ylabel("D_V/r_d ratio to CDM baseline")
    plt.title("BAO distance pressure: D_V/r_d")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("bao_distance_pressure_DV_ratio_plot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))

    for name in success_names:
        rows = bao_tables[name]

        z_values = []
        dm_ratios = []
        dh_ratios = []

        for row in rows:
            z = row["z"]
            base = base_by_z[z]
            z_values.append(z)
            dm_ratios.append(row["D_M_over_rd"] / base["D_M_over_rd"])
            dh_ratios.append(row["D_H_over_rd"] / base["D_H_over_rd"])

        plt.plot(z_values, dm_ratios, marker="o", label=name + " D_M/r_d", linewidth=1.1)
        plt.plot(z_values, dh_ratios, marker="x", label=name + " D_H/r_d", linewidth=1.1)

    plt.axhline(1.0, linewidth=1)
    plt.xlabel("z")
    plt.ylabel("ratio to CDM baseline")
    plt.title("BAO distance pressure: D_M/r_d and D_H/r_d")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("bao_distance_pressure_DM_DH_ratio_plot.png", dpi=160)
    plt.close()


print("")
print("TAIRID BAO distance pressure test complete.")
print("Created:")
print("  bao_distance_pressure_summary.json")
print("  bao_distance_pressure_summary.csv")
print("  bao_distance_ratios_to_cdm.csv")
print("  bao_distance_pressure_DV_ratio_plot.png")
print("  bao_distance_pressure_DM_DH_ratio_plot.png")
print("")
print("Read boundary:")
print("  This is an internal BAO-style distance-ratio pressure test.")
print("  It is not a full BAO likelihood and does not prove TAIRID cosmology.")
