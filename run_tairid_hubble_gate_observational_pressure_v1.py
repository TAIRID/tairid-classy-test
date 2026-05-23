#!/usr/bin/env python3
"""
TAIRID Hubble gate observational pressure test v1.

Purpose:
The Hubble gate window probe found that a localized recovery gate can
geometrically mimic the local-H0 side while recovering by BAO/CMB scales.

This script adds simple observational pressure:

1. Local-H0 pressure:
   The gate should make the low-z H0-equivalent close to a SH0ES-like target.

2. BAO recovery pressure:
   The gate should be close to 1 at common BAO redshifts.

3. CMB recovery pressure:
   The gate should be effectively 1 by z = 1100.

4. Supernova-shape pressure:
   The gate should not create a sharp or jagged distance-modulus feature across
   the supernova redshift range. This is not a real Pantheon likelihood; it is
   a smoothness and shape sanity check.

Boundary:
This is not a real SH0ES likelihood.
This is not a real BAO likelihood.
This is not a real Pantheon/Pantheon+ likelihood.
This is not a Planck likelihood.
This does not prove TAIRID cosmology.
It is a first observational-pressure diagnostic for the localized Hubble gate.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


OUTDIR = Path("hubble_gate_observational_pressure_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

C_LIGHT = 299792.458

H0_PLANCK_SIDE = 66.89318
H0_SHOES_TARGET = 73.04
SHOES_SIGMA = 1.04

OMEGA_B = 0.0223700000
OMEGA_CDM = 0.1200000000
OMEGA_M_PHYSICAL = OMEGA_B + OMEGA_CDM

# Old frozen monotonic gate from the previous joint low-z work.
FROZEN_GATE_A = 0.6580586049
FROZEN_GATE_ZT = 0.2224541370
FROZEN_GATE_P = 2.0

# Best shape from the clean v2 window probe.
BEST_GATE_A = 0.082
BEST_GATE_ZC = 0.2075
BEST_GATE_W = 0.038

# Pressure windows.
LOCAL_H0_Z = np.linspace(0.01, 0.15, 80)
SN_Z = np.unique(
    np.concatenate(
        [
            np.linspace(0.01, 0.10, 60),
            np.linspace(0.10, 0.35, 80),
            np.linspace(0.35, 1.50, 120),
            np.linspace(1.50, 2.30, 50),
        ]
    )
)
SN_Z.sort()

BAO_Z = np.array([0.38, 0.51, 0.61, 0.70, 0.85, 1.48, 2.33])
CMB_Z = np.array([1100.0])

PLOT_Z = np.unique(
    np.concatenate(
        [
            np.linspace(0.001, 0.5, 700),
            np.linspace(0.5, 2.5, 300),
            LOCAL_H0_Z,
            SN_Z,
            BAO_Z,
            np.array([0.01, 0.023, 0.05, 0.10, 0.15, 0.35]),
        ]
    )
)
PLOT_Z.sort()


def omega_m_from_h0(H0):
    h = H0 / 100.0
    return OMEGA_M_PHYSICAL / (h * h)


def e_z(z, H0):
    om = omega_m_from_h0(H0)
    ol = 1.0 - om
    z = np.asarray(z, dtype=float)
    return np.sqrt(om * (1.0 + z) ** 3 + ol)


def luminosity_distance(z_values, H0):
    z_values = np.asarray(z_values, dtype=float)
    zmax = max(float(np.max(z_values)), 0.001)

    grid = np.linspace(0.0, zmax, 9000)
    inv_e = 1.0 / e_z(grid, H0)

    integral = cumulative_trapezoid(inv_e, grid, initial=0.0)
    dc_grid = (C_LIGHT / H0) * integral

    dc = np.interp(z_values, grid, dc_grid)
    dl = (1.0 + z_values) * dc
    return dl


def distance_modulus_from_dl_mpc(dl_mpc):
    dl_mpc = np.asarray(dl_mpc, dtype=float)
    return 5.0 * np.log10(dl_mpc) + 25.0


def frozen_gate(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-FROZEN_GATE_A * (z / (z + FROZEN_GATE_ZT)) ** FROZEN_GATE_P)


def window_gate(z, amplitude, z_center, width):
    z = np.asarray(z, dtype=float)
    window = 0.5 * (1.0 + np.tanh((z_center - z) / width))
    return 1.0 - amplitude * window


def required_ratio(z):
    d_planck = luminosity_distance(z, H0_PLANCK_SIDE)
    d_shoes = luminosity_distance(z, H0_SHOES_TARGET)
    return d_shoes / d_planck


def h0_equivalent_from_gate(g):
    return H0_PLANCK_SIDE / np.asarray(g, dtype=float)


def effective_distance_modulus(z, gate_func):
    dl_planck = luminosity_distance(z, H0_PLANCK_SIDE)
    dl_eff = dl_planck * gate_func(z)
    return distance_modulus_from_dl_mpc(dl_eff)


def smoothness_metrics(z, mu_eff, mu_planck):
    residual = mu_eff - mu_planck

    d1 = np.gradient(residual, z)
    d2 = np.gradient(d1, z)

    return {
        "sn_mu_residual_max_abs": float(np.max(np.abs(residual))),
        "sn_mu_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "sn_mu_slope_max_abs": float(np.max(np.abs(d1))),
        "sn_mu_curvature_max_abs": float(np.max(np.abs(d2))),
        "sn_mu_residual_at_z_0p05": float(np.interp(0.05, z, residual)),
        "sn_mu_residual_at_z_0p15": float(np.interp(0.15, z, residual)),
        "sn_mu_residual_at_z_0p35": float(np.interp(0.35, z, residual)),
        "sn_mu_residual_at_z_1p0": float(np.interp(1.0, z, residual)),
    }


def evaluate_gate(name, family, params, gate_func):
    local_gate = gate_func(LOCAL_H0_Z)
    local_h0_eff = h0_equivalent_from_gate(local_gate)

    local_h0_mean = float(np.mean(local_h0_eff))
    local_h0_rms_error = float(np.sqrt(np.mean((local_h0_eff - H0_SHOES_TARGET) ** 2)))
    local_h0_chi2_proxy = float((local_h0_rms_error / SHOES_SIGMA) ** 2)

    bao_gate = gate_func(BAO_Z)
    cmb_gate = gate_func(CMB_Z)

    bao_max_drift = float(np.max(np.abs(bao_gate - 1.0)))
    bao_rms_drift = float(np.sqrt(np.mean((bao_gate - 1.0) ** 2)))
    cmb_drift = float(abs(cmb_gate[0] - 1.0))

    mu_eff = effective_distance_modulus(SN_Z, gate_func)
    mu_planck = effective_distance_modulus(SN_Z, lambda z: np.ones_like(np.asarray(z, dtype=float)))
    sn = smoothness_metrics(SN_Z, mu_eff, mu_planck)

    # Tolerances are diagnostic, not official likelihood errors.
    local_term = local_h0_chi2_proxy
    bao_term = (bao_max_drift / 0.0025) ** 2
    cmb_term = (cmb_drift / 0.0002) ** 2
    sn_smooth_term = (sn["sn_mu_curvature_max_abs"] / 25.0) ** 2
    sn_highz_term = (abs(sn["sn_mu_residual_at_z_1p0"]) / 0.005) ** 2

    total_score = local_term + bao_term + cmb_term + sn_smooth_term + sn_highz_term

    if (
        local_h0_rms_error <= 1.04
        and bao_max_drift <= 0.0025
        and cmb_drift <= 0.0002
        and abs(sn["sn_mu_residual_at_z_1p0"]) <= 0.005
    ):
        diagnostic = "passes_v1_pressure"
    elif (
        local_h0_rms_error <= 1.50
        and bao_max_drift <= 0.005
        and cmb_drift <= 0.0005
        and abs(sn["sn_mu_residual_at_z_1p0"]) <= 0.010
    ):
        diagnostic = "near_pass_needs_real_likelihood"
    elif local_h0_rms_error <= 1.50 and (bao_max_drift > 0.005 or cmb_drift > 0.0005):
        diagnostic = "matches_H0_but_fails_recovery"
    elif bao_max_drift <= 0.005 and cmb_drift <= 0.0005:
        diagnostic = "recovers_high_z_but_weak_H0"
    else:
        diagnostic = "fails_v1_pressure"

    return {
        "name": name,
        "family": family,
        "params": params,
        "diagnostic": diagnostic,
        "total_pressure_score": float(total_score),
        "local_h0_mean": local_h0_mean,
        "local_h0_rms_error": local_h0_rms_error,
        "local_h0_chi2_proxy": local_h0_chi2_proxy,
        "bao_max_drift_from_1": bao_max_drift,
        "bao_rms_drift_from_1": bao_rms_drift,
        "cmb_drift_from_1": cmb_drift,
        **sn,
        "gate_at_z_0p01": float(gate_func(np.array([0.01]))[0]),
        "gate_at_z_0p023": float(gate_func(np.array([0.023]))[0]),
        "gate_at_z_0p05": float(gate_func(np.array([0.05]))[0]),
        "gate_at_z_0p10": float(gate_func(np.array([0.10]))[0]),
        "gate_at_z_0p15": float(gate_func(np.array([0.15]))[0]),
        "gate_at_z_0p35": float(gate_func(np.array([0.35]))[0]),
        "gate_at_z_0p61": float(gate_func(np.array([0.61]))[0]),
        "gate_at_z_1p0": float(gate_func(np.array([1.0]))[0]),
        "gate_at_z_2p33": float(gate_func(np.array([2.33]))[0]),
        "gate_at_z_1100": float(gate_func(np.array([1100.0]))[0]),
    }


rows = []

rows.append(
    evaluate_gate(
        "no_gate_planck_side",
        "none",
        {},
        lambda z: np.ones_like(np.asarray(z, dtype=float)),
    )
)

rows.append(
    evaluate_gate(
        "frozen_joint_gate_v0p1",
        "frozen_monotonic",
        {
            "A": FROZEN_GATE_A,
            "z_t": FROZEN_GATE_ZT,
            "p": FROZEN_GATE_P,
        },
        frozen_gate,
    )
)

rows.append(
    evaluate_gate(
        "best_clean_v2_gate",
        "localized_recovery_window",
        {
            "amplitude": BEST_GATE_A,
            "z_center": BEST_GATE_ZC,
            "width": BEST_GATE_W,
        },
        lambda z: window_gate(z, BEST_GATE_A, BEST_GATE_ZC, BEST_GATE_W),
    )
)

# Refine around best clean v2 gate.
for amplitude in np.linspace(0.070, 0.094, 49):
    for z_center in np.linspace(0.160, 0.250, 73):
        for width in np.linspace(0.020, 0.060, 41):
            name = f"pressure_A{amplitude:.4f}_zc{z_center:.4f}_w{width:.4f}"
            params = {
                "amplitude": float(amplitude),
                "z_center": float(z_center),
                "width": float(width),
            }

            rows.append(
                evaluate_gate(
                    name,
                    "localized_recovery_window",
                    params,
                    lambda z, a=amplitude, zc=z_center, w=width: window_gate(z, a, zc, w),
                )
            )

rows_sorted = sorted(rows, key=lambda row: row["total_pressure_score"])
best = rows_sorted[0]

passes = [row for row in rows_sorted if row["diagnostic"] == "passes_v1_pressure"]
near = [row for row in rows_sorted if row["diagnostic"] == "near_pass_needs_real_likelihood"]
h0_fail_recovery = [row for row in rows_sorted if row["diagnostic"] == "matches_H0_but_fails_recovery"]

summary = {
    "boundary": "Observational-pressure diagnostic only. Not an official likelihood and not proof.",
    "H0_planck_side": H0_PLANCK_SIDE,
    "H0_SH0ES_like_target": H0_SHOES_TARGET,
    "SHOES_sigma_used": SHOES_SIGMA,
    "gate_convention": "D_eff(z) = D_Planck_LCDM(z) * G(z)",
    "best_case": best,
    "passes_v1_pressure_count": len(passes),
    "near_pass_count": len(near),
    "matches_H0_but_fails_recovery_count": len(h0_fail_recovery),
    "top_25": rows_sorted[:25],
}

(OUTDIR / "hubble_gate_observational_pressure_v1_summary.json").write_text(json.dumps(summary, indent=2))

header = [
    "rank",
    "name",
    "family",
    "diagnostic",
    "total_pressure_score",
    "local_h0_mean",
    "local_h0_rms_error",
    "local_h0_chi2_proxy",
    "bao_max_drift_from_1",
    "bao_rms_drift_from_1",
    "cmb_drift_from_1",
    "sn_mu_residual_max_abs",
    "sn_mu_residual_rms",
    "sn_mu_slope_max_abs",
    "sn_mu_curvature_max_abs",
    "sn_mu_residual_at_z_0p05",
    "sn_mu_residual_at_z_0p15",
    "sn_mu_residual_at_z_0p35",
    "sn_mu_residual_at_z_1p0",
    "gate_at_z_0p01",
    "gate_at_z_0p023",
    "gate_at_z_0p05",
    "gate_at_z_0p10",
    "gate_at_z_0p15",
    "gate_at_z_0p35",
    "gate_at_z_0p61",
    "gate_at_z_1p0",
    "gate_at_z_2p33",
    "gate_at_z_1100",
    "params_json",
]

with open(OUTDIR / "hubble_gate_observational_pressure_v1_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for rank, row in enumerate(rows_sorted[:800], start=1):
        writer.writerow(
            [
                rank,
                row["name"],
                row["family"],
                row["diagnostic"],
                row["total_pressure_score"],
                row["local_h0_mean"],
                row["local_h0_rms_error"],
                row["local_h0_chi2_proxy"],
                row["bao_max_drift_from_1"],
                row["bao_rms_drift_from_1"],
                row["cmb_drift_from_1"],
                row["sn_mu_residual_max_abs"],
                row["sn_mu_residual_rms"],
                row["sn_mu_slope_max_abs"],
                row["sn_mu_curvature_max_abs"],
                row["sn_mu_residual_at_z_0p05"],
                row["sn_mu_residual_at_z_0p15"],
                row["sn_mu_residual_at_z_0p35"],
                row["sn_mu_residual_at_z_1p0"],
                row["gate_at_z_0p01"],
                row["gate_at_z_0p023"],
                row["gate_at_z_0p05"],
                row["gate_at_z_0p10"],
                row["gate_at_z_0p15"],
                row["gate_at_z_0p35"],
                row["gate_at_z_0p61"],
                row["gate_at_z_1p0"],
                row["gate_at_z_2p33"],
                row["gate_at_z_1100"],
                json.dumps(row["params"]),
            ]
        )

with open(OUTDIR / "hubble_gate_observational_pressure_v1_summary.txt", "w") as f:
    f.write("TAIRID Hubble gate observational pressure test v1\n\n")
    f.write("Boundary: diagnostic only. Not SH0ES/BAO/SN/Planck likelihood and not proof.\n\n")
    f.write(f"Planck-side H0 used: {H0_PLANCK_SIDE}\n")
    f.write(f"SH0ES-like target H0 used: {H0_SHOES_TARGET} +/- {SHOES_SIGMA}\n\n")
    f.write("Best case:\n")
    f.write(json.dumps(best, indent=2) + "\n\n")
    f.write("Counts:\n")
    f.write(f"passes_v1_pressure_count: {len(passes)}\n")
    f.write(f"near_pass_count: {len(near)}\n")
    f.write(f"matches_H0_but_fails_recovery_count: {len(h0_fail_recovery)}\n\n")
    f.write("Interpretation guide:\n")
    f.write("- If v1 passes exist, the localized gate survives this first pressure screen.\n")
    f.write("- If only near passes exist, the shape may still be useful but needs real likelihoods.\n")
    f.write("- If only H0/fail-recovery cases exist, the gate cannot hide from BAO/CMB.\n")

# Plots.
z = PLOT_Z
target_ratio = required_ratio(z)
no_gate = np.ones_like(z)
frozen = frozen_gate(z)

best_params = best["params"]
if best["family"] == "localized_recovery_window":
    best_gate = window_gate(z, best_params["amplitude"], best_params["z_center"], best_params["width"])
else:
    best_gate = frozen if best["family"] == "frozen_monotonic" else no_gate

v2_gate = window_gate(z, BEST_GATE_A, BEST_GATE_ZC, BEST_GATE_W)

plt.figure(figsize=(10, 6))
plt.plot(z, target_ratio, label="Required local-H0 distance ratio")
plt.plot(z, no_gate, label="No gate")
plt.plot(z, frozen, label="Frozen monotonic gate")
plt.plot(z, v2_gate, label="Best clean v2 gate")
plt.plot(z, best_gate, label="Best v1 pressure gate")
plt.axvline(0.15, linewidth=1)
plt.axvline(0.35, linewidth=1)
plt.xlabel("z")
plt.ylabel("G(z)")
plt.title("Hubble gate observational pressure v1: distance gate")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "hubble_gate_pressure_v1_gate_plot.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(z, h0_equivalent_from_gate(target_ratio), label="Target equivalent H0")
plt.plot(z, h0_equivalent_from_gate(v2_gate), label="Best clean v2 gate")
plt.plot(z, h0_equivalent_from_gate(best_gate), label="Best v1 pressure gate")
plt.axhline(H0_PLANCK_SIDE, linewidth=1)
plt.axhline(H0_SHOES_TARGET, linewidth=1)
plt.axvline(0.15, linewidth=1)
plt.axvline(0.35, linewidth=1)
plt.xlabel("z")
plt.ylabel("H0-equivalent")
plt.title("Hubble gate observational pressure v1: H0-equivalent")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "hubble_gate_pressure_v1_h0_equivalent_plot.png", dpi=160)
plt.close()

mu_planck = effective_distance_modulus(z, lambda zz: np.ones_like(np.asarray(zz, dtype=float)))
mu_best = effective_distance_modulus(z, lambda zz: window_gate(zz, best_params["amplitude"], best_params["z_center"], best_params["width"]))
mu_v2 = effective_distance_modulus(z, lambda zz: window_gate(zz, BEST_GATE_A, BEST_GATE_ZC, BEST_GATE_W))
mu_frozen = effective_distance_modulus(z, frozen_gate)

plt.figure(figsize=(10, 6))
plt.plot(z, mu_frozen - mu_planck, label="Frozen monotonic residual")
plt.plot(z, mu_v2 - mu_planck, label="Best clean v2 residual")
plt.plot(z, mu_best - mu_planck, label="Best v1 pressure residual")
plt.axhline(0.0, linewidth=1)
plt.axvline(0.15, linewidth=1)
plt.axvline(0.35, linewidth=1)
plt.xlabel("z")
plt.ylabel("Distance modulus residual vs Planck-side LCDM")
plt.title("Hubble gate observational pressure v1: supernova-shape residual")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "hubble_gate_pressure_v1_sn_residual_plot.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(z, np.abs(frozen - 1.0), label="Frozen monotonic drift from 1")
plt.plot(z, np.abs(v2_gate - 1.0), label="Best clean v2 drift from 1")
plt.plot(z, np.abs(best_gate - 1.0), label="Best v1 pressure drift from 1")
plt.axvline(0.35, linewidth=1)
plt.xlabel("z")
plt.ylabel("|G(z) - 1|")
plt.title("Hubble gate observational pressure v1: recovery")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "hubble_gate_pressure_v1_recovery_plot.png", dpi=160)
plt.close()

print("")
print("TAIRID Hubble gate observational pressure v1 complete.")
print("Created:")
print("  hubble_gate_observational_pressure_v1_outputs/hubble_gate_observational_pressure_v1_summary.json")
print("  hubble_gate_observational_pressure_v1_outputs/hubble_gate_observational_pressure_v1_summary.csv")
print("  hubble_gate_observational_pressure_v1_outputs/hubble_gate_observational_pressure_v1_summary.txt")
print("  hubble_gate_observational_pressure_v1_outputs/hubble_gate_pressure_v1_gate_plot.png")
print("  hubble_gate_observational_pressure_v1_outputs/hubble_gate_pressure_v1_h0_equivalent_plot.png")
print("  hubble_gate_observational_pressure_v1_outputs/hubble_gate_pressure_v1_sn_residual_plot.png")
print("  hubble_gate_observational_pressure_v1_outputs/hubble_gate_pressure_v1_recovery_plot.png")
print("")
print("Boundary:")
print("  This is not a real likelihood.")
print("  This is a first observational-pressure screen for the localized Hubble gate.")
