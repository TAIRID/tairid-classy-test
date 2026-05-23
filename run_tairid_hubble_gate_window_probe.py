#!/usr/bin/env python3
"""
TAIRID Hubble gate window probe.

Purpose:
The warm/delayed-neutral proxy did not solve S8 or Hubble tension. This test
switches to the earlier TAIRID distance/accessibility idea.

Question:
Can a localized distance/accessibility gate mimic the local-H0 side while
recovering back to near-normal before BAO/CMB scales?

Interpretation:
This is not a physical proof. It is a geometry feasibility test.

The probe compares:
1. Planck-side LCDM distance curve.
2. SH0ES-like high-H0 distance curve.
3. Frozen TAIRID monotonic gate from the previous low-redshift work.
4. A searched localized recovery gate.

Gate convention:
D_eff(z) = D_Planck_LCDM(z) * G(z)

A gate G(z) < 1 at low redshift makes distances look shorter, which mimics
a higher local H0. For Planck/BAO safety, G(z) should recover close to 1 by
BAO/CMB redshifts.

Boundary:
This is not a Planck likelihood.
This is not a BAO likelihood.
This is not a supernova likelihood.
This is not proof of TAIRID cosmology.
It is a Hubble-branch shape diagnostic.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


OUTDIR = Path("hubble_gate_window_probe_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

C_LIGHT = 299792.458

H0_PLANCK_SIDE = 66.89318
H0_SHOES_TARGET = 73.04
SHOES_SIGMA = 1.04

OMEGA_B = 0.0223700000
OMEGA_CDM = 0.1200000000
OMEGA_M_PHYSICAL = OMEGA_B + OMEGA_CDM

FROZEN_GATE_A = 0.6580586049
FROZEN_GATE_ZT = 0.2224541370
FROZEN_GATE_P = 2.0

LOW_Z_GRID = np.linspace(0.01, 0.15, 80)
MID_Z_GRID = np.linspace(0.15, 0.35, 60)
BAO_Z = np.array([0.38, 0.51, 0.61, 0.70, 0.85, 1.48, 2.33])
CMB_Z = np.array([1100.0])

PLOT_Z = np.unique(
    np.concatenate(
        [
            np.linspace(0.001, 0.5, 500),
            np.linspace(0.5, 2.5, 200),
            LOW_Z_GRID,
            MID_Z_GRID,
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

    grid = np.linspace(0.0, zmax, 6000)
    inv_e = 1.0 / e_z(grid, H0)

    integral = cumulative_trapezoid(inv_e, grid, initial=0.0)
    dc_grid = (C_LIGHT / H0) * integral

    dc = np.interp(z_values, grid, dc_grid)
    dl = (1.0 + z_values) * dc
    return dl


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


def gate_metrics(name, family, params, gate_func):
    low_target = required_ratio(LOW_Z_GRID)
    low_gate = gate_func(LOW_Z_GRID)

    mid_gate = gate_func(MID_Z_GRID)
    bao_gate = gate_func(BAO_Z)
    cmb_gate = gate_func(CMB_Z)

    low_rms = float(np.sqrt(np.mean((low_gate - low_target) ** 2)))
    low_mean_abs = float(np.mean(np.abs(low_gate - low_target)))
    low_max_abs = float(np.max(np.abs(low_gate - low_target)))

    h0_eff_low = h0_equivalent_from_gate(low_gate)
    h0_eff_mean = float(np.mean(h0_eff_low))
    h0_eff_rms_error = float(np.sqrt(np.mean((h0_eff_low - H0_SHOES_TARGET) ** 2)))

    mid_max_drift = float(np.max(np.abs(mid_gate - 1.0)))
    bao_max_drift = float(np.max(np.abs(bao_gate - 1.0)))
    cmb_drift = float(abs(cmb_gate[0] - 1.0))

    # Lower is better. This intentionally rewards a low-z H0 match but strongly
    # penalizes failure to recover by BAO/CMB scales.
    score = (
        (low_rms / 0.005)
        + (bao_max_drift / 0.002)
        + (cmb_drift / 0.0001)
        + (h0_eff_rms_error / 1.0)
    )

    if low_rms <= 0.0075 and bao_max_drift <= 0.0025 and cmb_drift <= 0.0002:
        diagnostic = "localized_gate_possible"
    elif low_rms <= 0.015 and bao_max_drift <= 0.005 and cmb_drift <= 0.0005:
        diagnostic = "near_window_but_needs_physics"
    elif low_rms <= 0.015:
        diagnostic = "matches_local_H0_but_breaks_high_z"
    elif bao_max_drift <= 0.005 and cmb_drift <= 0.0005:
        diagnostic = "high_z_safe_but_weak_H0"
    else:
        diagnostic = "no_overlap_shape"

    return {
        "name": name,
        "family": family,
        "params": params,
        "diagnostic": diagnostic,
        "score": score,
        "low_z_gate_rms_vs_required": low_rms,
        "low_z_gate_mean_abs_vs_required": low_mean_abs,
        "low_z_gate_max_abs_vs_required": low_max_abs,
        "h0_eff_mean_low_z": h0_eff_mean,
        "h0_eff_rms_error_low_z": h0_eff_rms_error,
        "mid_z_max_drift_from_1": mid_max_drift,
        "bao_max_drift_from_1": bao_max_drift,
        "cmb_drift_from_1": cmb_drift,
        "gate_at_z_0p01": float(gate_func(np.array([0.01]))[0]),
        "gate_at_z_0p023": float(gate_func(np.array([0.023]))[0]),
        "gate_at_z_0p05": float(gate_func(np.array([0.05]))[0]),
        "gate_at_z_0p10": float(gate_func(np.array([0.10]))[0]),
        "gate_at_z_0p15": float(gate_func(np.array([0.15]))[0]),
        "gate_at_z_0p35": float(gate_func(np.array([0.35]))[0]),
        "gate_at_z_0p61": float(gate_func(np.array([0.61]))[0]),
        "gate_at_z_2p33": float(gate_func(np.array([2.33]))[0]),
        "gate_at_z_1100": float(gate_func(np.array([1100.0]))[0]),
    }


rows = []

# No-gate baseline.
rows.append(
    gate_metrics(
        "no_gate_planck_side",
        "none",
        {},
        lambda z: np.ones_like(np.asarray(z, dtype=float)),
    )
)

# Frozen monotonic gate from prior low-redshift work.
rows.append(
    gate_metrics(
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

# Local recovery window search.
for amplitude in np.linspace(0.04, 0.12, 81):
    for z_center in np.linspace(0.04, 0.28, 97):
        for width in np.linspace(0.004, 0.060, 57):
            name = f"window_A{amplitude:.3f}_zc{z_center:.3f}_w{width:.3f}"
            params = {
                "amplitude": float(amplitude),
                "z_center": float(z_center),
                "width": float(width),
            }

            rows.append(
                gate_metrics(
                    name,
                    "localized_recovery_window",
                    params,
                    lambda z, a=amplitude, zc=z_center, w=width: window_gate(z, a, zc, w),
                )
            )

rows_sorted = sorted(rows, key=lambda row: row["score"])
best = rows_sorted[0]

possible = [
    row for row in rows_sorted
    if row["diagnostic"] == "localized_gate_possible"
]

near = [
    row for row in rows_sorted
    if row["diagnostic"] == "near_window_but_needs_physics"
]

h0_breaks_high_z = [
    row for row in rows_sorted
    if row["diagnostic"] == "matches_local_H0_but_breaks_high_z"
]

summary = {
    "boundary": "Hubble-branch distance/accessibility gate shape diagnostic only. Not a likelihood and not proof.",
    "H0_planck_side": H0_PLANCK_SIDE,
    "H0_SH0ES_like_target": H0_SHOES_TARGET,
    "SH0ES_sigma_used": SHOES_SIGMA,
    "gate_convention": "D_eff(z) = D_Planck_LCDM(z) * G(z)",
    "best_case": best,
    "localized_gate_possible_count": len(possible),
    "near_window_count": len(near),
    "matches_local_H0_but_breaks_high_z_count": len(h0_breaks_high_z),
    "top_25": rows_sorted[:25],
}

(OUTDIR / "hubble_gate_window_probe_summary.json").write_text(json.dumps(summary, indent=2))

with open(OUTDIR / "hubble_gate_window_probe_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)

    header = [
        "rank",
        "name",
        "family",
        "diagnostic",
        "score",
        "low_z_gate_rms_vs_required",
        "low_z_gate_mean_abs_vs_required",
        "low_z_gate_max_abs_vs_required",
        "h0_eff_mean_low_z",
        "h0_eff_rms_error_low_z",
        "mid_z_max_drift_from_1",
        "bao_max_drift_from_1",
        "cmb_drift_from_1",
        "gate_at_z_0p01",
        "gate_at_z_0p023",
        "gate_at_z_0p05",
        "gate_at_z_0p10",
        "gate_at_z_0p15",
        "gate_at_z_0p35",
        "gate_at_z_0p61",
        "gate_at_z_2p33",
        "gate_at_z_1100",
        "params_json",
    ]

    writer.writerow(header)

    for rank, row in enumerate(rows_sorted[:500], start=1):
        writer.writerow(
            [
                rank,
                row["name"],
                row["family"],
                row["diagnostic"],
                row["score"],
                row["low_z_gate_rms_vs_required"],
                row["low_z_gate_mean_abs_vs_required"],
                row["low_z_gate_max_abs_vs_required"],
                row["h0_eff_mean_low_z"],
                row["h0_eff_rms_error_low_z"],
                row["mid_z_max_drift_from_1"],
                row["bao_max_drift_from_1"],
                row["cmb_drift_from_1"],
                row["gate_at_z_0p01"],
                row["gate_at_z_0p023"],
                row["gate_at_z_0p05"],
                row["gate_at_z_0p10"],
                row["gate_at_z_0p15"],
                row["gate_at_z_0p35"],
                row["gate_at_z_0p61"],
                row["gate_at_z_2p33"],
                row["gate_at_z_1100"],
                json.dumps(row["params"]),
            ]
        )

with open(OUTDIR / "hubble_gate_window_probe_summary.txt", "w") as f:
    f.write("TAIRID Hubble gate window probe\n\n")
    f.write("Boundary: shape diagnostic only. Not a Planck/BAO/SN likelihood and not proof.\n\n")
    f.write(f"Planck-side H0 used: {H0_PLANCK_SIDE}\n")
    f.write(f"SH0ES-like target H0 used: {H0_SHOES_TARGET} +/- {SHOES_SIGMA}\n\n")
    f.write("Best case:\n")
    f.write(json.dumps(best, indent=2) + "\n\n")
    f.write("Counts:\n")
    f.write(f"localized_gate_possible_count: {len(possible)}\n")
    f.write(f"near_window_count: {len(near)}\n")
    f.write(f"matches_local_H0_but_breaks_high_z_count: {len(h0_breaks_high_z)}\n\n")
    f.write("Interpretation guide:\n")
    f.write("- If the frozen monotonic gate ranks badly, the old gate is not the Hubble solution by itself.\n")
    f.write("- If localized window gates rank well, the Hubble branch needs a local/recovering accessibility boundary.\n")
    f.write("- If no localized window works, the simple gate idea is geometrically weak before likelihood testing.\n")

# Plot required ratio and best gates.
z = PLOT_Z
target = required_ratio(z)
no_gate = np.ones_like(z)
frozen = frozen_gate(z)

best_params = best["params"]
if best["family"] == "localized_recovery_window":
    best_gate = window_gate(
        z,
        best_params["amplitude"],
        best_params["z_center"],
        best_params["width"],
    )
else:
    best_gate = frozen if best["family"] == "frozen_monotonic" else no_gate

plt.figure(figsize=(10, 6))
plt.plot(z, target, label="Required ratio: D_L(H0=73.04)/D_L(H0=66.893)")
plt.plot(z, no_gate, label="No gate")
plt.plot(z, frozen, label="Frozen monotonic gate")
plt.plot(z, best_gate, label="Best searched gate")
plt.axvline(0.15, linewidth=1)
plt.axvline(0.35, linewidth=1)
plt.xlabel("z")
plt.ylabel("G(z)")
plt.title("Hubble gate window probe: distance ratio")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "hubble_gate_window_ratio_plot.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(z, h0_equivalent_from_gate(target), label="Target equivalent H0")
plt.plot(z, h0_equivalent_from_gate(frozen), label="Frozen monotonic gate equivalent H0")
plt.plot(z, h0_equivalent_from_gate(best_gate), label="Best searched gate equivalent H0")
plt.axhline(H0_PLANCK_SIDE, linewidth=1)
plt.axhline(H0_SHOES_TARGET, linewidth=1)
plt.axvline(0.15, linewidth=1)
plt.axvline(0.35, linewidth=1)
plt.xlabel("z")
plt.ylabel("H0-equivalent from gate")
plt.title("Hubble gate window probe: H0-equivalent")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "hubble_gate_window_h0_equivalent_plot.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(z, np.abs(frozen - 1.0), label="Frozen monotonic drift from 1")
plt.plot(z, np.abs(best_gate - 1.0), label="Best searched gate drift from 1")
plt.axvline(0.35, linewidth=1)
plt.xlabel("z")
plt.ylabel("|G(z) - 1|")
plt.title("High-z recovery behavior")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "hubble_gate_window_recovery_plot.png", dpi=160)
plt.close()

print("")
print("TAIRID Hubble gate window probe complete.")
print("Created:")
print("  hubble_gate_window_probe_outputs/hubble_gate_window_probe_summary.json")
print("  hubble_gate_window_probe_outputs/hubble_gate_window_probe_summary.csv")
print("  hubble_gate_window_probe_outputs/hubble_gate_window_probe_summary.txt")
print("  hubble_gate_window_probe_outputs/hubble_gate_window_ratio_plot.png")
print("  hubble_gate_window_probe_outputs/hubble_gate_window_h0_equivalent_plot.png")
print("  hubble_gate_window_probe_outputs/hubble_gate_window_recovery_plot.png")
print("")
print("Boundary:")
print("  This is not a likelihood.")
print("  This tests whether a local/recovering gate shape is even geometrically plausible.")
