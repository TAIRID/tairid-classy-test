#!/usr/bin/env python3
"""
TAIRID Pantheon+SH0ES Hubble gate pressure test v1.

Purpose:
The localized Hubble gate passed a geometry screen and a simple observational
pressure screen. This test adds a real public supernova data pressure step using
Pantheon+SH0ES distance data.

This is still NOT the full Pantheon+ likelihood.

It does:
- Download Pantheon+SH0ES.dat from the public DataRelease repository.
- Parse redshift, distance modulus, and diagonal uncertainty columns.
- Compare no-gate, frozen monotonic gate, prior best localized gates, and
  searched localized recovery gates.
- Fit one global magnitude offset for each gate, because supernova distances
  have calibration/absolute-magnitude degeneracy.
- Score whether the gate can match local H0 while avoiding a bad supernova
  distance-shape penalty and recovering by BAO/CMB scales.

Boundary:
This is not a full Pantheon+SH0ES likelihood.
This does not use the full covariance matrix.
This is not a BAO likelihood.
This is not a Planck likelihood.
This does not prove TAIRID cosmology.
It is a real-data diagonal pressure screen.
"""

import csv
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


OUTDIR = Path("pantheon_hubble_gate_pressure_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/refs/heads/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"

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

BEST_CLEAN_V2_A = 0.082
BEST_CLEAN_V2_ZC = 0.2075
BEST_CLEAN_V2_W = 0.038

BEST_PRESSURE_V1_A = 0.084
BEST_PRESSURE_V1_ZC = 0.22125
BEST_PRESSURE_V1_W = 0.060

LOCAL_H0_Z = np.linspace(0.01, 0.15, 80)
BAO_Z = np.array([0.38, 0.51, 0.61, 0.70, 0.85, 1.48, 2.33])
CMB_Z = np.array([1100.0])


def download_data():
    path = OUTDIR / "Pantheon_SH0ES_downloaded.dat"

    with urllib.request.urlopen(DATA_URL, timeout=120) as response:
        data = response.read()

    path.write_bytes(data)
    return path


def parse_table(path):
    lines = path.read_text(errors="replace").splitlines()
    rows = []

    header = None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if header is None:
            if stripped.startswith("#"):
                stripped = stripped[1:].strip()
            header = stripped.split()
            continue

        if stripped.startswith("#"):
            continue

        values = stripped.split()

        if len(values) < len(header):
            continue

        row = {}

        for key, value in zip(header, values):
            try:
                row[key] = float(value)
            except Exception:
                row[key] = value

        rows.append(row)

    if header is None or not rows:
        raise RuntimeError("Could not parse Pantheon+SH0ES table.")

    return header, rows


def choose_column(header, options):
    for option in options:
        if option in header:
            return option
    return None


def extract_sn_arrays(header, rows):
    z_col = choose_column(header, ["zHD", "zCMB", "zHEL", "zcmb", "z"])
    mu_col = choose_column(header, ["MU_SH0ES", "MU", "mu", "m_b_corr"])
    err_col = choose_column(header, ["MU_SH0ES_ERR_DIAG", "MUERR", "MU_ERR", "m_b_corr_err_DIAG", "MU_SH0ES_ERR"])

    if z_col is None:
        raise RuntimeError(f"No redshift column found. Header: {header}")

    if mu_col is None:
        raise RuntimeError(f"No distance-modulus-like column found. Header: {header}")

    if err_col is None:
        raise RuntimeError(f"No uncertainty column found. Header: {header}")

    z = []
    mu = []
    sigma = []

    for row in rows:
        try:
            zz = float(row[z_col])
            mm = float(row[mu_col])
            ss = float(row[err_col])
        except Exception:
            continue

        if not (np.isfinite(zz) and np.isfinite(mm) and np.isfinite(ss)):
            continue

        if zz <= 0.0 or ss <= 0.0:
            continue

        z.append(zz)
        mu.append(mm)
        sigma.append(ss)

    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Keep a broad Pantheon-style supernova range.
    mask = (z >= 0.01) & (z <= 2.30) & np.isfinite(z) & np.isfinite(mu) & np.isfinite(sigma)

    z = z[mask]
    mu = mu[mask]
    sigma = sigma[mask]

    # Diagonal-only screen. Cap unrealistically tiny errors so one point does not
    # dominate this diagnostic.
    sigma = np.maximum(sigma, 0.03)

    order = np.argsort(z)

    return {
        "z_col": z_col,
        "mu_col": mu_col,
        "err_col": err_col,
        "z": z[order],
        "mu": mu[order],
        "sigma": sigma[order],
    }


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


def h0_equivalent_from_gate(g):
    return H0_PLANCK_SIDE / np.asarray(g, dtype=float)


def model_mu(z, gate_func):
    dl = luminosity_distance(z, H0_PLANCK_SIDE)
    gate = gate_func(z)
    dl_eff = dl * gate
    return distance_modulus_from_dl_mpc(dl_eff)


def fit_offset_and_chi2(mu_obs, sigma, mu_model):
    w = 1.0 / (sigma * sigma)

    offset = np.sum(w * (mu_obs - mu_model)) / np.sum(w)
    residual = mu_obs - (mu_model + offset)

    chi2 = float(np.sum((residual / sigma) ** 2))
    dof = int(len(mu_obs) - 1)

    return offset, residual, chi2, dof


def binned_residuals(z, residual, bins):
    out = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (z >= lo) & (z < hi)

        if np.sum(mask) == 0:
            out.append([lo, hi, 0, float("nan"), float("nan")])
            continue

        vals = residual[mask]
        out.append(
            [
                lo,
                hi,
                int(np.sum(mask)),
                float(np.mean(vals)),
                float(np.std(vals)),
            ]
        )

    return out


def evaluate_gate(name, family, params, gate_func, z, mu_obs, sigma, baseline_chi2=None):
    mu_pred = model_mu(z, gate_func)
    offset, residual, chi2, dof = fit_offset_and_chi2(mu_obs, sigma, mu_pred)

    local_gate = gate_func(LOCAL_H0_Z)
    local_h0 = h0_equivalent_from_gate(local_gate)
    local_h0_mean = float(np.mean(local_h0))
    local_h0_rms_error = float(np.sqrt(np.mean((local_h0 - H0_SHOES_TARGET) ** 2)))
    local_h0_chi2_proxy = float((local_h0_rms_error / SHOES_SIGMA) ** 2)

    bao_gate = gate_func(BAO_Z)
    cmb_gate = gate_func(CMB_Z)

    bao_max_drift = float(np.max(np.abs(bao_gate - 1.0)))
    bao_rms_drift = float(np.sqrt(np.mean((bao_gate - 1.0) ** 2)))
    cmb_drift = float(abs(cmb_gate[0] - 1.0))

    delta_chi2_vs_no_gate = None if baseline_chi2 is None else float(chi2 - baseline_chi2)

    high_z_mask = z >= 0.35
    low_z_mask = z <= 0.15

    low_z_residual_rms = float(np.sqrt(np.mean(residual[low_z_mask] ** 2))) if np.any(low_z_mask) else float("nan")
    high_z_residual_rms = float(np.sqrt(np.mean(residual[high_z_mask] ** 2))) if np.any(high_z_mask) else float("nan")

    # Diagnostic score. This is deliberately conservative:
    # - local H0 match is rewarded,
    # - supernova diagonal chi2 penalty is penalized,
    # - BAO/CMB recovery failures are heavily penalized.
    sn_penalty = 0.0
    if delta_chi2_vs_no_gate is not None:
        sn_penalty = max(0.0, delta_chi2_vs_no_gate) / 20.0

    total_score = (
        local_h0_chi2_proxy
        + sn_penalty
        + (bao_max_drift / 0.0025) ** 2
        + (cmb_drift / 0.0002) ** 2
    )

    if (
        local_h0_rms_error <= 1.04
        and bao_max_drift <= 0.0025
        and cmb_drift <= 0.0002
        and delta_chi2_vs_no_gate is not None
        and delta_chi2_vs_no_gate <= 20.0
    ):
        diagnostic = "passes_pantheon_diag_pressure"
    elif (
        local_h0_rms_error <= 1.50
        and bao_max_drift <= 0.005
        and cmb_drift <= 0.0005
        and delta_chi2_vs_no_gate is not None
        and delta_chi2_vs_no_gate <= 50.0
    ):
        diagnostic = "near_pass_needs_full_covariance"
    elif local_h0_rms_error <= 1.50 and delta_chi2_vs_no_gate is not None and delta_chi2_vs_no_gate > 50.0:
        diagnostic = "matches_H0_but_bad_SN_shape"
    elif local_h0_rms_error <= 1.50 and (bao_max_drift > 0.005 or cmb_drift > 0.0005):
        diagnostic = "matches_H0_but_fails_recovery"
    elif bao_max_drift <= 0.005 and cmb_drift <= 0.0005:
        diagnostic = "recovers_high_z_but_weak_H0"
    else:
        diagnostic = "fails_pantheon_diag_pressure"

    return {
        "name": name,
        "family": family,
        "params": params,
        "diagnostic": diagnostic,
        "total_score": float(total_score),
        "chi2_diag": chi2,
        "dof_diag": dof,
        "reduced_chi2_diag": float(chi2 / dof) if dof > 0 else float("nan"),
        "delta_chi2_vs_no_gate": delta_chi2_vs_no_gate,
        "offset_fit_mag": float(offset),
        "local_h0_mean": local_h0_mean,
        "local_h0_rms_error": local_h0_rms_error,
        "local_h0_chi2_proxy": local_h0_chi2_proxy,
        "bao_max_drift_from_1": bao_max_drift,
        "bao_rms_drift_from_1": bao_rms_drift,
        "cmb_drift_from_1": cmb_drift,
        "low_z_residual_rms": low_z_residual_rms,
        "high_z_residual_rms": high_z_residual_rms,
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
        "_residual": residual,
        "_mu_model": mu_pred + offset,
    }


data_path = download_data()
header, parsed_rows = parse_table(data_path)
sn = extract_sn_arrays(header, parsed_rows)

z = sn["z"]
mu_obs = sn["mu"]
sigma = sn["sigma"]

base = evaluate_gate(
    "no_gate_planck_side",
    "none",
    {},
    lambda zz: np.ones_like(np.asarray(zz, dtype=float)),
    z,
    mu_obs,
    sigma,
    baseline_chi2=None,
)

baseline_chi2 = base["chi2_diag"]

rows = []

rows.append(
    evaluate_gate(
        "no_gate_planck_side",
        "none",
        {},
        lambda zz: np.ones_like(np.asarray(zz, dtype=float)),
        z,
        mu_obs,
        sigma,
        baseline_chi2=baseline_chi2,
    )
)

rows.append(
    evaluate_gate(
        "frozen_joint_gate_v0p1",
        "frozen_monotonic",
        {"A": FROZEN_GATE_A, "z_t": FROZEN_GATE_ZT, "p": FROZEN_GATE_P},
        frozen_gate,
        z,
        mu_obs,
        sigma,
        baseline_chi2=baseline_chi2,
    )
)

rows.append(
    evaluate_gate(
        "best_clean_v2_gate",
        "localized_recovery_window",
        {"amplitude": BEST_CLEAN_V2_A, "z_center": BEST_CLEAN_V2_ZC, "width": BEST_CLEAN_V2_W},
        lambda zz: window_gate(zz, BEST_CLEAN_V2_A, BEST_CLEAN_V2_ZC, BEST_CLEAN_V2_W),
        z,
        mu_obs,
        sigma,
        baseline_chi2=baseline_chi2,
    )
)

rows.append(
    evaluate_gate(
        "best_pressure_v1_gate",
        "localized_recovery_window",
        {"amplitude": BEST_PRESSURE_V1_A, "z_center": BEST_PRESSURE_V1_ZC, "width": BEST_PRESSURE_V1_W},
        lambda zz: window_gate(zz, BEST_PRESSURE_V1_A, BEST_PRESSURE_V1_ZC, BEST_PRESSURE_V1_W),
        z,
        mu_obs,
        sigma,
        baseline_chi2=baseline_chi2,
    )
)

# Search around the v1 pressure gate. Keep this moderate because real data rows
# are evaluated each time.
for amplitude in np.linspace(0.070, 0.094, 49):
    for z_center in np.linspace(0.175, 0.255, 65):
        for width in np.linspace(0.030, 0.075, 46):
            name = f"pantheon_A{amplitude:.4f}_zc{z_center:.4f}_w{width:.4f}"
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
                    lambda zz, a=amplitude, zc=z_center, w=width: window_gate(zz, a, zc, w),
                    z,
                    mu_obs,
                    sigma,
                    baseline_chi2=baseline_chi2,
                )
            )

rows_sorted = sorted(rows, key=lambda row: row["total_score"])
best = rows_sorted[0]

passes = [row for row in rows_sorted if row["diagnostic"] == "passes_pantheon_diag_pressure"]
near = [row for row in rows_sorted if row["diagnostic"] == "near_pass_needs_full_covariance"]
bad_sn = [row for row in rows_sorted if row["diagnostic"] == "matches_H0_but_bad_SN_shape"]

summary_clean_rows = []
for row in rows_sorted[:50]:
    clean = {k: v for k, v in row.items() if not k.startswith("_")}
    summary_clean_rows.append(clean)

summary = {
    "boundary": "Pantheon+SH0ES diagonal pressure screen only. Not full covariance and not proof.",
    "data_url": DATA_URL,
    "downloaded_file": str(data_path),
    "columns_used": {
        "z": sn["z_col"],
        "mu": sn["mu_col"],
        "sigma": sn["err_col"],
    },
    "n_supernova_rows_used": int(len(z)),
    "H0_planck_side": H0_PLANCK_SIDE,
    "H0_SH0ES_like_target": H0_SHOES_TARGET,
    "SHOES_sigma_used": SHOES_SIGMA,
    "gate_convention": "D_eff(z) = D_Planck_LCDM(z) * G(z)",
    "no_gate_baseline": {k: v for k, v in rows[0].items() if not k.startswith("_")},
    "best_case": {k: v for k, v in best.items() if not k.startswith("_")},
    "passes_pantheon_diag_pressure_count": len(passes),
    "near_pass_count": len(near),
    "matches_H0_but_bad_SN_shape_count": len(bad_sn),
    "top_50": summary_clean_rows,
}

(OUTDIR / "pantheon_hubble_gate_pressure_v1_summary.json").write_text(json.dumps(summary, indent=2))

header_out = [
    "rank",
    "name",
    "family",
    "diagnostic",
    "total_score",
    "chi2_diag",
    "dof_diag",
    "reduced_chi2_diag",
    "delta_chi2_vs_no_gate",
    "offset_fit_mag",
    "local_h0_mean",
    "local_h0_rms_error",
    "local_h0_chi2_proxy",
    "bao_max_drift_from_1",
    "bao_rms_drift_from_1",
    "cmb_drift_from_1",
    "low_z_residual_rms",
    "high_z_residual_rms",
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

with open(OUTDIR / "pantheon_hubble_gate_pressure_v1_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header_out)

    for rank, row in enumerate(rows_sorted[:1000], start=1):
        writer.writerow(
            [
                rank,
                row["name"],
                row["family"],
                row["diagnostic"],
                row["total_score"],
                row["chi2_diag"],
                row["dof_diag"],
                row["reduced_chi2_diag"],
                row["delta_chi2_vs_no_gate"],
                row["offset_fit_mag"],
                row["local_h0_mean"],
                row["local_h0_rms_error"],
                row["local_h0_chi2_proxy"],
                row["bao_max_drift_from_1"],
                row["bao_rms_drift_from_1"],
                row["cmb_drift_from_1"],
                row["low_z_residual_rms"],
                row["high_z_residual_rms"],
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

with open(OUTDIR / "pantheon_hubble_gate_pressure_v1_summary.txt", "w") as f:
    f.write("TAIRID Pantheon+SH0ES Hubble gate pressure v1\n\n")
    f.write("Boundary: diagonal-only real-data pressure screen. Not full covariance and not proof.\n\n")
    f.write(f"Data URL: {DATA_URL}\n")
    f.write(f"Rows used: {len(z)}\n")
    f.write(f"Columns used: z={sn['z_col']}, mu={sn['mu_col']}, sigma={sn['err_col']}\n\n")
    f.write("No-gate baseline:\n")
    f.write(json.dumps({k: v for k, v in rows[0].items() if not k.startswith("_")}, indent=2) + "\n\n")
    f.write("Best case:\n")
    f.write(json.dumps({k: v for k, v in best.items() if not k.startswith("_")}, indent=2) + "\n\n")
    f.write("Counts:\n")
    f.write(f"passes_pantheon_diag_pressure_count: {len(passes)}\n")
    f.write(f"near_pass_count: {len(near)}\n")
    f.write(f"matches_H0_but_bad_SN_shape_count: {len(bad_sn)}\n\n")
    f.write("Interpretation guide:\n")
    f.write("- If passes exist, the localized gate survives this diagonal Pantheon pressure screen.\n")
    f.write("- If only near passes exist, use full covariance before drawing conclusions.\n")
    f.write("- If H0 matching creates large SN chi2 penalty, the gate shape is too visible to supernovae.\n")

# Save binned residuals for no-gate and best.
bins = np.array([0.01, 0.023, 0.05, 0.10, 0.15, 0.25, 0.35, 0.60, 1.0, 1.5, 2.3])

with open(OUTDIR / "pantheon_hubble_gate_pressure_v1_binned_residuals.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case", "z_lo", "z_hi", "n", "mean_residual_mag", "std_residual_mag"])

    for row in [rows[0], best]:
        for b in binned_residuals(z, row["_residual"], bins):
            writer.writerow([row["name"], *b])

# Plots.
z_plot = np.unique(np.concatenate([np.linspace(0.001, 0.5, 800), np.linspace(0.5, 2.5, 300)]))
z_plot.sort()

if best["family"] == "localized_recovery_window":
    bp = best["params"]
    best_gate_func = lambda zz: window_gate(zz, bp["amplitude"], bp["z_center"], bp["width"])
elif best["family"] == "frozen_monotonic":
    best_gate_func = frozen_gate
else:
    best_gate_func = lambda zz: np.ones_like(np.asarray(zz, dtype=float))

plt.figure(figsize=(10, 6))
plt.plot(z_plot, np.ones_like(z_plot), label="No gate")
plt.plot(z_plot, frozen_gate(z_plot), label="Frozen monotonic gate")
plt.plot(z_plot, window_gate(z_plot, BEST_PRESSURE_V1_A, BEST_PRESSURE_V1_ZC, BEST_PRESSURE_V1_W), label="Best pressure v1 gate")
plt.plot(z_plot, best_gate_func(z_plot), label="Best Pantheon pressure gate")
plt.axvline(0.15, linewidth=1)
plt.axvline(0.35, linewidth=1)
plt.xlabel("z")
plt.ylabel("G(z)")
plt.title("Pantheon Hubble gate pressure v1: gate shape")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "pantheon_hubble_gate_pressure_v1_gate_plot.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 6))
sample = min(len(z), 4000)
plt.errorbar(z[:sample], rows[0]["_residual"][:sample], yerr=sigma[:sample], fmt=".", markersize=2, alpha=0.25, label="No gate residual")
plt.scatter(z[:sample], best["_residual"][:sample], s=3, alpha=0.25, label="Best gate residual")
plt.axhline(0.0, linewidth=1)
plt.xlabel("z")
plt.ylabel("Residual mag after fitted offset")
plt.title("Pantheon Hubble gate pressure v1: residual comparison")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "pantheon_hubble_gate_pressure_v1_residual_plot.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 6))
for row in [rows[0], best]:
    binned = binned_residuals(z, row["_residual"], bins)
    mids = []
    means = []

    for lo, hi, n, mean, std in binned:
        if n > 0 and np.isfinite(mean):
            mids.append((lo + hi) / 2.0)
            means.append(mean)

    plt.plot(mids, means, marker="o", label=row["name"])

plt.axhline(0.0, linewidth=1)
plt.xlabel("z bin midpoint")
plt.ylabel("Mean residual mag")
plt.title("Pantheon Hubble gate pressure v1: binned residuals")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "pantheon_hubble_gate_pressure_v1_binned_residuals.png", dpi=160)
plt.close()

print("")
print("TAIRID Pantheon+SH0ES Hubble gate pressure v1 complete.")
print("Created:")
print("  pantheon_hubble_gate_pressure_v1_outputs/pantheon_hubble_gate_pressure_v1_summary.json")
print("  pantheon_hubble_gate_pressure_v1_outputs/pantheon_hubble_gate_pressure_v1_summary.csv")
print("  pantheon_hubble_gate_pressure_v1_outputs/pantheon_hubble_gate_pressure_v1_summary.txt")
print("  pantheon_hubble_gate_pressure_v1_outputs/pantheon_hubble_gate_pressure_v1_binned_residuals.csv")
print("  pantheon_hubble_gate_pressure_v1_outputs/pantheon_hubble_gate_pressure_v1_gate_plot.png")
print("  pantheon_hubble_gate_pressure_v1_outputs/pantheon_hubble_gate_pressure_v1_residual_plot.png")
print("  pantheon_hubble_gate_pressure_v1_outputs/pantheon_hubble_gate_pressure_v1_binned_residuals.png")
print("")
print("Boundary:")
print("  This is not the full Pantheon+ covariance likelihood.")
print("  This is a diagonal-only real-data pressure screen.")
