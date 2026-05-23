#!/usr/bin/env python3
"""
TAIRID Pantheon+SH0ES ladder-offset Hubble gate test v3.

Purpose:
The previous ladder-split test used one global magnitude offset. That may be too
crude for a ladder-boundary idea.

This test allows different offset schemes:
- global_offset
- calibrator_vs_noncal_offset
- ladder_three_offsets: calibrator / SH0ES Hubble-flow / rest

Important:
If a calibrator-only gate becomes harmless only after a separate calibrator
offset is allowed, that is not proof. It means the supernova-only diagonal
screen cannot distinguish the gate from calibration-offset freedom. That result
must be labeled as offset-degenerate and handed to a real Cepheid/SH0ES
likelihood later.

Boundary:
This is not the full Pantheon+SH0ES likelihood.
This does not use the full covariance matrix.
This is not a Cepheid likelihood.
This is not a BAO likelihood.
This is not a Planck likelihood.
This does not prove TAIRID cosmology.
It is a diagonal-only ladder-offset pressure screen.
"""

import csv
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


OUTDIR = Path("pantheon_ladder_offsets_gate_v3_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"

C_LIGHT = 299792.458

H0_PLANCK_SIDE = 66.89318
H0_SHOES_TARGET = 73.04
SHOES_SIGMA = 1.04

OMEGA_B = 0.0223700000
OMEGA_CDM = 0.1200000000
OMEGA_M_PHYSICAL = OMEGA_B + OMEGA_CDM

BEST_LADDER_V2_A = 0.0875
BEST_LADDER_V2_ZC = 0.092
BEST_LADDER_V2_W = 0.068

BAO_Z = np.array([0.38, 0.51, 0.61, 0.70, 0.85, 1.48, 2.33])
CMB_Z = np.array([1100.0])

GATE_MODES = [
    "all_rows",
    "calibrator_only",
    "shoes_hf_only",
    "calibrator_and_hf",
    "lowz_noncal_only",
]

OFFSET_SCHEMES = [
    "global_offset",
    "calibrator_vs_noncal_offset",
    "ladder_three_offsets",
]


def download_data():
    path = OUTDIR / "Pantheon_SH0ES_downloaded.dat"
    with urllib.request.urlopen(DATA_URL, timeout=120) as response:
        data = response.read()
    path.write_bytes(data)
    return path


def parse_table(path):
    lines = path.read_text(errors="replace").splitlines()
    header = None
    rows = []

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


def extract_arrays(header, rows):
    z_col = choose_column(header, ["zHD", "zCMB", "zHEL", "zcmb", "z"])
    mu_col = choose_column(header, ["MU_SH0ES", "MU", "mu", "m_b_corr"])
    err_col = choose_column(header, ["MU_SH0ES_ERR_DIAG", "MUERR", "MU_ERR", "m_b_corr_err_DIAG", "MU_SH0ES_ERR"])

    if z_col is None:
        raise RuntimeError(f"No redshift column found. Header: {header}")

    if mu_col is None:
        raise RuntimeError(f"No distance-modulus-like column found. Header: {header}")

    if err_col is None:
        raise RuntimeError(f"No uncertainty column found. Header: {header}")

    for col in ["IS_CALIBRATOR", "USED_IN_SH0ES_HF"]:
        if col not in header:
            raise RuntimeError(f"Required split column missing: {col}")

    z = []
    mu = []
    sigma = []
    is_cal = []
    used_hf = []

    for row in rows:
        try:
            zz = float(row[z_col])
            mm = float(row[mu_col])
            ss = float(row[err_col])
            cal = int(float(row["IS_CALIBRATOR"]))
            hf = int(float(row["USED_IN_SH0ES_HF"]))
        except Exception:
            continue

        if not (np.isfinite(zz) and np.isfinite(mm) and np.isfinite(ss)):
            continue

        if zz <= 0.0 or ss <= 0.0:
            continue

        if not (cal == 1 or zz >= 0.01):
            continue

        if zz > 2.30:
            continue

        z.append(zz)
        mu.append(mm)
        sigma.append(max(ss, 0.03))
        is_cal.append(cal == 1)
        used_hf.append(hf == 1)

    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_cal = np.asarray(is_cal, dtype=bool)
    used_hf = np.asarray(used_hf, dtype=bool)

    order = np.argsort(z)

    return {
        "z_col": z_col,
        "mu_col": mu_col,
        "err_col": err_col,
        "z": z[order],
        "mu": mu[order],
        "sigma": sigma[order],
        "is_calibrator": is_cal[order],
        "used_in_shoes_hf": used_hf[order],
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


def window_gate(z, amplitude, z_center, width):
    z = np.asarray(z, dtype=float)
    window = 0.5 * (1.0 + np.tanh((z_center - z) / width))
    return 1.0 - amplitude * window


def h0_equivalent_from_gate(g):
    return H0_PLANCK_SIDE / np.asarray(g, dtype=float)


def mode_mask(mode, z, is_cal, used_hf):
    if mode == "all_rows":
        return np.ones_like(z, dtype=bool)

    if mode == "calibrator_only":
        return is_cal

    if mode == "shoes_hf_only":
        return used_hf

    if mode == "calibrator_and_hf":
        return is_cal | used_hf

    if mode == "lowz_noncal_only":
        return (z <= 0.15) & (~is_cal)

    raise ValueError(f"Unknown gate mode: {mode}")


def mechanism_mask(mode, z, is_cal, used_hf):
    if mode == "all_rows":
        return (z >= 0.01) & (z <= 0.15)

    if mode == "calibrator_only":
        return is_cal

    if mode == "shoes_hf_only":
        return used_hf

    if mode == "calibrator_and_hf":
        return is_cal | used_hf

    if mode == "lowz_noncal_only":
        return (z <= 0.15) & (~is_cal)

    raise ValueError(f"Unknown gate mode: {mode}")


def offset_groups(scheme, z, is_cal, used_hf):
    groups = np.full(len(z), "global", dtype=object)

    if scheme == "global_offset":
        return groups

    if scheme == "calibrator_vs_noncal_offset":
        groups[:] = "noncal"
        groups[is_cal] = "calibrator"
        return groups

    if scheme == "ladder_three_offsets":
        groups[:] = "rest"
        groups[used_hf] = "shoes_hf"
        groups[is_cal] = "calibrator"
        return groups

    raise ValueError(f"Unknown offset scheme: {scheme}")


def fit_group_offsets(mu_obs, sigma, mu_model, groups):
    w = 1.0 / (sigma * sigma)
    fitted = np.array(mu_model, dtype=float)
    offsets = {}

    for group in sorted(set(groups.tolist())):
        mask = groups == group
        offset = np.sum(w[mask] * (mu_obs[mask] - mu_model[mask])) / np.sum(w[mask])
        offsets[group] = float(offset)
        fitted[mask] = fitted[mask] + offset

    residual = mu_obs - fitted
    chi2 = float(np.sum((residual / sigma) ** 2))
    dof = int(len(mu_obs) - len(offsets))

    return offsets, residual, chi2, dof, fitted


def group_stats(name, mask, residual, sigma):
    if not np.any(mask):
        return {
            f"{name}_n": 0,
            f"{name}_mean_residual": float("nan"),
            f"{name}_rms_residual": float("nan"),
            f"{name}_chi2": float("nan"),
        }

    vals = residual[mask]
    sig = sigma[mask]

    return {
        f"{name}_n": int(np.sum(mask)),
        f"{name}_mean_residual": float(np.mean(vals)),
        f"{name}_rms_residual": float(np.sqrt(np.mean(vals * vals))),
        f"{name}_chi2": float(np.sum((vals / sig) ** 2)),
    }


def is_offset_degenerate(mode, offset_scheme):
    if offset_scheme == "global_offset":
        return False

    if offset_scheme == "calibrator_vs_noncal_offset" and mode == "calibrator_only":
        return True

    if offset_scheme == "ladder_three_offsets" and mode in ["calibrator_only", "shoes_hf_only", "calibrator_and_hf"]:
        return True

    return False


def evaluate_case(
    name,
    mode,
    offset_scheme,
    params,
    z,
    mu_obs,
    sigma,
    is_cal,
    used_hf,
    mu_planck,
    scheme_baseline_chi2,
):
    amplitude = params.get("amplitude", 0.0)
    z_center = params.get("z_center", 0.2)
    width = params.get("width", 0.05)

    applied = mode_mask(mode, z, is_cal, used_hf)

    gate_rows = np.ones_like(z, dtype=float)

    if amplitude > 0.0:
        gate_rows[applied] = window_gate(z[applied], amplitude, z_center, width)

    mu_model = mu_planck + 5.0 * np.log10(gate_rows)

    groups = offset_groups(offset_scheme, z, is_cal, used_hf)
    offsets, residual, chi2, dof, mu_fit = fit_group_offsets(mu_obs, sigma, mu_model, groups)

    delta_chi2 = None if scheme_baseline_chi2 is None else float(chi2 - scheme_baseline_chi2)

    mech = mechanism_mask(mode, z, is_cal, used_hf)

    if np.any(mech):
        mech_gate = np.ones_like(z[mech], dtype=float)
        if amplitude > 0.0:
            mech_gate = window_gate(z[mech], amplitude, z_center, width)
        h0_mechanism = h0_equivalent_from_gate(mech_gate)
        h0_mean = float(np.mean(h0_mechanism))
        h0_rms_error = float(np.sqrt(np.mean((h0_mechanism - H0_SHOES_TARGET) ** 2)))
    else:
        h0_mean = H0_PLANCK_SIDE
        h0_rms_error = abs(H0_PLANCK_SIDE - H0_SHOES_TARGET)

    h0_chi2_proxy = float((h0_rms_error / SHOES_SIGMA) ** 2)

    if mode == "all_rows":
        bao_gate = window_gate(BAO_Z, amplitude, z_center, width)
        cmb_gate = window_gate(CMB_Z, amplitude, z_center, width)
        applied_bao_max_drift = float(np.max(np.abs(bao_gate - 1.0)))
        applied_cmb_drift = float(abs(cmb_gate[0] - 1.0))
    else:
        applied_bao_max_drift = 0.0
        applied_cmb_drift = 0.0

    underlying_bao_gate = window_gate(BAO_Z, amplitude, z_center, width)
    underlying_cmb_gate = window_gate(CMB_Z, amplitude, z_center, width)

    underlying_bao_max_drift = float(np.max(np.abs(underlying_bao_gate - 1.0)))
    underlying_cmb_drift = float(abs(underlying_cmb_gate[0] - 1.0))

    cal = is_cal
    hf = used_hf
    lowz = (z <= 0.15) & (~is_cal)
    rest = (~is_cal) & (~used_hf)

    degenerate = is_offset_degenerate(mode, offset_scheme)

    if delta_chi2 is None:
        diagnostic = "baseline"
    elif degenerate and h0_rms_error <= 1.04 and delta_chi2 <= 5.0:
        diagnostic = "offset_absorbed_candidate_needs_cepheid_likelihood"
    elif h0_rms_error <= 1.04 and delta_chi2 <= 20.0 and applied_bao_max_drift <= 0.0025 and applied_cmb_drift <= 0.0002:
        diagnostic = "passes_split_offset_screen"
    elif h0_rms_error <= 1.50 and delta_chi2 <= 50.0 and applied_bao_max_drift <= 0.005 and applied_cmb_drift <= 0.0005:
        diagnostic = "near_pass_needs_full_covariance"
    elif h0_rms_error <= 1.50 and delta_chi2 > 50.0:
        diagnostic = "matches_H0_but_bad_SN_split_shape"
    elif delta_chi2 <= 20.0:
        diagnostic = "SN_safe_but_weak_H0"
    else:
        diagnostic = "fails_split_offset_pressure"

    sn_penalty = max(0.0, delta_chi2 or 0.0) / 20.0
    score = h0_chi2_proxy + sn_penalty + (applied_bao_max_drift / 0.0025) ** 2 + (applied_cmb_drift / 0.0002) ** 2

    row = {
        "name": name,
        "mode": mode,
        "offset_scheme": offset_scheme,
        "params": params,
        "diagnostic": diagnostic,
        "offset_degenerate": degenerate,
        "total_offset_score": float(score),
        "chi2_diag": chi2,
        "dof_diag": dof,
        "reduced_chi2_diag": float(chi2 / dof) if dof > 0 else float("nan"),
        "delta_chi2_vs_scheme_baseline": delta_chi2,
        "offsets": offsets,
        "h0_proxy_mean_for_mode": h0_mean,
        "h0_proxy_rms_error_for_mode": h0_rms_error,
        "h0_proxy_chi2_for_mode": h0_chi2_proxy,
        "applied_bao_max_drift_from_1": applied_bao_max_drift,
        "applied_cmb_drift_from_1": applied_cmb_drift,
        "underlying_window_bao_max_drift_from_1": underlying_bao_max_drift,
        "underlying_window_cmb_drift_from_1": underlying_cmb_drift,
        "gate_at_z_0p01": float(window_gate(np.array([0.01]), amplitude, z_center, width)[0]) if amplitude > 0 else 1.0,
        "gate_at_z_0p023": float(window_gate(np.array([0.023]), amplitude, z_center, width)[0]) if amplitude > 0 else 1.0,
        "gate_at_z_0p05": float(window_gate(np.array([0.05]), amplitude, z_center, width)[0]) if amplitude > 0 else 1.0,
        "gate_at_z_0p10": float(window_gate(np.array([0.10]), amplitude, z_center, width)[0]) if amplitude > 0 else 1.0,
        "gate_at_z_0p15": float(window_gate(np.array([0.15]), amplitude, z_center, width)[0]) if amplitude > 0 else 1.0,
        "gate_at_z_0p35": float(window_gate(np.array([0.35]), amplitude, z_center, width)[0]) if amplitude > 0 else 1.0,
        "gate_at_z_0p61": float(window_gate(np.array([0.61]), amplitude, z_center, width)[0]) if amplitude > 0 else 1.0,
        "gate_at_z_1p0": float(window_gate(np.array([1.0]), amplitude, z_center, width)[0]) if amplitude > 0 else 1.0,
        "gate_at_z_2p33": float(window_gate(np.array([2.33]), amplitude, z_center, width)[0]) if amplitude > 0 else 1.0,
        "gate_at_z_1100": float(window_gate(np.array([1100.0]), amplitude, z_center, width)[0]) if amplitude > 0 else 1.0,
        **group_stats("calibrator", cal, residual, sigma),
        **group_stats("shoes_hf", hf, residual, sigma),
        **group_stats("lowz_noncal", lowz, residual, sigma),
        **group_stats("rest", rest, residual, sigma),
    }

    row["_residual"] = residual
    row["_mu_fit"] = mu_fit
    row["_gate_rows"] = gate_rows

    return row


def binned_residuals(z, residual, bins):
    out = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (z >= lo) & (z < hi)

        if np.sum(mask) == 0:
            out.append([lo, hi, 0, float("nan"), float("nan")])
            continue

        vals = residual[mask]
        out.append([lo, hi, int(np.sum(mask)), float(np.mean(vals)), float(np.std(vals))])

    return out


data_path = download_data()
header, parsed_rows = parse_table(data_path)
sn = extract_arrays(header, parsed_rows)

z = sn["z"]
mu_obs = sn["mu"]
sigma = sn["sigma"]
is_cal = sn["is_calibrator"]
used_hf = sn["used_in_shoes_hf"]

dl_planck = luminosity_distance(z, H0_PLANCK_SIDE)
mu_planck = distance_modulus_from_dl_mpc(dl_planck)

scheme_baselines = {}
rows = []

for scheme in OFFSET_SCHEMES:
    base = evaluate_case(
        f"no_gate_{scheme}",
        "all_rows",
        scheme,
        {"amplitude": 0.0, "z_center": 0.2, "width": 0.05},
        z,
        mu_obs,
        sigma,
        is_cal,
        used_hf,
        mu_planck,
        scheme_baseline_chi2=None,
    )

    scheme_baselines[scheme] = base["chi2_diag"]

for scheme in OFFSET_SCHEMES:
    rows.append(
        evaluate_case(
            f"no_gate_{scheme}",
            "all_rows",
            scheme,
            {"amplitude": 0.0, "z_center": 0.2, "width": 0.05},
            z,
            mu_obs,
            sigma,
            is_cal,
            used_hf,
            mu_planck,
            scheme_baseline_chi2=scheme_baselines[scheme],
        )
    )

for scheme in OFFSET_SCHEMES:
    for mode in GATE_MODES:
        rows.append(
            evaluate_case(
                f"prior_best_{mode}_{scheme}",
                mode,
                scheme,
                {"amplitude": BEST_LADDER_V2_A, "z_center": BEST_LADDER_V2_ZC, "width": BEST_LADDER_V2_W},
                z,
                mu_obs,
                sigma,
                is_cal,
                used_hf,
                mu_planck,
                scheme_baseline_chi2=scheme_baselines[scheme],
            )
        )

for scheme in OFFSET_SCHEMES:
    for mode in GATE_MODES:
        for amplitude in np.linspace(0.060, 0.095, 15):
            for z_center in np.linspace(0.050, 0.160, 23):
                for width in np.linspace(0.030, 0.090, 17):
                    name = f"offset_{scheme}_{mode}_A{amplitude:.4f}_zc{z_center:.4f}_w{width:.4f}"
                    params = {
                        "amplitude": float(amplitude),
                        "z_center": float(z_center),
                        "width": float(width),
                    }

                    rows.append(
                        evaluate_case(
                            name,
                            mode,
                            scheme,
                            params,
                            z,
                            mu_obs,
                            sigma,
                            is_cal,
                            used_hf,
                            mu_planck,
                            scheme_baseline_chi2=scheme_baselines[scheme],
                        )
                    )

rows_sorted = sorted(rows, key=lambda row: row["total_offset_score"])
best = rows_sorted[0]

offset_absorbed = [
    row for row in rows_sorted
    if row["diagnostic"] == "offset_absorbed_candidate_needs_cepheid_likelihood"
]

passes = [
    row for row in rows_sorted
    if row["diagnostic"] == "passes_split_offset_screen"
]

near = [
    row for row in rows_sorted
    if row["diagnostic"] == "near_pass_needs_full_covariance"
]

bad_shape = [
    row for row in rows_sorted
    if row["diagnostic"] == "matches_H0_but_bad_SN_split_shape"
]

top_clean = []
for row in rows_sorted[:100]:
    top_clean.append({k: v for k, v in row.items() if not k.startswith("_")})

summary = {
    "boundary": "Pantheon+SH0ES diagonal ladder-offset screen only. Not full covariance and not proof.",
    "data_url": DATA_URL,
    "downloaded_file": str(data_path),
    "columns_used": {
        "z": sn["z_col"],
        "mu": sn["mu_col"],
        "sigma": sn["err_col"],
        "split_columns": ["IS_CALIBRATOR", "USED_IN_SH0ES_HF"],
    },
    "row_counts": {
        "total_rows_used": int(len(z)),
        "calibrator_rows": int(np.sum(is_cal)),
        "used_in_shoes_hf_rows": int(np.sum(used_hf)),
        "lowz_noncal_rows": int(np.sum((z <= 0.15) & (~is_cal))),
        "rest_rows": int(np.sum((~is_cal) & (~used_hf))),
    },
    "scheme_baseline_chi2": scheme_baselines,
    "H0_planck_side": H0_PLANCK_SIDE,
    "H0_SH0ES_like_target": H0_SHOES_TARGET,
    "SHOES_sigma_used": SHOES_SIGMA,
    "best_case": {k: v for k, v in best.items() if not k.startswith("_")},
    "offset_absorbed_count": len(offset_absorbed),
    "passes_count": len(passes),
    "near_pass_count": len(near),
    "matches_H0_but_bad_SN_split_shape_count": len(bad_shape),
    "top_100": top_clean,
}

(OUTDIR / "pantheon_ladder_offsets_gate_v3_summary.json").write_text(json.dumps(summary, indent=2))

header_out = [
    "rank",
    "name",
    "mode",
    "offset_scheme",
    "diagnostic",
    "offset_degenerate",
    "total_offset_score",
    "chi2_diag",
    "dof_diag",
    "reduced_chi2_diag",
    "delta_chi2_vs_scheme_baseline",
    "h0_proxy_mean_for_mode",
    "h0_proxy_rms_error_for_mode",
    "h0_proxy_chi2_for_mode",
    "applied_bao_max_drift_from_1",
    "applied_cmb_drift_from_1",
    "underlying_window_bao_max_drift_from_1",
    "underlying_window_cmb_drift_from_1",
    "calibrator_n",
    "calibrator_mean_residual",
    "calibrator_rms_residual",
    "calibrator_chi2",
    "shoes_hf_n",
    "shoes_hf_mean_residual",
    "shoes_hf_rms_residual",
    "shoes_hf_chi2",
    "lowz_noncal_n",
    "lowz_noncal_mean_residual",
    "lowz_noncal_rms_residual",
    "lowz_noncal_chi2",
    "rest_n",
    "rest_mean_residual",
    "rest_rms_residual",
    "rest_chi2",
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
    "offsets_json",
    "params_json",
]

with open(OUTDIR / "pantheon_ladder_offsets_gate_v3_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header_out)

    for rank, row in enumerate(rows_sorted[:2000], start=1):
        writer.writerow(
            [
                rank,
                row["name"],
                row["mode"],
                row["offset_scheme"],
                row["diagnostic"],
                row["offset_degenerate"],
                row["total_offset_score"],
                row["chi2_diag"],
                row["dof_diag"],
                row["reduced_chi2_diag"],
                row["delta_chi2_vs_scheme_baseline"],
                row["h0_proxy_mean_for_mode"],
                row["h0_proxy_rms_error_for_mode"],
                row["h0_proxy_chi2_for_mode"],
                row["applied_bao_max_drift_from_1"],
                row["applied_cmb_drift_from_1"],
                row["underlying_window_bao_max_drift_from_1"],
                row["underlying_window_cmb_drift_from_1"],
                row["calibrator_n"],
                row["calibrator_mean_residual"],
                row["calibrator_rms_residual"],
                row["calibrator_chi2"],
                row["shoes_hf_n"],
                row["shoes_hf_mean_residual"],
                row["shoes_hf_rms_residual"],
                row["shoes_hf_chi2"],
                row["lowz_noncal_n"],
                row["lowz_noncal_mean_residual"],
                row["lowz_noncal_rms_residual"],
                row["lowz_noncal_chi2"],
                row["rest_n"],
                row["rest_mean_residual"],
                row["rest_rms_residual"],
                row["rest_chi2"],
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
                json.dumps(row["offsets"]),
                json.dumps(row["params"]),
            ]
        )

with open(OUTDIR / "pantheon_ladder_offsets_gate_v3_summary.txt", "w") as f:
    f.write("TAIRID Pantheon+SH0ES ladder-offset Hubble gate v3\n\n")
    f.write("Boundary: diagonal ladder-offset screen only. Not full covariance and not proof.\n\n")
    f.write(f"Rows used: {len(z)}\n")
    f.write(f"Calibrator rows: {int(np.sum(is_cal))}\n")
    f.write(f"SH0ES Hubble-flow rows: {int(np.sum(used_hf))}\n\n")
    f.write("Scheme baseline chi2:\n")
    f.write(json.dumps(scheme_baselines, indent=2) + "\n\n")
    f.write("Best case:\n")
    f.write(json.dumps({k: v for k, v in best.items() if not k.startswith("_")}, indent=2) + "\n\n")
    f.write("Counts:\n")
    f.write(f"offset_absorbed_count: {len(offset_absorbed)}\n")
    f.write(f"passes_count: {len(passes)}\n")
    f.write(f"near_pass_count: {len(near)}\n")
    f.write(f"matches_H0_but_bad_SN_split_shape_count: {len(bad_shape)}\n\n")
    f.write("Interpretation guide:\n")
    f.write("- offset_absorbed means the SN-only diagonal screen cannot separate the gate from calibration-offset freedom.\n")
    f.write("- That is not proof; it means the next necessary test is real Cepheid/SH0ES calibration data.\n")
    f.write("- A non-degenerate pass would be stronger, but still requires full covariance.\n")

bins = np.array([0.0001, 0.005, 0.01, 0.023, 0.05, 0.10, 0.15, 0.25, 0.35, 0.60, 1.0, 1.5, 2.3])

with open(OUTDIR / "pantheon_ladder_offsets_gate_v3_binned_residuals.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case", "z_lo", "z_hi", "n", "mean_residual_mag", "std_residual_mag"])

    for row in [rows[0], best]:
        for b in binned_residuals(z, row["_residual"], bins):
            writer.writerow([row["name"], *b])

z_plot = np.unique(np.concatenate([np.linspace(0.0005, 0.5, 800), np.linspace(0.5, 2.5, 300)]))
z_plot.sort()

bp = best["params"]
best_gate_curve = window_gate(z_plot, bp["amplitude"], bp["z_center"], bp["width"])

plt.figure(figsize=(10, 6))
plt.plot(z_plot, np.ones_like(z_plot), label="No gate")
plt.plot(z_plot, window_gate(z_plot, BEST_LADDER_V2_A, BEST_LADDER_V2_ZC, BEST_LADDER_V2_W), label="Prior best ladder v2 window")
plt.plot(z_plot, best_gate_curve, label="Best ladder offset v3 window")
plt.axvline(0.15, linewidth=1)
plt.axvline(0.35, linewidth=1)
plt.xlabel("z")
plt.ylabel("Underlying G(z)")
plt.title("Pantheon ladder offsets gate v3: underlying gate window")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "pantheon_ladder_offsets_gate_v3_gate_plot.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(z, rows[0]["_residual"], s=5, alpha=0.25, label="Baseline residual")
plt.scatter(z, best["_residual"], s=5, alpha=0.25, label="Best v3 residual")
plt.axhline(0.0, linewidth=1)
plt.xlabel("z")
plt.ylabel("Residual mag after group offsets")
plt.title("Pantheon ladder offsets gate v3: residual comparison")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "pantheon_ladder_offsets_gate_v3_residual_plot.png", dpi=160)
plt.close()

group_names = ["calibrator", "shoes_hf", "lowz_noncal", "rest"]
x = np.arange(len(group_names))
base_means = [rows[0][f"{g}_mean_residual"] for g in group_names]
best_means = [best[f"{g}_mean_residual"] for g in group_names]

plt.figure(figsize=(10, 6))
plt.bar(x - 0.18, base_means, width=0.36, label="Baseline")
plt.bar(x + 0.18, best_means, width=0.36, label="Best v3")
plt.axhline(0.0, linewidth=1)
plt.xticks(x, group_names, rotation=20, ha="right")
plt.ylabel("Mean residual mag")
plt.title("Pantheon ladder offsets gate v3: group residual means")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "pantheon_ladder_offsets_gate_v3_group_residuals.png", dpi=160)
plt.close()

print("")
print("TAIRID Pantheon+SH0ES ladder-offset Hubble gate v3 complete.")
print("Created:")
print("  pantheon_ladder_offsets_gate_v3_outputs/pantheon_ladder_offsets_gate_v3_summary.json")
print("  pantheon_ladder_offsets_gate_v3_outputs/pantheon_ladder_offsets_gate_v3_summary.csv")
print("  pantheon_ladder_offsets_gate_v3_outputs/pantheon_ladder_offsets_gate_v3_summary.txt")
print("  pantheon_ladder_offsets_gate_v3_outputs/pantheon_ladder_offsets_gate_v3_binned_residuals.csv")
print("  pantheon_ladder_offsets_gate_v3_outputs/pantheon_ladder_offsets_gate_v3_gate_plot.png")
print("  pantheon_ladder_offsets_gate_v3_outputs/pantheon_ladder_offsets_gate_v3_residual_plot.png")
print("  pantheon_ladder_offsets_gate_v3_outputs/pantheon_ladder_offsets_gate_v3_group_residuals.png")
print("")
print("Boundary:")
print("  This is not the full Pantheon+ covariance likelihood.")
print("  This is a diagonal-only ladder-offset screen.")
