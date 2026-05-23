#!/usr/bin/env python3
"""
TAIRID Pantheon+SH0ES calibration-covariance gate test v1.

Purpose:
The full-covariance Pantheon+SH0ES test showed that the best calibrator-only
TAIRID gate remains offset-degenerate. This test asks a narrower question:

Is the calibrator-boundary gate aligned with the calibration covariance
direction, or is it only hidden by offset freedom?

Covariance models compared:
1. STATONLY
2. STATONLY + CALIB systematic covariance
3. STAT+SYS full covariance

Gate tested:
- calibrator_only localized gate around the previous best corridor

Offset schemes:
- global_offset
- calibrator_vs_noncal_offset
- ladder_three_offsets

Boundary:
This is not a Cepheid likelihood.
This is not the full SH0ES calibration-ladder likelihood.
This is not BAO.
This is not Planck.
This does not prove TAIRID cosmology.
It is a covariance-component isolation screen.
"""

import csv
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve


OUTDIR = Path("pantheon_calibration_covariance_gate_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
COV_STATONLY_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STATONLY.cov"
COV_STAT_SYS_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
COV_CALIB_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/sytematic_groupings/Pantheon%2BSH0ES_122221_CALIB.cov"

C_LIGHT = 299792.458

H0_PLANCK_SIDE = 66.89318
H0_SHOES_TARGET = 73.04
SHOES_SIGMA = 1.04

OMEGA_B = 0.0223700000
OMEGA_CDM = 0.1200000000
OMEGA_M_PHYSICAL = OMEGA_B + OMEGA_CDM

BEST_FULLCOV_V1_A = 0.085
BEST_FULLCOV_V1_ZC = 0.185
BEST_FULLCOV_V1_W = 0.070

BAO_Z = np.array([0.38, 0.51, 0.61, 0.70, 0.85, 1.48, 2.33])
CMB_Z = np.array([1100.0])

OFFSET_SCHEMES = [
    "global_offset",
    "calibrator_vs_noncal_offset",
    "ladder_three_offsets",
]


def download_file(url, local_name):
    path = OUTDIR / local_name
    with urllib.request.urlopen(url, timeout=180) as response:
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

    for col in ["IS_CALIBRATOR", "USED_IN_SHOES_HF", "USED_IN_SH0ES_HF"]:
        if col in header:
            pass

    hf_col = "USED_IN_SH0ES_HF" if "USED_IN_SH0ES_HF" in header else "USED_IN_SHOES_HF"

    for col in ["IS_CALIBRATOR", hf_col]:
        if col not in header:
            raise RuntimeError(f"Required split column missing: {col}")

    z = []
    mu = []
    sigma = []
    is_cal = []
    used_hf = []
    keep_mask = []

    for row in rows:
        keep = False

        try:
            zz = float(row[z_col])
            mm = float(row[mu_col])
            ss = float(row[err_col])
            cal = int(float(row["IS_CALIBRATOR"]))
            hf = int(float(row[hf_col]))
        except Exception:
            keep_mask.append(False)
            continue

        if np.isfinite(zz) and np.isfinite(mm) and np.isfinite(ss) and zz > 0.0 and ss > 0.0:
            if (cal == 1 or zz >= 0.01) and zz <= 2.30:
                keep = True

        keep_mask.append(keep)

        if keep:
            z.append(zz)
            mu.append(mm)
            sigma.append(max(ss, 0.03))
            is_cal.append(cal == 1)
            used_hf.append(hf == 1)

    return {
        "z_col": z_col,
        "mu_col": mu_col,
        "err_col": err_col,
        "hf_col": hf_col,
        "z": np.asarray(z, dtype=float),
        "mu": np.asarray(mu, dtype=float),
        "sigma": np.asarray(sigma, dtype=float),
        "is_calibrator": np.asarray(is_cal, dtype=bool),
        "used_in_shoes_hf": np.asarray(used_hf, dtype=bool),
        "keep_mask": np.asarray(keep_mask, dtype=bool),
        "raw_row_count": len(rows),
    }


def load_covariance(path, n_total, keep_mask):
    text = path.read_text(errors="replace").split()
    values = np.asarray([float(x) for x in text], dtype=float)

    if len(values) == 1 + n_total * n_total and int(round(values[0])) == n_total:
        flat = values[1:]
    elif len(values) == n_total * n_total:
        flat = values
    else:
        first = int(round(values[0])) if len(values) > 0 else None

        if first is not None and len(values) == 1 + first * first:
            n_total = first
            flat = values[1:]

            if len(keep_mask) != n_total:
                raise RuntimeError(
                    f"Covariance dimension {n_total} does not match data rows {len(keep_mask)}."
                )
        else:
            raise RuntimeError(
                f"Could not infer covariance shape. tokens={len(values)}, expected around {n_total*n_total}."
            )

    cov = flat.reshape((n_total, n_total))
    cov = cov[np.ix_(keep_mask, keep_mask)]

    return cov


def stable_cholesky(cov):
    jitter = 0.0

    for attempt in range(10):
        try:
            if jitter == 0.0:
                return cho_factor(cov, lower=True, check_finite=False), jitter
            return cho_factor(cov + np.eye(cov.shape[0]) * jitter, lower=True, check_finite=False), jitter
        except Exception:
            if jitter == 0.0:
                jitter = 1.0e-12
            else:
                jitter *= 10.0

    raise RuntimeError("Could not Cholesky-factor covariance even with jitter.")


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


def offset_design_matrix(scheme, z, is_cal, used_hf):
    cols = []
    names = []

    if scheme == "global_offset":
        cols.append(np.ones(len(z), dtype=float))
        names.append("global")
    elif scheme == "calibrator_vs_noncal_offset":
        cols.append(is_cal.astype(float))
        cols.append((~is_cal).astype(float))
        names.extend(["calibrator", "noncal"])
    elif scheme == "ladder_three_offsets":
        cal = is_cal
        hf = used_hf & (~is_cal)
        rest = (~cal) & (~hf)
        cols.extend([cal.astype(float), hf.astype(float), rest.astype(float)])
        names.extend(["calibrator", "shoes_hf", "rest"])
    else:
        raise ValueError(f"Unknown offset scheme: {scheme}")

    X = np.vstack(cols).T
    return X, names


def make_offset_solver(c_factor, X, names):
    cinv_X = cho_solve(c_factor, X, check_finite=False)
    A = X.T @ cinv_X
    A_inv = np.linalg.inv(A)

    return {
        "X": X,
        "names": names,
        "A_inv": A_inv,
    }


def fit_offsets_fullcov(y, c_factor, solver):
    X = solver["X"]
    names = solver["names"]
    A_inv = solver["A_inv"]

    cinv_y = cho_solve(c_factor, y, check_finite=False)
    b = X.T @ cinv_y
    beta = A_inv @ b
    residual = y - X @ beta

    chi2 = float(residual.T @ cho_solve(c_factor, residual, check_finite=False))
    dof = int(len(y) - len(beta))

    offsets = {name: float(value) for name, value in zip(names, beta)}

    return offsets, residual, chi2, dof


def group_stats(name, mask, residual):
    if not np.any(mask):
        return {
            f"{name}_n": 0,
            f"{name}_mean_residual": float("nan"),
            f"{name}_rms_residual": float("nan"),
        }

    vals = residual[mask]

    return {
        f"{name}_n": int(np.sum(mask)),
        f"{name}_mean_residual": float(np.mean(vals)),
        f"{name}_rms_residual": float(np.sqrt(np.mean(vals * vals))),
    }


def is_offset_degenerate(offset_scheme):
    return offset_scheme in ["calibrator_vs_noncal_offset", "ladder_three_offsets"]


def evaluate_case(
    name,
    covariance_model,
    offset_scheme,
    params,
    z,
    mu_obs,
    is_cal,
    used_hf,
    mu_planck,
    c_factor,
    solvers,
    scheme_baseline_chi2,
):
    amplitude = float(params.get("amplitude", 0.0))
    z_center = float(params.get("z_center", 0.2))
    width = float(params.get("width", 0.05))

    gate_rows = np.ones_like(z, dtype=float)

    if amplitude > 0.0:
        gate_rows[is_cal] = window_gate(z[is_cal], amplitude, z_center, width)

    mu_model = mu_planck + 5.0 * np.log10(gate_rows)
    y = mu_obs - mu_model

    offsets, residual, chi2, dof = fit_offsets_fullcov(y, c_factor, solvers[offset_scheme])

    delta_chi2 = None if scheme_baseline_chi2 is None else float(chi2 - scheme_baseline_chi2)

    if np.any(is_cal):
        gate_for_calibrators = np.ones_like(z[is_cal], dtype=float)

        if amplitude > 0.0:
            gate_for_calibrators = window_gate(z[is_cal], amplitude, z_center, width)

        h0_mechanism = h0_equivalent_from_gate(gate_for_calibrators)
        h0_mean = float(np.mean(h0_mechanism))
        h0_rms_error = float(np.sqrt(np.mean((h0_mechanism - H0_SHOES_TARGET) ** 2)))
    else:
        h0_mean = H0_PLANCK_SIDE
        h0_rms_error = abs(H0_PLANCK_SIDE - H0_SHOES_TARGET)

    h0_chi2_proxy = float((h0_rms_error / SHOES_SIGMA) ** 2)

    applied_bao_max_drift = 0.0
    applied_cmb_drift = 0.0

    underlying_bao_gate = window_gate(BAO_Z, amplitude, z_center, width)
    underlying_cmb_gate = window_gate(CMB_Z, amplitude, z_center, width)

    underlying_bao_max_drift = float(np.max(np.abs(underlying_bao_gate - 1.0)))
    underlying_cmb_drift = float(abs(underlying_cmb_gate[0] - 1.0))

    degenerate = is_offset_degenerate(offset_scheme)

    if delta_chi2 is None:
        diagnostic = "baseline"
    elif degenerate and h0_rms_error <= 1.04 and delta_chi2 <= 5.0:
        diagnostic = "offset_absorbed_candidate_needs_cepheid_likelihood"
    elif h0_rms_error <= 1.04 and delta_chi2 <= 20.0:
        diagnostic = "nondegenerate_covariance_pass"
    elif h0_rms_error <= 1.50 and delta_chi2 <= 50.0:
        diagnostic = "near_pass_needs_cepheid_likelihood"
    elif h0_rms_error <= 1.50 and delta_chi2 > 50.0:
        diagnostic = "matches_H0_but_bad_covariance_shape"
    elif delta_chi2 <= 20.0:
        diagnostic = "SN_safe_but_weak_H0"
    else:
        diagnostic = "fails_calibration_covariance_pressure"

    sn_penalty = max(0.0, delta_chi2 or 0.0) / 20.0
    score = h0_chi2_proxy + sn_penalty

    cal = is_cal
    hf = used_hf
    lowz = (z <= 0.15) & (~is_cal)
    rest = (~is_cal) & (~used_hf)

    row = {
        "name": name,
        "covariance_model": covariance_model,
        "mode": "calibrator_only",
        "offset_scheme": offset_scheme,
        "params": params,
        "diagnostic": diagnostic,
        "offset_degenerate": degenerate,
        "total_covariance_component_score": float(score),
        "chi2_fullcov": chi2,
        "dof_fullcov": dof,
        "reduced_chi2_fullcov": float(chi2 / dof) if dof > 0 else float("nan"),
        "delta_chi2_vs_scheme_baseline": delta_chi2,
        "offsets": offsets,
        "h0_proxy_mean_for_calibrators": h0_mean,
        "h0_proxy_rms_error_for_calibrators": h0_rms_error,
        "h0_proxy_chi2_for_calibrators": h0_chi2_proxy,
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
        **group_stats("calibrator", cal, residual),
        **group_stats("shoes_hf", hf, residual),
        **group_stats("lowz_noncal", lowz, residual),
        **group_stats("rest", rest, residual),
    }

    row["_residual"] = residual
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


data_path = download_file(DATA_URL, "Pantheon_SH0ES_downloaded.dat")
statonly_path = download_file(COV_STATONLY_URL, "Pantheon_SH0ES_STATONLY_downloaded.cov")
stat_sys_path = download_file(COV_STAT_SYS_URL, "Pantheon_SH0ES_STAT_SYS_downloaded.cov")
calib_path = download_file(COV_CALIB_URL, "Pantheon_SH0ES_CALIB_downloaded.cov")

header, parsed_rows = parse_table(data_path)
sn = extract_arrays(header, parsed_rows)

z = sn["z"]
mu_obs = sn["mu"]
sigma_diag = sn["sigma"]
is_cal = sn["is_calibrator"]
used_hf = sn["used_in_shoes_hf"]

cov_statonly = load_covariance(statonly_path, sn["raw_row_count"], sn["keep_mask"])
cov_stat_sys = load_covariance(stat_sys_path, sn["raw_row_count"], sn["keep_mask"])
cov_calib = load_covariance(calib_path, sn["raw_row_count"], sn["keep_mask"])

cov_models = {
    "STATONLY": cov_statonly,
    "STATONLY_PLUS_CALIB": cov_statonly + cov_calib,
    "STAT_PLUS_SYS": cov_stat_sys,
}

dl_planck = luminosity_distance(z, H0_PLANCK_SIDE)
mu_planck = distance_modulus_from_dl_mpc(dl_planck)

all_rows = []
model_summaries = {}

for cov_name, cov in cov_models.items():
    c_factor, jitter_used = stable_cholesky(cov)

    solvers = {}

    for scheme in OFFSET_SCHEMES:
        X, names = offset_design_matrix(scheme, z, is_cal, used_hf)
        solvers[scheme] = make_offset_solver(c_factor, X, names)

    scheme_baselines = {}

    for scheme in OFFSET_SCHEMES:
        base = evaluate_case(
            f"baseline_{cov_name}_{scheme}",
            cov_name,
            scheme,
            {"amplitude": 0.0, "z_center": 0.2, "width": 0.05},
            z,
            mu_obs,
            is_cal,
            used_hf,
            mu_planck,
            c_factor,
            solvers,
            scheme_baseline_chi2=None,
        )

        scheme_baselines[scheme] = base["chi2_fullcov"]

    rows_this_cov = []

    for scheme in OFFSET_SCHEMES:
        rows_this_cov.append(
            evaluate_case(
                f"baseline_{cov_name}_{scheme}",
                cov_name,
                scheme,
                {"amplitude": 0.0, "z_center": 0.2, "width": 0.05},
                z,
                mu_obs,
                is_cal,
                used_hf,
                mu_planck,
                c_factor,
                solvers,
                scheme_baseline_chi2=scheme_baselines[scheme],
            )
        )

        rows_this_cov.append(
            evaluate_case(
                f"prior_best_{cov_name}_{scheme}",
                cov_name,
                scheme,
                {"amplitude": BEST_FULLCOV_V1_A, "z_center": BEST_FULLCOV_V1_ZC, "width": BEST_FULLCOV_V1_W},
                z,
                mu_obs,
                is_cal,
                used_hf,
                mu_planck,
                c_factor,
                solvers,
                scheme_baseline_chi2=scheme_baselines[scheme],
            )
        )

        for amplitude in np.linspace(0.0775, 0.0925, 7):
            for z_center in np.linspace(0.145, 0.205, 9):
                for width in np.linspace(0.055, 0.085, 7):
                    name = f"calibcov_{cov_name}_{scheme}_A{amplitude:.4f}_zc{z_center:.4f}_w{width:.4f}"
                    params = {
                        "amplitude": float(amplitude),
                        "z_center": float(z_center),
                        "width": float(width),
                    }

                    rows_this_cov.append(
                        evaluate_case(
                            name,
                            cov_name,
                            scheme,
                            params,
                            z,
                            mu_obs,
                            is_cal,
                            used_hf,
                            mu_planck,
                            c_factor,
                            solvers,
                            scheme_baseline_chi2=scheme_baselines[scheme],
                        )
                    )

    rows_this_cov_sorted = sorted(rows_this_cov, key=lambda row: row["total_covariance_component_score"])
    best_this_cov = rows_this_cov_sorted[0]

    all_rows.extend(rows_this_cov)

    model_summaries[cov_name] = {
        "covariance_shape": list(cov.shape),
        "cholesky_jitter_used": jitter_used,
        "scheme_baseline_chi2": scheme_baselines,
        "best_case": {k: v for k, v in best_this_cov.items() if not k.startswith("_")},
        "offset_absorbed_count": int(
            sum(row["diagnostic"] == "offset_absorbed_candidate_needs_cepheid_likelihood" for row in rows_this_cov)
        ),
        "nondegenerate_pass_count": int(
            sum(row["diagnostic"] == "nondegenerate_covariance_pass" for row in rows_this_cov)
        ),
        "near_pass_count": int(
            sum(row["diagnostic"] == "near_pass_needs_cepheid_likelihood" for row in rows_this_cov)
        ),
        "bad_shape_count": int(
            sum(row["diagnostic"] == "matches_H0_but_bad_covariance_shape" for row in rows_this_cov)
        ),
    }

rows_sorted = sorted(all_rows, key=lambda row: row["total_covariance_component_score"])
best = rows_sorted[0]

prior_rows = [
    row for row in all_rows
    if row["name"].startswith("prior_best_")
]

prior_comparison = []

for row in sorted(prior_rows, key=lambda r: (r["offset_scheme"], r["covariance_model"])):
    prior_comparison.append({k: v for k, v in row.items() if not k.startswith("_")})

nondegenerate_passes = [
    row for row in rows_sorted
    if row["diagnostic"] == "nondegenerate_covariance_pass"
]

offset_absorbed = [
    row for row in rows_sorted
    if row["diagnostic"] == "offset_absorbed_candidate_needs_cepheid_likelihood"
]

top_clean = []

for row in rows_sorted[:120]:
    top_clean.append({k: v for k, v in row.items() if not k.startswith("_")})

summary = {
    "boundary": "Covariance-component isolation screen only. Not Cepheid/SH0ES likelihood and not proof.",
    "data_url": DATA_URL,
    "covariance_urls": {
        "STATONLY": COV_STATONLY_URL,
        "STAT_PLUS_SYS": COV_STAT_SYS_URL,
        "CALIB": COV_CALIB_URL,
    },
    "downloaded_files": {
        "data": str(data_path),
        "statonly_cov": str(statonly_path),
        "stat_sys_cov": str(stat_sys_path),
        "calib_cov": str(calib_path),
    },
    "columns_used": {
        "z": sn["z_col"],
        "mu": sn["mu_col"],
        "sigma_diag_for_group_stats": sn["err_col"],
        "split_columns": ["IS_CALIBRATOR", sn["hf_col"]],
    },
    "row_counts": {
        "raw_rows": int(sn["raw_row_count"]),
        "rows_used": int(len(z)),
        "calibrator_rows": int(np.sum(is_cal)),
        "used_in_shoes_hf_rows": int(np.sum(used_hf)),
        "lowz_noncal_rows": int(np.sum((z <= 0.15) & (~is_cal))),
        "rest_rows": int(np.sum((~is_cal) & (~used_hf))),
    },
    "H0_planck_side": H0_PLANCK_SIDE,
    "H0_SH0ES_like_target": H0_SHOES_TARGET,
    "SHOES_sigma_used": SHOES_SIGMA,
    "model_summaries": model_summaries,
    "best_case_overall": {k: v for k, v in best.items() if not k.startswith("_")},
    "nondegenerate_pass_count_total": len(nondegenerate_passes),
    "offset_absorbed_count_total": len(offset_absorbed),
    "prior_best_comparison": prior_comparison,
    "top_120": top_clean,
}

(OUTDIR / "pantheon_calibration_covariance_gate_v1_summary.json").write_text(json.dumps(summary, indent=2))

header_out = [
    "rank",
    "name",
    "covariance_model",
    "mode",
    "offset_scheme",
    "diagnostic",
    "offset_degenerate",
    "total_covariance_component_score",
    "chi2_fullcov",
    "dof_fullcov",
    "reduced_chi2_fullcov",
    "delta_chi2_vs_scheme_baseline",
    "h0_proxy_mean_for_calibrators",
    "h0_proxy_rms_error_for_calibrators",
    "h0_proxy_chi2_for_calibrators",
    "underlying_window_bao_max_drift_from_1",
    "underlying_window_cmb_drift_from_1",
    "calibrator_n",
    "calibrator_mean_residual",
    "calibrator_rms_residual",
    "shoes_hf_n",
    "shoes_hf_mean_residual",
    "shoes_hf_rms_residual",
    "lowz_noncal_n",
    "lowz_noncal_mean_residual",
    "lowz_noncal_rms_residual",
    "rest_n",
    "rest_mean_residual",
    "rest_rms_residual",
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

with open(OUTDIR / "pantheon_calibration_covariance_gate_v1_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header_out)

    for rank, row in enumerate(rows_sorted[:3000], start=1):
        writer.writerow(
            [
                rank,
                row["name"],
                row["covariance_model"],
                row["mode"],
                row["offset_scheme"],
                row["diagnostic"],
                row["offset_degenerate"],
                row["total_covariance_component_score"],
                row["chi2_fullcov"],
                row["dof_fullcov"],
                row["reduced_chi2_fullcov"],
                row["delta_chi2_vs_scheme_baseline"],
                row["h0_proxy_mean_for_calibrators"],
                row["h0_proxy_rms_error_for_calibrators"],
                row["h0_proxy_chi2_for_calibrators"],
                row["underlying_window_bao_max_drift_from_1"],
                row["underlying_window_cmb_drift_from_1"],
                row["calibrator_n"],
                row["calibrator_mean_residual"],
                row["calibrator_rms_residual"],
                row["shoes_hf_n"],
                row["shoes_hf_mean_residual"],
                row["shoes_hf_rms_residual"],
                row["lowz_noncal_n"],
                row["lowz_noncal_mean_residual"],
                row["lowz_noncal_rms_residual"],
                row["rest_n"],
                row["rest_mean_residual"],
                row["rest_rms_residual"],
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

with open(OUTDIR / "pantheon_calibration_covariance_gate_v1_summary.txt", "w") as f:
    f.write("TAIRID Pantheon+SH0ES calibration-covariance gate v1\n\n")
    f.write("Boundary: covariance-component isolation screen only. Not Cepheid/SH0ES likelihood and not proof.\n\n")
    f.write(f"Raw rows: {sn['raw_row_count']}\n")
    f.write(f"Rows used: {len(z)}\n")
    f.write(f"Calibrator rows: {int(np.sum(is_cal))}\n")
    f.write(f"SH0ES Hubble-flow rows: {int(np.sum(used_hf))}\n\n")
    f.write("Model summaries:\n")
    f.write(json.dumps(model_summaries, indent=2) + "\n\n")
    f.write("Best case overall:\n")
    f.write(json.dumps({k: v for k, v in best.items() if not k.startswith("_")}, indent=2) + "\n\n")
    f.write("Prior best comparison:\n")
    f.write(json.dumps(prior_comparison, indent=2) + "\n\n")
    f.write("Interpretation guide:\n")
    f.write("- If STATONLY strongly penalizes the gate but STATONLY_PLUS_CALIB relaxes it, the gate aligns with calibration covariance.\n")
    f.write("- If all covariance models absorb the gate only when calibrator offsets are free, offset freedom dominates.\n")
    f.write("- If a non-degenerate pass appears under global_offset, that is stronger but still not final.\n")
    f.write("- If the best result remains offset-degenerate, the next required wall remains real Cepheid/anchor likelihood data.\n")

bins = np.array([0.0001, 0.005, 0.01, 0.023, 0.05, 0.10, 0.15, 0.25, 0.35, 0.60, 1.0, 1.5, 2.3])

with open(OUTDIR / "pantheon_calibration_covariance_gate_v1_binned_residuals.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case", "covariance_model", "offset_scheme", "z_lo", "z_hi", "n", "mean_residual_mag", "std_residual_mag"])

    baseline_for_best_cov = None

    for row in rows_sorted:
        if (
            row["covariance_model"] == best["covariance_model"]
            and row["offset_scheme"] == best["offset_scheme"]
            and row["name"].startswith("baseline_")
        ):
            baseline_for_best_cov = row
            break

    for row in [baseline_for_best_cov, best]:
        if row is None:
            continue

        for b in binned_residuals(z, row["_residual"], bins):
            writer.writerow([row["name"], row["covariance_model"], row["offset_scheme"], *b])

z_plot = np.unique(np.concatenate([np.linspace(0.0005, 0.5, 800), np.linspace(0.5, 2.5, 300)]))
z_plot.sort()

bp = best["params"]
best_gate_curve = window_gate(z_plot, bp["amplitude"], bp["z_center"], bp["width"])

plt.figure(figsize=(10, 6))
plt.plot(z_plot, np.ones_like(z_plot), label="No gate")
plt.plot(z_plot, window_gate(z_plot, BEST_FULLCOV_V1_A, BEST_FULLCOV_V1_ZC, BEST_FULLCOV_V1_W), label="Prior best fullcov v1 window")
plt.plot(z_plot, best_gate_curve, label="Best calibration-covariance v1 window")
plt.axvline(0.15, linewidth=1)
plt.axvline(0.35, linewidth=1)
plt.xlabel("z")
plt.ylabel("Underlying G(z)")
plt.title("Pantheon calibration covariance gate v1: underlying gate window")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "pantheon_calibration_covariance_gate_v1_gate_plot.png", dpi=160)
plt.close()

plt.figure(figsize=(10, 6))
labels = []
values = []

for cov_name in ["STATONLY", "STATONLY_PLUS_CALIB", "STAT_PLUS_SYS"]:
    for row in prior_comparison:
        if row["covariance_model"] == cov_name and row["offset_scheme"] == "global_offset":
            labels.append(cov_name)
            values.append(row["delta_chi2_vs_scheme_baseline"])

plt.bar(np.arange(len(labels)), values)
plt.xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
plt.ylabel("Delta chi2 for prior calibrator gate, global offset")
plt.title("Does calibration covariance relax the non-degenerate gate?")
plt.tight_layout()
plt.savefig(OUTDIR / "pantheon_calibration_covariance_gate_v1_delta_by_covariance.png", dpi=160)
plt.close()

print("")
print("TAIRID Pantheon+SH0ES calibration-covariance gate v1 complete.")
print("Created:")
print("  pantheon_calibration_covariance_gate_v1_outputs/pantheon_calibration_covariance_gate_v1_summary.json")
print("  pantheon_calibration_covariance_gate_v1_outputs/pantheon_calibration_covariance_gate_v1_summary.csv")
print("  pantheon_calibration_covariance_gate_v1_outputs/pantheon_calibration_covariance_gate_v1_summary.txt")
print("  pantheon_calibration_covariance_gate_v1_outputs/pantheon_calibration_covariance_gate_v1_binned_residuals.csv")
print("  pantheon_calibration_covariance_gate_v1_outputs/pantheon_calibration_covariance_gate_v1_gate_plot.png")
print("  pantheon_calibration_covariance_gate_v1_outputs/pantheon_calibration_covariance_gate_v1_delta_by_covariance.png")
print("")
print("Boundary:")
print("  This is not the Cepheid/SH0ES ladder likelihood.")
print("  This is a covariance-component isolation screen.")
