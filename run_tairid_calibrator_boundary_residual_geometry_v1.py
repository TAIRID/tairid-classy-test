#!/usr/bin/env python3
"""
TAIRID calibrator-boundary residual geometry v1.

Purpose:
The previous field-relative residual geometry screen found generic residual shape
under a simple global offset baseline, but the signal weakened when calibrator /
noncalibrator / ladder-group offset freedom was allowed.

This test asks the next sharper question:

Does the residual geometry live at the calibrator / distance-ladder boundary,
or is it only ordinary offset freedom?

Data:
Pantheon+SH0ES distance table and full STAT+SYS covariance.

Groups:
1. calibrator
2. SH0ES Hubble-flow noncalibrator
3. low-z noncalibrator not SH0ES-HF
4. rest

Models:
1. offset-only baselines:
   - global offset
   - calibrator vs noncalibrator offset
   - boundary four-group offsets

2. global field-shape additions:
   - frozen TAIRID-like q(z)
   - generic x and x^2 curvature

3. boundary/group field-shape additions:
   - frozen q(z) per ladder group
   - generic x and x^2 per ladder group
   - calibrator-boundary contrast shapes

Boundary:
This is not proof of TAIRID.
This is not a full cosmology model.
This is not the SH0ES Cepheid likelihood.
This is not BAO or Planck.
This is a full-covariance Pantheon+SH0ES calibrator-boundary residual screen.

Interpretation:
A positive result means:
    residual geometry survives ladder-boundary offset freedom and deserves a sharper
    compact SH0ES/Cepheid-ladder likelihood test.

A negative result means:
    the supernova residual lane does not carry the field-relative bridge once
    ordinary ladder-boundary freedom is included.

Either result is useful.
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
from scipy.stats import chi2


OUTDIR = Path("tairid_calibrator_boundary_residual_geometry_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")
    return path


DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
COV_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"

C_LIGHT = 299792.458

H0_REF = 66.89318
OMEGA_B = 0.0223700000
OMEGA_CDM = 0.1200000000
OMEGA_M_PHYSICAL = OMEGA_B + OMEGA_CDM

# Frozen joint gate v0.1 constants carried forward from the earlier cosmology lane.
FROZEN_A = 0.6580586049
FROZEN_ZT = 0.2224541370
FROZEN_P = 2.0

BINS = np.array([
    0.0001, 0.005, 0.010, 0.023, 0.050, 0.075, 0.100, 0.150,
    0.250, 0.350, 0.500, 0.750, 1.000, 1.500, 2.300
])


def download_file(url, local_name):
    path = OUTDIR / local_name
    with urllib.request.urlopen(url, timeout=300) as response:
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
    err_col = choose_column(
        header,
        ["MU_SH0ES_ERR_DIAG", "MUERR", "MU_ERR", "m_b_corr_err_DIAG", "MU_SH0ES_ERR"],
    )

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
    keep_mask = []

    for row in rows:
        keep = False

        try:
            zz = float(row[z_col])
            mm = float(row[mu_col])
            ss = float(row[err_col])
            cal = int(float(row["IS_CALIBRATOR"]))
            hf = int(float(row["USED_IN_SH0ES_HF"]))
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
                jitter = 1.0e-10
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

    grid = np.linspace(0.0, zmax, 10000)
    inv_e = 1.0 / e_z(grid, H0)
    integral = cumulative_trapezoid(inv_e, grid, initial=0.0)
    dc_grid = (C_LIGHT / H0) * integral
    dc = np.interp(z_values, grid, dc_grid)

    return (1.0 + z_values) * dc


def distance_modulus_from_dl_mpc(dl_mpc):
    dl_mpc = np.asarray(dl_mpc, dtype=float)
    return 5.0 * np.log10(dl_mpc) + 25.0


def frozen_gate(z):
    z = np.asarray(z, dtype=float)
    x = z / (z + FROZEN_ZT)
    return np.exp(-FROZEN_A * (x ** FROZEN_P))


def frozen_gate_mu_shape(z):
    g = frozen_gate(z)
    return 5.0 * np.log10(g)


def exclusive_groups(z, is_cal, used_hf):
    calibrator = is_cal
    shoes_hf = used_hf & (~is_cal)
    lowz_noncal = (z <= 0.15) & (~is_cal) & (~used_hf)
    rest = ~(calibrator | shoes_hf | lowz_noncal)

    groups = {
        "calibrator": calibrator,
        "shoes_hf_noncal": shoes_hf,
        "lowz_noncal_nonhf": lowz_noncal,
        "rest": rest,
    }

    return groups


def design_offsets(kind, z, is_cal, used_hf, groups):
    cols = []
    names = []

    if kind == "global_offset":
        cols.append(np.ones(len(z), dtype=float))
        names.append("global")

    elif kind == "calibrator_vs_noncal_offset":
        cols.append(is_cal.astype(float))
        cols.append((~is_cal).astype(float))
        names.extend(["calibrator", "noncal"])

    elif kind == "boundary_four_offsets":
        for name, mask in groups.items():
            cols.append(mask.astype(float))
            names.append(name)

    else:
        raise ValueError(f"Unknown offset kind: {kind}")

    return np.vstack(cols).T, names


def center_within_mask(values, mask):
    out = np.zeros_like(values, dtype=float)
    mask = mask.astype(bool)

    if np.sum(mask) == 0:
        return out

    vals = values[mask]
    out[mask] = vals - float(np.mean(vals))

    return out


def build_shape_columns(kind, z, groups):
    x = z / (z + FROZEN_ZT)
    x1 = x - np.mean(x)
    x2 = x ** 2 - np.mean(x ** 2)
    q = frozen_gate_mu_shape(z)
    q_centered = q - np.mean(q)

    cols = []
    names = []

    if kind == "global_frozen_shape":
        cols.append(q_centered)
        names.append("global_frozen_q")

    elif kind == "global_generic_curvature":
        cols.extend([x1, x2])
        names.extend(["global_x", "global_x2"])

    elif kind == "group_frozen_shapes":
        for group_name, mask in groups.items():
            if np.sum(mask) >= 5:
                cols.append(center_within_mask(q, mask))
                names.append(f"{group_name}_frozen_q")

    elif kind == "group_generic_curvatures":
        for group_name, mask in groups.items():
            if np.sum(mask) >= 5:
                cols.append(center_within_mask(x, mask))
                names.append(f"{group_name}_x")
                cols.append(center_within_mask(x ** 2, mask))
                names.append(f"{group_name}_x2")

    elif kind == "boundary_contrast_frozen":
        q_cal = center_within_mask(q, groups["calibrator"])
        q_hf = center_within_mask(q, groups["shoes_hf_noncal"])
        q_lowz = center_within_mask(q, groups["lowz_noncal_nonhf"])
        q_rest = center_within_mask(q, groups["rest"])

        cols.extend([
            q_cal - q_hf,
            q_cal - q_lowz,
            q_hf - q_lowz,
            q_lowz - q_rest,
        ])
        names.extend([
            "contrast_calibrator_minus_shoes_hf_q",
            "contrast_calibrator_minus_lowz_noncal_q",
            "contrast_shoes_hf_minus_lowz_noncal_q",
            "contrast_lowz_noncal_minus_rest_q",
        ])

    elif kind == "boundary_contrast_generic":
        x_cal = center_within_mask(x, groups["calibrator"])
        x_hf = center_within_mask(x, groups["shoes_hf_noncal"])
        x_lowz = center_within_mask(x, groups["lowz_noncal_nonhf"])
        x_rest = center_within_mask(x, groups["rest"])

        x2_cal = center_within_mask(x ** 2, groups["calibrator"])
        x2_hf = center_within_mask(x ** 2, groups["shoes_hf_noncal"])
        x2_lowz = center_within_mask(x ** 2, groups["lowz_noncal_nonhf"])
        x2_rest = center_within_mask(x ** 2, groups["rest"])

        cols.extend([
            x_cal - x_hf,
            x2_cal - x2_hf,
            x_cal - x_lowz,
            x2_cal - x2_lowz,
            x_hf - x_lowz,
            x2_hf - x2_lowz,
            x_lowz - x_rest,
            x2_lowz - x2_rest,
        ])
        names.extend([
            "contrast_calibrator_minus_shoes_hf_x",
            "contrast_calibrator_minus_shoes_hf_x2",
            "contrast_calibrator_minus_lowz_noncal_x",
            "contrast_calibrator_minus_lowz_noncal_x2",
            "contrast_shoes_hf_minus_lowz_noncal_x",
            "contrast_shoes_hf_minus_lowz_noncal_x2",
            "contrast_lowz_noncal_minus_rest_x",
            "contrast_lowz_noncal_minus_rest_x2",
        ])

    else:
        raise ValueError(f"Unknown shape kind: {kind}")

    return cols, names


def gls_fit(y, c_factor, X, names):
    cinv_X = cho_solve(c_factor, X, check_finite=False)
    A = X.T @ cinv_X
    A_inv = np.linalg.pinv(A, rcond=1.0e-12)

    cinv_y = cho_solve(c_factor, y, check_finite=False)
    b = X.T @ cinv_y

    beta = A_inv @ b
    residual = y - X @ beta

    chi2_val = float(residual.T @ cho_solve(c_factor, residual, check_finite=False))
    dof = int(len(y) - len(beta))

    params = {name: float(value) for name, value in zip(names, beta)}
    param_errors = {
        f"{name}_err": float(math.sqrt(max(A_inv[i, i], 0.0)))
        for i, name in enumerate(names)
    }

    return {
        "params": params,
        "param_errors": param_errors,
        "beta": beta,
        "cov_beta": A_inv,
        "residual": residual,
        "chi2": chi2_val,
        "dof": dof,
        "reduced_chi2": float(chi2_val / dof) if dof > 0 else float("nan"),
    }


def fit_nested(y, c_factor, base_X, base_names, shape_cols, shape_names):
    base = gls_fit(y, c_factor, base_X, base_names)

    if not shape_cols:
        return base, base, {
            "delta_chi2_improvement": 0.0,
            "delta_dof": 0,
            "p_value_chi2_improvement": 1.0,
        }

    X_shape = np.column_stack([base_X] + shape_cols)
    names_shape = list(base_names) + list(shape_names)

    shaped = gls_fit(y, c_factor, X_shape, names_shape)

    delta = float(base["chi2"] - shaped["chi2"])
    dof_delta = max(len(shape_names), 1)
    p_value = float(chi2.sf(max(delta, 0.0), dof_delta))

    return base, shaped, {
        "delta_chi2_improvement": delta,
        "delta_dof": dof_delta,
        "p_value_chi2_improvement": p_value,
    }


def group_stats(groups, z, residual, sigma):
    rows = []

    for name, mask in groups.items():
        mask = mask.astype(bool)
        n = int(np.sum(mask))

        if n == 0:
            rows.append({"group": name, "n": 0})
            continue

        vals = residual[mask]
        zz = z[mask]
        sig = sigma[mask]

        rows.append(
            {
                "group": name,
                "n": n,
                "z_min": float(np.min(zz)),
                "z_median": float(np.median(zz)),
                "z_max": float(np.max(zz)),
                "mean_residual": float(np.mean(vals)),
                "median_residual": float(np.median(vals)),
                "rms_residual": float(np.sqrt(np.mean(vals * vals))),
                "mean_abs_residual": float(np.mean(np.abs(vals))),
                "std_residual": float(np.std(vals)),
                "mean_sigma_diag": float(np.mean(sig)),
            }
        )

    return rows


def binned_group_residuals(groups, z, residual, sigma, bins):
    rows = []

    for group_name, gmask in groups.items():
        gmask = gmask.astype(bool)

        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            mask = gmask & (z >= lo) & (z < hi)
            n = int(np.sum(mask))

            if n == 0:
                rows.append(
                    {
                        "group": group_name,
                        "bin_index": i,
                        "z_lo": float(lo),
                        "z_hi": float(hi),
                        "z_mid": float((lo + hi) / 2.0),
                        "n": 0,
                        "mean_residual": float("nan"),
                        "median_residual": float("nan"),
                        "rms_residual": float("nan"),
                        "mean_abs_residual": float("nan"),
                        "std_residual": float("nan"),
                        "mean_sigma_diag": float("nan"),
                    }
                )
                continue

            vals = residual[mask]
            sig = sigma[mask]

            rows.append(
                {
                    "group": group_name,
                    "bin_index": i,
                    "z_lo": float(lo),
                    "z_hi": float(hi),
                    "z_mid": float(np.mean(z[mask])),
                    "n": n,
                    "mean_residual": float(np.mean(vals)),
                    "median_residual": float(np.median(vals)),
                    "rms_residual": float(np.sqrt(np.mean(vals * vals))),
                    "mean_abs_residual": float(np.mean(np.abs(vals))),
                    "std_residual": float(np.std(vals)),
                    "mean_sigma_diag": float(np.mean(sig)),
                }
            )

    return rows


def write_rows_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fields = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return path


def flatten_model_row(row):
    return {
        "baseline_kind": row["baseline_kind"],
        "shape_kind": row["shape_kind"],
        "baseline_chi2": row["baseline_chi2"],
        "baseline_dof": row["baseline_dof"],
        "baseline_reduced_chi2": row["baseline_reduced_chi2"],
        "shape_chi2": row["shape_chi2"],
        "shape_dof": row["shape_dof"],
        "shape_reduced_chi2": row["shape_reduced_chi2"],
        "delta_chi2_improvement": row["delta_chi2_improvement"],
        "delta_dof": row["delta_dof"],
        "p_value_chi2_improvement": row["p_value_chi2_improvement"],
        "baseline_params_json": json.dumps(row["baseline_params"], default=json_default),
        "shape_params_json": json.dumps(row["shape_params"], default=json_default),
        "shape_param_errors_json": json.dumps(row["shape_param_errors"], default=json_default),
    }


def plot_group_residuals(group_rows, title, path):
    groups = [r["group"] for r in group_rows]
    means = [r.get("mean_residual", np.nan) for r in group_rows]
    rms = [r.get("rms_residual", np.nan) for r in group_rows]

    x = np.arange(len(groups))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, means, width=width, label="mean residual")
    plt.bar(x + width / 2, rms, width=width, label="RMS residual")
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, groups, rotation=25, ha="right")
    plt.ylabel("mag")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_binned_group_residuals(bin_rows, title, path):
    plt.figure(figsize=(11, 6))

    for group in sorted(set(r["group"] for r in bin_rows)):
        rows = [r for r in bin_rows if r["group"] == group and r["n"] > 0 and np.isfinite(r["mean_residual"])]
        if not rows:
            continue

        zmid = [r["z_mid"] for r in rows]
        mean = [r["mean_residual"] for r in rows]
        plt.plot(zmid, mean, marker="o", linewidth=1.5, label=group)

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("z")
    plt.ylabel("binned mean residual mag")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def decide_status(model_rows):
    def get_row(baseline_kind, shape_kind):
        for r in model_rows:
            if r["baseline_kind"] == baseline_kind and r["shape_kind"] == shape_kind:
                return r
        return None

    boundary_group_curv = get_row("boundary_four_offsets", "group_generic_curvatures")
    boundary_group_frozen = get_row("boundary_four_offsets", "group_frozen_shapes")
    boundary_contrast = get_row("boundary_four_offsets", "boundary_contrast_generic")
    global_curv_global = get_row("global_offset", "global_generic_curvature")
    global_curv_boundary = get_row("boundary_four_offsets", "global_generic_curvature")

    def p(row):
        if row is None:
            return 1.0
        val = row.get("p_value_chi2_improvement")
        return float(val) if val is not None else 1.0

    def delta(row):
        if row is None:
            return 0.0
        val = row.get("delta_chi2_improvement")
        return float(val) if val is not None else 0.0

    boundary_survives = (
        p(boundary_group_curv) <= 0.01
        or p(boundary_group_frozen) <= 0.01
        or p(boundary_contrast) <= 0.01
    )

    boundary_directional = (
        p(boundary_group_curv) <= 0.05
        or p(boundary_group_frozen) <= 0.05
        or p(boundary_contrast) <= 0.05
    )

    global_only = p(global_curv_global) <= 0.01 and p(global_curv_boundary) > 0.05

    best_rows = {
        "global_curvature_under_global_offset": global_curv_global,
        "global_curvature_under_boundary_offsets": global_curv_boundary,
        "boundary_group_curvatures": boundary_group_curv,
        "boundary_group_frozen_shapes": boundary_group_frozen,
        "boundary_contrast_generic": boundary_contrast,
    }

    if boundary_survives:
        return (
            "calibrator_boundary_residual_geometry_survives_ladder_offsets",
            8,
            "Move to compact SH0ES/Cepheid ladder likelihood: test y, L, C directly if full covariance can be retrieved.",
            best_rows,
        )

    if boundary_directional:
        return (
            "calibrator_boundary_residual_geometry_directional_not_locked",
            7,
            "Inspect group residual plots and rerun with alternative bins and robust covariance before promotion.",
            best_rows,
        )

    if global_only:
        return (
            "global_residual_geometry_absorbed_by_ladder_boundary_offsets",
            7,
            "Do not promote SN residual geometry; next wall is direct SH0ES ladder matrix likelihood.",
            best_rows,
        )

    return (
        "no_calibrator_boundary_residual_geometry_after_ladder_offsets",
        6,
        "The Pantheon+SH0ES SN residual lane does not currently support a field-relative boundary claim.",
        best_rows,
    )


def main():
    print("")
    print("TAIRID calibrator-boundary residual geometry v1 starting.")
    print("Boundary: full-covariance Pantheon+SH0ES calibrator-boundary residual screen only.")
    print("")

    data_path = download_file(DATA_URL, "Pantheon_SH0ES_downloaded.dat")
    cov_path = download_file(COV_URL, "Pantheon_SH0ES_STAT_SYS_downloaded.cov")

    header, parsed_rows = parse_table(data_path)
    sn = extract_arrays(header, parsed_rows)

    z = sn["z"]
    mu_obs = sn["mu"]
    sigma = sn["sigma"]
    is_cal = sn["is_calibrator"]
    used_hf = sn["used_in_shoes_hf"]

    cov = load_covariance(cov_path, sn["raw_row_count"], sn["keep_mask"])
    c_factor, jitter_used = stable_cholesky(cov)

    dl_ref = luminosity_distance(z, H0_REF)
    mu_ref = distance_modulus_from_dl_mpc(dl_ref)
    y = mu_obs - mu_ref

    groups = exclusive_groups(z, is_cal, used_hf)

    baseline_kinds = [
        "global_offset",
        "calibrator_vs_noncal_offset",
        "boundary_four_offsets",
    ]

    shape_kinds = [
        "global_frozen_shape",
        "global_generic_curvature",
        "group_frozen_shapes",
        "group_generic_curvatures",
        "boundary_contrast_frozen",
        "boundary_contrast_generic",
    ]

    model_rows = []

    baseline_cache = {}

    for baseline_kind in baseline_kinds:
        X_base, base_names = design_offsets(baseline_kind, z, is_cal, used_hf, groups)
        baseline = gls_fit(y, c_factor, X_base, base_names)
        baseline_cache[baseline_kind] = {
            "X": X_base,
            "names": base_names,
            "fit": baseline,
        }

        for shape_kind in shape_kinds:
            shape_cols, shape_names = build_shape_columns(shape_kind, z, groups)
            base, shaped, stats = fit_nested(
                y,
                c_factor,
                X_base,
                base_names,
                shape_cols,
                shape_names,
            )

            model_rows.append(
                {
                    "baseline_kind": baseline_kind,
                    "shape_kind": shape_kind,
                    "baseline_chi2": base["chi2"],
                    "baseline_dof": base["dof"],
                    "baseline_reduced_chi2": base["reduced_chi2"],
                    "baseline_params": base["params"],
                    "shape_chi2": shaped["chi2"],
                    "shape_dof": shaped["dof"],
                    "shape_reduced_chi2": shaped["reduced_chi2"],
                    "shape_params": shaped["params"],
                    "shape_param_errors": shaped["param_errors"],
                    "delta_chi2_improvement": stats["delta_chi2_improvement"],
                    "delta_dof": stats["delta_dof"],
                    "p_value_chi2_improvement": stats["p_value_chi2_improvement"],
                }
            )

    # Primary residual diagnostics are after boundary_four_offsets,
    # because that is the nuisance freedom that absorbed the previous signal.
    boundary_residual = baseline_cache["boundary_four_offsets"]["fit"]["residual"]
    group_summary_rows = group_stats(groups, z, boundary_residual, sigma)
    group_bin_rows = binned_group_residuals(groups, z, boundary_residual, sigma, BINS)

    model_csv_rows = [flatten_model_row(r) for r in model_rows]
    write_rows_csv(OUTDIR / "calibrator_boundary_model_results_v1.csv", model_csv_rows)
    write_rows_csv(OUTDIR / "calibrator_boundary_group_summary_v1.csv", group_summary_rows)
    write_rows_csv(OUTDIR / "calibrator_boundary_group_binned_residuals_v1.csv", group_bin_rows)

    plot_group_residuals(
        group_summary_rows,
        "Pantheon+SH0ES residuals after boundary four-group offsets",
        OUTDIR / "calibrator_boundary_group_residual_summary_v1.png",
    )

    plot_binned_group_residuals(
        group_bin_rows,
        "Pantheon+SH0ES binned residuals by ladder-boundary group",
        OUTDIR / "calibrator_boundary_group_binned_residuals_v1.png",
    )

    final_status, readiness_score, next_wall, best_cases = decide_status(model_rows)

    group_counts = {name: int(np.sum(mask)) for name, mask in groups.items()}

    summary = {
        "test_name": "TAIRID calibrator-boundary residual geometry v1",
        "boundary": (
            "Full-covariance Pantheon+SH0ES calibrator-boundary residual screen only. "
            "Not proof of TAIRID, not full cosmology, not SH0ES Cepheid likelihood, not BAO, not Planck."
        ),
        "data_url": DATA_URL,
        "covariance_url": COV_URL,
        "downloaded_data_file": str(data_path),
        "downloaded_covariance_file": str(cov_path),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "columns_used": {
            "z": sn["z_col"],
            "mu": sn["mu_col"],
            "sigma_diag_for_plots": sn["err_col"],
            "split_columns": ["IS_CALIBRATOR", "USED_IN_SH0ES_HF"],
        },
        "row_counts": {
            "raw_rows": int(sn["raw_row_count"]),
            "rows_used": int(len(z)),
            "calibrator_rows_original": int(np.sum(is_cal)),
            "used_in_shoes_hf_rows_original": int(np.sum(used_hf)),
            "exclusive_group_counts": group_counts,
        },
        "covariance": {
            "shape_used": list(cov.shape),
            "cholesky_jitter_used": jitter_used,
        },
        "frozen_gate_constants": {
            "A": FROZEN_A,
            "z_t": FROZEN_ZT,
            "p": FROZEN_P,
            "G_at_z_0p01": float(frozen_gate(np.array([0.01]))[0]),
            "G_at_z_0p023": float(frozen_gate(np.array([0.023]))[0]),
            "G_at_z_0p05": float(frozen_gate(np.array([0.05]))[0]),
            "G_at_z_0p10": float(frozen_gate(np.array([0.10]))[0]),
            "G_at_z_0p15": float(frozen_gate(np.array([0.15]))[0]),
            "G_at_z_0p35": float(frozen_gate(np.array([0.35]))[0]),
            "G_at_z_0p61": float(frozen_gate(np.array([0.61]))[0]),
            "G_at_z_1p0": float(frozen_gate(np.array([1.0]))[0]),
            "G_at_z_2p33": float(frozen_gate(np.array([2.33]))[0]),
            "G_at_z_1100": float(frozen_gate(np.array([1100.0]))[0]),
            "warning": (
                "These raw G(z) values are not a validated multi-observable cosmology. "
                "Naive all-observable use would need BAO/CMB/time-dilation checks."
            ),
        },
        "model_results": model_rows,
        "group_summary_after_boundary_offsets": group_summary_rows,
        "best_cases": best_cases,
        "output_files": {
            "summary_json": str(OUTDIR / "calibrator_boundary_residual_geometry_v1_summary.json"),
            "summary_txt": str(OUTDIR / "calibrator_boundary_residual_geometry_v1_summary.txt"),
            "model_results_csv": str(OUTDIR / "calibrator_boundary_model_results_v1.csv"),
            "group_summary_csv": str(OUTDIR / "calibrator_boundary_group_summary_v1.csv"),
            "group_binned_residuals_csv": str(OUTDIR / "calibrator_boundary_group_binned_residuals_v1.csv"),
            "plots": [
                str(OUTDIR / "calibrator_boundary_group_residual_summary_v1.png"),
                str(OUTDIR / "calibrator_boundary_group_binned_residuals_v1.png"),
            ],
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "Group-specific or boundary-contrast residual geometry improves full-covariance fit even after "
                "boundary four-group offsets are included."
            ),
            "what_weakens_TAIRID_here": (
                "Residual geometry appears only under global offset or calibrator/noncal offset, then disappears "
                "under boundary four-group offsets."
            ),
            "bridge_rule": (
                "This test applies the bridge rule to the exact location where the previous residual geometry was absorbed: "
                "the calibrator / SH0ES-HF / low-z / rest ladder boundary."
            ),
            "truth_boundary": (
                "Even a positive result only motivates direct SH0ES ladder-matrix testing. It does not prove TAIRID."
            ),
        },
    }

    write_json(OUTDIR / "calibrator_boundary_residual_geometry_v1_summary.json", summary)

    with open(OUTDIR / "calibrator_boundary_residual_geometry_v1_summary.txt", "w") as f:
        f.write("TAIRID calibrator-boundary residual geometry v1\n\n")
        f.write("Boundary: full-covariance Pantheon+SH0ES calibrator-boundary residual screen only.\n")
        f.write("Not proof of TAIRID. Not full cosmology. Not SH0ES Cepheid likelihood. Not BAO. Not Planck.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")
        f.write("Why this test exists:\n")
        f.write("- The previous field-relative residual screen found generic residual geometry under global offset.\n")
        f.write("- That geometry weakened when ladder-boundary offset freedom was added.\n")
        f.write("- This pass tests whether geometry survives specifically at the calibrator / ladder boundary.\n\n")
        f.write("Exclusive group counts:\n")
        f.write(json.dumps(group_counts, indent=2, default=json_default) + "\n\n")
        f.write("Best cases:\n")
        f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
        f.write("Group summary after boundary offsets:\n")
        f.write(json.dumps(group_summary_rows, indent=2, default=json_default) + "\n\n")
        f.write("Truth boundary:\n")
        f.write("- A positive result only motivates direct SH0ES ladder-matrix testing.\n")
        f.write("- A negative result means this SN residual boundary lane does not currently carry the bridge.\n")
        f.write("- This cannot prove or disprove TAIRID as a whole.\n")

    print("")
    print("TAIRID calibrator-boundary residual geometry v1 complete.")
    print("Created:")
    print("  tairid_calibrator_boundary_residual_geometry_v1_outputs/calibrator_boundary_residual_geometry_v1_summary.json")
    print("  tairid_calibrator_boundary_residual_geometry_v1_outputs/calibrator_boundary_residual_geometry_v1_summary.txt")
    print("  tairid_calibrator_boundary_residual_geometry_v1_outputs/calibrator_boundary_model_results_v1.csv")
    print("  tairid_calibrator_boundary_residual_geometry_v1_outputs/calibrator_boundary_group_summary_v1.csv")
    print("  tairid_calibrator_boundary_residual_geometry_v1_outputs/calibrator_boundary_group_binned_residuals_v1.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is a full-covariance Pantheon+SH0ES calibrator-boundary residual screen.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
