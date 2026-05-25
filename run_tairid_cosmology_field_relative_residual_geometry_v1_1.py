#!/usr/bin/env python3
"""
TAIRID cosmology field-relative residual geometry v1.1.

Purpose:
The bridge rule says TAIRID should not be tested as a simple offset.
This test asks whether Pantheon+SH0ES full-covariance supernova residuals retain
field-relative structure after ordinary offset/calibration freedom is removed.

This v1.1 fixes the v1 script bug where write_json was called but not defined.

It tests:
1. Static offset baselines:
   - global offset
   - calibrator vs non-calibrator offset
   - ladder three offsets: calibrator / SH0ES-HF / rest

2. Frozen TAIRID-like residual shape after offsets:
   q(z) = 5 log10(G(z)), with
   G(z) = exp[-A (z / (z + z_t))^2]
   using frozen joint-gate v0.1 constants:
   A = 0.6580586049
   z_t = 0.2224541370
   p = 2

3. Generic field-curvature shape after offsets:
   x = z / (z + z_t)
   residual shape terms x and x^2

4. Binned field geometry:
   - residual curvature across redshift
   - reach / stable-bin fraction
   - breach fraction
   - sign-cycle / re-entry count
   - lag-1 residual autocorrelation
   - permutation null by shuffling residuals across redshift bins

Boundary:
This is not proof of TAIRID.
This is not a full cosmology model.
This is not Cepheid likelihood.
This is not BAO or Planck.
This is a full-covariance Pantheon+SH0ES residual-geometry screen.
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


OUTDIR = Path("tairid_cosmology_field_relative_residual_geometry_v1_1_outputs")
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

# Planck-side reference used as a fixed background surface.
# Offsets are fitted, so the absolute H0 zero point is not the main test.
H0_REF = 66.89318
OMEGA_B = 0.0223700000
OMEGA_CDM = 0.1200000000
OMEGA_M_PHYSICAL = OMEGA_B + OMEGA_CDM

# Frozen TAIRID joint gate v0.1 constants carried from prior test lane.
FROZEN_A = 0.6580586049
FROZEN_ZT = 0.2224541370
FROZEN_P = 2.0

RANDOM_SEED = 42
PERMUTATIONS = 2000

OFFSET_SCHEMES = [
    "global_offset",
    "calibrator_vs_noncal_offset",
    "ladder_three_offsets",
]

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


def fit_baseline_and_shape(y, c_factor, base_X, base_names, shape_cols, shape_names):
    base = gls_fit(y, c_factor, base_X, base_names)

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


def binned_residuals(z, residual, sigma, bins):
    rows = []

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (z >= lo) & (z < hi)
        n = int(np.sum(mask))

        if n == 0:
            rows.append(
                {
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


def sign_change_count(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) < 3:
        return 0

    signs = np.sign(vals)
    signs = signs[signs != 0]

    if len(signs) < 2:
        return 0

    return int(np.sum(signs[1:] != signs[:-1]))


def longest_true_run(flags):
    best = 0
    current = 0

    for flag in flags:
        if bool(flag):
            current += 1
            best = max(best, current)
        else:
            current = 0

    return int(best)


def safe_polyfit(x, y, deg):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < deg + 1:
        return [float("nan")] * (deg + 1)

    try:
        return [float(v) for v in np.polyfit(x, y, deg)]
    except Exception:
        return [float("nan")] * (deg + 1)


def lag1_autocorr(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) < 3:
        return float("nan")

    a = vals[:-1]
    b = vals[1:]

    if np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def field_geometry_metrics(bin_rows):
    usable = [r for r in bin_rows if r["n"] > 0 and np.isfinite(r["mean_residual"])]

    if len(usable) < 4:
        return {
            "usable_bins": len(usable),
            "status": "not_enough_bins",
        }

    z_mid = np.asarray([r["z_mid"] for r in usable], dtype=float)
    mean = np.asarray([r["mean_residual"] for r in usable], dtype=float)
    abs_mean = np.abs(mean)

    x = z_mid / (z_mid + FROZEN_ZT)

    quad = safe_polyfit(x, mean, 2)
    linear = safe_polyfit(x, mean, 1)

    median_abs = float(np.median(abs_mean))
    mad_abs = float(np.median(np.abs(abs_mean - median_abs)))
    W = median_abs + mad_abs

    breached = abs_mean > W
    stable = ~breached
    transitions = np.abs(np.diff(breached.astype(int))) if len(breached) > 1 else np.asarray([])

    return {
        "usable_bins": int(len(usable)),
        "status": "ok",
        "window_W_median_abs_plus_mad": W,
        "mean_abs_binned_residual": float(np.mean(abs_mean)),
        "max_abs_binned_residual": float(np.max(abs_mean)),
        "binned_residual_range": float(np.max(mean) - np.min(mean)),
        "quad_coeff_x2": quad[0],
        "quad_coeff_x": quad[1],
        "quad_coeff_const": quad[2],
        "linear_slope_x": linear[0],
        "linear_const": linear[1],
        "lag1_autocorr": lag1_autocorr(mean),
        "sign_change_count": sign_change_count(mean),
        "breach_fraction": float(np.mean(breached)),
        "breach_transition_count": int(np.sum(transitions)) if len(transitions) else 0,
        "reach_stable_fraction": float(np.mean(stable)),
        "reach_longest_stable_run_fraction": float(longest_true_run(stable) / max(len(stable), 1)),
        "reach_longest_breach_run_fraction": float(longest_true_run(breached) / max(len(breached), 1)),
    }


def permutation_geometry(z, residual, sigma, bins, observed_metrics):
    rng = np.random.default_rng(RANDOM_SEED)
    metrics = []

    for _ in range(PERMUTATIONS):
        perm = residual.copy()
        rng.shuffle(perm)
        bins_perm = binned_residuals(z, perm, sigma, bins)
        m = field_geometry_metrics(bins_perm)

        if m.get("status") == "ok":
            metrics.append(m)

    if not metrics:
        return {
            "status": "no_valid_permutations",
            "n_perm": 0,
        }

    def p_ge(key, obs_abs=False):
        obs = observed_metrics.get(key)
        vals = np.asarray([m.get(key, np.nan) for m in metrics], dtype=float)
        vals = vals[np.isfinite(vals)]

        if obs is None or not np.isfinite(obs) or len(vals) == 0:
            return None

        if obs_abs:
            return float((1.0 + np.sum(np.abs(vals) >= abs(obs))) / (1.0 + len(vals)))

        return float((1.0 + np.sum(vals >= obs)) / (1.0 + len(vals)))

    quad_vals = np.asarray([m["quad_coeff_x2"] for m in metrics if np.isfinite(m["quad_coeff_x2"])])
    lag_vals = np.asarray([m["lag1_autocorr"] for m in metrics if np.isfinite(m["lag1_autocorr"])])

    return {
        "status": "ok",
        "n_perm": int(len(metrics)),
        "p_abs_quad_coeff_x2": p_ge("quad_coeff_x2", obs_abs=True),
        "p_abs_linear_slope_x": p_ge("linear_slope_x", obs_abs=True),
        "p_abs_lag1_autocorr": p_ge("lag1_autocorr", obs_abs=True),
        "p_breach_fraction": p_ge("breach_fraction", obs_abs=False),
        "p_reach_longest_stable_run_fraction": p_ge("reach_longest_stable_run_fraction", obs_abs=False),
        "p_breach_transition_count": p_ge("breach_transition_count", obs_abs=False),
        "perm_quad_abs_95": float(np.percentile(np.abs(quad_vals), 95)) if len(quad_vals) else None,
        "perm_lag1_abs_95": float(np.percentile(np.abs(lag_vals), 95)) if len(lag_vals) else None,
    }


def write_binned_csv(path, scheme, bin_rows):
    exists = path.exists()

    with open(path, "a", newline="") as f:
        fieldnames = ["scheme"] + list(bin_rows[0].keys()) if bin_rows else ["scheme"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        for r in bin_rows:
            out = {"scheme": scheme}
            out.update(r)
            writer.writerow(out)


def plot_residuals(z, residual, bin_rows, title, path):
    plt.figure(figsize=(11, 6))
    plt.scatter(z, residual, s=5, alpha=0.20, label="SN residuals after offsets")

    usable = [r for r in bin_rows if r["n"] > 0 and np.isfinite(r["mean_residual"])]
    if usable:
        zm = [r["z_mid"] for r in usable]
        mr = [r["mean_residual"] for r in usable]
        plt.plot(zm, mr, marker="o", linewidth=2, label="binned mean residual")

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("z")
    plt.ylabel("residual magnitude")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_frozen_shape(path_prefix):
    zp = np.linspace(0.001, 2.3, 1000)
    g = frozen_gate(zp)
    q = frozen_gate_mu_shape(zp)

    gate_path = OUTDIR / f"{path_prefix}_gate.png"
    shape_path = OUTDIR / f"{path_prefix}_mu_shape.png"

    plt.figure(figsize=(11, 6))
    plt.plot(zp, g, label="Frozen G(z)")
    plt.xlabel("z")
    plt.ylabel("G(z)")
    plt.title("Frozen TAIRID-like gate window")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(gate_path, dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(zp, q, label="q(z)=5 log10 G(z)")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("z")
    plt.ylabel("magnitude-shape q(z)")
    plt.title("Frozen TAIRID-like residual shape")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(shape_path, dpi=160)
    plt.close()

    return [str(gate_path), str(shape_path)]


def decide_status(rows):
    best_frozen = min(rows, key=lambda r: r.get("frozen_shape_p", 1.0))
    best_generic = min(rows, key=lambda r: r.get("generic_curvature_p", 1.0))

    def best_perm_key(row):
        vals = [
            row.get("perm_p_abs_quad_coeff_x2"),
            row.get("perm_p_abs_lag1_autocorr"),
            row.get("perm_p_breach_fraction"),
        ]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else 1.0

    best_perm = min(rows, key=best_perm_key)

    frozen_pass = best_frozen.get("frozen_shape_p") is not None and best_frozen["frozen_shape_p"] <= 0.01
    generic_pass = best_generic.get("generic_curvature_p") is not None and best_generic["generic_curvature_p"] <= 0.01
    perm_pass = best_perm_key(best_perm) <= 0.01

    if frozen_pass and perm_pass:
        return (
            "field_relative_frozen_shape_and_residual_geometry_supported",
            8,
            "Move to multi-observable coherence test: SN time dilation, Tolman/distance duality, BAO, and CMB acoustic-scale prototype.",
            {
                "best_frozen": best_frozen,
                "best_generic": best_generic,
                "best_permutation_geometry": best_perm,
            },
        )

    if frozen_pass:
        return (
            "frozen_tairid_shape_survives_offsets_but_geometry_needs_crosscheck",
            7,
            "Inspect residual plots and rerun with alternative binning before multi-observable promotion.",
            {
                "best_frozen": best_frozen,
                "best_generic": best_generic,
                "best_permutation_geometry": best_perm,
            },
        )

    if generic_pass or perm_pass:
        return (
            "generic_field_residual_geometry_detected_not_frozen_gate_specific",
            7,
            "Treat as residual-geometry clue, not TAIRID gate support; test alternative cosmology/nuisance baselines.",
            {
                "best_frozen": best_frozen,
                "best_generic": best_generic,
                "best_permutation_geometry": best_perm,
            },
        )

    return (
        "no_field_relative_sn_residual_support_after_offset_freedom",
        6,
        "Do not promote SN residual geometry. Return to multi-observable physics constraints or richer ladder likelihood.",
        {
            "best_frozen": best_frozen,
            "best_generic": best_generic,
            "best_permutation_geometry": best_perm,
        },
    )


def main():
    print("")
    print("TAIRID cosmology field-relative residual geometry v1.1 starting.")
    print("Boundary: full-covariance Pantheon+SH0ES residual-geometry screen only.")
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

    all_rows = []
    all_bin_rows_path = OUTDIR / "cosmology_field_relative_binned_residuals_v1_1.csv"

    if all_bin_rows_path.exists():
        all_bin_rows_path.unlink()

    plots = []

    for scheme in OFFSET_SCHEMES:
        X, names = offset_design_matrix(scheme, z, is_cal, used_hf)

        baseline = gls_fit(y, c_factor, X, names)

        q = frozen_gate_mu_shape(z)
        q_centered = q - np.mean(q)

        base2, frozen_fit, frozen_stats = fit_baseline_and_shape(
            y,
            c_factor,
            X,
            names,
            [q_centered],
            ["frozen_gate_mu_shape_alpha"],
        )

        x = z / (z + FROZEN_ZT)
        x1 = x - np.mean(x)
        x2 = x ** 2 - np.mean(x ** 2)

        base3, generic_fit, generic_stats = fit_baseline_and_shape(
            y,
            c_factor,
            X,
            names,
            [x1, x2],
            ["generic_x_linear", "generic_x2_curvature"],
        )

        bin_rows = binned_residuals(z, baseline["residual"], sigma, BINS)
        write_binned_csv(all_bin_rows_path, scheme, bin_rows)

        geom = field_geometry_metrics(bin_rows)
        perm = permutation_geometry(z, baseline["residual"], sigma, BINS, geom)

        residual_plot = OUTDIR / f"cosmology_field_relative_residuals_{scheme}_v1_1.png"
        plot_residuals(
            z,
            baseline["residual"],
            bin_rows,
            f"Pantheon+SH0ES residual geometry after {scheme}",
            residual_plot,
        )
        plots.append(str(residual_plot))

        row = {
            "scheme": scheme,
            "baseline_chi2": baseline["chi2"],
            "baseline_dof": baseline["dof"],
            "baseline_reduced_chi2": baseline["reduced_chi2"],
            "baseline_offsets": baseline["params"],
            "baseline_offset_errors": baseline["param_errors"],

            "frozen_shape_chi2": frozen_fit["chi2"],
            "frozen_shape_dof": frozen_fit["dof"],
            "frozen_shape_delta_chi2_improvement": frozen_stats["delta_chi2_improvement"],
            "frozen_shape_delta_dof": frozen_stats["delta_dof"],
            "frozen_shape_p": frozen_stats["p_value_chi2_improvement"],
            "frozen_shape_params": frozen_fit["params"],
            "frozen_shape_param_errors": frozen_fit["param_errors"],

            "generic_curvature_chi2": generic_fit["chi2"],
            "generic_curvature_dof": generic_fit["dof"],
            "generic_curvature_delta_chi2_improvement": generic_stats["delta_chi2_improvement"],
            "generic_curvature_delta_dof": generic_stats["delta_dof"],
            "generic_curvature_p": generic_stats["p_value_chi2_improvement"],
            "generic_curvature_params": generic_fit["params"],
            "generic_curvature_param_errors": generic_fit["param_errors"],

            "geometry_metrics": geom,
            "permutation_geometry": perm,

            "usable_bins": geom.get("usable_bins"),
            "quad_coeff_x2": geom.get("quad_coeff_x2"),
            "lag1_autocorr": geom.get("lag1_autocorr"),
            "breach_fraction": geom.get("breach_fraction"),
            "reach_stable_fraction": geom.get("reach_stable_fraction"),
            "sign_change_count": geom.get("sign_change_count"),

            "perm_p_abs_quad_coeff_x2": perm.get("p_abs_quad_coeff_x2"),
            "perm_p_abs_lag1_autocorr": perm.get("p_abs_lag1_autocorr"),
            "perm_p_breach_fraction": perm.get("p_breach_fraction"),
            "perm_p_reach_longest_stable_run_fraction": perm.get("p_reach_longest_stable_run_fraction"),
            "perm_p_breach_transition_count": perm.get("p_breach_transition_count"),
        }

        all_rows.append(row)

    plots.extend(plot_frozen_shape("cosmology_field_relative_frozen_shape_v1_1"))

    final_status, readiness_score, next_wall, best_cases = decide_status(all_rows)

    summary = {
        "test_name": "TAIRID cosmology field-relative residual geometry v1.1",
        "boundary": (
            "Full-covariance Pantheon+SH0ES residual-geometry screen only. "
            "Not proof of TAIRID, not full cosmology, not Cepheid likelihood, not BAO, not Planck."
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
            "calibrator_rows": int(np.sum(is_cal)),
            "used_in_shoes_hf_rows": int(np.sum(used_hf)),
            "lowz_noncal_rows": int(np.sum((z <= 0.15) & (~is_cal))),
            "rest_rows": int(np.sum((~is_cal) & (~used_hf))),
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
        "offset_scheme_results": all_rows,
        "best_cases": best_cases,
        "output_files": {
            "summary_json": str(OUTDIR / "cosmology_field_relative_residual_geometry_v1_1_summary.json"),
            "summary_txt": str(OUTDIR / "cosmology_field_relative_residual_geometry_v1_1_summary.txt"),
            "scheme_results_csv": str(OUTDIR / "cosmology_field_relative_scheme_results_v1_1.csv"),
            "binned_residuals_csv": str(all_bin_rows_path),
            "plots": plots,
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "A frozen or generic field-relative residual shape survives offset freedom and residual geometry "
                "is unlikely under permutation."
            ),
            "what_weakens_TAIRID_here": (
                "Frozen shape and field geometry fail after ordinary offset freedom is removed."
            ),
            "bridge_rule": (
                "This test applies the bridge rule: stop asking only for a different mean or offset; ask whether "
                "the path through the observational field shows reach, curvature, breach, cycling, or residual structure."
            ),
            "truth_boundary": (
                "Even a positive result only motivates a multi-observable coherence test. It does not prove TAIRID."
            ),
        },
    }

    write_json(OUTDIR / "cosmology_field_relative_residual_geometry_v1_1_summary.json", summary)

    csv_path = OUTDIR / "cosmology_field_relative_scheme_results_v1_1.csv"
    with open(csv_path, "w", newline="") as f:
        fields = [
            "scheme",
            "baseline_chi2",
            "baseline_dof",
            "baseline_reduced_chi2",
            "frozen_shape_delta_chi2_improvement",
            "frozen_shape_p",
            "generic_curvature_delta_chi2_improvement",
            "generic_curvature_p",
            "usable_bins",
            "quad_coeff_x2",
            "lag1_autocorr",
            "breach_fraction",
            "reach_stable_fraction",
            "sign_change_count",
            "perm_p_abs_quad_coeff_x2",
            "perm_p_abs_lag1_autocorr",
            "perm_p_breach_fraction",
            "perm_p_reach_longest_stable_run_fraction",
            "perm_p_breach_transition_count",
            "baseline_offsets_json",
            "frozen_shape_params_json",
            "generic_curvature_params_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in all_rows:
            writer.writerow(
                {
                    "scheme": r["scheme"],
                    "baseline_chi2": r["baseline_chi2"],
                    "baseline_dof": r["baseline_dof"],
                    "baseline_reduced_chi2": r["baseline_reduced_chi2"],
                    "frozen_shape_delta_chi2_improvement": r["frozen_shape_delta_chi2_improvement"],
                    "frozen_shape_p": r["frozen_shape_p"],
                    "generic_curvature_delta_chi2_improvement": r["generic_curvature_delta_chi2_improvement"],
                    "generic_curvature_p": r["generic_curvature_p"],
                    "usable_bins": r["usable_bins"],
                    "quad_coeff_x2": r["quad_coeff_x2"],
                    "lag1_autocorr": r["lag1_autocorr"],
                    "breach_fraction": r["breach_fraction"],
                    "reach_stable_fraction": r["reach_stable_fraction"],
                    "sign_change_count": r["sign_change_count"],
                    "perm_p_abs_quad_coeff_x2": r["perm_p_abs_quad_coeff_x2"],
                    "perm_p_abs_lag1_autocorr": r["perm_p_abs_lag1_autocorr"],
                    "perm_p_breach_fraction": r["perm_p_breach_fraction"],
                    "perm_p_reach_longest_stable_run_fraction": r["perm_p_reach_longest_stable_run_fraction"],
                    "perm_p_breach_transition_count": r["perm_p_breach_transition_count"],
                    "baseline_offsets_json": json.dumps(r["baseline_offsets"], default=json_default),
                    "frozen_shape_params_json": json.dumps(r["frozen_shape_params"], default=json_default),
                    "generic_curvature_params_json": json.dumps(r["generic_curvature_params"], default=json_default),
                }
            )

    with open(OUTDIR / "cosmology_field_relative_residual_geometry_v1_1_summary.txt", "w") as f:
        f.write("TAIRID cosmology field-relative residual geometry v1.1\n\n")
        f.write("Boundary: full-covariance Pantheon+SH0ES residual-geometry screen only.\n")
        f.write("Not proof of TAIRID. Not full cosmology. Not Cepheid likelihood. Not BAO. Not Planck.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")
        f.write("Why this test exists:\n")
        f.write("- The bridge rule says TAIRID should not be tested as a simple offset.\n")
        f.write("- This pass removes offset/calibration freedom first.\n")
        f.write("- It then checks whether frozen or generic field-relative residual geometry remains.\n\n")
        f.write("Best cases:\n")
        f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
        f.write("Scheme results:\n")
        f.write(json.dumps(all_rows, indent=2, default=json_default) + "\n\n")
        f.write("Truth boundary:\n")
        f.write("- A positive result only motivates a multi-observable coherence test.\n")
        f.write("- A negative result means this SN residual lane does not currently carry the bridge.\n")
        f.write("- This cannot prove or disprove TAIRID as a whole.\n")

    print("")
    print("TAIRID cosmology field-relative residual geometry v1.1 complete.")
    print("Created:")
    print("  tairid_cosmology_field_relative_residual_geometry_v1_1_outputs/cosmology_field_relative_residual_geometry_v1_1_summary.json")
    print("  tairid_cosmology_field_relative_residual_geometry_v1_1_outputs/cosmology_field_relative_residual_geometry_v1_1_summary.txt")
    print("  tairid_cosmology_field_relative_residual_geometry_v1_1_outputs/cosmology_field_relative_scheme_results_v1_1.csv")
    print("  tairid_cosmology_field_relative_residual_geometry_v1_1_outputs/cosmology_field_relative_binned_residuals_v1_1.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is a full-covariance SN residual-geometry screen.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
