#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder boundary-gate likelihood v1.

Purpose:
The compact SH0ES parser verified that the public ladder system is usable:
    y = ally data vector
    L = alll equation/design matrix
    C = allc covariance matrix

This test now asks the first bounded likelihood question:

Can a compact-ladder boundary/topology gate improve the SH0ES ladder likelihood
beyond the baseline y = X beta model, without merely behaving like another
free offset or degenerate nuisance parameter?

Boundary:
This is not proof of TAIRID.
This is not a replacement cosmology.
This is not BAO, Planck, or a full multi-observable model.
This is a compact SH0ES ladder pressure test using y, L, and C directly.

Important limit:
The compact matrices do not expose human-readable row labels here. Therefore this
v1 test uses matrix-topology coordinates derived from L and the baseline GLS
solution, not hand-labeled Cepheid/SN/calibrator row types.

The next step after a positive result is row-label recovery / SH0ES code mapping.
"""

import csv
import json
import math
import re
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


OUTDIR = Path("tairid_shoes_compact_ladder_boundary_gate_likelihood_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

OWNER = "PantheonPlusSH0ES"
REPO = "DataRelease"
BRANCH = "main"

COMPACT_FILES = {
    "allc": "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits",
    "alll": "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits",
    "ally": "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits",
}

AUX_FILES = [
    "SH0ES_Data/lstsq_results.txt",
    "SH0ES_Data/MCMC_utils.py",
    "SH0ES_Data/run_mcmc.py",
]

FROZEN_A = 0.6580586049
FROZEN_ZT = 0.2224541370
FROZEN_P = 2.0
EPS = 1.0e-12


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")
    return path


def write_csv(path, rows):
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

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return path


def safe_name(path):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(path))[:220]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def fetch_url(url, timeout=900):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-SH0ES-compact-ladder-boundary-gate-v1",
            "Accept": "*/*",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        status = getattr(response, "status", None)

    return data, final_url, content_type, status


def is_git_lfs_pointer(data):
    if data is None:
        return False

    head = data[:200].decode("utf-8", errors="replace")
    return "version https://git-lfs.github.com/spec/v1" in head and "oid sha256:" in head


def parse_lfs_pointer(data):
    text = data.decode("utf-8", errors="replace")
    out = {
        "is_lfs_pointer": True,
        "raw_text": text,
    }

    oid = re.search(r"oid sha256:([a-fA-F0-9]+)", text)
    size = re.search(r"size\s+([0-9]+)", text)

    if oid:
        out["oid_sha256"] = oid.group(1)

    if size:
        out["declared_size"] = int(size.group(1))

    return out


def candidate_urls(repo_path):
    quoted = urllib.parse.quote(repo_path, safe="/._-+")

    return [
        {
            "kind": "raw_githubusercontent",
            "url": f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{quoted}",
        },
        {
            "kind": "media_githubusercontent",
            "url": f"https://media.githubusercontent.com/media/{OWNER}/{REPO}/{BRANCH}/{quoted}",
        },
        {
            "kind": "github_raw",
            "url": f"https://github.com/{OWNER}/{REPO}/raw/{BRANCH}/{quoted}",
        },
    ]


def download_repo_path(repo_path, label=None, required=False):
    label = label or safe_name(repo_path)
    local = DOWNLOAD_DIR / safe_name(repo_path)

    attempts = []
    pointer_info = None

    for cand in candidate_urls(repo_path):
        url = cand["url"]

        try:
            data, final_url, content_type, status = fetch_url(url)

            attempt = {
                "repo_path": repo_path,
                "label": label,
                "candidate_kind": cand["kind"],
                "url": url,
                "final_url": final_url,
                "http_status": status,
                "content_type": content_type,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }

            if is_git_lfs_pointer(data):
                p = parse_lfs_pointer(data)
                attempt.update(p)
                attempt["status"] = "git_lfs_pointer_not_payload"
                pointer_info = p
                attempts.append(attempt)
                continue

            local.write_bytes(data)

            attempt["status"] = "downloaded_real_payload"
            attempt["local_path"] = str(local)
            attempt["file_sha256"] = sha256_file(local)
            attempts.append(attempt)

            return {
                "repo_path": repo_path,
                "label": label,
                "status": "downloaded",
                "local_path": str(local),
                "bytes": local.stat().st_size,
                "sha256": sha256_file(local),
                "attempts": attempts,
                "pointer_info": pointer_info,
            }

        except urllib.error.HTTPError as exc:
            attempts.append(
                {
                    "repo_path": repo_path,
                    "label": label,
                    "candidate_kind": cand["kind"],
                    "url": url,
                    "status": "http_error",
                    "http_code": exc.code,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "repo_path": repo_path,
                    "label": label,
                    "candidate_kind": cand["kind"],
                    "url": url,
                    "status": "download_failed",
                    "error": str(exc),
                }
            )

    return {
        "repo_path": repo_path,
        "label": label,
        "status": "failed_required" if required else "failed_optional",
        "local_path": None,
        "attempts": attempts,
        "pointer_info": pointer_info,
    }


def extract_first_numeric_fits_array(path):
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        for idx, hdu in enumerate(hdul):
            data = hdu.data

            if data is None:
                continue

            try:
                if getattr(data.dtype, "fields", None):
                    fields = list(data.dtype.fields.keys())

                    if len(fields) == 1:
                        arr = np.asarray(data[fields[0]])
                    else:
                        numeric = []

                        for name in fields:
                            vals = np.asarray(data[name])

                            if np.issubdtype(vals.dtype, np.number):
                                numeric.append(vals)

                        if len(numeric) == 1:
                            arr = np.asarray(numeric[0])
                        elif len(numeric) > 1:
                            arr = np.column_stack(
                                [np.asarray(v).reshape(len(v), -1) for v in numeric]
                            )
                        else:
                            continue
                else:
                    arr = np.asarray(data)

                arr = np.squeeze(arr)

                if not np.issubdtype(arr.dtype, np.number):
                    continue

                arr = arr.astype(np.float64)

                if arr.size == 0:
                    continue

                return arr, {
                    "selected_hdu_index": idx,
                    "selected_hdu_name": hdu.name,
                    "selected_shape": list(arr.shape),
                    "selected_dtype": str(arr.dtype),
                }

            except Exception:
                continue

    raise RuntimeError(f"No numeric FITS array found in {path}")


def determine_design_orientation(L, y_length):
    L = np.asarray(L, dtype=np.float64)

    if L.ndim != 2:
        return None, {
            "status": "L_not_2d",
            "L_shape": list(L.shape),
            "y_length": int(y_length),
        }

    if L.shape[0] == y_length and L.shape[1] != y_length:
        return L, {
            "status": "ok",
            "orientation": "L_is_observation_by_parameter",
            "X_shape": list(L.shape),
        }

    if L.shape[1] == y_length and L.shape[0] != y_length:
        X = L.T

        return X, {
            "status": "ok",
            "orientation": "L_transposed_to_observation_by_parameter",
            "original_L_shape": list(L.shape),
            "X_shape": list(X.shape),
        }

    if L.shape[0] == y_length and L.shape[1] == y_length:
        return L, {
            "status": "ambiguous_square_L",
            "orientation": "using_L_as_observation_by_parameter",
            "X_shape": list(L.shape),
        }

    return None, {
        "status": "no_L_axis_matches_y_length",
        "L_shape": list(L.shape),
        "y_length": int(y_length),
    }


def stable_cholesky_cov(C):
    C = np.asarray(C, dtype=np.float64)
    C_sym = 0.5 * (C + C.T)
    diag = np.diag(C_sym)

    scale = float(np.median(diag[diag > 0])) if np.any(diag > 0) else 1.0
    jitter = 0.0
    attempts = []

    for i in range(12):
        try:
            if jitter == 0.0:
                cf = cho_factor(C_sym, lower=True, check_finite=False)
            else:
                cf = cho_factor(
                    C_sym + np.eye(C_sym.shape[0]) * jitter,
                    lower=True,
                    check_finite=False,
                )

            attempts.append(
                {
                    "attempt": i,
                    "jitter": jitter,
                    "status": "success",
                }
            )

            return cf, C_sym, jitter, attempts

        except Exception as exc:
            attempts.append(
                {
                    "attempt": i,
                    "jitter": jitter,
                    "status": "failed",
                    "error": str(exc),
                }
            )

            if jitter == 0.0:
                jitter = max(scale * 1.0e-12, 1.0e-14)
            else:
                jitter *= 10.0

    raise RuntimeError("Cholesky failed even with jitter.")


def gls_fit(y, X, c_factor):
    Cinv_y = cho_solve(c_factor, y, check_finite=False)
    Cinv_X = cho_solve(c_factor, X, check_finite=False)

    normal = X.T @ Cinv_X
    rhs = X.T @ Cinv_y
    normal_inv = np.linalg.pinv(normal, rcond=1.0e-12)
    beta = normal_inv @ rhs

    residual = y - X @ beta
    chi2_value = float(residual.T @ cho_solve(c_factor, residual, check_finite=False))
    dof = int(len(y) - X.shape[1])
    beta_err = np.sqrt(np.maximum(np.diag(normal_inv), 0.0))

    return {
        "beta": beta,
        "beta_err": beta_err,
        "residual": residual,
        "normal": normal,
        "normal_inv": normal_inv,
        "chi2": chi2_value,
        "dof": dof,
        "k": int(X.shape[1]),
        "aic": float(chi2_value + 2.0 * X.shape[1]),
        "bic": float(chi2_value + X.shape[1] * math.log(len(y))),
        "reduced_chi2": float(chi2_value / dof) if dof > 0 else float("nan"),
    }


def weighted_residualize_columns(Z, X, c_factor):
    """
    Return Z residualized against X using C^-1 geometry.
    """

    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    Cinv_X = cho_solve(c_factor, X, check_finite=False)
    Cinv_Z = cho_solve(c_factor, Z, check_finite=False)

    A = X.T @ Cinv_X
    A_inv = np.linalg.pinv(A, rcond=1.0e-12)
    coef = A_inv @ (X.T @ Cinv_Z)

    return Z - X @ coef


def c_norm(v, c_factor):
    v = np.asarray(v, dtype=np.float64).reshape(-1)

    return float(math.sqrt(max(v.T @ cho_solve(c_factor, v, check_finite=False), 0.0)))


def standardize_col(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = v - np.mean(v)
    sd = np.std(v)

    if not np.isfinite(sd) or sd <= 1.0e-14:
        return np.zeros_like(v)

    return v / sd


def rank01(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    order = np.argsort(v)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(v), dtype=np.float64)

    if len(v) <= 1:
        return np.zeros_like(v)

    return ranks / (len(v) - 1.0)


def frozen_topology_gate(coord01):
    x = np.asarray(coord01, dtype=np.float64)
    g = np.exp(-FROZEN_A * ((x / (x + FROZEN_ZT + EPS)) ** FROZEN_P))

    return standardize_col(g)


def row_leverage_approx(X, baseline_fit):
    # Euclidean design leverage proxy. This avoids forming an n x n hat matrix.
    # It is structure-based, not residual-derived.
    normal_inv = baseline_fit["normal_inv"]

    return np.einsum("ij,jk,ik->i", X, normal_inv, X)


def build_topology_columns(X, baseline_fit):
    absX = np.abs(X)
    row_sum = np.sum(absX, axis=1) + EPS
    nonzero_count = np.sum(absX > 1.0e-12, axis=1).astype(float)

    beta = baseline_fit["beta"]
    beta_err = baseline_fit["beta_err"]

    distance_like_params = (beta > 20.0) & (beta < 40.0)
    high_uncert_params = beta_err >= np.nanpercentile(beta_err, 75.0)
    low_uncert_params = beta_err <= np.nanpercentile(beta_err, 25.0)
    extreme_beta_params = np.abs(beta - np.nanmedian(beta)) >= np.nanpercentile(
        np.abs(beta - np.nanmedian(beta)),
        75.0,
    )

    if np.any(distance_like_params):
        distance_weight = absX[:, distance_like_params].sum(axis=1) / row_sum
    else:
        distance_weight = np.zeros(X.shape[0])

    if np.any(high_uncert_params):
        high_uncert_weight = absX[:, high_uncert_params].sum(axis=1) / row_sum
    else:
        high_uncert_weight = np.zeros(X.shape[0])

    if np.any(low_uncert_params):
        low_uncert_weight = absX[:, low_uncert_params].sum(axis=1) / row_sum
    else:
        low_uncert_weight = np.zeros(X.shape[0])

    if np.any(extreme_beta_params):
        extreme_beta_weight = absX[:, extreme_beta_params].sum(axis=1) / row_sum
    else:
        extreme_beta_weight = np.zeros(X.shape[0])

    leverage = row_leverage_approx(X, baseline_fit)
    leverage01 = rank01(leverage)
    density01 = rank01(nonzero_count)
    distance01 = rank01(distance_weight)
    highunc01 = rank01(high_uncert_weight)

    boundary_mix = (
        0.45 * leverage01
        + 0.25 * density01
        + 0.20 * highunc01
        + 0.10 * distance01
    )

    columns = {
        "control_row_order_linear": standardize_col(np.linspace(-1.0, 1.0, X.shape[0])),
        "matrix_leverage_axis": standardize_col(leverage),
        "matrix_nonzero_density_axis": standardize_col(nonzero_count),
        "distance_like_parameter_weight_axis": standardize_col(distance_weight),
        "high_uncertainty_parameter_weight_axis": standardize_col(high_uncert_weight),
        "low_vs_high_uncertainty_contrast_axis": standardize_col(
            low_uncert_weight - high_uncert_weight
        ),
        "extreme_beta_parameter_weight_axis": standardize_col(extreme_beta_weight),
        "low_rank_boundary_topology_axis": standardize_col(boundary_mix),
        "frozen_topology_boundary_gate": frozen_topology_gate(rank01(boundary_mix)),
    }

    metadata = {
        "parameter_count": int(X.shape[1]),
        "distance_like_parameter_indices": [
            int(i) for i in np.where(distance_like_params)[0]
        ],
        "high_uncertainty_parameter_indices": [
            int(i) for i in np.where(high_uncert_params)[0]
        ],
        "low_uncertainty_parameter_indices": [
            int(i) for i in np.where(low_uncert_params)[0]
        ],
        "extreme_beta_parameter_indices": [
            int(i) for i in np.where(extreme_beta_params)[0]
        ],
        "leverage_min": float(np.min(leverage)),
        "leverage_max": float(np.max(leverage)),
        "nonzero_count_min": float(np.min(nonzero_count)),
        "nonzero_count_max": float(np.max(nonzero_count)),
    }

    return columns, metadata


def nested_model_test(y, X_base, c_factor, baseline_fit, col_map, model_name, column_names):
    Z_raw = np.column_stack([col_map[name] for name in column_names])
    Z_resid = weighted_residualize_columns(Z_raw, X_base, c_factor)

    resid_norms = [c_norm(Z_resid[:, i], c_factor) for i in range(Z_resid.shape[1])]
    raw_norms = [c_norm(Z_raw[:, i], c_factor) for i in range(Z_raw.shape[1])]

    nondegenerate_ratios = [
        float(r / max(a, EPS)) for r, a in zip(resid_norms, raw_norms)
    ]

    X_new = np.column_stack([X_base, Z_raw])
    fit = gls_fit(y, X_new, c_factor)

    delta_chi2 = float(baseline_fit["chi2"] - fit["chi2"])
    delta_dof = int(len(column_names))
    p_value = float(chi2.sf(max(delta_chi2, 0.0), delta_dof))

    delta_aic = float(fit["aic"] - baseline_fit["aic"])
    delta_bic = float(fit["bic"] - baseline_fit["bic"])

    beta0 = baseline_fit["beta"]
    beta1 = fit["beta"][: len(beta0)]
    beta0_err = baseline_fit["beta_err"]
    norm_shift = np.abs(beta1 - beta0) / np.maximum(beta0_err, EPS)

    added_beta = fit["beta"][-delta_dof:]
    added_err = fit["beta_err"][-delta_dof:]
    added_z = added_beta / np.maximum(added_err, EPS)

    return {
        "model_name": model_name,
        "column_names": list(column_names),
        "added_column_count": delta_dof,
        "baseline_chi2": baseline_fit["chi2"],
        "model_chi2": fit["chi2"],
        "delta_chi2_improvement": delta_chi2,
        "p_value_chi2_improvement": p_value,
        "baseline_aic": baseline_fit["aic"],
        "model_aic": fit["aic"],
        "delta_aic_model_minus_baseline": delta_aic,
        "baseline_bic": baseline_fit["bic"],
        "model_bic": fit["bic"],
        "delta_bic_model_minus_baseline": delta_bic,
        "model_dof": fit["dof"],
        "model_reduced_chi2": fit["reduced_chi2"],
        "nondegenerate_ratios": nondegenerate_ratios,
        "min_nondegenerate_ratio": float(np.min(nondegenerate_ratios)),
        "added_beta": [float(v) for v in added_beta],
        "added_beta_err": [float(v) for v in added_err],
        "added_beta_z": [float(v) for v in added_z],
        "max_abs_added_beta_z": float(np.max(np.abs(added_z))) if len(added_z) else None,
        "max_baseline_param_shift_sigma": float(np.max(norm_shift)) if len(norm_shift) else None,
        "median_baseline_param_shift_sigma": float(np.median(norm_shift)) if len(norm_shift) else None,
    }, fit


def permutation_column_test(y, X_base, c_factor, baseline_fit, col, observed_delta, repeats=500):
    rng = np.random.default_rng(42)
    deltas = []
    col = np.asarray(col, dtype=np.float64).reshape(-1)

    for _ in range(repeats):
        perm = col.copy()
        rng.shuffle(perm)

        X_new = np.column_stack([X_base, standardize_col(perm)])
        fit = gls_fit(y, X_new, c_factor)
        deltas.append(float(baseline_fit["chi2"] - fit["chi2"]))

    deltas = np.asarray(deltas, dtype=np.float64)

    return {
        "n_perm": int(repeats),
        "perm_delta_mean": float(np.mean(deltas)),
        "perm_delta_std": float(np.std(deltas)),
        "perm_delta_95": float(np.percentile(deltas, 95)),
        "perm_delta_99": float(np.percentile(deltas, 99)),
        "p_perm_delta_ge_observed": float(
            (1.0 + np.sum(deltas >= observed_delta)) / (1.0 + len(deltas))
        ),
    }


def plot_column(col_map, name, path):
    plt.figure(figsize=(11, 5))
    plt.plot(col_map[name], linewidth=1)
    plt.xlabel("compact ladder observation index")
    plt.ylabel("standardized column value")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_previews(y, X, baseline_fit, col_map, model_rows):
    rows = []
    residual = baseline_fit["residual"]

    for i in range(min(len(y), 500)):
        rows.append(
            {
                "observation_index": i,
                "y": float(y[i]),
                "baseline_residual": float(residual[i]),
            }
        )

    write_csv(OUTDIR / "compact_boundary_gate_residual_preview_v1.csv", rows)

    beta_rows = []
    for i, (b, e) in enumerate(zip(baseline_fit["beta"], baseline_fit["beta_err"])):
        beta_rows.append(
            {
                "parameter_index": i,
                "baseline_beta": float(b),
                "baseline_beta_err": float(e),
            }
        )

    write_csv(OUTDIR / "compact_boundary_gate_baseline_beta_v1.csv", beta_rows)

    col_rows = []
    for name, col in col_map.items():
        vals = np.asarray(col)
        col_rows.append(
            {
                "column": name,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "preview_first_12": json.dumps([float(v) for v in vals[:12]]),
            }
        )

    write_csv(OUTDIR / "compact_boundary_gate_candidate_column_stats_v1.csv", col_rows)

    for name in [
        "low_rank_boundary_topology_axis",
        "frozen_topology_boundary_gate",
        "matrix_leverage_axis",
    ]:
        if name in col_map:
            plot_column(col_map, name, OUTDIR / f"candidate_{name}_v1.png")

    plt.figure(figsize=(10, 6))
    plt.hist(residual, bins=80)
    plt.xlabel("baseline compact GLS residual")
    plt.ylabel("count")
    plt.title("SH0ES compact ladder baseline residuals")
    plt.tight_layout()
    plt.savefig(OUTDIR / "compact_boundary_gate_baseline_residual_hist_v1.png", dpi=160)
    plt.close()

    labels = [r["model_name"] for r in model_rows]
    vals = [r["delta_chi2_improvement"] for r in model_rows]

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(labels)), vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Delta chi2 improvement")
    plt.title("Compact ladder boundary-gate candidate improvements")
    plt.tight_layout()
    plt.savefig(OUTDIR / "compact_boundary_gate_delta_chi2_v1.png", dpi=160)
    plt.close()


def decide_status(model_rows):
    by_name = {r["model_name"]: r for r in model_rows}

    frozen = by_name.get("frozen_topology_boundary_gate")
    lowrank = by_name.get("low_rank_boundary_topology_axis")
    pair = by_name.get("low_rank_plus_frozen_topology_gate")
    control = by_name.get("control_row_order_linear")

    def strong(row):
        return bool(
            row
            and row.get("p_value_chi2_improvement", 1.0) <= 0.01
            and row.get("delta_bic_model_minus_baseline", 999.0) <= -6.0
            and row.get("min_nondegenerate_ratio", 0.0) >= 0.05
        )

    def directional(row):
        return bool(
            row
            and row.get("p_value_chi2_improvement", 1.0) <= 0.05
            and row.get("delta_aic_model_minus_baseline", 999.0) < 0.0
            and row.get("min_nondegenerate_ratio", 0.0) >= 0.02
        )

    control_strong = strong(control)
    topology_strong = strong(frozen) or strong(lowrank) or strong(pair)
    topology_directional = directional(frozen) or directional(lowrank) or directional(pair)

    best = min(
        model_rows,
        key=lambda r: r.get("delta_bic_model_minus_baseline", 999.0),
    ) if model_rows else None

    if topology_strong and not control_strong:
        return (
            "compact_ladder_boundary_gate_supported_not_offset_only",
            8,
            "Inspect row-label recovery and build v2 with explicit Cepheid/SN/calibrator row classes before any theory promotion.",
            {
                "best_bic_model": best,
                "frozen": frozen,
                "lowrank": lowrank,
                "pair": pair,
                "control": control,
            },
        )

    if topology_strong and control_strong:
        return (
            "compact_ladder_extra_column_support_confounded_by_row_order_control",
            7,
            "Do not promote. Row/order control also improves; recover row labels and rerun with shuffled/order-robust controls.",
            {
                "best_bic_model": best,
                "frozen": frozen,
                "lowrank": lowrank,
                "pair": pair,
                "control": control,
            },
        )

    if topology_directional:
        return (
            "compact_ladder_boundary_gate_directional_not_locked",
            7,
            "Treat as directional only; recover row labels and test explicit calibrator-boundary vectors.",
            {
                "best_bic_model": best,
                "frozen": frozen,
                "lowrank": lowrank,
                "pair": pair,
                "control": control,
            },
        )

    return (
        "compact_ladder_boundary_gate_not_supported_in_v1_topology_screen",
        6,
        "This topology-only v1 does not support a boundary gate; next move is row-label recovery, not theory promotion.",
        {
            "best_bic_model": best,
            "frozen": frozen,
            "lowrank": lowrank,
            "pair": pair,
            "control": control,
        },
    )


def main():
    print("")
    print("TAIRID SH0ES compact ladder boundary-gate likelihood v1 starting.")
    print("Boundary: compact SH0ES ladder pressure test only; not proof.")
    print("")

    downloads = {}
    ledger = []

    for label, repo_path in COMPACT_FILES.items():
        res = download_repo_path(repo_path, label=label, required=True)
        downloads[label] = res

        ledger.append(
            {
                "label": label,
                "repo_path": repo_path,
                "status": res.get("status"),
                "local_path": res.get("local_path"),
                "bytes": res.get("bytes"),
                "sha256": res.get("sha256"),
                "pointer_declared_size": (res.get("pointer_info") or {}).get("declared_size"),
                "pointer_oid_sha256": (res.get("pointer_info") or {}).get("oid_sha256"),
                "attempt_count": len(res.get("attempts", [])),
            }
        )

    aux_downloads = []

    for repo_path in AUX_FILES:
        res = download_repo_path(repo_path, label=safe_name(repo_path), required=False)
        aux_downloads.append(res)

        ledger.append(
            {
                "label": safe_name(repo_path),
                "repo_path": repo_path,
                "status": res.get("status"),
                "local_path": res.get("local_path"),
                "bytes": res.get("bytes"),
                "sha256": res.get("sha256"),
                "attempt_count": len(res.get("attempts", [])),
            }
        )

    write_csv(OUTDIR / "compact_boundary_gate_download_ledger_v1.csv", ledger)
    write_json(
        OUTDIR / "compact_boundary_gate_download_attempts_v1.json",
        {
            "compact": downloads,
            "auxiliary": aux_downloads,
        },
    )

    parsed = {}
    parse_meta = {}
    parse_errors = []

    for label in ["allc", "alll", "ally"]:
        res = downloads.get(label, {})

        if res.get("status") != "downloaded":
            parse_errors.append(
                {
                    "label": label,
                    "status": "not_downloaded",
                    "download_status": res.get("status"),
                }
            )
            continue

        try:
            arr, meta = extract_first_numeric_fits_array(Path(res["local_path"]))
            parsed[label] = arr
            parse_meta[label] = meta

        except Exception as exc:
            parse_errors.append(
                {
                    "label": label,
                    "status": "parse_failed",
                    "error": str(exc),
                }
            )

    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]):
        summary = {
            "test_name": "TAIRID SH0ES compact ladder boundary-gate likelihood v1",
            "final_status": "compact_ladder_boundary_gate_download_or_parse_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix matrix retrieval/parser before likelihood testing.",
            "parse_errors": parse_errors,
            "downloads": downloads,
        }

        write_json(OUTDIR / "compact_ladder_boundary_gate_likelihood_v1_summary.json", summary)
        print("Parse failed; see summary JSON.")
        return

    C = np.asarray(parsed["allc"], dtype=np.float64)
    L = np.asarray(parsed["alll"], dtype=np.float64)
    y = np.asarray(parsed["ally"], dtype=np.float64).reshape(-1)

    X, orient = determine_design_orientation(L, len(y))

    if X is None:
        raise RuntimeError(f"Could not orient L relative to y: {orient}")

    if C.ndim != 2 or C.shape[0] != len(y) or C.shape[1] != len(y):
        raise RuntimeError(f"C shape {C.shape} does not match y length {len(y)}")

    c_factor, C_sym, jitter, chol_attempts = stable_cholesky_cov(C)
    baseline = gls_fit(y, X, c_factor)

    col_map, topology_meta = build_topology_columns(X, baseline)

    candidate_models = [
        ("control_row_order_linear", ["control_row_order_linear"]),
        ("matrix_leverage_axis", ["matrix_leverage_axis"]),
        ("matrix_nonzero_density_axis", ["matrix_nonzero_density_axis"]),
        ("distance_like_parameter_weight_axis", ["distance_like_parameter_weight_axis"]),
        ("high_uncertainty_parameter_weight_axis", ["high_uncertainty_parameter_weight_axis"]),
        ("low_vs_high_uncertainty_contrast_axis", ["low_vs_high_uncertainty_contrast_axis"]),
        ("extreme_beta_parameter_weight_axis", ["extreme_beta_parameter_weight_axis"]),
        ("low_rank_boundary_topology_axis", ["low_rank_boundary_topology_axis"]),
        ("frozen_topology_boundary_gate", ["frozen_topology_boundary_gate"]),
        (
            "low_rank_plus_frozen_topology_gate",
            ["low_rank_boundary_topology_axis", "frozen_topology_boundary_gate"],
        ),
    ]

    model_rows = []
    model_fits = {}

    for model_name, cols in candidate_models:
        row, fit = nested_model_test(y, X, c_factor, baseline, col_map, model_name, cols)
        model_rows.append(row)
        model_fits[model_name] = fit

    # Permutation checks only for the two main topology-gate candidates and the control.
    perm_rows = []

    for model_name in [
        "control_row_order_linear",
        "low_rank_boundary_topology_axis",
        "frozen_topology_boundary_gate",
    ]:
        observed = next(r for r in model_rows if r["model_name"] == model_name)

        perm = permutation_column_test(
            y,
            X,
            c_factor,
            baseline,
            col_map[model_name],
            observed["delta_chi2_improvement"],
            repeats=500,
        )

        perm_rows.append({"model_name": model_name, **perm})
        observed.update({f"perm_{k}": v for k, v in perm.items()})

    final_status, readiness_score, next_wall, best_cases = decide_status(model_rows)

    write_csv(OUTDIR / "compact_ladder_boundary_gate_model_results_v1.csv", model_rows)
    write_csv(OUTDIR / "compact_ladder_boundary_gate_permutation_checks_v1.csv", perm_rows)
    write_json(OUTDIR / "compact_ladder_boundary_gate_parse_meta_v1.json", parse_meta)
    write_json(OUTDIR / "compact_ladder_boundary_gate_topology_metadata_v1.json", topology_meta)

    write_previews(y, X, baseline, col_map, model_rows)

    summary = {
        "test_name": "TAIRID SH0ES compact ladder boundary-gate likelihood v1",
        "boundary": (
            "Compact SH0ES ladder pressure test only. Not proof of TAIRID, not a replacement cosmology, "
            "not BAO, not Planck, and not a full multi-observable model."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "matrix_shapes": {
            "allc_C": list(C.shape),
            "alll_L": list(L.shape),
            "ally_y": list(np.asarray(parsed["ally"]).shape),
            "y_flat": list(y.shape),
            "X_design": list(X.shape),
            "L_orientation": orient,
        },
        "covariance": {
            "cholesky_jitter": jitter,
            "cholesky_attempts": chol_attempts,
            "diag_min": float(np.min(np.diag(C_sym))),
            "diag_max": float(np.max(np.diag(C_sym))),
            "diag_nonpositive_count": int(np.sum(np.diag(C_sym) <= 0)),
        },
        "baseline_gls": {
            "chi2": baseline["chi2"],
            "dof": baseline["dof"],
            "reduced_chi2": baseline["reduced_chi2"],
            "aic": baseline["aic"],
            "bic": baseline["bic"],
            "residual_mean": float(np.mean(baseline["residual"])),
            "residual_std": float(np.std(baseline["residual"])),
            "residual_rms": float(np.sqrt(np.mean(baseline["residual"] ** 2))),
            "parameter_count": int(len(baseline["beta"])),
            "normal_condition_estimate": float(np.linalg.cond(baseline["normal"])),
            "beta_preview_first_20": [float(v) for v in baseline["beta"][:20]],
            "beta_err_preview_first_20": [float(v) for v in baseline["beta_err"][:20]],
        },
        "frozen_topology_gate_constants": {
            "A": FROZEN_A,
            "z_t": FROZEN_ZT,
            "p": FROZEN_P,
            "note": "Applied to matrix-topology coordinate, not cosmological redshift, because compact row labels are not exposed yet.",
        },
        "topology_metadata": topology_meta,
        "model_results": model_rows,
        "permutation_checks": perm_rows,
        "best_cases": best_cases,
        "output_files": {
            "summary_json": str(OUTDIR / "compact_ladder_boundary_gate_likelihood_v1_summary.json"),
            "summary_txt": str(OUTDIR / "compact_ladder_boundary_gate_likelihood_v1_summary.txt"),
            "model_results_csv": str(OUTDIR / "compact_ladder_boundary_gate_model_results_v1.csv"),
            "permutation_checks_csv": str(OUTDIR / "compact_ladder_boundary_gate_permutation_checks_v1.csv"),
            "baseline_beta_csv": str(OUTDIR / "compact_boundary_gate_baseline_beta_v1.csv"),
            "residual_preview_csv": str(OUTDIR / "compact_boundary_gate_residual_preview_v1.csv"),
            "candidate_column_stats_csv": str(OUTDIR / "compact_boundary_gate_candidate_column_stats_v1.csv"),
            "plots": [
                str(OUTDIR / "candidate_low_rank_boundary_topology_axis_v1.png"),
                str(OUTDIR / "candidate_frozen_topology_boundary_gate_v1.png"),
                str(OUTDIR / "candidate_matrix_leverage_axis_v1.png"),
                str(OUTDIR / "compact_boundary_gate_baseline_residual_hist_v1.png"),
                str(OUTDIR / "compact_boundary_gate_delta_chi2_v1.png"),
            ],
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "A topology/boundary gate improves compact-ladder chi2 enough to overcome BIC penalty, remains nondegenerate "
                "against the original 47-parameter ladder model, and is not matched by the row-order control."
            ),
            "what_weakens_TAIRID_here": (
                "The candidate gate fails BIC/AIC, is degenerate with existing ladder parameters, or improves no better than a row-order control."
            ),
            "truth_boundary": (
                "Even a positive result only motivates row-label recovery and explicit Cepheid/SN/calibrator boundary testing. "
                "It does not prove TAIRID or resolve H0."
            ),
        },
    }

    write_json(OUTDIR / "compact_ladder_boundary_gate_likelihood_v1_summary.json", summary)

    with open(OUTDIR / "compact_ladder_boundary_gate_likelihood_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES compact ladder boundary-gate likelihood v1\n\n")
        f.write("Boundary: compact SH0ES ladder pressure test only. Not proof. Not BAO. Not Planck.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")

        f.write("Best cases:\n")
        f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")

        f.write("Model results:\n")
        f.write(json.dumps(model_rows, indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- A positive result only motivates row-label recovery and a stricter v2.\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not prove H0 resolution.\n")

    print("")
    print("TAIRID SH0ES compact ladder boundary-gate likelihood v1 complete.")
    print("Created:")
    print("  tairid_shoes_compact_ladder_boundary_gate_likelihood_v1_outputs/compact_ladder_boundary_gate_likelihood_v1_summary.json")
    print("  tairid_shoes_compact_ladder_boundary_gate_likelihood_v1_outputs/compact_ladder_boundary_gate_likelihood_v1_summary.txt")
    print("  tairid_shoes_compact_ladder_boundary_gate_likelihood_v1_outputs/compact_ladder_boundary_gate_model_results_v1.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is a compact SH0ES ladder boundary-gate pressure test.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
