#!/usr/bin/env python3
"""
TAIRID SH0ES residual-pressure map v1.

Purpose:
The prior SH0ES compact-ladder tests found:

1. The compact ladder is real and solvable:
   y = ally
   X = alll.T
   C = allc

2. The 42<->46 bridge is real matrix structure:
   277 rows connect parameter 42 and H0-like parameter 46.

3. But the tested TAIRID bridge directions are mostly gauge-null:
   they live inside the original 47-parameter SH0ES column space.

This test stops forcing the 42<->46 bridge and maps where nonabsorbed residual
pressure actually lives after the 47-parameter compact ladder is fitted.

It asks:
- Which row families still have residual pressure?
- Which signature clusters carry nonabsorbed residual pressure?
- Do high residual rows, high leverage rows, covariance-diagonal rows, or C^-1
  residual rows survive projection out of the 47-parameter model?
- Is any remaining pressure structured, or mostly ordinary residual/outlier behavior?

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This is a residual-pressure map after the compact SH0ES model has already been fitted.
"""

import csv
import json
import math
import re
import hashlib
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


OUTDIR = Path("tairid_shoes_residual_pressure_map_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

OWNER = "PantheonPlusSH0ES"
REPO = "DataRelease"
BRANCH = "main"

COMPACT = {
    "allc": "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits",
    "alll": "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits",
    "ally": "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits",
}

AUX = [
    "SH0ES_Data/MCMC_utils.py",
    "SH0ES_Data/run_mcmc.py",
    "SH0ES_Data/lstsq_results.txt",
    "SH0ES_Data/README.md",
    "SH0ES_Data/table2.tex",
    "SH0ES_Data/table2.README",
]

P42 = 42
P46 = 46
EPS = 1.0e-12
SEED = 42
RANDOM_CONTROL_REPEATS = 80


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


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def candidate_urls(repo_path):
    quoted = urllib.parse.quote(repo_path, safe="/._-+")
    return [
        (
            "raw_githubusercontent",
            f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{quoted}",
        ),
        (
            "media_githubusercontent",
            f"https://media.githubusercontent.com/media/{OWNER}/{REPO}/{BRANCH}/{quoted}",
        ),
        (
            "github_raw",
            f"https://github.com/{OWNER}/{REPO}/raw/{BRANCH}/{quoted}",
        ),
    ]


def fetch_url(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-SH0ES-residual-pressure-map-v1",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        status = getattr(response, "status", None)
    return data, final_url, content_type, status


def is_lfs_pointer(data):
    head = data[:220].decode("utf-8", errors="replace")
    return "version https://git-lfs.github.com/spec/v1" in head and "oid sha256:" in head


def parse_lfs_pointer(data):
    text = data.decode("utf-8", errors="replace")
    oid = re.search(r"oid sha256:([a-fA-F0-9]+)", text)
    size = re.search(r"size\s+([0-9]+)", text)
    return {
        "oid_sha256": oid.group(1) if oid else None,
        "declared_size": int(size.group(1)) if size else None,
        "raw_text": text,
    }


def download_repo_path(repo_path, label):
    local = DOWNLOAD_DIR / safe_name(repo_path)
    attempts = []
    pointer_info = None

    for kind, url in candidate_urls(repo_path):
        try:
            data, final_url, content_type, status = fetch_url(url)

            attempt = {
                "label": label,
                "repo_path": repo_path,
                "candidate_kind": kind,
                "url": url,
                "final_url": final_url,
                "http_status": status,
                "content_type": content_type,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }

            if is_lfs_pointer(data):
                pointer_info = parse_lfs_pointer(data)
                attempt.update(pointer_info)
                attempt["status"] = "git_lfs_pointer_not_payload"
                attempts.append(attempt)
                continue

            local.write_bytes(data)

            attempt["status"] = "downloaded_real_payload"
            attempt["local_path"] = str(local)
            attempt["file_sha256"] = sha256_file(local)
            attempts.append(attempt)

            return {
                "label": label,
                "repo_path": repo_path,
                "status": "downloaded",
                "local_path": str(local),
                "bytes": local.stat().st_size,
                "sha256": sha256_file(local),
                "pointer_info": pointer_info,
                "attempts": attempts,
            }

        except urllib.error.HTTPError as exc:
            attempts.append(
                {
                    "label": label,
                    "repo_path": repo_path,
                    "candidate_kind": kind,
                    "url": url,
                    "status": "http_error",
                    "http_code": exc.code,
                    "error": str(exc),
                }
            )

        except Exception as exc:
            attempts.append(
                {
                    "label": label,
                    "repo_path": repo_path,
                    "candidate_kind": kind,
                    "url": url,
                    "status": "download_failed",
                    "error": str(exc),
                }
            )

    return {
        "label": label,
        "repo_path": repo_path,
        "status": "failed",
        "local_path": None,
        "pointer_info": pointer_info,
        "attempts": attempts,
    }


def extract_first_numeric_fits_array(path):
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        for index, hdu in enumerate(hdul):
            data = hdu.data

            if data is None:
                continue

            try:
                if getattr(data.dtype, "fields", None):
                    numeric = []

                    for name in data.dtype.fields:
                        values = np.asarray(data[name])

                        if np.issubdtype(values.dtype, np.number):
                            numeric.append(values)

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

                arr = arr.astype(float)

                if arr.size:
                    return arr, {
                        "hdu_index": index,
                        "hdu_name": hdu.name,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }

            except Exception:
                continue

    raise RuntimeError(f"No numeric FITS array found in {path}")


def orient_design_matrix(L, y_length):
    L = np.asarray(L, float)

    if L.ndim != 2:
        return None, {"status": "L_not_2d", "L_shape": list(L.shape)}

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
            "X_shape": list(L.shape),
        }

    return None, {
        "status": "no_axis_matches_y",
        "L_shape": list(L.shape),
        "y_length": int(y_length),
    }


def stable_cholesky(C):
    C_sym = 0.5 * (C + C.T)
    diag = np.diag(C_sym)
    scale = float(np.median(diag[diag > 0])) if np.any(diag > 0) else 1.0

    jitter = 0.0
    attempts = []

    for attempt in range(12):
        try:
            if jitter == 0.0:
                factor = cho_factor(C_sym, lower=True, check_finite=False)
            else:
                factor = cho_factor(
                    C_sym + np.eye(C_sym.shape[0]) * jitter,
                    lower=True,
                    check_finite=False,
                )

            attempts.append(
                {
                    "attempt": attempt,
                    "jitter": jitter,
                    "status": "success",
                }
            )

            return factor, C_sym, jitter, attempts

        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "jitter": jitter,
                    "status": "failed",
                    "error": str(exc),
                }
            )

            if jitter == 0.0:
                jitter = max(scale * 1.0e-12, 1.0e-14)
            else:
                jitter *= 10.0

    raise RuntimeError("Cholesky failed")


def base_gls_fit(y, X, c_factor):
    c_inv_y = cho_solve(c_factor, y, check_finite=False)
    c_inv_x = cho_solve(c_factor, X, check_finite=False)

    y_cinv_y = float(y.T @ c_inv_y)
    normal = X.T @ c_inv_x
    rhs = X.T @ c_inv_y

    normal_inv = np.linalg.pinv(normal, rcond=1.0e-12)
    beta = normal_inv @ rhs
    chi2_value = float(y_cinv_y - 2.0 * beta.T @ rhs + beta.T @ normal @ beta)

    residual = y - X @ beta
    c_inv_residual = c_inv_y - c_inv_x @ beta

    n = len(y)
    k = X.shape[1]
    beta_err = np.sqrt(np.maximum(np.diag(normal_inv), 0.0))

    return {
        "Cinv_y": c_inv_y,
        "Cinv_X": c_inv_x,
        "Cinv_residual": c_inv_residual,
        "y_Cinv_y": y_cinv_y,
        "normal": normal,
        "rhs": rhs,
        "normal_inv": normal_inv,
        "beta": beta,
        "beta_err": beta_err,
        "residual": residual,
        "chi2": chi2_value,
        "dof": int(n - k),
        "k": int(k),
        "aic": float(chi2_value + 2 * k),
        "bic": float(chi2_value + k * math.log(n)),
        "reduced_chi2": float(chi2_value / (n - k)),
    }


def h0_like_from_beta(beta):
    if len(beta) <= P46:
        return None
    return float(10.0 ** (beta[P46] / 5.0))


def row_signature(row):
    active = np.where(np.abs(row) > 1.0e-12)[0]
    signs = np.sign(row[active]).astype(int)

    full_key = ",".join(f"{int(i)}:{int(s)}" for i, s in zip(active, signs))
    active_key = ",".join(str(int(i)) for i in active)
    sign_key = ",".join(str(int(s)) for s in signs)

    return active, signs, full_key, active_key, sign_key


def classify_row(active, signs, full_key):
    active_set = set(map(int, active))

    if len(active) == 1:
        return "single_parameter_prior_or_anchor_constraint"

    if full_key == "42:1,46:-1":
        return "explicit_42_46_bridge"

    if active_set == {P42, P46} and len(active) == 2:
        return "explicit_42_46_bridge"

    if len(active) == 2 and np.any(signs > 0) and np.any(signs < 0):
        return "two_parameter_difference_or_relative_constraint"

    if len(active) <= 4 and np.any(signs > 0) and np.any(signs < 0):
        return "sparse_ladder_relation"

    if len(active) <= 4:
        return "sparse_measurement_or_constraint"

    if len(active) >= 8:
        return "dense_calibration_or_ceph_sn_relation"

    return "medium_ladder_measurement"


def recover_rows_and_clusters(X, y, C_sym, baseline):
    keys = []
    row_cache = []

    for i in range(X.shape[0]):
        active, signs, full_key, active_key, sign_key = row_signature(X[i, :])
        keys.append(full_key)
        row_cache.append((active, signs, full_key, active_key, sign_key))

    cluster_counts = Counter(keys)
    signature_to_id = {key: idx for idx, key in enumerate(cluster_counts.keys())}

    residual = baseline["residual"]
    cinv_residual = baseline["Cinv_residual"]
    cov_diag = np.diag(C_sym)

    leverage = np.einsum(
        "ij,jk,ik->i",
        X,
        baseline["normal_inv"],
        baseline["Cinv_X"],
    )

    row_rows = []
    grouped = defaultdict(list)

    for i, (active, signs, full_key, active_key, sign_key) in enumerate(row_cache):
        active_set = set(map(int, active))
        family = classify_row(active, signs, full_key)

        touches_42 = P42 in active_set
        touches_46 = P46 in active_set
        bridge = family == "explicit_42_46_bridge"

        row = {
            "observation_index": i,
            "signature_cluster_id": int(signature_to_id[full_key]),
            "signature_cluster_size": int(cluster_counts[full_key]),
            "equation_family": family,
            "active_cols": active_key,
            "sign_pattern": sign_key,
            "full_signature": full_key,
            "nonzero_count": int(len(active)),
            "touches_param42": bool(touches_42),
            "touches_param46_H0_like": bool(touches_46),
            "bridges_param42_param46": bool(bridge),
            "y": float(y[i]),
            "baseline_residual": float(residual[i]),
            "abs_baseline_residual": float(abs(residual[i])),
            "cinv_residual": float(cinv_residual[i]),
            "abs_cinv_residual": float(abs(cinv_residual[i])),
            "cov_diag": float(cov_diag[i]),
            "leverage_proxy": float(leverage[i]),
            "abs_leverage_proxy": float(abs(leverage[i])),
        }

        row_rows.append(row)
        grouped[full_key].append(row)

    cluster_rows = []

    for key, rows in grouped.items():
        residuals = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        cinv_res = np.asarray([r["cinv_residual"] for r in rows], dtype=float)
        first = rows[0]

        cluster_rows.append(
            {
                "signature_cluster_id": first["signature_cluster_id"],
                "size": len(rows),
                "equation_family": first["equation_family"],
                "active_cols": first["active_cols"],
                "sign_pattern": first["sign_pattern"],
                "touches_param42": first["touches_param42"],
                "touches_param46_H0_like": first["touches_param46_H0_like"],
                "bridges_param42_param46": first["bridges_param42_param46"],
                "first_row": int(rows[0]["observation_index"]),
                "last_row": int(rows[-1]["observation_index"]),
                "mean_residual": float(np.mean(residuals)),
                "median_residual": float(np.median(residuals)),
                "rms_residual": float(np.sqrt(np.mean(residuals * residuals))),
                "mean_abs_residual": float(np.mean(np.abs(residuals))),
                "mean_cinv_residual": float(np.mean(cinv_res)),
                "mean_abs_cinv_residual": float(np.mean(np.abs(cinv_res))),
                "mean_cov_diag": float(np.mean([r["cov_diag"] for r in rows])),
                "mean_abs_leverage_proxy": float(np.mean([r["abs_leverage_proxy"] for r in rows])),
                "y_mean": float(np.mean([r["y"] for r in rows])),
                "y_std": float(np.std([r["y"] for r in rows])),
            }
        )

    cluster_rows = sorted(
        cluster_rows,
        key=lambda r: (-r["mean_abs_residual"], -r["size"], r["signature_cluster_id"]),
    )

    family_grouped = defaultdict(list)
    for row in row_rows:
        family_grouped[row["equation_family"]].append(row)

    family_rows = []

    for family, rows in family_grouped.items():
        residuals = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        cinv_res = np.asarray([r["cinv_residual"] for r in rows], dtype=float)

        family_rows.append(
            {
                "equation_family": family,
                "row_count": len(rows),
                "mean_residual": float(np.mean(residuals)),
                "median_residual": float(np.median(residuals)),
                "rms_residual": float(np.sqrt(np.mean(residuals * residuals))),
                "mean_abs_residual": float(np.mean(np.abs(residuals))),
                "mean_cinv_residual": float(np.mean(cinv_res)),
                "mean_abs_cinv_residual": float(np.mean(np.abs(cinv_res))),
                "touches_param46_count": int(sum(1 for r in rows if r["touches_param46_H0_like"])),
                "bridges_42_46_count": int(sum(1 for r in rows if r["bridges_param42_param46"])),
            }
        )

    family_rows = sorted(family_rows, key=lambda r: (-r["mean_abs_residual"], -r["row_count"]))

    return row_rows, cluster_rows, family_rows


def standardize(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    sd = float(np.std(v))
    if not np.isfinite(sd) or sd <= 1.0e-14:
        return np.zeros_like(v)
    return (v - float(np.mean(v))) / sd


def audit_vector(name, vector, kind, y, X, c_factor, baseline):
    raw = np.asarray(vector, dtype=float).reshape(-1)
    raw = np.where(np.isfinite(raw), raw, 0.0)

    z = standardize(raw)

    c_inv_z = cho_solve(c_factor, z, check_finite=False)
    raw_norm2 = float(z.T @ c_inv_z)

    if raw_norm2 <= EPS:
        return {
            "candidate": name,
            "candidate_kind": kind,
            "status": "zero_or_near_zero_raw_norm",
            "count_nonzero_raw": int(np.sum(np.abs(raw) > 1.0e-12)),
            "nondegenerate_ratio": 0.0,
            "delta_chi2_score": 0.0,
            "p_value_chi2_one_dof": 1.0,
            "delta_aic_if_added_column": 2.0,
            "delta_bic_if_added_column": float(math.log(len(y))),
        }

    x_t_cinv_z = X.T @ c_inv_z
    coeff = baseline["normal_inv"] @ x_t_cinv_z

    z_perp = z - X @ coeff
    c_inv_z_perp = c_inv_z - baseline["Cinv_X"] @ coeff

    residual_norm2 = float(z_perp.T @ c_inv_z_perp)
    raw_norm = float(math.sqrt(max(raw_norm2, 0.0)))
    residual_norm = float(math.sqrt(max(residual_norm2, 0.0)))
    ratio = float(residual_norm / max(raw_norm, EPS))

    if residual_norm2 <= EPS:
        score = 0.0
        delta = 0.0
        alpha_hat = 0.0
    else:
        score = float(z_perp.T @ baseline["Cinv_residual"])
        delta = float((score * score) / residual_norm2)
        alpha_hat = float(score / residual_norm2)

    p_value = float(chi2.sf(max(delta, 0.0), 1))
    delta_aic = float(2.0 - delta)
    delta_bic = float(math.log(len(y)) - delta)

    return {
        "candidate": name,
        "candidate_kind": kind,
        "status": "ok",
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "count_positive_raw": int(np.sum(raw > 0)),
        "count_negative_raw": int(np.sum(raw < 0)),
        "count_nonzero_raw": int(np.sum(np.abs(raw) > 1.0e-12)),
        "raw_Cinv_norm": raw_norm,
        "residualized_Cinv_norm": residual_norm,
        "nondegenerate_ratio": ratio,
        "projection_absorption_fraction": float(1.0 - ratio),
        "score": score,
        "alpha_hat_added_column": alpha_hat,
        "delta_chi2_score": delta,
        "p_value_chi2_one_dof": p_value,
        "delta_aic_if_added_column": delta_aic,
        "delta_bic_if_added_column": delta_bic,
        "would_improve_aic": bool(delta_aic < 0.0),
        "would_improve_bic": bool(delta_bic < 0.0),
    }


def pct_mask(values, percentile, side="high"):
    values = np.asarray(values, dtype=float)

    if side == "high":
        threshold = np.percentile(values, percentile)
        return values >= threshold

    if side == "low":
        threshold = np.percentile(values, percentile)
        return values <= threshold

    raise ValueError(side)


def build_candidate_vectors(row_rows, cluster_rows):
    n = len(row_rows)

    residual_abs = np.asarray([r["abs_baseline_residual"] for r in row_rows], dtype=float)
    cinv_abs = np.asarray([r["abs_cinv_residual"] for r in row_rows], dtype=float)
    cov_diag = np.asarray([r["cov_diag"] for r in row_rows], dtype=float)
    leverage_abs = np.asarray([r["abs_leverage_proxy"] for r in row_rows], dtype=float)

    families = sorted(set(r["equation_family"] for r in row_rows))

    vectors = []

    def add(name, values, kind):
        vectors.append(
            {
                "name": name,
                "values": np.asarray(values, dtype=float),
                "kind": kind,
            }
        )

    for family in families:
        mask = np.asarray([r["equation_family"] == family for r in row_rows], dtype=float)
        add(f"family_{family}", mask, "structural_family")

    add("rows_touch_param46", np.asarray([r["touches_param46_H0_like"] for r in row_rows], dtype=float), "structural_edge")
    add("rows_touch_param42", np.asarray([r["touches_param42"] for r in row_rows], dtype=float), "structural_edge")
    add("rows_bridge_42_46", np.asarray([r["bridges_param42_param46"] for r in row_rows], dtype=float), "structural_edge")
    add("rows_touch42_not_bridge", np.asarray([r["touches_param42"] and not r["bridges_param42_param46"] for r in row_rows], dtype=float), "structural_edge")

    add("diagnostic_abs_residual_top05", pct_mask(residual_abs, 95).astype(float), "diagnostic_residual_selected")
    add("diagnostic_abs_residual_top10", pct_mask(residual_abs, 90).astype(float), "diagnostic_residual_selected")
    add("diagnostic_abs_residual_top20", pct_mask(residual_abs, 80).astype(float), "diagnostic_residual_selected")

    add("diagnostic_abs_cinv_residual_top05", pct_mask(cinv_abs, 95).astype(float), "diagnostic_cinv_residual_selected")
    add("diagnostic_abs_cinv_residual_top10", pct_mask(cinv_abs, 90).astype(float), "diagnostic_cinv_residual_selected")

    add("diagnostic_cov_diag_top10", pct_mask(cov_diag, 90).astype(float), "diagnostic_covariance")
    add("diagnostic_abs_leverage_top10", pct_mask(leverage_abs, 90).astype(float), "diagnostic_leverage")
    add("diagnostic_abs_leverage_top05", pct_mask(leverage_abs, 95).astype(float), "diagnostic_leverage")

    add("row_order_linear_control", np.linspace(-1.0, 1.0, n), "control")
    add("row_order_quadratic_control", np.linspace(-1.0, 1.0, n) ** 2, "control")

    # Top residual-pressure clusters by mean absolute residual.
    top_clusters = [
        c for c in cluster_rows
        if c["size"] >= 5
    ][:20]

    for c in top_clusters:
        cid = c["signature_cluster_id"]
        mask = np.asarray([r["signature_cluster_id"] == cid for r in row_rows], dtype=float)
        name = f"cluster_{cid}_size_{c['size']}_meanabs_{c['mean_abs_residual']:.4g}"
        add(name, mask, "structural_cluster")

    # Cluster groups with high mean residual pressure.
    high_cluster_ids = set(c["signature_cluster_id"] for c in top_clusters[:10])
    add(
        "cluster_union_top10_mean_abs_residual_clusters",
        np.asarray([r["signature_cluster_id"] in high_cluster_ids for r in row_rows], dtype=float),
        "structural_cluster_union",
    )

    # Overlap maps.
    bridge_mask = np.asarray([r["bridges_param42_param46"] for r in row_rows], dtype=bool)
    dense_mask = np.asarray([r["equation_family"] == "dense_calibration_or_ceph_sn_relation" for r in row_rows], dtype=bool)
    high_resid10 = pct_mask(residual_abs, 90)
    high_lev10 = pct_mask(leverage_abs, 90)

    add("overlap_bridge_and_high_residual_top10", (bridge_mask & high_resid10).astype(float), "overlap")
    add("overlap_dense_and_high_residual_top10", (dense_mask & high_resid10).astype(float), "overlap")
    add("overlap_high_leverage_and_high_residual_top10", (high_lev10 & high_resid10).astype(float), "overlap")

    return vectors


def candidate_gram_audit(vectors, y, X, c_factor, baseline):
    selected = [
        item for item in vectors
        if item["kind"] not in ["control"] and np.std(item["values"]) > 1.0e-14
    ]

    if not selected:
        return {"status": "no_candidate_vectors"}, []

    names = [item["name"] for item in selected]
    Z = np.column_stack([standardize(item["values"]) for item in selected])

    c_inv_Z = cho_solve(c_factor, Z, check_finite=False)
    coeff = baseline["normal_inv"] @ (X.T @ c_inv_Z)

    Z_perp = Z - X @ coeff
    c_inv_Z_perp = c_inv_Z - baseline["Cinv_X"] @ coeff

    gram = Z_perp.T @ c_inv_Z_perp
    diag = np.maximum(np.diag(gram), 0.0)
    scale = np.sqrt(np.maximum(diag, EPS))
    corr = gram / np.outer(scale, scale)

    try:
        eigvals = np.linalg.eigvalsh(corr)
    except Exception:
        eigvals = np.asarray([])

    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "candidate": name,
                "candidate_kind": selected[i]["kind"],
                "residualized_gram_diag": float(diag[i]),
                "residualized_norm": float(math.sqrt(max(diag[i], 0.0))),
            }
        )

    summary = {
        "status": "ok",
        "candidate_count": len(names),
        "residualized_correlation_rank_tol_1e_minus_6": int(np.sum(eigvals > 1.0e-6)) if eigvals.size else 0,
        "eigen_min": float(np.min(eigvals)) if eigvals.size else None,
        "eigen_max": float(np.max(eigvals)) if eigvals.size else None,
        "eigenvalues": [float(v) for v in eigvals] if eigvals.size else [],
    }

    write_csv(OUTDIR / "residual_pressure_candidate_gram_diag_v1.csv", rows)
    write_json(OUTDIR / "residual_pressure_candidate_gram_summary_v1.json", summary)

    return summary, rows


def random_control_audit(y, X, c_factor, baseline, count, label):
    rng = np.random.default_rng(SEED)
    rows = []
    count = int(count)

    if count <= 0 or count >= len(y):
        return {"label": label, "status": "invalid_count", "count": count}, rows

    for i in range(RANDOM_CONTROL_REPEATS):
        mask = np.zeros(len(y), dtype=float)
        idx = rng.choice(len(y), size=count, replace=False)
        mask[idx] = 1.0
        rows.append(
            audit_vector(
                f"{label}_random_same_count_{i}",
                mask,
                "random_same_count_control",
                y,
                X,
                c_factor,
                baseline,
            )
        )

    for i in range(RANDOM_CONTROL_REPEATS):
        mask = np.zeros(len(y), dtype=float)
        start = int(rng.integers(0, len(y) - count + 1))
        mask[start:start + count] = 1.0
        rows.append(
            audit_vector(
                f"{label}_random_contiguous_block_{i}",
                mask,
                "random_contiguous_block_control",
                y,
                X,
                c_factor,
                baseline,
            )
        )

    same = [r for r in rows if r["candidate_kind"] == "random_same_count_control"]
    block = [r for r in rows if r["candidate_kind"] == "random_contiguous_block_control"]

    def summarize(group, prefix):
        ratios = np.asarray([r["nondegenerate_ratio"] for r in group], dtype=float)
        deltas = np.asarray([r["delta_chi2_score"] for r in group], dtype=float)

        return {
            f"{prefix}_ratio_mean": float(np.mean(ratios)),
            f"{prefix}_ratio_95": float(np.percentile(ratios, 95)),
            f"{prefix}_delta_mean": float(np.mean(deltas)),
            f"{prefix}_delta_95": float(np.percentile(deltas, 95)),
            f"{prefix}_delta_99": float(np.percentile(deltas, 99)),
        }

    summary = {
        "label": label,
        "status": "ok",
        "count": count,
        "repeats_per_control_type": RANDOM_CONTROL_REPEATS,
        **summarize(same, "same_count"),
        **summarize(block, "contiguous_block"),
    }

    return summary, rows


def code_context_search(aux_results):
    terms = [
        "fivelogH0", "H0", "alll", "ally", "allc", "theta", "samples",
        "parameter", "muhat", "intercept", "ceph", "cepheid", "anchor",
        "host", "calibrator", "pantheon", "hubble", "supernova", "sn",
    ]

    rows = []

    for item in aux_results:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["local_path"])

        if not path.name.lower().endswith((".py", ".txt", ".md", ".dat", ".out", ".tex", ".readme")):
            continue

        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue

        preview_path = OUTDIR / f"preview_{safe_name(item['repo_path'])}.txt"
        preview_path.write_text("\n".join(lines[:180]), encoding="utf-8")

        for i, line in enumerate(lines, start=1):
            hits = [term for term in terms if term.lower() in line.lower()]

            if hits:
                lo = max(1, i - 2)
                hi = min(len(lines), i + 2)
                context = "\n".join(f"{j}: {lines[j - 1]}" for j in range(lo, hi + 1))

                rows.append(
                    {
                        "repo_path": item["repo_path"],
                        "line_number": i,
                        "hit_terms": " | ".join(hits),
                        "line": line[:700],
                        "context": context[:1600],
                    }
                )

    write_csv(OUTDIR / "residual_pressure_code_context_hits_v1.csv", rows)
    return rows


def make_plots(row_rows, cluster_rows, family_rows, audit_rows, baseline):
    residual = np.asarray([r["baseline_residual"] for r in row_rows], dtype=float)
    abs_resid = np.abs(residual)
    cinv_abs = np.asarray([r["abs_cinv_residual"] for r in row_rows], dtype=float)
    leverage = np.asarray([r["abs_leverage_proxy"] for r in row_rows], dtype=float)

    plt.figure(figsize=(11, 5))
    plt.plot(residual, linewidth=0.8)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("compact row index")
    plt.ylabel("baseline residual")
    plt.title("SH0ES compact residual pressure by row")
    plt.tight_layout()
    plt.savefig(OUTDIR / "residual_pressure_by_row_v1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(abs_resid, bins=80)
    plt.xlabel("absolute baseline residual")
    plt.ylabel("count")
    plt.title("Absolute residual distribution")
    plt.tight_layout()
    plt.savefig(OUTDIR / "abs_residual_hist_v1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(leverage, abs_resid, s=8)
    plt.xlabel("absolute leverage proxy")
    plt.ylabel("absolute residual")
    plt.title("Residual pressure vs leverage proxy")
    plt.tight_layout()
    plt.savefig(OUTDIR / "residual_vs_leverage_v1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(cinv_abs, abs_resid, s=8)
    plt.xlabel("absolute C^-1 residual")
    plt.ylabel("absolute residual")
    plt.title("Residual pressure vs C^-1 residual")
    plt.tight_layout()
    plt.savefig(OUTDIR / "residual_vs_cinv_residual_v1.png", dpi=160)
    plt.close()

    fam_labels = [r["equation_family"] for r in family_rows]
    fam_vals = [r["mean_abs_residual"] for r in family_rows]

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(fam_labels)), fam_vals)
    plt.xticks(np.arange(len(fam_labels)), fam_labels, rotation=35, ha="right")
    plt.ylabel("mean absolute residual")
    plt.title("Mean residual pressure by equation family")
    plt.tight_layout()
    plt.savefig(OUTDIR / "family_mean_abs_residual_v1.png", dpi=160)
    plt.close()

    top_clusters = cluster_rows[:30]
    labels = [str(c["signature_cluster_id"]) for c in top_clusters]
    vals = [c["mean_abs_residual"] for c in top_clusters]

    plt.figure(figsize=(13, 6))
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("mean absolute residual")
    plt.title("Top residual-pressure signature clusters")
    plt.tight_layout()
    plt.savefig(OUTDIR / "top_cluster_mean_abs_residual_v1.png", dpi=160)
    plt.close()

    plot_rows = [
        r for r in audit_rows
        if r.get("candidate_kind") not in ["random_same_count_control", "random_contiguous_block_control"]
    ]

    labels = [r["candidate"] for r in plot_rows]
    ratios = [r.get("nondegenerate_ratio", 0.0) for r in plot_rows]
    deltas = [r.get("delta_chi2_score", 0.0) for r in plot_rows]

    plt.figure(figsize=(15, 6))
    plt.bar(np.arange(len(labels)), ratios)
    plt.xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=7)
    plt.ylabel("nondegenerate ratio")
    plt.title("Candidate residual-pressure direction survival")
    plt.tight_layout()
    plt.savefig(OUTDIR / "candidate_nondegenerate_ratios_v1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(15, 6))
    plt.bar(np.arange(len(labels)), deltas)
    plt.xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=7)
    plt.ylabel("delta chi2 score")
    plt.title("Candidate residual-pressure alignment")
    plt.tight_layout()
    plt.savefig(OUTDIR / "candidate_delta_chi2_scores_v1.png", dpi=160)
    plt.close()


def decide_status(audit_rows, random_summaries):
    structural_kinds = {
        "structural_family",
        "structural_edge",
        "structural_cluster",
        "structural_cluster_union",
        "overlap",
        "diagnostic_covariance",
        "diagnostic_leverage",
    }

    diagnostic_selected_kinds = {
        "diagnostic_residual_selected",
        "diagnostic_cinv_residual_selected",
    }

    structural = [
        r for r in audit_rows
        if r.get("candidate_kind") in structural_kinds and r.get("status") == "ok"
    ]

    diagnostic = [
        r for r in audit_rows
        if r.get("candidate_kind") in diagnostic_selected_kinds and r.get("status") == "ok"
    ]

    best_structural = max(structural, key=lambda r: r["delta_chi2_score"]) if structural else None
    best_diagnostic = max(diagnostic, key=lambda r: r["delta_chi2_score"]) if diagnostic else None
    best_any = max(
        [r for r in audit_rows if r.get("status") == "ok"],
        key=lambda r: r.get("delta_chi2_score", -1.0),
    ) if audit_rows else None

    strong_structured = bool(
        best_structural
        and best_structural["delta_bic_if_added_column"] < -2.0
        and best_structural["p_value_chi2_one_dof"] <= 0.01
        and best_structural["nondegenerate_ratio"] >= 0.05
    )

    directional_structured = bool(
        best_structural
        and best_structural["delta_aic_if_added_column"] < 0.0
        and best_structural["p_value_chi2_one_dof"] <= 0.05
        and best_structural["nondegenerate_ratio"] >= 0.02
    )

    diagnostic_pressure = bool(
        best_diagnostic
        and best_diagnostic["delta_chi2_score"] >= 3.84
        and best_diagnostic["nondegenerate_ratio"] >= 0.05
    )

    best_cases = {
        "best_structural_candidate": best_structural,
        "best_diagnostic_residual_selected_candidate": best_diagnostic,
        "best_any_candidate": best_any,
        "random_control_summaries": random_summaries,
    }

    if strong_structured:
        return (
            "structured_residual_pressure_detected_for_followup",
            8,
            "A nonabsorbed structural residual-pressure direction survived projection; build a stricter follow-up with external row labels.",
            best_cases,
        )

    if directional_structured:
        return (
            "structured_residual_pressure_directional_not_locked",
            7,
            "A structural direction is directional but not locked; inspect clusters and controls before any likelihood test.",
            best_cases,
        )

    if diagnostic_pressure:
        return (
            "residual_pressure_detected_but_diagnostic_selected",
            7,
            "Residual pressure exists outside the 47-parameter space, but the best signal is residual-selected and not theory-specific.",
            best_cases,
        )

    return (
        "no_structured_residual_pressure_detected",
        6,
        "No tested structural residual-pressure direction survives strongly enough to justify a new SH0ES boundary-likelihood test.",
        best_cases,
    )


def main():
    print("")
    print("TAIRID SH0ES residual-pressure map v1 starting.")
    print("Boundary: residual-pressure mapping only; not proof.")
    print("")

    downloads = {}
    aux_results = []
    ledger = []

    for label, repo_path in COMPACT.items():
        result = download_repo_path(repo_path, label)
        downloads[label] = result

        ledger.append(
            {
                "label": label,
                "repo_path": repo_path,
                "status": result.get("status"),
                "local_path": result.get("local_path"),
                "bytes": result.get("bytes"),
                "sha256": result.get("sha256"),
                "pointer_declared_size": (result.get("pointer_info") or {}).get("declared_size"),
                "pointer_oid_sha256": (result.get("pointer_info") or {}).get("oid_sha256"),
                "attempt_count": len(result.get("attempts", [])),
            }
        )

    for repo_path in AUX:
        result = download_repo_path(repo_path, safe_name(repo_path))
        aux_results.append(result)

        ledger.append(
            {
                "label": safe_name(repo_path),
                "repo_path": repo_path,
                "status": result.get("status"),
                "local_path": result.get("local_path"),
                "bytes": result.get("bytes"),
                "sha256": result.get("sha256"),
                "pointer_declared_size": (result.get("pointer_info") or {}).get("declared_size"),
                "pointer_oid_sha256": (result.get("pointer_info") or {}).get("oid_sha256"),
                "attempt_count": len(result.get("attempts", [])),
            }
        )

    write_csv(OUTDIR / "residual_pressure_download_ledger_v1.csv", ledger)
    write_json(
        OUTDIR / "residual_pressure_download_attempts_v1.json",
        {"compact": downloads, "auxiliary": aux_results},
    )

    code_hits = code_context_search(aux_results)

    parsed = {}
    parse_meta = {}
    parse_errors = []

    for label in ["allc", "alll", "ally"]:
        result = downloads.get(label, {})

        if result.get("status") != "downloaded":
            parse_errors.append(
                {
                    "label": label,
                    "status": "not_downloaded",
                    "download_status": result.get("status"),
                }
            )
            continue

        try:
            arr, meta = extract_first_numeric_fits_array(Path(result["local_path"]))
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

    write_json(OUTDIR / "residual_pressure_parse_meta_v1.json", parse_meta)
    write_json(OUTDIR / "residual_pressure_parse_errors_v1.json", parse_errors)

    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]):
        summary = {
            "test_name": "TAIRID SH0ES residual-pressure map v1",
            "boundary": "Matrix parse/download failure. No residual-pressure result.",
            "final_status": "residual_pressure_matrix_parse_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix parsing before residual-pressure mapping.",
            "parse_errors": parse_errors,
        }

        write_json(OUTDIR / "residual_pressure_map_v1_summary.json", summary)
        print("Parse failed. See summary JSON.")
        return

    C = np.asarray(parsed["allc"], float)
    L = np.asarray(parsed["alll"], float)
    y = np.asarray(parsed["ally"], float).reshape(-1)

    X, orientation = orient_design_matrix(L, len(y))

    if X is None:
        raise RuntimeError(f"Could not orient L: {orientation}")

    if C.ndim != 2 or C.shape[0] != len(y) or C.shape[1] != len(y):
        raise RuntimeError(f"C shape {C.shape} does not match y length {len(y)}")

    c_factor, C_sym, jitter, chol_attempts = stable_cholesky(C)
    baseline = base_gls_fit(y, X, c_factor)

    row_rows, cluster_rows, family_rows = recover_rows_and_clusters(X, y, C_sym, baseline)

    write_csv(OUTDIR / "residual_pressure_row_map_v1.csv", row_rows)
    write_csv(OUTDIR / "residual_pressure_signature_clusters_v1.csv", cluster_rows)
    write_csv(OUTDIR / "residual_pressure_family_summary_v1.csv", family_rows)

    candidate_vectors = build_candidate_vectors(row_rows, cluster_rows)

    audit_rows = []

    for item in candidate_vectors:
        audit_rows.append(
            audit_vector(
                item["name"],
                item["values"],
                item["kind"],
                y,
                X,
                c_factor,
                baseline,
            )
        )

    audit_rows = sorted(
        audit_rows,
        key=lambda r: (
            -float(r.get("delta_chi2_score", 0.0)),
            -float(r.get("nondegenerate_ratio", 0.0)),
        ),
    )

    write_csv(OUTDIR / "residual_pressure_candidate_audit_v1.csv", audit_rows)

    gram_summary, gram_rows = candidate_gram_audit(candidate_vectors, y, X, c_factor, baseline)

    random_summaries = []
    random_detail_rows = []

    # Run controls for the strongest non-control candidates by count.
    top_for_controls = [
        r for r in audit_rows
        if r.get("status") == "ok"
        and r.get("candidate_kind") not in ["control"]
        and 0 < int(r.get("count_nonzero_raw", 0)) < len(y)
    ][:3]

    seen_counts = set()

    for candidate in top_for_controls:
        count = int(candidate["count_nonzero_raw"])
        if count in seen_counts:
            continue
        seen_counts.add(count)

        summary_random, rows_random = random_control_audit(
            y,
            X,
            c_factor,
            baseline,
            count,
            f"count_{count}",
        )

        random_summaries.append(summary_random)
        random_detail_rows.extend(rows_random)

    write_json(OUTDIR / "residual_pressure_random_control_summaries_v1.json", random_summaries)
    write_csv(OUTDIR / "residual_pressure_random_control_details_v1.csv", random_detail_rows)

    final_status, readiness_score, next_wall, best_cases = decide_status(audit_rows, random_summaries)

    make_plots(row_rows, cluster_rows, family_rows, audit_rows, baseline)

    residual = baseline["residual"]
    abs_resid = np.abs(residual)

    edge_counts = {
        "rows_total": int(X.shape[0]),
        "bridge_42_46_rows": int(sum(1 for r in row_rows if r["bridges_param42_param46"])),
        "touch_param42_rows": int(sum(1 for r in row_rows if r["touches_param42"])),
        "touch_param46_rows": int(sum(1 for r in row_rows if r["touches_param46_H0_like"])),
    }

    summary = {
        "test_name": "TAIRID SH0ES residual-pressure map v1",
        "boundary": (
            "Residual-pressure mapping only. Not proof of TAIRID, not H0 resolution, "
            "not BAO, not Planck, and not a full cosmology model."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "matrix_shapes": {
            "allc_C": list(C.shape),
            "alll_L": list(L.shape),
            "ally_y_original": list(np.asarray(parsed["ally"]).shape),
            "y_flat": list(y.shape),
            "X_design": list(X.shape),
            "L_orientation": orientation,
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
            "parameter_count": int(len(baseline["beta"])),
            "param42_value": float(baseline["beta"][P42]),
            "param42_err": float(baseline["beta_err"][P42]),
            "param46_fivelogH0": float(baseline["beta"][P46]),
            "param46_err": float(baseline["beta_err"][P46]),
            "param46_H0_like": h0_like_from_beta(baseline["beta"]),
            "normal_condition_estimate": float(np.linalg.cond(baseline["normal"])),
            "residual_mean": float(np.mean(residual)),
            "residual_std": float(np.std(residual)),
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "abs_residual_90": float(np.percentile(abs_resid, 90)),
            "abs_residual_95": float(np.percentile(abs_resid, 95)),
            "abs_residual_99": float(np.percentile(abs_resid, 99)),
        },
        "edge_counts": edge_counts,
        "equation_family_summary": family_rows,
        "top_signature_clusters_by_residual_pressure": cluster_rows[:40],
        "candidate_audit": audit_rows,
        "candidate_gram_summary": gram_summary,
        "random_control_summaries": random_summaries,
        "best_cases": best_cases,
        "code_context_hits_count": len(code_hits),
        "output_files": {
            "summary_json": str(OUTDIR / "residual_pressure_map_v1_summary.json"),
            "summary_txt": str(OUTDIR / "residual_pressure_map_v1_summary.txt"),
            "row_map_csv": str(OUTDIR / "residual_pressure_row_map_v1.csv"),
            "cluster_summary_csv": str(OUTDIR / "residual_pressure_signature_clusters_v1.csv"),
            "family_summary_csv": str(OUTDIR / "residual_pressure_family_summary_v1.csv"),
            "candidate_audit_csv": str(OUTDIR / "residual_pressure_candidate_audit_v1.csv"),
            "candidate_gram_summary_json": str(OUTDIR / "residual_pressure_candidate_gram_summary_v1.json"),
            "candidate_gram_diag_csv": str(OUTDIR / "residual_pressure_candidate_gram_diag_v1.csv"),
            "random_control_summaries_json": str(OUTDIR / "residual_pressure_random_control_summaries_v1.json"),
            "random_control_details_csv": str(OUTDIR / "residual_pressure_random_control_details_v1.csv"),
            "code_context_hits_csv": str(OUTDIR / "residual_pressure_code_context_hits_v1.csv"),
            "plots": [
                str(OUTDIR / "residual_pressure_by_row_v1.png"),
                str(OUTDIR / "abs_residual_hist_v1.png"),
                str(OUTDIR / "residual_vs_leverage_v1.png"),
                str(OUTDIR / "residual_vs_cinv_residual_v1.png"),
                str(OUTDIR / "family_mean_abs_residual_v1.png"),
                str(OUTDIR / "top_cluster_mean_abs_residual_v1.png"),
                str(OUTDIR / "candidate_nondegenerate_ratios_v1.png"),
                str(OUTDIR / "candidate_delta_chi2_scores_v1.png"),
            ],
        },
        "interpretation": {
            "what_supports_a_next_model_test": (
                "A structural row family, cluster, leverage, covariance, or overlap direction survives projection out of the "
                "47-parameter SH0ES model and aligns with residual pressure enough to improve AIC/BIC."
            ),
            "what_does_not_support_a_next_model_test": (
                "Only residual-selected diagnostic masks improve, or structural candidates are absorbed by the compact model."
            ),
            "truth_boundary": (
                "This does not test TAIRID directly. It maps remaining residual pressure after the SH0ES compact ladder is fitted."
            ),
        },
    }

    write_json(OUTDIR / "residual_pressure_map_v1_summary.json", summary)

    with open(OUTDIR / "residual_pressure_map_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES residual-pressure map v1\n\n")
        f.write("Boundary: residual-pressure mapping only. Not proof. Not H0 resolution.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")

        f.write("Edge counts:\n")
        f.write(json.dumps(edge_counts, indent=2, default=json_default) + "\n\n")

        f.write("Best cases:\n")
        f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")

        f.write("Equation family summary:\n")
        f.write(json.dumps(family_rows, indent=2, default=json_default) + "\n\n")

        f.write("Top signature clusters by residual pressure:\n")
        f.write(json.dumps(cluster_rows[:25], indent=2, default=json_default) + "\n\n")

        f.write("Candidate audit:\n")
        f.write(json.dumps(audit_rows[:50], indent=2, default=json_default) + "\n\n")

        f.write("Random controls:\n")
        f.write(json.dumps(random_summaries, indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not prove H0 resolution.\n")
        f.write("- This only maps nonabsorbed residual pressure after the 47-parameter SH0ES compact ladder fit.\n")

    print("")
    print("TAIRID SH0ES residual-pressure map v1 complete.")
    print("Created:")
    print("  tairid_shoes_residual_pressure_map_v1_outputs/residual_pressure_map_v1_summary.json")
    print("  tairid_shoes_residual_pressure_map_v1_outputs/residual_pressure_map_v1_summary.txt")
    print("  tairid_shoes_residual_pressure_map_v1_outputs/residual_pressure_candidate_audit_v1.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is residual-pressure mapping only.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
