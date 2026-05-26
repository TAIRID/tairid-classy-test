#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder gauge-null audit v1.

Purpose:
The SH0ES compact-ladder tests have now shown three things:

1. The compact ladder is real and solvable:
   y = ally, X = alll.T, C = allc.

2. The 42<->46 bridge is real matrix structure:
   277 rows connect parameter 42 and H0-like parameter 46.

3. Simple row masks and coefficient scalings do not add likelihood support.
   They are mostly absorbed by the original 47-parameter ladder model.

This audit asks the cleaner structural question:

Which proposed TAIRID boundary directions survive projection out of the original
47-parameter SH0ES column space under the full C^-1 geometry?

Boundary:
This is not proof of TAIRID.
This is not a cosmology fit.
This is not H0 resolution.
This is a gauge/null-space audit of candidate boundary directions.
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


OUTDIR = Path("tairid_shoes_compact_ladder_gauge_null_audit_v1_outputs")
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


def fetch_url(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-SH0ES-gauge-null-audit-v1",
            "Accept": "*/*",
        },
    )

    with urllib.request.urlopen(req, timeout=900) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        status = getattr(response, "status", None)

    return data, final_url, content_type, status


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


def recover_edge_rows(X, y, baseline):
    keys = [row_signature(X[i, :])[2] for i in range(X.shape[0])]
    cluster_counts = {key: keys.count(key) for key in set(keys)}

    row_rows = []
    bridge_mask = np.zeros(X.shape[0], dtype=bool)
    touch42_mask = np.zeros(X.shape[0], dtype=bool)
    touch46_mask = np.zeros(X.shape[0], dtype=bool)

    for i in range(X.shape[0]):
        active, signs, full_key, active_key, sign_key = row_signature(X[i, :])
        active_set = set(map(int, active))

        touches_42 = P42 in active_set
        touches_46 = P46 in active_set
        bridge = full_key == "42:1,46:-1"

        if not bridge and active_set == {P42, P46} and len(active) == 2:
            bridge = True

        bridge_mask[i] = bridge
        touch42_mask[i] = touches_42
        touch46_mask[i] = touches_46

        row_rows.append(
            {
                "observation_index": i,
                "active_cols": active_key,
                "sign_pattern": sign_key,
                "full_signature": full_key,
                "signature_cluster_size": int(cluster_counts[full_key]),
                "nonzero_count": int(len(active)),
                "touches_param42": bool(touches_42),
                "touches_param46_H0_like": bool(touches_46),
                "bridges_param42_param46": bool(bridge),
                "y": float(y[i]),
                "baseline_residual": float(baseline["residual"][i]),
                "abs_baseline_residual": float(abs(baseline["residual"][i])),
            }
        )

    return row_rows, bridge_mask, touch42_mask, touch46_mask


def standardize(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    sd = float(np.std(v))

    if not np.isfinite(sd) or sd <= 1.0e-14:
        return np.zeros_like(v)

    return (v - float(np.mean(v))) / sd


def audit_vector(name, vector, kind, y, X, c_factor, baseline):
    z = np.asarray(vector, dtype=float).reshape(-1)

    if len(z) != len(y):
        raise ValueError(f"{name} length {len(z)} does not match y length {len(y)}")

    finite = np.isfinite(z)
    z = np.where(finite, z, 0.0)

    c_inv_z = cho_solve(c_factor, z, check_finite=False)
    raw_norm2 = float(z.T @ c_inv_z)

    if raw_norm2 <= EPS:
        return {
            "candidate": name,
            "candidate_kind": kind,
            "status": "zero_or_near_zero_raw_norm",
            "raw_Cinv_norm": float(math.sqrt(max(raw_norm2, 0.0))),
            "residualized_Cinv_norm": 0.0,
            "nondegenerate_ratio": 0.0,
            "score": 0.0,
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
    residual_norm = float(math.sqrt(max(residual_norm2, 0.0)))
    raw_norm = float(math.sqrt(max(raw_norm2, 0.0)))
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
        "raw_mean": float(np.mean(z)),
        "raw_std": float(np.std(z)),
        "raw_min": float(np.min(z)),
        "raw_max": float(np.max(z)),
        "count_positive": int(np.sum(z > 0)),
        "count_negative": int(np.sum(z < 0)),
        "count_nonzero": int(np.sum(np.abs(z) > 1.0e-12)),
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


def build_candidate_vectors(X, y, baseline, row_rows, bridge_mask, touch42_mask, touch46_mask):
    residual_abs = np.abs(baseline["residual"])
    global_top10 = residual_abs >= np.percentile(residual_abs, 90)

    bridge_residual_abs = residual_abs[bridge_mask]

    if bridge_residual_abs.size:
        bridge_top25_threshold = np.percentile(bridge_residual_abs, 75)
        bridge_top25 = bridge_mask & (residual_abs >= bridge_top25_threshold)
    else:
        bridge_top25 = np.zeros(len(y), dtype=bool)

    vectors = []

    def add(name, values, kind):
        vectors.append(
            {
                "name": name,
                "values": np.asarray(values, dtype=float),
                "kind": kind,
            }
        )

    add("row_mask_bridge_42_46_rows", bridge_mask.astype(float), "edge_candidate")
    add("row_mask_touch_param46", touch46_mask.astype(float), "edge_candidate")
    add("row_mask_touch_param42", touch42_mask.astype(float), "edge_candidate")
    add("row_mask_touch42_not_bridge", (touch42_mask & (~bridge_mask)).astype(float), "edge_candidate")
    add("row_mask_bridge_high_residual_top25_within_bridge", bridge_top25.astype(float), "diagnostic_residual_pressure")
    add("row_mask_global_high_residual_top10", global_top10.astype(float), "diagnostic_residual_pressure")

    add(
        "edge_beta_direction_bridge_param42_scale",
        bridge_mask.astype(float) * X[:, P42] * baseline["beta"][P42],
        "edge_candidate",
    )
    add(
        "edge_beta_direction_bridge_param46_scale",
        bridge_mask.astype(float) * X[:, P46] * baseline["beta"][P46],
        "edge_candidate",
    )
    add(
        "edge_beta_direction_bridge_antisym_42_46",
        bridge_mask.astype(float)
        * (X[:, P42] * baseline["beta"][P42] - X[:, P46] * baseline["beta"][P46]),
        "edge_candidate",
    )
    add(
        "edge_beta_direction_bridge_common_42_46",
        bridge_mask.astype(float)
        * (X[:, P42] * baseline["beta"][P42] + X[:, P46] * baseline["beta"][P46]),
        "edge_candidate",
    )

    add("edge_column_bridge_param42", bridge_mask.astype(float) * X[:, P42], "edge_candidate")
    add("edge_column_bridge_param46", bridge_mask.astype(float) * X[:, P46], "edge_candidate")
    add(
        "edge_beta_direction_touch46_param46",
        touch46_mask.astype(float) * X[:, P46] * baseline["beta"][P46],
        "edge_candidate",
    )
    add(
        "edge_beta_direction_touch42_param42",
        touch42_mask.astype(float) * X[:, P42] * baseline["beta"][P42],
        "edge_candidate",
    )

    add("global_beta_direction_param46", X[:, P46] * baseline["beta"][P46], "control_existing_parameter")
    add("global_column_param46", X[:, P46], "control_existing_parameter")
    add("row_order_linear_control", np.linspace(-1.0, 1.0, len(y)), "control")
    add("row_order_bridge_only", bridge_mask.astype(float) * np.linspace(-1.0, 1.0, len(y)), "control")

    return vectors


def candidate_gram_audit(vectors, y, X, c_factor, baseline):
    selected = [
        item
        for item in vectors
        if item["kind"] == "edge_candidate" and np.std(item["values"]) > 1.0e-14
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

    rank_tol = 1.0e-6
    rank = int(np.sum(eigvals > rank_tol)) if eigvals.size else 0

    rows = []

    for i, name in enumerate(names):
        rows.append(
            {
                "candidate": name,
                "residualized_gram_diag": float(diag[i]),
                "residualized_norm": float(math.sqrt(max(diag[i], 0.0))),
            }
        )

    summary = {
        "status": "ok",
        "candidate_count": len(names),
        "residualized_correlation_rank_tol_1e_minus_6": rank,
        "eigen_min": float(np.min(eigvals)) if eigvals.size else None,
        "eigen_max": float(np.max(eigvals)) if eigvals.size else None,
        "eigenvalues": [float(v) for v in eigvals] if eigvals.size else [],
        "interpretation": (
            "Low rank or tiny residualized diagonal values mean the candidate edge vectors "
            "mostly live inside the original 47-parameter ladder column space."
        ),
    }

    write_csv(OUTDIR / "gauge_null_candidate_gram_diag_v1.csv", rows)
    write_json(OUTDIR / "gauge_null_candidate_gram_summary_v1.json", summary)

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
        mask[start : start + count] = 1.0

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

    def summarize(group, group_name):
        if not group:
            return {}

        ratios = np.asarray([r["nondegenerate_ratio"] for r in group], dtype=float)
        deltas = np.asarray([r["delta_chi2_score"] for r in group], dtype=float)

        return {
            f"{group_name}_ratio_mean": float(np.mean(ratios)),
            f"{group_name}_ratio_95": float(np.percentile(ratios, 95)),
            f"{group_name}_delta_mean": float(np.mean(deltas)),
            f"{group_name}_delta_95": float(np.percentile(deltas, 95)),
            f"{group_name}_delta_99": float(np.percentile(deltas, 99)),
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
        "fivelogH0",
        "H0",
        "alll",
        "ally",
        "allc",
        "theta",
        "samples",
        "parameter",
        "muhat",
        "intercept",
        "ceph",
        "cepheid",
        "anchor",
        "host",
        "calibrator",
        "pantheon",
        "hubble",
        "supernova",
        "sn",
    ]

    rows = []

    for item in aux_results:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["local_path"])

        if not path.name.lower().endswith(
            (".py", ".txt", ".md", ".dat", ".out", ".tex", ".readme")
        ):
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

    write_csv(OUTDIR / "gauge_null_code_context_hits_v1.csv", rows)

    return rows


def make_plots(row_rows, audit_rows, baseline, bridge_mask, touch46_mask):
    residual = np.asarray([r["baseline_residual"] for r in row_rows], dtype=float)

    plt.figure(figsize=(11, 5))
    plt.plot(residual, linewidth=0.8, label="baseline residual")

    if np.any(touch46_mask):
        plt.scatter(
            np.where(touch46_mask)[0],
            residual[touch46_mask],
            s=8,
            label="touches param46",
        )

    if np.any(bridge_mask):
        plt.scatter(
            np.where(bridge_mask)[0],
            residual[bridge_mask],
            s=10,
            label="42-46 bridge",
        )

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("compact row index")
    plt.ylabel("baseline residual")
    plt.title("Gauge-null audit residual map")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTDIR / "gauge_null_residual_map_v1.png", dpi=160)
    plt.close()

    plot_rows = [
        r
        for r in audit_rows
        if r.get("candidate_kind") in ["edge_candidate", "control_existing_parameter", "control"]
    ]

    labels = [r["candidate"] for r in plot_rows]
    ratios = [r["nondegenerate_ratio"] for r in plot_rows]
    deltas = [r["delta_chi2_score"] for r in plot_rows]

    plt.figure(figsize=(13, 6))
    plt.bar(np.arange(len(labels)), ratios)
    plt.xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("nondegenerate ratio after C^-1 projection")
    plt.title("Candidate direction survival after projecting out SH0ES 47-parameter column space")
    plt.tight_layout()
    plt.savefig(OUTDIR / "gauge_null_nondegenerate_ratios_v1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.bar(np.arange(len(labels)), deltas)
    plt.xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("delta chi2 score if added as one column")
    plt.title("Candidate residual alignment after gauge projection")
    plt.tight_layout()
    plt.savefig(OUTDIR / "gauge_null_delta_chi2_scores_v1.png", dpi=160)
    plt.close()


def decide_status(audit_rows, random_summaries):
    edge_rows = [
        r
        for r in audit_rows
        if r.get("candidate_kind") == "edge_candidate" and r.get("status") == "ok"
    ]

    if not edge_rows:
        return (
            "gauge_null_audit_no_edge_candidates",
            5,
            "No valid edge candidates were auditable; recover equation edges again.",
            {},
        )

    best_ratio = max(edge_rows, key=lambda r: r["nondegenerate_ratio"])
    best_delta = max(edge_rows, key=lambda r: r["delta_chi2_score"])

    strong = (
        best_delta["delta_bic_if_added_column"] < -2.0
        and best_delta["p_value_chi2_one_dof"] <= 0.01
        and best_delta["nondegenerate_ratio"] >= 0.02
    )

    nondegenerate_but_not_aligned = (
        best_ratio["nondegenerate_ratio"] >= 0.05
        and best_delta["p_value_chi2_one_dof"] > 0.05
    )

    mostly_gauge_null = best_ratio["nondegenerate_ratio"] < 0.02

    best_cases = {
        "best_nondegenerate_ratio_candidate": best_ratio,
        "best_delta_chi2_candidate": best_delta,
        "random_control_summaries": random_summaries,
    }

    if strong:
        return (
            "nonabsorbed_boundary_direction_detected_provisional",
            8,
            "Build a strict likelihood test using the surviving residualized direction, with labels still provisional.",
            best_cases,
        )

    if mostly_gauge_null:
        return (
            "candidate_edge_directions_mostly_gauge_null",
            7,
            "The proposed boundary directions mostly live inside the original 47-parameter SH0ES column space.",
            best_cases,
        )

    if nondegenerate_but_not_aligned:
        return (
            "nondegenerate_edge_directions_exist_but_not_residual_aligned",
            7,
            "Some edge directions survive projection, but they do not align with current residual pressure.",
            best_cases,
        )

    return (
        "no_likelihood_aligned_nonabsorbed_edge_direction",
        6,
        "Candidate directions either absorb into the SH0ES model or do not improve residual likelihood.",
        best_cases,
    )


def main():
    print("")
    print("TAIRID SH0ES compact ladder gauge-null audit v1 starting.")
    print("Boundary: null-space / degeneracy audit only; not proof.")
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

    write_csv(OUTDIR / "gauge_null_download_ledger_v1.csv", ledger)
    write_json(
        OUTDIR / "gauge_null_download_attempts_v1.json",
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

    write_json(OUTDIR / "gauge_null_parse_meta_v1.json", parse_meta)
    write_json(OUTDIR / "gauge_null_parse_errors_v1.json", parse_errors)

    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]):
        summary = {
            "test_name": "TAIRID SH0ES compact ladder gauge-null audit v1",
            "boundary": "Matrix parse/download failure. No audit result.",
            "final_status": "gauge_null_matrix_parse_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix retrieval/parsing before gauge audit.",
            "parse_errors": parse_errors,
        }

        write_json(OUTDIR / "gauge_null_audit_v1_summary.json", summary)
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

    row_rows, bridge_mask, touch42_mask, touch46_mask = recover_edge_rows(X, y, baseline)
    write_csv(OUTDIR / "gauge_null_row_map_v1.csv", row_rows)

    candidate_vectors = build_candidate_vectors(
        X,
        y,
        baseline,
        row_rows,
        bridge_mask,
        touch42_mask,
        touch46_mask,
    )

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

    write_csv(OUTDIR / "gauge_null_candidate_audit_v1.csv", audit_rows)

    gram_summary, gram_diag_rows = candidate_gram_audit(candidate_vectors, y, X, c_factor, baseline)

    random_summaries = []
    random_detail_rows = []

    bridge_count = int(np.sum(bridge_mask))

    if bridge_count > 0:
        summary_bridge, rows_bridge = random_control_audit(
            y,
            X,
            c_factor,
            baseline,
            bridge_count,
            "bridge_42_46_count",
        )
        random_summaries.append(summary_bridge)
        random_detail_rows.extend(rows_bridge)

    touch46_count = int(np.sum(touch46_mask))

    if touch46_count > 0 and touch46_count != bridge_count:
        summary_touch, rows_touch = random_control_audit(
            y,
            X,
            c_factor,
            baseline,
            touch46_count,
            "touch_param46_count",
        )
        random_summaries.append(summary_touch)
        random_detail_rows.extend(rows_touch)

    write_csv(OUTDIR / "gauge_null_random_control_details_v1.csv", random_detail_rows)
    write_json(OUTDIR / "gauge_null_random_control_summaries_v1.json", random_summaries)

    final_status, readiness_score, next_wall, best_cases = decide_status(audit_rows, random_summaries)

    make_plots(row_rows, audit_rows, baseline, bridge_mask, touch46_mask)

    edge_counts = {
        "rows_total": int(X.shape[0]),
        "bridge_42_46_rows": int(np.sum(bridge_mask)),
        "touch_param42_rows": int(np.sum(touch42_mask)),
        "touch_param46_rows": int(np.sum(touch46_mask)),
    }

    summary = {
        "test_name": "TAIRID SH0ES compact ladder gauge-null audit v1",
        "boundary": (
            "Gauge/null-space audit only. Not proof of TAIRID, not H0 resolution, "
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
            "residual_mean": float(np.mean(baseline["residual"])),
            "residual_std": float(np.std(baseline["residual"])),
            "residual_rms": float(np.sqrt(np.mean(baseline["residual"] ** 2))),
        },
        "edge_counts": edge_counts,
        "candidate_audit": audit_rows,
        "candidate_gram_summary": gram_summary,
        "random_control_summaries": random_summaries,
        "best_cases": best_cases,
        "code_context_hits_count": len(code_hits),
        "output_files": {
            "summary_json": str(OUTDIR / "gauge_null_audit_v1_summary.json"),
            "summary_txt": str(OUTDIR / "gauge_null_audit_v1_summary.txt"),
            "row_map_csv": str(OUTDIR / "gauge_null_row_map_v1.csv"),
            "candidate_audit_csv": str(OUTDIR / "gauge_null_candidate_audit_v1.csv"),
            "candidate_gram_summary_json": str(OUTDIR / "gauge_null_candidate_gram_summary_v1.json"),
            "candidate_gram_diag_csv": str(OUTDIR / "gauge_null_candidate_gram_diag_v1.csv"),
            "random_control_summaries_json": str(OUTDIR / "gauge_null_random_control_summaries_v1.json"),
            "random_control_details_csv": str(OUTDIR / "gauge_null_random_control_details_v1.csv"),
            "code_context_hits_csv": str(OUTDIR / "gauge_null_code_context_hits_v1.csv"),
            "plots": [
                str(OUTDIR / "gauge_null_residual_map_v1.png"),
                str(OUTDIR / "gauge_null_nondegenerate_ratios_v1.png"),
                str(OUTDIR / "gauge_null_delta_chi2_scores_v1.png"),
            ],
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "A proposed boundary vector keeps a meaningful C^-1 residualized norm after projecting out the original "
                "47-parameter SH0ES column space and also aligns with residual pressure strongly enough to improve BIC."
            ),
            "what_weakens_TAIRID_here": (
                "Candidate directions have tiny nondegenerate ratios, meaning they are mostly gauge/reparameterization "
                "directions already absorbed by the compact ladder, or they survive projection but do not align with residuals."
            ),
            "truth_boundary": (
                "This test cannot prove TAIRID. It only says whether any tested boundary direction survives the existing "
                "SH0ES model geometry well enough to justify another likelihood test."
            ),
        },
    }

    write_json(OUTDIR / "gauge_null_audit_v1_summary.json", summary)

    with open(OUTDIR / "gauge_null_audit_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES compact ladder gauge-null audit v1\n\n")
        f.write("Boundary: gauge/null-space audit only. Not proof. Not H0 resolution.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Edge counts:\n")
        f.write(json.dumps(edge_counts, indent=2, default=json_default) + "\n\n")

        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")

        f.write("Best cases:\n")
        f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")

        f.write("Candidate audit:\n")
        f.write(json.dumps(audit_rows, indent=2, default=json_default) + "\n\n")

        f.write("Random controls:\n")
        f.write(json.dumps(random_summaries, indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not prove H0 resolution.\n")
        f.write("- This only audits whether candidate directions survive projection out of the SH0ES 47-parameter model.\n")

    print("")
    print("TAIRID SH0ES compact ladder gauge-null audit v1 complete.")
    print("Created:")
    print("  tairid_shoes_compact_ladder_gauge_null_audit_v1_outputs/gauge_null_audit_v1_summary.json")
    print("  tairid_shoes_compact_ladder_gauge_null_audit_v1_outputs/gauge_null_audit_v1_summary.txt")
    print("  tairid_shoes_compact_ladder_gauge_null_audit_v1_outputs/gauge_null_candidate_audit_v1.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is a gauge/null-space audit only.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
