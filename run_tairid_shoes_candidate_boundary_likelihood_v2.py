#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder candidate-boundary likelihood v2.

Purpose:
The table-aligned row-label recovery v1.1 recovered provisional compact-ladder
candidate row groups:

- 277 rows: candidate SH0ES Hubble-flow / SN-intercept block
- 77 rows: candidate calibrator-boundary block
- 8 rows: candidate anchor/prior rows

The prior topology-only gate failed because abstract topology columns were
absorbed by the original 47-parameter compact ladder model.

This v2 test asks the sharper question:

Do the recovered candidate row groups carry likelihood-relevant structure after
the original 47-parameter compact ladder model is already fitted?

Models:
1. Baseline compact SH0ES ladder:
   y = X beta, where X = L.T

2. Single candidate boundary vectors:
   - 277-row Hubble-flow candidate
   - 77-row calibrator candidate
   - 8-row anchor/prior candidate
   - calibrator minus Hubble-flow contrast

3. Combined candidate vectors:
   - Hubble-flow + calibrator separately
   - Hubble-flow + calibrator + anchor

4. Controls:
   - row-order linear control
   - same-size random masks
   - same-size contiguous random blocks

Decision checks:
- delta chi2
- AIC / BIC
- nondegeneracy after residualizing against original X under C^-1 geometry
- random/control p-values
- baseline parameter stability
- parameter 46 / H0-like shift audit

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full multi-observable cosmology.
Labels remain provisional.
A positive result only says candidate compact-ladder boundary rows carry
likelihood-relevant structure worth testing with better labels.
"""

import csv
import json
import math
import re
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


OUTDIR = Path("tairid_shoes_candidate_boundary_likelihood_v2_outputs")
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

EPS = 1.0e-12
RANDOM_SEED = 42
RANDOM_CONTROL_REPEATS = 300


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
            "User-Agent": "TAIRID-SH0ES-candidate-boundary-likelihood-v2",
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

    head = data[:220].decode("utf-8", errors="replace")
    return "version https://git-lfs.github.com/spec/v1" in head and "oid sha256:" in head


def parse_lfs_pointer(data):
    text = data.decode("utf-8", errors="replace")
    oid = re.search(r"oid sha256:([a-fA-F0-9]+)", text)
    size = re.search(r"size\s+([0-9]+)", text)

    return {
        "raw_text": text,
        "oid_sha256": oid.group(1) if oid else None,
        "declared_size": int(size.group(1)) if size else None,
    }


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


def download_repo_path(repo_path, label):
    local = DOWNLOAD_DIR / safe_name(repo_path)
    attempts = []
    pointer_info = None

    for cand in candidate_urls(repo_path):
        try:
            data, final_url, content_type, status = fetch_url(cand["url"])

            attempt = {
                "label": label,
                "repo_path": repo_path,
                "candidate_kind": cand["kind"],
                "url": cand["url"],
                "final_url": final_url,
                "http_status": status,
                "content_type": content_type,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }

            if is_git_lfs_pointer(data):
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
                    "candidate_kind": cand["kind"],
                    "url": cand["url"],
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
                    "candidate_kind": cand["kind"],
                    "url": cand["url"],
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
        for idx, hdu in enumerate(hdul):
            data = hdu.data

            if data is None:
                continue

            try:
                if getattr(data.dtype, "fields", None):
                    fields = list(data.dtype.fields.keys())
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
            "orientation": "using_L_as_observation_by_parameter",
            "X_shape": list(L.shape),
        }

    return None, {
        "status": "no_axis_matches_y",
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

            attempts.append({"attempt": i, "jitter": jitter, "status": "success"})
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
    Cinv_residual = cho_solve(c_factor, residual, check_finite=False)
    chi2_value = float(residual.T @ Cinv_residual)

    n = len(y)
    k = X.shape[1]
    dof = int(n - k)
    beta_err = np.sqrt(np.maximum(np.diag(normal_inv), 0.0))

    return {
        "beta": beta,
        "beta_err": beta_err,
        "residual": residual,
        "normal": normal,
        "normal_inv": normal_inv,
        "chi2": chi2_value,
        "dof": dof,
        "k": int(k),
        "aic": float(chi2_value + 2.0 * k),
        "bic": float(chi2_value + k * math.log(n)),
        "reduced_chi2": float(chi2_value / dof) if dof > 0 else float("nan"),
    }


def standardize_col(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = v - np.mean(v)
    sd = np.std(v)

    if not np.isfinite(sd) or sd <= 1.0e-14:
        return np.zeros_like(v)

    return v / sd


def c_norm(v, c_factor):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(math.sqrt(max(v.T @ cho_solve(c_factor, v, check_finite=False), 0.0)))


def weighted_residualize_columns(Z, X, c_factor):
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    Cinv_X = cho_solve(c_factor, X, check_finite=False)
    Cinv_Z = cho_solve(c_factor, Z, check_finite=False)

    A = X.T @ Cinv_X
    A_inv = np.linalg.pinv(A, rcond=1.0e-12)
    coeff = A_inv @ (X.T @ Cinv_Z)

    return Z - X @ coeff


def row_signature(row):
    active = np.where(np.abs(row) > 1.0e-12)[0]
    signs = np.sign(row[active]).astype(int)

    full_key = ",".join(f"{int(i)}:{int(s)}" for i, s in zip(active, signs))
    active_key = ",".join(str(int(i)) for i in active)
    sign_key = ",".join(str(int(s)) for s in signs)

    return active, signs, full_key, active_key, sign_key


def recover_candidate_row_labels(X, y, baseline):
    keys = []
    active_keys = []
    sign_keys = []
    nonzero_counts = []

    for i in range(X.shape[0]):
        active, signs, full_key, active_key, sign_key = row_signature(X[i, :])
        keys.append(full_key)
        active_keys.append(active_key)
        sign_keys.append(sign_key)
        nonzero_counts.append(len(active))

    cluster_counts = Counter(keys)

    # First, identify strong candidates by the exact table-aligned recovery clues.
    hf_keys = {
        key for key, count in cluster_counts.items()
        if count == 277 and key == "42:1,46:-1"
    }

    if not hf_keys:
        hf_keys = {
            key for key, count in cluster_counts.items()
            if count == 277
        }

    cal_keys = {
        key for key, count in cluster_counts.items()
        if count == 77
    }

    row_labels = []
    label_counts = Counter()

    for i, key in enumerate(keys):
        nz = nonzero_counts[i]

        if key in hf_keys:
            label = "candidate_SH0ES_Hubble_flow_or_SN_intercept_block"
        elif key in cal_keys:
            label = "candidate_calibrator_boundary_block"
        elif nz == 1:
            label = "candidate_anchor_or_prior_row"
        elif nz >= 8:
            label = "candidate_Cepheid_SN_calibration_relation"
        elif nz == 2 and "1,-1" in sign_keys[i]:
            label = "two_parameter_difference_or_relative_constraint"
        elif nz <= 4 and ("1,-1" in sign_keys[i] or "-1,1" in sign_keys[i]):
            label = "sparse_ladder_relation"
        elif nz <= 4:
            label = "sparse_measurement_or_constraint"
        else:
            label = "medium_ladder_measurement"

        row_labels.append(label)
        label_counts[label] += 1

    rows = []

    for i, label in enumerate(row_labels):
        rows.append(
            {
                "observation_index": i,
                "candidate_label": label,
                "cluster_size": int(cluster_counts[keys[i]]),
                "full_signature": keys[i],
                "active_cols": active_keys[i],
                "sign_pattern": sign_keys[i],
                "nonzero_count": int(nonzero_counts[i]),
                "y": float(y[i]),
                "baseline_residual": float(baseline["residual"][i]),
            }
        )

    return row_labels, rows, dict(label_counts)


def build_candidate_vectors(row_labels):
    labels = np.asarray(row_labels, dtype=object)
    n = len(labels)

    hf = (labels == "candidate_SH0ES_Hubble_flow_or_SN_intercept_block").astype(float)
    cal = (labels == "candidate_calibrator_boundary_block").astype(float)
    anchor = (labels == "candidate_anchor_or_prior_row").astype(float)
    ceph = (labels == "candidate_Cepheid_SN_calibration_relation").astype(float)

    vectors = {
        "candidate_hubble_flow_277": hf,
        "candidate_calibrator_77": cal,
        "candidate_anchor_prior_8": anchor,
        "candidate_ceph_sn_dense": ceph,
        "candidate_calibrator_minus_hubble_flow": cal - hf,
        "candidate_hubble_flow_union_calibrator": ((hf + cal) > 0).astype(float),
        "control_row_order_linear": np.linspace(-1.0, 1.0, n),
    }

    return vectors


def single_column_delta(y, X, c_factor, baseline, col):
    z = standardize_col(col)

    raw_norm = c_norm(z, c_factor)

    if raw_norm <= EPS:
        return {
            "delta_chi2": 0.0,
            "nondegenerate_ratio": 0.0,
        }

    z_res = weighted_residualize_columns(z, X, c_factor).reshape(-1)
    res_norm = c_norm(z_res, c_factor)
    ratio = float(res_norm / max(raw_norm, EPS))

    if res_norm <= EPS:
        return {
            "delta_chi2": 0.0,
            "nondegenerate_ratio": ratio,
        }

    Cinv_residual = cho_solve(c_factor, baseline["residual"], check_finite=False)
    numerator = float(z_res.T @ Cinv_residual)
    denominator = float(z_res.T @ cho_solve(c_factor, z_res, check_finite=False))

    delta = float((numerator * numerator) / max(denominator, EPS))

    return {
        "delta_chi2": delta,
        "nondegenerate_ratio": ratio,
    }


def nested_model_test(y, X_base, c_factor, baseline, col_map, model_name, column_names):
    Z_raw = np.column_stack([standardize_col(col_map[name]) for name in column_names])

    raw_norms = [c_norm(Z_raw[:, i], c_factor) for i in range(Z_raw.shape[1])]
    Z_resid = weighted_residualize_columns(Z_raw, X_base, c_factor)
    resid_norms = [c_norm(Z_resid[:, i], c_factor) for i in range(Z_resid.shape[1])]

    nondegenerate_ratios = [
        float(r / max(a, EPS)) for r, a in zip(resid_norms, raw_norms)
    ]

    X_new = np.column_stack([X_base, Z_raw])
    fit = gls_fit(y, X_new, c_factor)

    delta_chi2 = float(baseline["chi2"] - fit["chi2"])
    delta_dof = int(len(column_names))
    p_value = float(chi2.sf(max(delta_chi2, 0.0), delta_dof))

    delta_aic = float(fit["aic"] - baseline["aic"])
    delta_bic = float(fit["bic"] - baseline["bic"])

    base_beta = baseline["beta"]
    new_beta = fit["beta"][: len(base_beta)]
    base_err = baseline["beta_err"]

    shift_sigma = np.abs(new_beta - base_beta) / np.maximum(base_err, EPS)

    added_beta = fit["beta"][-delta_dof:]
    added_err = fit["beta_err"][-delta_dof:]
    added_z = added_beta / np.maximum(added_err, EPS)

    h0_like_baseline = float(10.0 ** (baseline["beta"][46] / 5.0)) if len(baseline["beta"]) > 46 else None
    h0_like_model = float(10.0 ** (fit["beta"][46] / 5.0)) if len(fit["beta"]) > 46 else None

    if h0_like_baseline is not None and h0_like_model is not None:
        h0_like_shift = float(h0_like_model - h0_like_baseline)
    else:
        h0_like_shift = None

    return {
        "model_name": model_name,
        "column_names": " | ".join(column_names),
        "added_column_count": delta_dof,
        "baseline_chi2": baseline["chi2"],
        "model_chi2": fit["chi2"],
        "delta_chi2_improvement": delta_chi2,
        "p_value_chi2_improvement": p_value,
        "baseline_aic": baseline["aic"],
        "model_aic": fit["aic"],
        "delta_aic_model_minus_baseline": delta_aic,
        "baseline_bic": baseline["bic"],
        "model_bic": fit["bic"],
        "delta_bic_model_minus_baseline": delta_bic,
        "model_dof": fit["dof"],
        "model_reduced_chi2": fit["reduced_chi2"],
        "nondegenerate_ratios_json": json.dumps(nondegenerate_ratios),
        "min_nondegenerate_ratio": float(np.min(nondegenerate_ratios)),
        "added_beta_json": json.dumps([float(v) for v in added_beta]),
        "added_beta_err_json": json.dumps([float(v) for v in added_err]),
        "added_beta_z_json": json.dumps([float(v) for v in added_z]),
        "max_abs_added_beta_z": float(np.max(np.abs(added_z))) if len(added_z) else None,
        "max_baseline_param_shift_sigma": float(np.max(shift_sigma)) if len(shift_sigma) else None,
        "median_baseline_param_shift_sigma": float(np.median(shift_sigma)) if len(shift_sigma) else None,
        "param46_fivelogH0_baseline": float(baseline["beta"][46]) if len(baseline["beta"]) > 46 else None,
        "param46_fivelogH0_model": float(fit["beta"][46]) if len(fit["beta"]) > 46 else None,
        "param46_H0_like_baseline": h0_like_baseline,
        "param46_H0_like_model": h0_like_model,
        "param46_H0_like_shift": h0_like_shift,
    }, fit


def random_same_count_control(y, X, c_factor, baseline, observed_delta, count, repeats, rng):
    deltas = []

    n = len(y)

    for _ in range(repeats):
        mask = np.zeros(n, dtype=float)
        idx = rng.choice(n, size=int(count), replace=False)
        mask[idx] = 1.0

        delta = single_column_delta(y, X, c_factor, baseline, mask)["delta_chi2"]
        deltas.append(delta)

    deltas = np.asarray(deltas, dtype=float)

    return {
        "control_type": "random_same_count_mask",
        "count": int(count),
        "repeats": int(repeats),
        "observed_delta_chi2": float(observed_delta),
        "control_delta_mean": float(np.mean(deltas)),
        "control_delta_std": float(np.std(deltas)),
        "control_delta_95": float(np.percentile(deltas, 95)),
        "control_delta_99": float(np.percentile(deltas, 99)),
        "p_control_delta_ge_observed": float((1.0 + np.sum(deltas >= observed_delta)) / (1.0 + len(deltas))),
    }


def random_contiguous_block_control(y, X, c_factor, baseline, observed_delta, count, repeats, rng):
    deltas = []
    n = len(y)
    count = int(count)

    if count <= 0 or count >= n:
        return {
            "control_type": "random_contiguous_block",
            "count": count,
            "repeats": 0,
            "observed_delta_chi2": float(observed_delta),
            "p_control_delta_ge_observed": None,
        }

    for _ in range(repeats):
        start = int(rng.integers(0, n - count + 1))
        mask = np.zeros(n, dtype=float)
        mask[start : start + count] = 1.0

        delta = single_column_delta(y, X, c_factor, baseline, mask)["delta_chi2"]
        deltas.append(delta)

    deltas = np.asarray(deltas, dtype=float)

    return {
        "control_type": "random_contiguous_block",
        "count": int(count),
        "repeats": int(repeats),
        "observed_delta_chi2": float(observed_delta),
        "control_delta_mean": float(np.mean(deltas)),
        "control_delta_std": float(np.std(deltas)),
        "control_delta_95": float(np.percentile(deltas, 95)),
        "control_delta_99": float(np.percentile(deltas, 99)),
        "p_control_delta_ge_observed": float((1.0 + np.sum(deltas >= observed_delta)) / (1.0 + len(deltas))),
    }


def attach_controls(y, X, c_factor, baseline, row, col_map, rng):
    names = row["column_names"].split(" | ")

    if len(names) != 1:
        return row, []

    name = names[0]
    vector = col_map[name]
    count = int(np.sum(np.asarray(vector) > 0.5))

    if count <= 0 or count >= len(y):
        return row, []

    observed = row["delta_chi2_improvement"]

    random_mask = random_same_count_control(
        y,
        X,
        c_factor,
        baseline,
        observed,
        count,
        RANDOM_CONTROL_REPEATS,
        rng,
    )

    block = random_contiguous_block_control(
        y,
        X,
        c_factor,
        baseline,
        observed,
        count,
        RANDOM_CONTROL_REPEATS,
        rng,
    )

    row["same_count_random_p"] = random_mask.get("p_control_delta_ge_observed")
    row["same_count_random_delta_95"] = random_mask.get("control_delta_95")
    row["contiguous_block_p"] = block.get("p_control_delta_ge_observed")
    row["contiguous_block_delta_95"] = block.get("control_delta_95")

    return row, [
        {"model_name": row["model_name"], **random_mask},
        {"model_name": row["model_name"], **block},
    ]


def summarize_candidate_vectors(col_map):
    rows = []

    for name, col in col_map.items():
        vals = np.asarray(col, dtype=float)

        rows.append(
            {
                "candidate_vector": name,
                "count_positive": int(np.sum(vals > 0.5)),
                "count_negative": int(np.sum(vals < -0.5)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )

    return rows


def write_previews(y, baseline, row_label_rows, col_map, model_rows):
    residual_rows = []

    for i in range(min(len(y), 800)):
        residual_rows.append(
            {
                "observation_index": i,
                "y": float(y[i]),
                "baseline_residual": float(baseline["residual"][i]),
            }
        )

    write_csv(OUTDIR / "candidate_boundary_residual_preview_v2.csv", residual_rows)
    write_csv(OUTDIR / "candidate_boundary_row_label_map_v2.csv", row_label_rows)
    write_csv(OUTDIR / "candidate_boundary_vector_stats_v2.csv", summarize_candidate_vectors(col_map))

    beta_rows = []

    for i, (b, e) in enumerate(zip(baseline["beta"], baseline["beta_err"])):
        beta_rows.append(
            {
                "parameter_index": i,
                "baseline_beta": float(b),
                "baseline_beta_err": float(e),
                "H0_like_if_param46": float(10.0 ** (b / 5.0)) if i == 46 else "",
            }
        )

    write_csv(OUTDIR / "candidate_boundary_baseline_beta_v2.csv", beta_rows)

    plt.figure(figsize=(10, 6))
    plt.hist(baseline["residual"], bins=80)
    plt.xlabel("baseline compact GLS residual")
    plt.ylabel("count")
    plt.title("SH0ES compact baseline residuals")
    plt.tight_layout()
    plt.savefig(OUTDIR / "candidate_boundary_baseline_residual_hist_v2.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    labels = [r["model_name"] for r in model_rows]
    vals = [r["delta_chi2_improvement"] for r in model_rows]
    plt.bar(np.arange(len(labels)), vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("delta chi2 improvement")
    plt.title("Candidate-boundary compact ladder improvements")
    plt.tight_layout()
    plt.savefig(OUTDIR / "candidate_boundary_delta_chi2_v2.png", dpi=160)
    plt.close()

    for name in [
        "candidate_hubble_flow_277",
        "candidate_calibrator_77",
        "candidate_calibrator_minus_hubble_flow",
        "control_row_order_linear",
    ]:
        if name not in col_map:
            continue

        plt.figure(figsize=(11, 4))
        plt.plot(col_map[name], linewidth=1)
        plt.xlabel("compact row index")
        plt.ylabel("candidate vector value")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(OUTDIR / f"{name}_v2.png", dpi=160)
        plt.close()


def decide_status(model_rows):
    by_name = {r["model_name"]: r for r in model_rows}

    key_models = [
        "candidate_hubble_flow_277",
        "candidate_calibrator_77",
        "candidate_calibrator_minus_hubble_flow",
        "candidate_hubble_plus_calibrator",
        "candidate_hubble_calibrator_anchor",
    ]

    control = by_name.get("control_row_order_linear")

    def strong(row):
        if not row:
            return False

        random_p = row.get("same_count_random_p")
        block_p = row.get("contiguous_block_p")

        if random_p is None:
            random_ok = True
        else:
            random_ok = random_p <= 0.05

        if block_p is None:
            block_ok = True
        else:
            block_ok = block_p <= 0.10

        return bool(
            row.get("p_value_chi2_improvement", 1.0) <= 0.01
            and row.get("delta_bic_model_minus_baseline", 999.0) <= -6.0
            and row.get("min_nondegenerate_ratio", 0.0) >= 0.02
            and random_ok
            and block_ok
        )

    def directional(row):
        if not row:
            return False

        return bool(
            row.get("p_value_chi2_improvement", 1.0) <= 0.05
            and row.get("delta_aic_model_minus_baseline", 999.0) < 0.0
            and row.get("min_nondegenerate_ratio", 0.0) >= 0.01
        )

    control_strong = strong(control)
    candidate_strong = [name for name in key_models if strong(by_name.get(name))]
    candidate_directional = [name for name in key_models if directional(by_name.get(name))]

    best_bic = min(
        model_rows,
        key=lambda r: r.get("delta_bic_model_minus_baseline", 999.0),
    ) if model_rows else None

    best_delta = max(
        model_rows,
        key=lambda r: r.get("delta_chi2_improvement", -999.0),
    ) if model_rows else None

    best_cases = {
        "best_bic_model": best_bic,
        "best_delta_chi2_model": best_delta,
        "control_row_order_linear": control,
        "candidate_strong_models": candidate_strong,
        "candidate_directional_models": candidate_directional,
    }

    if candidate_strong and not control_strong:
        return (
            "candidate_boundary_likelihood_supported_provisional_labels",
            8,
            "Recover stronger row/parameter labels, then rerun as explicit SH0ES boundary likelihood v3 before theory promotion.",
            best_cases,
        )

    if candidate_strong and control_strong:
        return (
            "candidate_boundary_signal_confounded_by_row_order_control",
            7,
            "Do not promote. Row-order control also passes; recover better row labels and order-robust controls.",
            best_cases,
        )

    if candidate_directional:
        return (
            "candidate_boundary_likelihood_directional_not_locked",
            7,
            "Treat as directional only. Inspect controls and recover labels before another gate test.",
            best_cases,
        )

    return (
        "candidate_boundary_likelihood_not_supported",
        6,
        "Candidate 277/77 row groups do not add enough compact-ladder likelihood support beyond the original 47-parameter model.",
        best_cases,
    )


def main():
    print("")
    print("TAIRID SH0ES compact ladder candidate-boundary likelihood v2 starting.")
    print("Boundary: compact candidate-boundary likelihood screen only; not proof.")
    print("")

    downloads = {}
    ledger = []

    for label, repo_path in COMPACT_FILES.items():
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

    write_csv(OUTDIR / "candidate_boundary_download_ledger_v2.csv", ledger)
    write_json(OUTDIR / "candidate_boundary_download_attempts_v2.json", downloads)

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

    write_json(OUTDIR / "candidate_boundary_parse_meta_v2.json", parse_meta)
    write_json(OUTDIR / "candidate_boundary_parse_errors_v2.json", parse_errors)

    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]):
        summary = {
            "test_name": "TAIRID SH0ES compact ladder candidate-boundary likelihood v2",
            "boundary": "Matrix parse/download failure. No likelihood result.",
            "final_status": "candidate_boundary_matrix_parse_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix retrieval/parsing before likelihood testing.",
            "parse_errors": parse_errors,
            "downloads": downloads,
        }
        write_json(OUTDIR / "candidate_boundary_likelihood_v2_summary.json", summary)
        print("Parse failed. See summary JSON.")
        return

    C = np.asarray(parsed["allc"], dtype=np.float64)
    L = np.asarray(parsed["alll"], dtype=np.float64)
    y = np.asarray(parsed["ally"], dtype=np.float64).reshape(-1)

    X, orientation = determine_design_orientation(L, len(y))

    if X is None:
        raise RuntimeError(f"Could not orient L relative to y: {orientation}")

    if C.ndim != 2 or C.shape[0] != len(y) or C.shape[1] != len(y):
        raise RuntimeError(f"C shape {C.shape} does not match y length {len(y)}")

    c_factor, C_sym, jitter, chol_attempts = stable_cholesky_cov(C)
    baseline = gls_fit(y, X, c_factor)

    row_labels, row_label_rows, label_counts = recover_candidate_row_labels(X, y, baseline)
    col_map = build_candidate_vectors(row_labels)

    candidate_models = [
        ("control_row_order_linear", ["control_row_order_linear"]),
        ("candidate_hubble_flow_277", ["candidate_hubble_flow_277"]),
        ("candidate_calibrator_77", ["candidate_calibrator_77"]),
        ("candidate_anchor_prior_8", ["candidate_anchor_prior_8"]),
        ("candidate_ceph_sn_dense", ["candidate_ceph_sn_dense"]),
        ("candidate_calibrator_minus_hubble_flow", ["candidate_calibrator_minus_hubble_flow"]),
        ("candidate_hubble_union_calibrator", ["candidate_hubble_flow_union_calibrator"]),
        ("candidate_hubble_plus_calibrator", ["candidate_hubble_flow_277", "candidate_calibrator_77"]),
        ("candidate_hubble_calibrator_anchor", ["candidate_hubble_flow_277", "candidate_calibrator_77", "candidate_anchor_prior_8"]),
    ]

    model_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    control_rows = []

    for model_name, cols in candidate_models:
        row, fit = nested_model_test(y, X, c_factor, baseline, col_map, model_name, cols)

        row, controls = attach_controls(y, X, c_factor, baseline, row, col_map, rng)

        model_rows.append(row)
        control_rows.extend(controls)

    final_status, readiness_score, next_wall, best_cases = decide_status(model_rows)

    write_csv(OUTDIR / "candidate_boundary_model_results_v2.csv", model_rows)
    write_csv(OUTDIR / "candidate_boundary_random_control_checks_v2.csv", control_rows)

    write_previews(y, baseline, row_label_rows, col_map, model_rows)

    summary = {
        "test_name": "TAIRID SH0ES compact ladder candidate-boundary likelihood v2",
        "boundary": (
            "Strict provisional compact-ladder candidate-boundary likelihood screen. "
            "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model. "
            "Recovered row labels remain provisional."
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
            "normal_condition_estimate": float(np.linalg.cond(baseline["normal"])),
            "residual_mean": float(np.mean(baseline["residual"])),
            "residual_std": float(np.std(baseline["residual"])),
            "residual_rms": float(np.sqrt(np.mean(baseline["residual"] ** 2))),
            "param46_fivelogH0": float(baseline["beta"][46]) if len(baseline["beta"]) > 46 else None,
            "param46_H0_like": float(10.0 ** (baseline["beta"][46] / 5.0)) if len(baseline["beta"]) > 46 else None,
            "beta_preview_first_20": [float(v) for v in baseline["beta"][:20]],
            "beta_err_preview_first_20": [float(v) for v in baseline["beta_err"][:20]],
        },
        "candidate_label_counts": label_counts,
        "model_results": model_rows,
        "random_control_checks": control_rows,
        "best_cases": best_cases,
        "output_files": {
            "summary_json": str(OUTDIR / "candidate_boundary_likelihood_v2_summary.json"),
            "summary_txt": str(OUTDIR / "candidate_boundary_likelihood_v2_summary.txt"),
            "model_results_csv": str(OUTDIR / "candidate_boundary_model_results_v2.csv"),
            "random_controls_csv": str(OUTDIR / "candidate_boundary_random_control_checks_v2.csv"),
            "row_label_map_csv": str(OUTDIR / "candidate_boundary_row_label_map_v2.csv"),
            "vector_stats_csv": str(OUTDIR / "candidate_boundary_vector_stats_v2.csv"),
            "baseline_beta_csv": str(OUTDIR / "candidate_boundary_baseline_beta_v2.csv"),
            "residual_preview_csv": str(OUTDIR / "candidate_boundary_residual_preview_v2.csv"),
            "plots": [
                str(OUTDIR / "candidate_boundary_baseline_residual_hist_v2.png"),
                str(OUTDIR / "candidate_boundary_delta_chi2_v2.png"),
                str(OUTDIR / "candidate_hubble_flow_277_v2.png"),
                str(OUTDIR / "candidate_calibrator_77_v2.png"),
                str(OUTDIR / "candidate_calibrator_minus_hubble_flow_v2.png"),
                str(OUTDIR / "control_row_order_linear_v2.png"),
            ],
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "A recovered candidate boundary vector improves compact-ladder chi2 enough to overcome BIC, "
                "remains nondegenerate against the original 47-parameter ladder model, beats random/count controls, "
                "and is not matched by row-order control."
            ),
            "what_weakens_TAIRID_here": (
                "Candidate vectors fail AIC/BIC, are absorbed by existing 47-parameter freedom, do no better than "
                "same-size/random/order controls, or mainly shift parameter 46 without independent structure."
            ),
            "truth_boundary": (
                "A positive result would only justify stricter row-label recovery and v3 testing. "
                "A negative result does not kill TAIRID; it says these provisional candidate labels are not enough."
            ),
        },
    }

    write_json(OUTDIR / "candidate_boundary_likelihood_v2_summary.json", summary)

    with open(OUTDIR / "candidate_boundary_likelihood_v2_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES compact ladder candidate-boundary likelihood v2\n\n")
        f.write("Boundary: strict provisional compact-ladder candidate-boundary likelihood screen only.\n")
        f.write("Not proof of TAIRID. Not H0 resolution. Not BAO. Not Planck.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Candidate label counts:\n")
        f.write(json.dumps(label_counts, indent=2, default=json_default) + "\n\n")

        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")

        f.write("Best cases:\n")
        f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")

        f.write("Model results:\n")
        f.write(json.dumps(model_rows, indent=2, default=json_default) + "\n\n")

        f.write("Random/control checks:\n")
        f.write(json.dumps(control_rows, indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- A positive result only motivates stricter row-label recovery and v3 testing.\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not prove H0 resolution.\n")
        f.write("- Labels remain provisional.\n")

    print("")
    print("TAIRID SH0ES compact ladder candidate-boundary likelihood v2 complete.")
    print("Created:")
    print("  tairid_shoes_candidate_boundary_likelihood_v2_outputs/candidate_boundary_likelihood_v2_summary.json")
    print("  tairid_shoes_candidate_boundary_likelihood_v2_outputs/candidate_boundary_likelihood_v2_summary.txt")
    print("  tairid_shoes_candidate_boundary_likelihood_v2_outputs/candidate_boundary_model_results_v2.csv")
    print("  tairid_shoes_candidate_boundary_likelihood_v2_outputs/candidate_boundary_random_control_checks_v2.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is a compact SH0ES candidate-boundary likelihood screen.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
