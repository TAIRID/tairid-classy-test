#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder equation-edge recovery v1.

Purpose:
The candidate-boundary likelihood v2 showed that simple recovered row masks
(277 Hubble-flow candidate rows, 77 calibrator candidate rows) do not add
independent likelihood power beyond the original 47-parameter compact ladder.

That means row membership was too blunt.

This test maps the equation-edge geometry of the compact SH0ES ladder matrix L:

    y = data vector
    X = L.T = compact equation/design matrix
    C = covariance matrix

It asks:
- Which compact rows touch parameter 46, the H0-like parameter?
- Which compact rows touch parameter 42?
- Which rows bridge 42 and 46?
- Which parameter pairs co-occur in equations?
- Which row-signature families form the ladder boundary?
- Which equation families have large residual pressure after baseline GLS?
- Which candidate edge-level boundary vectors should be tested in a later v2 likelihood?

Boundary:
This is not proof of TAIRID.
This is not a new cosmology fit.
This is not H0 resolution.
This is not a boundary-gate likelihood.
This is equation-edge recovery only.

A successful result prepares a later strict likelihood test using equation-edge
vectors instead of row-group masks.
"""

import csv
import json
import math
import re
import hashlib
import urllib.parse
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve


OUTDIR = Path("tairid_shoes_equation_edge_recovery_v1_outputs")
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
    "SH0ES_Data/MCMC_utils.py",
    "SH0ES_Data/run_mcmc.py",
    "SH0ES_Data/lstsq_results.txt",
    "SH0ES_Data/README.md",
    "SH0ES_Data/table2.tex",
    "SH0ES_Data/table2.README",
    "SH0ES_Data/optical_wes_R22_for19fromR16.dat",
    "SH0ES_Data/optical_wes_R22_for19fromR16.wM31.dat",
    "SH0ES_Data/R22_orig19_NIR.out",
    "SH0ES_Data/R22_orig19_NIR.wm31.out",
]

EPS = 1.0e-12
H0_LIKE_PARAM_INDEX = 46
KNOWN_BRIDGE_PARAM_INDEX = 42


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
            "User-Agent": "TAIRID-SH0ES-equation-edge-recovery-v1",
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
        "reduced_chi2": float(chi2_value / dof) if dof > 0 else float("nan"),
        "normal_condition_estimate": float(np.linalg.cond(normal)),
    }


def text_context_search(downloads):
    rows = []
    previews = []

    terms = [
        "fivelogH0", "H0", "theta", "samples", "alll", "ally", "allc", "lstsq",
        "parameter", "intercept", "muhat", "cepheid", "ceph", "anchor", "host",
        "calibrator", "pantheon", "hubble", "sn", "sne", "supernova",
    ]

    for item in downloads:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["local_path"])
        lower = path.name.lower()
        if not lower.endswith((".py", ".txt", ".md", ".dat", ".out", ".tex", ".readme")):
            continue

        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue

        lines = text.splitlines()
        preview_path = OUTDIR / f"preview_{safe_name(item['repo_path'])}.txt"
        preview_path.write_text("\n".join(lines[:200]), encoding="utf-8")
        previews.append(
            {
                "repo_path": item["repo_path"],
                "local_path": str(path),
                "line_count": len(lines),
                "preview_file": str(preview_path),
            }
        )

        for i, line in enumerate(lines, start=1):
            ll = line.lower()
            hits = [t for t in terms if t.lower() in ll]
            if not hits:
                continue

            lo = max(1, i - 2)
            hi = min(len(lines), i + 2)
            context = "\n".join(f"{j}: {lines[j-1]}" for j in range(lo, hi + 1))

            rows.append(
                {
                    "repo_path": item["repo_path"],
                    "line_number": i,
                    "hit_terms": " | ".join(hits),
                    "line": line[:700],
                    "context": context[:1600],
                }
            )

    write_csv(OUTDIR / "equation_edge_code_context_hits_v1.csv", rows)
    write_csv(OUTDIR / "equation_edge_aux_previews_v1.csv", previews)
    return rows, previews


def row_signature(row):
    active = np.where(np.abs(row) > 1.0e-12)[0]
    signs = np.sign(row[active]).astype(int)
    vals = row[active]

    full_key = ",".join(f"{int(i)}:{int(s)}" for i, s in zip(active, signs))
    active_key = ",".join(str(int(i)) for i in active)
    sign_key = ",".join(str(int(s)) for s in signs)

    return active, signs, vals, full_key, active_key, sign_key


def build_row_edge_map(X, y, residual):
    rows = []
    cluster_counter = Counter()
    pair_counter = Counter()
    pair_abs_weight = defaultdict(float)
    pair_signed_weight = defaultdict(float)
    param_touch_counter = Counter()
    param_abs_weight = defaultdict(float)

    row_active_cache = []

    for i in range(X.shape[0]):
        active, signs, vals, full_key, active_key, sign_key = row_signature(X[i, :])
        cluster_counter[full_key] += 1
        row_active_cache.append((active, signs, vals, full_key, active_key, sign_key))

        for col, val in zip(active, vals):
            param_touch_counter[int(col)] += 1
            param_abs_weight[int(col)] += float(abs(val))

        for a_idx in range(len(active)):
            for b_idx in range(a_idx + 1, len(active)):
                a = int(active[a_idx])
                b = int(active[b_idx])
                va = float(vals[a_idx])
                vb = float(vals[b_idx])
                key = (min(a, b), max(a, b))
                pair_counter[key] += 1
                pair_abs_weight[key] += abs(va) + abs(vb)
                pair_signed_weight[key] += va * vb

    signature_to_id = {}
    for idx, key in enumerate(cluster_counter.keys()):
        signature_to_id[key] = idx

    for i, (active, signs, vals, full_key, active_key, sign_key) in enumerate(row_active_cache):
        touches_46 = H0_LIKE_PARAM_INDEX in set(int(c) for c in active)
        touches_42 = KNOWN_BRIDGE_PARAM_INDEX in set(int(c) for c in active)
        bridges_42_46 = touches_42 and touches_46

        if len(active) == 1:
            equation_family = "single_parameter_prior_or_anchor_constraint"
        elif len(active) == 2 and np.any(signs > 0) and np.any(signs < 0):
            equation_family = "two_parameter_difference_or_relative_constraint"
        elif bridges_42_46:
            equation_family = "explicit_42_46_bridge_equation"
        elif touches_46:
            equation_family = "touches_H0_like_parameter_46"
        elif len(active) <= 4 and np.any(signs > 0) and np.any(signs < 0):
            equation_family = "sparse_ladder_relation"
        elif len(active) >= 8:
            equation_family = "dense_calibration_or_ceph_sn_relation"
        elif len(active) <= 4:
            equation_family = "sparse_measurement_or_constraint"
        else:
            equation_family = "medium_ladder_measurement"

        rows.append(
            {
                "observation_index": i,
                "signature_cluster_id": signature_to_id[full_key],
                "signature_cluster_size": int(cluster_counter[full_key]),
                "equation_family": equation_family,
                "touches_param46_H0_like": bool(touches_46),
                "touches_param42": bool(touches_42),
                "bridges_param42_param46": bool(bridges_42_46),
                "nonzero_count": int(len(active)),
                "active_cols": active_key,
                "sign_pattern": sign_key,
                "y": float(y[i]),
                "baseline_residual": float(residual[i]),
                "abs_residual": float(abs(residual[i])),
                "row_l1": float(np.sum(np.abs(vals))) if len(vals) else 0.0,
                "row_l2": float(np.sqrt(np.sum(vals * vals))) if len(vals) else 0.0,
                "row_sum": float(np.sum(vals)) if len(vals) else 0.0,
            }
        )

    pair_rows = []
    for (a, b), count in pair_counter.items():
        pair_rows.append(
            {
                "param_a": int(a),
                "param_b": int(b),
                "row_cooccurrence_count": int(count),
                "cooccurs_with_param46": bool(a == H0_LIKE_PARAM_INDEX or b == H0_LIKE_PARAM_INDEX),
                "cooccurs_with_param42": bool(a == KNOWN_BRIDGE_PARAM_INDEX or b == KNOWN_BRIDGE_PARAM_INDEX),
                "is_42_46_edge": bool(set([a, b]) == set([KNOWN_BRIDGE_PARAM_INDEX, H0_LIKE_PARAM_INDEX])),
                "sum_abs_weight": float(pair_abs_weight[(a, b)]),
                "sum_signed_product": float(pair_signed_weight[(a, b)]),
            }
        )

    pair_rows = sorted(
        pair_rows,
        key=lambda r: (
            not r["is_42_46_edge"],
            not r["cooccurs_with_param46"],
            -r["row_cooccurrence_count"],
            r["param_a"],
            r["param_b"],
        ),
    )

    param_rows = []
    for j in range(X.shape[1]):
        param_rows.append(
            {
                "parameter_index": j,
                "touching_row_count": int(param_touch_counter[j]),
                "sum_abs_design_weight": float(param_abs_weight[j]),
                "is_param46_H0_like": bool(j == H0_LIKE_PARAM_INDEX),
                "is_param42_known_bridge": bool(j == KNOWN_BRIDGE_PARAM_INDEX),
            }
        )

    return rows, pair_rows, param_rows


def summarize_equation_families(row_rows):
    grouped = defaultdict(list)
    for row in row_rows:
        grouped[row["equation_family"]].append(row)

    out = []
    for family, rows in grouped.items():
        residuals = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        abs_residuals = np.abs(residuals)
        out.append(
            {
                "equation_family": family,
                "row_count": len(rows),
                "mean_residual": float(np.mean(residuals)),
                "median_residual": float(np.median(residuals)),
                "rms_residual": float(np.sqrt(np.mean(residuals * residuals))),
                "mean_abs_residual": float(np.mean(abs_residuals)),
                "max_abs_residual": float(np.max(abs_residuals)),
                "touches_param46_count": int(sum(1 for r in rows if r["touches_param46_H0_like"])),
                "bridges_42_46_count": int(sum(1 for r in rows if r["bridges_param42_param46"])),
            }
        )

    return sorted(out, key=lambda r: (-r["row_count"], r["equation_family"]))


def summarize_signature_clusters(row_rows):
    grouped = defaultdict(list)
    for row in row_rows:
        grouped[row["signature_cluster_id"]].append(row)

    out = []
    for cid, rows in grouped.items():
        residuals = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        first = rows[0]
        out.append(
            {
                "signature_cluster_id": cid,
                "size": len(rows),
                "equation_family": first["equation_family"],
                "touches_param46_H0_like": first["touches_param46_H0_like"],
                "touches_param42": first["touches_param42"],
                "bridges_param42_param46": first["bridges_param42_param46"],
                "first_row": int(rows[0]["observation_index"]),
                "last_row": int(rows[-1]["observation_index"]),
                "active_cols": first["active_cols"],
                "sign_pattern": first["sign_pattern"],
                "mean_residual": float(np.mean(residuals)),
                "rms_residual": float(np.sqrt(np.mean(residuals * residuals))),
                "mean_abs_residual": float(np.mean(np.abs(residuals))),
                "y_mean": float(np.mean([r["y"] for r in rows])),
                "y_std": float(np.std([r["y"] for r in rows])),
            }
        )

    return sorted(
        out,
        key=lambda r: (
            not r["bridges_param42_param46"],
            not r["touches_param46_H0_like"],
            -r["size"],
            r["signature_cluster_id"],
        ),
    )


def contiguous_blocks(row_rows):
    blocks = []
    if not row_rows:
        return blocks

    start = 0
    prev_key = (
        row_rows[0]["signature_cluster_id"],
        row_rows[0]["equation_family"],
        row_rows[0]["active_cols"],
    )

    for idx in range(1, len(row_rows)):
        key = (
            row_rows[idx]["signature_cluster_id"],
            row_rows[idx]["equation_family"],
            row_rows[idx]["active_cols"],
        )
        if key != prev_key:
            blocks.append(summarize_block(len(blocks), row_rows[start:idx]))
            start = idx
            prev_key = key

    blocks.append(summarize_block(len(blocks), row_rows[start:]))
    return blocks


def summarize_block(block_id, rows):
    residuals = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
    first = rows[0]
    return {
        "block_id": block_id,
        "start_row": int(rows[0]["observation_index"]),
        "end_row": int(rows[-1]["observation_index"]),
        "size": len(rows),
        "signature_cluster_id": first["signature_cluster_id"],
        "equation_family": first["equation_family"],
        "touches_param46_H0_like": first["touches_param46_H0_like"],
        "touches_param42": first["touches_param42"],
        "bridges_param42_param46": first["bridges_param42_param46"],
        "active_cols": first["active_cols"],
        "sign_pattern": first["sign_pattern"],
        "mean_residual": float(np.mean(residuals)),
        "rms_residual": float(np.sqrt(np.mean(residuals * residuals))),
        "mean_abs_residual": float(np.mean(np.abs(residuals))),
    }


def build_candidate_edge_vectors(row_rows):
    n = len(row_rows)
    vectors = {}

    def mask(fn):
        return np.asarray([1.0 if fn(r) else 0.0 for r in row_rows], dtype=float)

    vectors["edge_rows_touch_param46"] = mask(lambda r: r["touches_param46_H0_like"])
    vectors["edge_rows_touch_param42"] = mask(lambda r: r["touches_param42"])
    vectors["edge_rows_bridge_42_46"] = mask(lambda r: r["bridges_param42_param46"])
    vectors["edge_rows_two_parameter_difference"] = mask(
        lambda r: r["equation_family"] == "two_parameter_difference_or_relative_constraint"
    )
    vectors["edge_rows_dense_calibration"] = mask(
        lambda r: r["equation_family"] == "dense_calibration_or_ceph_sn_relation"
    )
    vectors["edge_rows_sparse_ladder"] = mask(
        lambda r: r["equation_family"] == "sparse_ladder_relation"
    )

    abs_residual = np.asarray([r["abs_residual"] for r in row_rows], dtype=float)
    threshold = float(np.percentile(abs_residual, 90))
    vectors["edge_rows_high_residual_pressure_top10pct"] = (abs_residual >= threshold).astype(float)

    vectors["edge_bridge_42_46_high_residual_pressure"] = (
        vectors["edge_rows_bridge_42_46"] * vectors["edge_rows_high_residual_pressure_top10pct"]
    )

    stats = []
    for name, vals in vectors.items():
        stats.append(
            {
                "candidate_edge_vector": name,
                "nonzero_count": int(np.sum(vals > 0.5)),
                "fraction": float(np.mean(vals > 0.5)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "note": "Candidate only. This is for the next likelihood test, not theory promotion.",
            }
        )

    preview_rows = []
    for i in range(n):
        row = {"observation_index": i}
        for name, vals in vectors.items():
            row[name] = float(vals[i])
        preview_rows.append(row)

    write_csv(OUTDIR / "candidate_equation_edge_vectors_stats_v1.csv", stats)
    write_csv(OUTDIR / "candidate_equation_edge_vectors_by_row_v1.csv", preview_rows)

    return vectors, stats


def plot_outputs(row_rows, pair_rows, family_rows, cluster_rows, baseline):
    residuals = np.asarray([r["baseline_residual"] for r in row_rows], dtype=float)
    nz = np.asarray([r["nonzero_count"] for r in row_rows], dtype=float)
    bridge_mask = np.asarray([r["bridges_param42_param46"] for r in row_rows], dtype=bool)
    touch46_mask = np.asarray([r["touches_param46_H0_like"] for r in row_rows], dtype=bool)

    plt.figure(figsize=(10, 6))
    plt.hist(nz, bins=60)
    plt.xlabel("nonzero parameter count per equation row")
    plt.ylabel("count")
    plt.title("Compact ladder equation row sparsity")
    plt.tight_layout()
    plt.savefig(OUTDIR / "equation_row_sparsity_hist_v1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(residuals, linewidth=0.8, label="baseline residual")
    if np.any(touch46_mask):
        plt.scatter(np.where(touch46_mask)[0], residuals[touch46_mask], s=8, label="touches param46")
    if np.any(bridge_mask):
        plt.scatter(np.where(bridge_mask)[0], residuals[bridge_mask], s=10, label="bridges 42-46")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("compact row index")
    plt.ylabel("baseline residual")
    plt.title("Equation-edge residual map")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTDIR / "equation_edge_residual_map_v1.png", dpi=160)
    plt.close()

    top_pairs = pair_rows[:40]
    labels = [f"{r['param_a']}-{r['param_b']}" for r in top_pairs]
    counts = [r["row_cooccurrence_count"] for r in top_pairs]

    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(labels)), counts)
    plt.xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("row co-occurrence count")
    plt.title("Top parameter-pair equation edges")
    plt.tight_layout()
    plt.savefig(OUTDIR / "top_parameter_pair_edges_v1.png", dpi=160)
    plt.close()

    fam_labels = [r["equation_family"] for r in family_rows]
    fam_counts = [r["row_count"] for r in family_rows]
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(fam_labels)), fam_counts)
    plt.xticks(np.arange(len(fam_labels)), fam_labels, rotation=35, ha="right")
    plt.ylabel("row count")
    plt.title("Equation family counts")
    plt.tight_layout()
    plt.savefig(OUTDIR / "equation_family_counts_v1.png", dpi=160)
    plt.close()

    beta = baseline["beta"]
    beta_err = baseline["beta_err"]
    x = np.arange(len(beta))
    plt.figure(figsize=(11, 6))
    plt.errorbar(x, beta, yerr=beta_err, fmt="o", linewidth=1)
    plt.axvline(H0_LIKE_PARAM_INDEX, linewidth=1)
    plt.axvline(KNOWN_BRIDGE_PARAM_INDEX, linewidth=1)
    plt.xlabel("parameter index")
    plt.ylabel("beta")
    plt.title("GLS parameter map with 42 and 46 marked")
    plt.tight_layout()
    plt.savefig(OUTDIR / "parameter_map_42_46_v1.png", dpi=160)
    plt.close()


def decide_status(row_rows, pair_rows, cluster_rows, code_hits):
    bridge_rows = [r for r in row_rows if r["bridges_param42_param46"]]
    touch46_rows = [r for r in row_rows if r["touches_param46_H0_like"]]
    edge_42_46 = [r for r in pair_rows if r["is_42_46_edge"]]
    cluster_277_bridge = [
        c for c in cluster_rows
        if c["bridges_param42_param46"] and c["size"] == 277
    ]

    code_h0_hits = [
        r for r in code_hits
        if "fivelogH0" in r.get("line", "") or "fivelogH0" in r.get("context", "")
    ]

    if cluster_277_bridge and edge_42_46 and code_h0_hits:
        return (
            "equation_edge_42_46_hubble_flow_bridge_recovered",
            9,
            "Build strict equation-edge likelihood v2 using 42-46 bridge rows and high-pressure edge vectors.",
        )

    if bridge_rows and edge_42_46:
        return (
            "equation_edge_42_46_bridge_recovered_without_full_code_label",
            8,
            "Use bridge rows cautiously; recover stronger code/parameter labels before likelihood promotion.",
        )

    if touch46_rows:
        return (
            "equation_edges_touching_param46_recovered",
            7,
            "Build v1.1 to identify which param46 edges are Hubble-flow versus calibration edges.",
        )

    return (
        "equation_edge_recovery_partial_no_h0_bridge",
        6,
        "Matrix edge map created, but no reliable H0-like boundary bridge was recovered.",
    )


def main():
    print("")
    print("TAIRID SH0ES compact ladder equation-edge recovery v1 starting.")
    print("Boundary: equation-edge recovery only; not a TAIRID fit.")
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

    aux_downloads = []
    for repo_path in AUX_FILES:
        result = download_repo_path(repo_path, safe_name(repo_path))
        aux_downloads.append(result)
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

    write_csv(OUTDIR / "equation_edge_download_ledger_v1.csv", ledger)
    write_json(
        OUTDIR / "equation_edge_download_attempts_v1.json",
        {"compact": downloads, "auxiliary": aux_downloads},
    )

    code_hits, previews = text_context_search(aux_downloads)

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

    write_json(OUTDIR / "equation_edge_parse_meta_v1.json", parse_meta)
    write_json(OUTDIR / "equation_edge_parse_errors_v1.json", parse_errors)

    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]):
        summary = {
            "test_name": "TAIRID SH0ES compact ladder equation-edge recovery v1",
            "boundary": "Matrix parse/download failure. No edge recovery result.",
            "final_status": "equation_edge_matrix_parse_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix retrieval/parsing before edge recovery.",
            "parse_errors": parse_errors,
            "downloads": downloads,
        }
        write_json(OUTDIR / "equation_edge_recovery_v1_summary.json", summary)
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

    row_rows, pair_rows, param_rows = build_row_edge_map(X, y, baseline["residual"])
    family_rows = summarize_equation_families(row_rows)
    cluster_rows = summarize_signature_clusters(row_rows)
    block_rows = contiguous_blocks(row_rows)
    candidate_vectors, candidate_vector_stats = build_candidate_edge_vectors(row_rows)

    write_csv(OUTDIR / "equation_edge_row_map_v1.csv", row_rows)
    write_csv(OUTDIR / "equation_edge_parameter_pair_graph_v1.csv", pair_rows)
    write_csv(OUTDIR / "equation_edge_parameter_touch_summary_v1.csv", param_rows)
    write_csv(OUTDIR / "equation_edge_family_summary_v1.csv", family_rows)
    write_csv(OUTDIR / "equation_edge_signature_clusters_v1.csv", cluster_rows)
    write_csv(OUTDIR / "equation_edge_contiguous_blocks_v1.csv", block_rows)

    plot_outputs(row_rows, pair_rows, family_rows, cluster_rows, baseline)

    final_status, readiness_score, next_wall = decide_status(row_rows, pair_rows, cluster_rows, code_hits)

    bridge_rows = [r for r in row_rows if r["bridges_param42_param46"]]
    touch46_rows = [r for r in row_rows if r["touches_param46_H0_like"]]
    edge_42_46 = [r for r in pair_rows if r["is_42_46_edge"]]
    bridge_clusters = [r for r in cluster_rows if r["bridges_param42_param46"]]
    touch46_clusters = [r for r in cluster_rows if r["touches_param46_H0_like"]]

    summary = {
        "test_name": "TAIRID SH0ES compact ladder equation-edge recovery v1",
        "boundary": (
            "Equation-edge recovery only. Not proof of TAIRID, not a cosmology fit, "
            "not H0 resolution, and not a boundary-gate likelihood."
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
            "normal_condition_estimate": baseline["normal_condition_estimate"],
            "residual_mean": float(np.mean(baseline["residual"])),
            "residual_std": float(np.std(baseline["residual"])),
            "residual_rms": float(np.sqrt(np.mean(baseline["residual"] ** 2))),
            "parameter_count": int(len(baseline["beta"])),
            "param46_fivelogH0": float(baseline["beta"][46]) if len(baseline["beta"]) > 46 else None,
            "param46_H0_like": float(10.0 ** (baseline["beta"][46] / 5.0)) if len(baseline["beta"]) > 46 else None,
            "param42_value": float(baseline["beta"][42]) if len(baseline["beta"]) > 42 else None,
            "beta_preview_first_20": [float(v) for v in baseline["beta"][:20]],
            "beta_err_preview_first_20": [float(v) for v in baseline["beta_err"][:20]],
        },
        "edge_recovery_counts": {
            "row_count": len(row_rows),
            "parameter_count": X.shape[1],
            "parameter_pair_edge_count": len(pair_rows),
            "equation_family_count": len(family_rows),
            "signature_cluster_count": len(cluster_rows),
            "contiguous_block_count": len(block_rows),
            "rows_touching_param46": len(touch46_rows),
            "rows_touching_param42": int(sum(1 for r in row_rows if r["touches_param42"])),
            "rows_bridging_42_46": len(bridge_rows),
            "clusters_touching_param46": len(touch46_clusters),
            "clusters_bridging_42_46": len(bridge_clusters),
            "code_context_hits": len(code_hits),
            "auxiliary_previews": len(previews),
        },
        "top_42_46_edge": edge_42_46[:5],
        "bridge_42_46_clusters": bridge_clusters[:20],
        "touch46_clusters": touch46_clusters[:20],
        "top_parameter_pair_edges": pair_rows[:40],
        "equation_family_summary": family_rows,
        "candidate_edge_vector_stats": candidate_vector_stats,
        "output_files": {
            "summary_json": str(OUTDIR / "equation_edge_recovery_v1_summary.json"),
            "summary_txt": str(OUTDIR / "equation_edge_recovery_v1_summary.txt"),
            "row_map_csv": str(OUTDIR / "equation_edge_row_map_v1.csv"),
            "parameter_pair_graph_csv": str(OUTDIR / "equation_edge_parameter_pair_graph_v1.csv"),
            "parameter_touch_summary_csv": str(OUTDIR / "equation_edge_parameter_touch_summary_v1.csv"),
            "family_summary_csv": str(OUTDIR / "equation_edge_family_summary_v1.csv"),
            "signature_clusters_csv": str(OUTDIR / "equation_edge_signature_clusters_v1.csv"),
            "contiguous_blocks_csv": str(OUTDIR / "equation_edge_contiguous_blocks_v1.csv"),
            "candidate_edge_vectors_stats_csv": str(OUTDIR / "candidate_equation_edge_vectors_stats_v1.csv"),
            "candidate_edge_vectors_by_row_csv": str(OUTDIR / "candidate_equation_edge_vectors_by_row_v1.csv"),
            "code_context_hits_csv": str(OUTDIR / "equation_edge_code_context_hits_v1.csv"),
            "plots": [
                str(OUTDIR / "equation_row_sparsity_hist_v1.png"),
                str(OUTDIR / "equation_edge_residual_map_v1.png"),
                str(OUTDIR / "top_parameter_pair_edges_v1.png"),
                str(OUTDIR / "equation_family_counts_v1.png"),
                str(OUTDIR / "parameter_map_42_46_v1.png"),
            ],
        },
        "interpretation": {
            "what_success_means": (
                "The compact ladder equation edge linking parameter 42 and H0-like parameter 46 was recovered. "
                "The next test can use equation-edge vectors instead of blunt row masks."
            ),
            "what_partial_success_means": (
                "Rows touching parameter 46 were recovered, but the boundary bridge still needs sharper labeling."
            ),
            "what_failure_means": (
                "The compact matrix is accessible, but this pass did not identify a useful H0-side edge boundary."
            ),
            "truth_boundary": (
                "This is not a TAIRID fit. It only prepares a stricter equation-edge likelihood test."
            ),
        },
    }

    write_json(OUTDIR / "equation_edge_recovery_v1_summary.json", summary)

    with open(OUTDIR / "equation_edge_recovery_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES compact ladder equation-edge recovery v1\n\n")
        f.write("Boundary: equation-edge recovery only. Not proof. Not a cosmology fit.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Edge recovery counts:\n")
        f.write(json.dumps(summary["edge_recovery_counts"], indent=2, default=json_default) + "\n\n")

        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")

        f.write("Top 42-46 edge:\n")
        f.write(json.dumps(edge_42_46[:5], indent=2, default=json_default) + "\n\n")

        f.write("Bridge 42-46 clusters:\n")
        f.write(json.dumps(bridge_clusters[:20], indent=2, default=json_default) + "\n\n")

        f.write("Equation family summary:\n")
        f.write(json.dumps(family_rows, indent=2, default=json_default) + "\n\n")

        f.write("Candidate edge vector stats:\n")
        f.write(json.dumps(candidate_vector_stats, indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not test H0 resolution.\n")
        f.write("- This prepares an equation-edge likelihood test.\n")

    print("")
    print("TAIRID SH0ES compact ladder equation-edge recovery v1 complete.")
    print("Created:")
    print("  tairid_shoes_equation_edge_recovery_v1_outputs/equation_edge_recovery_v1_summary.json")
    print("  tairid_shoes_equation_edge_recovery_v1_outputs/equation_edge_recovery_v1_summary.txt")
    print("  tairid_shoes_equation_edge_recovery_v1_outputs/equation_edge_row_map_v1.csv")
    print("  tairid_shoes_equation_edge_recovery_v1_outputs/equation_edge_parameter_pair_graph_v1.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is equation-edge recovery only.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
