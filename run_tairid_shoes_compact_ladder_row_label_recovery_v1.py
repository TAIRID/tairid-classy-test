#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder row-label recovery v1.

Purpose:
The compact ladder boundary-gate v1 topology screen did not support the
abstract topology-only gate. The strongest next wall is row-label recovery.

This test asks:

Can we recover enough explicit compact-ladder structure to identify actual
row/parameter classes before testing a calibrator-boundary gate again?

Inputs:
- allc = C covariance matrix
- alll = L equation/design matrix
- ally = y data vector
- SH0ES auxiliary code/text/dat files

Outputs:
- parameter-order recovery attempts
- code/search evidence around y, L, C, parameters, Cepheids, SNe, anchors, hosts
- row structural clusters from L.T
- column structural features
- baseline GLS sanity solve
- guessed compact row classes
- candidate boundary vectors for v2, but no theory fit yet

Boundary:
This is not proof of TAIRID.
This is not a cosmology fit.
This is not H0 resolution.
This is a row-label / structure-recovery test.
"""

import csv
import json
import math
import re
import hashlib
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve


OUTDIR = Path("tairid_shoes_compact_ladder_row_label_recovery_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

OWNER = "PantheonPlusSH0ES"
REPO = "DataRelease"
BRANCH = "main"

GITHUB_CONTENTS_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/SH0ES_Data?ref={BRANCH}"

COMPACT_FILES = {
    "allc": "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits",
    "alll": "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits",
    "ally": "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits",
}

TEXT_PATTERNS = [
    "allc", "alll", "ally", "fits", "cov", "covariance", "lstsq",
    "theta", "param", "parameter", "par", "names", "labels",
    "cepheid", "ceph", "shoes", "pantheon", "supernova", "sne", "sn ",
    "anchor", "host", "calibrator", "hubble", "muhat", "intercept",
    "L", "C", "Y", "distance", "period", "metal", "zeropoint", "zero",
]

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


def fetch_url(url, timeout=900, accept="*/*"):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-SH0ES-row-label-recovery-v1",
            "Accept": accept,
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        status = getattr(response, "status", None)

    return data, final_url, content_type, status


def is_lfs_pointer(data):
    head = data[:200].decode("utf-8", errors="replace")
    return "version https://git-lfs.github.com/spec/v1" in head and "oid sha256:" in head


def parse_lfs_pointer(data):
    text = data.decode("utf-8", errors="replace")
    out = {"raw_text": text}

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


def download_repo_path(repo_path, required=False):
    local = DOWNLOAD_DIR / safe_name(repo_path)
    attempts = []
    pointer_info = None

    for cand in candidate_urls(repo_path):
        try:
            data, final_url, content_type, status = fetch_url(cand["url"])
            attempt = {
                "repo_path": repo_path,
                "candidate_kind": cand["kind"],
                "url": cand["url"],
                "final_url": final_url,
                "http_status": status,
                "content_type": content_type,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }

            if is_lfs_pointer(data):
                p = parse_lfs_pointer(data)
                pointer_info = p
                attempt.update(p)
                attempt["status"] = "git_lfs_pointer_not_payload"
                attempts.append(attempt)
                continue

            local.write_bytes(data)
            attempt["status"] = "downloaded_real_payload"
            attempt["local_path"] = str(local)
            attempt["file_sha256"] = sha256_file(local)
            attempts.append(attempt)

            return {
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
                    "repo_path": repo_path,
                    "candidate_kind": cand["kind"],
                    "url": cand["url"],
                    "status": "download_failed",
                    "error": str(exc),
                }
            )

    return {
        "repo_path": repo_path,
        "status": "failed_required" if required else "failed_optional",
        "local_path": None,
        "pointer_info": pointer_info,
        "attempts": attempts,
    }


def github_folder_inventory():
    try:
        data, final_url, content_type, status = fetch_url(GITHUB_CONTENTS_URL, accept="application/json")
        obj = json.loads(data.decode("utf-8", errors="replace"))

        rows = []
        for item in obj:
            rows.append(
                {
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "size": item.get("size"),
                    "type": item.get("type"),
                    "download_url": item.get("download_url"),
                    "html_url": item.get("html_url"),
                }
            )

        write_json(OUTDIR / "shoes_data_folder_inventory_raw_v1.json", obj)
        write_csv(OUTDIR / "shoes_data_folder_inventory_v1.csv", rows)

        return rows, {
            "status": "ok",
            "count": len(rows),
            "final_url": final_url,
            "content_type": content_type,
            "http_status": status,
        }

    except Exception as exc:
        return [], {"status": "failed", "error": str(exc)}


def is_textish(path):
    lower = path.name.lower()
    return lower.endswith((".txt", ".md", ".py", ".dat", ".csv", ".tsv", ".ipynb"))


def extract_numeric_fits(path):
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


def stable_cholesky(C):
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
    Cinv_res = cho_solve(c_factor, residual, check_finite=False)

    chi2_val = float(residual.T @ Cinv_res)
    dof = int(len(y) - X.shape[1])
    beta_err = np.sqrt(np.maximum(np.diag(normal_inv), 0.0))

    return {
        "beta": beta,
        "beta_err": beta_err,
        "residual": residual,
        "normal": normal,
        "normal_inv": normal_inv,
        "chi2": chi2_val,
        "dof": dof,
        "reduced_chi2": float(chi2_val / dof) if dof > 0 else float("nan"),
        "normal_condition_estimate": float(np.linalg.cond(normal)),
    }


def text_search_and_parameter_lists(download_rows):
    search_rows = []
    preview_rows = []
    parameter_list_candidates = []

    for item in download_rows:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["local_path"])

        if not is_textish(path):
            continue

        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue

        lines = text.splitlines()
        preview_path = OUTDIR / f"preview_{safe_name(item['repo_path'])}.txt"
        preview_path.write_text("\n".join(lines[:180]), encoding="utf-8")

        preview_rows.append(
            {
                "repo_path": item["repo_path"],
                "local_path": str(path),
                "line_count": len(lines),
                "char_count": len(text),
                "preview_file": str(preview_path),
            }
        )

        lower_patterns = [p.lower() for p in TEXT_PATTERNS]

        for i, line in enumerate(lines, start=1):
            ll = line.lower()
            hit_terms = [p for p in lower_patterns if p in ll]

            if hit_terms:
                search_rows.append(
                    {
                        "repo_path": item["repo_path"],
                        "line_number": i,
                        "hit_terms": " | ".join(hit_terms[:12]),
                        "line": line[:600],
                    }
                )

        # Recover quoted lists that may contain parameter names.
        for i, line in enumerate(lines, start=1):
            if "[" not in line or "]" not in line:
                continue

            quoted = re.findall(r"['\"]([^'\"]+)['\"]", line)

            if len(quoted) >= 5:
                parameter_list_candidates.append(
                    {
                        "repo_path": item["repo_path"],
                        "line_number": i,
                        "count": len(quoted),
                        "line": line[:1000],
                        "items_json": json.dumps(quoted),
                    }
                )

        # Multiline quoted lists.
        joined = "\n".join(lines)
        for match in re.finditer(r"([A-Za-z0-9_]*name[A-Za-z0-9_]*|labels?|pars?|params?)\s*=\s*\[(.*?)\]", joined, flags=re.I | re.S):
            body = match.group(2)
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", body)

            if len(quoted) >= 5:
                prefix_text = joined[: match.start()]
                line_number = prefix_text.count("\n") + 1
                parameter_list_candidates.append(
                    {
                        "repo_path": item["repo_path"],
                        "line_number": line_number,
                        "count": len(quoted),
                        "line": match.group(0)[:1000],
                        "items_json": json.dumps(quoted),
                    }
                )

    write_csv(OUTDIR / "row_label_text_search_hits_v1.csv", search_rows)
    write_csv(OUTDIR / "row_label_text_previews_v1.csv", preview_rows)
    write_csv(OUTDIR / "parameter_list_candidates_v1.csv", parameter_list_candidates)

    return search_rows, preview_rows, parameter_list_candidates


def parse_lstsq_results(download_rows):
    candidates = []

    for item in download_rows:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["local_path"])

        if "lstsq" not in path.name.lower():
            continue

        text = path.read_text(errors="replace")
        lines = text.splitlines()

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            # Try forms like: name value error, or index value error.
            tokens = stripped.replace("=", " ").replace(",", " ").split()
            nums = []
            words = []

            for tok in tokens:
                try:
                    nums.append(float(tok))
                except Exception:
                    words.append(tok)

            if nums:
                candidates.append(
                    {
                        "line_number": i,
                        "line": stripped[:500],
                        "word_prefix": " ".join(words[:4]),
                        "numeric_count": len(nums),
                        "numbers_json": json.dumps(nums[:8]),
                    }
                )

    write_csv(OUTDIR / "lstsq_results_parsed_lines_v1.csv", candidates)
    return candidates


def signature_from_row(row):
    active = np.where(np.abs(row) > 1.0e-12)[0]
    signs = np.sign(row[active]).astype(int)

    return {
        "active": active,
        "signs": signs,
        "active_key": ",".join(str(int(i)) for i in active),
        "sign_key": ",".join(str(int(s)) for s in signs),
        "full_key": ",".join(f"{int(i)}:{int(s)}" for i, s in zip(active, signs)),
    }


def guess_row_class(nz, active_cols, signs, y_value, residual, beta, beta_err):
    if nz == 0:
        return "empty_or_padding_row"

    if nz == 1:
        return "single_parameter_prior_or_anchor_constraint"

    if nz == 2 and np.any(signs > 0) and np.any(signs < 0):
        return "two_parameter_difference_or_relative_constraint"

    if nz <= 4 and np.any(signs > 0) and np.any(signs < 0):
        return "sparse_ladder_relation"

    if nz <= 4:
        return "sparse_measurement_or_constraint"

    distance_like = (beta > 20.0) & (beta < 40.0)
    high_uncert = beta_err >= np.nanpercentile(beta_err, 75.0)

    if len(active_cols) > 0:
        distance_weight = float(np.sum(distance_like[active_cols]) / len(active_cols))
        high_uncert_weight = float(np.sum(high_uncert[active_cols]) / len(active_cols))
    else:
        distance_weight = 0.0
        high_uncert_weight = 0.0

    if distance_weight >= 0.40 and nz >= 5:
        return "distance_ladder_measurement_candidate"

    if high_uncert_weight >= 0.40 and nz >= 5:
        return "high_uncertainty_boundary_candidate"

    if nz >= 8:
        return "dense_ceph_sn_or_calibration_relation"

    return "medium_ladder_measurement"


def row_and_cluster_features(X, y, residual, beta, beta_err):
    rows = []
    cluster_counter = Counter()
    signature_to_id = {}
    next_id = 0

    for i in range(X.shape[0]):
        row = X[i, :]
        sig = signature_from_row(row)

        key = sig["full_key"]

        if key not in signature_to_id:
            signature_to_id[key] = next_id
            next_id += 1

        cluster_id = signature_to_id[key]
        cluster_counter[cluster_id] += 1

        active = sig["active"]
        vals = row[active]

        row_class = guess_row_class(
            len(active),
            active,
            sig["signs"],
            y[i],
            residual[i],
            beta,
            beta_err,
        )

        rows.append(
            {
                "observation_index": i,
                "signature_cluster_id": cluster_id,
                "guessed_row_class": row_class,
                "y": float(y[i]),
                "baseline_residual": float(residual[i]),
                "nonzero_count": int(len(active)),
                "active_cols": sig["active_key"],
                "sign_pattern": sig["sign_key"],
                "row_l1": float(np.sum(np.abs(vals))) if len(vals) else 0.0,
                "row_l2": float(np.sqrt(np.sum(vals * vals))) if len(vals) else 0.0,
                "row_max_abs": float(np.max(np.abs(vals))) if len(vals) else 0.0,
                "row_sum": float(np.sum(vals)) if len(vals) else 0.0,
                "row_abs_sum": float(np.sum(np.abs(vals))) if len(vals) else 0.0,
            }
        )

    for r in rows:
        r["signature_cluster_size"] = int(cluster_counter[r["signature_cluster_id"]])

    cluster_rows = []
    grouped = defaultdict(list)

    for r in rows:
        grouped[r["signature_cluster_id"]].append(r)

    for cid, members in grouped.items():
        classes = Counter(m["guessed_row_class"] for m in members)
        nz_vals = [m["nonzero_count"] for m in members]
        y_vals = [m["y"] for m in members]
        res_vals = [m["baseline_residual"] for m in members]

        cluster_rows.append(
            {
                "signature_cluster_id": cid,
                "size": len(members),
                "dominant_class_guess": classes.most_common(1)[0][0],
                "class_counts_json": json.dumps(dict(classes)),
                "nonzero_count_median": float(np.median(nz_vals)),
                "nonzero_count_min": int(np.min(nz_vals)),
                "nonzero_count_max": int(np.max(nz_vals)),
                "y_mean": float(np.mean(y_vals)),
                "y_std": float(np.std(y_vals)),
                "residual_mean": float(np.mean(res_vals)),
                "residual_rms": float(np.sqrt(np.mean(np.asarray(res_vals) ** 2))),
                "example_active_cols": members[0]["active_cols"],
                "example_sign_pattern": members[0]["sign_pattern"],
            }
        )

    cluster_rows = sorted(cluster_rows, key=lambda r: (-r["size"], r["signature_cluster_id"]))

    return rows, cluster_rows


def column_features(X, beta, beta_err):
    rows = []

    for j in range(X.shape[1]):
        col = X[:, j]
        active = np.abs(col) > 1.0e-12
        vals = col[active]

        rows.append(
            {
                "parameter_index": j,
                "beta": float(beta[j]),
                "beta_err": float(beta_err[j]),
                "abs_beta_over_err": float(abs(beta[j]) / max(beta_err[j], EPS)),
                "nonzero_rows": int(np.sum(active)),
                "nonzero_fraction": float(np.mean(active)),
                "col_min": float(np.min(vals)) if len(vals) else 0.0,
                "col_max": float(np.max(vals)) if len(vals) else 0.0,
                "col_mean_nonzero": float(np.mean(vals)) if len(vals) else 0.0,
                "col_std_nonzero": float(np.std(vals)) if len(vals) else 0.0,
                "positive_count": int(np.sum(col > 1.0e-12)),
                "negative_count": int(np.sum(col < -1.0e-12)),
                "beta_family_guess": guess_parameter_family(beta[j], beta_err[j], np.sum(active), X.shape[0]),
            }
        )

    return rows


def guess_parameter_family(beta, beta_err, nonzero_rows, n_obs):
    if 20.0 <= beta <= 40.0:
        return "distance_modulus_like_parameter"

    if abs(beta) <= 0.20 and beta_err <= 0.20:
        return "small_global_or_nuisance_parameter"

    if nonzero_rows <= 10:
        return "sparse_anchor_or_prior_parameter"

    if nonzero_rows >= 0.40 * n_obs:
        return "global_or_broad_ladder_parameter"

    if beta_err >= 1.0:
        return "weakly_constrained_or_boundary_parameter"

    return "intermediate_ladder_parameter"


def build_candidate_boundary_vectors(row_rows, cluster_rows):
    n = len(row_rows)
    vectors = {}

    classes = [r["guessed_row_class"] for r in row_rows]
    nz = np.asarray([r["nonzero_count"] for r in row_rows], dtype=float)
    res = np.asarray([r["baseline_residual"] for r in row_rows], dtype=float)
    yvals = np.asarray([r["y"] for r in row_rows], dtype=float)
    cluster_size = np.asarray([r["signature_cluster_size"] for r in row_rows], dtype=float)

    def zscore(v):
        v = np.asarray(v, dtype=float)
        sd = np.std(v)
        if sd <= 1.0e-12:
            return np.zeros_like(v)
        return (v - np.mean(v)) / sd

    vectors["row_class_single_parameter_prior"] = np.asarray([
        1.0 if c == "single_parameter_prior_or_anchor_constraint" else 0.0 for c in classes
    ])

    vectors["row_class_two_parameter_difference"] = np.asarray([
        1.0 if c == "two_parameter_difference_or_relative_constraint" else 0.0 for c in classes
    ])

    vectors["row_class_dense_relation"] = np.asarray([
        1.0 if c == "dense_ceph_sn_or_calibration_relation" else 0.0 for c in classes
    ])

    vectors["row_class_distance_ladder_candidate"] = np.asarray([
        1.0 if c == "distance_ladder_measurement_candidate" else 0.0 for c in classes
    ])

    vectors["row_nonzero_count_z"] = zscore(nz)
    vectors["row_cluster_size_z"] = zscore(cluster_size)
    vectors["row_y_value_z"] = zscore(yvals)
    vectors["baseline_residual_z"] = zscore(res)

    vector_rows = []

    for name, vals in vectors.items():
        vals = np.asarray(vals, dtype=float)
        vector_rows.append(
            {
                "candidate_vector": name,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "nonzero_count": int(np.sum(np.abs(vals) > 1.0e-12)),
                "note": "Candidate only. Do not use for theory promotion until row-label mapping is validated.",
            }
        )

    write_csv(OUTDIR / "candidate_boundary_vectors_stats_v1.csv", vector_rows)

    preview_rows = []
    for i in range(n):
        row = {"observation_index": i}
        for name, vals in vectors.items():
            row[name] = float(vals[i])
        preview_rows.append(row)

    write_csv(OUTDIR / "candidate_boundary_vectors_by_row_v1.csv", preview_rows)

    return vectors, vector_rows


def plot_outputs(row_rows, cluster_rows, col_rows):
    nz = np.asarray([r["nonzero_count"] for r in row_rows], dtype=float)
    residual = np.asarray([r["baseline_residual"] for r in row_rows], dtype=float)
    yvals = np.asarray([r["y"] for r in row_rows], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.hist(nz, bins=60)
    plt.xlabel("nonzero parameter count per observation row")
    plt.ylabel("count")
    plt.title("Compact ladder row sparsity")
    plt.tight_layout()
    plt.savefig(OUTDIR / "row_nonzero_count_hist_v1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(residual, linewidth=0.8)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("observation index")
    plt.ylabel("baseline residual")
    plt.title("Baseline compact ladder residual by row index")
    plt.tight_layout()
    plt.savefig(OUTDIR / "baseline_residual_by_row_v1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(yvals, linewidth=0.8)
    plt.xlabel("observation index")
    plt.ylabel("y")
    plt.title("Compact ladder y vector by row index")
    plt.tight_layout()
    plt.savefig(OUTDIR / "y_vector_by_row_v1.png", dpi=160)
    plt.close()

    top_clusters = cluster_rows[:30]
    labels = [str(r["signature_cluster_id"]) for r in top_clusters]
    sizes = [r["size"] for r in top_clusters]

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(labels)), sizes)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("signature cluster id")
    plt.ylabel("cluster size")
    plt.title("Top compact ladder row signature clusters")
    plt.tight_layout()
    plt.savefig(OUTDIR / "top_signature_clusters_v1.png", dpi=160)
    plt.close()

    param_idx = [r["parameter_index"] for r in col_rows]
    beta = [r["beta"] for r in col_rows]
    beta_err = [r["beta_err"] for r in col_rows]

    plt.figure(figsize=(11, 6))
    plt.errorbar(param_idx, beta, yerr=beta_err, fmt="o", linewidth=1)
    plt.xlabel("parameter index")
    plt.ylabel("beta")
    plt.title("Compact ladder GLS parameters")
    plt.tight_layout()
    plt.savefig(OUTDIR / "gls_parameter_beta_with_errors_v1.png", dpi=160)
    plt.close()


def decide_status(parameter_candidates, row_rows, cluster_rows, col_rows, text_hits):
    exact_47_lists = [r for r in parameter_candidates if int(r.get("count", 0)) == 47]
    near_lists = [r for r in parameter_candidates if 35 <= int(r.get("count", 0)) <= 60]

    class_counts = Counter(r["guessed_row_class"] for r in row_rows)
    cluster_count = len(cluster_rows)

    has_code_hits = len(text_hits) > 20
    has_structure = len(row_rows) > 0 and cluster_count > 5 and len(col_rows) == 47

    if exact_47_lists and has_structure:
        return (
            "compact_ladder_parameter_order_recovered_row_classes_structural",
            8,
            "Use recovered parameter order plus row structural clusters to build explicit boundary-vector v2.",
        )

    if near_lists and has_structure:
        return (
            "compact_ladder_parameter_order_partially_recovered",
            7,
            "Inspect candidate parameter lists manually, then build row-label recovery v1.1 or boundary-vector v2.",
        )

    if has_structure and has_code_hits:
        return (
            "compact_ladder_row_structure_recovered_labels_still_partial",
            7,
            "Use structural clusters and code-search hits to target explicit row-label recovery in v1.1.",
        )

    if has_structure:
        return (
            "compact_ladder_structure_recovered_without_human_labels",
            6,
            "The matrix structure is mapped, but row labels are still missing. Do not build a boundary likelihood yet.",
        )

    return (
        "compact_ladder_row_label_recovery_failed",
        5,
        "Resolve parser or download issue before continuing.",
    )


def main():
    print("")
    print("TAIRID SH0ES compact ladder row-label recovery v1 starting.")
    print("Boundary: row-label / structure recovery only; not a TAIRID fit.")
    print("")

    folder_rows, folder_status = github_folder_inventory()

    repo_paths = sorted({r["path"] for r in folder_rows if r.get("path")})
    for required_path in COMPACT_FILES.values():
        if required_path not in repo_paths:
            repo_paths.append(required_path)

    downloads = []
    downloads_by_path = {}

    for repo_path in repo_paths:
        required = repo_path in COMPACT_FILES.values()
        result = download_repo_path(repo_path, required=required)
        downloads.append(result)
        downloads_by_path[repo_path] = result

    download_ledger = []
    for d in downloads:
        download_ledger.append(
            {
                "repo_path": d.get("repo_path"),
                "status": d.get("status"),
                "local_path": d.get("local_path"),
                "bytes": d.get("bytes"),
                "sha256": d.get("sha256"),
                "pointer_declared_size": (d.get("pointer_info") or {}).get("declared_size"),
                "pointer_oid_sha256": (d.get("pointer_info") or {}).get("oid_sha256"),
                "attempt_count": len(d.get("attempts", [])),
            }
        )

    write_csv(OUTDIR / "row_label_recovery_download_ledger_v1.csv", download_ledger)
    write_json(OUTDIR / "row_label_recovery_download_attempts_v1.json", downloads)

    text_hits, preview_rows, parameter_candidates = text_search_and_parameter_lists(downloads)
    lstsq_lines = parse_lstsq_results(downloads)

    parsed = {}
    parse_meta = {}
    parse_errors = []

    for label, repo_path in COMPACT_FILES.items():
        result = downloads_by_path.get(repo_path)

        if not result or result.get("status") != "downloaded":
            parse_errors.append(
                {
                    "label": label,
                    "repo_path": repo_path,
                    "status": "not_downloaded",
                }
            )
            continue

        try:
            arr, meta = extract_numeric_fits(Path(result["local_path"]))
            parsed[label] = arr
            parse_meta[label] = meta
        except Exception as exc:
            parse_errors.append(
                {
                    "label": label,
                    "repo_path": repo_path,
                    "status": "parse_failed",
                    "error": str(exc),
                }
            )

    write_json(OUTDIR / "row_label_recovery_fits_parse_meta_v1.json", parse_meta)
    write_json(OUTDIR / "row_label_recovery_parse_errors_v1.json", parse_errors)

    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]):
        summary = {
            "test_name": "TAIRID SH0ES compact ladder row-label recovery v1",
            "final_status": "compact_matrix_parse_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix parsing before row-label recovery.",
            "parse_errors": parse_errors,
            "folder_status": folder_status,
            "download_ledger": download_ledger,
        }
        write_json(OUTDIR / "shoes_compact_ladder_row_label_recovery_v1_summary.json", summary)
        print("Matrix parsing failed. See summary JSON.")
        return

    C = np.asarray(parsed["allc"], dtype=np.float64)
    L = np.asarray(parsed["alll"], dtype=np.float64)
    y = np.asarray(parsed["ally"], dtype=np.float64).reshape(-1)

    X, orientation = determine_design_orientation(L, len(y))

    if X is None:
        raise RuntimeError(f"Could not orient L: {orientation}")

    c_factor, C_sym, jitter, chol_attempts = stable_cholesky(C)
    baseline = gls_fit(y, X, c_factor)

    row_rows, cluster_rows = row_and_cluster_features(
        X,
        y,
        baseline["residual"],
        baseline["beta"],
        baseline["beta_err"],
    )

    col_rows = column_features(X, baseline["beta"], baseline["beta_err"])
    boundary_vectors, boundary_vector_stats = build_candidate_boundary_vectors(row_rows, cluster_rows)

    write_csv(OUTDIR / "compact_row_features_v1.csv", row_rows)
    write_csv(OUTDIR / "compact_signature_clusters_v1.csv", cluster_rows)
    write_csv(OUTDIR / "compact_parameter_column_features_v1.csv", col_rows)

    plot_outputs(row_rows, cluster_rows, col_rows)

    final_status, readiness_score, next_wall = decide_status(
        parameter_candidates,
        row_rows,
        cluster_rows,
        col_rows,
        text_hits,
    )

    row_class_counts = dict(Counter(r["guessed_row_class"] for r in row_rows))
    parameter_family_counts = dict(Counter(r["beta_family_guess"] for r in col_rows))

    summary = {
        "test_name": "TAIRID SH0ES compact ladder row-label recovery v1",
        "boundary": (
            "Row-label and compact-ladder structure recovery only. Not proof of TAIRID, "
            "not a cosmology fit, not H0 resolution."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "github": {
            "owner": OWNER,
            "repo": REPO,
            "branch": BRANCH,
            "folder_status": folder_status,
        },
        "matrix_shapes": {
            "allc_C": list(C.shape),
            "alll_L": list(L.shape),
            "ally_y_original": list(np.asarray(parsed["ally"]).shape),
            "y_flat": list(y.shape),
            "X_design": list(X.shape),
            "L_orientation": orientation,
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
            "beta_preview_first_20": [float(v) for v in baseline["beta"][:20]],
            "beta_err_preview_first_20": [float(v) for v in baseline["beta_err"][:20]],
        },
        "covariance": {
            "cholesky_jitter": jitter,
            "cholesky_attempts": chol_attempts,
            "diag_min": float(np.min(np.diag(C_sym))),
            "diag_max": float(np.max(np.diag(C_sym))),
            "diag_nonpositive_count": int(np.sum(np.diag(C_sym) <= 0)),
        },
        "recovery_counts": {
            "downloaded_files": int(sum(1 for d in downloads if d.get("status") == "downloaded")),
            "text_search_hits": len(text_hits),
            "text_previews": len(preview_rows),
            "parameter_list_candidates": len(parameter_candidates),
            "exact_47_parameter_lists": int(sum(1 for r in parameter_candidates if int(r.get("count", 0)) == 47)),
            "near_parameter_lists_35_to_60": int(sum(1 for r in parameter_candidates if 35 <= int(r.get("count", 0)) <= 60)),
            "lstsq_numeric_lines": len(lstsq_lines),
            "row_count": len(row_rows),
            "signature_cluster_count": len(cluster_rows),
            "parameter_column_count": len(col_rows),
            "row_class_counts": row_class_counts,
            "parameter_family_counts": parameter_family_counts,
        },
        "top_signature_clusters": cluster_rows[:30],
        "candidate_parameter_lists": parameter_candidates[:20],
        "candidate_boundary_vector_stats": boundary_vector_stats,
        "output_files": {
            "summary_json": str(OUTDIR / "shoes_compact_ladder_row_label_recovery_v1_summary.json"),
            "summary_txt": str(OUTDIR / "shoes_compact_ladder_row_label_recovery_v1_summary.txt"),
            "download_ledger_csv": str(OUTDIR / "row_label_recovery_download_ledger_v1.csv"),
            "text_search_hits_csv": str(OUTDIR / "row_label_text_search_hits_v1.csv"),
            "parameter_list_candidates_csv": str(OUTDIR / "parameter_list_candidates_v1.csv"),
            "lstsq_parsed_lines_csv": str(OUTDIR / "lstsq_results_parsed_lines_v1.csv"),
            "row_features_csv": str(OUTDIR / "compact_row_features_v1.csv"),
            "signature_clusters_csv": str(OUTDIR / "compact_signature_clusters_v1.csv"),
            "parameter_column_features_csv": str(OUTDIR / "compact_parameter_column_features_v1.csv"),
            "candidate_boundary_vectors_stats_csv": str(OUTDIR / "candidate_boundary_vectors_stats_v1.csv"),
            "candidate_boundary_vectors_by_row_csv": str(OUTDIR / "candidate_boundary_vectors_by_row_v1.csv"),
            "plots": [
                str(OUTDIR / "row_nonzero_count_hist_v1.png"),
                str(OUTDIR / "baseline_residual_by_row_v1.png"),
                str(OUTDIR / "y_vector_by_row_v1.png"),
                str(OUTDIR / "top_signature_clusters_v1.png"),
                str(OUTDIR / "gls_parameter_beta_with_errors_v1.png"),
            ],
        },
        "interpretation": {
            "what_success_means": (
                "Parameter names or row/column classes were recovered enough to build a stricter boundary-vector likelihood test."
            ),
            "what_partial_success_means": (
                "The compact matrix structure is mapped, but explicit human labels are still incomplete."
            ),
            "what_failure_means": (
                "The compact ladder is accessible but row-label mapping remains blocked."
            ),
            "truth_boundary": (
                "This test does not fit TAIRID. It only prepares the row-label structure needed for a valid compact-ladder boundary test."
            ),
        },
    }

    write_json(OUTDIR / "shoes_compact_ladder_row_label_recovery_v1_summary.json", summary)

    with open(OUTDIR / "shoes_compact_ladder_row_label_recovery_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES compact ladder row-label recovery v1\n\n")
        f.write("Boundary: row-label / structure recovery only. Not proof. Not a cosmology fit.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Matrix shapes:\n")
        f.write(json.dumps(summary["matrix_shapes"], indent=2, default=json_default) + "\n\n")

        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")

        f.write("Recovery counts:\n")
        f.write(json.dumps(summary["recovery_counts"], indent=2, default=json_default) + "\n\n")

        f.write("Top signature clusters:\n")
        f.write(json.dumps(cluster_rows[:20], indent=2, default=json_default) + "\n\n")

        f.write("Candidate parameter lists:\n")
        f.write(json.dumps(parameter_candidates[:10], indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not test H0 resolution.\n")
        f.write("- This prepares the compact ladder for a better boundary-vector likelihood test.\n")

    print("")
    print("TAIRID SH0ES compact ladder row-label recovery v1 complete.")
    print("Created:")
    print("  tairid_shoes_compact_ladder_row_label_recovery_v1_outputs/shoes_compact_ladder_row_label_recovery_v1_summary.json")
    print("  tairid_shoes_compact_ladder_row_label_recovery_v1_outputs/shoes_compact_ladder_row_label_recovery_v1_summary.txt")
    print("  tairid_shoes_compact_ladder_row_label_recovery_v1_outputs/compact_row_features_v1.csv")
    print("  tairid_shoes_compact_ladder_row_label_recovery_v1_outputs/compact_signature_clusters_v1.csv")
    print("  tairid_shoes_compact_ladder_row_label_recovery_v1_outputs/compact_parameter_column_features_v1.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is row-label / structure recovery only.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
