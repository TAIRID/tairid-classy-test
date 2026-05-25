#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder table-aligned row-label recovery v1.1.

Purpose:
The v1 row-label recovery mapped the compact SH0ES matrix skeleton:
- 3492 compact rows
- 47 parameters
- 114 signature clusters
- cluster 113 size = 277, active cols 42,46, likely related to SH0ES Hubble-flow rows

But v1 did not recover human-readable row labels or a 47-name parameter list.

This v1.1 test tries to align the compact matrix skeleton with public SH0ES auxiliary files:
- table-like data files
- README files
- MCMC_utils.py / run_mcmc.py
- lstsq_results.txt
- row-count matches
- contiguous matrix blocks
- signature clusters
- active-column patterns

Boundary:
This is not proof of TAIRID.
This is not a cosmology fit.
This is not a boundary-gate likelihood.
This is row-label / table-alignment recovery only.

The next likelihood test should only happen if we recover enough explicit row classes
to create real boundary vectors instead of topology guesses.
"""

import ast
import csv
import hashlib
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve


OUTDIR = Path("tairid_shoes_table_aligned_row_label_recovery_v1_1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

OWNER = "PantheonPlusSH0ES"
REPO = "DataRelease"
BRANCH = "main"
SHOES_API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/SH0ES_Data?ref={BRANCH}"

COMPACT_FILES = {
    "allc": "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits",
    "alll": "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits",
    "ally": "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits",
}

TARGET_FILE_HINTS = [
    "table2",
    "optical_wes",
    "R22",
    "NIR",
    "README",
    "readme",
    "MCMC_utils",
    "run_mcmc",
    "lstsq_results",
    "allc",
    "alll",
    "ally",
]

CODE_SEARCH_TERMS = [
    "allc", "alll", "ally", "fits", "lstsq", "theta", "param", "parameters",
    "ceph", "cepheid", "pantheon", "shoes", "anchor", "host", "calibrator",
    "muhat", "intercept", "hubble", "period", "metal", "distance", "sn",
    "supernova", "covariance", "design", "matrix", "equation", "topantheon",
]

EPS = 1e-12


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
            "User-Agent": "TAIRID-SH0ES-table-aligned-row-label-recovery-v1-1",
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
            attempts.append({
                "repo_path": repo_path,
                "candidate_kind": cand["kind"],
                "url": cand["url"],
                "status": "http_error",
                "http_code": exc.code,
                "error": str(exc),
            })
        except Exception as exc:
            attempts.append({
                "repo_path": repo_path,
                "candidate_kind": cand["kind"],
                "url": cand["url"],
                "status": "download_failed",
                "error": str(exc),
            })

    return {
        "repo_path": repo_path,
        "status": "failed_required" if required else "failed_optional",
        "local_path": None,
        "pointer_info": pointer_info,
        "attempts": attempts,
    }


def github_folder_inventory():
    try:
        data, final_url, content_type, status = fetch_url(SHOES_API_URL, accept="application/json")
        obj = json.loads(data.decode("utf-8", errors="replace"))

        rows = []
        for item in obj:
            rows.append({
                "name": item.get("name"),
                "path": item.get("path"),
                "size": item.get("size"),
                "type": item.get("type"),
                "download_url": item.get("download_url"),
                "html_url": item.get("html_url"),
            })

        write_json(OUTDIR / "shoes_data_folder_inventory_raw_v1_1.json", obj)
        write_csv(OUTDIR / "shoes_data_folder_inventory_v1_1.csv", rows)

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
    return lower.endswith((".txt", ".md", ".py", ".dat", ".csv", ".tsv", ".tex", ".out", ".ipynb"))


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
                        arr = np.column_stack([np.asarray(v).reshape(len(v), -1) for v in numeric])
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
            attempts.append({"attempt": i, "jitter": jitter, "status": "failed", "error": str(exc)})
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


def parse_text_tables(downloads):
    table_inventory = []
    parsed_table_rows = []

    for item in downloads:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["local_path"])

        if not is_textish(path):
            continue

        try:
            text = path.read_text(errors="replace")
        except Exception as exc:
            table_inventory.append({
                "repo_path": item["repo_path"],
                "local_path": str(path),
                "status": "read_failed",
                "error": str(exc),
            })
            continue

        lines = [ln.rstrip("\n") for ln in text.splitlines()]
        nonempty = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]

        preview_path = OUTDIR / f"preview_{safe_name(item['repo_path'])}.txt"
        preview_path.write_text("\n".join(lines[:220]), encoding="utf-8")

        table_inventory.append({
            "repo_path": item["repo_path"],
            "local_path": str(path),
            "status": "text_read",
            "line_count": len(lines),
            "nonempty_noncomment_lines": len(nonempty),
            "preview_file": str(preview_path),
        })

        # Try pandas parsing.
        dfs = []

        if path.suffix.lower() in [".csv", ".tsv"]:
            seps = ["," if path.suffix.lower() == ".csv" else "\t", r"\s+"]
        else:
            seps = [r"\s+", ",", "\t"]

        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, engine="python", comment="#")
                if df is not None and not df.empty and len(df.columns) >= 2:
                    df.columns = [str(c).strip() for c in df.columns]
                    dfs.append((sep, df))
                    break
            except Exception:
                continue

        for sep, df in dfs:
            parsed_table_rows.append({
                "repo_path": item["repo_path"],
                "local_path": str(path),
                "parse_sep": sep,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": " | ".join(map(str, df.columns[:40])),
                "host_like_columns": " | ".join([c for c in df.columns if "host" in str(c).lower() or "gal" in str(c).lower()][:20]),
                "ceph_like_columns": " | ".join([c for c in df.columns if "ceph" in str(c).lower() or "period" in str(c).lower()][:20]),
                "sn_like_columns": " | ".join([c for c in df.columns if str(c).lower() in ["sn", "sne"] or "super" in str(c).lower()][:20]),
            })

    write_csv(OUTDIR / "text_file_inventory_v1_1.csv", table_inventory)
    write_csv(OUTDIR / "parsed_aux_table_inventory_v1_1.csv", parsed_table_rows)

    return table_inventory, parsed_table_rows


def search_code_context(downloads):
    hit_rows = []
    parameter_candidates = []
    assignment_candidates = []

    for item in downloads:
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
        lower_terms = [t.lower() for t in CODE_SEARCH_TERMS]

        for i, line in enumerate(lines, start=1):
            ll = line.lower()
            hits = [t for t in lower_terms if t in ll]

            if hits:
                lo = max(1, i - 2)
                hi = min(len(lines), i + 2)
                context = "\n".join(f"{j}: {lines[j-1]}" for j in range(lo, hi + 1))

                hit_rows.append({
                    "repo_path": item["repo_path"],
                    "line_number": i,
                    "hit_terms": " | ".join(hits[:15]),
                    "line": line[:700],
                    "context": context[:1800],
                })

        # Quoted-list recovery.
        quoted_lists = re.finditer(
            r"([A-Za-z_][A-Za-z0-9_]*(?:names?|labels?|pars?|params?|columns?)?)\s*=\s*(\[[^\]]+\])",
            text,
            flags=re.I | re.S,
        )

        for match in quoted_lists:
            varname = match.group(1)
            raw_list = match.group(2)
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", raw_list)

            if len(quoted) >= 3:
                line_number = text[: match.start()].count("\n") + 1
                parameter_candidates.append({
                    "repo_path": item["repo_path"],
                    "line_number": line_number,
                    "varname": varname,
                    "count": len(quoted),
                    "items_json": json.dumps(quoted),
                    "raw": raw_list[:1200],
                })

        # Numeric array/list assignment recovery.
        for match in re.finditer(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(np\.array\()?(\[[^\]]{20,4000}\])",
            text,
            flags=re.S,
        ):
            varname = match.group(1)
            raw = match.group(3)
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", raw)

            if len(nums) >= 5:
                line_number = text[: match.start()].count("\n") + 1
                assignment_candidates.append({
                    "repo_path": item["repo_path"],
                    "line_number": line_number,
                    "varname": varname,
                    "numeric_count": len(nums),
                    "raw": raw[:1200],
                })

    write_csv(OUTDIR / "code_context_search_hits_v1_1.csv", hit_rows)
    write_csv(OUTDIR / "quoted_parameter_list_candidates_v1_1.csv", parameter_candidates)
    write_csv(OUTDIR / "numeric_assignment_candidates_v1_1.csv", assignment_candidates)

    return hit_rows, parameter_candidates, assignment_candidates


def parse_lstsq_results(downloads):
    rows = []

    for item in downloads:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["local_path"])

        if "lstsq" not in path.name.lower():
            continue

        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue

        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()

            if not stripped:
                continue

            nums = []
            tokens = stripped.replace(",", " ").replace("=", " ").split()
            words = []

            for tok in tokens:
                try:
                    nums.append(float(tok))
                except Exception:
                    words.append(tok)

            if nums:
                rows.append({
                    "line_number": i,
                    "line": stripped[:700],
                    "word_prefix": " ".join(words[:5]),
                    "numeric_count": len(nums),
                    "numbers_json": json.dumps(nums[:12]),
                })

    write_csv(OUTDIR / "lstsq_results_parsed_lines_v1_1.csv", rows)
    return rows


def row_signature(row):
    active = np.where(np.abs(row) > 1e-12)[0]
    signs = np.sign(row[active]).astype(int)
    full_key = ",".join(f"{int(i)}:{int(s)}" for i, s in zip(active, signs))

    return active, signs, full_key


def row_structural_features(X, y, residual, beta, beta_err):
    signature_to_id = {}
    next_id = 0
    rows = []

    distance_like = (beta > 20.0) & (beta < 40.0)
    high_uncert = beta_err >= np.nanpercentile(beta_err, 75.0)

    for i in range(X.shape[0]):
        row = X[i, :]
        active, signs, key = row_signature(row)

        if key not in signature_to_id:
            signature_to_id[key] = next_id
            next_id += 1

        cluster_id = signature_to_id[key]
        vals = row[active]
        nz = len(active)

        if nz == 1:
            row_class = "single_parameter_prior_or_anchor_constraint"
        elif nz == 2 and np.any(signs > 0) and np.any(signs < 0):
            row_class = "two_parameter_difference_or_relative_constraint"
        elif nz <= 4 and np.any(signs > 0) and np.any(signs < 0):
            row_class = "sparse_ladder_relation"
        elif nz <= 4:
            row_class = "sparse_measurement_or_constraint"
        elif nz >= 8:
            row_class = "dense_ceph_sn_or_calibration_relation"
        else:
            row_class = "medium_ladder_measurement"

        if nz:
            distance_weight = float(np.sum(distance_like[active]) / nz)
            high_uncert_weight = float(np.sum(high_uncert[active]) / nz)
        else:
            distance_weight = 0.0
            high_uncert_weight = 0.0

        rows.append({
            "observation_index": i,
            "signature_cluster_id": cluster_id,
            "guessed_row_class": row_class,
            "y": float(y[i]),
            "baseline_residual": float(residual[i]),
            "nonzero_count": int(nz),
            "active_cols": ",".join(str(int(c)) for c in active),
            "sign_pattern": ",".join(str(int(s)) for s in signs),
            "row_l1": float(np.sum(np.abs(vals))) if nz else 0.0,
            "row_l2": float(np.sqrt(np.sum(vals * vals))) if nz else 0.0,
            "row_sum": float(np.sum(vals)) if nz else 0.0,
            "distance_like_param_fraction": distance_weight,
            "high_uncert_param_fraction": high_uncert_weight,
        })

    cluster_sizes = Counter(r["signature_cluster_id"] for r in rows)

    for r in rows:
        r["signature_cluster_size"] = int(cluster_sizes[r["signature_cluster_id"]])

    clusters = []
    grouped = defaultdict(list)

    for r in rows:
        grouped[r["signature_cluster_id"]].append(r)

    for cid, members in grouped.items():
        class_counts = Counter(m["guessed_row_class"] for m in members)
        active_cols = members[0]["active_cols"]
        sign_pattern = members[0]["sign_pattern"]

        clusters.append({
            "signature_cluster_id": cid,
            "size": len(members),
            "dominant_class_guess": class_counts.most_common(1)[0][0],
            "class_counts_json": json.dumps(dict(class_counts)),
            "first_row_index": int(members[0]["observation_index"]),
            "last_row_index": int(members[-1]["observation_index"]),
            "example_active_cols": active_cols,
            "example_sign_pattern": sign_pattern,
            "y_mean": float(np.mean([m["y"] for m in members])),
            "y_std": float(np.std([m["y"] for m in members])),
            "residual_mean": float(np.mean([m["baseline_residual"] for m in members])),
            "residual_rms": float(np.sqrt(np.mean(np.asarray([m["baseline_residual"] for m in members]) ** 2))),
            "nonzero_count_median": float(np.median([m["nonzero_count"] for m in members])),
            "distance_like_param_fraction_mean": float(np.mean([m["distance_like_param_fraction"] for m in members])),
            "high_uncert_param_fraction_mean": float(np.mean([m["high_uncert_param_fraction"] for m in members])),
        })

    clusters = sorted(clusters, key=lambda r: (-r["size"], r["signature_cluster_id"]))

    return rows, clusters


def contiguous_blocks(row_rows):
    blocks = []

    if not row_rows:
        return blocks

    start = 0
    prev_cluster = row_rows[0]["signature_cluster_id"]
    prev_class = row_rows[0]["guessed_row_class"]

    for idx in range(1, len(row_rows)):
        cur_cluster = row_rows[idx]["signature_cluster_id"]
        cur_class = row_rows[idx]["guessed_row_class"]

        if cur_cluster != prev_cluster or cur_class != prev_class:
            members = row_rows[start:idx]
            blocks.append(summarize_block(len(blocks), members))
            start = idx
            prev_cluster = cur_cluster
            prev_class = cur_class

    members = row_rows[start:]
    blocks.append(summarize_block(len(blocks), members))

    return blocks


def summarize_block(block_id, members):
    yvals = np.asarray([m["y"] for m in members], dtype=float)
    res = np.asarray([m["baseline_residual"] for m in members], dtype=float)
    nz = np.asarray([m["nonzero_count"] for m in members], dtype=float)

    return {
        "block_id": block_id,
        "start_row": int(members[0]["observation_index"]),
        "end_row": int(members[-1]["observation_index"]),
        "size": int(len(members)),
        "signature_cluster_id": int(members[0]["signature_cluster_id"]),
        "guessed_row_class": members[0]["guessed_row_class"],
        "active_cols": members[0]["active_cols"],
        "sign_pattern": members[0]["sign_pattern"],
        "y_mean": float(np.mean(yvals)),
        "y_std": float(np.std(yvals)),
        "residual_mean": float(np.mean(res)),
        "residual_rms": float(np.sqrt(np.mean(res * res))),
        "nonzero_count_median": float(np.median(nz)),
    }


def parameter_features(X, beta, beta_err):
    rows = []

    for j in range(X.shape[1]):
        col = X[:, j]
        active = np.abs(col) > 1e-12
        vals = col[active]

        if 20.0 <= beta[j] <= 40.0:
            family = "distance_modulus_like_parameter"
        elif np.sum(active) <= 10:
            family = "sparse_anchor_or_prior_parameter"
        elif np.sum(active) >= 0.40 * X.shape[0]:
            family = "global_or_broad_ladder_parameter"
        elif beta_err[j] >= 1.0:
            family = "weakly_constrained_or_boundary_parameter"
        else:
            family = "intermediate_ladder_parameter"

        rows.append({
            "parameter_index": j,
            "beta": float(beta[j]),
            "beta_err": float(beta_err[j]),
            "abs_beta_over_err": float(abs(beta[j]) / max(beta_err[j], EPS)),
            "nonzero_rows": int(np.sum(active)),
            "nonzero_fraction": float(np.mean(active)),
            "positive_count": int(np.sum(col > 1e-12)),
            "negative_count": int(np.sum(col < -1e-12)),
            "col_min_nonzero": float(np.min(vals)) if len(vals) else 0.0,
            "col_max_nonzero": float(np.max(vals)) if len(vals) else 0.0,
            "beta_family_guess": family,
        })

    return rows


def align_blocks_to_aux_tables(blocks, clusters, aux_tables):
    alignments = []

    aux_counts = []
    for t in aux_tables:
        rows = int(t.get("rows", 0) or 0)
        if rows > 0:
            aux_counts.append((t, rows))

    for block in blocks:
        size = block["size"]
        matches = []

        for t, count in aux_counts:
            if count == size:
                match_type = "exact"
            elif abs(count - size) <= 2:
                match_type = "within_2"
            elif abs(count - size) <= 5:
                match_type = "within_5"
            elif count > 0 and abs(count - size) / count <= 0.05:
                match_type = "within_5_percent"
            else:
                continue

            matches.append({
                "repo_path": t.get("repo_path"),
                "rows": count,
                "columns": t.get("columns"),
                "column_names": t.get("column_names"),
                "match_type": match_type,
            })

        if matches or size in [277, 77, 428, 875]:
            alignments.append({
                "block_id": block["block_id"],
                "start_row": block["start_row"],
                "end_row": block["end_row"],
                "size": size,
                "signature_cluster_id": block["signature_cluster_id"],
                "guessed_row_class": block["guessed_row_class"],
                "active_cols": block["active_cols"],
                "special_count_hint": special_count_hint(size),
                "matches_json": json.dumps(matches[:10]),
                "match_count": len(matches),
            })

    # Also align large signature clusters by count.
    for cluster in clusters[:60]:
        size = int(cluster["size"])
        matches = []

        for t, count in aux_counts:
            if count == size:
                match_type = "exact"
            elif abs(count - size) <= 2:
                match_type = "within_2"
            elif abs(count - size) <= 5:
                match_type = "within_5"
            elif count > 0 and abs(count - size) / count <= 0.05:
                match_type = "within_5_percent"
            else:
                continue

            matches.append({
                "repo_path": t.get("repo_path"),
                "rows": count,
                "columns": t.get("columns"),
                "column_names": t.get("column_names"),
                "match_type": match_type,
            })

        if matches or size in [277, 77, 428, 875]:
            alignments.append({
                "block_id": "",
                "start_row": cluster["first_row_index"],
                "end_row": cluster["last_row_index"],
                "size": size,
                "signature_cluster_id": cluster["signature_cluster_id"],
                "guessed_row_class": cluster["dominant_class_guess"],
                "active_cols": cluster["example_active_cols"],
                "special_count_hint": special_count_hint(size),
                "matches_json": json.dumps(matches[:10]),
                "match_count": len(matches),
                "alignment_kind": "signature_cluster_count",
            })

    write_csv(OUTDIR / "block_and_cluster_aux_table_alignments_v1_1.csv", alignments)
    return alignments


def special_count_hint(size):
    hints = []
    if size == 277:
        hints.append("matches_prior_SH0ES_Hubble_flow_count_277")
    if size == 77:
        hints.append("matches_prior_calibrator_count_77")
    if size == 428:
        hints.append("matches_prior_lowz_noncal_nonHF_count_428")
    if size == 875:
        hints.append("matches_prior_rest_count_875")
    return " | ".join(hints)


def build_candidate_label_map(row_rows, blocks, alignments):
    block_by_row = {}

    for block in blocks:
        for i in range(block["start_row"], block["end_row"] + 1):
            block_by_row[i] = block

    cluster_hints = {}
    for a in alignments:
        hint = a.get("special_count_hint", "")
        if hint:
            cluster_hints[int(a["signature_cluster_id"])] = hint

    rows = []
    for r in row_rows:
        block = block_by_row.get(r["observation_index"], {})
        cluster_id = int(r["signature_cluster_id"])
        hint = cluster_hints.get(cluster_id, "")

        candidate_label = r["guessed_row_class"]

        if "Hubble_flow_count_277" in hint or "SH0ES_Hubble_flow" in hint:
            candidate_label = "candidate_SH0ES_Hubble_flow_or_SN_intercept_block"
        elif "calibrator_count_77" in hint:
            candidate_label = "candidate_calibrator_boundary_block"
        elif "lowz_noncal_nonHF" in hint:
            candidate_label = "candidate_lowz_noncal_nonHF_block"
        elif r["nonzero_count"] == 1:
            candidate_label = "candidate_anchor_or_prior_row"
        elif r["guessed_row_class"] == "dense_ceph_sn_or_calibration_relation":
            candidate_label = "candidate_Cepheid_SN_calibration_relation"

        rows.append({
            "observation_index": r["observation_index"],
            "candidate_label": candidate_label,
            "signature_cluster_id": cluster_id,
            "signature_cluster_size": r["signature_cluster_size"],
            "block_id": block.get("block_id", ""),
            "block_size": block.get("size", ""),
            "special_count_hint": hint,
            "guessed_row_class": r["guessed_row_class"],
            "active_cols": r["active_cols"],
            "sign_pattern": r["sign_pattern"],
            "y": r["y"],
            "baseline_residual": r["baseline_residual"],
        })

    write_csv(OUTDIR / "candidate_compact_row_label_map_v1_1.csv", rows)
    return rows


def build_candidate_boundary_vectors(label_map):
    names = [
        "candidate_SH0ES_Hubble_flow_or_SN_intercept_block",
        "candidate_calibrator_boundary_block",
        "candidate_lowz_noncal_nonHF_block",
        "candidate_Cepheid_SN_calibration_relation",
        "candidate_anchor_or_prior_row",
    ]

    preview = []
    stats = []

    for r in label_map:
        row = {"observation_index": r["observation_index"]}
        for name in names:
            row[name] = 1.0 if r["candidate_label"] == name else 0.0
        preview.append(row)

    for name in names:
        vals = np.asarray([p[name] for p in preview], dtype=float)
        stats.append({
            "candidate_vector": name,
            "nonzero_count": int(np.sum(vals > 0)),
            "fraction": float(np.mean(vals > 0)),
            "note": "Candidate only. Do not promote until validated against SH0ES code/table labels.",
        })

    write_csv(OUTDIR / "candidate_boundary_vectors_by_row_v1_1.csv", preview)
    write_csv(OUTDIR / "candidate_boundary_vectors_stats_v1_1.csv", stats)

    return stats


def plot_outputs(row_rows, clusters, blocks, label_map, parameter_rows):
    residual = np.asarray([r["baseline_residual"] for r in row_rows], dtype=float)
    nz = np.asarray([r["nonzero_count"] for r in row_rows], dtype=float)
    yvals = np.asarray([r["y"] for r in row_rows], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.hist(nz, bins=60)
    plt.xlabel("nonzero parameter count per compact row")
    plt.ylabel("count")
    plt.title("SH0ES compact ladder row sparsity v1.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "row_nonzero_count_hist_v1_1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(residual, linewidth=0.8)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("compact row index")
    plt.ylabel("baseline residual")
    plt.title("Baseline compact residual by row v1.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "baseline_residual_by_row_v1_1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(yvals, linewidth=0.8)
    plt.xlabel("compact row index")
    plt.ylabel("y")
    plt.title("Compact y vector by row v1.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "y_vector_by_row_v1_1.png", dpi=160)
    plt.close()

    top_clusters = clusters[:35]
    labels = [str(c["signature_cluster_id"]) for c in top_clusters]
    sizes = [c["size"] for c in top_clusters]

    plt.figure(figsize=(13, 6))
    plt.bar(np.arange(len(labels)), sizes)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("signature cluster id")
    plt.ylabel("cluster size")
    plt.title("Largest compact row signature clusters v1.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "top_signature_clusters_v1_1.png", dpi=160)
    plt.close()

    beta = [p["beta"] for p in parameter_rows]
    beta_err = [p["beta_err"] for p in parameter_rows]
    x = np.arange(len(beta))

    plt.figure(figsize=(11, 6))
    plt.errorbar(x, beta, yerr=beta_err, fmt="o", linewidth=1)
    plt.xlabel("parameter index")
    plt.ylabel("beta")
    plt.title("Compact GLS parameters v1.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "gls_parameter_beta_with_errors_v1_1.png", dpi=160)
    plt.close()

    label_counts = Counter(r["candidate_label"] for r in label_map)
    labs = list(label_counts.keys())
    vals = [label_counts[k] for k in labs]

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(labs)), vals)
    plt.xticks(np.arange(len(labs)), labs, rotation=35, ha="right")
    plt.ylabel("row count")
    plt.title("Candidate compact row labels v1.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "candidate_row_label_counts_v1_1.png", dpi=160)
    plt.close()


def decide_status(parameter_candidates, alignments, label_map, table_inventory, code_hits):
    exact_47 = [p for p in parameter_candidates if int(p.get("count", 0)) == 47]
    near_param = [p for p in parameter_candidates if 35 <= int(p.get("count", 0)) <= 60]

    label_counts = Counter(r["candidate_label"] for r in label_map)
    hf_count = label_counts.get("candidate_SH0ES_Hubble_flow_or_SN_intercept_block", 0)
    calibrator_count = label_counts.get("candidate_calibrator_boundary_block", 0)
    ceph_count = label_counts.get("candidate_Cepheid_SN_calibration_relation", 0)
    anchor_count = label_counts.get("candidate_anchor_or_prior_row", 0)

    alignment_count = len(alignments)
    table_count = len([t for t in table_inventory if t.get("status") == "text_read"])
    code_hit_count = len(code_hits)

    if exact_47 and hf_count == 277 and (calibrator_count > 0 or ceph_count > 0):
        return (
            "table_aligned_row_labels_and_parameter_order_recovered",
            9,
            "Build compact ladder explicit-boundary likelihood v2 using recovered row labels and parameter order.",
        )

    if hf_count == 277 and (calibrator_count > 0 or ceph_count > 0) and alignment_count > 0:
        return (
            "table_aligned_hubble_flow_and_boundary_candidates_recovered",
            8,
            "Build v2 as a strict candidate-boundary likelihood, but keep labels provisional.",
        )

    if hf_count == 277 and alignment_count > 0:
        return (
            "hubble_flow_candidate_recovered_labels_still_partial",
            7,
            "Use the 277-row candidate plus code/table hits to build row-label recovery v1.2 before fitting.",
        )

    if near_param or alignment_count > 0 or code_hit_count > 100:
        return (
            "table_alignment_partial_structure_recovered",
            7,
            "Inspect alignments manually, then target missing row classes in v1.2.",
        )

    if table_count > 0:
        return (
            "auxiliary_tables_recovered_but_not_aligned",
            6,
            "The tables are visible, but row-count/block alignment is not strong enough yet.",
        )

    return (
        "table_aligned_row_label_recovery_failed",
        5,
        "Resolve table parsing and code mapping before another likelihood test.",
    )


def main():
    print("")
    print("TAIRID SH0ES compact ladder table-aligned row-label recovery v1.1 starting.")
    print("Boundary: row-label/table-alignment only; not a TAIRID fit.")
    print("")

    folder_rows, folder_status = github_folder_inventory()

    repo_paths = sorted({r["path"] for r in folder_rows if r.get("path")})
    for p in COMPACT_FILES.values():
        if p not in repo_paths:
            repo_paths.append(p)

    downloads = []
    by_path = {}

    for repo_path in repo_paths:
        required = repo_path in COMPACT_FILES.values()
        result = download_repo_path(repo_path, required=required)
        downloads.append(result)
        by_path[repo_path] = result

    download_ledger = []
    for d in downloads:
        download_ledger.append({
            "repo_path": d.get("repo_path"),
            "status": d.get("status"),
            "local_path": d.get("local_path"),
            "bytes": d.get("bytes"),
            "sha256": d.get("sha256"),
            "pointer_declared_size": (d.get("pointer_info") or {}).get("declared_size"),
            "pointer_oid_sha256": (d.get("pointer_info") or {}).get("oid_sha256"),
            "attempt_count": len(d.get("attempts", [])),
        })

    write_csv(OUTDIR / "table_aligned_download_ledger_v1_1.csv", download_ledger)
    write_json(OUTDIR / "table_aligned_download_attempts_v1_1.json", downloads)

    table_inventory, aux_tables = parse_text_tables(downloads)
    code_hits, parameter_candidates, numeric_assignments = search_code_context(downloads)
    lstsq_lines = parse_lstsq_results(downloads)

    parsed = {}
    parse_meta = {}
    parse_errors = []

    for label, repo_path in COMPACT_FILES.items():
        result = by_path.get(repo_path)
        if not result or result.get("status") != "downloaded":
            parse_errors.append({"label": label, "repo_path": repo_path, "status": "not_downloaded"})
            continue

        try:
            arr, meta = extract_numeric_fits(Path(result["local_path"]))
            parsed[label] = arr
            parse_meta[label] = meta
        except Exception as exc:
            parse_errors.append({"label": label, "repo_path": repo_path, "status": "parse_failed", "error": str(exc)})

    write_json(OUTDIR / "fits_parse_meta_v1_1.json", parse_meta)
    write_json(OUTDIR / "fits_parse_errors_v1_1.json", parse_errors)

    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]):
        summary = {
            "test_name": "TAIRID SH0ES compact ladder table-aligned row-label recovery v1.1",
            "final_status": "compact_matrix_parse_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix parsing before table-alignment recovery.",
            "parse_errors": parse_errors,
            "folder_status": folder_status,
        }
        write_json(OUTDIR / "shoes_table_aligned_row_label_recovery_v1_1_summary.json", summary)
        print("Matrix parsing failed. See summary JSON.")
        return

    C = np.asarray(parsed["allc"], dtype=np.float64)
    L = np.asarray(parsed["alll"], dtype=np.float64)
    y = np.asarray(parsed["ally"], dtype=np.float64).reshape(-1)

    X, orientation = determine_design_orientation(L, len(y))
    if X is None:
        raise RuntimeError(f"Could not orient L relative to y: {orientation}")

    c_factor, C_sym, jitter, chol_attempts = stable_cholesky(C)
    baseline = gls_fit(y, X, c_factor)

    row_rows, clusters = row_structural_features(
        X,
        y,
        baseline["residual"],
        baseline["beta"],
        baseline["beta_err"],
    )

    blocks = contiguous_blocks(row_rows)
    parameter_rows = parameter_features(X, baseline["beta"], baseline["beta_err"])

    alignments = align_blocks_to_aux_tables(blocks, clusters, aux_tables)
    label_map = build_candidate_label_map(row_rows, blocks, alignments)
    vector_stats = build_candidate_boundary_vectors(label_map)

    write_csv(OUTDIR / "compact_row_features_v1_1.csv", row_rows)
    write_csv(OUTDIR / "compact_signature_clusters_v1_1.csv", clusters)
    write_csv(OUTDIR / "compact_contiguous_blocks_v1_1.csv", blocks)
    write_csv(OUTDIR / "compact_parameter_column_features_v1_1.csv", parameter_rows)

    plot_outputs(row_rows, clusters, blocks, label_map, parameter_rows)

    final_status, readiness_score, next_wall = decide_status(
        parameter_candidates,
        alignments,
        label_map,
        table_inventory,
        code_hits,
    )

    label_counts = dict(Counter(r["candidate_label"] for r in label_map))
    row_class_counts = dict(Counter(r["guessed_row_class"] for r in row_rows))
    parameter_family_counts = dict(Counter(r["beta_family_guess"] for r in parameter_rows))

    summary = {
        "test_name": "TAIRID SH0ES compact ladder table-aligned row-label recovery v1.1",
        "boundary": (
            "Row-label and table-alignment recovery only. Not proof of TAIRID, not a cosmology fit, "
            "not H0 resolution, and not a boundary-gate likelihood."
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
            "text_files_read": int(sum(1 for t in table_inventory if t.get("status") == "text_read")),
            "parsed_aux_tables": len(aux_tables),
            "code_context_hits": len(code_hits),
            "quoted_parameter_list_candidates": len(parameter_candidates),
            "exact_47_parameter_lists": int(sum(1 for p in parameter_candidates if int(p.get("count", 0)) == 47)),
            "near_parameter_lists_35_to_60": int(sum(1 for p in parameter_candidates if 35 <= int(p.get("count", 0)) <= 60)),
            "numeric_assignment_candidates": len(numeric_assignments),
            "lstsq_numeric_lines": len(lstsq_lines),
            "row_count": len(row_rows),
            "signature_cluster_count": len(clusters),
            "contiguous_block_count": len(blocks),
            "aux_alignment_count": len(alignments),
            "candidate_label_counts": label_counts,
            "row_class_counts": row_class_counts,
            "parameter_family_counts": parameter_family_counts,
        },
        "top_signature_clusters": clusters[:30],
        "largest_contiguous_blocks": sorted(blocks, key=lambda b: -b["size"])[:30],
        "top_aux_alignments": alignments[:40],
        "candidate_boundary_vector_stats": vector_stats,
        "output_files": {
            "summary_json": str(OUTDIR / "shoes_table_aligned_row_label_recovery_v1_1_summary.json"),
            "summary_txt": str(OUTDIR / "shoes_table_aligned_row_label_recovery_v1_1_summary.txt"),
            "download_ledger_csv": str(OUTDIR / "table_aligned_download_ledger_v1_1.csv"),
            "text_inventory_csv": str(OUTDIR / "text_file_inventory_v1_1.csv"),
            "parsed_aux_table_inventory_csv": str(OUTDIR / "parsed_aux_table_inventory_v1_1.csv"),
            "code_context_hits_csv": str(OUTDIR / "code_context_search_hits_v1_1.csv"),
            "parameter_list_candidates_csv": str(OUTDIR / "quoted_parameter_list_candidates_v1_1.csv"),
            "lstsq_parsed_lines_csv": str(OUTDIR / "lstsq_results_parsed_lines_v1_1.csv"),
            "row_features_csv": str(OUTDIR / "compact_row_features_v1_1.csv"),
            "signature_clusters_csv": str(OUTDIR / "compact_signature_clusters_v1_1.csv"),
            "contiguous_blocks_csv": str(OUTDIR / "compact_contiguous_blocks_v1_1.csv"),
            "parameter_column_features_csv": str(OUTDIR / "compact_parameter_column_features_v1_1.csv"),
            "aux_alignments_csv": str(OUTDIR / "block_and_cluster_aux_table_alignments_v1_1.csv"),
            "candidate_label_map_csv": str(OUTDIR / "candidate_compact_row_label_map_v1_1.csv"),
            "candidate_boundary_vectors_stats_csv": str(OUTDIR / "candidate_boundary_vectors_stats_v1_1.csv"),
            "candidate_boundary_vectors_by_row_csv": str(OUTDIR / "candidate_boundary_vectors_by_row_v1_1.csv"),
            "plots": [
                str(OUTDIR / "row_nonzero_count_hist_v1_1.png"),
                str(OUTDIR / "baseline_residual_by_row_v1_1.png"),
                str(OUTDIR / "y_vector_by_row_v1_1.png"),
                str(OUTDIR / "top_signature_clusters_v1_1.png"),
                str(OUTDIR / "gls_parameter_beta_with_errors_v1_1.png"),
                str(OUTDIR / "candidate_row_label_counts_v1_1.png"),
            ],
        },
        "interpretation": {
            "what_success_means": (
                "The compact matrix skeleton has been aligned to enough table/code evidence to build stricter candidate boundary vectors."
            ),
            "what_partial_success_means": (
                "One or more compact row families are identified, but labels remain provisional and should not be treated as final SH0ES metadata."
            ),
            "what_failure_means": (
                "Auxiliary tables or code hints could not be aligned strongly enough to improve over v1."
            ),
            "truth_boundary": (
                "This test does not fit TAIRID. It only prepares row labels for a future compact-ladder boundary likelihood."
            ),
        },
    }

    write_json(OUTDIR / "shoes_table_aligned_row_label_recovery_v1_1_summary.json", summary)

    with open(OUTDIR / "shoes_table_aligned_row_label_recovery_v1_1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES compact ladder table-aligned row-label recovery v1.1\n\n")
        f.write("Boundary: row-label / table-alignment only. Not proof. Not a cosmology fit.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Recovery counts:\n")
        f.write(json.dumps(summary["recovery_counts"], indent=2, default=json_default) + "\n\n")

        f.write("Top signature clusters:\n")
        f.write(json.dumps(clusters[:20], indent=2, default=json_default) + "\n\n")

        f.write("Largest contiguous blocks:\n")
        f.write(json.dumps(sorted(blocks, key=lambda b: -b["size"])[:20], indent=2, default=json_default) + "\n\n")

        f.write("Top auxiliary alignments:\n")
        f.write(json.dumps(alignments[:25], indent=2, default=json_default) + "\n\n")

        f.write("Candidate boundary vector stats:\n")
        f.write(json.dumps(vector_stats, indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not test H0 resolution.\n")
        f.write("- This prepares the compact ladder for a stricter boundary-vector likelihood test.\n")

    print("")
    print("TAIRID SH0ES compact ladder table-aligned row-label recovery v1.1 complete.")
    print("Created:")
    print("  tairid_shoes_table_aligned_row_label_recovery_v1_1_outputs/shoes_table_aligned_row_label_recovery_v1_1_summary.json")
    print("  tairid_shoes_table_aligned_row_label_recovery_v1_1_outputs/shoes_table_aligned_row_label_recovery_v1_1_summary.txt")
    print("  tairid_shoes_table_aligned_row_label_recovery_v1_1_outputs/candidate_compact_row_label_map_v1_1.csv")
    print("  tairid_shoes_table_aligned_row_label_recovery_v1_1_outputs/block_and_cluster_aux_table_alignments_v1_1.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is row-label / table-alignment recovery only.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
