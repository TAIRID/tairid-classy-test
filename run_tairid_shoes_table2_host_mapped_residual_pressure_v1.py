#!/usr/bin/env python3
"""
TAIRID SH0ES Table2 host-mapped residual-pressure audit v1.

Purpose:
The residual-pressure cluster identity pass found that the remaining SH0ES
compact-ladder pressure is not mainly the 42<->46 H0 bridge.

Instead, the main structure appears to be the 38/41/43 sparse-ladder spine:

    compact spine rows = 3130
    parsed Table2 Cepheid data rows = 3130

This audit directly aligns the compact 38/41/43 spine rows with Table2 Cepheid
rows, attaches host/table cells to compact residuals, and asks:

- Do compact spine rows align one-to-one with Table2 Cepheid rows?
- Which hosts / fields carry the highest residual pressure?
- Do host masks, table numeric columns, or high-pressure host blocks survive
  projection out of the original 47-parameter compact SH0ES model?
- Is remaining pressure host/table structured, or mostly residual-selected noise?

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This is a host/table residual-pressure audit after the SH0ES compact ladder fit.
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
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


OUTDIR = Path("tairid_shoes_table2_host_mapped_residual_pressure_v1_outputs")
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

TARGET_SPINE = {38, 41, 43}
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
        ("raw_githubusercontent", f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{quoted}"),
        ("media_githubusercontent", f"https://media.githubusercontent.com/media/{OWNER}/{REPO}/{BRANCH}/{quoted}"),
        ("github_raw", f"https://github.com/{OWNER}/{REPO}/raw/{BRANCH}/{quoted}"),
    ]


def fetch_url(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-SH0ES-table2-host-map-v1",
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
            attempts.append({
                "label": label,
                "repo_path": repo_path,
                "candidate_kind": kind,
                "url": url,
                "status": "http_error",
                "http_code": exc.code,
                "error": str(exc),
            })
        except Exception as exc:
            attempts.append({
                "label": label,
                "repo_path": repo_path,
                "candidate_kind": kind,
                "url": url,
                "status": "download_failed",
                "error": str(exc),
            })

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
                        arr = np.column_stack([np.asarray(v).reshape(len(v), -1) for v in numeric])
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

            attempts.append({"attempt": attempt, "jitter": jitter, "status": "success"})
            return factor, C_sym, jitter, attempts

        except Exception as exc:
            attempts.append({"attempt": attempt, "jitter": jitter, "status": "failed", "error": str(exc)})

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


def classify_row(active, signs):
    active_set = set(map(int, active))

    if len(active) == 1:
        return "single_parameter_prior_or_anchor_constraint"

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


def recover_compact_rows(X, y, C_sym, baseline):
    keys = []
    row_cache = []

    for i in range(X.shape[0]):
        active, signs, full_key, active_key, sign_key = row_signature(X[i, :])
        keys.append(full_key)
        row_cache.append((active, signs, full_key, active_key, sign_key))

    cluster_counts = Counter(keys)
    signature_to_id = {}
    for key in keys:
        if key not in signature_to_id:
            signature_to_id[key] = len(signature_to_id)

    residual = baseline["residual"]
    cinv_residual = baseline["Cinv_residual"]
    cov_diag = np.diag(C_sym)

    leverage = np.einsum("ij,jk,ik->i", X, baseline["normal_inv"], baseline["Cinv_X"])

    row_rows = []
    grouped = defaultdict(list)

    for i, (active, signs, full_key, active_key, sign_key) in enumerate(row_cache):
        active_set = set(map(int, active))
        family = classify_row(active, signs)
        contains_spine = TARGET_SPINE.issubset(active_set)
        varying_cols = sorted(active_set - TARGET_SPINE)

        row = {
            "observation_index": i,
            "signature_cluster_id": int(signature_to_id[full_key]),
            "signature_cluster_size": int(cluster_counts[full_key]),
            "equation_family": family,
            "active_cols": active_key,
            "sign_pattern": sign_key,
            "full_signature": full_key,
            "nonzero_count": int(len(active)),
            "contains_38_41_43_spine": bool(contains_spine),
            "varying_cols_outside_38_41_43": ",".join(str(v) for v in varying_cols),
            "touches_param42": bool(P42 in active_set),
            "touches_param46_H0_like": bool(P46 in active_set),
            "bridges_param42_param46": bool(family == "explicit_42_46_bridge"),
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

        cluster_rows.append({
            "signature_cluster_id": first["signature_cluster_id"],
            "size": len(rows),
            "equation_family": first["equation_family"],
            "active_cols": first["active_cols"],
            "sign_pattern": first["sign_pattern"],
            "contains_38_41_43_spine": first["contains_38_41_43_spine"],
            "varying_cols_outside_38_41_43": first["varying_cols_outside_38_41_43"],
            "touches_param42": first["touches_param42"],
            "touches_param46_H0_like": first["touches_param46_H0_like"],
            "bridges_param42_param46": first["bridges_param42_param46"],
            "first_row": int(rows[0]["observation_index"]),
            "last_row": int(rows[-1]["observation_index"]),
            "mean_residual": float(np.mean(residuals)),
            "median_residual": float(np.median(residuals)),
            "rms_residual": float(np.sqrt(np.mean(residuals * residuals))),
            "mean_abs_residual": float(np.mean(np.abs(residuals))),
            "max_abs_residual": float(np.max(np.abs(residuals))),
            "mean_cinv_residual": float(np.mean(cinv_res)),
            "mean_abs_cinv_residual": float(np.mean(np.abs(cinv_res))),
            "mean_cov_diag": float(np.mean([r["cov_diag"] for r in rows])),
            "mean_abs_leverage_proxy": float(np.mean([r["abs_leverage_proxy"] for r in rows])),
            "y_mean": float(np.mean([r["y"] for r in rows])),
            "y_std": float(np.std([r["y"] for r in rows])),
        })

    cluster_rows = sorted(cluster_rows, key=lambda r: (-r["mean_abs_residual"], -r["size"], r["signature_cluster_id"]))

    return row_rows, cluster_rows


def clean_latex_cell(cell):
    cell = re.sub(r"%.*$", "", cell)
    cell = re.sub(r"\\\\.*$", "", cell)
    cell = cell.strip()
    cell = re.sub(r"\\nodata", "", cell)
    cell = re.sub(r"\\pm", " +/- ", cell)
    cell = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", cell)
    cell = re.sub(r"\\text\{([^{}]*)\}", r"\1", cell)
    cell = re.sub(r"\\tablenotemark\{[^{}]*\}", "", cell)
    cell = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", cell)
    cell = re.sub(r"[{}$]", "", cell)
    cell = re.sub(r"\s+", " ", cell)
    return cell.strip()


def first_float(text):
    if text is None:
        return None
    text = str(text).replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def parse_table2_tex(path):
    text = path.read_text(errors="replace")
    all_rows = []
    data_rows = []
    header_like = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()

        if "&" not in raw:
            continue

        if raw.startswith("%"):
            continue

        if "\\begin" in raw or "\\end" in raw or "\\hline" in raw or "\\table" in raw:
            continue

        cells = [clean_latex_cell(c) for c in raw.split("&")]
        cells = [c for c in cells if c != ""]

        if len(cells) < 2:
            continue

        numeric_count = sum(1 for c in cells if first_float(c) is not None)
        joined = " ".join(cells).lower()
        is_header = any(term in joined for term in ["host", "field", "period", "m160", "f160", "cepheid", "metal"])

        row = {
            "table2_parse_index": len(all_rows),
            "source_line_number": line_number,
            "cell_count": len(cells),
            "numeric_count": numeric_count,
            "cells_json": json.dumps(cells),
            "row_text": raw[:1400],
        }

        all_rows.append(row)

        if is_header and numeric_count < 3:
            header_like.append(row)
            continue

        if numeric_count >= 3:
            data_rows.append(row)
        else:
            header_like.append(row)

    write_csv(OUTDIR / "table2_all_latex_rows_v1.csv", all_rows)
    write_csv(OUTDIR / "table2_header_like_rows_v1.csv", header_like)
    write_csv(OUTDIR / "table2_data_rows_v1.csv", data_rows)

    return all_rows, data_rows, header_like


def infer_host(cells):
    for cell in cells[:4]:
        text = str(cell).strip()
        if not text:
            continue

        patterns = [
            r"\bN\d{3,5}[A-Za-z]?\b",
            r"\bM\d{1,3}\b",
            r"\bLMC\b",
            r"\bSMC\b",
            r"\bM31\b",
            r"\bN4258\b",
            r"\bU\w+\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                return match.group(0).upper()

    for cell in cells[:4]:
        text = str(cell).strip()
        if text and first_float(text) is None:
            return re.sub(r"\s+", "_", text.upper())[:60]

    return "UNKNOWN"


def map_table2_to_spine_rows(row_rows, table2_data_rows):
    spine_rows = [r for r in row_rows if r["contains_38_41_43_spine"]]
    spine_rows = sorted(spine_rows, key=lambda r: r["observation_index"])

    table_rows = list(table2_data_rows)

    if len(table_rows) > len(spine_rows):
        # Prior pass found two extra header-like rows sometimes leak through.
        # Keep the last rows only as a fallback when counts differ.
        table_rows = table_rows[-len(spine_rows):]

    n_map = min(len(spine_rows), len(table_rows))
    mapped = []

    max_cells = 0
    parsed_cells_cache = []

    for row in table_rows[:n_map]:
        cells = json.loads(row["cells_json"])
        parsed_cells_cache.append(cells)
        max_cells = max(max_cells, len(cells))

    for i in range(n_map):
        compact = spine_rows[i]
        table = table_rows[i]
        cells = parsed_cells_cache[i]
        host = infer_host(cells)

        out = {
            "mapped_index": i,
            "compact_observation_index": compact["observation_index"],
            "compact_signature_cluster_id": compact["signature_cluster_id"],
            "compact_signature_cluster_size": compact["signature_cluster_size"],
            "compact_active_cols": compact["active_cols"],
            "compact_sign_pattern": compact["sign_pattern"],
            "compact_varying_cols_outside_38_41_43": compact["varying_cols_outside_38_41_43"],
            "host_guess": host,
            "table2_source_line_number": table["source_line_number"],
            "table2_cell_count": table["cell_count"],
            "table2_numeric_count": table["numeric_count"],
            "baseline_residual": compact["baseline_residual"],
            "abs_baseline_residual": compact["abs_baseline_residual"],
            "cinv_residual": compact["cinv_residual"],
            "abs_cinv_residual": compact["abs_cinv_residual"],
            "cov_diag": compact["cov_diag"],
            "leverage_proxy": compact["leverage_proxy"],
            "abs_leverage_proxy": compact["abs_leverage_proxy"],
        }

        for j in range(max_cells):
            cell = cells[j] if j < len(cells) else ""
            out[f"table2_cell_{j}"] = cell
            num = first_float(cell)
            out[f"table2_num_{j}"] = num if num is not None else ""

        mapped.append(out)

    write_csv(OUTDIR / "table2_compact_host_mapped_rows_v1.csv", mapped)

    map_status = {
        "compact_spine_rows": len(spine_rows),
        "table2_data_rows": len(table2_data_rows),
        "mapped_rows": n_map,
        "exact_count_match": bool(len(spine_rows) == len(table2_data_rows)),
        "first_compact_spine_row": int(spine_rows[0]["observation_index"]) if spine_rows else None,
        "last_compact_spine_row": int(spine_rows[-1]["observation_index"]) if spine_rows else None,
    }

    return mapped, map_status


def summarize_hosts(mapped_rows):
    grouped = defaultdict(list)
    for row in mapped_rows:
        grouped[row["host_guess"]].append(row)

    host_rows = []

    for host, rows in grouped.items():
        residual = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        abs_residual = np.abs(residual)
        cinv = np.asarray([r["cinv_residual"] for r in rows], dtype=float)

        cluster_counts = Counter(r["compact_signature_cluster_id"] for r in rows)
        varying_counts = Counter(r["compact_varying_cols_outside_38_41_43"] for r in rows)

        host_rows.append({
            "host_guess": host,
            "row_count": len(rows),
            "mean_residual": float(np.mean(residual)),
            "median_residual": float(np.median(residual)),
            "rms_residual": float(np.sqrt(np.mean(residual * residual))),
            "mean_abs_residual": float(np.mean(abs_residual)),
            "max_abs_residual": float(np.max(abs_residual)),
            "mean_cinv_residual": float(np.mean(cinv)),
            "mean_abs_cinv_residual": float(np.mean(np.abs(cinv))),
            "first_mapped_index": int(rows[0]["mapped_index"]),
            "last_mapped_index": int(rows[-1]["mapped_index"]),
            "dominant_signature_clusters_json": json.dumps(dict(cluster_counts.most_common(8))),
            "dominant_varying_cols_json": json.dumps(dict(varying_counts.most_common(8))),
        })

    host_rows = sorted(host_rows, key=lambda r: (-r["mean_abs_residual"], -r["row_count"], r["host_guess"]))
    write_csv(OUTDIR / "table2_host_residual_summary_v1.csv", host_rows)

    return host_rows


def summarize_clusters_with_hosts(mapped_rows):
    grouped = defaultdict(list)

    for row in mapped_rows:
        grouped[row["compact_signature_cluster_id"]].append(row)

    rows_out = []

    for cid, rows in grouped.items():
        residual = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        host_counts = Counter(r["host_guess"] for r in rows)
        first = rows[0]

        rows_out.append({
            "signature_cluster_id": cid,
            "row_count": len(rows),
            "active_cols": first["compact_active_cols"],
            "varying_cols_outside_38_41_43": first["compact_varying_cols_outside_38_41_43"],
            "dominant_host": host_counts.most_common(1)[0][0],
            "dominant_host_count": host_counts.most_common(1)[0][1],
            "host_counts_json": json.dumps(dict(host_counts.most_common(10))),
            "mean_residual": float(np.mean(residual)),
            "rms_residual": float(np.sqrt(np.mean(residual * residual))),
            "mean_abs_residual": float(np.mean(np.abs(residual))),
            "max_abs_residual": float(np.max(np.abs(residual))),
            "first_compact_row": int(rows[0]["compact_observation_index"]),
            "last_compact_row": int(rows[-1]["compact_observation_index"]),
        })

    rows_out = sorted(rows_out, key=lambda r: (-r["mean_abs_residual"], -r["row_count"], r["signature_cluster_id"]))
    write_csv(OUTDIR / "table2_cluster_host_alignment_v1.csv", rows_out)

    return rows_out


def numeric_feature_correlations(mapped_rows):
    if not mapped_rows:
        return []

    num_cols = sorted([k for k in mapped_rows[0].keys() if k.startswith("table2_num_")])
    residual = np.asarray([r["baseline_residual"] for r in mapped_rows], dtype=float)
    abs_residual = np.abs(residual)

    out = []

    for col in num_cols:
        vals = []
        keep_res = []
        keep_abs = []

        for row, res, abs_res in zip(mapped_rows, residual, abs_residual):
            value = row.get(col, "")
            if value == "" or value is None:
                continue
            try:
                v = float(value)
            except Exception:
                continue
            if not np.isfinite(v):
                continue

            vals.append(v)
            keep_res.append(res)
            keep_abs.append(abs_res)

        if len(vals) < 50:
            continue

        x = np.asarray(vals, dtype=float)
        y = np.asarray(keep_res, dtype=float)
        ay = np.asarray(keep_abs, dtype=float)

        if np.std(x) <= 1.0e-14:
            continue

        corr_res = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0
        corr_abs = float(np.corrcoef(x, ay)[0, 1]) if len(x) > 2 else 0.0

        out.append({
            "numeric_column": col,
            "valid_count": len(x),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "corr_with_residual": corr_res,
            "corr_with_abs_residual": corr_abs,
        })

    out = sorted(out, key=lambda r: -abs(r["corr_with_abs_residual"]))
    write_csv(OUTDIR / "table2_numeric_feature_correlations_v1.csv", out)
    return out


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


def build_audit_vectors(mapped_rows, host_summary, cluster_host_rows, numeric_corr, y_length):
    vectors = []

    def add(name, values, kind):
        if len(values) != y_length:
            raise ValueError(f"{name} has length {len(values)}, expected {y_length}")
        vectors.append({"name": name, "values": np.asarray(values, dtype=float), "kind": kind})

    # Host masks for strongest high-pressure hosts with enough rows.
    for host in host_summary[:20]:
        if host["row_count"] < 5:
            continue
        mask = np.zeros(y_length, dtype=float)
        for row in mapped_rows:
            if row["host_guess"] == host["host_guess"]:
                mask[int(row["compact_observation_index"])] = 1.0
        add(f"host_{safe_name(host['host_guess'])}_n{host['row_count']}", mask, "host_mask")

    # Union of top high-pressure hosts.
    top_hosts = {h["host_guess"] for h in host_summary[:10] if h["row_count"] >= 5}
    top_host_mask = np.zeros(y_length, dtype=float)
    for row in mapped_rows:
        if row["host_guess"] in top_hosts:
            top_host_mask[int(row["compact_observation_index"])] = 1.0
    add("host_union_top10_mean_abs_residual", top_host_mask, "host_union")

    # Cluster masks for strongest host-aligned clusters.
    for cluster in cluster_host_rows[:20]:
        if cluster["row_count"] < 5:
            continue
        mask = np.zeros(y_length, dtype=float)
        for row in mapped_rows:
            if row["compact_signature_cluster_id"] == cluster["signature_cluster_id"]:
                mask[int(row["compact_observation_index"])] = 1.0
        add(f"cluster_{cluster['signature_cluster_id']}_host_{safe_name(cluster['dominant_host'])}_n{cluster['row_count']}", mask, "cluster_host_mask")

    # Numeric feature vectors on compact rows.
    for item in numeric_corr[:20]:
        col = item["numeric_column"]
        values = np.zeros(y_length, dtype=float)
        present = np.zeros(y_length, dtype=float)

        vals = []
        for row in mapped_rows:
            value = row.get(col, "")
            try:
                v = float(value)
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(v)

        if not vals:
            continue

        fill = float(np.nanmean(vals))

        for row in mapped_rows:
            idx = int(row["compact_observation_index"])
            value = row.get(col, "")
            try:
                v = float(value)
            except Exception:
                v = fill
            if not np.isfinite(v):
                v = fill
            values[idx] = v
            present[idx] = 1.0

        add(f"numeric_{col}", values, "table2_numeric_feature")

        # Extremes can catch threshold-style behavior without forcing a slope.
        mapped_vals = np.asarray([values[int(r["compact_observation_index"])] for r in mapped_rows], dtype=float)
        high_threshold = np.percentile(mapped_vals, 90)
        low_threshold = np.percentile(mapped_vals, 10)

        high_mask = np.zeros(y_length, dtype=float)
        low_mask = np.zeros(y_length, dtype=float)

        for row in mapped_rows:
            idx = int(row["compact_observation_index"])
            if values[idx] >= high_threshold:
                high_mask[idx] = 1.0
            if values[idx] <= low_threshold:
                low_mask[idx] = 1.0

        add(f"numeric_{col}_top10", high_mask, "table2_numeric_extreme")
        add(f"numeric_{col}_bottom10", low_mask, "table2_numeric_extreme")

    # Controls.
    row_order = np.zeros(y_length, dtype=float)
    for row in mapped_rows:
        row_order[int(row["compact_observation_index"])] = float(row["mapped_index"])
    add("table2_mapped_row_order_control", row_order, "control")

    return vectors


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
        rows.append(audit_vector(f"{label}_random_same_count_{i}", mask, "random_same_count_control", y, X, c_factor, baseline))

    for i in range(RANDOM_CONTROL_REPEATS):
        mask = np.zeros(len(y), dtype=float)
        start = int(rng.integers(0, len(y) - count + 1))
        mask[start:start + count] = 1.0
        rows.append(audit_vector(f"{label}_random_contiguous_block_{i}", mask, "random_contiguous_block_control", y, X, c_factor, baseline))

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
        "table2", "host", "field", "ceph", "cepheid", "period", "metal",
        "F160", "m160", "color", "anchor", "calibrator", "muhat",
        "intercept", "fivelogH0", "H0", "alll", "ally", "allc",
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
        preview_path.write_text("\n".join(lines[:220]), encoding="utf-8")

        for i, line in enumerate(lines, start=1):
            hits = [term for term in terms if term.lower() in line.lower()]
            if hits:
                lo = max(1, i - 2)
                hi = min(len(lines), i + 2)
                context = "\n".join(f"{j}: {lines[j - 1]}" for j in range(lo, hi + 1))
                rows.append({
                    "repo_path": item["repo_path"],
                    "line_number": i,
                    "hit_terms": " | ".join(hits),
                    "line": line[:700],
                    "context": context[:1600],
                })

    write_csv(OUTDIR / "table2_host_map_context_hits_v1.csv", rows)
    return rows


def make_plots(mapped_rows, host_summary, cluster_host_rows, numeric_corr, audit_rows):
    residual = np.asarray([r["baseline_residual"] for r in mapped_rows], dtype=float)
    mapped_index = np.asarray([r["mapped_index"] for r in mapped_rows], dtype=float)

    plt.figure(figsize=(11, 5))
    plt.plot(mapped_index, residual, linewidth=0.8)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Table2 mapped row index")
    plt.ylabel("compact baseline residual")
    plt.title("Table2 mapped compact residuals")
    plt.tight_layout()
    plt.savefig(OUTDIR / "table2_mapped_residual_by_row_v1.png", dpi=160)
    plt.close()

    top_hosts = host_summary[:25]
    labels = [h["host_guess"] for h in top_hosts]
    vals = [h["mean_abs_residual"] for h in top_hosts]

    plt.figure(figsize=(13, 6))
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("mean absolute residual")
    plt.title("Highest-pressure Table2 hosts")
    plt.tight_layout()
    plt.savefig(OUTDIR / "table2_host_mean_abs_residual_v1.png", dpi=160)
    plt.close()

    top_clusters = cluster_host_rows[:30]
    labels = [f"{c['signature_cluster_id']}:{c['dominant_host']}" for c in top_clusters]
    vals = [c["mean_abs_residual"] for c in top_clusters]

    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("mean absolute residual")
    plt.title("High-pressure compact clusters aligned to Table2 hosts")
    plt.tight_layout()
    plt.savefig(OUTDIR / "table2_cluster_host_pressure_v1.png", dpi=160)
    plt.close()

    if numeric_corr:
        top_num = numeric_corr[:20]
        labels = [n["numeric_column"] for n in top_num]
        vals = [abs(n["corr_with_abs_residual"]) for n in top_num]

        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(labels)), vals)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("|corr with abs residual|")
        plt.title("Table2 numeric feature correlations")
        plt.tight_layout()
        plt.savefig(OUTDIR / "table2_numeric_feature_correlations_v1.png", dpi=160)
        plt.close()

    plot_rows = [r for r in audit_rows if r.get("candidate_kind") not in ["random_same_count_control", "random_contiguous_block_control"]]
    plot_rows = plot_rows[:40]

    labels = [r["candidate"] for r in plot_rows]
    deltas = [r.get("delta_chi2_score", 0.0) for r in plot_rows]
    ratios = [r.get("nondegenerate_ratio", 0.0) for r in plot_rows]

    plt.figure(figsize=(15, 6))
    plt.bar(np.arange(len(labels)), deltas)
    plt.xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=7)
    plt.ylabel("delta chi2 score")
    plt.title("Host/Table2 candidate residual alignment")
    plt.tight_layout()
    plt.savefig(OUTDIR / "table2_candidate_delta_chi2_scores_v1.png", dpi=160)
    plt.close()

    plt.figure(figsize=(15, 6))
    plt.bar(np.arange(len(labels)), ratios)
    plt.xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=7)
    plt.ylabel("nondegenerate ratio")
    plt.title("Host/Table2 candidate survival after SH0ES projection")
    plt.tight_layout()
    plt.savefig(OUTDIR / "table2_candidate_nondegenerate_ratios_v1.png", dpi=160)
    plt.close()


def decide_status(map_status, audit_rows, host_summary, numeric_corr, random_summaries):
    ok_rows = [r for r in audit_rows if r.get("status") == "ok"]

    host_candidates = [r for r in ok_rows if r.get("candidate_kind") in ["host_mask", "host_union"]]
    cluster_candidates = [r for r in ok_rows if r.get("candidate_kind") == "cluster_host_mask"]
    numeric_candidates = [r for r in ok_rows if r.get("candidate_kind") in ["table2_numeric_feature", "table2_numeric_extreme"]]

    best_host = max(host_candidates, key=lambda r: r["delta_chi2_score"]) if host_candidates else None
    best_cluster = max(cluster_candidates, key=lambda r: r["delta_chi2_score"]) if cluster_candidates else None
    best_numeric = max(numeric_candidates, key=lambda r: r["delta_chi2_score"]) if numeric_candidates else None
    best_any = max(ok_rows, key=lambda r: r["delta_chi2_score"]) if ok_rows else None

    def locked(row):
        return bool(
            row
            and row["p_value_chi2_one_dof"] <= 0.01
            and row["delta_bic_if_added_column"] < -2.0
            and row["nondegenerate_ratio"] >= 0.05
        )

    def directional(row):
        return bool(
            row
            and row["p_value_chi2_one_dof"] <= 0.05
            and row["delta_aic_if_added_column"] < 0.0
            and row["nondegenerate_ratio"] >= 0.02
        )

    best_cases = {
        "best_host_candidate": best_host,
        "best_cluster_candidate": best_cluster,
        "best_numeric_candidate": best_numeric,
        "best_any_candidate": best_any,
        "top_hosts": host_summary[:20],
        "top_numeric_correlations": numeric_corr[:20],
        "random_control_summaries": random_summaries,
    }

    if not map_status.get("exact_count_match"):
        return (
            "table2_host_map_partial_count_mismatch",
            6,
            "Table2 and compact spine rows were mapped only partially; inspect parsing before using results.",
            best_cases,
        )

    if locked(best_host) or locked(best_cluster) or locked(best_numeric):
        return (
            "table2_host_mapped_residual_pressure_detected_for_followup",
            8,
            "A host/table feature survived SH0ES projection and aligns with residual pressure; build a stricter validation pass.",
            best_cases,
        )

    if directional(best_host) or directional(best_cluster) or directional(best_numeric):
        return (
            "table2_host_mapped_residual_pressure_directional_not_locked",
            7,
            "Host/table residual pressure is directional but not locked; inspect controls and labels before modeling.",
            best_cases,
        )

    return (
        "table2_host_mapped_no_locked_residual_structure",
        6,
        "Table2 mapping succeeded, but no host/table direction is strong enough for a new likelihood model.",
        best_cases,
    )


def main():
    print("")
    print("TAIRID SH0ES Table2 host-mapped residual-pressure audit v1 starting.")
    print("Boundary: host/table residual-pressure audit only; not proof.")
    print("")

    downloads = {}
    aux_results = []
    ledger = []

    for label, repo_path in COMPACT.items():
        result = download_repo_path(repo_path, label)
        downloads[label] = result
        ledger.append({
            "label": label,
            "repo_path": repo_path,
            "status": result.get("status"),
            "local_path": result.get("local_path"),
            "bytes": result.get("bytes"),
            "sha256": result.get("sha256"),
            "pointer_declared_size": (result.get("pointer_info") or {}).get("declared_size"),
            "pointer_oid_sha256": (result.get("pointer_info") or {}).get("oid_sha256"),
            "attempt_count": len(result.get("attempts", [])),
        })

    for repo_path in AUX:
        result = download_repo_path(repo_path, safe_name(repo_path))
        aux_results.append(result)
        ledger.append({
            "label": safe_name(repo_path),
            "repo_path": repo_path,
            "status": result.get("status"),
            "local_path": result.get("local_path"),
            "bytes": result.get("bytes"),
            "sha256": result.get("sha256"),
            "pointer_declared_size": (result.get("pointer_info") or {}).get("declared_size"),
            "pointer_oid_sha256": (result.get("pointer_info") or {}).get("oid_sha256"),
            "attempt_count": len(result.get("attempts", [])),
        })

    write_csv(OUTDIR / "table2_host_map_download_ledger_v1.csv", ledger)
    write_json(OUTDIR / "table2_host_map_download_attempts_v1.json", {"compact": downloads, "auxiliary": aux_results})

    code_hits = code_context_search(aux_results)

    parsed = {}
    parse_meta = {}
    parse_errors = []

    for label in ["allc", "alll", "ally"]:
        result = downloads.get(label, {})
        if result.get("status") != "downloaded":
            parse_errors.append({"label": label, "status": "not_downloaded", "download_status": result.get("status")})
            continue

        try:
            arr, meta = extract_first_numeric_fits_array(Path(result["local_path"]))
            parsed[label] = arr
            parse_meta[label] = meta
        except Exception as exc:
            parse_errors.append({"label": label, "status": "parse_failed", "error": str(exc)})

    write_json(OUTDIR / "table2_host_map_parse_meta_v1.json", parse_meta)
    write_json(OUTDIR / "table2_host_map_parse_errors_v1.json", parse_errors)

    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]):
        summary = {
            "test_name": "TAIRID SH0ES Table2 host-mapped residual-pressure audit v1",
            "boundary": "Matrix parse/download failure. No host-map result.",
            "final_status": "table2_host_map_matrix_parse_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix parsing before host-mapped audit.",
            "parse_errors": parse_errors,
        }
        write_json(OUTDIR / "table2_host_mapped_residual_pressure_v1_summary.json", summary)
        print("Parse failed. See summary JSON.")
        return

    table2_result = next((x for x in aux_results if x.get("repo_path") == "SH0ES_Data/table2.tex" and x.get("status") == "downloaded"), None)
    if not table2_result:
        summary = {
            "test_name": "TAIRID SH0ES Table2 host-mapped residual-pressure audit v1",
            "boundary": "table2.tex failed to download. No host-map result.",
            "final_status": "table2_tex_download_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix table2.tex retrieval before host-mapped audit.",
        }
        write_json(OUTDIR / "table2_host_mapped_residual_pressure_v1_summary.json", summary)
        print("table2.tex download failed. See summary JSON.")
        return

    table2_all_rows, table2_data_rows, table2_header_like_rows = parse_table2_tex(Path(table2_result["local_path"]))

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

    row_rows, cluster_rows = recover_compact_rows(X, y, C_sym, baseline)
    write_csv(OUTDIR / "table2_host_map_compact_row_map_v1.csv", row_rows)
    write_csv(OUTDIR / "table2_host_map_compact_clusters_v1.csv", cluster_rows)

    mapped_rows, map_status = map_table2_to_spine_rows(row_rows, table2_data_rows)

    host_summary = summarize_hosts(mapped_rows)
    cluster_host_rows = summarize_clusters_with_hosts(mapped_rows)
    numeric_corr = numeric_feature_correlations(mapped_rows)

    vectors = build_audit_vectors(mapped_rows, host_summary, cluster_host_rows, numeric_corr, len(y))

    audit_rows = []
    for item in vectors:
        audit_rows.append(
            audit_vector(item["name"], item["values"], item["kind"], y, X, c_factor, baseline)
        )

    audit_rows = sorted(
        audit_rows,
        key=lambda r: (-float(r.get("delta_chi2_score", 0.0)), -float(r.get("nondegenerate_ratio", 0.0))),
    )

    write_csv(OUTDIR / "table2_host_mapped_candidate_audit_v1.csv", audit_rows)

    random_summaries = []
    random_detail_rows = []

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

        summary_random, rows_random = random_control_audit(y, X, c_factor, baseline, count, f"count_{count}")
        random_summaries.append(summary_random)
        random_detail_rows.extend(rows_random)

    write_json(OUTDIR / "table2_host_mapped_random_control_summaries_v1.json", random_summaries)
    write_csv(OUTDIR / "table2_host_mapped_random_control_details_v1.csv", random_detail_rows)

    final_status, readiness_score, next_wall, best_cases = decide_status(
        map_status,
        audit_rows,
        host_summary,
        numeric_corr,
        random_summaries,
    )

    make_plots(mapped_rows, host_summary, cluster_host_rows, numeric_corr, audit_rows)

    residual = baseline["residual"]
    abs_resid = np.abs(residual)

    summary = {
        "test_name": "TAIRID SH0ES Table2 host-mapped residual-pressure audit v1",
        "boundary": (
            "Host/table residual-pressure audit only. Not proof of TAIRID, not H0 resolution, "
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
            "param38_value": float(baseline["beta"][38]),
            "param38_err": float(baseline["beta_err"][38]),
            "param41_value": float(baseline["beta"][41]),
            "param41_err": float(baseline["beta_err"][41]),
            "param43_value": float(baseline["beta"][43]),
            "param43_err": float(baseline["beta_err"][43]),
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
        "table2_mapping": map_status,
        "table2_parse_counts": {
            "table2_all_latex_rows": len(table2_all_rows),
            "table2_data_rows": len(table2_data_rows),
            "table2_header_like_rows": len(table2_header_like_rows),
            "mapped_rows": len(mapped_rows),
            "unique_host_guesses": len(host_summary),
        },
        "top_hosts_by_residual_pressure": host_summary[:40],
        "top_cluster_host_alignments": cluster_host_rows[:40],
        "top_numeric_feature_correlations": numeric_corr[:40],
        "candidate_audit": audit_rows[:80],
        "random_control_summaries": random_summaries,
        "best_cases": best_cases,
        "code_context_hits_count": len(code_hits),
        "output_files": {
            "summary_json": str(OUTDIR / "table2_host_mapped_residual_pressure_v1_summary.json"),
            "summary_txt": str(OUTDIR / "table2_host_mapped_residual_pressure_v1_summary.txt"),
            "mapped_rows_csv": str(OUTDIR / "table2_compact_host_mapped_rows_v1.csv"),
            "host_summary_csv": str(OUTDIR / "table2_host_residual_summary_v1.csv"),
            "cluster_host_alignment_csv": str(OUTDIR / "table2_cluster_host_alignment_v1.csv"),
            "numeric_feature_correlations_csv": str(OUTDIR / "table2_numeric_feature_correlations_v1.csv"),
            "candidate_audit_csv": str(OUTDIR / "table2_host_mapped_candidate_audit_v1.csv"),
            "random_control_summaries_json": str(OUTDIR / "table2_host_mapped_random_control_summaries_v1.json"),
            "random_control_details_csv": str(OUTDIR / "table2_host_mapped_random_control_details_v1.csv"),
            "context_hits_csv": str(OUTDIR / "table2_host_map_context_hits_v1.csv"),
            "plots": [
                str(OUTDIR / "table2_mapped_residual_by_row_v1.png"),
                str(OUTDIR / "table2_host_mean_abs_residual_v1.png"),
                str(OUTDIR / "table2_cluster_host_pressure_v1.png"),
                str(OUTDIR / "table2_numeric_feature_correlations_v1.png"),
                str(OUTDIR / "table2_candidate_delta_chi2_scores_v1.png"),
                str(OUTDIR / "table2_candidate_nondegenerate_ratios_v1.png"),
            ],
        },
        "interpretation": {
            "what_success_means": (
                "Compact 38/41/43 spine rows align one-to-one with Table2 Cepheid rows, and at least one host/table "
                "direction survives the SH0ES 47-parameter projection enough to justify stricter validation."
            ),
            "what_failure_means": (
                "The table mapping works, but host/table directions do not survive strongly enough for a new model test."
            ),
            "truth_boundary": (
                "This does not fit or prove TAIRID. It maps Cepheid host/table residual pressure after the SH0ES compact ladder fit."
            ),
        },
    }

    write_json(OUTDIR / "table2_host_mapped_residual_pressure_v1_summary.json", summary)

    with open(OUTDIR / "table2_host_mapped_residual_pressure_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES Table2 host-mapped residual-pressure audit v1\n\n")
        f.write("Boundary: host/table residual-pressure audit only. Not proof. Not H0 resolution.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Table2 mapping:\n")
        f.write(json.dumps(map_status, indent=2, default=json_default) + "\n\n")

        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")

        f.write("Top hosts by residual pressure:\n")
        f.write(json.dumps(host_summary[:25], indent=2, default=json_default) + "\n\n")

        f.write("Top cluster-host alignments:\n")
        f.write(json.dumps(cluster_host_rows[:25], indent=2, default=json_default) + "\n\n")

        f.write("Top numeric feature correlations:\n")
        f.write(json.dumps(numeric_corr[:25], indent=2, default=json_default) + "\n\n")

        f.write("Best cases:\n")
        f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")

        f.write("Candidate audit:\n")
        f.write(json.dumps(audit_rows[:50], indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not prove H0 resolution.\n")
        f.write("- This only audits Cepheid/Table2 host-mapped residual pressure after the compact SH0ES fit.\n")

    print("")
    print("TAIRID SH0ES Table2 host-mapped residual-pressure audit v1 complete.")
    print("Created:")
    print("  tairid_shoes_table2_host_mapped_residual_pressure_v1_outputs/table2_host_mapped_residual_pressure_v1_summary.json")
    print("  tairid_shoes_table2_host_mapped_residual_pressure_v1_outputs/table2_host_mapped_residual_pressure_v1_summary.txt")
    print("  tairid_shoes_table2_host_mapped_residual_pressure_v1_outputs/table2_compact_host_mapped_rows_v1.csv")
    print("  tairid_shoes_table2_host_mapped_residual_pressure_v1_outputs/table2_host_residual_summary_v1.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is a host/table residual-pressure audit only.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
