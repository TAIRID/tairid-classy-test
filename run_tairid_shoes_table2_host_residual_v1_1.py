#!/usr/bin/env python3
"""
TAIRID SH0ES Table2 host residual audit v1.1.

Fresh-name replacement for the prior Table2 host-mapped residual-pressure audit.
It uses shorter filenames/workflow names to avoid GitHub Actions UI/name caching issues.

Boundary:
- Not proof of TAIRID.
- Not H0 resolution.
- Not a cosmology fit.
- Host/Table2 residual-pressure mapping after the original SH0ES compact ladder fit only.
"""

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

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

OUTDIR = Path("tairid_shoes_table2_host_residual_v1_1_outputs")
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
    "SH0ES_Data/table2.tex",
    "SH0ES_Data/table2.README",
    "SH0ES_Data/README.md",
    "SH0ES_Data/MCMC_utils.py",
    "SH0ES_Data/lstsq_results.txt",
]

SPINE = {38, 41, 43}
P42 = 42
P46 = 46
EPS = 1.0e-12
SEED = 42
RANDOM_CONTROL_REPEATS = 60


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
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:180]


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
        headers={"User-Agent": "TAIRID-SH0ES-table2-host-residual-v1-1", "Accept": "*/*"},
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        return response.read(), response.geturl(), response.headers.get("Content-Type", ""), getattr(response, "status", None)


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
            attempts.append({"label": label, "repo_path": repo_path, "candidate_kind": kind, "url": url, "status": "http_error", "http_code": exc.code, "error": str(exc)})
        except Exception as exc:
            attempts.append({"label": label, "repo_path": repo_path, "candidate_kind": kind, "url": url, "status": "download_failed", "error": str(exc)})
    return {"label": label, "repo_path": repo_path, "status": "failed", "local_path": None, "pointer_info": pointer_info, "attempts": attempts}


def extract_first_numeric_fits_array(path):
    from astropy.io import fits
    with fits.open(path, memmap=True) as hdul:
        for idx, hdu in enumerate(hdul):
            data = hdu.data
            if data is None:
                continue
            try:
                if getattr(data.dtype, "fields", None):
                    numeric = []
                    for name in data.dtype.fields:
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
                arr = arr.astype(float)
                if arr.size:
                    return arr, {"hdu_index": idx, "hdu_name": hdu.name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
            except Exception:
                continue
    raise RuntimeError(f"No numeric FITS array found in {path}")


def orient_design_matrix(L, y_length):
    L = np.asarray(L, float)
    if L.ndim != 2:
        return None, {"status": "L_not_2d", "L_shape": list(L.shape)}
    if L.shape[0] == y_length and L.shape[1] != y_length:
        return L, {"status": "ok", "orientation": "L_is_observation_by_parameter", "X_shape": list(L.shape)}
    if L.shape[1] == y_length and L.shape[0] != y_length:
        X = L.T
        return X, {"status": "ok", "orientation": "L_transposed_to_observation_by_parameter", "original_L_shape": list(L.shape), "X_shape": list(X.shape)}
    return None, {"status": "no_axis_matches_y", "L_shape": list(L.shape), "y_length": int(y_length)}


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
                factor = cho_factor(C_sym + np.eye(C_sym.shape[0]) * jitter, lower=True, check_finite=False)
            attempts.append({"attempt": attempt, "jitter": jitter, "status": "success"})
            return factor, C_sym, jitter, attempts
        except Exception as exc:
            attempts.append({"attempt": attempt, "jitter": jitter, "status": "failed", "error": str(exc)})
            jitter = max(scale * 1.0e-12, 1.0e-14) if jitter == 0.0 else jitter * 10.0
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


def h0_like(beta):
    return float(10.0 ** (beta[P46] / 5.0)) if len(beta) > P46 else None


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
    cache = []
    for i in range(X.shape[0]):
        active, signs, full_key, active_key, sign_key = row_signature(X[i, :])
        keys.append(full_key)
        cache.append((active, signs, full_key, active_key, sign_key))
    cluster_counts = Counter(keys)
    signature_to_id = {}
    for key in keys:
        if key not in signature_to_id:
            signature_to_id[key] = len(signature_to_id)
    residual = baseline["residual"]
    cinv_residual = baseline["Cinv_residual"]
    cov_diag = np.diag(C_sym)
    leverage = np.einsum("ij,jk,ik->i", X, baseline["normal_inv"], baseline["Cinv_X"])
    rows = []
    grouped = defaultdict(list)
    for i, (active, signs, full_key, active_key, sign_key) in enumerate(cache):
        active_set = set(map(int, active))
        family = classify_row(active, signs)
        contains_spine = SPINE.issubset(active_set)
        varying_cols = sorted(active_set - SPINE)
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
        rows.append(row)
        grouped[full_key].append(row)
    clusters = []
    for key, group in grouped.items():
        res = np.asarray([r["baseline_residual"] for r in group], dtype=float)
        first = group[0]
        clusters.append({
            "signature_cluster_id": first["signature_cluster_id"],
            "size": len(group),
            "equation_family": first["equation_family"],
            "active_cols": first["active_cols"],
            "sign_pattern": first["sign_pattern"],
            "contains_38_41_43_spine": first["contains_38_41_43_spine"],
            "varying_cols_outside_38_41_43": first["varying_cols_outside_38_41_43"],
            "first_row": int(group[0]["observation_index"]),
            "last_row": int(group[-1]["observation_index"]),
            "mean_residual": float(np.mean(res)),
            "rms_residual": float(np.sqrt(np.mean(res * res))),
            "mean_abs_residual": float(np.mean(np.abs(res))),
            "max_abs_residual": float(np.max(np.abs(res))),
            "dominant_host_guess": "",
        })
    clusters = sorted(clusters, key=lambda r: (-r["mean_abs_residual"], -r["size"], r["signature_cluster_id"]))
    return rows, clusters


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
    match = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", str(text).replace(",", ""))
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
    header_rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if "&" not in raw or raw.startswith("%"):
            continue
        if any(tok in raw for tok in ["\\begin", "\\end", "\\hline", "\\table"]):
            continue
        cells = [clean_latex_cell(c) for c in raw.split("&")]
        cells = [c for c in cells if c != ""]
        if len(cells) < 2:
            continue
        numeric_count = sum(1 for c in cells if first_float(c) is not None)
        joined = " ".join(cells).lower()
        header_like = any(term in joined for term in ["host", "field", "period", "f160", "m160", "cepheid", "metal"])
        rec = {
            "table2_parse_index": len(all_rows),
            "source_line_number": line_no,
            "cell_count": len(cells),
            "numeric_count": numeric_count,
            "cells_json": json.dumps(cells),
            "row_text": raw[:1400],
        }
        all_rows.append(rec)
        if numeric_count >= 3 and not (header_like and numeric_count < 4):
            data_rows.append(rec)
        else:
            header_rows.append(rec)
    write_csv(OUTDIR / "table2_all_rows_v1_1.csv", all_rows)
    write_csv(OUTDIR / "table2_data_rows_v1_1.csv", data_rows)
    write_csv(OUTDIR / "table2_header_rows_v1_1.csv", header_rows)
    return all_rows, data_rows, header_rows


def infer_host(cells):
    patterns = [r"\bN\d{3,5}[A-Za-z]?\b", r"\bM\d{1,3}\b", r"\bLMC\b", r"\bSMC\b", r"\bM31\b", r"\bN4258\b", r"\bU\w+\b"]
    for cell in cells[:5]:
        text = str(cell).strip()
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.I)
            if m:
                return m.group(0).upper()
    for cell in cells[:4]:
        text = str(cell).strip()
        if text and first_float(text) is None:
            return safe_name(text.upper())[:60]
    return "UNKNOWN"


def map_table2_to_spine(row_rows, table2_data_rows):
    spine_rows = [r for r in row_rows if r["contains_38_41_43_spine"]]
    spine_rows = sorted(spine_rows, key=lambda r: r["observation_index"])
    table_rows = list(table2_data_rows)
    if len(table_rows) > len(spine_rows):
        table_rows = table_rows[-len(spine_rows):]
    n = min(len(spine_rows), len(table_rows))
    parsed_cells = [json.loads(table_rows[i]["cells_json"]) for i in range(n)]
    max_cells = max([len(c) for c in parsed_cells], default=0)
    mapped = []
    for i in range(n):
        comp = spine_rows[i]
        tab = table_rows[i]
        cells = parsed_cells[i]
        host = infer_host(cells)
        out = {
            "mapped_index": i,
            "compact_observation_index": comp["observation_index"],
            "compact_signature_cluster_id": comp["signature_cluster_id"],
            "compact_signature_cluster_size": comp["signature_cluster_size"],
            "compact_active_cols": comp["active_cols"],
            "compact_sign_pattern": comp["sign_pattern"],
            "compact_varying_cols_outside_38_41_43": comp["varying_cols_outside_38_41_43"],
            "host_guess": host,
            "table2_source_line_number": tab["source_line_number"],
            "table2_cell_count": tab["cell_count"],
            "table2_numeric_count": tab["numeric_count"],
            "baseline_residual": comp["baseline_residual"],
            "abs_baseline_residual": comp["abs_baseline_residual"],
            "cinv_residual": comp["cinv_residual"],
            "abs_cinv_residual": comp["abs_cinv_residual"],
            "cov_diag": comp["cov_diag"],
            "leverage_proxy": comp["leverage_proxy"],
            "abs_leverage_proxy": comp["abs_leverage_proxy"],
        }
        for j in range(max_cells):
            cell = cells[j] if j < len(cells) else ""
            out[f"table2_cell_{j}"] = cell
            num = first_float(cell)
            out[f"table2_num_{j}"] = num if num is not None else ""
        mapped.append(out)
    write_csv(OUTDIR / "table2_compact_host_mapped_rows_v1_1.csv", mapped)
    status = {
        "compact_spine_rows": len(spine_rows),
        "table2_data_rows": len(table2_data_rows),
        "mapped_rows": n,
        "exact_count_match": bool(len(spine_rows) == len(table2_data_rows)),
        "first_compact_spine_row": int(spine_rows[0]["observation_index"]) if spine_rows else None,
        "last_compact_spine_row": int(spine_rows[-1]["observation_index"]) if spine_rows else None,
    }
    return mapped, status


def summarize_hosts(mapped):
    grouped = defaultdict(list)
    for row in mapped:
        grouped[row["host_guess"]].append(row)
    out = []
    for host, rows in grouped.items():
        res = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        cinv = np.asarray([r["cinv_residual"] for r in rows], dtype=float)
        cluster_counts = Counter(r["compact_signature_cluster_id"] for r in rows)
        out.append({
            "host_guess": host,
            "row_count": len(rows),
            "mean_residual": float(np.mean(res)),
            "median_residual": float(np.median(res)),
            "rms_residual": float(np.sqrt(np.mean(res * res))),
            "mean_abs_residual": float(np.mean(np.abs(res))),
            "max_abs_residual": float(np.max(np.abs(res))),
            "mean_cinv_residual": float(np.mean(cinv)),
            "mean_abs_cinv_residual": float(np.mean(np.abs(cinv))),
            "first_mapped_index": int(rows[0]["mapped_index"]),
            "last_mapped_index": int(rows[-1]["mapped_index"]),
            "dominant_signature_clusters_json": json.dumps(dict(cluster_counts.most_common(8))),
        })
    out = sorted(out, key=lambda r: (-r["mean_abs_residual"], -r["row_count"], r["host_guess"]))
    write_csv(OUTDIR / "table2_host_summary_v1_1.csv", out)
    return out


def summarize_cluster_hosts(mapped):
    grouped = defaultdict(list)
    for row in mapped:
        grouped[row["compact_signature_cluster_id"]].append(row)
    out = []
    for cid, rows in grouped.items():
        res = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        host_counts = Counter(r["host_guess"] for r in rows)
        first = rows[0]
        out.append({
            "signature_cluster_id": cid,
            "row_count": len(rows),
            "active_cols": first["compact_active_cols"],
            "varying_cols_outside_38_41_43": first["compact_varying_cols_outside_38_41_43"],
            "dominant_host": host_counts.most_common(1)[0][0],
            "dominant_host_count": host_counts.most_common(1)[0][1],
            "host_counts_json": json.dumps(dict(host_counts.most_common(10))),
            "mean_residual": float(np.mean(res)),
            "rms_residual": float(np.sqrt(np.mean(res * res))),
            "mean_abs_residual": float(np.mean(np.abs(res))),
            "max_abs_residual": float(np.max(np.abs(res))),
            "first_compact_row": int(rows[0]["compact_observation_index"]),
            "last_compact_row": int(rows[-1]["compact_observation_index"]),
        })
    out = sorted(out, key=lambda r: (-r["mean_abs_residual"], -r["row_count"], r["signature_cluster_id"]))
    write_csv(OUTDIR / "table2_cluster_host_summary_v1_1.csv", out)
    return out


def numeric_feature_correlations(mapped):
    if not mapped:
        return []
    num_cols = sorted([k for k in mapped[0].keys() if k.startswith("table2_num_")])
    res = np.asarray([r["baseline_residual"] for r in mapped], dtype=float)
    abs_res = np.abs(res)
    out = []
    for col in num_cols:
        vals = []
        keep_res = []
        keep_abs = []
        for row, rr, aa in zip(mapped, res, abs_res):
            value = row.get(col, "")
            try:
                v = float(value)
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(v)
                keep_res.append(rr)
                keep_abs.append(aa)
        if len(vals) < 50:
            continue
        x = np.asarray(vals, dtype=float)
        if np.std(x) <= 1.0e-14:
            continue
        y1 = np.asarray(keep_res, dtype=float)
        y2 = np.asarray(keep_abs, dtype=float)
        out.append({
            "numeric_column": col,
            "valid_count": len(x),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "corr_with_residual": float(np.corrcoef(x, y1)[0, 1]),
            "corr_with_abs_residual": float(np.corrcoef(x, y2)[0, 1]),
        })
    out = sorted(out, key=lambda r: -abs(r["corr_with_abs_residual"]))
    write_csv(OUTDIR / "table2_numeric_correlations_v1_1.csv", out)
    return out


def standardize(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    sd = float(np.std(v))
    if not np.isfinite(sd) or sd <= 1.0e-14:
        return np.zeros_like(v)
    return (v - float(np.mean(v))) / sd


def audit_vector(name, values, kind, y, X, c_factor, baseline):
    raw = np.asarray(values, dtype=float).reshape(-1)
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
    perp_norm2 = float(z_perp.T @ c_inv_z_perp)
    raw_norm = float(math.sqrt(max(raw_norm2, 0.0)))
    perp_norm = float(math.sqrt(max(perp_norm2, 0.0)))
    ratio = float(perp_norm / max(raw_norm, EPS))
    if perp_norm2 <= EPS:
        score = 0.0
        delta = 0.0
        alpha = 0.0
    else:
        score = float(z_perp.T @ baseline["Cinv_residual"])
        delta = float((score * score) / perp_norm2)
        alpha = float(score / perp_norm2)
    p = float(chi2.sf(max(delta, 0.0), 1))
    return {
        "candidate": name,
        "candidate_kind": kind,
        "status": "ok",
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "count_nonzero_raw": int(np.sum(np.abs(raw) > 1.0e-12)),
        "raw_Cinv_norm": raw_norm,
        "residualized_Cinv_norm": perp_norm,
        "nondegenerate_ratio": ratio,
        "projection_absorption_fraction": float(1.0 - ratio),
        "score": score,
        "alpha_hat_added_column": alpha,
        "delta_chi2_score": delta,
        "p_value_chi2_one_dof": p,
        "delta_aic_if_added_column": float(2.0 - delta),
        "delta_bic_if_added_column": float(math.log(len(y)) - delta),
        "would_improve_aic": bool(2.0 - delta < 0.0),
        "would_improve_bic": bool(math.log(len(y)) - delta < 0.0),
    }


def build_audit_vectors(mapped, host_summary, cluster_summary, numeric_corr, y_length):
    vectors = []
    def add(name, values, kind):
        values = np.asarray(values, dtype=float)
        if len(values) != y_length:
            raise ValueError(f"{name}: vector length {len(values)} != {y_length}")
        vectors.append({"name": name, "values": values, "kind": kind})
    for host in host_summary[:20]:
        if host["row_count"] < 5:
            continue
        mask = np.zeros(y_length, dtype=float)
        for row in mapped:
            if row["host_guess"] == host["host_guess"]:
                mask[int(row["compact_observation_index"])] = 1.0
        add(f"host_{safe_name(host['host_guess'])}_n{host['row_count']}", mask, "host_mask")
    top_hosts = {h["host_guess"] for h in host_summary[:10] if h["row_count"] >= 5}
    mask = np.zeros(y_length, dtype=float)
    for row in mapped:
        if row["host_guess"] in top_hosts:
            mask[int(row["compact_observation_index"])] = 1.0
    add("host_union_top10_pressure", mask, "host_union")
    for cluster in cluster_summary[:20]:
        if cluster["row_count"] < 5:
            continue
        mask = np.zeros(y_length, dtype=float)
        for row in mapped:
            if row["compact_signature_cluster_id"] == cluster["signature_cluster_id"]:
                mask[int(row["compact_observation_index"])] = 1.0
        add(f"cluster_{cluster['signature_cluster_id']}_host_{safe_name(cluster['dominant_host'])}_n{cluster['row_count']}", mask, "cluster_host_mask")
    for item in numeric_corr[:12]:
        col = item["numeric_column"]
        values = np.zeros(y_length, dtype=float)
        vals = []
        for row in mapped:
            try:
                v = float(row.get(col, ""))
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            continue
        fill = float(np.nanmean(vals))
        for row in mapped:
            idx = int(row["compact_observation_index"])
            try:
                v = float(row.get(col, fill))
            except Exception:
                v = fill
            values[idx] = v if np.isfinite(v) else fill
        add(f"numeric_{col}", values, "table2_numeric_feature")
        mapped_vals = np.asarray([values[int(r["compact_observation_index"])] for r in mapped], dtype=float)
        high = np.percentile(mapped_vals, 90)
        low = np.percentile(mapped_vals, 10)
        high_mask = np.zeros(y_length, dtype=float)
        low_mask = np.zeros(y_length, dtype=float)
        for row in mapped:
            idx = int(row["compact_observation_index"])
            if values[idx] >= high:
                high_mask[idx] = 1.0
            if values[idx] <= low:
                low_mask[idx] = 1.0
        add(f"numeric_{col}_top10", high_mask, "table2_numeric_extreme")
        add(f"numeric_{col}_bottom10", low_mask, "table2_numeric_extreme")
    order = np.zeros(y_length, dtype=float)
    for row in mapped:
        order[int(row["compact_observation_index"])] = float(row["mapped_index"])
    add("table2_mapped_row_order_control", order, "control")
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
    def stats(group, prefix):
        ratios = np.asarray([r["nondegenerate_ratio"] for r in group], dtype=float)
        deltas = np.asarray([r["delta_chi2_score"] for r in group], dtype=float)
        return {
            f"{prefix}_ratio_mean": float(np.mean(ratios)),
            f"{prefix}_ratio_95": float(np.percentile(ratios, 95)),
            f"{prefix}_delta_mean": float(np.mean(deltas)),
            f"{prefix}_delta_95": float(np.percentile(deltas, 95)),
            f"{prefix}_delta_99": float(np.percentile(deltas, 99)),
        }
    return {"label": label, "status": "ok", "count": count, "repeats_per_control_type": RANDOM_CONTROL_REPEATS, **stats(same, "same_count"), **stats(block, "contiguous_block")}, rows


def code_context_search(aux_results):
    terms = ["table2", "host", "field", "ceph", "cepheid", "period", "metal", "F160", "m160", "color", "anchor", "calibrator", "muhat", "intercept", "fivelogH0", "H0", "alll", "ally", "allc"]
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
        (OUTDIR / f"preview_{safe_name(item['repo_path'])}.txt").write_text("\n".join(lines[:220]), encoding="utf-8")
        for i, line in enumerate(lines, start=1):
            hits = [term for term in terms if term.lower() in line.lower()]
            if hits:
                lo = max(1, i - 2)
                hi = min(len(lines), i + 2)
                context = "\n".join(f"{j}: {lines[j-1]}" for j in range(lo, hi + 1))
                rows.append({"repo_path": item["repo_path"], "line_number": i, "hit_terms": " | ".join(hits), "line": line[:700], "context": context[:1600]})
    write_csv(OUTDIR / "table2_host_context_hits_v1_1.csv", rows)
    return rows


def make_plots(mapped, host_summary, cluster_summary, numeric_corr, audit_rows):
    if not mapped:
        return
    x = np.asarray([r["mapped_index"] for r in mapped], dtype=float)
    res = np.asarray([r["baseline_residual"] for r in mapped], dtype=float)
    plt.figure(figsize=(11, 5))
    plt.plot(x, res, linewidth=0.8)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Table2 mapped row index")
    plt.ylabel("compact baseline residual")
    plt.title("Table2 mapped compact residuals v1.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "table2_mapped_residual_by_row_v1_1.png", dpi=160)
    plt.close()
    top = host_summary[:25]
    plt.figure(figsize=(13, 6))
    plt.bar(np.arange(len(top)), [h["mean_abs_residual"] for h in top])
    plt.xticks(np.arange(len(top)), [h["host_guess"] for h in top], rotation=45, ha="right")
    plt.ylabel("mean absolute residual")
    plt.title("Highest-pressure Table2 hosts v1.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "table2_host_pressure_v1_1.png", dpi=160)
    plt.close()
    topc = cluster_summary[:30]
    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(topc)), [c["mean_abs_residual"] for c in topc])
    plt.xticks(np.arange(len(topc)), [f"{c['signature_cluster_id']}:{c['dominant_host']}" for c in topc], rotation=60, ha="right", fontsize=8)
    plt.ylabel("mean absolute residual")
    plt.title("High-pressure compact clusters aligned to Table2 hosts v1.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "table2_cluster_host_pressure_v1_1.png", dpi=160)
    plt.close()
    plot_rows = [r for r in audit_rows if r.get("candidate_kind") not in ["random_same_count_control", "random_contiguous_block_control"]][:40]
    if plot_rows:
        plt.figure(figsize=(15, 6))
        plt.bar(np.arange(len(plot_rows)), [r.get("delta_chi2_score", 0.0) for r in plot_rows])
        plt.xticks(np.arange(len(plot_rows)), [r["candidate"] for r in plot_rows], rotation=65, ha="right", fontsize=7)
        plt.ylabel("delta chi2 score")
        plt.title("Host/Table2 candidate residual alignment v1.1")
        plt.tight_layout()
        plt.savefig(OUTDIR / "table2_candidate_delta_chi2_v1_1.png", dpi=160)
        plt.close()
        plt.figure(figsize=(15, 6))
        plt.bar(np.arange(len(plot_rows)), [r.get("nondegenerate_ratio", 0.0) for r in plot_rows])
        plt.xticks(np.arange(len(plot_rows)), [r["candidate"] for r in plot_rows], rotation=65, ha="right", fontsize=7)
        plt.ylabel("nondegenerate ratio")
        plt.title("Host/Table2 candidate survival after SH0ES projection v1.1")
        plt.tight_layout()
        plt.savefig(OUTDIR / "table2_candidate_nondegenerate_v1_1.png", dpi=160)
        plt.close()


def decide_status(map_status, audit_rows, host_summary, numeric_corr, random_summaries):
    ok = [r for r in audit_rows if r.get("status") == "ok"]
    hosts = [r for r in ok if r.get("candidate_kind") in ["host_mask", "host_union"]]
    clusters = [r for r in ok if r.get("candidate_kind") == "cluster_host_mask"]
    numerics = [r for r in ok if r.get("candidate_kind") in ["table2_numeric_feature", "table2_numeric_extreme"]]
    best_host = max(hosts, key=lambda r: r["delta_chi2_score"]) if hosts else None
    best_cluster = max(clusters, key=lambda r: r["delta_chi2_score"]) if clusters else None
    best_numeric = max(numerics, key=lambda r: r["delta_chi2_score"]) if numerics else None
    best_any = max(ok, key=lambda r: r["delta_chi2_score"]) if ok else None
    def locked(row):
        return bool(row and row["p_value_chi2_one_dof"] <= 0.01 and row["delta_bic_if_added_column"] < -2.0 and row["nondegenerate_ratio"] >= 0.05)
    def directional(row):
        return bool(row and row["p_value_chi2_one_dof"] <= 0.05 and row["delta_aic_if_added_column"] < 0.0 and row["nondegenerate_ratio"] >= 0.02)
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
        return "table2_host_map_count_mismatch_but_audited", 6, "Table2 and compact spine rows did not match exactly; inspect parser before relying on host labels.", best_cases
    if locked(best_host) or locked(best_cluster) or locked(best_numeric):
        return "table2_host_mapped_residual_pressure_detected_for_followup", 8, "A host/table feature survived SH0ES projection and aligns with residual pressure; build a stricter validation pass.", best_cases
    if directional(best_host) or directional(best_cluster) or directional(best_numeric):
        return "table2_host_mapped_residual_pressure_directional_not_locked", 7, "Host/table residual pressure is directional but not locked; inspect controls and labels before modeling.", best_cases
    return "table2_host_mapped_no_locked_residual_structure", 6, "Table2 mapping succeeded, but no host/table direction is strong enough for a new likelihood model.", best_cases


def main():
    print("TAIRID SH0ES Table2 host residual audit v1.1 starting.")
    print("Boundary: host/table residual-pressure audit only; not proof.")

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
            "attempt_count": len(result.get("attempts", [])),
        })

    write_csv(OUTDIR / "download_ledger_v1_1.csv", ledger)
    write_json(OUTDIR / "download_attempts_v1_1.json", {"compact": downloads, "auxiliary": aux_results})
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
    write_json(OUTDIR / "parse_meta_v1_1.json", parse_meta)
    write_json(OUTDIR / "parse_errors_v1_1.json", parse_errors)

    table2 = next((x for x in aux_results if x.get("repo_path") == "SH0ES_Data/table2.tex" and x.get("status") == "downloaded"), None)
    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]) or not table2:
        summary = {
            "test_name": "TAIRID SH0ES Table2 host residual audit v1.1",
            "boundary": "Download/parse failure. No host-map result.",
            "final_status": "table2_host_residual_v1_1_parse_or_download_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix or table2 retrieval before host-mapped audit.",
            "parse_errors": parse_errors,
            "table2_downloaded": bool(table2),
        }
        write_json(OUTDIR / "table2_host_residual_v1_1_summary.json", summary)
        print("Parse/download failed. See summary JSON.")
        return

    table2_all, table2_data, table2_headers = parse_table2_tex(Path(table2["local_path"]))

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
    write_csv(OUTDIR / "compact_row_map_v1_1.csv", row_rows)
    write_csv(OUTDIR / "compact_cluster_map_v1_1.csv", cluster_rows)

    mapped, map_status = map_table2_to_spine(row_rows, table2_data)
    host_summary = summarize_hosts(mapped)
    cluster_host_summary = summarize_cluster_hosts(mapped)
    numeric_corr = numeric_feature_correlations(mapped)

    vectors = build_audit_vectors(mapped, host_summary, cluster_host_summary, numeric_corr, len(y))
    audit_rows = [audit_vector(v["name"], v["values"], v["kind"], y, X, c_factor, baseline) for v in vectors]
    audit_rows = sorted(audit_rows, key=lambda r: (-float(r.get("delta_chi2_score", 0.0)), -float(r.get("nondegenerate_ratio", 0.0))))
    write_csv(OUTDIR / "candidate_audit_v1_1.csv", audit_rows)

    random_summaries = []
    random_detail_rows = []
    seen_counts = set()
    for candidate in [r for r in audit_rows if r.get("status") == "ok" and r.get("candidate_kind") != "control" and 0 < int(r.get("count_nonzero_raw", 0)) < len(y)][:3]:
        count = int(candidate["count_nonzero_raw"])
        if count in seen_counts:
            continue
        seen_counts.add(count)
        summary_random, rows_random = random_control_audit(y, X, c_factor, baseline, count, f"count_{count}")
        random_summaries.append(summary_random)
        random_detail_rows.extend(rows_random)
    write_json(OUTDIR / "random_control_summaries_v1_1.json", random_summaries)
    write_csv(OUTDIR / "random_control_details_v1_1.csv", random_detail_rows)

    final_status, readiness_score, next_wall, best_cases = decide_status(map_status, audit_rows, host_summary, numeric_corr, random_summaries)

    make_plots(mapped, host_summary, cluster_host_summary, numeric_corr, audit_rows)

    residual = baseline["residual"]
    abs_resid = np.abs(residual)
    summary = {
        "test_name": "TAIRID SH0ES Table2 host residual audit v1.1",
        "boundary": "Host/table residual-pressure audit only. Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model.",
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
            "param46_H0_like": h0_like(baseline["beta"]),
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
            "table2_all_rows": len(table2_all),
            "table2_data_rows": len(table2_data),
            "table2_header_rows": len(table2_headers),
            "mapped_rows": len(mapped),
            "unique_host_guesses": len(host_summary),
        },
        "top_hosts_by_residual_pressure": host_summary[:40],
        "top_cluster_host_alignments": cluster_host_summary[:40],
        "top_numeric_feature_correlations": numeric_corr[:40],
        "candidate_audit": audit_rows[:80],
        "random_control_summaries": random_summaries,
        "best_cases": best_cases,
        "code_context_hits_count": len(code_hits),
        "output_files": {
            "summary_json": str(OUTDIR / "table2_host_residual_v1_1_summary.json"),
            "summary_txt": str(OUTDIR / "table2_host_residual_v1_1_summary.txt"),
            "mapped_rows_csv": str(OUTDIR / "table2_compact_host_mapped_rows_v1_1.csv"),
            "host_summary_csv": str(OUTDIR / "table2_host_summary_v1_1.csv"),
            "cluster_host_summary_csv": str(OUTDIR / "table2_cluster_host_summary_v1_1.csv"),
            "numeric_correlations_csv": str(OUTDIR / "table2_numeric_correlations_v1_1.csv"),
            "candidate_audit_csv": str(OUTDIR / "candidate_audit_v1_1.csv"),
            "random_control_summaries_json": str(OUTDIR / "random_control_summaries_v1_1.json"),
            "random_control_details_csv": str(OUTDIR / "random_control_details_v1_1.csv"),
            "context_hits_csv": str(OUTDIR / "table2_host_context_hits_v1_1.csv"),
        },
        "interpretation": {
            "success_condition": "Compact 38/41/43 spine rows align with Table2 Cepheid rows and at least one host/table direction survives projection enough to justify stricter validation.",
            "failure_condition": "The mapping works, but host/table directions are absorbed or not residual-aligned strongly enough for a new model.",
            "truth_boundary": "This does not prove TAIRID. It only audits Cepheid/Table2 host-mapped residual pressure after the SH0ES compact ladder fit.",
        },
    }
    write_json(OUTDIR / "table2_host_residual_v1_1_summary.json", summary)
    with open(OUTDIR / "table2_host_residual_v1_1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES Table2 host residual audit v1.1\n\n")
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
        f.write(json.dumps(cluster_host_summary[:25], indent=2, default=json_default) + "\n\n")
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
    print("TAIRID SH0ES Table2 host residual audit v1.1 complete.")
    print("Created:")
    print("  tairid_shoes_table2_host_residual_v1_1_outputs/table2_host_residual_v1_1_summary.json")
    print("  tairid_shoes_table2_host_residual_v1_1_outputs/table2_host_residual_v1_1_summary.txt")
    print("  tairid_shoes_table2_host_residual_v1_1_outputs/table2_compact_host_mapped_rows_v1_1.csv")
    print("  tairid_shoes_table2_host_residual_v1_1_outputs/table2_host_summary_v1_1.csv")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
