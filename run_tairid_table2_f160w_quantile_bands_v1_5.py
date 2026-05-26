#!/usr/bin/env python3
"""
TAIRID Table2 F160W quantile-band localization audit v1.5.

Purpose:
v1.4 found a directional faint-side F160W threshold signal, but cumulative
thresholds could not tell whether the pressure was a sharp tail, a broad
faint-side regime, or ordinary measurement-selection behavior.

This test uses non-overlapping F160W rank bands:

    0-5, 5-10, 10-15, ... 95-100 percent

Because F160W is a magnitude-like column:
    lower F160W = brighter
    higher F160W = fainter

So:
    0-5% is the brightest tail
    95-100% is the faintest tail

This test asks:
- Which non-overlapping F160W bands carry residual pressure?
- Does the pressure localize to the faintest tail?
- Does it spread across the full faint side?
- Does it remain after host, row-order, uncertainty, and measurement controls?
- Does it beat permutation controls?

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only tests where the SH0ES Table2 F160W residual pressure localizes.
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
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


OUTDIR = Path("tairid_table2_f160w_quantile_bands_v1_5_outputs")
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
    "SH0ES_Data/table2.tex",
    "SH0ES_Data/table2.README",
    "SH0ES_Data/README.md",
    "SH0ES_Data/MCMC_utils.py",
    "SH0ES_Data/lstsq_results.txt",
]

SPINE_COLS = {38, 41, 43}
P42 = 42
P46 = 46
EPS = 1.0e-12
SEED = 42
PERMUTATION_REPEATS = 60
BAND_EDGES = list(range(0, 105, 5))


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


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))[:180]


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
            "User-Agent": "TAIRID-Table2-F160W-quantile-bands-v1-5",
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
        for hdu_index, hdu in enumerate(hdul):
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
                        "hdu_index": hdu_index,
                        "hdu_name": hdu.name,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }

            except Exception:
                continue

    raise RuntimeError(f"No numeric FITS array found in {path}")


def orient_design_matrix(L, y_length):
    L = np.asarray(L, dtype=float)

    if L.ndim != 2:
        return None, {
            "status": "L_not_2d",
            "L_shape": list(L.shape),
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
            "status": "ambiguous_square_using_L",
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

    raise RuntimeError("Cholesky failed even after jitter attempts.")


def gls_fit(y, D, c_factor, rcond=1.0e-12):
    D = np.asarray(D, dtype=float)
    c_inv_y = cho_solve(c_factor, y, check_finite=False)
    c_inv_D = cho_solve(c_factor, D, check_finite=False)

    y_cinv_y = float(y.T @ c_inv_y)
    normal = D.T @ c_inv_D
    rhs = D.T @ c_inv_y

    normal_inv = np.linalg.pinv(normal, rcond=rcond)
    beta = normal_inv @ rhs

    chi2_value = float(y_cinv_y - 2.0 * beta.T @ rhs + beta.T @ normal @ beta)
    residual = y - D @ beta
    c_inv_residual = c_inv_y - c_inv_D @ beta

    n = len(y)
    k = D.shape[1]

    return {
        "D": D,
        "Cinv_y": c_inv_y,
        "Cinv_D": c_inv_D,
        "y_Cinv_y": y_cinv_y,
        "normal": normal,
        "normal_inv": normal_inv,
        "rhs": rhs,
        "beta": beta,
        "residual": residual,
        "Cinv_residual": c_inv_residual,
        "chi2": chi2_value,
        "dof": int(n - k),
        "k": int(k),
        "aic": float(chi2_value + 2 * k),
        "bic": float(chi2_value + k * math.log(n)),
        "reduced_chi2": float(chi2_value / max(n - k, 1)),
    }


def h0_like(beta):
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


def recover_compact_rows(X, y, C_sym, baseline_fit):
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

    residual = baseline_fit["residual"]
    cinv_residual = baseline_fit["Cinv_residual"]
    cov_diag = np.diag(C_sym)

    leverage = np.einsum(
        "ij,jk,ik->i",
        X,
        baseline_fit["normal_inv"],
        baseline_fit["Cinv_D"],
    )

    rows = []
    grouped = defaultdict(list)

    for i, (active, signs, full_key, active_key, sign_key) in enumerate(cache):
        active_set = set(map(int, active))
        family = classify_row(active, signs)
        contains_spine = SPINE_COLS.issubset(active_set)
        varying_cols = sorted(active_set - SPINE_COLS)

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
        abs_res = np.abs(res)
        first = group[0]

        clusters.append(
            {
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
                "median_residual": float(np.median(res)),
                "rms_residual": float(np.sqrt(np.mean(res * res))),
                "mean_abs_residual": float(np.mean(abs_res)),
                "max_abs_residual": float(np.max(abs_res)),
            }
        )

    clusters = sorted(
        clusters,
        key=lambda r: (-r["mean_abs_residual"], -r["size"], r["signature_cluster_id"]),
    )

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

    match = re.search(
        r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?",
        str(text).replace(",", ""),
    )

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

    for line_number, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()

        if "&" not in raw:
            continue

        if raw.startswith("%"):
            continue

        if any(token in raw for token in ["\\begin", "\\end", "\\hline", "\\table"]):
            continue

        cells = [clean_latex_cell(c) for c in raw.split("&")]
        cells = [c for c in cells if c != ""]

        if len(cells) < 2:
            continue

        numeric_count = sum(1 for c in cells if first_float(c) is not None)
        joined = " ".join(cells).lower()

        header_like = any(
            term in joined
            for term in ["host", "field", "period", "f160", "m160", "cepheid", "metal"]
        )

        row = {
            "table2_parse_index": len(all_rows),
            "source_line_number": line_number,
            "cell_count": len(cells),
            "numeric_count": numeric_count,
            "cells_json": json.dumps(cells),
            "row_text": raw[:1400],
        }

        all_rows.append(row)

        if numeric_count >= 3 and not (header_like and numeric_count < 4):
            data_rows.append(row)
        else:
            header_rows.append(row)

    write_csv(OUTDIR / "table2_all_rows_v1_5.csv", all_rows)
    write_csv(OUTDIR / "table2_data_rows_v1_5.csv", data_rows)
    write_csv(OUTDIR / "table2_header_rows_v1_5.csv", header_rows)

    return all_rows, data_rows, header_rows


def infer_host(cells):
    patterns = [
        r"\bN\d{3,5}[A-Za-z]?\b",
        r"\bM\d{1,3}\b",
        r"\bLMC\b",
        r"\bSMC\b",
        r"\bM31\b",
        r"\bN4258\b",
        r"\bU\w+\b",
    ]

    for cell in cells[:5]:
        text = str(cell).strip()

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)

            if match:
                return match.group(0).upper()

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

    mapped_count = min(len(spine_rows), len(table_rows))
    parsed_cells = []

    for i in range(mapped_count):
        parsed_cells.append(json.loads(table_rows[i]["cells_json"]))

    max_cells = max([len(cells) for cells in parsed_cells], default=0)
    mapped = []

    for i in range(mapped_count):
        compact = spine_rows[i]
        table = table_rows[i]
        cells = parsed_cells[i]
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

        for cell_index in range(max_cells):
            cell = cells[cell_index] if cell_index < len(cells) else ""
            number = first_float(cell)

            out[f"table2_cell_{cell_index}"] = cell
            out[f"table2_num_{cell_index}"] = number if number is not None else ""

        mapped.append(out)

    write_csv(OUTDIR / "table2_compact_host_mapped_rows_v1_5.csv", mapped)

    status = {
        "compact_spine_rows": len(spine_rows),
        "table2_data_rows": len(table2_data_rows),
        "mapped_rows": mapped_count,
        "exact_count_match": bool(len(spine_rows) == len(table2_data_rows)),
        "first_compact_spine_row": int(spine_rows[0]["observation_index"]) if spine_rows else None,
        "last_compact_spine_row": int(spine_rows[-1]["observation_index"]) if spine_rows else None,
    }

    return mapped, status


def summarize_hosts(mapped_rows):
    grouped = defaultdict(list)

    for row in mapped_rows:
        grouped[row["host_guess"]].append(row)

    host_rows = []

    for host, rows in grouped.items():
        residual = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        cluster_counts = Counter(r["compact_signature_cluster_id"] for r in rows)

        host_rows.append(
            {
                "host_guess": host,
                "row_count": len(rows),
                "mean_residual": float(np.mean(residual)),
                "median_residual": float(np.median(residual)),
                "rms_residual": float(np.sqrt(np.mean(residual * residual))),
                "mean_abs_residual": float(np.mean(np.abs(residual))),
                "max_abs_residual": float(np.max(np.abs(residual))),
                "first_mapped_index": int(rows[0]["mapped_index"]),
                "last_mapped_index": int(rows[-1]["mapped_index"]),
                "dominant_signature_clusters_json": json.dumps(dict(cluster_counts.most_common(8))),
            }
        )

    host_rows = sorted(
        host_rows,
        key=lambda r: (-r["mean_abs_residual"], -r["row_count"], r["host_guess"]),
    )

    write_csv(OUTDIR / "table2_host_summary_v1_5.csv", host_rows)

    return host_rows


def numeric_feature_summary(mapped_rows):
    if not mapped_rows:
        return [], {}

    numeric_columns = sorted([key for key in mapped_rows[0].keys() if key.startswith("table2_num_")])

    residual = np.asarray([r["baseline_residual"] for r in mapped_rows], dtype=float)
    abs_residual = np.abs(residual)

    rows = []

    for column in numeric_columns:
        values = []
        keep_residual = []
        keep_abs_residual = []

        for row, residual_value, abs_value in zip(mapped_rows, residual, abs_residual):
            value = row.get(column, "")

            try:
                numeric_value = float(value)
            except Exception:
                continue

            if np.isfinite(numeric_value):
                values.append(numeric_value)
                keep_residual.append(residual_value)
                keep_abs_residual.append(abs_value)

        if len(values) < 50:
            continue

        x = np.asarray(values, dtype=float)

        if np.std(x) <= 1.0e-14:
            continue

        y_signed = np.asarray(keep_residual, dtype=float)
        y_abs = np.asarray(keep_abs_residual, dtype=float)

        rows.append(
            {
                "numeric_column": column,
                "valid_count": len(x),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "corr_with_residual": float(np.corrcoef(x, y_signed)[0, 1]),
                "corr_with_abs_residual": float(np.corrcoef(x, y_abs)[0, 1]),
            }
        )

    rows = sorted(rows, key=lambda r: -abs(r["corr_with_abs_residual"]))

    likely = {
        "period": "table2_num_4",
        "color": "table2_num_5",
        "color_or_pre_f160w_sigma": "table2_num_6",
        "f160w": "table2_num_7",
        "f160w_sigma": "table2_num_8",
        "metallicity": "table2_num_9",
    }

    write_csv(OUTDIR / "table2_numeric_summary_v1_5.csv", rows)
    write_json(OUTDIR / "table2_likely_numeric_labels_v1_5.json", likely)

    return rows, likely


def standardize(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    std = float(np.std(values))

    if not np.isfinite(std) or std <= 1.0e-14:
        return np.zeros_like(values)

    return (values - float(np.mean(values))) / std


def nonzero_std(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return float(np.std(values)) > 1.0e-14


def vector_from_mapped_numeric(mapped_rows, y_length, column, fill_strategy="mean"):
    values = np.zeros(y_length, dtype=float)
    observed = []

    for row in mapped_rows:
        try:
            value = float(row.get(column, ""))
        except Exception:
            continue

        if np.isfinite(value):
            observed.append(value)

    if not observed:
        return values, {
            "column": column,
            "status": "no_observed_values",
            "observed_count": 0,
        }

    fill = float(np.mean(observed)) if fill_strategy == "mean" else 0.0

    for row in mapped_rows:
        idx = int(row["compact_observation_index"])

        try:
            value = float(row.get(column, fill))
        except Exception:
            value = fill

        values[idx] = value if np.isfinite(value) else fill

    meta = {
        "column": column,
        "status": "ok",
        "observed_count": len(observed),
        "fill": fill,
        "min": float(np.min(observed)),
        "max": float(np.max(observed)),
        "mean": float(np.mean(observed)),
        "std": float(np.std(observed)),
    }

    return values, meta


def host_control_matrix(mapped_rows, host_summary, y_length, top_n=10):
    controls = []
    names = []

    for host in host_summary[:top_n]:
        if host["row_count"] < 5:
            continue

        mask = np.zeros(y_length, dtype=float)

        for row in mapped_rows:
            if row["host_guess"] == host["host_guess"]:
                mask[int(row["compact_observation_index"])] = 1.0

        if nonzero_std(mask):
            controls.append(standardize(mask))
            names.append(f"host_control_{safe_name(host['host_guess'])}_n{host['row_count']}")

    if not controls:
        return np.empty((y_length, 0)), []

    return np.column_stack(controls), names


def controls_from_columns(mapped_rows, y_length, columns, label_prefix):
    controls = []
    names = []
    metadata = []

    for column in columns:
        values, meta = vector_from_mapped_numeric(mapped_rows, y_length, column)

        if nonzero_std(values):
            controls.append(standardize(values))
            names.append(f"{label_prefix}_{column}")
            metadata.append(meta)

    if not controls:
        return np.empty((y_length, 0)), [], metadata

    return np.column_stack(controls), names, metadata


def row_order_control(mapped_rows, y_length):
    values = np.zeros(y_length, dtype=float)

    for row in mapped_rows:
        values[int(row["compact_observation_index"])] = float(row["mapped_index"])

    return standardize(values).reshape(-1, 1), ["table2_row_order"]


def build_designs(X, mapped_rows, host_summary, y_length):
    designs = {}

    row_order, row_order_names = row_order_control(mapped_rows, y_length)
    host_controls, host_names = host_control_matrix(mapped_rows, host_summary, y_length, top_n=10)

    uncertainty_cols = ["table2_num_6", "table2_num_8"]
    uncertainty_controls, uncertainty_names, uncertainty_meta = controls_from_columns(
        mapped_rows,
        y_length,
        uncertainty_cols,
        "uncertainty",
    )

    measurement_cols = ["table2_num_4", "table2_num_5", "table2_num_6", "table2_num_8", "table2_num_9"]
    measurement_controls, measurement_names, measurement_meta = controls_from_columns(
        mapped_rows,
        y_length,
        measurement_cols,
        "measurement",
    )

    def make(name, blocks, block_names):
        valid_blocks = [block for block in blocks if block.shape[1] > 0]
        D = np.column_stack([X] + valid_blocks) if valid_blocks else X.copy()
        names = [f"original_param_{i}" for i in range(X.shape[1])]

        for current_names in block_names:
            names.extend(current_names)

        designs[name] = {
            "D": D,
            "names": names,
            "added_column_count": int(D.shape[1] - X.shape[1]),
        }

    make("original_47", [], [])
    make("plus_host_top10", [host_controls], [host_names])
    make("plus_uncertainty", [uncertainty_controls], [uncertainty_names])
    make("plus_measurement_controls", [measurement_controls], [measurement_names])
    make(
        "plus_host_top10_and_measurement_controls",
        [host_controls, measurement_controls],
        [host_names, measurement_names],
    )
    make(
        "plus_host_top10_row_order_measurement_controls",
        [host_controls, row_order, measurement_controls],
        [host_names, row_order_names, measurement_names],
    )

    metadata = {
        "host_control_names": host_names,
        "uncertainty_names": uncertainty_names,
        "uncertainty_metadata": uncertainty_meta,
        "measurement_names": measurement_names,
        "measurement_metadata": measurement_meta,
        "designs": {
            key: {
                "columns": value["D"].shape[1],
                "added_column_count": value["added_column_count"],
            }
            for key, value in designs.items()
        },
    }

    write_json(OUTDIR / "control_design_metadata_v1_5.json", metadata)

    return designs, metadata


def build_quantile_band_candidates(mapped_rows, y_length):
    f160w_values = []
    finite_rows = []

    for row in mapped_rows:
        try:
            value = float(row.get("table2_num_7", ""))
        except Exception:
            continue

        if np.isfinite(value):
            f160w_values.append(value)
            finite_rows.append(row)

    order = np.argsort(np.asarray(f160w_values, dtype=float))
    sorted_rows = [finite_rows[int(i)] for i in order]
    sorted_values = np.asarray([f160w_values[int(i)] for i in order], dtype=float)

    n = len(sorted_rows)
    candidates = []

    for low, high in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        start = int(math.floor(n * low / 100.0))
        end = int(math.floor(n * high / 100.0))

        if high == 100:
            end = n

        if end <= start:
            continue

        band_rows = sorted_rows[start:end]
        band_values = sorted_values[start:end]

        mask = np.zeros(y_length, dtype=float)

        for row in band_rows:
            mask[int(row["compact_observation_index"])] = 1.0

        center = 0.5 * (low + high)

        if high <= 25:
            region = "bright_tail"
        elif low >= 75:
            region = "faint_tail"
        elif center < 50:
            region = "bright_middle"
        else:
            region = "faint_middle"

        candidates.append(
            {
                "name": f"f160w_band_{low:02d}_{high:02d}",
                "values": mask,
                "kind": "f160w_quantile_band",
                "low_percentile": low,
                "high_percentile": high,
                "center_percentile": center,
                "region": region,
                "count": int(np.sum(mask > 0)),
                "f160w_min": float(np.min(band_values)),
                "f160w_max": float(np.max(band_values)),
                "f160w_mean": float(np.mean(band_values)),
            }
        )

    inventory = []

    for candidate in candidates:
        inventory.append(
            {
                "name": candidate["name"],
                "kind": candidate["kind"],
                "low_percentile": candidate["low_percentile"],
                "high_percentile": candidate["high_percentile"],
                "center_percentile": candidate["center_percentile"],
                "region": candidate["region"],
                "count": candidate["count"],
                "f160w_min": candidate["f160w_min"],
                "f160w_max": candidate["f160w_max"],
                "f160w_mean": candidate["f160w_mean"],
                "std": float(np.std(candidate["values"])),
            }
        )

    write_json(OUTDIR / "quantile_band_candidate_inventory_v1_5.json", inventory)

    return candidates


def audit_candidate_against_fit(candidate, y, c_factor, fit, design_name):
    raw = np.asarray(candidate["values"], dtype=float).reshape(-1)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    z = standardize(raw)

    c_inv_z = cho_solve(c_factor, z, check_finite=False)
    raw_norm2 = float(z.T @ c_inv_z)

    if raw_norm2 <= EPS:
        return {
            "design": design_name,
            "candidate": candidate["name"],
            "candidate_kind": candidate["kind"],
            "region": candidate["region"],
            "low_percentile": candidate["low_percentile"],
            "high_percentile": candidate["high_percentile"],
            "center_percentile": candidate["center_percentile"],
            "status": "zero_or_near_zero_raw_norm",
            "delta_chi2_score": 0.0,
            "p_value_chi2_one_dof": 1.0,
            "nondegenerate_ratio": 0.0,
            "delta_aic_if_added_column": 2.0,
            "delta_bic_if_added_column": float(math.log(len(y))),
            "count_nonzero_raw": int(np.sum(np.abs(raw) > 1.0e-12)),
        }

    x_t_cinv_z = fit["D"].T @ c_inv_z
    coeff = fit["normal_inv"] @ x_t_cinv_z

    z_perp = z - fit["D"] @ coeff
    c_inv_z_perp = c_inv_z - fit["Cinv_D"] @ coeff

    perp_norm2 = float(z_perp.T @ c_inv_z_perp)
    raw_norm = float(math.sqrt(max(raw_norm2, 0.0)))
    perp_norm = float(math.sqrt(max(perp_norm2, 0.0)))
    ratio = float(perp_norm / max(raw_norm, EPS))

    if perp_norm2 <= EPS:
        score = 0.0
        delta = 0.0
        alpha = 0.0
    else:
        score = float(z_perp.T @ fit["Cinv_residual"])
        delta = float((score * score) / perp_norm2)
        alpha = float(score / perp_norm2)

    p_value = float(chi2.sf(max(delta, 0.0), 1))

    return {
        "design": design_name,
        "candidate": candidate["name"],
        "candidate_kind": candidate["kind"],
        "region": candidate["region"],
        "low_percentile": candidate["low_percentile"],
        "high_percentile": candidate["high_percentile"],
        "center_percentile": candidate["center_percentile"],
        "f160w_min": candidate["f160w_min"],
        "f160w_max": candidate["f160w_max"],
        "f160w_mean": candidate["f160w_mean"],
        "status": "ok",
        "base_design_k": fit["k"],
        "base_chi2": fit["chi2"],
        "base_reduced_chi2": fit["reduced_chi2"],
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "count_nonzero_raw": int(np.sum(np.abs(raw) > 1.0e-12)),
        "raw_Cinv_norm": raw_norm,
        "residualized_Cinv_norm": perp_norm,
        "nondegenerate_ratio": ratio,
        "projection_absorption_fraction": float(1.0 - ratio),
        "score": score,
        "alpha_hat_added_column": alpha,
        "delta_chi2_score": delta,
        "p_value_chi2_one_dof": p_value,
        "delta_aic_if_added_column": float(2.0 - delta),
        "delta_bic_if_added_column": float(math.log(len(y)) - delta),
        "would_improve_aic": bool(2.0 - delta < 0.0),
        "would_improve_bic": bool(math.log(len(y)) - delta < 0.0),
    }


def run_audits(y, c_factor, design_fits, candidates):
    rows = []

    for design_name, fit in design_fits.items():
        for candidate in candidates:
            rows.append(
                audit_candidate_against_fit(candidate, y, c_factor, fit, design_name)
            )

    rows = sorted(
        rows,
        key=lambda r: (
            r["design"],
            float(r.get("center_percentile", -1.0)),
        ),
    )

    write_csv(OUTDIR / "quantile_band_audit_all_designs_v1_5.csv", rows)

    return rows


def permutation_controls(mapped_rows, y, c_factor, fit, candidate, design_name):
    rng = np.random.default_rng(SEED)
    rows = []

    raw = np.asarray(candidate["values"], dtype=float).reshape(-1)

    spine_indices = np.asarray(
        [int(row["compact_observation_index"]) for row in mapped_rows],
        dtype=int,
    )

    if len(spine_indices) == 0:
        return [], {}

    spine_values = raw[spine_indices].copy()

    for i in range(PERMUTATION_REPEATS):
        permuted = raw.copy()
        permuted_values = spine_values.copy()
        rng.shuffle(permuted_values)
        permuted[spine_indices] = permuted_values

        pseudo = {
            **candidate,
            "name": f"{candidate['name']}_permutation_{i}",
            "values": permuted,
            "kind": "permutation_control",
        }

        rows.append(audit_candidate_against_fit(pseudo, y, c_factor, fit, design_name))

    deltas = np.asarray([r["delta_chi2_score"] for r in rows], dtype=float)
    ratios = np.asarray([r["nondegenerate_ratio"] for r in rows], dtype=float)

    observed = audit_candidate_against_fit(candidate, y, c_factor, fit, design_name)

    summary = {
        "design": design_name,
        "candidate": candidate["name"],
        "region": candidate["region"],
        "low_percentile": candidate["low_percentile"],
        "high_percentile": candidate["high_percentile"],
        "center_percentile": candidate["center_percentile"],
        "repeats": PERMUTATION_REPEATS,
        "observed_delta_chi2": observed["delta_chi2_score"],
        "observed_nondegenerate_ratio": observed["nondegenerate_ratio"],
        "permutation_delta_mean": float(np.mean(deltas)),
        "permutation_delta_95": float(np.percentile(deltas, 95)),
        "permutation_delta_99": float(np.percentile(deltas, 99)),
        "permutation_ratio_mean": float(np.mean(ratios)),
        "permutation_ratio_95": float(np.percentile(ratios, 95)),
        "observed_exceeds_95_percent_permutation_delta": bool(
            observed["delta_chi2_score"] > float(np.percentile(deltas, 95))
        ),
        "observed_exceeds_99_percent_permutation_delta": bool(
            observed["delta_chi2_score"] > float(np.percentile(deltas, 99))
        ),
    }

    return rows, summary


def run_selected_permutation_controls(mapped_rows, y, c_factor, design_fits, candidates):
    selected_designs = [
        "original_47",
        "plus_host_top10_row_order_measurement_controls",
    ]

    all_rows = []
    summaries = []

    for design_name in selected_designs:
        if design_name not in design_fits:
            continue

        for candidate in candidates:
            rows, summary = permutation_controls(
                mapped_rows,
                y,
                c_factor,
                design_fits[design_name],
                candidate,
                design_name,
            )

            all_rows.extend(rows)
            summaries.append(summary)

    write_csv(OUTDIR / "quantile_band_permutation_control_details_v1_5.csv", all_rows)
    write_json(OUTDIR / "quantile_band_permutation_control_summaries_v1_5.json", summaries)

    return all_rows, summaries


def summarize_band_profiles(audit_rows, permutation_summaries):
    perm_key = {
        (p["design"], p["candidate"]): p
        for p in permutation_summaries
    }

    profile_rows = []

    for row in audit_rows:
        if row.get("candidate_kind") != "f160w_quantile_band":
            continue

        perm = perm_key.get((row["design"], row["candidate"]), {})

        profile_rows.append(
            {
                "design": row["design"],
                "candidate": row["candidate"],
                "region": row["region"],
                "low_percentile": row["low_percentile"],
                "high_percentile": row["high_percentile"],
                "center_percentile": row["center_percentile"],
                "f160w_min": row["f160w_min"],
                "f160w_max": row["f160w_max"],
                "f160w_mean": row["f160w_mean"],
                "count_nonzero_raw": row["count_nonzero_raw"],
                "delta_chi2_score": row["delta_chi2_score"],
                "p_value_chi2_one_dof": row["p_value_chi2_one_dof"],
                "delta_aic_if_added_column": row["delta_aic_if_added_column"],
                "delta_bic_if_added_column": row["delta_bic_if_added_column"],
                "nondegenerate_ratio": row["nondegenerate_ratio"],
                "observed_exceeds_95_percent_permutation_delta": perm.get("observed_exceeds_95_percent_permutation_delta"),
                "observed_exceeds_99_percent_permutation_delta": perm.get("observed_exceeds_99_percent_permutation_delta"),
                "permutation_delta_99": perm.get("permutation_delta_99"),
            }
        )

    profile_rows = sorted(
        profile_rows,
        key=lambda r: (
            r["design"],
            r["center_percentile"],
        ),
    )

    write_csv(OUTDIR / "quantile_band_profile_summary_v1_5.csv", profile_rows)

    by_design = {}

    for design in sorted(set(r["design"] for r in profile_rows)):
        rows = [r for r in profile_rows if r["design"] == design]
        rows_sorted = sorted(rows, key=lambda r: r["center_percentile"])

        best = max(rows_sorted, key=lambda r: r["delta_chi2_score"]) if rows_sorted else None
        bright_tail = [r for r in rows_sorted if r["region"] == "bright_tail"]
        faint_tail = [r for r in rows_sorted if r["region"] == "faint_tail"]
        bright_middle = [r for r in rows_sorted if r["region"] == "bright_middle"]
        faint_middle = [r for r in rows_sorted if r["region"] == "faint_middle"]

        def group_stats(group):
            if not group:
                return {
                    "count": 0,
                    "mean_delta": None,
                    "max_delta": None,
                    "sum_delta": None,
                    "best": None,
                }

            deltas = np.asarray([r["delta_chi2_score"] for r in group], dtype=float)

            return {
                "count": len(group),
                "mean_delta": float(np.mean(deltas)),
                "max_delta": float(np.max(deltas)),
                "sum_delta": float(np.sum(deltas)),
                "best": max(group, key=lambda r: r["delta_chi2_score"]),
            }

        deltas_all = np.asarray([r["delta_chi2_score"] for r in rows_sorted], dtype=float)
        median_delta = float(np.median(deltas_all)) if len(deltas_all) else None

        by_design[design] = {
            "best_band": best,
            "median_delta": median_delta,
            "bright_tail": group_stats(bright_tail),
            "bright_middle": group_stats(bright_middle),
            "faint_middle": group_stats(faint_middle),
            "faint_tail": group_stats(faint_tail),
            "all_bands": rows_sorted,
        }

        if by_design[design]["faint_tail"]["sum_delta"] is not None and by_design[design]["bright_tail"]["sum_delta"] is not None:
            by_design[design]["faint_tail_minus_bright_tail_sum_delta"] = (
                by_design[design]["faint_tail"]["sum_delta"] - by_design[design]["bright_tail"]["sum_delta"]
            )
        else:
            by_design[design]["faint_tail_minus_bright_tail_sum_delta"] = None

    write_json(OUTDIR / "quantile_band_profile_by_design_v1_5.json", by_design)

    return profile_rows, by_design


def decide_status(profile_by_design):
    full_design = "plus_host_top10_row_order_measurement_controls"
    original_design = "original_47"

    full = profile_by_design.get(full_design, {})
    original = profile_by_design.get(original_design, {})

    best_full = full.get("best_band")
    best_original = original.get("best_band")

    faint_tail = full.get("faint_tail", {})
    bright_tail = full.get("bright_tail", {})
    faint_minus_bright = full.get("faint_tail_minus_bright_tail_sum_delta")
    median_delta = full.get("median_delta")

    best_cases = {
        "original_design": original,
        "full_control_design": full,
        "best_original_band": best_original,
        "best_full_control_band": best_full,
        "faint_tail_minus_bright_tail_sum_delta_full_controls": faint_minus_bright,
    }

    def strong_perm(row):
        return bool(
            row
            and row.get("observed_exceeds_99_percent_permutation_delta") is True
            and row.get("delta_chi2_score", 0.0) >= 25.0
        )

    def directional_perm(row):
        return bool(
            row
            and row.get("observed_exceeds_95_percent_permutation_delta") is True
            and row.get("delta_chi2_score", 0.0) >= 10.0
        )

    if best_full and best_full["region"] == "faint_tail" and strong_perm(best_full):
        if faint_minus_bright is not None and faint_minus_bright > 100.0:
            return (
                "f160w_quantile_pressure_localizes_to_faint_tail",
                8,
                "The strongest non-overlapping F160W residual band remains in the faint tail after full controls.",
                best_cases,
            )

    if best_full and best_full["region"] in ["faint_tail", "faint_middle"] and directional_perm(best_full):
        return (
            "f160w_quantile_pressure_broad_faint_side_directional",
            7,
            "F160W residual pressure remains stronger on the faint side after full controls, but localization is broad.",
            best_cases,
        )

    if best_original and best_full and best_original["delta_chi2_score"] > 25.0 and best_full["delta_chi2_score"] < 10.0:
        return (
            "f160w_quantile_pressure_collapses_under_full_controls",
            7,
            "F160W band pressure is strong before controls but collapses under host, row-order, and measurement controls.",
            best_cases,
        )

    if best_full and best_full["region"] in ["bright_tail", "bright_middle"]:
        return (
            "f160w_quantile_pressure_not_faint_specific",
            6,
            "The strongest controlled F160W band is not on the faint side, so the faint-boundary interpretation weakens.",
            best_cases,
        )

    return (
        "no_locked_f160w_quantile_localization",
        6,
        "The non-overlapping F160W bands do not lock a clean localization pattern after controls.",
        best_cases,
    )


def make_plots(mapped_rows, profile_rows):
    if not mapped_rows:
        return

    x = np.asarray([r["mapped_index"] for r in mapped_rows], dtype=float)
    residual = np.asarray([r["baseline_residual"] for r in mapped_rows], dtype=float)
    abs_residual = np.abs(residual)

    f160w = []

    for row in mapped_rows:
        try:
            f160w.append(float(row.get("table2_num_7", np.nan)))
        except Exception:
            f160w.append(np.nan)

    f160w = np.asarray(f160w, dtype=float)
    mask = np.isfinite(f160w)

    plt.figure(figsize=(11, 5))
    plt.plot(x, residual, linewidth=0.8)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Table2 mapped row index")
    plt.ylabel("compact baseline residual")
    plt.title("Table2 residuals by mapped row v1.5")
    plt.tight_layout()
    plt.savefig(OUTDIR / "table2_residual_by_row_v1_5.png", dpi=160)
    plt.close()

    if np.sum(mask) > 50:
        plt.figure(figsize=(8, 6))
        plt.scatter(f160w[mask], abs_residual[mask], s=8)
        plt.xlabel("F160W-like column table2_num_7")
        plt.ylabel("absolute compact residual")
        plt.title("F160W-like magnitude vs absolute residual v1.5")
        plt.tight_layout()
        plt.savefig(OUTDIR / "f160w_vs_abs_residual_v1_5.png", dpi=160)
        plt.close()

    for design in sorted(set(r["design"] for r in profile_rows)):
        rows = sorted(
            [r for r in profile_rows if r["design"] == design],
            key=lambda r: r["center_percentile"],
        )

        if not rows:
            continue

        labels = [f"{int(r['low_percentile'])}-{int(r['high_percentile'])}" for r in rows]
        values = [r["delta_chi2_score"] for r in rows]

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(rows)), values)
        plt.xticks(np.arange(len(rows)), labels, rotation=45, ha="right")
        plt.xlabel("F160W rank band; left = bright, right = faint")
        plt.ylabel("delta chi2 score")
        plt.title(f"Non-overlapping F160W band pressure: {design}")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"quantile_band_pressure_{safe_name(design)}_v1_5.png", dpi=160)
        plt.close()

    selected = "plus_host_top10_row_order_measurement_controls"
    rows = sorted(
        [r for r in profile_rows if r["design"] == selected],
        key=lambda r: r["center_percentile"],
    )

    if rows:
        labels = [f"{int(r['low_percentile'])}-{int(r['high_percentile'])}" for r in rows]

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(rows)), [r["delta_chi2_score"] for r in rows])
        plt.xticks(np.arange(len(rows)), labels, rotation=45, ha="right")
        plt.xlabel("F160W rank band; left = bright, right = faint")
        plt.ylabel("delta chi2 score")
        plt.title("F160W quantile-band localization after full controls")
        plt.tight_layout()
        plt.savefig(OUTDIR / "quantile_band_full_controls_v1_5.png", dpi=160)
        plt.close()


def code_context_search(aux_results):
    terms = [
        "table2",
        "host",
        "field",
        "ceph",
        "cepheid",
        "period",
        "metal",
        "F160",
        "m160",
        "color",
        "anchor",
        "calibrator",
        "muhat",
        "intercept",
        "fivelogH0",
        "H0",
        "alll",
        "ally",
        "allc",
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

                rows.append(
                    {
                        "repo_path": item["repo_path"],
                        "line_number": i,
                        "hit_terms": " | ".join(hits),
                        "line": line[:700],
                        "context": context[:1600],
                    }
                )

    write_csv(OUTDIR / "table2_quantile_context_hits_v1_5.csv", rows)

    return rows


def main():
    print("")
    print("TAIRID Table2 F160W quantile-band localization audit v1.5 starting.")
    print("Boundary: F160W band-localization audit only; not proof.")
    print("")

    downloads = {}
    aux_results = []
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
                "attempt_count": len(result.get("attempts", [])),
            }
        )

    for repo_path in AUX_FILES:
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
                "attempt_count": len(result.get("attempts", [])),
            }
        )

    write_csv(OUTDIR / "download_ledger_v1_5.csv", ledger)
    write_json(OUTDIR / "download_attempts_v1_5.json", {"compact": downloads, "auxiliary": aux_results})

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

    write_json(OUTDIR / "parse_meta_v1_5.json", parse_meta)
    write_json(OUTDIR / "parse_errors_v1_5.json", parse_errors)

    table2_result = next(
        (
            item for item in aux_results
            if item.get("repo_path") == "SH0ES_Data/table2.tex"
            and item.get("status") == "downloaded"
        ),
        None,
    )

    if parse_errors or not all(key in parsed for key in ["allc", "alll", "ally"]) or not table2_result:
        summary = {
            "test_name": "TAIRID Table2 F160W quantile-band localization audit v1.5",
            "boundary": "Download/parse failure. No quantile-band result.",
            "final_status": "table2_f160w_quantile_bands_v1_5_parse_or_download_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix or table2 retrieval before quantile-band audit.",
            "parse_errors": parse_errors,
            "table2_downloaded": bool(table2_result),
        }

        write_json(OUTDIR / "table2_f160w_quantile_bands_v1_5_summary.json", summary)
        print("Parse/download failed. See summary JSON.")
        return

    table2_all_rows, table2_data_rows, table2_header_rows = parse_table2_tex(Path(table2_result["local_path"]))

    C = np.asarray(parsed["allc"], dtype=float)
    L = np.asarray(parsed["alll"], dtype=float)
    y = np.asarray(parsed["ally"], dtype=float).reshape(-1)

    X, orientation = orient_design_matrix(L, len(y))

    if X is None:
        raise RuntimeError(f"Could not orient L: {orientation}")

    if C.ndim != 2 or C.shape[0] != len(y) or C.shape[1] != len(y):
        raise RuntimeError(f"C shape {C.shape} does not match y length {len(y)}")

    c_factor, C_sym, jitter, chol_attempts = stable_cholesky(C)

    baseline = gls_fit(y, X, c_factor)
    row_rows, cluster_rows = recover_compact_rows(X, y, C_sym, baseline)

    write_csv(OUTDIR / "compact_row_map_v1_5.csv", row_rows)
    write_csv(OUTDIR / "compact_cluster_map_v1_5.csv", cluster_rows)

    mapped_rows, map_status = map_table2_to_spine(row_rows, table2_data_rows)
    host_summary = summarize_hosts(mapped_rows)
    numeric_rows, likely_numeric_labels = numeric_feature_summary(mapped_rows)

    designs, control_metadata = build_designs(X, mapped_rows, host_summary, len(y))

    design_fits = {}

    for design_name, design in designs.items():
        design_fits[design_name] = gls_fit(y, design["D"], c_factor)

    design_fit_rows = []

    for design_name, fit in design_fits.items():
        design_fit_rows.append(
            {
                "design": design_name,
                "k": fit["k"],
                "dof": fit["dof"],
                "chi2": fit["chi2"],
                "reduced_chi2": fit["reduced_chi2"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "delta_chi2_vs_original": baseline["chi2"] - fit["chi2"],
                "delta_aic_vs_original": fit["aic"] - baseline["aic"],
                "delta_bic_vs_original": fit["bic"] - baseline["bic"],
            }
        )

    write_csv(OUTDIR / "design_fit_comparison_v1_5.csv", design_fit_rows)

    candidates = build_quantile_band_candidates(mapped_rows, len(y))
    audit_rows = run_audits(y, c_factor, design_fits, candidates)

    permutation_rows, permutation_summaries = run_selected_permutation_controls(
        mapped_rows,
        y,
        c_factor,
        design_fits,
        candidates,
    )

    profile_rows, profile_by_design = summarize_band_profiles(
        audit_rows,
        permutation_summaries,
    )

    final_status, readiness_score, next_wall, best_cases = decide_status(profile_by_design)

    make_plots(mapped_rows, profile_rows)

    residual = baseline["residual"]
    abs_residual = np.abs(residual)

    edge_counts = {
        "rows_total": int(X.shape[0]),
        "spine_38_41_43_rows": int(sum(1 for r in row_rows if r["contains_38_41_43_spine"])),
        "bridge_42_46_rows": int(sum(1 for r in row_rows if r["bridges_param42_param46"])),
        "touch_param42_rows": int(sum(1 for r in row_rows if r["touches_param42"])),
        "touch_param46_rows": int(sum(1 for r in row_rows if r["touches_param46_H0_like"])),
    }

    summary = {
        "test_name": "TAIRID Table2 F160W quantile-band localization audit v1.5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "F160W band-localization audit only. Not proof of TAIRID, not H0 resolution, "
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
            "parameter_count": int(X.shape[1]),
            "param38_value": float(baseline["beta"][38]),
            "param41_value": float(baseline["beta"][41]),
            "param43_value": float(baseline["beta"][43]),
            "param42_value": float(baseline["beta"][P42]),
            "param46_fivelogH0": float(baseline["beta"][P46]),
            "param46_H0_like": h0_like(baseline["beta"]),
            "normal_condition_estimate": float(np.linalg.cond(baseline["normal"])),
            "residual_mean": float(np.mean(residual)),
            "residual_std": float(np.std(residual)),
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "abs_residual_90": float(np.percentile(abs_residual, 90)),
            "abs_residual_95": float(np.percentile(abs_residual, 95)),
            "abs_residual_99": float(np.percentile(abs_residual, 99)),
        },
        "edge_counts": edge_counts,
        "table2_mapping": map_status,
        "table2_parse_counts": {
            "table2_all_rows": len(table2_all_rows),
            "table2_data_rows": len(table2_data_rows),
            "table2_header_rows": len(table2_header_rows),
            "mapped_rows": len(mapped_rows),
            "unique_host_guesses": len(host_summary),
        },
        "likely_numeric_labels": likely_numeric_labels,
        "design_fit_comparison": design_fit_rows,
        "top_numeric_correlations": numeric_rows[:40],
        "top_hosts_by_residual_pressure": host_summary[:30],
        "quantile_band_profile_by_design": profile_by_design,
        "quantile_band_profile_summary": profile_rows,
        "quantile_band_audit_rows": audit_rows,
        "permutation_control_summaries": permutation_summaries,
        "best_cases": best_cases,
        "code_context_hits_count": len(code_hits),
        "output_files": {
            "summary_json": str(OUTDIR / "table2_f160w_quantile_bands_v1_5_summary.json"),
            "summary_txt": str(OUTDIR / "table2_f160w_quantile_bands_v1_5_summary.txt"),
            "mapped_rows_csv": str(OUTDIR / "table2_compact_host_mapped_rows_v1_5.csv"),
            "host_summary_csv": str(OUTDIR / "table2_host_summary_v1_5.csv"),
            "numeric_summary_csv": str(OUTDIR / "table2_numeric_summary_v1_5.csv"),
            "design_fit_comparison_csv": str(OUTDIR / "design_fit_comparison_v1_5.csv"),
            "quantile_band_audit_csv": str(OUTDIR / "quantile_band_audit_all_designs_v1_5.csv"),
            "quantile_band_profile_csv": str(OUTDIR / "quantile_band_profile_summary_v1_5.csv"),
            "quantile_band_profile_json": str(OUTDIR / "quantile_band_profile_by_design_v1_5.json"),
            "permutation_summaries_json": str(OUTDIR / "quantile_band_permutation_control_summaries_v1_5.json"),
            "permutation_details_csv": str(OUTDIR / "quantile_band_permutation_control_details_v1_5.csv"),
            "control_design_metadata_json": str(OUTDIR / "control_design_metadata_v1_5.json"),
            "context_hits_csv": str(OUTDIR / "table2_quantile_context_hits_v1_5.csv"),
            "plots": [
                str(OUTDIR / "table2_residual_by_row_v1_5.png"),
                str(OUTDIR / "f160w_vs_abs_residual_v1_5.png"),
                str(OUTDIR / "quantile_band_full_controls_v1_5.png"),
            ],
        },
        "interpretation": {
            "what_supports_faint_tail_localization": (
                "The strongest non-overlapping band remains in the faint tail after full controls and beats permutation checks."
            ),
            "what_supports_broad_faint_regime": (
                "Several adjacent faint-side bands carry elevated pressure rather than one narrow 95-100% tail."
            ),
            "what_supports_stopping": (
                "Pressure shifts away from the faint side, becomes symmetric, or collapses under full controls."
            ),
            "truth_boundary": (
                "This test cannot prove TAIRID. It only localizes the SH0ES Table2 F160W residual pressure across non-overlapping bands."
            ),
        },
    }

    write_json(OUTDIR / "table2_f160w_quantile_bands_v1_5_summary.json", summary)

    with open(OUTDIR / "table2_f160w_quantile_bands_v1_5_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID Table2 F160W quantile-band localization audit v1.5\n\n")
        f.write("Boundary: F160W band-localization audit only. Not proof. Not H0 resolution.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Table2 mapping:\n")
        f.write(json.dumps(map_status, indent=2, default=json_default) + "\n\n")

        f.write("Edge counts:\n")
        f.write(json.dumps(edge_counts, indent=2, default=json_default) + "\n\n")

        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")

        f.write("Design fit comparison:\n")
        f.write(json.dumps(design_fit_rows, indent=2, default=json_default) + "\n\n")

        f.write("Quantile band profile by design:\n")
        f.write(json.dumps(profile_by_design, indent=2, default=json_default) + "\n\n")

        f.write("Quantile band profile summary rows:\n")
        f.write(json.dumps(profile_rows, indent=2, default=json_default) + "\n\n")

        f.write("Permutation summaries:\n")
        f.write(json.dumps(permutation_summaries, indent=2, default=json_default) + "\n\n")

        f.write("Best cases:\n")
        f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not prove H0 resolution.\n")
        f.write("- This only localizes the F160W residual pressure across non-overlapping bands.\n")

    print("")
    print("TAIRID Table2 F160W quantile-band localization audit v1.5 complete.")
    print("Created:")
    print("  tairid_table2_f160w_quantile_bands_v1_5_outputs/table2_f160w_quantile_bands_v1_5_summary.json")
    print("  tairid_table2_f160w_quantile_bands_v1_5_outputs/table2_f160w_quantile_bands_v1_5_summary.txt")
    print("  tairid_table2_f160w_quantile_bands_v1_5_outputs/quantile_band_profile_summary_v1_5.csv")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


OUTDIR = Path("tairid_table2_f160w_quantile_bands_v1_5_outputs")
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
    "SH0ES_Data/table2.tex",
    "SH0ES_Data/table2.README",
    "SH0ES_Data/README.md",
    "SH0ES_Data/MCMC_utils.py",
    "SH0ES_Data/lstsq_results.txt",
]

SPINE_COLS = {38, 41, 43}
P42 = 42
P46 = 46
EPS = 1.0e-12
SEED = 42
PERMUTATION_REPEATS = 60
BAND_EDGES = list(range(0, 105, 5))


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


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))[:180]


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
            "User-Agent": "TAIRID-Table2-F160W-quantile-bands-v1-5",
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
        for hdu_index, hdu in enumerate(hdul):
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
                        "hdu_index": hdu_index,
                        "hdu_name": hdu.name,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }

            except Exception:
                continue

    raise RuntimeError(f"No numeric FITS array found in {path}")


def orient_design_matrix(L, y_length):
    L = np.asarray(L, dtype=float)

    if L.ndim != 2:
        return None, {
            "status": "L_not_2d",
            "L_shape": list(L.shape),
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
            "status": "ambiguous_square_using_L",
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

    raise RuntimeError("Cholesky failed even after jitter attempts.")


def gls_fit(y, D, c_factor, rcond=1.0e-12):
    D = np.asarray(D, dtype=float)
    c_inv_y = cho_solve(c_factor, y, check_finite=False)
    c_inv_D = cho_solve(c_factor, D, check_finite=False)

    y_cinv_y = float(y.T @ c_inv_y)
    normal = D.T @ c_inv_D
    rhs = D.T @ c_inv_y

    normal_inv = np.linalg.pinv(normal, rcond=rcond)
    beta = normal_inv @ rhs

    chi2_value = float(y_cinv_y - 2.0 * beta.T @ rhs + beta.T @ normal @ beta)
    residual = y - D @ beta
    c_inv_residual = c_inv_y - c_inv_D @ beta

    n = len(y)
    k = D.shape[1]

    return {
        "D": D,
        "Cinv_y": c_inv_y,
        "Cinv_D": c_inv_D,
        "y_Cinv_y": y_cinv_y,
        "normal": normal,
        "normal_inv": normal_inv,
        "rhs": rhs,
        "beta": beta,
        "residual": residual,
        "Cinv_residual": c_inv_residual,
        "chi2": chi2_value,
        "dof": int(n - k),
        "k": int(k),
        "aic": float(chi2_value + 2 * k),
        "bic": float(chi2_value + k * math.log(n)),
        "reduced_chi2": float(chi2_value / max(n - k, 1)),
    }


def h0_like(beta):
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


def recover_compact_rows(X, y, C_sym, baseline_fit):
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

    residual = baseline_fit["residual"]
    cinv_residual = baseline_fit["Cinv_residual"]
    cov_diag = np.diag(C_sym)

    leverage = np.einsum(
        "ij,jk,ik->i",
        X,
        baseline_fit["normal_inv"],
        baseline_fit["Cinv_D"],
    )

    rows = []
    grouped = defaultdict(list)

    for i, (active, signs, full_key, active_key, sign_key) in enumerate(cache):
        active_set = set(map(int, active))
        family = classify_row(active, signs)
        contains_spine = SPINE_COLS.issubset(active_set)
        varying_cols = sorted(active_set - SPINE_COLS)

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
        abs_res = np.abs(res)
        first = group[0]

        clusters.append(
            {
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
                "median_residual": float(np.median(res)),
                "rms_residual": float(np.sqrt(np.mean(res * res))),
                "mean_abs_residual": float(np.mean(abs_res)),
                "max_abs_residual": float(np.max(abs_res)),
            }
        )

    clusters = sorted(
        clusters,
        key=lambda r: (-r["mean_abs_residual"], -r["size"], r["signature_cluster_id"]),
    )

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

    match = re.search(
        r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?",
        str(text).replace(",", ""),
    )

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

    for line_number, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()

        if "&" not in raw:
            continue

        if raw.startswith("%"):
            continue

        if any(token in raw for token in ["\\begin", "\\end", "\\hline", "\\table"]):
            continue

        cells = [clean_latex_cell(c) for c in raw.split("&")]
        cells = [c for c in cells if c != ""]

        if len(cells) < 2:
            continue

        numeric_count = sum(1 for c in cells if first_float(c) is not None)
        joined = " ".join(cells).lower()

        header_like = any(
            term in joined
            for term in ["host", "field", "period", "f160", "m160", "cepheid", "metal"]
        )

        row = {
            "table2_parse_index": len(all_rows),
            "source_line_number": line_number,
            "cell_count": len(cells),
            "numeric_count": numeric_count,
            "cells_json": json.dumps(cells),
            "row_text": raw[:1400],
        }

        all_rows.append(row)

        if numeric_count >= 3 and not (header_like and numeric_count < 4):
            data_rows.append(row)
        else:
            header_rows.append(row)

    write_csv(OUTDIR / "table2_all_rows_v1_5.csv", all_rows)
    write_csv(OUTDIR / "table2_data_rows_v1_5.csv", data_rows)
    write_csv(OUTDIR / "table2_header_rows_v1_5.csv", header_rows)

    return all_rows, data_rows, header_rows


def infer_host(cells):
    patterns = [
        r"\bN\d{3,5}[A-Za-z]?\b",
        r"\bM\d{1,3}\b",
        r"\bLMC\b",
        r"\bSMC\b",
        r"\bM31\b",
        r"\bN4258\b",
        r"\bU\w+\b",
    ]

    for cell in cells[:5]:
        text = str(cell).strip()

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)

            if match:
                return match.group(0).upper()

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

    mapped_count = min(len(spine_rows), len(table_rows))
    parsed_cells = []

    for i in range(mapped_count):
        parsed_cells.append(json.loads(table_rows[i]["cells_json"]))

    max_cells = max([len(cells) for cells in parsed_cells], default=0)
    mapped = []

    for i in range(mapped_count):
        compact = spine_rows[i]
        table = table_rows[i]
        cells = parsed_cells[i]
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

        for cell_index in range(max_cells):
            cell = cells[cell_index] if cell_index < len(cells) else ""
            number = first_float(cell)

            out[f"table2_cell_{cell_index}"] = cell
            out[f"table2_num_{cell_index}"] = number if number is not None else ""

        mapped.append(out)

    write_csv(OUTDIR / "table2_compact_host_mapped_rows_v1_5.csv", mapped)

    status = {
        "compact_spine_rows": len(spine_rows),
        "table2_data_rows": len(table2_data_rows),
        "mapped_rows": mapped_count,
        "exact_count_match": bool(len(spine_rows) == len(table2_data_rows)),
        "first_compact_spine_row": int(spine_rows[0]["observation_index"]) if spine_rows else None,
        "last_compact_spine_row": int(spine_rows[-1]["observation_index"]) if spine_rows else None,
    }

    return mapped, status


def summarize_hosts(mapped_rows):
    grouped = defaultdict(list)

    for row in mapped_rows:
        grouped[row["host_guess"]].append(row)

    host_rows = []

    for host, rows in grouped.items():
        residual = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        cluster_counts = Counter(r["compact_signature_cluster_id"] for r in rows)

        host_rows.append(
            {
                "host_guess": host,
                "row_count": len(rows),
                "mean_residual": float(np.mean(residual)),
                "median_residual": float(np.median(residual)),
                "rms_residual": float(np.sqrt(np.mean(residual * residual))),
                "mean_abs_residual": float(np.mean(np.abs(residual))),
                "max_abs_residual": float(np.max(np.abs(residual))),
                "first_mapped_index": int(rows[0]["mapped_index"]),
                "last_mapped_index": int(rows[-1]["mapped_index"]),
                "dominant_signature_clusters_json": json.dumps(dict(cluster_counts.most_common(8))),
            }
        )

    host_rows = sorted(
        host_rows,
        key=lambda r: (-r["mean_abs_residual"], -r["row_count"], r["host_guess"]),
    )

    write_csv(OUTDIR / "table2_host_summary_v1_5.csv", host_rows)

    return host_rows


def numeric_feature_summary(mapped_rows):
    if not mapped_rows:
        return [], {}

    numeric_columns = sorted([key for key in mapped_rows[0].keys() if key.startswith("table2_num_")])

    residual = np.asarray([r["baseline_residual"] for r in mapped_rows], dtype=float)
    abs_residual = np.abs(residual)

    rows = []

    for column in numeric_columns:
        values = []
        keep_residual = []
        keep_abs_residual = []

        for row, residual_value, abs_value in zip(mapped_rows, residual, abs_residual):
            value = row.get(column, "")

            try:
                numeric_value = float(value)
            except Exception:
                continue

            if np.isfinite(numeric_value):
                values.append(numeric_value)
                keep_residual.append(residual_value)
                keep_abs_residual.append(abs_value)

        if len(values) < 50:
            continue

        x = np.asarray(values, dtype=float)

        if np.std(x) <= 1.0e-14:
            continue

        y_signed = np.asarray(keep_residual, dtype=float)
        y_abs = np.asarray(keep_abs_residual, dtype=float)

        rows.append(
            {
                "numeric_column": column,
                "valid_count": len(x),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "corr_with_residual": float(np.corrcoef(x, y_signed)[0, 1]),
                "corr_with_abs_residual": float(np.corrcoef(x, y_abs)[0, 1]),
            }
        )

    rows = sorted(rows, key=lambda r: -abs(r["corr_with_abs_residual"]))

    likely = {
        "period": "table2_num_4",
        "color": "table2_num_5",
        "color_or_pre_f160w_sigma": "table2_num_6",
        "f160w": "table2_num_7",
        "f160w_sigma": "table2_num_8",
        "metallicity": "table2_num_9",
    }

    write_csv(OUTDIR / "table2_numeric_summary_v1_5.csv", rows)
    write_json(OUTDIR / "table2_likely_numeric_labels_v1_5.json", likely)

    return rows, likely


def standardize(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    std = float(np.std(values))

    if not np.isfinite(std) or std <= 1.0e-14:
        return np.zeros_like(values)

    return (values - float(np.mean(values))) / std


def nonzero_std(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return float(np.std(values)) > 1.0e-14


def vector_from_mapped_numeric(mapped_rows, y_length, column, fill_strategy="mean"):
    values = np.zeros(y_length, dtype=float)
    observed = []

    for row in mapped_rows:
        try:
            value = float(row.get(column, ""))
        except Exception:
            continue

        if np.isfinite(value):
            observed.append(value)

    if not observed:
        return values, {
            "column": column,
            "status": "no_observed_values",
            "observed_count": 0,
        }

    fill = float(np.mean(observed)) if fill_strategy == "mean" else 0.0

    for row in mapped_rows:
        idx = int(row["compact_observation_index"])

        try:
            value = float(row.get(column, fill))
        except Exception:
            value = fill

        values[idx] = value if np.isfinite(value) else fill

    meta = {
        "column": column,
        "status": "ok",
        "observed_count": len(observed),
        "fill": fill,
        "min": float(np.min(observed)),
        "max": float(np.max(observed)),
        "mean": float(np.mean(observed)),
        "std": float(np.std(observed)),
    }

    return values, meta


def host_control_matrix(mapped_rows, host_summary, y_length, top_n=10):
    controls = []
    names = []

    for host in host_summary[:top_n]:
        if host["row_count"] < 5:
            continue

        mask = np.zeros(y_length, dtype=float)

        for row in mapped_rows:
            if row["host_guess"] == host["host_guess"]:
                mask[int(row["compact_observation_index"])] = 1.0

        if nonzero_std(mask):
            controls.append(standardize(mask))
            names.append(f"host_control_{safe_name(host['host_guess'])}_n{host['row_count']}")

    if not controls:
        return np.empty((y_length, 0)), []

    return np.column_stack(controls), names


def controls_from_columns(mapped_rows, y_length, columns, label_prefix):
    controls = []
    names = []
    metadata = []

    for column in columns:
        values, meta = vector_from_mapped_numeric(mapped_rows, y_length, column)

        if nonzero_std(values):
            controls.append(standardize(values))
            names.append(f"{label_prefix}_{column}")
            metadata.append(meta)

    if not controls:
        return np.empty((y_length, 0)), [], metadata

    return np.column_stack(controls), names, metadata


def row_order_control(mapped_rows, y_length):
    values = np.zeros(y_length, dtype=float)

    for row in mapped_rows:
        values[int(row["compact_observation_index"])] = float(row["mapped_index"])

    return standardize(values).reshape(-1, 1), ["table2_row_order"]


def build_designs(X, mapped_rows, host_summary, y_length):
    designs = {}

    row_order, row_order_names = row_order_control(mapped_rows, y_length)
    host_controls, host_names = host_control_matrix(mapped_rows, host_summary, y_length, top_n=10)

    uncertainty_cols = ["table2_num_6", "table2_num_8"]
    uncertainty_controls, uncertainty_names, uncertainty_meta = controls_from_columns(
        mapped_rows,
        y_length,
        uncertainty_cols,
        "uncertainty",
    )

    measurement_cols = ["table2_num_4", "table2_num_5", "table2_num_6", "table2_num_8", "table2_num_9"]
    measurement_controls, measurement_names, measurement_meta = controls_from_columns(
        mapped_rows,
        y_length,
        measurement_cols,
        "measurement",
    )

    def make(name, blocks, block_names):
        valid_blocks = [block for block in blocks if block.shape[1] > 0]
        D = np.column_stack([X] + valid_blocks) if valid_blocks else X.copy()
        names = [f"original_param_{i}" for i in range(X.shape[1])]

        for current_names in block_names:
            names.extend(current_names)

        designs[name] = {
            "D": D,
            "names": names,
            "added_column_count": int(D.shape[1] - X.shape[1]),
        }

    make("original_47", [], [])
    make("plus_host_top10", [host_controls], [host_names])
    make("plus_uncertainty", [uncertainty_controls], [uncertainty_names])
    make("plus_measurement_controls", [measurement_controls], [measurement_names])
    make(
        "plus_host_top10_and_measurement_controls",
        [host_controls, measurement_controls],
        [host_names, measurement_names],
    )
    make(
        "plus_host_top10_row_order_measurement_controls",
        [host_controls, row_order, measurement_controls],
        [host_names, row_order_names, measurement_names],
    )

    metadata = {
        "host_control_names": host_names,
        "uncertainty_names": uncertainty_names,
        "uncertainty_metadata": uncertainty_meta,
        "measurement_names": measurement_names,
        "measurement_metadata": measurement_meta,
        "designs": {
            key: {
                "columns": value["D"].shape[1],
                "added_column_count": value["added_column_count"],
            }
            for key, value in designs.items()
        },
    }

    write_json(OUTDIR / "control_design_metadata_v1_5.json", metadata)

    return designs, metadata


def build_quantile_band_candidates(mapped_rows, y_length):
    f160w_values = []
    finite_rows = []

    for row in mapped_rows:
        try:
            value = float(row.get("table2_num_7", ""))
        except Exception:
            continue

        if np.isfinite(value):
            f160w_values.append(value)
            finite_rows.append(row)

    order = np.argsort(np.asarray(f160w_values, dtype=float))
    sorted_rows = [finite_rows[int(i)] for i in order]
    sorted_values = np.asarray([f160w_values[int(i)] for i in order], dtype=float)

    n = len(sorted_rows)
    candidates = []

    for low, high in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        start = int(math.floor(n * low / 100.0))
        end = int(math.floor(n * high / 100.0))

        if high == 100:
            end = n

        if end <= start:
            continue

        band_rows = sorted_rows[start:end]
        band_values = sorted_values[start:end]

        mask = np.zeros(y_length, dtype=float)

        for row in band_rows:
            mask[int(row["compact_observation_index"])] = 1.0

        center = 0.5 * (low + high)

        if high <= 25:
            region = "bright_tail"
        elif low >= 75:
            region = "faint_tail"
        elif center < 50:
            region = "bright_middle"
        else:
            region = "faint_middle"

        candidates.append(
            {
                "name": f"f160w_band_{low:02d}_{high:02d}",
                "values": mask,
                "kind": "f160w_quantile_band",
                "low_percentile": low,
                "high_percentile": high,
                "center_percentile": center,
                "region": region,
                "count": int(np.sum(mask > 0)),
                "f160w_min": float(np.min(band_values)),
                "f160w_max": float(np.max(band_values)),
                "f160w_mean": float(np.mean(band_values)),
            }
        )

    inventory = []

    for candidate in candidates:
        inventory.append(
            {
                "name": candidate["name"],
                "kind": candidate["kind"],
                "low_percentile": candidate["low_percentile"],
                "high_percentile": candidate["high_percentile"],
                "center_percentile": candidate["center_percentile"],
                "region": candidate["region"],
                "count": candidate["count"],
                "f160w_min": candidate["f160w_min"],
                "f160w_max": candidate["f160w_max"],
                "f160w_mean": candidate["f160w_mean"],
                "std": float(np.std(candidate["values"])),
            }
        )

    write_json(OUTDIR / "quantile_band_candidate_inventory_v1_5.json", inventory)

    return candidates


def audit_candidate_against_fit(candidate, y, c_factor, fit, design_name):
    raw = np.asarray(candidate["values"], dtype=float).reshape(-1)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    z = standardize(raw)

    c_inv_z = cho_solve(c_factor, z, check_finite=False)
    raw_norm2 = float(z.T @ c_inv_z)

    if raw_norm2 <= EPS:
        return {
            "design": design_name,
            "candidate": candidate["name"],
            "candidate_kind": candidate["kind"],
            "region": candidate["region"],
            "low_percentile": candidate["low_percentile"],
            "high_percentile": candidate["high_percentile"],
            "center_percentile": candidate["center_percentile"],
            "status": "zero_or_near_zero_raw_norm",
            "delta_chi2_score": 0.0,
            "p_value_chi2_one_dof": 1.0,
            "nondegenerate_ratio": 0.0,
            "delta_aic_if_added_column": 2.0,
            "delta_bic_if_added_column": float(math.log(len(y))),
            "count_nonzero_raw": int(np.sum(np.abs(raw) > 1.0e-12)),
        }

    x_t_cinv_z = fit["D"].T @ c_inv_z
    coeff = fit["normal_inv"] @ x_t_cinv_z

    z_perp = z - fit["D"] @ coeff
    c_inv_z_perp = c_inv_z - fit["Cinv_D"] @ coeff

    perp_norm2 = float(z_perp.T @ c_inv_z_perp)
    raw_norm = float(math.sqrt(max(raw_norm2, 0.0)))
    perp_norm = float(math.sqrt(max(perp_norm2, 0.0)))
    ratio = float(perp_norm / max(raw_norm, EPS))

    if perp_norm2 <= EPS:
        score = 0.0
        delta = 0.0
        alpha = 0.0
    else:
        score = float(z_perp.T @ fit["Cinv_residual"])
        delta = float((score * score) / perp_norm2)
        alpha = float(score / perp_norm2)

    p_value = float(chi2.sf(max(delta, 0.0), 1))

    return {
        "design": design_name,
        "candidate": candidate["name"],
        "candidate_kind": candidate["kind"],
        "region": candidate["region"],
        "low_percentile": candidate["low_percentile"],
        "high_percentile": candidate["high_percentile"],
        "center_percentile": candidate["center_percentile"],
        "f160w_min": candidate["f160w_min"],
        "f160w_max": candidate["f160w_max"],
        "f160w_mean": candidate["f160w_mean"],
        "status": "ok",
        "base_design_k": fit["k"],
        "base_chi2": fit["chi2"],
        "base_reduced_chi2": fit["reduced_chi2"],
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "count_nonzero_raw": int(np.sum(np.abs(raw) > 1.0e-12)),
        "raw_Cinv_norm": raw_norm,
        "residualized_Cinv_norm": perp_norm,
        "nondegenerate_ratio": ratio,
        "projection_absorption_fraction": float(1.0 - ratio),
        "score": score,
        "alpha_hat_added_column": alpha,
        "delta_chi2_score": delta,
        "p_value_chi2_one_dof": p_value,
        "delta_aic_if_added_column": float(2.0 - delta),
        "delta_bic_if_added_column": float(math.log(len(y)) - delta),
        "would_improve_aic": bool(2.0 - delta < 0.0),
        "would_improve_bic": bool(math.log(len(y)) - delta < 0.0),
    }


def run_audits(y, c_factor, design_fits, candidates):
    rows = []

    for design_name, fit in design_fits.items():
        for candidate in candidates:
            rows.append(
                audit_candidate_against_fit(candidate, y, c_factor, fit, design_name)
            )

    rows = sorted(
        rows,
        key=lambda r: (
            r["design"],
            float(r.get("center_percentile", -1.0)),
        ),
    )

    write_csv(OUTDIR / "quantile_band_audit_all_designs_v1_5.csv", rows)

    return rows


def permutation_controls(mapped_rows, y, c_factor, fit, candidate, design_name):
    rng = np.random.default_rng(SEED)
    rows = []

    raw = np.asarray(candidate["values"], dtype=float).reshape(-1)

    spine_indices = np.asarray(
        [int(row["compact_observation_index"]) for row in mapped_rows],
        dtype=int,
    )

    if len(spine_indices) == 0:
        return [], {}

    spine_values = raw[spine_indices].copy()

    for i in range(PERMUTATION_REPEATS):
        permuted = raw.copy()
        permuted_values = spine_values.copy()
        rng.shuffle(permuted_values)
        permuted[spine_indices] = permuted_values

        pseudo = {
            **candidate,
            "name": f"{candidate['name']}_permutation_{i}",
            "values": permuted,
            "kind": "permutation_control",
        }

        rows.append(audit_candidate_against_fit(pseudo, y, c_factor, fit, design_name))

    deltas = np.asarray([r["delta_chi2_score"] for r in rows], dtype=float)
    ratios = np.asarray([r["nondegenerate_ratio"] for r in rows], dtype=float)

    observed = audit_candidate_against_fit(candidate, y, c_factor, fit, design_name)

    summary = {
        "design": design_name,
        "candidate": candidate["name"],
        "region": candidate["region"],
        "low_percentile": candidate["low_percentile"],
        "high_percentile": candidate["high_percentile"],
        "center_percentile": candidate["center_percentile"],
        "repeats": PERMUTATION_REPEATS,
        "observed_delta_chi2": observed["delta_chi2_score"],
        "observed_nondegenerate_ratio": observed["nondegenerate_ratio"],
        "permutation_delta_mean": float(np.mean(deltas)),
        "permutation_delta_95": float(np.percentile(deltas, 95)),
        "permutation_delta_99": float(np.percentile(deltas, 99)),
        "permutation_ratio_mean": float(np.mean(ratios)),
        "permutation_ratio_95": float(np.percentile(ratios, 95)),
        "observed_exceeds_95_percent_permutation_delta": bool(
            observed["delta_chi2_score"] > float(np.percentile(deltas, 95))
        ),
        "observed_exceeds_99_percent_permutation_delta": bool(
            observed["delta_chi2_score"] > float(np.percentile(deltas, 99))
        ),
    }

    return rows, summary


def run_selected_permutation_controls(mapped_rows, y, c_factor, design_fits, candidates):
    selected_designs = [
        "original_47",
        "plus_host_top10_row_order_measurement_controls",
    ]

    all_rows = []
    summaries = []

    for design_name in selected_designs:
        if design_name not in design_fits:
            continue

        for candidate in candidates:
            rows, summary = permutation_controls(
                mapped_rows,
                y,
                c_factor,
                design_fits[design_name],
                candidate,
                design_name,
            )

            all_rows.extend(rows)
            summaries.append(summary)

    write_csv(OUTDIR / "quantile_band_permutation_control_details_v1_5.csv", all_rows)
    write_json(OUTDIR / "quantile_band_permutation_control_summaries_v1_5.json", summaries)

    return all_rows, summaries


def summarize_band_profiles(audit_rows, permutation_summaries):
    perm_key = {
        (p["design"], p["candidate"]): p
        for p in permutation_summaries
    }

    profile_rows = []

    for row in audit_rows:
        if row.get("candidate_kind") != "f160w_quantile_band":
            continue

        perm = perm_key.get((row["design"], row["candidate"]), {})

        profile_rows.append(
            {
                "design": row["design"],
                "candidate": row["candidate"],
                "region": row["region"],
                "low_percentile": row["low_percentile"],
                "high_percentile": row["high_percentile"],
                "center_percentile": row["center_percentile"],
                "f160w_min": row["f160w_min"],
                "f160w_max": row["f160w_max"],
                "f160w_mean": row["f160w_mean"],
                "count_nonzero_raw": row["count_nonzero_raw"],
                "delta_chi2_score": row["delta_chi2_score"],
                "p_value_chi2_one_dof": row["p_value_chi2_one_dof"],
                "delta_aic_if_added_column": row["delta_aic_if_added_column"],
                "delta_bic_if_added_column": row["delta_bic_if_added_column"],
                "nondegenerate_ratio": row["nondegenerate_ratio"],
                "observed_exceeds_95_percent_permutation_delta": perm.get("observed_exceeds_95_percent_permutation_delta"),
                "observed_exceeds_99_percent_permutation_delta": perm.get("observed_exceeds_99_percent_permutation_delta"),
                "permutation_delta_99": perm.get("permutation_delta_99"),
            }
        )

    profile_rows = sorted(
        profile_rows,
        key=lambda r: (
            r["design"],
            r["center_percentile"],
        ),
    )

    write_csv(OUTDIR / "quantile_band_profile_summary_v1_5.csv", profile_rows)

    by_design = {}

    for design in sorted(set(r["design"] for r in profile_rows)):
        rows = [r for r in profile_rows if r["design"] == design]
        rows_sorted = sorted(rows, key=lambda r: r["center_percentile"])

        best = max(rows_sorted, key=lambda r: r["delta_chi2_score"]) if rows_sorted else None
        bright_tail = [r for r in rows_sorted if r["region"] == "bright_tail"]
        faint_tail = [r for r in rows_sorted if r["region"] == "faint_tail"]
        bright_middle = [r for r in rows_sorted if r["region"] == "bright_middle"]
        faint_middle = [r for r in rows_sorted if r["region"] == "faint_middle"]

        def group_stats(group):
            if not group:
                return {
                    "count": 0,
                    "mean_delta": None,
                    "max_delta": None,
                    "sum_delta": None,
                    "best": None,
                }

            deltas = np.asarray([r["delta_chi2_score"] for r in group], dtype=float)

            return {
                "count": len(group),
                "mean_delta": float(np.mean(deltas)),
                "max_delta": float(np.max(deltas)),
                "sum_delta": float(np.sum(deltas)),
                "best": max(group, key=lambda r: r["delta_chi2_score"]),
            }

        deltas_all = np.asarray([r["delta_chi2_score"] for r in rows_sorted], dtype=float)
        median_delta = float(np.median(deltas_all)) if len(deltas_all) else None

        by_design[design] = {
            "best_band": best,
            "median_delta": median_delta,
            "bright_tail": group_stats(bright_tail),
            "bright_middle": group_stats(bright_middle),
            "faint_middle": group_stats(faint_middle),
            "faint_tail": group_stats(faint_tail),
            "all_bands": rows_sorted,
        }

        if by_design[design]["faint_tail"]["sum_delta"] is not None and by_design[design]["bright_tail"]["sum_delta"] is not None:
            by_design[design]["faint_tail_minus_bright_tail_sum_delta"] = (
                by_design[design]["faint_tail"]["sum_delta"] - by_design[design]["bright_tail"]["sum_delta"]
            )
        else:
            by_design[design]["faint_tail_minus_bright_tail_sum_delta"] = None

    write_json(OUTDIR / "quantile_band_profile_by_design_v1_5.json", by_design)

    return profile_rows, by_design


def decide_status(profile_by_design):
    full_design = "plus_host_top10_row_order_measurement_controls"
    original_design = "original_47"

    full = profile_by_design.get(full_design, {})
    original = profile_by_design.get(original_design, {})

    best_full = full.get("best_band")
    best_original = original.get("best_band")

    faint_tail = full.get("faint_tail", {})
    bright_tail = full.get("bright_tail", {})
    faint_minus_bright = full.get("faint_tail_minus_bright_tail_sum_delta")
    median_delta = full.get("median_delta")

    best_cases = {
        "original_design": original,
        "full_control_design": full,
        "best_original_band": best_original,
        "best_full_control_band": best_full,
        "faint_tail_minus_bright_tail_sum_delta_full_controls": faint_minus_bright,
    }

    def strong_perm(row):
        return bool(
            row
            and row.get("observed_exceeds_99_percent_permutation_delta") is True
            and row.get("delta_chi2_score", 0.0) >= 25.0
        )

    def directional_perm(row):
        return bool(
            row
            and row.get("observed_exceeds_95_percent_permutation_delta") is True
            and row.get("delta_chi2_score", 0.0) >= 10.0
        )

    if best_full and best_full["region"] == "faint_tail" and strong_perm(best_full):
        if faint_minus_bright is not None and faint_minus_bright > 100.0:
            return (
                "f160w_quantile_pressure_localizes_to_faint_tail",
                8,
                "The strongest non-overlapping F160W residual band remains in the faint tail after full controls.",
                best_cases,
            )

    if best_full and best_full["region"] in ["faint_tail", "faint_middle"] and directional_perm(best_full):
        return (
            "f160w_quantile_pressure_broad_faint_side_directional",
            7,
            "F160W residual pressure remains stronger on the faint side after full controls, but localization is broad.",
            best_cases,
        )

    if best_original and best_full and best_original["delta_chi2_score"] > 25.0 and best_full["delta_chi2_score"] < 10.0:
        return (
            "f160w_quantile_pressure_collapses_under_full_controls",
            7,
            "F160W band pressure is strong before controls but collapses under host, row-order, and measurement controls.",
            best_cases,
        )

    if best_full and best_full["region"] in ["bright_tail", "bright_middle"]:
        return (
            "f160w_quantile_pressure_not_faint_specific",
            6,
            "The strongest controlled F160W band is not on the faint side, so the faint-boundary interpretation weakens.",
            best_cases,
        )

    return (
        "no_locked_f160w_quantile_localization",
        6,
        "The non-overlapping F160W bands do not lock a clean localization pattern after controls.",
        best_cases,
    )


def make_plots(mapped_rows, profile_rows):
    if not mapped_rows:
        return

    x = np.asarray([r["mapped_index"] for r in mapped_rows], dtype=float)
    residual = np.asarray([r["baseline_residual"] for r in mapped_rows], dtype=float)
    abs_residual = np.abs(residual)

    f160w = []

    for row in mapped_rows:
        try:
            f160w.append(float(row.get("table2_num_7", np.nan)))
        except Exception:
            f160w.append(np.nan)

    f160w = np.asarray(f160w, dtype=float)
    mask = np.isfinite(f160w)

    plt.figure(figsize=(11, 5))
    plt.plot(x, residual, linewidth=0.8)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Table2 mapped row index")
    plt.ylabel("compact baseline residual")
    plt.title("Table2 residuals by mapped row v1.5")
    plt.tight_layout()
    plt.savefig(OUTDIR / "table2_residual_by_row_v1_5.png", dpi=160)
    plt.close()

    if np.sum(mask) > 50:
        plt.figure(figsize=(8, 6))
        plt.scatter(f160w[mask], abs_residual[mask], s=8)
        plt.xlabel("F160W-like column table2_num_7")
        plt.ylabel("absolute compact residual")
        plt.title("F160W-like magnitude vs absolute residual v1.5")
        plt.tight_layout()
        plt.savefig(OUTDIR / "f160w_vs_abs_residual_v1_5.png", dpi=160)
        plt.close()

    for design in sorted(set(r["design"] for r in profile_rows)):
        rows = sorted(
            [r for r in profile_rows if r["design"] == design],
            key=lambda r: r["center_percentile"],
        )

        if not rows:
            continue

        labels = [f"{int(r['low_percentile'])}-{int(r['high_percentile'])}" for r in rows]
        values = [r["delta_chi2_score"] for r in rows]

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(rows)), values)
        plt.xticks(np.arange(len(rows)), labels, rotation=45, ha="right")
        plt.xlabel("F160W rank band; left = bright, right = faint")
        plt.ylabel("delta chi2 score")
        plt.title(f"Non-overlapping F160W band pressure: {design}")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"quantile_band_pressure_{safe_name(design)}_v1_5.png", dpi=160)
        plt.close()

    selected = "plus_host_top10_row_order_measurement_controls"
    rows = sorted(
        [r for r in profile_rows if r["design"] == selected],
        key=lambda r: r["center_percentile"],
    )

    if rows:
        labels = [f"{int(r['low_percentile'])}-{int(r['high_percentile'])}" for r in rows]

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(rows)), [r["delta_chi2_score"] for r in rows])
        plt.xticks(np.arange(len(rows)), labels, rotation=45, ha="right")
        plt.xlabel("F160W rank band; left = bright, right = faint")
        plt.ylabel("delta chi2 score")
        plt.title("F160W quantile-band localization after full controls")
        plt.tight_layout()
        plt.savefig(OUTDIR / "quantile_band_full_controls_v1_5.png", dpi=160)
        plt.close()


def code_context_search(aux_results):
    terms = [
        "table2",
        "host",
        "field",
        "ceph",
        "cepheid",
        "period",
        "metal",
        "F160",
        "m160",
        "color",
        "anchor",
        "calibrator",
        "muhat",
        "intercept",
        "fivelogH0",
        "H0",
        "alll",
        "ally",
        "allc",
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

                rows.append(
                    {
                        "repo_path": item["repo_path"],
                        "line_number": i,
                        "hit_terms": " | ".join(hits),
                        "line": line[:700],
                        "context": context[:1600],
                    }
                )

    write_csv(OUTDIR / "table2_quantile_context_hits_v1_5.csv", rows)

    return rows


def main():
    print("")
    print("TAIRID Table2 F160W quantile-band localization audit v1.5 starting.")
    print("Boundary: F160W band-localization audit only; not proof.")
    print("")

    downloads = {}
    aux_results = []
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
                "attempt_count": len(result.get("attempts", [])),
            }
        )

    for repo_path in AUX_FILES:
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
                "attempt_count": len(result.get("attempts", [])),
            }
        )

    write_csv(OUTDIR / "download_ledger_v1_5.csv", ledger)
    write_json(OUTDIR / "download_attempts_v1_5.json", {"compact": downloads, "auxiliary": aux_results})

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

    write_json(OUTDIR / "parse_meta_v1_5.json", parse_meta)
    write_json(OUTDIR / "parse_errors_v1_5.json", parse_errors)

    table2_result = next(
        (
            item for item in aux_results
            if item.get("repo_path") == "SH0ES_Data/table2.tex"
            and item.get("status") == "downloaded"
        ),
        None,
    )

    if parse_errors or not all(key in parsed for key in ["allc", "alll", "ally"]) or not table2_result:
        summary = {
            "test_name": "TAIRID Table2 F160W quantile-band localization audit v1.5",
            "boundary": "Download/parse failure. No quantile-band result.",
            "final_status": "table2_f160w_quantile_bands_v1_5_parse_or_download_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix or table2 retrieval before quantile-band audit.",
            "parse_errors": parse_errors,
            "table2_downloaded": bool(table2_result),
        }

        write_json(OUTDIR / "table2_f160w_quantile_bands_v1_5_summary.json", summary)
        print("Parse/download failed. See summary JSON.")
        return

    table2_all_rows, table2_data_rows, table2_header_rows = parse_table2_tex(Path(table2_result["local_path"]))

    C = np.asarray(parsed["allc"], dtype=float)
    L = np.asarray(parsed["alll"], dtype=float)
    y = np.asarray(parsed["ally"], dtype=float).reshape(-1)

    X, orientation = orient_design_matrix(L, len(y))

    if X is None:
        raise RuntimeError(f"Could not orient L: {orientation}")

    if C.ndim != 2 or C.shape[0] != len(y) or C.shape[1] != len(y):
        raise RuntimeError(f"C shape {C.shape} does not match y length {len(y)}")

    c_factor, C_sym, jitter, chol_attempts = stable_cholesky(C)

    baseline = gls_fit(y, X, c_factor)
    row_rows, cluster_rows = recover_compact_rows(X, y, C_sym, baseline)

    write_csv(OUTDIR / "compact_row_map_v1_5.csv", row_rows)
    write_csv(OUTDIR / "compact_cluster_map_v1_5.csv", cluster_rows)

    mapped_rows, map_status = map_table2_to_spine(row_rows, table2_data_rows)
    host_summary = summarize_hosts(mapped_rows)
    numeric_rows, likely_numeric_labels = numeric_feature_summary(mapped_rows)

    designs, control_metadata = build_designs(X, mapped_rows, host_summary, len(y))

    design_fits = {}

    for design_name, design in designs.items():
        design_fits[design_name] = gls_fit(y, design["D"], c_factor)

    design_fit_rows = []

    for design_name, fit in design_fits.items():
        design_fit_rows.append(
            {
                "design": design_name,
                "k": fit["k"],
                "dof": fit["dof"],
                "chi2": fit["chi2"],
                "reduced_chi2": fit["reduced_chi2"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "delta_chi2_vs_original": baseline["chi2"] - fit["chi2"],
                "delta_aic_vs_original": fit["aic"] - baseline["aic"],
                "delta_bic_vs_original": fit["bic"] - baseline["bic"],
            }
        )

    write_csv(OUTDIR / "design_fit_comparison_v1_5.csv", design_fit_rows)

    candidates = build_quantile_band_candidates(mapped_rows, len(y))
    audit_rows = run_audits(y, c_factor, design_fits, candidates)

    permutation_rows, permutation_summaries = run_selected_permutation_controls(
        mapped_rows,
        y,
        c_factor,
        design_fits,
        candidates,
    )

    profile_rows, profile_by_design = summarize_band_profiles(
        audit_rows,
        permutation_summaries,
    )

    final_status, readiness_score, next_wall, best_cases = decide_status(profile_by_design)

    make_plots(mapped_rows, profile_rows)

    residual = baseline["residual"]
    abs_residual = np.abs(residual)

    edge_counts = {
        "rows_total": int(X.shape[0]),
        "spine_38_41_43_rows": int(sum(1 for r in row_rows if r["contains_38_41_43_spine"])),
        "bridge_42_46_rows": int(sum(1 for r in row_rows if r["bridges_param42_param46"])),
        "touch_param42_rows": int(sum(1 for r in row_rows if r["touches_param42"])),
        "touch_param46_rows": int(sum(1 for r in row_rows if r["touches_param46_H0_like"])),
    }

    summary = {
        "test_name": "TAIRID Table2 F160W quantile-band localization audit v1.5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "F160W band-localization audit only. Not proof of TAIRID, not H0 resolution, "
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
            "parameter_count": int(X.shape[1]),
            "param38_value": float(baseline["beta"][38]),
            "param41_value": float(baseline["beta"][41]),
            "param43_value": float(baseline["beta"][43]),
            "param42_value": float(baseline["beta"][P42]),
            "param46_fivelogH0": float(baseline["beta"][P46]),
            "param46_H0_like": h0_like(baseline["beta"]),
            "normal_condition_estimate": float(np.linalg.cond(baseline["normal"])),
            "residual_mean": float(np.mean(residual)),
            "residual_std": float(np.std(residual)),
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "abs_residual_90": float(np.percentile(abs_residual, 90)),
            "abs_residual_95": float(np.percentile(abs_residual, 95)),
            "abs_residual_99": float(np.percentile(abs_residual, 99)),
        },
        "edge_counts": edge_counts,
        "table2_mapping": map_status,
        "table2_parse_counts": {
            "table2_all_rows": len(table2_all_rows),
            "table2_data_rows": len(table2_data_rows),
            "table2_header_rows": len(table2_header_rows),
            "mapped_rows": len(mapped_rows),
            "unique_host_guesses": len(host_summary),
        },
        "likely_numeric_labels": likely_numeric_labels,
        "design_fit_comparison": design_fit_rows,
        "top_numeric_correlations": numeric_rows[:40],
        "top_hosts_by_residual_pressure": host_summary[:30],
        "quantile_band_profile_by_design": profile_by_design,
        "quantile_band_profile_summary": profile_rows,
        "quantile_band_audit_rows": audit_rows,
        "permutation_control_summaries": permutation_summaries,
        "best_cases": best_cases,
        "code_context_hits_count": len(code_hits),
        "output_files": {
            "summary_json": str(OUTDIR / "table2_f160w_quantile_bands_v1_5_summary.json"),
            "summary_txt": str(OUTDIR / "table2_f160w_quantile_bands_v1_5_summary.txt"),
            "mapped_rows_csv": str(OUTDIR / "table2_compact_host_mapped_rows_v1_5.csv"),
            "host_summary_csv": str(OUTDIR / "table2_host_summary_v1_5.csv"),
            "numeric_summary_csv": str(OUTDIR / "table2_numeric_summary_v1_5.csv"),
            "design_fit_comparison_csv": str(OUTDIR / "design_fit_comparison_v1_5.csv"),
            "quantile_band_audit_csv": str(OUTDIR / "quantile_band_audit_all_designs_v1_5.csv"),
            "quantile_band_profile_csv": str(OUTDIR / "quantile_band_profile_summary_v1_5.csv"),
            "quantile_band_profile_json": str(OUTDIR / "quantile_band_profile_by_design_v1_5.json"),
            "permutation_summaries_json": str(OUTDIR / "quantile_band_permutation_control_summaries_v1_5.json"),
            "permutation_details_csv": str(OUTDIR / "quantile_band_permutation_control_details_v1_5.csv"),
            "control_design_metadata_json": str(OUTDIR / "control_design_metadata_v1_5.json"),
            "context_hits_csv": str(OUTDIR / "table2_quantile_context_hits_v1_5.csv"),
            "plots": [
                str(OUTDIR / "table2_residual_by_row_v1_5.png"),
                str(OUTDIR / "f160w_vs_abs_residual_v1_5.png"),
                str(OUTDIR / "quantile_band_full_controls_v1_5.png"),
            ],
        },
        "interpretation": {
            "what_supports_faint_tail_localization": (
                "The strongest non-overlapping band remains in the faint tail after full controls and beats permutation checks."
            ),
            "what_supports_broad_faint_regime": (
                "Several adjacent faint-side bands carry elevated pressure rather than one narrow 95-100% tail."
            ),
            "what_supports_stopping": (
                "Pressure shifts away from the faint side, becomes symmetric, or collapses under full controls."
            ),
            "truth_boundary": (
                "This test cannot prove TAIRID. It only localizes the SH0ES Table2 F160W residual pressure across non-overlapping bands."
            ),
        },
    }

    write_json(OUTDIR / "table2_f160w_quantile_bands_v1_5_summary.json", summary)

    with open(OUTDIR / "table2_f160w_quantile_bands_v1_5_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID Table2 F160W quantile-band localization audit v1.5\n\n")
        f.write("Boundary: F160W band-localization audit only. Not proof. Not H0 resolution.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Table2 mapping:\n")
        f.write(json.dumps(map_status, indent=2, default=json_default) + "\n\n")

        f.write("Edge counts:\n")
        f.write(json.dumps(edge_counts, indent=2, default=json_default) + "\n\n")

        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")

        f.write("Design fit comparison:\n")
        f.write(json.dumps(design_fit_rows, indent=2, default=json_default) + "\n\n")

        f.write("Quantile band profile by design:\n")
        f.write(json.dumps(profile_by_design, indent=2, default=json_default) + "\n\n")

        f.write("Quantile band profile summary rows:\n")
        f.write(json.dumps(profile_rows, indent=2, default=json_default) + "\n\n")

        f.write("Permutation summaries:\n")
        f.write(json.dumps(permutation_summaries, indent=2, default=json_default) + "\n\n")

        f.write("Best cases:\n")
        f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not prove H0 resolution.\n")
        f.write("- This only localizes the F160W residual pressure across non-overlapping bands.\n")

    print("")
    print("TAIRID Table2 F160W quantile-band localization audit v1.5 complete.")
    print("Created:")
    print("  tairid_table2_f160w_quantile_bands_v1_5_outputs/table2_f160w_quantile_bands_v1_5_summary.json")
    print("  tairid_table2_f160w_quantile_bands_v1_5_outputs/table2_f160w_quantile_bands_v1_5_summary.txt")
    print("  tairid_table2_f160w_quantile_bands_v1_5_outputs/quantile_band_profile_summary_v1_5.csv")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
