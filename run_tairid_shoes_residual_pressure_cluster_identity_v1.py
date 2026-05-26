#!/usr/bin/env python3
"""
TAIRID SH0ES residual-pressure cluster identity recovery v1.

Purpose:
The SH0ES residual-pressure map found that leftover pressure after the
47-parameter compact ladder fit does not mainly live in the 42<->46 H0 bridge.

Instead, pressure appeared in sparse measurement / sparse ladder clusters,
especially repeated active-column forms involving columns:

    38, 41, 43

with a varying first column, such as:

    21,38,41,43
    4,38,41,43
    20,38,41,43
    35,38,41,43

This test does not fit a new model.

It asks:
- Which high-pressure signature clusters dominate the residual map?
- Are columns 38,41,43 acting like a repeated sparse-ladder spine?
- Which varying columns connect into that spine?
- Do high-pressure clusters align with auxiliary table sizes, row blocks, or table2/data files?
- Can we recover better identity labels before any new likelihood test?

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not a cosmology fit.
This is cluster identity recovery only.
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


OUTDIR = Path("tairid_shoes_residual_pressure_cluster_identity_v1_outputs")
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
    "SH0ES_Data/optical_wes_R22_for19fromR16.dat",
    "SH0ES_Data/optical_wes_R22_for19fromR16.wM31.dat",
    "SH0ES_Data/R22_orig19_NIR.out",
    "SH0ES_Data/R22_orig19_NIR.wm31.out",
]

TARGET_SPINE = {38, 41, 43}
P42 = 42
P46 = 46
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
            "User-Agent": "TAIRID-SH0ES-cluster-identity-v1",
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


def recover_rows_clusters_and_columns(X, y, C_sym, baseline):
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

        cluster_rows.append(
            {
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
                "spine_38_41_43_count": int(sum(1 for r in rows if r["contains_38_41_43_spine"])),
                "touches_param46_count": int(sum(1 for r in rows if r["touches_param46_H0_like"])),
                "bridges_42_46_count": int(sum(1 for r in rows if r["bridges_param42_param46"])),
            }
        )

    family_rows = sorted(family_rows, key=lambda r: (-r["mean_abs_residual"], -r["row_count"]))

    column_rows = []

    for j in range(X.shape[1]):
        col = X[:, j]
        active = np.abs(col) > 1.0e-12
        touched_rows = [row_rows[i] for i in range(len(row_rows)) if active[i]]

        if touched_rows:
            residuals = np.asarray([r["baseline_residual"] for r in touched_rows], dtype=float)
            families = Counter(r["equation_family"] for r in touched_rows)
        else:
            residuals = np.asarray([], dtype=float)
            families = Counter()

        column_rows.append(
            {
                "parameter_index": j,
                "beta": float(baseline["beta"][j]),
                "beta_err": float(baseline["beta_err"][j]),
                "nonzero_row_count": int(np.sum(active)),
                "nonzero_fraction": float(np.mean(active)),
                "mean_abs_residual_touched": float(np.mean(np.abs(residuals))) if residuals.size else 0.0,
                "rms_residual_touched": float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else 0.0,
                "touches_spine_clusters_count": int(sum(1 for r in touched_rows if r["contains_38_41_43_spine"])),
                "dominant_families_json": json.dumps(dict(families.most_common(6))),
                "is_target_spine_column_38_41_43": bool(j in TARGET_SPINE),
                "is_param42": bool(j == P42),
                "is_param46_H0_like": bool(j == P46),
            }
        )

    return row_rows, cluster_rows, family_rows, column_rows


def parse_latex_table_rows(text):
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if "&" not in stripped:
            continue
        if stripped.startswith("%"):
            continue
        if "\\hline" in stripped or "\\begin" in stripped or "\\end" in stripped:
            continue

        cells = [re.sub(r"\\\\.*$", "", c).strip() for c in stripped.split("&")]
        clean_cells = []
        for cell in cells:
            cell = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", cell)
            cell = re.sub(r"[{}$]", "", cell)
            clean_cells.append(cell.strip())

        if len(clean_cells) >= 2:
            rows.append(
                {
                    "line_number": line_number,
                    "cell_count": len(clean_cells),
                    "cells_json": json.dumps(clean_cells),
                    "row_text": stripped[:1200],
                }
            )
    return rows


def parse_auxiliary_tables(aux_results):
    inventory = []
    table_rows = []
    latex_rows = []
    readme_hits = []

    for item in aux_results:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["local_path"])
        lower = path.name.lower()

        if not lower.endswith((".py", ".txt", ".md", ".dat", ".out", ".tex", ".readme")):
            continue

        try:
            text = path.read_text(errors="replace")
        except Exception as exc:
            inventory.append(
                {
                    "repo_path": item["repo_path"],
                    "local_path": str(path),
                    "status": "read_failed",
                    "error": str(exc),
                }
            )
            continue

        lines = text.splitlines()
        preview_path = OUTDIR / f"preview_{safe_name(item['repo_path'])}.txt"
        preview_path.write_text("\n".join(lines[:220]), encoding="utf-8")

        nonempty_noncomment = [
            line for line in lines
            if line.strip() and not line.lstrip().startswith("#") and not line.lstrip().startswith("%")
        ]

        inventory.append(
            {
                "repo_path": item["repo_path"],
                "local_path": str(path),
                "status": "text_read",
                "line_count": len(lines),
                "nonempty_noncomment_lines": len(nonempty_noncomment),
                "preview_file": str(preview_path),
            }
        )

        if lower.endswith(".tex"):
            parsed = parse_latex_table_rows(text)
            for row in parsed:
                out = {"repo_path": item["repo_path"], **row}
                latex_rows.append(out)

        # Generic numeric/table parse attempts.
        if lower.endswith((".dat", ".out", ".csv", ".tsv", ".txt")):
            for sep in [r"\s+", ",", "\t"]:
                try:
                    df = pd.read_csv(path, sep=sep, engine="python", comment="#")
                    if df is not None and len(df) > 0 and len(df.columns) >= 2:
                        table_rows.append(
                            {
                                "repo_path": item["repo_path"],
                                "local_path": str(path),
                                "parse_sep": sep,
                                "rows": int(len(df)),
                                "columns": int(len(df.columns)),
                                "column_names": " | ".join(map(str, df.columns[:50])),
                                "host_like_columns": " | ".join(
                                    [str(c) for c in df.columns if "host" in str(c).lower() or "gal" in str(c).lower()][:20]
                                ),
                                "ceph_like_columns": " | ".join(
                                    [str(c) for c in df.columns if "ceph" in str(c).lower() or "period" in str(c).lower()][:20]
                                ),
                                "sn_like_columns": " | ".join(
                                    [str(c) for c in df.columns if "sn" in str(c).lower() or "super" in str(c).lower()][:20]
                                ),
                            }
                        )
                        break
                except Exception:
                    continue

        # README / code context hits for target terms.
        terms = [
            "table2", "host", "ceph", "cepheid", "anchor", "calibrator",
            "muhat", "intercept", "hubble", "sn", "supernova", "fivelogH0",
            "H0", "alll", "ally", "allc", "parameter", "theta",
            "38", "41", "43",
        ]
        for i, line in enumerate(lines, start=1):
            hits = [term for term in terms if term.lower() in line.lower()]
            if hits:
                lo = max(1, i - 2)
                hi = min(len(lines), i + 2)
                context = "\n".join(f"{j}: {lines[j - 1]}" for j in range(lo, hi + 1))
                readme_hits.append(
                    {
                        "repo_path": item["repo_path"],
                        "line_number": i,
                        "hit_terms": " | ".join(hits),
                        "line": line[:700],
                        "context": context[:1600],
                    }
                )

    write_csv(OUTDIR / "aux_text_inventory_v1.csv", inventory)
    write_csv(OUTDIR / "aux_parsed_table_inventory_v1.csv", table_rows)
    write_csv(OUTDIR / "table2_latex_rows_v1.csv", latex_rows)
    write_csv(OUTDIR / "aux_context_hits_v1.csv", readme_hits)

    return inventory, table_rows, latex_rows, readme_hits


def build_table_alignment_candidates(cluster_rows, aux_table_rows, latex_rows):
    alignments = []

    latex_count = len(latex_rows)
    aux_counts = []

    if latex_count > 0:
        aux_counts.append(
            {
                "source": "table2_latex_rows",
                "repo_path": "SH0ES_Data/table2.tex",
                "rows": latex_count,
                "columns": "",
                "column_names": "latex row parse",
            }
        )

    for t in aux_table_rows:
        aux_counts.append(
            {
                "source": "parsed_aux_table",
                "repo_path": t.get("repo_path"),
                "rows": int(t.get("rows", 0) or 0),
                "columns": t.get("columns"),
                "column_names": t.get("column_names"),
            }
        )

    for cluster in cluster_rows[:80]:
        size = int(cluster["size"])
        matches = []

        for t in aux_counts:
            count = int(t["rows"])

            if count <= 0:
                continue

            if count == size:
                match_type = "exact"
            elif abs(count - size) <= 2:
                match_type = "within_2"
            elif abs(count - size) <= 5:
                match_type = "within_5"
            elif abs(count - size) / max(count, 1) <= 0.05:
                match_type = "within_5_percent"
            else:
                continue

            matches.append(
                {
                    "source": t["source"],
                    "repo_path": t["repo_path"],
                    "rows": count,
                    "columns": t["columns"],
                    "column_names": t["column_names"],
                    "match_type": match_type,
                }
            )

        if matches or cluster["contains_38_41_43_spine"]:
            alignments.append(
                {
                    "signature_cluster_id": cluster["signature_cluster_id"],
                    "size": size,
                    "active_cols": cluster["active_cols"],
                    "contains_38_41_43_spine": cluster["contains_38_41_43_spine"],
                    "varying_cols_outside_38_41_43": cluster["varying_cols_outside_38_41_43"],
                    "mean_abs_residual": cluster["mean_abs_residual"],
                    "first_row": cluster["first_row"],
                    "last_row": cluster["last_row"],
                    "match_count": len(matches),
                    "matches_json": json.dumps(matches[:12]),
                }
            )

    write_csv(OUTDIR / "cluster_aux_table_alignment_candidates_v1.csv", alignments)
    return alignments


def build_spine_report(row_rows, cluster_rows, column_rows):
    spine_clusters = [
        c for c in cluster_rows
        if c["contains_38_41_43_spine"]
    ]

    varying_counter = Counter()
    varying_residuals = defaultdict(list)
    varying_sizes = Counter()

    for c in spine_clusters:
        cols = [
            int(v) for v in c["varying_cols_outside_38_41_43"].split(",")
            if v.strip().isdigit()
        ]
        for col in cols:
            varying_counter[col] += 1
            varying_sizes[col] += int(c["size"])
            varying_residuals[col].append(float(c["mean_abs_residual"]))

    varying_rows = []

    for col, count in varying_counter.most_common():
        beta = next((r["beta"] for r in column_rows if r["parameter_index"] == col), None)
        beta_err = next((r["beta_err"] for r in column_rows if r["parameter_index"] == col), None)

        varying_rows.append(
            {
                "varying_column": col,
                "spine_cluster_count": int(count),
                "total_rows_in_spine_clusters": int(varying_sizes[col]),
                "mean_cluster_abs_residual": float(np.mean(varying_residuals[col])),
                "max_cluster_abs_residual": float(np.max(varying_residuals[col])),
                "beta": beta,
                "beta_err": beta_err,
            }
        )

    spine_summary = {
        "spine_columns": sorted(TARGET_SPINE),
        "spine_cluster_count": len(spine_clusters),
        "spine_total_rows": int(sum(c["size"] for c in spine_clusters)),
        "spine_mean_abs_residual_mean": float(np.mean([c["mean_abs_residual"] for c in spine_clusters])) if spine_clusters else 0.0,
        "top_spine_clusters": spine_clusters[:30],
        "interpretation": (
            "Clusters containing columns 38,41,43 plus one or more varying columns "
            "are treated as a candidate sparse-ladder spine until human labels are recovered."
        ),
    }

    write_csv(OUTDIR / "spine_38_41_43_varying_columns_v1.csv", varying_rows)
    write_json(OUTDIR / "spine_38_41_43_summary_v1.json", spine_summary)

    return spine_summary, varying_rows


def make_plots(cluster_rows, family_rows, column_rows, spine_summary):
    top_clusters = cluster_rows[:35]
    labels = [str(c["signature_cluster_id"]) for c in top_clusters]
    vals = [c["mean_abs_residual"] for c in top_clusters]

    plt.figure(figsize=(13, 6))
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("mean absolute residual")
    plt.title("Top high-pressure signature clusters")
    plt.tight_layout()
    plt.savefig(OUTDIR / "top_high_pressure_clusters_v1.png", dpi=160)
    plt.close()

    fam_labels = [r["equation_family"] for r in family_rows]
    fam_vals = [r["mean_abs_residual"] for r in family_rows]

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(fam_labels)), fam_vals)
    plt.xticks(np.arange(len(fam_labels)), fam_labels, rotation=35, ha="right")
    plt.ylabel("mean absolute residual")
    plt.title("Mean residual pressure by equation family")
    plt.tight_layout()
    plt.savefig(OUTDIR / "family_residual_pressure_v1.png", dpi=160)
    plt.close()

    spine_clusters = [c for c in cluster_rows if c["contains_38_41_43_spine"]]
    if spine_clusters:
        labels = [str(c["signature_cluster_id"]) for c in spine_clusters[:40]]
        vals = [c["mean_abs_residual"] for c in spine_clusters[:40]]

        plt.figure(figsize=(13, 6))
        plt.bar(np.arange(len(labels)), vals)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("mean absolute residual")
        plt.title("38/41/43 spine clusters by residual pressure")
        plt.tight_layout()
        plt.savefig(OUTDIR / "spine_38_41_43_clusters_v1.png", dpi=160)
        plt.close()

    x = [r["parameter_index"] for r in column_rows]
    y = [r["mean_abs_residual_touched"] for r in column_rows]

    plt.figure(figsize=(12, 6))
    plt.bar(x, y)
    for col in [38, 41, 43, 42, 46]:
        plt.axvline(col, linewidth=1)
    plt.xlabel("parameter index")
    plt.ylabel("mean abs residual of touched rows")
    plt.title("Column-level residual pressure")
    plt.tight_layout()
    plt.savefig(OUTDIR / "column_residual_pressure_v1.png", dpi=160)
    plt.close()


def decide_status(spine_summary, alignments, readme_hits, cluster_rows):
    spine_count = int(spine_summary.get("spine_cluster_count", 0))
    spine_rows = int(spine_summary.get("spine_total_rows", 0))
    spine_top = spine_summary.get("top_spine_clusters", [])

    top10 = cluster_rows[:10]
    top10_spine = sum(1 for c in top10 if c.get("contains_38_41_43_spine"))

    alignment_hits = [a for a in alignments if int(a.get("match_count", 0)) > 0]
    useful_context_hits = [
        h for h in readme_hits
        if any(term in h.get("hit_terms", "") for term in ["host", "ceph", "cepheid", "calibrator", "table2", "parameter"])
    ]

    best_cases = {
        "spine_summary": spine_summary,
        "top_alignment_hits": alignment_hits[:20],
        "top_context_hits_count": len(useful_context_hits),
        "top10_high_pressure_spine_count": top10_spine,
    }

    if spine_count >= 10 and top10_spine >= 3 and useful_context_hits:
        return (
            "residual_pressure_spine_recovered_labels_partial",
            8,
            "The 38/41/43 sparse-ladder spine is recovered as the main high-pressure structure, but labels remain partial.",
            best_cases,
        )

    if spine_count >= 5 and top10_spine >= 2:
        return (
            "residual_pressure_spine_directional_not_labeled",
            7,
            "A 38/41/43 spine pattern appears in high-pressure clusters, but table/code identity is still incomplete.",
            best_cases,
        )

    if alignment_hits:
        return (
            "residual_pressure_table_alignment_partial",
            7,
            "Some cluster/table count alignments exist, but the 38/41/43 spine is not strong enough yet.",
            best_cases,
        )

    return (
        "residual_pressure_cluster_identity_still_unresolved",
        6,
        "High-pressure clusters were mapped, but identity labels remain unresolved.",
        best_cases,
    )


def main():
    print("")
    print("TAIRID SH0ES residual-pressure cluster identity recovery v1 starting.")
    print("Boundary: cluster identity recovery only; not proof.")
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

    write_csv(OUTDIR / "cluster_identity_download_ledger_v1.csv", ledger)
    write_json(
        OUTDIR / "cluster_identity_download_attempts_v1.json",
        {"compact": downloads, "auxiliary": aux_results},
    )

    aux_inventory, aux_table_rows, latex_rows, context_hits = parse_auxiliary_tables(aux_results)

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

    write_json(OUTDIR / "cluster_identity_parse_meta_v1.json", parse_meta)
    write_json(OUTDIR / "cluster_identity_parse_errors_v1.json", parse_errors)

    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]):
        summary = {
            "test_name": "TAIRID SH0ES residual-pressure cluster identity recovery v1",
            "boundary": "Matrix parse/download failure. No identity result.",
            "final_status": "cluster_identity_matrix_parse_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix parsing before cluster identity recovery.",
            "parse_errors": parse_errors,
        }
        write_json(OUTDIR / "cluster_identity_v1_summary.json", summary)
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

    row_rows, cluster_rows, family_rows, column_rows = recover_rows_clusters_and_columns(
        X,
        y,
        C_sym,
        baseline,
    )

    write_csv(OUTDIR / "cluster_identity_row_map_v1.csv", row_rows)
    write_csv(OUTDIR / "cluster_identity_signature_clusters_v1.csv", cluster_rows)
    write_csv(OUTDIR / "cluster_identity_family_summary_v1.csv", family_rows)
    write_csv(OUTDIR / "cluster_identity_column_summary_v1.csv", column_rows)

    alignments = build_table_alignment_candidates(cluster_rows, aux_table_rows, latex_rows)
    spine_summary, varying_rows = build_spine_report(row_rows, cluster_rows, column_rows)

    final_status, readiness_score, next_wall, best_cases = decide_status(
        spine_summary,
        alignments,
        context_hits,
        cluster_rows,
    )

    make_plots(cluster_rows, family_rows, column_rows, spine_summary)

    residual = baseline["residual"]
    abs_resid = np.abs(residual)

    edge_counts = {
        "rows_total": int(X.shape[0]),
        "bridge_42_46_rows": int(sum(1 for r in row_rows if r["bridges_param42_param46"])),
        "touch_param42_rows": int(sum(1 for r in row_rows if r["touches_param42"])),
        "touch_param46_rows": int(sum(1 for r in row_rows if r["touches_param46_H0_like"])),
        "spine_38_41_43_rows": int(sum(1 for r in row_rows if r["contains_38_41_43_spine"])),
    }

    summary = {
        "test_name": "TAIRID SH0ES residual-pressure cluster identity recovery v1",
        "boundary": (
            "Cluster identity recovery only. Not proof of TAIRID, not H0 resolution, "
            "not BAO, not Planck, and not a cosmology fit."
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
        "edge_counts": edge_counts,
        "spine_38_41_43_summary": spine_summary,
        "top_spine_varying_columns": varying_rows[:30],
        "equation_family_summary": family_rows,
        "top_signature_clusters_by_residual_pressure": cluster_rows[:40],
        "top_table_alignments": alignments[:40],
        "auxiliary_recovery_counts": {
            "aux_text_files_read": int(sum(1 for x in aux_inventory if x.get("status") == "text_read")),
            "parsed_aux_tables": len(aux_table_rows),
            "table2_latex_rows": len(latex_rows),
            "context_hits": len(context_hits),
            "alignment_candidates": len(alignments),
        },
        "best_cases": best_cases,
        "output_files": {
            "summary_json": str(OUTDIR / "cluster_identity_v1_summary.json"),
            "summary_txt": str(OUTDIR / "cluster_identity_v1_summary.txt"),
            "row_map_csv": str(OUTDIR / "cluster_identity_row_map_v1.csv"),
            "cluster_summary_csv": str(OUTDIR / "cluster_identity_signature_clusters_v1.csv"),
            "family_summary_csv": str(OUTDIR / "cluster_identity_family_summary_v1.csv"),
            "column_summary_csv": str(OUTDIR / "cluster_identity_column_summary_v1.csv"),
            "spine_summary_json": str(OUTDIR / "spine_38_41_43_summary_v1.json"),
            "spine_varying_columns_csv": str(OUTDIR / "spine_38_41_43_varying_columns_v1.csv"),
            "table_alignments_csv": str(OUTDIR / "cluster_aux_table_alignment_candidates_v1.csv"),
            "aux_context_hits_csv": str(OUTDIR / "aux_context_hits_v1.csv"),
            "table2_latex_rows_csv": str(OUTDIR / "table2_latex_rows_v1.csv"),
            "plots": [
                str(OUTDIR / "top_high_pressure_clusters_v1.png"),
                str(OUTDIR / "family_residual_pressure_v1.png"),
                str(OUTDIR / "spine_38_41_43_clusters_v1.png"),
                str(OUTDIR / "column_residual_pressure_v1.png"),
            ],
        },
        "interpretation": {
            "what_success_means": (
                "The high-pressure residual clusters show a repeated 38/41/43 sparse-ladder spine, "
                "giving a sharper identity target for the next label-recovery pass."
            ),
            "what_failure_means": (
                "The high-pressure clusters remain structurally mapped but not identifiable enough for a new likelihood test."
            ),
            "truth_boundary": (
                "This test does not fit TAIRID. It only tries to identify the row/column structure behind the residual pressure."
            ),
        },
    }

    write_json(OUTDIR / "cluster_identity_v1_summary.json", summary)

    with open(OUTDIR / "cluster_identity_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES residual-pressure cluster identity recovery v1\n\n")
        f.write("Boundary: cluster identity recovery only. Not proof. Not H0 resolution.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")

        f.write("Edge / spine counts:\n")
        f.write(json.dumps(edge_counts, indent=2, default=json_default) + "\n\n")

        f.write("38/41/43 spine summary:\n")
        f.write(json.dumps(spine_summary, indent=2, default=json_default) + "\n\n")

        f.write("Top spine varying columns:\n")
        f.write(json.dumps(varying_rows[:25], indent=2, default=json_default) + "\n\n")

        f.write("Top high-pressure clusters:\n")
        f.write(json.dumps(cluster_rows[:25], indent=2, default=json_default) + "\n\n")

        f.write("Top table alignments:\n")
        f.write(json.dumps(alignments[:25], indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not prove H0 resolution.\n")
        f.write("- This only identifies high-pressure cluster structure for possible later testing.\n")

    print("")
    print("TAIRID SH0ES residual-pressure cluster identity recovery v1 complete.")
    print("Created:")
    print("  tairid_shoes_residual_pressure_cluster_identity_v1_outputs/cluster_identity_v1_summary.json")
    print("  tairid_shoes_residual_pressure_cluster_identity_v1_outputs/cluster_identity_v1_summary.txt")
    print("  tairid_shoes_residual_pressure_cluster_identity_v1_outputs/cluster_identity_signature_clusters_v1.csv")
    print("  tairid_shoes_residual_pressure_cluster_identity_v1_outputs/spine_38_41_43_summary_v1.json")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is cluster identity recovery only.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
