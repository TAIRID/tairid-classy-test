#!/usr/bin/env python3
"""
TAIRID SH0ES Cepheid FITS target parser v2.

Purpose:
The previous SH0ES Cepheid/anchor parser found partial public-data readiness, but it did
not extract a clean source-level Cepheid table with host + period + magnitude + uncertainty.
This test is not a cosmology fit. It is a targeted parser for the compact SH0ES ladder files
identified by the handoff.

Primary target files:
- SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits
- SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits
- SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits

Support files to inspect when available:
- SH0ES_Data/MCMC_utils.py
- SH0ES_Data/run_mcmc.py
- SH0ES_Data/lstsq_results.txt
- SH0ES_Data/optical_wes_R22_for19fromR16.dat

What this test asks:
1. Do the SH0ES FITS files contain source-level Cepheid rows?
2. Do they contain enough fields for a period-luminosity proxy?
3. Do L, y, and C align as a compact linear ladder system?
4. Can the support scripts/text explain parameter names or ladder roles?
5. Is a first public Cepheid/anchor proxy likelihood possible after this parser?

Boundary:
This is not a SH0ES likelihood.
This is not a Cepheid fit.
This is not a cosmology fit.
This does not prove TAIRID cosmology.
It is a data-readiness and structure-inspection test for the next source-level wall.
"""

import csv
import hashlib
import json
import math
import os
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


OUTDIR = Path("shoes_ceph_fits_target_parser_v2_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_DIR = OUTDIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

REPO_OWNER = "PantheonPlusSH0ES"
REPO_NAME = "DataRelease"
BRANCH = "main"
SHOES_DIR = "SH0ES_Data"

GITHUB_API_SHOES_DIR = (
    f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{SHOES_DIR}?ref={BRANCH}"
)

RAW_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{SHOES_DIR}"
MEDIA_BASE = f"https://media.githubusercontent.com/media/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{SHOES_DIR}"
GITHUB_RAW_BASE = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/raw/{BRANCH}/{SHOES_DIR}"

TARGET_FITS = [
    "allc_shoes_ceph_topantheonwt6.0_112221.fits",
    "alll_shoes_ceph_topantheonwt6.0_112221.fits",
    "ally_shoes_ceph_topantheonwt6.0_112221.fits",
]

SUPPORT_FILES = [
    "MCMC_utils.py",
    "run_mcmc.py",
    "lstsq_results.txt",
    "optical_wes_R22_for19fromR16.dat",
]

ALL_TARGETS = TARGET_FITS + SUPPORT_FILES

CORE_FIELD_ALIASES = {
    "host": [
        "host", "galaxy", "gal", "sn_host", "hostname", "host_name", "target", "field"
    ],
    "cepheid_or_source_id": [
        "id", "cephid", "cepheid", "source", "star", "name", "objid", "object"
    ],
    "period_or_logp": [
        "period", "per", "logp", "log_p", "p", "puls", "cepper"
    ],
    "magnitude_or_photometry": [
        "mag", "m_", "meanmag", "phot", "f160w", "f555w", "f814w", "hmag",
        "vmag", "imag", "w", "wvi", "wfc3", "flux", "intensity"
    ],
    "uncertainty": [
        "err", "error", "sigma", "sig", "unc", "uncert", "std", "cov", "ivar",
        "weight", "e_"
    ],
    "anchor_or_distance": [
        "anchor", "dist", "distance", "mu", "ceph_dist", "maser", "parallax",
        "lmc", "n4258", "mw", "calib", "calibrator", "zp", "zeropoint"
    ],
    "metallicity": [
        "metal", "feh", "z", "oh", "8.69", "met"
    ],
}

PARAMETER_HINTS = [
    "period",
    "logp",
    "cepheid",
    "ceph",
    "host",
    "anchor",
    "distance",
    "metal",
    "zpt",
    "zeropoint",
    "slope",
    "m_h",
    "h0",
    "cov",
    "lstsq",
    "calib",
    "calibrator",
    "n4258",
    "lmc",
    "mw",
    "parallax",
]


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        if value.size > 30:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sample": json_safe(value.ravel()[:30]),
            }
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace").strip()
        except Exception:
            return repr(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_url(url, timeout=180):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-SH0ES-parser-v2",
            "Accept": "application/vnd.github.v3+json, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        final_url = response.geturl()
        status = getattr(response, "status", None)
        content_type = response.headers.get("Content-Type", "")
    return data, final_url, status, content_type


def is_git_lfs_pointer_bytes(data):
    head = data[:300].decode("utf-8", errors="ignore")
    return (
        "version https://git-lfs.github.com/spec/v1" in head
        and "oid sha256:" in head
    )


def parse_git_lfs_pointer(data):
    text = data.decode("utf-8", errors="replace")
    out = {"is_pointer": is_git_lfs_pointer_bytes(data), "raw_text": text}
    oid_match = re.search(r"oid\s+sha256:([0-9a-fA-F]+)", text)
    size_match = re.search(r"size\s+([0-9]+)", text)
    if oid_match:
        out["oid_sha256"] = oid_match.group(1)
    if size_match:
        out["size"] = int(size_match.group(1))
    return out


def github_api_list_shoes_dir():
    try:
        data, final_url, status, content_type = read_url(GITHUB_API_SHOES_DIR)
        listing = json.loads(data.decode("utf-8"))
        if not isinstance(listing, list):
            raise RuntimeError("GitHub API listing was not a list.")
        return listing
    except Exception as exc:
        return {
            "error": str(exc),
            "url": GITHUB_API_SHOES_DIR,
        }


def build_url_candidates(filename, api_listing_by_name):
    encoded = urllib.parse.quote(filename)
    candidates = []

    api_entry = api_listing_by_name.get(filename)
    if isinstance(api_entry, dict):
        download_url = api_entry.get("download_url")
        html_url = api_entry.get("html_url")
        if download_url:
            candidates.append(download_url)
        if html_url:
            candidates.append(html_url.replace("/blob/", "/raw/"))

    candidates.extend(
        [
            f"{MEDIA_BASE}/{encoded}",
            f"{GITHUB_RAW_BASE}/{encoded}",
            f"{RAW_BASE}/{encoded}",
        ]
    )

    deduped = []
    seen = set()
    for url in candidates:
        if url and url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def download_with_fallback(filename, api_listing_by_name):
    local_path = DOWNLOAD_DIR / filename
    candidates = build_url_candidates(filename, api_listing_by_name)
    attempts = []
    pointer_best = None

    for url in candidates:
        try:
            data, final_url, status, content_type = read_url(url)
            is_pointer = is_git_lfs_pointer_bytes(data)
            attempt = {
                "url": url,
                "final_url": final_url,
                "status": status,
                "content_type": content_type,
                "size_bytes": len(data),
                "git_lfs_pointer": is_pointer,
            }

            if is_pointer:
                attempt["pointer"] = parse_git_lfs_pointer(data)
                attempts.append(attempt)
                pointer_best = data
                continue

            local_path.write_bytes(data)
            attempt["sha256"] = sha256_file(local_path)
            attempts.append(attempt)

            return {
                "filename": filename,
                "status": "downloaded",
                "path": str(local_path),
                "size_bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
                "attempts": attempts,
            }

        except Exception as exc:
            attempts.append(
                {
                    "url": url,
                    "status": "error",
                    "error": str(exc),
                }
            )

    if pointer_best is not None:
        pointer_path = DOWNLOAD_DIR / f"{filename}.git_lfs_pointer.txt"
        pointer_path.write_bytes(pointer_best)
        return {
            "filename": filename,
            "status": "git_lfs_pointer_only",
            "path": str(pointer_path),
            "size_bytes": pointer_path.stat().st_size,
            "pointer": parse_git_lfs_pointer(pointer_best),
            "attempts": attempts,
        }

    return {
        "filename": filename,
        "status": "missing_or_download_failed",
        "path": None,
        "attempts": attempts,
    }


def norm_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_core_field_matches(column_names):
    matches = {key: [] for key in CORE_FIELD_ALIASES}
    normalized = {col: norm_name(col) for col in column_names}

    for col, col_norm in normalized.items():
        for field_type, aliases in CORE_FIELD_ALIASES.items():
            for alias in aliases:
                alias_norm = norm_name(alias)
                if alias_norm and alias_norm in col_norm:
                    matches[field_type].append(col)
                    break

    return matches


def core_score_from_matches(matches):
    required_groups = [
        "host",
        "cepheid_or_source_id",
        "period_or_logp",
        "magnitude_or_photometry",
        "uncertainty",
        "anchor_or_distance",
    ]
    present = {key: bool(matches.get(key)) for key in required_groups}
    score = sum(1 for key in required_groups if present[key])
    pass_condition = all(
        present[key]
        for key in [
            "host",
            "cepheid_or_source_id",
            "period_or_logp",
            "magnitude_or_photometry",
            "uncertainty",
        ]
    ) and present["anchor_or_distance"]

    if pass_condition:
        readiness = "pass_possible_source_level_ladder_proxy"
    elif score >= 4:
        readiness = "partial_candidate_needs_manual_mapping"
    elif score >= 2:
        readiness = "weak_candidate_probably_preprocessed"
    else:
        readiness = "no_source_level_core_fields_detected"

    return {
        "score_out_of_6": score,
        "present": present,
        "pass_condition": pass_condition,
        "readiness": readiness,
    }


def summarize_numeric_array(data):
    arr = np.asarray(data)

    out = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }

    if arr.size == 0:
        return out

    if arr.dtype.fields:
        out["structured_fields"] = list(arr.dtype.fields.keys())
        return out

    if np.issubdtype(arr.dtype, np.number):
        finite = arr[np.isfinite(arr)]
        out["finite_count"] = int(finite.size)
        if finite.size > 0:
            out.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                }
            )

        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            diag = np.diag(arr)
            finite_diag = diag[np.isfinite(diag)]
            out["square_matrix"] = True
            out["diag_finite_count"] = int(finite_diag.size)
            if finite_diag.size > 0:
                out["diag_min"] = float(np.min(finite_diag))
                out["diag_max"] = float(np.max(finite_diag))
                out["diag_mean"] = float(np.mean(finite_diag))
            try:
                denom = max(1.0e-30, float(np.max(np.abs(arr))))
                asym = float(np.max(np.abs(arr - arr.T)) / denom)
                out["max_relative_asymmetry"] = asym
            except Exception as exc:
                out["symmetry_check_error"] = str(exc)

    return out


def sample_table_rows(data, max_rows=5):
    samples = []
    try:
        n = min(max_rows, len(data))
        names = list(data.names or [])
        for i in range(n):
            row = {}
            for name in names:
                try:
                    row[name] = json_safe(data[name][i])
                except Exception as exc:
                    row[name] = f"<error: {exc}>"
            samples.append(row)
    except Exception as exc:
        samples.append({"sample_error": str(exc)})
    return samples


def inspect_fits_file(path, filename):
    file_summary = {
        "filename": filename,
        "path": str(path),
        "status": "inspect_started",
        "hdus": [],
        "table_candidates": [],
        "matrix_candidates": [],
        "core_field_candidates": [],
    }

    try:
        with fits.open(path, memmap=True) as hdul:
            file_summary["status"] = "fits_opened"
            file_summary["hdu_count"] = len(hdul)

            for idx, hdu in enumerate(hdul):
                hdu_name = getattr(hdu, "name", "")
                hdu_class = hdu.__class__.__name__
                data = getattr(hdu, "data", None)
                header = getattr(hdu, "header", None)

                hdu_info = {
                    "index": idx,
                    "name": hdu_name,
                    "class": hdu_class,
                    "has_data": data is not None,
                }

                if header is not None:
                    selected_header = {}
                    for key in [
                        "EXTNAME", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
                        "BITPIX", "XTENSION", "TFIELDS", "OBJECT", "COMMENT",
                    ]:
                        if key in header:
                            try:
                                selected_header[key] = json_safe(header[key])
                            except Exception:
                                selected_header[key] = str(header[key])
                    hdu_info["selected_header"] = selected_header

                if data is not None:
                    try:
                        hdu_info["data_summary"] = summarize_numeric_array(data)
                    except Exception as exc:
                        hdu_info["data_summary_error"] = str(exc)

                if hasattr(hdu, "columns") and hdu.columns is not None and data is not None:
                    columns = []
                    column_names = list(hdu.columns.names or [])

                    for col in hdu.columns:
                        columns.append(
                            {
                                "name": col.name,
                                "format": str(col.format),
                                "unit": str(col.unit) if col.unit else "",
                                "disp": str(col.disp) if col.disp else "",
                                "dim": str(col.dim) if col.dim else "",
                            }
                        )

                    matches = find_core_field_matches(column_names)
                    score = core_score_from_matches(matches)
                    row_count = len(data) if hasattr(data, "__len__") else None

                    table_info = {
                        "filename": filename,
                        "hdu_index": idx,
                        "hdu_name": hdu_name,
                        "row_count": int(row_count) if row_count is not None else None,
                        "column_count": len(column_names),
                        "columns": columns,
                        "core_matches": matches,
                        "core_score": score,
                    }

                    hdu_info["columns"] = columns
                    hdu_info["core_matches"] = matches
                    hdu_info["core_score"] = score

                    sample_rows = sample_table_rows(data, max_rows=5)
                    sample_path = SAMPLES_DIR / f"{filename}_hdu{idx}_sample_rows.json"
                    sample_path.write_text(json.dumps(sample_rows, indent=2))
                    table_info["sample_rows_file"] = str(sample_path)

                    file_summary["table_candidates"].append(table_info)
                    file_summary["core_field_candidates"].append(
                        {
                            "filename": filename,
                            "hdu_index": idx,
                            "hdu_name": hdu_name,
                            "score": score,
                            "matches": matches,
                        }
                    )

                elif data is not None:
                    arr = np.asarray(data)
                    if np.issubdtype(arr.dtype, np.number):
                        matrix_info = {
                            "filename": filename,
                            "hdu_index": idx,
                            "hdu_name": hdu_name,
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "summary": summarize_numeric_array(arr),
                        }

                        if arr.size > 0:
                            sample = arr.ravel()[:20]
                            matrix_info["flat_sample_first_20"] = json_safe(sample)

                        file_summary["matrix_candidates"].append(matrix_info)

                file_summary["hdus"].append(hdu_info)

    except Exception as exc:
        file_summary["status"] = "fits_open_failed"
        file_summary["error"] = str(exc)

    return file_summary


def inspect_text_file(path, filename):
    try:
        text = path.read_text(errors="replace")
    except Exception as exc:
        return {
            "filename": filename,
            "status": "text_read_failed",
            "error": str(exc),
        }

    lower = text.lower()
    lines = text.splitlines()
    hints = {}

    for hint in PARAMETER_HINTS:
        count = lower.count(hint.lower())
        if count:
            hints[hint] = count

    matched_lines = []
    for i, line in enumerate(lines, start=1):
        lowered = line.lower()
        if any(h in lowered for h in PARAMETER_HINTS):
            matched_lines.append(
                {
                    "line": i,
                    "text": line[:500],
                }
            )
        if len(matched_lines) >= 200:
            break

    excerpt = "\n".join(lines[:200])
    excerpt_path = SAMPLES_DIR / f"{filename}_first_200_lines.txt"
    excerpt_path.write_text(excerpt)

    hints_path = SAMPLES_DIR / f"{filename}_matched_hint_lines.json"
    hints_path.write_text(json.dumps(matched_lines, indent=2))

    return {
        "filename": filename,
        "status": "text_inspected",
        "line_count": len(lines),
        "size_bytes": path.stat().st_size,
        "hint_counts": hints,
        "first_200_lines_file": str(excerpt_path),
        "matched_hint_lines_file": str(hints_path),
    }


def classify_l_y_c(fits_summaries):
    matrix_entries = []

    for summary in fits_summaries:
        for matrix in summary.get("matrix_candidates", []):
            matrix_entries.append(matrix)

    by_file = {entry["filename"]: entry for entry in matrix_entries}

    c_entry = by_file.get("allc_shoes_ceph_topantheonwt6.0_112221.fits")
    l_entry = by_file.get("alll_shoes_ceph_topantheonwt6.0_112221.fits")
    y_entry = by_file.get("ally_shoes_ceph_topantheonwt6.0_112221.fits")

    out = {
        "found_allc_matrix": c_entry is not None,
        "found_alll_matrix": l_entry is not None,
        "found_ally_vector_or_matrix": y_entry is not None,
        "allc": c_entry,
        "alll": l_entry,
        "ally": y_entry,
        "alignment": {},
        "diagnostic": "not_enough_matrix_data",
    }

    y_size = None
    if y_entry is not None:
        y_shape = y_entry.get("shape", [])
        if len(y_shape) == 1:
            y_size = y_shape[0]
        elif len(y_shape) == 2 and 1 in y_shape:
            y_size = max(y_shape)
        else:
            y_size = int(np.prod(y_shape)) if y_shape else None
        out["alignment"]["y_size_inferred"] = y_size

    if l_entry is not None and y_size is not None:
        l_shape = l_entry.get("shape", [])
        out["alignment"]["L_shape"] = l_shape
        out["alignment"]["L_has_dimension_matching_y"] = y_size in l_shape
        if len(l_shape) == 2 and y_size in l_shape:
            other_dim = l_shape[0] if l_shape[1] == y_size else l_shape[1]
            out["alignment"]["parameter_count_or_equation_count_candidate"] = int(other_dim)

    if c_entry is not None and y_size is not None:
        c_shape = c_entry.get("shape", [])
        out["alignment"]["C_shape"] = c_shape
        out["alignment"]["C_matches_y_square"] = c_shape == [y_size, y_size]

    if c_entry is None:
        out["diagnostic"] = "C_missing_or_pointer_blocked_real_compact_chi_square"
    elif l_entry is None or y_entry is None:
        out["diagnostic"] = "L_or_y_missing_compact_system_incomplete"
    elif (
        out["alignment"].get("L_has_dimension_matching_y")
        and out["alignment"].get("C_matches_y_square")
    ):
        out["diagnostic"] = "compact_linear_system_shapes_align"
    else:
        out["diagnostic"] = "compact_linear_system_shapes_do_not_align"

    return out


def write_hdu_ledger(fits_summaries):
    path = OUTDIR / "shoes_ceph_fits_target_parser_v2_hdu_ledger.csv"
    rows = []

    for summary in fits_summaries:
        for hdu in summary.get("hdus", []):
            data_summary = hdu.get("data_summary", {})
            rows.append(
                {
                    "filename": summary.get("filename"),
                    "status": summary.get("status"),
                    "hdu_index": hdu.get("index"),
                    "hdu_name": hdu.get("name"),
                    "hdu_class": hdu.get("class"),
                    "has_data": hdu.get("has_data"),
                    "shape": json.dumps(data_summary.get("shape")),
                    "dtype": data_summary.get("dtype"),
                    "size": data_summary.get("size"),
                    "column_count": len(hdu.get("columns", [])),
                    "core_score": hdu.get("core_score", {}).get("score_out_of_6"),
                    "core_readiness": hdu.get("core_score", {}).get("readiness"),
                }
            )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "status", "hdu_index", "hdu_name", "hdu_class",
                "has_data", "shape", "dtype", "size", "column_count",
                "core_score", "core_readiness",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_column_candidate_ledger(fits_summaries):
    path = OUTDIR / "shoes_ceph_fits_target_parser_v2_column_candidates.csv"
    rows = []

    for summary in fits_summaries:
        for table in summary.get("table_candidates", []):
            matches = table.get("core_matches", {})
            for field_type, columns in matches.items():
                for col in columns:
                    rows.append(
                        {
                            "filename": table.get("filename"),
                            "hdu_index": table.get("hdu_index"),
                            "hdu_name": table.get("hdu_name"),
                            "field_type": field_type,
                            "matched_column": col,
                            "row_count": table.get("row_count"),
                            "column_count": table.get("column_count"),
                            "score_out_of_6": table.get("core_score", {}).get("score_out_of_6"),
                            "readiness": table.get("core_score", {}).get("readiness"),
                        }
                    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "hdu_index", "hdu_name", "field_type",
                "matched_column", "row_count", "column_count",
                "score_out_of_6", "readiness",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_download_ledger(downloads):
    path = OUTDIR / "shoes_ceph_fits_target_parser_v2_download_ledger.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "status", "path", "size_bytes", "sha256",
                "git_lfs_pointer", "pointer_size", "pointer_oid_sha256",
                "attempt_count",
            ],
        )
        writer.writeheader()

        for item in downloads:
            pointer = item.get("pointer", {})
            writer.writerow(
                {
                    "filename": item.get("filename"),
                    "status": item.get("status"),
                    "path": item.get("path"),
                    "size_bytes": item.get("size_bytes"),
                    "sha256": item.get("sha256"),
                    "git_lfs_pointer": bool(pointer),
                    "pointer_size": pointer.get("size"),
                    "pointer_oid_sha256": pointer.get("oid_sha256"),
                    "attempt_count": len(item.get("attempts", [])),
                }
            )

    return path


def plot_matrix_shapes(fits_summaries):
    matrix_entries = []
    for summary in fits_summaries:
        for matrix in summary.get("matrix_candidates", []):
            shape = matrix.get("shape", [])
            if shape:
                matrix_entries.append(
                    {
                        "label": matrix.get("filename", "")[:24],
                        "x": shape[-1] if len(shape) >= 1 else 0,
                        "y": shape[-2] if len(shape) >= 2 else 1,
                    }
                )

    if not matrix_entries:
        return None

    labels = [m["label"] for m in matrix_entries]
    x_vals = [m["x"] for m in matrix_entries]
    y_vals = [m["y"] for m in matrix_entries]
    pos = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(pos - 0.18, x_vals, width=0.36, label="last dimension")
    plt.bar(pos + 0.18, y_vals, width=0.36, label="second-last dimension")
    plt.xticks(pos, labels, rotation=25, ha="right")
    plt.ylabel("Dimension size")
    plt.title("SH0ES targeted FITS parser v2: matrix/vector dimensions")
    plt.legend(fontsize=8)
    plt.tight_layout()

    outpath = OUTDIR / "shoes_ceph_fits_target_parser_v2_matrix_shapes.png"
    plt.savefig(outpath, dpi=160)
    plt.close()
    return outpath


def plot_core_scores(fits_summaries):
    rows = []
    for summary in fits_summaries:
        for candidate in summary.get("core_field_candidates", []):
            rows.append(
                {
                    "label": f"{candidate.get('filename')} hdu{candidate.get('hdu_index')}",
                    "score": candidate.get("score", {}).get("score_out_of_6", 0),
                }
            )

    if not rows:
        return None

    rows = sorted(rows, key=lambda x: x["score"], reverse=True)[:25]

    labels = [r["label"][:45] for r in rows]
    scores = [r["score"] for r in rows]
    pos = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    plt.bar(pos, scores)
    plt.xticks(pos, labels, rotation=35, ha="right")
    plt.ylim(0, 6)
    plt.ylabel("Detected core field groups out of 6")
    plt.title("SH0ES targeted FITS parser v2: source-level Cepheid field score")
    plt.tight_layout()

    outpath = OUTDIR / "shoes_ceph_fits_target_parser_v2_core_field_scores.png"
    plt.savefig(outpath, dpi=160)
    plt.close()
    return outpath


def main():
    print("")
    print("TAIRID SH0ES Cepheid FITS target parser v2 starting.")
    print("Boundary: data-readiness parser only, not a SH0ES likelihood and not proof.")
    print("")

    api_listing = github_api_list_shoes_dir()

    if isinstance(api_listing, dict) and "error" in api_listing:
        api_listing_by_name = {}
        api_summary = api_listing
    else:
        api_listing_by_name = {item.get("name"): item for item in api_listing if isinstance(item, dict)}
        api_summary = {
            "status": "ok",
            "url": GITHUB_API_SHOES_DIR,
            "file_count": len(api_listing_by_name),
            "files": sorted(api_listing_by_name.keys()),
        }

    (OUTDIR / "shoes_ceph_fits_target_parser_v2_github_api_listing.json").write_text(
        json.dumps(api_summary, indent=2)
    )

    downloads = []
    for filename in ALL_TARGETS:
        print(f"Downloading / locating {filename} ...")
        result = download_with_fallback(filename, api_listing_by_name)
        downloads.append(result)
        print(f"  {result.get('status')}")

    write_download_ledger(downloads)

    fits_summaries = []
    text_summaries = []

    for item in downloads:
        filename = item.get("filename")
        status = item.get("status")
        path_str = item.get("path")

        if not path_str or status not in ["downloaded"]:
            continue

        path = Path(path_str)

        if filename.lower().endswith(".fits"):
            print(f"Inspecting FITS: {filename}")
            fits_summary = inspect_fits_file(path, filename)
            fits_summaries.append(fits_summary)
            (OUTDIR / f"{filename}_inspection.json").write_text(json.dumps(fits_summary, indent=2))
        else:
            print(f"Inspecting text/support: {filename}")
            text_summary = inspect_text_file(path, filename)
            text_summaries.append(text_summary)
            (OUTDIR / f"{filename}_inspection.json").write_text(json.dumps(text_summary, indent=2))

    hdu_ledger = write_hdu_ledger(fits_summaries)
    column_ledger = write_column_candidate_ledger(fits_summaries)

    matrix_alignment = classify_l_y_c(fits_summaries)
    matrix_alignment_path = OUTDIR / "shoes_ceph_fits_target_parser_v2_matrix_alignment.json"
    matrix_alignment_path.write_text(json.dumps(matrix_alignment, indent=2))

    all_core_candidates = []
    for summary in fits_summaries:
        all_core_candidates.extend(summary.get("core_field_candidates", []))

    all_core_candidates_sorted = sorted(
        all_core_candidates,
        key=lambda c: c.get("score", {}).get("score_out_of_6", 0),
        reverse=True,
    )

    best_core_candidate = all_core_candidates_sorted[0] if all_core_candidates_sorted else None

    if best_core_candidate:
        best_score = best_core_candidate.get("score", {})
        best_readiness = best_score.get("readiness")
        best_pass = bool(best_score.get("pass_condition"))
    else:
        best_readiness = "no_table_candidates_found"
        best_pass = False

    allc_download = next((d for d in downloads if d.get("filename") == TARGET_FITS[0]), None)
    allc_status = allc_download.get("status") if allc_download else "not_attempted"

    if best_pass:
        final_status = "source_level_proxy_possible"
        readiness_score = 9
    elif matrix_alignment.get("diagnostic") == "compact_linear_system_shapes_align":
        final_status = "compact_matrix_ladder_possible_source_fields_not_yet_mapped"
        readiness_score = 8
    elif allc_status == "git_lfs_pointer_only":
        final_status = "partial_parser_blocked_by_allc_lfs_payload"
        readiness_score = 7
    elif best_core_candidate and best_core_candidate.get("score", {}).get("score_out_of_6", 0) >= 4:
        final_status = "partial_parser_needed_manual_field_mapping"
        readiness_score = 7
    elif fits_summaries:
        final_status = "fits_opened_but_source_level_fields_not_detected"
        readiness_score = 5
    else:
        final_status = "download_or_fits_open_failed"
        readiness_score = 3

    plot_paths = []
    for plotter in [plot_matrix_shapes, plot_core_scores]:
        try:
            plot_path = plotter(fits_summaries)
            if plot_path:
                plot_paths.append(str(plot_path))
        except Exception as exc:
            plot_paths.append(f"plot_error: {exc}")

    summary = {
        "test_name": "TAIRID SH0ES Cepheid FITS target parser v2",
        "boundary": "Data-readiness parser only. Not a Cepheid likelihood, not a cosmology fit, not proof.",
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "primary_targets": TARGET_FITS,
        "support_files": SUPPORT_FILES,
        "github_api_listing": api_summary,
        "downloads": downloads,
        "fits_file_count_opened": len(fits_summaries),
        "support_file_count_inspected": len(text_summaries),
        "matrix_alignment": matrix_alignment,
        "best_core_candidate": best_core_candidate,
        "top_core_candidates": all_core_candidates_sorted[:20],
        "text_support_summaries": text_summaries,
        "output_files": {
            "download_ledger_csv": str(OUTDIR / "shoes_ceph_fits_target_parser_v2_download_ledger.csv"),
            "hdu_ledger_csv": str(hdu_ledger),
            "column_candidates_csv": str(column_ledger),
            "matrix_alignment_json": str(matrix_alignment_path),
            "plots": plot_paths,
        },
        "interpretation": {
            "if_source_level_proxy_possible": (
                "A first public Cepheid/anchor proxy likelihood can be attempted next."
            ),
            "if_compact_matrix_ladder_possible_source_fields_not_yet_mapped": (
                "L/y/C can likely support a compact ladder chi-square, but the next script must map parameter/equation meanings."
            ),
            "if_partial_parser_blocked_by_allc_lfs_payload": (
                "The covariance C is still blocked by Git LFS payload retrieval. Retrieve the real allc FITS object before compact chi-square."
            ),
            "if_fits_opened_but_source_level_fields_not_detected": (
                "The public compact files may be preprocessed rather than source-level Cepheid rows. Inspect support scripts and parameter names before fitting."
            ),
        },
    }

    summary_path = OUTDIR / "shoes_ceph_fits_target_parser_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    with open(OUTDIR / "shoes_ceph_fits_target_parser_v2_summary.txt", "w") as f:
        f.write("TAIRID SH0ES Cepheid FITS target parser v2\n\n")
        f.write("Boundary: data-readiness parser only. Not a SH0ES likelihood and not proof.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n\n")
        f.write("Primary target FITS files:\n")
        for name in TARGET_FITS:
            f.write(f"- {name}\n")
        f.write("\nSupport files:\n")
        for name in SUPPORT_FILES:
            f.write(f"- {name}\n")
        f.write("\nDownload statuses:\n")
        for item in downloads:
            f.write(f"- {item.get('filename')}: {item.get('status')}")
            pointer = item.get("pointer")
            if pointer:
                f.write(
                    f" | Git LFS pointer size={pointer.get('size')} oid={pointer.get('oid_sha256')}"
                )
            f.write("\n")
        f.write("\nMatrix alignment diagnostic:\n")
        f.write(json.dumps(matrix_alignment, indent=2))
        f.write("\n\nBest source-level core candidate:\n")
        f.write(json.dumps(best_core_candidate, indent=2))
        f.write("\n\nInterpretation guide:\n")
        f.write("- If allc/L/y align, the next wall is compact ladder chi-square mapping.\n")
        f.write("- If allc is only a Git LFS pointer, retrieve the real allc FITS payload before chi-square.\n")
        f.write("- If source-level Cepheid columns appear, build a first period-luminosity proxy.\n")
        f.write("- If only preprocessed arrays appear, inspect MCMC_utils.py and run_mcmc.py to map the equation system.\n")
        f.write("- Do not claim TAIRID solved Hubble tension from this parser.\n")

    print("")
    print("TAIRID SH0ES Cepheid FITS target parser v2 complete.")
    print("Created:")
    print("  shoes_ceph_fits_target_parser_v2_outputs/shoes_ceph_fits_target_parser_v2_summary.json")
    print("  shoes_ceph_fits_target_parser_v2_outputs/shoes_ceph_fits_target_parser_v2_summary.txt")
    print("  shoes_ceph_fits_target_parser_v2_outputs/shoes_ceph_fits_target_parser_v2_download_ledger.csv")
    print("  shoes_ceph_fits_target_parser_v2_outputs/shoes_ceph_fits_target_parser_v2_hdu_ledger.csv")
    print("  shoes_ceph_fits_target_parser_v2_outputs/shoes_ceph_fits_target_parser_v2_column_candidates.csv")
    print("  shoes_ceph_fits_target_parser_v2_outputs/shoes_ceph_fits_target_parser_v2_matrix_alignment.json")
    print("")
    print("Boundary:")
    print("  This is not the Cepheid/SH0ES ladder likelihood.")
    print("  This is not a cosmology fit.")
    print("  This is a targeted data-readiness parser.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
