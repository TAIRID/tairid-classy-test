#!/usr/bin/env python3
"""
TAIRID NIR FITS Alignment Probe v1.4.2
New-file-name version to avoid GitHub workflow/file collision with prior v1.4 runs.

Purpose:
v1.2 proved the SH0ES R22 NIR Cepheid .out files can be parsed as H/F160W-like
surface tables.
v1.3 located likely high-level SH0ES ladder candidates.
v1.4 failed because at least one candidate path did not open as a valid FITS
payload and the script crashed.

v1.4.2 does the same audit with new filenames and runtime-safe handling:
    1. download NIR surface files,
    2. parse row counts and host inventory,
    3. download the C/L/y FITS candidate paths through media/raw URLs,
    4. classify payloads before opening,
    5. inspect FITS HDUs only when safe,
    6. decide whether any lawful row map to the NIR surface is exposed.

This test does NOT validate TAIRID.
This test does NOT tune the frozen v1.0 rule.
This test does NOT create residuals from magnitudes.
This test does NOT replay the frozen edge rule.
This test does NOT claim H0 correction or new physics.

Truth boundary:
A path ending in .fits is not a row map. A readable y/L/C ladder system is not a
row-labeled NIR residual surface. Replay remains blocked unless explicit labels,
mapping metadata, or a lawful reconstructable row map is found.
"""

import csv
import gzip
import io
import json
import math
import re
import traceback
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


OUTDIR = Path("tairid_nir_fits_alignment_probe_v1_4_2_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO = "PantheonPlusSH0ES/DataRelease"
BRANCH_CANDIDATES = ["main", "master"]

NIR_SURFACE_FILES = [
    {"label": "nir_orig19", "repo_path": "SH0ES_Data/R22_orig19_NIR.out"},
    {"label": "nir_orig19_wm31", "repo_path": "SH0ES_Data/R22_orig19_NIR.wm31.out"},
]

FITS_CANDIDATES = [
    {
        "label": "C_covariance_matrix",
        "role": "C",
        "repo_path": "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits",
        "expected_meaning": "C covariance matrix for high-level SH0ES/Pantheon ladder system",
    },
    {
        "label": "L_equation_matrix",
        "role": "L",
        "repo_path": "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits",
        "expected_meaning": "L equation matrix for high-level SH0ES/Pantheon ladder system",
    },
    {
        "label": "y_data_vector",
        "role": "y",
        "repo_path": "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits",
        "expected_meaning": "y data vector for high-level SH0ES/Pantheon ladder system",
    },
]

REFERENCE_CONTEXT_FILES = [
    "SH0ES_Data/README.md",
    "SH0ES_Data/table2.README",
    "SH0ES_Data/table2.tex",
]

FROZEN_RULE_CARRIED_FORWARD = {
    "source_status": "locked by v1.0, schema-checked by v1.2, locator-scanned by v1.3",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "frozen_variable": "F160W-like Table2 numeric column, table2_num_7",
    "external_proxy_candidate": "H column in R22 NIR files, treated only as H/F160W-like surface candidate",
    "edge_rule": "within-host high 5% H/F160W-like magnitude minus within-host low 5% H/F160W-like magnitude",
    "low_alpha_reference_hosts": ["LMC", "SMC", "N4536"],
    "sign_break_quarantine_hosts": ["M31"],
    "hard_boundary": [
        "Do not tune host regimes in v1.4.2.",
        "Do not invent residuals.",
        "Do not claim validation from FITS dimensionality.",
        "Do not treat y/L/C as NIR row-aligned unless explicit labels, mapping metadata, or a lawful row map is found.",
        "Do not replay the frozen edge rule in v1.4.2.",
        "Do not claim H0 correction or new physics.",
    ],
}

CLAIMS_V1_4_2 = {
    "battery_name": "TAIRID NIR FITS Alignment Probe v1.4.2",
    "scope": "New-file-name runtime-safe FITS schema and NIR row-alignment proof attempt",
    "primary_question": (
        "Can the SH0ES high-level y/L/C FITS candidates be downloaded as valid FITS files and row-aligned "
        "to the v1.2 NIR Cepheid H-band surface without inventing residuals or row labels?"
    ),
    "truth_boundary": (
        "This is schema/dimensionality/row-map audit only. It does not validate TAIRID, H0 correction, or new physics."
    ),
}

ROW_LABEL_TERMS = [
    "host", "field", "galaxy", "id", "cepheid", "ceph", "ra", "dec",
    "period", "per", "f160w", "h", "hmag", "mag", "nir"
]

OUTCOME_TERMS = [
    "resid", "residual", "mu", "distance", "dist", "mag", "calib",
    "model", "fit", "sigma", "cov", "err", "error"
]


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
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


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:180]


def raw_url(branch, repo_path):
    return f"https://raw.githubusercontent.com/{REPO}/{branch}/{repo_path}"


def media_url(branch, repo_path):
    return f"https://media.githubusercontent.com/media/{REPO}/{branch}/{repo_path}"


def fetch_url_bytes(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TAIRID-v1.4.2-nir-fits-alignment-probe"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()
        return data, response.geturl(), response.headers.get("Content-Type", "")


def payload_kind(data):
    head = data[:512]
    text = head.decode("utf-8", errors="replace").lower()
    if head.startswith(b"SIMPLE") or head.startswith(b"XTENSION"):
        return "fits_like"
    if head.startswith(b"\x1f\x8b"):
        return "gzip_payload"
    if "version https://git-lfs.github.com/spec" in text:
        return "git_lfs_pointer"
    if text.lstrip().startswith("<!doctype html") or text.lstrip().startswith("<html"):
        return "html_payload"
    if "404: not found" in text or "not found" in text[:100]:
        return "not_found_payload"
    if len(data) < 2048 and all((32 <= b <= 126) or b in (9, 10, 13) for b in data):
        return "small_text_payload"
    return "unknown_binary_or_text_payload"


def maybe_decompress_gzip(data):
    if not data.startswith(b"\x1f\x8b"):
        return data, False, None
    try:
        return gzip.decompress(data), True, None
    except Exception as exc:
        return data, False, repr(exc)


def fetch_bytes_for_path(repo_path, prefer_media=False):
    errors = []
    for branch in BRANCH_CANDIDATES:
        urls = [media_url(branch, repo_path), raw_url(branch, repo_path)] if prefer_media else [raw_url(branch, repo_path), media_url(branch, repo_path)]
        for url in urls:
            try:
                data, final_url, content_type = fetch_url_bytes(url)
                kind = payload_kind(data)
                if kind in {"html_payload", "not_found_payload"}:
                    errors.append({"branch": branch, "url": url, "final_url": final_url, "content_type": content_type, "payload_kind": kind})
                    continue
                if kind in {"git_lfs_pointer", "small_text_payload"} and "raw.githubusercontent.com" in url:
                    errors.append({"branch": branch, "url": url, "final_url": final_url, "content_type": content_type, "payload_kind": kind, "note": "trying media URL"})
                    continue
                return {
                    "status": "downloaded",
                    "branch": branch,
                    "url": url,
                    "final_url": final_url,
                    "content_type": content_type,
                    "bytes": len(data),
                    "payload_kind": kind,
                    "data": data,
                    "errors": errors,
                }
            except Exception as exc:
                errors.append({"branch": branch, "url": url, "error": repr(exc)})
    return {
        "status": "failed",
        "branch": None,
        "url": None,
        "final_url": None,
        "content_type": None,
        "bytes": 0,
        "payload_kind": "download_failed",
        "data": b"",
        "errors": errors,
    }


def fetch_text_for_path(repo_path):
    fetched = fetch_bytes_for_path(repo_path, prefer_media=False)
    if fetched["status"] != "downloaded":
        return {**fetched, "text": ""}
    return {**fetched, "text": fetched["data"].decode("utf-8", errors="replace")}


def to_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_nir_text(label, repo_path, text):
    rows = []
    errors = []
    skipped_count = 0
    header_line = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if not stripped or stripped.startswith("-"):
            skipped_count += 1
            continue
        tokens = stripped.split()
        if tokens and tokens[0].lower() == "host":
            header_line = stripped
            continue
        if len(tokens) < 11:
            skipped_count += 1
            continue
        ra = to_float(tokens[1])
        dec = to_float(tokens[2])
        period = to_float(tokens[4])
        v_i = to_float(tokens[5])
        sigma_v_i = to_float(tokens[6])
        h_mag = to_float(tokens[7])
        sigma_h = to_float(tokens[8])
        metal = to_float(tokens[9])
        if None in [ra, dec, period, v_i, sigma_v_i, h_mag, sigma_h, metal]:
            errors.append({"line_number": line_number, "reason": "numeric_parse_failed", "line": raw[:240]})
            continue
        rows.append(
            {
                "dataset_label": label,
                "repo_path": repo_path,
                "source_line_number": line_number,
                "host": tokens[0],
                "ra": ra,
                "dec": dec,
                "id": tokens[3],
                "period": period,
                "log_period": math.log10(period) if period > 0 else None,
                "v_i": v_i,
                "sigma_v_i": sigma_v_i,
                "h_mag": h_mag,
                "sigma_h": sigma_h,
                "metal_minus_8_69": metal,
                "hst_flag": tokens[10],
                "raw_line": raw,
            }
        )
    return {
        "label": label,
        "repo_path": repo_path,
        "header_line": header_line,
        "skipped_count": skipped_count,
        "row_count": len(rows),
        "parse_error_count": len(errors),
        "rows": rows,
        "errors": errors,
    }


def summarize_nir_rows(rows):
    counts = Counter(row["host"] for row in rows)
    return {
        "row_count": len(rows),
        "host_count": len(counts),
        "host_counts": dict(sorted(counts.items())),
        "contains_LMC": counts.get("LMC", 0) > 0,
        "contains_SMC": counts.get("SMC", 0) > 0,
        "contains_N4536": counts.get("N4536", 0) > 0,
        "contains_M31": counts.get("M31", 0) > 0,
    }


def header_text(header):
    return "\n".join(str(card) for card in header.cards)


def selected_header(header):
    keep = {}
    for key in header.keys():
        lower = str(key).lower()
        if key in {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "EXTNAME", "XTENSION", "TFIELDS"}:
            keep[key] = header.get(key)
        elif any(term in lower for term in ["row", "col", "ceph", "nir", "f160", "sh0es", "pantheon", "cov", "fit", "data", "dim"]):
            value = header.get(key)
            if isinstance(value, (str, int, float, bool)):
                keep[key] = value
    return keep


def sample_table_values(data, columns, max_rows=3):
    sample = []
    if data is None or not columns:
        return sample
    n = min(max_rows, len(data))
    for i in range(n):
        row = {}
        for col in columns[:20]:
            try:
                value = data[col][i]
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                elif hasattr(value, "item"):
                    try:
                        value = value.item()
                    except Exception:
                        pass
                row[col] = value
            except Exception as exc:
                row[col] = f"ERROR:{repr(exc)}"
        sample.append(row)
    return sample


def non_fits_summary(label, role, repo_path, local_path, data_bytes, fetched, note):
    return {
        "label": label,
        "role": role,
        "repo_path": repo_path,
        "local_path": str(local_path),
        "bytes": len(data_bytes),
        "payload_kind": fetched.get("payload_kind"),
        "download_status": fetched.get("status"),
        "url": fetched.get("url"),
        "final_url": fetched.get("final_url"),
        "content_type": fetched.get("content_type"),
        "is_valid_fits": False,
        "fits_open_error": note,
        "hdu_rows": [],
        "possible_label_evidence": [],
        "possible_outcome_evidence": [],
        "dimensional_evidence": [],
        "payload_head_text": data_bytes[:800].decode("utf-8", errors="replace"),
    }


def inspect_fits_bytes(label, role, repo_path, data_bytes, fetched):
    local_path = DOWNLOAD_DIR / safe_name(repo_path)
    local_path.write_bytes(data_bytes)
    active_bytes, was_gzip, gzip_error = maybe_decompress_gzip(data_bytes)
    active_kind = payload_kind(active_bytes)
    if gzip_error:
        return non_fits_summary(label, role, repo_path, local_path, data_bytes, fetched, f"gzip_decompress_failed:{gzip_error}")
    if active_kind not in {"fits_like", "unknown_binary_or_text_payload"}:
        return non_fits_summary(label, role, repo_path, local_path, data_bytes, fetched, f"payload_not_fits_like:{active_kind}")

    hdu_rows = []
    possible_label_evidence = []
    possible_outcome_evidence = []
    dimensional_evidence = []
    try:
        try:
            hdul = fits.open(io.BytesIO(active_bytes), memmap=False)
        except OSError:
            hdul = fits.open(io.BytesIO(active_bytes), memmap=False, ignore_missing_simple=True)

        with hdul:
            for idx, hdu in enumerate(hdul):
                header = hdu.header
                data = hdu.data
                htxt = header_text(header).lower()
                shape = None
                ndim = None
                size = None
                dtype = None
                is_square_matrix = False
                is_vector_like = False
                is_table = False
                column_names = []
                label_like_columns = []
                outcome_like_columns = []
                sample_values = []

                if data is not None:
                    try:
                        shape = tuple(int(x) for x in data.shape)
                        ndim = int(np.ndim(data))
                        size = int(np.size(data))
                        dtype = str(data.dtype)
                        if len(shape) == 2 and shape[0] == shape[1]:
                            is_square_matrix = True
                        if len(shape) == 1 or (len(shape) == 2 and (shape[0] == 1 or shape[1] == 1)):
                            is_vector_like = True
                    except Exception:
                        shape = None

                if hasattr(hdu, "columns") and getattr(hdu, "columns", None) is not None:
                    try:
                        column_names = list(hdu.columns.names)
                        if column_names:
                            is_table = True
                            label_like_columns = [
                                c for c in column_names
                                if any(term == str(c).lower() or term in str(c).lower() for term in ROW_LABEL_TERMS)
                            ]
                            outcome_like_columns = [
                                c for c in column_names
                                if any(term == str(c).lower() or term in str(c).lower() for term in OUTCOME_TERMS)
                            ]
                            sample_values = sample_table_values(data, column_names)
                    except Exception:
                        column_names = []

                header_hits = {
                    "mentions_nir": "nir" in htxt,
                    "mentions_f160w": "f160w" in htxt or "f160" in htxt,
                    "mentions_cepheid": "cepheid" in htxt or "ceph" in htxt,
                    "mentions_covariance": "cov" in htxt or "covar" in htxt,
                    "mentions_residual": "resid" in htxt or "residual" in htxt,
                    "mentions_distance": "distance" in htxt or "dist" in htxt or "mu" in htxt,
                    "mentions_fit": "fit" in htxt or "model" in htxt,
                }

                hdu_row = {
                    "fits_label": label,
                    "role": role,
                    "repo_path": repo_path,
                    "local_path": str(local_path),
                    "hdu_index": idx,
                    "hdu_name": hdu.name,
                    "hdu_type": type(hdu).__name__,
                    "shape": json.dumps(shape),
                    "ndim": ndim,
                    "size": size,
                    "dtype": dtype,
                    "is_square_matrix": is_square_matrix,
                    "is_vector_like": is_vector_like,
                    "is_table": is_table,
                    "column_count": len(column_names),
                    "column_names": json.dumps(column_names, default=json_default),
                    "label_like_columns": json.dumps(label_like_columns, default=json_default),
                    "outcome_like_columns": json.dumps(outcome_like_columns, default=json_default),
                    "header_hits": json.dumps(header_hits, default=json_default),
                    "selected_header": json.dumps(selected_header(header), default=json_default),
                    "sample_values": json.dumps(sample_values, default=json_default),
                }
                hdu_rows.append(hdu_row)
                if label_like_columns:
                    possible_label_evidence.append(hdu_row)
                if outcome_like_columns or header_hits["mentions_residual"] or header_hits["mentions_distance"]:
                    possible_outcome_evidence.append(hdu_row)
                if shape is not None:
                    dimensional_evidence.append(hdu_row)

        return {
            "label": label,
            "role": role,
            "repo_path": repo_path,
            "local_path": str(local_path),
            "bytes": len(data_bytes),
            "payload_kind": fetched.get("payload_kind"),
            "download_status": fetched.get("status"),
            "url": fetched.get("url"),
            "final_url": fetched.get("final_url"),
            "content_type": fetched.get("content_type"),
            "was_gzip_decompressed": was_gzip,
            "is_valid_fits": True,
            "hdu_rows": hdu_rows,
            "possible_label_evidence": possible_label_evidence,
            "possible_outcome_evidence": possible_outcome_evidence,
            "dimensional_evidence": dimensional_evidence,
        }
    except Exception as exc:
        return non_fits_summary(label, role, repo_path, local_path, data_bytes, fetched, f"fits_open_failed:{repr(exc)}")


def compare_dimensions_to_nir(fits_summaries, nir_summaries):
    nir_row_counts = {label: summary["row_count"] for label, summary in nir_summaries.items()}
    all_counts = set(nir_row_counts.values())
    if "nir_orig19" in nir_row_counts and "nir_orig19_wm31" in nir_row_counts:
        all_counts.add(nir_row_counts["nir_orig19_wm31"] - nir_row_counts["nir_orig19"])
    rows = []
    for fs in fits_summaries:
        for hdu in fs.get("hdu_rows", []):
            try:
                shape = json.loads(hdu["shape"]) if hdu["shape"] else None
            except Exception:
                shape = None
            if not shape:
                continue
            matched_counts = []
            for dim in shape:
                for nir_label, count in nir_row_counts.items():
                    if dim == count:
                        matched_counts.append(f"{nir_label}:{count}")
                if dim in all_counts and not any(str(dim) in m for m in matched_counts):
                    matched_counts.append(f"delta_or_known_count:{dim}")
            rows.append(
                {
                    "fits_label": fs["label"],
                    "role": fs["role"],
                    "hdu_index": hdu["hdu_index"],
                    "hdu_name": hdu["hdu_name"],
                    "shape": hdu["shape"],
                    "is_square_matrix": hdu["is_square_matrix"],
                    "is_vector_like": hdu["is_vector_like"],
                    "matched_nir_counts": ";".join(matched_counts),
                    "has_any_nir_dimension_match": bool(matched_counts),
                    "interpretation": "dimension_match_only_not_row_alignment" if matched_counts else "no_dimension_match_to_v1_2_nir_row_counts",
                }
            )
    return rows


def detect_matrix_system(fits_summaries):
    role_shapes = {}
    for fs in fits_summaries:
        role_shapes[fs["role"]] = []
        for hdu in fs.get("hdu_rows", []):
            try:
                shape = json.loads(hdu["shape"]) if hdu["shape"] else None
            except Exception:
                shape = None
            if shape:
                role_shapes[fs["role"]].append(shape)
    system_candidates = []
    for c in role_shapes.get("C", []):
        if len(c) != 2 or c[0] != c[1]:
            continue
        n = c[0]
        for l in role_shapes.get("L", []):
            if len(l) != 2:
                continue
            for y in role_shapes.get("y", []):
                y_len = None
                if len(y) == 1:
                    y_len = y[0]
                elif len(y) == 2 and 1 in y:
                    y_len = max(y)
                if y_len is None:
                    continue
                system_candidates.append(
                    {
                        "C_shape": c,
                        "L_shape": l,
                        "y_shape": y,
                        "C_square_N": n,
                        "L_rows": l[0],
                        "L_cols": l[1],
                        "y_length": y_len,
                        "C_matches_y": n == y_len,
                        "L_rows_match_y": l[0] == y_len,
                        "full_yLC_dimensional_consistency": n == y_len and l[0] == y_len,
                        "parameter_count_candidate": l[1],
                    }
                )
    return system_candidates


def evaluate_row_alignment(fits_summaries, dimension_rows, matrix_system_candidates):
    valid_fits_count = sum(1 for fs in fits_summaries if fs.get("is_valid_fits"))
    non_fits_count = sum(1 for fs in fits_summaries if not fs.get("is_valid_fits"))
    label_evidence = []
    outcome_evidence = []
    shape_only = []
    for fs in fits_summaries:
        label_evidence.extend(fs.get("possible_label_evidence", []))
        outcome_evidence.extend(fs.get("possible_outcome_evidence", []))
    for row in dimension_rows:
        if row["has_any_nir_dimension_match"]:
            shape_only.append(row)
    has_explicit_row_labels = len(label_evidence) > 0
    has_outcome_evidence = len(outcome_evidence) > 0
    has_consistent_yLC = any(c.get("full_yLC_dimensional_consistency") for c in matrix_system_candidates)
    has_dimensional_match = len(shape_only) > 0

    if valid_fits_count == 0 and non_fits_count > 0:
        final_status = "fits_candidates_not_downloaded_as_valid_fits_no_row_alignment"
        readiness = 4
        next_wall = "Candidate .fits paths did not yield readable FITS products in this runtime. No row alignment can be proven from non-FITS payloads."
    elif has_explicit_row_labels and has_outcome_evidence:
        final_status = "explicit_row_label_and_outcome_candidate_found_manual_mapping_required"
        readiness = 8
        next_wall = "FITS files expose row-label and outcome-like fields. Next test must build and verify a row map before replay."
    elif has_explicit_row_labels:
        final_status = "explicit_row_label_candidate_found_but_outcome_not_proven"
        readiness = 7
        next_wall = "Some row-label-like fields exist, but a valid residual/outcome field is not proven. Manual mapping required."
    elif has_consistent_yLC:
        final_status = "ladder_matrix_system_confirmed_but_no_nir_row_labels"
        readiness = 6
        next_wall = "The high-level y/L/C ladder system appears dimensionally coherent, but no explicit NIR row labels or residual map were found. Do not replay frozen NIR edge surfaces yet."
    elif has_dimensional_match:
        final_status = "dimension_match_only_no_row_alignment_proof"
        readiness = 5
        next_wall = "At least one FITS dimension matches a v1.2 NIR row count, but dimension matching is not row alignment. Do not validate from this alone."
    else:
        final_status = "no_nir_row_alignment_proof_available"
        readiness = 4
        next_wall = "FITS files do not expose a lawful row map to NIR Cepheid surface rows. Treat the NIR lane as surface-only unless external documentation supplies a mapping."

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "evidence_counts": {
            "valid_fits_candidate_count": valid_fits_count,
            "non_fits_candidate_count": non_fits_count,
            "explicit_row_label_evidence_count": len(label_evidence),
            "outcome_evidence_count": len(outcome_evidence),
            "dimension_match_only_count": len(shape_only),
            "yLC_dimensional_system_candidate_count": len(matrix_system_candidates),
            "yLC_dimensional_system_consistent_count": sum(1 for c in matrix_system_candidates if c.get("full_yLC_dimensional_consistency")),
        },
        "truth_boundary": CLAIMS_V1_4_2["truth_boundary"],
    }


def context_sniff(text):
    lower = text.lower()
    return {
        "mentions_y_l_c": "data vector" in lower and "equation matrix" in lower and "covariance" in lower,
        "mentions_fits_files": ".fits" in lower or "fits" in lower,
        "mentions_table2_not_recommended_for_refit": "not to use this table" in lower,
        "mentions_covariance_missing_from_table2": "do not include the covariance" in lower or "covariance" in lower,
        "mentions_nir": "nir" in lower,
        "mentions_cepheid": "cepheid" in lower or "cepheids" in lower,
        "first_1200_chars": text[:1200],
    }


def holographic_surface_ledger(decision):
    return {
        "observable_surface": {
            "name": "SH0ES high-level FITS candidates plus v1.2 NIR H-band Cepheid rows",
            "nir_surface_files": NIR_SURFACE_FILES,
            "fits_candidate_files": FITS_CANDIDATES,
        },
        "hidden_depth_sought": {
            "name": "A lawful row map from NIR Cepheid surface rows into y/L/C outcome depth",
            "why_needed": "The frozen high/low H-edge surface cannot be replayed against the ladder system unless the outcome/covariance rows corresponding to NIR observations are known.",
        },
        "boundary_that_forms_surface": {
            "release_boundary": REPO,
            "data_boundary": "Public SH0ES/PantheonPlus release files only",
            "method_boundary": "Schema, header, dimension, and label audit only; no model fitting",
        },
        "what_information_is_lost_or_missing_if_no_map_found": [
            "Which y-vector rows correspond to NIR Cepheids.",
            "Which covariance rows/columns correspond to NIR Cepheids.",
            "Which equation-matrix rows encode the F160W/H-band Cepheid surface.",
            "Whether high/low H-edge Cepheids have any direct residual/outcome representation.",
            "Whether frozen v1.0 Table2 residual behavior transfers to the high-level ladder system.",
        ],
        "what_can_be_reconstructed_now": [
            "NIR surface row counts and host inventory.",
            "Whether candidate .fits paths downloaded as valid FITS payloads.",
            "FITS HDU schemas if readable.",
            "Matrix/vector dimensional consistency if readable.",
            "Presence or absence of row-label-like metadata if readable.",
            "Whether a replay is legally allowed under the truth boundary.",
        ],
        "what_cannot_be_reconstructed_now": [
            "Frozen-rule predictive validation.",
            "H0 correction.",
            "New physics.",
            "Row-level residuals if they are not exposed or mappable.",
        ],
        "surface_noise_definition": [
            "File paths ending in .fits that download as non-FITS payloads.",
            "Dimension matches without row labels.",
            "README mentions of NIR/covariance that do not expose row-level data.",
            "Full ladder matrices treated as if they were NIR-only residual surfaces.",
            "Residuals created by subtracting magnitudes without an accepted outcome definition.",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v1.4.2 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v1.4.2 did")
    lines.append("")
    lines.append("- Created a new-file-name version so GitHub does not reuse or collide with old v1.4 files.")
    lines.append("- Downloaded the v1.2 NIR Cepheid surface files.")
    lines.append("- Attempted to download the three high-level SH0ES ladder FITS candidates: C, L, and y.")
    lines.append("- Checked whether payloads were valid FITS before opening them with Astropy.")
    lines.append("- If readable, inspected HDUs, headers, dimensions, table columns, and possible label/outcome fields.")
    lines.append("- Did not replay the frozen edge rule.")
    lines.append("")
    lines.append("## Frozen rule carried forward")
    lines.append("")
    lines.append("- F160W/H-like surface only.")
    lines.append("- Within-host high 5% H edge minus within-host low 5% H edge.")
    lines.append("- Low-alpha/reference hosts: LMC, SMC, N4536.")
    lines.append("- M31 sign-break quarantine.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v1.4.2 is schema/dimensionality/row-map audit only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not create residuals from H magnitudes.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append("v1.5 should build a row-map candidate and test whether NIR high/low edge rows can be located in the outcome system. Replay is still not allowed until that map passes.")
    elif decision["readiness_score_0_to_10"] >= 6:
        lines.append("v1.5 should inspect the ladder equation matrix terms and documentation to determine whether any column/row metadata can identify the Cepheid/NIR subset. Do not replay the frozen edge rule yet.")
    else:
        lines.append("v1.5 should classify the NIR lane as surface-only context unless external documentation supplies a lawful row map. The next validation lane should return to a data product with explicit residual/outcome columns.")
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID NIR FITS Alignment Probe v1.4.2 starting.")
    print("Boundary: schema/dimensionality/row-map audit only; no validation and no tuning.")

    write_json(OUTDIR / "claims_v1_4_2.json", CLAIMS_V1_4_2)
    write_json(OUTDIR / "frozen_rule_carried_forward_v1_4_2.json", FROZEN_RULE_CARRIED_FORWARD)

    try:
        download_ledger = []
        nir_summaries = {}
        all_nir_rows = []
        context_files = []

        for target in NIR_SURFACE_FILES:
            fetched = fetch_text_for_path(target["repo_path"])
            download_ledger.append(
                {
                    "kind": "nir_surface",
                    "label": target["label"],
                    "repo_path": target["repo_path"],
                    "status": fetched["status"],
                    "branch": fetched["branch"],
                    "url": fetched["url"],
                    "final_url": fetched.get("final_url"),
                    "content_type": fetched.get("content_type"),
                    "payload_kind": fetched.get("payload_kind"),
                    "bytes": fetched["bytes"],
                    "errors": json.dumps(fetched["errors"], default=json_default),
                }
            )
            if fetched["status"] == "downloaded":
                local_path = DOWNLOAD_DIR / safe_name(target["repo_path"])
                local_path.write_text(fetched["text"], encoding="utf-8")
                parsed = parse_nir_text(target["label"], target["repo_path"], fetched["text"])
                nir_summaries[target["label"]] = summarize_nir_rows(parsed["rows"])
                all_nir_rows.extend(parsed["rows"])
                write_csv(OUTDIR / f"{target['label']}_parsed_rows_v1_4_2.csv", parsed["rows"])
                write_json(OUTDIR / f"{target['label']}_parse_errors_v1_4_2.json", parsed["errors"])

        write_csv(OUTDIR / "all_nir_surface_rows_v1_4_2.csv", all_nir_rows)
        write_json(OUTDIR / "nir_surface_summaries_v1_4_2.json", nir_summaries)

        for repo_path in REFERENCE_CONTEXT_FILES:
            fetched = fetch_text_for_path(repo_path)
            download_ledger.append(
                {
                    "kind": "reference_context",
                    "label": Path(repo_path).name,
                    "repo_path": repo_path,
                    "status": fetched["status"],
                    "branch": fetched["branch"],
                    "url": fetched["url"],
                    "final_url": fetched.get("final_url"),
                    "content_type": fetched.get("content_type"),
                    "payload_kind": fetched.get("payload_kind"),
                    "bytes": fetched["bytes"],
                    "errors": json.dumps(fetched["errors"], default=json_default),
                }
            )
            if fetched["status"] == "downloaded":
                local_path = DOWNLOAD_DIR / safe_name(repo_path)
                local_path.write_text(fetched["text"], encoding="utf-8")
                context_files.append({"repo_path": repo_path, "sniff": context_sniff(fetched["text"])})
        write_json(OUTDIR / "reference_context_sniffs_v1_4_2.json", context_files)

        fits_summaries = []
        all_hdu_rows = []
        for target in FITS_CANDIDATES:
            fetched = fetch_bytes_for_path(target["repo_path"], prefer_media=True)
            download_ledger.append(
                {
                    "kind": "fits_candidate",
                    "label": target["label"],
                    "repo_path": target["repo_path"],
                    "status": fetched["status"],
                    "branch": fetched["branch"],
                    "url": fetched["url"],
                    "final_url": fetched.get("final_url"),
                    "content_type": fetched.get("content_type"),
                    "payload_kind": fetched.get("payload_kind"),
                    "bytes": fetched["bytes"],
                    "errors": json.dumps(fetched["errors"], default=json_default),
                }
            )
            if fetched["status"] != "downloaded":
                fs = non_fits_summary(target["label"], target["role"], target["repo_path"], DOWNLOAD_DIR / safe_name(target["repo_path"]), b"", fetched, "download_failed")
            else:
                fs = inspect_fits_bytes(target["label"], target["role"], target["repo_path"], fetched["data"], fetched)
            fits_summaries.append(fs)
            all_hdu_rows.extend(fs.get("hdu_rows", []))

        write_csv(OUTDIR / "download_ledger_v1_4_2.csv", download_ledger)
        write_json(OUTDIR / "fits_schema_full_v1_4_2.json", fits_summaries)
        write_csv(OUTDIR / "fits_hdu_schema_rows_v1_4_2.csv", all_hdu_rows)

        non_fits_rows = [
            {
                "label": fs["label"],
                "role": fs["role"],
                "repo_path": fs["repo_path"],
                "payload_kind": fs.get("payload_kind"),
                "bytes": fs.get("bytes"),
                "url": fs.get("url"),
                "final_url": fs.get("final_url"),
                "content_type": fs.get("content_type"),
                "fits_open_error": fs.get("fits_open_error"),
                "payload_head_text": fs.get("payload_head_text", "")[:500],
            }
            for fs in fits_summaries
            if not fs.get("is_valid_fits")
        ]
        write_csv(OUTDIR / "non_fits_payloads_v1_4_2.csv", non_fits_rows)

        dimension_rows = compare_dimensions_to_nir(fits_summaries, nir_summaries)
        write_csv(OUTDIR / "fits_dimension_vs_nir_counts_v1_4_2.csv", dimension_rows)

        matrix_system_candidates = detect_matrix_system(fits_summaries)
        write_json(OUTDIR / "yLC_matrix_system_candidates_v1_4_2.json", matrix_system_candidates)

        decision = evaluate_row_alignment(fits_summaries, dimension_rows, matrix_system_candidates)
        write_json(OUTDIR / "decision_v1_4_2.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v1_4_2.json", ledger)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v1_4_2.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v1_4_2.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID NIR FITS Alignment Probe v1.4.2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Schema/dimensionality/row-map audit only. No validation, no tuning, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "download_ledger": download_ledger,
            "nir_surface_summaries": nir_summaries,
            "fits_candidate_files": FITS_CANDIDATES,
            "fits_summaries": fits_summaries,
            "fits_hdu_schema_rows": all_hdu_rows,
            "non_fits_payloads": non_fits_rows,
            "dimension_vs_nir_counts": dimension_rows,
            "matrix_system_candidates": matrix_system_candidates,
            "decision": decision,
            "claims_v1_4_2": CLAIMS_V1_4_2,
            "frozen_rule_carried_forward": FROZEN_RULE_CARRIED_FORWARD,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "nir_fits_alignment_probe_v1_4_2_summary.json"),
                "summary_txt": str(OUTDIR / "nir_fits_alignment_probe_v1_4_2_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v1_4_2.csv"),
                "non_fits_payloads_csv": str(OUTDIR / "non_fits_payloads_v1_4_2.csv"),
                "nir_surface_summaries_json": str(OUTDIR / "nir_surface_summaries_v1_4_2.json"),
                "fits_schema_full_json": str(OUTDIR / "fits_schema_full_v1_4_2.json"),
                "fits_hdu_schema_rows_csv": str(OUTDIR / "fits_hdu_schema_rows_v1_4_2.csv"),
                "dimension_vs_nir_counts_csv": str(OUTDIR / "fits_dimension_vs_nir_counts_v1_4_2.csv"),
                "matrix_system_candidates_json": str(OUTDIR / "yLC_matrix_system_candidates_v1_4_2.json"),
                "decision_json": str(OUTDIR / "decision_v1_4_2.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v1_4_2.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v1_4_2.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v1_4_2.txt"),
            },
            "interpretation": {
                "what_success_means": "A lawful row map may exist between NIR surface rows and the high-level ladder y/L/C system.",
                "what_success_does_not_mean": "Even if a row map candidate is found, this does not validate the frozen Table2 edge rule.",
                "what_failure_means": "If no explicit row map is exposed, the NIR lane remains surface-only context for this testing path.",
                "truth_boundary": CLAIMS_V1_4_2["truth_boundary"],
            },
        }
        write_json(OUTDIR / "nir_fits_alignment_probe_v1_4_2_summary.json", summary)

        with open(OUTDIR / "nir_fits_alignment_probe_v1_4_2_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID NIR FITS Alignment Probe v1.4.2\n\n")
            f.write("Boundary: schema/dimensionality/row-map audit only. No validation. No tuning.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Evidence counts:\n")
            f.write(json.dumps(decision["evidence_counts"], indent=2, default=json_default) + "\n\n")
            f.write("NIR surface summaries:\n")
            f.write(json.dumps(nir_summaries, indent=2, default=json_default) + "\n\n")
            f.write("Non-FITS payloads:\n")
            f.write(json.dumps(non_fits_rows, indent=2, default=json_default) + "\n\n")
            f.write("y/L/C matrix system candidates:\n")
            f.write(json.dumps(matrix_system_candidates, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not create residuals from H magnitudes.\n")
            f.write("- Do not replay without a lawful row map.\n")

        print("TAIRID NIR FITS Alignment Probe v1.4.2 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID NIR FITS Alignment Probe v1.4.2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "nir_fits_alignment_probe_v1_4_2_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V1_4_2["truth_boundary"],
        }
        write_json(OUTDIR / "nir_fits_alignment_probe_v1_4_2_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
