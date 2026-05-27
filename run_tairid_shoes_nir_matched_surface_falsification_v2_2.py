#!/usr/bin/env python3
"""
TAIRID SH0ES NIR Matched Surface Falsification v2.2

Why this test exists:
v2.1 showed that the locked F160W residual edge surface stayed stronger than
predeclared Table2 negative-control variables. The next clean move is an
adjacent external-lane falsification attempt, not a tuned model upgrade.

v2.2 asks whether the locked residual pattern can be explained away by the
adjacent public NIR Cepheid tables rather than by the residual layer itself.

It uses the same public SH0ES ladder files:

    table2.tex
    lstsq_results.txt
    ally_shoes_ceph_topantheonwt6.0_112221.fits
    alll_shoes_ceph_topantheonwt6.0_112221.fits
    allc_shoes_ceph_topantheonwt6.0_112221.fits

and the adjacent public NIR tables:

    R22_orig19_NIR.out
    R22_orig19_NIR.wm31.out

Core checks:

    1. Rebuild the same Table2 ladder residuals: residual = y - theta @ L.
    2. Parse the adjacent NIR H/F160W-like tables.
    3. Match NIR rows to Table2 rows by host + Cepheid ID, with coordinate/period diagnostics.
    4. Re-run the locked F160W residual edge on the matched subset.
    5. Re-run the same residual edge sorted by external NIR H magnitude.
    6. Test whether H - F160W mismatch across frozen edges is large enough to explain the locked residual pattern.
    7. Report NIR-only raw H edge surfaces without treating them as residual validation.

This test does NOT validate TAIRID.
This test does NOT tune the frozen rule.
This test does NOT search a new variable.
This test does NOT add new regimes.
This test does NOT claim H0 correction or new physics.
This test does NOT prove SH0ES is wrong.

Truth boundary:
If the NIR matched surface does not explain the locked residual pattern, that is
only an external-lane falsification pass. It is not proof. If the NIR surface
does explain it, that is a real fragility result and must be reported honestly.
"""

import csv
import io
import json
import math
import re
import traceback
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


OUTDIR = Path("tairid_shoes_nir_matched_surface_falsification_v2_2_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO = "PantheonPlusSH0ES/DataRelease"
BRANCH_CANDIDATES = ["main", "master"]

TABLE2_PATH = "SH0ES_Data/table2.tex"
README_PATH = "SH0ES_Data/README.md"
TABLE2_README_PATH = "SH0ES_Data/table2.README"
LSTSQ_PATH = "SH0ES_Data/lstsq_results.txt"

Y_FITS_PATH = "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits"
L_FITS_PATH = "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits"
C_FITS_PATH = "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits"

NIR_TARGETS = [
    {
        "label": "nir_orig19",
        "repo_path": "SH0ES_Data/R22_orig19_NIR.out",
    },
    {
        "label": "nir_orig19_wm31",
        "repo_path": "SH0ES_Data/R22_orig19_NIR.wm31.out",
    },
]

EDGE_PERCENTILE = 5.0
MIN_HOST_ROWS_FOR_EDGE = 20
LOW_ALPHA_REFERENCE_HOSTS = {"LMC", "SMC", "N4536"}
SIGN_BREAK_QUARANTINE_HOSTS = {"M31"}

FROZEN_RULE = {
    "source_status": "v2.2 adjacent NIR matched-surface falsification of locked v1.9/v2.0/v2.1 frozen replay",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "locked_variable": "f160w parsed from Table2.tex",
    "adjacent_external_surface": "R22 NIR H/F160W-like tables",
    "edge_rule": "within-host high 5% sort-variable edge minus within-host low 5% sort-variable edge",
    "edge_percentile": EDGE_PERCENTILE,
    "minimum_host_rows_for_edge": MIN_HOST_ROWS_FOR_EDGE,
    "low_alpha_reference_hosts": sorted(LOW_ALPHA_REFERENCE_HOSTS),
    "sign_break_quarantine_hosts": sorted(SIGN_BREAK_QUARANTINE_HOSTS),
    "clean_high_alpha_rule": "all active Table2 hosts except LMC, SMC, N4536, and M31",
    "hard_boundary": [
        "Do not tune the F160W edge percentile in v2.2.",
        "Do not replace F160W with NIR H as a new locked variable.",
        "Do not add or remove frozen regimes in v2.2.",
        "Do not transfer M31 correction; M31 is quarantine/report-only.",
        "Do not treat raw NIR H as a residual layer.",
        "Do not claim H0 correction or new physics.",
        "Do not claim TAIRID is proven.",
    ],
}

CLAIMS_V2_2 = {
    "battery_name": "TAIRID SH0ES NIR Matched Surface Falsification v2.2",
    "scope": "Adjacent NIR row-match falsification of the locked F160W residual edge surface",
    "primary_question": (
        "Can the adjacent NIR H/F160W-like surface explain away the locked Table2 F160W residual edge pattern, "
        "or does the pattern remain specific to the ladder residual surface?"
    ),
    "truth_boundary": (
        "This is adjacent-lane falsification only. It does not validate TAIRID, H0 correction, or new physics."
    ),
}


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
        for key in row:
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
        headers={"User-Agent": "TAIRID-v2.2-shoes-nir-matched-surface-falsification"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()
        return data, response.geturl(), response.headers.get("Content-Type", "")


def payload_kind(data):
    head = data[:512]
    text = head.decode("utf-8", errors="replace").lower()
    if head.startswith(b"SIMPLE") or head.startswith(b"XTENSION"):
        return "fits_like"
    if "version https://git-lfs.github.com/spec" in text:
        return "git_lfs_pointer"
    if text.lstrip().startswith("<!doctype html") or text.lstrip().startswith("<html"):
        return "html_payload"
    if "404: not found" in text or "not found" in text[:100]:
        return "not_found_payload"
    if len(data) < 4096 and all((32 <= b <= 126) or b in (9, 10, 13) for b in data):
        return "small_text_payload"
    return "unknown_binary_or_text_payload"


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
                if kind in {"git_lfs_pointer", "small_text_payload"} and "raw.githubusercontent.com" in url and repo_path.lower().endswith(".fits"):
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


def parse_float(token):
    try:
        value = float(str(token).replace("D", "E"))
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def parse_table2_tex(text):
    rows = []
    errors = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        raw = line.rstrip("\n")
        stripped = raw.strip()

        if not stripped:
            continue
        if "&" not in stripped or not stripped.endswith("\\\\"):
            continue
        if stripped.startswith("\\") or "colhead" in stripped or "startdata" in stripped or "enddata" in stripped:
            continue

        parts = [p.strip() for p in stripped.replace("\\\\", "").split("&")]
        if len(parts) < 11:
            continue

        host = parts[0]
        ra = parse_float(parts[1])
        dec = parse_float(parts[2])
        period = parse_float(parts[4])
        v_i = parse_float(parts[5])
        sigma_v_i = parse_float(parts[6])
        f160w = parse_float(parts[7])
        sigma_f160w = parse_float(parts[8])
        metal = parse_float(parts[9])

        if None in [ra, dec, period, v_i, sigma_v_i, f160w, sigma_f160w, metal]:
            errors.append({"line_number": line_number, "reason": "numeric_parse_failed", "line": raw[:240]})
            continue

        rows.append(
            {
                "table2_row_index_0_based": len(rows),
                "source_line_number": line_number,
                "host": host,
                "ra": ra,
                "dec": dec,
                "id": parts[3],
                "period": period,
                "log_period": math.log10(period) if period > 0 else None,
                "v_i": v_i,
                "sigma_v_i": sigma_v_i,
                "f160w": f160w,
                "sigma_f160w": sigma_f160w,
                "metal_minus_8_69": metal,
                "note": parts[10],
                "raw_line": raw,
            }
        )
    return {"row_count": len(rows), "rows": rows, "errors": errors}


def parse_nir_text(label, repo_path, text):
    rows = []
    errors = []
    header_line = None
    skipped_count = 0

    for line_number, line in enumerate(text.splitlines(), start=1):
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if not stripped:
            skipped_count += 1
            continue
        if stripped.startswith("-"):
            skipped_count += 1
            continue
        tokens = stripped.split()
        if tokens and tokens[0].lower() == "host":
            header_line = stripped
            continue
        if len(tokens) < 11:
            skipped_count += 1
            continue

        host = tokens[0]
        ra = parse_float(tokens[1])
        dec = parse_float(tokens[2])
        period = parse_float(tokens[4])
        v_i = parse_float(tokens[5])
        sigma_v_i = parse_float(tokens[6])
        h_mag = parse_float(tokens[7])
        sigma_h = parse_float(tokens[8])
        metal = parse_float(tokens[9])

        if None in [ra, dec, period, v_i, sigma_v_i, h_mag, sigma_h, metal]:
            errors.append({"line_number": line_number, "reason": "numeric_parse_failed", "line": raw[:240]})
            continue

        rows.append(
            {
                "nir_dataset_label": label,
                "nir_repo_path": repo_path,
                "nir_row_index_0_based": len(rows),
                "nir_source_line_number": line_number,
                "host": host,
                "ra_nir": ra,
                "dec_nir": dec,
                "id": tokens[3],
                "period_nir": period,
                "log_period_nir": math.log10(period) if period > 0 else None,
                "v_i_nir": v_i,
                "sigma_v_i_nir": sigma_v_i,
                "h_mag_nir": h_mag,
                "sigma_h_nir": sigma_h,
                "metal_minus_8_69_nir": metal,
                "hst_flag_nir": tokens[10],
                "raw_line_nir": raw,
            }
        )

    return {
        "label": label,
        "repo_path": repo_path,
        "header_line": header_line,
        "row_count": len(rows),
        "skipped_count": skipped_count,
        "parse_error_count": len(errors),
        "rows": rows,
        "errors": errors,
    }


def read_primary_fits_array(data_bytes):
    with fits.open(io.BytesIO(data_bytes), memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=float)


def parse_lstsq_results(text):
    rows = []
    errors = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        value = parse_float(tokens[0]) if tokens else None
        sigma = parse_float(tokens[1]) if len(tokens) > 1 else None
        if value is None:
            errors.append({"line_number": line_number, "line": line[:240], "reason": "first_token_not_float"})
            continue
        rows.append(
            {
                "theta_index_0_based": len(rows),
                "theta_value": value,
                "theta_sigma_or_release_second_column": sigma,
                "source_line_number": line_number,
                "raw_line": line,
            }
        )
    return {"theta_count": len(rows), "rows": rows, "errors": errors}


def summarize_numeric(values):
    arr = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if len(arr) == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_ladder_residual(theta, L, y):
    theta = np.asarray(theta, dtype=float)
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)

    if L.ndim != 2:
        return None, None, "L_not_2D"
    if y.ndim != 1:
        return None, None, "y_not_1D"

    if L.shape[0] == len(theta) and L.shape[1] == len(y):
        predicted = np.dot(theta, L)
        orientation = "theta_dot_L"
    elif L.shape[1] == len(theta) and L.shape[0] == len(y):
        predicted = np.dot(L, theta)
        orientation = "L_dot_theta"
    else:
        return None, None, "theta_L_y_shape_mismatch"

    return predicted, y - predicted, orientation


def covariance_diag_info(C):
    if C is None or not isinstance(C, np.ndarray) or C.ndim != 2 or C.shape[0] != C.shape[1]:
        return None, {"covariance_valid_square": False, "reason": "C missing or not square"}
    diag = np.diag(C)
    return diag, {
        "covariance_valid_square": True,
        "C_shape": list(C.shape),
        "diag_count": int(len(diag)),
        "diag_positive_count": int(np.sum(diag > 0)),
        "diag_summary": summarize_numeric(diag.tolist()),
        "sigma_summary": summarize_numeric(np.sqrt(diag[diag > 0]).tolist() if np.any(diag > 0) else []),
    }


def frozen_role_for_host(host):
    if host in LOW_ALPHA_REFERENCE_HOSTS:
        return "low_alpha_reference"
    if host in SIGN_BREAK_QUARANTINE_HOSTS:
        return "sign_break_quarantine"
    return "clean_high_alpha_candidate"


def attach_table2_residuals(table_rows, y, predicted_y, residual, covariance_diag):
    out = []
    for i, r in enumerate(table_rows):
        sigma_c = None
        standardized = None
        if covariance_diag is not None and i < len(covariance_diag) and covariance_diag[i] > 0:
            sigma_c = float(math.sqrt(covariance_diag[i]))
            standardized = float(residual[i] / sigma_c)
        o = dict(r)
        o["y_vector_index_0_based"] = i
        o["y_value"] = float(y[i])
        o["predicted_y_from_theta_L"] = float(predicted_y[i])
        o["ladder_residual_y_minus_thetaL"] = float(residual[i])
        o["covariance_diag"] = float(covariance_diag[i]) if covariance_diag is not None and i < len(covariance_diag) else None
        o["covariance_sigma"] = sigma_c
        o["standardized_residual"] = standardized
        o["frozen_regime_role"] = frozen_role_for_host(r["host"])
        out.append(o)
    return out


def match_nir_to_table2(table_residual_rows, nir_rows):
    by_key = defaultdict(list)
    for row in table_residual_rows:
        by_key[(row["host"], row["id"])].append(row)

    used_table_indices = set()
    matched = []
    unmatched_nir = []

    for nir in nir_rows:
        candidates = [
            row for row in by_key.get((nir["host"], nir["id"]), [])
            if row["table2_row_index_0_based"] not in used_table_indices
        ]

        if not candidates:
            unmatched_nir.append({**nir, "unmatched_reason": "no_unused_host_id_match"})
            continue

        def score(row):
            period_score = abs((row["period"] or 0.0) - (nir["period_nir"] or 0.0))
            ra_score = abs((row["ra"] or 0.0) - (nir["ra_nir"] or 0.0))
            dec_score = abs((row["dec"] or 0.0) - (nir["dec_nir"] or 0.0))
            return (period_score, ra_score + dec_score)

        best = sorted(candidates, key=score)[0]
        used_table_indices.add(best["table2_row_index_0_based"])

        out = dict(best)
        for k, v in nir.items():
            if k in {"host", "id"}:
                out[f"{k}_nir"] = v
            else:
                out[k] = v
        out["match_method"] = "host_id_best_period_coordinate"
        out["period_abs_diff"] = abs(best["period"] - nir["period_nir"])
        out["ra_abs_diff"] = abs(best["ra"] - nir["ra_nir"])
        out["dec_abs_diff"] = abs(best["dec"] - nir["dec_nir"])
        out["h_minus_f160w"] = nir["h_mag_nir"] - best["f160w"]
        out["abs_h_minus_f160w"] = abs(out["h_minus_f160w"])
        matched.append(out)

    unmatched_table = [
        row for row in table_residual_rows
        if row["table2_row_index_0_based"] not in used_table_indices
    ]

    return {
        "matched_rows": matched,
        "unmatched_nir_rows": unmatched_nir,
        "unmatched_table2_rows": unmatched_table,
    }


def edge_covariance_diagnostic(low_side, high_side, C):
    if C is None or C.ndim != 2 or C.shape[0] != C.shape[1]:
        return {"covariance_edge_variance_full": None, "covariance_edge_sigma_full": None, "covariance_edge_z_full": None}
    low_idx = [r["y_vector_index_0_based"] for r in low_side]
    high_idx = [r["y_vector_index_0_based"] for r in high_side]
    idxs = high_idx + low_idx
    k_high = len(high_idx)
    k_low = len(low_idx)
    weights = np.asarray(([1.0 / k_high] * k_high) + ([-1.0 / k_low] * k_low), dtype=float)
    delta = float(np.mean([r["ladder_residual_y_minus_thetaL"] for r in high_side]) - np.mean([r["ladder_residual_y_minus_thetaL"] for r in low_side]))
    sub = C[np.ix_(idxs, idxs)]
    var_full = float(weights @ sub @ weights)
    sigma_full = math.sqrt(var_full) if var_full > 0 else None
    return {
        "covariance_edge_variance_full": var_full,
        "covariance_edge_sigma_full": sigma_full,
        "covariance_edge_z_full": float(delta / sigma_full) if sigma_full else None,
    }


def build_residual_edges(rows, C, sort_variable, value_variable="ladder_residual_y_minus_thetaL", label=""):
    by_host = defaultdict(list)
    for row in rows:
        value = row.get(sort_variable)
        if value is not None and math.isfinite(float(value)):
            by_host[row["host"]].append(row)

    edge_rows = []
    inventory = []

    for host, items in sorted(by_host.items()):
        items = sorted(items, key=lambda r: r[sort_variable])
        n = len(items)
        role = frozen_role_for_host(host)

        if n < MIN_HOST_ROWS_FOR_EDGE:
            inventory.append(
                {
                    "analysis_label": label,
                    "sort_variable": sort_variable,
                    "value_variable": value_variable,
                    "host": host,
                    "row_count": n,
                    "edge_status": "not_enough_rows",
                    "edge_count_each_side": 0,
                    "frozen_regime_role": role,
                }
            )
            continue

        k = max(1, int(math.floor(n * EDGE_PERCENTILE / 100.0)))
        low_side = items[:k]
        high_side = items[-k:]

        for side_name, selected in [("low_sort_edge", low_side), ("high_sort_edge", high_side)]:
            for r in selected:
                out = {
                    "analysis_label": label,
                    "sort_variable": sort_variable,
                    "value_variable": value_variable,
                    "host": host,
                    "edge_side": side_name,
                    "table2_row_index_0_based": r.get("table2_row_index_0_based"),
                    "nir_row_index_0_based": r.get("nir_row_index_0_based"),
                    "sort_value": r[sort_variable],
                    "value": r[value_variable],
                    "frozen_regime_role": role,
                }
                edge_rows.append(out)

        low_values = np.asarray([r[value_variable] for r in low_side], dtype=float)
        high_values = np.asarray([r[value_variable] for r in high_side], dtype=float)
        low_sort = np.asarray([r[sort_variable] for r in low_side], dtype=float)
        high_sort = np.asarray([r[sort_variable] for r in high_side], dtype=float)

        cov = {}
        if value_variable == "ladder_residual_y_minus_thetaL" and all("y_vector_index_0_based" in r for r in low_side + high_side):
            cov = edge_covariance_diagnostic(low_side, high_side, C)

        inventory.append(
            {
                "analysis_label": label,
                "sort_variable": sort_variable,
                "value_variable": value_variable,
                "host": host,
                "row_count": n,
                "edge_status": "edge_surface_built",
                "edge_count_each_side": k,
                "low_edge_mean_sort_value": float(np.mean(low_sort)),
                "high_edge_mean_sort_value": float(np.mean(high_sort)),
                "high_minus_low_mean_sort_value": float(np.mean(high_sort) - np.mean(low_sort)),
                "low_edge_mean_value": float(np.mean(low_values)),
                "high_edge_mean_value": float(np.mean(high_values)),
                "high_minus_low_mean_value": float(np.mean(high_values) - np.mean(low_values)),
                "low_edge_median_value": float(np.median(low_values)),
                "high_edge_median_value": float(np.median(high_values)),
                "high_minus_low_median_value": float(np.median(high_values) - np.median(low_values)),
                "frozen_regime_role": role,
                **cov,
            }
        )

    return {
        "edge_rows": edge_rows,
        "edge_inventory": inventory,
    }


def build_nir_only_raw_h_edges(nir_rows, label):
    rows = []
    for r in nir_rows:
        out = dict(r)
        out["frozen_regime_role"] = frozen_role_for_host(r["host"])
        rows.append(out)
    return build_residual_edges(rows, None, "h_mag_nir", value_variable="h_mag_nir", label=label)


def aggregate_edges(edge_inventory):
    roles = defaultdict(list)
    for row in edge_inventory:
        if row.get("edge_status") == "edge_surface_built":
            roles[row["frozen_regime_role"]].append(row)

    out = {}
    for role, rows in roles.items():
        values = np.asarray([r["high_minus_low_mean_value"] for r in rows], dtype=float)
        weights = np.asarray([r["edge_count_each_side"] for r in rows], dtype=float)
        zvals = np.asarray([r["covariance_edge_z_full"] for r in rows if r.get("covariance_edge_z_full") is not None], dtype=float)
        out[role] = {
            "host_count": int(len(rows)),
            "hosts": [r["host"] for r in rows],
            "weighted_mean_high_minus_low_value": float(np.average(values, weights=weights)) if len(values) and np.sum(weights) > 0 else None,
            "unweighted_mean_high_minus_low_value": float(np.mean(values)) if len(values) else None,
            "unweighted_median_high_minus_low_value": float(np.median(values)) if len(values) else None,
            "std_high_minus_low_value": float(np.std(values)) if len(values) else None,
            "mean_covariance_edge_z_full": float(np.mean(zvals)) if len(zvals) else None,
        }

    clean = out.get("clean_high_alpha_candidate", {})
    ref = out.get("low_alpha_reference", {})
    out["clean_minus_reference"] = {
        "weighted_delta": None,
        "unweighted_delta": None,
        "truth_boundary": "descriptive_external_falsification_contrast_only_not_H0_correction",
    }
    if clean and ref:
        if clean.get("weighted_mean_high_minus_low_value") is not None and ref.get("weighted_mean_high_minus_low_value") is not None:
            out["clean_minus_reference"]["weighted_delta"] = clean["weighted_mean_high_minus_low_value"] - ref["weighted_mean_high_minus_low_value"]
        if clean.get("unweighted_mean_high_minus_low_value") is not None and ref.get("unweighted_mean_high_minus_low_value") is not None:
            out["clean_minus_reference"]["unweighted_delta"] = clean["unweighted_mean_high_minus_low_value"] - ref["unweighted_mean_high_minus_low_value"]
    return out


def make_dataset_analysis(dataset_label, table_residual_rows, nir_rows, C):
    matched = match_nir_to_table2(table_residual_rows, nir_rows)
    matched_rows = matched["matched_rows"]

    full_f160w = build_residual_edges(table_residual_rows, C, "f160w", "ladder_residual_y_minus_thetaL", "table2_full_f160w_residual")
    matched_f160w = build_residual_edges(matched_rows, C, "f160w", "ladder_residual_y_minus_thetaL", f"{dataset_label}_matched_f160w_residual")
    matched_nir_h = build_residual_edges(matched_rows, C, "h_mag_nir", "ladder_residual_y_minus_thetaL", f"{dataset_label}_matched_nir_h_residual")
    matched_h_minus_f = build_residual_edges(matched_rows, None, "f160w", "h_minus_f160w", f"{dataset_label}_matched_h_minus_f160w_by_f160w_edges")
    nir_only_raw_h = build_nir_only_raw_h_edges(nir_rows, f"{dataset_label}_nir_only_raw_h_edges")

    analyses = {
        "table2_full_f160w_residual": full_f160w,
        "matched_f160w_residual": matched_f160w,
        "matched_nir_h_residual": matched_nir_h,
        "matched_h_minus_f160w_by_f160w_edges": matched_h_minus_f,
        "nir_only_raw_h_edges": nir_only_raw_h,
    }

    aggregates = {name: aggregate_edges(result["edge_inventory"]) for name, result in analyses.items()}

    match_summary = {
        "dataset_label": dataset_label,
        "nir_row_count": len(nir_rows),
        "table2_row_count": len(table_residual_rows),
        "matched_row_count": len(matched_rows),
        "match_fraction_of_table2": len(matched_rows) / len(table_residual_rows) if table_residual_rows else None,
        "match_fraction_of_nir": len(matched_rows) / len(nir_rows) if nir_rows else None,
        "matched_host_count": len(set(r["host"] for r in matched_rows)),
        "unmatched_nir_count": len(matched["unmatched_nir_rows"]),
        "unmatched_table2_count": len(matched["unmatched_table2_rows"]),
        "period_abs_diff_summary": summarize_numeric([r["period_abs_diff"] for r in matched_rows]),
        "ra_abs_diff_summary": summarize_numeric([r["ra_abs_diff"] for r in matched_rows]),
        "dec_abs_diff_summary": summarize_numeric([r["dec_abs_diff"] for r in matched_rows]),
        "h_minus_f160w_summary": summarize_numeric([r["h_minus_f160w"] for r in matched_rows]),
        "abs_h_minus_f160w_summary": summarize_numeric([r["abs_h_minus_f160w"] for r in matched_rows]),
    }

    return {
        "dataset_label": dataset_label,
        "match": matched,
        "match_summary": match_summary,
        "analyses": analyses,
        "aggregates": aggregates,
    }


def decide(all_dataset_results):
    rows = []
    for result in all_dataset_results:
        label = result["dataset_label"]
        match = result["match_summary"]
        agg = result["aggregates"]

        full_delta = agg["table2_full_f160w_residual"].get("clean_minus_reference", {}).get("weighted_delta")
        matched_delta = agg["matched_f160w_residual"].get("clean_minus_reference", {}).get("weighted_delta")
        nir_h_delta = agg["matched_nir_h_residual"].get("clean_minus_reference", {}).get("weighted_delta")
        h_minus_f_delta = agg["matched_h_minus_f160w_by_f160w_edges"].get("clean_minus_reference", {}).get("weighted_delta")
        raw_h_delta = agg["nir_only_raw_h_edges"].get("clean_minus_reference", {}).get("weighted_delta")

        rows.append(
            {
                "dataset_label": label,
                "matched_row_count": match["matched_row_count"],
                "match_fraction_of_table2": match["match_fraction_of_table2"],
                "matched_host_count": match["matched_host_count"],
                "full_f160w_residual_delta": full_delta,
                "matched_f160w_residual_delta": matched_delta,
                "matched_nir_h_residual_delta": nir_h_delta,
                "h_minus_f160w_edge_delta": h_minus_f_delta,
                "nir_only_raw_h_regime_delta": raw_h_delta,
                "matched_delta_fraction_of_full": matched_delta / full_delta if full_delta not in (None, 0) and matched_delta is not None else None,
                "nir_h_delta_fraction_of_full": nir_h_delta / full_delta if full_delta not in (None, 0) and nir_h_delta is not None else None,
                "abs_h_minus_f_fraction_of_full": abs(h_minus_f_delta) / abs(full_delta) if full_delta not in (None, 0) and h_minus_f_delta is not None else None,
            }
        )

    best = sorted(rows, key=lambda r: (-(r["matched_row_count"] or 0), r["dataset_label"]))[0] if rows else None

    if not best:
        return {
            "final_status": "nir_matched_surface_falsification_not_ready",
            "readiness_score_0_to_10": 4,
            "next_wall": "No NIR dataset result was available.",
            "dataset_decisions": rows,
            "gates": [],
            "failed_gates": ["no_dataset_result"],
            "truth_boundary": CLAIMS_V2_2["truth_boundary"],
        }

    match_ok = (best["matched_row_count"] or 0) >= 1000 and (best["matched_host_count"] or 0) >= 10
    matched_reproduces_direction = (
        best["full_f160w_residual_delta"] is not None
        and best["matched_f160w_residual_delta"] is not None
        and best["full_f160w_residual_delta"] * best["matched_f160w_residual_delta"] > 0
    )
    h_sort_same_direction = (
        best["full_f160w_residual_delta"] is not None
        and best["matched_nir_h_residual_delta"] is not None
        and best["full_f160w_residual_delta"] * best["matched_nir_h_residual_delta"] > 0
    )
    mismatch_small = (
        best["abs_h_minus_f_fraction_of_full"] is not None
        and best["abs_h_minus_f_fraction_of_full"] < 0.25
    )
    mismatch_not_large = (
        best["abs_h_minus_f_fraction_of_full"] is not None
        and best["abs_h_minus_f_fraction_of_full"] < 0.75
    )

    gates = [
        {
            "gate": "G1_nir_table_parsed_and_matched",
            "passed": match_ok,
            "evidence": {
                "best_dataset": best["dataset_label"],
                "matched_row_count": best["matched_row_count"],
                "matched_host_count": best["matched_host_count"],
                "match_fraction_of_table2": best["match_fraction_of_table2"],
            },
        },
        {
            "gate": "G2_matched_subset_preserves_f160w_residual_direction",
            "passed": matched_reproduces_direction,
            "evidence": {
                "full_f160w_residual_delta": best["full_f160w_residual_delta"],
                "matched_f160w_residual_delta": best["matched_f160w_residual_delta"],
                "matched_delta_fraction_of_full": best["matched_delta_fraction_of_full"],
            },
        },
        {
            "gate": "G3_external_nir_h_sort_report_completed",
            "passed": best["matched_nir_h_residual_delta"] is not None,
            "evidence": {
                "matched_nir_h_residual_delta": best["matched_nir_h_residual_delta"],
                "same_direction_as_f160w": h_sort_same_direction,
                "nir_h_delta_fraction_of_full": best["nir_h_delta_fraction_of_full"],
            },
        },
        {
            "gate": "G4_h_minus_f160w_mismatch_not_large_enough_to_explain_locked_delta",
            "passed": mismatch_not_large,
            "evidence": {
                "h_minus_f160w_edge_delta": best["h_minus_f160w_edge_delta"],
                "abs_h_minus_f_fraction_of_full": best["abs_h_minus_f_fraction_of_full"],
                "strict_small_threshold_0_25_passed": mismatch_small,
                "large_threshold_0_75_passed": mismatch_not_large,
            },
        },
        {
            "gate": "G5_nir_raw_h_surface_reported_not_used_as_residual_validation",
            "passed": best["nir_only_raw_h_regime_delta"] is not None,
            "evidence": {
                "nir_only_raw_h_regime_delta": best["nir_only_raw_h_regime_delta"],
                "used_as_residual_validation": False,
            },
        },
    ]

    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed:
        if mismatch_small:
            final_status = "adjacent_nir_surface_does_not_explain_locked_residual_pattern"
            readiness = 9
            next_wall = (
                "The adjacent NIR matched surface did not explain away the locked residual edge pattern. "
                "Next step can be a bounded technical note plus one more independent external dataset search."
            )
        else:
            final_status = "adjacent_nir_surface_does_not_fully_explain_pattern_but_mismatch_caution"
            readiness = 8
            next_wall = (
                "The adjacent NIR surface did not fully explain the locked residual pattern, but H-F160W mismatch was not tiny. "
                "Report the caution before further escalation."
            )
    elif "G4_h_minus_f160w_mismatch_not_large_enough_to_explain_locked_delta" in failed:
        final_status = "adjacent_nir_surface_mismatch_can_explain_too_much_caution"
        readiness = 6
        next_wall = (
            "H-F160W mismatch across frozen edges is large enough to be a fragility source. Do not escalate without explaining this."
        )
    elif len(failed) <= 2 and "G1_nir_table_parsed_and_matched" not in failed:
        final_status = "adjacent_nir_falsification_completed_with_cautions"
        readiness = 7
        next_wall = (
            "The NIR falsification lane mostly completed, but one or more interpretation gates failed. Report the boundary."
        )
    else:
        final_status = "adjacent_nir_falsification_not_ready_for_interpretation"
        readiness = 5
        next_wall = (
            "NIR matching or core reporting failed. Do not interpret the external-lane result."
        )

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "best_dataset": best,
        "dataset_decisions": rows,
        "gates": gates,
        "failed_gates": failed,
        "truth_boundary": CLAIMS_V2_2["truth_boundary"],
    }


def source_sniffs(readme_text, table2_readme_text):
    combined = (readme_text or "") + "\n\n" + (table2_readme_text or "")
    lower = combined.lower()
    patterns = [
        "lstsq_results.txt",
        "y, data vector",
        "l, equation matrix",
        "c, covariance matrix",
        "equation 6",
        "not to use this table",
        "already packaged",
        "covariance",
        "initial slope term of -3.285",
        "R22_orig19_NIR",
    ]
    rows = []
    snippets = []
    for pattern in patterns:
        pat = pattern.lower()
        count = lower.count(pat)
        rows.append({"pattern": pattern, "count": count})
        if count:
            pos = lower.find(pat)
            start = max(0, pos - 350)
            end = min(len(combined), pos + 900)
            snippets.append({"pattern": pattern, "snippet": combined[start:end].replace("\n", " ")[:1300]})
    return {"pattern_counts": rows, "snippets": snippets}


def holographic_surface_ledger(decision):
    return {
        "observable_surface": {
            "name": "Adjacent NIR H/F160W-like matched surface compared against locked Table2 residual edge surface",
            "table2_path": TABLE2_PATH,
            "theta_path": LSTSQ_PATH,
            "y_path": Y_FITS_PATH,
            "l_path": L_FITS_PATH,
            "c_path": C_FITS_PATH,
            "nir_paths": [t["repo_path"] for t in NIR_TARGETS],
        },
        "hidden_depth_sought": {
            "name": "Whether adjacent NIR raw surface explains locked residual pattern",
            "allowed_claim": "The NIR matched surface can falsify or fail to falsify a trivial raw-surface explanation.",
            "not_allowed_claim": "Do not treat NIR raw H as a residual layer or as proof of H0 correction, SH0ES error, TAIRID proof, or new physics.",
        },
        "boundary_that_forms_surface": {
            "row_boundary": "first 3130 y entries attached to Table2 rows; NIR rows matched by host and Cepheid ID",
            "locked_edge_boundary": "within-host 5 percent bright/faint F160W edges",
            "external_surface_boundary": "adjacent R22 NIR H magnitude table, not a residual table",
            "regime_boundary": "frozen regimes; M31 quarantine not transferred",
            "method_boundary": "no variable replacement, no tuning, no post-hoc regime editing",
        },
        "what_can_be_reconstructed_now": [
            "NIR-to-Table2 row match coverage",
            "Matched F160W residual edge contrast",
            "Matched NIR-H-sorted residual edge contrast",
            "H-minus-F160W mismatch across frozen edges",
            "NIR-only raw H edge surfaces",
        ],
        "what_cannot_be_reconstructed_now": [
            "A new H0 solution",
            "New physics",
            "A causal explanation",
            "A tuned likelihood model",
            "External residual validation from raw NIR H alone",
        ],
        "surface_noise_definition": [
            "Any use of raw NIR H as residuals",
            "Any claim that NIR-only raw H validates the locked residual pattern",
            "Any M31 correction transfer",
            "Any post-hoc matching or regime edits",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v2.2 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v2.2 did")
    lines.append("")
    lines.append("- Rebuilt the same public SH0ES ladder residual surface.")
    lines.append("- Parsed the adjacent R22 NIR Cepheid tables.")
    lines.append("- Matched NIR rows to Table2 rows by host and Cepheid ID with period/coordinate diagnostics.")
    lines.append("- Re-ran the locked F160W residual edge on the matched subset.")
    lines.append("- Re-ran the matched residual edge sorted by external NIR H magnitude.")
    lines.append("- Tested whether H - F160W mismatch across frozen edges could explain the locked residual pattern.")
    lines.append("- Reported NIR-only raw H edge surfaces without treating them as residual validation.")
    lines.append("- Did not tune, refit, or claim H0 correction.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v2.2 is adjacent-lane falsification only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not prove SH0ES is wrong.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append(
            "v2.3 should either write a bounded technical note from v1.9-v2.2 or perform one more independent public-data search for a true external residual/outcome layer."
        )
    elif decision["readiness_score_0_to_10"] >= 6:
        lines.append(
            "v2.3 should document the NIR fragility/caution and avoid any H0 or physics escalation."
        )
    else:
        lines.append(
            "v2.3 should stop escalation and write the boundary result honestly."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES NIR Matched Surface Falsification v2.2 starting.")
    print("Boundary: adjacent-lane falsification only; no tuning, no H0 claim, no new physics claim.")

    write_json(OUTDIR / "claims_v2_2.json", CLAIMS_V2_2)
    write_json(OUTDIR / "frozen_rule_v2_2.json", FROZEN_RULE)

    try:
        download_ledger = []
        text_files = {}

        for repo_path in [TABLE2_PATH, TABLE2_README_PATH, README_PATH, LSTSQ_PATH]:
            fetched = fetch_text_for_path(repo_path)
            download_ledger.append(
                {
                    "kind": "text",
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
                text_files[repo_path] = fetched["text"]

        nir_parsed_results = []
        for target in NIR_TARGETS:
            fetched = fetch_text_for_path(target["repo_path"])
            download_ledger.append(
                {
                    "kind": "nir_text",
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
            else:
                parsed = {
                    "label": target["label"],
                    "repo_path": target["repo_path"],
                    "row_count": 0,
                    "parse_error_count": 0,
                    "rows": [],
                    "errors": [{"reason": "download_failed"}],
                }
            nir_parsed_results.append(parsed)
            write_csv(OUTDIR / f"{target['label']}_nir_rows_v2_2.csv", parsed["rows"])
            write_json(OUTDIR / f"{target['label']}_nir_parse_errors_v2_2.json", parsed["errors"])

        fits_arrays = {}
        for repo_path in [Y_FITS_PATH, L_FITS_PATH, C_FITS_PATH]:
            fetched = fetch_bytes_for_path(repo_path, prefer_media=True)
            download_ledger.append(
                {
                    "kind": "fits",
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
                local_path.write_bytes(fetched["data"])
                fits_arrays[repo_path] = read_primary_fits_array(fetched["data"])

        write_csv(OUTDIR / "download_ledger_v2_2.csv", download_ledger)

        table2 = parse_table2_tex(text_files.get(TABLE2_PATH, ""))
        table_rows = table2["rows"]
        write_csv(OUTDIR / "table2_parsed_rows_v2_2.csv", table_rows)
        write_json(OUTDIR / "table2_parse_errors_v2_2.json", table2["errors"])

        theta_parse = parse_lstsq_results(text_files.get(LSTSQ_PATH, ""))
        theta_rows = theta_parse["rows"]
        theta = np.asarray([r["theta_value"] for r in theta_rows], dtype=float)
        write_csv(OUTDIR / "theta_lstsq_vector_v2_2.csv", theta_rows)
        write_json(OUTDIR / "theta_lstsq_parse_errors_v2_2.json", theta_parse["errors"])

        y = np.asarray(fits_arrays.get(Y_FITS_PATH, np.asarray([])), dtype=float)
        L = np.asarray(fits_arrays.get(L_FITS_PATH, np.asarray([[]])), dtype=float)
        C = np.asarray(fits_arrays.get(C_FITS_PATH, np.asarray([[]])), dtype=float)

        predicted_y, residual, orientation = compute_ladder_residual(theta, L, y)
        if predicted_y is None:
            predicted_y = np.asarray([])
            residual = np.asarray([])

        covariance_diag, cov_info = covariance_diag_info(C)
        write_json(OUTDIR / "covariance_diag_summary_v2_2.json", cov_info)

        table_residual_rows = []
        if len(residual) >= len(table_rows) and len(table_rows) > 0:
            table_residual_rows = attach_table2_residuals(table_rows, y, predicted_y, residual, covariance_diag)
        write_csv(OUTDIR / "table2_ladder_residual_rows_v2_2.csv", table_residual_rows)

        dataset_results = []
        all_match_summaries = []

        for parsed in nir_parsed_results:
            result = make_dataset_analysis(parsed["label"], table_residual_rows, parsed["rows"], C)
            dataset_results.append(result)
            all_match_summaries.append(result["match_summary"])

            write_csv(OUTDIR / f"{parsed['label']}_matched_rows_v2_2.csv", result["match"]["matched_rows"])
            write_csv(OUTDIR / f"{parsed['label']}_unmatched_nir_rows_v2_2.csv", result["match"]["unmatched_nir_rows"])
            write_csv(OUTDIR / f"{parsed['label']}_unmatched_table2_rows_v2_2.csv", result["match"]["unmatched_table2_rows"])
            write_json(OUTDIR / f"{parsed['label']}_match_summary_v2_2.json", result["match_summary"])

            for analysis_name, analysis_result in result["analyses"].items():
                write_csv(OUTDIR / f"{parsed['label']}_{analysis_name}_edge_rows_v2_2.csv", analysis_result["edge_rows"])
                write_csv(OUTDIR / f"{parsed['label']}_{analysis_name}_edge_inventory_v2_2.csv", analysis_result["edge_inventory"])
                write_json(OUTDIR / f"{parsed['label']}_{analysis_name}_aggregate_v2_2.json", result["aggregates"][analysis_name])

        write_csv(OUTDIR / "all_dataset_match_summaries_v2_2.csv", all_match_summaries)

        decision = decide(dataset_results)
        write_json(OUTDIR / "decision_v2_2.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v2_2.json", ledger)

        source_info = source_sniffs(text_files.get(README_PATH, ""), text_files.get(TABLE2_README_PATH, ""))
        write_json(OUTDIR / "source_sniffs_v2_2.json", source_info)
        write_csv(OUTDIR / "source_pattern_counts_v2_2.csv", source_info["pattern_counts"])
        write_csv(OUTDIR / "source_snippets_v2_2.csv", source_info["snippets"])

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v2_2.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v2_2.txt").write_text(handoff, encoding="utf-8")

        residual_summary = {
            "theta_count": int(len(theta)),
            "y_shape": list(y.shape),
            "L_shape": list(L.shape),
            "C_shape": list(C.shape),
            "orientation": orientation,
            "predicted_y_count": int(len(predicted_y)),
            "residual_count": int(len(residual)),
            "table2_row_count": int(len(table_rows)),
            "all_residual_summary": summarize_numeric(residual.tolist() if len(residual) else []),
            "table2_residual_summary": summarize_numeric([r["ladder_residual_y_minus_thetaL"] for r in table_residual_rows]),
        }
        write_json(OUTDIR / "residual_summary_v2_2.json", residual_summary)

        compact_dataset_outputs = []
        for result in dataset_results:
            compact_dataset_outputs.append(
                {
                    "dataset_label": result["dataset_label"],
                    "match_summary": result["match_summary"],
                    "aggregates": result["aggregates"],
                }
            )

        summary = {
            "test_name": "TAIRID SH0ES NIR Matched Surface Falsification v2.2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Adjacent NIR matched-surface falsification only. No tuning, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "download_ledger": download_ledger,
            "residual_summary": residual_summary,
            "covariance_diag_summary": cov_info,
            "nir_parse_summaries": [
                {
                    "label": p["label"],
                    "repo_path": p["repo_path"],
                    "row_count": p["row_count"],
                    "parse_error_count": p.get("parse_error_count", len(p.get("errors", []))),
                }
                for p in nir_parsed_results
            ],
            "dataset_outputs": compact_dataset_outputs,
            "decision": decision,
            "claims_v2_2": CLAIMS_V2_2,
            "frozen_rule": FROZEN_RULE,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_nir_matched_surface_falsification_v2_2_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_nir_matched_surface_falsification_v2_2_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v2_2.csv"),
                "table2_residual_rows_csv": str(OUTDIR / "table2_ladder_residual_rows_v2_2.csv"),
                "match_summaries_csv": str(OUTDIR / "all_dataset_match_summaries_v2_2.csv"),
                "decision_json": str(OUTDIR / "decision_v2_2.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v2_2.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v2_2.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v2_2.txt"),
            },
            "interpretation": {
                "what_success_means": "The adjacent NIR matched surface did not explain away the locked residual edge pattern.",
                "what_success_does_not_mean": "This does not prove TAIRID, H0 correction, new physics, or SH0ES error.",
                "what_caution_means": "If H-F160W mismatch is large, report it as a fragility source.",
                "truth_boundary": CLAIMS_V2_2["truth_boundary"],
            },
        }

        write_json(OUTDIR / "shoes_nir_matched_surface_falsification_v2_2_summary.json", summary)

        with open(OUTDIR / "shoes_nir_matched_surface_falsification_v2_2_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES NIR Matched Surface Falsification v2.2\n\n")
            f.write("Boundary: adjacent-lane falsification only. No tuning. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Best dataset decision:\n")
            f.write(json.dumps(decision.get("best_dataset"), indent=2, default=json_default) + "\n\n")
            f.write("All dataset decisions:\n")
            f.write(json.dumps(decision.get("dataset_decisions"), indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not prove SH0ES is wrong.\n")
            f.write("- This does not tune or replace the frozen rule.\n")
            f.write("- Raw NIR H is not treated as a residual layer.\n")

        print("TAIRID SH0ES NIR Matched Surface Falsification v2.2 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES NIR Matched Surface Falsification v2.2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_nir_matched_surface_falsification_v2_2_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V2_2["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_nir_matched_surface_falsification_v2_2_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
