#!/usr/bin/env python3
"""
TAIRID SH0ES Frozen Replay Negative Controls v2.1

Why this test exists:
v2.0 showed that the locked v1.9 Table2 F160W residual edge replay survived
jackknife, covariance-aware, and regime-label permutation stress tests.

v2.1 does NOT change the frozen rule. It asks a stricter falsification question:

    Is the locked F160W edge result specific to the frozen F160W surface, or do
    ordinary Table2 columns produce the same clean-minus-reference contrast when
    the same within-host 5 percent edge machinery is applied?

This is a negative-control test. It reruns the same residual construction and
same frozen regime definitions, then compares the locked F160W result against
predeclared control sort variables:

    period
    log_period
    V-I
    sigma_V-I
    sigma_F160W
    metal_minus_8_69
    ra
    dec

Controls are report-only. They are not replacements for the frozen rule.

This test does NOT validate TAIRID.
This test does NOT tune the frozen v1.0/v1.9 rule.
This test does NOT search for a better variable.
This test does NOT add new regimes.
This test does NOT claim H0 correction or new physics.
This test does NOT prove SH0ES is wrong.

Truth boundary:
If F160W remains stronger than the controls, that supports specificity of the
locked residual surface pattern. If a control matches or exceeds it, that is a
fragility result that must be reported honestly. Either outcome is not proof of
TAIRID or new physics.
"""

import csv
import io
import json
import math
import re
import traceback
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


OUTDIR = Path("tairid_shoes_frozen_replay_negative_controls_v2_1_outputs")
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

EDGE_PERCENTILE = 5.0
MIN_HOST_ROWS_FOR_EDGE = 20

LOW_ALPHA_REFERENCE_HOSTS = {"LMC", "SMC", "N4536"}
SIGN_BREAK_QUARANTINE_HOSTS = {"M31"}

LOCKED_VARIABLE = "f160w"
CONTROL_VARIABLES = [
    "period",
    "log_period",
    "v_i",
    "sigma_v_i",
    "sigma_f160w",
    "metal_minus_8_69",
    "ra",
    "dec",
]

FROZEN_RULE = {
    "source_status": "v2.1 negative-control specificity test of locked v1.9/v2.0 frozen replay",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "locked_variable": LOCKED_VARIABLE,
    "negative_control_variables": CONTROL_VARIABLES,
    "edge_rule": "within-host high 5% sort-variable edge minus within-host low 5% sort-variable edge",
    "locked_interpretation": "Only f160w is the frozen rule. Control variables are falsification/report-only surfaces.",
    "edge_percentile": EDGE_PERCENTILE,
    "minimum_host_rows_for_edge": MIN_HOST_ROWS_FOR_EDGE,
    "low_alpha_reference_hosts": sorted(LOW_ALPHA_REFERENCE_HOSTS),
    "sign_break_quarantine_hosts": sorted(SIGN_BREAK_QUARANTINE_HOSTS),
    "clean_high_alpha_rule": "all active Table2 hosts except LMC, SMC, N4536, and M31",
    "hard_boundary": [
        "Do not tune the F160W edge percentile in v2.1.",
        "Do not replace F160W with a better-looking variable.",
        "Do not add or remove frozen regimes in v2.1.",
        "Do not transfer M31 correction; M31 is quarantine/report-only.",
        "Do not claim H0 correction or new physics.",
        "Do not claim TAIRID is proven.",
    ],
}

CLAIMS_V2_1 = {
    "battery_name": "TAIRID SH0ES Frozen Replay Negative Controls v2.1",
    "scope": "Specificity/falsification test comparing locked F160W edge replay to predeclared Table2 control variables",
    "primary_question": (
        "Does the locked F160W residual edge contrast remain stronger or more specific than ordinary Table2 control-variable edge contrasts?"
    ),
    "truth_boundary": (
        "This is a negative-control specificity test only. It does not validate TAIRID, H0 correction, or new physics."
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
        headers={"User-Agent": "TAIRID-v2.1-shoes-frozen-negative-controls"},
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


def edge_covariance_diagnostic(low_side, high_side, C):
    if C is None or C.ndim != 2 or C.shape[0] != C.shape[1]:
        return {"covariance_edge_variance_full": None, "covariance_edge_sigma_full": None, "covariance_edge_z_full": None}

    low_idx = [r["y_vector_index_0_based"] for r in low_side]
    high_idx = [r["y_vector_index_0_based"] for r in high_side]
    idxs = high_idx + low_idx
    k_high = len(high_idx)
    k_low = len(low_idx)
    weights = np.asarray(([1.0 / k_high] * k_high) + ([-1.0 / k_low] * k_low), dtype=float)
    delta = float(
        np.mean([r["ladder_residual_y_minus_thetaL"] for r in high_side])
        - np.mean([r["ladder_residual_y_minus_thetaL"] for r in low_side])
    )
    sub = C[np.ix_(idxs, idxs)]
    var_full = float(weights @ sub @ weights)
    sigma_full = math.sqrt(var_full) if var_full > 0 else None
    return {
        "covariance_edge_variance_full": var_full,
        "covariance_edge_sigma_full": sigma_full,
        "covariance_edge_z_full": float(delta / sigma_full) if sigma_full else None,
    }


def build_edges_for_variable(residual_rows, C, sort_variable):
    by_host = defaultdict(list)
    for row in residual_rows:
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
                    "sort_variable": sort_variable,
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
                    "sort_variable": sort_variable,
                    "host": host,
                    "edge_side": side_name,
                    "y_vector_index_0_based": r["y_vector_index_0_based"],
                    "table2_row_index_0_based": r["table2_row_index_0_based"],
                    "sort_value": r[sort_variable],
                    "f160w": r["f160w"],
                    "ladder_residual_y_minus_thetaL": r["ladder_residual_y_minus_thetaL"],
                    "standardized_residual": r["standardized_residual"],
                    "frozen_regime_role": role,
                }
                edge_rows.append(out)

        low_resid = np.asarray([r["ladder_residual_y_minus_thetaL"] for r in low_side], dtype=float)
        high_resid = np.asarray([r["ladder_residual_y_minus_thetaL"] for r in high_side], dtype=float)
        low_value = np.asarray([r[sort_variable] for r in low_side], dtype=float)
        high_value = np.asarray([r[sort_variable] for r in high_side], dtype=float)
        low_z = np.asarray([r["standardized_residual"] for r in low_side if r["standardized_residual"] is not None], dtype=float)
        high_z = np.asarray([r["standardized_residual"] for r in high_side if r["standardized_residual"] is not None], dtype=float)

        cov = edge_covariance_diagnostic(low_side, high_side, C)

        inventory.append(
            {
                "sort_variable": sort_variable,
                "host": host,
                "row_count": n,
                "edge_status": "edge_surface_built",
                "edge_count_each_side": k,
                "low_edge_mean_sort_value": float(np.mean(low_value)),
                "high_edge_mean_sort_value": float(np.mean(high_value)),
                "high_minus_low_mean_sort_value": float(np.mean(high_value) - np.mean(low_value)),
                "low_edge_mean_residual": float(np.mean(low_resid)),
                "high_edge_mean_residual": float(np.mean(high_resid)),
                "high_minus_low_mean_residual": float(np.mean(high_resid) - np.mean(low_resid)),
                "low_edge_median_residual": float(np.median(low_resid)),
                "high_edge_median_residual": float(np.median(high_resid)),
                "high_minus_low_median_residual": float(np.median(high_resid) - np.median(low_resid)),
                "low_edge_mean_standardized_residual": float(np.mean(low_z)) if len(low_z) else None,
                "high_edge_mean_standardized_residual": float(np.mean(high_z)) if len(high_z) else None,
                "high_minus_low_mean_standardized_residual": float(np.mean(high_z) - np.mean(low_z)) if len(low_z) and len(high_z) else None,
                "frozen_regime_role": role,
                **cov,
            }
        )

    return {"edge_rows": edge_rows, "edge_inventory": inventory}


def aggregate_regimes(edge_inventory):
    roles = defaultdict(list)
    for row in edge_inventory:
        if row.get("edge_status") == "edge_surface_built":
            roles[row["frozen_regime_role"]].append(row)

    out = {}
    for role, rows in roles.items():
        values = np.asarray([r["high_minus_low_mean_residual"] for r in rows], dtype=float)
        weights = np.asarray([r["edge_count_each_side"] for r in rows], dtype=float)
        z_full = np.asarray([r["covariance_edge_z_full"] for r in rows if r.get("covariance_edge_z_full") is not None], dtype=float)
        out[role] = {
            "host_count": int(len(rows)),
            "hosts": [r["host"] for r in rows],
            "unweighted_mean_high_minus_low_residual": float(np.mean(values)) if len(values) else None,
            "unweighted_median_high_minus_low_residual": float(np.median(values)) if len(values) else None,
            "weighted_mean_high_minus_low_residual": float(np.average(values, weights=weights)) if len(values) and np.sum(weights) > 0 else None,
            "std_high_minus_low_residual": float(np.std(values)) if len(values) else None,
            "mean_covariance_edge_z_full": float(np.mean(z_full)) if len(z_full) else None,
        }

    clean = out.get("clean_high_alpha_candidate", {})
    ref = out.get("low_alpha_reference", {})
    out["clean_minus_reference"] = {
        "unweighted_mean_delta": None,
        "weighted_mean_delta": None,
        "truth_boundary": "negative_control_contrast_only_not_H0_correction",
    }
    if clean and ref:
        if clean.get("unweighted_mean_high_minus_low_residual") is not None and ref.get("unweighted_mean_high_minus_low_residual") is not None:
            out["clean_minus_reference"]["unweighted_mean_delta"] = clean["unweighted_mean_high_minus_low_residual"] - ref["unweighted_mean_high_minus_low_residual"]
        if clean.get("weighted_mean_high_minus_low_residual") is not None and ref.get("weighted_mean_high_minus_low_residual") is not None:
            out["clean_minus_reference"]["weighted_mean_delta"] = clean["weighted_mean_high_minus_low_residual"] - ref["weighted_mean_high_minus_low_residual"]
    return out


def selected_index_sets(edge_rows):
    out = defaultdict(set)
    for row in edge_rows:
        key = (row["host"], row["edge_side"])
        out[key].add(row["table2_row_index_0_based"])
    return out


def edge_overlap_rows(locked_edge_rows, all_variable_edge_results):
    locked_sets = selected_index_sets(locked_edge_rows)
    rows = []

    for variable, result in all_variable_edge_results.items():
        if variable == LOCKED_VARIABLE:
            continue
        control_sets = selected_index_sets(result["edge_rows"])
        host_scores = []
        for key, locked_set in locked_sets.items():
            control_set = control_sets.get(key, set())
            if not locked_set or not control_set:
                continue
            intersection = len(locked_set & control_set)
            union = len(locked_set | control_set)
            host_scores.append(intersection / union if union else 0.0)

        rows.append(
            {
                "control_variable": variable,
                "host_edge_overlap_count": len(host_scores),
                "mean_jaccard_overlap_with_f160w_edges": float(np.mean(host_scores)) if host_scores else None,
                "median_jaccard_overlap_with_f160w_edges": float(np.median(host_scores)) if host_scores else None,
                "max_jaccard_overlap_with_f160w_edges": float(np.max(host_scores)) if host_scores else None,
            }
        )
    return rows


def control_variable_summary(all_variable_edge_results):
    rows = []
    aggregates = {}

    for variable, result in all_variable_edge_results.items():
        agg = aggregate_regimes(result["edge_inventory"])
        aggregates[variable] = agg
        clean = agg.get("clean_high_alpha_candidate", {})
        ref = agg.get("low_alpha_reference", {})
        quarantine = agg.get("sign_break_quarantine", {})
        contrast = agg.get("clean_minus_reference", {})
        delta = contrast.get("weighted_mean_delta")

        rows.append(
            {
                "sort_variable": variable,
                "is_locked_variable": variable == LOCKED_VARIABLE,
                "clean_edge_host_count": clean.get("host_count", 0),
                "reference_edge_host_count": ref.get("host_count", 0),
                "m31_quarantine_reported": bool(quarantine.get("host_count", 0)),
                "clean_weighted_edge_residual": clean.get("weighted_mean_high_minus_low_residual"),
                "reference_weighted_edge_residual": ref.get("weighted_mean_high_minus_low_residual"),
                "clean_minus_reference_weighted_delta": delta,
                "absolute_clean_minus_reference_weighted_delta": abs(delta) if delta is not None else None,
                "clean_unweighted_edge_residual": clean.get("unweighted_mean_high_minus_low_residual"),
                "reference_unweighted_edge_residual": ref.get("unweighted_mean_high_minus_low_residual"),
                "clean_minus_reference_unweighted_delta": contrast.get("unweighted_mean_delta"),
                "clean_mean_covariance_edge_z_full": clean.get("mean_covariance_edge_z_full"),
                "reference_mean_covariance_edge_z_full": ref.get("mean_covariance_edge_z_full"),
            }
        )

    rows = sorted(rows, key=lambda r: (-(r["absolute_clean_minus_reference_weighted_delta"] or -999), r["sort_variable"]))
    for rank, row in enumerate(rows, start=1):
        row["absolute_delta_rank_1_is_largest"] = rank
    return rows, aggregates


def decide(summary_rows, overlap_rows, residual_ok):
    locked = next((r for r in summary_rows if r["sort_variable"] == LOCKED_VARIABLE), None)
    controls = [r for r in summary_rows if r["sort_variable"] != LOCKED_VARIABLE]
    controls_with_delta = [r for r in controls if r["absolute_clean_minus_reference_weighted_delta"] is not None]
    locked_abs = locked.get("absolute_clean_minus_reference_weighted_delta") if locked else None
    max_control_abs = max([r["absolute_clean_minus_reference_weighted_delta"] for r in controls_with_delta], default=None)
    stronger_controls = [
        r["sort_variable"] for r in controls_with_delta
        if locked_abs is not None and r["absolute_clean_minus_reference_weighted_delta"] >= locked_abs
    ]
    near_controls = [
        r["sort_variable"] for r in controls_with_delta
        if locked_abs is not None and r["absolute_clean_minus_reference_weighted_delta"] >= 0.75 * locked_abs
    ]

    overlap_values = [
        r["mean_jaccard_overlap_with_f160w_edges"]
        for r in overlap_rows
        if r["mean_jaccard_overlap_with_f160w_edges"] is not None
    ]
    max_overlap = max(overlap_values, default=None)

    gates = [
        {
            "gate": "G1_residual_surface_rebuilt",
            "passed": bool(residual_ok),
            "evidence": {"residual_surface_rebuilt": bool(residual_ok)},
        },
        {
            "gate": "G2_locked_f160w_result_rebuilt",
            "passed": locked is not None and locked.get("clean_minus_reference_weighted_delta") is not None,
            "evidence": {
                "locked_delta": locked.get("clean_minus_reference_weighted_delta") if locked else None,
                "locked_abs_delta": locked_abs,
            },
        },
        {
            "gate": "G3_control_variables_completed",
            "passed": len(controls_with_delta) >= len(CONTROL_VARIABLES),
            "evidence": {
                "completed_control_count": len(controls_with_delta),
                "expected_control_count": len(CONTROL_VARIABLES),
            },
        },
        {
            "gate": "G4_f160w_abs_delta_exceeds_all_controls",
            "passed": locked_abs is not None and max_control_abs is not None and locked_abs > max_control_abs,
            "evidence": {
                "locked_abs_delta": locked_abs,
                "max_control_abs_delta": max_control_abs,
                "controls_matching_or_exceeding_locked": stronger_controls,
            },
        },
        {
            "gate": "G5_no_control_within_75_percent_of_locked",
            "passed": locked_abs is not None and len(near_controls) == 0,
            "evidence": {
                "controls_at_or_above_75_percent_of_locked": near_controls,
            },
        },
        {
            "gate": "G6_edge_overlap_report_completed",
            "passed": len(overlap_rows) >= len(CONTROL_VARIABLES),
            "evidence": {
                "overlap_row_count": len(overlap_rows),
                "max_mean_jaccard_overlap": max_overlap,
            },
        },
    ]

    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed:
        final_status = "f160w_locked_surface_specific_against_negative_controls"
        readiness = 9
        next_wall = (
            "F160W remained stronger than all predeclared control variables. Next step should be an external-lane falsification or a formal write-up, not tuning."
        )
    elif "G4_f160w_abs_delta_exceeds_all_controls" in failed or "G5_no_control_within_75_percent_of_locked" in failed:
        final_status = "f160w_surface_detected_but_negative_controls_show_specificity_caution"
        readiness = 7
        next_wall = (
            "The locked F160W surface remains measurable, but one or more controls approached or exceeded it. This is a fragility boundary, not a failure to hide."
        )
    elif len(failed) <= 2 and "G1_residual_surface_rebuilt" not in failed:
        final_status = "negative_control_test_completed_with_reporting_cautions"
        readiness = 7
        next_wall = (
            "Negative controls mostly completed, but reporting gates need review. Do not escalate until the caution is explained."
        )
    else:
        final_status = "negative_control_test_not_ready_for_interpretation"
        readiness = 5
        next_wall = (
            "Core rebuild or control completion failed. Do not interpret specificity."
        )

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "gates": gates,
        "failed_gates": failed,
        "specificity_core": {
            "locked_variable": LOCKED_VARIABLE,
            "locked_abs_delta": locked_abs,
            "max_control_abs_delta": max_control_abs,
            "controls_matching_or_exceeding_locked": stronger_controls,
            "controls_at_or_above_75_percent_of_locked": near_controls,
            "locked_rank_1_is_largest": locked.get("absolute_delta_rank_1_is_largest") if locked else None,
            "control_count": len(controls_with_delta),
            "max_mean_jaccard_overlap_with_f160w_edges": max_overlap,
        },
        "truth_boundary": CLAIMS_V2_1["truth_boundary"],
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
            "name": "Locked F160W residual edge surface compared against Table2 negative-control surfaces",
            "table2_path": TABLE2_PATH,
            "theta_path": LSTSQ_PATH,
            "y_path": Y_FITS_PATH,
            "l_path": L_FITS_PATH,
            "c_path": C_FITS_PATH,
        },
        "hidden_depth_sought": {
            "name": "Specificity of frozen residual surface pattern",
            "allowed_claim": "The locked result can be compared against predeclared control variables without tuning.",
            "not_allowed_claim": "Do not treat control comparison as proof of H0 correction, SH0ES error, TAIRID proof, or new physics.",
        },
        "boundary_that_forms_surface": {
            "row_boundary": "first 3130 y entries attached to Table2 rows",
            "locked_edge_boundary": "within-host 5 percent bright/faint F160W edges",
            "control_boundary": "same edge machinery applied to predeclared non-F160W Table2 columns",
            "regime_boundary": "frozen v1.0/v1.9 regimes; M31 quarantine not transferred",
            "method_boundary": "no variable replacement, no tuning, no post-hoc control promotion",
        },
        "what_can_be_reconstructed_now": [
            "Locked F160W contrast",
            "Predeclared control-variable contrasts",
            "Rank of F160W against controls",
            "Edge-row overlap between F160W and control surfaces",
            "Specificity or fragility status",
        ],
        "what_cannot_be_reconstructed_now": [
            "A new H0 solution",
            "New physics",
            "A causal explanation",
            "A tuned likelihood model",
            "Generalization beyond Table2 without a separate replay",
        ],
        "surface_noise_definition": [
            "Any control chosen after seeing results",
            "Any replacement of F160W by a stronger control",
            "Any regime editing",
            "Any M31 correction transfer",
            "Any H0 interpretation from a control comparison",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v2.1 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v2.1 did")
    lines.append("")
    lines.append("- Rebuilt the same public SH0ES ladder residual surface.")
    lines.append("- Re-ran the locked F160W within-host 5 percent edge rule.")
    lines.append("- Kept regimes fixed: clean high-alpha, LMC+SMC+N4536 reference, M31 quarantine.")
    lines.append("- Ran predeclared negative-control edge surfaces: period, log_period, V-I, sigma_V-I, sigma_F160W, metallicity, RA, and DEC.")
    lines.append("- Ranked F160W against the controls by absolute clean-minus-reference delta.")
    lines.append("- Reported overlap between control edge rows and locked F160W edge rows.")
    lines.append("- Did not promote any control variable or change the frozen rule.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v2.1 is a negative-control specificity test only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not prove SH0ES is wrong.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 9:
        lines.append(
            "v2.2 should attempt an external falsification lane or write the result as a bounded technical note before any further model work."
        )
    elif decision["readiness_score_0_to_10"] >= 7:
        lines.append(
            "v2.2 should document the specificity caution and run an external falsification lane rather than forcing interpretation."
        )
    else:
        lines.append(
            "v2.2 should stop escalation and write the negative-control boundary result honestly."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Frozen Replay Negative Controls v2.1 starting.")
    print("Boundary: negative-control specificity only; no tuning, no H0 claim, no new physics claim.")

    write_json(OUTDIR / "claims_v2_1.json", CLAIMS_V2_1)
    write_json(OUTDIR / "frozen_rule_v2_1.json", FROZEN_RULE)

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

        write_csv(OUTDIR / "download_ledger_v2_1.csv", download_ledger)

        table2 = parse_table2_tex(text_files.get(TABLE2_PATH, ""))
        table_rows = table2["rows"]
        write_csv(OUTDIR / "table2_parsed_rows_v2_1.csv", table_rows)
        write_json(OUTDIR / "table2_parse_errors_v2_1.json", table2["errors"])

        theta_parse = parse_lstsq_results(text_files.get(LSTSQ_PATH, ""))
        theta_rows = theta_parse["rows"]
        theta = np.asarray([r["theta_value"] for r in theta_rows], dtype=float)
        write_csv(OUTDIR / "theta_lstsq_vector_v2_1.csv", theta_rows)
        write_json(OUTDIR / "theta_lstsq_parse_errors_v2_1.json", theta_parse["errors"])

        y = np.asarray(fits_arrays.get(Y_FITS_PATH, np.asarray([])), dtype=float)
        L = np.asarray(fits_arrays.get(L_FITS_PATH, np.asarray([[]])), dtype=float)
        C = np.asarray(fits_arrays.get(C_FITS_PATH, np.asarray([[]])), dtype=float)

        predicted_y, residual, orientation = compute_ladder_residual(theta, L, y)
        if predicted_y is None:
            predicted_y = np.asarray([])
            residual = np.asarray([])

        covariance_diag, cov_info = covariance_diag_info(C)
        write_json(OUTDIR / "covariance_diag_summary_v2_1.json", cov_info)

        residual_rows = []
        if len(residual) >= len(table_rows) and len(table_rows) > 0:
            residual_rows = attach_table2_residuals(table_rows, y, predicted_y, residual, covariance_diag)

        write_csv(OUTDIR / "table2_ladder_residual_rows_v2_1.csv", residual_rows)

        variables = [LOCKED_VARIABLE] + CONTROL_VARIABLES
        all_variable_edge_results = {}
        all_edge_rows = []
        all_edge_inventory = []

        for variable in variables:
            result = build_edges_for_variable(residual_rows, C, variable)
            all_variable_edge_results[variable] = result
            all_edge_rows.extend(result["edge_rows"])
            all_edge_inventory.extend(result["edge_inventory"])
            write_csv(OUTDIR / f"edge_rows_{variable}_v2_1.csv", result["edge_rows"])
            write_csv(OUTDIR / f"edge_inventory_{variable}_v2_1.csv", result["edge_inventory"])
            write_json(OUTDIR / f"regime_aggregate_{variable}_v2_1.json", aggregate_regimes(result["edge_inventory"]))

        write_csv(OUTDIR / "all_variable_edge_rows_v2_1.csv", all_edge_rows)
        write_csv(OUTDIR / "all_variable_edge_inventory_v2_1.csv", all_edge_inventory)

        summary_rows, aggregates = control_variable_summary(all_variable_edge_results)
        write_csv(OUTDIR / "control_variable_summary_v2_1.csv", summary_rows)
        write_json(OUTDIR / "control_variable_regime_aggregates_v2_1.json", aggregates)

        overlap_rows = edge_overlap_rows(
            all_variable_edge_results[LOCKED_VARIABLE]["edge_rows"],
            all_variable_edge_results,
        )
        write_csv(OUTDIR / "control_edge_overlap_with_f160w_v2_1.csv", overlap_rows)

        source_info = source_sniffs(text_files.get(README_PATH, ""), text_files.get(TABLE2_README_PATH, ""))
        write_json(OUTDIR / "source_sniffs_v2_1.json", source_info)
        write_csv(OUTDIR / "source_pattern_counts_v2_1.csv", source_info["pattern_counts"])
        write_csv(OUTDIR / "source_snippets_v2_1.csv", source_info["snippets"])

        residual_ok = (
            predicted_y is not None
            and residual is not None
            and len(residual) == len(y)
            and len(table_rows) == 3130
            and len(residual_rows) == 3130
        )

        decision = decide(summary_rows, overlap_rows, residual_ok)
        write_json(OUTDIR / "decision_v2_1.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v2_1.json", ledger)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v2_1.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v2_1.txt").write_text(handoff, encoding="utf-8")

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
            "table2_residual_summary": summarize_numeric([r["ladder_residual_y_minus_thetaL"] for r in residual_rows]),
        }
        write_json(OUTDIR / "residual_summary_v2_1.json", residual_summary)

        summary = {
            "test_name": "TAIRID SH0ES Frozen Replay Negative Controls v2.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Negative-control specificity testing only. No tuning, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "download_ledger": download_ledger,
            "residual_summary": residual_summary,
            "covariance_diag_summary": cov_info,
            "control_variable_summary": summary_rows,
            "control_edge_overlap_with_f160w": overlap_rows,
            "decision": decision,
            "claims_v2_1": CLAIMS_V2_1,
            "frozen_rule": FROZEN_RULE,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_frozen_replay_negative_controls_v2_1_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_frozen_replay_negative_controls_v2_1_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v2_1.csv"),
                "residual_rows_csv": str(OUTDIR / "table2_ladder_residual_rows_v2_1.csv"),
                "all_edge_inventory_csv": str(OUTDIR / "all_variable_edge_inventory_v2_1.csv"),
                "control_variable_summary_csv": str(OUTDIR / "control_variable_summary_v2_1.csv"),
                "control_edge_overlap_csv": str(OUTDIR / "control_edge_overlap_with_f160w_v2_1.csv"),
                "control_variable_regime_aggregates_json": str(OUTDIR / "control_variable_regime_aggregates_v2_1.json"),
                "decision_json": str(OUTDIR / "decision_v2_1.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v2_1.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v2_1.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v2_1.txt"),
            },
            "interpretation": {
                "what_success_means": "The locked F160W residual surface remains specific against the predeclared Table2 control variables.",
                "what_success_does_not_mean": "This does not prove TAIRID, H0 correction, new physics, or SH0ES error.",
                "what_caution_means": "If controls approach or exceed F160W, report fragility and do not force interpretation.",
                "truth_boundary": CLAIMS_V2_1["truth_boundary"],
            },
        }

        write_json(OUTDIR / "shoes_frozen_replay_negative_controls_v2_1_summary.json", summary)

        with open(OUTDIR / "shoes_frozen_replay_negative_controls_v2_1_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Frozen Replay Negative Controls v2.1\n\n")
            f.write("Boundary: negative-control specificity only. No tuning. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Specificity core:\n")
            f.write(json.dumps(decision["specificity_core"], indent=2, default=json_default) + "\n\n")
            f.write("Control variable summary:\n")
            f.write(json.dumps(summary_rows, indent=2, default=json_default) + "\n\n")
            f.write("Control edge overlap with F160W:\n")
            f.write(json.dumps(overlap_rows, indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not prove SH0ES is wrong.\n")
            f.write("- This does not tune or replace the frozen rule.\n")

        print("TAIRID SH0ES Frozen Replay Negative Controls v2.1 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Frozen Replay Negative Controls v2.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_frozen_replay_negative_controls_v2_1_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V2_1["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_frozen_replay_negative_controls_v2_1_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
