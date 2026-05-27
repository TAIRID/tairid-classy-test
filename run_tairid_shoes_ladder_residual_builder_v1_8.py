#!/usr/bin/env python3
"""
TAIRID SH0ES Ladder Residual Builder v1.8

Why this test exists:
v1.7 confirmed a lawful row-identity/provenance bridge:

    Table2.tex Cepheid rows map to the first 3130 entries of the SH0ES y data vector.

But v1.7 correctly stopped before residual replay because y is data, not residuals.

This v1.8 test performs the next lawful pre-replay step:

    Build the actual ladder residual vector using the public SH0ES least-squares
    parameter vector and the public y/L/C files.

Core residual construction:

    predicted_y = theta @ L
    residual = y - predicted_y

where:
    - theta is parsed from SH0ES_Data/lstsq_results.txt.
    - L is SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits.
    - y is SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits.
    - C is SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits.

This test then attaches the first 3130 residual entries to Table2 rows and builds
a frozen-edge preflight inventory. It does NOT run the final frozen-rule replay.

This test does NOT validate TAIRID.
This test does NOT tune the frozen v1.0 rule.
This test does NOT claim H0 correction or new physics.
This test does NOT treat y as residuals.
This test does NOT promote the edge result as validation.

Truth boundary:
Residual construction is necessary before replay, but not sufficient for model
validation. v1.8 may say "residual surface ready for frozen replay" only if the
theta/L/y/C dimensions are coherent and the first 3130 residuals attach cleanly
to Table2 rows. The actual frozen-rule statistical replay belongs in v1.9.
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


OUTDIR = Path("tairid_shoes_ladder_residual_builder_v1_8_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO = "PantheonPlusSH0ES/DataRelease"
BRANCH_CANDIDATES = ["main", "master"]

TABLE2_PATH = "SH0ES_Data/table2.tex"
TABLE2_README_PATH = "SH0ES_Data/table2.README"
README_PATH = "SH0ES_Data/README.md"

Y_FITS_PATH = "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits"
L_FITS_PATH = "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits"
C_FITS_PATH = "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits"
LSTSQ_PATH = "SH0ES_Data/lstsq_results.txt"

EDGE_PERCENTILE = 5.0
MIN_HOST_ROWS_FOR_EDGE = 20

FROZEN_RULE_CARRIED_FORWARD = {
    "source_status": "locked by v1.0; row identity confirmed by v1.7; residual builder preflight in v1.8",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "frozen_variable": "F160W-like Table2 numeric column, table2_num_7",
    "edge_rule": "within-host high 5% F160W-like magnitude minus within-host low 5% F160W-like magnitude",
    "low_alpha_reference_hosts": ["LMC", "SMC", "N4536"],
    "sign_break_quarantine_hosts": ["M31"],
    "clean_high_alpha_rule": "all Table2 hosts except LMC, SMC, N4536, and M31",
    "hard_boundary": [
        "Do not tune host regimes in v1.8.",
        "Do not invent residuals.",
        "Do not claim validation from residual construction.",
        "Do not claim H0 correction or new physics.",
        "Do not treat the frozen-edge preflight as the final replay.",
    ],
}

CLAIMS_V1_8 = {
    "battery_name": "TAIRID SH0ES Ladder Residual Builder v1.8",
    "scope": "Construct ladder residuals r = y - theta@L and attach Table2 first-block residual surface",
    "primary_question": (
        "Can the public SH0ES y/L/C files and lstsq_results parameter vector construct a finite residual vector "
        "that attaches cleanly to the first 3130 Table2 Cepheid rows?"
    ),
    "truth_boundary": (
        "This is residual construction and replay readiness only. It does not validate TAIRID, H0 correction, or new physics."
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
        headers={"User-Agent": "TAIRID-v1.8-shoes-ladder-residual-builder"},
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
        arr = np.asarray(hdul[0].data, dtype=float)
    return arr


def parse_lstsq_results(text):
    rows = []
    errors = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 1:
            continue
        value = parse_float(tokens[0])
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


def rmse(values):
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else None


def median_abs(values):
    arr = np.asarray(values, dtype=float)
    return float(np.median(np.abs(arr))) if len(arr) else None


def max_abs(values):
    arr = np.asarray(values, dtype=float)
    return float(np.max(np.abs(arr))) if len(arr) else None


def frozen_role_for_host(host):
    if host in {"LMC", "SMC", "N4536"}:
        return "low_alpha_reference"
    if host == "M31":
        return "sign_break_quarantine"
    return "clean_high_alpha_candidate"


def build_residual_rows(table_rows, y, predicted_y, residual, covariance_diag):
    out = []
    n = len(table_rows)
    for i, r in enumerate(table_rows):
        sigma_c = None
        standardized = None
        if covariance_diag is not None and i < len(covariance_diag):
            if covariance_diag[i] > 0:
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


def host_residual_summary(residual_rows):
    by_host = defaultdict(list)
    for row in residual_rows:
        by_host[row["host"]].append(row)

    rows = []
    for host, items in by_host.items():
        residuals = [r["ladder_residual_y_minus_thetaL"] for r in items]
        zvals = [r["standardized_residual"] for r in items if r["standardized_residual"] is not None]
        f160w = [r["f160w"] for r in items]
        rows.append(
            {
                "host": host,
                "row_count": len(items),
                "start_y_index": min(r["y_vector_index_0_based"] for r in items),
                "end_y_index": max(r["y_vector_index_0_based"] for r in items),
                "f160w_min": min(f160w),
                "f160w_median": float(np.median(f160w)),
                "f160w_max": max(f160w),
                "residual_mean": float(np.mean(residuals)),
                "residual_median": float(np.median(residuals)),
                "residual_rmse": rmse(residuals),
                "residual_median_abs": median_abs(residuals),
                "residual_std": float(np.std(residuals)),
                "standardized_residual_rmse": rmse(zvals) if zvals else None,
                "standardized_residual_median_abs": median_abs(zvals) if zvals else None,
                "frozen_regime_role": frozen_role_for_host(host),
                "has_enough_rows_for_edge": len(items) >= MIN_HOST_ROWS_FOR_EDGE,
            }
        )
    return sorted(rows, key=lambda r: (-r["row_count"], r["host"]))


def build_frozen_edge_preflight(residual_rows):
    by_host = defaultdict(list)
    for row in residual_rows:
        by_host[row["host"]].append(row)

    edge_rows = []
    inventory = []

    for host, items in sorted(by_host.items()):
        items = sorted(items, key=lambda r: r["f160w"])
        n = len(items)
        role = frozen_role_for_host(host)

        if n < MIN_HOST_ROWS_FOR_EDGE:
            inventory.append(
                {
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

        for side_name, selected in [
            ("low_F160W_bright_edge", low_side),
            ("high_F160W_faint_edge", high_side),
        ]:
            for r in selected:
                out = dict(r)
                out["edge_side"] = side_name
                out["edge_percentile"] = EDGE_PERCENTILE
                out["host_row_count"] = n
                out["edge_count_each_side"] = k
                edge_rows.append(out)

        low_resid = [r["ladder_residual_y_minus_thetaL"] for r in low_side]
        high_resid = [r["ladder_residual_y_minus_thetaL"] for r in high_side]
        low_f = [r["f160w"] for r in low_side]
        high_f = [r["f160w"] for r in high_side]
        low_z = [r["standardized_residual"] for r in low_side if r["standardized_residual"] is not None]
        high_z = [r["standardized_residual"] for r in high_side if r["standardized_residual"] is not None]

        inventory.append(
            {
                "host": host,
                "row_count": n,
                "edge_status": "edge_surface_built",
                "edge_count_each_side": k,
                "low_edge_mean_F160W": float(np.mean(low_f)),
                "high_edge_mean_F160W": float(np.mean(high_f)),
                "high_minus_low_mean_F160W": float(np.mean(high_f) - np.mean(low_f)),
                "low_edge_mean_residual": float(np.mean(low_resid)),
                "high_edge_mean_residual": float(np.mean(high_resid)),
                "high_minus_low_mean_residual": float(np.mean(high_resid) - np.mean(low_resid)),
                "low_edge_median_residual": float(np.median(low_resid)),
                "high_edge_median_residual": float(np.median(high_resid)),
                "high_minus_low_median_residual": float(np.median(high_resid) - np.median(low_resid)),
                "low_edge_mean_standardized_residual": float(np.mean(low_z)) if low_z else None,
                "high_edge_mean_standardized_residual": float(np.mean(high_z)) if high_z else None,
                "high_minus_low_mean_standardized_residual": float(np.mean(high_z) - np.mean(low_z)) if low_z and high_z else None,
                "frozen_regime_role": role,
                "truth_boundary": "preflight_metric_only_not_final_replay",
            }
        )

    return {
        "edge_rows": sorted(edge_rows, key=lambda r: (r["host"], r["edge_side"], r["f160w"])),
        "edge_inventory": sorted(inventory, key=lambda r: (-r["row_count"], r["host"])),
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

    residual = y - predicted
    return predicted, residual, orientation


def covariance_diag_info(C):
    if C is None or not isinstance(C, np.ndarray) or C.ndim != 2 or C.shape[0] != C.shape[1]:
        return None, {
            "covariance_valid_square": False,
            "reason": "C missing or not square",
        }
    diag = np.diag(C)
    finite = np.isfinite(diag)
    positive = diag > 0
    return diag, {
        "covariance_valid_square": True,
        "C_shape": list(C.shape),
        "diag_count": int(len(diag)),
        "diag_finite_count": int(np.sum(finite)),
        "diag_positive_count": int(np.sum(positive)),
        "diag_nonpositive_count": int(np.sum(~positive)),
        "diag_summary": summarize_numeric(diag.tolist()),
        "sigma_summary": summarize_numeric(np.sqrt(diag[positive]).tolist() if np.any(positive) else []),
    }


def residual_construction_gates(theta_rows, y, L, C, predicted, residual, orientation, table_rows, covariance_diag, edge_result):
    n_table = len(table_rows)
    active_edge_hosts = [
        r for r in edge_result["edge_inventory"]
        if r.get("edge_status") == "edge_surface_built"
    ]

    gates = [
        {
            "gate": "G1_theta_vector_available",
            "passed": len(theta_rows) > 0,
            "detail": "lstsq_results.txt must parse into parameter values.",
            "evidence": {"theta_count": len(theta_rows)},
        },
        {
            "gate": "G2_theta_matches_L_parameter_axis",
            "passed": L.ndim == 2 and len(theta_rows) in set(L.shape),
            "detail": "theta length must match one axis of L.",
            "evidence": {"theta_count": len(theta_rows), "L_shape": list(L.shape)},
        },
        {
            "gate": "G3_y_L_orientation_resolves",
            "passed": orientation in {"theta_dot_L", "L_dot_theta"},
            "detail": "A lawful y prediction orientation must be available.",
            "evidence": {"orientation": orientation, "y_shape": list(y.shape), "L_shape": list(L.shape)},
        },
        {
            "gate": "G4_residual_vector_finite",
            "passed": residual is not None and np.all(np.isfinite(residual)),
            "detail": "Residual vector y - theta*L must be finite.",
            "evidence": {"residual_count": int(len(residual)) if residual is not None else 0},
        },
        {
            "gate": "G5_covariance_square_and_positive_diag",
            "passed": C.ndim == 2 and C.shape[0] == C.shape[1] and covariance_diag is not None and int(np.sum(covariance_diag > 0)) == len(covariance_diag),
            "detail": "C must be square with positive diagonal entries.",
            "evidence": {
                "C_shape": list(C.shape),
                "diag_count": int(len(covariance_diag)) if covariance_diag is not None else 0,
                "positive_diag_count": int(np.sum(covariance_diag > 0)) if covariance_diag is not None else 0,
            },
        },
        {
            "gate": "G6_table2_first_block_attaches",
            "passed": residual is not None and n_table > 0 and len(residual) >= n_table,
            "detail": "The Table2 rows must attach to the first block established in v1.7.",
            "evidence": {"table2_row_count": n_table, "residual_length": int(len(residual)) if residual is not None else 0},
        },
        {
            "gate": "G7_frozen_edge_preflight_available",
            "passed": len(active_edge_hosts) >= 10,
            "detail": "Enough hosts should support within-host frozen-edge preflight surfaces.",
            "evidence": {"active_edge_host_count": len(active_edge_hosts)},
        },
    ]

    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed:
        final_status = "residual_surface_ready_for_frozen_replay_v1_9"
        readiness = 8
        next_wall = (
            "Residuals attach to Table2 and frozen-edge preflight surfaces exist. "
            "v1.9 may run the locked frozen residual replay without tuning."
        )
    elif all(g != "G1_theta_vector_available" and g != "G2_theta_matches_L_parameter_axis" and g != "G3_y_L_orientation_resolves" for g in failed):
        final_status = "residual_surface_partially_ready_with_cautions"
        readiness = 6
        next_wall = (
            "Residual construction works, but covariance/Table2/edge preflight gates need review before replay."
        )
    else:
        final_status = "residual_surface_not_ready"
        readiness = 4
        next_wall = (
            "Residual construction failed at a core theta/L/y gate. Do not replay."
        )

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "gates": gates,
        "failed_gates": failed,
        "truth_boundary": CLAIMS_V1_8["truth_boundary"],
    }


def holographic_surface_ledger(decision):
    return {
        "observable_surface": {
            "name": "SH0ES y/L/C ladder residual surface attached to Table2 rows",
            "table2_path": TABLE2_PATH,
            "y_path": Y_FITS_PATH,
            "l_path": L_FITS_PATH,
            "c_path": C_FITS_PATH,
            "theta_path": LSTSQ_PATH,
        },
        "hidden_depth_sought": {
            "name": "Residual layer for locked Table2 boundary-polarity replay",
            "why_needed": (
                "The frozen high/low F160W edge rule can only be replayed against actual residuals, not raw y data."
            ),
        },
        "boundary_that_forms_surface": {
            "release_boundary": REPO,
            "method_boundary": "Use only public y/L/C and lstsq_results; no tuning and no refit",
            "row_boundary": "First 3130 y entries are the Table2 block established in v1.7",
        },
        "what_can_be_reconstructed_now": [
            "theta parameter vector from lstsq_results.txt",
            "predicted_y from theta @ L",
            "residual vector y - theta @ L",
            "Table2-attached residual rows",
            "host residual summaries",
            "frozen edge preflight inventory",
        ],
        "what_cannot_be_reconstructed_now": [
            "A final TAIRID validation claim",
            "A new H0 correction",
            "New physics",
            "A custom covariance for alternative row selections",
            "A tuned host regime",
        ],
        "surface_noise_definition": [
            "Any use of y as residuals",
            "Any residual built from hand-made or invented theta values",
            "Any host/regime change made after seeing residuals",
            "Any final replay claim made from v1.8 preflight metrics",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v1.8 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v1.8 did")
    lines.append("")
    lines.append("- Parsed the release-provided lstsq_results.txt parameter vector.")
    lines.append("- Read the public SH0ES y/L/C FITS files.")
    lines.append("- Built predicted_y from theta and L.")
    lines.append("- Built the actual residual vector r = y - theta*L.")
    lines.append("- Attached the first 3130 residuals to Table2 rows using the v1.7 row bridge.")
    lines.append("- Built host residual summaries and frozen-edge preflight surfaces.")
    lines.append("- Did not perform the final frozen residual replay.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v1.8 is residual construction and replay readiness only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not tune the frozen rule.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append(
            "v1.9 should run the locked frozen residual replay: within-host high 5% F160W edge minus low 5% F160W edge, with regimes fixed as clean high-alpha, low-alpha/reference LMC+SMC+N4536, and M31 sign-break quarantine. It must report but not tune."
        )
    elif decision["readiness_score_0_to_10"] >= 6:
        lines.append(
            "v1.9 should first repair the failed readiness gates before any replay."
        )
    else:
        lines.append(
            "v1.9 should stop this route and return to a data product with explicit residual columns."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Ladder Residual Builder v1.8 starting.")
    print("Boundary: residual construction only; no validation and no final replay.")

    write_json(OUTDIR / "claims_v1_8.json", CLAIMS_V1_8)
    write_json(OUTDIR / "frozen_rule_carried_forward_v1_8.json", FROZEN_RULE_CARRIED_FORWARD)

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

        write_csv(OUTDIR / "download_ledger_v1_8.csv", download_ledger)

        table2 = parse_table2_tex(text_files.get(TABLE2_PATH, ""))
        table_rows = table2["rows"]
        write_csv(OUTDIR / "table2_parsed_rows_v1_8.csv", table_rows)
        write_json(OUTDIR / "table2_parse_errors_v1_8.json", table2["errors"])

        theta_parse = parse_lstsq_results(text_files.get(LSTSQ_PATH, ""))
        theta_rows = theta_parse["rows"]
        theta = np.asarray([r["theta_value"] for r in theta_rows], dtype=float)
        write_csv(OUTDIR / "theta_lstsq_vector_v1_8.csv", theta_rows)
        write_json(OUTDIR / "theta_lstsq_parse_errors_v1_8.json", theta_parse["errors"])

        y = np.asarray(fits_arrays.get(Y_FITS_PATH, np.asarray([])), dtype=float)
        L = np.asarray(fits_arrays.get(L_FITS_PATH, np.asarray([[]])), dtype=float)
        C = np.asarray(fits_arrays.get(C_FITS_PATH, np.asarray([[]])), dtype=float)

        predicted_y, residual, orientation = compute_ladder_residual(theta, L, y)
        if predicted_y is None:
            predicted_y = np.asarray([])
            residual = np.asarray([])

        covariance_diag, cov_info = covariance_diag_info(C)
        write_json(OUTDIR / "covariance_diag_summary_v1_8.json", cov_info)

        residual_construction = {
            "theta_count": int(len(theta)),
            "y_shape": list(y.shape),
            "L_shape": list(L.shape),
            "C_shape": list(C.shape),
            "orientation": orientation,
            "predicted_y_count": int(len(predicted_y)),
            "residual_count": int(len(residual)),
            "residual_summary_all": summarize_numeric(residual.tolist() if len(residual) else []),
            "residual_rmse_all": rmse(residual) if len(residual) else None,
            "residual_median_abs_all": median_abs(residual) if len(residual) else None,
            "table2_row_count": len(table_rows),
            "table2_block_start_index": 0,
            "table2_block_end_index": len(table_rows) - 1 if table_rows else None,
        }
        write_json(OUTDIR / "ladder_residual_construction_v1_8.json", residual_construction)

        residual_rows = []
        host_summary = []
        edge_result = {"edge_rows": [], "edge_inventory": []}

        if len(residual) >= len(table_rows) and len(table_rows) > 0:
            residual_rows = build_residual_rows(table_rows, y, predicted_y, residual, covariance_diag)
            host_summary = host_residual_summary(residual_rows)
            edge_result = build_frozen_edge_preflight(residual_rows)

        write_csv(OUTDIR / "table2_ladder_residual_rows_v1_8.csv", residual_rows)
        write_csv(OUTDIR / "table2_host_residual_summary_v1_8.csv", host_summary)
        write_csv(OUTDIR / "frozen_edge_preflight_rows_v1_8.csv", edge_result["edge_rows"])
        write_csv(OUTDIR / "frozen_edge_preflight_inventory_v1_8.csv", edge_result["edge_inventory"])

        regime_summary = {}
        for row in edge_result["edge_inventory"]:
            role = row.get("frozen_regime_role", "unknown")
            regime_summary.setdefault(role, {"edge_hosts": 0, "not_enough_rows": 0, "hosts": []})
            if row.get("edge_status") == "edge_surface_built":
                regime_summary[role]["edge_hosts"] += 1
                regime_summary[role]["hosts"].append(row["host"])
            else:
                regime_summary[role]["not_enough_rows"] += 1
        write_json(OUTDIR / "frozen_regime_edge_summary_v1_8.json", regime_summary)

        source_info = source_sniffs(
            text_files.get(README_PATH, ""),
            text_files.get(TABLE2_README_PATH, ""),
        )
        write_json(OUTDIR / "source_sniffs_v1_8.json", source_info)
        write_csv(OUTDIR / "source_pattern_counts_v1_8.csv", source_info["pattern_counts"])
        write_csv(OUTDIR / "source_snippets_v1_8.csv", source_info["snippets"])

        decision = residual_construction_gates(
            theta_rows=theta_rows,
            y=y,
            L=L,
            C=C,
            predicted=predicted_y,
            residual=residual,
            orientation=orientation,
            table_rows=table_rows,
            covariance_diag=covariance_diag,
            edge_result=edge_result,
        )
        write_json(OUTDIR / "decision_v1_8.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v1_8.json", ledger)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v1_8.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v1_8.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID SH0ES Ladder Residual Builder v1.8",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Residual construction and replay readiness only. No final replay, no tuning, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "download_ledger": download_ledger,
            "theta_parse": {
                "theta_count": theta_parse["theta_count"],
                "parse_error_count": len(theta_parse["errors"]),
            },
            "ladder_residual_construction": residual_construction,
            "covariance_diag_summary": cov_info,
            "host_residual_summary_count": len(host_summary),
            "frozen_edge_preflight_host_count": len(edge_result["edge_inventory"]),
            "frozen_regime_edge_summary": regime_summary,
            "decision": decision,
            "claims_v1_8": CLAIMS_V1_8,
            "frozen_rule_carried_forward": FROZEN_RULE_CARRIED_FORWARD,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_ladder_residual_builder_v1_8_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_ladder_residual_builder_v1_8_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v1_8.csv"),
                "theta_vector_csv": str(OUTDIR / "theta_lstsq_vector_v1_8.csv"),
                "residual_construction_json": str(OUTDIR / "ladder_residual_construction_v1_8.json"),
                "covariance_diag_summary_json": str(OUTDIR / "covariance_diag_summary_v1_8.json"),
                "table2_residual_rows_csv": str(OUTDIR / "table2_ladder_residual_rows_v1_8.csv"),
                "host_residual_summary_csv": str(OUTDIR / "table2_host_residual_summary_v1_8.csv"),
                "edge_preflight_rows_csv": str(OUTDIR / "frozen_edge_preflight_rows_v1_8.csv"),
                "edge_preflight_inventory_csv": str(OUTDIR / "frozen_edge_preflight_inventory_v1_8.csv"),
                "regime_edge_summary_json": str(OUTDIR / "frozen_regime_edge_summary_v1_8.json"),
                "decision_json": str(OUTDIR / "decision_v1_8.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v1_8.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v1_8.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v1_8.txt"),
            },
            "interpretation": {
                "what_success_means": "The public SH0ES ladder residual surface can be built and attached to Table2 rows.",
                "what_success_does_not_mean": "This does not validate the frozen Table2 boundary-polarity rule.",
                "next_required_step": "Run the locked frozen residual replay in v1.9 only if readiness gates pass.",
                "truth_boundary": CLAIMS_V1_8["truth_boundary"],
            },
        }

        write_json(OUTDIR / "shoes_ladder_residual_builder_v1_8_summary.json", summary)

        with open(OUTDIR / "shoes_ladder_residual_builder_v1_8_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Ladder Residual Builder v1.8\n\n")
            f.write("Boundary: residual construction and replay readiness only. No final replay. No validation.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Residual construction:\n")
            f.write(json.dumps(residual_construction, indent=2, default=json_default) + "\n\n")
            f.write("Covariance diagonal summary:\n")
            f.write(json.dumps(cov_info, indent=2, default=json_default) + "\n\n")
            f.write("Frozen regime edge summary:\n")
            f.write(json.dumps(regime_summary, indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not tune the frozen rule.\n")
            f.write("- v1.8 edge metrics are preflight only; final replay belongs in v1.9.\n")

        print("TAIRID SH0ES Ladder Residual Builder v1.8 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Ladder Residual Builder v1.8",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_ladder_residual_builder_v1_8_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V1_8["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_ladder_residual_builder_v1_8_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
