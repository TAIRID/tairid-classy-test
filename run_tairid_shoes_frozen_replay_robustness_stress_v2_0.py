#!/usr/bin/env python3
"""
TAIRID SH0ES Frozen Replay Robustness Stress v2.0

Why this test exists:
v1.9 completed the locked frozen residual replay and detected a nonzero
clean-minus-reference residual edge contrast.

v2.0 does NOT change the frozen rule. It stress-tests the v1.9 result without
tuning:

    1. rebuild the same public SH0ES ladder residual surface,
    2. rerun the same frozen within-host 5 percent F160W edge rule,
    3. perform leave-one-host-out jackknife diagnostics,
    4. perform leave-one-reference-host-out diagnostics,
    5. compute covariance-aware edge block diagnostics from the public C matrix,
    6. run a report-only regime-label permutation diagnostic,
    7. produce report-only H0 sensitivity bounds.

This test does NOT validate TAIRID.
This test does NOT tune the frozen v1.0/v1.9 rule.
This test does NOT search new variables.
This test does NOT add new regimes.
This test does NOT claim H0 correction or new physics.
This test does NOT prove SH0ES is wrong.

Truth boundary:
A robust v2.0 result means only that the locked residual surface pattern survives
specified stress tests. H0 sensitivity output is a report-only scale translation,
not an applied correction.
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


OUTDIR = Path("tairid_shoes_frozen_replay_robustness_stress_v2_0_outputs")
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
RANDOM_REPEATS = 500
RANDOM_SEED = 1101980

LOW_ALPHA_REFERENCE_HOSTS = {"LMC", "SMC", "N4536"}
SIGN_BREAK_QUARANTINE_HOSTS = {"M31"}

FROZEN_RULE = {
    "source_status": "v2.0 robustness stress of the locked v1.9 frozen replay",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "frozen_variable": "f160w parsed from Table2.tex",
    "edge_rule": "within-host high 5% F160W/faint edge minus within-host low 5% F160W/bright edge",
    "edge_percentile": EDGE_PERCENTILE,
    "minimum_host_rows_for_edge": MIN_HOST_ROWS_FOR_EDGE,
    "low_alpha_reference_hosts": sorted(LOW_ALPHA_REFERENCE_HOSTS),
    "sign_break_quarantine_hosts": sorted(SIGN_BREAK_QUARANTINE_HOSTS),
    "clean_high_alpha_rule": "all active Table2 hosts except LMC, SMC, N4536, and M31",
    "hard_boundary": [
        "Do not tune the F160W edge percentile in v2.0.",
        "Do not search new variables in v2.0.",
        "Do not add or remove frozen regimes in v2.0.",
        "Do not transfer M31 correction; M31 is quarantine/report-only.",
        "Do not claim H0 correction or new physics.",
        "Do not claim TAIRID is proven.",
    ],
}

CLAIMS_V2_0 = {
    "battery_name": "TAIRID SH0ES Frozen Replay Robustness Stress v2.0",
    "scope": "Jackknife, covariance-aware, permutation, and report-only H0 sensitivity stress tests for locked v1.9 replay",
    "primary_question": (
        "Does the locked within-host F160W residual edge pattern survive robustness stress tests without tuning?"
    ),
    "truth_boundary": (
        "This is robustness stress testing only. It does not validate TAIRID, H0 correction, or new physics."
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
        headers={"User-Agent": "TAIRID-v2.0-shoes-frozen-replay-robustness-stress"},
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
        return {
            "covariance_edge_variance_full": None,
            "covariance_edge_sigma_full": None,
            "covariance_edge_z_full": None,
            "covariance_edge_variance_diag_only": None,
            "covariance_edge_sigma_diag_only": None,
            "covariance_edge_z_diag_only": None,
        }

    low_idx = [r["y_vector_index_0_based"] for r in low_side]
    high_idx = [r["y_vector_index_0_based"] for r in high_side]
    idxs = high_idx + low_idx
    k_high = len(high_idx)
    k_low = len(low_idx)
    weights = np.asarray(([1.0 / k_high] * k_high) + ([-1.0 / k_low] * k_low), dtype=float)

    delta = float(np.mean([r["ladder_residual_y_minus_thetaL"] for r in high_side]) - np.mean([r["ladder_residual_y_minus_thetaL"] for r in low_side]))

    sub = C[np.ix_(idxs, idxs)]
    var_full = float(weights @ sub @ weights)
    diag = np.diag(C)[idxs]
    var_diag = float(np.sum((weights ** 2) * diag))

    sigma_full = math.sqrt(var_full) if var_full > 0 else None
    sigma_diag = math.sqrt(var_diag) if var_diag > 0 else None

    return {
        "covariance_edge_variance_full": var_full,
        "covariance_edge_sigma_full": sigma_full,
        "covariance_edge_z_full": float(delta / sigma_full) if sigma_full else None,
        "covariance_edge_variance_diag_only": var_diag,
        "covariance_edge_sigma_diag_only": sigma_diag,
        "covariance_edge_z_diag_only": float(delta / sigma_diag) if sigma_diag else None,
    }


def build_frozen_edges(residual_rows, C):
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

        for side_name, selected in [("low_F160W_bright_edge", low_side), ("high_F160W_faint_edge", high_side)]:
            for r in selected:
                out = dict(r)
                out["edge_side"] = side_name
                out["edge_percentile"] = EDGE_PERCENTILE
                out["host_row_count"] = n
                out["edge_count_each_side"] = k
                edge_rows.append(out)

        low_resid = np.asarray([r["ladder_residual_y_minus_thetaL"] for r in low_side], dtype=float)
        high_resid = np.asarray([r["ladder_residual_y_minus_thetaL"] for r in high_side], dtype=float)
        low_z = np.asarray([r["standardized_residual"] for r in low_side if r["standardized_residual"] is not None], dtype=float)
        high_z = np.asarray([r["standardized_residual"] for r in high_side if r["standardized_residual"] is not None], dtype=float)
        low_f = np.asarray([r["f160w"] for r in low_side], dtype=float)
        high_f = np.asarray([r["f160w"] for r in high_side], dtype=float)

        cov_diag = edge_covariance_diagnostic(low_side, high_side, C)

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
                "low_edge_mean_standardized_residual": float(np.mean(low_z)) if len(low_z) else None,
                "high_edge_mean_standardized_residual": float(np.mean(high_z)) if len(high_z) else None,
                "high_minus_low_mean_standardized_residual": float(np.mean(high_z) - np.mean(low_z)) if len(low_z) and len(high_z) else None,
                "frozen_regime_role": role,
                **cov_diag,
            }
        )

    return {
        "edge_rows": sorted(edge_rows, key=lambda r: (r["host"], r["edge_side"], r["f160w"])),
        "edge_inventory": sorted(inventory, key=lambda r: (-r["row_count"], r["host"])),
    }


def aggregate_regimes(edge_inventory, omit_hosts=None):
    omit_hosts = set(omit_hosts or [])
    roles = defaultdict(list)
    for row in edge_inventory:
        if row.get("edge_status") == "edge_surface_built" and row["host"] not in omit_hosts:
            roles[row["frozen_regime_role"]].append(row)

    out = {}
    for role, rows in roles.items():
        values = np.asarray([r["high_minus_low_mean_residual"] for r in rows], dtype=float)
        weights = np.asarray([r["edge_count_each_side"] for r in rows], dtype=float)
        z_full = np.asarray([r["covariance_edge_z_full"] for r in rows if r.get("covariance_edge_z_full") is not None], dtype=float)

        inv_var_values = []
        inv_var_weights = []
        for r in rows:
            var = r.get("covariance_edge_variance_full")
            if var is not None and var > 0:
                inv_var_values.append(r["high_minus_low_mean_residual"])
                inv_var_weights.append(1.0 / var)

        out[role] = {
            "host_count": int(len(rows)),
            "hosts": [r["host"] for r in rows],
            "unweighted_mean_high_minus_low_residual": float(np.mean(values)) if len(values) else None,
            "unweighted_median_high_minus_low_residual": float(np.median(values)) if len(values) else None,
            "weighted_mean_high_minus_low_residual": float(np.average(values, weights=weights)) if len(values) and np.sum(weights) > 0 else None,
            "inverse_variance_weighted_mean_high_minus_low_residual": float(np.average(inv_var_values, weights=inv_var_weights)) if inv_var_values and np.sum(inv_var_weights) > 0 else None,
            "std_high_minus_low_residual": float(np.std(values)) if len(values) else None,
            "mean_covariance_edge_z_full": float(np.mean(z_full)) if len(z_full) else None,
        }

    clean = out.get("clean_high_alpha_candidate", {})
    ref = out.get("low_alpha_reference", {})
    out["clean_minus_reference"] = {
        "unweighted_mean_delta": None,
        "weighted_mean_delta": None,
        "inverse_variance_weighted_delta": None,
        "truth_boundary": "descriptive_robustness_contrast_only_not_H0_correction",
    }
    if clean and ref:
        if clean.get("unweighted_mean_high_minus_low_residual") is not None and ref.get("unweighted_mean_high_minus_low_residual") is not None:
            out["clean_minus_reference"]["unweighted_mean_delta"] = clean["unweighted_mean_high_minus_low_residual"] - ref["unweighted_mean_high_minus_low_residual"]
        if clean.get("weighted_mean_high_minus_low_residual") is not None and ref.get("weighted_mean_high_minus_low_residual") is not None:
            out["clean_minus_reference"]["weighted_mean_delta"] = clean["weighted_mean_high_minus_low_residual"] - ref["weighted_mean_high_minus_low_residual"]
        if clean.get("inverse_variance_weighted_mean_high_minus_low_residual") is not None and ref.get("inverse_variance_weighted_mean_high_minus_low_residual") is not None:
            out["clean_minus_reference"]["inverse_variance_weighted_delta"] = clean["inverse_variance_weighted_mean_high_minus_low_residual"] - ref["inverse_variance_weighted_mean_high_minus_low_residual"]
    return out


def jackknife_diagnostics(edge_inventory):
    full = aggregate_regimes(edge_inventory)
    full_delta = full.get("clean_minus_reference", {}).get("weighted_mean_delta")
    built_hosts = [r["host"] for r in edge_inventory if r.get("edge_status") == "edge_surface_built"]
    rows = []

    for host in sorted(built_hosts):
        agg = aggregate_regimes(edge_inventory, omit_hosts={host})
        delta = agg.get("clean_minus_reference", {}).get("weighted_mean_delta")
        rows.append(
            {
                "omitted_host": host,
                "omitted_role": frozen_role_for_host(host),
                "full_weighted_delta": full_delta,
                "jackknife_weighted_delta": delta,
                "delta_shift_from_full": (delta - full_delta) if delta is not None and full_delta is not None else None,
                "clean_host_count_after_omit": agg.get("clean_high_alpha_candidate", {}).get("host_count", 0),
                "reference_host_count_after_omit": agg.get("low_alpha_reference", {}).get("host_count", 0),
            }
        )

    clean_j = [r["jackknife_weighted_delta"] for r in rows if r["omitted_role"] == "clean_high_alpha_candidate" and r["jackknife_weighted_delta"] is not None]
    ref_j = [r["jackknife_weighted_delta"] for r in rows if r["omitted_role"] == "low_alpha_reference" and r["jackknife_weighted_delta"] is not None]
    all_j = [r["jackknife_weighted_delta"] for r in rows if r["jackknife_weighted_delta"] is not None]

    summary = {
        "full_weighted_delta": full_delta,
        "all_jackknife": summarize_numeric(all_j),
        "clean_host_omits": summarize_numeric(clean_j),
        "reference_host_omits": summarize_numeric(ref_j),
        "all_jackknife_positive_count": int(sum(1 for v in all_j if v > 0)),
        "all_jackknife_count": int(len(all_j)),
        "clean_jackknife_positive_count": int(sum(1 for v in clean_j if v > 0)),
        "clean_jackknife_count": int(len(clean_j)),
        "reference_jackknife_positive_count": int(sum(1 for v in ref_j if v > 0)),
        "reference_jackknife_count": int(len(ref_j)),
        "largest_absolute_shift": max([abs(r["delta_shift_from_full"]) for r in rows if r["delta_shift_from_full"] is not None], default=None),
    }
    return {"rows": rows, "summary": summary}


def permutation_regime_control(edge_inventory):
    rng = np.random.default_rng(RANDOM_SEED)
    active = [
        r for r in edge_inventory
        if r.get("edge_status") == "edge_surface_built"
        and r["frozen_regime_role"] != "sign_break_quarantine"
    ]
    ref_hosts = [r["host"] for r in active if r["frozen_regime_role"] == "low_alpha_reference"]
    ref_n = len(ref_hosts)
    if ref_n == 0:
        return {"rows": [], "summary": {"reason": "no_reference_hosts"}}

    host_to_row = {r["host"]: r for r in active}
    hosts = sorted(host_to_row)
    observed = aggregate_regimes(edge_inventory).get("clean_minus_reference", {}).get("weighted_mean_delta")

    rows = []
    for repeat in range(RANDOM_REPEATS):
        pseudo_ref = set(rng.choice(hosts, size=ref_n, replace=False).tolist())
        pseudo_clean = [h for h in hosts if h not in pseudo_ref]

        ref_rows = [host_to_row[h] for h in pseudo_ref]
        clean_rows = [host_to_row[h] for h in pseudo_clean]

        def weighted_mean(rs):
            vals = np.asarray([r["high_minus_low_mean_residual"] for r in rs], dtype=float)
            w = np.asarray([r["edge_count_each_side"] for r in rs], dtype=float)
            return float(np.average(vals, weights=w)) if len(vals) and np.sum(w) > 0 else None

        ref_mean = weighted_mean(ref_rows)
        clean_mean = weighted_mean(clean_rows)
        delta = clean_mean - ref_mean if clean_mean is not None and ref_mean is not None else None

        rows.append(
            {
                "repeat": repeat,
                "pseudo_reference_hosts": ",".join(sorted(pseudo_ref)),
                "pseudo_clean_host_count": len(pseudo_clean),
                "pseudo_reference_host_count": len(pseudo_ref),
                "pseudo_clean_minus_reference_weighted_delta": delta,
                "observed_frozen_delta": observed,
            }
        )

    vals = np.asarray([r["pseudo_clean_minus_reference_weighted_delta"] for r in rows if r["pseudo_clean_minus_reference_weighted_delta"] is not None], dtype=float)
    summary = {
        "observed_frozen_delta": observed,
        "random_repeat_count": int(len(vals)),
        "pseudo_delta_summary": summarize_numeric(vals.tolist()),
        "two_sided_empirical_p": float((np.sum(np.abs(vals) >= abs(observed)) + 1) / (len(vals) + 1)) if observed is not None and len(vals) else None,
        "one_sided_high_empirical_p": float((np.sum(vals >= observed) + 1) / (len(vals) + 1)) if observed is not None and len(vals) else None,
        "truth_boundary": "regime_label_permutation_is_report_only_not_regime_tuning",
    }
    return {"rows": rows, "summary": summary}


def h0_sensitivity(theta, full_delta, jackknife_summary):
    h0_like = float(theta[46]) if len(theta) > 46 else None
    fractions = [0.01, 0.05, 0.10, 0.25, 1.00]
    rows = []

    if h0_like is None or full_delta is None:
        return {"rows": [], "summary": {"h0_like_from_theta_index_46": h0_like, "reason": "missing_h0_or_delta"}}

    for frac in fractions:
        mu_delta = full_delta * frac
        ratio = 10 ** (-mu_delta / 5.0)
        rows.append(
            {
                "fraction_of_observed_delta_applied": frac,
                "mu_delta_mag_report_only": mu_delta,
                "h0_ratio_if_interpreted_as_distance_modulus": ratio,
                "h0_like_start_theta46": h0_like,
                "h0_like_after_report_only_translation": h0_like * ratio,
                "h0_like_shift": h0_like * ratio - h0_like,
                "truth_boundary": "report_only_not_applied_correction",
            }
        )

    jack = jackknife_summary.get("all_jackknife", {})
    return {
        "rows": rows,
        "summary": {
            "h0_like_from_theta_index_46": h0_like,
            "full_clean_minus_reference_weighted_delta": full_delta,
            "jackknife_delta_min": jack.get("min"),
            "jackknife_delta_max": jack.get("max"),
            "formula": "H0_ratio = 10^(-mu_delta/5)",
            "truth_boundary": "This is scale translation only, not an H0 correction.",
        },
    }


def decide(full_aggregate, jackknife_summary, permutation_summary, edge_inventory):
    full_delta = full_aggregate.get("clean_minus_reference", {}).get("weighted_mean_delta")
    inv_delta = full_aggregate.get("clean_minus_reference", {}).get("inverse_variance_weighted_delta")
    clean = full_aggregate.get("clean_high_alpha_candidate", {})
    ref = full_aggregate.get("low_alpha_reference", {})
    built_edges = [r for r in edge_inventory if r.get("edge_status") == "edge_surface_built"]

    all_j_count = jackknife_summary.get("all_jackknife_count", 0)
    all_j_pos = jackknife_summary.get("all_jackknife_positive_count", 0)
    clean_j_count = jackknife_summary.get("clean_jackknife_count", 0)
    clean_j_pos = jackknife_summary.get("clean_jackknife_positive_count", 0)
    ref_j_count = jackknife_summary.get("reference_jackknife_count", 0)
    ref_j_pos = jackknife_summary.get("reference_jackknife_positive_count", 0)
    perm_p = permutation_summary.get("one_sided_high_empirical_p")

    gates = [
        {
            "gate": "G1_locked_replay_rebuilt",
            "passed": full_delta is not None and len(built_edges) >= 10,
            "evidence": {"built_edge_host_count": len(built_edges), "weighted_delta": full_delta},
        },
        {
            "gate": "G2_clean_and_reference_present",
            "passed": (clean.get("host_count", 0) or 0) >= 10 and (ref.get("host_count", 0) or 0) >= 3,
            "evidence": {"clean_hosts": clean.get("host_count", 0), "reference_hosts": ref.get("host_count", 0)},
        },
        {
            "gate": "G3_leave_one_clean_host_positive",
            "passed": clean_j_count > 0 and clean_j_pos == clean_j_count,
            "evidence": {"clean_jackknife_positive_count": clean_j_pos, "clean_jackknife_count": clean_j_count},
        },
        {
            "gate": "G4_leave_one_reference_host_positive",
            "passed": ref_j_count > 0 and ref_j_pos == ref_j_count,
            "evidence": {"reference_jackknife_positive_count": ref_j_pos, "reference_jackknife_count": ref_j_count},
        },
        {
            "gate": "G5_covariance_aware_delta_same_direction",
            "passed": inv_delta is not None and full_delta is not None and inv_delta * full_delta > 0,
            "evidence": {"weighted_delta": full_delta, "inverse_variance_weighted_delta": inv_delta},
        },
        {
            "gate": "G6_permutation_report_completed",
            "passed": perm_p is not None,
            "evidence": {"one_sided_high_empirical_p": perm_p, "random_repeats": RANDOM_REPEATS},
        },
    ]

    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed:
        if perm_p is not None and perm_p <= 0.05:
            final_status = "locked_replay_robust_under_jackknife_and_permutation"
            readiness = 9
            next_wall = (
                "The locked residual surface survived jackknife, covariance-aware, and permutation stress tests. "
                "Next step should be a write-up plus an external-lane falsification test, not rule tuning."
            )
        else:
            final_status = "locked_replay_robust_under_jackknife_covariance_permutation_not_extreme"
            readiness = 8
            next_wall = (
                "The locked residual surface survived jackknife and covariance-aware stress tests, while permutation extremity was not strong. "
                "Next step should document this boundary and run an external falsification lane."
            )
    elif len(failed) <= 2 and "G1_locked_replay_rebuilt" not in failed:
        final_status = "locked_replay_partially_robust_with_cautions"
        readiness = 7
        next_wall = (
            "The locked replay remains measurable but failed one or two stress gates. Report the fragility directly."
        )
    else:
        final_status = "locked_replay_not_robust_enough_for_escalation"
        readiness = 5
        next_wall = (
            "The locked replay did not survive enough robustness gates. Do not escalate."
        )

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "gates": gates,
        "failed_gates": failed,
        "robustness_core": {
            "full_weighted_delta": full_delta,
            "inverse_variance_weighted_delta": inv_delta,
            "clean_host_count": clean.get("host_count", 0),
            "reference_host_count": ref.get("host_count", 0),
            "all_jackknife_positive_count": all_j_pos,
            "all_jackknife_count": all_j_count,
            "clean_jackknife_positive_count": clean_j_pos,
            "clean_jackknife_count": clean_j_count,
            "reference_jackknife_positive_count": ref_j_pos,
            "reference_jackknife_count": ref_j_count,
            "permutation_one_sided_high_p": perm_p,
        },
        "truth_boundary": CLAIMS_V2_0["truth_boundary"],
    }


def holographic_surface_ledger(decision):
    return {
        "observable_surface": {
            "name": "Frozen Table2 F160W edge replay under robustness stress",
            "table2_path": TABLE2_PATH,
            "theta_path": LSTSQ_PATH,
            "y_path": Y_FITS_PATH,
            "l_path": L_FITS_PATH,
            "c_path": C_FITS_PATH,
        },
        "hidden_depth_sought": {
            "name": "Stress stability of frozen residual surface pattern",
            "allowed_claim": "The locked replay can be stress-tested for stability without tuning.",
            "not_allowed_claim": "Do not treat robustness as proof of H0 correction, SH0ES error, TAIRID proof, or new physics.",
        },
        "boundary_that_forms_surface": {
            "row_boundary": "first 3130 y entries attached to Table2 rows",
            "edge_boundary": "within-host 5 percent bright/faint F160W edges",
            "regime_boundary": "frozen v1.0/v1.9 regimes; M31 quarantine not transferred",
            "stress_boundary": "jackknife/covariance/permutation diagnostics only; no rule change",
        },
        "what_can_be_reconstructed_now": [
            "Leave-one-host-out stability",
            "Leave-one-reference-host-out stability",
            "Covariance-aware edge z diagnostics",
            "Inverse-variance aggregate diagnostics",
            "Report-only H0 sensitivity scale",
        ],
        "what_cannot_be_reconstructed_now": [
            "A new H0 solution",
            "New physics",
            "A causal explanation",
            "A tuned likelihood model",
            "Generalization beyond Table2 without a separate replay",
        ],
        "surface_noise_definition": [
            "Any stress result obtained by changing the frozen edge rule",
            "Any M31 correction transfer",
            "Any H0 sensitivity row treated as an applied correction",
            "Any post-hoc regime editing",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v2.0 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v2.0 did")
    lines.append("")
    lines.append("- Rebuilt the same public SH0ES ladder residual surface.")
    lines.append("- Re-ran the locked within-host 5 percent F160W edge rule.")
    lines.append("- Kept regimes fixed: clean high-alpha, LMC+SMC+N4536 reference, M31 quarantine.")
    lines.append("- Ran leave-one-host-out and leave-one-reference-host-out jackknife diagnostics.")
    lines.append("- Computed covariance-aware edge-block diagnostics from the public C matrix.")
    lines.append("- Ran report-only regime-label permutation diagnostics.")
    lines.append("- Computed report-only H0 sensitivity scale translations.")
    lines.append("- Did not tune, refit, or claim H0 correction.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v2.0 is robustness stress testing only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not prove SH0ES is wrong.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append(
            "v2.1 should be an external falsification lane: replay the same locked surface against a separate public SH0ES-adjacent surface or run a negative-control variable within Table2 without changing the original frozen result."
        )
    elif decision["readiness_score_0_to_10"] >= 7:
        lines.append(
            "v2.1 should document the fragility points and rerun only the failed robustness diagnostics, without changing the frozen rule."
        )
    else:
        lines.append(
            "v2.1 should stop escalation and write the boundary result honestly."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Frozen Replay Robustness Stress v2.0 starting.")
    print("Boundary: robustness stress only; no tuning, no H0 claim, no new physics claim.")

    write_json(OUTDIR / "claims_v2_0.json", CLAIMS_V2_0)
    write_json(OUTDIR / "frozen_rule_v2_0.json", FROZEN_RULE)

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

        write_csv(OUTDIR / "download_ledger_v2_0.csv", download_ledger)

        table2 = parse_table2_tex(text_files.get(TABLE2_PATH, ""))
        table_rows = table2["rows"]
        write_csv(OUTDIR / "table2_parsed_rows_v2_0.csv", table_rows)
        write_json(OUTDIR / "table2_parse_errors_v2_0.json", table2["errors"])

        theta_parse = parse_lstsq_results(text_files.get(LSTSQ_PATH, ""))
        theta_rows = theta_parse["rows"]
        theta = np.asarray([r["theta_value"] for r in theta_rows], dtype=float)
        write_csv(OUTDIR / "theta_lstsq_vector_v2_0.csv", theta_rows)
        write_json(OUTDIR / "theta_lstsq_parse_errors_v2_0.json", theta_parse["errors"])

        y = np.asarray(fits_arrays.get(Y_FITS_PATH, np.asarray([])), dtype=float)
        L = np.asarray(fits_arrays.get(L_FITS_PATH, np.asarray([[]])), dtype=float)
        C = np.asarray(fits_arrays.get(C_FITS_PATH, np.asarray([[]])), dtype=float)

        predicted_y, residual, orientation = compute_ladder_residual(theta, L, y)
        if predicted_y is None:
            predicted_y = np.asarray([])
            residual = np.asarray([])

        covariance_diag, cov_info = covariance_diag_info(C)
        write_json(OUTDIR / "covariance_diag_summary_v2_0.json", cov_info)

        residual_rows = []
        edge_result = {"edge_rows": [], "edge_inventory": []}
        if len(residual) >= len(table_rows) and len(table_rows) > 0:
            residual_rows = attach_table2_residuals(table_rows, y, predicted_y, residual, covariance_diag)
            edge_result = build_frozen_edges(residual_rows, C)

        write_csv(OUTDIR / "table2_ladder_residual_rows_v2_0.csv", residual_rows)
        write_csv(OUTDIR / "frozen_edge_rows_v2_0.csv", edge_result["edge_rows"])
        write_csv(OUTDIR / "frozen_edge_inventory_covariance_v2_0.csv", edge_result["edge_inventory"])

        full_aggregate = aggregate_regimes(edge_result["edge_inventory"])
        write_json(OUTDIR / "frozen_regime_aggregate_covariance_v2_0.json", full_aggregate)

        jackknife = jackknife_diagnostics(edge_result["edge_inventory"])
        write_csv(OUTDIR / "leave_one_host_jackknife_v2_0.csv", jackknife["rows"])
        write_json(OUTDIR / "leave_one_host_jackknife_summary_v2_0.json", jackknife["summary"])

        permutation = permutation_regime_control(edge_result["edge_inventory"])
        write_csv(OUTDIR / "regime_label_permutation_rows_v2_0.csv", permutation["rows"])
        write_json(OUTDIR / "regime_label_permutation_summary_v2_0.json", permutation["summary"])

        full_delta = full_aggregate.get("clean_minus_reference", {}).get("weighted_mean_delta")
        h0 = h0_sensitivity(theta, full_delta, jackknife["summary"])
        write_csv(OUTDIR / "h0_sensitivity_report_only_rows_v2_0.csv", h0["rows"])
        write_json(OUTDIR / "h0_sensitivity_report_only_summary_v2_0.json", h0["summary"])

        decision = decide(
            full_aggregate=full_aggregate,
            jackknife_summary=jackknife["summary"],
            permutation_summary=permutation["summary"],
            edge_inventory=edge_result["edge_inventory"],
        )
        write_json(OUTDIR / "decision_v2_0.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v2_0.json", ledger)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v2_0.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v2_0.txt").write_text(handoff, encoding="utf-8")

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
        write_json(OUTDIR / "residual_summary_v2_0.json", residual_summary)

        summary = {
            "test_name": "TAIRID SH0ES Frozen Replay Robustness Stress v2.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Robustness stress testing only. No tuning, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "download_ledger": download_ledger,
            "residual_summary": residual_summary,
            "covariance_diag_summary": cov_info,
            "frozen_regime_aggregate_covariance": full_aggregate,
            "jackknife_summary": jackknife["summary"],
            "permutation_summary": permutation["summary"],
            "h0_sensitivity_report_only": h0["summary"],
            "decision": decision,
            "claims_v2_0": CLAIMS_V2_0,
            "frozen_rule": FROZEN_RULE,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_frozen_replay_robustness_stress_v2_0_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_frozen_replay_robustness_stress_v2_0_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v2_0.csv"),
                "residual_rows_csv": str(OUTDIR / "table2_ladder_residual_rows_v2_0.csv"),
                "edge_inventory_covariance_csv": str(OUTDIR / "frozen_edge_inventory_covariance_v2_0.csv"),
                "regime_aggregate_json": str(OUTDIR / "frozen_regime_aggregate_covariance_v2_0.json"),
                "jackknife_csv": str(OUTDIR / "leave_one_host_jackknife_v2_0.csv"),
                "jackknife_summary_json": str(OUTDIR / "leave_one_host_jackknife_summary_v2_0.json"),
                "permutation_rows_csv": str(OUTDIR / "regime_label_permutation_rows_v2_0.csv"),
                "permutation_summary_json": str(OUTDIR / "regime_label_permutation_summary_v2_0.json"),
                "h0_sensitivity_rows_csv": str(OUTDIR / "h0_sensitivity_report_only_rows_v2_0.csv"),
                "h0_sensitivity_summary_json": str(OUTDIR / "h0_sensitivity_report_only_summary_v2_0.json"),
                "decision_json": str(OUTDIR / "decision_v2_0.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v2_0.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v2_0.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v2_0.txt"),
            },
            "interpretation": {
                "what_success_means": "The locked residual surface pattern survived predeclared robustness stress tests.",
                "what_success_does_not_mean": "This does not prove TAIRID, H0 correction, new physics, or SH0ES error.",
                "next_required_step": "External falsification or negative-control testing without changing the frozen rule.",
                "truth_boundary": CLAIMS_V2_0["truth_boundary"],
            },
        }

        write_json(OUTDIR / "shoes_frozen_replay_robustness_stress_v2_0_summary.json", summary)

        with open(OUTDIR / "shoes_frozen_replay_robustness_stress_v2_0_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Frozen Replay Robustness Stress v2.0\n\n")
            f.write("Boundary: robustness stress only. No tuning. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Robustness core:\n")
            f.write(json.dumps(decision["robustness_core"], indent=2, default=json_default) + "\n\n")
            f.write("Full aggregate:\n")
            f.write(json.dumps(full_aggregate, indent=2, default=json_default) + "\n\n")
            f.write("Jackknife summary:\n")
            f.write(json.dumps(jackknife["summary"], indent=2, default=json_default) + "\n\n")
            f.write("Permutation summary:\n")
            f.write(json.dumps(permutation["summary"], indent=2, default=json_default) + "\n\n")
            f.write("H0 sensitivity report-only summary:\n")
            f.write(json.dumps(h0["summary"], indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not prove SH0ES is wrong.\n")
            f.write("- This does not tune the frozen rule.\n")

        print("TAIRID SH0ES Frozen Replay Robustness Stress v2.0 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Frozen Replay Robustness Stress v2.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_frozen_replay_robustness_stress_v2_0_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V2_0["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_frozen_replay_robustness_stress_v2_0_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
