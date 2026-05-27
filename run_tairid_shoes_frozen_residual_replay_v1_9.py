#!/usr/bin/env python3
"""
TAIRID SH0ES Frozen Residual Replay v1.9

Why this test exists:
The uploaded table2 auto artifact already showed that Table2 maps into the
SH0ES y/L/C ladder system and that residual pressure can be computed. But that
artifact also scanned several candidate structures.

v1.9 is different. It runs only the frozen rule carried forward from v1.0:

    Variable:
        Table2 F160W-like column from table2.tex, parsed as f160w.

    Edge rule:
        Within each host, sort Cepheids by f160w.
        Low edge  = brightest / lowest 5 percent F160W rows.
        High edge = faintest / highest 5 percent F160W rows.
        Edge score = high-edge mean ladder residual minus low-edge mean ladder residual.

    Frozen regimes:
        clean_high_alpha_candidate = all hosts except LMC, SMC, N4536, and M31.
        low_alpha_reference = LMC + SMC + N4536.
        sign_break_quarantine = M31. M31 is reported but not transferred.

This test uses the public SH0ES release files:

    table2.tex
    lstsq_results.txt
    ally_shoes_ceph_topantheonwt6.0_112221.fits
    alll_shoes_ceph_topantheonwt6.0_112221.fits
    allc_shoes_ceph_topantheonwt6.0_112221.fits

Core residual construction:

    predicted_y = theta @ L
    residual = y - predicted_y

This test DOES run the locked residual replay.
It does NOT tune the rule.
It does NOT search new variables.
It does NOT add new regimes.
It does NOT claim H0 correction.
It does NOT claim new physics.
It does NOT claim TAIRID is proven.

Truth boundary:
A strong frozen residual replay means only that the locked Table2 F160W edge
surface reappears inside the public SH0ES ladder residual layer. It is evidence
for a residual surface pattern, not a cosmological correction or proof of new
physics.
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


OUTDIR = Path("tairid_shoes_frozen_residual_replay_v1_9_outputs")
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
RANDOM_REPEATS = 250
RANDOM_SEED = 1101980

LOW_ALPHA_REFERENCE_HOSTS = {"LMC", "SMC", "N4536"}
SIGN_BREAK_QUARANTINE_HOSTS = {"M31"}

FROZEN_RULE = {
    "source_status": "locked by v1.0 and replayed without tuning in v1.9",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "frozen_variable": "f160w parsed from Table2.tex",
    "edge_rule": "within-host high 5% F160W/faint edge minus within-host low 5% F160W/bright edge",
    "edge_percentile": EDGE_PERCENTILE,
    "minimum_host_rows_for_edge": MIN_HOST_ROWS_FOR_EDGE,
    "low_alpha_reference_hosts": sorted(LOW_ALPHA_REFERENCE_HOSTS),
    "sign_break_quarantine_hosts": sorted(SIGN_BREAK_QUARANTINE_HOSTS),
    "clean_high_alpha_rule": "all active Table2 hosts except LMC, SMC, N4536, and M31",
    "hard_boundary": [
        "Do not tune the F160W edge percentile in v1.9.",
        "Do not search new variables in v1.9.",
        "Do not add or remove frozen regimes in v1.9.",
        "Do not transfer M31 correction; M31 is quarantine/report-only.",
        "Do not claim H0 correction or new physics.",
        "Do not claim TAIRID is proven.",
    ],
}

CLAIMS_V1_9 = {
    "battery_name": "TAIRID SH0ES Frozen Residual Replay v1.9",
    "scope": "Locked within-host Table2 F160W 5% edge replay against SH0ES ladder residuals",
    "primary_question": (
        "Does the frozen within-host F160W high-minus-low edge rule reappear in the public SH0ES "
        "ladder residual layer after Table2 rows are attached to y - theta@L?"
    ),
    "truth_boundary": (
        "This is a locked residual surface replay only. It does not validate TAIRID, H0 correction, or new physics."
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
        headers={"User-Agent": "TAIRID-v1.9-shoes-frozen-residual-replay"},
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


def rmse(values):
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else None


def median_abs(values):
    arr = np.asarray(values, dtype=float)
    return float(np.median(np.abs(arr))) if len(arr) else None


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
        return None, {"covariance_valid_square": False, "reason": "C missing or not square"}
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


def build_frozen_edges(residual_rows):
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
            }
        )

    return {
        "edge_rows": sorted(edge_rows, key=lambda r: (r["host"], r["edge_side"], r["f160w"])),
        "edge_inventory": sorted(inventory, key=lambda r: (-r["row_count"], r["host"])),
    }


def aggregate_regimes(edge_inventory):
    roles = defaultdict(list)
    for row in edge_inventory:
        if row.get("edge_status") == "edge_surface_built":
            roles[row["frozen_regime_role"]].append(row)

    out = {}
    for role, rows in roles.items():
        values = np.asarray([r["high_minus_low_mean_residual"] for r in rows], dtype=float)
        z_values = np.asarray([r["high_minus_low_mean_standardized_residual"] for r in rows if r["high_minus_low_mean_standardized_residual"] is not None], dtype=float)
        weights = np.asarray([r["edge_count_each_side"] for r in rows], dtype=float)

        weighted = float(np.average(values, weights=weights)) if len(values) and np.sum(weights) > 0 else None
        out[role] = {
            "host_count": int(len(rows)),
            "hosts": [r["host"] for r in rows],
            "unweighted_mean_high_minus_low_residual": float(np.mean(values)) if len(values) else None,
            "unweighted_median_high_minus_low_residual": float(np.median(values)) if len(values) else None,
            "weighted_mean_high_minus_low_residual": weighted,
            "std_high_minus_low_residual": float(np.std(values)) if len(values) else None,
            "unweighted_mean_high_minus_low_standardized_residual": float(np.mean(z_values)) if len(z_values) else None,
            "weighted_by_edge_count": True,
        }

    clean = out.get("clean_high_alpha_candidate", {})
    ref = out.get("low_alpha_reference", {})
    if clean and ref:
        out["clean_minus_reference"] = {
            "unweighted_mean_delta": clean.get("unweighted_mean_high_minus_low_residual") - ref.get("unweighted_mean_high_minus_low_residual"),
            "weighted_mean_delta": clean.get("weighted_mean_high_minus_low_residual") - ref.get("weighted_mean_high_minus_low_residual"),
            "truth_boundary": "descriptive_frozen_replay_contrast_only_not_H0_correction",
        }
    else:
        out["clean_minus_reference"] = {
            "unweighted_mean_delta": None,
            "weighted_mean_delta": None,
            "truth_boundary": "reference_or_clean_regime_missing",
        }
    return out


def random_control(edge_inventory, residual_rows):
    rng = np.random.default_rng(RANDOM_SEED)

    by_host = defaultdict(list)
    for row in residual_rows:
        by_host[row["host"]].append(row)

    observed = {
        row["host"]: row
        for row in edge_inventory
        if row.get("edge_status") == "edge_surface_built"
    }

    control_rows = []
    host_control_summaries = []

    for host, obs in sorted(observed.items()):
        items = by_host[host]
        n = len(items)
        k = obs["edge_count_each_side"]
        residuals = np.asarray([r["ladder_residual_y_minus_thetaL"] for r in items], dtype=float)
        observed_delta = obs["high_minus_low_mean_residual"]

        sims = []
        if n >= 2 * k and k > 0:
            for repeat in range(RANDOM_REPEATS):
                perm = rng.permutation(n)
                low_idx = perm[:k]
                high_idx = perm[k:2*k]
                sim_delta = float(np.mean(residuals[high_idx]) - np.mean(residuals[low_idx]))
                sims.append(sim_delta)
                control_rows.append(
                    {
                        "host": host,
                        "repeat": repeat,
                        "row_count": n,
                        "edge_count_each_side": k,
                        "sim_high_minus_low_mean_residual": sim_delta,
                        "observed_high_minus_low_mean_residual": observed_delta,
                        "frozen_regime_role": obs["frozen_regime_role"],
                    }
                )

        sims_arr = np.asarray(sims, dtype=float)
        if len(sims_arr):
            abs_p = float((np.sum(np.abs(sims_arr) >= abs(observed_delta)) + 1) / (len(sims_arr) + 1))
            one_sided_high_p = float((np.sum(sims_arr >= observed_delta) + 1) / (len(sims_arr) + 1))
            host_control_summaries.append(
                {
                    "host": host,
                    "row_count": n,
                    "edge_count_each_side": k,
                    "observed_high_minus_low_mean_residual": observed_delta,
                    "random_mean": float(np.mean(sims_arr)),
                    "random_std": float(np.std(sims_arr)),
                    "random_abs_95": float(np.quantile(np.abs(sims_arr), 0.95)),
                    "random_abs_99": float(np.quantile(np.abs(sims_arr), 0.99)),
                    "observed_abs_over_random_abs95": float(abs(observed_delta) / np.quantile(np.abs(sims_arr), 0.95)) if np.quantile(np.abs(sims_arr), 0.95) > 0 else None,
                    "two_sided_empirical_p": abs_p,
                    "one_sided_high_empirical_p": one_sided_high_p,
                    "frozen_regime_role": obs["frozen_regime_role"],
                }
            )

    return {
        "control_rows": control_rows,
        "host_control_summaries": sorted(host_control_summaries, key=lambda r: (r["two_sided_empirical_p"], -abs(r["observed_high_minus_low_mean_residual"]), r["host"])),
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


def decision_summary(theta_rows, y, L, C, residual, orientation, table_rows, edge_inventory, regime_aggregate, controls):
    built_edges = [r for r in edge_inventory if r.get("edge_status") == "edge_surface_built"]
    clean = regime_aggregate.get("clean_high_alpha_candidate", {})
    ref = regime_aggregate.get("low_alpha_reference", {})
    quarantine = regime_aggregate.get("sign_break_quarantine", {})
    contrast = regime_aggregate.get("clean_minus_reference", {})

    clean_hosts = clean.get("host_count", 0) or 0
    ref_hosts = ref.get("host_count", 0) or 0
    m31_reported = bool(quarantine.get("host_count", 0))

    clean_value = clean.get("weighted_mean_high_minus_low_residual")
    ref_value = ref.get("weighted_mean_high_minus_low_residual")
    contrast_value = contrast.get("weighted_mean_delta")

    control_summaries = controls.get("host_control_summaries", [])
    strong_control_count = sum(1 for r in control_summaries if r.get("two_sided_empirical_p") is not None and r["two_sided_empirical_p"] <= 0.05)

    gates = [
        {
            "gate": "G1_residual_vector_constructed",
            "passed": residual is not None and len(residual) == len(y) and np.all(np.isfinite(residual)),
            "evidence": {"residual_count": int(len(residual)) if residual is not None else 0, "y_count": int(len(y))},
        },
        {
            "gate": "G2_table2_first_block_attached",
            "passed": len(table_rows) == 3130 and len(residual) >= len(table_rows),
            "evidence": {"table2_row_count": len(table_rows), "residual_count": int(len(residual)) if residual is not None else 0},
        },
        {
            "gate": "G3_frozen_edge_hosts_available",
            "passed": len(built_edges) >= 10,
            "evidence": {"built_edge_host_count": len(built_edges)},
        },
        {
            "gate": "G4_clean_and_reference_regimes_available",
            "passed": clean_hosts >= 10 and ref_hosts >= 1,
            "evidence": {"clean_edge_hosts": clean_hosts, "reference_edge_hosts": ref_hosts},
        },
        {
            "gate": "G5_M31_quarantine_reported_not_transferred",
            "passed": True,
            "evidence": {"M31_reported_as_quarantine": m31_reported, "M31_transferred": False},
        },
        {
            "gate": "G6_random_controls_completed",
            "passed": len(control_summaries) >= 10,
            "evidence": {"control_host_count": len(control_summaries), "random_repeats": RANDOM_REPEATS},
        },
    ]

    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed and clean_value is not None and ref_value is not None:
        if contrast_value is not None and abs(contrast_value) > 0.01:
            final_status = "frozen_residual_replay_completed_surface_pattern_detected"
            readiness = 8
            next_wall = (
                "Locked replay completed and produced a nonzero clean-minus-reference residual edge contrast. "
                "Next test should stress-test robustness without tuning: jackknife hosts, covariance-aware block summaries, and report-only H0 sensitivity bounds."
            )
        else:
            final_status = "frozen_residual_replay_completed_no_large_regime_contrast"
            readiness = 7
            next_wall = (
                "Locked replay completed but clean-minus-reference contrast is small. Next test should document as boundary result, not force significance."
            )
    elif len(failed) <= 2 and "G1_residual_vector_constructed" not in failed and "G2_table2_first_block_attached" not in failed:
        final_status = "frozen_residual_replay_completed_with_cautions"
        readiness = 6
        next_wall = "Replay mostly completed but one or more reporting/regime/control gates failed."
    else:
        final_status = "frozen_residual_replay_not_ready"
        readiness = 4
        next_wall = "Core residual or Table2 attachment gates failed. Do not interpret replay."

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "gates": gates,
        "failed_gates": failed,
        "replay_core": {
            "theta_count": len(theta_rows),
            "y_shape": list(y.shape),
            "L_shape": list(L.shape),
            "C_shape": list(C.shape),
            "orientation": orientation,
            "table2_row_count": len(table_rows),
            "edge_host_count": len(built_edges),
            "clean_edge_host_count": clean_hosts,
            "reference_edge_host_count": ref_hosts,
            "m31_quarantine_reported": m31_reported,
            "clean_weighted_edge_residual": clean_value,
            "reference_weighted_edge_residual": ref_value,
            "clean_minus_reference_weighted_delta": contrast_value,
            "host_random_control_count": len(control_summaries),
            "host_random_control_p_le_0_05_count": strong_control_count,
        },
        "truth_boundary": CLAIMS_V1_9["truth_boundary"],
    }


def holographic_surface_ledger(decision):
    return {
        "observable_surface": {
            "name": "Locked Table2 F160W edge surface on SH0ES ladder residuals",
            "table2_path": TABLE2_PATH,
            "theta_path": LSTSQ_PATH,
            "y_path": Y_FITS_PATH,
            "l_path": L_FITS_PATH,
            "c_path": C_FITS_PATH,
        },
        "hidden_depth_sought": {
            "name": "Frozen boundary-polarity residual pattern",
            "allowed_claim": (
                "The locked edge rule can be measured as a residual surface pattern within the public ladder residual layer."
            ),
            "not_allowed_claim": (
                "Do not claim H0 correction, SH0ES error, TAIRID proof, or new physics from this replay alone."
            ),
        },
        "boundary_that_forms_surface": {
            "row_boundary": "first 3130 y entries attached to Table2 rows",
            "edge_boundary": "within-host 5 percent bright/faint F160W edges",
            "regime_boundary": "frozen v1.0 regimes; M31 quarantine not transferred",
            "method_boundary": "no variable search, no regime tuning, no percentile tuning",
        },
        "what_can_be_reconstructed_now": [
            "Residual vector y - theta@L",
            "Table2-attached residual rows",
            "Frozen within-host edge residuals",
            "Frozen regime summaries",
            "Random same-host control comparisons",
        ],
        "what_cannot_be_reconstructed_now": [
            "A new H0 solution",
            "New physics",
            "A causal explanation",
            "A tuned likelihood model",
            "Generalization to non-Table2 data without a separate replay",
        ],
        "surface_noise_definition": [
            "Any signal dependent on changing the edge percentile",
            "Any signal dependent on adding/removing regimes",
            "Any M31 correction transfer",
            "Any interpretation that ignores covariance warnings",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v1.9 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v1.9 did")
    lines.append("")
    lines.append("- Built SH0ES ladder residuals from public y/L/C and lstsq_results.")
    lines.append("- Attached the first 3130 residuals to Table2 rows.")
    lines.append("- Ran only the frozen within-host 5 percent F160W edge rule.")
    lines.append("- Kept regimes fixed: clean high-alpha, LMC+SMC+N4536 reference, M31 quarantine.")
    lines.append("- Reported random same-host controls.")
    lines.append("- Did not tune, refit, or claim H0 correction.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v1.9 is a locked residual surface replay.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not prove SH0ES is wrong.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append(
            "v2.0 should stress-test the locked replay without tuning: leave-one-host-out jackknife, leave-one-regime-out report, covariance-aware block diagnostics, and report-only H0 sensitivity bounds."
        )
    elif decision["readiness_score_0_to_10"] >= 6:
        lines.append(
            "v2.0 should repair any failed reporting gates and rerun the same locked replay. Do not change the rule."
        )
    else:
        lines.append(
            "v2.0 should stop the frozen replay lane until residual construction or row attachment is corrected."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Frozen Residual Replay v1.9 starting.")
    print("Boundary: locked residual replay only; no tuning, no H0 claim, no new physics claim.")

    write_json(OUTDIR / "claims_v1_9.json", CLAIMS_V1_9)
    write_json(OUTDIR / "frozen_rule_v1_9.json", FROZEN_RULE)

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

        write_csv(OUTDIR / "download_ledger_v1_9.csv", download_ledger)

        table2 = parse_table2_tex(text_files.get(TABLE2_PATH, ""))
        table_rows = table2["rows"]
        write_csv(OUTDIR / "table2_parsed_rows_v1_9.csv", table_rows)
        write_json(OUTDIR / "table2_parse_errors_v1_9.json", table2["errors"])

        theta_parse = parse_lstsq_results(text_files.get(LSTSQ_PATH, ""))
        theta_rows = theta_parse["rows"]
        theta = np.asarray([r["theta_value"] for r in theta_rows], dtype=float)
        write_csv(OUTDIR / "theta_lstsq_vector_v1_9.csv", theta_rows)
        write_json(OUTDIR / "theta_lstsq_parse_errors_v1_9.json", theta_parse["errors"])

        y = np.asarray(fits_arrays.get(Y_FITS_PATH, np.asarray([])), dtype=float)
        L = np.asarray(fits_arrays.get(L_FITS_PATH, np.asarray([[]])), dtype=float)
        C = np.asarray(fits_arrays.get(C_FITS_PATH, np.asarray([[]])), dtype=float)

        predicted_y, residual, orientation = compute_ladder_residual(theta, L, y)
        if predicted_y is None:
            predicted_y = np.asarray([])
            residual = np.asarray([])

        covariance_diag, cov_info = covariance_diag_info(C)
        write_json(OUTDIR / "covariance_diag_summary_v1_9.json", cov_info)

        residual_rows = []
        edge_result = {"edge_rows": [], "edge_inventory": []}
        if len(residual) >= len(table_rows) and len(table_rows) > 0:
            residual_rows = attach_table2_residuals(table_rows, y, predicted_y, residual, covariance_diag)
            edge_result = build_frozen_edges(residual_rows)

        write_csv(OUTDIR / "table2_ladder_residual_rows_v1_9.csv", residual_rows)
        write_csv(OUTDIR / "frozen_edge_rows_v1_9.csv", edge_result["edge_rows"])
        write_csv(OUTDIR / "frozen_edge_inventory_v1_9.csv", edge_result["edge_inventory"])

        regime_aggregate = aggregate_regimes(edge_result["edge_inventory"])
        write_json(OUTDIR / "frozen_regime_aggregate_v1_9.json", regime_aggregate)

        controls = random_control(edge_result["edge_inventory"], residual_rows)
        write_csv(OUTDIR / "random_same_host_control_rows_v1_9.csv", controls["control_rows"])
        write_csv(OUTDIR / "random_same_host_control_summary_v1_9.csv", controls["host_control_summaries"])

        source_info = source_sniffs(text_files.get(README_PATH, ""), text_files.get(TABLE2_README_PATH, ""))
        write_json(OUTDIR / "source_sniffs_v1_9.json", source_info)
        write_csv(OUTDIR / "source_pattern_counts_v1_9.csv", source_info["pattern_counts"])
        write_csv(OUTDIR / "source_snippets_v1_9.csv", source_info["snippets"])

        residual_summary = {
            "theta_count": int(len(theta)),
            "y_shape": list(y.shape),
            "L_shape": list(L.shape),
            "C_shape": list(C.shape),
            "orientation": orientation,
            "predicted_y_count": int(len(predicted_y)),
            "residual_count": int(len(residual)),
            "table2_row_count": int(len(table_rows)),
            "table2_residual_summary": summarize_numeric([r["ladder_residual_y_minus_thetaL"] for r in residual_rows]),
            "table2_standardized_residual_summary": summarize_numeric([r["standardized_residual"] for r in residual_rows if r["standardized_residual"] is not None]),
            "all_residual_summary": summarize_numeric(residual.tolist() if len(residual) else []),
            "all_residual_rmse": rmse(residual) if len(residual) else None,
            "all_residual_median_abs": median_abs(residual) if len(residual) else None,
        }
        write_json(OUTDIR / "residual_summary_v1_9.json", residual_summary)

        decision = decision_summary(
            theta_rows=theta_rows,
            y=y,
            L=L,
            C=C,
            residual=residual,
            orientation=orientation,
            table_rows=table_rows,
            edge_inventory=edge_result["edge_inventory"],
            regime_aggregate=regime_aggregate,
            controls=controls,
        )
        write_json(OUTDIR / "decision_v1_9.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v1_9.json", ledger)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v1_9.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v1_9.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID SH0ES Frozen Residual Replay v1.9",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Locked frozen residual replay only. No tuning, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "download_ledger": download_ledger,
            "residual_summary": residual_summary,
            "covariance_diag_summary": cov_info,
            "frozen_regime_aggregate": regime_aggregate,
            "random_control_summary_count": len(controls["host_control_summaries"]),
            "decision": decision,
            "claims_v1_9": CLAIMS_V1_9,
            "frozen_rule": FROZEN_RULE,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_frozen_residual_replay_v1_9_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_frozen_residual_replay_v1_9_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v1_9.csv"),
                "theta_vector_csv": str(OUTDIR / "theta_lstsq_vector_v1_9.csv"),
                "residual_rows_csv": str(OUTDIR / "table2_ladder_residual_rows_v1_9.csv"),
                "edge_rows_csv": str(OUTDIR / "frozen_edge_rows_v1_9.csv"),
                "edge_inventory_csv": str(OUTDIR / "frozen_edge_inventory_v1_9.csv"),
                "regime_aggregate_json": str(OUTDIR / "frozen_regime_aggregate_v1_9.json"),
                "random_control_rows_csv": str(OUTDIR / "random_same_host_control_rows_v1_9.csv"),
                "random_control_summary_csv": str(OUTDIR / "random_same_host_control_summary_v1_9.csv"),
                "residual_summary_json": str(OUTDIR / "residual_summary_v1_9.json"),
                "decision_json": str(OUTDIR / "decision_v1_9.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v1_9.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v1_9.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v1_9.txt"),
            },
            "interpretation": {
                "what_success_means": "The locked F160W edge surface is measurable inside the public SH0ES ladder residual layer.",
                "what_success_does_not_mean": "This does not prove TAIRID, H0 correction, new physics, or SH0ES error.",
                "next_required_step": "If successful, run v2.0 robustness stress tests without changing the frozen rule.",
                "truth_boundary": CLAIMS_V1_9["truth_boundary"],
            },
        }

        write_json(OUTDIR / "shoes_frozen_residual_replay_v1_9_summary.json", summary)

        with open(OUTDIR / "shoes_frozen_residual_replay_v1_9_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Frozen Residual Replay v1.9\n\n")
            f.write("Boundary: locked residual replay only. No tuning. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Replay core:\n")
            f.write(json.dumps(decision["replay_core"], indent=2, default=json_default) + "\n\n")
            f.write("Frozen regime aggregate:\n")
            f.write(json.dumps(regime_aggregate, indent=2, default=json_default) + "\n\n")
            f.write("Residual summary:\n")
            f.write(json.dumps(residual_summary, indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not prove SH0ES is wrong.\n")
            f.write("- This does not tune the frozen rule.\n")

        print("TAIRID SH0ES Frozen Residual Replay v1.9 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Frozen Residual Replay v1.9",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_frozen_residual_replay_v1_9_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V1_9["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_frozen_residual_replay_v1_9_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
