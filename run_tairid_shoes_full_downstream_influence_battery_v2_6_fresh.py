#!/usr/bin/env python3
"""
TAIRID SH0ES Full Downstream Influence Battery v2.6 Fresh

Why this test exists:
v2.5 proved the bridge exists from the locked SH0ES Table2 F160W residual
host-edge surface to Pantheon+SH0ES downstream calibrator rows. But v2.5 stopped
before validation and only reported weak descriptive downstream signal.

v2.6 is the first larger bundled downstream battery. It uses one workflow to run
many predeclared checks, so we stop wasting setup time on tiny 30-second gates.

Core question:
    Does the frozen Table2 host-edge residual surface carry a measurable
    downstream calibrator-distance influence signal, beyond simple controls and
    permutation nulls?

This test DOES:
    1. Rebuild the locked Table2 residual surface.
    2. Rebuild host-level F160W edge summaries.
    3. Rebuild the downstream calibrator bridge.
    4. Run predeclared weighted regressions.
    5. Run clean/reference regime contrasts.
    6. Run leave-one-host-out sensitivity.
    7. Run bootstrap confidence intervals.
    8. Run random host-label permutation nulls.
    9. Run regime-label permutation nulls.
    10. Run M31 quarantine/report-only check.
    11. Run predeclared false-positive controls.
    12. Produce an effect-size summary and pass/fail decision.
    13. Write a handoff for the next thread.

This test DOES NOT:
    - validate TAIRID,
    - claim H0 correction,
    - claim new physics,
    - claim SH0ES is wrong,
    - tune the frozen rule,
    - change the edge percentile,
    - refit the distance ladder,
    - use M31 as a correction,
    - promote a result based on one lucky statistic.

Truth boundary:
v2.6 is a predeclared downstream influence battery. A positive result would be
evidence of a downstream association worth deeper testing, not proof. A weak or
failed result must be reported as a limitation of the current lane.
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

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


OUTDIR = Path("tairid_shoes_full_downstream_influence_battery_v2_6_fresh_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO = "PantheonPlusSH0ES/DataRelease"
BRANCH_CANDIDATES = ["main", "master"]

TABLE2_PATH = "SH0ES_Data/table2.tex"
LSTSQ_PATH = "SH0ES_Data/lstsq_results.txt"
Y_FITS_PATH = "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits"
L_FITS_PATH = "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits"
C_FITS_PATH = "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits"

DISTANCE_TABLE_PATH = "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
DISTANCE_COV_PATH = "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"
REDSHIFT_METADATA_PATH = "Pantheon+_Data/1_DATA/all_redshifts_PVs.csv"

EDGE_PERCENTILE = 5.0
MIN_HOST_ROWS_FOR_EDGE = 20
RANDOM_SEED = 112221
BOOTSTRAP_N = 1000
PERMUTATION_N = 2000

LOW_ALPHA_REFERENCE_HOSTS = {"LMC", "SMC", "N4536"}
SIGN_BREAK_QUARANTINE_HOSTS = {"M31"}

CLAIMS_V2_6 = {
    "battery_name": "TAIRID SH0ES Full Downstream Influence Battery v2.6 Fresh",
    "scope": "Predeclared downstream influence battery after v2.5 bridge audit",
    "primary_question": (
        "Does the frozen Table2 F160W residual host-edge surface carry a measurable downstream calibrator-distance "
        "influence signal beyond simple controls and permutation nulls?"
    ),
    "truth_boundary": (
        "This is an association battery only. It does not validate TAIRID, H0 correction, or new physics."
    ),
}

FROZEN_RULE_CARRY_FORWARD = {
    "locked_variable": "Table2 F160W/f160w",
    "locked_edge_rule": "within-host high 5% F160W/faint edge minus within-host low 5% F160W/bright edge",
    "locked_regimes": {
        "clean_high_alpha_candidate": "all active Table2 hosts except LMC, SMC, N4536, and M31",
        "low_alpha_reference": ["LMC", "SMC", "N4536"],
        "sign_break_quarantine": ["M31"],
    },
    "hard_boundary": [
        "Do not tune the frozen rule in v2.6.",
        "Do not change EDGE_PERCENTILE.",
        "Do not fit a new distance-ladder model.",
        "Do not transfer M31 correction; M31 remains quarantine/report-only.",
        "Do not claim H0 correction or new physics.",
    ],
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
        headers={"User-Agent": "TAIRID-v2.6-fresh-full-downstream-influence-battery"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
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
    if "404: not found" in text or "not found" in text[:120]:
        return "not_found_payload"
    if len(data) < 8192 and all((32 <= b <= 126) or b in (9, 10, 13) for b in data):
        return "small_text_payload"
    return "unknown_binary_or_text_payload"


def fetch_bytes_for_path(repo_path, prefer_media=False):
    errors = []
    for branch in BRANCH_CANDIDATES:
        urls = (
            [media_url(branch, repo_path), raw_url(branch, repo_path)]
            if prefer_media
            else [raw_url(branch, repo_path), media_url(branch, repo_path)]
        )
        for url in urls:
            try:
                data, final_url, content_type = fetch_url_bytes(url)
                kind = payload_kind(data)
                if kind in {"html_payload", "not_found_payload"}:
                    errors.append(
                        {
                            "branch": branch,
                            "url": url,
                            "final_url": final_url,
                            "content_type": content_type,
                            "payload_kind": kind,
                        }
                    )
                    continue
                if kind == "git_lfs_pointer" and "raw.githubusercontent.com" in url:
                    errors.append(
                        {
                            "branch": branch,
                            "url": url,
                            "final_url": final_url,
                            "content_type": content_type,
                            "payload_kind": kind,
                            "note": "trying media URL",
                        }
                    )
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


def to_float(value):
    try:
        out = float(str(value).replace("D", "E"))
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


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
        ra = to_float(parts[1])
        dec = to_float(parts[2])
        period = to_float(parts[4])
        v_i = to_float(parts[5])
        sigma_v_i = to_float(parts[6])
        f160w = to_float(parts[7])
        sigma_f160w = to_float(parts[8])
        metal = to_float(parts[9])
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


def parse_lstsq_results(text):
    rows = []
    errors = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        value = to_float(tokens[0]) if tokens else None
        sigma = to_float(tokens[1]) if len(tokens) > 1 else None
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


def read_primary_fits_array(data_bytes):
    with fits.open(io.BytesIO(data_bytes), memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=float)


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


def weighted_mean(values, weights):
    pairs = [
        (float(v), float(w))
        for v, w in zip(values, weights)
        if v is not None and w is not None and math.isfinite(float(v)) and math.isfinite(float(w)) and float(w) > 0
    ]
    if not pairs:
        return None
    vals = np.asarray([p[0] for p in pairs], dtype=float)
    wts = np.asarray([p[1] for p in pairs], dtype=float)
    return float(np.average(vals, weights=wts))


def pearson_corr(x, y):
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return {"n": len(pairs), "r": None}
    a = np.asarray([p[0] for p in pairs], dtype=float)
    b = np.asarray([p[1] for p in pairs], dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return {"n": len(pairs), "r": None}
    return {"n": len(pairs), "r": float(np.corrcoef(a, b)[0, 1])}


def build_host_f160w_edges(residual_rows):
    by_host = defaultdict(list)
    for row in residual_rows:
        by_host[row["host"]].append(row)

    edge_rows = []
    host_summary = []

    for host, rows in sorted(by_host.items()):
        rows = sorted(rows, key=lambda r: r["f160w"])
        n = len(rows)
        role = frozen_role_for_host(host)

        if n < MIN_HOST_ROWS_FOR_EDGE:
            host_summary.append(
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
        low_side = rows[:k]
        high_side = rows[-k:]

        for side_name, selected in [("low_f160w_bright_edge", low_side), ("high_f160w_faint_edge", high_side)]:
            for r in selected:
                edge_row = dict(r)
                edge_row["edge_side"] = side_name
                edge_row["edge_percentile"] = EDGE_PERCENTILE
                edge_row["host_row_count"] = n
                edge_row["edge_count_each_side"] = k
                edge_rows.append(edge_row)

        low_resid = [r["ladder_residual_y_minus_thetaL"] for r in low_side]
        high_resid = [r["ladder_residual_y_minus_thetaL"] for r in high_side]
        low_f160w = [r["f160w"] for r in low_side]
        high_f160w = [r["f160w"] for r in high_side]
        all_period = [r["period"] for r in rows]
        all_color = [r["v_i"] for r in rows]
        all_metal = [r["metal_minus_8_69"] for r in rows]

        host_summary.append(
            {
                "host": host,
                "row_count": n,
                "edge_status": "edge_surface_built",
                "edge_count_each_side": k,
                "frozen_regime_role": role,
                "low_edge_mean_f160w": float(np.mean(low_f160w)),
                "high_edge_mean_f160w": float(np.mean(high_f160w)),
                "high_minus_low_mean_f160w": float(np.mean(high_f160w) - np.mean(low_f160w)),
                "low_edge_mean_residual": float(np.mean(low_resid)),
                "high_edge_mean_residual": float(np.mean(high_resid)),
                "high_minus_low_mean_residual": float(np.mean(high_resid) - np.mean(low_resid)),
                "low_edge_median_residual": float(np.median(low_resid)),
                "high_edge_median_residual": float(np.median(high_resid)),
                "high_minus_low_median_residual": float(np.median(high_resid) - np.median(low_resid)),
                "host_mean_period": float(np.mean(all_period)),
                "host_mean_color_v_i": float(np.mean(all_color)),
                "host_mean_metal": float(np.mean(all_metal)),
                "host_f160w_std": float(np.std([r["f160w"] for r in rows])),
                "host_period_std": float(np.std(all_period)),
            }
        )

    return {
        "edge_rows": edge_rows,
        "host_summary": sorted(host_summary, key=lambda r: (-r["row_count"], r["host"])),
    }


def parse_distance_table(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"columns": [], "rows": []}
    columns = lines[0].split()
    rows = []
    for index, line in enumerate(lines[1:], start=0):
        parts = line.split()
        if len(parts) < len(columns):
            continue
        row = dict(zip(columns, parts[:len(columns)]))
        row["distance_row_index_0_based"] = index
        for key in [
            "zHD", "zHDERR", "zCMB", "zCMBERR", "zHEL", "zHELERR",
            "m_b_corr", "m_b_corr_err_DIAG", "MU_SH0ES", "MU_SH0ES_ERR_DIAG",
            "CEPH_DIST", "IS_CALIBRATOR", "USED_IN_SH0ES_HF",
            "RA", "DEC", "HOST_RA", "HOST_DEC", "VPEC", "VPECERR", "HOST_LOGMASS",
        ]:
            if key in row:
                val = to_float(row[key])
                if val is not None:
                    row[key] = val
        rows.append(row)
    return {"columns": columns, "rows": rows}


def parse_redshift_metadata(text):
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def normalize_host_name(host_name):
    if host_name is None:
        return None
    text = str(host_name).strip()
    if not text:
        return None
    u = text.upper()
    u = u.replace("MESSIER", "M")
    u = u.replace("MRK", "M")
    u = u.replace("MARKARIAN", "M")
    u = u.replace("NGC", "N")
    u = u.replace("UGC", "U")
    u = re.sub(r"[^A-Z0-9]", "", u)
    match = re.match(r"^([MNU])0+([0-9]+[A-Z]?)$", u)
    if match:
        u = match.group(1) + match.group(2)
    return u if u else None


def build_table_host_normalizer(table_hosts):
    norm_map = {}
    for host in sorted(table_hosts):
        norm = normalize_host_name(host)
        if norm:
            norm_map[norm] = host
    return norm_map


def match_normalized_host_to_table(host_name, norm_map):
    norm = normalize_host_name(host_name)
    tried = []
    if norm is None:
        return None, None, tried
    candidates = [norm]
    if re.match(r"^[MNU]\d+[A-Z]$", norm):
        candidates.append(norm[:-1])
    if re.match(r"^[MNU]\d+$", norm):
        candidates.append(norm + "A")
    manual = {
        "N105": "N105A",
        "N976": "N976A",
    }
    if norm in manual:
        candidates.insert(0, manual[norm])
    for candidate in candidates:
        tried.append(candidate)
        if candidate in norm_map:
            return norm_map[candidate], norm, tried
        if candidate in norm_map.values():
            return candidate, norm, tried
    return None, norm, tried


def parse_cov_dimension(text):
    tokens = text.split()
    first_int = None
    if tokens:
        try:
            first_int = int(float(tokens[0]))
        except Exception:
            first_int = None
    expected = first_int * first_int if first_int is not None else None
    return {
        "first_token_as_int": first_int,
        "payload_element_count_after_first": len(tokens) - 1 if tokens else 0,
        "expected_square_elements": expected,
        "has_expected_square_payload": expected is not None and len(tokens) - 1 == expected,
    }


def build_calibrator_bridge(distance_rows, redshift_rows, host_edge_summary):
    redshift_by_snid = {}
    for row in redshift_rows:
        snid = str(row.get("SNID", "")).strip().lower()
        iauc = str(row.get("IAUC", "")).strip().lower()
        if snid:
            redshift_by_snid[snid] = row
        if iauc and iauc not in redshift_by_snid:
            redshift_by_snid[iauc] = row

    table_hosts = sorted(set(row["host"] for row in host_edge_summary))
    norm_map = build_table_host_normalizer(table_hosts)
    edge_by_host = {row["host"]: row for row in host_edge_summary}

    calibrator_rows = []
    unmatched_rows = []

    for row in distance_rows:
        is_calibrator = int(row.get("IS_CALIBRATOR", 0)) == 1 if row.get("IS_CALIBRATOR") is not None else False
        if not is_calibrator:
            continue

        cid = str(row.get("CID", "")).strip()
        meta = redshift_by_snid.get(cid.lower(), {})
        host_name = meta.get("host", "")

        table_host, normalized_host, tried = match_normalized_host_to_table(host_name, norm_map)
        mu = row.get("MU_SH0ES")
        ceph = row.get("CEPH_DIST")
        mu_err = row.get("MU_SH0ES_ERR_DIAG")
        mu_minus_ceph = mu - ceph if isinstance(mu, float) and isinstance(ceph, float) and ceph > 0 else None
        weight = 1.0 / (mu_err * mu_err) if isinstance(mu_err, float) and mu_err > 0 else None

        out = {
            "CID": cid,
            "IDSURVEY": row.get("IDSURVEY"),
            "distance_row_index_0_based": row.get("distance_row_index_0_based"),
            "metadata_host": host_name,
            "metadata_host_normalized": normalized_host,
            "table2_host_match": table_host,
            "host_match_tried": "|".join(tried),
            "host_bridge_status": "matched_to_table2_host" if table_host else "no_table2_host_match",
            "zHD": row.get("zHD"),
            "zCMB": row.get("zCMB"),
            "MU_SH0ES": mu,
            "MU_SH0ES_ERR_DIAG": mu_err,
            "CEPH_DIST": ceph,
            "mu_minus_cepheid_distance": mu_minus_ceph,
            "inverse_muerr2_weight": weight,
            "USED_IN_SH0ES_HF": row.get("USED_IN_SH0ES_HF"),
            "distance_HOST_LOGMASS": row.get("HOST_LOGMASS"),
            "metadata_zHD": to_float(meta.get("zHD")) if meta else None,
            "metadata_PV": to_float(meta.get("PV")) if meta else None,
        }

        if table_host and table_host in edge_by_host:
            edge = edge_by_host[table_host]
            out["edge_status"] = edge.get("edge_status")
            out["frozen_regime_role"] = edge.get("frozen_regime_role")
            out["host_row_count"] = edge.get("row_count")
            out["host_edge_count_each_side"] = edge.get("edge_count_each_side")
            out["host_high_minus_low_f160w"] = edge.get("high_minus_low_mean_f160w")
            out["host_high_minus_low_residual"] = edge.get("high_minus_low_mean_residual")
            out["host_high_minus_low_median_residual"] = edge.get("high_minus_low_median_residual")
            out["host_mean_period"] = edge.get("host_mean_period")
            out["host_mean_color_v_i"] = edge.get("host_mean_color_v_i")
            out["host_mean_metal"] = edge.get("host_mean_metal")
            out["host_f160w_std"] = edge.get("host_f160w_std")
            out["host_period_std"] = edge.get("host_period_std")

        calibrator_rows.append(out)
        if not table_host:
            unmatched_rows.append(out)

    return {
        "calibrator_rows": calibrator_rows,
        "unmatched_rows": unmatched_rows,
    }


def aggregate_host_bridge(calibrator_rows):
    by_host = defaultdict(list)
    for row in calibrator_rows:
        host = row.get("table2_host_match")
        if host and row.get("edge_status") == "edge_surface_built":
            by_host[host].append(row)

    host_rows = []
    for host, rows in sorted(by_host.items()):
        unique_cids = sorted(set(r["CID"] for r in rows))
        mu_resids = [r.get("mu_minus_cepheid_distance") for r in rows]
        weights = [r.get("inverse_muerr2_weight") for r in rows]
        used_hf = [r.get("USED_IN_SH0ES_HF") for r in rows if r.get("USED_IN_SH0ES_HF") is not None]
        exemplar = rows[0]
        host_rows.append(
            {
                "table2_host": host,
                "metadata_hosts_seen": "|".join(sorted(set(str(r.get("metadata_host", "")) for r in rows))),
                "unique_cid_count": len(unique_cids),
                "unique_cids": "|".join(unique_cids),
                "calibrator_row_count": len(rows),
                "frozen_regime_role": exemplar.get("frozen_regime_role"),
                "host_row_count": exemplar.get("host_row_count"),
                "host_high_minus_low_f160w": exemplar.get("host_high_minus_low_f160w"),
                "host_high_minus_low_residual": exemplar.get("host_high_minus_low_residual"),
                "host_high_minus_low_median_residual": exemplar.get("host_high_minus_low_median_residual"),
                "host_mean_period": exemplar.get("host_mean_period"),
                "host_mean_color_v_i": exemplar.get("host_mean_color_v_i"),
                "host_mean_metal": exemplar.get("host_mean_metal"),
                "host_f160w_std": exemplar.get("host_f160w_std"),
                "host_period_std": exemplar.get("host_period_std"),
                "mean_mu_minus_cepheid_distance": summarize_numeric(mu_resids)["mean"],
                "median_mu_minus_cepheid_distance": summarize_numeric(mu_resids)["median"],
                "weighted_mean_mu_minus_cepheid_distance": weighted_mean(mu_resids, weights),
                "mean_MU_SH0ES": summarize_numeric([r.get("MU_SH0ES") for r in rows])["mean"],
                "mean_CEPH_DIST": summarize_numeric([r.get("CEPH_DIST") for r in rows])["mean"],
                "mean_zHD": summarize_numeric([r.get("zHD") for r in rows])["mean"],
                "mean_distance_HOST_LOGMASS": summarize_numeric([r.get("distance_HOST_LOGMASS") for r in rows])["mean"],
                "mean_mu_err_diag": summarize_numeric([r.get("MU_SH0ES_ERR_DIAG") for r in rows])["mean"],
                "used_in_shoes_hf_values": "|".join(sorted(set(str(v) for v in used_hf))),
            }
        )
    return host_rows


def design_matrix(rows, predictors, include_intercept=True):
    y = []
    w = []
    X = []
    kept = []
    for row in rows:
        target = row.get("weighted_mean_mu_minus_cepheid_distance")
        weight = row.get("calibrator_row_count")
        vals = []
        ok = target is not None and weight is not None and weight > 0
        for p in predictors:
            val = row.get(p)
            if val is None or not math.isfinite(float(val)):
                ok = False
                break
            vals.append(float(val))
        if ok:
            y.append(float(target))
            w.append(float(weight))
            X.append(([1.0] if include_intercept else []) + vals)
            kept.append(row)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(w, dtype=float), kept


def weighted_ols(rows, predictors, model_name):
    X, y, w, kept = design_matrix(rows, predictors, include_intercept=True)
    n = int(len(y))
    p = int(X.shape[1]) if n else 0
    if n <= p or n < 4:
        return {
            "model_name": model_name,
            "predictors": predictors,
            "n": n,
            "status": "not_enough_rows",
        }

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        yhat = X @ beta
        resid = y - yhat
        weighted_y_mean = np.average(y, weights=w)
        ss_res = float(np.sum(w * resid * resid))
        ss_tot = float(np.sum(w * (y - weighted_y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

        dof = n - p
        sigma2 = ss_res / dof if dof > 0 else None
        xtwx_inv = np.linalg.pinv(X.T @ (w[:, None] * X))
        se = np.sqrt(np.diag(xtwx_inv) * sigma2) if sigma2 is not None else np.full(p, np.nan)
        tvals = beta / se

        names = ["intercept"] + predictors
        coef_rows = []
        for name, b, s, t in zip(names, beta, se, tvals):
            coef_rows.append(
                {
                    "model_name": model_name,
                    "term": name,
                    "estimate": float(b),
                    "std_error": float(s) if math.isfinite(float(s)) else None,
                    "t_value": float(t) if math.isfinite(float(t)) else None,
                }
            )

        return {
            "model_name": model_name,
            "predictors": predictors,
            "n": n,
            "status": "ok",
            "r2": r2,
            "weighted_sse": ss_res,
            "coef_rows": coef_rows,
            "primary_slope": coef_rows[1]["estimate"] if len(coef_rows) > 1 else None,
            "primary_t": coef_rows[1]["t_value"] if len(coef_rows) > 1 else None,
            "kept_hosts": [r["table2_host"] for r in kept],
            "truth_boundary": "weighted_OLS_association_only_not_causal_not_H0_correction",
        }
    except Exception as exc:
        return {
            "model_name": model_name,
            "predictors": predictors,
            "n": n,
            "status": "failed",
            "error": repr(exc),
        }


def clean_reference_contrast(host_rows):
    clean = [r for r in host_rows if r.get("frozen_regime_role") == "clean_high_alpha_candidate"]
    ref = [r for r in host_rows if r.get("frozen_regime_role") == "low_alpha_reference"]
    quarantine = [r for r in host_rows if r.get("frozen_regime_role") == "sign_break_quarantine"]

    clean_mu = weighted_mean(
        [r.get("weighted_mean_mu_minus_cepheid_distance") for r in clean],
        [r.get("calibrator_row_count") for r in clean],
    )
    ref_mu = weighted_mean(
        [r.get("weighted_mean_mu_minus_cepheid_distance") for r in ref],
        [r.get("calibrator_row_count") for r in ref],
    )

    clean_edge = weighted_mean(
        [r.get("host_high_minus_low_residual") for r in clean],
        [r.get("calibrator_row_count") for r in clean],
    )
    ref_edge = weighted_mean(
        [r.get("host_high_minus_low_residual") for r in ref],
        [r.get("calibrator_row_count") for r in ref],
    )

    return {
        "clean_host_count": len(clean),
        "reference_host_count": len(ref),
        "quarantine_host_count": len(quarantine),
        "clean_weighted_mean_mu_minus_cepheid": clean_mu,
        "reference_weighted_mean_mu_minus_cepheid": ref_mu,
        "clean_minus_reference_mu_difference": clean_mu - ref_mu if clean_mu is not None and ref_mu is not None else None,
        "clean_weighted_mean_edge_residual": clean_edge,
        "reference_weighted_mean_edge_residual": ref_edge,
        "clean_minus_reference_edge_difference": clean_edge - ref_edge if clean_edge is not None and ref_edge is not None else None,
        "truth_boundary": "regime_contrast_association_only",
    }


def leave_one_host_out(host_rows):
    rows = []
    for held in host_rows:
        subset = [r for r in host_rows if r["table2_host"] != held["table2_host"]]
        corr = pearson_corr(
            [r.get("host_high_minus_low_residual") for r in subset],
            [r.get("weighted_mean_mu_minus_cepheid_distance") for r in subset],
        )
        model = weighted_ols(subset, ["host_high_minus_low_residual"], "loho_edge_only")
        contrast = clean_reference_contrast(subset)
        rows.append(
            {
                "held_out_host": held["table2_host"],
                "held_out_role": held.get("frozen_regime_role"),
                "n_remaining": len(subset),
                "corr_r": corr.get("r"),
                "corr_n": corr.get("n"),
                "edge_only_slope": model.get("primary_slope"),
                "edge_only_t": model.get("primary_t"),
                "clean_minus_reference_mu_difference": contrast.get("clean_minus_reference_mu_difference"),
            }
        )
    return rows


def bootstrap_battery(host_rows, rng):
    rows = []
    n = len(host_rows)
    if n < 5:
        return {"iterations": 0, "summary": {}, "rows": rows}
    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        sample = [host_rows[int(j)] for j in idx]
        corr = pearson_corr(
            [r.get("host_high_minus_low_residual") for r in sample],
            [r.get("weighted_mean_mu_minus_cepheid_distance") for r in sample],
        )
        model = weighted_ols(sample, ["host_high_minus_low_residual"], "bootstrap_edge_only")
        contrast = clean_reference_contrast(sample)
        rows.append(
            {
                "iteration": i,
                "corr_r": corr.get("r"),
                "edge_only_slope": model.get("primary_slope"),
                "edge_only_t": model.get("primary_t"),
                "clean_minus_reference_mu_difference": contrast.get("clean_minus_reference_mu_difference"),
            }
        )

    def ci(values):
        arr = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
        if len(arr) == 0:
            return {"count": 0, "mean": None, "p025": None, "p500": None, "p975": None, "sign_positive_fraction": None}
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "p025": float(np.percentile(arr, 2.5)),
            "p500": float(np.percentile(arr, 50.0)),
            "p975": float(np.percentile(arr, 97.5)),
            "sign_positive_fraction": float(np.mean(arr > 0)),
        }

    summary = {
        "corr_r": ci([r.get("corr_r") for r in rows]),
        "edge_only_slope": ci([r.get("edge_only_slope") for r in rows]),
        "clean_minus_reference_mu_difference": ci([r.get("clean_minus_reference_mu_difference") for r in rows]),
    }
    return {"iterations": BOOTSTRAP_N, "summary": summary, "rows": rows}


def permutation_battery(host_rows, rng):
    rows = []
    n = len(host_rows)
    if n < 5:
        return {"iterations": 0, "summary": {}, "rows": rows}

    observed_corr = pearson_corr(
        [r.get("host_high_minus_low_residual") for r in host_rows],
        [r.get("weighted_mean_mu_minus_cepheid_distance") for r in host_rows],
    ).get("r")
    observed_model = weighted_ols(host_rows, ["host_high_minus_low_residual"], "observed_edge_only")
    observed_slope = observed_model.get("primary_slope")
    observed_contrast = clean_reference_contrast(host_rows).get("clean_minus_reference_mu_difference")

    mu_values = [r.get("weighted_mean_mu_minus_cepheid_distance") for r in host_rows]
    roles = [r.get("frozen_regime_role") for r in host_rows]

    for i in range(PERMUTATION_N):
        shuffled_mu = list(mu_values)
        rng.shuffle(shuffled_mu)

        shuffled_rows = []
        for row, mu in zip(host_rows, shuffled_mu):
            o = dict(row)
            o["weighted_mean_mu_minus_cepheid_distance"] = mu
            shuffled_rows.append(o)

        corr = pearson_corr(
            [r.get("host_high_minus_low_residual") for r in shuffled_rows],
            [r.get("weighted_mean_mu_minus_cepheid_distance") for r in shuffled_rows],
        )
        model = weighted_ols(shuffled_rows, ["host_high_minus_low_residual"], "perm_host_label_edge_only")

        shuffled_roles = list(roles)
        rng.shuffle(shuffled_roles)
        role_rows = []
        for row, role in zip(host_rows, shuffled_roles):
            o = dict(row)
            o["frozen_regime_role"] = role
            role_rows.append(o)
        contrast = clean_reference_contrast(role_rows)

        rows.append(
            {
                "iteration": i,
                "host_label_perm_corr_r": corr.get("r"),
                "host_label_perm_edge_slope": model.get("primary_slope"),
                "regime_label_perm_clean_minus_reference_mu_difference": contrast.get("clean_minus_reference_mu_difference"),
            }
        )

    def p_two_sided(null_values, observed):
        if observed is None:
            return None
        arr = np.asarray([v for v in null_values if v is not None and math.isfinite(float(v))], dtype=float)
        if len(arr) == 0:
            return None
        return float((1.0 + np.sum(np.abs(arr) >= abs(float(observed)))) / (len(arr) + 1.0))

    summary = {
        "iterations": PERMUTATION_N,
        "observed_corr_r": observed_corr,
        "observed_edge_only_slope": observed_slope,
        "observed_clean_minus_reference_mu_difference": observed_contrast,
        "host_label_perm_corr_r_p_two_sided": p_two_sided([r.get("host_label_perm_corr_r") for r in rows], observed_corr),
        "host_label_perm_edge_slope_p_two_sided": p_two_sided([r.get("host_label_perm_edge_slope") for r in rows], observed_slope),
        "regime_label_perm_contrast_p_two_sided": p_two_sided(
            [r.get("regime_label_perm_clean_minus_reference_mu_difference") for r in rows],
            observed_contrast,
        ),
    }
    return {"iterations": PERMUTATION_N, "summary": summary, "rows": rows}


def control_variable_tests(host_rows):
    predictors = [
        "host_high_minus_low_residual",
        "host_high_minus_low_f160w",
        "host_mean_period",
        "host_mean_color_v_i",
        "host_mean_metal",
        "host_f160w_std",
        "host_period_std",
        "mean_zHD",
        "mean_distance_HOST_LOGMASS",
    ]
    corr_rows = []
    regression_rows = []
    coef_rows = []
    for predictor in predictors:
        corr = pearson_corr(
            [r.get(predictor) for r in host_rows],
            [r.get("weighted_mean_mu_minus_cepheid_distance") for r in host_rows],
        )
        corr_rows.append(
            {
                "predictor": predictor,
                "corr_n": corr.get("n"),
                "corr_r": corr.get("r"),
                "truth_boundary": "control_association_only",
            }
        )
        model = weighted_ols(host_rows, [predictor], f"control_{predictor}")
        regression_rows.append(
            {
                "model_name": model.get("model_name"),
                "predictor": predictor,
                "n": model.get("n"),
                "status": model.get("status"),
                "r2": model.get("r2"),
                "primary_slope": model.get("primary_slope"),
                "primary_t": model.get("primary_t"),
            }
        )
        for cr in model.get("coef_rows", []):
            coef_rows.append(cr)

    multivar_sets = [
        ("edge_plus_period_color_metal", ["host_high_minus_low_residual", "host_mean_period", "host_mean_color_v_i", "host_mean_metal"]),
        ("edge_plus_downstream_context", ["host_high_minus_low_residual", "mean_zHD", "mean_distance_HOST_LOGMASS"]),
        ("edge_plus_f160w_span", ["host_high_minus_low_residual", "host_high_minus_low_f160w"]),
    ]
    multivar_rows = []
    for name, preds in multivar_sets:
        model = weighted_ols(host_rows, preds, name)
        multivar_rows.append(
            {
                "model_name": model.get("model_name"),
                "predictors": "|".join(preds),
                "n": model.get("n"),
                "status": model.get("status"),
                "r2": model.get("r2"),
                "primary_slope": model.get("primary_slope"),
                "primary_t": model.get("primary_t"),
            }
        )
        for cr in model.get("coef_rows", []):
            coef_rows.append(cr)

    return {
        "corr_rows": corr_rows,
        "regression_rows": regression_rows,
        "multivar_rows": multivar_rows,
        "coef_rows": coef_rows,
    }


def m31_quarantine_check(host_rows):
    m31 = [r for r in host_rows if r.get("frozen_regime_role") == "sign_break_quarantine" or r.get("table2_host") == "M31"]
    non_m31 = [r for r in host_rows if r not in m31]
    return {
        "m31_rows": m31,
        "m31_host_count": len(m31),
        "non_m31_host_count": len(non_m31),
        "non_m31_corr": pearson_corr(
            [r.get("host_high_minus_low_residual") for r in non_m31],
            [r.get("weighted_mean_mu_minus_cepheid_distance") for r in non_m31],
        ),
        "with_m31_corr": pearson_corr(
            [r.get("host_high_minus_low_residual") for r in host_rows],
            [r.get("weighted_mean_mu_minus_cepheid_distance") for r in host_rows],
        ),
        "m31_truth_boundary": "M31 remains quarantine/report-only and is not used as a correction.",
    }


def effect_size_summary(host_rows, contrast, models, bootstrap, permutation, controls, m31_check):
    edge_corr = pearson_corr(
        [r.get("host_high_minus_low_residual") for r in host_rows],
        [r.get("weighted_mean_mu_minus_cepheid_distance") for r in host_rows],
    )
    edge_model = models.get("edge_only", {})
    primary_p = permutation.get("summary", {}).get("host_label_perm_edge_slope_p_two_sided")
    corr_p = permutation.get("summary", {}).get("host_label_perm_corr_r_p_two_sided")
    contrast_p = permutation.get("summary", {}).get("regime_label_perm_contrast_p_two_sided")
    boot_slope = bootstrap.get("summary", {}).get("edge_only_slope", {})
    boot_contrast = bootstrap.get("summary", {}).get("clean_minus_reference_mu_difference", {})

    signal_strength = "weak_or_absent"
    positive_flags = []
    caution_flags = []

    if edge_corr.get("r") is not None and abs(edge_corr["r"]) >= 0.30:
        positive_flags.append("edge_downstream_correlation_abs_ge_0_30")
    else:
        caution_flags.append("edge_downstream_correlation_small")

    if primary_p is not None and primary_p <= 0.05:
        positive_flags.append("edge_slope_permutation_p_le_0_05")
    else:
        caution_flags.append("edge_slope_permutation_not_significant")

    if boot_slope.get("p025") is not None and boot_slope.get("p975") is not None:
        if boot_slope["p025"] > 0 or boot_slope["p975"] < 0:
            positive_flags.append("bootstrap_slope_ci_excludes_zero")
        else:
            caution_flags.append("bootstrap_slope_ci_includes_zero")

    if contrast_p is not None and contrast_p <= 0.05:
        positive_flags.append("regime_contrast_permutation_p_le_0_05")
    else:
        caution_flags.append("regime_contrast_not_significant")

    if len(positive_flags) >= 3:
        signal_strength = "moderate_downstream_association_worth_followup"
    elif len(positive_flags) >= 2:
        signal_strength = "limited_downstream_association_with_cautions"

    return {
        "host_count": len(host_rows),
        "primary_edge_corr": edge_corr,
        "primary_edge_slope": edge_model.get("primary_slope"),
        "primary_edge_t": edge_model.get("primary_t"),
        "primary_edge_r2": edge_model.get("r2"),
        "primary_edge_perm_p_two_sided": primary_p,
        "primary_corr_perm_p_two_sided": corr_p,
        "clean_minus_reference_mu_difference": contrast.get("clean_minus_reference_mu_difference"),
        "regime_contrast_perm_p_two_sided": contrast_p,
        "bootstrap_edge_slope": boot_slope,
        "bootstrap_regime_contrast": boot_contrast,
        "m31_quarantine_summary": {
            "m31_host_count": m31_check.get("m31_host_count"),
            "with_m31_corr": m31_check.get("with_m31_corr"),
            "non_m31_corr": m31_check.get("non_m31_corr"),
        },
        "positive_flags": positive_flags,
        "caution_flags": caution_flags,
        "signal_strength": signal_strength,
        "truth_boundary": "effect_summary_is_association_only_not_validation",
    }


def decide(residual_summary, distance_summary, cov_shape, bridge_rows, host_rows, effect_summary, permutation):
    calibrator_rows = bridge_rows["calibrator_rows"]
    matched_rows = [r for r in calibrator_rows if r.get("host_bridge_status") == "matched_to_table2_host"]
    unique_cids = sorted(set(r["CID"] for r in calibrator_rows))
    matched_unique_cids = sorted(set(r["CID"] for r in matched_rows))
    bridged_hosts = sorted(set(r["table2_host_match"] for r in matched_rows if r.get("table2_host_match")))
    edge_qualified_hosts = sorted(set(r["table2_host"] for r in host_rows))

    signal_strength = effect_summary.get("signal_strength")
    strong_enough = signal_strength in {
        "moderate_downstream_association_worth_followup",
        "limited_downstream_association_with_cautions",
    }

    gates = [
        {
            "gate": "G1_table2_residual_surface_rebuilt",
            "passed": residual_summary.get("table2_row_count") == 3130 and residual_summary.get("residual_count", 0) >= 3130,
            "evidence": residual_summary,
        },
        {
            "gate": "G2_downstream_distance_and_covariance_ready",
            "passed": distance_summary.get("distance_row_count") == 1701 and cov_shape.get("has_expected_square_payload") is True,
            "evidence": {
                "distance_row_count": distance_summary.get("distance_row_count"),
                "covariance_dimension": cov_shape.get("first_token_as_int"),
                "covariance_shape_valid": cov_shape.get("has_expected_square_payload"),
            },
        },
        {
            "gate": "G3_bridge_coverage_ready",
            "passed": len(matched_unique_cids) >= 30 and len(bridged_hosts) >= 25 and len(edge_qualified_hosts) >= 20,
            "evidence": {
                "calibrator_row_count": len(calibrator_rows),
                "unique_calibrator_cid_count": len(unique_cids),
                "matched_unique_cid_count": len(matched_unique_cids),
                "bridged_table2_host_count": len(bridged_hosts),
                "edge_qualified_bridged_host_count": len(edge_qualified_hosts),
            },
        },
        {
            "gate": "G4_full_battery_completed",
            "passed": permutation.get("iterations", 0) >= PERMUTATION_N,
            "evidence": {
                "bootstrap_iterations": BOOTSTRAP_N,
                "permutation_iterations": permutation.get("iterations"),
            },
        },
        {
            "gate": "G5_signal_strength_clears_followup_threshold",
            "passed": strong_enough,
            "evidence": effect_summary,
        },
        {
            "gate": "G6_no_validation_claim_allowed",
            "passed": True,
            "evidence": {
                "validation_claim": False,
                "h0_claim": False,
                "new_physics_claim": False,
            },
        },
    ]

    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed:
        final_status = "downstream_influence_signal_present_but_not_validation"
        readiness = 9
        next_wall = (
            "The bundled downstream battery found a follow-up-worthy association. Next step should be an independent external check or a formal paper note with strict limits."
        )
    elif failed == ["G5_signal_strength_clears_followup_threshold"]:
        final_status = "downstream_bridge_passes_but_signal_weak_or_absent"
        readiness = 7
        next_wall = (
            "The bridge and battery worked, but the downstream influence signal did not clear the conservative follow-up threshold. "
            "Do not escalate this as a physics result."
        )
    elif len(failed) <= 2 and "G1_table2_residual_surface_rebuilt" not in failed:
        final_status = "downstream_influence_battery_completed_with_cautions"
        readiness = 6
        next_wall = (
            "The battery completed but at least one evidence gate failed. Interpret as caution, not validation."
        )
    else:
        final_status = "downstream_influence_battery_not_ready"
        readiness = 5
        next_wall = (
            "Core rebuild, bridge, or battery gates failed. Do not interpret."
        )

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "gates": gates,
        "failed_gates": failed,
        "bridge_core": {
            "calibrator_row_count": len(calibrator_rows),
            "unique_calibrator_cid_count": len(unique_cids),
            "matched_unique_cid_count": len(matched_unique_cids),
            "bridged_table2_host_count": len(bridged_hosts),
            "edge_qualified_bridged_host_count": len(edge_qualified_hosts),
            "unmatched_calibrator_row_count": len(bridge_rows["unmatched_rows"]),
        },
        "effect_core": effect_summary,
        "truth_boundary": CLAIMS_V2_6["truth_boundary"],
    }


def make_plots(host_rows):
    try:
        xs = [r.get("host_high_minus_low_residual") for r in host_rows]
        ys = [r.get("weighted_mean_mu_minus_cepheid_distance") for r in host_rows]
        labels = [r.get("table2_host") for r in host_rows]
        pairs = [(x, y, label) for x, y, label in zip(xs, ys, labels) if x is not None and y is not None]
        if pairs:
            plt.figure(figsize=(8, 6))
            plt.scatter([p[0] for p in pairs], [p[1] for p in pairs])
            plt.xlabel("Host F160W edge residual delta")
            plt.ylabel("Weighted mean MU_SH0ES - CEPH_DIST")
            plt.title("v2.6 downstream bridge host-level influence diagnostic")
            plt.tight_layout()
            plt.savefig(OUTDIR / "host_edge_vs_downstream_mu_v2_6_fresh.png", dpi=160)
            plt.close()

        role_counts = defaultdict(int)
        for row in host_rows:
            role_counts[row.get("frozen_regime_role", "unknown")] += 1
        if role_counts:
            plt.figure(figsize=(8, 5))
            plt.bar(list(role_counts.keys()), list(role_counts.values()))
            plt.ylabel("host count")
            plt.title("v2.6 bridged hosts by frozen regime")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(OUTDIR / "bridged_hosts_by_regime_v2_6_fresh.png", dpi=160)
            plt.close()
    except Exception as exc:
        write_json(OUTDIR / "plot_error_v2_6_fresh.json", {"error": repr(exc), "traceback": traceback.format_exc()})


def holographic_surface_ledger(decision):
    return {
        "observable_surface": {
            "name": "Full downstream influence battery over locked Table2 host-edge surface and Pantheon+SH0ES calibrator bridge",
            "table2_path": TABLE2_PATH,
            "distance_table_path": DISTANCE_TABLE_PATH,
            "redshift_metadata_path": REDSHIFT_METADATA_PATH,
            "distance_covariance_path": DISTANCE_COV_PATH,
        },
        "hidden_depth_sought": {
            "name": "Whether locked host-edge structure carries downstream calibrator-distance association",
            "allowed_claim": "v2.6 can report whether a predeclared downstream association battery passes or fails.",
            "not_allowed_claim": "v2.6 cannot validate TAIRID, claim H0 correction, prove new physics, or claim SH0ES is wrong.",
        },
        "boundary_that_forms_surface": {
            "locked_lane_boundary": "v1.9-v2.5 Table2 residual F160W edge rule remains frozen",
            "downstream_boundary": "Pantheon+SH0ES distance rows are SN-level outcomes, bridged by host only",
            "statistical_boundary": "weighted OLS, bootstrap, permutation, controls, and contrasts are association tests only",
            "method_boundary": "no distance-ladder refit and no H0 correction model",
        },
        "what_can_be_reconstructed_now": [
            "Host-level edge/outcome association",
            "Control-variable comparison",
            "Bootstrap uncertainty",
            "Permutation null strength",
            "Regime contrast behavior",
            "M31 quarantine sensitivity",
        ],
        "what_cannot_be_reconstructed_now": [
            "Causality",
            "H0 correction",
            "New physics",
            "Proof of TAIRID",
            "A claim that SH0ES is wrong",
        ],
        "surface_noise_definition": [
            "Treating a weak association as proof",
            "Changing the frozen rule after seeing downstream outcomes",
            "Promoting M31 from quarantine to correction",
            "Ignoring false-positive controls",
            "Calling host-level bridge analysis a full ladder refit",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v2.6 Fresh Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## Bridge core")
    lines.append("")
    for key, value in decision.get("bridge_core", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Effect core")
    lines.append("")
    effect = decision.get("effect_core", {})
    lines.append(f"- signal_strength: `{effect.get('signal_strength')}`")
    lines.append(f"- primary_edge_corr: `{effect.get('primary_edge_corr')}`")
    lines.append(f"- primary_edge_slope: `{effect.get('primary_edge_slope')}`")
    lines.append(f"- primary_edge_perm_p_two_sided: `{effect.get('primary_edge_perm_p_two_sided')}`")
    lines.append(f"- clean_minus_reference_mu_difference: `{effect.get('clean_minus_reference_mu_difference')}`")
    lines.append(f"- regime_contrast_perm_p_two_sided: `{effect.get('regime_contrast_perm_p_two_sided')}`")
    lines.append("")
    lines.append("## What v2.6 did")
    lines.append("")
    lines.append("- Rebuilt the locked Table2 residual surface.")
    lines.append("- Rebuilt host-level F160W edge summaries without tuning.")
    lines.append("- Rebuilt the Pantheon+SH0ES downstream calibrator bridge.")
    lines.append("- Ran weighted regressions, regime contrasts, leave-one-host-out checks, bootstrap, permutations, M31 quarantine checks, and control-variable tests.")
    lines.append("- Produced an effect-size summary and conservative pass/fail decision.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v2.6 is an association battery only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not prove SH0ES is wrong.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["final_status"] == "downstream_influence_signal_present_but_not_validation":
        lines.append("Next step: run an independent external check or write a bounded technical note with strict limitations.")
    elif decision["final_status"] == "downstream_bridge_passes_but_signal_weak_or_absent":
        lines.append("Next step: stop escalation in this lane or pivot to a genuinely independent dataset. Do not sell this as a downstream physics result.")
    else:
        lines.append("Next step: repair failed gates before interpreting further.")
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Full Downstream Influence Battery v2.6 Fresh starting.")
    print("Boundary: association battery only; no downstream refit, no tuning, no H0 claim.")

    rng = np.random.default_rng(RANDOM_SEED)

    write_json(OUTDIR / "claims_v2_6_fresh.json", CLAIMS_V2_6)
    write_json(OUTDIR / "frozen_rule_carry_forward_v2_6_fresh.json", FROZEN_RULE_CARRY_FORWARD)

    try:
        download_ledger = []
        text_files = {}
        fits_arrays = {}

        paths = [
            TABLE2_PATH,
            LSTSQ_PATH,
            Y_FITS_PATH,
            L_FITS_PATH,
            C_FITS_PATH,
            DISTANCE_TABLE_PATH,
            DISTANCE_COV_PATH,
            REDSHIFT_METADATA_PATH,
        ]

        for path in paths:
            fetched = fetch_bytes_for_path(path, prefer_media=path.lower().endswith(".fits"))
            download_ledger.append(
                {
                    "repo_path": path,
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
                continue
            local_path = DOWNLOAD_DIR / safe_name(path)
            local_path.write_bytes(fetched["data"])
            if path.lower().endswith(".fits"):
                fits_arrays[path] = read_primary_fits_array(fetched["data"])
            else:
                text_files[path] = fetched["data"].decode("utf-8", errors="replace")

        write_csv(OUTDIR / "download_ledger_v2_6_fresh.csv", download_ledger)

        table2 = parse_table2_tex(text_files.get(TABLE2_PATH, ""))
        write_csv(OUTDIR / "table2_parsed_rows_v2_6_fresh.csv", table2["rows"])
        write_json(OUTDIR / "table2_parse_errors_v2_6_fresh.json", table2["errors"])

        theta_parse = parse_lstsq_results(text_files.get(LSTSQ_PATH, ""))
        theta = np.asarray([r["theta_value"] for r in theta_parse["rows"]], dtype=float)
        write_csv(OUTDIR / "theta_lstsq_vector_v2_6_fresh.csv", theta_parse["rows"])
        write_json(OUTDIR / "theta_lstsq_parse_errors_v2_6_fresh.json", theta_parse["errors"])

        y = np.asarray(fits_arrays.get(Y_FITS_PATH, np.asarray([])), dtype=float)
        L = np.asarray(fits_arrays.get(L_FITS_PATH, np.asarray([[]])), dtype=float)
        C = np.asarray(fits_arrays.get(C_FITS_PATH, np.asarray([[]])), dtype=float)

        predicted_y, residual, orientation = compute_ladder_residual(theta, L, y)
        if predicted_y is None:
            predicted_y = np.asarray([])
            residual = np.asarray([])

        covariance_diag, cov_info = covariance_diag_info(C)
        table_residual_rows = []
        if len(residual) >= len(table2["rows"]) and len(table2["rows"]) > 0:
            table_residual_rows = attach_table2_residuals(table2["rows"], y, predicted_y, residual, covariance_diag)

        write_csv(OUTDIR / "table2_ladder_residual_rows_v2_6_fresh.csv", table_residual_rows)
        write_json(OUTDIR / "table2_covariance_diag_summary_v2_6_fresh.json", cov_info)

        host_edges = build_host_f160w_edges(table_residual_rows)
        write_csv(OUTDIR / "table2_f160w_edge_rows_v2_6_fresh.csv", host_edges["edge_rows"])
        write_csv(OUTDIR / "table2_host_edge_summary_v2_6_fresh.csv", host_edges["host_summary"])

        distance = parse_distance_table(text_files.get(DISTANCE_TABLE_PATH, ""))
        distance_rows = distance["rows"]
        write_csv(OUTDIR / "pantheon_shoes_distance_rows_v2_6_fresh.csv", distance_rows[:5000])

        redshift_rows = parse_redshift_metadata(text_files.get(REDSHIFT_METADATA_PATH, ""))
        write_csv(OUTDIR / "redshift_metadata_rows_v2_6_fresh.csv", redshift_rows[:5000])

        cov_shape = parse_cov_dimension(text_files.get(DISTANCE_COV_PATH, ""))
        write_json(OUTDIR / "downstream_covariance_shape_v2_6_fresh.json", cov_shape)

        bridge = build_calibrator_bridge(distance_rows, redshift_rows, host_edges["host_summary"])
        write_csv(OUTDIR / "calibrator_bridge_rows_v2_6_fresh.csv", bridge["calibrator_rows"])
        write_csv(OUTDIR / "calibrator_unmatched_rows_v2_6_fresh.csv", bridge["unmatched_rows"])

        host_bridge = aggregate_host_bridge(bridge["calibrator_rows"])
        write_csv(OUTDIR / "bridged_host_influence_dataset_v2_6_fresh.csv", host_bridge)

        models = {
            "edge_only": weighted_ols(host_bridge, ["host_high_minus_low_residual"], "edge_only"),
            "edge_plus_period_color_metal": weighted_ols(
                host_bridge,
                ["host_high_minus_low_residual", "host_mean_period", "host_mean_color_v_i", "host_mean_metal"],
                "edge_plus_period_color_metal",
            ),
            "edge_plus_downstream_context": weighted_ols(
                host_bridge,
                ["host_high_minus_low_residual", "mean_zHD", "mean_distance_HOST_LOGMASS"],
                "edge_plus_downstream_context",
            ),
        }
        model_rows = []
        coef_rows = []
        for name, model in models.items():
            model_rows.append(
                {
                    "model_key": name,
                    "model_name": model.get("model_name"),
                    "predictors": "|".join(model.get("predictors", [])),
                    "n": model.get("n"),
                    "status": model.get("status"),
                    "r2": model.get("r2"),
                    "primary_slope": model.get("primary_slope"),
                    "primary_t": model.get("primary_t"),
                }
            )
            for cr in model.get("coef_rows", []):
                coef_rows.append(cr)
        write_csv(OUTDIR / "weighted_regression_models_v2_6_fresh.csv", model_rows)
        write_csv(OUTDIR / "weighted_regression_coefficients_v2_6_fresh.csv", coef_rows)

        contrast = clean_reference_contrast(host_bridge)
        write_json(OUTDIR / "clean_reference_contrast_v2_6_fresh.json", contrast)

        loho = leave_one_host_out(host_bridge)
        write_csv(OUTDIR / "leave_one_host_out_sensitivity_v2_6_fresh.csv", loho)

        bootstrap = bootstrap_battery(host_bridge, rng)
        write_json(OUTDIR / "bootstrap_summary_v2_6_fresh.json", bootstrap["summary"])
        write_csv(OUTDIR / "bootstrap_iterations_v2_6_fresh.csv", bootstrap["rows"])

        permutation = permutation_battery(host_bridge, rng)
        write_json(OUTDIR / "permutation_summary_v2_6_fresh.json", permutation["summary"])
        write_csv(OUTDIR / "permutation_iterations_v2_6_fresh.csv", permutation["rows"])

        controls = control_variable_tests(host_bridge)
        write_csv(OUTDIR / "control_correlations_v2_6_fresh.csv", controls["corr_rows"])
        write_csv(OUTDIR / "control_regressions_v2_6_fresh.csv", controls["regression_rows"])
        write_csv(OUTDIR / "control_multivar_regressions_v2_6_fresh.csv", controls["multivar_rows"])
        write_csv(OUTDIR / "control_regression_coefficients_v2_6_fresh.csv", controls["coef_rows"])

        m31 = m31_quarantine_check(host_bridge)
        write_json(
            OUTDIR / "m31_quarantine_check_v2_6_fresh.json",
            {
                "m31_host_count": m31["m31_host_count"],
                "non_m31_host_count": m31["non_m31_host_count"],
                "non_m31_corr": m31["non_m31_corr"],
                "with_m31_corr": m31["with_m31_corr"],
                "m31_truth_boundary": m31["m31_truth_boundary"],
            },
        )
        write_csv(OUTDIR / "m31_quarantine_rows_v2_6_fresh.csv", m31["m31_rows"])

        residual_summary = {
            "theta_count": int(len(theta)),
            "y_shape": list(y.shape),
            "L_shape": list(L.shape),
            "C_shape": list(C.shape),
            "orientation": orientation,
            "predicted_y_count": int(len(predicted_y)),
            "residual_count": int(len(residual)),
            "table2_row_count": int(len(table2["rows"])),
            "table2_residual_row_count": int(len(table_residual_rows)),
            "table2_host_count": int(len(set(r["host"] for r in table_residual_rows))),
        }

        distance_summary = {
            "distance_column_count": len(distance["columns"]),
            "distance_columns": distance["columns"],
            "distance_row_count": len(distance_rows),
            "calibrator_row_count": len([r for r in distance_rows if int(r.get("IS_CALIBRATOR", 0)) == 1]),
            "unique_distance_cid_count": len(set(str(r.get("CID", "")) for r in distance_rows)),
        }

        effect_summary = effect_size_summary(host_bridge, contrast, models, bootstrap, permutation, controls, m31)
        write_json(OUTDIR / "effect_size_summary_v2_6_fresh.json", effect_summary)

        decision = decide(residual_summary, distance_summary, cov_shape, bridge, host_bridge, effect_summary, permutation)
        write_json(OUTDIR / "decision_v2_6_fresh.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v2_6_fresh.json", ledger)

        make_plots(host_bridge)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v2_6_fresh.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v2_6_fresh.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID SH0ES Full Downstream Influence Battery v2.6 Fresh",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Association battery only. No tuning, no downstream refit, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "random_seed": RANDOM_SEED,
            "bootstrap_iterations": BOOTSTRAP_N,
            "permutation_iterations": PERMUTATION_N,
            "download_ledger": download_ledger,
            "residual_summary": residual_summary,
            "distance_summary": distance_summary,
            "downstream_covariance_shape": cov_shape,
            "bridge_core": decision["bridge_core"],
            "models": model_rows,
            "clean_reference_contrast": contrast,
            "bootstrap_summary": bootstrap["summary"],
            "permutation_summary": permutation["summary"],
            "control_correlation_summary": controls["corr_rows"],
            "m31_quarantine_check": {
                "m31_host_count": m31["m31_host_count"],
                "non_m31_host_count": m31["non_m31_host_count"],
                "non_m31_corr": m31["non_m31_corr"],
                "with_m31_corr": m31["with_m31_corr"],
            },
            "effect_size_summary": effect_summary,
            "decision": decision,
            "claims_v2_6": CLAIMS_V2_6,
            "frozen_rule_carry_forward": FROZEN_RULE_CARRY_FORWARD,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_full_downstream_influence_battery_v2_6_fresh_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_full_downstream_influence_battery_v2_6_fresh_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v2_6_fresh.csv"),
                "host_dataset_csv": str(OUTDIR / "bridged_host_influence_dataset_v2_6_fresh.csv"),
                "weighted_models_csv": str(OUTDIR / "weighted_regression_models_v2_6_fresh.csv"),
                "control_correlations_csv": str(OUTDIR / "control_correlations_v2_6_fresh.csv"),
                "loho_csv": str(OUTDIR / "leave_one_host_out_sensitivity_v2_6_fresh.csv"),
                "bootstrap_summary_json": str(OUTDIR / "bootstrap_summary_v2_6_fresh.json"),
                "permutation_summary_json": str(OUTDIR / "permutation_summary_v2_6_fresh.json"),
                "effect_summary_json": str(OUTDIR / "effect_size_summary_v2_6_fresh.json"),
                "decision_json": str(OUTDIR / "decision_v2_6_fresh.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v2_6_fresh.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v2_6_fresh.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v2_6_fresh.txt"),
                "plots": [
                    str(OUTDIR / "host_edge_vs_downstream_mu_v2_6_fresh.png"),
                    str(OUTDIR / "bridged_hosts_by_regime_v2_6_fresh.png"),
                ],
            },
            "interpretation": {
                "what_success_means": "A downstream association survived a larger predeclared battery and is worth follow-up.",
                "what_success_does_not_mean": "This does not validate TAIRID, H0 correction, new physics, or SH0ES error.",
                "what_weak_result_means": "The bridge may be real while the downstream influence signal is weak or absent.",
                "truth_boundary": CLAIMS_V2_6["truth_boundary"],
            },
        }

        write_json(OUTDIR / "shoes_full_downstream_influence_battery_v2_6_fresh_summary.json", summary)

        with open(OUTDIR / "shoes_full_downstream_influence_battery_v2_6_fresh_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Full Downstream Influence Battery v2.6 Fresh\n\n")
            f.write("Boundary: association battery only. No tuning. No downstream refit. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Bridge core:\n")
            f.write(json.dumps(decision["bridge_core"], indent=2, default=json_default) + "\n\n")
            f.write("Effect summary:\n")
            f.write(json.dumps(effect_summary, indent=2, default=json_default) + "\n\n")
            f.write("Weighted models:\n")
            f.write(json.dumps(model_rows, indent=2, default=json_default) + "\n\n")
            f.write("Bootstrap summary:\n")
            f.write(json.dumps(bootstrap["summary"], indent=2, default=json_default) + "\n\n")
            f.write("Permutation summary:\n")
            f.write(json.dumps(permutation["summary"], indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not prove SH0ES is wrong.\n")
            f.write("- This does not tune or replace the frozen rule.\n")
            f.write("- This does not refit the distance ladder.\n")

        print("TAIRID SH0ES Full Downstream Influence Battery v2.6 Fresh complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Full Downstream Influence Battery v2.6 Fresh",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_full_downstream_influence_battery_v2_6_fresh_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V2_6["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_full_downstream_influence_battery_v2_6_fresh_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
