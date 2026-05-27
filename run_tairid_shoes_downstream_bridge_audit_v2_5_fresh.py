#!/usr/bin/env python3
"""
TAIRID SH0ES Downstream Bridge Audit v2.5 Fresh

Why this test exists:
v2.4.1 confirmed that a legitimate downstream Pantheon+SH0ES distance table
exists, with covariance support, but it is not a direct Table2 Cepheid replay
surface. A bridge is required before any downstream frozen-rule replay.

v2.5 builds that bridge and stops before validation.

Core questions:
    1. Can the locked Table2 F160W residual host-edge surface be rebuilt?
    2. Can Pantheon+SH0ES calibrator SN rows be joined to host metadata?
    3. Can calibrator hosts be normalized and matched back to Table2 hosts?
    4. Do enough mapped hosts also have frozen F160W edge summaries?
    5. What downstream influence variables are available for a later test?
    6. Is v2.6 allowed to run a predeclared influence test, or must the bridge
       stop here?

This test DOES:
    - rebuild the same Table2 ladder residual rows,
    - rebuild host-level frozen F160W edge summaries,
    - parse Pantheon+SH0ES downstream distance rows,
    - parse redshift/host metadata,
    - bridge calibrator SN rows to Table2 hosts by normalized host name,
    - compute descriptive host-level downstream quantities,
    - compute report-only correlations/contrasts,
    - write a bridge audit and next-test handoff.

This test DOES NOT:
    - validate TAIRID,
    - claim H0 correction,
    - claim new physics,
    - claim SH0ES is wrong,
    - tune the frozen rule,
    - fit a downstream correction model,
    - replace F160W with another variable,
    - treat bridge success as validation.

Truth boundary:
v2.5 is a bridge and influence audit only. If it passes, v2.6 may run a
predeclared downstream influence test. v2.5 itself is not that test.
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


OUTDIR = Path("tairid_shoes_downstream_bridge_audit_v2_5_fresh_outputs")
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

LOW_ALPHA_REFERENCE_HOSTS = {"LMC", "SMC", "N4536"}
SIGN_BREAK_QUARANTINE_HOSTS = {"M31"}

CLAIMS_V2_5 = {
    "battery_name": "TAIRID SH0ES Downstream Bridge Audit v2.5 Fresh",
    "scope": "Bridge audit from locked Table2 F160W residual host-edge surface to Pantheon+SH0ES downstream calibrator distance rows",
    "primary_question": (
        "Can downstream Pantheon+SH0ES calibrator rows be bridged to Table2 hosts with enough coverage "
        "to justify a later predeclared influence test?"
    ),
    "truth_boundary": (
        "This is bridge/influence auditing only. It does not validate TAIRID, H0 correction, or new physics."
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
        "Do not tune the frozen rule in v2.5.",
        "Do not fit a downstream correction in v2.5.",
        "Do not claim H0 correction or new physics.",
        "Do not treat bridge coverage as validation.",
        "Do not transfer M31 correction; M31 remains quarantine/report-only.",
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
        headers={"User-Agent": "TAIRID-v2.5-fresh-downstream-bridge-audit"},
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


def fetch_text_for_path(repo_path):
    fetched = fetch_bytes_for_path(repo_path, prefer_media=repo_path.lower().endswith(".fits"))
    if fetched["status"] != "downloaded":
        return {**fetched, "text": ""}
    return {**fetched, "text": fetched["data"].decode("utf-8", errors="replace")}


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
                "mean_mu_minus_cepheid_distance": summarize_numeric(mu_resids)["mean"],
                "median_mu_minus_cepheid_distance": summarize_numeric(mu_resids)["median"],
                "weighted_mean_mu_minus_cepheid_distance": weighted_mean(mu_resids, weights),
                "mean_MU_SH0ES": summarize_numeric([r.get("MU_SH0ES") for r in rows])["mean"],
                "mean_CEPH_DIST": summarize_numeric([r.get("CEPH_DIST") for r in rows])["mean"],
                "mean_zHD": summarize_numeric([r.get("zHD") for r in rows])["mean"],
                "used_in_shoes_hf_values": "|".join(sorted(set(str(v) for v in used_hf))),
            }
        )

    return host_rows


def regime_bridge_summary(host_rows):
    by_role = defaultdict(list)
    for row in host_rows:
        by_role[row.get("frozen_regime_role", "unknown")].append(row)

    out = []
    for role, rows in sorted(by_role.items()):
        out.append(
            {
                "frozen_regime_role": role,
                "bridged_host_count": len(rows),
                "unique_cid_count": sum(int(r.get("unique_cid_count", 0)) for r in rows),
                "calibrator_row_count": sum(int(r.get("calibrator_row_count", 0)) for r in rows),
                "mean_host_edge_residual": summarize_numeric([r.get("host_high_minus_low_residual") for r in rows])["mean"],
                "mean_mu_minus_cepheid_distance": summarize_numeric([r.get("mean_mu_minus_cepheid_distance") for r in rows])["mean"],
                "weighted_mean_mu_minus_cepheid_distance_across_hosts": weighted_mean(
                    [r.get("weighted_mean_mu_minus_cepheid_distance") for r in rows],
                    [r.get("calibrator_row_count") for r in rows],
                ),
            }
        )
    return out


def influence_report(host_rows):
    clean_hosts = [r for r in host_rows if r.get("frozen_regime_role") == "clean_high_alpha_candidate"]
    reference_hosts = [r for r in host_rows if r.get("frozen_regime_role") == "low_alpha_reference"]
    quarantine_hosts = [r for r in host_rows if r.get("frozen_regime_role") == "sign_break_quarantine"]

    edge = [r.get("host_high_minus_low_residual") for r in host_rows]
    mu = [r.get("weighted_mean_mu_minus_cepheid_distance") for r in host_rows]
    fspan = [r.get("host_high_minus_low_f160w") for r in host_rows]

    clean_mu = weighted_mean(
        [r.get("weighted_mean_mu_minus_cepheid_distance") for r in clean_hosts],
        [r.get("calibrator_row_count") for r in clean_hosts],
    )
    ref_mu = weighted_mean(
        [r.get("weighted_mean_mu_minus_cepheid_distance") for r in reference_hosts],
        [r.get("calibrator_row_count") for r in reference_hosts],
    )

    return {
        "host_edge_residual_vs_downstream_mu_corr": pearson_corr(edge, mu),
        "host_f160w_span_vs_downstream_mu_corr": pearson_corr(fspan, mu),
        "clean_bridged_host_count": len(clean_hosts),
        "reference_bridged_host_count": len(reference_hosts),
        "quarantine_bridged_host_count": len(quarantine_hosts),
        "clean_weighted_mean_mu_minus_cepheid": clean_mu,
        "reference_weighted_mean_mu_minus_cepheid": ref_mu,
        "clean_minus_reference_mu_difference": clean_mu - ref_mu if clean_mu is not None and ref_mu is not None else None,
        "truth_boundary": "report_only_bridge_diagnostic_not_validation",
    }


def decide(residual_summary, distance_summary, cov_summary, bridge_rows, host_rows, report):
    calibrator_rows = bridge_rows["calibrator_rows"]
    matched_rows = [r for r in calibrator_rows if r.get("host_bridge_status") == "matched_to_table2_host"]
    unique_cids = sorted(set(r["CID"] for r in calibrator_rows))
    matched_unique_cids = sorted(set(r["CID"] for r in matched_rows))
    bridged_hosts = sorted(set(r["table2_host_match"] for r in matched_rows if r.get("table2_host_match")))
    edge_qualified_hosts = sorted(set(r["table2_host"] for r in host_rows))
    corr_n = report.get("host_edge_residual_vs_downstream_mu_corr", {}).get("n", 0)

    gates = [
        {
            "gate": "G1_table2_residual_surface_rebuilt",
            "passed": residual_summary.get("table2_row_count") == 3130 and residual_summary.get("residual_count", 0) >= 3130,
            "evidence": residual_summary,
        },
        {
            "gate": "G2_downstream_distance_and_covariance_ready",
            "passed": distance_summary.get("distance_row_count") == 1701 and cov_summary.get("has_expected_square_payload") is True,
            "evidence": {
                "distance_row_count": distance_summary.get("distance_row_count"),
                "covariance_dimension": cov_summary.get("first_token_as_int"),
                "covariance_shape_valid": cov_summary.get("has_expected_square_payload"),
            },
        },
        {
            "gate": "G3_calibrator_rows_identified",
            "passed": len(calibrator_rows) >= 60 and len(unique_cids) >= 35,
            "evidence": {
                "calibrator_row_count": len(calibrator_rows),
                "unique_calibrator_cid_count": len(unique_cids),
            },
        },
        {
            "gate": "G4_calibrator_hosts_bridge_to_table2",
            "passed": len(matched_unique_cids) >= 30 and len(bridged_hosts) >= 25,
            "evidence": {
                "matched_unique_cid_count": len(matched_unique_cids),
                "bridged_table2_host_count": len(bridged_hosts),
                "unmatched_calibrator_row_count": len(bridge_rows["unmatched_rows"]),
            },
        },
        {
            "gate": "G5_edge_qualified_bridge_hosts_available",
            "passed": len(edge_qualified_hosts) >= 20 and corr_n >= 20,
            "evidence": {
                "edge_qualified_bridged_host_count": len(edge_qualified_hosts),
                "correlation_pair_count": corr_n,
            },
        },
        {
            "gate": "G6_stop_before_downstream_replay",
            "passed": True,
            "evidence": {
                "downstream_replay_run": False,
                "reason": "v2.5 is bridge/influence audit only",
            },
        },
    ]

    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed:
        final_status = "downstream_bridge_ready_for_predeclared_influence_test_not_validation"
        readiness = 9
        next_wall = (
            "Bridge coverage is strong enough for v2.6. v2.6 may run a predeclared downstream influence test, "
            "but v2.5 itself is not validation."
        )
    elif len(failed) <= 2 and "G1_table2_residual_surface_rebuilt" not in failed:
        final_status = "downstream_bridge_built_with_cautions"
        readiness = 7
        next_wall = (
            "The bridge exists but coverage or covariance gates need review before v2.6."
        )
    else:
        final_status = "downstream_bridge_not_ready_for_influence_test"
        readiness = 5
        next_wall = (
            "Bridge coverage is not good enough. Do not run downstream influence testing yet."
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
            "report_only_correlation": report.get("host_edge_residual_vs_downstream_mu_corr"),
            "clean_minus_reference_mu_difference": report.get("clean_minus_reference_mu_difference"),
        },
        "truth_boundary": CLAIMS_V2_5["truth_boundary"],
    }


def holographic_surface_ledger(decision):
    return {
        "observable_surface": {
            "name": "Bridge from locked Table2 F160W residual host-edge surface to downstream Pantheon+SH0ES calibrator rows",
            "table2_path": TABLE2_PATH,
            "distance_table_path": DISTANCE_TABLE_PATH,
            "redshift_metadata_path": REDSHIFT_METADATA_PATH,
            "distance_covariance_path": DISTANCE_COV_PATH,
        },
        "hidden_depth_sought": {
            "name": "Whether downstream calibrator rows can be host-bridged before any replay",
            "allowed_claim": "v2.5 can establish bridge coverage and report influence diagnostics.",
            "not_allowed_claim": "v2.5 cannot validate TAIRID, claim H0 correction, or fit a downstream correction.",
        },
        "boundary_that_forms_surface": {
            "locked_lane_boundary": "v1.9-v2.2 Table2 residual F160W edge rule remains frozen",
            "bridge_boundary": "calibrator SN rows joined by CID to host metadata, then normalized to Table2 host names",
            "downstream_boundary": "Pantheon+SH0ES distance rows are SN-level outcomes, not Cepheid rows",
            "method_boundary": "no downstream replay, no tuning, no model fitting in v2.5",
        },
        "what_can_be_reconstructed_now": [
            "Table2 host edge summary",
            "Calibrator SN host bridge",
            "Host-level downstream MU_SH0ES minus CEPH_DIST diagnostics",
            "Report-only correlations and regime summaries",
            "Readiness for a later predeclared influence test",
        ],
        "what_cannot_be_reconstructed_now": [
            "A validated downstream correction",
            "H0 correction",
            "New physics",
            "A proof of TAIRID",
            "A claim that SH0ES is wrong",
        ],
        "surface_noise_definition": [
            "Treating bridge coverage as validation",
            "Changing host regimes after seeing downstream outcomes",
            "Using unmatched hosts to force coverage",
            "Fitting a correction model in the bridge audit",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v2.5 Fresh Handoff")
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
    lines.append("## What v2.5 did")
    lines.append("")
    lines.append("- Rebuilt the locked Table2 residual surface.")
    lines.append("- Rebuilt host-level F160W edge summaries without tuning.")
    lines.append("- Parsed the Pantheon+SH0ES downstream distance table and covariance shape.")
    lines.append("- Joined calibrator SNe to redshift/host metadata.")
    lines.append("- Normalized host names and bridged calibrator rows to Table2 hosts.")
    lines.append("- Produced host-level downstream influence diagnostics.")
    lines.append("- Stopped before any downstream replay or correction model.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v2.5 is a bridge/influence audit only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not prove SH0ES is wrong.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append(
            "v2.6 may run a predeclared downstream influence test using only the frozen host-edge summary and the already bridged calibrator outcome variables. It must not tune the edge rule or refit a new model."
        )
    else:
        lines.append(
            "v2.6 should repair the bridge or stop escalation before downstream testing."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Downstream Bridge Audit v2.5 Fresh starting.")
    print("Boundary: bridge/influence audit only; no downstream replay, no tuning, no H0 claim.")

    write_json(OUTDIR / "claims_v2_5_fresh.json", CLAIMS_V2_5)
    write_json(OUTDIR / "frozen_rule_carry_forward_v2_5_fresh.json", FROZEN_RULE_CARRY_FORWARD)

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

        write_csv(OUTDIR / "download_ledger_v2_5_fresh.csv", download_ledger)

        table2 = parse_table2_tex(text_files.get(TABLE2_PATH, ""))
        write_csv(OUTDIR / "table2_parsed_rows_v2_5_fresh.csv", table2["rows"])
        write_json(OUTDIR / "table2_parse_errors_v2_5_fresh.json", table2["errors"])

        theta_parse = parse_lstsq_results(text_files.get(LSTSQ_PATH, ""))
        theta = np.asarray([r["theta_value"] for r in theta_parse["rows"]], dtype=float)
        write_csv(OUTDIR / "theta_lstsq_vector_v2_5_fresh.csv", theta_parse["rows"])
        write_json(OUTDIR / "theta_lstsq_parse_errors_v2_5_fresh.json", theta_parse["errors"])

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

        write_csv(OUTDIR / "table2_ladder_residual_rows_v2_5_fresh.csv", table_residual_rows)
        write_json(OUTDIR / "table2_covariance_diag_summary_v2_5_fresh.json", cov_info)

        host_edges = build_host_f160w_edges(table_residual_rows)
        write_csv(OUTDIR / "table2_f160w_edge_rows_v2_5_fresh.csv", host_edges["edge_rows"])
        write_csv(OUTDIR / "table2_host_edge_summary_v2_5_fresh.csv", host_edges["host_summary"])

        distance = parse_distance_table(text_files.get(DISTANCE_TABLE_PATH, ""))
        distance_rows = distance["rows"]
        write_csv(OUTDIR / "pantheon_shoes_distance_rows_v2_5_fresh.csv", distance_rows[:5000])

        redshift_rows = parse_redshift_metadata(text_files.get(REDSHIFT_METADATA_PATH, ""))
        write_csv(OUTDIR / "redshift_metadata_rows_v2_5_fresh.csv", redshift_rows[:5000])

        cov_shape = parse_cov_dimension(text_files.get(DISTANCE_COV_PATH, ""))
        write_json(OUTDIR / "downstream_covariance_shape_v2_5_fresh.json", cov_shape)

        bridge = build_calibrator_bridge(distance_rows, redshift_rows, host_edges["host_summary"])
        write_csv(OUTDIR / "calibrator_bridge_rows_v2_5_fresh.csv", bridge["calibrator_rows"])
        write_csv(OUTDIR / "calibrator_unmatched_rows_v2_5_fresh.csv", bridge["unmatched_rows"])

        host_bridge = aggregate_host_bridge(bridge["calibrator_rows"])
        regime_summary = regime_bridge_summary(host_bridge)
        report = influence_report(host_bridge)

        write_csv(OUTDIR / "bridged_host_influence_audit_v2_5_fresh.csv", host_bridge)
        write_csv(OUTDIR / "regime_bridge_summary_v2_5_fresh.csv", regime_summary)
        write_json(OUTDIR / "report_only_influence_diagnostics_v2_5_fresh.json", report)

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

        decision = decide(residual_summary, distance_summary, cov_shape, bridge, host_bridge, report)
        write_json(OUTDIR / "decision_v2_5_fresh.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v2_5_fresh.json", ledger)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v2_5_fresh.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v2_5_fresh.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID SH0ES Downstream Bridge Audit v2.5 Fresh",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Bridge/influence audit only. No tuning, no downstream replay, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "download_ledger": download_ledger,
            "residual_summary": residual_summary,
            "distance_summary": distance_summary,
            "downstream_covariance_shape": cov_shape,
            "bridge_core": decision["bridge_core"],
            "regime_bridge_summary": regime_summary,
            "report_only_influence_diagnostics": report,
            "decision": decision,
            "claims_v2_5": CLAIMS_V2_5,
            "frozen_rule_carry_forward": FROZEN_RULE_CARRY_FORWARD,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_downstream_bridge_audit_v2_5_fresh_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_downstream_bridge_audit_v2_5_fresh_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v2_5_fresh.csv"),
                "host_edge_summary_csv": str(OUTDIR / "table2_host_edge_summary_v2_5_fresh.csv"),
                "calibrator_bridge_rows_csv": str(OUTDIR / "calibrator_bridge_rows_v2_5_fresh.csv"),
                "bridged_host_influence_csv": str(OUTDIR / "bridged_host_influence_audit_v2_5_fresh.csv"),
                "regime_bridge_summary_csv": str(OUTDIR / "regime_bridge_summary_v2_5_fresh.csv"),
                "influence_diagnostics_json": str(OUTDIR / "report_only_influence_diagnostics_v2_5_fresh.json"),
                "decision_json": str(OUTDIR / "decision_v2_5_fresh.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v2_5_fresh.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v2_5_fresh.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v2_5_fresh.txt"),
            },
            "interpretation": {
                "what_success_means": "There is enough host bridge coverage to run a later predeclared downstream influence test.",
                "what_success_does_not_mean": "This does not validate the frozen residual replay or prove any physics claim.",
                "next_required_step": "v2.6 may run a predeclared downstream influence test without tuning the frozen F160W rule.",
                "truth_boundary": CLAIMS_V2_5["truth_boundary"],
            },
        }
        write_json(OUTDIR / "shoes_downstream_bridge_audit_v2_5_fresh_summary.json", summary)

        with open(OUTDIR / "shoes_downstream_bridge_audit_v2_5_fresh_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Downstream Bridge Audit v2.5 Fresh\n\n")
            f.write("Boundary: bridge/influence audit only. No tuning. No downstream replay. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Bridge core:\n")
            f.write(json.dumps(decision["bridge_core"], indent=2, default=json_default) + "\n\n")
            f.write("Report-only influence diagnostics:\n")
            f.write(json.dumps(report, indent=2, default=json_default) + "\n\n")
            f.write("Regime bridge summary:\n")
            f.write(json.dumps(regime_summary, indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not prove SH0ES is wrong.\n")
            f.write("- This does not tune or replace the frozen rule.\n")
            f.write("- This does not run a downstream replay.\n")

        print("TAIRID SH0ES Downstream Bridge Audit v2.5 Fresh complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Downstream Bridge Audit v2.5 Fresh",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_downstream_bridge_audit_v2_5_fresh_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V2_5["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_downstream_bridge_audit_v2_5_fresh_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
