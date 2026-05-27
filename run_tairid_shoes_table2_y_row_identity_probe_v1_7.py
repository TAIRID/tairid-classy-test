#!/usr/bin/env python3
"""
TAIRID SH0ES Table2 Y Row Identity Probe v1.7

Why this test exists:
v1.6 confirmed that the high-level SH0ES ladder system is coherent:

    C = 3492 x 3492
    L = 47 x 3492
    y = 3492

But v1.6 did not find a simple optical+NIR block-total explanation for all
3492 rows. It also showed that Table2.tex has 3130 Cepheid rows and that the
release documentation says Table 2 is already packaged into the y vector and
related to the L matrix and C covariance matrix.

This v1.7 test checks the next lawful row-identity bridge:

    Can the first 3130 y-vector entries be reconstructed from Table2.tex
    Cepheid rows using the public SH0ES transformation surface?

The transformation tested is deliberately narrow and source-bound:

    y_candidate = F160W + 3.285 * (log10(period) - 1) - R * (V-I) + host_offset

where:
    - 3.285 is the initial slope term described in the SH0ES README.
    - R is scanned near the expected NIR Wesenheit color coefficient.
    - host_offset is estimated only as a row-identity/provenance diagnostic,
      not as a physical fit or validation result.

This test does NOT validate TAIRID.
This test does NOT tune the frozen v1.0 rule.
This test does NOT replay the frozen edge rule.
This test does NOT claim H0 correction or new physics.
This test does NOT treat y as residuals.

Truth boundary:
A successful Table2 -> y row-identity bridge proves only that Table2 rows are
lawfully mappable into the y data vector. It does not prove the TAIRID boundary
polarity correction. A later test must compute real residuals from y - theta*L
using an accepted published parameter vector before replay is allowed.
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


OUTDIR = Path("tairid_shoes_table2_y_row_identity_probe_v1_7_outputs")
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

COLOR_COEFFICIENT_GRID = [
    0.000, 0.300, 0.350, 0.386, 0.390, 0.400, 0.450, 0.500
]

SLOPE_TERM = 3.285
LOGP_CENTER = 1.0

FROZEN_RULE_CARRIED_FORWARD = {
    "source_status": "locked by v1.0, schema-checked by v1.2, locator-scanned by v1.3, FITS-checked by v1.4.2, provenance-crawled by v1.6",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "frozen_variable": "F160W-like Table2 numeric column, table2_num_7",
    "edge_rule": "within-host high 5% F160W-like magnitude minus within-host low 5% F160W-like magnitude",
    "low_alpha_reference_hosts": ["LMC", "SMC", "N4536"],
    "sign_break_quarantine_hosts": ["M31"],
    "hard_boundary": [
        "Do not tune host regimes in v1.7.",
        "Do not invent residuals.",
        "Do not replay the frozen edge rule in v1.7.",
        "Do not treat y as residuals.",
        "Do not claim H0 correction or new physics.",
    ],
}

CLAIMS_V1_7 = {
    "battery_name": "TAIRID SH0ES Table2 Y Row Identity Probe v1.7",
    "scope": "Table2-to-y vector row identity and host-offset provenance bridge",
    "primary_question": (
        "Can the Table2 Cepheid rows be reconstructed as the first 3130 rows of the SH0ES y data vector "
        "using the public initial slope and NIR Wesenheit-style transformation surface?"
    ),
    "truth_boundary": (
        "This is row identity / provenance only. It does not validate TAIRID, H0 correction, or new physics."
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
        headers={"User-Agent": "TAIRID-v1.7-table2-y-row-identity-probe"},
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
                "row_index_0_based": len(rows),
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


def build_candidate_base(rows, color_coeff):
    out = []
    for r in rows:
        logp = r["log_period"]
        value = r["f160w"] + SLOPE_TERM * (logp - LOGP_CENTER) - color_coeff * r["v_i"]
        out.append(value)
    return np.asarray(out, dtype=float)


def host_offset_fit(rows, y_segment, base_values):
    by_host = defaultdict(list)
    for i, r in enumerate(rows):
        by_host[r["host"]].append(i)

    host_offsets = {}
    host_rows = []
    corrected = np.zeros_like(base_values, dtype=float)

    for host, idxs in by_host.items():
        diffs = y_segment[idxs] - base_values[idxs]
        offset = float(np.median(diffs))
        host_offsets[host] = offset
        corrected[idxs] = base_values[idxs] + offset
        residuals = y_segment[idxs] - corrected[idxs]
        host_rows.append(
            {
                "host": host,
                "start_index": int(min(idxs)),
                "end_index": int(max(idxs)),
                "row_count": int(len(idxs)),
                "host_offset_median": offset,
                "host_offset_mean_raw": float(np.mean(diffs)),
                "raw_diff_std": float(np.std(diffs)),
                "corrected_residual_rmse": rmse(residuals),
                "corrected_residual_median_abs": median_abs(residuals),
                "corrected_residual_max_abs": max_abs(residuals),
            }
        )

    all_resid = y_segment - corrected
    host_rows = sorted(host_rows, key=lambda r: r["start_index"])
    return host_offsets, host_rows, corrected, all_resid


def scan_candidate_models(rows, y_array):
    n = len(rows)
    max_start = max(0, len(y_array) - n)

    model_rows = []
    detailed_best = None

    for color_coeff in COLOR_COEFFICIENT_GRID:
        base = build_candidate_base(rows, color_coeff)

        # Lightweight scan: evaluate start 0 plus a coarse grid and local near start.
        starts = set([0])
        starts.update(range(0, max_start + 1, 25))
        starts.update(range(0, min(max_start, 100) + 1))
        if max_start not in starts:
            starts.add(max_start)

        best_for_color = None
        for start in sorted(starts):
            seg = y_array[start:start+n]
            if len(seg) != n:
                continue
            _, _, _, resid = host_offset_fit(rows, seg, base)
            score = rmse(resid)
            row = {
                "color_coeff": color_coeff,
                "start_index": start,
                "end_index": start + n - 1,
                "row_count": n,
                "rmse_after_host_offsets": score,
                "median_abs_after_host_offsets": median_abs(resid),
                "max_abs_after_host_offsets": max_abs(resid),
                "correlation_base_to_y": float(np.corrcoef(base, seg)[0, 1]) if len(base) > 1 else None,
            }
            if best_for_color is None or row["rmse_after_host_offsets"] < best_for_color["rmse_after_host_offsets"]:
                best_for_color = row

        if best_for_color:
            model_rows.append(best_for_color)
            if detailed_best is None or best_for_color["rmse_after_host_offsets"] < detailed_best["rmse_after_host_offsets"]:
                detailed_best = best_for_color

    model_rows = sorted(model_rows, key=lambda r: (r["rmse_after_host_offsets"], r["start_index"]))
    return model_rows, detailed_best


def classify_offsets(host_rows):
    near_zero = []
    large_negative = []
    moderate_negative = []
    small_positive = []
    other = []

    for row in host_rows:
        off = row["host_offset_median"]
        if abs(off) < 0.05:
            near_zero.append(row["host"])
        elif off < -25:
            large_negative.append(row["host"])
        elif -22 < off < -15:
            moderate_negative.append(row["host"])
        elif 0.005 < off < 0.05:
            small_positive.append(row["host"])
        else:
            other.append(row["host"])

    return {
        "near_zero_offset_hosts": near_zero,
        "large_negative_offset_hosts": large_negative,
        "moderate_negative_offset_hosts": moderate_negative,
        "small_positive_offset_hosts": small_positive,
        "other_offset_hosts": other,
    }


def source_sniffs(readme_text, table2_readme_text):
    combined = (readme_text or "") + "\n\n" + (table2_readme_text or "")
    lower = combined.lower()

    patterns = [
        "initial slope term of -3.285",
        "y, data vector",
        "l, equation matrix",
        "c, covariance matrix",
        "table 2",
        "not to use this table",
        "already packaged",
        "complete nir wesenheit magnitudes",
        "covariance",
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
            end = min(len(combined), pos + 800)
            snippets.append({"pattern": pattern, "snippet": combined[start:end].replace("\n", " ")[:1200]})

    return {"pattern_counts": rows, "snippets": snippets}


def make_reconstruction_rows(rows, y_segment, base, corrected, residuals, host_offsets):
    out = []
    for i, r in enumerate(rows):
        o = dict(r)
        o["y_vector_index_0_based"] = i
        o["y_value"] = float(y_segment[i])
        o["base_transformed_value_no_host_offset"] = float(base[i])
        o["host_offset_applied"] = float(host_offsets[r["host"]])
        o["reconstructed_y_value"] = float(corrected[i])
        o["reconstruction_residual"] = float(residuals[i])
        out.append(o)
    return out


def decide(best_model, host_rows, offset_classes, n_rows, y_len, source_info):
    if not best_model:
        return {
            "final_status": "table2_y_identity_model_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "No viable transformation candidate was found.",
            "truth_boundary": CLAIMS_V1_7["truth_boundary"],
            "evidence_counts": {},
        }

    rmse_val = best_model["rmse_after_host_offsets"]
    med_abs = best_model["median_abs_after_host_offsets"]
    max_abs_val = best_model["max_abs_after_host_offsets"]
    start_ok = best_model["start_index"] == 0
    color_ok = abs(best_model["color_coeff"] - 0.386) < 0.01
    source_has_slope = any(r["pattern"] == "initial slope term of -3.285" and r["count"] > 0 for r in source_info["pattern_counts"])
    source_has_packaged = any(r["pattern"] == "already packaged" and r["count"] > 0 for r in source_info["pattern_counts"])
    anchor_offsets_present = (
        "N4258" in offset_classes["large_negative_offset_hosts"]
        and "LMC" in offset_classes["moderate_negative_offset_hosts"]
    )
    many_zero_hosts = len(offset_classes["near_zero_offset_hosts"]) >= 25

    if start_ok and color_ok and rmse_val is not None and rmse_val < 0.01 and med_abs is not None and med_abs < 0.005 and max_abs_val is not None and max_abs_val < 0.03 and anchor_offsets_present and many_zero_hosts and source_has_slope:
        final_status = "table2_y_row_identity_bridge_confirmed_no_residual_replay_yet"
        readiness = 8
        next_wall = (
            "Table2 rows appear lawfully mapped to the first y-vector block after SH0ES transformation and host offsets. "
            "Next test must compute actual residuals from y - theta*L using an accepted parameter vector before replay."
        )
    elif start_ok and rmse_val is not None and rmse_val < 0.05:
        final_status = "table2_y_row_identity_candidate_found_needs_manual_review"
        readiness = 7
        next_wall = (
            "A strong Table2-to-y identity candidate exists, but at least one strict evidence gate did not pass. "
            "Manual review or a stricter reconstruction pass is needed before residual replay."
        )
    else:
        final_status = "table2_y_row_identity_not_proven"
        readiness = 5
        next_wall = (
            "The Table2-to-y row bridge was not proven strongly enough. Do not replay frozen edge rule."
        )

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "evidence_counts": {
            "table2_row_count": n_rows,
            "y_length": y_len,
            "best_start_index": best_model["start_index"],
            "best_color_coeff": best_model["color_coeff"],
            "best_rmse_after_host_offsets": rmse_val,
            "best_median_abs_after_host_offsets": med_abs,
            "best_max_abs_after_host_offsets": max_abs_val,
            "start_index_zero": start_ok,
            "color_coeff_near_0_386": color_ok,
            "source_mentions_initial_slope": source_has_slope,
            "source_mentions_packaged_y_vector": source_has_packaged,
            "anchor_offsets_present": anchor_offsets_present,
            "near_zero_host_count": len(offset_classes["near_zero_offset_hosts"]),
            "host_offset_row_count": len(host_rows),
        },
        "truth_boundary": CLAIMS_V1_7["truth_boundary"],
    }


def holographic_surface_ledger(decision):
    return {
        "observable_surface": {
            "name": "Table2 Cepheid rows and SH0ES y data vector",
            "table2_path": TABLE2_PATH,
            "y_path": Y_FITS_PATH,
            "l_path": L_FITS_PATH,
            "c_path": C_FITS_PATH,
        },
        "hidden_depth_sought": {
            "name": "Row identity bridge from Table2 Cepheid rows into y-vector observation space",
            "why_needed": (
                "The frozen Table2 edge rule cannot be replayed against ladder residuals until Table2 rows are located in y."
            ),
        },
        "boundary_that_forms_surface": {
            "release_boundary": REPO,
            "method_boundary": "Row reconstruction identity only; no residual replay and no model validation",
        },
        "what_can_be_reconstructed_now": [
            "Table2 row count and order.",
            "First-block y-vector identity candidate.",
            "SH0ES transformed y surface from F160W, period, color, and host offsets.",
            "Host-offset classes that distinguish SN hosts from anchors.",
        ],
        "what_cannot_be_reconstructed_now": [
            "Residual vector until theta is chosen from an accepted parameter source.",
            "Frozen-rule validation.",
            "H0 correction.",
            "New physics.",
            "A new covariance characterization for custom selections.",
        ],
        "surface_noise_definition": [
            "Treating y as residuals.",
            "Treating host-offset reconstruction as physical validation.",
            "Ignoring the Table2 README warning about covariance/custom selections.",
            "Replaying the frozen edge rule before residuals are computed.",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v1.7 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v1.7 did")
    lines.append("")
    lines.append("- Parsed Table2.tex into Cepheid rows.")
    lines.append("- Read the SH0ES y data vector.")
    lines.append("- Reconstructed Table2 rows into y-space using F160W, period, color, the README slope term, and host offsets.")
    lines.append("- Tested whether the best row identity bridge starts at y index 0.")
    lines.append("- Classified host-offset structure as provenance evidence.")
    lines.append("- Did not compute residuals.")
    lines.append("- Did not replay the frozen edge rule.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v1.7 is row identity / provenance only.")
    lines.append("- y is data, not residuals.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append(
            "v1.8 should compute the actual ladder residual vector r = y - theta*L using a published or release-provided parameter vector, then inspect whether Table2 rows 0..3129 can be safely replayed against residuals. The frozen edge rule is still not replayed until residual construction passes."
        )
    elif decision["readiness_score_0_to_10"] >= 7:
        lines.append(
            "v1.8 should run a stricter reconstruction review and then compute residuals only if the row bridge passes."
        )
    else:
        lines.append(
            "v1.8 should close this route and return to a data product with explicit residual/outcome columns."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Table2 Y Row Identity Probe v1.7 starting.")
    print("Boundary: row identity only; no residuals, no validation, no replay.")

    write_json(OUTDIR / "claims_v1_7.json", CLAIMS_V1_7)
    write_json(OUTDIR / "frozen_rule_carried_forward_v1_7.json", FROZEN_RULE_CARRIED_FORWARD)

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

        write_csv(OUTDIR / "download_ledger_v1_7.csv", download_ledger)

        table2 = parse_table2_tex(text_files.get(TABLE2_PATH, ""))
        table_rows = table2["rows"]
        write_csv(OUTDIR / "table2_parsed_rows_v1_7.csv", table_rows)
        write_json(OUTDIR / "table2_parse_errors_v1_7.json", table2["errors"])

        y_array = np.asarray(fits_arrays.get(Y_FITS_PATH, np.asarray([])), dtype=float)
        l_array = np.asarray(fits_arrays.get(L_FITS_PATH, np.asarray([])), dtype=float)
        c_array = np.asarray(fits_arrays.get(C_FITS_PATH, np.asarray([])), dtype=float)

        fits_shapes = {
            "y_shape": list(y_array.shape),
            "L_shape": list(l_array.shape),
            "C_shape": list(c_array.shape),
            "y_length": int(len(y_array)) if y_array.ndim == 1 else None,
            "table2_row_count": len(table_rows),
        }
        write_json(OUTDIR / "fits_and_table_shapes_v1_7.json", fits_shapes)

        source_info = source_sniffs(
            text_files.get(README_PATH, ""),
            text_files.get(TABLE2_README_PATH, ""),
        )
        write_json(OUTDIR / "source_sniffs_v1_7.json", source_info)
        write_csv(OUTDIR / "source_pattern_counts_v1_7.csv", source_info["pattern_counts"])
        write_csv(OUTDIR / "source_snippets_v1_7.csv", source_info["snippets"])

        model_scan, best_model = scan_candidate_models(table_rows, y_array)
        write_csv(OUTDIR / "model_scan_summary_v1_7.csv", model_scan)

        if best_model:
            best_color = best_model["color_coeff"]
            best_start = best_model["start_index"]
            n = len(table_rows)
            y_segment = y_array[best_start:best_start+n]
            base = build_candidate_base(table_rows, best_color)
            host_offsets, host_rows, corrected, residuals = host_offset_fit(table_rows, y_segment, base)
            offset_classes = classify_offsets(host_rows)

            reconstruction_rows = make_reconstruction_rows(table_rows, y_segment, base, corrected, residuals, host_offsets)
        else:
            host_offsets = {}
            host_rows = []
            offset_classes = {
                "near_zero_offset_hosts": [],
                "large_negative_offset_hosts": [],
                "moderate_negative_offset_hosts": [],
                "small_positive_offset_hosts": [],
                "other_offset_hosts": [],
            }
            reconstruction_rows = []
            residuals = np.asarray([])

        write_json(OUTDIR / "host_offsets_v1_7.json", host_offsets)
        write_csv(OUTDIR / "host_offset_summary_v1_7.csv", host_rows)
        write_json(OUTDIR / "host_offset_classes_v1_7.json", offset_classes)
        write_csv(OUTDIR / "table2_y_reconstruction_rows_v1_7.csv", reconstruction_rows)

        residual_summary = {
            "count": int(len(residuals)),
            "rmse": rmse(residuals) if len(residuals) else None,
            "median_abs": median_abs(residuals) if len(residuals) else None,
            "max_abs": max_abs(residuals) if len(residuals) else None,
            "summary": summarize_numeric(residuals.tolist() if len(residuals) else []),
        }
        write_json(OUTDIR / "reconstruction_residual_summary_v1_7.json", residual_summary)

        decision = decide(
            best_model,
            host_rows,
            offset_classes,
            len(table_rows),
            int(len(y_array)) if y_array.ndim == 1 else None,
            source_info,
        )
        write_json(OUTDIR / "decision_v1_7.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v1_7.json", ledger)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v1_7.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v1_7.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID SH0ES Table2 Y Row Identity Probe v1.7",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Row identity/provenance only. No residual replay, no tuning, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "download_ledger": download_ledger,
            "fits_and_table_shapes": fits_shapes,
            "source_sniffs": source_info,
            "model_scan_summary": model_scan,
            "best_model": best_model,
            "host_offsets": host_offsets,
            "host_offset_classes": offset_classes,
            "reconstruction_residual_summary": residual_summary,
            "decision": decision,
            "claims_v1_7": CLAIMS_V1_7,
            "frozen_rule_carried_forward": FROZEN_RULE_CARRIED_FORWARD,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_table2_y_row_identity_probe_v1_7_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_table2_y_row_identity_probe_v1_7_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v1_7.csv"),
                "table2_rows_csv": str(OUTDIR / "table2_parsed_rows_v1_7.csv"),
                "fits_and_table_shapes_json": str(OUTDIR / "fits_and_table_shapes_v1_7.json"),
                "model_scan_csv": str(OUTDIR / "model_scan_summary_v1_7.csv"),
                "host_offsets_json": str(OUTDIR / "host_offsets_v1_7.json"),
                "host_offset_summary_csv": str(OUTDIR / "host_offset_summary_v1_7.csv"),
                "host_offset_classes_json": str(OUTDIR / "host_offset_classes_v1_7.json"),
                "reconstruction_rows_csv": str(OUTDIR / "table2_y_reconstruction_rows_v1_7.csv"),
                "residual_summary_json": str(OUTDIR / "reconstruction_residual_summary_v1_7.json"),
                "decision_json": str(OUTDIR / "decision_v1_7.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v1_7.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v1_7.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v1_7.txt"),
            },
            "interpretation": {
                "what_success_means": "The public Table2 rows can be lawfully identified inside the y-vector observation surface.",
                "what_success_does_not_mean": "This does not validate the frozen Table2 edge rule because y is data, not residuals.",
                "next_required_step": "Compute residuals r = y - theta*L using an accepted parameter vector before any frozen replay.",
                "truth_boundary": CLAIMS_V1_7["truth_boundary"],
            },
        }

        write_json(OUTDIR / "shoes_table2_y_row_identity_probe_v1_7_summary.json", summary)

        with open(OUTDIR / "shoes_table2_y_row_identity_probe_v1_7_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Table2 Y Row Identity Probe v1.7\n\n")
            f.write("Boundary: row identity/provenance only. No residual replay. No validation.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Best model:\n")
            f.write(json.dumps(best_model, indent=2, default=json_default) + "\n\n")
            f.write("Host offset classes:\n")
            f.write(json.dumps(offset_classes, indent=2, default=json_default) + "\n\n")
            f.write("Reconstruction residual summary:\n")
            f.write(json.dumps(residual_summary, indent=2, default=json_default) + "\n\n")
            f.write("Evidence counts:\n")
            f.write(json.dumps(decision["evidence_counts"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- y is data, not residuals.\n")
            f.write("- Do not replay until residuals are computed from y - theta*L.\n")

        print("TAIRID SH0ES Table2 Y Row Identity Probe v1.7 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Table2 Y Row Identity Probe v1.7",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_table2_y_row_identity_probe_v1_7_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V1_7["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_table2_y_row_identity_probe_v1_7_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
