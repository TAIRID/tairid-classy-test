#!/usr/bin/env python3
"""
TAIRID SH0ES Ladder Block Map Audit v1.5

Why this test exists:
v1.4.2 succeeded in downloading the SH0ES high-level FITS candidates as valid
FITS products. It found:

    C covariance matrix shape: 3492 x 3492
    L equation matrix shape: 47 x 3492
    y data vector shape: 3492

v1.4.2 also found no explicit row labels inside those FITS files, so the NIR
lane could not be replayed yet.

This v1.5 test checks the next lawful possibility:

    Does the 3492 observation-space length correspond to a concatenation of
    SH0ES optical Cepheid rows plus SH0ES NIR Cepheid rows?

If a block-length match exists, v1.5 records it as a candidate row-map surface.
It still does NOT validate TAIRID. It still does NOT replay the frozen rule.

This test does NOT validate TAIRID.
This test does NOT tune the frozen v1.0 rule.
This test does NOT create residuals from magnitudes.
This test does NOT replay the frozen edge rule.
This test does NOT claim H0 correction or new physics.

Truth boundary:
A block-length match is not a row-level proof. It is only a candidate surface.
Replay remains blocked unless a later test proves ordering, row identity, and
which portion of y/C/L corresponds to the NIR Cepheid observations.
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


OUTDIR = Path("tairid_shoes_ladder_block_map_audit_v1_5_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO = "PantheonPlusSH0ES/DataRelease"
BRANCH_CANDIDATES = ["main", "master"]

TABLE_FILES = [
    {
        "label": "nir_orig19",
        "kind": "nir",
        "repo_path": "SH0ES_Data/R22_orig19_NIR.out",
    },
    {
        "label": "nir_orig19_wm31",
        "kind": "nir",
        "repo_path": "SH0ES_Data/R22_orig19_NIR.wm31.out",
    },
    {
        "label": "optical_orig19",
        "kind": "optical",
        "repo_path": "SH0ES_Data/optical_wes_R22_for19fromR16.dat",
    },
    {
        "label": "optical_orig19_wm31",
        "kind": "optical",
        "repo_path": "SH0ES_Data/optical_wes_R22_for19fromR16.wM31.dat",
    },
]

FITS_CANDIDATES = [
    {
        "label": "C_covariance_matrix",
        "role": "C",
        "repo_path": "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits",
    },
    {
        "label": "L_equation_matrix",
        "role": "L",
        "repo_path": "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits",
    },
    {
        "label": "y_data_vector",
        "role": "y",
        "repo_path": "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits",
    },
]

SOURCE_CONTEXT_FILES = [
    "SH0ES_Data/README.md",
    "SH0ES_Data/table2.README",
    "SH0ES_Data/table2.tex",
    "SH0ES_Data/MCMC_utils.py",
    "SH0ES_Data/run_mcmc.py",
    "SH0ES_Data/read_chains_example.py",
    "SH0ES_Data/lstsq_results.txt",
]

FROZEN_RULE_CARRIED_FORWARD = {
    "source_status": "locked by v1.0, schema-checked by v1.2, locator-scanned by v1.3, FITS-checked by v1.4.2",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "frozen_variable": "F160W-like Table2 numeric column, table2_num_7",
    "external_proxy_candidate": "H column in R22 NIR files, treated only as H/F160W-like surface candidate",
    "edge_rule": "within-host high 5% H/F160W-like magnitude minus within-host low 5% H/F160W-like magnitude",
    "low_alpha_reference_hosts": ["LMC", "SMC", "N4536"],
    "sign_break_quarantine_hosts": ["M31"],
    "hard_boundary": [
        "Do not tune host regimes in v1.5.",
        "Do not invent residuals.",
        "Do not replay the frozen edge rule in v1.5.",
        "Do not treat a block-length match as row-level proof.",
        "Do not claim H0 correction or new physics.",
    ],
}

CLAIMS_V1_5 = {
    "battery_name": "TAIRID SH0ES Ladder Block Map Audit v1.5",
    "scope": "Optical + NIR table row-count block audit against y/C/L observation-space length",
    "primary_question": (
        "Does the 3492-length SH0ES ladder observation space match a lawful optical+NIR table block "
        "candidate, and is there source/documentation evidence for the ordering?"
    ),
    "truth_boundary": (
        "This is block-map feasibility only. It does not validate TAIRID, H0 correction, or new physics."
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
    req = urllib.request.Request(url, headers={"User-Agent": "TAIRID-v1.5-shoes-block-map-audit"})
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


def maybe_decompress_gzip(data):
    if not data.startswith(b"\x1f\x8b"):
        return data, False, None
    try:
        return gzip.decompress(data), True, None
    except Exception as exc:
        return data, False, repr(exc)


def parse_table_rows(label, kind, repo_path, text):
    rows = []
    skipped = 0
    errors = []
    header_lines = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if not stripped:
            skipped += 1
            continue
        if stripped.startswith("#") or stripped.startswith("%") or stripped.startswith("-"):
            skipped += 1
            continue
        if stripped.startswith("\\") or stripped.startswith("&") or stripped.endswith("\\\\"):
            skipped += 1
            continue

        tokens = stripped.split()
        lower0 = tokens[0].lower() if tokens else ""
        if lower0 in {"host", "field", "galaxy", "name"}:
            header_lines.append(stripped)
            continue
        if len(tokens) < 4:
            skipped += 1
            continue

        numeric_count = 0
        for token in tokens[1:]:
            try:
                value = float(token.replace("D", "E"))
                if math.isfinite(value):
                    numeric_count += 1
            except Exception:
                pass

        if numeric_count < 2:
            skipped += 1
            continue

        rows.append(
            {
                "dataset_label": label,
                "kind": kind,
                "repo_path": repo_path,
                "source_line_number": line_number,
                "host_candidate": tokens[0],
                "id_candidate": tokens[3] if len(tokens) > 3 else "",
                "token_count": len(tokens),
                "numeric_token_count_after_first": numeric_count,
                "raw_line": raw,
            }
        )

    return {
        "label": label,
        "kind": kind,
        "repo_path": repo_path,
        "row_count": len(rows),
        "skipped_count": skipped,
        "parse_error_count": len(errors),
        "header_lines": header_lines[:5],
        "rows": rows,
        "errors": errors,
        "host_counts": dict(sorted(Counter(r["host_candidate"] for r in rows).items())),
    }


def inspect_fits_shape(label, role, repo_path, data_bytes, fetched):
    local_path = DOWNLOAD_DIR / safe_name(repo_path)
    local_path.write_bytes(data_bytes)
    active_bytes, was_gzip, gzip_error = maybe_decompress_gzip(data_bytes)
    active_kind = payload_kind(active_bytes)

    result = {
        "label": label,
        "role": role,
        "repo_path": repo_path,
        "local_path": str(local_path),
        "download_status": fetched.get("status"),
        "payload_kind": fetched.get("payload_kind"),
        "content_type": fetched.get("content_type"),
        "bytes": len(data_bytes),
        "was_gzip_decompressed": was_gzip,
        "is_valid_fits": False,
        "shape": None,
        "ndim": None,
        "dtype": None,
        "hdu_count": 0,
        "error": None,
    }

    if gzip_error:
        result["error"] = f"gzip_decompress_failed:{gzip_error}"
        return result

    if active_kind not in {"fits_like", "unknown_binary_or_text_payload"}:
        result["error"] = f"payload_not_fits_like:{active_kind}"
        return result

    try:
        try:
            hdul = fits.open(io.BytesIO(active_bytes), memmap=False)
        except OSError:
            hdul = fits.open(io.BytesIO(active_bytes), memmap=False, ignore_missing_simple=True)
        with hdul:
            result["hdu_count"] = len(hdul)
            hdu = hdul[0]
            data = hdu.data
            result["is_valid_fits"] = True
            if data is not None:
                result["shape"] = list(data.shape)
                result["ndim"] = int(np.ndim(data))
                result["dtype"] = str(data.dtype)
        return result
    except Exception as exc:
        result["error"] = f"fits_open_failed:{repr(exc)}"
        return result


def derive_ladder_observation_space(fits_results):
    shape_by_role = {r["role"]: r.get("shape") for r in fits_results if r.get("is_valid_fits")}
    y_shape = shape_by_role.get("y")
    c_shape = shape_by_role.get("C")
    l_shape = shape_by_role.get("L")

    y_len = y_shape[0] if isinstance(y_shape, list) and len(y_shape) == 1 else None
    c_dim = c_shape[0] if isinstance(c_shape, list) and len(c_shape) == 2 and c_shape[0] == c_shape[1] else None

    l_rows = None
    l_cols = None
    if isinstance(l_shape, list) and len(l_shape) == 2:
        l_rows, l_cols = l_shape

    return {
        "C_shape": c_shape,
        "L_shape": l_shape,
        "y_shape": y_shape,
        "y_length": y_len,
        "C_square_dimension": c_dim,
        "L_rows": l_rows,
        "L_cols": l_cols,
        "C_matches_y": c_dim is not None and y_len is not None and c_dim == y_len,
        "L_cols_match_y": l_cols is not None and y_len is not None and l_cols == y_len,
        "L_rows_match_y": l_rows is not None and y_len is not None and l_rows == y_len,
        "corrected_orientation_consistent": (
            c_dim is not None and y_len is not None and l_cols is not None and c_dim == y_len and l_cols == y_len
        ),
        "parameter_count_candidate": l_rows,
        "observation_count_candidate": y_len,
    }


def source_hints(text_map):
    rows = []
    combined = "\n\n".join(f"===== {path} =====\n{text}" for path, text in text_map.items())
    lower = combined.lower()

    patterns = [
        "r22_orig19_nir",
        "r22_orig19_nir.wm31",
        "optical_wes_r22_for19fromr16",
        "optical_wes_r22_for19fromr16.wm31",
        "allc_shoes_ceph",
        "alll_shoes_ceph",
        "ally_shoes_ceph",
        "concatenate",
        "append",
        "hstack",
        "vstack",
        "ceph",
        "nir",
        "optical",
        "equation matrix",
        "data vector",
        "covariance matrix",
    ]

    for pat in patterns:
        count = lower.count(pat)
        if count:
            rows.append({"pattern": pat, "count": count})

    window_rows = []
    for path, text in text_map.items():
        low = text.lower()
        for pat in ["r22_orig19_nir", "optical_wes", "ally_shoes", "alll_shoes", "allc_shoes", "concatenate", "append"]:
            pos = low.find(pat)
            if pos >= 0:
                start = max(0, pos - 300)
                end = min(len(text), pos + 500)
                window_rows.append(
                    {
                        "repo_path": path,
                        "pattern": pat,
                        "snippet": text[start:end].replace("\n", " ")[:900],
                    }
                )

    optical_pos = lower.find("optical")
    nir_pos = lower.find("nir")
    order_hint = "no_order_hint"
    if optical_pos >= 0 and nir_pos >= 0:
        order_hint = "optical_appears_before_nir_in_combined_context" if optical_pos < nir_pos else "nir_appears_before_optical_in_combined_context"

    return {
        "pattern_counts": rows,
        "snippets": window_rows,
        "order_hint": order_hint,
        "mentions_both_optical_and_nir": optical_pos >= 0 and nir_pos >= 0,
        "mentions_all_yLC": all(term in lower for term in ["allc_shoes", "alll_shoes", "ally_shoes"]),
    }


def build_block_candidates(table_summaries, ladder):
    y_len = ladder.get("y_length")
    c_dim = ladder.get("C_square_dimension")
    l_cols = ladder.get("L_cols")

    by_label = {r["label"]: r for r in table_summaries}
    pairs = [
        ("optical_orig19", "nir_orig19"),
        ("optical_orig19_wm31", "nir_orig19_wm31"),
        ("optical_orig19", "nir_orig19_wm31"),
        ("optical_orig19_wm31", "nir_orig19"),
    ]

    candidates = []
    for optical_label, nir_label in pairs:
        optical = by_label.get(optical_label)
        nir = by_label.get(nir_label)
        if not optical or not nir:
            continue
        optical_n = optical["row_count"]
        nir_n = nir["row_count"]
        total = optical_n + nir_n
        matches_y = y_len is not None and total == y_len
        matches_c = c_dim is not None and total == c_dim
        matches_l_cols = l_cols is not None and total == l_cols

        candidates.append(
            {
                "optical_label": optical_label,
                "nir_label": nir_label,
                "optical_row_count": optical_n,
                "nir_row_count": nir_n,
                "total_rows": total,
                "y_length": y_len,
                "C_square_dimension": c_dim,
                "L_cols": l_cols,
                "matches_y_length": matches_y,
                "matches_C_dimension": matches_c,
                "matches_L_cols": matches_l_cols,
                "full_observation_length_match": matches_y and matches_c and matches_l_cols,
                "candidate_order_optical_then_nir": {
                    "optical_start_index_0_based": 0,
                    "optical_end_index_0_based": optical_n - 1 if optical_n else None,
                    "nir_start_index_0_based": optical_n,
                    "nir_end_index_0_based": total - 1 if total else None,
                },
                "candidate_order_nir_then_optical": {
                    "nir_start_index_0_based": 0,
                    "nir_end_index_0_based": nir_n - 1 if nir_n else None,
                    "optical_start_index_0_based": nir_n,
                    "optical_end_index_0_based": total - 1 if total else None,
                },
            }
        )
    return candidates


def decide(table_summaries, ladder, block_candidates, hints):
    exact = [c for c in block_candidates if c.get("full_observation_length_match")]
    orientation_ok = bool(ladder.get("corrected_orientation_consistent"))
    code_has_both = bool(hints.get("mentions_both_optical_and_nir"))
    code_has_yLC = bool(hints.get("mentions_all_yLC"))

    if exact and orientation_ok and code_has_both and code_has_yLC:
        final_status = "optical_plus_nir_block_length_candidate_found_source_context_present"
        readiness = 7
        next_wall = (
            "The ladder observation length matches at least one optical+NIR block candidate and source context mentions "
            "the relevant files. This is still not row-level proof. Next test must prove block order and row identity."
        )
    elif exact and orientation_ok:
        final_status = "optical_plus_nir_block_length_candidate_found_no_order_proof"
        readiness = 6
        next_wall = (
            "The ladder observation length matches at least one optical+NIR block candidate, but source/order proof is missing. "
            "Do not replay yet."
        )
    elif orientation_ok:
        final_status = "ladder_observation_space_confirmed_no_optical_nir_block_match"
        readiness = 5
        next_wall = (
            "The y/C/L observation-space orientation is coherent, but optical+NIR row totals did not match. "
            "NIR replay remains blocked."
        )
    else:
        final_status = "ladder_orientation_or_download_not_ready"
        readiness = 4
        next_wall = (
            "Could not confirm a coherent y/C/L observation-space orientation. NIR replay remains blocked."
        )

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "evidence_counts": {
            "table_summary_count": len(table_summaries),
            "block_candidate_count": len(block_candidates),
            "exact_block_length_candidate_count": len(exact),
            "corrected_yLC_orientation_consistent": orientation_ok,
            "source_mentions_both_optical_and_nir": code_has_both,
            "source_mentions_all_yLC_files": code_has_yLC,
        },
        "truth_boundary": CLAIMS_V1_5["truth_boundary"],
    }


def holographic_surface_ledger(decision):
    return {
        "observable_surface": {
            "name": "SH0ES optical/NIR Cepheid row tables plus y/C/L observation-space dimensions",
            "table_files": TABLE_FILES,
            "fits_candidate_files": FITS_CANDIDATES,
        },
        "hidden_depth_sought": {
            "name": "Block-level row-map candidate from table rows into 3492-length ladder observation space",
            "why_needed": (
                "A row-map is required before the frozen H/F160W edge rule can be compared to any y/C/L outcome surface."
            ),
        },
        "boundary_that_forms_surface": {
            "release_boundary": REPO,
            "data_boundary": "Public SH0ES/PantheonPlus release files only",
            "method_boundary": "Row-count/block audit only; no fitting and no replay",
        },
        "what_can_be_reconstructed_now": [
            "Optical and NIR table row counts.",
            "Host inventories for table surfaces.",
            "Whether optical+NIR row totals match y/C/L observation-space length.",
            "Whether source files mention relevant inputs and ladder matrices.",
            "A candidate block interval only if row totals match.",
        ],
        "what_cannot_be_reconstructed_now": [
            "Final row-level identity inside y.",
            "Residual values for NIR rows.",
            "Covariance submatrix validity for NIR rows.",
            "Frozen-rule predictive validation.",
            "H0 correction or new physics.",
        ],
        "surface_noise_definition": [
            "A row-count match without order proof.",
            "Code mentions without executable row-map recovery.",
            "Treating y as residuals without proof.",
            "Treating an optical+NIR block as NIR-only.",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v1.5 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v1.5 did")
    lines.append("")
    lines.append("- Parsed SH0ES NIR Cepheid row tables.")
    lines.append("- Parsed SH0ES optical Cepheid row tables.")
    lines.append("- Rechecked the y/C/L FITS observation-space dimensions.")
    lines.append("- Corrected the v1.4.2 orientation check by testing whether L columns match y length.")
    lines.append("- Tested whether optical+NIR row totals match the 3492-length ladder observation space.")
    lines.append("- Scanned source/context files for ordering hints.")
    lines.append("- Did not replay the frozen edge rule.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v1.5 is block-map feasibility only.")
    lines.append("- A block-length match is not row-level proof.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not create residuals from H magnitudes.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 7:
        lines.append(
            "v1.6 should prove or reject block order by inspecting source code execution paths and, if possible, reconstructing the exact y-row assembly sequence. Replay remains blocked until order and row identity are proven."
        )
    elif decision["readiness_score_0_to_10"] >= 6:
        lines.append(
            "v1.6 should search source code and documentation for exact row assembly order. Without order proof, the NIR lane remains a candidate surface only."
        )
    else:
        lines.append(
            "v1.6 should stop the NIR validation lane and return to a data product with explicit residual/outcome columns."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Ladder Block Map Audit v1.5 starting.")
    print("Boundary: block-map feasibility only; no validation and no replay.")

    write_json(OUTDIR / "claims_v1_5.json", CLAIMS_V1_5)
    write_json(OUTDIR / "frozen_rule_carried_forward_v1_5.json", FROZEN_RULE_CARRIED_FORWARD)

    try:
        download_ledger = []
        table_summaries = []
        all_table_rows = []
        source_text_map = {}

        for target in TABLE_FILES:
            fetched = fetch_text_for_path(target["repo_path"])
            download_ledger.append(
                {
                    "kind": "table",
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
                continue
            local_path = DOWNLOAD_DIR / safe_name(target["repo_path"])
            local_path.write_text(fetched["text"], encoding="utf-8")
            parsed = parse_table_rows(target["label"], target["kind"], target["repo_path"], fetched["text"])
            summary = {k: v for k, v in parsed.items() if k not in {"rows", "errors"}}
            table_summaries.append(summary)
            all_table_rows.extend(parsed["rows"])
            write_csv(OUTDIR / f"{target['label']}_parsed_rows_v1_5.csv", parsed["rows"])
            write_json(OUTDIR / f"{target['label']}_parse_errors_v1_5.json", parsed["errors"])

        write_csv(OUTDIR / "all_table_rows_v1_5.csv", all_table_rows)
        write_json(OUTDIR / "table_summaries_v1_5.json", table_summaries)

        for repo_path in SOURCE_CONTEXT_FILES:
            fetched = fetch_text_for_path(repo_path)
            download_ledger.append(
                {
                    "kind": "source_context",
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
                source_text_map[repo_path] = fetched["text"]

        hints = source_hints(source_text_map)
        write_json(OUTDIR / "source_order_hints_v1_5.json", hints)
        write_csv(OUTDIR / "source_pattern_counts_v1_5.csv", hints["pattern_counts"])
        write_csv(OUTDIR / "source_snippets_v1_5.csv", hints["snippets"])

        fits_results = []
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
                fits_results.append(
                    {
                        "label": target["label"],
                        "role": target["role"],
                        "repo_path": target["repo_path"],
                        "is_valid_fits": False,
                        "shape": None,
                        "error": "download_failed",
                    }
                )
            else:
                fits_results.append(inspect_fits_shape(target["label"], target["role"], target["repo_path"], fetched["data"], fetched))

        write_csv(OUTDIR / "download_ledger_v1_5.csv", download_ledger)
        write_json(OUTDIR / "fits_shape_results_v1_5.json", fits_results)

        ladder = derive_ladder_observation_space(fits_results)
        write_json(OUTDIR / "ladder_observation_space_v1_5.json", ladder)

        block_candidates = build_block_candidates(table_summaries, ladder)
        write_json(OUTDIR / "block_map_candidates_v1_5.json", block_candidates)

        decision = decide(table_summaries, ladder, block_candidates, hints)
        write_json(OUTDIR / "decision_v1_5.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v1_5.json", ledger)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v1_5.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v1_5.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID SH0ES Ladder Block Map Audit v1.5",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Block-map feasibility only. No validation, no tuning, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "download_ledger": download_ledger,
            "table_summaries": table_summaries,
            "fits_shape_results": fits_results,
            "ladder_observation_space": ladder,
            "block_map_candidates": block_candidates,
            "source_hints": hints,
            "decision": decision,
            "claims_v1_5": CLAIMS_V1_5,
            "frozen_rule_carried_forward": FROZEN_RULE_CARRIED_FORWARD,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_ladder_block_map_audit_v1_5_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_ladder_block_map_audit_v1_5_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v1_5.csv"),
                "table_summaries_json": str(OUTDIR / "table_summaries_v1_5.json"),
                "all_table_rows_csv": str(OUTDIR / "all_table_rows_v1_5.csv"),
                "fits_shape_results_json": str(OUTDIR / "fits_shape_results_v1_5.json"),
                "ladder_observation_space_json": str(OUTDIR / "ladder_observation_space_v1_5.json"),
                "block_map_candidates_json": str(OUTDIR / "block_map_candidates_v1_5.json"),
                "source_order_hints_json": str(OUTDIR / "source_order_hints_v1_5.json"),
                "source_pattern_counts_csv": str(OUTDIR / "source_pattern_counts_v1_5.csv"),
                "source_snippets_csv": str(OUTDIR / "source_snippets_v1_5.csv"),
                "decision_json": str(OUTDIR / "decision_v1_5.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v1_5.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v1_5.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v1_5.txt"),
            },
            "interpretation": {
                "what_success_means": "A candidate block map may exist between optical+NIR tables and the high-level ladder observation space.",
                "what_success_does_not_mean": "A block-length match does not validate the frozen Table2 edge rule.",
                "what_failure_means": "If no block match exists, the NIR lane remains surface-only for this path.",
                "truth_boundary": CLAIMS_V1_5["truth_boundary"],
            },
        }

        write_json(OUTDIR / "shoes_ladder_block_map_audit_v1_5_summary.json", summary)

        with open(OUTDIR / "shoes_ladder_block_map_audit_v1_5_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Ladder Block Map Audit v1.5\n\n")
            f.write("Boundary: block-map feasibility only. No validation. No replay.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Ladder observation space:\n")
            f.write(json.dumps(ladder, indent=2, default=json_default) + "\n\n")
            f.write("Table summaries:\n")
            f.write(json.dumps(table_summaries, indent=2, default=json_default) + "\n\n")
            f.write("Block map candidates:\n")
            f.write(json.dumps(block_candidates, indent=2, default=json_default) + "\n\n")
            f.write("Source hints:\n")
            f.write(json.dumps(hints, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not create residuals from H magnitudes.\n")
            f.write("- Do not replay without order and row identity proof.\n")

        print("TAIRID SH0ES Ladder Block Map Audit v1.5 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Ladder Block Map Audit v1.5",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_ladder_block_map_audit_v1_5_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V1_5["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_ladder_block_map_audit_v1_5_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
