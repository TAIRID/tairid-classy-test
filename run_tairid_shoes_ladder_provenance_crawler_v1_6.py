#!/usr/bin/env python3
"""
TAIRID SH0ES Ladder Provenance Crawler v1.6

Why this test exists:
v1.5 confirmed the high-level SH0ES ladder observation-space orientation:

    C = 3492 x 3492
    L = 47 x 3492
    y = 3492

It also showed that the simple optical+NIR table row totals do not equal 3492,
so the NIR lane still cannot be replayed. The next lawful question is whether
the public release contains any source, table, or documentation evidence that
explains how the 3492 y/C/L observation rows are assembled.

This v1.6 test performs a wider repository crawl:

    1. fetch the public repository tree,
    2. inspect SH0ES text/source/table files,
    3. inspect SH0ES FITS shapes,
    4. count candidate table surfaces,
    5. search for source/documentation provenance hints,
    6. search row-count combinations that could explain the 3492 observation space,
    7. decide whether any lawful row-provenance path exists.

This test does NOT validate TAIRID.
This test does NOT tune the frozen v1.0 rule.
This test does NOT create residuals from magnitudes.
This test does NOT replay the frozen edge rule.
This test does NOT claim H0 correction or new physics.

Truth boundary:
A provenance hint is not validation. A row-count combination is not row-level
identity. A y/C/L observation-space vector cannot be used for the NIR edge replay
unless the NIR rows can be lawfully identified.
"""

import csv
import gzip
import io
import itertools
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


OUTDIR = Path("tairid_shoes_ladder_provenance_crawler_v1_6_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO = "PantheonPlusSH0ES/DataRelease"
BRANCH_CANDIDATES = ["main", "master"]

TEXT_EXTENSIONS = {
    ".md", ".txt", ".tex", ".dat", ".out", ".csv", ".tsv",
    ".py", ".readme", ".README", ".yml", ".yaml", ".ini", ".log"
}

KNOWN_FITS = [
    "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits",
    "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits",
    "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits",
]

PATTERNS = [
    "3492",
    "ally_shoes_ceph_topantheonwt6.0_112221.fits",
    "alll_shoes_ceph_topantheonwt6.0_112221.fits",
    "allc_shoes_ceph_topantheonwt6.0_112221.fits",
    "data vector",
    "equation matrix",
    "covariance matrix",
    "equation 6",
    "compact matrix",
    "linear equations",
    "np.dot(theta,L)",
    "Y-np.dot(theta,L)",
    "R22_orig19_NIR.out",
    "R22_orig19_NIR.wm31.out",
    "optical_wes_R22_for19fromR16.dat",
    "optical_wes_R22_for19fromR16.wM31.dat",
    "table 2",
    "WFC3-IR Cepheids",
    "not to use this table",
    "covariance",
    "row",
    "ceph",
    "nir",
    "optical",
    "pantheon",
    "lstsq",
]

FROZEN_RULE_CARRIED_FORWARD = {
    "source_status": "locked by v1.0, schema-checked by v1.2, locator-scanned by v1.3, FITS-checked by v1.4.2, block-checked by v1.5",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "frozen_variable": "F160W-like Table2 numeric column, table2_num_7",
    "external_proxy_candidate": "H column in R22 NIR files, treated only as H/F160W-like surface candidate",
    "edge_rule": "within-host high 5% H/F160W-like magnitude minus within-host low 5% H/F160W-like magnitude",
    "low_alpha_reference_hosts": ["LMC", "SMC", "N4536"],
    "sign_break_quarantine_hosts": ["M31"],
    "hard_boundary": [
        "Do not tune host regimes in v1.6.",
        "Do not invent residuals.",
        "Do not replay the frozen edge rule in v1.6.",
        "Do not treat source mentions as row-level proof.",
        "Do not treat row-count combinations as row-level proof.",
        "Do not claim H0 correction or new physics.",
    ],
}

CLAIMS_V1_6 = {
    "battery_name": "TAIRID SH0ES Ladder Provenance Crawler v1.6",
    "scope": "Wide source/table/FITS crawl for 3492-row ladder observation-space provenance",
    "primary_question": (
        "Does the public release contain enough source, table, or documentation evidence to explain the 3492-row "
        "y/C/L observation space and identify whether NIR Cepheids can be lawfully row-mapped?"
    ),
    "truth_boundary": (
        "This is provenance discovery only. It does not validate TAIRID, H0 correction, or new physics."
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


def api_tree_url(branch):
    return f"https://api.github.com/repos/{REPO}/git/trees/{branch}?recursive=1"


def raw_url(branch, repo_path):
    return f"https://raw.githubusercontent.com/{REPO}/{branch}/{repo_path}"


def media_url(branch, repo_path):
    return f"https://media.githubusercontent.com/media/{REPO}/{branch}/{repo_path}"


def fetch_url_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "TAIRID-v1.6-shoes-provenance-crawler"})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()
        return data, response.geturl(), response.headers.get("Content-Type", "")


def fetch_json_url(url):
    data, final_url, content_type = fetch_url_bytes(url)
    return json.loads(data.decode("utf-8", errors="replace")), final_url, content_type


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
    if len(data) < 4096 and all((32 <= b <= 126) or b in (9, 10, 13) for b in data):
        return "small_text_payload"
    return "unknown_binary_or_text_payload"


def fetch_repo_tree():
    errors = []
    for branch in BRANCH_CANDIDATES:
        url = api_tree_url(branch)
        try:
            payload, final_url, content_type = fetch_json_url(url)
            rows = []
            for item in payload.get("tree", []):
                if item.get("type") == "blob":
                    rows.append(
                        {
                            "branch": branch,
                            "path": item.get("path", ""),
                            "size": item.get("size"),
                            "sha": item.get("sha", ""),
                            "url": raw_url(branch, item.get("path", "")),
                        }
                    )
            return {"status": "downloaded", "branch": branch, "rows": rows, "errors": errors}
        except Exception as exc:
            errors.append({"branch": branch, "url": url, "error": repr(exc)})
    return {"status": "failed", "branch": None, "rows": [], "errors": errors}


def fetch_bytes_for_path(repo_path, branch, prefer_media=False):
    errors = []
    urls = [media_url(branch, repo_path), raw_url(branch, repo_path)] if prefer_media else [raw_url(branch, repo_path), media_url(branch, repo_path)]
    for url in urls:
        try:
            data, final_url, content_type = fetch_url_bytes(url)
            kind = payload_kind(data)
            if kind in {"html_payload", "not_found_payload"}:
                errors.append({"url": url, "final_url": final_url, "content_type": content_type, "payload_kind": kind})
                continue
            if kind in {"git_lfs_pointer", "small_text_payload"} and "raw.githubusercontent.com" in url and repo_path.lower().endswith(".fits"):
                errors.append({"url": url, "final_url": final_url, "content_type": content_type, "payload_kind": kind, "note": "trying media URL"})
                continue
            return {
                "status": "downloaded",
                "url": url,
                "final_url": final_url,
                "content_type": content_type,
                "bytes": len(data),
                "payload_kind": kind,
                "data": data,
                "errors": errors,
            }
        except Exception as exc:
            errors.append({"url": url, "error": repr(exc)})
    return {
        "status": "failed",
        "url": None,
        "final_url": None,
        "content_type": None,
        "bytes": 0,
        "payload_kind": "download_failed",
        "data": b"",
        "errors": errors,
    }


def extension(path):
    name = Path(path).name
    suffix = Path(name).suffix
    return suffix if suffix else ""


def is_text_candidate(path):
    ext = extension(path)
    if ext in TEXT_EXTENSIONS:
        return True
    low = path.lower()
    return low.endswith(".readme") or low.endswith("readme")


def is_shoes_candidate(path):
    return path.startswith("SH0ES_Data/")


def decode_text(data):
    return data.decode("utf-8", errors="replace")


def count_latex_table_rows(text):
    count = 0
    host_counts = Counter()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("\\") or s.startswith("%"):
            continue
        if "&" in s and s.endswith("\\\\"):
            if "colhead" in s.lower() or "table" in s.lower():
                continue
            parts = [p.strip() for p in s.replace("\\\\", "").split("&")]
            if len(parts) >= 4:
                count += 1
                host_counts[parts[0]] += 1
    return count, dict(sorted(host_counts.items()))


def count_whitespace_table_rows(text):
    count = 0
    host_counts = Counter()
    token_hist = Counter()
    skipped = 0
    for line in text.splitlines():
        raw = line.rstrip("\n")
        s = raw.strip()
        if not s:
            skipped += 1
            continue
        if s.startswith("#") or s.startswith("%") or s.startswith("-") or s.startswith("\\"):
            skipped += 1
            continue
        tokens = s.split()
        if not tokens or tokens[0].lower() in {"host", "field", "galaxy", "name"}:
            skipped += 1
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
        if numeric_count >= 2:
            count += 1
            host_counts[tokens[0]] += 1
            token_hist[len(tokens)] += 1
        else:
            skipped += 1
    return count, dict(sorted(host_counts.items())), dict(sorted(token_hist.items())), skipped


def search_patterns(path, text):
    low = text.lower()
    rows = []
    snippets = []
    for pattern in PATTERNS:
        pat = pattern.lower()
        count = low.count(pat)
        if count:
            rows.append({"repo_path": path, "pattern": pattern, "count": count})
            pos = low.find(pat)
            start = max(0, pos - 350)
            end = min(len(text), pos + 700)
            snippets.append(
                {
                    "repo_path": path,
                    "pattern": pattern,
                    "snippet": text[start:end].replace("\n", " ")[:1200],
                }
            )
    return rows, snippets


def maybe_decompress_gzip(data):
    if not data.startswith(b"\x1f\x8b"):
        return data, False, None
    try:
        return gzip.decompress(data), True, None
    except Exception as exc:
        return data, False, repr(exc)


def inspect_fits_shape(path, data_bytes, fetched):
    active_bytes, was_gzip, gzip_error = maybe_decompress_gzip(data_bytes)
    active_kind = payload_kind(active_bytes)
    role = "unknown"
    low = path.lower()
    if "allc_" in low:
        role = "C"
    elif "alll_" in low:
        role = "L"
    elif "ally_" in low:
        role = "y"

    result = {
        "repo_path": path,
        "role": role,
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


def derive_ladder(fits_rows):
    shape_by_role = {}
    for row in fits_rows:
        if row.get("is_valid_fits") and row.get("role") in {"C", "L", "y"}:
            shape_by_role[row["role"]] = row.get("shape")
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
        "corrected_orientation_consistent": c_dim is not None and y_len is not None and l_cols is not None and c_dim == y_len and l_cols == y_len,
        "parameter_count_candidate": l_rows,
        "observation_count_candidate": y_len,
    }


def build_surface_inventory(text_rows):
    inventory = []
    for row in text_rows:
        counts = []
        if row.get("whitespace_row_count", 0) > 0:
            counts.append(("whitespace", row["whitespace_row_count"]))
        if row.get("latex_table_row_count", 0) > 0:
            counts.append(("latex", row["latex_table_row_count"]))
        for method, count in counts:
            inventory.append(
                {
                    "repo_path": row["repo_path"],
                    "method": method,
                    "row_count": count,
                    "extension": row.get("extension", ""),
                    "host_count": len(row.get("host_counts", {})) if isinstance(row.get("host_counts"), dict) else None,
                }
            )
    return sorted(inventory, key=lambda r: (-r["row_count"], r["repo_path"], r["method"]))


def combo_search(surface_inventory, target):
    candidates = []
    usable = [r for r in surface_inventory if r["row_count"] and r["row_count"] > 0]
    compact = []
    seen = set()
    for row in usable:
        key = (row["repo_path"], row["method"], row["row_count"])
        if key not in seen:
            compact.append(row)
            seen.add(key)

    for k in [1, 2, 3]:
        for combo in itertools.combinations(compact, k):
            total = sum(c["row_count"] for c in combo)
            delta = total - target if target is not None else None
            if target is None:
                continue
            if delta == 0 or abs(delta) <= 100:
                candidates.append(
                    {
                        "combo_size": k,
                        "total_rows": total,
                        "target": target,
                        "delta": delta,
                        "absolute_delta": abs(delta),
                        "files": " + ".join(f"{c['repo_path']}[{c['method']}:{c['row_count']}]" for c in combo),
                        "exact_match": delta == 0,
                    }
                )
    return sorted(candidates, key=lambda r: (r["absolute_delta"], r["combo_size"], r["files"]))[:200]


def decide(ladder, combo_candidates, pattern_rows):
    target = ladder.get("y_length")
    orientation_ok = bool(ladder.get("corrected_orientation_consistent"))
    exact = [c for c in combo_candidates if c.get("exact_match")]
    near = [c for c in combo_candidates if not c.get("exact_match")]
    pattern_count = len(pattern_rows)
    explicit_3492_mentions = [r for r in pattern_rows if r["pattern"] == "3492"]

    if exact and orientation_ok and explicit_3492_mentions:
        final_status = "exact_row_count_provenance_candidate_found_needs_row_identity"
        readiness = 7
        next_wall = "An exact 3492 row-count combination exists with source provenance hints. Next test must prove actual row order and identity before replay."
    elif exact and orientation_ok:
        final_status = "exact_row_count_combination_found_without_source_order"
        readiness = 6
        next_wall = "An exact 3492 row-count combination exists, but source/order proof is missing. Replay remains blocked."
    elif near and orientation_ok:
        final_status = "near_row_count_candidates_only_no_provenance"
        readiness = 5
        next_wall = "Only near row-count candidates exist. No lawful row map has been proven. NIR replay remains blocked."
    elif orientation_ok:
        final_status = "ladder_dimensions_confirmed_no_row_provenance"
        readiness = 4
        next_wall = "The y/C/L ladder dimensions are coherent, but no row-count provenance path was found. Close NIR validation unless external documentation supplies row mapping."
    else:
        final_status = "ladder_dimensions_not_confirmed"
        readiness = 3
        next_wall = "Could not confirm y/C/L dimensions in this run. Retry only if download failed."

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "evidence_counts": {
            "target_y_length": target,
            "corrected_yLC_orientation_consistent": orientation_ok,
            "combo_candidate_count_within_100": len(combo_candidates),
            "exact_combo_count": len(exact),
            "near_combo_count": len(near),
            "pattern_hit_count": pattern_count,
            "explicit_3492_pattern_count": len(explicit_3492_mentions),
        },
        "truth_boundary": CLAIMS_V1_6["truth_boundary"],
    }


def holographic_surface_ledger(decision):
    return {
        "observable_surface": {
            "name": "Public SH0ES release tree, text/source/table files, and y/C/L FITS dimensions",
            "repo": REPO,
        },
        "hidden_depth_sought": {
            "name": "3492-row observation-space provenance and possible NIR row-map path",
            "why_needed": "Without row provenance, the NIR high/low edge surface cannot be tied to y/C/L outcome depth.",
        },
        "boundary_that_forms_surface": {
            "release_boundary": REPO,
            "method_boundary": "Repository crawl, row counting, FITS dimension audit, and source pattern search only",
        },
        "what_can_be_reconstructed_now": [
            "Candidate table row counts across public SH0ES files.",
            "FITS y/C/L dimensions.",
            "Source/documentation snippets mentioning ladder files and table surfaces.",
            "Near or exact row-count combinations up to three surfaces.",
            "Whether a later row-identity test is justified.",
        ],
        "what_cannot_be_reconstructed_now": [
            "NIR row identity inside y.",
            "Residual values for NIR rows.",
            "A valid covariance submatrix for NIR rows.",
            "Frozen-rule predictive validation.",
            "H0 correction or new physics.",
        ],
        "surface_noise_definition": [
            "Any row-count sum without source order.",
            "Any source mention without executable row-map construction.",
            "Any use of y as if it were a NIR residual vector.",
            "Any inferred residual from H magnitudes.",
        ],
        "decision": decision,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v1.6 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v1.6 did")
    lines.append("")
    lines.append("- Crawled the public repository tree.")
    lines.append("- Downloaded SH0ES text/source/table candidates.")
    lines.append("- Counted candidate whitespace and LaTeX table surfaces.")
    lines.append("- Downloaded SH0ES FITS candidates and rechecked y/C/L dimensions.")
    lines.append("- Searched source/documentation snippets for 3492-row provenance and matrix assembly hints.")
    lines.append("- Searched row-count combinations up to three surfaces against the y length.")
    lines.append("- Did not replay the frozen edge rule.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v1.6 is provenance discovery only.")
    lines.append("- Row-count combinations are not row identity.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not create residuals from H magnitudes.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 7:
        lines.append("v1.7 should attempt exact row-order reconstruction from the strongest provenance candidate. Replay is still blocked until row identity is proven.")
    elif decision["readiness_score_0_to_10"] >= 6:
        lines.append("v1.7 should inspect the exact candidate combination and search for source order. Replay remains blocked.")
    elif decision["readiness_score_0_to_10"] >= 5:
        lines.append("v1.7 should either obtain external documentation for the row map or close the NIR validation lane as surface-only.")
    else:
        lines.append("v1.7 should close the NIR validation lane and return to a data product with explicit residual/outcome columns.")
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Ladder Provenance Crawler v1.6 starting.")
    print("Boundary: provenance discovery only; no validation and no replay.")

    write_json(OUTDIR / "claims_v1_6.json", CLAIMS_V1_6)
    write_json(OUTDIR / "frozen_rule_carried_forward_v1_6.json", FROZEN_RULE_CARRIED_FORWARD)

    try:
        download_ledger = []

        tree = fetch_repo_tree()
        write_json(OUTDIR / "repo_tree_fetch_v1_6.json", tree)
        if tree["status"] != "downloaded":
            decision = {
                "final_status": "repo_tree_download_failed",
                "readiness_score_0_to_10": 3,
                "next_wall": "Repository tree could not be fetched.",
                "evidence_counts": {},
                "truth_boundary": CLAIMS_V1_6["truth_boundary"],
            }
            write_json(OUTDIR / "decision_v1_6.json", decision)
            return

        branch = tree["branch"]
        repo_rows = tree["rows"]
        write_csv(OUTDIR / "repo_tree_inventory_v1_6.csv", repo_rows)

        text_paths = [
            r["path"] for r in repo_rows
            if is_shoes_candidate(r["path"]) and is_text_candidate(r["path"]) and (r.get("size") is None or r.get("size") <= 8_000_000)
        ]

        fits_paths = sorted(set(KNOWN_FITS + [r["path"] for r in repo_rows if is_shoes_candidate(r["path"]) and r["path"].lower().endswith(".fits")]))

        text_summary_rows = []
        pattern_rows = []
        snippet_rows = []

        for path in sorted(text_paths):
            fetched = fetch_bytes_for_path(path, branch, prefer_media=False)
            download_ledger.append(
                {
                    "kind": "text_candidate",
                    "repo_path": path,
                    "status": fetched["status"],
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
            text = decode_text(fetched["data"])

            latex_count, latex_hosts = count_latex_table_rows(text)
            white_count, white_hosts, token_hist, skipped = count_whitespace_table_rows(text)
            prow, srow = search_patterns(path, text)
            pattern_rows.extend(prow)
            snippet_rows.extend(srow)

            host_counts = white_hosts if white_count >= latex_count else latex_hosts

            text_summary_rows.append(
                {
                    "repo_path": path,
                    "extension": extension(path),
                    "bytes": fetched["bytes"],
                    "whitespace_row_count": white_count,
                    "latex_table_row_count": latex_count,
                    "selected_row_count": max(white_count, latex_count),
                    "selected_method": "whitespace" if white_count >= latex_count else "latex",
                    "host_counts": host_counts,
                    "host_count": len(host_counts),
                    "token_histogram": token_hist,
                    "skipped_line_count": skipped,
                    "pattern_hit_count": sum(r["count"] for r in prow),
                }
            )

        write_csv(OUTDIR / "download_ledger_v1_6.csv", download_ledger)
        write_json(OUTDIR / "text_surface_summaries_v1_6.json", text_summary_rows)
        write_csv(OUTDIR / "source_pattern_counts_v1_6.csv", pattern_rows)
        write_csv(OUTDIR / "source_snippets_v1_6.csv", snippet_rows)

        fits_rows = []
        for path in fits_paths:
            fetched = fetch_bytes_for_path(path, branch, prefer_media=True)
            download_ledger.append(
                {
                    "kind": "fits_candidate",
                    "repo_path": path,
                    "status": fetched["status"],
                    "url": fetched["url"],
                    "final_url": fetched.get("final_url"),
                    "content_type": fetched.get("content_type"),
                    "payload_kind": fetched.get("payload_kind"),
                    "bytes": fetched["bytes"],
                    "errors": json.dumps(fetched["errors"], default=json_default),
                }
            )
            if fetched["status"] == "downloaded":
                local_path = DOWNLOAD_DIR / safe_name(path)
                local_path.write_bytes(fetched["data"])
                fits_rows.append(inspect_fits_shape(path, fetched["data"], fetched))
            else:
                fits_rows.append({"repo_path": path, "role": "unknown", "is_valid_fits": False, "shape": None, "error": "download_failed"})

        write_csv(OUTDIR / "download_ledger_v1_6.csv", download_ledger)
        write_json(OUTDIR / "fits_shape_inventory_v1_6.json", fits_rows)

        ladder = derive_ladder(fits_rows)
        write_json(OUTDIR / "ladder_observation_space_v1_6.json", ladder)

        surface_inventory = build_surface_inventory(text_summary_rows)
        write_csv(OUTDIR / "surface_row_count_inventory_v1_6.csv", surface_inventory)

        combo_candidates = combo_search(surface_inventory, ladder.get("y_length"))
        write_csv(OUTDIR / "row_count_combo_candidates_v1_6.csv", combo_candidates)

        decision = decide(ladder, combo_candidates, pattern_rows)
        write_json(OUTDIR / "decision_v1_6.json", decision)

        ledger = holographic_surface_ledger(decision)
        write_json(OUTDIR / "holographic_surface_ledger_v1_6.json", ledger)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v1_6.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v1_6.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID SH0ES Ladder Provenance Crawler v1.6",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Provenance discovery only. No validation, no tuning, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "branch": branch,
            "repo_file_count": len(repo_rows),
            "text_candidate_count": len(text_paths),
            "fits_candidate_count": len(fits_paths),
            "text_surface_summaries": text_summary_rows,
            "fits_shape_inventory": fits_rows,
            "ladder_observation_space": ladder,
            "surface_row_count_inventory": surface_inventory,
            "row_count_combo_candidates": combo_candidates,
            "source_pattern_counts": pattern_rows,
            "source_snippets": snippet_rows[:200],
            "decision": decision,
            "claims_v1_6": CLAIMS_V1_6,
            "frozen_rule_carried_forward": FROZEN_RULE_CARRIED_FORWARD,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_ladder_provenance_crawler_v1_6_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_ladder_provenance_crawler_v1_6_summary.txt"),
                "repo_tree_csv": str(OUTDIR / "repo_tree_inventory_v1_6.csv"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v1_6.csv"),
                "text_surface_summaries_json": str(OUTDIR / "text_surface_summaries_v1_6.json"),
                "fits_shape_inventory_json": str(OUTDIR / "fits_shape_inventory_v1_6.json"),
                "ladder_observation_space_json": str(OUTDIR / "ladder_observation_space_v1_6.json"),
                "surface_row_count_inventory_csv": str(OUTDIR / "surface_row_count_inventory_v1_6.csv"),
                "row_count_combo_candidates_csv": str(OUTDIR / "row_count_combo_candidates_v1_6.csv"),
                "source_pattern_counts_csv": str(OUTDIR / "source_pattern_counts_v1_6.csv"),
                "source_snippets_csv": str(OUTDIR / "source_snippets_v1_6.csv"),
                "decision_json": str(OUTDIR / "decision_v1_6.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v1_6.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v1_6.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v1_6.txt"),
            },
            "interpretation": {
                "what_success_means": "A public provenance path may exist for the 3492-row ladder observation space.",
                "what_success_does_not_mean": "A provenance hint or row-count combination does not validate the frozen Table2 edge rule.",
                "what_failure_means": "If no provenance path exists, the NIR lane should be closed as surface-only for validation purposes.",
                "truth_boundary": CLAIMS_V1_6["truth_boundary"],
            },
        }

        write_json(OUTDIR / "shoes_ladder_provenance_crawler_v1_6_summary.json", summary)

        with open(OUTDIR / "shoes_ladder_provenance_crawler_v1_6_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Ladder Provenance Crawler v1.6\n\n")
            f.write("Boundary: provenance discovery only. No validation. No replay.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Ladder observation space:\n")
            f.write(json.dumps(ladder, indent=2, default=json_default) + "\n\n")
            f.write("Top surface row-count inventory:\n")
            f.write(json.dumps(surface_inventory[:50], indent=2, default=json_default) + "\n\n")
            f.write("Row-count combination candidates:\n")
            f.write(json.dumps(combo_candidates[:50], indent=2, default=json_default) + "\n\n")
            f.write("Evidence counts:\n")
            f.write(json.dumps(decision["evidence_counts"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not create residuals from H magnitudes.\n")
            f.write("- Do not replay without row identity proof.\n")

        print("TAIRID SH0ES Ladder Provenance Crawler v1.6 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Ladder Provenance Crawler v1.6",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_ladder_provenance_crawler_v1_6_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V1_6["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_ladder_provenance_crawler_v1_6_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
