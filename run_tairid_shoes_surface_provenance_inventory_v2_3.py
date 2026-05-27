#!/usr/bin/env python3
"""
TAIRID SH0ES Surface Provenance Inventory v2.3

Why this test exists:
v1.9-v2.2 produced a locked residual-surface result that survived robustness,
negative controls, and adjacent NIR matching. v2.2 also showed the important
surface-ledger lesson: the adjacent NIR H/F160W-like surface is not a truly
independent residual layer when matched rows carry identical H/F160W values.

v2.3 does not try to validate the model. It does not tune the rule. It does not
run another replay. It performs a provenance and independent-lane inventory.

Core questions:
    1. Which public release files are core ladder surfaces, adjacent surfaces,
       duplicate/overlap surfaces, covariance surfaces, documentation surfaces,
       or possible independent outcome/residual lanes?
    2. Which files are exact Git-blob duplicates or likely reused surfaces?
    3. Which SH0ES/Pantheon+ release files are eligible for the next test as a
       true external residual/outcome layer candidate?
    4. Does the NIR lane remain classified as adjacent/overlap rather than
       independent validation?
    5. What should v2.4 test next without tuning the frozen F160W result?

This test DOES:
    - fetch the public DataRelease repository tree,
    - classify files by provenance role,
    - audit known SH0ES Table2/NIR/y/L/C/lstsq surfaces,
    - compute selected file hashes,
    - parse Table2 and NIR rows for overlap diagnostics,
    - write an independent-lane candidate inventory,
    - recommend the next bounded test.

This test DOES NOT:
    - validate TAIRID,
    - claim H0 correction,
    - claim new physics,
    - claim SH0ES is wrong,
    - search for a better variable,
    - tune the frozen rule,
    - count duplicate/adjacent surfaces as independent external validation.

Truth boundary:
v2.3 is an evidence-hygiene and provenance test. A good result only means the
next lane can be chosen more honestly. It is not model validation.
"""

import csv
import hashlib
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


OUTDIR = Path("tairid_shoes_surface_provenance_inventory_v2_3_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO_OWNER = "PantheonPlusSH0ES"
REPO_NAME = "DataRelease"
REPO = f"{REPO_OWNER}/{REPO_NAME}"
BRANCH_CANDIDATES = ["main", "master"]

TABLE2_PATH = "SH0ES_Data/table2.tex"
TABLE2_README_PATH = "SH0ES_Data/table2.README"
README_PATH = "SH0ES_Data/README.md"

LSTSQ_PATH = "SH0ES_Data/lstsq_results.txt"
Y_FITS_PATH = "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits"
L_FITS_PATH = "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits"
C_FITS_PATH = "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits"

NIR_PATHS = [
    "SH0ES_Data/R22_orig19_NIR.out",
    "SH0ES_Data/R22_orig19_NIR.wm31.out",
]

KNOWN_CORE_PATHS = [
    TABLE2_PATH,
    TABLE2_README_PATH,
    README_PATH,
    LSTSQ_PATH,
    Y_FITS_PATH,
    L_FITS_PATH,
    C_FITS_PATH,
] + NIR_PATHS

CLAIMS_V2_3 = {
    "battery_name": "TAIRID SH0ES Surface Provenance Inventory v2.3",
    "scope": "Repository surface provenance, duplicate/overlap audit, and independent-lane candidate inventory",
    "primary_question": (
        "Which public release surfaces are duplicates, adjacent surfaces, core ladder surfaces, or eligible "
        "independent residual/outcome candidates for the next frozen-rule test?"
    ),
    "truth_boundary": (
        "This is evidence hygiene and provenance only. It does not validate TAIRID, H0 correction, or new physics."
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
        "Do not tune the frozen rule in v2.3.",
        "Do not treat NIR duplicate/overlap surfaces as independent validation.",
        "Do not promote a candidate lane until its residual/outcome semantics are verified.",
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


def api_url(path):
    return f"https://api.github.com/repos/{REPO}/{path}"


def raw_url(branch, repo_path):
    return f"https://raw.githubusercontent.com/{REPO}/{branch}/{repo_path}"


def media_url(branch, repo_path):
    return f"https://media.githubusercontent.com/media/{REPO}/{branch}/{repo_path}"


def fetch_url_bytes(url, timeout=120):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-v2.3-shoes-surface-provenance-inventory",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        return data, response.geturl(), response.headers.get("Content-Type", "")


def fetch_json(url, timeout=120):
    data, final_url, content_type = fetch_url_bytes(url, timeout=timeout)
    return json.loads(data.decode("utf-8", errors="replace")), final_url, content_type


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
    if len(data) < 8192 and all((32 <= b <= 126) or b in (9, 10, 13) for b in data):
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


def get_repo_tree():
    errors = []
    for branch in BRANCH_CANDIDATES:
        url = api_url(f"git/trees/{branch}?recursive=1")
        try:
            payload, final_url, content_type = fetch_json(url)
            if payload.get("tree"):
                return {
                    "status": "downloaded",
                    "branch": branch,
                    "url": url,
                    "final_url": final_url,
                    "content_type": content_type,
                    "tree": payload["tree"],
                    "truncated": payload.get("truncated"),
                    "errors": errors,
                }
            errors.append({"branch": branch, "url": url, "reason": "no_tree_in_payload"})
        except Exception as exc:
            errors.append({"branch": branch, "url": url, "error": repr(exc)})
    return {
        "status": "failed",
        "branch": None,
        "url": None,
        "final_url": None,
        "content_type": None,
        "tree": [],
        "truncated": None,
        "errors": errors,
    }


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

        values = {
            "ra": parse_float(parts[1]),
            "dec": parse_float(parts[2]),
            "period": parse_float(parts[4]),
            "v_i": parse_float(parts[5]),
            "sigma_v_i": parse_float(parts[6]),
            "f160w": parse_float(parts[7]),
            "sigma_f160w": parse_float(parts[8]),
            "metal_minus_8_69": parse_float(parts[9]),
        }

        if any(v is None for v in values.values()):
            errors.append({"line_number": line_number, "reason": "numeric_parse_failed", "line": raw[:240]})
            continue

        rows.append(
            {
                "table2_row_index_0_based": len(rows),
                "source_line_number": line_number,
                "host": parts[0],
                "id": parts[3],
                "note": parts[10],
                "raw_line": raw,
                **values,
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
        if not stripped or stripped.startswith("-"):
            skipped_count += 1
            continue
        tokens = stripped.split()
        if tokens and tokens[0].lower() == "host":
            header_line = stripped
            continue
        if len(tokens) < 11:
            skipped_count += 1
            continue

        values = {
            "ra_nir": parse_float(tokens[1]),
            "dec_nir": parse_float(tokens[2]),
            "period_nir": parse_float(tokens[4]),
            "v_i_nir": parse_float(tokens[5]),
            "sigma_v_i_nir": parse_float(tokens[6]),
            "h_mag_nir": parse_float(tokens[7]),
            "sigma_h_nir": parse_float(tokens[8]),
            "metal_minus_8_69_nir": parse_float(tokens[9]),
        }

        if any(v is None for v in values.values()):
            errors.append({"line_number": line_number, "reason": "numeric_parse_failed", "line": raw[:240]})
            continue

        rows.append(
            {
                "nir_dataset_label": label,
                "nir_repo_path": repo_path,
                "nir_row_index_0_based": len(rows),
                "nir_source_line_number": line_number,
                "host": tokens[0],
                "id": tokens[3],
                "hst_flag_nir": tokens[10],
                "raw_line_nir": raw,
                **values,
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


def match_nir_to_table2(table_rows, nir_rows):
    by_key = defaultdict(list)
    for row in table_rows:
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
        out["period_abs_diff"] = abs(best["period"] - nir["period_nir"])
        out["ra_abs_diff"] = abs(best["ra"] - nir["ra_nir"])
        out["dec_abs_diff"] = abs(best["dec"] - nir["dec_nir"])
        out["h_minus_f160w"] = nir["h_mag_nir"] - best["f160w"]
        out["abs_h_minus_f160w"] = abs(out["h_minus_f160w"])
        matched.append(out)

    unmatched_table = [
        row for row in table_rows
        if row["table2_row_index_0_based"] not in used_table_indices
    ]

    return {
        "matched_rows": matched,
        "unmatched_nir_rows": unmatched_nir,
        "unmatched_table2_rows": unmatched_table,
        "summary": {
            "nir_row_count": len(nir_rows),
            "table2_row_count": len(table_rows),
            "matched_row_count": len(matched),
            "match_fraction_of_table2": len(matched) / len(table_rows) if table_rows else None,
            "match_fraction_of_nir": len(matched) / len(nir_rows) if nir_rows else None,
            "matched_host_count": len(set(r["host"] for r in matched)),
            "unmatched_nir_count": len(unmatched_nir),
            "unmatched_table2_count": len(unmatched_table),
            "period_abs_diff_summary": summarize_numeric([r["period_abs_diff"] for r in matched]),
            "ra_abs_diff_summary": summarize_numeric([r["ra_abs_diff"] for r in matched]),
            "dec_abs_diff_summary": summarize_numeric([r["dec_abs_diff"] for r in matched]),
            "h_minus_f160w_summary": summarize_numeric([r["h_minus_f160w"] for r in matched]),
            "abs_h_minus_f160w_summary": summarize_numeric([r["abs_h_minus_f160w"] for r in matched]),
            "exact_h_equals_f160w_count": int(sum(1 for r in matched if abs(r["h_minus_f160w"]) < 1e-12)),
        },
    }


def fits_shape_summary(data):
    try:
        with fits.open(io.BytesIO(data), memmap=False) as hdul:
            rows = []
            for idx, hdu in enumerate(hdul):
                shape = None
                dtype = None
                if getattr(hdu, "data", None) is not None:
                    arr = np.asarray(hdu.data)
                    shape = list(arr.shape)
                    dtype = str(arr.dtype)
                rows.append(
                    {
                        "hdu_index": idx,
                        "hdu_name": hdu.name,
                        "hdu_type": type(hdu).__name__,
                        "shape": shape,
                        "dtype": dtype,
                    }
                )
            return {"status": "readable_fits", "hdu_summaries": rows}
    except Exception as exc:
        return {"status": "fits_read_failed", "error": repr(exc)}


def classify_path(path, size=None):
    lower = path.lower()
    name = path.split("/")[-1].lower()
    ext = Path(path).suffix.lower()

    tags = []
    role = "unclassified_surface"
    independence_score = 0
    validation_role = "unknown"
    caution = []

    if path == TABLE2_PATH:
        role = "locked_table2_surface"
        tags.extend(["table2", "cepheid", "locked_surface"])
        validation_role = "locked_internal_surface"
        caution.append("Used for frozen rule; not independent of the current lane.")
    elif path in NIR_PATHS:
        role = "adjacent_nir_overlap_surface"
        tags.extend(["nir", "cepheid", "adjacent_surface"])
        validation_role = "adjacent_overlap_not_independent"
        caution.append("v2.2 showed matched H/F160W equality; do not count as independent validation.")
    elif path in {Y_FITS_PATH, L_FITS_PATH, C_FITS_PATH, LSTSQ_PATH}:
        role = "core_ladder_fit_surface"
        tags.extend(["ladder_core", "fit_system"])
        validation_role = "core_internal_ladder_surface"
        caution.append("Necessary for residual construction, but already part of the tested internal lane.")
    elif path.endswith("README.md") or "readme" in name:
        role = "documentation_surface"
        tags.append("documentation")
        validation_role = "source_semantics"
    elif "cov" in lower or name.startswith("allc_") or ext in {".cov"}:
        role = "covariance_surface"
        tags.append("covariance")
        independence_score += 1
        validation_role = "uncertainty_structure_candidate"
    elif any(term in lower for term in ["resid", "residual", "fitres"]):
        role = "residual_outcome_candidate"
        tags.append("residual_candidate")
        independence_score += 4
        validation_role = "possible_true_external_residual_or_outcome"
    elif any(term in lower for term in ["pantheon", "sne", "sn", "hubble", "hd"]):
        role = "distance_ladder_or_supernova_outcome_candidate"
        tags.append("downstream_outcome_candidate")
        independence_score += 3
        validation_role = "possible_downstream_external_outcome"
    elif any(term in lower for term in ["ceph", "cepheid", "nir", "f160w", "r22", "table"]):
        role = "cepheid_measurement_surface"
        tags.append("cepheid_measurement")
        independence_score += 1
        validation_role = "adjacent_or_related_surface"
        caution.append("Measurement surface may overlap with Table2; needs row/provenance audit before use.")
    elif ext in {".fits", ".fit", ".dat", ".txt", ".csv", ".out", ".tex"}:
        role = "data_surface_candidate"
        tags.append("data_candidate")
        independence_score += 1
        validation_role = "needs_semantics"

    if ext:
        tags.append(f"ext:{ext}")
    if "shoes" in lower:
        tags.append("shoes")
    if "pantheon" in lower:
        tags.append("pantheon")
    if "anchor" in lower or any(anchor in lower for anchor in ["lmc", "smc", "n4258", "m31"]):
        tags.append("anchor_or_reference")

    if path in KNOWN_CORE_PATHS:
        independence_score -= 2
    if role in {"locked_table2_surface", "core_ladder_fit_surface", "adjacent_nir_overlap_surface"}:
        independence_score = min(independence_score, 0)
    if role == "documentation_surface":
        independence_score = 0

    return {
        "path": path,
        "name": path.split("/")[-1],
        "extension": ext,
        "role": role,
        "validation_role": validation_role,
        "independence_score": independence_score,
        "tags": ";".join(sorted(set(tags))),
        "caution": " | ".join(caution),
        "size": size,
    }


def rank_independent_candidates(classified_rows):
    candidates = []
    for row in classified_rows:
        score = row["independence_score"]
        lower = row["path"].lower()

        if row["role"] in {
            "residual_outcome_candidate",
            "distance_ladder_or_supernova_outcome_candidate",
            "covariance_surface",
            "data_surface_candidate",
        }:
            score += 0

        if any(term in lower for term in ["resid", "fitres", "hubble", "pantheon", "cov", "stat", "sys"]):
            score += 1

        if row["path"] in KNOWN_CORE_PATHS:
            score -= 5
        if row["role"] == "adjacent_nir_overlap_surface":
            score -= 5

        if score > 0:
            candidate = dict(row)
            candidate["candidate_score"] = score
            candidate["candidate_reason"] = candidate_reason(row)
            candidates.append(candidate)

    return sorted(candidates, key=lambda r: (-r["candidate_score"], r["path"]))


def candidate_reason(row):
    role = row["role"]
    path = row["path"]
    if role == "residual_outcome_candidate":
        return "Name suggests residual/FITRES-like outcome. Verify schema before replay."
    if role == "distance_ladder_or_supernova_outcome_candidate":
        return "Name suggests downstream distance-ladder or supernova outcome surface. Verify independence from Table2 residual lane."
    if role == "covariance_surface":
        return "Covariance/uncertainty structure may support a later likelihood or influence test."
    if role == "data_surface_candidate":
        return "Generic data surface; inspect schema and documentation before using."
    return f"Candidate from role {role}: {path}"


def git_duplicate_groups(tree):
    by_sha = defaultdict(list)
    for item in tree:
        if item.get("type") == "blob":
            sha = item.get("sha")
            if sha:
                by_sha[sha].append(item.get("path"))
    groups = []
    for sha, paths in by_sha.items():
        if len(paths) > 1:
            groups.append({"git_blob_sha": sha, "path_count": len(paths), "paths": sorted(paths)})
    return sorted(groups, key=lambda r: (-r["path_count"], r["git_blob_sha"]))


def source_sniffs(text_files):
    combined = "\n\n".join(text_files.get(path, "") for path in [README_PATH, TABLE2_README_PATH])
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
        "Pantheon",
        "SH0ES",
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


def selected_file_hashes_and_shapes():
    rows = []
    text_files = {}
    fits_shapes = {}
    download_rows = []

    for path in KNOWN_CORE_PATHS:
        prefer_media = path.lower().endswith(".fits")
        fetched = fetch_bytes_for_path(path, prefer_media=prefer_media)
        download_rows.append(
            {
                "path": path,
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
            rows.append({"path": path, "status": "download_failed"})
            continue

        local_path = DOWNLOAD_DIR / safe_name(path)
        local_path.write_bytes(fetched["data"])

        sha256 = hashlib.sha256(fetched["data"]).hexdigest()
        rows.append(
            {
                "path": path,
                "status": "downloaded",
                "bytes": len(fetched["data"]),
                "sha256": sha256,
                "payload_kind": fetched["payload_kind"],
                "local_path": str(local_path),
            }
        )

        if path.lower().endswith((".txt", ".md", ".tex", ".out", ".readme")) or fetched["payload_kind"] in {"small_text_payload"}:
            text_files[path] = fetched["data"].decode("utf-8", errors="replace")
        if path.lower().endswith(".fits"):
            fits_shapes[path] = fits_shape_summary(fetched["data"])

    return rows, text_files, fits_shapes, download_rows


def provenance_decision(tree_status, classified_rows, candidates, duplicate_groups, table2, nir_audits, fits_shapes):
    core_paths = {row["path"] for row in classified_rows if row["path"] in KNOWN_CORE_PATHS}
    missing_core = [p for p in KNOWN_CORE_PATHS if p not in core_paths]
    true_candidate_count = len(candidates)

    nir_overlap_ready = False
    nir_duplicate_like = False
    best_nir = None
    if nir_audits:
        best_nir = sorted(nir_audits, key=lambda r: (-(r.get("matched_row_count") or 0), r["dataset_label"]))[0]
        nir_overlap_ready = (best_nir.get("matched_row_count") or 0) >= 1000
        exact = best_nir.get("exact_h_equals_f160w_count") or 0
        matched = best_nir.get("matched_row_count") or 0
        nir_duplicate_like = matched > 0 and exact / matched > 0.95

    ylc_shapes_ok = all(
        path in fits_shapes and fits_shapes[path].get("status") == "readable_fits"
        for path in [Y_FITS_PATH, L_FITS_PATH, C_FITS_PATH]
    )

    gates = [
        {
            "gate": "G1_repository_tree_fetched",
            "passed": tree_status == "downloaded",
            "evidence": {"tree_status": tree_status},
        },
        {
            "gate": "G2_known_core_surfaces_present",
            "passed": len(missing_core) == 0,
            "evidence": {"missing_core_paths": missing_core, "known_core_count": len(KNOWN_CORE_PATHS)},
        },
        {
            "gate": "G3_ylc_fits_shapes_readable",
            "passed": ylc_shapes_ok,
            "evidence": {"fits_shape_keys": sorted(fits_shapes.keys())},
        },
        {
            "gate": "G4_table2_parsed",
            "passed": table2.get("row_count") == 3130,
            "evidence": {"table2_row_count": table2.get("row_count"), "parse_error_count": len(table2.get("errors", []))},
        },
        {
            "gate": "G5_nir_overlap_classified_not_independent",
            "passed": nir_overlap_ready and nir_duplicate_like,
            "evidence": {"best_nir_overlap": best_nir},
        },
        {
            "gate": "G6_independent_candidate_inventory_nonempty",
            "passed": true_candidate_count > 0,
            "evidence": {"candidate_count": true_candidate_count},
        },
        {
            "gate": "G7_duplicate_blob_audit_completed",
            "passed": duplicate_groups is not None,
            "evidence": {"duplicate_blob_group_count": len(duplicate_groups)},
        },
    ]

    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed:
        final_status = "surface_provenance_inventory_complete_true_external_lane_needed"
        readiness = 9
        next_wall = (
            "The release surfaces are inventoried, NIR is correctly classified as adjacent/overlap, and candidate external lanes exist. "
            "v2.4 should inspect the top candidate schemas before any replay."
        )
    elif "G5_nir_overlap_classified_not_independent" in failed and len(failed) <= 2:
        final_status = "surface_provenance_inventory_complete_nir_overlap_caution"
        readiness = 8
        next_wall = (
            "Inventory completed, but the NIR overlap/duplication classification needs review. Do not count NIR as independent until resolved."
        )
    elif len(failed) <= 2 and "G1_repository_tree_fetched" not in failed:
        final_status = "surface_provenance_inventory_complete_with_cautions"
        readiness = 7
        next_wall = (
            "Inventory mostly completed, but one or two evidence-hygiene gates failed. Review before selecting v2.4."
        )
    else:
        final_status = "surface_provenance_inventory_not_ready"
        readiness = 5
        next_wall = (
            "Core repository or parsing gates failed. Do not choose an external lane yet."
        )

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "failed_gates": failed,
        "gates": gates,
        "truth_boundary": CLAIMS_V2_3["truth_boundary"],
    }


def holographic_surface_ledger(decision, candidate_count, duplicate_count):
    return {
        "observable_surface": {
            "name": "Public DataRelease file tree plus selected SH0ES ladder/Table2/NIR surfaces",
            "repo": REPO,
            "known_core_paths": KNOWN_CORE_PATHS,
        },
        "hidden_depth_sought": {
            "name": "Provenance separation between duplicate, adjacent, core, and independent candidate surfaces",
            "allowed_claim": "v2.3 can classify surfaces and choose a safer next candidate lane.",
            "not_allowed_claim": "v2.3 cannot validate the frozen residual replay or prove new physics.",
        },
        "boundary_that_forms_surface": {
            "release_boundary": "PantheonPlusSH0ES/DataRelease public repository",
            "core_lane_boundary": "Table2 + y/L/C + lstsq_results used in v1.9-v2.2",
            "adjacent_boundary": "R22 NIR tables overlap Table2 F160W/H on matched rows",
            "candidate_boundary": "Files with residual/outcome/covariance/downstream semantics need schema validation before replay",
        },
        "what_can_be_reconstructed_now": [
            "Repository file inventory",
            "Git blob duplicate groups",
            "Known-core file hashes",
            "FITS shape summaries for y/L/C",
            "Table2/NIR overlap diagnostics",
            "Candidate external-lane ranking",
        ],
        "what_cannot_be_reconstructed_now": [
            "External validation",
            "H0 correction",
            "New physics",
            "A tuned likelihood model",
            "A final claim that any candidate is independent without schema verification",
        ],
        "surface_noise_definition": [
            "Counting duplicate Git blobs as separate evidence",
            "Counting NIR overlap as independent validation",
            "Promoting a candidate lane before reading its schema",
            "Changing the frozen F160W rule after inventory",
        ],
        "candidate_count": candidate_count,
        "duplicate_blob_group_count": duplicate_count,
        "decision": decision,
    }


def write_handoff(decision, candidates):
    top = candidates[:5]
    lines = []
    lines.append("# TAIRID v2.3 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v2.3 did")
    lines.append("")
    lines.append("- Fetched the public DataRelease repository tree.")
    lines.append("- Classified public files by surface/provenance role.")
    lines.append("- Audited known core Table2, NIR, y/L/C, and lstsq surfaces.")
    lines.append("- Computed selected file hashes and FITS shape summaries.")
    lines.append("- Rechecked Table2/NIR matched-surface overlap so NIR is not counted as independent validation.")
    lines.append("- Built an independent-lane candidate list for v2.4.")
    lines.append("- Did not tune, replay, or claim H0 correction.")
    lines.append("")
    lines.append("## Top candidate lanes for v2.4")
    lines.append("")
    if top:
        for candidate in top:
            lines.append(f"- `{candidate['path']}` — score `{candidate['candidate_score']}` — {candidate['candidate_reason']}")
    else:
        lines.append("- No candidate lane was ranked. v2.4 should repair the inventory first.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v2.3 is provenance and evidence hygiene only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not prove SH0ES is wrong.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append(
            "v2.4 should inspect and parse the highest-ranked candidate lanes, then stop before replay unless a legitimate residual/outcome schema is found."
        )
    else:
        lines.append(
            "v2.4 should repair the failed provenance gates before selecting any external lane."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Surface Provenance Inventory v2.3 starting.")
    print("Boundary: provenance/evidence hygiene only; no tuning, no validation, no H0 claim.")

    write_json(OUTDIR / "claims_v2_3.json", CLAIMS_V2_3)
    write_json(OUTDIR / "frozen_rule_carry_forward_v2_3.json", FROZEN_RULE_CARRY_FORWARD)

    try:
        tree_result = get_repo_tree()
        tree = tree_result["tree"]
        write_json(
            OUTDIR / "repo_tree_fetch_v2_3.json",
            {
                "status": tree_result["status"],
                "branch": tree_result["branch"],
                "url": tree_result["url"],
                "final_url": tree_result["final_url"],
                "truncated": tree_result["truncated"],
                "tree_count": len(tree),
                "errors": tree_result["errors"],
            },
        )

        tree_rows = []
        for item in tree:
            tree_rows.append(
                {
                    "path": item.get("path"),
                    "type": item.get("type"),
                    "size": item.get("size"),
                    "git_blob_sha": item.get("sha"),
                    "url": item.get("url"),
                }
            )
        write_csv(OUTDIR / "repo_tree_rows_v2_3.csv", tree_rows)

        duplicate_groups = git_duplicate_groups(tree)
        write_json(OUTDIR / "git_blob_duplicate_groups_v2_3.json", duplicate_groups)

        classified_rows = []
        for item in tree:
            if item.get("type") != "blob":
                continue
            classified = classify_path(item.get("path"), item.get("size"))
            classified["git_blob_sha"] = item.get("sha")
            classified_rows.append(classified)
        write_csv(OUTDIR / "surface_provenance_classification_v2_3.csv", classified_rows)

        candidates = rank_independent_candidates(classified_rows)
        write_csv(OUTDIR / "independent_lane_candidates_ranked_v2_3.csv", candidates)

        selected_hashes, text_files, fits_shapes, selected_downloads = selected_file_hashes_and_shapes()
        write_csv(OUTDIR / "selected_core_file_hashes_v2_3.csv", selected_hashes)
        write_json(OUTDIR / "selected_fits_shape_summaries_v2_3.json", fits_shapes)
        write_csv(OUTDIR / "selected_core_download_ledger_v2_3.csv", selected_downloads)

        table2 = parse_table2_tex(text_files.get(TABLE2_PATH, ""))
        write_csv(OUTDIR / "table2_parsed_rows_v2_3.csv", table2["rows"])
        write_json(OUTDIR / "table2_parse_errors_v2_3.json", table2["errors"])

        nir_audits = []
        all_matched_rows = []
        for nir_path in NIR_PATHS:
            label = safe_name(nir_path).replace("SH0ES_Data_", "").replace("_out", "")
            parsed = parse_nir_text(label, nir_path, text_files.get(nir_path, ""))
            write_csv(OUTDIR / f"{label}_parsed_nir_rows_v2_3.csv", parsed["rows"])
            write_json(OUTDIR / f"{label}_parse_errors_v2_3.json", parsed["errors"])

            match = match_nir_to_table2(table2["rows"], parsed["rows"])
            summary = {"dataset_label": label, "repo_path": nir_path, **match["summary"]}
            nir_audits.append(summary)
            for row in match["matched_rows"]:
                row["dataset_label"] = label
                all_matched_rows.append(row)
            write_csv(OUTDIR / f"{label}_matched_table2_nir_rows_v2_3.csv", match["matched_rows"])
            write_csv(OUTDIR / f"{label}_unmatched_nir_rows_v2_3.csv", match["unmatched_nir_rows"])
            write_json(OUTDIR / f"{label}_nir_overlap_summary_v2_3.json", summary)

        write_csv(OUTDIR / "all_nir_overlap_summaries_v2_3.csv", nir_audits)
        write_csv(OUTDIR / "all_nir_matched_rows_v2_3.csv", all_matched_rows)

        source_info = source_sniffs(text_files)
        write_json(OUTDIR / "source_sniffs_v2_3.json", source_info)
        write_csv(OUTDIR / "source_pattern_counts_v2_3.csv", source_info["pattern_counts"])
        write_csv(OUTDIR / "source_snippets_v2_3.csv", source_info["snippets"])

        role_counts = Counter(row["role"] for row in classified_rows)
        validation_role_counts = Counter(row["validation_role"] for row in classified_rows)
        write_json(
            OUTDIR / "surface_role_counts_v2_3.json",
            {
                "role_counts": dict(sorted(role_counts.items())),
                "validation_role_counts": dict(sorted(validation_role_counts.items())),
            },
        )

        decision = provenance_decision(
            tree_status=tree_result["status"],
            classified_rows=classified_rows,
            candidates=candidates,
            duplicate_groups=duplicate_groups,
            table2=table2,
            nir_audits=nir_audits,
            fits_shapes=fits_shapes,
        )
        write_json(OUTDIR / "decision_v2_3.json", decision)

        ledger = holographic_surface_ledger(decision, len(candidates), len(duplicate_groups))
        write_json(OUTDIR / "holographic_surface_ledger_v2_3.json", ledger)

        handoff = write_handoff(decision, candidates)
        (OUTDIR / "next_thread_handoff_after_v2_3.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v2_3.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID SH0ES Surface Provenance Inventory v2.3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Provenance/evidence hygiene only. No tuning, no replay, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "repo_tree": {
                "status": tree_result["status"],
                "branch": tree_result["branch"],
                "tree_count": len(tree),
                "truncated": tree_result["truncated"],
            },
            "surface_role_counts": dict(sorted(role_counts.items())),
            "validation_role_counts": dict(sorted(validation_role_counts.items())),
            "duplicate_blob_group_count": len(duplicate_groups),
            "selected_core_file_hash_count": len(selected_hashes),
            "fits_shape_summaries": fits_shapes,
            "table2_summary": {
                "row_count": table2["row_count"],
                "parse_error_count": len(table2["errors"]),
            },
            "nir_overlap_summaries": nir_audits,
            "candidate_count": len(candidates),
            "top_candidates": candidates[:10],
            "decision": decision,
            "claims_v2_3": CLAIMS_V2_3,
            "frozen_rule_carry_forward": FROZEN_RULE_CARRY_FORWARD,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_surface_provenance_inventory_v2_3_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_surface_provenance_inventory_v2_3_summary.txt"),
                "repo_tree_csv": str(OUTDIR / "repo_tree_rows_v2_3.csv"),
                "surface_classification_csv": str(OUTDIR / "surface_provenance_classification_v2_3.csv"),
                "independent_candidates_csv": str(OUTDIR / "independent_lane_candidates_ranked_v2_3.csv"),
                "duplicate_groups_json": str(OUTDIR / "git_blob_duplicate_groups_v2_3.json"),
                "core_hashes_csv": str(OUTDIR / "selected_core_file_hashes_v2_3.csv"),
                "fits_shapes_json": str(OUTDIR / "selected_fits_shape_summaries_v2_3.json"),
                "nir_overlap_csv": str(OUTDIR / "all_nir_overlap_summaries_v2_3.csv"),
                "source_sniffs_json": str(OUTDIR / "source_sniffs_v2_3.json"),
                "decision_json": str(OUTDIR / "decision_v2_3.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v2_3.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v2_3.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v2_3.txt"),
            },
            "interpretation": {
                "what_success_means": "The release surfaces were classified and a safer candidate inventory exists for v2.4.",
                "what_success_does_not_mean": "This does not validate the frozen residual replay or prove any physics claim.",
                "next_required_step": "v2.4 should parse top candidate lanes and stop before replay unless a legitimate residual/outcome schema is found.",
                "truth_boundary": CLAIMS_V2_3["truth_boundary"],
            },
        }
        write_json(OUTDIR / "shoes_surface_provenance_inventory_v2_3_summary.json", summary)

        with open(OUTDIR / "shoes_surface_provenance_inventory_v2_3_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Surface Provenance Inventory v2.3\n\n")
            f.write("Boundary: provenance/evidence hygiene only. No tuning. No replay. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Surface role counts:\n")
            f.write(json.dumps(dict(sorted(role_counts.items())), indent=2, default=json_default) + "\n\n")
            f.write("NIR overlap summaries:\n")
            f.write(json.dumps(nir_audits, indent=2, default=json_default) + "\n\n")
            f.write("Top candidates:\n")
            f.write(json.dumps(candidates[:10], indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not prove SH0ES is wrong.\n")
            f.write("- This does not tune or replace the frozen rule.\n")

        print("TAIRID SH0ES Surface Provenance Inventory v2.3 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Surface Provenance Inventory v2.3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_surface_provenance_inventory_v2_3_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V2_3["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_surface_provenance_inventory_v2_3_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
