#!/usr/bin/env python3
"""
TAIRID SH0ES Candidate Schema Scout v2.4

Why this test exists:
v2.3 completed the surface provenance inventory and found that the NIR lane is
adjacent/overlap, not independent validation. It also found many candidate files,
but many of the top-ranked residual-looking files are simulations, not the next
right external lane.

v2.4 does not replay the frozen rule. It inspects the highest-value candidate
schemas and stops before any model claim.

Core question:
    Is there a legitimate external residual/outcome schema in the public
    Pantheon+SH0ES release that can support the next bounded test?

This test DOES:
    - download and inspect selected downstream distance/covariance files,
    - inspect selected redshift/host metadata surfaces,
    - inspect selected SH0ES optical Cepheid surfaces,
    - inspect one simulation FITRES file only as a schema-control lane,
    - classify each target as direct replay, bridge-required, simulation-only,
      covariance-only, metadata-only, or not usable,
    - choose the next bounded test.

This test DOES NOT:
    - validate TAIRID,
    - claim H0 correction,
    - claim new physics,
    - claim SH0ES is wrong,
    - tune the frozen rule,
    - search for a better F160W-like variable,
    - replay the locked F160W residual edge against downstream distances yet.

Truth boundary:
v2.4 is a schema scout only. A successful result means we found a safer next
surface to inspect or bridge. It is not model validation.
"""

import csv
import gzip
import io
import json
import math
import re
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


OUTDIR = Path("tairid_shoes_candidate_schema_scout_v2_4_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO = "PantheonPlusSH0ES/DataRelease"
BRANCH_CANDIDATES = ["main", "master"]

TARGETS = [
    {
        "label": "pantheon_shoes_distance_table",
        "repo_path": "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",
        "target_role": "downstream_distance_outcome_candidate",
        "priority": 1,
    },
    {
        "label": "pantheon_shoes_distance_readme",
        "repo_path": "Pantheon+_Data/4_DISTANCES_AND_COVAR/README",
        "target_role": "documentation_semantics",
        "priority": 1,
    },
    {
        "label": "pantheon_shoes_stat_sys_cov",
        "repo_path": "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov",
        "target_role": "downstream_covariance_candidate",
        "priority": 2,
    },
    {
        "label": "pantheon_shoes_statonly_cov",
        "repo_path": "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STATONLY.cov",
        "target_role": "downstream_covariance_candidate",
        "priority": 2,
    },
    {
        "label": "all_redshifts_pvs",
        "repo_path": "Pantheon+_Data/1_DATA/all_redshifts_PVs.csv",
        "target_role": "redshift_velocity_metadata_candidate",
        "priority": 3,
    },
    {
        "label": "hostgal_logmass",
        "repo_path": "Pantheon+_Data/1_DATA/header_overrides/HOSTGAL_LOGMASS.txt",
        "target_role": "host_metadata_candidate",
        "priority": 4,
    },
    {
        "label": "redshift_cmb_override",
        "repo_path": "Pantheon+_Data/1_DATA/header_overrides/REDSHIFT_CMB.txt",
        "target_role": "redshift_metadata_candidate",
        "priority": 4,
    },
    {
        "label": "shoes_optical_wes",
        "repo_path": "SH0ES_Data/optical_wes_R22_for19fromR16.dat",
        "target_role": "adjacent_cepheid_measurement_candidate",
        "priority": 5,
    },
    {
        "label": "shoes_optical_wes_wm31",
        "repo_path": "SH0ES_Data/optical_wes_R22_for19fromR16.wM31.dat",
        "target_role": "adjacent_cepheid_measurement_candidate",
        "priority": 5,
    },
    {
        "label": "simulation_fitres_schema_control",
        "repo_path": "Pantheon+_Data/6_SIMULATIONS/LCFITS/SIM_0001.FITRES.gz",
        "target_role": "simulation_residual_schema_control",
        "priority": 6,
    },
]

CLAIMS_V2_4 = {
    "battery_name": "TAIRID SH0ES Candidate Schema Scout v2.4",
    "scope": "Schema inspection for external/downstream candidate lanes after v2.3 provenance inventory",
    "primary_question": (
        "Which candidate files contain real downstream outcome, covariance, metadata, or simulation-only schemas, "
        "and what next test is allowed without tuning the frozen F160W rule?"
    ),
    "truth_boundary": (
        "This is schema scouting only. It does not validate TAIRID, H0 correction, or new physics."
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
        "Do not tune the frozen rule in v2.4.",
        "Do not run a replay in v2.4.",
        "Do not treat simulations as real external validation.",
        "Do not treat downstream distance rows as direct Cepheid rows unless a bridge is explicitly established.",
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
        headers={"User-Agent": "TAIRID-v2.4-shoes-candidate-schema-scout"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        data = response.read()
        return data, response.geturl(), response.headers.get("Content-Type", "")


def payload_kind(data):
    head = data[:512]
    text = head.decode("utf-8", errors="replace").lower()
    if head.startswith(b"\x1f\x8b"):
        return "gzip_payload"
    if "version https://git-lfs.github.com/spec" in text:
        return "git_lfs_pointer"
    if text.lstrip().startswith("<!doctype html") or text.lstrip().startswith("<html"):
        return "html_payload"
    if "404: not found" in text or "not found" in text[:100]:
        return "not_found_payload"
    if len(data) < 8192 and all((32 <= b <= 126) or b in (9, 10, 13) for b in data):
        return "small_text_payload"
    return "unknown_binary_or_text_payload"


def fetch_bytes_for_path(repo_path, prefer_media=True):
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
                if kind == "git_lfs_pointer" and "raw.githubusercontent.com" in url:
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


def decode_text(data, repo_path):
    if repo_path.lower().endswith(".gz") or data[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(data).decode("utf-8", errors="replace"), True
        except Exception:
            return data.decode("utf-8", errors="replace"), False
    return data.decode("utf-8", errors="replace"), False


def token_is_number(token):
    try:
        float(str(token).replace("D", "E"))
        return True
    except Exception:
        return False


def detect_semantic_columns(columns):
    low = [str(c).lower() for c in columns]
    return {
        "has_id_like": any(c in {"cid", "snid", "name", "id"} or "cid" in c or "snid" in c for c in low),
        "has_redshift_like": any(c in {"z", "zhd", "zcmb", "zhel", "redshift"} or "redshift" in c for c in low),
        "has_mu_like": any(c in {"mu", "muerr", "mu_sh0es"} or "mu" in c for c in low),
        "has_distance_like": any("dist" in c or "dl" == c for c in low),
        "has_residual_like": any("resid" in c or "residual" in c or "pull" in c for c in low),
        "has_calibrator_like": any("calib" in c or "cepheid" in c or "sh0es" in c for c in low),
        "has_host_like": any("host" in c for c in low),
        "has_mass_like": any("mass" in c for c in low),
        "has_covariance_like": any("cov" in c for c in low),
        "has_photometry_like": any(c in {"mjd", "flux", "fluxerr", "flt", "band", "mag", "magerr"} or "flux" in c or "mag" in c for c in low),
        "has_table2_direct_keys": (
            any(c == "host" for c in low)
            and any(c in {"id", "cepheid_id", "cephid"} for c in low)
            and any(c in {"f160w", "h", "h_mag"} or "f160w" in c for c in low)
        ),
    }


def parse_snana_or_table_text(text, max_rows=10000):
    lines = text.splitlines()
    columns = []
    data_rows = []
    comments = []
    varnames_line = None
    first_noncomment = None
    data_prefixes = {}

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comments.append(stripped[:500])
            continue
        if stripped.upper().startswith("VARNAMES:"):
            varnames_line = stripped
            columns = stripped.split()[1:]
            continue
        if stripped.upper().startswith("SN:") or stripped.upper().startswith("ROW:"):
            prefix = stripped.split()[0].rstrip(":")
            data_prefixes[prefix] = data_prefixes.get(prefix, 0) + 1
            parts = stripped.split()[1:]
            if columns and len(parts) >= len(columns):
                data_rows.append(dict(zip(columns, parts[:len(columns)])))
            elif not columns:
                data_rows.append({"raw": stripped})
            if len(data_rows) >= max_rows:
                break
            continue
        if first_noncomment is None:
            first_noncomment = stripped
            maybe_cols = stripped.split()
            numeric_ratio = sum(token_is_number(t) for t in maybe_cols) / len(maybe_cols) if maybe_cols else 0
            if numeric_ratio < 0.5 and len(maybe_cols) >= 2:
                columns = maybe_cols
                continue
        if columns:
            parts = stripped.split()
            if len(parts) >= len(columns):
                data_rows.append(dict(zip(columns, parts[:len(columns)])))
                if len(data_rows) >= max_rows:
                    break

    if not columns and first_noncomment:
        parts = first_noncomment.split()
        if parts and sum(token_is_number(t) for t in parts) / len(parts) >= 0.5:
            columns = [f"col_{i}" for i in range(len(parts))]
            data_rows.append(dict(zip(columns, parts)))

    return {
        "line_count": len(lines),
        "comment_count": len(comments),
        "varnames_line": varnames_line,
        "first_noncomment": first_noncomment,
        "columns": columns,
        "column_count": len(columns),
        "parsed_preview_row_count": len(data_rows),
        "data_prefixes": data_prefixes,
        "preview_rows": data_rows[:20],
        "comments_preview": comments[:20],
        "semantic_flags": detect_semantic_columns(columns),
    }


def parse_csv_text(text, max_rows=10000):
    f = io.StringIO(text)
    try:
        reader = csv.DictReader(f)
        rows = []
        for idx, row in enumerate(reader):
            rows.append(row)
            if idx + 1 >= max_rows:
                break
        columns = reader.fieldnames or []
        return {
            "columns": columns,
            "column_count": len(columns),
            "parsed_preview_row_count": len(rows),
            "preview_rows": rows[:20],
            "semantic_flags": detect_semantic_columns(columns),
        }
    except Exception as exc:
        return {
            "columns": [],
            "column_count": 0,
            "parsed_preview_row_count": 0,
            "preview_rows": [],
            "semantic_flags": detect_semantic_columns([]),
            "parse_error": repr(exc),
        }


def parse_cov_text(text):
    # Keep this as shape inference only. Do not materialize the covariance matrix.
    tokens = text.split()
    first_int = None
    if tokens:
        try:
            first_int = int(float(tokens[0]))
        except Exception:
            first_int = None

    token_count = len(tokens)
    numeric_first_50 = sum(1 for t in tokens[:50] if token_is_number(t))
    expected_square = first_int * first_int if first_int is not None else None
    has_expected_square_payload = expected_square is not None and (token_count - 1 == expected_square)

    return {
        "token_count": token_count,
        "first_token_as_int": first_int,
        "expected_square_elements": expected_square,
        "payload_element_count_after_first": token_count - 1 if token_count else 0,
        "has_expected_square_payload": has_expected_square_payload,
        "numeric_first_50_count": numeric_first_50,
        "semantic_flags": {
            "has_covariance_like": True,
            "has_table2_direct_keys": False,
            "has_mu_like": False,
            "has_residual_like": False,
            "has_redshift_like": False,
        },
    }


def inspect_target(target, data):
    label = target["label"]
    repo_path = target["repo_path"]
    lower = repo_path.lower()

    text, decompressed = decode_text(data, repo_path)

    if lower.endswith(".cov"):
        kind = "covariance_matrix_text"
        parsed = parse_cov_text(text)
    elif lower.endswith(".csv"):
        kind = "csv_table"
        parsed = parse_csv_text(text)
    else:
        kind = "snana_or_whitespace_table"
        parsed = parse_snana_or_table_text(text)

    semantic = parsed.get("semantic_flags", {})
    columns = parsed.get("columns", [])

    direct_replay_possible = bool(semantic.get("has_table2_direct_keys"))
    downstream_outcome_possible = bool(semantic.get("has_mu_like") and semantic.get("has_redshift_like"))
    simulation_only = "6_simulations" in lower or "sim_" in lower or "simulation" in target.get("target_role", "")
    covariance_only = kind == "covariance_matrix_text"
    metadata_only = target.get("target_role", "").endswith("metadata_candidate") or "header_overrides" in lower
    adjacent_measurement = "optical_wes" in lower or "cepheid_measurement" in target.get("target_role", "")

    if simulation_only:
        recommended_use = "schema_control_only_not_external_validation"
        next_test_allowed = False
        bridge_status = "simulation_only"
    elif covariance_only:
        recommended_use = "covariance_support_for_downstream_likelihood_if_dimension_matches"
        next_test_allowed = False
        bridge_status = "covariance_only"
    elif downstream_outcome_possible:
        recommended_use = "downstream_outcome_schema_found_bridge_required_before_replay"
        next_test_allowed = True
        bridge_status = "bridge_required_not_direct_table2_replay"
    elif adjacent_measurement:
        recommended_use = "adjacent_measurement_schema_only_possible_overlap_lane"
        next_test_allowed = False
        bridge_status = "adjacent_measurement_not_independent_by_default"
    elif metadata_only:
        recommended_use = "metadata_support_only"
        next_test_allowed = False
        bridge_status = "metadata_only"
    elif direct_replay_possible:
        recommended_use = "direct_table2_like_schema_candidate_requires_careful_duplicate_audit"
        next_test_allowed = False
        bridge_status = "possible_direct_but_duplicate_audit_required"
    else:
        recommended_use = "schema_not_sufficient_for_replay"
        next_test_allowed = False
        bridge_status = "insufficient_schema"

    return {
        "label": label,
        "repo_path": repo_path,
        "target_role": target["target_role"],
        "priority": target["priority"],
        "file_kind": kind,
        "decompressed_gzip": decompressed,
        "columns_detected": "|".join(columns[:80]),
        "column_count": parsed.get("column_count"),
        "line_count": parsed.get("line_count"),
        "parsed_preview_row_count": parsed.get("parsed_preview_row_count"),
        "cov_first_token_as_int": parsed.get("first_token_as_int"),
        "cov_expected_square_elements": parsed.get("expected_square_elements"),
        "cov_payload_element_count_after_first": parsed.get("payload_element_count_after_first"),
        "cov_has_expected_square_payload": parsed.get("has_expected_square_payload"),
        "has_id_like": semantic.get("has_id_like"),
        "has_redshift_like": semantic.get("has_redshift_like"),
        "has_mu_like": semantic.get("has_mu_like"),
        "has_residual_like": semantic.get("has_residual_like"),
        "has_calibrator_like": semantic.get("has_calibrator_like"),
        "has_host_like": semantic.get("has_host_like"),
        "has_table2_direct_keys": semantic.get("has_table2_direct_keys"),
        "direct_replay_possible": direct_replay_possible,
        "downstream_outcome_possible": downstream_outcome_possible,
        "simulation_only": simulation_only,
        "covariance_only": covariance_only,
        "metadata_only": metadata_only,
        "adjacent_measurement": adjacent_measurement,
        "bridge_status": bridge_status,
        "recommended_use": recommended_use,
        "next_test_allowed_by_schema": next_test_allowed,
        "truth_boundary": "schema_only_no_replay_no_validation",
        "parsed_detail": parsed,
    }


def choose_next_lane(inspections):
    distance = next((x for x in inspections if x["label"] == "pantheon_shoes_distance_table"), None)
    statsys = next((x for x in inspections if x["label"] == "pantheon_shoes_stat_sys_cov"), None)
    statonly = next((x for x in inspections if x["label"] == "pantheon_shoes_statonly_cov"), None)
    sim = next((x for x in inspections if x["label"] == "simulation_fitres_schema_control"), None)

    downstream_ok = bool(distance and distance["downstream_outcome_possible"])
    cov_ok = bool(
        statsys
        and statsys["covariance_only"]
        and statsys.get("cov_has_expected_square_payload") is True
    )
    statonly_ok = bool(
        statonly
        and statonly["covariance_only"]
        and statonly.get("cov_has_expected_square_payload") is True
    )
    no_direct_table2_keys = bool(distance and not distance["has_table2_direct_keys"])
    sim_not_validation = bool(sim and sim["simulation_only"])

    gates = [
        {
            "gate": "G1_downstream_distance_table_schema_found",
            "passed": downstream_ok,
            "evidence": {
                "distance_columns": distance["columns_detected"] if distance else None,
                "has_mu_like": distance["has_mu_like"] if distance else None,
                "has_redshift_like": distance["has_redshift_like"] if distance else None,
                "parsed_preview_row_count": distance["parsed_preview_row_count"] if distance else None,
            },
        },
        {
            "gate": "G2_stat_sys_covariance_shape_valid",
            "passed": cov_ok,
            "evidence": {
                "dimension": statsys.get("cov_first_token_as_int") if statsys else None,
                "has_expected_square_payload": statsys.get("cov_has_expected_square_payload") if statsys else None,
            },
        },
        {
            "gate": "G3_statonly_covariance_shape_valid",
            "passed": statonly_ok,
            "evidence": {
                "dimension": statonly.get("cov_first_token_as_int") if statonly else None,
                "has_expected_square_payload": statonly.get("cov_has_expected_square_payload") if statonly else None,
            },
        },
        {
            "gate": "G4_downstream_table_is_not_direct_table2_replay_surface",
            "passed": no_direct_table2_keys,
            "evidence": {
                "has_table2_direct_keys": distance["has_table2_direct_keys"] if distance else None,
                "bridge_required": True,
            },
        },
        {
            "gate": "G5_simulation_fitres_classified_as_schema_control_only",
            "passed": sim_not_validation,
            "evidence": {
                "simulation_only": sim["simulation_only"] if sim else None,
                "recommended_use": sim["recommended_use"] if sim else None,
            },
        },
    ]

    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed:
        final_status = "downstream_distance_schema_found_bridge_required_before_replay"
        readiness = 9
        next_wall = (
            "A legitimate downstream distance/covariance schema exists, but it is not a direct Table2 Cepheid replay surface. "
            "v2.5 should build a calibrator/host bridge and influence audit before any frozen-rule replay downstream."
        )
    elif downstream_ok and no_direct_table2_keys:
        final_status = "downstream_distance_schema_found_covariance_or_controls_need_review"
        readiness = 8
        next_wall = (
            "The downstream distance table is usable, but covariance or control classification needs review before v2.5."
        )
    elif downstream_ok:
        final_status = "downstream_schema_found_with_direct_key_caution"
        readiness = 7
        next_wall = (
            "The downstream table parsed, but direct-key semantics need review before any bridge."
        )
    else:
        final_status = "candidate_schema_scout_not_ready_for_next_replay"
        readiness = 5
        next_wall = (
            "No legitimate downstream outcome schema was confirmed. Do not replay."
        )

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "gates": gates,
        "failed_gates": failed,
        "truth_boundary": CLAIMS_V2_4["truth_boundary"],
    }


def holographic_surface_ledger(decision, inspections):
    return {
        "observable_surface": {
            "name": "Selected candidate schemas after v2.3 provenance inventory",
            "repo": REPO,
            "targets": TARGETS,
        },
        "hidden_depth_sought": {
            "name": "Whether an external/downstream residual or outcome lane exists",
            "allowed_claim": "v2.4 can identify downstream outcome/covariance schemas and whether a bridge is required.",
            "not_allowed_claim": "v2.4 cannot validate TAIRID, replay the frozen rule, or claim H0 correction.",
        },
        "boundary_that_forms_surface": {
            "locked_lane_boundary": "v1.9-v2.2 Table2 residual edge lane remains frozen",
            "schema_boundary": "v2.4 inspects file columns and covariance shapes only",
            "bridge_boundary": "downstream distance rows are not Cepheid Table2 rows unless a host/calibrator bridge is proven",
            "simulation_boundary": "simulation FITRES is schema control only, not external validation",
        },
        "what_can_be_reconstructed_now": [
            "Which candidate files parse",
            "Which files contain MU/redshift/downstream outcome columns",
            "Which files are covariance-only",
            "Which files are metadata-only",
            "Which files are simulation-only",
            "Whether v2.5 should be a bridge audit or replay",
        ],
        "what_cannot_be_reconstructed_now": [
            "A downstream F160W replay",
            "A causal H0 correction",
            "New physics",
            "A claim that SH0ES is wrong",
            "A proof of TAIRID",
        ],
        "surface_noise_definition": [
            "Treating simulation FITRES as real external validation",
            "Treating distance table rows as direct Cepheid rows",
            "Skipping the calibrator/host bridge",
            "Changing the frozen F160W rule",
        ],
        "inspection_count": len(inspections),
        "decision": decision,
    }


def write_handoff(decision, inspections):
    usable = [x for x in inspections if x.get("next_test_allowed_by_schema")]
    lines = []
    lines.append("# TAIRID v2.4 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v2.4 did")
    lines.append("")
    lines.append("- Inspected downstream Pantheon+SH0ES distance/covariance schemas.")
    lines.append("- Inspected selected redshift and host metadata files.")
    lines.append("- Inspected selected SH0ES optical Cepheid measurement files.")
    lines.append("- Inspected one simulation FITRES file as schema-control only.")
    lines.append("- Classified candidate surfaces as downstream outcome, covariance, metadata, adjacent measurement, simulation-only, or unusable.")
    lines.append("- Did not run a replay or tune the frozen rule.")
    lines.append("")
    lines.append("## Schema surfaces allowed forward")
    lines.append("")
    if usable:
        for item in usable:
            lines.append(f"- `{item['repo_path']}` — {item['recommended_use']}")
    else:
        lines.append("- No schema was allowed forward.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v2.4 is schema scouting only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not prove SH0ES is wrong.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append(
            "v2.5 should build a calibrator/host bridge and downstream influence audit. It should not directly replay the F160W edge against Pantheon+SH0ES distances until the bridge is proven."
        )
    else:
        lines.append(
            "v2.5 should repair schema parsing before any bridge or replay."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID SH0ES Candidate Schema Scout v2.4 starting.")
    print("Boundary: schema scouting only; no replay, no tuning, no H0 claim.")

    write_json(OUTDIR / "claims_v2_4.json", CLAIMS_V2_4)
    write_json(OUTDIR / "frozen_rule_carry_forward_v2_4.json", FROZEN_RULE_CARRY_FORWARD)

    try:
        download_ledger = []
        inspections = []

        for target in TARGETS:
            fetched = fetch_bytes_for_path(target["repo_path"], prefer_media=True)
            download_ledger.append(
                {
                    "label": target["label"],
                    "repo_path": target["repo_path"],
                    "target_role": target["target_role"],
                    "priority": target["priority"],
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
                inspections.append(
                    {
                        "label": target["label"],
                        "repo_path": target["repo_path"],
                        "target_role": target["target_role"],
                        "priority": target["priority"],
                        "status": "download_failed",
                        "recommended_use": "not_usable_download_failed",
                        "next_test_allowed_by_schema": False,
                        "truth_boundary": "schema_only_no_replay_no_validation",
                    }
                )
                continue

            local_path = DOWNLOAD_DIR / safe_name(target["repo_path"])
            local_path.write_bytes(fetched["data"])

            inspection = inspect_target(target, fetched["data"])
            inspection["status"] = "downloaded"
            inspection["bytes"] = fetched["bytes"]
            inspection["payload_kind"] = fetched["payload_kind"]
            inspection["local_path"] = str(local_path)
            inspections.append(inspection)

            detail_path = OUTDIR / f"{target['label']}_schema_detail_v2_4.json"
            write_json(detail_path, inspection)

            preview_rows = inspection.get("parsed_detail", {}).get("preview_rows", [])
            if preview_rows and isinstance(preview_rows, list):
                write_csv(OUTDIR / f"{target['label']}_preview_rows_v2_4.csv", preview_rows)

        write_csv(OUTDIR / "download_ledger_v2_4.csv", download_ledger)

        flat_inspections = []
        for item in inspections:
            flat = dict(item)
            flat.pop("parsed_detail", None)
            flat_inspections.append(flat)
        write_csv(OUTDIR / "candidate_schema_inventory_v2_4.csv", flat_inspections)

        decision = choose_next_lane(inspections)
        write_json(OUTDIR / "decision_v2_4.json", decision)

        ledger = holographic_surface_ledger(decision, inspections)
        write_json(OUTDIR / "holographic_surface_ledger_v2_4.json", ledger)

        handoff = write_handoff(decision, inspections)
        (OUTDIR / "next_thread_handoff_after_v2_4.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v2_4.txt").write_text(handoff, encoding="utf-8")

        status_counts = {}
        recommended_counts = {}
        bridge_counts = {}
        for item in inspections:
            status_counts[item.get("status", "unknown")] = status_counts.get(item.get("status", "unknown"), 0) + 1
            recommended_counts[item.get("recommended_use", "unknown")] = recommended_counts.get(item.get("recommended_use", "unknown"), 0) + 1
            bridge_counts[item.get("bridge_status", "unknown")] = bridge_counts.get(item.get("bridge_status", "unknown"), 0) + 1

        summary = {
            "test_name": "TAIRID SH0ES Candidate Schema Scout v2.4",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Schema scouting only. No tuning, no replay, no H0 claim, no new-physics claim.",
            "repo": REPO,
            "target_count": len(TARGETS),
            "status_counts": status_counts,
            "recommended_use_counts": recommended_counts,
            "bridge_status_counts": bridge_counts,
            "download_ledger": download_ledger,
            "candidate_schema_inventory": flat_inspections,
            "decision": decision,
            "claims_v2_4": CLAIMS_V2_4,
            "frozen_rule_carry_forward": FROZEN_RULE_CARRY_FORWARD,
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "shoes_candidate_schema_scout_v2_4_summary.json"),
                "summary_txt": str(OUTDIR / "shoes_candidate_schema_scout_v2_4_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v2_4.csv"),
                "schema_inventory_csv": str(OUTDIR / "candidate_schema_inventory_v2_4.csv"),
                "decision_json": str(OUTDIR / "decision_v2_4.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v2_4.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v2_4.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v2_4.txt"),
            },
            "interpretation": {
                "what_success_means": "A downstream outcome/covariance schema was identified, with bridge requirements stated before replay.",
                "what_success_does_not_mean": "This does not validate the frozen residual replay or prove any physics claim.",
                "next_required_step": "v2.5 should build a calibrator/host bridge and downstream influence audit.",
                "truth_boundary": CLAIMS_V2_4["truth_boundary"],
            },
        }
        write_json(OUTDIR / "shoes_candidate_schema_scout_v2_4_summary.json", summary)

        with open(OUTDIR / "shoes_candidate_schema_scout_v2_4_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID SH0ES Candidate Schema Scout v2.4\n\n")
            f.write("Boundary: schema scouting only. No tuning. No replay. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Recommended use counts:\n")
            f.write(json.dumps(recommended_counts, indent=2, default=json_default) + "\n\n")
            f.write("Bridge status counts:\n")
            f.write(json.dumps(bridge_counts, indent=2, default=json_default) + "\n\n")
            f.write("Candidate schema inventory:\n")
            f.write(json.dumps(flat_inspections, indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not prove SH0ES is wrong.\n")
            f.write("- This does not tune or replace the frozen rule.\n")

        print("TAIRID SH0ES Candidate Schema Scout v2.4 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID SH0ES Candidate Schema Scout v2.4",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "shoes_candidate_schema_scout_v2_4_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V2_4["truth_boundary"],
        }
        write_json(OUTDIR / "shoes_candidate_schema_scout_v2_4_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
