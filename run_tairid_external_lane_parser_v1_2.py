#!/usr/bin/env python3
"""
TAIRID External Lane Parser v1.2
SH0ES NIR Cepheid schema audit and frozen-rule replay feasibility.

Purpose:
v1.0 locked the SH0ES Table2 frozen F160W model.
v1.1 searched for an adjacent/external replay lane and pointed us toward the
SH0ES NIR Cepheid files:

    SH0ES_Data/R22_orig19_NIR.out
    SH0ES_Data/R22_orig19_NIR.wm31.out

This v1.2 test does NOT validate the frozen model.
It does NOT tune regimes.
It does NOT invent residuals.

It only asks:
    1. Can the NIR Cepheid files be parsed cleanly?
    2. Do they contain host-field + H/F160W-like surface structure?
    3. Can within-host high/low H-band edge surfaces be built?
    4. Which frozen-regime hosts are present or missing?
    5. Are residual/covariance fields present, or must validation wait?

Boundary:
This does not prove TAIRID.
This does not prove H0 correction.
This does not prove new physics.
This is a schema / surface feasibility audit only.
"""

import csv
import json
import math
import re
import traceback
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path("tairid_external_lane_parser_v1_2_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO = "PantheonPlusSH0ES/DataRelease"
BRANCH_CANDIDATES = ["main", "master"]

TARGET_FILES = [
    {
        "label": "nir_orig19",
        "repo_path": "SH0ES_Data/R22_orig19_NIR.out",
        "meaning": "R22 original 19 host NIR Cepheid table, plus listed anchor content from release notes.",
    },
    {
        "label": "nir_orig19_wm31",
        "repo_path": "SH0ES_Data/R22_orig19_NIR.wm31.out",
        "meaning": "R22 NIR Cepheid table variant with M31 included/expanded.",
    },
]

FROZEN_RULE = {
    "source_status": "locked by v1.0",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "frozen_variable": "F160W-like Table2 numeric column, table2_num_7",
    "external_proxy_candidate": "H column in R22 NIR files, treated only as H/F160W-like surface candidate",
    "edge_rule": "within-host high 5% H/F160W-like magnitude minus within-host low 5% H/F160W-like magnitude",
    "low_alpha_reference_hosts": ["LMC", "SMC", "N4536"],
    "sign_break_quarantine_hosts": ["M31"],
    "clean_high_alpha_rule": "all active hosts except LMC, SMC, N4536, and M31",
    "hard_boundary": [
        "Do not tune host regimes in v1.2.",
        "Do not invent residuals.",
        "Do not claim validation from schema feasibility.",
        "Do not add SMC or any other host if absent from the parsed files.",
        "Do not transfer the Table2 result into NIR files unless a later test defines a valid outcome/residual layer.",
    ],
}

CLAIMS_V1_2 = {
    "battery_name": "TAIRID External Lane Parser v1.2",
    "scope": "SH0ES NIR Cepheid schema audit and frozen-rule replay feasibility",
    "primary_question": (
        "Can R22_orig19_NIR.out and R22_orig19_NIR.wm31.out support a frozen H/F160W within-host edge surface replay, "
        "and what is missing before validation?"
    ),
    "holographic_surface_question": (
        "Does the observable NIR Cepheid table provide a readable surface that can carry host-field structure, "
        "without pretending that missing residual/covariance depth is visible?"
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only checks parser/schema/surface feasibility."
    ),
}

MIN_HOST_ROWS_FOR_EDGE = 20
EDGE_PERCENTILE = 5.0
EXPECTED_COLUMNS = [
    "host",
    "ra",
    "dec",
    "id",
    "period",
    "v_i",
    "sigma_v_i",
    "h_mag",
    "sigma_h",
    "metal_minus_8_69",
    "hst_flag",
]


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
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
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:160]


def raw_url(branch, repo_path):
    return f"https://raw.githubusercontent.com/{REPO}/{branch}/{repo_path}"


def fetch_text_for_path(repo_path):
    errors = []

    for branch in BRANCH_CANDIDATES:
        url = raw_url(branch, repo_path)

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "TAIRID-v1.2-external-lane-parser"},
            )
            with urllib.request.urlopen(req, timeout=45) as response:
                text = response.read().decode("utf-8", errors="replace")
            return {
                "status": "downloaded",
                "branch": branch,
                "url": url,
                "text": text,
                "bytes": len(text.encode("utf-8")),
                "errors": errors,
            }
        except Exception as exc:
            errors.append({"branch": branch, "url": url, "error": repr(exc)})

    return {
        "status": "failed",
        "branch": None,
        "url": None,
        "text": "",
        "bytes": 0,
        "errors": errors,
    }


def to_float(value):
    try:
        out = float(value)
    except Exception:
        return None

    if not math.isfinite(out):
        return None

    return out


def parse_nir_text(label, repo_path, text):
    rows = []
    errors = []
    header_line = None
    separator_count = 0
    skipped_count = 0

    for line_number, line in enumerate(text.splitlines(), start=1):
        raw = line.rstrip("\n")
        stripped = raw.strip()

        if not stripped:
            skipped_count += 1
            continue

        if stripped.startswith("-"):
            separator_count += 1
            continue

        tokens = stripped.split()

        if tokens and tokens[0].lower() == "host":
            header_line = stripped
            continue

        if len(tokens) < 11:
            skipped_count += 1
            continue

        host = tokens[0]
        ra = to_float(tokens[1])
        dec = to_float(tokens[2])
        period = to_float(tokens[4])
        v_i = to_float(tokens[5])
        sigma_v_i = to_float(tokens[6])
        h_mag = to_float(tokens[7])
        sigma_h = to_float(tokens[8])
        metal = to_float(tokens[9])

        if None in [ra, dec, period, v_i, sigma_v_i, h_mag, sigma_h, metal]:
            errors.append(
                {
                    "line_number": line_number,
                    "reason": "numeric_parse_failed",
                    "line": raw[:240],
                }
            )
            continue

        rows.append(
            {
                "dataset_label": label,
                "repo_path": repo_path,
                "source_line_number": line_number,
                "host": host,
                "ra": ra,
                "dec": dec,
                "id": tokens[3],
                "period": period,
                "log_period": math.log10(period) if period > 0 else None,
                "v_i": v_i,
                "sigma_v_i": sigma_v_i,
                "h_mag": h_mag,
                "sigma_h": sigma_h,
                "metal_minus_8_69": metal,
                "hst_flag": tokens[10],
                "raw_line": raw,
            }
        )

    return {
        "label": label,
        "repo_path": repo_path,
        "header_line": header_line,
        "separator_count": separator_count,
        "skipped_count": skipped_count,
        "row_count": len(rows),
        "parse_error_count": len(errors),
        "rows": rows,
        "errors": errors,
    }


def summarize_numeric(values):
    arr = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)

    if len(arr) == 0:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "std": None,
        }

    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def host_summary(rows):
    by_host = defaultdict(list)

    for row in rows:
        by_host[row["host"]].append(row)

    out = []

    for host, host_rows in by_host.items():
        out.append(
            {
                "host": host,
                "row_count": len(host_rows),
                "period_min": summarize_numeric([r["period"] for r in host_rows])["min"],
                "period_median": summarize_numeric([r["period"] for r in host_rows])["median"],
                "period_max": summarize_numeric([r["period"] for r in host_rows])["max"],
                "h_min": summarize_numeric([r["h_mag"] for r in host_rows])["min"],
                "h_median": summarize_numeric([r["h_mag"] for r in host_rows])["median"],
                "h_max": summarize_numeric([r["h_mag"] for r in host_rows])["max"],
                "sigma_h_median": summarize_numeric([r["sigma_h"] for r in host_rows])["median"],
                "v_i_median": summarize_numeric([r["v_i"] for r in host_rows])["median"],
                "metal_median": summarize_numeric([r["metal_minus_8_69"] for r in host_rows])["median"],
                "has_enough_rows_for_edge": len(host_rows) >= MIN_HOST_ROWS_FOR_EDGE,
                "frozen_regime_role": frozen_role_for_host(host),
            }
        )

    return sorted(out, key=lambda r: (-r["row_count"], r["host"]))


def frozen_role_for_host(host):
    if host in set(FROZEN_RULE["low_alpha_reference_hosts"]):
        return "low_alpha_reference"
    if host in set(FROZEN_RULE["sign_break_quarantine_hosts"]):
        return "sign_break_quarantine"
    return "clean_high_alpha_candidate"


def build_edge_surface(rows):
    by_host = defaultdict(list)

    for row in rows:
        by_host[row["host"]].append(row)

    edge_rows = []
    edge_inventory = []

    for host, host_rows in sorted(by_host.items()):
        host_rows = sorted(host_rows, key=lambda r: r["h_mag"])
        n = len(host_rows)

        if n < MIN_HOST_ROWS_FOR_EDGE:
            edge_inventory.append(
                {
                    "host": host,
                    "row_count": n,
                    "edge_status": "not_enough_rows",
                    "edge_count_each_side": 0,
                    "frozen_regime_role": frozen_role_for_host(host),
                }
            )
            continue

        k = max(1, int(math.floor(n * EDGE_PERCENTILE / 100.0)))
        low_side = host_rows[:k]
        high_side = host_rows[-k:]

        for side_name, side_value, selected in [
            ("low_H_bright_edge", -1, low_side),
            ("high_H_faint_edge", 1, high_side),
        ]:
            for r in selected:
                out = dict(r)
                out["edge_side"] = side_name
                out["edge_value_high_minus_low"] = side_value
                out["edge_percentile"] = EDGE_PERCENTILE
                out["host_row_count"] = n
                out["edge_count_each_side"] = k
                out["frozen_regime_role"] = frozen_role_for_host(host)
                edge_rows.append(out)

        edge_inventory.append(
            {
                "host": host,
                "row_count": n,
                "edge_status": "edge_surface_built",
                "edge_count_each_side": k,
                "low_H_max_in_low_edge": max(r["h_mag"] for r in low_side),
                "high_H_min_in_high_edge": min(r["h_mag"] for r in high_side),
                "low_edge_mean_H": float(np.mean([r["h_mag"] for r in low_side])),
                "high_edge_mean_H": float(np.mean([r["h_mag"] for r in high_side])),
                "high_minus_low_mean_H": float(
                    np.mean([r["h_mag"] for r in high_side])
                    - np.mean([r["h_mag"] for r in low_side])
                ),
                "frozen_regime_role": frozen_role_for_host(host),
            }
        )

    return {
        "edge_rows": sorted(edge_rows, key=lambda r: (r["host"], r["edge_side"], r["h_mag"])),
        "edge_inventory": sorted(edge_inventory, key=lambda r: (-r["row_count"], r["host"])),
    }


def regime_presence(rows, edge_inventory):
    counts = Counter(row["host"] for row in rows)
    edge_hosts = {
        row["host"]
        for row in edge_inventory
        if row.get("edge_status") == "edge_surface_built"
    }

    required_hosts = (
        FROZEN_RULE["low_alpha_reference_hosts"]
        + FROZEN_RULE["sign_break_quarantine_hosts"]
    )

    host_rows = []

    for host in required_hosts:
        host_rows.append(
            {
                "host": host,
                "frozen_role": frozen_role_for_host(host),
                "present_in_rows": host in counts,
                "row_count": counts.get(host, 0),
                "edge_surface_built": host in edge_hosts,
            }
        )

    clean_hosts = sorted(
        host for host in counts
        if frozen_role_for_host(host) == "clean_high_alpha_candidate"
    )

    return {
        "required_frozen_host_rows": host_rows,
        "clean_candidate_host_count": len(clean_hosts),
        "clean_candidate_hosts": clean_hosts,
        "low_alpha_hosts_present": [
            h for h in FROZEN_RULE["low_alpha_reference_hosts"]
            if counts.get(h, 0) > 0
        ],
        "low_alpha_hosts_missing": [
            h for h in FROZEN_RULE["low_alpha_reference_hosts"]
            if counts.get(h, 0) <= 0
        ],
        "sign_break_hosts_present": [
            h for h in FROZEN_RULE["sign_break_quarantine_hosts"]
            if counts.get(h, 0) > 0
        ],
        "sign_break_hosts_missing": [
            h for h in FROZEN_RULE["sign_break_quarantine_hosts"]
            if counts.get(h, 0) <= 0
        ],
    }


def compare_datasets(parsed_a, parsed_b):
    rows_a = parsed_a["rows"]
    rows_b = parsed_b["rows"]

    key_a = {
        (r["host"], r["id"], round(float(r["period"]), 6), round(float(r["h_mag"]), 6))
        for r in rows_a
    }
    key_b = {
        (r["host"], r["id"], round(float(r["period"]), 6), round(float(r["h_mag"]), 6))
        for r in rows_b
    }

    counts_a = Counter(r["host"] for r in rows_a)
    counts_b = Counter(r["host"] for r in rows_b)

    all_hosts = sorted(set(counts_a) | set(counts_b))

    host_deltas = []

    for host in all_hosts:
        host_deltas.append(
            {
                "host": host,
                "count_in_a": counts_a.get(host, 0),
                "count_in_b": counts_b.get(host, 0),
                "b_minus_a": counts_b.get(host, 0) - counts_a.get(host, 0),
            }
        )

    return {
        "dataset_a": parsed_a["label"],
        "dataset_b": parsed_b["label"],
        "row_count_a": len(rows_a),
        "row_count_b": len(rows_b),
        "b_minus_a_row_count": len(rows_b) - len(rows_a),
        "unique_key_count_a": len(key_a),
        "unique_key_count_b": len(key_b),
        "keys_added_in_b": len(key_b - key_a),
        "keys_removed_in_b": len(key_a - key_b),
        "hosts_added_in_b": sorted(set(counts_b) - set(counts_a)),
        "hosts_removed_in_b": sorted(set(counts_a) - set(counts_b)),
        "host_count_deltas": sorted(host_deltas, key=lambda r: (-abs(r["b_minus_a"]), r["host"])),
    }


def holographic_surface_ledger(dataset_label, rows, edge_result, regime):
    edge_hosts = [
        row["host"] for row in edge_result["edge_inventory"]
        if row.get("edge_status") == "edge_surface_built"
    ]

    return {
        "dataset_label": dataset_label,
        "observable_surface": {
            "surface_name": "R22 NIR Cepheid H-band row table",
            "available_columns": EXPECTED_COLUMNS,
            "row_count": len(rows),
            "host_count": len(set(row["host"] for row in rows)),
            "edge_surface_host_count": len(edge_hosts),
            "edge_rule_applied": "within-host high 5% H minus within-host low 5% H",
        },
        "hidden_depth_claimed": {
            "allowed_claim": (
                "The H-band surface may carry host-field and measurement-boundary information, "
                "but v1.2 does not claim it contains the Table2 residual structure."
            ),
            "not_allowed_claim": (
                "Do not claim the NIR table validates the locked Table2 residual model unless a later test supplies "
                "a valid residual/outcome layer."
            ),
        },
        "boundary_that_forms_surface": {
            "local_boundary": "host field",
            "measurement_boundary": "NIR H/F160W-like Cepheid measurement",
            "edge_boundary": "within-host high/low H magnitude tails",
        },
        "encoded_memory_candidate": [
            "host-specific H magnitude distribution",
            "period and color context",
            "uncertainty context",
            "metallicity context",
            "anchor/sign-break host presence or absence",
        ],
        "lost_or_missing_interior": [
            "No Table2 compact residual vector is directly present in these .out files.",
            "No covariance matrix is directly present in these .out files.",
            "No direct held-out residual pressure metric is present in these .out files.",
            "SMC may be absent in this NIR lane, so the frozen low-alpha regime is only partially replayable if absent.",
        ],
        "reconstructive_power_allowed_now": [
            "Parser validation",
            "schema mapping",
            "host/regime presence audit",
            "within-host H-edge surface construction",
            "feasibility decision for a later replay",
        ],
        "reconstructive_power_not_allowed_now": [
            "predictive validation",
            "H0 correction",
            "new physics claim",
            "claim that the frozen Table2 model transfers",
        ],
        "surface_noise_check": {
            "would_fail_if": [
                "H column cannot be parsed",
                "host fields are missing",
                "too few hosts have enough rows for within-host edges",
                "frozen regime hosts are mostly absent",
                "no later residual/outcome layer can be identified",
            ]
        },
        "regime_presence": regime,
    }


def feasibility_gates(label, parsed, edge_result, regime):
    rows = parsed["rows"]
    host_count = len(set(row["host"] for row in rows))
    active_edge_hosts = [
        row for row in edge_result["edge_inventory"]
        if row.get("edge_status") == "edge_surface_built"
    ]

    low_present = set(regime["low_alpha_hosts_present"])
    sign_present = set(regime["sign_break_hosts_present"])

    gates = [
        {
            "gate": "G1_parse_rows",
            "passed": len(rows) > 0 and parsed["parse_error_count"] == 0,
            "detail": "NIR file must parse into rows without numeric parse errors.",
            "evidence": {
                "row_count": len(rows),
                "parse_error_count": parsed["parse_error_count"],
            },
        },
        {
            "gate": "G2_schema_columns",
            "passed": all(key in rows[0] for key in EXPECTED_COLUMNS) if rows else False,
            "detail": "Expected host/period/color/H/sigma/metallicity schema must be present after parsing.",
            "evidence": {
                "expected_columns": EXPECTED_COLUMNS,
                "first_row_keys": list(rows[0].keys()) if rows else [],
            },
        },
        {
            "gate": "G3_host_field_available",
            "passed": host_count >= 10,
            "detail": "Host field must contain enough distinct hosts for field-relative audit.",
            "evidence": {"host_count": host_count},
        },
        {
            "gate": "G4_h_edge_surface_available",
            "passed": len(active_edge_hosts) >= 10,
            "detail": "At least 10 hosts should support within-host high/low H edge surfaces.",
            "evidence": {"active_edge_host_count": len(active_edge_hosts)},
        },
        {
            "gate": "G5_frozen_regime_partial_presence",
            "passed": "LMC" in low_present and "N4536" in low_present and "M31" in sign_present,
            "detail": "LMC, N4536, and M31 should be present for partial frozen-regime replay; SMC is separately reported if missing.",
            "evidence": {
                "low_alpha_hosts_present": sorted(low_present),
                "low_alpha_hosts_missing": regime["low_alpha_hosts_missing"],
                "sign_break_hosts_present": sorted(sign_present),
                "sign_break_hosts_missing": regime["sign_break_hosts_missing"],
            },
        },
        {
            "gate": "G6_residual_layer_absent_stop_before_validation",
            "passed": True,
            "detail": "The NIR .out schema does not include the Table2 residual/covariance layer, so v1.2 must stop before validation.",
            "evidence": {
                "validation_allowed_now": False,
                "reason": "schema/surface only; no residual/covariance outcome identified in this file",
            },
        },
        {
            "gate": "G7_holographic_surface_ledger_required",
            "passed": True,
            "detail": "v1.2 records surface/depth/lost-interior limits before any future replay.",
            "evidence": {"ledger_written": True},
        },
    ]

    failed = [g for g in gates if not g["passed"]]

    if not failed:
        final_status = "nir_surface_schema_feasible_residual_validation_not_yet_possible"
        readiness = 8
        next_wall = (
            "NIR H-band host-field surface is parser-ready, but validation needs a legitimate residual/outcome layer. "
            "Do not invent residuals."
        )
    elif len(failed) <= 2 and all(g["gate"] != "G1_parse_rows" for g in failed):
        final_status = "nir_surface_schema_partially_feasible_with_cautions"
        readiness = 7
        next_wall = "Parser works, but frozen-rule replay is incomplete or host/regime coverage is partial."
    else:
        final_status = "nir_surface_schema_not_ready_for_frozen_replay"
        readiness = 5
        next_wall = "NIR schema did not pass enough gates for frozen-rule replay feasibility."

    return {
        "dataset_label": label,
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "gates": gates,
        "failed_gates": [g["gate"] for g in failed],
    }


def make_plots(all_host_summaries, all_edge_inventories):
    try:
        if all_host_summaries:
            top = sorted(all_host_summaries, key=lambda r: (-r["row_count"], r["dataset_label"], r["host"]))[:30]
            labels = [f"{r['dataset_label']}:{r['host']}" for r in top]
            values = [r["row_count"] for r in top]
            x = np.arange(len(top))

            plt.figure(figsize=(14, 5))
            plt.bar(x, values)
            plt.xticks(x, labels, rotation=70, ha="right", fontsize=8)
            plt.ylabel("row count")
            plt.title("v1.2 NIR host row counts")
            plt.tight_layout()
            plt.savefig(OUTDIR / "nir_host_row_counts_v1_2.png", dpi=160)
            plt.close()

        if all_edge_inventories:
            active = [
                r for r in all_edge_inventories
                if r.get("edge_status") == "edge_surface_built"
            ]
            top = sorted(active, key=lambda r: (-r["row_count"], r["dataset_label"], r["host"]))[:30]
            labels = [f"{r['dataset_label']}:{r['host']}" for r in top]
            values = [r.get("high_minus_low_mean_H", 0.0) for r in top]
            x = np.arange(len(top))

            plt.figure(figsize=(14, 5))
            plt.bar(x, values)
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, labels, rotation=70, ha="right", fontsize=8)
            plt.ylabel("high-edge mean H minus low-edge mean H")
            plt.title("v1.2 within-host H-edge surface separation")
            plt.tight_layout()
            plt.savefig(OUTDIR / "nir_h_edge_surface_separation_v1_2.png", dpi=160)
            plt.close()

    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v1_2.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )


def write_handoff(final_status, readiness, next_wall, dataset_decisions):
    lines = []
    lines.append("# TAIRID v1.2 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{final_status}`")
    lines.append(f"- Readiness score: `{readiness}/10`")
    lines.append(f"- Next wall: {next_wall}")
    lines.append("")
    lines.append("## Frozen rule carried forward")
    lines.append("")
    lines.append("- F160W/H-like surface only; no variable search.")
    lines.append("- Within-host high 5% H edge minus within-host low 5% H edge.")
    lines.append("- Low-alpha/reference hosts: LMC, SMC, N4536.")
    lines.append("- M31 sign-break quarantine.")
    lines.append("")
    lines.append("## v1.2 result")
    lines.append("")
    for decision in dataset_decisions:
        lines.append(f"### {decision['dataset_label']}")
        lines.append(f"- Status: `{decision['final_status']}`")
        lines.append(f"- Readiness: `{decision['readiness_score_0_to_10']}/10`")
        lines.append(f"- Next wall: {decision['next_wall']}")
        lines.append(f"- Failed gates: {', '.join(decision['failed_gates']) if decision['failed_gates'] else 'none'}")
        lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v1.2 is schema and surface feasibility only.")
    lines.append("- It does not validate the frozen Table2 model.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not invent residuals.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    lines.append(
        "v1.3 should identify or build a legitimate residual/outcome layer for the NIR Cepheid files. "
        "If no such layer exists, v1.3 should stop and classify the NIR lane as surface-only context, not validation."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID External Lane Parser v1.2 starting.")
    print("Boundary: schema/surface feasibility only; no validation and no tuning.")

    write_json(OUTDIR / "claims_v1_2.json", CLAIMS_V1_2)
    write_json(OUTDIR / "frozen_rule_v1_2.json", FROZEN_RULE)

    try:
        download_rows = []
        parsed_results = []
        all_rows = []
        all_host_rows = []
        all_edge_rows = []
        all_edge_inventory = []
        all_ledgers = []
        dataset_decisions = []

        for target in TARGET_FILES:
            label = target["label"]
            repo_path = target["repo_path"]

            fetched = fetch_text_for_path(repo_path)
            local_path = DOWNLOAD_DIR / safe_name(repo_path)

            if fetched["status"] == "downloaded":
                local_path.write_text(fetched["text"], encoding="utf-8")

            download_rows.append(
                {
                    "label": label,
                    "repo_path": repo_path,
                    "status": fetched["status"],
                    "branch": fetched["branch"],
                    "url": fetched["url"],
                    "bytes": fetched["bytes"],
                    "local_path": str(local_path) if fetched["status"] == "downloaded" else "",
                    "errors": json.dumps(fetched["errors"], default=json_default),
                }
            )

            if fetched["status"] != "downloaded":
                parsed_results.append(
                    {
                        "label": label,
                        "repo_path": repo_path,
                        "row_count": 0,
                        "parse_error_count": 0,
                        "download_failed": True,
                    }
                )
                dataset_decisions.append(
                    {
                        "dataset_label": label,
                        "final_status": "download_failed",
                        "readiness_score_0_to_10": 4,
                        "next_wall": "Could not download target NIR file.",
                        "gates": [],
                        "failed_gates": ["download_failed"],
                    }
                )
                continue

            parsed = parse_nir_text(label, repo_path, fetched["text"])
            parsed_results.append(
                {
                    "label": label,
                    "repo_path": repo_path,
                    "header_line": parsed["header_line"],
                    "row_count": parsed["row_count"],
                    "parse_error_count": parsed["parse_error_count"],
                    "separator_count": parsed["separator_count"],
                    "skipped_count": parsed["skipped_count"],
                }
            )

            rows = parsed["rows"]
            for r in rows:
                all_rows.append(r)

            write_csv(OUTDIR / f"{label}_parsed_rows_v1_2.csv", rows)
            write_json(OUTDIR / f"{label}_parse_errors_v1_2.json", parsed["errors"])

            h_summary = host_summary(rows)
            for r in h_summary:
                r["dataset_label"] = label
                all_host_rows.append(r)
            write_csv(OUTDIR / f"{label}_host_summary_v1_2.csv", h_summary)

            edge_result = build_edge_surface(rows)
            for r in edge_result["edge_rows"]:
                r["dataset_label"] = label
                all_edge_rows.append(r)
            for r in edge_result["edge_inventory"]:
                r["dataset_label"] = label
                all_edge_inventory.append(r)

            write_csv(OUTDIR / f"{label}_h_edge_rows_v1_2.csv", edge_result["edge_rows"])
            write_csv(OUTDIR / f"{label}_h_edge_inventory_v1_2.csv", edge_result["edge_inventory"])

            regime = regime_presence(rows, edge_result["edge_inventory"])
            write_json(OUTDIR / f"{label}_regime_presence_v1_2.json", regime)

            ledger = holographic_surface_ledger(label, rows, edge_result, regime)
            all_ledgers.append(ledger)
            write_json(OUTDIR / f"{label}_holographic_surface_ledger_v1_2.json", ledger)

            decision = feasibility_gates(label, parsed, edge_result, regime)
            dataset_decisions.append(decision)
            write_json(OUTDIR / f"{label}_feasibility_gates_v1_2.json", decision)

        write_csv(OUTDIR / "download_ledger_v1_2.csv", download_rows)
        write_csv(OUTDIR / "all_parsed_nir_rows_v1_2.csv", all_rows)
        write_csv(OUTDIR / "all_host_summary_v1_2.csv", all_host_rows)
        write_csv(OUTDIR / "all_h_edge_rows_v1_2.csv", all_edge_rows)
        write_csv(OUTDIR / "all_h_edge_inventory_v1_2.csv", all_edge_inventory)
        write_json(OUTDIR / "all_holographic_surface_ledgers_v1_2.json", all_ledgers)

        comparison = None
        if len(parsed_results) >= 2:
            parsed_full = []
            for target in TARGET_FILES:
                label = target["label"]
                path = OUTDIR / f"{label}_parsed_rows_v1_2.csv"
                # Reuse in-memory rows by filtering all_rows.
                parsed_full.append(
                    {
                        "label": label,
                        "rows": [r for r in all_rows if r["dataset_label"] == label],
                    }
                )
            comparison = compare_datasets(parsed_full[0], parsed_full[1])
            write_json(OUTDIR / "nir_orig19_vs_wm31_comparison_v1_2.json", comparison)

        ready_scores = [d["readiness_score_0_to_10"] for d in dataset_decisions]
        max_ready = max(ready_scores) if ready_scores else 4

        if any(d["final_status"] == "nir_surface_schema_feasible_residual_validation_not_yet_possible" for d in dataset_decisions):
            final_status = "nir_schema_surface_ready_but_residual_validation_missing"
            readiness = 8
            next_wall = (
                "Parser and H-edge surface are ready. The next barrier is a legitimate residual/outcome layer; "
                "do not invent one."
            )
        elif max_ready >= 7:
            final_status = "nir_schema_partially_ready_with_cautions"
            readiness = 7
            next_wall = "Parser mostly works, but host/regime coverage or edge feasibility has cautions."
        else:
            final_status = "nir_schema_not_ready"
            readiness = max_ready
            next_wall = "NIR files did not pass enough schema gates."

        make_plots(all_host_rows, all_edge_inventory)

        handoff = write_handoff(final_status, readiness, next_wall, dataset_decisions)
        (OUTDIR / "next_thread_handoff_after_v1_2.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v1_2.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID External Lane Parser v1.2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Schema and H/F160W-like surface feasibility only. "
                "No validation, no tuning, no H0 claim, no new-physics claim."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness,
            "next_wall": next_wall,
            "claims_v1_2": CLAIMS_V1_2,
            "frozen_rule": FROZEN_RULE,
            "target_files": TARGET_FILES,
            "download_ledger": download_rows,
            "parse_summaries": parsed_results,
            "dataset_decisions": dataset_decisions,
            "nir_orig19_vs_wm31_comparison": comparison,
            "holographic_surface_ledgers": all_ledgers,
            "output_files": {
                "summary_json": str(OUTDIR / "external_lane_parser_v1_2_summary.json"),
                "summary_txt": str(OUTDIR / "external_lane_parser_v1_2_summary.txt"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v1_2.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v1_2.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v1_2.csv"),
                "all_rows_csv": str(OUTDIR / "all_parsed_nir_rows_v1_2.csv"),
                "all_host_summary_csv": str(OUTDIR / "all_host_summary_v1_2.csv"),
                "all_edge_inventory_csv": str(OUTDIR / "all_h_edge_inventory_v1_2.csv"),
                "all_edge_rows_csv": str(OUTDIR / "all_h_edge_rows_v1_2.csv"),
                "holographic_ledgers_json": str(OUTDIR / "all_holographic_surface_ledgers_v1_2.json"),
                "comparison_json": str(OUTDIR / "nir_orig19_vs_wm31_comparison_v1_2.json"),
                "plots": [
                    str(OUTDIR / "nir_host_row_counts_v1_2.png"),
                    str(OUTDIR / "nir_h_edge_surface_separation_v1_2.png"),
                ],
            },
            "interpretation": {
                "what_success_means": (
                    "The adjacent NIR Cepheid table can be parsed and can support a frozen H-edge surface construction."
                ),
                "what_success_does_not_mean": (
                    "This does not validate the Table2 frozen model because residual/covariance depth is not present in the NIR .out files."
                ),
                "next_required_step": (
                    "v1.3 must locate or define a legitimate residual/outcome layer before any predictive replay."
                ),
                "truth_boundary": CLAIMS_V1_2["truth_boundary"],
            },
        }

        write_json(OUTDIR / "external_lane_parser_v1_2_summary.json", summary)

        with open(OUTDIR / "external_lane_parser_v1_2_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID External Lane Parser v1.2\n\n")
            f.write("Boundary: schema/surface feasibility only. No validation. No tuning.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Dataset decisions:\n")
            f.write(json.dumps(dataset_decisions, indent=2, default=json_default) + "\n\n")
            f.write("NIR orig19 vs wm31 comparison:\n")
            f.write(json.dumps(comparison, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This only checks parser/schema/surface feasibility.\n")
            f.write("- Do not invent residuals.\n")

        print("TAIRID External Lane Parser v1.2 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID External Lane Parser v1.2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "external_lane_parser_v1_2_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V1_2["truth_boundary"],
        }
        write_json(OUTDIR / "external_lane_parser_v1_2_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
