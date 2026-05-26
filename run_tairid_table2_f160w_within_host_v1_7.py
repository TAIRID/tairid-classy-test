#!/usr/bin/env python3
"""
TAIRID Table2 F160W within-host faint-tail audit v1.7.

Purpose:
v1.6 showed that the global F160W faint-tail residual pressure is distributed
across multiple host/field labels rather than collapsing to one host.

This test separates global faintness from field-relative / within-host faintness.
It asks whether Cepheids that are faint relative to their own host also carry
residual pressure after full controls.

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only audits whether the faint-tail residual pressure is global or within-host-relative.
"""

import csv
import json
import math
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_tairid_table2_f160w_faint_tail_host_field_v1_6 as v16


OUTDIR = Path("tairid_table2_f160w_within_host_v1_7_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

F160W_COLUMN = "table2_num_7"
FULL_DESIGN = "plus_host_top10_row_order_measurement_controls"
ORIGINAL_DESIGN = "original_47"
PERCENTILES = [5, 10, 15, 20, 25]
MIN_HOST_ROWS = 20
PERMUTATION_REPEATS = 80
SEED = 42


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


def safe_name(value):
    return v16.safe_name(value)


def clean_entity(row, key):
    return v16.get_entity(row, key)


def numeric(row, key):
    return v16.get_numeric(row, key)


def finite_host_groups(mapped_rows):
    groups = defaultdict(list)

    for row in mapped_rows:
        value = numeric(row, F160W_COLUMN)

        if not np.isfinite(value):
            continue

        host = clean_entity(row, "host_guess")
        groups[host].append((float(value), row))

    cleaned = {}

    for host, items in groups.items():
        items = sorted(items, key=lambda item: item[0])

        if len(items) >= MIN_HOST_ROWS:
            cleaned[host] = items

    return cleaned


def group_rows_by_key(rows, key):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row.get(key, "UNKNOWN")].append(row)

    return grouped


def build_global_faint_mask(mapped_rows, y_length):
    mask, tail_rows, meta = v16.build_faint_tail_mask(mapped_rows, y_length)

    return {
        "name": "global_f160w_faint_95_100",
        "values": mask,
        "kind": "global_faint_tail_reference",
        "side": "global_faint",
        "percentile": 5,
        "selected_by_host": Counter(row["host_guess_clean"] for row in tail_rows),
        "selected_indices_by_host": {
            host: [int(row["compact_observation_index"]) for row in rows]
            for host, rows in group_rows_by_key(tail_rows, "host_guess_clean").items()
        },
        "selected_rows": tail_rows,
        "metadata": meta,
    }


def build_within_host_candidate(mapped_rows, y_length, host_groups, side, percentile):
    mask = np.zeros(y_length, dtype=float)
    selected_rows = []
    selected_by_host = Counter()
    selected_indices_by_host = defaultdict(list)
    host_meta = []

    for host, items in host_groups.items():
        n = len(items)
        k = max(1, int(math.ceil(n * percentile / 100.0)))

        if side == "within_host_faint":
            chosen = items[-k:]
            rank_side = "faintest"
        elif side == "within_host_bright":
            chosen = items[:k]
            rank_side = "brightest"
        else:
            raise ValueError(side)

        for value, row in chosen:
            idx = int(row["compact_observation_index"])
            mask[idx] = 1.0
            selected_by_host[host] += 1
            selected_indices_by_host[host].append(idx)

            enriched = dict(row)
            enriched["host_guess_clean"] = host
            enriched["f160w_value"] = float(value)
            enriched["within_host_side"] = side
            enriched["within_host_percentile"] = percentile
            enriched["host_row_count_used_for_rank"] = n
            enriched["host_selected_count_for_rank"] = k
            selected_rows.append(enriched)

        values = np.asarray([value for value, _ in items], dtype=float)
        chosen_values = np.asarray([value for value, _ in chosen], dtype=float)

        host_meta.append(
            {
                "host": host,
                "host_row_count": n,
                "selected_count": k,
                "side": side,
                "percentile": percentile,
                "host_f160w_min": float(np.min(values)),
                "host_f160w_max": float(np.max(values)),
                "selected_f160w_min": float(np.min(chosen_values)),
                "selected_f160w_max": float(np.max(chosen_values)),
            }
        )

    selected_rows = sorted(
        selected_rows,
        key=lambda row: (
            row["host_guess_clean"],
            row["f160w_value"],
            row["compact_observation_index"],
        ),
    )

    name = f"{side}_{percentile:02d}"

    return {
        "name": name,
        "values": mask,
        "kind": "within_host_f160w_rank_mask",
        "side": side,
        "percentile": percentile,
        "selected_by_host": dict(selected_by_host),
        "selected_indices_by_host": dict(selected_indices_by_host),
        "selected_rows": selected_rows,
        "host_meta": host_meta,
        "metadata": {
            "side": side,
            "percentile": percentile,
            "host_count_used": len(host_groups),
            "min_host_rows": MIN_HOST_ROWS,
            "selected_total_rows": int(np.sum(mask > 0)),
            "plain_meaning": (
                f"Within each host with at least {MIN_HOST_ROWS} Cepheids, "
                f"select the {rank_side} {percentile}% by F160W-like magnitude."
            ),
        },
    }


def build_candidates(mapped_rows, y_length):
    host_groups = finite_host_groups(mapped_rows)
    candidates = [build_global_faint_mask(mapped_rows, y_length)]

    for percentile in PERCENTILES:
        candidates.append(
            build_within_host_candidate(
                mapped_rows,
                y_length,
                host_groups,
                "within_host_faint",
                percentile,
            )
        )

        candidates.append(
            build_within_host_candidate(
                mapped_rows,
                y_length,
                host_groups,
                "within_host_bright",
                percentile,
            )
        )

    inventory = []

    for candidate in candidates:
        inventory.append(
            {
                "name": candidate["name"],
                "kind": candidate["kind"],
                "side": candidate["side"],
                "percentile": candidate["percentile"],
                "selected_total_rows": int(np.sum(candidate["values"] > 0)),
                "host_count_with_selected_rows": len(candidate["selected_by_host"]),
                "selected_by_host_json": json.dumps(candidate["selected_by_host"], sort_keys=True),
                **candidate["metadata"],
            }
        )

    write_json(OUTDIR / "within_host_candidate_inventory_v1_7.json", inventory)

    all_selected_rows = []

    for candidate in candidates:
        for row in candidate["selected_rows"]:
            out = dict(row)
            out["candidate"] = candidate["name"]
            all_selected_rows.append(out)

    write_csv(OUTDIR / "within_host_selected_rows_v1_7.csv", all_selected_rows)

    return candidates, host_groups


def audit_candidate(candidate, y, c_factor, fit, design_name):
    return v16.audit_vector(
        name=candidate["name"],
        values=candidate["values"],
        kind=candidate["kind"],
        y=y,
        c_factor=c_factor,
        fit=fit,
        design_name=design_name,
        metadata={
            "side": candidate["side"],
            "percentile": candidate["percentile"],
            "selected_total_rows": int(np.sum(candidate["values"] > 0)),
            "host_count_with_selected_rows": len(candidate["selected_by_host"]),
        },
    )


def run_candidate_audits(candidates, y, c_factor, design_fits):
    rows = []

    for design_name in [ORIGINAL_DESIGN, FULL_DESIGN]:
        fit = design_fits[design_name]

        for candidate in candidates:
            rows.append(audit_candidate(candidate, y, c_factor, fit, design_name))

    rows = sorted(
        rows,
        key=lambda row: (
            row["design"],
            row.get("side", ""),
            row.get("percentile") if row.get("percentile") is not None else 999,
        ),
    )

    write_csv(OUTDIR / "within_host_candidate_audit_v1_7.csv", rows)

    return rows


def host_preserving_permutation(candidate, host_groups, y, c_factor, fit, design_name):
    rng = np.random.default_rng(SEED)
    selected_counts = {
        host: len(indices)
        for host, indices in candidate["selected_indices_by_host"].items()
    }
    rows = []

    for repeat in range(PERMUTATION_REPEATS):
        mask = np.zeros(len(y), dtype=float)

        for host, k in selected_counts.items():
            items = host_groups.get(host, [])

            if not items or k <= 0:
                continue

            k_eff = min(k, len(items))
            chosen_positions = rng.choice(len(items), size=k_eff, replace=False)

            for pos in chosen_positions:
                _, row = items[int(pos)]
                mask[int(row["compact_observation_index"])] = 1.0

        rows.append(
            v16.audit_vector(
                name=f"{candidate['name']}_host_preserving_permutation_{repeat}",
                values=mask,
                kind="host_preserving_permutation_control",
                y=y,
                c_factor=c_factor,
                fit=fit,
                design_name=design_name,
                metadata={
                    "source_candidate": candidate["name"],
                    "side": candidate["side"],
                    "percentile": candidate["percentile"],
                },
            )
        )

    observed = audit_candidate(candidate, y, c_factor, fit, design_name)
    deltas = np.asarray([row["delta_chi2_score"] for row in rows], dtype=float)
    ratios = np.asarray([row["nondegenerate_ratio"] for row in rows], dtype=float)

    summary = {
        "design": design_name,
        "candidate": candidate["name"],
        "side": candidate["side"],
        "percentile": candidate["percentile"],
        "repeats": PERMUTATION_REPEATS,
        "observed_delta_chi2": observed["delta_chi2_score"],
        "observed_nondegenerate_ratio": observed["nondegenerate_ratio"],
        "permutation_delta_mean": float(np.mean(deltas)),
        "permutation_delta_95": float(np.percentile(deltas, 95)),
        "permutation_delta_99": float(np.percentile(deltas, 99)),
        "permutation_ratio_mean": float(np.mean(ratios)),
        "permutation_ratio_95": float(np.percentile(ratios, 95)),
        "observed_exceeds_95_percent_permutation_delta": bool(
            observed["delta_chi2_score"] > float(np.percentile(deltas, 95))
        ),
        "observed_exceeds_99_percent_permutation_delta": bool(
            observed["delta_chi2_score"] > float(np.percentile(deltas, 99))
        ),
    }

    return rows, summary


def run_permutation_controls(candidates, host_groups, y, c_factor, design_fits):
    all_rows = []
    summaries = []
    fit = design_fits[FULL_DESIGN]

    for candidate in candidates:
        if candidate["side"] == "global_faint":
            continue

        rows, summary = host_preserving_permutation(
            candidate,
            host_groups,
            y,
            c_factor,
            fit,
            FULL_DESIGN,
        )

        all_rows.extend(rows)
        summaries.append(summary)

    write_csv(OUTDIR / "within_host_host_preserving_permutation_details_v1_7.csv", all_rows)
    write_json(OUTDIR / "within_host_host_preserving_permutation_summaries_v1_7.json", summaries)

    return all_rows, summaries


def host_component_audit(candidate, y, c_factor, fit, design_name):
    component_rows = []
    leaveout_rows = []
    full = audit_candidate(candidate, y, c_factor, fit, design_name)
    full_delta = full["delta_chi2_score"]

    for host, indices in candidate["selected_indices_by_host"].items():
        component = np.zeros(len(y), dtype=float)
        component[np.asarray(indices, dtype=int)] = 1.0

        component_rows.append(
            v16.audit_vector(
                name=f"{candidate['name']}_host_component_{host}",
                values=component,
                kind="within_host_component",
                y=y,
                c_factor=c_factor,
                fit=fit,
                design_name=design_name,
                metadata={
                    "source_candidate": candidate["name"],
                    "host": host,
                    "selected_count_in_host": len(indices),
                },
            )
        )

        reduced = np.asarray(candidate["values"], dtype=float).copy()
        reduced[np.asarray(indices, dtype=int)] = 0.0

        leave = v16.audit_vector(
            name=f"{candidate['name']}_without_host_{host}",
            values=reduced,
            kind="within_host_leave_one_host_out",
            y=y,
            c_factor=c_factor,
            fit=fit,
            design_name=design_name,
            metadata={
                "source_candidate": candidate["name"],
                "host": host,
                "removed_count_in_host": len(indices),
            },
        )

        drop = full_delta - leave["delta_chi2_score"]
        leave["delta_drop_from_full_candidate"] = float(drop)
        leave["delta_drop_fraction_from_full_candidate"] = float(drop / full_delta) if full_delta else None
        leaveout_rows.append(leave)

    component_rows = sorted(
        component_rows,
        key=lambda row: (
            -row.get("delta_chi2_score", 0.0),
            -row.get("selected_count_in_host", 0),
            row.get("host", ""),
        ),
    )

    leaveout_rows = sorted(
        leaveout_rows,
        key=lambda row: -row.get("delta_drop_fraction_from_full_candidate", 0.0),
    )

    return full, component_rows, leaveout_rows


def summarize_profiles(audit_rows, permutation_summaries):
    perm_key = {
        (row["candidate"], row["design"]): row
        for row in permutation_summaries
    }

    profile_rows = []

    for row in audit_rows:
        if row["design"] != FULL_DESIGN:
            continue

        perm = perm_key.get((row["candidate"], row["design"]), {})
        out = dict(row)
        out["permutation_delta_95"] = perm.get("permutation_delta_95")
        out["permutation_delta_99"] = perm.get("permutation_delta_99")
        out["observed_exceeds_95_percent_permutation_delta"] = perm.get(
            "observed_exceeds_95_percent_permutation_delta"
        )
        out["observed_exceeds_99_percent_permutation_delta"] = perm.get(
            "observed_exceeds_99_percent_permutation_delta"
        )
        profile_rows.append(out)

    profile_rows = sorted(
        profile_rows,
        key=lambda row: (
            row.get("side", ""),
            row.get("percentile") if row.get("percentile") is not None else 999,
        ),
    )

    write_csv(OUTDIR / "within_host_profile_summary_v1_7.csv", profile_rows)

    by_side = {}

    for side in sorted(set(row.get("side", "") for row in profile_rows)):
        rows = [row for row in profile_rows if row.get("side") == side]
        best = max(rows, key=lambda row: row.get("delta_chi2_score", 0.0)) if rows else None

        by_side[side] = {
            "best": best,
            "rows": rows,
            "max_delta": best.get("delta_chi2_score") if best else None,
            "sum_delta": float(np.sum([row.get("delta_chi2_score", 0.0) for row in rows])) if rows else None,
        }

    write_json(OUTDIR / "within_host_profile_by_side_v1_7.json", by_side)

    return profile_rows, by_side


def decide_status(by_side, global_audit, best_component_info):
    faint = by_side.get("within_host_faint", {})
    bright = by_side.get("within_host_bright", {})
    best_faint = faint.get("best")
    best_bright = bright.get("best")

    best_cases = {
        "global_faint_reference": global_audit,
        "best_within_host_faint": best_faint,
        "best_within_host_bright": best_bright,
        "best_component_info": best_component_info,
    }

    def strong(row):
        return bool(
            row
            and row.get("delta_chi2_score", 0.0) >= 25.0
            and row.get("observed_exceeds_99_percent_permutation_delta") is True
            and row.get("p_value_chi2_one_dof", 1.0) <= 0.01
        )

    def directional(row):
        return bool(
            row
            and row.get("delta_chi2_score", 0.0) >= 10.0
            and row.get("observed_exceeds_95_percent_permutation_delta") is True
            and row.get("p_value_chi2_one_dof", 1.0) <= 0.05
        )

    faint_delta = best_faint.get("delta_chi2_score", 0.0) if best_faint else 0.0
    bright_delta = best_bright.get("delta_chi2_score", 0.0) if best_bright else 0.0
    global_delta = global_audit.get("delta_chi2_score", 0.0)

    if strong(best_faint) and faint_delta > bright_delta + 20.0:
        return (
            "within_host_faint_tail_boundary_survives_controls",
            8,
            "Within-host faint-tail ranks survive full controls and host-preserving permutation checks.",
            best_cases,
        )

    if directional(best_faint) and faint_delta > bright_delta + 5.0:
        return (
            "within_host_faint_tail_directional_not_locked",
            7,
            "Within-host faint-tail ranks remain directional, but not strong enough to lock a field-relative boundary model.",
            best_cases,
        )

    if global_delta >= 25.0 and faint_delta < 10.0:
        return (
            "global_faint_tail_survives_but_within_host_faintness_collapses",
            7,
            "The signal is mainly global faintness, not within-host relative faintness.",
            best_cases,
        )

    if best_faint and best_bright and abs(faint_delta - bright_delta) < 5.0:
        return (
            "within_host_rank_signal_not_faint_specific",
            6,
            "Within-host bright and faint ranks are too similar to call a faint-side boundary.",
            best_cases,
        )

    return (
        "no_locked_within_host_faint_tail_structure",
        6,
        "Within-host faintness does not produce a locked residual structure after controls.",
        best_cases,
    )


def make_plots(profile_rows, component_rows, leaveout_rows):
    if profile_rows:
        for side in ["within_host_faint", "within_host_bright"]:
            rows = sorted(
                [row for row in profile_rows if row.get("side") == side],
                key=lambda row: row.get("percentile") or 0,
            )

            if not rows:
                continue

            plt.figure(figsize=(8, 5))
            plt.plot(
                [row["percentile"] for row in rows],
                [row["delta_chi2_score"] for row in rows],
                marker="o",
            )
            plt.xlabel("within-host percentile selected")
            plt.ylabel("delta chi2 score")
            plt.title(f"{side} residual pressure after full controls")
            plt.tight_layout()
            plt.savefig(OUTDIR / f"{side}_profile_v1_7.png", dpi=160)
            plt.close()

        faint_rows = sorted(
            [row for row in profile_rows if row.get("side") == "within_host_faint"],
            key=lambda row: row.get("percentile") or 0,
        )
        bright_rows = sorted(
            [row for row in profile_rows if row.get("side") == "within_host_bright"],
            key=lambda row: row.get("percentile") or 0,
        )

        if faint_rows and bright_rows:
            plt.figure(figsize=(8, 5))
            plt.plot(
                [row["percentile"] for row in faint_rows],
                [row["delta_chi2_score"] for row in faint_rows],
                marker="o",
                label="within-host faint",
            )
            plt.plot(
                [row["percentile"] for row in bright_rows],
                [row["delta_chi2_score"] for row in bright_rows],
                marker="o",
                label="within-host bright",
            )
            plt.xlabel("within-host percentile selected")
            plt.ylabel("delta chi2 score")
            plt.title("Within-host faint vs bright rank pressure")
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUTDIR / "within_host_faint_vs_bright_v1_7.png", dpi=160)
            plt.close()

    if component_rows:
        rows = component_rows[:25]
        labels = [row.get("host", "UNKNOWN") for row in rows]
        values = [row.get("delta_chi2_score", 0.0) for row in rows]

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(rows)), values)
        plt.xticks(np.arange(len(rows)), labels, rotation=55, ha="right", fontsize=8)
        plt.ylabel("delta chi2 score")
        plt.title("Top within-host faint components by host")
        plt.tight_layout()
        plt.savefig(OUTDIR / "within_host_top_host_components_v1_7.png", dpi=160)
        plt.close()

    if leaveout_rows:
        rows = leaveout_rows[:25]
        labels = [row.get("host", "UNKNOWN") for row in rows]
        values = [row.get("delta_drop_fraction_from_full_candidate", 0.0) for row in rows]

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(rows)), values)
        plt.xticks(np.arange(len(rows)), labels, rotation=55, ha="right", fontsize=8)
        plt.ylabel("fraction of candidate delta removed")
        plt.title("Leave-one-host-out drop for best within-host faint candidate")
        plt.tight_layout()
        plt.savefig(OUTDIR / "within_host_leave_one_host_drop_v1_7.png", dpi=160)
        plt.close()


def main():
    print("TAIRID Table2 F160W within-host faint-tail audit v1.7 starting.")
    print("Boundary: within-host rank audit only; not proof.")

    repair_summary = {}

    try:
        v16.OUTDIR = OUTDIR
        v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        ns, repair_summary = v16.load_v15_helpers()
        write_json(OUTDIR / "v15_import_repair_summary_v1_7.json", repair_summary)

        download_repo_path = ns["download_repo_path"]
        code_context_search = ns["code_context_search"]
        extract_first_numeric_fits_array = ns["extract_first_numeric_fits_array"]
        parse_table2_tex = ns["parse_table2_tex"]
        orient_design_matrix = ns["orient_design_matrix"]
        stable_cholesky = ns["stable_cholesky"]
        gls_fit = ns["gls_fit"]
        recover_compact_rows = ns["recover_compact_rows"]
        map_table2_to_spine = ns["map_table2_to_spine"]
        summarize_hosts = ns["summarize_hosts"]
        numeric_feature_summary = ns["numeric_feature_summary"]
        build_designs = ns["build_designs"]
        h0_like = ns["h0_like"]

        downloads = {}
        aux_results = []
        ledger = []

        for label, repo_path in ns["COMPACT_FILES"].items():
            result = download_repo_path(repo_path, label)
            downloads[label] = result
            ledger.append(
                {
                    "label": label,
                    "repo_path": repo_path,
                    "status": result.get("status"),
                    "local_path": result.get("local_path"),
                    "bytes": result.get("bytes"),
                    "sha256": result.get("sha256"),
                    "attempt_count": len(result.get("attempts", [])),
                }
            )

        for repo_path in ns["AUX_FILES"]:
            result = download_repo_path(repo_path, safe_name(repo_path))
            aux_results.append(result)
            ledger.append(
                {
                    "label": safe_name(repo_path),
                    "repo_path": repo_path,
                    "status": result.get("status"),
                    "local_path": result.get("local_path"),
                    "bytes": result.get("bytes"),
                    "sha256": result.get("sha256"),
                    "attempt_count": len(result.get("attempts", [])),
                }
            )

        write_csv(OUTDIR / "download_ledger_v1_7.csv", ledger)
        write_json(OUTDIR / "download_attempts_v1_7.json", {"compact": downloads, "auxiliary": aux_results})

        code_hits = code_context_search(aux_results)

        parsed = {}
        parse_meta = {}
        parse_errors = []

        for label in ["allc", "alll", "ally"]:
            result = downloads.get(label, {})

            if result.get("status") != "downloaded":
                parse_errors.append(
                    {
                        "label": label,
                        "status": "not_downloaded",
                        "download_status": result.get("status"),
                    }
                )
                continue

            try:
                arr, meta = extract_first_numeric_fits_array(Path(result["local_path"]))
                parsed[label] = arr
                parse_meta[label] = meta
            except Exception as exc:
                parse_errors.append(
                    {
                        "label": label,
                        "status": "parse_failed",
                        "error": str(exc),
                    }
                )

        write_json(OUTDIR / "parse_meta_v1_7.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v1_7.json", parse_errors)

        table2_result = next(
            (
                item for item in aux_results
                if item.get("repo_path") == "SH0ES_Data/table2.tex"
                and item.get("status") == "downloaded"
            ),
            None,
        )

        if parse_errors or not all(key in parsed for key in ["allc", "alll", "ally"]) or not table2_result:
            summary = {
                "test_name": "TAIRID Table2 F160W within-host faint-tail audit v1.7",
                "final_status": "table2_f160w_within_host_v1_7_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or table2 retrieval before within-host audit.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "table2_f160w_within_host_v1_7_summary.json", summary)
            print("Parse/download failed. See summary JSON.")
            return

        table2_all_rows, table2_data_rows, table2_header_rows = parse_table2_tex(
            Path(table2_result["local_path"])
        )

        C = np.asarray(parsed["allc"], dtype=float)
        L = np.asarray(parsed["alll"], dtype=float)
        y = np.asarray(parsed["ally"], dtype=float).reshape(-1)

        X, orientation = orient_design_matrix(L, len(y))

        if X is None:
            raise RuntimeError(f"Could not orient L: {orientation}")

        if C.ndim != 2 or C.shape[0] != len(y) or C.shape[1] != len(y):
            raise RuntimeError(f"C shape {C.shape} does not match y length {len(y)}")

        c_factor, C_sym, jitter, chol_attempts = stable_cholesky(C)

        baseline = gls_fit(y, X, c_factor)
        row_rows, cluster_rows = recover_compact_rows(X, y, C_sym, baseline)

        write_csv(OUTDIR / "compact_row_map_v1_7.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v1_7.csv", cluster_rows)

        mapped_rows, map_status = map_table2_to_spine(row_rows, table2_data_rows)
        host_summary = summarize_hosts(mapped_rows)
        numeric_rows, likely_numeric_labels = numeric_feature_summary(mapped_rows)

        designs, control_metadata = build_designs(X, mapped_rows, host_summary, len(y))
        design_fits = {
            name: gls_fit(y, design["D"], c_factor)
            for name, design in designs.items()
        }

        candidates, host_groups = build_candidates(mapped_rows, len(y))
        audit_rows = run_candidate_audits(candidates, y, c_factor, design_fits)
        permutation_rows, permutation_summaries = run_permutation_controls(
            candidates,
            host_groups,
            y,
            c_factor,
            design_fits,
        )
        profile_rows, by_side = summarize_profiles(audit_rows, permutation_summaries)

        global_candidate = next(c for c in candidates if c["side"] == "global_faint")
        global_audit = audit_candidate(
            global_candidate,
            y,
            c_factor,
            design_fits[FULL_DESIGN],
            FULL_DESIGN,
        )

        best_faint_row = by_side.get("within_host_faint", {}).get("best")
        best_faint_candidate = (
            next((c for c in candidates if c["name"] == best_faint_row["candidate"]), None)
            if best_faint_row
            else None
        )

        component_rows = []
        leaveout_rows = []
        best_component_info = {}

        if best_faint_candidate:
            best_full, component_rows, leaveout_rows = host_component_audit(
                best_faint_candidate,
                y,
                c_factor,
                design_fits[FULL_DESIGN],
                FULL_DESIGN,
            )
            write_csv(OUTDIR / "within_host_best_faint_host_components_v1_7.csv", component_rows)
            write_csv(OUTDIR / "within_host_best_faint_leave_one_host_out_v1_7.csv", leaveout_rows)
            best_component_info = {
                "best_full_candidate_audit": best_full,
                "top_components": component_rows[:30],
                "top_leaveout_drops": leaveout_rows[:30],
            }

        final_status, readiness_score, next_wall, best_cases = decide_status(
            by_side,
            global_audit,
            best_component_info,
        )

        make_plots(profile_rows, component_rows, leaveout_rows)

        residual = baseline["residual"]
        abs_residual = np.abs(residual)

        design_fit_rows = []

        for design_name, fit in design_fits.items():
            design_fit_rows.append(
                {
                    "design": design_name,
                    "k": fit["k"],
                    "dof": fit["dof"],
                    "chi2": fit["chi2"],
                    "reduced_chi2": fit["reduced_chi2"],
                    "aic": fit["aic"],
                    "bic": fit["bic"],
                    "delta_chi2_vs_original": baseline["chi2"] - fit["chi2"],
                    "delta_aic_vs_original": fit["aic"] - baseline["aic"],
                    "delta_bic_vs_original": fit["bic"] - baseline["bic"],
                }
            )

        write_csv(OUTDIR / "design_fit_comparison_v1_7.csv", design_fit_rows)

        edge_counts = {
            "rows_total": int(X.shape[0]),
            "spine_38_41_43_rows": int(sum(1 for row in row_rows if row["contains_38_41_43_spine"])),
            "bridge_42_46_rows": int(sum(1 for row in row_rows if row["bridges_param42_param46"])),
            "touch_param42_rows": int(sum(1 for row in row_rows if row["touches_param42"])),
            "touch_param46_rows": int(sum(1 for row in row_rows if row["touches_param46_H0_like"])),
        }

        summary = {
            "test_name": "TAIRID Table2 F160W within-host faint-tail audit v1.7",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Within-host F160W rank audit only. Not proof of TAIRID, not H0 resolution, "
                "not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "repair_summary": repair_summary,
            "matrix_shapes": {
                "allc_C": list(C.shape),
                "alll_L": list(L.shape),
                "ally_y_original": list(np.asarray(parsed["ally"]).shape),
                "y_flat": list(y.shape),
                "X_design": list(X.shape),
                "L_orientation": orientation,
            },
            "covariance": {
                "cholesky_jitter": jitter,
                "cholesky_attempts": chol_attempts,
                "diag_min": float(np.min(np.diag(C_sym))),
                "diag_max": float(np.max(np.diag(C_sym))),
                "diag_nonpositive_count": int(np.sum(np.diag(C_sym) <= 0)),
            },
            "baseline_gls": {
                "chi2": baseline["chi2"],
                "dof": baseline["dof"],
                "reduced_chi2": baseline["reduced_chi2"],
                "aic": baseline["aic"],
                "bic": baseline["bic"],
                "parameter_count": int(X.shape[1]),
                "param38_value": float(baseline["beta"][38]),
                "param41_value": float(baseline["beta"][41]),
                "param43_value": float(baseline["beta"][43]),
                "param42_value": float(baseline["beta"][42]),
                "param46_fivelogH0": float(baseline["beta"][46]),
                "param46_H0_like": h0_like(baseline["beta"]),
                "normal_condition_estimate": float(np.linalg.cond(baseline["normal"])),
                "residual_mean": float(np.mean(residual)),
                "residual_std": float(np.std(residual)),
                "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
                "abs_residual_90": float(np.percentile(abs_residual, 90)),
                "abs_residual_95": float(np.percentile(abs_residual, 95)),
                "abs_residual_99": float(np.percentile(abs_residual, 99)),
            },
            "edge_counts": edge_counts,
            "table2_mapping": map_status,
            "table2_parse_counts": {
                "table2_all_rows": len(table2_all_rows),
                "table2_data_rows": len(table2_data_rows),
                "table2_header_rows": len(table2_header_rows),
                "mapped_rows": len(mapped_rows),
                "unique_host_guesses": len(host_summary),
            },
            "likely_numeric_labels": likely_numeric_labels,
            "design_fit_comparison": design_fit_rows,
            "top_numeric_correlations": numeric_rows[:40],
            "top_hosts_by_residual_pressure": host_summary[:30],
            "host_group_count_used_for_within_host_ranks": len(host_groups),
            "min_host_rows_for_within_host_rank": MIN_HOST_ROWS,
            "global_faint_reference_audit_full_controls": global_audit,
            "within_host_profile_by_side": by_side,
            "within_host_profile_summary": profile_rows,
            "within_host_candidate_audits": audit_rows,
            "host_preserving_permutation_summaries": permutation_summaries,
            "best_component_info": best_component_info,
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "table2_f160w_within_host_v1_7_summary.json"),
                "summary_txt": str(OUTDIR / "table2_f160w_within_host_v1_7_summary.txt"),
                "candidate_inventory_json": str(OUTDIR / "within_host_candidate_inventory_v1_7.json"),
                "candidate_audit_csv": str(OUTDIR / "within_host_candidate_audit_v1_7.csv"),
                "profile_summary_csv": str(OUTDIR / "within_host_profile_summary_v1_7.csv"),
                "profile_by_side_json": str(OUTDIR / "within_host_profile_by_side_v1_7.json"),
                "permutation_summaries_json": str(OUTDIR / "within_host_host_preserving_permutation_summaries_v1_7.json"),
                "selected_rows_csv": str(OUTDIR / "within_host_selected_rows_v1_7.csv"),
                "plots": [
                    str(OUTDIR / "within_host_faint_vs_bright_v1_7.png"),
                    str(OUTDIR / "within_host_top_host_components_v1_7.png"),
                    str(OUTDIR / "within_host_leave_one_host_drop_v1_7.png"),
                ],
            },
            "interpretation": {
                "field_relative_boundary_condition": (
                    "Within-host faint ranks beat host-preserving random controls and remain stronger than within-host bright ranks."
                ),
                "global_only_condition": (
                    "Global faint tail remains strong while within-host faint ranks collapse."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only separates global faintness from host-relative faintness in the SH0ES Table2 residual layer."
                ),
            },
        }

        write_json(OUTDIR / "table2_f160w_within_host_v1_7_summary.json", summary)

        with open(OUTDIR / "table2_f160w_within_host_v1_7_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Table2 F160W within-host faint-tail audit v1.7\n\n")
            f.write("Boundary: within-host F160W rank audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Global faint reference audit, full controls:\n")
            f.write(json.dumps(global_audit, indent=2, default=json_default) + "\n\n")
            f.write("Within-host profile by side:\n")
            f.write(json.dumps(by_side, indent=2, default=json_default) + "\n\n")
            f.write("Host-preserving permutation summaries:\n")
            f.write(json.dumps(permutation_summaries, indent=2, default=json_default) + "\n\n")
            f.write("Best component info:\n")
            f.write(json.dumps(best_component_info, indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only separates global faintness from within-host faintness.\n")

        print("TAIRID Table2 F160W within-host faint-tail audit v1.7 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Table2 F160W within-host faint-tail audit v1.7",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "v1_7_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "table2_f160w_within_host_v1_7_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
