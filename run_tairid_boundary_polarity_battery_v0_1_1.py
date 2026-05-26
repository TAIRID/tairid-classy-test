#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.1.1
SH0ES Table2 F160W signed edge-pair polarity audit.

Purpose:
v0.1 asked the right question but used a classifier that was too blunt.

The old classifier partly judged polarity by whether the faint tail had a larger
standalone chi-square score than the bright tail. That is not the clean TAIRID
boundary-polarity question.

This v0.1.1 classifier asks the correct edge-pair question:

    Does the faint edge and bright edge separate signed residual direction?

For this audit:
    positive group = F160W faint side
    negative group = F160W bright side

A polarity-supporting result should show:
    1. faint-minus-bright contrast survives full controls,
    2. contrast beats same-count random contrast permutations,
    3. signed score beats permutation,
    4. faint mean residual is higher than bright mean residual.

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This is a classifier-correction and replication gate for signed F160W polarity.
"""

import csv
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_tairid_table2_f160w_faint_tail_host_field_v1_6 as v16
import run_tairid_boundary_polarity_battery_v0_1 as b01


OUTDIR = Path("tairid_boundary_polarity_battery_v0_1_1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

FULL_DESIGN = "plus_host_top10_row_order_measurement_controls"
ORIGINAL_DESIGN = "original_47"

PERMUTATION_REPEATS = 120


CLAIMS_V0_1_1 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.1.1",
    "status": "classifier-correction replication gate",
    "reason_for_v0_1_1": (
        "v0.1 found strong contrast evidence, but the decision classifier was "
        "too focused on standalone faint-vs-bright chi-square dominance. "
        "v0.1.1 locks edge-pair polarity as the actual test object."
    ),
    "native_tairid_claim": (
        "A boundary surface should produce directional separation across opposing edges. "
        "For the SH0ES Table2 F160W layer, if the global faint edge is structurally active, "
        "the faint-minus-bright contrast should survive controls and random contrast checks."
    ),
    "edge_pair_rule": (
        "Polarity is judged by the signed contrast vector: faint edge = +1, bright edge = -1. "
        "A pass requires contrast survival, permutation survival, and signed residual separation."
    ),
    "anti_ad_hoc_boundary": (
        "This does not retroactively prove v0.1. It corrects the classifier and establishes "
        "the rule to carry forward into v0.2 and cross-domain batteries."
    ),
    "failure_rule": (
        "If contrast does not beat permutation or signed residual separation is weak, "
        "TAIRID must narrow this lane back to residual localization only."
    ),
}


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


def contrast_direction_stats(name, contrast_values, fit):
    stats = b01.signed_residual_stats(name, contrast_values, fit)

    pos = stats["positive_group"]
    neg = stats["negative_group"]

    mean_diff = stats["signed_mean_residual_difference_pos_minus_neg"]
    cinv_diff = stats["signed_mean_cinv_residual_difference_pos_minus_neg"]

    out = {
        "candidate": name,
        "positive_group_plain": "faint side",
        "negative_group_plain": "bright side",
        "positive_count": pos["count"],
        "negative_count": neg["count"],
        "positive_mean_residual": pos["mean_residual"],
        "negative_mean_residual": neg["mean_residual"],
        "positive_mean_abs_residual": pos["mean_abs_residual"],
        "negative_mean_abs_residual": neg["mean_abs_residual"],
        "signed_mean_residual_difference_faint_minus_bright": mean_diff,
        "signed_mean_cinv_residual_difference_faint_minus_bright": cinv_diff,
        "expected_tairid_direction_met": bool(
            mean_diff is not None and mean_diff > 0.0
        ),
    }

    return out


def fixed_classifier(global_results, within_results):
    global_faint = global_results["global_faint_audit_full"]
    global_bright = global_results["global_bright_audit_full"]
    global_contrast = global_results["global_contrast_audit_full_with_permutation"]
    global_direction = global_results["global_contrast_direction_stats"]

    within_faint = within_results["within_host_faint_audit_full"]
    within_bright = within_results["within_host_bright_audit_full"]
    within_contrast = within_results["within_host_contrast_audit_full_with_permutation"]
    within_direction = within_results["within_host_contrast_direction_stats"]

    def edge_pair_pass(row, direction):
        return bool(
            row.get("delta_chi2_score", 0.0) >= 25.0
            and row.get("p_value_chi2_one_dof", 1.0) <= 0.01
            and row.get("observed_exceeds_99_percent_permutation_delta") is True
            and row.get("observed_abs_score_exceeds_99_percent_permutation") is True
            and direction.get("expected_tairid_direction_met") is True
        )

    def edge_pair_directional(row, direction):
        return bool(
            row.get("delta_chi2_score", 0.0) >= 10.0
            and row.get("p_value_chi2_one_dof", 1.0) <= 0.05
            and row.get("observed_exceeds_95_percent_permutation_delta") is True
            and row.get("observed_abs_score_exceeds_95_percent_permutation") is True
            and direction.get("expected_tairid_direction_met") is True
        )

    global_pass = edge_pair_pass(global_contrast, global_direction)
    within_pass = edge_pair_pass(within_contrast, within_direction)

    global_directional = edge_pair_directional(global_contrast, global_direction)
    within_directional = edge_pair_directional(within_contrast, within_direction)

    global_faint_magnitude_stronger = (
        global_faint.get("delta_chi2_score", 0.0)
        > global_bright.get("delta_chi2_score", 0.0) + 20.0
    )

    within_faint_magnitude_stronger = (
        within_faint.get("delta_chi2_score", 0.0)
        > within_bright.get("delta_chi2_score", 0.0) + 20.0
    )

    best_cases = {
        "global_faint": global_faint,
        "global_bright": global_bright,
        "global_contrast": global_contrast,
        "global_direction": global_direction,
        "within_host_faint": within_faint,
        "within_host_bright": within_bright,
        "within_host_contrast": within_contrast,
        "within_host_direction": within_direction,
        "global_pass": global_pass,
        "within_pass": within_pass,
        "global_directional": global_directional,
        "within_directional": within_directional,
        "global_faint_magnitude_stronger": global_faint_magnitude_stronger,
        "within_faint_magnitude_stronger": within_faint_magnitude_stronger,
    }

    if global_pass and within_pass:
        return (
            "edge_pair_polarity_supported_global_and_within_host",
            8,
            "Both global and within-host F160W edge-pair contrasts survive full controls, permutation checks, and signed direction tests.",
            best_cases,
        )

    if global_pass and not within_pass:
        return (
            "edge_pair_polarity_supported_global_system_scale",
            8,
            "Global F160W edge-pair polarity survives, but within-host polarity does not lock. Boundary is system-scale here.",
            best_cases,
        )

    if within_pass and not global_pass:
        return (
            "edge_pair_polarity_supported_within_host_only",
            7,
            "Within-host edge-pair polarity survives, but the global contrast does not lock.",
            best_cases,
        )

    if global_directional or within_directional:
        return (
            "edge_pair_polarity_directional_not_locked",
            7,
            "At least one edge-pair contrast is directional, but it does not meet the locked v0.1.1 pass gate.",
            best_cases,
        )

    if global_faint_magnitude_stronger:
        return (
            "boundary_magnitude_without_signed_polarity",
            7,
            "The faint edge remains stronger in magnitude, but signed edge-pair polarity does not pass.",
            best_cases,
        )

    return (
        "no_signed_edge_pair_polarity",
        6,
        "The v0.1.1 classifier does not find a locked signed F160W edge-pair polarity.",
        best_cases,
    )


def make_v011_plots(global_rows, within_rows):
    if global_rows:
        labels = [row["candidate"] for row in global_rows]
        deltas = [row.get("delta_chi2_score", 0.0) for row in global_rows]
        scores = [row.get("score", 0.0) for row in global_rows]

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(labels)), deltas)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("delta chi2 score")
        plt.title("Global F160W edge-pair polarity candidates v0.1.1")
        plt.tight_layout()
        plt.savefig(OUTDIR / "global_edge_pair_delta_chi2_v0_1_1.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(labels)), scores)
        plt.axhline(0.0, linewidth=1)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("signed score")
        plt.title("Global F160W edge-pair signed scores v0.1.1")
        plt.tight_layout()
        plt.savefig(OUTDIR / "global_edge_pair_signed_scores_v0_1_1.png", dpi=160)
        plt.close()

    if within_rows:
        labels = [row["candidate"] for row in within_rows]
        deltas = [row.get("delta_chi2_score", 0.0) for row in within_rows]
        scores = [row.get("score", 0.0) for row in within_rows]

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(labels)), deltas)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("delta chi2 score")
        plt.title("Within-host F160W edge-pair polarity candidates v0.1.1")
        plt.tight_layout()
        plt.savefig(OUTDIR / "within_host_edge_pair_delta_chi2_v0_1_1.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(labels)), scores)
        plt.axhline(0.0, linewidth=1)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("signed score")
        plt.title("Within-host F160W edge-pair signed scores v0.1.1")
        plt.tight_layout()
        plt.savefig(OUTDIR / "within_host_edge_pair_signed_scores_v0_1_1.png", dpi=160)
        plt.close()


def main():
    print("TAIRID Boundary Prediction Battery v0.1.1 starting.")
    print("Boundary: classifier-correction signed edge-pair audit only; not proof.")

    write_json(OUTDIR / "claims_v0_1_1.json", CLAIMS_V0_1_1)

    repair_summary = {}

    try:
        v16.OUTDIR = OUTDIR
        v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        b01.OUTDIR = OUTDIR
        b01.DOWNLOAD_DIR = DOWNLOAD_DIR
        b01.PERMUTATION_REPEATS = PERMUTATION_REPEATS

        ns, repair_summary = v16.load_v15_helpers()
        write_json(OUTDIR / "v15_import_repair_summary_v0_1_1.json", repair_summary)

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

        write_csv(OUTDIR / "download_ledger_v0_1_1.csv", ledger)
        write_json(OUTDIR / "download_attempts_v0_1_1.json", {"compact": downloads, "auxiliary": aux_results})

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

        write_json(OUTDIR / "parse_meta_v0_1_1.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v0_1_1.json", parse_errors)

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
                "test_name": "TAIRID Boundary Prediction Battery v0.1.1",
                "final_status": "boundary_polarity_battery_v0_1_1_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or Table2 retrieval before v0.1.1.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "boundary_polarity_battery_v0_1_1_summary.json", summary)
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

        write_csv(OUTDIR / "compact_row_map_v0_1_1.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v0_1_1.csv", cluster_rows)

        mapped_rows, map_status = map_table2_to_spine(row_rows, table2_data_rows)
        host_summary = summarize_hosts(mapped_rows)
        numeric_rows, likely_numeric_labels = numeric_feature_summary(mapped_rows)

        designs, control_metadata = build_designs(X, mapped_rows, host_summary, len(y))
        design_fits = {
            name: gls_fit(y, design["D"], c_factor)
            for name, design in designs.items()
        }

        full_fit = design_fits[FULL_DESIGN]
        original_fit = design_fits[ORIGINAL_DESIGN]

        # Global edge-pair masks.
        bright_mask, bright_rows, bright_meta = b01.build_global_band_mask(mapped_rows, len(y), 0, 5)
        faint_mask, faint_rows, faint_meta = b01.build_global_band_mask(mapped_rows, len(y), 95, 100)
        global_contrast = faint_mask - bright_mask

        write_csv(OUTDIR / "global_bright_00_05_rows_v0_1_1.csv", bright_rows)
        write_csv(OUTDIR / "global_faint_95_100_rows_v0_1_1.csv", faint_rows)

        global_faint_original = b01.audit_vector(
            "global_f160w_faint_95_100",
            faint_mask,
            "global_faint_mask",
            y,
            c_factor,
            original_fit,
            ORIGINAL_DESIGN,
            metadata=faint_meta,
        )
        global_bright_original = b01.audit_vector(
            "global_f160w_bright_00_05",
            bright_mask,
            "global_bright_mask",
            y,
            c_factor,
            original_fit,
            ORIGINAL_DESIGN,
            metadata=bright_meta,
        )
        global_contrast_original = b01.audit_vector(
            "global_f160w_faint95_minus_bright05_contrast",
            global_contrast,
            "global_faint_minus_bright_contrast",
            y,
            c_factor,
            original_fit,
            ORIGINAL_DESIGN,
            metadata={
                "positive_group": "global_f160w_faint_95_100",
                "negative_group": "global_f160w_bright_00_05",
                "positive_count": int(np.sum(global_contrast > 0)),
                "negative_count": int(np.sum(global_contrast < 0)),
            },
        )

        global_faint_full = b01.audit_vector(
            "global_f160w_faint_95_100",
            faint_mask,
            "global_faint_mask",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=faint_meta,
        )
        global_bright_full = b01.audit_vector(
            "global_f160w_bright_00_05",
            bright_mask,
            "global_bright_mask",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=bright_meta,
        )
        global_contrast_full = b01.audit_vector(
            "global_f160w_faint95_minus_bright05_contrast",
            global_contrast,
            "global_faint_minus_bright_contrast",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata={
                "positive_group": "global_f160w_faint_95_100",
                "negative_group": "global_f160w_bright_00_05",
                "positive_count": int(np.sum(global_contrast > 0)),
                "negative_count": int(np.sum(global_contrast < 0)),
            },
        )

        global_perm_rows, global_perm_summary = b01.same_count_contrast_permutation(
            "global_f160w_faint95_minus_bright05_contrast",
            int(np.sum(global_contrast > 0)),
            int(np.sum(global_contrast < 0)),
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            mapped_rows,
        )
        write_csv(OUTDIR / "global_contrast_permutation_details_v0_1_1.csv", global_perm_rows)
        write_json(OUTDIR / "global_contrast_permutation_summary_v0_1_1.json", global_perm_summary)

        global_contrast_full_with_perm = b01.add_permutation_flags(global_contrast_full, global_perm_summary)
        global_direction = contrast_direction_stats(
            "global_f160w_faint95_minus_bright05_contrast",
            global_contrast,
            full_fit,
        )

        # Within-host edge-pair masks.
        within_faint, within_bright, within_faint_rows, within_bright_rows, within_meta = b01.build_within_host_tail_masks(
            mapped_rows,
            len(y),
            percentile=5,
        )
        within_contrast = within_faint - within_bright

        write_csv(OUTDIR / "within_host_faint_rows_v0_1_1.csv", within_faint_rows)
        write_csv(OUTDIR / "within_host_bright_rows_v0_1_1.csv", within_bright_rows)
        write_json(OUTDIR / "within_host_tail_meta_v0_1_1.json", within_meta)

        within_faint_full = b01.audit_vector(
            "within_host_f160w_faint_05",
            within_faint,
            "within_host_faint_mask",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=within_meta,
        )
        within_bright_full = b01.audit_vector(
            "within_host_f160w_bright_05",
            within_bright,
            "within_host_bright_mask",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=within_meta,
        )
        within_contrast_full = b01.audit_vector(
            "within_host_f160w_faint05_minus_bright05_contrast",
            within_contrast,
            "within_host_faint_minus_bright_contrast",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata={
                "positive_group": "within_host_faint_05",
                "negative_group": "within_host_bright_05",
                "positive_count": int(np.sum(within_contrast > 0)),
                "negative_count": int(np.sum(within_contrast < 0)),
                **within_meta,
            },
        )

        within_perm_rows, within_perm_summary = b01.same_count_contrast_permutation(
            "within_host_f160w_faint05_minus_bright05_contrast",
            int(np.sum(within_contrast > 0)),
            int(np.sum(within_contrast < 0)),
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            mapped_rows,
        )
        write_csv(OUTDIR / "within_host_contrast_permutation_details_v0_1_1.csv", within_perm_rows)
        write_json(OUTDIR / "within_host_contrast_permutation_summary_v0_1_1.json", within_perm_summary)

        within_contrast_full_with_perm = b01.add_permutation_flags(within_contrast_full, within_perm_summary)
        within_direction = contrast_direction_stats(
            "within_host_f160w_faint05_minus_bright05_contrast",
            within_contrast,
            full_fit,
        )

        global_results = {
            "global_faint_audit_original": global_faint_original,
            "global_bright_audit_original": global_bright_original,
            "global_contrast_audit_original": global_contrast_original,
            "global_faint_audit_full": global_faint_full,
            "global_bright_audit_full": global_bright_full,
            "global_contrast_audit_full": global_contrast_full,
            "global_contrast_audit_full_with_permutation": global_contrast_full_with_perm,
            "global_contrast_permutation_summary": global_perm_summary,
            "global_contrast_direction_stats": global_direction,
        }

        within_results = {
            "within_host_faint_audit_full": within_faint_full,
            "within_host_bright_audit_full": within_bright_full,
            "within_host_contrast_audit_full": within_contrast_full,
            "within_host_contrast_audit_full_with_permutation": within_contrast_full_with_perm,
            "within_host_contrast_permutation_summary": within_perm_summary,
            "within_host_contrast_direction_stats": within_direction,
        }

        final_status, readiness_score, next_wall, best_cases = fixed_classifier(
            global_results,
            within_results,
        )

        candidate_rows = [
            global_faint_original,
            global_bright_original,
            global_contrast_original,
            global_faint_full,
            global_bright_full,
            global_contrast_full_with_perm,
            within_faint_full,
            within_bright_full,
            within_contrast_full_with_perm,
        ]
        write_csv(OUTDIR / "edge_pair_polarity_candidate_audit_v0_1_1.csv", candidate_rows)

        direction_rows = [global_direction, within_direction]
        write_csv(OUTDIR / "edge_pair_direction_stats_v0_1_1.csv", direction_rows)

        make_v011_plots(
            [global_faint_full, global_bright_full, global_contrast_full_with_perm],
            [within_faint_full, within_bright_full, within_contrast_full_with_perm],
        )

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

        write_csv(OUTDIR / "design_fit_comparison_v0_1_1.csv", design_fit_rows)

        residual = baseline["residual"]
        abs_residual = np.abs(residual)

        edge_counts = {
            "rows_total": int(X.shape[0]),
            "spine_38_41_43_rows": int(sum(1 for row in row_rows if row["contains_38_41_43_spine"])),
            "bridge_42_46_rows": int(sum(1 for row in row_rows if row["bridges_param42_param46"])),
            "touch_param42_rows": int(sum(1 for row in row_rows if row["touches_param42"])),
            "touch_param46_rows": int(sum(1 for row in row_rows if row["touches_param46_H0_like"])),
        }

        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.1.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Classifier-correction signed F160W edge-pair audit only. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_1_1": CLAIMS_V0_1_1,
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
            "global_edge_pair_results": global_results,
            "within_host_edge_pair_results": within_results,
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_1_1_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_1_1_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v0_1_1.json"),
                "candidate_audit_csv": str(OUTDIR / "edge_pair_polarity_candidate_audit_v0_1_1.csv"),
                "direction_stats_csv": str(OUTDIR / "edge_pair_direction_stats_v0_1_1.csv"),
                "global_permutation_summary_json": str(OUTDIR / "global_contrast_permutation_summary_v0_1_1.json"),
                "within_host_permutation_summary_json": str(OUTDIR / "within_host_contrast_permutation_summary_v0_1_1.json"),
                "plots": [
                    str(OUTDIR / "global_edge_pair_delta_chi2_v0_1_1.png"),
                    str(OUTDIR / "global_edge_pair_signed_scores_v0_1_1.png"),
                    str(OUTDIR / "within_host_edge_pair_delta_chi2_v0_1_1.png"),
                    str(OUTDIR / "within_host_edge_pair_signed_scores_v0_1_1.png"),
                ],
            },
            "interpretation": {
                "what_counts_as_support": (
                    "Faint-minus-bright contrast survives controls, beats permutation, beats signed-score permutation, "
                    "and has positive faint-minus-bright residual direction."
                ),
                "what_counts_as_narrowing": (
                    "Faint magnitude survives but signed contrast does not; TAIRID remains a residual-localization lens here."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only tests whether the SH0ES Table2 F160W boundary has signed edge-pair polarity."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_1_1_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_1_1_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.1.1\n\n")
            f.write("Boundary: classifier-correction signed edge-pair audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Claims v0.1.1:\n")
            f.write(json.dumps(CLAIMS_V0_1_1, indent=2, default=json_default) + "\n\n")
            f.write("Global edge-pair results:\n")
            f.write(json.dumps(global_results, indent=2, default=json_default) + "\n\n")
            f.write("Within-host edge-pair results:\n")
            f.write(json.dumps(within_results, indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only tests signed F160W edge-pair polarity.\n")

        print("TAIRID Boundary Prediction Battery v0.1.1 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.1.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_1_1_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_1_1_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
