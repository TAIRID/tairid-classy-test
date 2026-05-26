#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.1
SH0ES Table2 F160W signed polarity audit.

Purpose:
The prior SH0ES/Table2 tests found a repeated residual-pressure localization:

1. Table2 compact mapping is exact.
2. Residual pressure follows F160W / uncertainty / selection geometry.
3. The strongest non-overlapping band is the global F160W faintest 5%.
4. The global faint-tail signal is distributed across many host labels.
5. Within-host faintness did not lock as the main explanation.

This v0.1 battery changes the question.

Instead of only asking:
    "Where is residual pressure large?"

It asks:
    "Does the boundary have signed polarity?"

TAIRID pre-registered claim for this battery:
    If the F160W faint tail is a real boundary-pressure surface, then it should
    not only show elevated absolute residual pressure. It should also separate
    signed residual direction from the bright tail after controls.

Native TAIRID language:
    - boundary pressure should localize near a viability / measurement edge
    - unresolved pressure should show direction when compared across opposite edges
    - if only magnitude survives but sign does not, the claim narrows to
      measurement-selection localization rather than a stronger boundary-polarity rule

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This tests whether the SH0ES Table2 F160W boundary result has signed polarity.
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


OUTDIR = Path("tairid_boundary_polarity_battery_v0_1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

F160W_COLUMN = "table2_num_7"
FULL_DESIGN = "plus_host_top10_row_order_measurement_controls"
ORIGINAL_DESIGN = "original_47"

SEED = 42
PERMUTATION_REPEATS = 120
MIN_HOST_ROWS = 20
EPS = 1.0e-12


PREREGISTERED_TAIRID_CLAIMS = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.1",
    "claim_scope": "SH0ES Table2 F160W signed polarity only",
    "native_tairid_claim": (
        "A residual boundary surface should not only concentrate pressure near an edge. "
        "If the edge is structurally active, opposite sides of the measurement boundary "
        "should separate signed residual direction after the original model and declared controls."
    ),
    "prediction_1_location": (
        "The global F160W faintest 5% should remain stronger than the global brightest 5% "
        "after host, row-order, and measurement controls."
    ),
    "prediction_2_signed_polarity": (
        "The contrast vector [global faintest 5% minus global brightest 5%] should survive "
        "full controls and beat same-count random contrast permutations."
    ),
    "prediction_3_scale_boundary": (
        "If global contrast survives but within-host contrast does not, the boundary is at the "
        "measurement-system scale rather than each host's local faint edge."
    ),
    "failure_rule": (
        "If the bright tail is stronger, or the faint-bright contrast fails permutation, TAIRID "
        "cannot claim signed boundary polarity here and must narrow the result to measurement "
        "or selection localization only."
    ),
    "anti_ad_hoc_rule": (
        "The pass/fail logic is decided before reading the v0.1 result. Failed polarity cannot "
        "be rescued by changing the meaning of polarity after the run."
    ),
}


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Counter):
        return dict(obj)
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


def numeric(row, key):
    return v16.get_numeric(row, key)


def entity(row, key):
    return v16.get_entity(row, key)


def standardize(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    std = float(np.std(values))

    if not np.isfinite(std) or std <= 1.0e-14:
        return np.zeros_like(values)

    return (values - float(np.mean(values))) / std


def build_global_band_mask(mapped_rows, y_length, low_percentile, high_percentile):
    finite = []

    for row in mapped_rows:
        value = numeric(row, F160W_COLUMN)

        if np.isfinite(value):
            finite.append((float(value), row))

    finite = sorted(finite, key=lambda item: item[0])
    n = len(finite)

    start = int(math.floor(n * low_percentile / 100.0))
    end = int(math.floor(n * high_percentile / 100.0))

    if high_percentile == 100:
        end = n

    band = finite[start:end]
    mask = np.zeros(y_length, dtype=float)
    rows = []

    for value, row in band:
        idx = int(row["compact_observation_index"])
        mask[idx] = 1.0

        enriched = dict(row)
        enriched["f160w_value"] = float(value)
        enriched["host_guess_clean"] = entity(row, "host_guess")
        enriched["band_low_percentile"] = low_percentile
        enriched["band_high_percentile"] = high_percentile
        rows.append(enriched)

    values = np.asarray([value for value, _ in band], dtype=float)

    meta = {
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "count": int(np.sum(mask > 0)),
        "finite_count": n,
        "f160w_min": float(np.min(values)) if len(values) else None,
        "f160w_max": float(np.max(values)) if len(values) else None,
        "f160w_mean": float(np.mean(values)) if len(values) else None,
    }

    return mask, rows, meta


def finite_host_groups(mapped_rows):
    groups = defaultdict(list)

    for row in mapped_rows:
        value = numeric(row, F160W_COLUMN)

        if not np.isfinite(value):
            continue

        host = entity(row, "host_guess")
        groups[host].append((float(value), row))

    cleaned = {}

    for host, items in groups.items():
        items = sorted(items, key=lambda item: item[0])

        if len(items) >= MIN_HOST_ROWS:
            cleaned[host] = items

    return cleaned


def build_within_host_tail_masks(mapped_rows, y_length, percentile=5):
    host_groups = finite_host_groups(mapped_rows)

    faint = np.zeros(y_length, dtype=float)
    bright = np.zeros(y_length, dtype=float)

    faint_rows = []
    bright_rows = []

    selected_by_host = {}

    for host, items in host_groups.items():
        n = len(items)
        k = max(1, int(math.ceil(n * percentile / 100.0)))

        bright_items = items[:k]
        faint_items = items[-k:]

        selected_by_host[host] = {
            "host_row_count": n,
            "selected_each_side": k,
        }

        for value, row in bright_items:
            idx = int(row["compact_observation_index"])
            bright[idx] = 1.0
            out = dict(row)
            out["host_guess_clean"] = host
            out["f160w_value"] = float(value)
            out["within_host_side"] = "bright"
            bright_rows.append(out)

        for value, row in faint_items:
            idx = int(row["compact_observation_index"])
            faint[idx] = 1.0
            out = dict(row)
            out["host_guess_clean"] = host
            out["f160w_value"] = float(value)
            out["within_host_side"] = "faint"
            faint_rows.append(out)

    meta = {
        "percentile": percentile,
        "min_host_rows": MIN_HOST_ROWS,
        "host_count_used": len(host_groups),
        "faint_count": int(np.sum(faint > 0)),
        "bright_count": int(np.sum(bright > 0)),
        "selected_by_host": selected_by_host,
    }

    return faint, bright, faint_rows, bright_rows, meta


def signed_residual_stats(name, values, fit):
    raw = np.asarray(values, dtype=float).reshape(-1)
    residual = np.asarray(fit["residual"], dtype=float).reshape(-1)
    cinv_residual = np.asarray(fit["Cinv_residual"], dtype=float).reshape(-1)

    pos = raw > 0
    neg = raw < 0
    nonzero = np.abs(raw) > 0

    def stat_block(mask):
        if not np.any(mask):
            return {
                "count": 0,
                "mean_residual": None,
                "median_residual": None,
                "mean_abs_residual": None,
                "mean_cinv_residual": None,
            }

        r = residual[mask]
        cr = cinv_residual[mask]

        return {
            "count": int(np.sum(mask)),
            "mean_residual": float(np.mean(r)),
            "median_residual": float(np.median(r)),
            "mean_abs_residual": float(np.mean(np.abs(r))),
            "mean_cinv_residual": float(np.mean(cr)),
        }

    pos_stats = stat_block(pos)
    neg_stats = stat_block(neg)
    nonzero_stats = stat_block(nonzero)

    return {
        "candidate": name,
        "positive_group": pos_stats,
        "negative_group": neg_stats,
        "nonzero_group": nonzero_stats,
        "signed_mean_residual_difference_pos_minus_neg": (
            float(pos_stats["mean_residual"] - neg_stats["mean_residual"])
            if pos_stats["count"] and neg_stats["count"]
            else None
        ),
        "signed_mean_cinv_residual_difference_pos_minus_neg": (
            float(pos_stats["mean_cinv_residual"] - neg_stats["mean_cinv_residual"])
            if pos_stats["count"] and neg_stats["count"]
            else None
        ),
    }


def audit_vector(name, values, kind, y, c_factor, fit, design_name, metadata=None):
    metadata = metadata or {}

    return v16.audit_vector(
        name=name,
        values=values,
        kind=kind,
        y=y,
        c_factor=c_factor,
        fit=fit,
        design_name=design_name,
        metadata=metadata,
    )


def same_count_contrast_permutation(name, pos_count, neg_count, y, c_factor, fit, design_name, mapped_rows):
    rng = np.random.default_rng(SEED)
    spine_indices = np.asarray(
        [int(row["compact_observation_index"]) for row in mapped_rows],
        dtype=int,
    )

    rows = []

    for repeat in range(PERMUTATION_REPEATS):
        values = np.zeros(len(y), dtype=float)
        chosen = rng.choice(
            spine_indices,
            size=min(pos_count + neg_count, len(spine_indices)),
            replace=False,
        )

        pos = chosen[:pos_count]
        neg = chosen[pos_count:pos_count + neg_count]

        values[pos] = 1.0
        values[neg] = -1.0

        rows.append(
            audit_vector(
                name=f"{name}_same_count_permutation_{repeat}",
                values=values,
                kind="same_count_contrast_permutation",
                y=y,
                c_factor=c_factor,
                fit=fit,
                design_name=design_name,
                metadata={
                    "source_candidate": name,
                    "repeat": repeat,
                    "positive_count": pos_count,
                    "negative_count": neg_count,
                },
            )
        )

    deltas = np.asarray([row["delta_chi2_score"] for row in rows], dtype=float)
    scores = np.asarray([row["score"] for row in rows], dtype=float)

    summary = {
        "candidate": name,
        "design": design_name,
        "repeats": PERMUTATION_REPEATS,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "permutation_delta_mean": float(np.mean(deltas)),
        "permutation_delta_95": float(np.percentile(deltas, 95)),
        "permutation_delta_99": float(np.percentile(deltas, 99)),
        "permutation_abs_score_mean": float(np.mean(np.abs(scores))),
        "permutation_abs_score_95": float(np.percentile(np.abs(scores), 95)),
        "permutation_abs_score_99": float(np.percentile(np.abs(scores), 99)),
    }

    return rows, summary


def add_permutation_flags(observed, perm_summary):
    out = dict(observed)
    out["permutation_delta_95"] = perm_summary["permutation_delta_95"]
    out["permutation_delta_99"] = perm_summary["permutation_delta_99"]
    out["permutation_abs_score_95"] = perm_summary["permutation_abs_score_95"]
    out["permutation_abs_score_99"] = perm_summary["permutation_abs_score_99"]
    out["observed_exceeds_95_percent_permutation_delta"] = bool(
        observed["delta_chi2_score"] > perm_summary["permutation_delta_95"]
    )
    out["observed_exceeds_99_percent_permutation_delta"] = bool(
        observed["delta_chi2_score"] > perm_summary["permutation_delta_99"]
    )
    out["observed_abs_score_exceeds_95_percent_permutation"] = bool(
        abs(observed["score"]) > perm_summary["permutation_abs_score_95"]
    )
    out["observed_abs_score_exceeds_99_percent_permutation"] = bool(
        abs(observed["score"]) > perm_summary["permutation_abs_score_99"]
    )

    return out


def decide_status(global_results, within_results):
    global_faint = global_results["global_faint_audit_full"]
    global_bright = global_results["global_bright_audit_full"]
    global_contrast = global_results["global_contrast_audit_full_with_permutation"]

    within_contrast = within_results["within_host_contrast_audit_full_with_permutation"]
    within_faint = within_results["within_host_faint_audit_full"]
    within_bright = within_results["within_host_bright_audit_full"]

    best_cases = {
        "global_faint": global_faint,
        "global_bright": global_bright,
        "global_contrast": global_contrast,
        "within_host_faint": within_faint,
        "within_host_bright": within_bright,
        "within_host_contrast": within_contrast,
    }

    global_faint_stronger = global_faint["delta_chi2_score"] > global_bright["delta_chi2_score"] + 20.0
    global_contrast_pass = (
        global_contrast["delta_chi2_score"] >= 25.0
        and global_contrast["p_value_chi2_one_dof"] <= 0.01
        and global_contrast["observed_exceeds_99_percent_permutation_delta"] is True
        and global_contrast["observed_abs_score_exceeds_99_percent_permutation"] is True
    )

    within_contrast_pass = (
        within_contrast["delta_chi2_score"] >= 25.0
        and within_contrast["p_value_chi2_one_dof"] <= 0.01
        and within_contrast["observed_exceeds_99_percent_permutation_delta"] is True
        and within_contrast["observed_abs_score_exceeds_99_percent_permutation"] is True
    )

    within_faint_stronger = within_faint["delta_chi2_score"] > within_bright["delta_chi2_score"] + 20.0

    if global_faint_stronger and global_contrast_pass and not within_contrast_pass:
        return (
            "global_boundary_polarity_supported_system_scale_not_within_host",
            8,
            "The global F160W faint-vs-bright contrast has signed polarity after controls, but within-host polarity does not lock.",
            best_cases,
        )

    if global_faint_stronger and global_contrast_pass and within_contrast_pass and within_faint_stronger:
        return (
            "global_and_within_host_boundary_polarity_supported",
            8,
            "Both global F160W boundary contrast and within-host faint-side contrast survive signed polarity controls.",
            best_cases,
        )

    if global_faint_stronger and not global_contrast_pass:
        return (
            "global_faint_boundary_magnitude_without_signed_polarity",
            7,
            "The global faint tail remains stronger in magnitude, but signed polarity does not pass permutation controls.",
            best_cases,
        )

    if global_bright["delta_chi2_score"] >= global_faint["delta_chi2_score"]:
        return (
            "polarity_test_not_faint_specific",
            6,
            "The bright side is as strong as or stronger than the faint side, so the faint-boundary polarity claim weakens.",
            best_cases,
        )

    return (
        "no_locked_boundary_polarity",
        6,
        "The polarity battery does not lock a signed boundary structure after controls.",
        best_cases,
    )


def make_plots(global_plot_rows, within_plot_rows, signed_stats_rows):
    if global_plot_rows:
        labels = [row["candidate"] for row in global_plot_rows]
        deltas = [row["delta_chi2_score"] for row in global_plot_rows]
        scores = [row["score"] for row in global_plot_rows]

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(labels)), deltas)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("delta chi2 score")
        plt.title("Global F160W boundary candidates")
        plt.tight_layout()
        plt.savefig(OUTDIR / "global_boundary_delta_chi2_v0_1.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(labels)), scores)
        plt.axhline(0.0, linewidth=1)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("signed score")
        plt.title("Global F160W boundary signed scores")
        plt.tight_layout()
        plt.savefig(OUTDIR / "global_boundary_signed_scores_v0_1.png", dpi=160)
        plt.close()

    if within_plot_rows:
        labels = [row["candidate"] for row in within_plot_rows]
        deltas = [row["delta_chi2_score"] for row in within_plot_rows]
        scores = [row["score"] for row in within_plot_rows]

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(labels)), deltas)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("delta chi2 score")
        plt.title("Within-host F160W boundary candidates")
        plt.tight_layout()
        plt.savefig(OUTDIR / "within_host_boundary_delta_chi2_v0_1.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(labels)), scores)
        plt.axhline(0.0, linewidth=1)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("signed score")
        plt.title("Within-host F160W boundary signed scores")
        plt.tight_layout()
        plt.savefig(OUTDIR / "within_host_boundary_signed_scores_v0_1.png", dpi=160)
        plt.close()


def main():
    print("TAIRID Boundary Prediction Battery v0.1 starting.")
    print("Boundary: signed polarity audit only; not proof.")

    write_json(OUTDIR / "preregistered_tairid_claims_v0_1.json", PREREGISTERED_TAIRID_CLAIMS)

    repair_summary = {}

    try:
        v16.OUTDIR = OUTDIR
        v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        ns, repair_summary = v16.load_v15_helpers()
        write_json(OUTDIR / "v15_import_repair_summary_v0_1.json", repair_summary)

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

        write_csv(OUTDIR / "download_ledger_v0_1.csv", ledger)
        write_json(OUTDIR / "download_attempts_v0_1.json", {"compact": downloads, "auxiliary": aux_results})

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

        write_json(OUTDIR / "parse_meta_v0_1.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v0_1.json", parse_errors)

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
                "test_name": "TAIRID Boundary Prediction Battery v0.1",
                "final_status": "boundary_polarity_battery_v0_1_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or Table2 retrieval before polarity audit.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "boundary_polarity_battery_v0_1_summary.json", summary)
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

        write_csv(OUTDIR / "compact_row_map_v0_1.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v0_1.csv", cluster_rows)

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

        # Global bright/faint tail masks.
        bright_mask, bright_rows, bright_meta = build_global_band_mask(mapped_rows, len(y), 0, 5)
        faint_mask, faint_rows, faint_meta = build_global_band_mask(mapped_rows, len(y), 95, 100)
        global_contrast = faint_mask - bright_mask

        write_csv(OUTDIR / "global_bright_00_05_rows_v0_1.csv", bright_rows)
        write_csv(OUTDIR / "global_faint_95_100_rows_v0_1.csv", faint_rows)

        global_faint_full = audit_vector(
            "global_f160w_faint_95_100",
            faint_mask,
            "global_faint_mask",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=faint_meta,
        )
        global_bright_full = audit_vector(
            "global_f160w_bright_00_05",
            bright_mask,
            "global_bright_mask",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=bright_meta,
        )
        global_contrast_full = audit_vector(
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

        global_faint_original = audit_vector(
            "global_f160w_faint_95_100",
            faint_mask,
            "global_faint_mask",
            y,
            c_factor,
            original_fit,
            ORIGINAL_DESIGN,
            metadata=faint_meta,
        )
        global_bright_original = audit_vector(
            "global_f160w_bright_00_05",
            bright_mask,
            "global_bright_mask",
            y,
            c_factor,
            original_fit,
            ORIGINAL_DESIGN,
            metadata=bright_meta,
        )
        global_contrast_original = audit_vector(
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

        global_perm_rows, global_perm_summary = same_count_contrast_permutation(
            "global_f160w_faint95_minus_bright05_contrast",
            int(np.sum(global_contrast > 0)),
            int(np.sum(global_contrast < 0)),
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            mapped_rows,
        )
        write_csv(OUTDIR / "global_contrast_permutation_details_v0_1.csv", global_perm_rows)
        write_json(OUTDIR / "global_contrast_permutation_summary_v0_1.json", global_perm_summary)

        global_contrast_full_with_perm = add_permutation_flags(global_contrast_full, global_perm_summary)

        global_signed_stats_full = [
            signed_residual_stats("global_f160w_faint_95_100", faint_mask, full_fit),
            signed_residual_stats("global_f160w_bright_00_05", bright_mask, full_fit),
            signed_residual_stats("global_f160w_faint95_minus_bright05_contrast", global_contrast, full_fit),
        ]
        write_json(OUTDIR / "global_signed_residual_stats_full_controls_v0_1.json", global_signed_stats_full)

        # Within-host contrast.
        within_faint, within_bright, within_faint_rows, within_bright_rows, within_meta = build_within_host_tail_masks(
            mapped_rows,
            len(y),
            percentile=5,
        )
        within_contrast = within_faint - within_bright

        write_csv(OUTDIR / "within_host_faint_rows_v0_1.csv", within_faint_rows)
        write_csv(OUTDIR / "within_host_bright_rows_v0_1.csv", within_bright_rows)
        write_json(OUTDIR / "within_host_tail_meta_v0_1.json", within_meta)

        within_faint_full = audit_vector(
            "within_host_f160w_faint_05",
            within_faint,
            "within_host_faint_mask",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=within_meta,
        )
        within_bright_full = audit_vector(
            "within_host_f160w_bright_05",
            within_bright,
            "within_host_bright_mask",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=within_meta,
        )
        within_contrast_full = audit_vector(
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

        within_perm_rows, within_perm_summary = same_count_contrast_permutation(
            "within_host_f160w_faint05_minus_bright05_contrast",
            int(np.sum(within_contrast > 0)),
            int(np.sum(within_contrast < 0)),
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            mapped_rows,
        )
        write_csv(OUTDIR / "within_host_contrast_permutation_details_v0_1.csv", within_perm_rows)
        write_json(OUTDIR / "within_host_contrast_permutation_summary_v0_1.json", within_perm_summary)

        within_contrast_full_with_perm = add_permutation_flags(within_contrast_full, within_perm_summary)

        within_signed_stats_full = [
            signed_residual_stats("within_host_f160w_faint_05", within_faint, full_fit),
            signed_residual_stats("within_host_f160w_bright_05", within_bright, full_fit),
            signed_residual_stats("within_host_f160w_faint05_minus_bright05_contrast", within_contrast, full_fit),
        ]
        write_json(OUTDIR / "within_host_signed_residual_stats_full_controls_v0_1.json", within_signed_stats_full)

        global_results = {
            "global_faint_audit_original": global_faint_original,
            "global_bright_audit_original": global_bright_original,
            "global_contrast_audit_original": global_contrast_original,
            "global_faint_audit_full": global_faint_full,
            "global_bright_audit_full": global_bright_full,
            "global_contrast_audit_full": global_contrast_full,
            "global_contrast_audit_full_with_permutation": global_contrast_full_with_perm,
            "global_contrast_permutation_summary": global_perm_summary,
            "global_signed_residual_stats_full": global_signed_stats_full,
        }

        within_results = {
            "within_host_faint_audit_full": within_faint_full,
            "within_host_bright_audit_full": within_bright_full,
            "within_host_contrast_audit_full": within_contrast_full,
            "within_host_contrast_audit_full_with_permutation": within_contrast_full_with_perm,
            "within_host_contrast_permutation_summary": within_perm_summary,
            "within_host_signed_residual_stats_full": within_signed_stats_full,
        }

        final_status, readiness_score, next_wall, best_cases = decide_status(
            global_results,
            within_results,
        )

        global_plot_rows = [
            global_faint_full,
            global_bright_full,
            global_contrast_full_with_perm,
        ]
        within_plot_rows = [
            within_faint_full,
            within_bright_full,
            within_contrast_full_with_perm,
        ]

        make_plots(global_plot_rows, within_plot_rows, global_signed_stats_full + within_signed_stats_full)

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

        write_csv(OUTDIR / "design_fit_comparison_v0_1.csv", design_fit_rows)

        all_candidate_rows = [
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
        write_csv(OUTDIR / "boundary_polarity_candidate_audit_v0_1.csv", all_candidate_rows)

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
            "test_name": "TAIRID Boundary Prediction Battery v0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Signed F160W boundary-polarity audit only. Not proof of TAIRID, not H0 resolution, "
                "not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "preregistered_tairid_claims": PREREGISTERED_TAIRID_CLAIMS,
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
            "global_boundary_results": global_results,
            "within_host_boundary_results": within_results,
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_1_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_1_summary.txt"),
                "preregistered_claims_json": str(OUTDIR / "preregistered_tairid_claims_v0_1.json"),
                "candidate_audit_csv": str(OUTDIR / "boundary_polarity_candidate_audit_v0_1.csv"),
                "global_contrast_permutation_summary_json": str(OUTDIR / "global_contrast_permutation_summary_v0_1.json"),
                "within_host_contrast_permutation_summary_json": str(OUTDIR / "within_host_contrast_permutation_summary_v0_1.json"),
                "global_signed_stats_json": str(OUTDIR / "global_signed_residual_stats_full_controls_v0_1.json"),
                "within_host_signed_stats_json": str(OUTDIR / "within_host_signed_residual_stats_full_controls_v0_1.json"),
                "plots": [
                    str(OUTDIR / "global_boundary_delta_chi2_v0_1.png"),
                    str(OUTDIR / "global_boundary_signed_scores_v0_1.png"),
                    str(OUTDIR / "within_host_boundary_delta_chi2_v0_1.png"),
                    str(OUTDIR / "within_host_boundary_signed_scores_v0_1.png"),
                ],
            },
            "interpretation": {
                "what_supports_tairid_polarity": (
                    "Global faint tail is stronger than bright tail and the faint-minus-bright signed contrast "
                    "survives full controls and same-count permutation checks."
                ),
                "what_narrows_tairid": (
                    "Global faint magnitude survives, but signed contrast fails. That would keep TAIRID as a "
                    "residual-localization lens here, not a signed boundary-polarity result."
                ),
                "what_fails_this_lane": (
                    "Bright tail is stronger, contrast fails controls, or within-host/global results become inconsistent "
                    "without a stable scale interpretation."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only tests whether the SH0ES Table2 F160W boundary "
                    "has signed polarity after controls."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_1_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_1_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.1\n\n")
            f.write("Boundary: signed F160W polarity audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")

            f.write("Pre-registered TAIRID claims:\n")
            f.write(json.dumps(PREREGISTERED_TAIRID_CLAIMS, indent=2, default=json_default) + "\n\n")

            f.write("Global boundary results:\n")
            f.write(json.dumps(global_results, indent=2, default=json_default) + "\n\n")

            f.write("Within-host boundary results:\n")
            f.write(json.dumps(within_results, indent=2, default=json_default) + "\n\n")

            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")

            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only tests signed polarity in the SH0ES Table2 F160W boundary.\n")

        print("TAIRID Boundary Prediction Battery v0.1 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_1_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_1_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
