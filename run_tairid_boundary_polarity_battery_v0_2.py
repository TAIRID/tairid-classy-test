#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.2
SH0ES Table2 multi-variable edge-pair polarity audit.

Purpose:
v0.1.1 found signed edge-pair polarity in the F160W layer:

    global F160W faint edge - global F160W bright edge
    within-host F160W faint edge - within-host F160W bright edge

This v0.2 battery asks the anti-ad-hoc question:

    Is F160W uniquely boundary-active,
    or can every Table2 numeric variable produce similar polarity?

Native TAIRID claim:
    TAIRID does not get to claim a boundary result merely because a variable can
    be split into high and low groups. The F160W edge should remain a primary
    boundary candidate when compared against nearby measurement variables.

This test audits Table2 numeric columns:
    table2_num_4  likely period
    table2_num_5  likely color
    table2_num_6  likely color/pre-F160W uncertainty
    table2_num_7  likely F160W
    table2_num_8  likely F160W uncertainty
    table2_num_9  likely metallicity

For each variable:
    1. Global high 5% minus low 5% contrast
    2. Within-host high 5% minus low 5% contrast
    3. Full controls:
       original SH0ES compact model + host top10 + row order + measurement controls
    4. Same-count random contrast permutations
    5. Signed residual direction:
       high side mean residual minus low side mean residual

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only tests whether F160W edge-pair polarity is specific or generic.
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
import run_tairid_boundary_polarity_battery_v0_1 as b01


OUTDIR = Path("tairid_boundary_polarity_battery_v0_2_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

FULL_DESIGN = "plus_host_top10_row_order_measurement_controls"
ORIGINAL_DESIGN = "original_47"

EDGE_PERCENTILE = 5
MIN_HOST_ROWS = 20
PERMUTATION_REPEATS = 120

VARIABLES = [
    {
        "column": "table2_num_4",
        "label": "period_like",
        "plain": "Period-like Table2 numeric column",
        "tairid_role": "pacing / cycle proxy control",
    },
    {
        "column": "table2_num_5",
        "label": "color_like",
        "plain": "Color-like Table2 numeric column",
        "tairid_role": "measurement/color proxy control",
    },
    {
        "column": "table2_num_6",
        "label": "pre_f160w_sigma_or_color_uncertainty_like",
        "plain": "Pre-F160W sigma / color-side uncertainty-like column",
        "tairid_role": "uncertainty boundary control",
    },
    {
        "column": "table2_num_7",
        "label": "f160w_like",
        "plain": "F160W-like magnitude column",
        "tairid_role": "primary predicted measurement boundary",
        "primary_prediction": True,
    },
    {
        "column": "table2_num_8",
        "label": "f160w_sigma_like",
        "plain": "F160W uncertainty-like column",
        "tairid_role": "uncertainty boundary control",
    },
    {
        "column": "table2_num_9",
        "label": "metallicity_like",
        "plain": "Metallicity-like Table2 numeric column",
        "tairid_role": "calibration/environment proxy control",
    },
]

CLAIMS_V0_2 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.2",
    "scope": "SH0ES Table2 multi-variable edge-pair polarity specificity",
    "native_tairid_claim": (
        "If TAIRID is doing more than variable hunting, the F160W boundary should remain "
        "a primary edge-pair polarity candidate when compared against other Table2 numeric columns."
    ),
    "primary_prediction": (
        "F160W-like table2_num_7 should pass global and/or within-host signed edge-pair polarity "
        "and should remain among the strongest variables after full controls and permutations."
    ),
    "specificity_gate": (
        "If many unrelated columns pass equally or stronger, the result narrows from F160W boundary "
        "specificity to a broader Table2 measurement-gradient effect."
    ),
    "anti_ad_hoc_rule": (
        "This test declares F160W as the primary predicted boundary before comparison. "
        "A stronger non-F160W result cannot simply be renamed as the intended TAIRID result."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only tests whether the signed polarity found "
        "in v0.1.1 is specific to the F160W measurement edge or generic across Table2 columns."
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


def finite_values_for_column(mapped_rows, column):
    finite = []

    for row in mapped_rows:
        value = numeric(row, column)

        if np.isfinite(value):
            finite.append((float(value), row))

    return sorted(finite, key=lambda item: item[0])


def build_global_edge_pair(mapped_rows, y_length, column, label, percentile=EDGE_PERCENTILE):
    finite = finite_values_for_column(mapped_rows, column)
    n = len(finite)

    if n < 100:
        raise RuntimeError(f"Not enough finite rows for {column}: {n}")

    low_end = int(math.floor(n * percentile / 100.0))
    high_start = int(math.floor(n * (100.0 - percentile) / 100.0))

    low_items = finite[:low_end]
    high_items = finite[high_start:]

    low_mask = np.zeros(y_length, dtype=float)
    high_mask = np.zeros(y_length, dtype=float)
    low_rows = []
    high_rows = []

    for value, row in low_items:
        idx = int(row["compact_observation_index"])
        low_mask[idx] = 1.0
        out = dict(row)
        out["edge_variable"] = label
        out["edge_column"] = column
        out["edge_side"] = "low"
        out["edge_value"] = float(value)
        out["host_guess_clean"] = entity(row, "host_guess")
        low_rows.append(out)

    for value, row in high_items:
        idx = int(row["compact_observation_index"])
        high_mask[idx] = 1.0
        out = dict(row)
        out["edge_variable"] = label
        out["edge_column"] = column
        out["edge_side"] = "high"
        out["edge_value"] = float(value)
        out["host_guess_clean"] = entity(row, "host_guess")
        high_rows.append(out)

    contrast = high_mask - low_mask

    low_vals = np.asarray([value for value, _ in low_items], dtype=float)
    high_vals = np.asarray([value for value, _ in high_items], dtype=float)

    meta = {
        "column": column,
        "label": label,
        "edge_type": "global_high_minus_low",
        "percentile_each_side": percentile,
        "finite_count": n,
        "low_count": int(np.sum(low_mask > 0)),
        "high_count": int(np.sum(high_mask > 0)),
        "low_min": float(np.min(low_vals)),
        "low_max": float(np.max(low_vals)),
        "low_mean": float(np.mean(low_vals)),
        "high_min": float(np.min(high_vals)),
        "high_max": float(np.max(high_vals)),
        "high_mean": float(np.mean(high_vals)),
    }

    return low_mask, high_mask, contrast, low_rows, high_rows, meta


def build_within_host_edge_pair(mapped_rows, y_length, column, label, percentile=EDGE_PERCENTILE):
    groups = defaultdict(list)

    for row in mapped_rows:
        value = numeric(row, column)

        if not np.isfinite(value):
            continue

        host = entity(row, "host_guess")
        groups[host].append((float(value), row))

    usable = {
        host: sorted(items, key=lambda item: item[0])
        for host, items in groups.items()
        if len(items) >= MIN_HOST_ROWS
    }

    low_mask = np.zeros(y_length, dtype=float)
    high_mask = np.zeros(y_length, dtype=float)
    low_rows = []
    high_rows = []
    host_meta = []

    for host, items in usable.items():
        n = len(items)
        k = max(1, int(math.ceil(n * percentile / 100.0)))

        low_items = items[:k]
        high_items = items[-k:]

        for value, row in low_items:
            idx = int(row["compact_observation_index"])
            low_mask[idx] = 1.0
            out = dict(row)
            out["edge_variable"] = label
            out["edge_column"] = column
            out["edge_side"] = "within_host_low"
            out["edge_value"] = float(value)
            out["host_guess_clean"] = host
            out["host_rank_count"] = n
            out["host_selected_each_side"] = k
            low_rows.append(out)

        for value, row in high_items:
            idx = int(row["compact_observation_index"])
            high_mask[idx] = 1.0
            out = dict(row)
            out["edge_variable"] = label
            out["edge_column"] = column
            out["edge_side"] = "within_host_high"
            out["edge_value"] = float(value)
            out["host_guess_clean"] = host
            out["host_rank_count"] = n
            out["host_selected_each_side"] = k
            high_rows.append(out)

        values = np.asarray([value for value, _ in items], dtype=float)
        host_meta.append(
            {
                "host": host,
                "host_row_count": n,
                "selected_each_side": k,
                "column": column,
                "label": label,
                "host_value_min": float(np.min(values)),
                "host_value_max": float(np.max(values)),
            }
        )

    contrast = high_mask - low_mask

    meta = {
        "column": column,
        "label": label,
        "edge_type": "within_host_high_minus_low",
        "percentile_each_side": percentile,
        "min_host_rows": MIN_HOST_ROWS,
        "host_count_used": len(usable),
        "low_count": int(np.sum(low_mask > 0)),
        "high_count": int(np.sum(high_mask > 0)),
        "host_meta": host_meta,
    }

    return low_mask, high_mask, contrast, low_rows, high_rows, meta


def audit_vector(name, values, kind, y, c_factor, fit, design_name, metadata=None):
    return v16.audit_vector(
        name=name,
        values=values,
        kind=kind,
        y=y,
        c_factor=c_factor,
        fit=fit,
        design_name=design_name,
        metadata=metadata or {},
    )


def direction_stats(name, contrast_values, fit):
    raw = np.asarray(contrast_values, dtype=float).reshape(-1)
    residual = np.asarray(fit["residual"], dtype=float).reshape(-1)
    cinv_residual = np.asarray(fit["Cinv_residual"], dtype=float).reshape(-1)

    pos = raw > 0
    neg = raw < 0

    def block(mask):
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

    high = block(pos)
    low = block(neg)

    mean_diff = (
        float(high["mean_residual"] - low["mean_residual"])
        if high["count"] and low["count"]
        else None
    )

    cinv_diff = (
        float(high["mean_cinv_residual"] - low["mean_cinv_residual"])
        if high["count"] and low["count"]
        else None
    )

    return {
        "candidate": name,
        "positive_group_plain": "high edge",
        "negative_group_plain": "low edge",
        "high_count": high["count"],
        "low_count": low["count"],
        "high_mean_residual": high["mean_residual"],
        "low_mean_residual": low["mean_residual"],
        "high_mean_abs_residual": high["mean_abs_residual"],
        "low_mean_abs_residual": low["mean_abs_residual"],
        "signed_mean_residual_difference_high_minus_low": mean_diff,
        "signed_mean_cinv_residual_difference_high_minus_low": cinv_diff,
        "high_side_more_positive": bool(mean_diff is not None and mean_diff > 0.0),
    }


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


def same_count_contrast_permutation(name, contrast, y, c_factor, fit, design_name, mapped_rows):
    b01.PERMUTATION_REPEATS = PERMUTATION_REPEATS

    pos_count = int(np.sum(np.asarray(contrast) > 0))
    neg_count = int(np.sum(np.asarray(contrast) < 0))

    rows, summary = b01.same_count_contrast_permutation(
        name,
        pos_count,
        neg_count,
        y,
        c_factor,
        fit,
        design_name,
        mapped_rows,
    )

    return rows, summary


def pass_gate(audit_row, direction_row, require_positive_direction=False):
    if not audit_row:
        return False

    core = bool(
        audit_row.get("delta_chi2_score", 0.0) >= 25.0
        and audit_row.get("p_value_chi2_one_dof", 1.0) <= 0.01
        and audit_row.get("observed_exceeds_99_percent_permutation_delta") is True
        and audit_row.get("observed_abs_score_exceeds_99_percent_permutation") is True
    )

    if not core:
        return False

    if require_positive_direction:
        return bool(direction_row.get("high_side_more_positive") is True)

    return True


def analyze_variable(variable, mapped_rows, y, c_factor, full_fit, original_fit):
    column = variable["column"]
    label = variable["label"]
    y_length = len(y)

    result = {
        "column": column,
        "label": label,
        "plain": variable["plain"],
        "tairid_role": variable["tairid_role"],
        "primary_prediction": bool(variable.get("primary_prediction", False)),
        "status": "started",
    }

    all_audit_rows = []
    direction_rows = []
    permutation_summaries = []
    permutation_rows_all = []

    # Global high-low edge pair.
    global_low, global_high, global_contrast, global_low_rows, global_high_rows, global_meta = build_global_edge_pair(
        mapped_rows,
        y_length,
        column,
        label,
    )

    write_csv(OUTDIR / f"{safe_name(label)}_global_low_rows_v0_2.csv", global_low_rows)
    write_csv(OUTDIR / f"{safe_name(label)}_global_high_rows_v0_2.csv", global_high_rows)

    global_high_audit = audit_vector(
        f"{label}_global_high_{100 - EDGE_PERCENTILE:02d}_100",
        global_high,
        "global_high_edge_mask",
        y,
        c_factor,
        full_fit,
        FULL_DESIGN,
        metadata=global_meta,
    )
    global_low_audit = audit_vector(
        f"{label}_global_low_00_{EDGE_PERCENTILE:02d}",
        global_low,
        "global_low_edge_mask",
        y,
        c_factor,
        full_fit,
        FULL_DESIGN,
        metadata=global_meta,
    )
    global_contrast_audit = audit_vector(
        f"{label}_global_high_minus_low_contrast",
        global_contrast,
        "global_high_minus_low_contrast",
        y,
        c_factor,
        full_fit,
        FULL_DESIGN,
        metadata=global_meta,
    )

    global_perm_rows, global_perm_summary = same_count_contrast_permutation(
        f"{label}_global_high_minus_low_contrast",
        global_contrast,
        y,
        c_factor,
        full_fit,
        FULL_DESIGN,
        mapped_rows,
    )
    global_contrast_with_perm = add_permutation_flags(global_contrast_audit, global_perm_summary)
    global_direction = direction_stats(
        f"{label}_global_high_minus_low_contrast",
        global_contrast,
        full_fit,
    )

    # Original model contrast, for reference.
    global_contrast_original = audit_vector(
        f"{label}_global_high_minus_low_contrast_original",
        global_contrast,
        "global_high_minus_low_contrast_original",
        y,
        c_factor,
        original_fit,
        ORIGINAL_DESIGN,
        metadata=global_meta,
    )

    # Within-host high-low edge pair.
    within_low, within_high, within_contrast, within_low_rows, within_high_rows, within_meta = build_within_host_edge_pair(
        mapped_rows,
        y_length,
        column,
        label,
    )

    write_csv(OUTDIR / f"{safe_name(label)}_within_host_low_rows_v0_2.csv", within_low_rows)
    write_csv(OUTDIR / f"{safe_name(label)}_within_host_high_rows_v0_2.csv", within_high_rows)

    within_high_audit = audit_vector(
        f"{label}_within_host_high_{EDGE_PERCENTILE:02d}",
        within_high,
        "within_host_high_edge_mask",
        y,
        c_factor,
        full_fit,
        FULL_DESIGN,
        metadata=within_meta,
    )
    within_low_audit = audit_vector(
        f"{label}_within_host_low_{EDGE_PERCENTILE:02d}",
        within_low,
        "within_host_low_edge_mask",
        y,
        c_factor,
        full_fit,
        FULL_DESIGN,
        metadata=within_meta,
    )
    within_contrast_audit = audit_vector(
        f"{label}_within_host_high_minus_low_contrast",
        within_contrast,
        "within_host_high_minus_low_contrast",
        y,
        c_factor,
        full_fit,
        FULL_DESIGN,
        metadata=within_meta,
    )

    within_perm_rows, within_perm_summary = same_count_contrast_permutation(
        f"{label}_within_host_high_minus_low_contrast",
        within_contrast,
        y,
        c_factor,
        full_fit,
        FULL_DESIGN,
        mapped_rows,
    )
    within_contrast_with_perm = add_permutation_flags(within_contrast_audit, within_perm_summary)
    within_direction = direction_stats(
        f"{label}_within_host_high_minus_low_contrast",
        within_contrast,
        full_fit,
    )

    within_contrast_original = audit_vector(
        f"{label}_within_host_high_minus_low_contrast_original",
        within_contrast,
        "within_host_high_minus_low_contrast_original",
        y,
        c_factor,
        original_fit,
        ORIGINAL_DESIGN,
        metadata=within_meta,
    )

    all_audit_rows.extend(
        [
            global_low_audit,
            global_high_audit,
            global_contrast_original,
            global_contrast_with_perm,
            within_low_audit,
            within_high_audit,
            within_contrast_original,
            within_contrast_with_perm,
        ]
    )
    direction_rows.extend([global_direction, within_direction])
    permutation_summaries.extend([global_perm_summary, within_perm_summary])
    permutation_rows_all.extend(global_perm_rows)
    permutation_rows_all.extend(within_perm_rows)

    require_positive = bool(variable.get("primary_prediction", False))

    global_pass = pass_gate(global_contrast_with_perm, global_direction, require_positive_direction=require_positive)
    within_pass = pass_gate(within_contrast_with_perm, within_direction, require_positive_direction=require_positive)

    # For non-F160W controls, strong polarity can be either sign.
    control_global_strong = pass_gate(global_contrast_with_perm, global_direction, require_positive_direction=False)
    control_within_strong = pass_gate(within_contrast_with_perm, within_direction, require_positive_direction=False)

    combined_delta = float(
        global_contrast_with_perm.get("delta_chi2_score", 0.0)
        + within_contrast_with_perm.get("delta_chi2_score", 0.0)
    )

    result.update(
        {
            "status": "ok",
            "global_contrast": global_contrast_with_perm,
            "global_direction": global_direction,
            "global_contrast_original": global_contrast_original,
            "within_host_contrast": within_contrast_with_perm,
            "within_host_direction": within_direction,
            "within_host_contrast_original": within_contrast_original,
            "global_pass": global_pass,
            "within_host_pass": within_pass,
            "control_global_strong": control_global_strong,
            "control_within_host_strong": control_within_strong,
            "combined_delta_chi2": combined_delta,
            "global_delta_chi2": global_contrast_with_perm.get("delta_chi2_score", 0.0),
            "within_host_delta_chi2": within_contrast_with_perm.get("delta_chi2_score", 0.0),
            "global_signed_mean_residual_difference_high_minus_low": global_direction.get(
                "signed_mean_residual_difference_high_minus_low"
            ),
            "within_host_signed_mean_residual_difference_high_minus_low": within_direction.get(
                "signed_mean_residual_difference_high_minus_low"
            ),
            "audit_rows": all_audit_rows,
            "direction_rows": direction_rows,
            "permutation_summaries": permutation_summaries,
            "permutation_rows": permutation_rows_all,
        }
    )

    return result


def decide_status(variable_results):
    ok = [r for r in variable_results if r.get("status") == "ok"]
    ranked = sorted(ok, key=lambda r: -r.get("combined_delta_chi2", 0.0))

    f160w = next((r for r in ok if r["column"] == "table2_num_7"), None)
    f160w_rank = next(
        (idx + 1 for idx, row in enumerate(ranked) if row["column"] == "table2_num_7"),
        None,
    )

    control_strong = [
        r for r in ok
        if r["column"] != "table2_num_7"
        and (r.get("control_global_strong") or r.get("control_within_host_strong"))
    ]

    both_pass = bool(f160w and f160w.get("global_pass") and f160w.get("within_host_pass"))
    one_pass = bool(f160w and (f160w.get("global_pass") or f160w.get("within_host_pass")))

    next_best = None
    for row in ranked:
        if row["column"] != "table2_num_7":
            next_best = row
            break

    f160w_combined = f160w.get("combined_delta_chi2", 0.0) if f160w else 0.0
    next_combined = next_best.get("combined_delta_chi2", 0.0) if next_best else 0.0

    best_cases = {
        "ranked_variables": [
            {
                "rank": i + 1,
                "column": row["column"],
                "label": row["label"],
                "combined_delta_chi2": row["combined_delta_chi2"],
                "global_delta_chi2": row["global_delta_chi2"],
                "within_host_delta_chi2": row["within_host_delta_chi2"],
                "global_direction": row["global_signed_mean_residual_difference_high_minus_low"],
                "within_host_direction": row["within_host_signed_mean_residual_difference_high_minus_low"],
                "global_pass": row["global_pass"],
                "within_host_pass": row["within_host_pass"],
                "control_global_strong": row["control_global_strong"],
                "control_within_host_strong": row["control_within_host_strong"],
            }
            for i, row in enumerate(ranked)
        ],
        "f160w_result": f160w,
        "f160w_rank": f160w_rank,
        "next_best_non_f160w": next_best,
        "strong_control_variables": control_strong,
    }

    if both_pass and f160w_rank == 1 and (next_combined <= 0 or f160w_combined >= 1.25 * next_combined):
        return (
            "f160w_edge_pair_polarity_primary_and_specific",
            9,
            "F160W is the strongest signed edge-pair polarity variable and exceeds the next-best control by the specificity gate.",
            best_cases,
        )

    if both_pass and f160w_rank in [1, 2]:
        return (
            "f160w_edge_pair_polarity_primary_but_not_unique",
            8,
            "F160W passes global and within-host polarity and remains among the strongest variables, but other Table2 variables also carry polarity.",
            best_cases,
        )

    if one_pass and f160w_rank in [1, 2, 3]:
        return (
            "f160w_edge_pair_polarity_directional_partial",
            7,
            "F160W remains a strong polarity candidate, but it does not pass both global and within-host gates.",
            best_cases,
        )

    if f160w and f160w_rank and f160w_rank > 3:
        return (
            "f160w_not_primary_other_variables_stronger",
            6,
            "Other Table2 variables produce stronger edge-pair polarity than F160W, weakening F160W specificity.",
            best_cases,
        )

    return (
        "no_f160w_specific_edge_pair_polarity",
        6,
        "The v0.2 battery does not support F160W-specific edge-pair polarity.",
        best_cases,
    )


def make_plots(variable_summary_rows):
    if not variable_summary_rows:
        return

    rows = sorted(variable_summary_rows, key=lambda r: -r["combined_delta_chi2"])
    labels = [r["label"] for r in rows]

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(rows)), [r["combined_delta_chi2"] for r in rows])
    plt.xticks(np.arange(len(rows)), labels, rotation=45, ha="right")
    plt.ylabel("combined delta chi2")
    plt.title("TAIRID v0.2 combined edge-pair polarity strength")
    plt.tight_layout()
    plt.savefig(OUTDIR / "variable_combined_delta_chi2_v0_2.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    x = np.arange(len(rows))
    width = 0.35
    plt.bar(x - width / 2, [r["global_delta_chi2"] for r in rows], width, label="global")
    plt.bar(x + width / 2, [r["within_host_delta_chi2"] for r in rows], width, label="within-host")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("delta chi2")
    plt.title("Global vs within-host edge-pair polarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "variable_global_vs_within_delta_chi2_v0_2.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(rows)), [r["global_direction"] for r in rows])
    plt.axhline(0.0, linewidth=1)
    plt.xticks(np.arange(len(rows)), labels, rotation=45, ha="right")
    plt.ylabel("high-minus-low mean residual")
    plt.title("Global signed direction by variable")
    plt.tight_layout()
    plt.savefig(OUTDIR / "variable_global_signed_direction_v0_2.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(rows)), [r["within_host_direction"] for r in rows])
    plt.axhline(0.0, linewidth=1)
    plt.xticks(np.arange(len(rows)), labels, rotation=45, ha="right")
    plt.ylabel("within-host high-minus-low mean residual")
    plt.title("Within-host signed direction by variable")
    plt.tight_layout()
    plt.savefig(OUTDIR / "variable_within_host_signed_direction_v0_2.png", dpi=160)
    plt.close()


def main():
    print("TAIRID Boundary Prediction Battery v0.2 starting.")
    print("Boundary: multi-variable edge-pair polarity specificity audit only; not proof.")

    write_json(OUTDIR / "claims_v0_2.json", CLAIMS_V0_2)

    repair_summary = {}

    try:
        v16.OUTDIR = OUTDIR
        v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        b01.OUTDIR = OUTDIR
        b01.DOWNLOAD_DIR = DOWNLOAD_DIR
        b01.PERMUTATION_REPEATS = PERMUTATION_REPEATS

        ns, repair_summary = v16.load_v15_helpers()
        write_json(OUTDIR / "v15_import_repair_summary_v0_2.json", repair_summary)

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

        write_csv(OUTDIR / "download_ledger_v0_2.csv", ledger)
        write_json(OUTDIR / "download_attempts_v0_2.json", {"compact": downloads, "auxiliary": aux_results})

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

        write_json(OUTDIR / "parse_meta_v0_2.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v0_2.json", parse_errors)

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
                "test_name": "TAIRID Boundary Prediction Battery v0.2",
                "final_status": "boundary_polarity_battery_v0_2_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or Table2 retrieval before v0.2.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "boundary_polarity_battery_v0_2_summary.json", summary)
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

        write_csv(OUTDIR / "compact_row_map_v0_2.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v0_2.csv", cluster_rows)

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

        variable_results = []
        all_audit_rows = []
        all_direction_rows = []
        all_permutation_summaries = []

        for variable in VARIABLES:
            print(f"Analyzing {variable['label']} ({variable['column']})")
            try:
                result = analyze_variable(
                    variable,
                    mapped_rows,
                    y,
                    c_factor,
                    full_fit,
                    original_fit,
                )
            except Exception as exc:
                result = {
                    "column": variable["column"],
                    "label": variable["label"],
                    "plain": variable["plain"],
                    "tairid_role": variable["tairid_role"],
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }

            variable_results.append(result)

            if result.get("status") == "ok":
                all_audit_rows.extend(result["audit_rows"])
                all_direction_rows.extend(result["direction_rows"])
                all_permutation_summaries.extend(result["permutation_summaries"])

        variable_summary_rows = []

        for result in variable_results:
            if result.get("status") != "ok":
                variable_summary_rows.append(
                    {
                        "column": result["column"],
                        "label": result["label"],
                        "status": result["status"],
                        "error": result.get("error", ""),
                    }
                )
                continue

            variable_summary_rows.append(
                {
                    "column": result["column"],
                    "label": result["label"],
                    "plain": result["plain"],
                    "tairid_role": result["tairid_role"],
                    "primary_prediction": result["primary_prediction"],
                    "status": result["status"],
                    "combined_delta_chi2": result["combined_delta_chi2"],
                    "global_delta_chi2": result["global_delta_chi2"],
                    "within_host_delta_chi2": result["within_host_delta_chi2"],
                    "global_signed_mean_residual_difference_high_minus_low": result[
                        "global_signed_mean_residual_difference_high_minus_low"
                    ],
                    "within_host_signed_mean_residual_difference_high_minus_low": result[
                        "within_host_signed_mean_residual_difference_high_minus_low"
                    ],
                    "global_pass": result["global_pass"],
                    "within_host_pass": result["within_host_pass"],
                    "control_global_strong": result["control_global_strong"],
                    "control_within_host_strong": result["control_within_host_strong"],
                    "global_p_value": result["global_contrast"].get("p_value_chi2_one_dof"),
                    "within_host_p_value": result["within_host_contrast"].get("p_value_chi2_one_dof"),
                    "global_nondegenerate_ratio": result["global_contrast"].get("nondegenerate_ratio"),
                    "within_host_nondegenerate_ratio": result["within_host_contrast"].get("nondegenerate_ratio"),
                }
            )

        write_csv(OUTDIR / "variable_polarity_summary_v0_2.csv", variable_summary_rows)
        write_csv(OUTDIR / "edge_pair_polarity_candidate_audit_v0_2.csv", all_audit_rows)
        write_csv(OUTDIR / "edge_pair_direction_stats_v0_2.csv", all_direction_rows)
        write_json(OUTDIR / "permutation_summaries_v0_2.json", all_permutation_summaries)

        final_status, readiness_score, next_wall, best_cases = decide_status(variable_results)

        make_plots([row for row in variable_summary_rows if row.get("status") == "ok"])

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

        write_csv(OUTDIR / "design_fit_comparison_v0_2.csv", design_fit_rows)

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
            "test_name": "TAIRID Boundary Prediction Battery v0.2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Multi-variable signed edge-pair polarity specificity audit only. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_2": CLAIMS_V0_2,
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
            "variable_summary": variable_summary_rows,
            "variable_results": variable_results,
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_2_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_2_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v0_2.json"),
                "variable_summary_csv": str(OUTDIR / "variable_polarity_summary_v0_2.csv"),
                "candidate_audit_csv": str(OUTDIR / "edge_pair_polarity_candidate_audit_v0_2.csv"),
                "direction_stats_csv": str(OUTDIR / "edge_pair_direction_stats_v0_2.csv"),
                "permutation_summaries_json": str(OUTDIR / "permutation_summaries_v0_2.json"),
                "plots": [
                    str(OUTDIR / "variable_combined_delta_chi2_v0_2.png"),
                    str(OUTDIR / "variable_global_vs_within_delta_chi2_v0_2.png"),
                    str(OUTDIR / "variable_global_signed_direction_v0_2.png"),
                    str(OUTDIR / "variable_within_host_signed_direction_v0_2.png"),
                ],
            },
            "interpretation": {
                "what_supports_f160w_specificity": (
                    "F160W remains the strongest or near-strongest variable and passes both global and within-host polarity gates."
                ),
                "what_narrows_tairid": (
                    "F160W passes, but many other variables pass too. That means the result is broader measurement-gradient polarity, not F160W-specific."
                ),
                "what_fails_f160w_specificity": (
                    "F160W is weaker than several controls or fails polarity while other variables pass."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only checks whether the F160W signed polarity result is specific or generic."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_2_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_2_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.2\n\n")
            f.write("Boundary: multi-variable signed edge-pair polarity specificity audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Claims v0.2:\n")
            f.write(json.dumps(CLAIMS_V0_2, indent=2, default=json_default) + "\n\n")
            f.write("Variable summary:\n")
            f.write(json.dumps(variable_summary_rows, indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only tests whether F160W polarity is specific or generic across Table2 variables.\n")

        print("TAIRID Boundary Prediction Battery v0.2 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_2_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_2_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
