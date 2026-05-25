#!/usr/bin/env python3
"""
TAIRID ASD artifact-control response-shape v2.

Purpose:
The ASD eye-tracking response-shape replication v1 was positive but provisional.
Some high-performing feature families appeared to include technical/device/export/session
columns such as Unnamed_0, Tracking_Ratio, Port_Status, Export_Start/End_Trial_Time_ms,
RecordingTime_ms, index-like columns, and eye-position/pupil-position technical fields.

This cleanup test asks:
After removing obvious technical/artifact columns, does the TAIRID response-shape signal
still persist?

This script intentionally reuses the accepted v1 ASD parser:
    run_tairid_asd_eye_tracking_response_shape_replication_v1.py

Keep that v1 parser file in the repository beside this script.

Boundary:
This is not proof of TAIRID.
This is not clinical diagnosis.
This is not a cosmology result.
This is an artifact-control test of a second-neurotype response-shape replication.
"""

import csv
import json
import math
import importlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTDIR = Path("tairid_asd_artifact_control_response_shape_v2_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_DIR = OUTDIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CONTROL_LEVELS = [
    "original_rebuild",
    "artifact_clean",
    "strict_gaze_clean",
    "pupil_removed_clean",
    "gaze_aoi_only_clean",
]

TECHNICAL_EXCLUDE_TERMS = [
    "unnamed",
    "port_status",
    "tracking_ratio",
    "export_start",
    "export_end",
    "recordingtime",
    "recording_time",
    "timestamp",
    "systemtime",
    "system_time",
    "index_right",
    "index_left",
    "__index",
    "row_count",
    "n_rows",
    "session",
    "calibration",
    "validity",
    "valid_",
    "device",
    "camera",
]

POSITION_TECHNICAL_EXCLUDE_TERMS = [
    "eye_position",
    "pupil_position",
    "pupil_size",
]

PUPIL_EXCLUDE_TERMS = [
    "pupil",
]

STRICT_ALLOWED_TERMS = [
    "point_of_regard",
    "gaze_vector",
    "aoi_order",
    "fixation",
    "saccade",
    "sacc",
    "amplitude",
    "dispersion",
    "duration",
    "dwell",
    "velocity",
    "speed",
]

Gaze_AOI_ALLOWED_TERMS = [
    "point_of_regard",
    "gaze_vector",
    "aoi_order",
    "aoi",
    "fixation",
    "saccade",
    "sacc",
    "amplitude",
    "dispersion",
]


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path


def load_v1_module():
    try:
        base = importlib.import_module("run_tairid_asd_eye_tracking_response_shape_replication_v1")
    except Exception as exc:
        raise RuntimeError(
            "Could not import run_tairid_asd_eye_tracking_response_shape_replication_v1.py. "
            "Keep the accepted ASD v1 parser in the repository beside this cleanup script. "
            f"Import error: {exc}"
        )

    base.OUTDIR = OUTDIR
    base.DOWNLOAD_DIR = DOWNLOAD_DIR
    base.EXTRACT_DIR = EXTRACT_DIR

    OUTDIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    return base


def lower_name(col):
    return str(col).lower()


def is_key_col(col):
    return col in ["participant", "group", "context"]


def has_any(name, terms):
    name = lower_name(name)
    return any(term in name for term in terms)


def is_artifact_col(col):
    if is_key_col(col):
        return False

    name = lower_name(col)

    if has_any(name, TECHNICAL_EXCLUDE_TERMS):
        return True

    if has_any(name, POSITION_TECHNICAL_EXCLUDE_TERMS):
        return True

    return False


def keep_col_for_level(col, level):
    if is_key_col(col):
        return True

    name = lower_name(col)

    if level == "original_rebuild":
        return True

    if is_artifact_col(col):
        return False

    if level == "artifact_clean":
        # Keep biological pupil diameter if present, but remove pupil position/size above.
        return True

    if level == "pupil_removed_clean":
        if has_any(name, PUPIL_EXCLUDE_TERMS):
            return False
        return True

    if level == "strict_gaze_clean":
        if has_any(name, PUPIL_EXCLUDE_TERMS):
            return False
        return has_any(name, STRICT_ALLOWED_TERMS)

    if level == "gaze_aoi_only_clean":
        if has_any(name, PUPIL_EXCLUDE_TERMS):
            return False
        return has_any(name, Gaze_AOI_ALLOWED_TERMS)

    return True


def clean_participant_context_df(df, level):
    if df.empty:
        return df.copy(), {
            "control_level": level,
            "original_column_count": 0,
            "kept_column_count": 0,
            "removed_column_count": 0,
            "removed_columns": [],
            "kept_columns": [],
        }

    keep_cols = [c for c in df.columns if keep_col_for_level(c, level)]
    removed_cols = [c for c in df.columns if c not in keep_cols]

    cleaned = df[keep_cols].copy()

    meta = {
        "control_level": level,
        "original_column_count": int(len(df.columns)),
        "kept_column_count": int(len(keep_cols)),
        "removed_column_count": int(len(removed_cols)),
        "removed_columns": removed_cols,
        "kept_columns": keep_cols,
    }

    return cleaned, meta


def model_dict(rows):
    return {r["model_name"]: r for r in rows}


def perm_dict(rows):
    return {r["model_name"]: r for r in rows}


def best_shape_auc(model_rows):
    m = model_dict(model_rows)
    values = []

    for key in [
        "dynamic_context_mismatch_model",
        "viability_context_breach_model",
        "combined_context_response_model",
    ]:
        val = m.get(key, {}).get("auc_mean")
        if val is not None and np.isfinite(float(val)):
            values.append(float(val))

    return max(values) if values else None


def best_shape_p(permutation_rows):
    p = perm_dict(permutation_rows)
    values = []

    for key in [
        "dynamic_context_mismatch_model",
        "viability_context_breach_model",
        "combined_context_response_model",
    ]:
        val = p.get(key, {}).get("p_value_ge_observed")
        if val is not None and np.isfinite(float(val)):
            values.append(float(val))

    return min(values) if values else None


def run_level(base, participant_context_df, level):
    cleaned_df, clean_meta = clean_participant_context_df(participant_context_df, level)

    context_df, role_meta = base.build_context_shape_features(cleaned_df, family_filter="all")
    subject_df, subject_meta = base.build_subject_response_features(context_df)

    model_rows, permutation_rows = base.run_models(subject_df)
    feature_rows = base.add_bh_q(base.feature_tests(subject_df))

    family_rows = []
    family_meta = []

    for family in ["all", "timing", "gaze_spatial", "aoi_social", "general"]:
        fam_context_df, fam_role_meta = base.build_context_shape_features(cleaned_df, family_filter=family)
        fam_subject_df, fam_subject_meta = base.build_subject_response_features(fam_context_df)

        family_meta.append(
            {
                "control_level": level,
                "family": family,
                **fam_role_meta,
                **fam_subject_meta,
                "context_rows": int(len(fam_context_df)) if not fam_context_df.empty else 0,
                "subject_rows": int(len(fam_subject_df)) if not fam_subject_df.empty else 0,
            }
        )

        if fam_subject_df.empty:
            continue

        fam_model_rows, fam_perm_rows = base.run_models(fam_subject_df)

        for row in fam_model_rows:
            family_rows.append(
                {
                    "control_level": level,
                    "family": family,
                    **row,
                }
            )

    if not context_df.empty:
        context_path = OUTDIR / f"asd_artifact_{level}_context_shape_features.csv"
        context_df.to_csv(context_path, index=False)
    else:
        context_path = None

    if not subject_df.empty:
        subject_path = OUTDIR / f"asd_artifact_{level}_subject_response_features.csv"
        subject_df.to_csv(subject_path, index=False)
    else:
        subject_path = None

    model_path = write_csv(OUTDIR / f"asd_artifact_{level}_model_results.csv", model_rows)
    permutation_path = write_csv(OUTDIR / f"asd_artifact_{level}_permutation_results.csv", permutation_rows)
    family_path = write_csv(OUTDIR / f"asd_artifact_{level}_family_results.csv", family_rows)
    family_meta_path = write_csv(OUTDIR / f"asd_artifact_{level}_family_meta.csv", family_meta)
    feature_path = write_csv(OUTDIR / f"asd_artifact_{level}_feature_tests_bh_fdr.csv", feature_rows)

    m = model_dict(model_rows)
    p = perm_dict(permutation_rows)

    static_auc = m.get("static_level_model", {}).get("auc_mean")
    dynamic_auc = m.get("dynamic_context_mismatch_model", {}).get("auc_mean")
    viability_auc = m.get("viability_context_breach_model", {}).get("auc_mean")
    combined_auc = m.get("combined_context_response_model", {}).get("auc_mean")

    dynamic_p = p.get("dynamic_context_mismatch_model", {}).get("p_value_ge_observed")
    viability_p = p.get("viability_context_breach_model", {}).get("p_value_ge_observed")
    combined_p = p.get("combined_context_response_model", {}).get("p_value_ge_observed")

    group_counts = {}
    if not subject_df.empty:
        group_counts = {str(k): int(v) for k, v in Counter(subject_df["group"]).items()}

    level_summary = {
        "control_level": level,
        "clean_meta": clean_meta,
        "role_meta": role_meta,
        "subject_meta": subject_meta,
        "subject_rows": int(len(subject_df)) if not subject_df.empty else 0,
        "group_counts": group_counts,
        "static_auc": static_auc,
        "dynamic_auc": dynamic_auc,
        "viability_auc": viability_auc,
        "combined_auc": combined_auc,
        "best_shape_auc": best_shape_auc(model_rows),
        "dynamic_permutation_p": dynamic_p,
        "viability_permutation_p": viability_p,
        "combined_permutation_p": combined_p,
        "best_shape_permutation_p": best_shape_p(permutation_rows),
        "model_rows": model_rows,
        "permutation_rows": permutation_rows,
        "top_features": feature_rows[:15],
        "output_files": {
            "context_shape_features_csv": str(context_path) if context_path else None,
            "subject_response_features_csv": str(subject_path) if subject_path else None,
            "model_results_csv": str(model_path),
            "permutation_results_csv": str(permutation_path),
            "family_results_csv": str(family_path),
            "family_meta_csv": str(family_meta_path),
            "feature_tests_bh_fdr_csv": str(feature_path),
        },
    }

    return level_summary, family_rows, family_meta


def plot_control_summary(level_summaries):
    rows = []

    for s in level_summaries:
        rows.append(
            {
                "control_level": s["control_level"],
                "static_auc": s.get("static_auc"),
                "dynamic_auc": s.get("dynamic_auc"),
                "viability_auc": s.get("viability_auc"),
                "combined_auc": s.get("combined_auc"),
            }
        )

    if not rows:
        return None

    labels = [r["control_level"] for r in rows]
    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(13, 7))

    for i, key in enumerate(["static_auc", "dynamic_auc", "viability_auc", "combined_auc"]):
        vals = []
        for r in rows:
            v = r.get(key)
            vals.append(float(v) if v is not None and np.isfinite(float(v)) else 0.0)
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=key.replace("_auc", ""))

    plt.axhline(0.5, linewidth=1)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Repeated CV AUC")
    plt.title("ASD artifact-control response-shape v2: AUC by cleanup level")
    plt.legend()
    plt.tight_layout()

    path = OUTDIR / "asd_artifact_control_auc_by_cleanup_level.png"
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def plot_shape_minus_static(level_summaries):
    labels = []
    vals = []

    for s in level_summaries:
        static = s.get("static_auc")
        shape = s.get("best_shape_auc")

        if static is None or shape is None:
            continue

        labels.append(s["control_level"])
        vals.append(float(shape) - float(static))

    if not labels:
        return None

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Best shape AUC - static AUC")
    plt.title("ASD artifact-control response-shape v2: shape advantage after cleanup")
    plt.tight_layout()

    path = OUTDIR / "asd_artifact_control_shape_minus_static.png"
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def decide_status(level_summaries):
    artifact = next((s for s in level_summaries if s["control_level"] == "artifact_clean"), None)
    strict = next((s for s in level_summaries if s["control_level"] == "strict_gaze_clean"), None)
    pupil_removed = next((s for s in level_summaries if s["control_level"] == "pupil_removed_clean"), None)

    if artifact is None:
        return (
            "artifact_control_failed_to_run",
            5,
            "Inspect output files and parser dependency.",
        )

    static_auc = artifact.get("static_auc")
    best_shape = artifact.get("best_shape_auc")
    best_p = artifact.get("best_shape_permutation_p")

    strict_shape = strict.get("best_shape_auc") if strict else None
    pupil_shape = pupil_removed.get("best_shape_auc") if pupil_removed else None

    shape_beats_static = (
        static_auc is not None
        and best_shape is not None
        and float(best_shape) > float(static_auc) + 0.05
    )

    shape_perm_locked = best_p is not None and float(best_p) <= 0.05

    clean_shape_present = best_shape is not None and float(best_shape) >= 0.60
    strict_or_pupil_present = (
        (strict_shape is not None and float(strict_shape) >= 0.58)
        or (pupil_shape is not None and float(pupil_shape) >= 0.58)
    )

    if shape_beats_static and shape_perm_locked and strict_or_pupil_present:
        return (
            "artifact_control_supports_clean_asd_response_shape",
            8,
            "Use ASD as a provisional second-neurotype replication and write the cross-neurotype axis map.",
        )

    if clean_shape_present and shape_perm_locked:
        return (
            "artifact_clean_shape_persists_but_static_or_strict_controls_limit_claim",
            7,
            "Treat ASD as supportive but not locked; compare cleaned feature columns before axis-map writeup.",
        )

    if clean_shape_present:
        return (
            "artifact_clean_shape_directional_not_permutation_locked",
            7,
            "Signal persists directionally after cleanup but needs stronger validation.",
        )

    return (
        "artifact_control_weakens_asd_replication",
        6,
        "Do not use ASD as locked replication yet; refine parser or choose another ASD dataset.",
    )


def main():
    print("")
    print("TAIRID ASD artifact-control response-shape v2 starting.")
    print("Boundary: artifact-control test only; not proof or diagnosis.")
    print("")

    base = load_v1_module()

    article, downloads = base.download_figshare_article()
    extraction = base.extract_archives(downloads)

    write_csv(OUTDIR / "asd_artifact_download_ledger.csv", downloads)
    write_csv(OUTDIR / "asd_artifact_extraction_ledger.csv", extraction)

    participant_map, participant_sources = base.load_participant_metadata()
    participant_map_rows = [
        {"participant_key": k, "group": v}
        for k, v in sorted(participant_map.items())
    ]

    write_csv(OUTDIR / "asd_artifact_participant_sources.csv", participant_sources)
    write_csv(OUTDIR / "asd_artifact_participant_map.csv", participant_map_rows)

    table_inventory, feature_rows = base.parse_experiment_tables(participant_map)

    inventory_path = write_csv(OUTDIR / "asd_artifact_table_inventory.csv", table_inventory)
    feature_rows_path = write_csv(OUTDIR / "asd_artifact_context_feature_rows.csv", feature_rows)

    participant_context_df = base.aggregate_participant_context(feature_rows)

    if not participant_context_df.empty:
        participant_context_path = OUTDIR / "asd_artifact_participant_context_features_unfiltered.csv"
        participant_context_df.to_csv(participant_context_path, index=False)
    else:
        participant_context_path = None

    level_summaries = []
    all_family_rows = []
    all_family_meta = []

    for level in CONTROL_LEVELS:
        print(f"Running cleanup level: {level}")
        level_summary, family_rows, family_meta = run_level(base, participant_context_df, level)
        level_summaries.append(level_summary)
        all_family_rows.extend(family_rows)
        all_family_meta.extend(family_meta)

    cleanup_summary_rows = []
    for s in level_summaries:
        cleanup_summary_rows.append(
            {
                "control_level": s["control_level"],
                "subject_rows": s.get("subject_rows"),
                "group_counts_json": json.dumps(s.get("group_counts")),
                "kept_column_count": s.get("clean_meta", {}).get("kept_column_count"),
                "removed_column_count": s.get("clean_meta", {}).get("removed_column_count"),
                "static_auc": s.get("static_auc"),
                "dynamic_auc": s.get("dynamic_auc"),
                "viability_auc": s.get("viability_auc"),
                "combined_auc": s.get("combined_auc"),
                "best_shape_auc": s.get("best_shape_auc"),
                "best_shape_permutation_p": s.get("best_shape_permutation_p"),
                "dynamic_permutation_p": s.get("dynamic_permutation_p"),
                "viability_permutation_p": s.get("viability_permutation_p"),
                "combined_permutation_p": s.get("combined_permutation_p"),
            }
        )

    cleanup_summary_path = write_csv(
        OUTDIR / "asd_artifact_control_cleanup_summary.csv",
        cleanup_summary_rows,
    )

    all_family_path = write_csv(
        OUTDIR / "asd_artifact_control_all_family_results.csv",
        all_family_rows,
    )

    all_family_meta_path = write_csv(
        OUTDIR / "asd_artifact_control_all_family_meta.csv",
        all_family_meta,
    )

    plots = []

    p = plot_control_summary(level_summaries)
    if p:
        plots.append(p)

    p = plot_shape_minus_static(level_summaries)
    if p:
        plots.append(p)

    final_status, readiness_score, next_wall = decide_status(level_summaries)

    artifact = next((s for s in level_summaries if s["control_level"] == "artifact_clean"), None)

    summary = {
        "test_name": "TAIRID ASD artifact-control response-shape v2",
        "boundary": (
            "Artifact-control test of second-neurotype response-shape replication only. "
            "Not clinical diagnosis, not proof of TAIRID, and not a cosmology result."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "dataset": {
            "figshare_article_id": base.FIGSHARE_ARTICLE_ID,
            "figshare_api_url": base.FIGSHARE_API_URL,
            "article_title": article.get("title"),
            "doi": article.get("doi"),
        },
        "parser_counts": {
            "downloads_count": len(downloads),
            "extraction_count": len(extraction),
            "participant_source_count": len(participant_sources),
            "participant_map_count": len(participant_map),
            "table_inventory_count": len(table_inventory),
            "context_feature_row_count": len(feature_rows),
            "participant_context_rows": int(len(participant_context_df)) if not participant_context_df.empty else 0,
            "participant_context_columns": int(len(participant_context_df.columns)) if not participant_context_df.empty else 0,
        },
        "cleanup_levels": level_summaries,
        "artifact_clean_key_result": artifact,
        "output_files": {
            "table_inventory_csv": str(inventory_path),
            "context_feature_rows_csv": str(feature_rows_path),
            "participant_context_unfiltered_csv": str(participant_context_path) if participant_context_path else None,
            "cleanup_summary_csv": str(cleanup_summary_path),
            "all_family_results_csv": str(all_family_path),
            "all_family_meta_csv": str(all_family_meta_path),
            "plots": plots,
        },
        "interpretation": {
            "why_this_test_exists": (
                "ASD v1 was positive but had technical columns that could inflate static or family-specific results."
            ),
            "what_supports_TAIRID_here": (
                "After obvious technical columns are removed, dynamic mismatch or viability breach still beats static level "
                "and remains stronger than permutation expectation."
            ),
            "what_weakens_the_replication": (
                "The response-shape signal collapses after removing device/export/session columns, or static artifact-like "
                "features remain stronger than dynamic mismatch."
            ),
            "truth_boundary": (
                "A clean positive result supports ASD as a provisional second-neurotype axis replication; it cannot prove "
                "TAIRID, diagnose ASD, or prove cosmology."
            ),
        },
    }

    summary_path = OUTDIR / "asd_artifact_control_response_shape_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(OUTDIR / "asd_artifact_control_response_shape_v2_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID ASD artifact-control response-shape v2\n\n")
        f.write("Boundary: artifact-control test only. Not diagnosis. Not proof.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Why this test exists:\n")
        f.write("- ASD v1 was positive but possibly contaminated by device/export/session columns.\n")
        f.write("- This pass removes obvious artifact columns and reruns the response-shape models.\n")
        f.write("- The result tells us whether ASD remains useful as a second neurotype axis replication.\n\n")

        f.write("Cleanup summary:\n")
        f.write(json.dumps(cleanup_summary_rows, indent=2) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This can support a TAIRID response-shape translation.\n")
        f.write("- It cannot prove TAIRID.\n")
        f.write("- It cannot diagnose ASD.\n")
        f.write("- It cannot prove any cosmology claim.\n")

    print("")
    print("TAIRID ASD artifact-control response-shape v2 complete.")
    print("Created:")
    print("  tairid_asd_artifact_control_response_shape_v2_outputs/asd_artifact_control_response_shape_v2_summary.json")
    print("  tairid_asd_artifact_control_response_shape_v2_outputs/asd_artifact_control_response_shape_v2_summary.txt")
    print("  tairid_asd_artifact_control_response_shape_v2_outputs/asd_artifact_control_cleanup_summary.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is not clinical diagnosis.")
    print("  This is an artifact-control test.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
