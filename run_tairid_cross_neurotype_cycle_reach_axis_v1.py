#!/usr/bin/env python3
"""
TAIRID cross-neurotype cycle/reach axis test v1.

Purpose:
ETDD70 and ASD eye-tracking tests supported a TAIRID response-shape translation:
T pacing, I constraint, M mismatch, W viability window, B breach, slope, and curvature.

This test adds two explicit derived axes:

1. Cycling / recurrence:
   How often the system reverses, loops, re-enters, breaches/re-enters, or shifts phase
   across task demand or context.

2. Reach / coherent propagation span:
   How far the system can extend across tasks/contexts before mismatch leaves the
   viability window.

Why this is the next test:
Cosmology taught that simple offset-shaped TAIRID translations get absorbed.
Neurotype datasets let us test relative response geometry more directly.
This pass asks whether cycling and reach add stable explanatory structure across
two neurotype datasets rather than being hidden inside mismatch alone.

Datasets:
- ETDD70 dyslexia eye-tracking, via the accepted ETDD70 viability-window v2 parser.
- ASD eye-tracking, via the accepted ASD response-shape v1 parser plus artifact cleanup.

Required repository files:
- run_tairid_etdd70_viability_window_v2.py
- run_tairid_asd_eye_tracking_response_shape_replication_v1.py

Boundary:
This is not proof of TAIRID.
This is not diagnosis.
This is not a cosmology result.
This is a cross-neurotype operational axis test.
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
from scipy import stats


OUTDIR = Path("tairid_cross_neurotype_cycle_reach_axis_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

ETDD70_DIR = OUTDIR / "etdd70_rebuild"
ETDD70_DOWNLOAD_DIR = ETDD70_DIR / "downloaded"
ETDD70_EXTRACT_DIR = ETDD70_DIR / "extracted"

ASD_DIR = OUTDIR / "asd_rebuild"
ASD_DOWNLOAD_DIR = ASD_DIR / "downloaded"
ASD_EXTRACT_DIR = ASD_DIR / "extracted"

for p in [
    ETDD70_DIR,
    ETDD70_DOWNLOAD_DIR,
    ETDD70_EXTRACT_DIR,
    ASD_DIR,
    ASD_DOWNLOAD_DIR,
    ASD_EXTRACT_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
CV_REPEATS = 80
PERMUTATIONS = 150
PERM_REPEATS = 20
RIDGE = 1.0e-3

TOLERANCE_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0]

ASD_TECHNICAL_EXCLUDE_TERMS = [
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
    "eye_position",
    "pupil_position",
    "pupil_size",
]

np.random.seed(RANDOM_SEED)


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


def load_etdd70_module():
    try:
        mod = importlib.import_module("run_tairid_etdd70_viability_window_v2")
    except Exception as exc:
        raise RuntimeError(
            "Could not import run_tairid_etdd70_viability_window_v2.py. "
            "Keep the accepted ETDD70 v2 parser in the repo beside this script. "
            f"Import error: {exc}"
        )

    mod.OUTDIR = ETDD70_DIR
    mod.DOWNLOAD_DIR = ETDD70_DOWNLOAD_DIR
    mod.EXTRACT_DIR = ETDD70_EXTRACT_DIR
    return mod


def load_asd_module():
    try:
        mod = importlib.import_module("run_tairid_asd_eye_tracking_response_shape_replication_v1")
    except Exception as exc:
        raise RuntimeError(
            "Could not import run_tairid_asd_eye_tracking_response_shape_replication_v1.py. "
            "Keep the accepted ASD v1 parser in the repo beside this script. "
            f"Import error: {exc}"
        )

    mod.OUTDIR = ASD_DIR
    mod.DOWNLOAD_DIR = ASD_DOWNLOAD_DIR
    mod.EXTRACT_DIR = ASD_EXTRACT_DIR
    return mod


def has_any(text, terms):
    s = str(text).lower()
    return any(term in s for term in terms)


def clean_asd_participant_context_df(df):
    if df.empty:
        return df.copy(), {
            "original_columns": 0,
            "kept_columns": 0,
            "removed_columns": 0,
            "removed_column_names": [],
        }

    key_cols = {"participant", "group", "context"}

    keep = []
    removed = []

    for col in df.columns:
        if col in key_cols:
            keep.append(col)
        elif has_any(col, ASD_TECHNICAL_EXCLUDE_TERMS):
            removed.append(col)
        else:
            keep.append(col)

    return df[keep].copy(), {
        "original_columns": int(len(df.columns)),
        "kept_columns": int(len(keep)),
        "removed_columns": int(len(removed)),
        "removed_column_names": removed,
    }


def nonzero_signs(vals):
    vals = np.asarray(vals, dtype=float)
    signs = np.sign(vals)
    signs = signs[signs != 0]
    return signs


def sign_change_count(vals):
    signs = nonzero_signs(vals)
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def longest_true_run(flags):
    flags = list(bool(x) for x in flags)
    best = 0
    current = 0

    for x in flags:
        if x:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return int(best)


def safe_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return np.nan


def safe_curvature(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    try:
        return float(np.polyfit(x, y, 2)[0])
    except Exception:
        return np.nan


def add_cycle_reach_features_from_series(base, x, T, I, M, C, group_key):
    x = np.asarray(x, dtype=float)
    T = np.asarray(T, dtype=float)
    I = np.asarray(I, dtype=float)
    M = np.asarray(M, dtype=float)
    C = np.asarray(C, dtype=float)

    dT = np.diff(T)
    dI = np.diff(I)
    dM = np.diff(M)
    dC = np.diff(C)

    n = len(M)

    base["Axis_context_count"] = int(n)
    base["Cyc_M_derivative_turns"] = sign_change_count(dM)
    base["Cyc_M_derivative_turn_rate"] = float(sign_change_count(dM) / max(len(dM) - 1, 1))
    base["Cyc_T_derivative_turns"] = sign_change_count(dT)
    base["Cyc_I_derivative_turns"] = sign_change_count(dI)
    base["Cyc_C_derivative_turns"] = sign_change_count(dC)

    if len(dT) and len(dI):
        opposition = np.sign(dT) * np.sign(dI) < 0
        base["Cyc_TI_opposition_fraction"] = float(np.mean(opposition))
    else:
        base["Cyc_TI_opposition_fraction"] = np.nan

    M_med = float(np.median(M)) if len(M) else np.nan
    centered = M - M_med
    base["Cyc_M_center_crossings"] = sign_change_count(centered)
    base["Cyc_M_center_crossing_rate"] = float(sign_change_count(centered) / max(n - 1, 1))

    base["Reach_M_span"] = float(np.max(M) - np.min(M)) if len(M) else np.nan
    base["Reach_C_span"] = float(np.max(C) - np.min(C)) if len(C) else np.nan
    base["Reach_T_span"] = float(np.max(T) - np.min(T)) if len(T) else np.nan
    base["Reach_I_span"] = float(np.max(I) - np.min(I)) if len(I) else np.nan
    base["Reach_TI_vector_span"] = float(
        math.sqrt(base["Reach_T_span"] ** 2 + base["Reach_I_span"] ** 2)
    ) if np.isfinite(base["Reach_T_span"]) and np.isfinite(base["Reach_I_span"]) else np.nan

    base["Axis_M_slope_recomputed"] = safe_slope(x, M)
    base["Axis_M_curvature_recomputed"] = safe_curvature(x, M)
    base["Axis_T_slope_recomputed"] = safe_slope(x, T)
    base["Axis_I_slope_recomputed"] = safe_slope(x, I)
    base["Axis_C_slope_recomputed"] = safe_slope(x, C)

    if np.isfinite(base["Axis_T_slope_recomputed"]) and np.isfinite(base["Axis_I_slope_recomputed"]):
        base["Axis_TI_slope_gap_recomputed"] = float(
            abs(base["Axis_T_slope_recomputed"] - base["Axis_I_slope_recomputed"])
        )
    else:
        base["Axis_TI_slope_gap_recomputed"] = np.nan

    for mult in TOLERANCE_MULTIPLIERS:
        suffix = f"tol{str(mult).replace('.', 'p')}"

        if group_key == "etdd70":
            W = float(M[0] + mult * base.get("control_task1_mismatch_sd_used", 0.0))
        else:
            W = float(np.percentile(M, 25) + mult * base.get("td_baseline_sd_used", 0.0))

        B = np.maximum(0.0, M - W)
        breached = B > 0.0
        stable = ~breached

        base[f"W_axis_{suffix}"] = W
        base[f"B_axis_total_{suffix}"] = float(np.sum(B))
        base[f"B_axis_mean_{suffix}"] = float(np.mean(B))
        base[f"B_axis_max_{suffix}"] = float(np.max(B))
        base[f"B_axis_fraction_{suffix}"] = float(np.mean(breached))
        base[f"B_axis_count_{suffix}"] = int(np.sum(breached))

        base[f"Cyc_breach_transition_count_{suffix}"] = int(
            np.sum(np.abs(np.diff(breached.astype(int))))
        ) if len(breached) > 1 else 0

        reentry = 0
        if len(breached) > 1:
            for a, b in zip(breached[:-1], breached[1:]):
                if a and not b:
                    reentry += 1

        base[f"Cyc_breach_reentry_count_{suffix}"] = int(reentry)
        base[f"Reach_stable_fraction_{suffix}"] = float(np.mean(stable))
        base[f"Reach_breach_fraction_{suffix}"] = float(np.mean(breached))
        base[f"Reach_longest_stable_run_{suffix}"] = longest_true_run(stable)
        base[f"Reach_longest_breach_run_{suffix}"] = longest_true_run(breached)

        if group_key == "etdd70":
            stable_x = x[stable]
            base[f"Reach_highest_stable_task_{suffix}"] = float(np.max(stable_x)) if len(stable_x) else 0.0
        else:
            base[f"Reach_stable_context_count_{suffix}"] = int(np.sum(stable))

    return base


def build_etdd70_cycle_reach_features(et):
    record, downloads = et.download_zenodo_record()
    extraction = et.extract_archives(downloads)

    label_map, label_sources = et.load_label_map()
    metrics_inventory, metric_feature_rows = et.parse_metrics_files(label_map)
    subject_task_df = et.aggregate_subject_task(metric_feature_rows)
    task_df, ti_meta = et.build_tairid_task_features(subject_task_df)
    subject_vw_df, vw_meta = et.build_subject_viability_features(task_df)

    rows = []

    for subject, sub in task_df.groupby("subject"):
        sub = sub.sort_values("task_rank")

        if set(sub["task_rank"].astype(int).values) != {1, 2, 3}:
            continue

        group_vals = sub["group"].dropna().unique()
        group = group_vals[0] if len(group_vals) else None

        if group not in ["control", "dyslexic"]:
            continue

        subject_row = subject_vw_df[subject_vw_df["subject"].astype(str) == str(subject)]
        if subject_row.empty:
            continue

        base = subject_row.iloc[0].to_dict()
        base["dataset"] = "ETDD70"
        base["label_name"] = group
        base["label"] = 1 if group == "dyslexic" else 0

        x = sub["task_rank"].astype(float).values
        T = sub["T_pacing_proxy"].astype(float).values
        I = sub["I_constraint_proxy"].astype(float).values
        M = sub["M_mismatch_abs"].astype(float).values
        C = sub["collapse_load_proxy"].astype(float).values

        base = add_cycle_reach_features_from_series(base, x, T, I, M, C, "etdd70")
        rows.append(base)

    meta = {
        "dataset": "ETDD70",
        "record_title": record.get("metadata", {}).get("title"),
        "downloads_count": len(downloads),
        "extraction_count": len(extraction),
        "label_source_count": len(label_sources),
        "metrics_inventory_count": len(metrics_inventory),
        "metric_feature_rows": len(metric_feature_rows),
        "subject_task_rows": int(len(subject_task_df)) if not subject_task_df.empty else 0,
        "task_shape_rows": int(len(task_df)) if not task_df.empty else 0,
        "subject_vw_rows": int(len(subject_vw_df)) if not subject_vw_df.empty else 0,
        "cycle_reach_rows": len(rows),
        "ti_meta": ti_meta,
        "vw_meta": vw_meta,
    }

    return pd.DataFrame(rows), meta, {
        "downloads": downloads,
        "label_sources": label_sources,
        "metrics_inventory": metrics_inventory,
        "subject_task_df": subject_task_df,
        "task_df": task_df,
        "subject_vw_df": subject_vw_df,
    }


def build_asd_cycle_reach_features(asd):
    article, downloads = asd.download_figshare_article()
    extraction = asd.extract_archives(downloads)

    participant_map, participant_sources = asd.load_participant_metadata()
    table_inventory, feature_rows = asd.parse_experiment_tables(participant_map)
    participant_context_df = asd.aggregate_participant_context(feature_rows)

    clean_df, clean_meta = clean_asd_participant_context_df(participant_context_df)

    context_df, role_meta = asd.build_context_shape_features(clean_df, family_filter="all")
    subject_df, subject_meta = asd.build_subject_response_features(context_df)

    rows = []

    for participant, sub in context_df.groupby("participant"):
        sub = sub.sort_values("context")

        group_vals = sub["group"].dropna().unique()
        group = group_vals[0] if len(group_vals) else None

        if group not in ["asd", "td"]:
            continue

        subject_row = subject_df[subject_df["participant"].astype(str) == str(participant)]
        if subject_row.empty:
            continue

        base = subject_row.iloc[0].to_dict()
        base["dataset"] = "ASD"
        base["label_name"] = group
        base["label"] = 1 if group == "asd" else 0

        x = np.arange(len(sub), dtype=float)
        T = sub["T_pacing_proxy"].astype(float).values
        I = sub["I_constraint_proxy"].astype(float).values
        M = sub["M_mismatch_abs"].astype(float).values
        C = sub["collapse_load_proxy"].astype(float).values

        base = add_cycle_reach_features_from_series(base, x, T, I, M, C, "asd")
        rows.append(base)

    meta = {
        "dataset": "ASD",
        "article_title": article.get("title"),
        "doi": article.get("doi"),
        "downloads_count": len(downloads),
        "extraction_count": len(extraction),
        "participant_source_count": len(participant_sources),
        "participant_map_count": len(participant_map),
        "table_inventory_count": len(table_inventory),
        "context_feature_rows": len(feature_rows),
        "participant_context_rows": int(len(participant_context_df)) if not participant_context_df.empty else 0,
        "participant_context_clean_rows": int(len(clean_df)) if not clean_df.empty else 0,
        "context_shape_rows": int(len(context_df)) if not context_df.empty else 0,
        "subject_response_rows": int(len(subject_df)) if not subject_df.empty else 0,
        "cycle_reach_rows": len(rows),
        "artifact_clean_meta": clean_meta,
        "role_meta": role_meta,
        "subject_meta": subject_meta,
    }

    return pd.DataFrame(rows), meta, {
        "downloads": downloads,
        "participant_sources": participant_sources,
        "table_inventory": table_inventory,
        "participant_context_df": participant_context_df,
        "clean_participant_context_df": clean_df,
        "context_df": context_df,
        "subject_df": subject_df,
    }


def auc_score(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]

    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return np.nan

    pos = scores[labels == 1]
    neg = scores[labels == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    ranks = stats.rankdata(np.concatenate([pos, neg]))
    rpos = np.sum(ranks[:len(pos)])

    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stratified_folds(y, k, rng):
    y = np.asarray(y, dtype=int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    parts0 = np.array_split(idx0, k)
    parts1 = np.array_split(idx1, k)

    folds = []
    for i in range(k):
        test = np.concatenate([parts0[i], parts1[i]])
        rng.shuffle(test)
        folds.append(test)

    return folds


def lda_scores(X_train, y_train, X_test):
    y_train = np.asarray(y_train, dtype=int)

    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]

    if len(X0) < 2 or len(X1) < 2:
        return np.zeros(len(X_test), dtype=float)

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    if X_train.shape[1] == 1:
        var = float(np.var(X_train[:, 0]) + RIDGE)
        w = np.asarray([(mu1[0] - mu0[0]) / var])
    else:
        cov = np.cov(X_train.T, bias=False)
        cov = np.atleast_2d(cov)
        cov = cov + np.eye(cov.shape[0]) * RIDGE
        w = np.linalg.pinv(cov, rcond=1.0e-8) @ (mu1 - mu0)

    b = -0.5 * float((mu1 + mu0) @ w)
    return X_test @ w + b


def repeated_cv(df, feature_cols, repeats=CV_REPEATS, y_override=None):
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return {"status": "no_features", "n": 0, "feature_cols": []}

    data = df[feature_cols + ["label"]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 12 or data["label"].nunique() < 2:
        return {
            "status": "not_enough_data",
            "n": int(len(data)),
            "feature_cols": feature_cols,
        }

    X_raw = data[feature_cols].astype(float).values
    y = data["label"].astype(int).values

    if y_override is not None:
        y = np.asarray(y_override, dtype=int)

    if len(np.unique(y)) < 2:
        return {"status": "one_class", "n": int(len(y)), "feature_cols": feature_cols}

    counts = np.bincount(y)
    k = min(5, int(np.min(counts[counts > 0])))
    k = max(2, k)

    rng = np.random.default_rng(RANDOM_SEED)
    aucs = []

    for _ in range(repeats):
        folds = stratified_folds(y, k, rng)
        preds = np.zeros(len(y), dtype=float)

        for test in folds:
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test] = False

            X_train = X_raw[train_mask]
            X_test = X_raw[test]
            y_train = y[train_mask]

            mu = X_train.mean(axis=0)
            sd = X_train.std(axis=0)
            sd[sd <= 1.0e-12] = 1.0

            X_train_z = (X_train - mu) / sd
            X_test_z = (X_test - mu) / sd

            preds[test] = lda_scores(X_train_z, y_train, X_test_z)

        auc = auc_score(preds, y)
        if np.isfinite(auc):
            aucs.append(float(auc))

    if not aucs:
        return {"status": "auc_failed", "n": int(len(y)), "feature_cols": feature_cols}

    return {
        "status": "ok",
        "n": int(len(y)),
        "repeats": repeats,
        "feature_cols": feature_cols,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_min": float(np.min(aucs)),
        "auc_max": float(np.max(aucs)),
    }


def permutation_test(df, feature_cols, observed_auc):
    feature_cols = [c for c in feature_cols if c in df.columns]
    data = df[feature_cols + ["label"]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 12 or data["label"].nunique() < 2 or observed_auc is None:
        return {
            "status": "not_enough_data",
            "n_perm": 0,
            "p_value_ge_observed": None,
        }

    y = data["label"].astype(int).values
    rng = np.random.default_rng(RANDOM_SEED + 99)
    perm_aucs = []

    for _ in range(PERMUTATIONS):
        yp = y.copy()
        rng.shuffle(yp)

        res = repeated_cv(data, feature_cols, repeats=PERM_REPEATS, y_override=yp)
        auc = res.get("auc_mean")
        if auc is not None and np.isfinite(float(auc)):
            perm_aucs.append(float(auc))

    if not perm_aucs:
        return {
            "status": "no_valid_permutations",
            "n_perm": 0,
            "p_value_ge_observed": None,
        }

    perm_aucs = np.asarray(perm_aucs, dtype=float)
    p = float((1.0 + np.sum(perm_aucs >= observed_auc)) / (1.0 + len(perm_aucs)))

    return {
        "status": "ok",
        "n_perm": int(len(perm_aucs)),
        "p_value_ge_observed": p,
        "perm_auc_mean": float(np.mean(perm_aucs)),
        "perm_auc_std": float(np.std(perm_aucs)),
        "perm_auc_95": float(np.percentile(perm_aucs, 95)),
        "perm_auc_99": float(np.percentile(perm_aucs, 99)),
    }


def feature_sets(df):
    static = [
        "M_mean",
        "M_max",
        "collapse_load_mean",
        "collapse_load_max",
    ]

    response = [
        "M_range",
        "M_slope",
        "M_curvature",
        "T_I_slope_gap",
        "collapse_load_slope",
        "M_context_slope",
        "M_context_curvature",
        "T_I_context_slope_gap",
        "collapse_load_context_slope",
        "Axis_M_slope_recomputed",
        "Axis_M_curvature_recomputed",
        "Axis_TI_slope_gap_recomputed",
    ]

    viability = []
    for mult in TOLERANCE_MULTIPLIERS:
        suffix = f"tol{str(mult).replace('.', 'p')}"
        viability.extend(
            [
                f"B_total_{suffix}",
                f"B_mean_harder_{suffix}",
                f"B_max_{suffix}",
                f"B_slope_{suffix}",
                f"B_curvature_{suffix}",
                f"B_fraction_{suffix}",
                f"B_axis_total_{suffix}",
                f"B_axis_mean_{suffix}",
                f"B_axis_max_{suffix}",
                f"B_axis_fraction_{suffix}",
            ]
        )

    cycling = [
        "Cyc_M_derivative_turns",
        "Cyc_M_derivative_turn_rate",
        "Cyc_T_derivative_turns",
        "Cyc_I_derivative_turns",
        "Cyc_C_derivative_turns",
        "Cyc_TI_opposition_fraction",
        "Cyc_M_center_crossings",
        "Cyc_M_center_crossing_rate",
    ]

    reach = [
        "Reach_M_span",
        "Reach_C_span",
        "Reach_T_span",
        "Reach_I_span",
        "Reach_TI_vector_span",
    ]

    for mult in TOLERANCE_MULTIPLIERS:
        suffix = f"tol{str(mult).replace('.', 'p')}"
        cycling.extend(
            [
                f"Cyc_breach_transition_count_{suffix}",
                f"Cyc_breach_reentry_count_{suffix}",
            ]
        )
        reach.extend(
            [
                f"Reach_stable_fraction_{suffix}",
                f"Reach_breach_fraction_{suffix}",
                f"Reach_longest_stable_run_{suffix}",
                f"Reach_longest_breach_run_{suffix}",
                f"Reach_highest_stable_task_{suffix}",
                f"Reach_stable_context_count_{suffix}",
            ]
        )

    # Keep only columns present in this dataset.
    return {
        "static_level_model": [c for c in static if c in df.columns],
        "response_dynamic_model": [c for c in response if c in df.columns],
        "viability_window_model": [c for c in viability if c in df.columns],
        "cycling_model": [c for c in cycling if c in df.columns],
        "reach_model": [c for c in reach if c in df.columns],
        "cycling_reach_model": [c for c in cycling + reach if c in df.columns],
        "combined_axis_model": [c for c in static + response + viability + cycling + reach if c in df.columns],
    }


def run_model_suite(df, dataset_name):
    rows = []
    perms = []

    fsets = feature_sets(df)

    for name, cols in fsets.items():
        res = repeated_cv(df, cols)
        auc = res.get("auc_mean")

        rows.append(
            {
                "dataset": dataset_name,
                "model_name": name,
                **{k: v for k, v in res.items() if k != "feature_cols"},
                "feature_cols": " | ".join(res.get("feature_cols", [])),
            }
        )

        perm = permutation_test(df, cols, observed_auc=auc)
        perms.append(
            {
                "dataset": dataset_name,
                "model_name": name,
                "observed_auc_mean": auc,
                "permutation_status": perm.get("status"),
                "n_perm": perm.get("n_perm"),
                "p_value_ge_observed": perm.get("p_value_ge_observed"),
                "perm_auc_mean": perm.get("perm_auc_mean"),
                "perm_auc_std": perm.get("perm_auc_std"),
                "perm_auc_95": perm.get("perm_auc_95"),
                "perm_auc_99": perm.get("perm_auc_99"),
            }
        )

    return rows, perms


def feature_tests(df, dataset_name):
    rows = []

    numeric_cols = [
        c for c in df.columns
        if c not in ["dataset", "label", "label_name", "group", "subject", "participant"]
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notnull().sum() >= 8
    ]

    for c in numeric_cols:
        pos = df[df["label"] == 1][c].dropna().astype(float).values
        neg = df[df["label"] == 0][c].dropna().astype(float).values

        if len(pos) < 3 or len(neg) < 3:
            continue

        tstat, pval = stats.ttest_ind(pos, neg, equal_var=False, nan_policy="omit")

        pooled = math.sqrt(
            ((len(pos) - 1) * np.var(pos, ddof=1) + (len(neg) - 1) * np.var(neg, ddof=1))
            / max(len(pos) + len(neg) - 2, 1)
        )
        d = 0.0 if pooled <= 1.0e-12 else float((np.mean(pos) - np.mean(neg)) / pooled)

        scores = df[c].astype(float).values
        labels = df["label"].astype(int).values

        if np.nanmean(pos) < np.nanmean(neg):
            scores = -scores

        auc = auc_score(scores, labels)

        rows.append(
            {
                "dataset": dataset_name,
                "feature": c,
                "positive_label_mean": float(np.mean(pos)),
                "negative_label_mean": float(np.mean(neg)),
                "cohen_d_positive_minus_negative": d,
                "welch_t": float(tstat),
                "welch_p": float(pval),
                "oriented_auc": auc,
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
            }
        )

    return sorted(
        rows,
        key=lambda r: (
            -(abs(r["cohen_d_positive_minus_negative"]) if np.isfinite(r["cohen_d_positive_minus_negative"]) else 0),
            r["welch_p"] if np.isfinite(r["welch_p"]) else 999,
        ),
    )


def add_bh_q(rows):
    rows = list(rows)
    pvals = []
    idxs = []

    for i, r in enumerate(rows):
        p = r.get("welch_p")
        if p is not None and np.isfinite(float(p)):
            pvals.append(float(p))
            idxs.append(i)

    if not pvals:
        return rows

    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(ranked)
    q = np.empty(m, dtype=float)

    prev = 1.0
    for j in range(m - 1, -1, -1):
        val = ranked[j] * m / (j + 1)
        prev = min(prev, val)
        q[j] = prev

    q_orig = np.empty(m, dtype=float)
    q_orig[order] = q

    for idx, qv in zip(idxs, q_orig):
        rows[idx]["bh_fdr_q"] = float(min(qv, 1.0))
        rows[idx]["bh_fdr_significant_0p05"] = bool(qv <= 0.05)

    return rows


def plot_models(model_rows, path):
    ok = [r for r in model_rows if r.get("status") == "ok" and r.get("auc_mean") is not None]

    if not ok:
        return None

    labels = [f"{r['dataset']}\n{r['model_name'].replace('_model','')}" for r in ok]
    vals = [float(r["auc_mean"]) for r in ok]
    x = np.arange(len(labels))

    plt.figure(figsize=(18, 7))
    plt.bar(x, vals)
    plt.axhline(0.5, linewidth=1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Repeated CV AUC")
    plt.title("TAIRID cross-neurotype cycle/reach axis v1")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def plot_axis_advantage(model_rows, path):
    by = {}
    for r in model_rows:
        if r.get("status") == "ok" and r.get("auc_mean") is not None:
            by.setdefault(r["dataset"], {})[r["model_name"]] = float(r["auc_mean"])

    labels = []
    vals = []

    for dataset, m in by.items():
        static = m.get("static_level_model")
        axis = max(
            [
                v
                for k, v in m.items()
                if k in ["cycling_model", "reach_model", "cycling_reach_model", "combined_axis_model"]
            ] or [np.nan]
        )

        if static is not None and np.isfinite(axis):
            labels.append(dataset)
            vals.append(axis - static)

    if not labels:
        return None

    x = np.arange(len(labels))

    plt.figure(figsize=(8, 6))
    plt.bar(x, vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("Best cycle/reach/combined AUC - static AUC")
    plt.title("Cycle/reach axis advantage over static level")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def decide_status(all_model_rows, all_perm_rows):
    by = {}
    byp = {}

    for r in all_model_rows:
        by.setdefault(r["dataset"], {})[r["model_name"]] = r

    for r in all_perm_rows:
        byp.setdefault(r["dataset"], {})[r["model_name"]] = r

    dataset_status = {}

    for dataset, models in by.items():
        static_auc = models.get("static_level_model", {}).get("auc_mean")

        axis_names = ["cycling_model", "reach_model", "cycling_reach_model", "combined_axis_model"]
        axis_aucs = [
            models.get(name, {}).get("auc_mean")
            for name in axis_names
            if models.get(name, {}).get("auc_mean") is not None
        ]
        best_axis_auc = max(axis_aucs) if axis_aucs else None

        axis_ps = [
            byp.get(dataset, {}).get(name, {}).get("p_value_ge_observed")
            for name in axis_names
            if byp.get(dataset, {}).get(name, {}).get("p_value_ge_observed") is not None
        ]
        best_axis_p = min(axis_ps) if axis_ps else None

        response_auc = models.get("response_dynamic_model", {}).get("auc_mean")
        viability_auc = models.get("viability_window_model", {}).get("auc_mean")

        support = False
        if static_auc is not None and best_axis_auc is not None:
            support = bool(best_axis_auc > static_auc + 0.03 and best_axis_auc >= 0.60)

        locked = bool(support and best_axis_p is not None and best_axis_p <= 0.05)

        dataset_status[dataset] = {
            "static_auc": static_auc,
            "response_auc": response_auc,
            "viability_auc": viability_auc,
            "best_axis_auc": best_axis_auc,
            "best_axis_permutation_p": best_axis_p,
            "axis_support": support,
            "axis_locked": locked,
        }

    locked_count = sum(1 for v in dataset_status.values() if v["axis_locked"])
    support_count = sum(1 for v in dataset_status.values() if v["axis_support"])

    if locked_count >= 2:
        return (
            "cycle_reach_axes_cross_neurotype_locked",
            9,
            "Write the cross-neurotype TAIRID axis map and define the next replication dataset.",
            dataset_status,
        )

    if locked_count >= 1 and support_count >= 2:
        return (
            "cycle_reach_axes_supported_but_partly_provisional",
            8,
            "Write the axis map with ETDD70/ASD status labels and avoid overclaiming.",
            dataset_status,
        )

    if support_count >= 1:
        return (
            "cycle_reach_axes_directional_in_one_or_more_datasets",
            7,
            "Treat cycling/reach as useful derived axes, but test another dataset before locking.",
            dataset_status,
        )

    return (
        "cycle_reach_axes_not_yet_supported",
        6,
        "Keep mismatch/viability as primary axes and refine cycling/reach definitions.",
        dataset_status,
    )


def main():
    print("")
    print("TAIRID cross-neurotype cycle/reach axis test v1 starting.")
    print("Boundary: operational axis test only; not proof or diagnosis.")
    print("")

    et = load_etdd70_module()
    asd = load_asd_module()

    et_df, et_meta, et_artifacts = build_etdd70_cycle_reach_features(et)
    asd_df, asd_meta, asd_artifacts = build_asd_cycle_reach_features(asd)

    if not et_df.empty:
        et_path = OUTDIR / "etdd70_cycle_reach_subject_features.csv"
        et_df.to_csv(et_path, index=False)
    else:
        et_path = None

    if not asd_df.empty:
        asd_path = OUTDIR / "asd_cycle_reach_subject_features.csv"
        asd_df.to_csv(asd_path, index=False)
    else:
        asd_path = None

    all_model_rows = []
    all_perm_rows = []
    all_feature_rows = []

    for dataset_name, df in [("ETDD70", et_df), ("ASD", asd_df)]:
        if df.empty:
            continue

        model_rows, perm_rows = run_model_suite(df, dataset_name)
        feat_rows = add_bh_q(feature_tests(df, dataset_name))

        all_model_rows.extend(model_rows)
        all_perm_rows.extend(perm_rows)
        all_feature_rows.extend(feat_rows)

        write_csv(OUTDIR / f"{dataset_name.lower()}_cycle_reach_model_results.csv", model_rows)
        write_csv(OUTDIR / f"{dataset_name.lower()}_cycle_reach_permutation_results.csv", perm_rows)
        write_csv(OUTDIR / f"{dataset_name.lower()}_cycle_reach_feature_tests_bh_fdr.csv", feat_rows)

    model_path = write_csv(OUTDIR / "cross_neurotype_cycle_reach_model_results.csv", all_model_rows)
    perm_path = write_csv(OUTDIR / "cross_neurotype_cycle_reach_permutation_results.csv", all_perm_rows)
    feature_path = write_csv(OUTDIR / "cross_neurotype_cycle_reach_feature_tests_bh_fdr.csv", all_feature_rows)

    plots = []

    p = plot_models(all_model_rows, OUTDIR / "cross_neurotype_cycle_reach_model_auc.png")
    if p:
        plots.append(p)

    p = plot_axis_advantage(all_model_rows, OUTDIR / "cross_neurotype_cycle_reach_axis_advantage.png")
    if p:
        plots.append(p)

    final_status, readiness_score, next_wall, dataset_axis_status = decide_status(
        all_model_rows,
        all_perm_rows,
    )

    combined_counts = {
        "ETDD70_rows": int(len(et_df)) if not et_df.empty else 0,
        "ASD_rows": int(len(asd_df)) if not asd_df.empty else 0,
        "ETDD70_group_counts": {str(k): int(v) for k, v in Counter(et_df["label_name"]).items()} if not et_df.empty else {},
        "ASD_group_counts": {str(k): int(v) for k, v in Counter(asd_df["label_name"]).items()} if not asd_df.empty else {},
    }

    summary = {
        "test_name": "TAIRID cross-neurotype cycle/reach axis test v1",
        "boundary": (
            "Operational cross-neurotype axis test only. Not proof of TAIRID, "
            "not clinical diagnosis, and not a cosmology result."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "combined_counts": combined_counts,
        "dataset_axis_status": dataset_axis_status,
        "etdd70_meta": et_meta,
        "asd_meta": asd_meta,
        "model_results": all_model_rows,
        "permutation_results": all_perm_rows,
        "top_feature_tests": all_feature_rows[:30],
        "output_files": {
            "etdd70_cycle_reach_subject_features_csv": str(et_path) if et_path else None,
            "asd_cycle_reach_subject_features_csv": str(asd_path) if asd_path else None,
            "model_results_csv": str(model_path),
            "permutation_results_csv": str(perm_path),
            "feature_tests_bh_fdr_csv": str(feature_path),
            "plots": plots,
        },
        "axis_map_interpretation": {
            "T": "pacing / timing / propagation rhythm",
            "I": "constraint / stabilization / task or context organization",
            "M": "|T - I| mismatch",
            "W": "viability window / baseline tolerated mismatch",
            "B": "breach outside the viability window",
            "Cycling": "recurrence, reversal, breach/re-entry, oscillatory or phase-turn behavior",
            "Reach": "coherent propagation span across task/context before breach or instability",
            "cosmology_lesson": (
                "Do not translate TAIRID as a simple offset. Search for slope, curvature, covariance, "
                "cycle/reach, and viability-window behavior."
            ),
            "truth_boundary": (
                "If supported, cycling/reach become derived operational axes for TAIRID. "
                "They do not prove TAIRID or transfer proof between neuroscience and cosmology."
            ),
        },
    }

    summary_path = OUTDIR / "cross_neurotype_cycle_reach_axis_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(OUTDIR / "cross_neurotype_cycle_reach_axis_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID cross-neurotype cycle/reach axis test v1\n\n")
        f.write("Boundary: operational axis test only. Not proof. Not diagnosis. Not cosmology result.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Why this test exists:\n")
        f.write("- ETDD70 supported dynamic mismatch and viability breach.\n")
        f.write("- ASD artifact-control supported the same axis family provisionally.\n")
        f.write("- Cycling and reach were present implicitly but needed explicit axis tests.\n\n")

        f.write("Dataset axis status:\n")
        f.write(json.dumps(dataset_axis_status, indent=2) + "\n\n")

        f.write("Model results:\n")
        f.write(json.dumps(all_model_rows, indent=2) + "\n\n")

        f.write("Permutation results:\n")
        f.write(json.dumps(all_perm_rows, indent=2) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This can support cycling/reach as derived operational axes.\n")
        f.write("- It cannot prove TAIRID.\n")
        f.write("- It cannot diagnose neurotypes.\n")
        f.write("- It cannot prove any cosmology claim.\n")

    print("")
    print("TAIRID cross-neurotype cycle/reach axis test v1 complete.")
    print("Created:")
    print("  tairid_cross_neurotype_cycle_reach_axis_v1_outputs/cross_neurotype_cycle_reach_axis_v1_summary.json")
    print("  tairid_cross_neurotype_cycle_reach_axis_v1_outputs/cross_neurotype_cycle_reach_axis_v1_summary.txt")
    print("  tairid_cross_neurotype_cycle_reach_axis_v1_outputs/cross_neurotype_cycle_reach_model_results.csv")
    print("  tairid_cross_neurotype_cycle_reach_axis_v1_outputs/cross_neurotype_cycle_reach_permutation_results.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is not clinical diagnosis.")
    print("  This is a cross-neurotype operational axis test.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
