#!/usr/bin/env python3
"""
TAIRID ETDD70 viability-window robustness test v1.

Purpose:
The ETDD70 viability-window v2 test found that dynamic mismatch and viability-breach
features carried more group-relevant signal than simple static level features.

This robustness pass asks whether that result holds under:
1. repeated cross-validation,
2. label permutation,
3. tolerance-window stability,
4. trial-only / AOI-only / general / combined feature-family checks,
5. multiple-comparison correction for individual features.

Boundary:
This is not proof of TAIRID.
This is not clinical diagnosis.
This is not a cosmology result.
This is a robustness test of the TAIRID response-shape translation.

Dependency:
This script intentionally reuses the accepted v2 parser:
    run_tairid_etdd70_viability_window_v2.py
Keep that file in the repository beside this file.
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


OUTDIR = Path("tairid_etdd70_viability_robustness_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_DIR = OUTDIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
CV_REPEATS = 100
PERMUTATIONS = 250
PERM_REPEATS = 25
RIDGE = 1.0e-3

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


def load_v2_module():
    try:
        vw2 = importlib.import_module("run_tairid_etdd70_viability_window_v2")
    except Exception as exc:
        raise RuntimeError(
            "Could not import run_tairid_etdd70_viability_window_v2.py. "
            "Keep the accepted v2 parser file in the repository beside this script. "
            f"Import error: {exc}"
        )

    # Redirect v2 parser outputs into this robustness folder.
    vw2.OUTDIR = OUTDIR
    vw2.DOWNLOAD_DIR = DOWNLOAD_DIR
    vw2.EXTRACT_DIR = EXTRACT_DIR

    OUTDIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    return vw2


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


def bootstrap_auc(scores, labels, n_boot=500):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan

    base = auc_score(scores, labels)
    rng = np.random.default_rng(RANDOM_SEED)
    boots = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        if len(np.unique(labels[idx])) < 2:
            continue
        boots.append(auc_score(scores[idx], labels[idx]))

    if not boots:
        return base, np.nan, np.nan

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return base, float(lo), float(hi)


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


def lda_fold_scores(X_train, y_train, X_test):
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


def repeated_cv_predictions(df, feature_cols, repeats=CV_REPEATS, k=5, y_override=None):
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return {
            "status": "no_features",
            "feature_cols": [],
            "n": 0,
            "auc_mean": None,
            "auc_std": None,
        }

    data = df[feature_cols + ["label"]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 20 or data["label"].nunique() < 2:
        return {
            "status": "not_enough_data",
            "feature_cols": feature_cols,
            "n": int(len(data)),
            "auc_mean": None,
            "auc_std": None,
        }

    X_raw = data[feature_cols].astype(float).values
    y = data["label"].astype(int).values

    if y_override is not None:
        y = np.asarray(y_override, dtype=int)
        if len(y) != len(data):
            raise ValueError("y_override length mismatch")

    if len(np.unique(y)) < 2:
        return {
            "status": "one_class",
            "feature_cols": feature_cols,
            "n": int(len(data)),
            "auc_mean": None,
            "auc_std": None,
        }

    rng = np.random.default_rng(RANDOM_SEED)
    aucs = []
    all_preds = []
    all_labels = []

    k_eff = min(k, int(np.min(np.bincount(y))) if len(np.bincount(y)) > 1 else k)
    k_eff = max(2, k_eff)

    for _ in range(repeats):
        folds = stratified_folds(y, k_eff, rng)
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

            preds[test] = lda_fold_scores(X_train_z, y_train, X_test_z)

        aucs.append(auc_score(preds, y))
        all_preds.append(preds)
        all_labels.append(y.copy())

    aucs = np.asarray([a for a in aucs if np.isfinite(a)], dtype=float)

    if len(aucs) == 0:
        return {
            "status": "auc_failed",
            "feature_cols": feature_cols,
            "n": int(len(data)),
            "auc_mean": None,
            "auc_std": None,
        }

    pooled_preds = np.mean(np.vstack(all_preds), axis=0)
    pooled_labels = all_labels[0]
    pooled_auc, lo, hi = bootstrap_auc(pooled_preds, pooled_labels, n_boot=500)

    return {
        "status": "ok",
        "feature_cols": feature_cols,
        "n": int(len(data)),
        "repeats": repeats,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_min": float(np.min(aucs)),
        "auc_max": float(np.max(aucs)),
        "pooled_auc": pooled_auc,
        "pooled_auc_ci_2p5": lo,
        "pooled_auc_ci_97p5": hi,
    }


def permutation_test(df, feature_cols, observed_auc, repeats=PERM_REPEATS, n_perm=PERMUTATIONS):
    feature_cols = [c for c in feature_cols if c in df.columns]

    data = df[feature_cols + ["label"]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 20 or data["label"].nunique() < 2 or observed_auc is None:
        return {
            "status": "not_enough_data",
            "n_perm": 0,
            "p_value_ge_observed": None,
            "perm_auc_mean": None,
            "perm_auc_std": None,
        }

    y = data["label"].astype(int).values
    rng = np.random.default_rng(RANDOM_SEED + 99)
    perm_aucs = []

    for _ in range(n_perm):
        y_perm = y.copy()
        rng.shuffle(y_perm)

        res = repeated_cv_predictions(
            data,
            feature_cols,
            repeats=repeats,
            k=5,
            y_override=y_perm,
        )

        auc = res.get("auc_mean")
        if auc is not None and np.isfinite(auc):
            perm_aucs.append(float(auc))

    if not perm_aucs:
        return {
            "status": "no_valid_permutations",
            "n_perm": 0,
            "p_value_ge_observed": None,
            "perm_auc_mean": None,
            "perm_auc_std": None,
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
        "perm_auc_values": [float(x) for x in perm_aucs],
    }


def add_bh_q_values(feature_rows):
    rows = list(feature_rows)

    pvals = []
    idxs = []

    for i, row in enumerate(rows):
        p = row.get("welch_p")
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

    q_original = np.empty(m, dtype=float)
    q_original[order] = q

    for idx, qv in zip(idxs, q_original):
        rows[idx]["bh_fdr_q"] = float(min(qv, 1.0))
        rows[idx]["bh_fdr_significant_0p05"] = bool(qv <= 0.05)

    return rows


def model_feature_sets():
    static_cols = [
        "M_mean",
        "M_max",
        "collapse_load_mean",
        "collapse_load_max",
    ]

    mismatch_dynamic_cols = [
        "M_slope",
        "M_curvature",
        "M_range",
        "T_I_slope_gap",
        "collapse_load_slope",
        "collapse_load_curvature",
    ]

    viability_cols = []
    for mult in [0.0, 0.25, 0.5, 1.0]:
        suffix = f"tol{str(mult).replace('.', 'p')}"
        viability_cols.extend(
            [
                f"B_total_{suffix}",
                f"B_mean_harder_{suffix}",
                f"B_max_{suffix}",
                f"B_slope_{suffix}",
                f"B_curvature_{suffix}",
                f"B_pseudotext_minus_text_{suffix}",
            ]
        )

    combined_cols = static_cols + mismatch_dynamic_cols + viability_cols

    return {
        "static_level_model": static_cols,
        "mismatch_dynamic_model": mismatch_dynamic_cols,
        "viability_breach_model": viability_cols,
        "combined_response_viability_model": combined_cols,
    }


def tolerance_feature_sets():
    out = {}

    for mult in [0.0, 0.25, 0.5, 1.0]:
        suffix = f"tol{str(mult).replace('.', 'p')}"
        out[f"viability_breach_{suffix}"] = [
            f"B_total_{suffix}",
            f"B_mean_harder_{suffix}",
            f"B_max_{suffix}",
            f"B_slope_{suffix}",
            f"B_curvature_{suffix}",
            f"B_pseudotext_minus_text_{suffix}",
        ]

    return out


def build_tairid_task_features_family(vw2, subject_task_df, family):
    if subject_task_df.empty:
        return pd.DataFrame(), {
            "family": family,
            "numeric_cols_count": 0,
            "pacing_cols_count": 0,
            "constraint_cols_count": 0,
        }

    numeric_cols = [
        c for c in subject_task_df.columns
        if c not in ["subject", "group", "task_rank", "task_name"]
        and not vw2.is_identifier_artifact_col(c)
        and not c.endswith("dedup_trial_count")
        and "row_count" not in vw2.norm(c)
        and "nrows" not in vw2.norm(c)
        and pd.api.types.is_numeric_dtype(subject_task_df[c])
        and subject_task_df[c].notnull().sum() >= 10
    ]

    if family != "all":
        numeric_cols = [c for c in numeric_cols if c.startswith(family + "__")]

    pacing_cols = []
    constraint_cols = []

    for c in numeric_cols:
        role = vw2.column_role(c)
        if role in ["pacing", "both"]:
            pacing_cols.append(c)
        if role in ["constraint", "both"]:
            constraint_cols.append(c)

    pacing_cols = pacing_cols[:40]
    constraint_cols = constraint_cols[:40]

    if not pacing_cols and numeric_cols:
        pacing_cols = numeric_cols[: max(1, len(numeric_cols) // 3)]

    if not constraint_cols and numeric_cols:
        constraint_cols = numeric_cols[max(1, len(numeric_cols) // 3):] or numeric_cols

    if not numeric_cols:
        return pd.DataFrame(), {
            "family": family,
            "numeric_cols_count": 0,
            "pacing_cols_count": 0,
            "constraint_cols_count": 0,
        }

    df = vw2.zscore_by_task(subject_task_df, numeric_cols)

    pacing_z = [c + "__z_by_task" for c in pacing_cols if c + "__z_by_task" in df.columns]
    constraint_z = [c + "__z_by_task" for c in constraint_cols if c + "__z_by_task" in df.columns]

    if not pacing_z:
        pacing_z = [c + "__z_by_task" for c in numeric_cols[: max(1, len(numeric_cols) // 3)]]
    if not constraint_z:
        constraint_z = [c + "__z_by_task" for c in numeric_cols[max(1, len(numeric_cols) // 3):]]

    df["T_pacing_proxy"] = df[pacing_z].mean(axis=1)
    df["I_constraint_proxy"] = df[constraint_z].mean(axis=1)
    df["M_mismatch_abs"] = np.abs(df["T_pacing_proxy"] - df["I_constraint_proxy"])
    df["collapse_load_proxy"] = np.sqrt(df["T_pacing_proxy"] ** 2 + df["I_constraint_proxy"] ** 2)
    df["interaction_TI"] = df["T_pacing_proxy"] * df["I_constraint_proxy"]

    meta = {
        "family": family,
        "numeric_cols_count": len(numeric_cols),
        "pacing_cols_count": len(pacing_cols),
        "constraint_cols_count": len(constraint_cols),
        "pacing_cols": pacing_cols,
        "constraint_cols": constraint_cols,
    }

    return df, meta


def prepare_labeled_subject_df(subject_viability_df):
    if subject_viability_df.empty:
        return pd.DataFrame()

    df = subject_viability_df.copy()
    df = df[df["group"].isin(["control", "dyslexic"])]
    df["label"] = (df["group"] == "dyslexic").astype(int)
    return df


def run_model_suite(subject_df, repeats=CV_REPEATS):
    df = prepare_labeled_subject_df(subject_df)
    rows = []

    for name, cols in model_feature_sets().items():
        res = repeated_cv_predictions(df, cols, repeats=repeats)
        row = {
            "model_name": name,
            **{k: v for k, v in res.items() if k != "feature_cols"},
            "feature_cols": " | ".join(res.get("feature_cols", [])),
        }
        rows.append(row)

    return rows


def run_tolerance_suite(subject_df, repeats=CV_REPEATS):
    df = prepare_labeled_subject_df(subject_df)
    rows = []

    for name, cols in tolerance_feature_sets().items():
        res = repeated_cv_predictions(df, cols, repeats=repeats)
        row = {
            "model_name": name,
            **{k: v for k, v in res.items() if k != "feature_cols"},
            "feature_cols": " | ".join(res.get("feature_cols", [])),
        }
        rows.append(row)

    return rows


def run_permutation_suite(subject_df, model_rows):
    df = prepare_labeled_subject_df(subject_df)
    rows = []

    feature_map = model_feature_sets()

    for model_row in model_rows:
        name = model_row["model_name"]
        cols = feature_map.get(name, [])
        observed = model_row.get("auc_mean")

        perm = permutation_test(df, cols, observed_auc=observed)

        rows.append(
            {
                "model_name": name,
                "observed_auc_mean": observed,
                "permutation_status": perm.get("status"),
                "n_perm": perm.get("n_perm"),
                "p_value_ge_observed": perm.get("p_value_ge_observed"),
                "perm_auc_mean": perm.get("perm_auc_mean"),
                "perm_auc_std": perm.get("perm_auc_std"),
                "perm_auc_95": perm.get("perm_auc_95"),
                "perm_auc_99": perm.get("perm_auc_99"),
            }
        )

    return rows


def run_family_suite(vw2, subject_task_df):
    family_rows = []
    family_meta = []

    for family in ["all", "trial", "aoi", "general"]:
        task_df, meta = build_tairid_task_features_family(vw2, subject_task_df, family)
        subject_df, vw_meta = vw2.build_subject_viability_features(task_df)

        family_meta.append(
            {
                "family": family,
                **meta,
                "subject_viability_rows": int(len(subject_df)) if not subject_df.empty else 0,
                "complete_subject_count": vw_meta.get("complete_subject_count"),
            }
        )

        if subject_df.empty:
            continue

        suite = run_model_suite(subject_df, repeats=CV_REPEATS)

        for row in suite:
            family_rows.append(
                {
                    "family": family,
                    **row,
                }
            )

    return family_rows, family_meta


def plot_model_bars(model_rows, title, path):
    ok = [r for r in model_rows if r.get("status") == "ok" and r.get("auc_mean") is not None]

    if not ok:
        return None

    labels = [r["model_name"] for r in ok]
    vals = [float(r["auc_mean"]) for r in ok]
    errs = [float(r.get("auc_std") or 0.0) for r in ok]
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, vals, yerr=errs)
    plt.axhline(0.5, linewidth=1)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Repeated CV AUC")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def plot_family_bars(family_rows, path):
    ok = [r for r in family_rows if r.get("status") == "ok" and r.get("auc_mean") is not None]

    if not ok:
        return None

    labels = [f"{r['family']}\n{r['model_name'].replace('_model','')}" for r in ok]
    vals = [float(r["auc_mean"]) for r in ok]
    x = np.arange(len(labels))

    plt.figure(figsize=(16, 7))
    plt.bar(x, vals)
    plt.axhline(0.5, linewidth=1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Repeated CV AUC")
    plt.title("ETDD70 viability robustness v1: feature-family stability")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def plot_permutation_rows(permutation_rows, path):
    ok = [r for r in permutation_rows if r.get("permutation_status") == "ok"]

    if not ok:
        return None

    labels = [r["model_name"] for r in ok]
    obs = [float(r["observed_auc_mean"]) for r in ok]
    perm95 = [float(r["perm_auc_95"]) for r in ok]
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.18, obs, width=0.36, label="Observed")
    plt.bar(x + 0.18, perm95, width=0.36, label="Permutation 95th percentile")
    plt.axhline(0.5, linewidth=1)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("AUC")
    plt.title("ETDD70 viability robustness v1: observed vs permutation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def decide_status(model_rows, permutation_rows, tolerance_rows, family_rows):
    by_model = {r["model_name"]: r for r in model_rows}
    by_perm = {r["model_name"]: r for r in permutation_rows}

    static_auc = by_model.get("static_level_model", {}).get("auc_mean")
    dynamic_auc = by_model.get("mismatch_dynamic_model", {}).get("auc_mean")
    viability_auc = by_model.get("viability_breach_model", {}).get("auc_mean")
    combined_auc = by_model.get("combined_response_viability_model", {}).get("auc_mean")

    dynamic_p = by_perm.get("mismatch_dynamic_model", {}).get("p_value_ge_observed")
    viability_p = by_perm.get("viability_breach_model", {}).get("p_value_ge_observed")
    combined_p = by_perm.get("combined_response_viability_model", {}).get("p_value_ge_observed")

    shape_auc = max(
        [x for x in [dynamic_auc, viability_auc, combined_auc] if x is not None] or [None]
    )

    shape_p = min(
        [p for p in [dynamic_p, viability_p, combined_p] if p is not None] or [1.0]
    )

    tolerance_ok = [
        r for r in tolerance_rows
        if r.get("status") == "ok" and r.get("auc_mean") is not None and float(r["auc_mean"]) >= 0.58
    ]

    family_ok = [
        r for r in family_rows
        if r.get("status") == "ok"
        and r.get("auc_mean") is not None
        and float(r["auc_mean"]) >= 0.58
        and r.get("model_name") in ["mismatch_dynamic_model", "viability_breach_model", "combined_response_viability_model"]
    ]

    if (
        static_auc is not None
        and shape_auc is not None
        and shape_auc > static_auc + 0.05
        and shape_p <= 0.05
        and len(tolerance_ok) >= 2
    ):
        return (
            "robustness_supports_tairid_response_shape_translation",
            8,
            "Run a second-dataset replication across another neurotype/task-demand dataset.",
        )

    if (
        static_auc is not None
        and shape_auc is not None
        and shape_auc > static_auc + 0.05
        and len(tolerance_ok) >= 2
    ):
        return (
            "directional_response_shape_signal_not_permutation_locked",
            7,
            "Signal remains above static, but permutation or family stability is not strong enough to lock.",
        )

    if shape_auc is not None and shape_auc >= 0.60 and len(family_ok) >= 2:
        return (
            "response_shape_signal_present_but_needs_stronger_validation",
            7,
            "Feature-family stability is promising; improve proxy definitions before replication.",
        )

    return (
        "robustness_does_not_yet_lock_response_shape",
        6,
        "Treat v2 as exploratory; refine T/I proxy construction or use richer event-level features.",
    )


def main():
    print("")
    print("TAIRID ETDD70 viability-window robustness test v1 starting.")
    print("Boundary: robustness of response-shape translation only; not proof.")
    print("")

    vw2 = load_v2_module()

    record, downloads = vw2.download_zenodo_record()
    extraction = vw2.extract_archives(downloads)

    write_csv(OUTDIR / "etdd70_vwr_download_ledger.csv", downloads)
    write_csv(OUTDIR / "etdd70_vwr_extraction_ledger.csv", extraction)

    label_map, label_sources = vw2.load_label_map()
    label_map_rows = [{"subject_key": k, "group": v} for k, v in sorted(label_map.items())]

    write_csv(OUTDIR / "etdd70_vwr_label_sources.csv", label_sources)
    write_csv(OUTDIR / "etdd70_vwr_label_map.csv", label_map_rows)

    metrics_inventory, metric_feature_rows = vw2.parse_metrics_files(label_map)
    write_csv(OUTDIR / "etdd70_vwr_metrics_inventory.csv", metrics_inventory)
    write_csv(OUTDIR / "etdd70_vwr_metric_feature_rows.csv", metric_feature_rows)

    subject_task_df = vw2.aggregate_subject_task(metric_feature_rows)
    if not subject_task_df.empty:
        subject_task_path = OUTDIR / "etdd70_vwr_subject_task_features.csv"
        subject_task_df.to_csv(subject_task_path, index=False)
    else:
        subject_task_path = None

    task_df, ti_meta = vw2.build_tairid_task_features(subject_task_df)
    if not task_df.empty:
        task_path = OUTDIR / "etdd70_vwr_tairid_task_features.csv"
        task_df.to_csv(task_path, index=False)
    else:
        task_path = None

    subject_viability_df, vw_meta = vw2.build_subject_viability_features(task_df)
    if not subject_viability_df.empty:
        subject_vw_path = OUTDIR / "etdd70_vwr_subject_viability_features.csv"
        subject_viability_df.to_csv(subject_vw_path, index=False)
    else:
        subject_vw_path = None

    model_rows = run_model_suite(subject_viability_df, repeats=CV_REPEATS)
    tolerance_rows = run_tolerance_suite(subject_viability_df, repeats=CV_REPEATS)
    permutation_rows = run_permutation_suite(subject_viability_df, model_rows)
    family_rows, family_meta = run_family_suite(vw2, subject_task_df)

    feature_rows = add_bh_q_values(vw2.feature_tests(subject_viability_df))

    model_path = write_csv(OUTDIR / "etdd70_vwr_repeated_cv_models.csv", model_rows)
    tolerance_path = write_csv(OUTDIR / "etdd70_vwr_tolerance_stability.csv", tolerance_rows)
    permutation_path = write_csv(OUTDIR / "etdd70_vwr_permutation_tests.csv", permutation_rows)
    family_path = write_csv(OUTDIR / "etdd70_vwr_feature_family_stability.csv", family_rows)
    family_meta_path = write_csv(OUTDIR / "etdd70_vwr_feature_family_meta.csv", family_meta)
    feature_path = write_csv(OUTDIR / "etdd70_vwr_feature_tests_bh_fdr.csv", feature_rows)

    plots = []

    p = plot_model_bars(
        model_rows,
        "ETDD70 viability robustness v1: repeated CV model comparison",
        OUTDIR / "etdd70_vwr_model_auc_bars.png",
    )
    if p:
        plots.append(p)

    p = plot_model_bars(
        tolerance_rows,
        "ETDD70 viability robustness v1: tolerance stability",
        OUTDIR / "etdd70_vwr_tolerance_auc_bars.png",
    )
    if p:
        plots.append(p)

    p = plot_family_bars(
        family_rows,
        OUTDIR / "etdd70_vwr_family_auc_bars.png",
    )
    if p:
        plots.append(p)

    p = plot_permutation_rows(
        permutation_rows,
        OUTDIR / "etdd70_vwr_permutation_auc_bars.png",
    )
    if p:
        plots.append(p)

    final_status, readiness_score, next_wall = decide_status(
        model_rows,
        permutation_rows,
        tolerance_rows,
        family_rows,
    )

    task_counts = {}
    if not subject_task_df.empty:
        task_counts = {
            str(k): int(v)
            for k, v in subject_task_df["task_rank"].value_counts().sort_index().items()
        }

    group_counts = {}
    if not subject_viability_df.empty:
        group_counts = {
            str(k): int(v)
            for k, v in Counter(subject_viability_df["group"]).items()
        }

    top_features = feature_rows[:20]

    summary = {
        "test_name": "TAIRID ETDD70 viability-window robustness test v1",
        "boundary": (
            "Robustness test of cross-domain viability-window response-shape translation only. "
            "Not clinical diagnosis, not proof of TAIRID, and not a cosmology result."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "dataset": {
            "zenodo_record_id": vw2.ZENODO_RECORD_ID,
            "zenodo_api_url": vw2.ZENODO_API_URL,
            "record_title": record.get("metadata", {}).get("title"),
        },
        "parser_counts": {
            "downloads_count": len(downloads),
            "extraction_count": len(extraction),
            "label_source_count": len(label_sources),
            "label_map_count": len(label_map),
            "metrics_inventory_count": len(metrics_inventory),
            "metric_feature_row_count": len(metric_feature_rows),
            "subject_task_feature_count": int(len(subject_task_df)) if not subject_task_df.empty else 0,
            "subject_task_task_counts": task_counts,
            "subject_viability_row_count": int(len(subject_viability_df)) if not subject_viability_df.empty else 0,
            "group_counts": group_counts,
        },
        "robustness_settings": {
            "cv_repeats": CV_REPEATS,
            "permutations": PERMUTATIONS,
            "permutation_cv_repeats": PERM_REPEATS,
            "classifier": "ridge-regularized LDA repeated stratified CV",
        },
        "model_results": model_rows,
        "tolerance_stability": tolerance_rows,
        "permutation_tests": [
            {k: v for k, v in row.items() if k != "perm_auc_values"}
            for row in permutation_rows
        ],
        "feature_family_stability": family_rows,
        "feature_family_meta": family_meta,
        "top_feature_tests_bh_fdr": top_features,
        "ti_proxy_meta": ti_meta,
        "viability_window_meta": vw_meta,
        "output_files": {
            "subject_task_features_csv": str(subject_task_path) if subject_task_path else None,
            "tairid_task_features_csv": str(task_path) if task_path else None,
            "subject_viability_features_csv": str(subject_vw_path) if subject_vw_path else None,
            "repeated_cv_models_csv": str(model_path),
            "tolerance_stability_csv": str(tolerance_path),
            "permutation_tests_csv": str(permutation_path),
            "feature_family_stability_csv": str(family_path),
            "feature_family_meta_csv": str(family_meta_path),
            "feature_tests_bh_fdr_csv": str(feature_path),
            "plots": plots,
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "Dynamic mismatch or viability-window breach remains stronger than static level features "
                "under repeated CV, permutation, tolerance, and feature-family checks."
            ),
            "what_weakens_this_translation": (
                "The signal drops to chance under permutation, depends on one tolerance setting only, "
                "or appears only in one fragile feature family."
            ),
            "axis_mapping_note": (
                "Neurotype data is being used to map TAIRID axes: T pacing, I constraint, "
                "M mismatch, W viability window, B breach, plus slope/curvature/recovery. "
                "Cosmology taught that offset-shaped translations are absorbed; neurotype tasks test "
                "the relative differential response shape more directly."
            ),
        },
    }

    summary_path = OUTDIR / "etdd70_viability_robustness_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(OUTDIR / "etdd70_viability_robustness_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID ETDD70 viability-window robustness test v1\n\n")
        f.write("Boundary: robustness of response-shape translation only. Not diagnosis. Not proof.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Why this test exists:\n")
        f.write("- SH0ES showed simple offset-shaped TAIRID translations can be absorbed by existing model freedom.\n")
        f.write("- ETDD70 v2 suggested dynamic mismatch / viability breach carries more signal than static level.\n")
        f.write("- This pass checks whether that result is stable enough to guide the next neurotype replication.\n\n")

        f.write("Model results:\n")
        f.write(json.dumps(model_rows, indent=2) + "\n\n")

        f.write("Permutation tests:\n")
        f.write(json.dumps([{k: v for k, v in row.items() if k != "perm_auc_values"} for row in permutation_rows], indent=2) + "\n\n")

        f.write("Tolerance stability:\n")
        f.write(json.dumps(tolerance_rows, indent=2) + "\n\n")

        f.write("Feature-family stability:\n")
        f.write(json.dumps(family_rows, indent=2) + "\n\n")

        f.write("Top feature tests with BH-FDR:\n")
        f.write(json.dumps(top_features, indent=2) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This can support a TAIRID response-shape translation.\n")
        f.write("- It cannot prove TAIRID.\n")
        f.write("- It cannot diagnose dyslexia.\n")
        f.write("- It cannot prove any cosmology claim.\n")

    print("")
    print("TAIRID ETDD70 viability-window robustness test v1 complete.")
    print("Created:")
    print("  tairid_etdd70_viability_robustness_v1_outputs/etdd70_viability_robustness_v1_summary.json")
    print("  tairid_etdd70_viability_robustness_v1_outputs/etdd70_viability_robustness_v1_summary.txt")
    print("  tairid_etdd70_viability_robustness_v1_outputs/etdd70_vwr_repeated_cv_models.csv")
    print("  tairid_etdd70_viability_robustness_v1_outputs/etdd70_vwr_permutation_tests.csv")
    print("  tairid_etdd70_viability_robustness_v1_outputs/etdd70_vwr_tolerance_stability.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is not a clinical diagnostic result.")
    print("  This is a robustness test of the response-shape translation.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
