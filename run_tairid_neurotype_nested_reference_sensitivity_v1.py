#!/usr/bin/env python3
"""Fold-contained sensitivity replay of the frozen ETDD70 / ASD cycle-reach tests.

This is an ADDITIVE methodology audit. Do not alter, relabel, or overwrite the
historical scripts or outputs. No permutation significance is claimed here.
All task/context z-scores and control-group reference scales are trained on
outer-fold training participants only.
"""
from __future__ import annotations

import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

import run_tairid_cross_neurotype_cycle_reach_axis_v1 as legacy

OUT = Path("tairid_neurotype_nested_reference_sensitivity_v1_outputs")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261011
REPEATS = 8
EXPECTED_N = {"ETDD70": 70, "ASD": 56}
HISTORICAL = {
    "ETDD70": {"static_level_model": 0.5871, "reach_model": 0.7388,
                "viability_window_model": 0.6118, "cycling_model": 0.5604,
                "response_dynamic_model": 0.6605, "cycling_reach_model": 0.6314,
                "combined_axis_model": 0.5583},
    "ASD": {"static_level_model": 0.4509, "reach_model": 0.6960,
            "viability_window_model": 0.7541, "cycling_model": 0.5433,
            "response_dynamic_model": 0.5398, "cycling_reach_model": 0.6795,
            "combined_axis_model": 0.6172},
}

def compact_hash(df, id_col, context_col):
    sample = df[[id_col, context_col]].astype(str).sort_values([id_col, context_col])
    return hashlib.sha256(sample.to_csv(index=False).encode()).hexdigest()

def get_raw(domain):
    if domain == "ETDD70":
        et = legacy.load_etdd70_module()
        _, meta, aux = legacy.build_etdd70_cycle_reach_features(et)
        raw = aux["subject_task_df"].copy()
        id_col, context_col, control, case, engine = (
            "subject", "task_rank", "control", "dyslexic", et)
    else:
        asd = legacy.load_asd_module()
        _, meta, aux = legacy.build_asd_cycle_reach_features(asd)
        raw, _ = legacy.clean_asd_participant_context_df(aux["participant_context_df"])
        id_col, context_col, control, case, engine = (
            "participant", "context", "td", "asd", asd)
    raw[id_col] = raw[id_col].astype(str)
    raw = raw[raw["group"].isin([control, case])].copy()
    raw = raw.replace([np.inf, -np.inf], np.nan)
    # One row per participant/context and at least three contexts per participant.
    if raw.duplicated([id_col, context_col]).any():
        raise ValueError(f"{domain}: participant/context duplicates at raw input")
    complete = raw.groupby(id_col)[context_col].nunique()
    ids = sorted(complete[complete >= 3].index.tolist())
    raw = raw[raw[id_col].isin(ids)].copy()
    if len(ids) != EXPECTED_N[domain]:
        raise ValueError(
            f"{domain}: source population changed: {len(ids)} != {EXPECTED_N[domain]}. "
            "Do not silently score against frozen historical counts.")
    groupmap = raw.groupby(id_col)["group"].first().to_dict()
    if raw.groupby(id_col)["group"].nunique().max() != 1:
        raise ValueError(f"{domain}: subject group inconsistent by context")
    y = np.array([int(groupmap[s] == case) for s in ids], dtype=int)
    return raw, ids, y, (id_col, context_col, control, case, engine), meta

def transform_fold(raw, ids, test_ids, spec, domain):
    id_col, context_col, control, case, eng = spec
    test_set = set(test_ids)
    train_set = set(ids) - test_set
    train = raw[raw[id_col].isin(train_set)].copy()
    if domain == "ETDD70":
        numcols, pacing_cols, constraint_cols = eng.choose_role_columns(train)
        suffix = "__z_by_task"
    else:
        numcols, pacing_cols, constraint_cols = eng.choose_role_columns(
            train, family_filter="all")
        suffix = "__z_by_context"
    if not numcols or not pacing_cols or not constraint_cols:
        raise ValueError(f"{domain}: no usable role columns in training fold")
    full = raw.copy()
    # Source role selection and all z-score references are TRAINING-ONLY.
    for col in numcols:
        zcol = col + suffix
        full[zcol] = np.nan
        for context in sorted(full[context_col].dropna().unique().tolist()):
            masktr = train[context_col] == context
            sample = pd.to_numeric(train.loc[masktr, col], errors="coerce")
            if sample.empty or not np.isfinite(sample).any():
                # The frozen zscore_by_context helper uses 0 when a context
                # has no finite reference scale. Never inspect heldout values
                # to fill such a column.
                full.loc[full[context_col] == context, zcol] = 0.0
                continue
            mu = float(sample.mean())
            sd = float(sample.std(ddof=0))
            mask = full[context_col] == context
            vals = pd.to_numeric(full.loc[mask, col], errors="coerce")
            full.loc[mask, zcol] = ((vals - mu) / sd) if sd > 1e-12 else 0.0
    pcols = [c + suffix for c in pacing_cols]
    icols = [c + suffix for c in constraint_cols]
    full["T_pacing_proxy"] = full[pcols].mean(axis=1)
    full["I_constraint_proxy"] = full[icols].mean(axis=1)
    full["M_mismatch_abs"] = abs(full["T_pacing_proxy"] - full["I_constraint_proxy"])
    full["collapse_load_proxy"] = np.hypot(
        full["T_pacing_proxy"], full["I_constraint_proxy"])
    full["interaction_TI"] = full["T_pacing_proxy"] * full["I_constraint_proxy"]
    if not np.isfinite(full["M_mismatch_abs"].to_numpy(float)).all():
        raise ValueError(f"{domain}: nonfinite mismatch proxies")

    # Historical viability builders calculate control/TD reference SD from
    # group labels. Mask TEST labels before calling them, so the reference SD
    # comes exclusively from training controls (test outcomes remain withheld).
    masked = full.copy()
    masked.loc[masked[id_col].isin(test_set), "group"] = "__held_out__"
    if domain == "ETDD70":
        frame, _ = eng.build_subject_viability_features(masked)
    else:
        frame, _ = eng.build_subject_response_features(masked)
    base_map = frame.set_index(id_col).to_dict("index")
    true_group = raw.groupby(id_col)["group"].first().to_dict()
    rows = []
    for subject, grp in full.groupby(id_col):
        if subject not in base_map:
            raise ValueError(f"{domain}: missing complete feature row for {subject}")
        sub = grp.sort_values(context_col)
        if sub[context_col].nunique() < 3:
            raise ValueError(f"{domain}: incomplete contexts {subject}")
        x = (sub[context_col].to_numpy(float) if domain == "ETDD70"
             else np.arange(len(sub), dtype=float))
        t = sub["T_pacing_proxy"].to_numpy(float)
        i = sub["I_constraint_proxy"].to_numpy(float)
        m = sub["M_mismatch_abs"].to_numpy(float)
        c = sub["collapse_load_proxy"].to_numpy(float)
        base = dict(base_map[subject])
        base["group"] = true_group[subject]
        base[id_col] = subject
        base["label"] = int(true_group[subject] == case)
        base["dataset"] = domain
        base = legacy.add_cycle_reach_features_from_series(
            base, x, t, i, m, c,
            "etdd70" if domain == "ETDD70" else "asd")
        rows.append(base)
    out = pd.DataFrame(rows).set_index(id_col).reindex(ids).reset_index()
    if out["label"].isna().any():
        raise ValueError(f"{domain}: feature alignment failed")
    return out, {
        "n_train": len(train_set), "n_test": len(test_set),
        "role_column_count": len(numcols),
        "pacing_columns": len(pacing_cols),
        "constraint_columns": len(constraint_cols),
    }

def score_fold(train_rows, test_rows, y_train, columns):
    # Imputation of unavailable features is derived from train only; all
    # columns are fixed by train feature-family existence, not outcomes.
    xtr = train_rows[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    xte = test_rows[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    xtr[~np.isfinite(xtr)] = np.nan
    xte[~np.isfinite(xte)] = np.nan
    mu = np.nanmean(xtr, axis=0)
    mu[~np.isfinite(mu)] = 0.0
    n_missing = int(np.isnan(xtr).sum() + np.isnan(xte).sum())
    xtr = np.where(np.isnan(xtr), mu, xtr)
    xte = np.where(np.isnan(xte), mu, xte)
    sd = xtr.std(axis=0)
    sd[sd <= 1e-12] = 1.0
    score = legacy.lda_scores((xtr - mu) / sd, y_train, (xte - mu) / sd)
    return score, n_missing

def run_domain(domain):
    raw, ids, y, spec, metadata = get_raw(domain)
    n = len(ids)
    rng = np.random.default_rng(SEED)
    labels = dict(zip(ids, y))
    by_model = {name: [] for name in HISTORICAL[domain]}
    fold_records = []
    prediction_records = []
    for repeat in range(REPEATS):
        folds = legacy.stratified_folds(y, 5, rng)
        preds = {name: np.full(n, np.nan) for name in by_model}
        for fold, test_pos in enumerate(folds):
            test_ids = [ids[int(ix)] for ix in test_pos]
            train_ids = [s for s in ids if s not in set(test_ids)]
            if set(test_ids) & set(train_ids):
                raise AssertionError("participant train/test overlap")
            tab, info = transform_fold(raw, ids, test_ids, spec, domain)
            tid = spec[0]
            train_rows = tab.set_index(tid).loc[train_ids].reset_index()
            test_rows = tab.set_index(tid).loc[test_ids].reset_index()
            family_columns = legacy.feature_sets(train_rows)
            row = {"dataset": domain, "repeat": repeat, "fold": fold,
                   "n_train": len(train_ids), "n_test": len(test_ids),
                   "test_id_sha256": hashlib.sha256(
                       ",".join(sorted(test_ids)).encode()).hexdigest(),
                   **info}
            for name in by_model:
                cols = [col for col in family_columns[name]
                        if col in train_rows and col in test_rows]
                if not cols:
                    raise ValueError(f"{domain}: no features for {name}")
                val, missing = score_fold(
                    train_rows, test_rows,
                    np.array([labels[s] for s in train_ids]), cols)
                preds[name][test_pos] = val
                row[name + "_features"] = len(cols)
                row[name + "_imputed_cells"] = missing
            fold_records.append(row)
        for name, scores in preds.items():
            if not np.isfinite(scores).all():
                raise ValueError(f"{domain} {name}: missing predictions")
            auc = legacy.auc_score(scores, y)
            by_model[name].append(float(auc))
        if repeat == 0:
            for j, subject in enumerate(ids):
                prediction_records.append({
                    "dataset": domain, "subject": subject,
                    "label": int(y[j]),
                    **{name: float(preds[name][j]) for name in by_model},
                })
        print(f"{domain} outer split repeat {repeat+1}/{REPEATS} done", flush=True)
    rows = []
    for name, aucs in by_model.items():
        hist = HISTORICAL[domain][name]
        rows.append({
            "dataset": domain, "model": name,
            "historic_mean_auc": hist,
            "nested_train_only_reference_mean_auc": float(np.mean(aucs)),
            "nested_auc_sd_across_repeat_splits": float(np.std(aucs)),
            "descriptive_difference_nested_minus_historic":
                float(np.mean(aucs) - hist),
            "repeat_aucs": aucs,
            "n_subjects": n,
            "n_repeats": REPEATS,
            "n_outer_folds": 5,
            "permutation_p": None,
        })
    return {
        "dataset": domain, "n_subjects": n,
        "n_controls": int(sum(y == 0)), "n_cases": int(sum(y == 1)),
        "raw_subject_context_hash": compact_hash(raw, spec[0], spec[1]),
        "source_metadata": {k: v for k, v in metadata.items()
                            if isinstance(v, (int, float, str, bool, type(None)))},
        "results": rows, "fold_audit": fold_records,
        "repeat0_predictions": prediction_records,
    }

def main():
    outputs = []
    for domain in ["ETDD70", "ASD"]:
        print(f"Starting leakage-controlled sensitivity: {domain}", flush=True)
        outputs.append(run_domain(domain))
    envelope = {
        "status": "NESTED_TRAIN_ONLY_REFERENCE_SENSITIVITY_COMPLETED",
        "historical_results_changed": False,
        "permutation_significance_claimed": False,
        "independent_validation_claimed": False,
        "CV_repeats": REPEATS,
        "random_seed": SEED,
        "source_module": "run_tairid_cross_neurotype_cycle_reach_axis_v1.py",
        "source_comparison_warning": (
            "Historical numbers used 80 outer repeats and different preprocessing; "
            "the 8-repeat nested replay is a sensitivity diagnostic, not a paired "
            "identically split hypothesis test. No historical p-values are carried over."),
        "results": outputs,
    }
    path = OUT / "sensitivity_summary.json"
    path.write_text(json.dumps(envelope, indent=2, default=str) + "\n")
    lines = [
        "# Nested train-only-reference sensitivity v1",
        "",
        "Status: exploratory leakage-control comparison only; original studies unchanged.",
        "All participant/task zscores, selected raw input columns and control-group "
        "reference SDs were fitted in the outer training fold.",
        "No new independent holdout, permutation p-value, neurotype validation, "
        "or clinical conclusions are claimed.",
        "",
        "| Dataset | Model | Frozen AUC | Train-only AUC | Delta (descriptive) |",
        "|---|---|---:|---:|---:|",
    ]
    for data in outputs:
        for row in data["results"]:
            lines.append(
                f"| {data['dataset']} | {row['model']} "
                f"| {row['historic_mean_auc']:.4f} "
                f"| {row['nested_train_only_reference_mean_auc']:.4f} "
                f"| {row['descriptive_difference_nested_minus_historic']:+.4f} |")
    lines += [
        "", "Caveats: comparison uses 8 outer CV repeats versus the original 80;",
        "sources and raw subject counts are checked; raw observations stay",
        "inside the executed workflow artifact. Differences also reflect",
        "split variation and are not solely attributable to leakage.",
        "Perform an independently preregistered full-pipeline nested permutation",
        "and stronger external validation before promoting any finding.",
    ]
    (OUT / "sensitivity_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(json.dumps({"status": envelope["status"],
                      "datasets": [{k:d[k] for k in ["dataset","n_subjects",
                                                     "n_controls","n_cases"]}
                                   for d in outputs]}, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
