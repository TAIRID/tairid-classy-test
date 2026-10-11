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

OUT = Path("tairid_neurotype_paired_reference_sensitivity_v2_outputs")
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

def transform_fold(raw, ids, test_ids, spec, domain, reference_scope="train_only"):
    id_col, context_col, control, case, eng = spec
    test_set = set(test_ids)
    train_set = set(ids) - test_set
    train = raw[raw[id_col].isin(train_set)].copy()
    assert reference_scope in ("train_only", "full_cohort")
    reference = train if reference_scope == "train_only" else raw
    if domain == "ETDD70":
        numcols, pacing_cols, constraint_cols = eng.choose_role_columns(reference)
        suffix = "__z_by_task"
    else:
        numcols, pacing_cols, constraint_cols = eng.choose_role_columns(
            reference, family_filter="all")
        suffix = "__z_by_context"
    if not numcols or not pacing_cols or not constraint_cols:
        raise ValueError(f"{domain}: no usable role columns in training fold")
    full = raw.copy()
    # Paired counterfactual: same fold, intentionally full-cohort reference in
    # legacy-like arm vs legitimate train-only reference in corrected arm.
    for col in numcols:
        zcol = col + suffix
        full[zcol] = np.nan
        for context in sorted(full[context_col].dropna().unique().tolist()):
            maskref = reference[context_col] == context
            sample = pd.to_numeric(reference.loc[maskref, col], errors="coerce")
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
    if reference_scope == "train_only":
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
        "reference_scope": reference_scope,
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
    rng = np.random.default_rng(SEED)
    labels = dict(zip(ids, y))
    method_names = ["train_only", "full_cohort"]
    family_names = list(HISTORICAL[domain])
    outcomes = {mode:{name:[] for name in family_names} for mode in method_names}
    fold_records = []
    first_repeat = []
    for repeat in range(REPEATS):
        folds = legacy.stratified_folds(y, 5, rng)
        preds = {mode:{name:np.full(len(ids), np.nan) for name in family_names}
                 for mode in method_names}
        for fold, test_pos in enumerate(folds):
            test_ids = [ids[int(x)] for x in test_pos]
            train_ids = [subject for subject in ids if subject not in set(test_ids)]
            if set(train_ids).intersection(test_ids):
                raise AssertionError("subject overlap")
            for mode in method_names:
                tab, info = transform_fold(raw, ids, test_ids, spec, domain, mode)
                index_col = spec[0]
                train_rows = tab.set_index(index_col).loc[train_ids].reset_index()
                test_rows = tab.set_index(index_col).loc[test_ids].reset_index()
                family_columns = legacy.feature_sets(train_rows)
                item = {"dataset":domain, "repeat":repeat, "fold":fold,
                        "reference_scope":mode,
                        "test_id_sha256":hashlib.sha256(
                            ",".join(sorted(test_ids)).encode()).hexdigest(),
                        **info}
                for name in family_names:
                    cols = [c for c in family_columns[name]
                            if c in train_rows and c in test_rows]
                    if not cols:
                        raise ValueError(f"no columns: {domain} {mode} {name}")
                    values, missing = score_fold(
                        train_rows, test_rows,
                        np.asarray([labels[s] for s in train_ids],dtype=int),cols)
                    preds[mode][name][test_pos] = values
                    item[name+"_nfeatures"] = len(cols)
                    item[name+"_missing"] = missing
                fold_records.append(item)
        for mode in method_names:
            for name in family_names:
                if not np.isfinite(preds[mode][name]).all():
                    raise ValueError(f"missing predictions: {domain} {mode} {name}")
                outcomes[mode][name].append(float(legacy.auc_score(preds[mode][name],y)))
        if repeat == 0:
            for j,subject in enumerate(ids):
                first_repeat.append(
                    {"dataset":domain,"subject":subject,"label":int(y[j]),
                    "models":{mode:{name:float(preds[mode][name][j])
                                     for name in family_names}
                              for mode in method_names}})
        print(f"Paired {domain} {repeat+1}/{REPEATS} complete",flush=True)
    rows=[]
    for name in family_names:
        train=np.asarray(outcomes["train_only"][name], dtype=float)
        full=np.asarray(outcomes["full_cohort"][name], dtype=float)
        paired=train-full
        # Repeated CV repeats are not independent samples. Descriptive only.
        rows.append({"dataset":domain,"model":name,
                     "train_only_mean_auc":float(train.mean()),
                     "full_cohort_mean_auc":float(full.mean()),
                     "train_minus_full_paired_mean":float(paired.mean()),
                     "train_minus_full_min":float(paired.min()),
                     "train_minus_full_max":float(paired.max()),
                     "train_only_per_repeat_auc":train.tolist(),
                     "full_cohort_per_repeat_auc":full.tolist(),
                     "paired_per_repeat_deltas":paired.tolist(),
                     "historic_unpaired_mean_auc":HISTORICAL[domain][name],
                     "permutation_p":None,"confidence_interval":None})
    return {"dataset":domain,"n_subjects":len(ids),
            "n_controls":int(sum(y==0)),"n_cases":int(sum(y==1)),
            "raw_subject_context_hash":compact_hash(raw,spec[0],spec[1]),
            "results":rows,"fold_audit":fold_records,
            "repeat0_predictions":first_repeat}


def main():
    outputs=[]
    for domain in ("ETDD70","ASD"):
        print(f"Paired calibration scope audit begins {domain}",flush=True)
        outputs.append(run_domain(domain))
    bundle={"status":"PAIRED_SAME_SPLIT_REFERENCE_SCOPE_SENSITIVITY_COMPLETED",
            "version":"NREF-2-v0.1", "seed":SEED,
            "outer_repeats":REPEATS, "outer_folds":5,
            "same_subject_folds_both_arms":True,
            "models_retrained_each_fold":True,
            "historical_unaltered":True,
            "neither_arm_independent_external_holdout":True,
            "full_cohort_arm_uses_heldout_data_and_labels_for_reference":True,
            "permutation_significance":None,
            "ci_interpretation":"No CI; repeat splits not independent units",
            "note":("This is an exploratory paired sensitivity, not a clean "
                    "replication of historical 80-repeat results, nor a "
                    "validated neurotype model. Both arms share same "
                    "train/test split and downstream LDA/imputation."),
            "datasets":outputs}
    (OUT/"paired_summary.json").write_text(json.dumps(bundle,indent=2,default=str)+"\n")
    lines=["# NREF-2 paired same-split reference-scope sensitivity",
           "","Status: exploratory paired sensitivity, no p-value or CI.",
           "Full-cohort arm is deliberately contaminated for diagnostic comparison.",
           "",
           "| Dataset | Model | Full-cohort AUC | Train-only AUC | Train minus full |",
           "|---|---|---:|---:|---:|"]
    for d in outputs:
        for x in d["results"]:
            lines.append(
                f"| {d['dataset']} | {x['model']} "
                f"| {x['full_cohort_mean_auc']:.4f} "
                f"| {x['train_only_mean_auc']:.4f} "
                f"| {x['train_minus_full_paired_mean']:+.4f} |")
    lines+=["","Both arms use identical participants, folds, predictor families,",
            "and LDA/imputation. Only upstream reference scope changes.",
            "The full-cohort arm explicitly sees held-out subject data and",
            "group labels for reference construction; it is NOT a valid",
            "predictive validation. Reference scope can also change",
            "which feature columns exist in some folds.",
            "Historical 80-repeat AUCs are retained for orientation only.",
            "No full-pipeline label permutations were executed.",
            "Repeat-level differences are descriptives, not independent CIs."]
    (OUT/"paired_summary.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    return 0

if __name__=="__main__":
    sys.exit(main())
