#!/usr/bin/env python3
"""
TAIRID ADHD inhibition / selection cycle-reach test v1.

Purpose:
ETDD70 and ASD eye-tracking tests supported TAIRID axes:
T pacing, I constraint, M mismatch, W viability window, B breach, reach, and cycling.

This ADHD lane asks a sharper question:
Does ADHD task-performance data show stronger cycling / switching / reach limitation
than static level differences?

Dataset:
OpenNeuroDatasets/ds003500
Functional MRI task-performance events for selective attention and response inhibition.
This script uses the public GitHub mirror and reads events.tsv files only, not BOLD images.

Boundary:
This is not proof of TAIRID.
This is not clinical diagnosis.
This is not a cosmology result.
This is an operational axis test using public task-performance event tables.

Primary analysis:
child_only cohort:
- ADHD children
- age-matched non-ADHD children

Secondary analysis:
all participants:
- includes adult non-ADHD controls, useful as a stress check but not the primary ADHD comparison.

TAIRID translation:
T = pacing / response timing / response-time variability
I = constraint / inhibition pressure / error load / non-response / go/no-go mismatch
M = |T - I|
W = baseline viability window
B = breach outside W
Reach = stable span across task contexts before breach
Cycling = reversals, alternations, breach/re-entry, T/I opposition across task contexts
"""

import csv
import io
import json
import math
import re
import urllib.request
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


OUTDIR = Path("tairid_adhd_inhibition_cycle_reach_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_OWNER = "OpenNeuroDatasets"
GITHUB_REPO = "ds003500"
GITHUB_BRANCH = "master"
GITHUB_TREE_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}"

RANDOM_SEED = 42
CV_REPEATS = 100
PERMUTATIONS = 250
PERM_REPEATS = 25
RIDGE = 1.0e-3
TOLERANCE_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0]

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


def read_url_text(url, timeout=300):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-ADHD-inhibition-cycle-reach-v1",
            "Accept": "application/json, text/plain, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def safe_float(x):
    try:
        if pd.isna(x) or str(x).strip().lower() in ["n/a", "na", "nan", "none", ""]:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def download_repo_tree():
    text = read_url_text(GITHUB_TREE_URL)
    tree = json.loads(text)
    (OUTDIR / "github_tree_ds003500.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    return tree


def download_raw_file(path):
    url = f"{RAW_BASE}/{path}"
    local = DOWNLOAD_DIR / path.replace("/", "__")
    local.parent.mkdir(parents=True, exist_ok=True)

    text = read_url_text(url, timeout=300)
    local.write_text(text, encoding="utf-8")

    return text, local


def read_table_from_text(text):
    try:
        return pd.read_csv(io.StringIO(text), sep="\t")
    except Exception:
        return pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")


def load_participants():
    text, local = download_raw_file("participants.tsv")
    participants = read_table_from_text(text)

    # Robust fallback: GitHub preview sometimes displays whitespace-flattened TSV.
    if participants.shape[1] <= 1:
        participants = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")

    participants.columns = [str(c).strip() for c in participants.columns]

    for required in ["participant_id", "adhd"]:
        if required not in participants.columns:
            raise RuntimeError(f"participants.tsv missing required column: {required}")

    rows = []
    for _, r in participants.iterrows():
        pid = str(r["participant_id"]).strip()
        adhd = int(float(r["adhd"]))

        birth = pd.to_datetime(str(r.get("birthdate_shifted", "")), errors="coerce")
        a_date = pd.to_datetime(str(r.get("a_date", "")), errors="coerce")
        b_date = pd.to_datetime(str(r.get("b_date", "")), errors="coerce")

        age_a = np.nan
        if pd.notnull(birth) and pd.notnull(a_date):
            age_a = float((a_date - birth).days / 365.25)

        rows.append(
            {
                "participant_id": pid,
                "adhd": adhd,
                "label": adhd,
                "group": "adhd" if adhd == 1 else "non_adhd",
                "sex": safe_float(r.get("sex")),
                "handedness": safe_float(r.get("handedness")),
                "age_at_acq_a_years": age_a,
                "cohort_primary": "child" if np.isfinite(age_a) and age_a < 18 else "adult_or_unclear",
                "birthdate_shifted": str(r.get("birthdate_shifted", "")),
                "a_date": str(r.get("a_date", "")),
                "b_date": str(r.get("b_date", "")),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "participants_parsed.csv", index=False)

    return df


def list_event_paths(tree):
    paths = []
    for item in tree.get("tree", []):
        path = item.get("path", "")
        if path.endswith("_events.tsv") and "/func/" in path:
            paths.append(path)

    return sorted(paths)


def parse_task_from_path(path):
    m = re.search(r"_task-([^_]+)_", path)
    task = m.group(1) if m else "unknown"

    acq_match = re.search(r"_acq-([^_]+)_", path)
    acq = acq_match.group(1) if acq_match else "unknown"

    subject_match = re.search(r"(sub-[0-9A-Za-z]+)", path)
    subject = subject_match.group(1) if subject_match else "unknown"

    n = norm(task)

    feature_mode = "unknown"
    if n.startswith("feat"):
        feature_mode = "feature"
    elif n.startswith("conj"):
        feature_mode = "conjunction"

    paradigm = "unknown"
    if n.endswith("inh"):
        paradigm = "inhibition"
    elif n.endswith("sel"):
        paradigm = "selection"

    load = np.nan
    load_match = re.search(r"([0-9]+)", task)
    if load_match:
        load = float(load_match.group(1))

    # Task-demand index is a rough operational axis:
    # more items/features and selection/conjunction increase context pressure.
    demand = 0.0
    if np.isfinite(load):
        demand += math.log1p(load)
    if feature_mode == "conjunction":
        demand += 0.75
    if paradigm == "selection":
        demand += 0.50
    if paradigm == "inhibition":
        demand += 0.35

    return {
        "subject": subject,
        "task": task,
        "acq": acq,
        "feature_mode": feature_mode,
        "paradigm": paradigm,
        "load_n": load,
        "demand_index": demand,
    }


def block_type_code(value):
    s = str(value).strip().lower()
    if s in ["go", "respond", "response"]:
        return 0.0
    if s in ["no-go", "nogo", "no_go", "inhibit", "inhibition"]:
        return 1.0
    if s in ["nr", "nonresponse", "non-response"]:
        return 2.0
    return np.nan


def sign_change_count(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return 0

    d = np.diff(vals)
    signs = np.sign(d)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0

    return int(np.sum(signs[1:] != signs[:-1]))


def longest_true_run(flags):
    best = 0
    cur = 0
    for f in flags:
        if bool(f):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def summarize_values(vals, prefix):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    out = {}

    if len(vals) == 0:
        return out

    out[f"{prefix}_mean"] = float(np.mean(vals))
    out[f"{prefix}_median"] = float(np.median(vals))
    out[f"{prefix}_sum"] = float(np.sum(vals))
    out[f"{prefix}_max"] = float(np.max(vals))
    out[f"{prefix}_min"] = float(np.min(vals))

    if len(vals) >= 2:
        out[f"{prefix}_std"] = float(np.std(vals))
        out[f"{prefix}_cv"] = float(np.std(vals) / max(abs(np.mean(vals)), 1.0e-9))
        out[f"{prefix}_range"] = float(np.max(vals) - np.min(vals))
        out[f"{prefix}_turns"] = sign_change_count(vals)
        x = np.arange(len(vals), dtype=float)
        try:
            out[f"{prefix}_block_slope"] = float(np.polyfit(x, vals, 1)[0])
        except Exception:
            out[f"{prefix}_block_slope"] = np.nan

    return out


def parse_events_file(path, participant_lookup):
    task_meta = parse_task_from_path(path)
    text, local = download_raw_file(path)
    df = read_table_from_text(text)
    df.columns = [str(c).strip() for c in df.columns]

    subject = task_meta["subject"]
    if subject not in participant_lookup:
        return None, {"path": path, "status": "subject_not_in_participants", **task_meta}

    p = participant_lookup[subject]

    row = {
        "source_path": path,
        "local_file": str(local),
        "participant_id": subject,
        "group": p["group"],
        "label": int(p["label"]),
        "age_at_acq_a_years": p["age_at_acq_a_years"],
        "cohort_primary": p["cohort_primary"],
        **task_meta,
        "event_rows": int(len(df)),
    }

    numeric_cols = [
        "duration",
        "response_time_avg",
        "correct_total",
        "errors_total",
        "false-no_go",
        "false-go",
        "NR",
    ]

    for col in numeric_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).values
            row.update(summarize_values(vals, col.replace("-", "_")))

    correct = pd.to_numeric(df.get("correct_total", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(float)
    errors = pd.to_numeric(df.get("errors_total", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(float)
    false_nogo = pd.to_numeric(df.get("false-no_go", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(float)
    false_go = pd.to_numeric(df.get("false-go", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(float)
    nr = pd.to_numeric(df.get("NR", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(float)

    denom = correct + errors + false_nogo + false_go + nr
    denom = denom.replace(0, np.nan)

    row["total_correct"] = float(np.nansum(correct.values))
    row["total_errors"] = float(np.nansum(errors.values))
    row["total_false_nogo"] = float(np.nansum(false_nogo.values))
    row["total_false_go"] = float(np.nansum(false_go.values))
    row["total_nr"] = float(np.nansum(nr.values))
    row["total_response_opportunities_proxy"] = float(np.nansum(denom.values))

    total_denom = row["total_response_opportunities_proxy"]
    row["error_rate_proxy"] = float((row["total_errors"] + row["total_false_nogo"] + row["total_false_go"] + row["total_nr"]) / max(total_denom, 1.0))
    row["commission_error_proxy"] = float(row["total_false_go"] / max(total_denom, 1.0))
    row["omission_or_nr_proxy"] = float((row["total_false_nogo"] + row["total_nr"]) / max(total_denom, 1.0))

    if "block_type_intended" in df.columns and "block_type_performed" in df.columns:
        intended = df["block_type_intended"].astype(str).str.lower()
        performed = df["block_type_performed"].astype(str).str.lower()
        mismatch = intended != performed
        row["block_type_mismatch_fraction"] = float(np.mean(mismatch))
        row["block_type_mismatch_count"] = int(np.sum(mismatch))

        intended_code = np.asarray([block_type_code(v) for v in intended], dtype=float)
        performed_code = np.asarray([block_type_code(v) for v in performed], dtype=float)

        row["intended_block_switches"] = int(np.sum(np.diff(intended_code[np.isfinite(intended_code)]) != 0)) if np.sum(np.isfinite(intended_code)) > 1 else 0
        row["performed_block_switches"] = int(np.sum(np.diff(performed_code[np.isfinite(performed_code)]) != 0)) if np.sum(np.isfinite(performed_code)) > 1 else 0
        row["performed_block_turns"] = sign_change_count(performed_code)

    if "comments" in df.columns:
        comments = df["comments"].astype(str).str.lower()
        row["comment_nr_fraction"] = float(np.mean(comments.str.contains("non") | comments.str.contains("nr")))

    return row, {"path": path, "status": "parsed", **task_meta, "columns": " | ".join(df.columns)}


def build_context_features(event_rows):
    if not event_rows:
        return pd.DataFrame()

    df = pd.DataFrame(event_rows)

    # Aggregate possible duplicate acq/task rows by participant-task-acq.
    group_cols = [
        "participant_id",
        "group",
        "label",
        "age_at_acq_a_years",
        "cohort_primary",
        "task",
        "acq",
        "feature_mode",
        "paradigm",
        "load_n",
        "demand_index",
    ]

    numeric_cols = [
        c for c in df.columns
        if c not in group_cols + ["source_path", "local_file", "subject"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    out = df.groupby(group_cols, dropna=False)[numeric_cols].median().reset_index()
    out["context"] = (
        out["task"].astype(str)
        + "_acq-"
        + out["acq"].astype(str)
    )

    return out


def zscore_by_context(df, cols):
    out = df.copy()

    for c in cols:
        zc = c + "__z_by_context"
        out[zc] = np.nan

        for context, sub in out.groupby("context"):
            vals = pd.to_numeric(sub[c], errors="coerce")
            mu = vals.mean()
            sd = vals.std(ddof=0)

            if not np.isfinite(sd) or sd <= 1.0e-12:
                out.loc[sub.index, zc] = 0.0
            else:
                out.loc[sub.index, zc] = (vals - mu) / sd

    return out


def choose_axis_columns(context_df):
    numeric_cols = [
        c for c in context_df.columns
        if c not in [
            "participant_id", "group", "label", "age_at_acq_a_years", "cohort_primary",
            "task", "acq", "feature_mode", "paradigm", "load_n", "demand_index", "context"
        ]
        and pd.api.types.is_numeric_dtype(context_df[c])
        and context_df[c].notnull().sum() >= 10
    ]

    pacing_hints = [
        "response_time_avg",
        "duration",
        "cv",
        "block_slope",
    ]

    constraint_hints = [
        "error",
        "false",
        "nr",
        "mismatch",
        "commission",
        "omission",
        "switch",
        "turn",
    ]

    pacing_cols = [c for c in numeric_cols if any(h in c.lower() for h in pacing_hints)]
    constraint_cols = [c for c in numeric_cols if any(h in c.lower() for h in constraint_hints)]

    if not pacing_cols and numeric_cols:
        pacing_cols = numeric_cols[: max(1, len(numeric_cols) // 3)]
    if not constraint_cols and numeric_cols:
        constraint_cols = numeric_cols[max(1, len(numeric_cols) // 3):] or numeric_cols

    return numeric_cols, pacing_cols[:40], constraint_cols[:40]


def build_tairid_context_shape(context_df):
    if context_df.empty:
        return pd.DataFrame(), {}

    numeric_cols, pacing_cols, constraint_cols = choose_axis_columns(context_df)

    if not numeric_cols:
        return pd.DataFrame(), {
            "numeric_cols_count": 0,
            "pacing_cols": [],
            "constraint_cols": [],
        }

    df = zscore_by_context(context_df, numeric_cols)

    pacing_z = [c + "__z_by_context" for c in pacing_cols if c + "__z_by_context" in df.columns]
    constraint_z = [c + "__z_by_context" for c in constraint_cols if c + "__z_by_context" in df.columns]

    df["T_pacing_proxy"] = df[pacing_z].mean(axis=1)
    df["I_constraint_proxy"] = df[constraint_z].mean(axis=1)
    df["M_mismatch_abs"] = np.abs(df["T_pacing_proxy"] - df["I_constraint_proxy"])
    df["collapse_load_proxy"] = np.sqrt(df["T_pacing_proxy"] ** 2 + df["I_constraint_proxy"] ** 2)
    df["interaction_TI"] = df["T_pacing_proxy"] * df["I_constraint_proxy"]

    meta = {
        "numeric_cols_count": len(numeric_cols),
        "pacing_cols_count": len(pacing_cols),
        "constraint_cols_count": len(constraint_cols),
        "pacing_cols": pacing_cols,
        "constraint_cols": constraint_cols,
    }

    return df, meta


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


def safe_curve(x, y):
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


def sign_turns(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return 0

    d = np.diff(vals)
    signs = np.sign(d)
    signs = signs[signs != 0]

    if len(signs) < 2:
        return 0

    return int(np.sum(signs[1:] != signs[:-1]))


def build_subject_axis_features(shape_df):
    if shape_df.empty:
        return pd.DataFrame(), {}

    rows = []

    # Use non-ADHD child baseline mismatch as population window scale.
    child_df = shape_df[shape_df["cohort_primary"] == "child"].copy()
    low_demand = child_df.groupby("participant_id")["M_mismatch_abs"].quantile(0.25).dropna()
    child_control_ids = child_df[child_df["label"] == 0]["participant_id"].unique().tolist()
    control_low = low_demand[low_demand.index.isin(child_control_ids)].values

    if len(control_low) >= 2:
        control_baseline_sd = float(np.std(control_low, ddof=1))
    else:
        control_baseline_sd = float(shape_df["M_mismatch_abs"].std(ddof=1))

    if not np.isfinite(control_baseline_sd) or control_baseline_sd <= 1.0e-12:
        control_baseline_sd = 0.0

    for pid, sub in shape_df.groupby("participant_id"):
        sub = sub.sort_values(["demand_index", "task", "acq"])

        if len(sub) < 3:
            continue

        group_vals = sub["group"].dropna().unique()
        group = group_vals[0] if len(group_vals) else None

        label_vals = sub["label"].dropna().unique()
        label = int(label_vals[0]) if len(label_vals) else None

        cohort_vals = sub["cohort_primary"].dropna().unique()
        cohort = cohort_vals[0] if len(cohort_vals) else "unknown"

        x = sub["demand_index"].astype(float).values
        T = sub["T_pacing_proxy"].astype(float).values
        I = sub["I_constraint_proxy"].astype(float).values
        M = sub["M_mismatch_abs"].astype(float).values
        C = sub["collapse_load_proxy"].astype(float).values

        baseline = float(np.percentile(M, 25))

        dT = np.diff(T)
        dI = np.diff(I)
        dM = np.diff(M)
        dC = np.diff(C)

        base = {
            "participant_id": pid,
            "group": group,
            "label": label,
            "cohort_primary": cohort,
            "context_count": int(len(sub)),
            "demand_min": float(np.min(x)),
            "demand_max": float(np.max(x)),
            "demand_span": float(np.max(x) - np.min(x)),
            "M_mean": float(np.mean(M)),
            "M_median": float(np.median(M)),
            "M_max": float(np.max(M)),
            "M_range": float(np.max(M) - np.min(M)),
            "M_iqr": float(np.percentile(M, 75) - np.percentile(M, 25)),
            "M_std": float(np.std(M)),
            "M_slope": safe_slope(x, M),
            "M_curvature": safe_curve(x, M),
            "T_mean": float(np.mean(T)),
            "I_mean": float(np.mean(I)),
            "T_slope": safe_slope(x, T),
            "I_slope": safe_slope(x, I),
            "T_I_slope_gap": float(abs(safe_slope(x, T) - safe_slope(x, I))) if np.isfinite(safe_slope(x, T)) and np.isfinite(safe_slope(x, I)) else np.nan,
            "collapse_load_mean": float(np.mean(C)),
            "collapse_load_max": float(np.max(C)),
            "collapse_load_range": float(np.max(C) - np.min(C)),
            "collapse_load_slope": safe_slope(x, C),
            "context_baseline_M_q25": baseline,
            "control_child_baseline_sd_used": control_baseline_sd,
            "Cyc_M_derivative_turns": sign_turns(M),
            "Cyc_T_derivative_turns": sign_turns(T),
            "Cyc_I_derivative_turns": sign_turns(I),
            "Cyc_C_derivative_turns": sign_turns(C),
            "Cyc_M_derivative_turn_rate": float(sign_turns(M) / max(len(M) - 2, 1)),
            "Cyc_TI_opposition_fraction": float(np.mean(np.sign(dT) * np.sign(dI) < 0)) if len(dT) and len(dI) else np.nan,
            "Reach_M_span": float(np.max(M) - np.min(M)),
            "Reach_C_span": float(np.max(C) - np.min(C)),
            "Reach_T_span": float(np.max(T) - np.min(T)),
            "Reach_I_span": float(np.max(I) - np.min(I)),
            "Reach_TI_vector_span": float(math.sqrt((np.max(T) - np.min(T)) ** 2 + (np.max(I) - np.min(I)) ** 2)),
        }

        for mult in TOLERANCE_MULTIPLIERS:
            suffix = f"tol{str(mult).replace('.', 'p')}"
            W = baseline + mult * control_baseline_sd
            B = np.maximum(0.0, M - W)
            breached = B > 0.0
            stable = ~breached

            base[f"W_{suffix}"] = float(W)
            base[f"B_total_{suffix}"] = float(np.sum(B))
            base[f"B_mean_{suffix}"] = float(np.mean(B))
            base[f"B_max_{suffix}"] = float(np.max(B))
            base[f"B_fraction_{suffix}"] = float(np.mean(breached))
            base[f"B_count_{suffix}"] = int(np.sum(breached))
            base[f"B_slope_{suffix}"] = safe_slope(x, B)
            base[f"B_curvature_{suffix}"] = safe_curve(x, B)

            base[f"Cyc_breach_transition_count_{suffix}"] = int(np.sum(np.abs(np.diff(breached.astype(int))))) if len(breached) > 1 else 0
            base[f"Cyc_breach_reentry_count_{suffix}"] = int(np.sum((breached[:-1] == True) & (breached[1:] == False))) if len(breached) > 1 else 0

            base[f"Reach_stable_fraction_{suffix}"] = float(np.mean(stable))
            base[f"Reach_breach_fraction_{suffix}"] = float(np.mean(breached))
            base[f"Reach_longest_stable_run_{suffix}"] = longest_true_run(stable)
            base[f"Reach_longest_breach_run_{suffix}"] = longest_true_run(breached)

            stable_x = x[stable]
            base[f"Reach_highest_stable_demand_{suffix}"] = float(np.max(stable_x)) if len(stable_x) else 0.0

        rows.append(base)

    meta = {
        "subject_rows": len(rows),
        "control_child_baseline_sd": control_baseline_sd,
    }

    return pd.DataFrame(rows), meta


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
        cov = np.atleast_2d(cov) + np.eye(X_train.shape[1]) * RIDGE
        w = np.linalg.pinv(cov, rcond=1.0e-8) @ (mu1 - mu0)

    b = -0.5 * float((mu1 + mu0) @ w)
    return X_test @ w + b


def repeated_cv(df, feature_cols, repeats=CV_REPEATS, y_override=None):
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return {"status": "no_features", "n": 0, "feature_cols": []}

    data = df[feature_cols + ["label"]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 12 or data["label"].nunique() < 2:
        return {"status": "not_enough_data", "n": int(len(data)), "feature_cols": feature_cols}

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
        return {"status": "not_enough_data", "n_perm": 0, "p_value_ge_observed": None}

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
        return {"status": "no_valid_permutations", "n_perm": 0, "p_value_ge_observed": None}

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


def model_feature_sets(df):
    static = [
        "M_mean",
        "M_max",
        "collapse_load_mean",
        "collapse_load_max",
        "T_mean",
        "I_mean",
    ]

    response = [
        "M_range",
        "M_iqr",
        "M_std",
        "M_slope",
        "M_curvature",
        "T_I_slope_gap",
        "collapse_load_range",
        "collapse_load_slope",
    ]

    viability = []
    cycling = [
        "Cyc_M_derivative_turns",
        "Cyc_T_derivative_turns",
        "Cyc_I_derivative_turns",
        "Cyc_C_derivative_turns",
        "Cyc_M_derivative_turn_rate",
        "Cyc_TI_opposition_fraction",
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
        viability.extend(
            [
                f"B_total_{suffix}",
                f"B_mean_{suffix}",
                f"B_max_{suffix}",
                f"B_fraction_{suffix}",
                f"B_slope_{suffix}",
                f"B_curvature_{suffix}",
            ]
        )
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
                f"Reach_highest_stable_demand_{suffix}",
            ]
        )

    return {
        "static_level_model": [c for c in static if c in df.columns],
        "response_dynamic_model": [c for c in response if c in df.columns],
        "viability_window_model": [c for c in viability if c in df.columns],
        "cycling_model": [c for c in cycling if c in df.columns],
        "reach_model": [c for c in reach if c in df.columns],
        "cycling_reach_model": [c for c in cycling + reach if c in df.columns],
        "combined_axis_model": [c for c in static + response + viability + cycling + reach if c in df.columns],
    }


def run_models(df, cohort_name):
    rows = []
    perms = []

    for name, cols in model_feature_sets(df).items():
        res = repeated_cv(df, cols)
        auc = res.get("auc_mean")

        rows.append(
            {
                "cohort": cohort_name,
                "model_name": name,
                **{k: v for k, v in res.items() if k != "feature_cols"},
                "feature_cols": " | ".join(res.get("feature_cols", [])),
            }
        )

        perm = permutation_test(df, cols, observed_auc=auc)
        perms.append(
            {
                "cohort": cohort_name,
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


def feature_tests(df, cohort_name):
    rows = []

    numeric_cols = [
        c for c in df.columns
        if c not in ["participant_id", "group", "label", "cohort_primary"]
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

        rows.append(
            {
                "cohort": cohort_name,
                "feature": c,
                "adhd_mean": float(np.mean(pos)),
                "non_adhd_mean": float(np.mean(neg)),
                "cohen_d_adhd_minus_nonadhd": d,
                "welch_t": float(tstat),
                "welch_p": float(pval),
                "oriented_auc": auc_score(scores, labels),
                "n_adhd": int(len(pos)),
                "n_non_adhd": int(len(neg)),
            }
        )

    return sorted(
        rows,
        key=lambda r: (
            -(abs(r["cohen_d_adhd_minus_nonadhd"]) if np.isfinite(r["cohen_d_adhd_minus_nonadhd"]) else 0),
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


def plot_model_auc(model_rows, path):
    ok = [r for r in model_rows if r.get("status") == "ok" and r.get("auc_mean") is not None]
    if not ok:
        return None

    labels = [f"{r['cohort']}\n{r['model_name'].replace('_model', '')}" for r in ok]
    vals = [float(r["auc_mean"]) for r in ok]
    x = np.arange(len(labels))

    plt.figure(figsize=(18, 7))
    plt.bar(x, vals)
    plt.axhline(0.5, linewidth=1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Repeated CV AUC")
    plt.title("TAIRID ADHD inhibition cycle/reach v1")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def plot_axis_advantage(model_rows, path):
    by = {}
    for r in model_rows:
        if r.get("status") == "ok" and r.get("auc_mean") is not None:
            by.setdefault(r["cohort"], {})[r["model_name"]] = float(r["auc_mean"])

    labels = []
    vals = []
    for cohort, m in by.items():
        static = m.get("static_level_model")
        axis = max(
            [
                m.get("cycling_model", np.nan),
                m.get("reach_model", np.nan),
                m.get("cycling_reach_model", np.nan),
                m.get("combined_axis_model", np.nan),
            ]
        )
        if static is not None and np.isfinite(axis):
            labels.append(cohort)
            vals.append(axis - static)

    if not labels:
        return None

    x = np.arange(len(labels))
    plt.figure(figsize=(9, 6))
    plt.bar(x, vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("Best cycle/reach/combined AUC - static AUC")
    plt.title("ADHD cycle/reach advantage over static level")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def decide_status(model_rows, perm_rows):
    by_model = {}
    by_perm = {}

    for r in model_rows:
        by_model.setdefault(r["cohort"], {})[r["model_name"]] = r
    for r in perm_rows:
        by_perm.setdefault(r["cohort"], {})[r["model_name"]] = r

    cohort_status = {}

    for cohort, m in by_model.items():
        static_auc = m.get("static_level_model", {}).get("auc_mean")

        axis_names = ["cycling_model", "reach_model", "cycling_reach_model", "combined_axis_model"]
        axis_aucs = [
            m.get(name, {}).get("auc_mean")
            for name in axis_names
            if m.get(name, {}).get("auc_mean") is not None
        ]
        best_axis_auc = max(axis_aucs) if axis_aucs else None

        axis_ps = [
            by_perm.get(cohort, {}).get(name, {}).get("p_value_ge_observed")
            for name in axis_names
            if by_perm.get(cohort, {}).get(name, {}).get("p_value_ge_observed") is not None
        ]
        best_axis_p = min(axis_ps) if axis_ps else None

        cycling_auc = m.get("cycling_model", {}).get("auc_mean")
        reach_auc = m.get("reach_model", {}).get("auc_mean")

        axis_support = (
            static_auc is not None
            and best_axis_auc is not None
            and best_axis_auc >= 0.60
            and best_axis_auc > static_auc + 0.03
        )
        axis_locked = bool(axis_support and best_axis_p is not None and best_axis_p <= 0.05)

        cohort_status[cohort] = {
            "static_auc": static_auc,
            "cycling_auc": cycling_auc,
            "reach_auc": reach_auc,
            "best_axis_auc": best_axis_auc,
            "best_axis_permutation_p": best_axis_p,
            "axis_support": axis_support,
            "axis_locked": axis_locked,
        }

    primary = cohort_status.get("child_only", {})
    if primary.get("axis_locked") and primary.get("cycling_auc", 0) >= primary.get("static_auc", 1) + 0.03:
        return (
            "adhd_child_cycle_reach_axis_locked",
            9,
            "Add ADHD as the cycling/reach stress-test lane in the cross-neurotype axis map.",
            cohort_status,
        )

    if primary.get("axis_support"):
        return (
            "adhd_child_cycle_reach_axis_supported_not_locked",
            8,
            "Treat ADHD as supportive for cycling/reach, but inspect feature tests before locking.",
            cohort_status,
        )

    if any(v.get("axis_support") for v in cohort_status.values()):
        return (
            "adhd_cycle_reach_directional_in_secondary_cohort",
            7,
            "Signal appears outside the primary child-only lane; check adult-control confounding.",
            cohort_status,
        )

    return (
        "adhd_cycle_reach_not_yet_supported",
        6,
        "Keep ADHD lane exploratory; inspect parser, task ordering, and event columns.",
        cohort_status,
    )


def main():
    print("")
    print("TAIRID ADHD inhibition / selection cycle-reach test v1 starting.")
    print("Boundary: operational axis test only; not proof or diagnosis.")
    print("")

    tree = download_repo_tree()
    participants = load_participants()
    participant_lookup = {r["participant_id"]: r for _, r in participants.iterrows()}

    event_paths = list_event_paths(tree)

    event_rows = []
    inventory = []

    for path in event_paths:
        try:
            row, inv = parse_events_file(path, participant_lookup)
            inventory.append(inv)
            if row is not None:
                event_rows.append(row)
        except Exception as exc:
            inventory.append({"path": path, "status": "parse_failed", "error": str(exc)})

    write_csv(OUTDIR / "adhd_events_inventory.csv", inventory)
    write_csv(OUTDIR / "adhd_event_context_raw_features.csv", event_rows)

    context_df = build_context_features(event_rows)
    if not context_df.empty:
        context_df.to_csv(OUTDIR / "adhd_context_features.csv", index=False)

    shape_df, axis_meta = build_tairid_context_shape(context_df)
    if not shape_df.empty:
        shape_df.to_csv(OUTDIR / "adhd_tairid_context_shape_features.csv", index=False)

    subject_df, subject_meta = build_subject_axis_features(shape_df)
    if not subject_df.empty:
        subject_df.to_csv(OUTDIR / "adhd_subject_cycle_reach_features.csv", index=False)

    cohorts = {
        "child_only": subject_df[subject_df["cohort_primary"] == "child"].copy() if not subject_df.empty else pd.DataFrame(),
        "all_participants": subject_df.copy() if not subject_df.empty else pd.DataFrame(),
    }

    all_model_rows = []
    all_perm_rows = []
    all_feature_rows = []

    for cohort_name, df in cohorts.items():
        if df.empty:
            continue

        model_rows, perm_rows = run_models(df, cohort_name)
        feats = add_bh_q(feature_tests(df, cohort_name))

        all_model_rows.extend(model_rows)
        all_perm_rows.extend(perm_rows)
        all_feature_rows.extend(feats)

        write_csv(OUTDIR / f"adhd_{cohort_name}_model_results.csv", model_rows)
        write_csv(OUTDIR / f"adhd_{cohort_name}_permutation_results.csv", perm_rows)
        write_csv(OUTDIR / f"adhd_{cohort_name}_feature_tests_bh_fdr.csv", feats)

    model_path = write_csv(OUTDIR / "adhd_cycle_reach_model_results.csv", all_model_rows)
    perm_path = write_csv(OUTDIR / "adhd_cycle_reach_permutation_results.csv", all_perm_rows)
    feature_path = write_csv(OUTDIR / "adhd_cycle_reach_feature_tests_bh_fdr.csv", all_feature_rows)

    plots = []
    p = plot_model_auc(all_model_rows, OUTDIR / "adhd_cycle_reach_model_auc.png")
    if p:
        plots.append(p)

    p = plot_axis_advantage(all_model_rows, OUTDIR / "adhd_cycle_reach_axis_advantage.png")
    if p:
        plots.append(p)

    final_status, readiness_score, next_wall, cohort_axis_status = decide_status(all_model_rows, all_perm_rows)

    group_counts_child = {}
    group_counts_all = {}
    if not subject_df.empty:
        child = subject_df[subject_df["cohort_primary"] == "child"]
        group_counts_child = {str(k): int(v) for k, v in Counter(child["group"]).items()}
        group_counts_all = {str(k): int(v) for k, v in Counter(subject_df["group"]).items()}

    summary = {
        "test_name": "TAIRID ADHD inhibition / selection cycle-reach test v1",
        "boundary": (
            "Operational ADHD axis test only. Not proof of TAIRID, not diagnosis, and not a cosmology result."
        ),
        "dataset": {
            "github_repo": f"{GITHUB_OWNER}/{GITHUB_REPO}",
            "github_tree_url": GITHUB_TREE_URL,
            "raw_base": RAW_BASE,
            "note": "Uses events.tsv task-performance tables only, not BOLD images.",
        },
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "parser_counts": {
            "participants": int(len(participants)),
            "event_paths_found": int(len(event_paths)),
            "event_context_rows": int(len(event_rows)),
            "context_feature_rows": int(len(context_df)) if not context_df.empty else 0,
            "shape_context_rows": int(len(shape_df)) if not shape_df.empty else 0,
            "subject_axis_rows": int(len(subject_df)) if not subject_df.empty else 0,
            "child_group_counts": group_counts_child,
            "all_group_counts": group_counts_all,
        },
        "axis_meta": axis_meta,
        "subject_meta": subject_meta,
        "cohort_axis_status": cohort_axis_status,
        "model_results": all_model_rows,
        "permutation_results": all_perm_rows,
        "top_feature_tests": all_feature_rows[:30],
        "output_files": {
            "participants_parsed_csv": str(OUTDIR / "participants_parsed.csv"),
            "events_inventory_csv": str(OUTDIR / "adhd_events_inventory.csv"),
            "event_context_raw_features_csv": str(OUTDIR / "adhd_event_context_raw_features.csv"),
            "context_features_csv": str(OUTDIR / "adhd_context_features.csv"),
            "tairid_context_shape_features_csv": str(OUTDIR / "adhd_tairid_context_shape_features.csv"),
            "subject_cycle_reach_features_csv": str(OUTDIR / "adhd_subject_cycle_reach_features.csv"),
            "model_results_csv": str(model_path),
            "permutation_results_csv": str(perm_path),
            "feature_tests_bh_fdr_csv": str(feature_path),
            "plots": plots,
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "Cycling/reach or combined axis features outperform static level in the child-only ADHD comparison "
                "and beat permutation expectation."
            ),
            "what_weakens_this_lane": (
                "The signal appears only when adult controls are included, or static level beats cycling/reach."
            ),
            "axis_prediction": (
                "ADHD should stress cycling, breach transition, T/I opposition, and stable reach across inhibition/selection contexts."
            ),
            "truth_boundary": (
                "A positive result supports ADHD as a cycling/reach stress-test lane. It cannot prove TAIRID or diagnose ADHD."
            ),
        },
    }

    summary_path = OUTDIR / "adhd_inhibition_cycle_reach_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(OUTDIR / "adhd_inhibition_cycle_reach_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID ADHD inhibition / selection cycle-reach test v1\n\n")
        f.write("Boundary: operational axis test only. Not proof. Not diagnosis. Not cosmology result.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Why this test exists:\n")
        f.write("- ETDD70 and ASD supported mismatch / viability / reach axes.\n")
        f.write("- ADHD should be a stronger stress test for cycling and reach.\n")
        f.write("- This pass uses public ds003500 events.tsv task-performance tables.\n\n")

        f.write("Cohort axis status:\n")
        f.write(json.dumps(cohort_axis_status, indent=2) + "\n\n")

        f.write("Model results:\n")
        f.write(json.dumps(all_model_rows, indent=2) + "\n\n")

        f.write("Permutation results:\n")
        f.write(json.dumps(all_perm_rows, indent=2) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This can support ADHD as a cycling/reach stress-test lane.\n")
        f.write("- It cannot prove TAIRID.\n")
        f.write("- It cannot diagnose ADHD.\n")
        f.write("- It cannot prove any cosmology claim.\n")

    print("")
    print("TAIRID ADHD inhibition / selection cycle-reach test v1 complete.")
    print("Created:")
    print("  tairid_adhd_inhibition_cycle_reach_v1_outputs/adhd_inhibition_cycle_reach_v1_summary.json")
    print("  tairid_adhd_inhibition_cycle_reach_v1_outputs/adhd_inhibition_cycle_reach_v1_summary.txt")
    print("  tairid_adhd_inhibition_cycle_reach_v1_outputs/adhd_cycle_reach_model_results.csv")
    print("  tairid_adhd_inhibition_cycle_reach_v1_outputs/adhd_cycle_reach_permutation_results.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is not clinical diagnosis.")
    print("  This is an ADHD operational axis test.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
