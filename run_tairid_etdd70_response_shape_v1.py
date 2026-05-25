#!/usr/bin/env python3
"""
TAIRID ETDD70 response-shape test v1.

Purpose:
The SH0ES compact-ladder tests taught us that a simple TAIRID gate/offset is too easily
absorbed by ordinary calibration and nuisance freedom. The next translation should not
look for a simple mean shift. It should look for a response shape.

This test moves to neuroscience-style data and asks whether the TAIRID
Pacing-Constraint Mismatch Response shape is visible in ETDD70 eye-tracking data.

Dataset:
ETDD70 Eye-Tracking Dyslexia Dataset, Zenodo record 13332134.
The dataset contains eye-tracking recordings from 70 Czech children, 35 dyslexic and
35 non-dyslexic readers, during three reading tasks.

Question:
Does a pacing/constraint response-shape model carry group-relevant signal beyond simple
mean differences?

TAIRID translation:
- T / pacing proxy:
  eye-movement tempo, fixation timing, sample cadence, saccade timing, reading progression.
- I / constraint proxy:
  task demand, regression/backtracking, fixation dispersion, sequence instability.
- O / stable output proxy:
  lower variability, smoother progression, lower mismatch.
- SDR / mismatch proxy:
  deviation between pacing and constraint response across task demand.

This is not a clinical diagnostic model.
This is not proof of TAIRID.
This is a first cross-domain response-shape test.
"""

import csv
import io
import json
import math
import os
import re
import zipfile
import tarfile
import hashlib
import urllib.request
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
from scipy.special import expit


OUTDIR = Path("tairid_etdd70_response_shape_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_DIR = OUTDIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD_ID = "13332134"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TASK_ORDER_HINTS = {
    "task1": 1,
    "syll": 1,
    "syllable": 1,
    "slab": 1,
    "task4": 2,
    "meaning": 2,
    "text": 2,
    "story": 2,
    "task5": 3,
    "pseudo": 3,
    "pseudotext": 3,
}

GROUP_DYS_HINTS = ["dys", "dyslex", "diagnosed", "yes", "true", "1"]
GROUP_CONTROL_HINTS = ["control", "td", "typical", "non", "no", "false", "0", "healthy"]

COLUMN_HINTS = {
    "subject": [
        "subject", "participant", "subj", "id", "user", "person", "child", "student", "pupil"
    ],
    "group": [
        "group", "class", "label", "diagnosis", "dyslexia", "reader", "type", "condition"
    ],
    "task": [
        "task", "text", "stimulus", "trial", "condition", "reading"
    ],
    "time": [
        "time", "timestamp", "t", "ms", "sec", "second"
    ],
    "x": [
        "x", "gaze_x", "gazex", "fix_x", "fixx", "posx", "screen_x"
    ],
    "y": [
        "y", "gaze_y", "gazey", "fix_y", "fixy", "posy", "screen_y"
    ],
    "duration": [
        "duration", "dur", "fixation_duration", "fixdur", "fix_dur", "dwell"
    ],
    "saccade": [
        "saccade", "sacc", "amplitude", "velocity", "sacamp"
    ],
    "event": [
        "event", "type", "event_type", "label"
    ],
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_url(url, timeout=240):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-ETDD70-response-shape-v1",
            "Accept": "application/json, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        final_url = response.geturl()
        status = getattr(response, "status", None)
        content_type = response.headers.get("Content-Type", "")
    return data, final_url, status, content_type


def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    return name[:180]


def download_zenodo_record():
    data, final_url, status, content_type = read_url(ZENODO_API_URL)
    record = json.loads(data.decode("utf-8"))

    record_path = OUTDIR / "zenodo_record_13332134.json"
    record_path.write_text(json.dumps(record, indent=2))

    files = record.get("files", [])
    downloads = []

    for f in files:
        key = f.get("key") or f.get("filename") or "unknown"
        links = f.get("links", {})
        url = links.get("self") or links.get("download")
        size = f.get("size")

        if not url:
            downloads.append({"key": key, "status": "no_download_url", "size": size})
            continue

        local_path = DOWNLOAD_DIR / safe_name(key)

        try:
            print(f"Downloading {key} ...")
            payload, got_url, got_status, got_content_type = read_url(url, timeout=600)
            local_path.write_bytes(payload)
            downloads.append(
                {
                    "key": key,
                    "status": "downloaded",
                    "url": url,
                    "final_url": got_url,
                    "content_type": got_content_type,
                    "size_declared": size,
                    "size_downloaded": local_path.stat().st_size,
                    "path": str(local_path),
                    "sha256": sha256_file(local_path),
                }
            )
        except Exception as exc:
            downloads.append(
                {
                    "key": key,
                    "status": "download_failed",
                    "url": url,
                    "size_declared": size,
                    "error": str(exc),
                }
            )

    return record, downloads


def extract_archives(downloads):
    extracted = []

    for item in downloads:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["path"])
        lower = path.name.lower()

        if lower.endswith(".zip"):
            target = EXTRACT_DIR / path.stem
            target.mkdir(parents=True, exist_ok=True)

            try:
                with zipfile.ZipFile(path, "r") as z:
                    z.extractall(target)
                extracted.append({"archive": str(path), "status": "zip_extracted", "target": str(target)})
            except Exception as exc:
                extracted.append({"archive": str(path), "status": "zip_extract_failed", "error": str(exc)})

        elif lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
            target = EXTRACT_DIR / path.stem.replace(".tar", "")
            target.mkdir(parents=True, exist_ok=True)

            try:
                with tarfile.open(path, "r:*") as t:
                    t.extractall(target)
                extracted.append({"archive": str(path), "status": "tar_extracted", "target": str(target)})
            except Exception as exc:
                extracted.append({"archive": str(path), "status": "tar_extract_failed", "error": str(exc)})

    return extracted


def list_candidate_tables():
    candidates = []

    roots = [DOWNLOAD_DIR, EXTRACT_DIR]

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            lower = path.name.lower()

            if lower.endswith((".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls")):
                candidates.append(path)

    return sorted(set(candidates))


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_column(columns, role):
    cols = list(columns)
    normalized = {c: norm(c) for c in cols}

    for hint in COLUMN_HINTS.get(role, []):
        h = norm(hint)
        for c, cn in normalized.items():
            if h and h == cn:
                return c

    for hint in COLUMN_HINTS.get(role, []):
        h = norm(hint)
        for c, cn in normalized.items():
            if h and h in cn:
                return c

    return None


def read_table(path, max_rows=None):
    lower = path.name.lower()

    try:
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            df = pd.read_excel(path, nrows=max_rows)
        elif lower.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", nrows=max_rows)
        elif lower.endswith(".csv"):
            df = pd.read_csv(path, nrows=max_rows)
        else:
            try:
                df = pd.read_csv(path, sep=None, engine="python", nrows=max_rows)
            except Exception:
                df = pd.read_csv(path, sep=r"\s+", engine="python", nrows=max_rows)
    except Exception as exc:
        return None, str(exc)

    if df is None or df.empty:
        return None, "empty"

    df.columns = [str(c).strip() for c in df.columns]
    return df, None


def infer_task_from_path_or_value(path, value=None):
    text = f"{path} {value if value is not None else ''}".lower()

    for key, rank in TASK_ORDER_HINTS.items():
        if key in text:
            return rank

    nums = re.findall(r"task[_ -]?([0-9]+)", text)
    if nums:
        n = int(nums[0])
        if n <= 1:
            return 1
        if n <= 4:
            return 2
        return 3

    return None


def normalize_group(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None

    s = str(value).strip().lower()

    if not s:
        return None

    if any(h in s for h in ["dys", "dyslex"]):
        return "dyslexic"

    if s in ["1", "true", "yes", "y"]:
        return "dyslexic"

    if any(h in s for h in ["control", "typical", "td", "non", "healthy"]):
        return "control"

    if s in ["0", "false", "no", "n"]:
        return "control"

    return None


def infer_subject_from_path(path):
    parts = [p.lower() for p in Path(path).parts]
    joined = "/".join(parts)

    patterns = [
        r"sub[-_]?([0-9A-Za-z]+)",
        r"subject[-_]?([0-9A-Za-z]+)",
        r"participant[-_]?([0-9A-Za-z]+)",
        r"pupil[-_]?([0-9A-Za-z]+)",
    ]

    for pat in patterns:
        m = re.search(pat, joined)
        if m:
            return m.group(1)

    stem = Path(path).stem
    m = re.search(r"([0-9]{2,4})", stem)
    if m:
        return m.group(1)

    return stem[:40]


def numeric_series(df, col):
    if col is None or col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def table_inventory_and_features():
    table_paths = list_candidate_tables()
    inventory = []
    feature_rows = []

    for path in table_paths:
        df, err = read_table(path)

        if df is None:
            inventory.append(
                {
                    "path": str(path),
                    "status": "read_failed",
                    "error": err,
                }
            )
            continue

        cols = list(df.columns)
        sub_col = find_column(cols, "subject")
        group_col = find_column(cols, "group")
        task_col = find_column(cols, "task")
        time_col = find_column(cols, "time")
        x_col = find_column(cols, "x")
        y_col = find_column(cols, "y")
        duration_col = find_column(cols, "duration")
        sacc_col = find_column(cols, "saccade")
        event_col = find_column(cols, "event")

        subject = None
        if sub_col and sub_col in df.columns:
            nonnull = df[sub_col].dropna()
            if len(nonnull):
                subject = str(nonnull.iloc[0])
        if subject is None:
            subject = infer_subject_from_path(path)

        group = None
        if group_col and group_col in df.columns:
            vals = df[group_col].dropna().astype(str)
            if len(vals):
                counts = Counter(normalize_group(v) for v in vals)
                counts.pop(None, None)
                if counts:
                    group = counts.most_common(1)[0][0]

        if group is None:
            group = normalize_group(str(path))

        task = None
        if task_col and task_col in df.columns:
            vals = df[task_col].dropna().astype(str)
            if len(vals):
                task = infer_task_from_path_or_value(path, vals.iloc[0])
        if task is None:
            task = infer_task_from_path_or_value(path)

        n = len(df)

        time = numeric_series(df, time_col)
        x = numeric_series(df, x_col)
        y = numeric_series(df, y_col)
        dur = numeric_series(df, duration_col)
        sacc = numeric_series(df, sacc_col)

        row = {
            "source_file": str(path),
            "subject": subject,
            "group": group,
            "task_rank": task,
            "row_count": int(n),
            "columns": " | ".join(cols[:80]),
            "subject_col": sub_col,
            "group_col": group_col,
            "task_col": task_col,
            "time_col": time_col,
            "x_col": x_col,
            "y_col": y_col,
            "duration_col": duration_col,
            "saccade_col": sacc_col,
            "event_col": event_col,
            "usable_score": 0,
        }

        if group is not None:
            row["usable_score"] += 2
        if task is not None:
            row["usable_score"] += 2
        if subject is not None:
            row["usable_score"] += 1
        if duration_col is not None:
            row["usable_score"] += 2
        if time_col is not None:
            row["usable_score"] += 1
        if x_col is not None and y_col is not None:
            row["usable_score"] += 2
        if event_col is not None:
            row["usable_score"] += 1

        inventory.append(row)

        # Build feature row when there is at least subject and some numeric signal.
        features = {
            "source_file": str(path),
            "subject": subject,
            "group": group,
            "task_rank": task,
            "n_rows": int(n),
        }

        got_any_signal = False

        if dur is not None:
            d = dur.dropna().astype(float).values
            if d.size >= 5:
                got_any_signal = True
                features.update(
                    {
                        "duration_mean": float(np.mean(d)),
                        "duration_median": float(np.median(d)),
                        "duration_std": float(np.std(d)),
                        "duration_cv": float(np.std(d) / max(abs(np.mean(d)), 1.0e-9)),
                        "duration_iqr": float(np.percentile(d, 75) - np.percentile(d, 25)),
                    }
                )

        if time is not None:
            t = time.dropna().astype(float).values
            if t.size >= 5:
                dt = np.diff(np.sort(t))
                dt = dt[np.isfinite(dt)]
                dt = dt[dt > 0]
                if dt.size >= 3:
                    got_any_signal = True
                    features.update(
                        {
                            "time_delta_mean": float(np.mean(dt)),
                            "time_delta_std": float(np.std(dt)),
                            "time_delta_cv": float(np.std(dt) / max(abs(np.mean(dt)), 1.0e-9)),
                            "sample_rate_proxy": float(1.0 / max(np.median(dt), 1.0e-9)),
                        }
                    )

        if x is not None and y is not None:
            xy = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(xy) >= 5:
                xv = xy["x"].astype(float).values
                yv = xy["y"].astype(float).values
                dx = np.diff(xv)
                dy = np.diff(yv)
                step = np.sqrt(dx * dx + dy * dy)
                step = step[np.isfinite(step)]
                got_any_signal = True
                features.update(
                    {
                        "x_std": float(np.std(xv)),
                        "y_std": float(np.std(yv)),
                        "xy_dispersion": float(np.sqrt(np.var(xv) + np.var(yv))),
                        "path_step_mean": float(np.mean(step)) if step.size else None,
                        "path_step_std": float(np.std(step)) if step.size else None,
                        "path_step_cv": float(np.std(step) / max(abs(np.mean(step)), 1.0e-9)) if step.size else None,
                        "backward_step_fraction": float(np.mean(dx < 0)) if dx.size else None,
                    }
                )

        if sacc is not None:
            sv = sacc.dropna().astype(float).values
            if sv.size >= 5:
                got_any_signal = True
                features.update(
                    {
                        "saccade_mean": float(np.mean(sv)),
                        "saccade_std": float(np.std(sv)),
                        "saccade_cv": float(np.std(sv) / max(abs(np.mean(sv)), 1.0e-9)),
                    }
                )

        if got_any_signal:
            feature_rows.append(features)

    return inventory, feature_rows


def aggregate_subject_task_features(feature_rows):
    if not feature_rows:
        return pd.DataFrame()

    df = pd.DataFrame(feature_rows)

    # Keep only rows with group and task if possible.
    usable = df.copy()

    if "group" in usable.columns:
        usable = usable[usable["group"].isin(["dyslexic", "control"])]

    if "task_rank" in usable.columns:
        usable = usable[pd.notnull(usable["task_rank"])]

    if usable.empty:
        return pd.DataFrame()

    numeric_cols = [
        c for c in usable.columns
        if c not in ["source_file", "subject", "group"]
        and pd.api.types.is_numeric_dtype(usable[c])
    ]

    group_cols = ["subject", "group", "task_rank"]

    agg = usable.groupby(group_cols, dropna=False)[numeric_cols].median().reset_index()

    return agg


def zscore_by_task(df, cols):
    out = df.copy()
    for c in cols:
        zc = c + "_z_by_task"
        out[zc] = np.nan
        for task, sub in out.groupby("task_rank"):
            vals = pd.to_numeric(sub[c], errors="coerce")
            mu = vals.mean()
            sd = vals.std(ddof=0)
            if not np.isfinite(sd) or sd <= 1.0e-12:
                out.loc[sub.index, zc] = 0.0
            else:
                out.loc[sub.index, zc] = (vals - mu) / sd
    return out


def build_tairid_shape_features(agg):
    if agg.empty:
        return pd.DataFrame(), []

    df = agg.copy()

    candidate_cols = [
        "duration_median",
        "duration_mean",
        "duration_cv",
        "duration_iqr",
        "time_delta_cv",
        "xy_dispersion",
        "path_step_cv",
        "backward_step_fraction",
        "saccade_cv",
        "saccade_std",
    ]

    usable_cols = [c for c in candidate_cols if c in df.columns and df[c].notnull().sum() >= 10]

    if not usable_cols:
        return pd.DataFrame(), []

    df = zscore_by_task(df, usable_cols)

    zcols = [c + "_z_by_task" for c in usable_cols]

    # TAIRID proxies:
    # Pacing load: timing/fixation duration features.
    pacing_parts = []
    for c in ["duration_median", "duration_mean", "time_delta_cv"]:
        zc = c + "_z_by_task"
        if zc in df.columns:
            pacing_parts.append(zc)

    # Constraint load: variability, dispersion, backward/regression-style features.
    constraint_parts = []
    for c in ["duration_cv", "duration_iqr", "xy_dispersion", "path_step_cv", "backward_step_fraction", "saccade_cv", "saccade_std"]:
        zc = c + "_z_by_task"
        if zc in df.columns:
            constraint_parts.append(zc)

    if not pacing_parts:
        pacing_parts = zcols[: max(1, len(zcols) // 2)]
    if not constraint_parts:
        constraint_parts = zcols[max(1, len(zcols) // 2):] or zcols

    df["T_pacing_proxy"] = df[pacing_parts].mean(axis=1)
    df["I_constraint_proxy"] = df[constraint_parts].mean(axis=1)

    df["mismatch_abs"] = np.abs(df["T_pacing_proxy"] - df["I_constraint_proxy"])
    df["collapse_load_proxy"] = np.sqrt(df["T_pacing_proxy"] ** 2 + df["I_constraint_proxy"] ** 2)
    df["interaction_TI"] = df["T_pacing_proxy"] * df["I_constraint_proxy"]

    # Subject-level response-shape across task demand.
    subject_rows = []

    for subject, sub in df.groupby("subject"):
        if len(sub) < 2:
            continue

        sub = sub.sort_values("task_rank")
        group_vals = sub["group"].dropna().unique()
        group = group_vals[0] if len(group_vals) else None

        tasks = sub["task_rank"].astype(float).values
        T = sub["T_pacing_proxy"].astype(float).values
        I = sub["I_constraint_proxy"].astype(float).values
        M = sub["mismatch_abs"].astype(float).values
        C = sub["collapse_load_proxy"].astype(float).values

        def slope(vals):
            if len(vals) < 2:
                return np.nan
            try:
                return float(np.polyfit(tasks, vals, 1)[0])
            except Exception:
                return np.nan

        def curvature(vals):
            if len(vals) < 3:
                return np.nan
            try:
                return float(np.polyfit(tasks, vals, 2)[0])
            except Exception:
                return np.nan

        subject_rows.append(
            {
                "subject": subject,
                "group": group,
                "task_count": int(len(sub)),
                "T_slope_over_task": slope(T),
                "I_slope_over_task": slope(I),
                "mismatch_slope_over_task": slope(M),
                "collapse_load_slope_over_task": slope(C),
                "mismatch_curvature_over_task": curvature(M),
                "T_range": float(np.max(T) - np.min(T)),
                "I_range": float(np.max(I) - np.min(I)),
                "mismatch_range": float(np.max(M) - np.min(M)),
                "collapse_load_range": float(np.max(C) - np.min(C)),
                "T_I_slope_gap": float(abs(slope(T) - slope(I))) if np.isfinite(slope(T)) and np.isfinite(slope(I)) else np.nan,
                "mean_mismatch": float(np.mean(M)),
                "max_mismatch": float(np.max(M)),
                "mean_collapse_load": float(np.mean(C)),
            }
        )

    subject_df = pd.DataFrame(subject_rows)

    return df, subject_df


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / max(len(a)+len(b)-2, 1))
    if pooled <= 1.0e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def bootstrap_auc(x, y, n_boot=500):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan

    def auc_score(scores, labels):
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        ranks = stats.rankdata(np.concatenate([pos, neg]))
        rpos = np.sum(ranks[:len(pos)])
        auc = (rpos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
        return float(auc)

    base = auc_score(x, y)
    boots = []
    n = len(x)

    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(auc_score(x[idx], y[idx]))

    if not boots:
        return base, np.nan, np.nan

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return base, float(lo), float(hi)


def simple_logistic_cv(df, feature_cols, label_col="label", k=5):
    data = df[feature_cols + [label_col]].dropna()
    if len(data) < 20 or data[label_col].nunique() < 2:
        return {"status": "not_enough_data", "n": len(data)}

    X = data[feature_cols].astype(float).values
    y = data[label_col].astype(int).values

    # Standardize.
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd <= 1.0e-12] = 1.0
    X = (X - mu) / sd

    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    folds = np.array_split(idx, k)
    preds = np.zeros(len(y), dtype=float)

    for fold in folds:
        train = np.setdiff1d(idx, fold)
        Xt = X[train]
        yt = y[train]

        # Ridge logistic by simple gradient descent.
        w = np.zeros(X.shape[1])
        b = 0.0
        lr = 0.05
        lam = 0.1

        for _ in range(800):
            z = Xt @ w + b
            p = expit(z)
            grad_w = Xt.T @ (p - yt) / len(yt) + lam * w
            grad_b = float(np.mean(p - yt))
            w -= lr * grad_w
            b -= lr * grad_b

        preds[fold] = expit(X[fold] @ w + b)

    auc, lo, hi = bootstrap_auc(preds, y, n_boot=500)
    acc = float(np.mean((preds >= 0.5).astype(int) == y))

    return {
        "status": "ok",
        "n": int(len(y)),
        "feature_cols": feature_cols,
        "auc": auc,
        "auc_ci_2p5": lo,
        "auc_ci_97p5": hi,
        "accuracy_threshold_0p5": acc,
    }


def analyze_response_shape(subject_df, task_df):
    results = {
        "status": "started",
        "feature_tests": [],
        "models": {},
    }

    if subject_df.empty:
        results["status"] = "no_subject_shape_features"
        return results

    df = subject_df.copy()
    df = df[df["group"].isin(["dyslexic", "control"])]
    df["label"] = (df["group"] == "dyslexic").astype(int)

    if df.empty or df["label"].nunique() < 2:
        results["status"] = "missing_groups"
        return results

    shape_cols = [
        "T_slope_over_task",
        "I_slope_over_task",
        "mismatch_slope_over_task",
        "collapse_load_slope_over_task",
        "mismatch_curvature_over_task",
        "T_I_slope_gap",
        "mean_mismatch",
        "max_mismatch",
        "mismatch_range",
        "mean_collapse_load",
    ]

    shape_cols = [c for c in shape_cols if c in df.columns and df[c].notnull().sum() >= 10]

    for c in shape_cols:
        dys = df[df["label"] == 1][c].dropna()
        con = df[df["label"] == 0][c].dropna()
        if len(dys) < 3 or len(con) < 3:
            continue

        tstat, pval = stats.ttest_ind(dys, con, equal_var=False, nan_policy="omit")
        d = cohen_d(dys, con)

        # Orient AUC so higher score means dyslexic-like if dys mean > control mean.
        scores = df[c].astype(float).values
        if np.nanmean(dys) < np.nanmean(con):
            scores = -scores

        auc, lo, hi = bootstrap_auc(scores, df["label"].values, n_boot=500)

        results["feature_tests"].append(
            {
                "feature": c,
                "dyslexic_mean": float(np.mean(dys)),
                "control_mean": float(np.mean(con)),
                "cohen_d_dys_minus_control": d,
                "welch_t": float(tstat),
                "welch_p": float(pval),
                "oriented_auc": auc,
                "auc_ci_2p5": lo,
                "auc_ci_97p5": hi,
                "n_dyslexic": int(len(dys)),
                "n_control": int(len(con)),
            }
        )

    # Compare simple static mean-load model to response-shape model.
    static_cols = [c for c in ["mean_mismatch", "mean_collapse_load", "max_mismatch"] if c in shape_cols]
    dynamic_cols = [c for c in shape_cols if c not in static_cols]

    results["models"]["static_level_model"] = simple_logistic_cv(df, static_cols) if static_cols else {"status": "no_static_cols"}
    results["models"]["dynamic_shape_model"] = simple_logistic_cv(df, dynamic_cols) if dynamic_cols else {"status": "no_dynamic_cols"}
    results["models"]["combined_shape_model"] = simple_logistic_cv(df, shape_cols) if shape_cols else {"status": "no_shape_cols"}

    results["status"] = "ok"
    results["subject_count"] = int(len(df))
    results["group_counts"] = {str(k): int(v) for k, v in Counter(df["group"]).items()}
    results["shape_cols_used"] = shape_cols

    return results


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return path

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path


def plot_outputs(task_df, subject_df, analysis):
    plots = []

    if not task_df.empty and "T_pacing_proxy" in task_df.columns:
        plt.figure(figsize=(10, 6))
        for group, sub in task_df.groupby("group"):
            means = sub.groupby("task_rank")["mismatch_abs"].mean()
            plt.plot(means.index, means.values, marker="o", label=str(group))
        plt.xlabel("Task rank / demand proxy")
        plt.ylabel("Mean |T - I| mismatch proxy")
        plt.title("TAIRID ETDD70 response shape v1: mismatch across task demand")
        plt.legend()
        plt.tight_layout()
        path = OUTDIR / "etdd70_mismatch_across_task.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plots.append(str(path))

        plt.figure(figsize=(10, 6))
        for group, sub in task_df.groupby("group"):
            means = sub.groupby("task_rank")["collapse_load_proxy"].mean()
            plt.plot(means.index, means.values, marker="o", label=str(group))
        plt.xlabel("Task rank / demand proxy")
        plt.ylabel("Mean collapse-load proxy")
        plt.title("TAIRID ETDD70 response shape v1: collapse load across task demand")
        plt.legend()
        plt.tight_layout()
        path = OUTDIR / "etdd70_collapse_load_across_task.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plots.append(str(path))

    if not subject_df.empty and "mismatch_slope_over_task" in subject_df.columns:
        plt.figure(figsize=(10, 6))
        groups = []
        values = []
        for group, sub in subject_df.groupby("group"):
            vals = sub["mismatch_slope_over_task"].dropna().values
            if len(vals):
                groups.append(str(group))
                values.append(vals)
        if values:
            plt.boxplot(values, labels=groups)
            plt.ylabel("Mismatch slope over task demand")
            plt.title("TAIRID ETDD70 response shape v1: subject-level mismatch slope")
            plt.tight_layout()
            path = OUTDIR / "etdd70_subject_mismatch_slope_boxplot.png"
            plt.savefig(path, dpi=160)
            plt.close()
            plots.append(str(path))

    return plots


def main():
    print("")
    print("TAIRID ETDD70 response-shape test v1 starting.")
    print("Boundary: cross-domain response-shape test only; not proof.")
    print("")

    record, downloads = download_zenodo_record()
    extraction = extract_archives(downloads)

    inventory, feature_rows = table_inventory_and_features()

    inventory_path = write_csv(OUTDIR / "etdd70_table_inventory.csv", inventory)
    features_path = write_csv(OUTDIR / "etdd70_raw_feature_rows.csv", feature_rows)

    subject_task = aggregate_subject_task_features(feature_rows)
    if not subject_task.empty:
        subject_task_path = OUTDIR / "etdd70_subject_task_features.csv"
        subject_task.to_csv(subject_task_path, index=False)
    else:
        subject_task_path = None

    task_shape_df, subject_shape_df = build_tairid_shape_features(subject_task)

    if not task_shape_df.empty:
        task_shape_path = OUTDIR / "etdd70_tairid_task_shape_features.csv"
        task_shape_df.to_csv(task_shape_path, index=False)
    else:
        task_shape_path = None

    if not subject_shape_df.empty:
        subject_shape_path = OUTDIR / "etdd70_tairid_subject_response_shape.csv"
        subject_shape_df.to_csv(subject_shape_path, index=False)
    else:
        subject_shape_path = None

    analysis = analyze_response_shape(subject_shape_df, task_shape_df)

    analysis_path = OUTDIR / "etdd70_response_shape_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2))

    plots = plot_outputs(task_shape_df, subject_shape_df, analysis)

    # Decide status.
    if analysis.get("status") != "ok":
        final_status = "data_inventory_complete_response_shape_not_estimated"
        readiness_score = 5
        next_wall = "Inspect downloaded files and column map; update parser for ETDD70 file structure."
    else:
        models = analysis.get("models", {})
        dyn_auc = models.get("dynamic_shape_model", {}).get("auc")
        static_auc = models.get("static_level_model", {}).get("auc")
        combined_auc = models.get("combined_shape_model", {}).get("auc")

        if dyn_auc is not None and static_auc is not None and dyn_auc > static_auc + 0.05:
            final_status = "dynamic_response_shape_outperforms_static_level_proxy"
            readiness_score = 8
            next_wall = "Replicate on a second neurotype or task-switching dataset."
        elif combined_auc is not None and combined_auc >= 0.70:
            final_status = "response_shape_contains_group_relevant_signal"
            readiness_score = 7
            next_wall = "Check robustness and compare to simpler features."
        else:
            final_status = "response_shape_measurable_but_not_strongly_discriminative"
            readiness_score = 6
            next_wall = "Refine T/I proxies or use richer event-level parsing."

    summary = {
        "test_name": "TAIRID ETDD70 response-shape test v1",
        "boundary": (
            "Cross-domain response-shape test only. Not clinical diagnosis, not proof of TAIRID, "
            "and not a cosmology result."
        ),
        "dataset": {
            "zenodo_record_id": ZENODO_RECORD_ID,
            "zenodo_api_url": ZENODO_API_URL,
            "record_title": record.get("metadata", {}).get("title"),
            "record_description_first_500": str(record.get("metadata", {}).get("description", ""))[:500],
        },
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "downloads": downloads,
        "extraction": extraction,
        "table_inventory_count": len(inventory),
        "raw_feature_row_count": len(feature_rows),
        "subject_task_feature_count": int(len(subject_task)) if not subject_task.empty else 0,
        "subject_response_shape_count": int(len(subject_shape_df)) if not subject_shape_df.empty else 0,
        "analysis": analysis,
        "output_files": {
            "table_inventory_csv": str(inventory_path),
            "raw_feature_rows_csv": str(features_path),
            "subject_task_features_csv": str(subject_task_path) if subject_task_path else None,
            "task_shape_features_csv": str(task_shape_path) if task_shape_path else None,
            "subject_response_shape_csv": str(subject_shape_path) if subject_shape_path else None,
            "analysis_json": str(analysis_path),
            "plots": plots,
        },
        "interpretation": {
            "what_would_support_TAIRID_shape": (
                "Dynamic mismatch/curvature/slope features carry signal beyond static mean-load proxies."
            ),
            "what_would_weaken_this_translation": (
                "Only simple mean differences appear, or parser cannot recover task-level timing/constraint structure."
            ),
            "translation_rule": (
                "Do not translate objects across domains. Translate operator roles: pacing, constraint, mismatch, "
                "stability, boundary transition."
            ),
        },
    }

    summary_path = OUTDIR / "etdd70_response_shape_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    with open(OUTDIR / "etdd70_response_shape_v1_summary.txt", "w") as f:
        f.write("TAIRID ETDD70 response-shape test v1\n\n")
        f.write("Boundary: cross-domain response-shape test only. Not clinical diagnosis. Not proof.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("What this tests:\n")
        f.write("- Whether pacing/constraint mismatch dynamics are measurable in eye-tracking data.\n")
        f.write("- Whether dynamic response-shape features outperform simple static level proxies.\n")
        f.write("- Whether TAIRID translation is useful as a testable operator map, not a metaphor.\n\n")

        f.write("What this does not prove:\n")
        f.write("- It does not prove TAIRID.\n")
        f.write("- It does not diagnose dyslexia.\n")
        f.write("- It does not prove any cosmology claim.\n")
        f.write("- It does not show brains and cosmology are the same object.\n\n")

        f.write("Analysis:\n")
        f.write(json.dumps(analysis, indent=2) + "\n\n")

    print("")
    print("TAIRID ETDD70 response-shape test v1 complete.")
    print("Created:")
    print("  tairid_etdd70_response_shape_v1_outputs/etdd70_response_shape_v1_summary.json")
    print("  tairid_etdd70_response_shape_v1_outputs/etdd70_response_shape_v1_summary.txt")
    print("  tairid_etdd70_response_shape_v1_outputs/etdd70_table_inventory.csv")
    print("  tairid_etdd70_response_shape_v1_outputs/etdd70_raw_feature_rows.csv")
    print("  tairid_etdd70_response_shape_v1_outputs/etdd70_response_shape_analysis.json")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is not a clinical diagnostic result.")
    print("  This is a cross-domain response-shape test.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
