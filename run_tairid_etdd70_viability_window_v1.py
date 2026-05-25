#!/usr/bin/env python3
"""
TAIRID ETDD70 viability-window breach test v1.

Purpose:
The SH0ES compact-ladder tests showed that simple offset/gate translations are too easily
absorbed by ordinary model freedom. The ETDD70 response-shape pass suggested that a better
TAIRID translation should look for dynamic pacing/constraint mismatch under task demand.

This test adds the TAIRID viability window.

Core question:
When reading-task demand increases, does each reader's pacing/constraint relationship
remain inside a recoverable viability window, or does it breach that window?

TAIRID translation:
- T / pacing proxy:
  timing, fixation duration, reading tempo, movement/saccade timing.
- I / constraint proxy:
  fixation variability, regression/backtracking, dispersion, instability, count/load.
- M / mismatch:
  |T - I| after task-normalization.
- W / viability window:
  subject baseline mismatch on the easiest task plus a tolerance term.
- B / breach:
  max(0, M_task - W_subject).

Dataset:
ETDD70 Eye-Tracking Dyslexia Dataset, Zenodo record 13332134.

Important parser corrections:
- "non-dyslexic" must map to control before checking for "dyslexic".
- Use exact label file when available.
- Prefer subject identifiers like sid/subject_id over generic row id.
- Use Subject_*_metrics.csv files for this first clean pass.
- Ignore __MACOSX artifacts.

Boundary:
This is not proof of TAIRID.
This is not clinical diagnosis.
This is not a cosmology result.
This is a cross-domain viability-window response-shape test.
"""

import csv
import json
import math
import re
import zipfile
import tarfile
import hashlib
import urllib.request
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit


OUTDIR = Path("tairid_etdd70_viability_window_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_DIR = OUTDIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD_ID = "13332134"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TASK_NAME_TO_RANK = {
    "t1_syllables": 1,
    "task1_syllables": 1,
    "syllables": 1,
    "syllable": 1,
    "slabiky": 1,
    "t4_meaningful_text": 2,
    "task4_meaningful_text": 2,
    "meaningful_text": 2,
    "meaningful": 2,
    "text": 2,
    "t5_pseudo_text": 3,
    "task5_pseudo_text": 3,
    "pseudo_text": 3,
    "pseudotext": 3,
    "pseudo": 3,
}

TASK_RANK_TO_NAME = {
    1: "T1_Syllables",
    2: "T4_Meaningful_Text",
    3: "T5_Pseudo_Text",
}

TOLERANCE_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0]

SUBJECT_COL_PREFERENCE = [
    "sid",
    "subject_id",
    "subjectid",
    "subject",
    "participant_id",
    "participant",
    "subj",
    "child_id",
    "child",
    "student_id",
    "pupil_id",
]

AVOID_SUBJECT_COLS = {
    "id",
    "fixation_id",
    "saccade_id",
    "event_id",
    "trial_id",
    "item_id",
    "word_id",
    "row_id",
    "index",
    "unnamed:0",
}

GROUP_COL_HINTS = [
    "group",
    "label",
    "class",
    "diagnosis",
    "dyslexia",
    "reader",
    "type",
    "condition",
]

TASK_COL_HINTS = [
    "task",
    "text",
    "stimulus",
    "trial",
    "condition",
    "reading",
]

# Pacing/load names: slower time, longer fixation, larger reading-time values are treated
# as higher pacing pressure/load. These are standardized by task before building T.
PACING_NAME_HINTS = [
    "duration",
    "dur",
    "time",
    "latency",
    "fixation",
    "fix",
    "dwell",
    "readingtime",
    "reading_time",
    "rt",
    "tempo",
    "speed",
    "velocity",
]

# Constraint/instability names: dispersion, regression, counts, variability, amplitude,
# and backtracking-like fields are treated as constraint/instability pressure.
CONSTRAINT_NAME_HINTS = [
    "regression",
    "regress",
    "back",
    "return",
    "refix",
    "refixation",
    "count",
    "num",
    "number",
    "n_",
    "std",
    "sd",
    "var",
    "variance",
    "cv",
    "dispersion",
    "spread",
    "saccade",
    "sacc",
    "amplitude",
    "path",
    "error",
    "skip",
    "jump",
]


def write_csv(path, rows):
    """Robust CSV writer using the union of keys across mixed rows."""
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
            "User-Agent": "TAIRID-ETDD70-viability-window-v1",
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

    (OUTDIR / "zenodo_record_13332134.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    files = record.get("files", [])
    downloads = []

    for f in files:
        key = f.get("key") or f.get("filename") or "unknown"
        links = f.get("links", {})
        url = links.get("self") or links.get("download")
        declared_size = f.get("size")

        if not url:
            downloads.append(
                {
                    "key": key,
                    "status": "no_download_url",
                    "size_declared": declared_size,
                }
            )
            continue

        local_path = DOWNLOAD_DIR / safe_name(key)

        try:
            print(f"Downloading {key} ...")
            payload, got_url, got_status, got_content_type = read_url(url, timeout=900)
            local_path.write_bytes(payload)

            downloads.append(
                {
                    "key": key,
                    "status": "downloaded",
                    "url": url,
                    "final_url": got_url,
                    "content_type": got_content_type,
                    "size_declared": declared_size,
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
                    "size_declared": declared_size,
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
                extracted.append(
                    {
                        "archive": str(path),
                        "status": "zip_extracted",
                        "target": str(target),
                    }
                )
            except Exception as exc:
                extracted.append(
                    {
                        "archive": str(path),
                        "status": "zip_extract_failed",
                        "error": str(exc),
                    }
                )

        elif lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
            target = EXTRACT_DIR / path.stem.replace(".tar", "")
            target.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(path, "r:*") as t:
                    t.extractall(target)
                extracted.append(
                    {
                        "archive": str(path),
                        "status": "tar_extracted",
                        "target": str(target),
                    }
                )
            except Exception as exc:
                extracted.append(
                    {
                        "archive": str(path),
                        "status": "tar_extract_failed",
                        "error": str(exc),
                    }
                )

    return extracted


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


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


def normalize_group(value):
    """
    Critical correction:
    Check non-dyslexic/control before checking dyslexic, because 'non-dyslexic'
    contains the substring 'dyslexic'.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None

    s = str(value).strip().lower()
    s_norm = re.sub(r"[_\-]+", " ", s)

    if not s_norm:
        return None

    if (
        "non dyslexic" in s_norm
        or "non-dyslexic" in s
        or "nondyslexic" in norm(s)
        or "control" in s_norm
        or "typical" in s_norm
        or s_norm in ["0", "false", "no", "n", "healthy", "td"]
    ):
        return "control"

    if "dyslexic" in s_norm or "dyslexia" in s_norm or "dys" == s_norm or s_norm in ["1", "true", "yes", "y"]:
        return "dyslexic"

    return None


def find_col_by_exact_or_hint(columns, hints, avoid=None):
    avoid = avoid or set()
    cols = list(columns)
    by_norm = {c: norm(c) for c in cols}

    avoid_norm = {norm(a) for a in avoid}

    for hint in hints:
        h = norm(hint)
        for c, cn in by_norm.items():
            if cn in avoid_norm:
                continue
            if h and cn == h:
                return c

    for hint in hints:
        h = norm(hint)
        for c, cn in by_norm.items():
            if cn in avoid_norm:
                continue
            if h and h in cn:
                return c

    return None


def find_subject_column(columns):
    cols = list(columns)
    by_norm = {c: norm(c) for c in cols}

    for preferred in SUBJECT_COL_PREFERENCE:
        p = norm(preferred)
        for c, cn in by_norm.items():
            if cn == p and cn not in {norm(a) for a in AVOID_SUBJECT_COLS}:
                return c

    for preferred in SUBJECT_COL_PREFERENCE:
        p = norm(preferred)
        for c, cn in by_norm.items():
            if p and p in cn and cn not in {norm(a) for a in AVOID_SUBJECT_COLS}:
                return c

    return None


def infer_subject_from_path(path):
    text = str(path)

    patterns = [
        r"Subject[_\- ]?([0-9A-Za-z]+)",
        r"subject[_\- ]?([0-9A-Za-z]+)",
        r"sub[_\- ]?([0-9A-Za-z]+)",
        r"sid[_\- ]?([0-9A-Za-z]+)",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)

    stem = Path(path).stem
    m = re.search(r"([0-9]{2,4})", stem)
    if m:
        return m.group(1)

    return stem[:40]


def infer_task_rank(path, value=None):
    text = f"{path} {value if value is not None else ''}".lower()
    text_norm = re.sub(r"[^a-z0-9]+", "_", text)

    for key, rank in TASK_NAME_TO_RANK.items():
        if key in text or key in text_norm:
            return rank

    m = re.search(r"t([145])", text_norm)
    if m:
        task_num = int(m.group(1))
        if task_num == 1:
            return 1
        if task_num == 4:
            return 2
        if task_num == 5:
            return 3

    m = re.search(r"task[_\- ]?([0-9]+)", text)
    if m:
        n = int(m.group(1))
        if n == 1:
            return 1
        if n == 4:
            return 2
        if n == 5:
            return 3

    return None


def is_mac_or_junk(path):
    parts = [p.lower() for p in Path(path).parts]
    name = Path(path).name.lower()

    if "__macosx" in parts:
        return True
    if name.startswith("._"):
        return True
    if name.startswith("."):
        return True

    return False


def list_label_candidate_tables():
    candidates = []
    for root in [DOWNLOAD_DIR, EXTRACT_DIR]:
        for path in root.rglob("*"):
            if not path.is_file() or is_mac_or_junk(path):
                continue
            lower = path.name.lower()
            if lower.endswith((".csv", ".tsv", ".txt", ".xlsx", ".xls")):
                if "label" in lower or "class" in lower or "dys" in lower:
                    candidates.append(path)
    return sorted(set(candidates))


def list_metrics_tables():
    candidates = []
    for root in [DOWNLOAD_DIR, EXTRACT_DIR]:
        for path in root.rglob("*"):
            if not path.is_file() or is_mac_or_junk(path):
                continue

            lower = path.name.lower()

            if not lower.endswith((".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls")):
                continue

            if "metrics" in lower and "subject" in str(path).lower():
                candidates.append(path)
            elif lower.endswith("_metrics.csv"):
                candidates.append(path)
            elif "metrics" in lower:
                candidates.append(path)

    return sorted(set(candidates))


def load_label_map():
    label_map = {}
    label_sources = []

    for path in list_label_candidate_tables():
        df, err = read_table(path)

        if df is None:
            label_sources.append({"path": str(path), "status": "read_failed", "error": err})
            continue

        columns = list(df.columns)
        sub_col = find_subject_column(columns)
        group_col = find_col_by_exact_or_hint(columns, GROUP_COL_HINTS)

        if sub_col is None and len(columns) >= 1:
            sub_col = columns[0]
        if group_col is None and len(columns) >= 2:
            group_col = columns[-1]

        parsed = 0

        for _, row in df.iterrows():
            subj = str(row.get(sub_col, "")).strip() if sub_col else ""
            grp = normalize_group(row.get(group_col)) if group_col else None

            if subj and grp:
                label_map[subj] = grp
                label_map[norm(subj)] = grp
                m = re.search(r"([0-9]{1,4})", subj)
                if m:
                    label_map[m.group(1)] = grp
                parsed += 1

        label_sources.append(
            {
                "path": str(path),
                "status": "parsed",
                "columns": " | ".join(columns),
                "subject_col": sub_col,
                "group_col": group_col,
                "parsed_labels": parsed,
            }
        )

    return label_map, label_sources


def infer_group(subject, label_map):
    if subject is None:
        return None

    s = str(subject).strip()
    if not s:
        return None

    for key in [s, norm(s)]:
        if key in label_map:
            return label_map[key]

    m = re.search(r"([0-9]{1,4})", s)
    if m and m.group(1) in label_map:
        return label_map[m.group(1)]

    return None


def numeric_columns(df):
    out = []

    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        finite_count = int(np.isfinite(vals).sum())
        if finite_count >= 1:
            out.append(c)

    return out


def column_role(name):
    n = name.lower()
    n_norm = norm(name)

    is_pacing = any(norm(h) in n_norm for h in PACING_NAME_HINTS)
    is_constraint = any(norm(h) in n_norm for h in CONSTRAINT_NAME_HINTS)

    # Avoid putting generic x/y into pacing unless it has stronger timing text.
    if n_norm in ["x", "y", "sid", "subject", "subjectid", "taskrank"]:
        return "ignore"

    if is_pacing and not is_constraint:
        return "pacing"
    if is_constraint and not is_pacing:
        return "constraint"
    if is_pacing and is_constraint:
        return "both"

    return "other_numeric"


def parse_metrics_files(label_map):
    inventory = []
    rows = []

    for path in list_metrics_tables():
        df, err = read_table(path)

        if df is None:
            inventory.append({"path": str(path), "status": "read_failed", "error": err})
            continue

        columns = list(df.columns)
        subject_col = find_subject_column(columns)
        group_col = find_col_by_exact_or_hint(columns, GROUP_COL_HINTS)
        task_col = find_col_by_exact_or_hint(columns, TASK_COL_HINTS)

        subject_from_path = infer_subject_from_path(path)
        task_from_path = infer_task_rank(path)

        usable_numeric = numeric_columns(df)

        inventory.append(
            {
                "path": str(path),
                "status": "read_ok",
                "row_count": int(len(df)),
                "columns": " | ".join(columns),
                "subject_col": subject_col,
                "group_col": group_col,
                "task_col": task_col,
                "subject_from_path": subject_from_path,
                "task_from_path": task_from_path,
                "numeric_column_count": len(usable_numeric),
            }
        )

        if not usable_numeric:
            continue

        if subject_col and subject_col in df.columns:
            subject_values = df[subject_col].dropna().astype(str).unique().tolist()
        else:
            subject_values = [subject_from_path]

        if not subject_values:
            subject_values = [subject_from_path]

        for subject in subject_values:
            subdf = df.copy()

            if subject_col and subject_col in df.columns:
                subdf = subdf[subdf[subject_col].astype(str) == str(subject)]

            if subdf.empty:
                continue

            group = None

            if group_col and group_col in subdf.columns:
                grp_counts = Counter(normalize_group(v) for v in subdf[group_col].dropna().astype(str).values)
                grp_counts.pop(None, None)
                if grp_counts:
                    group = grp_counts.most_common(1)[0][0]

            if group is None:
                group = infer_group(subject, label_map)

            task_rank = None

            if task_col and task_col in subdf.columns:
                vals = subdf[task_col].dropna().astype(str).values
                for v in vals[:20]:
                    task_rank = infer_task_rank(path, v)
                    if task_rank is not None:
                        break

            if task_rank is None:
                task_rank = task_from_path

            if group not in ["control", "dyslexic"] or task_rank not in [1, 2, 3]:
                continue

            feature_row = {
                "source_file": str(path),
                "subject": str(subject),
                "group": group,
                "task_rank": int(task_rank),
                "task_name": TASK_RANK_TO_NAME[int(task_rank)],
                "n_rows_in_file_subject_slice": int(len(subdf)),
            }

            for col in usable_numeric:
                if col == subject_col:
                    continue

                vals = pd.to_numeric(subdf[col], errors="coerce").dropna().astype(float).values
                if vals.size == 0:
                    continue

                role = column_role(col)

                if role == "ignore":
                    continue

                safe_col = re.sub(r"[^A-Za-z0-9_]+", "_", col).strip("_")[:80]

                feature_row[f"{safe_col}__median"] = float(np.median(vals))
                feature_row[f"{safe_col}__mean"] = float(np.mean(vals))

                if vals.size >= 2:
                    feature_row[f"{safe_col}__std"] = float(np.std(vals))
                    feature_row[f"{safe_col}__cv"] = float(np.std(vals) / max(abs(np.mean(vals)), 1.0e-9))
                    feature_row[f"{safe_col}__iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))

            rows.append(feature_row)

    return inventory, rows


def aggregate_subject_task(feature_rows):
    if not feature_rows:
        return pd.DataFrame()

    df = pd.DataFrame(feature_rows)

    df = df[df["group"].isin(["control", "dyslexic"])]
    df = df[df["task_rank"].isin([1, 2, 3])]

    if df.empty:
        return pd.DataFrame()

    group_cols = ["subject", "group", "task_rank", "task_name"]

    numeric_cols = [
        c for c in df.columns
        if c not in ["source_file", "subject", "group", "task_rank", "task_name"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not numeric_cols:
        return pd.DataFrame()

    agg = df.groupby(group_cols, dropna=False)[numeric_cols].median().reset_index()

    return agg


def choose_role_columns(df):
    numeric_cols = [
        c for c in df.columns
        if c not in ["subject", "group", "task_rank", "task_name"]
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notnull().sum() >= 10
    ]

    pacing_cols = []
    constraint_cols = []

    for c in numeric_cols:
        base = c.split("__")[0]
        role = column_role(base)

        if role in ["pacing", "both"]:
            pacing_cols.append(c)
        if role in ["constraint", "both"]:
            constraint_cols.append(c)

    if not pacing_cols and numeric_cols:
        pacing_cols = numeric_cols[: max(1, len(numeric_cols) // 3)]

    if not constraint_cols and numeric_cols:
        constraint_cols = numeric_cols[max(1, len(numeric_cols) // 3):] or numeric_cols

    # Limit to avoid over-weighting hundreds of generated columns.
    pacing_cols = pacing_cols[:30]
    constraint_cols = constraint_cols[:30]

    return numeric_cols, pacing_cols, constraint_cols


def zscore_by_task(df, cols):
    out = df.copy()

    for c in cols:
        zc = c + "__z_by_task"
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


def build_tairid_task_features(subject_task_df):
    if subject_task_df.empty:
        return pd.DataFrame(), {
            "numeric_cols": [],
            "pacing_cols": [],
            "constraint_cols": [],
        }

    numeric_cols, pacing_cols, constraint_cols = choose_role_columns(subject_task_df)

    if not numeric_cols:
        return pd.DataFrame(), {
            "numeric_cols": [],
            "pacing_cols": [],
            "constraint_cols": [],
        }

    df = zscore_by_task(subject_task_df, numeric_cols)

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
        "numeric_cols": numeric_cols,
        "pacing_cols": pacing_cols,
        "constraint_cols": constraint_cols,
        "pacing_z_cols_used": pacing_z,
        "constraint_z_cols_used": constraint_z,
    }

    return df, meta


def build_subject_viability_features(task_df):
    if task_df.empty:
        return pd.DataFrame(), {}

    df = task_df.copy()

    # Need all 3 tasks for the cleanest viability-window response curve.
    complete_subjects = []
    for subject, sub in df.groupby("subject"):
        tasks = set(int(x) for x in sub["task_rank"].dropna().astype(int).values)
        if {1, 2, 3}.issubset(tasks):
            complete_subjects.append(subject)

    df = df[df["subject"].isin(complete_subjects)]

    if df.empty:
        return pd.DataFrame(), {
            "complete_subject_count": 0,
            "control_task1_mismatch_sd": None,
        }

    task1 = df[df["task_rank"] == 1]
    control_task1 = task1[task1["group"] == "control"]["M_mismatch_abs"].astype(float).dropna().values

    if control_task1.size >= 2:
        control_baseline_sd = float(np.std(control_task1, ddof=1))
    else:
        control_baseline_sd = float(task1["M_mismatch_abs"].astype(float).std(ddof=1))

    if not np.isfinite(control_baseline_sd) or control_baseline_sd <= 1.0e-12:
        control_baseline_sd = 0.0

    rows = []

    for subject, sub in df.groupby("subject"):
        sub = sub.sort_values("task_rank")
        group_vals = sub["group"].dropna().unique()
        group = group_vals[0] if len(group_vals) else None

        by_task = {int(r["task_rank"]): r for _, r in sub.iterrows()}

        if 1 not in by_task or 2 not in by_task or 3 not in by_task:
            continue

        M1 = float(by_task[1]["M_mismatch_abs"])
        M2 = float(by_task[2]["M_mismatch_abs"])
        M3 = float(by_task[3]["M_mismatch_abs"])

        T_vals = np.asarray([float(by_task[i]["T_pacing_proxy"]) for i in [1, 2, 3]], dtype=float)
        I_vals = np.asarray([float(by_task[i]["I_constraint_proxy"]) for i in [1, 2, 3]], dtype=float)
        M_vals = np.asarray([M1, M2, M3], dtype=float)
        C_vals = np.asarray([float(by_task[i]["collapse_load_proxy"]) for i in [1, 2, 3]], dtype=float)
        tasks = np.asarray([1.0, 2.0, 3.0], dtype=float)

        def slope(vals):
            try:
                return float(np.polyfit(tasks, vals, 1)[0])
            except Exception:
                return np.nan

        def curve(vals):
            try:
                return float(np.polyfit(tasks, vals, 2)[0])
            except Exception:
                return np.nan

        base = {
            "subject": subject,
            "group": group,
            "task_count": 3,
            "M_task1_baseline": M1,
            "M_task2": M2,
            "M_task3": M3,
            "M_mean": float(np.mean(M_vals)),
            "M_max": float(np.max(M_vals)),
            "M_range": float(np.max(M_vals) - np.min(M_vals)),
            "M_slope": slope(M_vals),
            "M_curvature": curve(M_vals),
            "T_slope": slope(T_vals),
            "I_slope": slope(I_vals),
            "T_I_slope_gap": float(abs(slope(T_vals) - slope(I_vals))),
            "collapse_load_mean": float(np.mean(C_vals)),
            "collapse_load_max": float(np.max(C_vals)),
            "collapse_load_slope": slope(C_vals),
            "collapse_load_curvature": curve(C_vals),
            "control_task1_mismatch_sd_used": control_baseline_sd,
        }

        for mult in TOLERANCE_MULTIPLIERS:
            W = M1 + mult * control_baseline_sd

            B1 = max(0.0, M1 - W)
            B2 = max(0.0, M2 - W)
            B3 = max(0.0, M3 - W)

            B_vals = np.asarray([B1, B2, B3], dtype=float)

            suffix = f"tol{str(mult).replace('.', 'p')}"

            base[f"W_{suffix}"] = float(W)
            base[f"B1_{suffix}"] = float(B1)
            base[f"B2_{suffix}"] = float(B2)
            base[f"B3_{suffix}"] = float(B3)
            base[f"B_total_{suffix}"] = float(B2 + B3)
            base[f"B_mean_harder_{suffix}"] = float((B2 + B3) / 2.0)
            base[f"B_max_{suffix}"] = float(np.max(B_vals))
            base[f"B_slope_{suffix}"] = slope(B_vals)
            base[f"B_curvature_{suffix}"] = curve(B_vals)
            base[f"B_any_harder_{suffix}"] = int((B2 > 0.0) or (B3 > 0.0))
            base[f"B_pseudotext_minus_text_{suffix}"] = float(B3 - B2)

        rows.append(base)

    meta = {
        "complete_subject_count": len(rows),
        "control_task1_mismatch_sd": control_baseline_sd,
        "tolerance_multipliers": TOLERANCE_MULTIPLIERS,
    }

    return pd.DataFrame(rows), meta


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / max(len(a) + len(b) - 2, 1)
    )

    if pooled <= 1.0e-12:
        return 0.0

    return float((np.mean(a) - np.mean(b)) / pooled)


def auc_score(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 2:
        return np.nan

    pos = scores[labels == 1]
    neg = scores[labels == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    ranks = stats.rankdata(np.concatenate([pos, neg]))
    rpos = np.sum(ranks[:len(pos)])
    auc = (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    return float(auc)


def bootstrap_auc(scores, labels, n_boot=500):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan

    base = auc_score(scores, labels)
    boots = []
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(scores)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        boots.append(auc_score(scores[idx], labels[idx]))

    if not boots:
        return base, np.nan, np.nan

    lo, hi = np.percentile(boots, [2.5, 97.5])

    return base, float(lo), float(hi)


def simple_logistic_cv(df, feature_cols, label_col="label", k=5):
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return {"status": "no_features", "n": 0, "feature_cols": []}

    data = df[feature_cols + [label_col]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 20 or data[label_col].nunique() < 2:
        return {"status": "not_enough_data", "n": int(len(data)), "feature_cols": feature_cols}

    X = data[feature_cols].astype(float).values
    y = data[label_col].astype(int).values

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd <= 1.0e-12] = 1.0
    X = (X - mu) / sd

    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    folds = np.array_split(idx, min(k, len(idx)))
    preds = np.zeros(len(y), dtype=float)

    for fold in folds:
        train = np.setdiff1d(idx, fold)
        Xt = X[train]
        yt = y[train]

        w = np.zeros(X.shape[1])
        b = 0.0
        lr = 0.05
        lam = 0.1

        for _ in range(900):
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


def feature_tests(subject_df):
    if subject_df.empty:
        return []

    df = subject_df.copy()
    df = df[df["group"].isin(["control", "dyslexic"])]

    tests = []

    numeric_cols = [
        c for c in df.columns
        if c not in ["subject", "group"]
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notnull().sum() >= 10
    ]

    for c in numeric_cols:
        dys = df[df["group"] == "dyslexic"][c].dropna().astype(float).values
        con = df[df["group"] == "control"][c].dropna().astype(float).values

        if len(dys) < 3 or len(con) < 3:
            continue

        tstat, pval = stats.ttest_ind(dys, con, equal_var=False, nan_policy="omit")
        d = cohen_d(dys, con)

        scores = df[c].astype(float).values
        labels = (df["group"] == "dyslexic").astype(int).values

        if np.nanmean(dys) < np.nanmean(con):
            scores = -scores

        auc, lo, hi = bootstrap_auc(scores, labels, n_boot=500)

        tests.append(
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

    tests = sorted(
        tests,
        key=lambda r: (
            -(abs(r["cohen_d_dys_minus_control"]) if np.isfinite(r["cohen_d_dys_minus_control"]) else 0),
            r["welch_p"] if np.isfinite(r["welch_p"]) else 999,
        ),
    )

    return tests


def analyze_models(subject_df):
    if subject_df.empty:
        return {"status": "no_subject_viability_features"}

    df = subject_df.copy()
    df = df[df["group"].isin(["control", "dyslexic"])]

    if df.empty or df["group"].nunique() < 2:
        return {
            "status": "missing_groups",
            "n": int(len(df)),
            "group_counts": dict(Counter(df["group"])) if "group" in df.columns else {},
        }

    df["label"] = (df["group"] == "dyslexic").astype(int)

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
    for mult in TOLERANCE_MULTIPLIERS:
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

    models = {
        "static_level_model": simple_logistic_cv(df, static_cols),
        "mismatch_dynamic_model": simple_logistic_cv(df, mismatch_dynamic_cols),
        "viability_breach_model": simple_logistic_cv(df, viability_cols),
        "combined_response_viability_model": simple_logistic_cv(
            df, static_cols + mismatch_dynamic_cols + viability_cols
        ),
    }

    tests = feature_tests(df)

    return {
        "status": "ok",
        "n": int(len(df)),
        "group_counts": {str(k): int(v) for k, v in Counter(df["group"]).items()},
        "models": models,
        "feature_tests": tests,
    }


def plot_outputs(task_df, subject_df, analysis):
    plots = []

    if not task_df.empty:
        plt.figure(figsize=(10, 6))
        for group, sub in task_df.groupby("group"):
            means = sub.groupby("task_rank")["M_mismatch_abs"].mean()
            plt.plot(means.index, means.values, marker="o", label=str(group))
        plt.xlabel("Task demand rank")
        plt.ylabel("Mean mismatch |T - I|")
        plt.title("TAIRID ETDD70 viability window v1: mismatch across demand")
        plt.legend()
        plt.tight_layout()
        path = OUTDIR / "etdd70_vw_mismatch_across_demand.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plots.append(str(path))

        plt.figure(figsize=(10, 6))
        for group, sub in task_df.groupby("group"):
            means = sub.groupby("task_rank")["collapse_load_proxy"].mean()
            plt.plot(means.index, means.values, marker="o", label=str(group))
        plt.xlabel("Task demand rank")
        plt.ylabel("Mean collapse-load proxy")
        plt.title("TAIRID ETDD70 viability window v1: collapse load across demand")
        plt.legend()
        plt.tight_layout()
        path = OUTDIR / "etdd70_vw_collapse_load_across_demand.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plots.append(str(path))

    if not subject_df.empty:
        for mult in TOLERANCE_MULTIPLIERS:
            suffix = f"tol{str(mult).replace('.', 'p')}"
            col = f"B_total_{suffix}"
            if col not in subject_df.columns:
                continue

            groups = []
            values = []
            for group, sub in subject_df.groupby("group"):
                vals = sub[col].dropna().astype(float).values
                if len(vals):
                    groups.append(str(group))
                    values.append(vals)

            if values:
                plt.figure(figsize=(8, 6))
                plt.boxplot(values, labels=groups)
                plt.ylabel(f"Total breach score ({suffix})")
                plt.title(f"TAIRID ETDD70 viability breach: {suffix}")
                plt.tight_layout()
                path = OUTDIR / f"etdd70_vw_breach_boxplot_{suffix}.png"
                plt.savefig(path, dpi=160)
                plt.close()
                plots.append(str(path))

    return plots


def decide_status(analysis):
    if analysis.get("status") != "ok":
        return (
            "data_inventory_complete_viability_window_not_estimated",
            5,
            "Inspect metrics inventory and update exact ETDD70 parser if needed.",
        )

    models = analysis.get("models", {})
    static = models.get("static_level_model", {})
    dynamic = models.get("mismatch_dynamic_model", {})
    viability = models.get("viability_breach_model", {})
    combined = models.get("combined_response_viability_model", {})

    static_auc = static.get("auc") if static.get("status") == "ok" else None
    dynamic_auc = dynamic.get("auc") if dynamic.get("status") == "ok" else None
    viability_auc = viability.get("auc") if viability.get("status") == "ok" else None
    combined_auc = combined.get("auc") if combined.get("status") == "ok" else None

    if viability_auc is not None and static_auc is not None and viability_auc > static_auc + 0.05:
        return (
            "viability_window_breach_outperforms_static_level_proxy",
            8,
            "Replicate on another neurotype/task-switching dataset and test stability of tolerance choice.",
        )

    if combined_auc is not None and combined_auc >= 0.70:
        return (
            "viability_response_shape_contains_group_relevant_signal",
            7,
            "Check against simpler non-TAIRID features and run a confirmatory exact parser pass.",
        )

    if dynamic_auc is not None and static_auc is not None and dynamic_auc > static_auc + 0.05:
        return (
            "dynamic_mismatch_shape_outperforms_static_level_proxy",
            7,
            "Viability breach needs refinement, but relative response shape is useful.",
        )

    return (
        "viability_window_measurable_but_not_strongly_discriminative",
        6,
        "Refine T/I proxy assignment or use richer event-level timing data.",
    )


def main():
    print("")
    print("TAIRID ETDD70 viability-window breach test v1 starting.")
    print("Boundary: cross-domain viability-window test only; not proof.")
    print("")

    record, downloads = download_zenodo_record()
    extraction = extract_archives(downloads)

    write_csv(OUTDIR / "etdd70_vw_download_ledger.csv", downloads)
    write_csv(OUTDIR / "etdd70_vw_extraction_ledger.csv", extraction)

    label_map, label_sources = load_label_map()
    label_map_rows = [{"subject_key": k, "group": v} for k, v in sorted(label_map.items())]

    write_csv(OUTDIR / "etdd70_vw_label_sources.csv", label_sources)
    write_csv(OUTDIR / "etdd70_vw_label_map.csv", label_map_rows)

    metrics_inventory, metric_feature_rows = parse_metrics_files(label_map)

    inventory_path = write_csv(OUTDIR / "etdd70_vw_metrics_inventory.csv", metrics_inventory)
    metric_features_path = write_csv(OUTDIR / "etdd70_vw_metric_feature_rows.csv", metric_feature_rows)

    subject_task_df = aggregate_subject_task(metric_feature_rows)

    if not subject_task_df.empty:
        subject_task_path = OUTDIR / "etdd70_vw_subject_task_features.csv"
        subject_task_df.to_csv(subject_task_path, index=False)
    else:
        subject_task_path = None

    task_df, ti_meta = build_tairid_task_features(subject_task_df)

    if not task_df.empty:
        task_path = OUTDIR / "etdd70_vw_tairid_task_features.csv"
        task_df.to_csv(task_path, index=False)
    else:
        task_path = None

    subject_viability_df, vw_meta = build_subject_viability_features(task_df)

    if not subject_viability_df.empty:
        subject_vw_path = OUTDIR / "etdd70_vw_subject_viability_features.csv"
        subject_viability_df.to_csv(subject_vw_path, index=False)
    else:
        subject_vw_path = None

    analysis = analyze_models(subject_viability_df)

    analysis_path = OUTDIR / "etdd70_vw_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    plots = plot_outputs(task_df, subject_viability_df, analysis)

    final_status, readiness_score, next_wall = decide_status(analysis)

    summary = {
        "test_name": "TAIRID ETDD70 viability-window breach test v1",
        "boundary": (
            "Cross-domain viability-window response-shape test only. Not clinical diagnosis, "
            "not proof of TAIRID, and not a cosmology result."
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
        "downloads_count": len(downloads),
        "extraction_count": len(extraction),
        "label_source_count": len(label_sources),
        "label_map_count": len(label_map),
        "metrics_inventory_count": len(metrics_inventory),
        "metric_feature_row_count": len(metric_feature_rows),
        "subject_task_feature_count": int(len(subject_task_df)) if not subject_task_df.empty else 0,
        "task_shape_row_count": int(len(task_df)) if not task_df.empty else 0,
        "subject_viability_row_count": int(len(subject_viability_df)) if not subject_viability_df.empty else 0,
        "ti_proxy_meta": ti_meta,
        "viability_window_meta": vw_meta,
        "analysis": analysis,
        "output_files": {
            "download_ledger_csv": str(OUTDIR / "etdd70_vw_download_ledger.csv"),
            "extraction_ledger_csv": str(OUTDIR / "etdd70_vw_extraction_ledger.csv"),
            "label_sources_csv": str(OUTDIR / "etdd70_vw_label_sources.csv"),
            "label_map_csv": str(OUTDIR / "etdd70_vw_label_map.csv"),
            "metrics_inventory_csv": str(inventory_path),
            "metric_feature_rows_csv": str(metric_features_path),
            "subject_task_features_csv": str(subject_task_path) if subject_task_path else None,
            "tairid_task_features_csv": str(task_path) if task_path else None,
            "subject_viability_features_csv": str(subject_vw_path) if subject_vw_path else None,
            "analysis_json": str(analysis_path),
            "plots": plots,
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "Viability-window breach or dynamic mismatch features outperform simple static level proxies."
            ),
            "what_weakens_this_translation": (
                "Breach features do not outperform simple static measures, or the inferred T/I proxies are unstable."
            ),
            "translation_rule": (
                "Translate operator roles, not objects: pacing, constraint, mismatch, viability window, breach, recovery."
            ),
            "truth_boundary": (
                "This can support a TAIRID response-shape translation, but it cannot prove TAIRID or any cosmology claim."
            ),
        },
    }

    summary_path = OUTDIR / "etdd70_viability_window_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(OUTDIR / "etdd70_viability_window_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID ETDD70 viability-window breach test v1\n\n")
        f.write("Boundary: cross-domain viability-window response-shape test only. Not clinical diagnosis. Not proof.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("What this tests:\n")
        f.write("- Whether pacing/constraint mismatch breaches a subject-specific viability window as task demand rises.\n")
        f.write("- Whether breach features outperform simple static level proxies.\n")
        f.write("- Whether TAIRID translation is useful as a testable operator map, not a metaphor.\n\n")

        f.write("What this does not prove:\n")
        f.write("- It does not prove TAIRID.\n")
        f.write("- It does not diagnose dyslexia.\n")
        f.write("- It does not prove any cosmology claim.\n")
        f.write("- It does not show brains and cosmology are the same object.\n\n")

        f.write("Viability-window definition used:\n")
        f.write("M = |T_pacing_proxy - I_constraint_proxy|\n")
        f.write("W_subject = M_task1 + tolerance_multiplier * SD_control_task1\n")
        f.write("B_task = max(0, M_task - W_subject)\n\n")

        f.write("Analysis:\n")
        f.write(json.dumps(analysis, indent=2) + "\n\n")

    print("")
    print("TAIRID ETDD70 viability-window breach test v1 complete.")
    print("Created:")
    print("  tairid_etdd70_viability_window_v1_outputs/etdd70_viability_window_v1_summary.json")
    print("  tairid_etdd70_viability_window_v1_outputs/etdd70_viability_window_v1_summary.txt")
    print("  tairid_etdd70_viability_window_v1_outputs/etdd70_vw_metrics_inventory.csv")
    print("  tairid_etdd70_viability_window_v1_outputs/etdd70_vw_subject_viability_features.csv")
    print("  tairid_etdd70_viability_window_v1_outputs/etdd70_vw_analysis.json")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is not a clinical diagnostic result.")
    print("  This is a cross-domain viability-window response-shape test.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
