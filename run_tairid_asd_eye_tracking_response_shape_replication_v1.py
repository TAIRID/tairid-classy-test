#!/usr/bin/env python3
"""
TAIRID ASD eye-tracking response-shape replication v1.

Purpose:
ETDD70 dyslexia testing supported a TAIRID response-shape translation:
dynamic pacing/constraint mismatch and viability-window breach carried more signal than
simple static level features.

This test asks whether the same axis family appears in a second neurotype dataset:
ASD vs typically developing eye-tracking data.

Dataset:
Figshare article 20113592:
"Eye-Tracking Dataset to Support the Research on Autism Spectrum Disorder"

Important boundary:
This is not proof of TAIRID.
This is not clinical diagnosis.
This is not a cosmology result.
This is a second-dataset neurotype response-shape replication.

Translation:
- T / pacing proxy:
  timing, duration, velocity, fixation tempo, scan rhythm.
- I / constraint proxy:
  gaze dispersion, AOI instability, saccade/fixation load, revisit/backtracking pressure.
- M / mismatch:
  |T - I| after context-normalization.
- W / viability window:
  subject baseline mismatch across the lower-load context range plus tolerance.
- B / breach:
  max(0, M_context - W_subject).

Unlike ETDD70, this dataset does not have the same clean ordered reading-demand tasks.
So this script tests context-window stability across repeated eye-tracking experiment files.
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
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


OUTDIR = Path("tairid_asd_eye_tracking_response_shape_replication_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_DIR = OUTDIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

FIGSHARE_ARTICLE_ID = "20113592"
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

RANDOM_SEED = 42
CV_REPEATS = 100
PERMUTATIONS = 250
PERM_REPEATS = 25
RIDGE = 1.0e-3

TOLERANCE_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0]

SUBJECT_HINTS = [
    "participant",
    "participant_id",
    "participantid",
    "subject",
    "subject_id",
    "subjectid",
    "subj",
    "sid",
    "child",
    "idparticipant",
]

GROUP_HINTS = [
    "group",
    "class",
    "label",
    "diagnosis",
    "asd",
    "autism",
    "condition",
    "type",
]

TIME_HINTS = [
    "time",
    "timestamp",
    "duration",
    "dur",
    "fixationduration",
    "fix_dur",
    "latency",
    "velocity",
    "speed",
    "rt",
    "reaction",
    "dwell",
]

CONSTRAINT_HINTS = [
    "x",
    "y",
    "gaze",
    "fixation",
    "fix",
    "saccade",
    "sacc",
    "aoi",
    "areaofinterest",
    "amplitude",
    "dispersion",
    "distance",
    "pupil",
    "count",
    "number",
    "num",
    "visit",
    "revisit",
    "back",
    "regress",
    "std",
    "sd",
    "variance",
    "var",
    "spread",
]

AOI_SOCIAL_HINTS = [
    "face",
    "eye",
    "eyes",
    "mouth",
    "body",
    "person",
    "human",
    "social",
    "aoi",
    "areaofinterest",
]

ID_ARTIFACT_HINTS = [
    "id",
    "index",
    "row",
    "trial",
    "stimulus",
    "participant",
    "subject",
    "file",
    "session",
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


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_url(url, timeout=300):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-ASD-eye-tracking-response-shape-v1",
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


def download_figshare_article():
    data, final_url, status, content_type = read_url(FIGSHARE_API_URL)
    article = json.loads(data.decode("utf-8"))

    (OUTDIR / "figshare_article_20113592.json").write_text(
        json.dumps(article, indent=2), encoding="utf-8"
    )

    downloads = []

    for f in article.get("files", []):
        name = f.get("name") or f.get("filename") or f.get("id") or "unknown"
        url = f.get("download_url") or f.get("supplied_md5") or None

        if not url:
            links = f.get("links", {})
            url = links.get("download") or links.get("self")

        if not url:
            downloads.append(
                {
                    "name": name,
                    "status": "no_download_url",
                    "size": f.get("size"),
                }
            )
            continue

        local_path = DOWNLOAD_DIR / safe_name(name)

        try:
            print(f"Downloading {name} ...")
            payload, got_url, got_status, got_content_type = read_url(url, timeout=1200)
            local_path.write_bytes(payload)

            downloads.append(
                {
                    "name": name,
                    "status": "downloaded",
                    "url": url,
                    "final_url": got_url,
                    "content_type": got_content_type,
                    "declared_size": f.get("size"),
                    "size_downloaded": local_path.stat().st_size,
                    "path": str(local_path),
                    "sha256": sha256_file(local_path),
                }
            )
        except Exception as exc:
            downloads.append(
                {
                    "name": name,
                    "status": "download_failed",
                    "url": url,
                    "declared_size": f.get("size"),
                    "error": str(exc),
                }
            )

    return article, downloads


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


def is_junk_path(path):
    parts = [p.lower() for p in Path(path).parts]
    name = Path(path).name.lower()
    return "__macosx" in parts or name.startswith("._") or name.startswith(".")


def list_table_files():
    out = []

    for root in [DOWNLOAD_DIR, EXTRACT_DIR]:
        for path in root.rglob("*"):
            if not path.is_file() or is_junk_path(path):
                continue

            lower = path.name.lower()
            if lower.endswith((".csv", ".tsv", ".txt", ".xlsx", ".xls")):
                out.append(path)

    return sorted(set(out))


def read_table(path, max_rows=None):
    lower = path.name.lower()

    try:
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            df = pd.read_excel(path, nrows=max_rows)
        elif lower.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", nrows=max_rows)
        elif lower.endswith(".csv"):
            try:
                df = pd.read_csv(path, nrows=max_rows)
            except Exception:
                df = pd.read_csv(path, sep=";", nrows=max_rows)
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


def find_col(columns, hints, avoid=None):
    avoid = avoid or set()
    avoid_norm = {norm(a) for a in avoid}
    by_norm = {c: norm(c) for c in columns}

    for h0 in hints:
        h = norm(h0)
        for c, cn in by_norm.items():
            if cn in avoid_norm:
                continue
            if cn == h:
                return c

    for h0 in hints:
        h = norm(h0)
        for c, cn in by_norm.items():
            if cn in avoid_norm:
                continue
            if h and h in cn:
                return c

    return None


def normalize_group(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None

    s = str(value).strip().lower()
    compact = norm(s)

    if not compact:
        return None

    # Check TD/control before ASD, to avoid strings like "non-ASD" being misread.
    if (
        "typicallydeveloping" in compact
        or compact in ["td", "typical", "control", "controls", "healthy", "normal"]
        or "typically" in compact
        or "control" in compact
        or "nonasd" in compact
        or "notasd" in compact
        or "noasd" in compact
    ):
        return "td"

    if (
        compact == "asd"
        or "autism" in compact
        or "autistic" in compact
        or "spectrum" in compact
        or compact.startswith("asd")
        or compact.endswith("asd")
    ):
        return "asd"

    return None


def infer_group_from_participant_id(value):
    s = str(value).strip().lower()
    compact = norm(s)

    if not compact:
        return None

    if compact.startswith("td") or "typicallydeveloping" in compact:
        return "td"

    if compact.startswith("asd") or "autism" in compact:
        return "asd"

    return None


def load_participant_metadata():
    participant_map = {}
    sources = []

    for path in list_table_files():
        lower = path.name.lower()
        if "participant" not in lower and "metadata" not in lower and "subject" not in lower:
            continue

        df, err = read_table(path)
        if df is None:
            sources.append({"path": str(path), "status": "read_failed", "error": err})
            continue

        columns = list(df.columns)
        subject_col = find_col(columns, SUBJECT_HINTS)
        group_col = find_col(columns, GROUP_HINTS)

        if subject_col is None:
            subject_col = find_col(columns, ["participant"])

        if subject_col is None:
            sources.append(
                {
                    "path": str(path),
                    "status": "no_subject_col",
                    "columns": " | ".join(columns),
                }
            )
            continue

        parsed = 0
        for _, row in df.iterrows():
            subj = str(row.get(subject_col, "")).strip()
            if not subj:
                continue

            grp = None
            if group_col is not None:
                grp = normalize_group(row.get(group_col))

            if grp is None:
                grp = infer_group_from_participant_id(subj)

            if grp in ["asd", "td"]:
                participant_map[subj] = grp
                participant_map[norm(subj)] = grp
                parsed += 1

        sources.append(
            {
                "path": str(path),
                "status": "parsed",
                "columns": " | ".join(columns),
                "subject_col": subject_col,
                "group_col": group_col,
                "parsed": parsed,
            }
        )

    return participant_map, sources


def infer_context_from_path(path):
    stem = Path(path).stem
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", stem)
    return cleaned[:80]


def is_identifier_col(col):
    n = norm(col)

    if n in {norm(x) for x in ID_ARTIFACT_HINTS}:
        return True

    if n.endswith("id") or n.startswith("id"):
        return True

    if "participant" in n or "subject" in n:
        return True

    return False


def numeric_columns(df):
    cols = []
    for c in df.columns:
        if is_identifier_col(c):
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if int(np.isfinite(vals).sum()) >= 2:
            cols.append(c)
    return cols


def feature_family(name):
    n = norm(name)

    if any(norm(h) in n for h in AOI_SOCIAL_HINTS):
        return "aoi_social"

    if any(norm(h) in n for h in ["x", "y", "gaze", "distance", "amplitude", "dispersion"]):
        return "gaze_spatial"

    if any(norm(h) in n for h in TIME_HINTS):
        return "timing"

    return "general"


def feature_role(name):
    n = norm(name)

    if is_identifier_col(name):
        return "ignore"

    pacing = any(norm(h) in n for h in TIME_HINTS)
    constraint = any(norm(h) in n for h in CONSTRAINT_HINTS)

    if pacing and not constraint:
        return "pacing"
    if constraint and not pacing:
        return "constraint"
    if pacing and constraint:
        return "both"
    return "other"


def summarize_numeric(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {}

    out = {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
    }

    if vals.size >= 2:
        out["std"] = float(np.std(vals))
        out["cv"] = float(np.std(vals) / max(abs(np.mean(vals)), 1.0e-9))
        out["iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))

    return out


def parse_experiment_tables(participant_map):
    inventory = []
    rows = []

    for path in list_table_files():
        lower = path.name.lower()

        if "participant" in lower and ("metadata" in lower or lower == "participants.csv"):
            continue

        df, err = read_table(path)

        if df is None:
            inventory.append({"path": str(path), "status": "read_failed", "error": err})
            continue

        columns = list(df.columns)
        subject_col = find_col(columns, SUBJECT_HINTS)
        group_col = find_col(columns, GROUP_HINTS)
        num_cols = numeric_columns(df)

        context = infer_context_from_path(path)

        inv = {
            "path": str(path),
            "status": "read_ok",
            "context": context,
            "row_count": int(len(df)),
            "columns": " | ".join(columns),
            "subject_col": subject_col,
            "group_col": group_col,
            "numeric_col_count": len(num_cols),
        }

        if subject_col is None:
            inv["status"] = "read_ok_no_subject_col"
            inventory.append(inv)
            continue

        if not num_cols:
            inv["status"] = "read_ok_no_numeric_cols"
            inventory.append(inv)
            continue

        inventory.append(inv)

        subject_values = df[subject_col].dropna().astype(str).unique().tolist()

        for subj in subject_values:
            subdf = df[df[subject_col].astype(str) == str(subj)]

            if subdf.empty:
                continue

            grp = None

            if group_col is not None:
                counts = Counter(normalize_group(v) for v in subdf[group_col].dropna().astype(str).values)
                counts.pop(None, None)
                if counts:
                    grp = counts.most_common(1)[0][0]

            if grp is None:
                grp = participant_map.get(str(subj)) or participant_map.get(norm(subj))

            if grp is None:
                grp = infer_group_from_participant_id(subj)

            if grp not in ["asd", "td"]:
                continue

            row = {
                "source_file": str(path),
                "context": context,
                "participant": str(subj),
                "group": grp,
                "n_rows": int(len(subdf)),
            }

            for col in num_cols:
                vals = pd.to_numeric(subdf[col], errors="coerce").dropna().astype(float).values
                if vals.size < 2:
                    continue

                safe = re.sub(r"[^A-Za-z0-9_]+", "_", col).strip("_")[:80]
                fam = feature_family(col)

                for stat, value in summarize_numeric(vals).items():
                    row[f"{fam}__{safe}__{stat}"] = value

            rows.append(row)

    return inventory, rows


def aggregate_participant_context(feature_rows):
    if not feature_rows:
        return pd.DataFrame()

    df = pd.DataFrame(feature_rows)
    df = df[df["group"].isin(["asd", "td"])]

    if df.empty:
        return pd.DataFrame()

    group_cols = ["participant", "group", "context"]

    numeric_cols = [
        c for c in df.columns
        if c not in ["source_file", "participant", "group", "context"]
        and not is_identifier_col(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not numeric_cols:
        return pd.DataFrame()

    return df.groupby(group_cols, dropna=False)[numeric_cols].median().reset_index()


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


def choose_role_columns(df, family_filter="all"):
    numeric_cols = [
        c for c in df.columns
        if c not in ["participant", "group", "context"]
        and not is_identifier_col(c)
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notnull().sum() >= 10
    ]

    if family_filter != "all":
        numeric_cols = [c for c in numeric_cols if c.startswith(family_filter + "__")]

    pacing = []
    constraint = []

    for c in numeric_cols:
        role = feature_role(c)
        if role in ["pacing", "both"]:
            pacing.append(c)
        if role in ["constraint", "both"]:
            constraint.append(c)

    pacing = pacing[:50]
    constraint = constraint[:50]

    if not pacing and numeric_cols:
        pacing = numeric_cols[: max(1, len(numeric_cols) // 3)]

    if not constraint and numeric_cols:
        constraint = numeric_cols[max(1, len(numeric_cols) // 3):] or numeric_cols

    return numeric_cols, pacing, constraint


def build_context_shape_features(participant_context_df, family_filter="all"):
    if participant_context_df.empty:
        return pd.DataFrame(), {}

    numeric_cols, pacing_cols, constraint_cols = choose_role_columns(participant_context_df, family_filter)

    if not numeric_cols:
        return pd.DataFrame(), {
            "family_filter": family_filter,
            "numeric_cols_count": 0,
            "pacing_cols_count": 0,
            "constraint_cols_count": 0,
        }

    df = zscore_by_context(participant_context_df, numeric_cols)

    pacing_z = [c + "__z_by_context" for c in pacing_cols if c + "__z_by_context" in df.columns]
    constraint_z = [c + "__z_by_context" for c in constraint_cols if c + "__z_by_context" in df.columns]

    if not pacing_z:
        pacing_z = [c + "__z_by_context" for c in numeric_cols[: max(1, len(numeric_cols) // 3)]]

    if not constraint_z:
        constraint_z = [c + "__z_by_context" for c in numeric_cols[max(1, len(numeric_cols) // 3):]]

    df["T_pacing_proxy"] = df[pacing_z].mean(axis=1)
    df["I_constraint_proxy"] = df[constraint_z].mean(axis=1)
    df["M_mismatch_abs"] = np.abs(df["T_pacing_proxy"] - df["I_constraint_proxy"])
    df["collapse_load_proxy"] = np.sqrt(df["T_pacing_proxy"] ** 2 + df["I_constraint_proxy"] ** 2)
    df["interaction_TI"] = df["T_pacing_proxy"] * df["I_constraint_proxy"]

    meta = {
        "family_filter": family_filter,
        "numeric_cols_count": len(numeric_cols),
        "pacing_cols_count": len(pacing_cols),
        "constraint_cols_count": len(constraint_cols),
        "pacing_cols": pacing_cols,
        "constraint_cols": constraint_cols,
    }

    return df, meta


def build_subject_response_features(context_df):
    if context_df.empty:
        return pd.DataFrame(), {}

    rows = []

    all_context_counts = context_df.groupby("participant")["context"].nunique()
    min_contexts = 3

    complete = all_context_counts[all_context_counts >= min_contexts].index.tolist()
    df = context_df[context_df["participant"].isin(complete)].copy()

    if df.empty:
        return pd.DataFrame(), {
            "complete_subject_count": 0,
            "minimum_contexts_required": min_contexts,
        }

    td_context_low = []
    for participant, sub in df[df["group"] == "td"].groupby("participant"):
        vals = sub["M_mismatch_abs"].dropna().astype(float).values
        if vals.size >= min_contexts:
            td_context_low.append(float(np.percentile(vals, 25)))

    if len(td_context_low) >= 2:
        td_baseline_sd = float(np.std(td_context_low, ddof=1))
    else:
        td_baseline_sd = float(df["M_mismatch_abs"].astype(float).std(ddof=1))

    if not np.isfinite(td_baseline_sd) or td_baseline_sd <= 1.0e-12:
        td_baseline_sd = 0.0

    for participant, sub in df.groupby("participant"):
        sub = sub.sort_values("context")
        group_vals = sub["group"].dropna().unique()
        group = group_vals[0] if len(group_vals) else None

        M = sub["M_mismatch_abs"].dropna().astype(float).values
        T = sub["T_pacing_proxy"].dropna().astype(float).values
        I = sub["I_constraint_proxy"].dropna().astype(float).values
        C = sub["collapse_load_proxy"].dropna().astype(float).values

        if len(M) < min_contexts or len(T) < min_contexts or len(I) < min_contexts:
            continue

        idx = np.arange(len(M), dtype=float)

        def slope(vals):
            try:
                return float(np.polyfit(idx, vals, 1)[0])
            except Exception:
                return np.nan

        def curve(vals):
            if len(vals) < 3:
                return np.nan
            try:
                return float(np.polyfit(idx, vals, 2)[0])
            except Exception:
                return np.nan

        baseline = float(np.percentile(M, 25))

        base = {
            "participant": participant,
            "group": group,
            "context_count": int(len(M)),
            "M_mean": float(np.mean(M)),
            "M_median": float(np.median(M)),
            "M_max": float(np.max(M)),
            "M_min": float(np.min(M)),
            "M_range": float(np.max(M) - np.min(M)),
            "M_iqr": float(np.percentile(M, 75) - np.percentile(M, 25)),
            "M_std": float(np.std(M)),
            "M_cv": float(np.std(M) / max(abs(np.mean(M)), 1.0e-9)),
            "M_context_slope": slope(M),
            "M_context_curvature": curve(M),
            "T_mean": float(np.mean(T)),
            "I_mean": float(np.mean(I)),
            "T_context_slope": slope(T),
            "I_context_slope": slope(I),
            "T_I_context_slope_gap": float(abs(slope(T) - slope(I))) if np.isfinite(slope(T)) and np.isfinite(slope(I)) else np.nan,
            "collapse_load_mean": float(np.mean(C)),
            "collapse_load_max": float(np.max(C)),
            "collapse_load_range": float(np.max(C) - np.min(C)),
            "collapse_load_context_slope": slope(C),
            "context_baseline_M_q25": baseline,
            "td_baseline_sd_used": td_baseline_sd,
        }

        for mult in TOLERANCE_MULTIPLIERS:
            W = baseline + mult * td_baseline_sd
            B = np.maximum(0.0, M - W)
            suffix = f"tol{str(mult).replace('.', 'p')}"

            base[f"W_{suffix}"] = float(W)
            base[f"B_total_{suffix}"] = float(np.sum(B))
            base[f"B_mean_{suffix}"] = float(np.mean(B))
            base[f"B_max_{suffix}"] = float(np.max(B))
            base[f"B_count_{suffix}"] = int(np.sum(B > 0.0))
            base[f"B_fraction_{suffix}"] = float(np.mean(B > 0.0))
            base[f"B_slope_{suffix}"] = slope(B)
            base[f"B_curvature_{suffix}"] = curve(B)

        rows.append(base)

    meta = {
        "complete_subject_count": len(rows),
        "minimum_contexts_required": min_contexts,
        "td_baseline_sd": td_baseline_sd,
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


def repeated_cv(df, feature_cols, repeats=CV_REPEATS, k=5, y_override=None):
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
    k_eff = min(k, int(np.min(counts[counts > 0])))
    k_eff = max(2, k_eff)

    rng = np.random.default_rng(RANDOM_SEED)
    aucs = []

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

        auc = auc_score(preds, y)
        if np.isfinite(auc):
            aucs.append(auc)

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
        y_perm = y.copy()
        rng.shuffle(y_perm)

        res = repeated_cv(data, feature_cols, repeats=PERM_REPEATS, y_override=y_perm)
        auc = res.get("auc_mean")

        if auc is not None and np.isfinite(auc):
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


def model_sets():
    static = [
        "M_mean",
        "M_max",
        "collapse_load_mean",
        "collapse_load_max",
        "T_mean",
        "I_mean",
    ]

    dynamic = [
        "M_range",
        "M_iqr",
        "M_std",
        "M_cv",
        "M_context_slope",
        "M_context_curvature",
        "T_I_context_slope_gap",
        "collapse_load_range",
        "collapse_load_context_slope",
    ]

    viability = []
    for mult in TOLERANCE_MULTIPLIERS:
        suffix = f"tol{str(mult).replace('.', 'p')}"
        viability.extend(
            [
                f"B_total_{suffix}",
                f"B_mean_{suffix}",
                f"B_max_{suffix}",
                f"B_count_{suffix}",
                f"B_fraction_{suffix}",
                f"B_slope_{suffix}",
                f"B_curvature_{suffix}",
            ]
        )

    return {
        "static_level_model": static,
        "dynamic_context_mismatch_model": dynamic,
        "viability_context_breach_model": viability,
        "combined_context_response_model": static + dynamic + viability,
    }


def feature_tests(subject_df):
    if subject_df.empty:
        return []

    df = subject_df.copy()
    df = df[df["group"].isin(["asd", "td"])]

    tests = []

    numeric_cols = [
        c for c in df.columns
        if c not in ["participant", "group"]
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notnull().sum() >= 8
    ]

    for c in numeric_cols:
        asd = df[df["group"] == "asd"][c].dropna().astype(float).values
        td = df[df["group"] == "td"][c].dropna().astype(float).values

        if len(asd) < 3 or len(td) < 3:
            continue

        tstat, pval = stats.ttest_ind(asd, td, equal_var=False, nan_policy="omit")
        d = cohen_d(asd, td)

        labels = (df["group"] == "asd").astype(int).values
        scores = df[c].astype(float).values

        if np.nanmean(asd) < np.nanmean(td):
            scores = -scores

        auc = auc_score(scores, labels)

        tests.append(
            {
                "feature": c,
                "asd_mean": float(np.mean(asd)),
                "td_mean": float(np.mean(td)),
                "cohen_d_asd_minus_td": d,
                "welch_t": float(tstat),
                "welch_p": float(pval),
                "oriented_auc": auc,
                "n_asd": int(len(asd)),
                "n_td": int(len(td)),
            }
        )

    return sorted(
        tests,
        key=lambda r: (
            -(abs(r["cohen_d_asd_minus_td"]) if np.isfinite(r["cohen_d_asd_minus_td"]) else 0),
            r["welch_p"] if np.isfinite(r["welch_p"]) else 999,
        ),
    )


def add_bh_q(feature_rows):
    rows = list(feature_rows)
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


def run_models(subject_df):
    df = subject_df.copy()
    df = df[df["group"].isin(["asd", "td"])]
    df["label"] = (df["group"] == "asd").astype(int)

    model_rows = []
    permutation_rows = []

    for name, cols in model_sets().items():
        res = repeated_cv(df, cols, repeats=CV_REPEATS)
        auc = res.get("auc_mean")

        model_rows.append(
            {
                "model_name": name,
                **{k: v for k, v in res.items() if k != "feature_cols"},
                "feature_cols": " | ".join(res.get("feature_cols", [])),
            }
        )

        perm = permutation_test(df, cols, observed_auc=auc)

        permutation_rows.append(
            {
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

    return model_rows, permutation_rows


def run_family_suite(participant_context_df):
    rows = []
    meta = []

    for family in ["all", "timing", "gaze_spatial", "aoi_social", "general"]:
        context_df, role_meta = build_context_shape_features(participant_context_df, family_filter=family)
        subject_df, subject_meta = build_subject_response_features(context_df)

        meta.append(
            {
                "family": family,
                **role_meta,
                **subject_meta,
                "context_rows": int(len(context_df)) if not context_df.empty else 0,
                "subject_rows": int(len(subject_df)) if not subject_df.empty else 0,
            }
        )

        if subject_df.empty:
            continue

        model_rows, _ = run_models(subject_df)

        for r in model_rows:
            rows.append({"family": family, **r})

    return rows, meta


def plot_model_rows(model_rows, path, title):
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


def plot_permutation(permutation_rows, path):
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
    plt.title("ASD eye-tracking response-shape replication v1: observed vs permutation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def plot_context_mismatch(context_df, path):
    if context_df.empty:
        return None

    summary = context_df.groupby(["group", "context"])["M_mismatch_abs"].mean().reset_index()

    plt.figure(figsize=(14, 7))
    for group, sub in summary.groupby("group"):
        vals = sub.sort_values("context")["M_mismatch_abs"].values
        plt.plot(np.arange(len(vals)), vals, marker="o", label=str(group))

    plt.xlabel("Context index")
    plt.ylabel("Mean mismatch |T - I|")
    plt.title("ASD eye-tracking response-shape replication v1: mismatch across contexts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def decide_status(model_rows, permutation_rows, family_rows):
    model = {r["model_name"]: r for r in model_rows}
    perm = {r["model_name"]: r for r in permutation_rows}

    static_auc = model.get("static_level_model", {}).get("auc_mean")
    dynamic_auc = model.get("dynamic_context_mismatch_model", {}).get("auc_mean")
    viability_auc = model.get("viability_context_breach_model", {}).get("auc_mean")
    combined_auc = model.get("combined_context_response_model", {}).get("auc_mean")

    dynamic_p = perm.get("dynamic_context_mismatch_model", {}).get("p_value_ge_observed")
    viability_p = perm.get("viability_context_breach_model", {}).get("p_value_ge_observed")
    combined_p = perm.get("combined_context_response_model", {}).get("p_value_ge_observed")

    shape_auc = max([x for x in [dynamic_auc, viability_auc, combined_auc] if x is not None] or [None])
    shape_p = min([p for p in [dynamic_p, viability_p, combined_p] if p is not None] or [1.0])

    family_shape_ok = [
        r for r in family_rows
        if r.get("status") == "ok"
        and r.get("auc_mean") is not None
        and float(r["auc_mean"]) >= 0.58
        and r.get("model_name") in [
            "dynamic_context_mismatch_model",
            "viability_context_breach_model",
            "combined_context_response_model",
        ]
    ]

    if static_auc is not None and shape_auc is not None and shape_auc > static_auc + 0.05 and shape_p <= 0.05:
        return (
            "asd_replication_supports_tairid_response_shape_translation",
            8,
            "Compare axis behavior against ETDD70 and write a cross-neurotype axis map.",
        )

    if static_auc is not None and shape_auc is not None and shape_auc > static_auc + 0.05:
        return (
            "asd_response_shape_directional_not_permutation_locked",
            7,
            "Signal is above static but needs stronger permutation or feature-family stability.",
        )

    if shape_auc is not None and shape_auc >= 0.60 and len(family_shape_ok) >= 2:
        return (
            "asd_response_shape_present_but_needs_refinement",
            7,
            "Feature-family stability is promising; refine T/I proxy assignment.",
        )

    return (
        "asd_response_shape_not_yet_supported",
        6,
        "Treat as weak replication; inspect parser and feature families before drawing conclusions.",
    )


def main():
    print("")
    print("TAIRID ASD eye-tracking response-shape replication v1 starting.")
    print("Boundary: second neurotype response-shape test only; not proof.")
    print("")

    article, downloads = download_figshare_article()
    extraction = extract_archives(downloads)

    write_csv(OUTDIR / "asd_et_download_ledger.csv", downloads)
    write_csv(OUTDIR / "asd_et_extraction_ledger.csv", extraction)

    participant_map, participant_sources = load_participant_metadata()
    participant_map_rows = [{"participant_key": k, "group": v} for k, v in sorted(participant_map.items())]

    write_csv(OUTDIR / "asd_et_participant_sources.csv", participant_sources)
    write_csv(OUTDIR / "asd_et_participant_map.csv", participant_map_rows)

    table_inventory, feature_rows = parse_experiment_tables(participant_map)

    inventory_path = write_csv(OUTDIR / "asd_et_table_inventory.csv", table_inventory)
    feature_rows_path = write_csv(OUTDIR / "asd_et_context_feature_rows.csv", feature_rows)

    participant_context_df = aggregate_participant_context(feature_rows)

    if not participant_context_df.empty:
        participant_context_path = OUTDIR / "asd_et_participant_context_features.csv"
        participant_context_df.to_csv(participant_context_path, index=False)
    else:
        participant_context_path = None

    context_df, role_meta = build_context_shape_features(participant_context_df, family_filter="all")

    if not context_df.empty:
        context_path = OUTDIR / "asd_et_tairid_context_shape_features.csv"
        context_df.to_csv(context_path, index=False)
    else:
        context_path = None

    subject_df, subject_meta = build_subject_response_features(context_df)

    if not subject_df.empty:
        subject_path = OUTDIR / "asd_et_subject_response_shape_features.csv"
        subject_df.to_csv(subject_path, index=False)
    else:
        subject_path = None

    model_rows, permutation_rows = run_models(subject_df)
    family_rows, family_meta = run_family_suite(participant_context_df)
    features = add_bh_q(feature_tests(subject_df))

    model_path = write_csv(OUTDIR / "asd_et_model_results.csv", model_rows)
    permutation_path = write_csv(OUTDIR / "asd_et_permutation_results.csv", permutation_rows)
    family_path = write_csv(OUTDIR / "asd_et_family_results.csv", family_rows)
    family_meta_path = write_csv(OUTDIR / "asd_et_family_meta.csv", family_meta)
    feature_tests_path = write_csv(OUTDIR / "asd_et_feature_tests_bh_fdr.csv", features)

    plots = []

    p = plot_model_rows(
        model_rows,
        OUTDIR / "asd_et_model_auc_bars.png",
        "ASD eye-tracking response-shape replication v1: model comparison",
    )
    if p:
        plots.append(p)

    p = plot_permutation(permutation_rows, OUTDIR / "asd_et_permutation_auc_bars.png")
    if p:
        plots.append(p)

    p = plot_model_rows(
        family_rows,
        OUTDIR / "asd_et_family_auc_bars.png",
        "ASD eye-tracking response-shape replication v1: feature-family comparison",
    )
    if p:
        plots.append(p)

    p = plot_context_mismatch(context_df, OUTDIR / "asd_et_mismatch_across_contexts.png")
    if p:
        plots.append(p)

    final_status, readiness_score, next_wall = decide_status(model_rows, permutation_rows, family_rows)

    group_counts = {}
    if not subject_df.empty:
        group_counts = {str(k): int(v) for k, v in Counter(subject_df["group"]).items()}

    context_counts = {}
    if not participant_context_df.empty:
        context_counts = {
            "participant_count": int(participant_context_df["participant"].nunique()),
            "context_count": int(participant_context_df["context"].nunique()),
            "participant_context_rows": int(len(participant_context_df)),
        }

    top_features = features[:20]

    summary = {
        "test_name": "TAIRID ASD eye-tracking response-shape replication v1",
        "boundary": (
            "Second neurotype response-shape replication only. Not clinical diagnosis, "
            "not proof of TAIRID, and not a cosmology result."
        ),
        "dataset": {
            "figshare_article_id": FIGSHARE_ARTICLE_ID,
            "figshare_api_url": FIGSHARE_API_URL,
            "article_title": article.get("title"),
            "doi": article.get("doi"),
        },
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "parser_counts": {
            "downloads_count": len(downloads),
            "extraction_count": len(extraction),
            "participant_source_count": len(participant_sources),
            "participant_map_count": len(participant_map),
            "table_inventory_count": len(table_inventory),
            "context_feature_row_count": len(feature_rows),
            **context_counts,
            "subject_response_shape_rows": int(len(subject_df)) if not subject_df.empty else 0,
            "group_counts": group_counts,
        },
        "settings": {
            "cv_repeats": CV_REPEATS,
            "permutations": PERMUTATIONS,
            "permutation_cv_repeats": PERM_REPEATS,
            "classifier": "ridge-regularized LDA repeated stratified CV",
            "context_window_note": (
                "This ASD dataset uses repeated experiment contexts rather than the clean ordered reading-demand axis in ETDD70."
            ),
        },
        "role_meta": role_meta,
        "subject_response_meta": subject_meta,
        "model_results": model_rows,
        "permutation_results": permutation_rows,
        "family_results": family_rows,
        "family_meta": family_meta,
        "top_feature_tests_bh_fdr": top_features,
        "output_files": {
            "table_inventory_csv": str(inventory_path),
            "context_feature_rows_csv": str(feature_rows_path),
            "participant_context_features_csv": str(participant_context_path) if participant_context_path else None,
            "tairid_context_shape_features_csv": str(context_path) if context_path else None,
            "subject_response_shape_features_csv": str(subject_path) if subject_path else None,
            "model_results_csv": str(model_path),
            "permutation_results_csv": str(permutation_path),
            "family_results_csv": str(family_path),
            "family_meta_csv": str(family_meta_path),
            "feature_tests_bh_fdr_csv": str(feature_tests_path),
            "plots": plots,
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "Dynamic context mismatch or viability-context breach beats static level features "
                "and remains stronger than permutation expectation."
            ),
            "what_weakens_this_replication": (
                "Signal does not beat static features, permutation null is comparable, or the parser cannot recover reliable ASD/TD context rows."
            ),
            "axis_mapping_note": (
                "This test uses ASD eye tracking to see whether TAIRID axes learned from ETDD70 generalize: "
                "T pacing, I constraint, M mismatch, W viability window, B breach, slope/curvature/context range."
            ),
            "truth_boundary": (
                "A positive result supports response-shape translation only. It does not prove TAIRID, diagnose ASD, or prove cosmology."
            ),
        },
    }

    summary_path = OUTDIR / "asd_eye_tracking_response_shape_replication_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(OUTDIR / "asd_eye_tracking_response_shape_replication_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID ASD eye-tracking response-shape replication v1\n\n")
        f.write("Boundary: second neurotype response-shape replication only. Not diagnosis. Not proof.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Why this test exists:\n")
        f.write("- ETDD70 supported dynamic mismatch / viability-window response shape in dyslexia eye tracking.\n")
        f.write("- This tests whether the same axis family appears in ASD vs TD eye tracking.\n")
        f.write("- Cosmology taught us not to chase simple offsets; this tests relative response geometry instead.\n\n")

        f.write("Model results:\n")
        f.write(json.dumps(model_rows, indent=2) + "\n\n")

        f.write("Permutation results:\n")
        f.write(json.dumps(permutation_rows, indent=2) + "\n\n")

        f.write("Feature-family results:\n")
        f.write(json.dumps(family_rows, indent=2) + "\n\n")

        f.write("Top feature tests with BH-FDR:\n")
        f.write(json.dumps(top_features, indent=2) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This can support a TAIRID response-shape translation.\n")
        f.write("- It cannot prove TAIRID.\n")
        f.write("- It cannot diagnose ASD.\n")
        f.write("- It cannot prove any cosmology claim.\n")

    print("")
    print("TAIRID ASD eye-tracking response-shape replication v1 complete.")
    print("Created:")
    print("  tairid_asd_eye_tracking_response_shape_replication_v1_outputs/asd_eye_tracking_response_shape_replication_v1_summary.json")
    print("  tairid_asd_eye_tracking_response_shape_replication_v1_outputs/asd_eye_tracking_response_shape_replication_v1_summary.txt")
    print("  tairid_asd_eye_tracking_response_shape_replication_v1_outputs/asd_et_model_results.csv")
    print("  tairid_asd_eye_tracking_response_shape_replication_v1_outputs/asd_et_permutation_results.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is not clinical diagnosis.")
    print("  This is a second neurotype response-shape replication.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
