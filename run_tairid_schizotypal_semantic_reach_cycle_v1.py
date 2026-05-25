#!/usr/bin/env python3
"""
TAIRID schizotypal semantic reach / cycle test v1.

Purpose:
The current TAIRID neurotype axis map has:

- ETDD70 / dyslexia:
  locked for dynamic mismatch, viability, and reach across symbolic task demand.

- ASD eye-tracking:
  supportive provisional for mismatch, context-window viability, breach, and reach.

- ADHD ds003500:
  static difference detected, but cycling/reach not supported in block-summary events.

- Mood / bipolar actigraphy:
  locked as the primary time-series cycling / hysteresis / reach lane.

This schizotypal lane asks a different question:
Can schizotypal traits help separate semantic/reference reach from cycling?

Prediction:
Schizotypal semantic data should test reach-first behavior:
semantic/reference-field spread, indirect association reach, boundary permeability,
and possible re-entry/cycle behavior when reach exceeds stable semantic constraint.

Dataset:
OSF project j29fn:
"Semantic priming and schizotypal personality: reassessing the link between thought disorder
and enhanced spreading of semantic activation."

Boundary:
This is not proof of TAIRID.
This is not diagnosis.
This is not medical advice.
This is not a cosmology result.
It is an operational semantic-reach / cycle axis test.

TAIRID translation:
T = response pacing / lexical decision timing
I = semantic constraint / error / variability / relation-control pressure
M = |T - I|
W = semantic viability window / tolerated mismatch
B = breach beyond W
Reach = indirect/direct semantic propagation span
Cycling = reversals, re-entry, sign changes across SOA/directness/context
"""

import csv
import io
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


OUTDIR = Path("tairid_schizotypal_semantic_reach_cycle_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_DIR = OUTDIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

OSF_NODE_ID = "j29fn"
OSF_FILES_URL = f"https://api.osf.io/v2/nodes/{OSF_NODE_ID}/files/"

RANDOM_SEED = 42
CV_REPEATS = 100
PERMUTATIONS = 250
PERM_REPEATS = 25
RIDGE = 1.0e-3

TOLERANCE_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0]

SUBJECT_HINTS = [
    "subject", "subject_id", "subjectid", "participant", "participant_id",
    "participantid", "subj", "sid", "id", "pid", "code"
]

RT_HINTS = [
    "rt", "reactiontime", "reaction_time", "responsetime", "response_time",
    "latency", "duration"
]

ACCURACY_HINTS = [
    "accuracy", "accurate", "correct", "responsecorrect", "response_correct",
    "error", "errors", "incorrect", "acc"
]

SOA_HINTS = [
    "soa", "stimulusonsetasynchrony", "stimulus_onset_asynchrony"
]

RELATED_HINTS = [
    "related", "relatedness", "relation", "semanticrelation", "semantic_relation",
    "condition", "cond"
]

DIRECT_HINTS = [
    "direct", "directness", "indirect", "associationtype", "association_type",
    "relationtype", "relation_type", "condition", "cond"
]

PRIME_HINTS = ["prime", "cue"]
TARGET_HINTS = ["target", "word", "stimulus"]

SCORE_HINTS_PRIMARY = [
    "disorg", "disorganization", "cognitivedisorganization", "cogdis",
    "odd", "oddspeech", "thought", "spq", "schizotypy", "olife",
    "unusual", "positive", "negative", "introvert", "anhedonia"
]

ID_LIKE_HINTS = [
    "trial", "item", "stimulus", "prime", "target", "word", "block", "session",
    "run", "condition", "cond", "soa", "related", "relation", "direct"
]


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fields, seen = [], set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:180]


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def read_url(url, timeout=900, accept="application/json, */*"):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-schizotypal-semantic-reach-cycle-v1",
            "Accept": accept,
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")

    return data, final_url, content_type


def read_json_url(url, timeout=900):
    data, final_url, content_type = read_url(url, timeout=timeout)
    return json.loads(data.decode("utf-8", errors="replace"))


def collect_osf_files():
    queue = [OSF_FILES_URL]
    seen_urls = set()
    files = []
    inventory = []

    while queue:
        url = queue.pop(0)

        if url in seen_urls:
            continue

        seen_urls.add(url)

        try:
            page = read_json_url(url)
        except Exception as exc:
            inventory.append(
                {
                    "url": url,
                    "status": "api_read_failed",
                    "error": str(exc),
                }
            )
            continue

        for item in page.get("data", []):
            attrs = item.get("attributes", {}) or {}
            links = item.get("links", {}) or {}
            rels = item.get("relationships", {}) or {}

            name = attrs.get("name") or item.get("id") or "unknown"
            kind = attrs.get("kind") or "unknown"
            path = attrs.get("path") or name

            row = {
                "id": item.get("id"),
                "name": name,
                "kind": kind,
                "path": path,
                "materialized_path": attrs.get("materialized_path"),
                "size": attrs.get("size"),
                "download": links.get("download"),
            }

            inventory.append(row)

            if kind == "file" and links.get("download"):
                files.append(row)
            else:
                related = (
                    rels.get("files", {})
                    .get("links", {})
                    .get("related", {})
                    .get("href")
                )
                if related:
                    queue.append(related)

        next_url = page.get("links", {}).get("next")

        if next_url:
            queue.append(next_url)

    return files, inventory


def download_osf_files(files):
    downloads = []

    for f in files:
        name = f.get("name") or f.get("id") or "unknown"
        url = f.get("download")

        if not url:
            downloads.append(
                {
                    "name": name,
                    "status": "no_download_url",
                    "osf_path": f.get("materialized_path") or f.get("path"),
                }
            )
            continue

        local = DOWNLOAD_DIR / safe_name(name)

        # Avoid collisions.
        if local.exists():
            stem = local.stem
            suffix = local.suffix
            local = DOWNLOAD_DIR / f"{stem}_{safe_name(str(f.get('id')))}{suffix}"

        try:
            print(f"Downloading OSF file: {name}")
            payload, final_url, content_type = read_url(url, timeout=1200, accept="*/*")
            local.write_bytes(payload)

            downloads.append(
                {
                    "name": name,
                    "status": "downloaded",
                    "url": url,
                    "final_url": final_url,
                    "content_type": content_type,
                    "osf_path": f.get("materialized_path") or f.get("path"),
                    "size_declared": f.get("size"),
                    "size_downloaded": local.stat().st_size,
                    "path": str(local),
                    "sha256": sha256_file(local),
                }
            )

        except Exception as exc:
            downloads.append(
                {
                    "name": name,
                    "status": "download_failed",
                    "url": url,
                    "osf_path": f.get("materialized_path") or f.get("path"),
                    "error": str(exc),
                }
            )

    return downloads


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

        elif lower.endswith((".tar", ".tar.gz", ".tgz")):
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


def list_candidate_data_files():
    out = []

    for root in [DOWNLOAD_DIR, EXTRACT_DIR]:
        for path in root.rglob("*"):
            if not path.is_file() or is_junk_path(path):
                continue

            lower = path.name.lower()

            if lower.endswith((
                ".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls",
                ".sav", ".dta", ".por", ".xpt", ".rds", ".rda", ".rdata"
            )):
                out.append(path)

    return sorted(set(out))


def read_any_table(path):
    lower = path.name.lower()
    tables = []

    try:
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            sheets = pd.read_excel(path, sheet_name=None)
            for name, df in sheets.items():
                if df is not None and not df.empty:
                    df.columns = [str(c).strip() for c in df.columns]
                    tables.append((f"{path.name}::{name}", df))

        elif lower.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t")
            df.columns = [str(c).strip() for c in df.columns]
            tables.append((path.name, df))

        elif lower.endswith(".csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.read_csv(path, sep=";")
            df.columns = [str(c).strip() for c in df.columns]
            tables.append((path.name, df))

        elif lower.endswith(".txt") or lower.endswith(".dat"):
            try:
                df = pd.read_csv(path, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(path, sep=r"\s+", engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            tables.append((path.name, df))

        elif lower.endswith((".sav", ".dta", ".por", ".xpt")):
            import pyreadstat

            if lower.endswith(".sav"):
                df, meta = pyreadstat.read_sav(path)
            elif lower.endswith(".dta"):
                df, meta = pyreadstat.read_dta(path)
            elif lower.endswith(".por"):
                df, meta = pyreadstat.read_por(path)
            else:
                df, meta = pyreadstat.read_xport(path)

            df.columns = [str(c).strip() for c in df.columns]
            tables.append((path.name, df))

        elif lower.endswith((".rds", ".rda", ".rdata")):
            import pyreadr

            result = pyreadr.read_r(path)

            for name, df in result.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.columns = [str(c).strip() for c in df.columns]
                    tables.append((f"{path.name}::{name}", df))

    except Exception as exc:
        return [], str(exc)

    return tables, None


def find_col(columns, hints, avoid_hints=None):
    avoid_hints = avoid_hints or []
    by_norm = {c: norm(c) for c in columns}

    for hint in hints:
        h = norm(hint)
        for c, cn in by_norm.items():
            if any(norm(a) in cn for a in avoid_hints):
                continue
            if cn == h:
                return c

    for hint in hints:
        h = norm(hint)
        for c, cn in by_norm.items():
            if any(norm(a) in cn for a in avoid_hints):
                continue
            if h and h in cn:
                return c

    return None


def find_subject_col(columns):
    return find_col(columns, SUBJECT_HINTS, avoid_hints=["trial", "item", "stimulus", "prime", "target"])


def is_numeric_series(s):
    vals = pd.to_numeric(s, errors="coerce")
    return int(np.isfinite(vals).sum()) >= max(5, int(0.10 * len(s)))


def score_candidate_columns(df):
    cols = []

    for c in df.columns:
        cn = norm(c)

        if any(norm(h) in cn for h in ID_LIKE_HINTS):
            continue

        if any(norm(h) in cn for h in SCORE_HINTS_PRIMARY) and is_numeric_series(df[c]):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notnull().sum() >= 5 and vals.nunique(dropna=True) >= 3:
                cols.append(c)

    return cols


def rt_candidate_col(df):
    candidates = []

    for c in df.columns:
        cn = norm(c)

        if any(norm(h) in cn for h in RT_HINTS):
            vals = pd.to_numeric(df[c], errors="coerce")
            finite = vals[np.isfinite(vals)]

            if len(finite) >= max(20, int(0.20 * len(df))):
                median = float(np.nanmedian(finite))
                score = 0.0

                if "rt" == cn or cn.endswith("rt"):
                    score += 5.0
                if 100 <= median <= 5000:
                    score += 5.0
                if "reaction" in cn or "response" in cn:
                    score += 3.0

                score += min(len(finite) / 1000.0, 5.0)
                candidates.append((score, c))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def accuracy_candidate_col(df):
    for c in df.columns:
        cn = norm(c)
        if any(norm(h) in cn for h in ACCURACY_HINTS):
            return c
    return None


def soa_candidate_col(df):
    return find_col(df.columns, SOA_HINTS)


def related_candidate_col(df):
    return find_col(df.columns, RELATED_HINTS)


def direct_candidate_col(df):
    return find_col(df.columns, DIRECT_HINTS)


def normalize_related(value):
    s = str(value).strip().lower()
    compact = norm(s)

    if compact in ["1", "true", "yes"]:
        return "related"

    if compact in ["0", "false", "no"]:
        return "unrelated"

    if "unrelated" in compact or compact in ["unrel", "un"]:
        return "unrelated"

    if "related" in compact or compact in ["rel", "r"]:
        return "related"

    if "weak" in compact:
        return "weak_related"

    return "unknown"


def normalize_directness(value):
    s = str(value).strip().lower()
    compact = norm(s)

    if "indirect" in compact:
        return "indirect"

    if "direct" in compact:
        return "direct"

    if compact in ["1", "true", "yes"]:
        return "direct"

    if compact in ["0", "false", "no"]:
        return "indirect"

    return "unknown"


def parse_soa(value):
    s = str(value)
    nums = re.findall(r"\d+\.?\d*", s)

    if nums:
        return float(nums[0])

    return np.nan


def normalize_accuracy(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return np.nan

    s = str(value).strip().lower()
    compact = norm(s)

    if compact in ["true", "yes", "y", "correct", "corr", "1"]:
        return 1.0

    if compact in ["false", "no", "n", "incorrect", "error", "err", "0"]:
        return 0.0

    try:
        val = float(s)

        if val in [0.0, 1.0]:
            return val

        # If this is an error column, handled later by name.
        return val

    except Exception:
        return np.nan


def table_inventory_and_candidates():
    inventory = []
    tables = []

    for path in list_candidate_data_files():
        read_tables, err = read_any_table(path)

        if err:
            inventory.append(
                {
                    "path": str(path),
                    "status": "read_failed",
                    "error": err,
                }
            )
            continue

        for table_name, df in read_tables:
            if df is None or df.empty:
                continue

            columns = list(df.columns)
            subj = find_subject_col(columns)
            rt_col = rt_candidate_col(df)
            score_cols = score_candidate_columns(df)
            acc_col = accuracy_candidate_col(df)
            soa_col = soa_candidate_col(df)
            rel_col = related_candidate_col(df)
            direct_col = direct_candidate_col(df)

            trial_score = 0

            if subj:
                trial_score += 3
            if rt_col:
                trial_score += 5
            if acc_col:
                trial_score += 1
            if soa_col:
                trial_score += 1
            if rel_col:
                trial_score += 1
            if direct_col:
                trial_score += 1
            if len(df) >= 100:
                trial_score += 2

            score_score = 0

            if subj:
                score_score += 3
            if score_cols:
                score_score += 5
            if 5 <= len(df) <= 1000:
                score_score += 1

            item = {
                "path": str(path),
                "table_name": table_name,
                "status": "read_ok",
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "columns": " | ".join(map(str, df.columns)),
                "subject_col": subj,
                "rt_col": rt_col,
                "accuracy_col": acc_col,
                "soa_col": soa_col,
                "related_col": rel_col,
                "direct_col": direct_col,
                "score_cols": " | ".join(score_cols),
                "trial_table_score": trial_score,
                "score_table_score": score_score,
            }

            inventory.append(item)
            tables.append((item, df))

    return inventory, tables


def build_score_table(tables):
    score_frames = []

    for item, df in tables:
        subj_col = item.get("subject_col")
        score_cols = [x.strip() for x in str(item.get("score_cols", "")).split("|") if x.strip()]

        if not subj_col or not score_cols:
            continue

        keep = [subj_col] + score_cols
        temp = df[keep].copy()
        temp = temp.rename(columns={subj_col: "subject"})
        temp["subject"] = temp["subject"].astype(str).str.strip()
        temp = temp[temp["subject"] != ""]

        for c in score_cols:
            temp[c] = pd.to_numeric(temp[c], errors="coerce")

        grouped = temp.groupby("subject", dropna=False)[score_cols].median().reset_index()

        if len(grouped) >= 5:
            score_frames.append(
                {
                    "source": item.get("table_name"),
                    "path": item.get("path"),
                    "df": grouped,
                    "score_cols": score_cols,
                    "n_subjects": int(grouped["subject"].nunique()),
                }
            )

    if not score_frames:
        return pd.DataFrame(), []

    # Merge score frames outer by subject.
    merged = None
    sources = []

    for frame in sorted(score_frames, key=lambda x: -x["n_subjects"]):
        temp = frame["df"].copy()

        # Prefix duplicate score column names by source stem if needed.
        if merged is None:
            merged = temp
        else:
            overlap = [c for c in temp.columns if c != "subject" and c in merged.columns]
            if overlap:
                rename = {c: f"{c}__{safe_name(frame['source'])[:20]}" for c in overlap}
                temp = temp.rename(columns=rename)

            merged = pd.merge(merged, temp, on="subject", how="outer")

        sources.append(
            {
                "source": frame["source"],
                "path": frame["path"],
                "n_subjects": frame["n_subjects"],
                "score_cols": " | ".join(frame["score_cols"]),
            }
        )

    if merged is None:
        return pd.DataFrame(), sources

    return merged, sources


def build_trial_table(tables, score_table):
    trial_candidates = sorted(
        [(item, df) for item, df in tables if item.get("trial_table_score", 0) >= 8],
        key=lambda x: -x[0].get("trial_table_score", 0),
    )

    rows = []
    trial_sources = []

    score_cols = []
    if score_table is not None and not score_table.empty:
        score_cols = [c for c in score_table.columns if c != "subject"]

    for item, df in trial_candidates:
        subj_col = item.get("subject_col")
        rt_col = item.get("rt_col")

        if not subj_col or not rt_col:
            continue

        acc_col = item.get("accuracy_col")
        soa_col = item.get("soa_col")
        rel_col = item.get("related_col")
        direct_col = item.get("direct_col")

        local_score_cols = score_candidate_columns(df)

        parsed = 0

        for _, r in df.iterrows():
            subject = str(r.get(subj_col, "")).strip()
            if not subject:
                continue

            rt = pd.to_numeric(pd.Series([r.get(rt_col)]), errors="coerce").iloc[0]

            if not np.isfinite(rt) or rt <= 0:
                continue

            # Keep plausible lexical-decision RT range, but do not over-prune.
            if rt > 20000:
                continue

            acc = np.nan

            if acc_col:
                acc = normalize_accuracy(r.get(acc_col))

                # If the column is named error/incorrect, invert 0/1.
                acn = norm(acc_col)
                if np.isfinite(acc) and any(x in acn for x in ["error", "incorrect"]):
                    if acc in [0.0, 1.0]:
                        acc = 1.0 - acc

            rel_val = r.get(rel_col) if rel_col else ""
            direct_val = r.get(direct_col) if direct_col else ""

            # Use all condition-like text to infer relation/directness.
            context_text = " ".join(str(r.get(c, "")) for c in df.columns if norm(c) in ["condition", "cond", "relation", "relatedness", "directness", "type"])

            related = normalize_related(str(rel_val) + " " + context_text)
            directness = normalize_directness(str(direct_val) + " " + context_text)
            soa = parse_soa(r.get(soa_col)) if soa_col else parse_soa(context_text)

            row = {
                "source_table": item.get("table_name"),
                "source_path": item.get("path"),
                "subject": subject,
                "rt": float(rt),
                "accuracy": float(acc) if np.isfinite(acc) else np.nan,
                "relatedness": related,
                "directness": directness,
                "soa": float(soa) if np.isfinite(soa) else np.nan,
            }

            for sc in local_score_cols:
                val = pd.to_numeric(pd.Series([r.get(sc)]), errors="coerce").iloc[0]
                if np.isfinite(val):
                    row[sc] = float(val)

            rows.append(row)
            parsed += 1

        trial_sources.append(
            {
                "source_table": item.get("table_name"),
                "source_path": item.get("path"),
                "parsed_rows": parsed,
                "trial_table_score": item.get("trial_table_score"),
                "subject_col": subj_col,
                "rt_col": rt_col,
                "accuracy_col": acc_col,
                "soa_col": soa_col,
                "related_col": rel_col,
                "direct_col": direct_col,
            }
        )

    trial = pd.DataFrame(rows)

    if trial.empty:
        return trial, trial_sources

    if score_table is not None and not score_table.empty:
        # Merge external scores where missing.
        trial = pd.merge(trial, score_table, on="subject", how="left", suffixes=("", "__scoretable"))

        for c in score_cols:
            c2 = f"{c}__scoretable"
            if c2 in trial.columns:
                if c in trial.columns:
                    trial[c] = trial[c].combine_first(trial[c2])
                    trial = trial.drop(columns=[c2])
                else:
                    trial = trial.rename(columns={c2: c})

    trial["soa"] = pd.to_numeric(trial["soa"], errors="coerce")
    trial["rt"] = pd.to_numeric(trial["rt"], errors="coerce")
    trial["accuracy"] = pd.to_numeric(trial["accuracy"], errors="coerce")

    return trial, trial_sources


def choose_primary_scores(trial_df):
    score_cols = []

    for c in trial_df.columns:
        if c in [
            "source_table", "source_path", "subject", "rt", "accuracy",
            "relatedness", "directness", "soa"
        ]:
            continue

        if not is_numeric_series(trial_df[c]):
            continue

        cn = norm(c)

        if any(norm(h) in cn for h in SCORE_HINTS_PRIMARY):
            score_cols.append(c)

    priority_terms = [
        "disorg", "disorganization", "cogdis", "cognitivedisorganization",
        "spqtotal", "totalspq", "spq", "olife", "unusual", "positive", "negative"
    ]

    def score_priority(c):
        cn = norm(c)
        score = 0
        for i, term in enumerate(priority_terms):
            if term in cn:
                score += 100 - i
        nonnull = trial_df.groupby("subject")[c].median().notnull().sum()
        score += min(nonnull, 50)
        return score

    score_cols = sorted(set(score_cols), key=score_priority, reverse=True)

    return score_cols


def zscore(vals):
    vals = np.asarray(vals, dtype=float)
    mu = np.nanmean(vals)
    sd = np.nanstd(vals)

    if not np.isfinite(sd) or sd <= 1.0e-12:
        return np.zeros_like(vals, dtype=float)

    return (vals - mu) / sd


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

    for flag in flags:
        if bool(flag):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return int(best)


def safe_div(a, b):
    if b is None or not np.isfinite(b) or abs(b) <= 1.0e-12:
        return np.nan
    return float(a / b)


def slope(x, y):
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


def curve(x, y):
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


def build_subject_semantic_axis_features(trial_df, score_cols):
    rows = []

    if trial_df.empty:
        return pd.DataFrame()

    for subject, sub in trial_df.groupby("subject"):
        if len(sub) < 20:
            continue

        sub = sub.copy()

        score_values = {}
        for sc in score_cols:
            vals = pd.to_numeric(sub[sc], errors="coerce").dropna()
            score_values[sc] = float(vals.median()) if len(vals) else np.nan

        # Basic static features.
        rt = pd.to_numeric(sub["rt"], errors="coerce").dropna().astype(float).values
        acc = pd.to_numeric(sub["accuracy"], errors="coerce").dropna().astype(float).values

        if len(rt) < 20:
            continue

        base = {
            "subject": subject,
            "trial_count": int(len(sub)),
            "rt_mean": float(np.mean(rt)),
            "rt_median": float(np.median(rt)),
            "rt_std": float(np.std(rt)),
            "rt_cv": float(np.std(rt) / max(abs(np.mean(rt)), 1.0e-9)),
            "rt_iqr": float(np.percentile(rt, 75) - np.percentile(rt, 25)),
            "accuracy_mean": float(np.mean(acc)) if len(acc) else np.nan,
            "error_rate": float(1.0 - np.mean(acc)) if len(acc) else np.nan,
            **score_values,
        }

        # Context-level features.
        context_rows = []

        context_cols = ["directness", "relatedness"]
        if sub["soa"].notnull().sum() >= 10:
            context_cols.append("soa")

        grouped = sub.groupby(context_cols, dropna=False)

        for keys, g in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)

            ctx = dict(zip(context_cols, keys))
            rt_vals = pd.to_numeric(g["rt"], errors="coerce").dropna().astype(float).values
            acc_vals = pd.to_numeric(g["accuracy"], errors="coerce").dropna().astype(float).values

            if len(rt_vals) < 3:
                continue

            context_rows.append(
                {
                    **ctx,
                    "n": int(len(rt_vals)),
                    "rt_median": float(np.median(rt_vals)),
                    "rt_mean": float(np.mean(rt_vals)),
                    "rt_cv": float(np.std(rt_vals) / max(abs(np.mean(rt_vals)), 1.0e-9)),
                    "error_rate": float(1.0 - np.mean(acc_vals)) if len(acc_vals) else np.nan,
                }
            )

        ctx_df = pd.DataFrame(context_rows)

        if ctx_df.empty or len(ctx_df) < 3:
            continue

        # TAIRID context axes.
        T_raw = ctx_df["rt_median"].astype(float).values
        I_raw = (
            ctx_df["rt_cv"].fillna(ctx_df["rt_cv"].median()).astype(float).values
            + ctx_df["error_rate"].fillna(ctx_df["error_rate"].median()).astype(float).values
        )

        # Add a small constraint term for unrelated/unknown contexts.
        rel_norm = ctx_df["relatedness"].astype(str).str.lower().values
        I_raw = I_raw + np.asarray([0.20 if r == "unrelated" else 0.05 if r == "unknown" else 0.0 for r in rel_norm])

        T = zscore(T_raw)
        I = zscore(I_raw)
        M = np.abs(T - I)
        C = np.sqrt(T ** 2 + I ** 2)

        ctx_df["T_pacing_proxy"] = T
        ctx_df["I_constraint_proxy"] = I
        ctx_df["M_mismatch_abs"] = M
        ctx_df["collapse_load_proxy"] = C

        # Sort contexts by semantic reach: direct related -> direct unrelated -> indirect related -> indirect unrelated,
        # then by SOA.
        def context_order(row):
            direct = str(row.get("directness", "unknown")).lower()
            rel = str(row.get("relatedness", "unknown")).lower()
            soa = row.get("soa", 0.0)
            if not np.isfinite(pd.to_numeric(pd.Series([soa]), errors="coerce").iloc[0]):
                soa = 0.0
            direct_rank = {"direct": 0, "unknown": 1, "indirect": 2}.get(direct, 1)
            rel_rank = {"related": 0, "weak_related": 1, "unknown": 2, "unrelated": 3}.get(rel, 2)
            return (direct_rank, rel_rank, float(soa))

        ctx_df["_order"] = ctx_df.apply(context_order, axis=1)
        ctx_df = ctx_df.sort_values("_order").reset_index(drop=True)

        T_ord = ctx_df["T_pacing_proxy"].values
        I_ord = ctx_df["I_constraint_proxy"].values
        M_ord = ctx_df["M_mismatch_abs"].values
        C_ord = ctx_df["collapse_load_proxy"].values
        x_ord = np.arange(len(ctx_df), dtype=float)

        base.update(
            {
                "context_count": int(len(ctx_df)),
                "M_mean": float(np.mean(M_ord)),
                "M_median": float(np.median(M_ord)),
                "M_max": float(np.max(M_ord)),
                "M_range": float(np.max(M_ord) - np.min(M_ord)),
                "M_iqr": float(np.percentile(M_ord, 75) - np.percentile(M_ord, 25)),
                "M_slope_context": slope(x_ord, M_ord),
                "M_curvature_context": curve(x_ord, M_ord),
                "T_context_slope": slope(x_ord, T_ord),
                "I_context_slope": slope(x_ord, I_ord),
                "T_I_slope_gap": float(abs(slope(x_ord, T_ord) - slope(x_ord, I_ord)))
                if np.isfinite(slope(x_ord, T_ord)) and np.isfinite(slope(x_ord, I_ord))
                else np.nan,
                "collapse_load_mean": float(np.mean(C_ord)),
                "collapse_load_max": float(np.max(C_ord)),
                "collapse_load_range": float(np.max(C_ord) - np.min(C_ord)),
                "Cyc_M_turns": sign_change_count(M_ord),
                "Cyc_T_turns": sign_change_count(T_ord),
                "Cyc_I_turns": sign_change_count(I_ord),
                "Cyc_C_turns": sign_change_count(C_ord),
                "Cyc_M_turn_rate": float(sign_change_count(M_ord) / max(len(M_ord) - 2, 1)),
                "Cyc_TI_opposition_fraction": float(np.mean(np.sign(np.diff(T_ord)) * np.sign(np.diff(I_ord)) < 0))
                if len(T_ord) > 2 and len(I_ord) > 2
                else np.nan,
            }
        )

        # Priming / semantic reach features.
        priming_rows = []

        for (directness, soa), g in sub.groupby(["directness", "soa"], dropna=False):
            related_rt = pd.to_numeric(
                g[g["relatedness"] == "related"]["rt"],
                errors="coerce",
            ).dropna().astype(float).values

            weak_rt = pd.to_numeric(
                g[g["relatedness"] == "weak_related"]["rt"],
                errors="coerce",
            ).dropna().astype(float).values

            unrelated_rt = pd.to_numeric(
                g[g["relatedness"] == "unrelated"]["rt"],
                errors="coerce",
            ).dropna().astype(float).values

            if len(unrelated_rt) >= 3 and len(related_rt) >= 3:
                priming_rows.append(
                    {
                        "directness": directness,
                        "soa": float(soa) if np.isfinite(soa) else np.nan,
                        "kind": "related",
                        "priming_ms": float(np.median(unrelated_rt) - np.median(related_rt)),
                    }
                )

            if len(unrelated_rt) >= 3 and len(weak_rt) >= 3:
                priming_rows.append(
                    {
                        "directness": directness,
                        "soa": float(soa) if np.isfinite(soa) else np.nan,
                        "kind": "weak_related",
                        "priming_ms": float(np.median(unrelated_rt) - np.median(weak_rt)),
                    }
                )

        prim_df = pd.DataFrame(priming_rows)

        direct_vals = []
        indirect_vals = []
        weak_vals = []

        if not prim_df.empty:
            direct_vals = prim_df[(prim_df["directness"] == "direct") & (prim_df["kind"] == "related")]["priming_ms"].dropna().astype(float).values
            indirect_vals = prim_df[(prim_df["directness"] == "indirect") & (prim_df["kind"] == "related")]["priming_ms"].dropna().astype(float).values
            weak_vals = prim_df[prim_df["kind"] == "weak_related"]["priming_ms"].dropna().astype(float).values

        direct_mean = float(np.mean(direct_vals)) if len(direct_vals) else np.nan
        indirect_mean = float(np.mean(indirect_vals)) if len(indirect_vals) else np.nan
        weak_mean = float(np.mean(weak_vals)) if len(weak_vals) else np.nan

        base.update(
            {
                "Reach_direct_priming_mean_ms": direct_mean,
                "Reach_indirect_priming_mean_ms": indirect_mean,
                "Reach_weak_priming_mean_ms": weak_mean,
                "Reach_indirect_minus_direct_ms": float(indirect_mean - direct_mean)
                if np.isfinite(indirect_mean) and np.isfinite(direct_mean)
                else np.nan,
                "Reach_indirect_to_direct_ratio": safe_div(indirect_mean, abs(direct_mean))
                if np.isfinite(indirect_mean) and np.isfinite(direct_mean)
                else np.nan,
                "Reach_semantic_priming_span_ms": float(np.nanmax([direct_mean, indirect_mean, weak_mean]) - np.nanmin([direct_mean, indirect_mean, weak_mean]))
                if np.sum(np.isfinite([direct_mean, indirect_mean, weak_mean])) >= 2
                else np.nan,
                "Reach_indirect_positive": int(indirect_mean > 0) if np.isfinite(indirect_mean) else np.nan,
                "Reach_direct_positive": int(direct_mean > 0) if np.isfinite(direct_mean) else np.nan,
            }
        )

        if not prim_df.empty:
            prim_df = prim_df.sort_values(["directness", "kind", "soa"])
            prim_vals = prim_df["priming_ms"].astype(float).values
            base["Cyc_priming_sign_changes"] = sign_change_count(prim_vals)
            base["Cyc_priming_turn_rate"] = float(sign_change_count(prim_vals) / max(len(prim_vals) - 2, 1))
            base["Reach_priming_max_ms"] = float(np.nanmax(prim_vals))
            base["Reach_priming_min_ms"] = float(np.nanmin(prim_vals))
        else:
            base["Cyc_priming_sign_changes"] = np.nan
            base["Cyc_priming_turn_rate"] = np.nan
            base["Reach_priming_max_ms"] = np.nan
            base["Reach_priming_min_ms"] = np.nan

        # Viability window over semantic contexts.
        baseline = float(np.percentile(M_ord, 25))
        local_sd = float(np.std(M_ord))

        for mult in TOLERANCE_MULTIPLIERS:
            suffix = f"tol{str(mult).replace('.', 'p')}"
            W = baseline + mult * local_sd
            B = np.maximum(0.0, M_ord - W)
            breached = B > 0.0
            stable = ~breached

            base[f"W_{suffix}"] = float(W)
            base[f"B_mean_{suffix}"] = float(np.mean(B))
            base[f"B_max_{suffix}"] = float(np.max(B))
            base[f"B_fraction_{suffix}"] = float(np.mean(breached))
            base[f"Reach_stable_fraction_{suffix}"] = float(np.mean(stable))
            base[f"Reach_breach_fraction_{suffix}"] = float(np.mean(breached))
            base[f"Reach_longest_stable_run_fraction_{suffix}"] = float(longest_true_run(stable) / max(len(stable), 1))
            base[f"Reach_longest_breach_run_fraction_{suffix}"] = float(longest_true_run(breached) / max(len(breached), 1))

            transitions = np.abs(np.diff(breached.astype(int))) if len(breached) > 1 else np.asarray([])
            base[f"Cyc_breach_transition_rate_{suffix}"] = float(np.mean(transitions)) if len(transitions) else 0.0
            base[f"Cyc_breach_reentry_rate_{suffix}"] = (
                float(np.sum((breached[:-1] == True) & (breached[1:] == False)) / max(len(breached) - 1, 1))
                if len(breached) > 1
                else 0.0
            )

        rows.append(base)

    return pd.DataFrame(rows)


def make_high_low_contrast(features_df, score_col):
    df = features_df.copy()
    vals = pd.to_numeric(df[score_col], errors="coerce")
    df = df[np.isfinite(vals)].copy()
    vals = pd.to_numeric(df[score_col], errors="coerce")

    if len(df) < 20 or vals.nunique() < 5:
        return pd.DataFrame(), {"status": "not_enough_score_range"}

    lo = float(vals.quantile(1 / 3))
    hi = float(vals.quantile(2 / 3))

    low = df[vals <= lo].copy()
    high = df[vals >= hi].copy()

    out = pd.concat([low, high], ignore_index=True)
    out["label"] = (pd.to_numeric(out[score_col], errors="coerce") >= hi).astype(int)

    meta = {
        "status": "ok",
        "score_col": score_col,
        "low_threshold": lo,
        "high_threshold": hi,
        "n_total_with_score": int(len(df)),
        "n_low": int(len(low)),
        "n_high": int(len(high)),
    }

    return out, meta


def model_feature_sets(df):
    static = [
        "rt_mean", "rt_median", "rt_std", "rt_cv", "rt_iqr",
        "accuracy_mean", "error_rate"
    ]

    semantic_reach = [
        "Reach_direct_priming_mean_ms",
        "Reach_indirect_priming_mean_ms",
        "Reach_weak_priming_mean_ms",
        "Reach_indirect_minus_direct_ms",
        "Reach_indirect_to_direct_ratio",
        "Reach_semantic_priming_span_ms",
        "Reach_indirect_positive",
        "Reach_direct_positive",
        "Reach_priming_max_ms",
        "Reach_priming_min_ms",
    ]

    mismatch = [
        "M_mean", "M_median", "M_max", "M_range", "M_iqr",
        "M_slope_context", "M_curvature_context", "T_I_slope_gap",
        "collapse_load_mean", "collapse_load_max", "collapse_load_range",
    ]

    viability = []
    cycling = [
        "Cyc_M_turns", "Cyc_T_turns", "Cyc_I_turns", "Cyc_C_turns",
        "Cyc_M_turn_rate", "Cyc_TI_opposition_fraction",
        "Cyc_priming_sign_changes", "Cyc_priming_turn_rate",
    ]

    reach_window = []

    for mult in TOLERANCE_MULTIPLIERS:
        suffix = f"tol{str(mult).replace('.', 'p')}"
        viability += [
            f"B_mean_{suffix}",
            f"B_max_{suffix}",
            f"B_fraction_{suffix}",
        ]
        reach_window += [
            f"Reach_stable_fraction_{suffix}",
            f"Reach_breach_fraction_{suffix}",
            f"Reach_longest_stable_run_fraction_{suffix}",
            f"Reach_longest_breach_run_fraction_{suffix}",
        ]
        cycling += [
            f"Cyc_breach_transition_rate_{suffix}",
            f"Cyc_breach_reentry_rate_{suffix}",
        ]

    sets = {
        "static_rt_accuracy_model": static,
        "semantic_reach_model": semantic_reach,
        "mismatch_dynamic_model": mismatch,
        "viability_window_model": viability,
        "reach_window_model": reach_window,
        "cycling_reentry_model": cycling,
        "reach_cycle_model": semantic_reach + reach_window + cycling,
        "combined_axis_model": static + semantic_reach + mismatch + viability + reach_window + cycling,
    }

    return {k: [c for c in v if c in df.columns] for k, v in sets.items()}


def auc_score(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[mask], labels[mask]

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
    k = max(2, min(5, int(np.min(counts[counts > 0]))))
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


def run_models_for_score(features_df, score_col):
    contrast_df, contrast_meta = make_high_low_contrast(features_df, score_col)

    model_rows = []
    perm_rows = []

    if contrast_df.empty or contrast_meta.get("status") != "ok":
        return {
            "score_col": score_col,
            "contrast_meta": contrast_meta,
            "model_rows": [],
            "permutation_rows": [],
            "status": "not_enough_score_data",
        }

    for name, cols in model_feature_sets(contrast_df).items():
        res = repeated_cv(contrast_df, cols)
        auc = res.get("auc_mean")

        model_rows.append(
            {
                "score_col": score_col,
                "model_name": name,
                **{k: v for k, v in res.items() if k != "feature_cols"},
                "feature_cols": " | ".join(res.get("feature_cols", [])),
            }
        )

        perm = permutation_test(contrast_df, cols, observed_auc=auc)

        perm_rows.append(
            {
                "score_col": score_col,
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

    return {
        "score_col": score_col,
        "contrast_meta": contrast_meta,
        "model_rows": model_rows,
        "permutation_rows": perm_rows,
        "status": "ok",
    }


def spearman_feature_tests(features_df, score_col):
    rows = []

    if score_col not in features_df.columns:
        return rows

    score = pd.to_numeric(features_df[score_col], errors="coerce")

    numeric_cols = [
        c for c in features_df.columns
        if c not in ["subject", score_col]
        and pd.api.types.is_numeric_dtype(features_df[c])
        and features_df[c].notnull().sum() >= 8
    ]

    for c in numeric_cols:
        vals = pd.to_numeric(features_df[c], errors="coerce")
        mask = np.isfinite(vals) & np.isfinite(score)

        if int(mask.sum()) < 8:
            continue

        rho, p = stats.spearmanr(vals[mask], score[mask], nan_policy="omit")

        if not np.isfinite(rho):
            continue

        rows.append(
            {
                "score_col": score_col,
                "feature": c,
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "n": int(mask.sum()),
            }
        )

    return add_bh_q_spearman(
        sorted(
            rows,
            key=lambda r: (
                -(abs(r["spearman_rho"]) if np.isfinite(r["spearman_rho"]) else 0),
                r["spearman_p"] if np.isfinite(r.get("spearman_p", np.nan)) else 999,
            ),
        )
    )


def add_bh_q_spearman(rows):
    rows = list(rows)
    pvals = []
    idxs = []

    for i, r in enumerate(rows):
        p = r.get("spearman_p")
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


def decide_status(model_rows, perm_rows, score_cols):
    by_score = defaultdict(dict)
    by_perm = defaultdict(dict)

    for r in model_rows:
        if r.get("status") == "ok":
            by_score[r["score_col"]][r["model_name"]] = r

    for r in perm_rows:
        if r.get("permutation_status") == "ok":
            by_perm[r["score_col"]][r["model_name"]] = r

    score_status = {}

    axis_names = [
        "semantic_reach_model",
        "mismatch_dynamic_model",
        "viability_window_model",
        "reach_window_model",
        "cycling_reentry_model",
        "reach_cycle_model",
        "combined_axis_model",
    ]

    for sc in score_cols:
        models = by_score.get(sc, {})
        perms = by_perm.get(sc, {})
        static_auc = models.get("static_rt_accuracy_model", {}).get("auc_mean")

        axis_values = [
            (models.get(name, {}).get("auc_mean"), name)
            for name in axis_names
            if models.get(name, {}).get("auc_mean") is not None
        ]

        best_axis_auc, best_axis_name = max(axis_values, key=lambda x: x[0]) if axis_values else (None, None)

        axis_ps = [
            perms.get(name, {}).get("p_value_ge_observed")
            for name in axis_names
            if perms.get(name, {}).get("p_value_ge_observed") is not None
        ]

        best_axis_p = min(axis_ps) if axis_ps else None

        semantic_auc = models.get("semantic_reach_model", {}).get("auc_mean")
        cycle_auc = models.get("cycling_reentry_model", {}).get("auc_mean")

        axis_support = (
            static_auc is not None
            and best_axis_auc is not None
            and best_axis_auc >= 0.60
            and best_axis_auc > static_auc + 0.03
        )

        axis_locked = bool(axis_support and best_axis_p is not None and best_axis_p <= 0.05)

        score_status[sc] = {
            "static_auc": static_auc,
            "semantic_reach_auc": semantic_auc,
            "cycling_reentry_auc": cycle_auc,
            "best_axis_auc": best_axis_auc,
            "best_axis_model": best_axis_name,
            "best_axis_permutation_p": best_axis_p,
            "axis_support": axis_support,
            "axis_locked": axis_locked,
        }

    locked_scores = [k for k, v in score_status.items() if v.get("axis_locked")]
    supported_scores = [k for k, v in score_status.items() if v.get("axis_support")]

    reach_locked = [
        k for k, v in score_status.items()
        if v.get("axis_locked") and v.get("best_axis_model") in ["semantic_reach_model", "reach_window_model", "reach_cycle_model"]
    ]

    if reach_locked:
        return (
            "schizotypal_semantic_reach_axis_locked",
            9,
            "Add schizotypal semantic reach as the reference-field reach lane and compare reach-first vs cycle-first fingerprints.",
            score_status,
        )

    if locked_scores:
        return (
            "schizotypal_semantic_axis_locked_nonreach_primary",
            8,
            "Use schizotypal data as semantic-axis support, but inspect whether the locked model is mismatch, viability, or cycling rather than reach.",
            score_status,
        )

    if supported_scores:
        return (
            "schizotypal_semantic_axis_supported_not_locked",
            7,
            "Treat the schizotypal lane as directional; inspect score mapping and table parsing before promotion.",
            score_status,
        )

    return (
        "schizotypal_semantic_reach_not_supported_in_this_pass",
        6,
        "This pass does not support a reach/cycle claim; preserve it as a constrained negative unless parser review finds missing score columns.",
        score_status,
    )


def plot_model_bars(model_rows, path):
    ok = [r for r in model_rows if r.get("status") == "ok" and r.get("auc_mean") is not None]

    if not ok:
        return None

    labels = [f"{r['score_col']}\n{r['model_name'].replace('_model','')}" for r in ok]
    vals = [float(r["auc_mean"]) for r in ok]
    x = np.arange(len(labels))

    plt.figure(figsize=(22, 8))
    plt.bar(x, vals)
    plt.axhline(0.5, linewidth=1)
    plt.xticks(x, labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Repeated CV AUC")
    plt.title("TAIRID schizotypal semantic reach / cycle v1")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def plot_axis_advantage(score_status, path):
    labels, vals = [], []

    for sc, st in score_status.items():
        static = st.get("static_auc")
        axis = st.get("best_axis_auc")
        if static is None or axis is None:
            continue
        labels.append(sc)
        vals.append(float(axis) - float(static))

    if not labels:
        return None

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    plt.bar(x, vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Best axis AUC - static RT/accuracy AUC")
    plt.title("Schizotypal semantic axis advantage over static")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def main():
    print("")
    print("TAIRID schizotypal semantic reach / cycle v1 starting.")
    print("Boundary: operational semantic axis test only; not proof, diagnosis, or medical advice.")
    print("")

    osf_files, osf_inventory = collect_osf_files()
    write_csv(OUTDIR / "schizotypal_osf_file_inventory.csv", osf_inventory)

    downloads = download_osf_files(osf_files)
    write_csv(OUTDIR / "schizotypal_download_ledger.csv", downloads)

    extraction = extract_archives(downloads)
    write_csv(OUTDIR / "schizotypal_extraction_ledger.csv", extraction)

    table_inventory, tables = table_inventory_and_candidates()
    write_csv(OUTDIR / "schizotypal_table_inventory.csv", table_inventory)

    score_table, score_sources = build_score_table(tables)

    if score_table is not None and not score_table.empty:
        score_table.to_csv(OUTDIR / "schizotypal_score_table_merged.csv", index=False)

    write_csv(OUTDIR / "schizotypal_score_sources.csv", score_sources)

    trial_df, trial_sources = build_trial_table(tables, score_table)
    write_csv(OUTDIR / "schizotypal_trial_sources.csv", trial_sources)

    if not trial_df.empty:
        trial_df.to_csv(OUTDIR / "schizotypal_trial_table_merged.csv", index=False)

    score_cols = choose_primary_scores(trial_df) if not trial_df.empty else []

    # Limit to strongest score columns for runtime and interpretability.
    score_cols = score_cols[:6]

    features_df = build_subject_semantic_axis_features(trial_df, score_cols) if not trial_df.empty else pd.DataFrame()

    if not features_df.empty:
        features_df.to_csv(OUTDIR / "schizotypal_subject_semantic_axis_features.csv", index=False)

    all_model_rows = []
    all_perm_rows = []
    all_spearman_rows = []
    contrast_meta = []

    for sc in score_cols:
        result = run_models_for_score(features_df, sc) if not features_df.empty else {
            "score_col": sc,
            "status": "no_features",
            "contrast_meta": {},
            "model_rows": [],
            "permutation_rows": [],
        }

        all_model_rows.extend(result.get("model_rows", []))
        all_perm_rows.extend(result.get("permutation_rows", []))

        contrast_meta.append(
            {
                "score_col": sc,
                "status": result.get("status"),
                **result.get("contrast_meta", {}),
            }
        )

        spearman_rows = spearman_feature_tests(features_df, sc) if not features_df.empty else []
        all_spearman_rows.extend(spearman_rows)

        write_csv(OUTDIR / f"schizotypal_{safe_name(sc)}_model_results.csv", result.get("model_rows", []))
        write_csv(OUTDIR / f"schizotypal_{safe_name(sc)}_permutation_results.csv", result.get("permutation_rows", []))
        write_csv(OUTDIR / f"schizotypal_{safe_name(sc)}_spearman_feature_tests.csv", spearman_rows)

    model_path = write_csv(OUTDIR / "schizotypal_all_model_results.csv", all_model_rows)
    perm_path = write_csv(OUTDIR / "schizotypal_all_permutation_results.csv", all_perm_rows)
    spearman_path = write_csv(OUTDIR / "schizotypal_all_spearman_feature_tests.csv", all_spearman_rows)
    contrast_meta_path = write_csv(OUTDIR / "schizotypal_contrast_meta.csv", contrast_meta)

    final_status, readiness_score, next_wall, score_status = decide_status(
        all_model_rows,
        all_perm_rows,
        score_cols,
    )

    plots = []

    p = plot_model_bars(all_model_rows, OUTDIR / "schizotypal_model_auc_bars.png")
    if p:
        plots.append(p)

    p = plot_axis_advantage(score_status, OUTDIR / "schizotypal_axis_advantage.png")
    if p:
        plots.append(p)

    summary = {
        "test_name": "TAIRID schizotypal semantic reach / cycle v1",
        "boundary": (
            "Operational semantic reach/cycle axis test only. Not proof of TAIRID, "
            "not diagnosis, not medical advice, and not a cosmology result."
        ),
        "dataset": {
            "osf_node_id": OSF_NODE_ID,
            "osf_files_url": OSF_FILES_URL,
            "note": (
                "This pass uses OSF project files for semantic priming and schizotypal personality, "
                "then builds trial-level lexical decision / semantic priming features where parsable."
            ),
        },
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "parser_counts": {
            "osf_file_inventory_count": len(osf_inventory),
            "download_count": len(downloads),
            "extraction_count": len(extraction),
            "table_count": len(tables),
            "score_source_count": len(score_sources),
            "trial_source_count": len(trial_sources),
            "trial_rows": int(len(trial_df)) if not trial_df.empty else 0,
            "subject_feature_rows": int(len(features_df)) if not features_df.empty else 0,
            "score_cols_used": score_cols,
        },
        "score_status": score_status,
        "contrast_meta": contrast_meta,
        "model_results": all_model_rows,
        "permutation_results": all_perm_rows,
        "top_spearman_feature_tests": all_spearman_rows[:50],
        "output_files": {
            "osf_file_inventory_csv": str(OUTDIR / "schizotypal_osf_file_inventory.csv"),
            "download_ledger_csv": str(OUTDIR / "schizotypal_download_ledger.csv"),
            "table_inventory_csv": str(OUTDIR / "schizotypal_table_inventory.csv"),
            "score_table_csv": str(OUTDIR / "schizotypal_score_table_merged.csv") if score_table is not None and not score_table.empty else None,
            "trial_table_csv": str(OUTDIR / "schizotypal_trial_table_merged.csv") if not trial_df.empty else None,
            "subject_semantic_axis_features_csv": str(OUTDIR / "schizotypal_subject_semantic_axis_features.csv") if not features_df.empty else None,
            "model_results_csv": str(model_path),
            "permutation_results_csv": str(perm_path),
            "spearman_feature_tests_csv": str(spearman_path),
            "contrast_meta_csv": str(contrast_meta_path),
            "plots": plots,
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "Semantic reach, reach-window, cycling/re-entry, mismatch, or viability models outperform "
                "static RT/accuracy and beat permutation expectation for high-vs-low schizotypy scores."
            ),
            "what_weakens_the_lane": (
                "Static RT/accuracy remains strongest, score columns cannot be recovered, or semantic reach/cycle "
                "models fail permutation."
            ),
            "axis_prediction": (
                "Schizotypal traits should stress semantic/reference reach before cycling: association spread, "
                "indirect priming, boundary permeability, and possible re-entry when reach exceeds stable constraint."
            ),
            "truth_boundary": (
                "A positive result supports schizotypal semantic reach as an operational TAIRID lane. "
                "It cannot diagnose schizotypy, prove TAIRID, or transfer proof to cosmology."
            ),
        },
    }

    write_json(OUTDIR / "schizotypal_semantic_reach_cycle_v1_summary.json", summary)

    with open(OUTDIR / "schizotypal_semantic_reach_cycle_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID schizotypal semantic reach / cycle v1\n\n")
        f.write("Boundary: operational semantic axis test only. Not proof. Not diagnosis. Not medical advice.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Why this test exists:\n")
        f.write("- Bipolar/mood actigraphy helped lock cycling/hysteresis in real time-series data.\n")
        f.write("- Schizotypal semantic priming should help separate reach-first semantic propagation from cycle-first rhythm behavior.\n")
        f.write("- This pass tests semantic reach, indirect/direct priming, mismatch, viability, and re-entry/cycling against schizotypy scores.\n\n")

        f.write("Score status:\n")
        f.write(json.dumps(score_status, indent=2, default=str) + "\n\n")

        f.write("Parser counts:\n")
        f.write(json.dumps(summary["parser_counts"], indent=2, default=str) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This can support a schizotypal semantic reach/cycle lane.\n")
        f.write("- It cannot prove TAIRID.\n")
        f.write("- It cannot diagnose schizotypy or schizotypal personality disorder.\n")
        f.write("- It cannot prove any cosmology claim.\n")

    print("")
    print("TAIRID schizotypal semantic reach / cycle v1 complete.")
    print("Created:")
    print("  tairid_schizotypal_semantic_reach_cycle_v1_outputs/schizotypal_semantic_reach_cycle_v1_summary.json")
    print("  tairid_schizotypal_semantic_reach_cycle_v1_outputs/schizotypal_semantic_reach_cycle_v1_summary.txt")
    print("  tairid_schizotypal_semantic_reach_cycle_v1_outputs/schizotypal_all_model_results.csv")
    print("  tairid_schizotypal_semantic_reach_cycle_v1_outputs/schizotypal_all_permutation_results.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is not diagnosis or medical advice.")
    print("  This is a semantic reach/cycle axis test.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
