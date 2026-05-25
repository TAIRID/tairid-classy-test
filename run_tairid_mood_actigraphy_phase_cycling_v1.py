#!/usr/bin/env python3
"""
TAIRID mood / actigraphy phase-cycling test v1.

Purpose:
The ADHD event-table test detected static differences but did not lock cycling/reach.
That likely happened because block-summary events flattened the time sequence needed
to test cycling.

This test moves the cycling lane to longitudinal motor-activity data.

Dataset:
Zenodo record 13754984:
OBF-Psychiatric motor-activity dataset, including bipolar and unipolar depression,
schizophrenia, ADHD, clinical samples, and healthy controls.

Core question:
Do longitudinal motor-activity recordings show TAIRID cycling / reach / hysteresis axes
better than simple static activity averages?

TAIRID translation:
T = activity pacing / rhythm / propagation tempo
I = instability / constraint / rhythm-fragmentation pressure
M = |T - I|
W = viability window / baseline tolerated mismatch
B = breach beyond W
Reach = longest stable span / stable fraction before breach
Cycling = oscillation, turns, recurrence, breach transitions
H = hysteresis / recovery asymmetry after breach

Boundary:
This is not proof of TAIRID.
This is not diagnosis.
This is not medical advice.
This is not a cosmology result.
It is an operational time-series axis test.
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
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


OUTDIR = Path("tairid_mood_actigraphy_phase_cycling_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_DIR = OUTDIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD_ID = "13754984"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

RANDOM_SEED = 42
CV_REPEATS = 100
PERMUTATIONS = 250
PERM_REPEATS = 25
RIDGE = 1.0e-3

TOLERANCE_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0]

TIME_HINTS = [
    "time", "timestamp", "datetime", "date", "minute", "day", "hour"
]

ACTIVITY_HINTS = [
    "activity", "activity_count", "activitycount", "motor", "actigraph",
    "counts", "count", "zcm", "pim", "steps", "accel", "movement",
    "mad", "vm", "vector"
]

SUBJECT_HINTS = [
    "subject", "subject_id", "subjectid", "participant", "participant_id",
    "participantid", "patient", "patient_id", "id", "number", "subj"
]

GROUP_HINTS = [
    "group", "diagnosis", "diagnostic", "condition", "class", "label",
    "afftype", "type", "category", "cohort"
]

ID_ARTIFACT_HINTS = [
    "id", "number", "index", "row", "unnamed", "subject", "participant",
    "patient", "date", "time", "timestamp", "datetime"
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


def read_url(url, timeout=600):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-mood-actigraphy-phase-cycling-v1",
            "Accept": "application/json, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
    return data, final_url, content_type


def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    return name[:180]


def download_zenodo_record():
    data, final_url, content_type = read_url(ZENODO_API_URL)
    record = json.loads(data.decode("utf-8", errors="replace"))

    (OUTDIR / "zenodo_record_13754984.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    downloads = []

    for f in record.get("files", []):
        key = f.get("key") or f.get("filename") or "unknown"
        links = f.get("links", {})
        url = links.get("self") or links.get("download")

        if not url:
            downloads.append(
                {
                    "key": key,
                    "status": "no_download_url",
                    "size_declared": f.get("size"),
                }
            )
            continue

        local = DOWNLOAD_DIR / safe_name(key)

        try:
            print(f"Downloading {key} ...")
            payload, got_url, got_content_type = read_url(url, timeout=1200)
            local.write_bytes(payload)

            downloads.append(
                {
                    "key": key,
                    "status": "downloaded",
                    "url": url,
                    "final_url": got_url,
                    "content_type": got_content_type,
                    "size_declared": f.get("size"),
                    "size_downloaded": local.stat().st_size,
                    "path": str(local),
                    "sha256": sha256_file(local),
                }
            )
        except Exception as exc:
            downloads.append(
                {
                    "key": key,
                    "status": "download_failed",
                    "url": url,
                    "size_declared": f.get("size"),
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


def is_junk_path(path):
    parts = [p.lower() for p in Path(path).parts]
    name = Path(path).name.lower()
    return "__macosx" in parts or name.startswith("._") or name.startswith(".")


def list_table_files():
    files = []

    for root in [DOWNLOAD_DIR, EXTRACT_DIR]:
        for path in root.rglob("*"):
            if not path.is_file() or is_junk_path(path):
                continue

            lower = path.name.lower()
            if lower.endswith((".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls")):
                files.append(path)

    return sorted(set(files))


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

    for hint in hints:
        h = norm(hint)
        for c, cn in by_norm.items():
            if cn in avoid_norm:
                continue
            if cn == h:
                return c

    for hint in hints:
        h = norm(hint)
        for c, cn in by_norm.items():
            if cn in avoid_norm:
                continue
            if h and h in cn:
                return c

    return None


def infer_group_from_text(text):
    s = str(text).lower()
    compact = norm(s)

    # Control first, to avoid "non-control" mistakes.
    if any(x in compact for x in ["healthycontrol", "control", "hc", "healthy"]):
        return "control"

    if any(x in compact for x in ["bipolar", "bipolardepression", "bipolardisorder", "bd", "bpd"]):
        return "bipolar"

    if any(x in compact for x in ["unipolar", "mdd", "majordepression", "depression", "depressed", "depresjon"]):
        return "unipolar_or_depression"

    if any(x in compact for x in ["schizophrenia", "schizo", "sz"]):
        return "schizophrenia"

    if any(x in compact for x in ["adhd", "attentiondeficit"]):
        return "adhd"

    if any(x in compact for x in ["clinical", "patient", "condition"]):
        return "clinical_or_condition"

    return None


def infer_subject_from_path(path):
    text = str(path)
    stem = Path(path).stem

    patterns = [
        r"(sub[-_ ]?[A-Za-z0-9]+)",
        r"(subject[-_ ]?[A-Za-z0-9]+)",
        r"(patient[-_ ]?[A-Za-z0-9]+)",
        r"(condition[-_ ]?[A-Za-z0-9]+)",
        r"(control[-_ ]?[A-Za-z0-9]+)",
        r"(bipolar[-_ ]?[A-Za-z0-9]+)",
        r"(unipolar[-_ ]?[A-Za-z0-9]+)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return re.sub(r"[^A-Za-z0-9]+", "_", m.group(1)).strip("_")

    m = re.search(r"([0-9]{1,5})", stem)
    if m:
        return m.group(1)

    return stem[:80]


def normalize_subject_key(value):
    return norm(str(value))


def build_metadata_map():
    maps = {}
    sources = []

    for path in list_table_files():
        df, err = read_table(path)
        if df is None:
            continue

        if len(df) > 1000:
            continue

        columns = list(df.columns)
        subject_col = find_col(columns, SUBJECT_HINTS)
        group_col = find_col(columns, GROUP_HINTS)

        if subject_col is None:
            continue

        if group_col is None:
            # Try path-level group as metadata source.
            path_group = infer_group_from_text(path)
        else:
            path_group = None

        parsed = 0

        for _, row in df.iterrows():
            sid = str(row.get(subject_col, "")).strip()
            if not sid:
                continue

            group = None
            if group_col is not None:
                group = infer_group_from_text(row.get(group_col))

            if group is None:
                group = path_group

            if group is None:
                continue

            for key in [sid, normalize_subject_key(sid)]:
                maps[key] = group

            # Also map numeric part when present.
            m = re.search(r"([0-9]{1,5})", sid)
            if m:
                maps[m.group(1)] = group

            parsed += 1

        if parsed:
            sources.append(
                {
                    "path": str(path),
                    "status": "parsed_metadata",
                    "subject_col": subject_col,
                    "group_col": group_col,
                    "parsed": parsed,
                    "columns": " | ".join(columns),
                }
            )

    return maps, sources


def choose_activity_column(df):
    numeric_cols = []

    for c in df.columns:
        if any(h in norm(c) for h in [norm(x) for x in ID_ARTIFACT_HINTS]):
            continue

        vals = pd.to_numeric(df[c], errors="coerce")
        finite_count = int(np.isfinite(vals).sum())

        if finite_count < max(20, int(0.50 * len(df))):
            continue

        variance = float(np.nanvar(vals))
        if not np.isfinite(variance) or variance <= 0:
            continue

        score = 0
        cn = norm(c)

        if any(norm(h) in cn for h in ACTIVITY_HINTS):
            score += 10

        score += min(finite_count / 1000.0, 5.0)
        score += min(math.log1p(max(variance, 0.0)), 5.0)

        numeric_cols.append((score, c, finite_count, variance))

    if not numeric_cols:
        return None

    numeric_cols.sort(reverse=True, key=lambda x: x[0])
    return numeric_cols[0][1]


def parse_time_index(df):
    columns = list(df.columns)
    by_norm = {c: norm(c) for c in columns}

    datetime_candidates = [
        c for c in columns
        if any(h in by_norm[c] for h in ["datetime", "timestamp", "date_time", "time"])
    ]

    for c in datetime_candidates:
        vals = pd.to_datetime(df[c], errors="coerce")
        if vals.notnull().sum() >= max(20, int(0.50 * len(df))):
            return vals, True, c

    date_col = None
    time_col = None

    for c in columns:
        cn = by_norm[c]
        if date_col is None and "date" in cn:
            date_col = c
        if time_col is None and "time" in cn:
            time_col = c

    if date_col is not None and time_col is not None and date_col != time_col:
        vals = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str),
            errors="coerce",
        )
        if vals.notnull().sum() >= max(20, int(0.50 * len(df))):
            return vals, True, f"{date_col}+{time_col}"

    if date_col is not None:
        vals = pd.to_datetime(df[date_col], errors="coerce")
        if vals.notnull().sum() >= max(20, int(0.50 * len(df))):
            return vals, True, date_col

    # Fallback to minute index.
    vals = pd.date_range("2000-01-01", periods=len(df), freq="min")
    return pd.Series(vals), False, "synthetic_minute_index"


def load_activity_series(metadata_map):
    inventory = []
    series_rows = []

    for path in list_table_files():
        df, err = read_table(path)

        if df is None:
            inventory.append({"path": str(path), "status": "read_failed", "error": err})
            continue

        if len(df) < 100:
            inventory.append({"path": str(path), "status": "too_few_rows", "rows": int(len(df))})
            continue

        act_col = choose_activity_column(df)
        if act_col is None:
            inventory.append({"path": str(path), "status": "no_activity_col", "rows": int(len(df)), "columns": " | ".join(df.columns)})
            continue

        times, has_real_time, time_source = parse_time_index(df)
        activity = pd.to_numeric(df[act_col], errors="coerce")

        temp = pd.DataFrame({"time": times, "activity": activity})
        temp = temp.dropna()
        temp = temp[np.isfinite(temp["activity"])]
        temp = temp.sort_values("time")

        if len(temp) < 100:
            inventory.append({"path": str(path), "status": "activity_series_too_short_after_clean", "rows": int(len(temp))})
            continue

        subject = infer_subject_from_path(path)

        # Try subject columns.
        subject_col = find_col(df.columns, SUBJECT_HINTS)
        if subject_col is not None:
            nonnull = df[subject_col].dropna().astype(str)
            if len(nonnull):
                subject = str(nonnull.iloc[0])

        group = None
        for key in [subject, normalize_subject_key(subject)]:
            if key in metadata_map:
                group = metadata_map[key]
                break

        if group is None:
            m = re.search(r"([0-9]{1,5})", str(subject))
            if m and m.group(1) in metadata_map:
                group = metadata_map[m.group(1)]

        if group is None:
            group = infer_group_from_text(path)

        if group is None:
            group = "unknown"

        local_csv = OUTDIR / "normalized_series" / f"{safe_name(str(subject))}__{safe_name(path.stem)}.csv"
        local_csv.parent.mkdir(parents=True, exist_ok=True)
        temp.to_csv(local_csv, index=False)

        inventory.append(
            {
                "path": str(path),
                "status": "activity_series_parsed",
                "subject": str(subject),
                "group": group,
                "rows": int(len(temp)),
                "activity_col": act_col,
                "time_source": time_source,
                "has_real_time": bool(has_real_time),
                "normalized_series_csv": str(local_csv),
            }
        )

        series_rows.append(
            {
                "subject": str(subject),
                "group": group,
                "source_file": str(path),
                "activity_col": act_col,
                "time_source": time_source,
                "has_real_time": bool(has_real_time),
                "series_csv": str(local_csv),
                "row_count": int(len(temp)),
            }
        )

    return inventory, series_rows


def zscore(vals):
    vals = np.asarray(vals, dtype=float)
    mu = np.nanmean(vals)
    sd = np.nanstd(vals)
    if not np.isfinite(sd) or sd <= 1.0e-12:
        return np.zeros_like(vals)
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
    current = 0
    for x in flags:
        if bool(x):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def autocorr(vals, lag):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) <= lag + 2:
        return np.nan

    a = vals[:-lag]
    b = vals[lag:]

    if np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def spectral_features(vals, sample_period_hours=1.0):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) < 48 or np.std(vals) <= 1.0e-12:
        return {
            "fft_dominant_period_hours": np.nan,
            "fft_dominant_power_fraction": np.nan,
            "fft_24h_power_fraction": np.nan,
        }

    centered = vals - np.mean(vals)
    power = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=sample_period_hours)

    if len(power) <= 2 or np.sum(power[1:]) <= 1.0e-12:
        return {
            "fft_dominant_period_hours": np.nan,
            "fft_dominant_power_fraction": np.nan,
            "fft_24h_power_fraction": np.nan,
        }

    power[0] = 0.0
    idx = int(np.argmax(power))
    dom_freq = freqs[idx]

    dom_period = float(1.0 / dom_freq) if dom_freq > 0 else np.nan
    dom_frac = float(power[idx] / np.sum(power))

    target_freq = 1.0 / 24.0
    idx24 = int(np.argmin(np.abs(freqs - target_freq)))
    frac24 = float(power[idx24] / np.sum(power))

    return {
        "fft_dominant_period_hours": dom_period,
        "fft_dominant_power_fraction": dom_frac,
        "fft_24h_power_fraction": frac24,
    }


def intradaily_variability(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) < 3:
        return np.nan

    var = np.var(vals)
    if var <= 1.0e-12:
        return np.nan

    return float(np.mean(np.diff(vals) ** 2) / var)


def interdaily_stability(hourly_series):
    if hourly_series.empty:
        return np.nan

    s = hourly_series.copy().dropna()
    if len(s) < 48 or np.var(s.values) <= 1.0e-12:
        return np.nan

    by_hour = s.groupby(s.index.hour).mean()
    if len(by_hour) < 12:
        return np.nan

    return float(np.var(by_hour.values) / np.var(s.values))


def build_subject_features(series_rows):
    raw_features = []
    series_cache = []

    for row in series_rows:
        path = Path(row["series_csv"])
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna().sort_values("time")

        if len(df) < 100:
            continue

        activity = pd.to_numeric(df["activity"], errors="coerce").astype(float)
        activity = np.maximum(activity.values, 0.0)
        log_activity = np.log1p(activity)

        ts = pd.Series(log_activity, index=df["time"])
        ts = ts[~ts.index.duplicated(keep="first")]

        # Hourly regularized series for rhythm measures.
        hourly = ts.resample("60min").mean().interpolate(limit=3).dropna()
        if len(hourly) < 24:
            # Fallback to chunking if real timestamps were not meaningful.
            chunks = []
            step = 60
            for i in range(0, len(log_activity), step):
                chunk = log_activity[i:i + step]
                if len(chunk) >= 10:
                    chunks.append(float(np.mean(chunk)))
            hourly = pd.Series(
                chunks,
                index=pd.date_range("2000-01-01", periods=len(chunks), freq="60min"),
            )

        if len(hourly) < 24:
            continue

        vals = hourly.values.astype(float)

        # TAIRID internal series:
        # T = activity pacing / tempo.
        # I = local instability / rhythm-fragmentation pressure.
        T = zscore(vals)

        local_diff = np.abs(np.diff(vals, prepend=vals[0]))
        rolling_vol = pd.Series(vals).rolling(6, min_periods=2).std().fillna(0.0).values
        I = zscore(local_diff + rolling_vol)

        M = np.abs(T - I)
        C = np.sqrt(T ** 2 + I ** 2)

        subject = row["subject"]
        group = row["group"]

        base = {
            "subject": subject,
            "group": group,
            "source_file": row["source_file"],
            "row_count": int(row["row_count"]),
            "hourly_count": int(len(vals)),
            "duration_hours": float(len(vals)),
            "duration_days": float(len(vals) / 24.0),
            "activity_mean": float(np.mean(vals)),
            "activity_median": float(np.median(vals)),
            "activity_std": float(np.std(vals)),
            "activity_cv": float(np.std(vals) / max(abs(np.mean(vals)), 1.0e-9)),
            "activity_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "activity_range": float(np.max(vals) - np.min(vals)),
            "activity_active_fraction": float(np.mean(vals > np.percentile(vals, 50))),
            "T_mean": float(np.mean(T)),
            "I_mean": float(np.mean(I)),
            "M_mean": float(np.mean(M)),
            "M_median": float(np.median(M)),
            "M_max": float(np.max(M)),
            "M_std": float(np.std(M)),
            "M_iqr": float(np.percentile(M, 75) - np.percentile(M, 25)),
            "M_range": float(np.max(M) - np.min(M)),
            "collapse_load_mean": float(np.mean(C)),
            "collapse_load_max": float(np.max(C)),
            "collapse_load_range": float(np.max(C) - np.min(C)),
            "Cyc_activity_turns": sign_change_count(vals),
            "Cyc_M_turns": sign_change_count(M),
            "Cyc_T_turns": sign_change_count(T),
            "Cyc_I_turns": sign_change_count(I),
            "Cyc_activity_turn_rate": float(sign_change_count(vals) / max(len(vals) - 2, 1)),
            "Cyc_M_turn_rate": float(sign_change_count(M) / max(len(M) - 2, 1)),
            "Cyc_TI_opposition_fraction": float(np.mean(np.sign(np.diff(T)) * np.sign(np.diff(I)) < 0)) if len(T) > 2 else np.nan,
            "Rhythm_autocorr_1h": autocorr(vals, 1),
            "Rhythm_autocorr_24h": autocorr(vals, 24),
            "Rhythm_intradaily_variability": intradaily_variability(vals),
            "Rhythm_interdaily_stability": interdaily_stability(hourly),
            **spectral_features(vals, sample_period_hours=1.0),
        }

        # Hysteresis / recovery preliminary based on M derivative.
        dM = np.diff(M)
        rising = dM[dM > 0]
        falling = -dM[dM < 0]

        base["H_mismatch_rise_mean"] = float(np.mean(rising)) if len(rising) else 0.0
        base["H_mismatch_fall_mean"] = float(np.mean(falling)) if len(falling) else 0.0
        base["H_rise_fall_asymmetry"] = float(base["H_mismatch_rise_mean"] - base["H_mismatch_fall_mean"])

        series_cache.append(
            {
                "subject": subject,
                "group": group,
                "T": T,
                "I": I,
                "M": M,
                "C": C,
                "vals": vals,
                "base": base,
            }
        )

    # Population window scale from controls if available.
    control_q25 = [
        float(np.percentile(x["M"], 25))
        for x in series_cache
        if x["group"] == "control" and len(x["M"]) > 10
    ]

    if len(control_q25) >= 2:
        control_m_q25_sd = float(np.std(control_q25, ddof=1))
    else:
        all_q25 = [float(np.percentile(x["M"], 25)) for x in series_cache if len(x["M"]) > 10]
        control_m_q25_sd = float(np.std(all_q25, ddof=1)) if len(all_q25) >= 2 else 0.0

    if not np.isfinite(control_m_q25_sd):
        control_m_q25_sd = 0.0

    for x in series_cache:
        M = x["M"]
        C = x["C"]
        base = x["base"]
        baseline = float(np.percentile(M, 25))

        base["W_baseline_M_q25"] = baseline
        base["W_control_M_q25_sd_used"] = control_m_q25_sd

        for mult in TOLERANCE_MULTIPLIERS:
            suffix = f"tol{str(mult).replace('.', 'p')}"
            W = baseline + mult * control_m_q25_sd
            B = np.maximum(0.0, M - W)
            breached = B > 0.0
            stable = ~breached

            base[f"W_{suffix}"] = float(W)
            base[f"B_total_{suffix}"] = float(np.sum(B))
            base[f"B_mean_{suffix}"] = float(np.mean(B))
            base[f"B_max_{suffix}"] = float(np.max(B))
            base[f"B_fraction_{suffix}"] = float(np.mean(breached))
            base[f"B_count_{suffix}"] = int(np.sum(breached))

            base[f"Reach_stable_fraction_{suffix}"] = float(np.mean(stable))
            base[f"Reach_breach_fraction_{suffix}"] = float(np.mean(breached))
            base[f"Reach_longest_stable_run_hours_{suffix}"] = longest_true_run(stable)
            base[f"Reach_longest_breach_run_hours_{suffix}"] = longest_true_run(breached)

            transitions = np.abs(np.diff(breached.astype(int)))
            base[f"Cyc_breach_transition_count_{suffix}"] = int(np.sum(transitions)) if len(transitions) else 0
            base[f"Cyc_breach_transition_rate_{suffix}"] = float(np.mean(transitions)) if len(transitions) else 0.0
            base[f"Cyc_breach_reentry_count_{suffix}"] = int(np.sum((breached[:-1] == True) & (breached[1:] == False))) if len(breached) > 1 else 0

            # Hysteresis: average time from breach start to next stable re-entry.
            recovery_lengths = []
            in_breach = False
            length = 0

            for flag in breached:
                if flag and not in_breach:
                    in_breach = True
                    length = 1
                elif flag and in_breach:
                    length += 1
                elif (not flag) and in_breach:
                    recovery_lengths.append(length)
                    in_breach = False
                    length = 0

            base[f"H_breach_episode_mean_hours_{suffix}"] = float(np.mean(recovery_lengths)) if recovery_lengths else 0.0
            base[f"H_breach_episode_max_hours_{suffix}"] = float(np.max(recovery_lengths)) if recovery_lengths else 0.0

        raw_features.append(base)

    return pd.DataFrame(raw_features)


def group_to_label_rows(features_df, contrast):
    df = features_df.copy()

    if contrast == "bipolar_vs_control":
        df = df[df["group"].isin(["bipolar", "control"])].copy()
        df["label"] = (df["group"] == "bipolar").astype(int)
        positive = "bipolar"
        negative = "control"

    elif contrast == "mood_vs_control":
        mood_groups = ["bipolar", "unipolar_or_depression", "clinical_or_condition"]
        df = df[df["group"].isin(mood_groups + ["control"])].copy()
        df["label"] = df["group"].isin(mood_groups).astype(int)
        positive = "mood_or_depression"
        negative = "control"

    elif contrast == "bipolar_vs_unipolar":
        df = df[df["group"].isin(["bipolar", "unipolar_or_depression"])].copy()
        df["label"] = (df["group"] == "bipolar").astype(int)
        positive = "bipolar"
        negative = "unipolar_or_depression"

    else:
        raise ValueError(f"Unknown contrast: {contrast}")

    return df, positive, negative


def model_feature_sets(df):
    static = [
        "activity_mean", "activity_median", "activity_std", "activity_cv",
        "activity_iqr", "M_mean", "M_max", "collapse_load_mean",
    ]

    rhythm = [
        "Rhythm_autocorr_1h", "Rhythm_autocorr_24h",
        "Rhythm_intradaily_variability", "Rhythm_interdaily_stability",
        "fft_dominant_period_hours", "fft_dominant_power_fraction",
        "fft_24h_power_fraction",
    ]

    cycling = [
        "Cyc_activity_turns", "Cyc_M_turns", "Cyc_T_turns", "Cyc_I_turns",
        "Cyc_activity_turn_rate", "Cyc_M_turn_rate", "Cyc_TI_opposition_fraction",
    ]

    reach = []
    hysteresis = [
        "H_mismatch_rise_mean", "H_mismatch_fall_mean", "H_rise_fall_asymmetry",
    ]

    viability = []

    for mult in TOLERANCE_MULTIPLIERS:
        suffix = f"tol{str(mult).replace('.', 'p')}"
        viability.extend(
            [
                f"B_total_{suffix}", f"B_mean_{suffix}", f"B_max_{suffix}",
                f"B_fraction_{suffix}", f"B_count_{suffix}",
            ]
        )
        reach.extend(
            [
                f"Reach_stable_fraction_{suffix}",
                f"Reach_breach_fraction_{suffix}",
                f"Reach_longest_stable_run_hours_{suffix}",
                f"Reach_longest_breach_run_hours_{suffix}",
            ]
        )
        cycling.extend(
            [
                f"Cyc_breach_transition_count_{suffix}",
                f"Cyc_breach_transition_rate_{suffix}",
                f"Cyc_breach_reentry_count_{suffix}",
            ]
        )
        hysteresis.extend(
            [
                f"H_breach_episode_mean_hours_{suffix}",
                f"H_breach_episode_max_hours_{suffix}",
            ]
        )

    all_sets = {
        "static_activity_model": static,
        "rhythm_model": rhythm,
        "cycling_model": cycling,
        "viability_breach_model": viability,
        "reach_model": reach,
        "hysteresis_model": hysteresis,
        "cycle_reach_hysteresis_model": cycling + reach + hysteresis,
        "combined_axis_model": static + rhythm + cycling + viability + reach + hysteresis,
    }

    return {k: [c for c in v if c in df.columns] for k, v in all_sets.items()}


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


def run_models_for_contrast(features_df, contrast):
    df, positive, negative = group_to_label_rows(features_df, contrast)

    counts = {str(k): int(v) for k, v in Counter(df["group"]).items()}
    usable = df["label"].nunique() == 2 and min(Counter(df["label"]).values()) >= 6

    model_rows = []
    perm_rows = []

    if not usable:
        return {
            "contrast": contrast,
            "status": "not_enough_labeled_rows",
            "positive_label": positive,
            "negative_label": negative,
            "group_counts": counts,
            "n_rows": int(len(df)),
            "model_rows": [],
            "permutation_rows": [],
        }

    for name, cols in model_feature_sets(df).items():
        res = repeated_cv(df, cols)
        auc = res.get("auc_mean")

        model_rows.append(
            {
                "contrast": contrast,
                "model_name": name,
                **{k: v for k, v in res.items() if k != "feature_cols"},
                "feature_cols": " | ".join(res.get("feature_cols", [])),
            }
        )

        perm = permutation_test(df, cols, observed_auc=auc)

        perm_rows.append(
            {
                "contrast": contrast,
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
        "contrast": contrast,
        "status": "ok",
        "positive_label": positive,
        "negative_label": negative,
        "group_counts": counts,
        "n_rows": int(len(df)),
        "model_rows": model_rows,
        "permutation_rows": perm_rows,
    }


def feature_tests_for_contrast(features_df, contrast):
    df, positive, negative = group_to_label_rows(features_df, contrast)

    if df["label"].nunique() < 2:
        return []

    rows = []
    numeric_cols = [
        c for c in df.columns
        if c not in ["subject", "group", "source_file", "activity_col", "time_source", "label"]
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
                "contrast": contrast,
                "feature": c,
                "positive_label": positive,
                "negative_label": negative,
                "positive_mean": float(np.mean(pos)),
                "negative_mean": float(np.mean(neg)),
                "cohen_d_positive_minus_negative": d,
                "welch_t": float(tstat),
                "welch_p": float(pval),
                "oriented_auc": auc_score(scores, labels),
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
            }
        )

    return add_bh_q(
        sorted(
            rows,
            key=lambda r: (
                -(abs(r["cohen_d_positive_minus_negative"]) if np.isfinite(r["cohen_d_positive_minus_negative"]) else 0),
                r["welch_p"] if np.isfinite(r["welch_p"]) else 999,
            ),
        )
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


def decide_status(contrast_results):
    status_map = {}

    for result in contrast_results:
        if result.get("status") != "ok":
            status_map[result["contrast"]] = {
                "status": result.get("status"),
                "n_rows": result.get("n_rows"),
                "group_counts": result.get("group_counts"),
            }
            continue

        models = {r["model_name"]: r for r in result["model_rows"]}
        perms = {r["model_name"]: r for r in result["permutation_rows"]}

        static_auc = models.get("static_activity_model", {}).get("auc_mean")

        axis_names = [
            "rhythm_model",
            "cycling_model",
            "viability_breach_model",
            "reach_model",
            "hysteresis_model",
            "cycle_reach_hysteresis_model",
            "combined_axis_model",
        ]

        axis_aucs = [
            models.get(name, {}).get("auc_mean")
            for name in axis_names
            if models.get(name, {}).get("auc_mean") is not None
        ]
        best_axis_auc = max(axis_aucs) if axis_aucs else None

        axis_ps = [
            perms.get(name, {}).get("p_value_ge_observed")
            for name in axis_names
            if perms.get(name, {}).get("p_value_ge_observed") is not None
        ]
        best_axis_p = min(axis_ps) if axis_ps else None

        axis_support = (
            static_auc is not None
            and best_axis_auc is not None
            and best_axis_auc >= 0.60
            and best_axis_auc > static_auc + 0.03
        )
        axis_locked = bool(axis_support and best_axis_p is not None and best_axis_p <= 0.05)

        status_map[result["contrast"]] = {
            "status": "ok",
            "n_rows": result.get("n_rows"),
            "group_counts": result.get("group_counts"),
            "static_auc": static_auc,
            "best_axis_auc": best_axis_auc,
            "best_axis_permutation_p": best_axis_p,
            "axis_support": axis_support,
            "axis_locked": axis_locked,
            "best_model": max(
                [
                    (r.get("auc_mean") if r.get("auc_mean") is not None else -1, r.get("model_name"))
                    for r in result["model_rows"]
                ],
                key=lambda x: x[0],
            )[1] if result["model_rows"] else None,
        }

    bipolar = status_map.get("bipolar_vs_control", {})
    mood = status_map.get("mood_vs_control", {})
    bipolar_unipolar = status_map.get("bipolar_vs_unipolar", {})

    if bipolar.get("axis_locked"):
        return (
            "bipolar_actigraphy_phase_cycling_axis_locked",
            9,
            "Add bipolar/mood actigraphy as the primary cycling/hysteresis lane in the cross-neurotype axis map.",
            status_map,
        )

    if bipolar.get("axis_support"):
        return (
            "bipolar_actigraphy_phase_cycling_supported_not_locked",
            8,
            "Treat bipolar cycling as supportive; inspect feature tests before locking.",
            status_map,
        )

    if mood.get("axis_locked"):
        return (
            "mood_actigraphy_phase_cycling_locked_not_bipolar_specific",
            8,
            "Use mood-actigraphy as a cycling/hysteresis lane, but do not call it bipolar-specific yet.",
            status_map,
        )

    if mood.get("axis_support") or bipolar_unipolar.get("axis_support"):
        return (
            "mood_actigraphy_phase_cycling_directional",
            7,
            "Signal is directional; refine labels and bipolar/unipolar split before axis-map promotion.",
            status_map,
        )

    return (
        "mood_actigraphy_phase_cycling_not_yet_supported",
        6,
        "Parser completed but cycling/reach/hysteresis did not beat static activity averages.",
        status_map,
    )


def plot_model_bars(all_model_rows, path):
    ok = [r for r in all_model_rows if r.get("status") == "ok" and r.get("auc_mean") is not None]
    if not ok:
        return None

    labels = [f"{r['contrast']}\n{r['model_name'].replace('_model','')}" for r in ok]
    vals = [float(r["auc_mean"]) for r in ok]
    x = np.arange(len(labels))

    plt.figure(figsize=(18, 8))
    plt.bar(x, vals)
    plt.axhline(0.5, linewidth=1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Repeated CV AUC")
    plt.title("TAIRID mood actigraphy phase-cycling v1")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def plot_axis_advantage(status_map, path):
    labels = []
    vals = []

    for contrast, s in status_map.items():
        static = s.get("static_auc")
        axis = s.get("best_axis_auc")
        if static is None or axis is None:
            continue
        labels.append(contrast)
        vals.append(float(axis) - float(static))

    if not labels:
        return None

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Best axis AUC - static AUC")
    plt.title("Mood actigraphy: cycling/reach/hysteresis advantage over static")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def main():
    print("")
    print("TAIRID mood actigraphy phase-cycling v1 starting.")
    print("Boundary: operational time-series axis test only; not proof, diagnosis, or medical advice.")
    print("")

    record, downloads = download_zenodo_record()
    extraction = extract_archives(downloads)

    write_csv(OUTDIR / "mood_actigraphy_download_ledger.csv", downloads)
    write_csv(OUTDIR / "mood_actigraphy_extraction_ledger.csv", extraction)

    metadata_map, metadata_sources = build_metadata_map()
    metadata_rows = [{"subject_key": k, "group": v} for k, v in sorted(metadata_map.items())]
    write_csv(OUTDIR / "mood_actigraphy_metadata_sources.csv", metadata_sources)
    write_csv(OUTDIR / "mood_actigraphy_metadata_map.csv", metadata_rows)

    inventory, series_rows = load_activity_series(metadata_map)
    write_csv(OUTDIR / "mood_actigraphy_table_inventory.csv", inventory)
    write_csv(OUTDIR / "mood_actigraphy_series_rows.csv", series_rows)

    features_df = build_subject_features(series_rows)

    if not features_df.empty:
        features_path = OUTDIR / "mood_actigraphy_subject_axis_features.csv"
        features_df.to_csv(features_path, index=False)
    else:
        features_path = None

    all_contrasts = [
        "bipolar_vs_control",
        "mood_vs_control",
        "bipolar_vs_unipolar",
    ]

    contrast_results = []
    all_model_rows = []
    all_perm_rows = []
    all_feature_rows = []

    for contrast in all_contrasts:
        result = run_models_for_contrast(features_df, contrast) if not features_df.empty else {
            "contrast": contrast,
            "status": "no_features",
            "model_rows": [],
            "permutation_rows": [],
        }
        contrast_results.append(result)

        all_model_rows.extend(result.get("model_rows", []))
        all_perm_rows.extend(result.get("permutation_rows", []))

        feats = feature_tests_for_contrast(features_df, contrast) if not features_df.empty else []
        all_feature_rows.extend(feats)

        write_csv(OUTDIR / f"mood_actigraphy_{contrast}_model_results.csv", result.get("model_rows", []))
        write_csv(OUTDIR / f"mood_actigraphy_{contrast}_permutation_results.csv", result.get("permutation_rows", []))
        write_csv(OUTDIR / f"mood_actigraphy_{contrast}_feature_tests_bh_fdr.csv", feats)

    model_path = write_csv(OUTDIR / "mood_actigraphy_all_model_results.csv", all_model_rows)
    perm_path = write_csv(OUTDIR / "mood_actigraphy_all_permutation_results.csv", all_perm_rows)
    feature_path = write_csv(OUTDIR / "mood_actigraphy_all_feature_tests_bh_fdr.csv", all_feature_rows)

    final_status, readiness_score, next_wall, contrast_status = decide_status(contrast_results)

    plots = []

    p = plot_model_bars(all_model_rows, OUTDIR / "mood_actigraphy_model_auc_bars.png")
    if p:
        plots.append(p)

    p = plot_axis_advantage(contrast_status, OUTDIR / "mood_actigraphy_axis_advantage.png")
    if p:
        plots.append(p)

    group_counts = {}
    if not features_df.empty and "group" in features_df.columns:
        group_counts = {str(k): int(v) for k, v in Counter(features_df["group"]).items()}

    summary = {
        "test_name": "TAIRID mood actigraphy phase-cycling v1",
        "boundary": (
            "Operational time-series axis test only. Not proof of TAIRID, not diagnosis, "
            "not medical advice, and not a cosmology result."
        ),
        "dataset": {
            "zenodo_record_id": ZENODO_RECORD_ID,
            "zenodo_api_url": ZENODO_API_URL,
            "record_title": record.get("metadata", {}).get("title"),
            "doi": record.get("doi"),
            "note": "Uses downloaded motor-activity tables only.",
        },
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "parser_counts": {
            "downloads_count": len(downloads),
            "extraction_count": len(extraction),
            "metadata_source_count": len(metadata_sources),
            "metadata_map_count": len(metadata_map),
            "activity_series_count": len(series_rows),
            "subject_axis_rows": int(len(features_df)) if not features_df.empty else 0,
            "group_counts": group_counts,
        },
        "contrast_status": contrast_status,
        "model_results": all_model_rows,
        "permutation_results": all_perm_rows,
        "top_feature_tests": all_feature_rows[:40],
        "output_files": {
            "download_ledger_csv": str(OUTDIR / "mood_actigraphy_download_ledger.csv"),
            "extraction_ledger_csv": str(OUTDIR / "mood_actigraphy_extraction_ledger.csv"),
            "metadata_sources_csv": str(OUTDIR / "mood_actigraphy_metadata_sources.csv"),
            "metadata_map_csv": str(OUTDIR / "mood_actigraphy_metadata_map.csv"),
            "table_inventory_csv": str(OUTDIR / "mood_actigraphy_table_inventory.csv"),
            "series_rows_csv": str(OUTDIR / "mood_actigraphy_series_rows.csv"),
            "subject_axis_features_csv": str(features_path) if features_path else None,
            "model_results_csv": str(model_path),
            "permutation_results_csv": str(perm_path),
            "feature_tests_bh_fdr_csv": str(feature_path),
            "plots": plots,
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "Rhythm, cycling, reach, breach, or hysteresis models outperform static activity averages "
                "and beat permutation expectation."
            ),
            "what_weakens_the_lane": (
                "Static activity averages outperform cycling/reach/hysteresis, or bipolar labels cannot be recovered cleanly."
            ),
            "axis_prediction": (
                "Bipolar/mood actigraphy should stress phase cycling, stable reach, breach/re-entry, and recovery asymmetry "
                "more directly than block-summary ADHD events."
            ),
            "truth_boundary": (
                "A positive result supports mood-actigraphy as a TAIRID cycling/hysteresis lane. It cannot prove TAIRID, "
                "diagnose bipolar disorder, or prove any cosmology claim."
            ),
        },
    }

    summary_path = OUTDIR / "mood_actigraphy_phase_cycling_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(OUTDIR / "mood_actigraphy_phase_cycling_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID mood actigraphy phase-cycling v1\n\n")
        f.write("Boundary: operational time-series axis test only. Not proof. Not diagnosis. Not medical advice.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Why this test exists:\n")
        f.write("- ADHD block-summary events detected static differences but did not lock cycling/reach.\n")
        f.write("- Mood/bipolar actigraphy provides longitudinal motor-activity time series.\n")
        f.write("- This tests cycling, reach, breach, rhythm, and hysteresis directly.\n\n")

        f.write("Contrast status:\n")
        f.write(json.dumps(contrast_status, indent=2) + "\n\n")

        f.write("Model results:\n")
        f.write(json.dumps(all_model_rows, indent=2) + "\n\n")

        f.write("Permutation results:\n")
        f.write(json.dumps(all_perm_rows, indent=2) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- This can support mood-actigraphy as a cycling/hysteresis lane.\n")
        f.write("- It cannot prove TAIRID.\n")
        f.write("- It cannot diagnose bipolar disorder or depression.\n")
        f.write("- It cannot prove any cosmology claim.\n")

    print("")
    print("TAIRID mood actigraphy phase-cycling v1 complete.")
    print("Created:")
    print("  tairid_mood_actigraphy_phase_cycling_v1_outputs/mood_actigraphy_phase_cycling_v1_summary.json")
    print("  tairid_mood_actigraphy_phase_cycling_v1_outputs/mood_actigraphy_phase_cycling_v1_summary.txt")
    print("  tairid_mood_actigraphy_phase_cycling_v1_outputs/mood_actigraphy_all_model_results.csv")
    print("  tairid_mood_actigraphy_phase_cycling_v1_outputs/mood_actigraphy_all_permutation_results.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is not diagnosis or medical advice.")
    print("  This is a mood-actigraphy operational axis test.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
