#!/usr/bin/env python3
"""
TAIRID mood actigraphy label + duration control v2.

Purpose:
v1 showed a strong mood/clinical actigraphy cycling signal, but it was not
bipolar-specific and some features were duration/count-sensitive. v2 uses the
OBF-Psychiatric info files directly, builds diagnosis-flag contrasts, removes
raw duration/count artifacts from model feature sets, and adds duration-adjusted
and duration-overlap sensitivity analyses.

Boundary:
This is not proof of TAIRID.
This is not diagnosis.
This is not medical advice.
This is not a cosmology result.
It is an operational time-series axis test.
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


OUTDIR = Path("tairid_mood_actigraphy_label_duration_control_v2_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR = OUTDIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
SERIES_DIR = OUTDIR / "normalized_series"
SERIES_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD_ID = "13754984"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

RANDOM_SEED = 42
CV_REPEATS = 100
PERMUTATIONS = 250
PERM_REPEATS = 25
RIDGE = 1.0e-3
TOLERANCE_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0]

DIAGNOSIS_FLAG_COLS = [
    "adhd", "add", "bipolar", "unipolar", "anxiety", "substance", "other", "ct",
]
MED_FLAG_COLS = [
    "med", "med_antidepr", "med_moodstab", "med_antipsych", "med_anxiety_benzo",
    "med_sleep", "med_stimulants",
]
SCORE_COLS = ["mdq_pos", "wurs", "asrs", "madrs", "hads_a", "hads_d"]


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


def read_url(url, timeout=900):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-mood-actigraphy-label-duration-control-v2",
            "Accept": "application/json, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
    return data, final_url, content_type


def download_zenodo_record():
    data, final_url, content_type = read_url(ZENODO_API_URL)
    record = json.loads(data.decode("utf-8", errors="replace"))

    (OUTDIR / "zenodo_record_13754984.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
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


def list_table_files():
    out = []

    for root in [DOWNLOAD_DIR, EXTRACT_DIR]:
        for path in root.rglob("*"):
            if (
                path.is_file()
                and not is_junk_path(path)
                and path.name.lower().endswith((".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls"))
            ):
                out.append(path)

    return sorted(set(out))


def read_table(path, max_rows=None):
    lower = path.name.lower()

    try:
        if lower.endswith((".xlsx", ".xls")):
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


def flag_value(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 0

    s = str(x).strip().lower()

    if s in ["", "na", "nan", "none", "n/a"]:
        return 0

    try:
        return 1 if float(s) != 0 else 0
    except Exception:
        return 1 if s in ["yes", "true", "y", "present", "positive", "pos"] else 0


def num_value(x):
    if x is None:
        return np.nan

    try:
        s = str(x).strip().upper()

        if s in ["", "NA", "N/A", "NAN", "NONE"]:
            return np.nan

        return float(s)
    except Exception:
        return np.nan


def age_midpoint(value):
    s = str(value).strip()
    nums = [float(x) for x in re.findall(r"\d+", s)]

    if len(nums) >= 2:
        return float(np.mean(nums[:2]))

    if len(nums) == 1:
        return nums[0]

    return np.nan


def primary_label_from_flags(row):
    if flag_value(row.get("bipolar")) == 1:
        return "bipolar_flag"

    if flag_value(row.get("unipolar")) == 1:
        return "unipolar_flag"

    if flag_value(row.get("adhd")) == 1 or flag_value(row.get("add")) == 1:
        return "adhd_or_add_flag"

    if flag_value(row.get("anxiety")) == 1:
        return "anxiety_flag"

    if flag_value(row.get("substance")) == 1:
        return "substance_flag"

    if flag_value(row.get("other")) == 1:
        return "other_clinical_flag"

    return "clinical_no_specific_flag"


def parse_info_files():
    metadata = {}
    sources = []
    rows = []

    for path in list_table_files():
        lower = path.name.lower()

        if not lower.endswith("-info.csv"):
            continue

        df, err = read_table(path)

        if df is None:
            sources.append({"path": str(path), "status": "read_failed", "error": err})
            continue

        if "number" not in df.columns:
            sources.append(
                {
                    "path": str(path),
                    "status": "missing_number",
                    "columns": " | ".join(df.columns),
                }
            )
            continue

        folder_label = "unknown"

        if "control-info" in lower:
            folder_label = "control"
        elif "adhd-info" in lower:
            folder_label = "adhd_folder"
        elif "clinical-info" in lower:
            folder_label = "clinical_folder"

        parsed = 0

        for _, r in df.iterrows():
            number = str(r.get("number", "")).strip()

            if not number:
                continue

            meta = {
                "number": number,
                "subject_key": norm(number),
                "source_info_file": str(path),
                "folder_label": folder_label,
                "age_raw": str(r.get("age", "")),
                "age_midpoint": age_midpoint(r.get("age")),
                "gender": num_value(r.get("gender")),
                "days_info": num_value(r.get("days")),
                "acc_time": str(r.get("acc_time", "")),
            }

            for col in DIAGNOSIS_FLAG_COLS + MED_FLAG_COLS:
                meta[col] = flag_value(r.get(col)) if col in df.columns else 0

            for col in SCORE_COLS:
                meta[col] = num_value(r.get(col)) if col in df.columns else np.nan

            if folder_label == "control":
                meta["is_control"] = 1
                meta["primary_label"] = "control"
            else:
                meta["is_control"] = 0
                meta["primary_label"] = primary_label_from_flags(meta)

            meta["mood_any"] = 1 if (meta.get("bipolar", 0) == 1 or meta.get("unipolar", 0) == 1) else 0
            meta["clinical_any"] = 0 if meta["is_control"] == 1 else 1
            meta["bipolar_only"] = 1 if meta.get("bipolar", 0) == 1 and meta.get("unipolar", 0) == 0 else 0
            meta["unipolar_only"] = 1 if meta.get("unipolar", 0) == 1 and meta.get("bipolar", 0) == 0 else 0
            meta["bipolar_and_unipolar"] = 1 if meta.get("bipolar", 0) == 1 and meta.get("unipolar", 0) == 1 else 0

            metadata[norm(number)] = meta
            rows.append(meta)
            parsed += 1

        sources.append(
            {
                "path": str(path),
                "status": "parsed_info",
                "folder_label": folder_label,
                "rows": int(len(df)),
                "parsed": parsed,
                "columns": " | ".join(df.columns),
            }
        )

    return metadata, sources, rows


def infer_number_from_activity_path(path):
    stem = Path(path).stem
    return stem


def choose_activity_column(df):
    candidates = []

    for c in df.columns:
        cn = norm(c)

        if any(x in cn for x in ["timestamp", "datetime", "date", "time", "number", "id", "index", "unnamed"]):
            continue

        vals = pd.to_numeric(df[c], errors="coerce")
        finite = int(np.isfinite(vals).sum())

        if finite < max(20, int(0.50 * len(df))):
            continue

        var = float(np.nanvar(vals))

        if not np.isfinite(var) or var <= 0:
            continue

        score = 0.0

        if "activity" in cn or "count" in cn or "motor" in cn or "accel" in cn or "steps" in cn:
            score += 10.0

        score += min(math.log1p(var), 5.0)
        score += min(finite / 1000.0, 5.0)

        candidates.append((score, c, finite, var))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def parse_time_index(df):
    cols = list(df.columns)

    for c in cols:
        cn = norm(c)

        if any(h in cn for h in ["timestamp", "datetime", "date", "time"]):
            vals = pd.to_datetime(df[c], errors="coerce")

            if vals.notnull().sum() >= max(20, int(0.50 * len(df))):
                return vals, True, c

    vals = pd.date_range("2000-01-01", periods=len(df), freq="min")
    return pd.Series(vals), False, "synthetic_minute_index"


def load_activity_series(metadata):
    inventory = []
    series_rows = []

    for path in list_table_files():
        lower = path.name.lower()

        if lower.endswith("-info.csv") or "readme" in lower:
            continue

        df, err = read_table(path)

        if df is None:
            inventory.append({"path": str(path), "status": "read_failed", "error": err})
            continue

        if len(df) < 100:
            inventory.append({"path": str(path), "status": "too_few_rows", "rows": int(len(df))})
            continue

        act_col = choose_activity_column(df)

        if act_col is None:
            inventory.append(
                {
                    "path": str(path),
                    "status": "no_activity_col",
                    "rows": int(len(df)),
                    "columns": " | ".join(df.columns),
                }
            )
            continue

        number = infer_number_from_activity_path(path)
        meta = metadata.get(norm(number))

        if meta is None:
            inventory.append(
                {
                    "path": str(path),
                    "status": "no_matching_info_metadata",
                    "number": number,
                    "rows": int(len(df)),
                    "activity_col": act_col,
                }
            )
            continue

        times, has_real_time, time_source = parse_time_index(df)
        activity = pd.to_numeric(df[act_col], errors="coerce")
        temp = pd.DataFrame({"time": times, "activity": activity}).dropna()
        temp = temp[np.isfinite(temp["activity"])]
        temp = temp.sort_values("time")

        if len(temp) < 100:
            inventory.append(
                {
                    "path": str(path),
                    "status": "activity_series_too_short_after_clean",
                    "number": number,
                    "rows": int(len(temp)),
                }
            )
            continue

        local_csv = SERIES_DIR / f"{safe_name(number)}.csv"
        temp.to_csv(local_csv, index=False)

        row = {
            "number": number,
            "subject_key": norm(number),
            "source_file": str(path),
            "series_csv": str(local_csv),
            "activity_col": act_col,
            "time_source": time_source,
            "has_real_time": bool(has_real_time),
            "raw_rows": int(len(df)),
            "clean_rows": int(len(temp)),
            **meta,
        }

        series_rows.append(row)

        inventory.append(
            {
                "path": str(path),
                "status": "activity_series_parsed",
                "number": number,
                "primary_label": meta.get("primary_label"),
                "folder_label": meta.get("folder_label"),
                "rows": int(len(temp)),
                "activity_col": act_col,
                "time_source": time_source,
                "has_real_time": bool(has_real_time),
                "series_csv": str(local_csv),
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
    cur = 0

    for f in flags:
        if bool(f):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

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
    s = hourly_series.copy().dropna()

    if len(s) < 48 or np.var(s.values) <= 1.0e-12:
        return np.nan

    by_hour = s.groupby(s.index.hour).mean()

    if len(by_hour) < 12:
        return np.nan

    return float(np.var(by_hour.values) / np.var(s.values))


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


def slope(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) < 2:
        return np.nan

    try:
        return float(np.polyfit(np.arange(len(vals), dtype=float), vals, 1)[0])
    except Exception:
        return np.nan


def build_subject_features(series_rows):
    cache = []

    for row in series_rows:
        df = pd.read_csv(row["series_csv"])
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna().sort_values("time")

        if len(df) < 100:
            continue

        raw_activity = pd.to_numeric(df["activity"], errors="coerce").astype(float).values
        raw_activity = np.maximum(raw_activity, 0.0)
        ts = pd.Series(np.log1p(raw_activity), index=df["time"])
        ts = ts[~ts.index.duplicated(keep="first")]
        hourly = ts.resample("60min").mean().interpolate(limit=3).dropna()

        if len(hourly) < 24:
            chunks = []

            for i in range(0, len(raw_activity), 60):
                chunk = raw_activity[i:i + 60]

                if len(chunk) >= 10:
                    chunks.append(float(np.mean(np.log1p(chunk))))

            hourly = pd.Series(
                chunks,
                index=pd.date_range("2000-01-01", periods=len(chunks), freq="60min"),
            )

        if len(hourly) < 24:
            continue

        vals = hourly.values.astype(float)

        T = zscore(vals)
        local_diff = np.abs(np.diff(vals, prepend=vals[0]))
        rolling_vol = pd.Series(vals).rolling(6, min_periods=2).std().fillna(0.0).values
        I = zscore(local_diff + rolling_vol)
        M = np.abs(T - I)
        C = np.sqrt(T ** 2 + I ** 2)
        dT = np.diff(T)
        dI = np.diff(I)
        dM = np.diff(M)
        rising = dM[dM > 0]
        falling = -dM[dM < 0]

        base = {
            **{
                k: row.get(k)
                for k in [
                    "number", "subject_key", "folder_label", "primary_label", "source_file",
                    "age_raw", "age_midpoint", "gender", "days_info",
                ]
            },
            **{
                k: row.get(k, 0)
                for k in DIAGNOSIS_FLAG_COLS
                + MED_FLAG_COLS
                + SCORE_COLS
                + [
                    "is_control", "clinical_any", "mood_any", "bipolar_only",
                    "unipolar_only", "bipolar_and_unipolar",
                ]
            },
            "hourly_count": int(len(vals)),
            "duration_hours": float(len(vals)),
            "duration_days": float(len(vals) / 24.0),
            "activity_mean": float(np.mean(vals)),
            "activity_median": float(np.median(vals)),
            "activity_std": float(np.std(vals)),
            "activity_cv": float(np.std(vals) / max(abs(np.mean(vals)), 1e-9)),
            "activity_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "activity_range": float(np.max(vals) - np.min(vals)),
            "T_mean": float(np.mean(T)),
            "I_mean": float(np.mean(I)),
            "M_mean": float(np.mean(M)),
            "M_median": float(np.median(M)),
            "M_max": float(np.max(M)),
            "M_std": float(np.std(M)),
            "M_iqr": float(np.percentile(M, 75) - np.percentile(M, 25)),
            "M_range": float(np.max(M) - np.min(M)),
            "M_slope": slope(M),
            "collapse_load_mean": float(np.mean(C)),
            "collapse_load_max": float(np.max(C)),
            "collapse_load_range": float(np.max(C) - np.min(C)),
            "collapse_load_slope": slope(C),
            "Cyc_activity_turn_rate": float(sign_change_count(vals) / max(len(vals) - 2, 1)),
            "Cyc_M_turn_rate": float(sign_change_count(M) / max(len(M) - 2, 1)),
            "Cyc_T_turn_rate": float(sign_change_count(T) / max(len(T) - 2, 1)),
            "Cyc_I_turn_rate": float(sign_change_count(I) / max(len(I) - 2, 1)),
            "Cyc_TI_opposition_fraction": float(np.mean(np.sign(dT) * np.sign(dI) < 0)) if len(dT) and len(dI) else np.nan,
            "Rhythm_autocorr_1h": autocorr(vals, 1),
            "Rhythm_autocorr_24h": autocorr(vals, 24),
            "Rhythm_intradaily_variability": intradaily_variability(vals),
            "Rhythm_interdaily_stability": interdaily_stability(hourly),
            **spectral_features(vals, sample_period_hours=1.0),
            "H_mismatch_rise_mean": float(np.mean(rising)) if len(rising) else 0.0,
            "H_mismatch_fall_mean": float(np.mean(falling)) if len(falling) else 0.0,
        }

        base["H_rise_fall_asymmetry"] = float(
            base["H_mismatch_rise_mean"] - base["H_mismatch_fall_mean"]
        )

        cache.append({"base": base, "M": M})

    control_q25 = [
        float(np.percentile(x["M"], 25))
        for x in cache
        if x["base"].get("is_control") == 1 and len(x["M"]) > 10
    ]

    if len(control_q25) >= 2:
        control_sd = float(np.std(control_q25, ddof=1))
    else:
        all_q25 = [float(np.percentile(x["M"], 25)) for x in cache if len(x["M"]) > 10]
        control_sd = float(np.std(all_q25, ddof=1)) if len(all_q25) >= 2 else 0.0

    if not np.isfinite(control_sd):
        control_sd = 0.0

    rows = []

    for x in cache:
        M = x["M"]
        base = x["base"]
        baseline = float(np.percentile(M, 25))
        base["W_baseline_M_q25"] = baseline
        base["W_control_M_q25_sd_used"] = control_sd

        for mult in TOLERANCE_MULTIPLIERS:
            suffix = f"tol{str(mult).replace('.', 'p')}"
            W = baseline + mult * control_sd
            B = np.maximum(0.0, M - W)
            breached = B > 0.0
            stable = ~breached
            transitions = np.abs(np.diff(breached.astype(int))) if len(breached) > 1 else np.asarray([])

            episodes = []
            in_breach, length = False, 0

            for flag in breached:
                if flag and not in_breach:
                    in_breach, length = True, 1
                elif flag and in_breach:
                    length += 1
                elif (not flag) and in_breach:
                    episodes.append(length)
                    in_breach, length = False, 0

            base[f"W_{suffix}"] = float(W)
            base[f"B_mean_{suffix}"] = float(np.mean(B))
            base[f"B_max_{suffix}"] = float(np.max(B))
            base[f"B_fraction_{suffix}"] = float(np.mean(breached))
            base[f"B_rate_per_hour_{suffix}"] = float(np.sum(B) / max(len(B), 1))
            base[f"Reach_stable_fraction_{suffix}"] = float(np.mean(stable))
            base[f"Reach_breach_fraction_{suffix}"] = float(np.mean(breached))
            base[f"Reach_longest_stable_run_fraction_{suffix}"] = float(longest_true_run(stable) / max(len(stable), 1))
            base[f"Reach_longest_breach_run_fraction_{suffix}"] = float(longest_true_run(breached) / max(len(breached), 1))
            base[f"Cyc_breach_transition_rate_{suffix}"] = float(np.mean(transitions)) if len(transitions) else 0.0
            base[f"Cyc_breach_reentry_rate_{suffix}"] = (
                float(np.sum((breached[:-1] == True) & (breached[1:] == False)) / max(len(breached) - 1, 1))
                if len(breached) > 1
                else 0.0
            )
            base[f"H_breach_episode_mean_fraction_{suffix}"] = (
                float(np.mean(episodes) / max(len(breached), 1)) if episodes else 0.0
            )
            base[f"H_breach_episode_max_fraction_{suffix}"] = (
                float(np.max(episodes) / max(len(breached), 1)) if episodes else 0.0
            )

        rows.append(base)

    return pd.DataFrame(rows)


def make_contrast_df(features, contrast):
    df = features.copy()

    if contrast == "bipolar_any_vs_control":
        df = df[(df["bipolar"] == 1) | (df["is_control"] == 1)].copy()
        df["label"] = (df["bipolar"] == 1).astype(int)

    elif contrast == "unipolar_any_vs_control":
        df = df[(df["unipolar"] == 1) | (df["is_control"] == 1)].copy()
        df["label"] = (df["unipolar"] == 1).astype(int)

    elif contrast == "mood_any_vs_control":
        df = df[(df["mood_any"] == 1) | (df["is_control"] == 1)].copy()
        df["label"] = (df["mood_any"] == 1).astype(int)

    elif contrast == "bipolar_only_vs_unipolar_only":
        df = df[((df["bipolar_only"] == 1) | (df["unipolar_only"] == 1))].copy()
        df["label"] = (df["bipolar_only"] == 1).astype(int)

    elif contrast == "bipolar_any_vs_unipolar_any_nonoverlap":
        df = df[((df["bipolar"] == 1) ^ (df["unipolar"] == 1))].copy()
        df["label"] = (df["bipolar"] == 1).astype(int)

    elif contrast == "adhd_any_vs_control":
        df = df[((df["adhd"] == 1) | (df["add"] == 1)) | (df["is_control"] == 1)].copy()
        df["label"] = ((df["adhd"] == 1) | (df["add"] == 1)).astype(int)

    elif contrast == "clinical_any_vs_control":
        df = df[(df["clinical_any"] == 1) | (df["is_control"] == 1)].copy()
        df["label"] = (df["clinical_any"] == 1).astype(int)

    else:
        raise ValueError(f"unknown contrast {contrast}")

    return df


def duration_overlap_subset(df):
    if df.empty or df["label"].nunique() < 2:
        return df.copy(), {"status": "not_enough_groups"}

    pos = df[df["label"] == 1]["duration_days"].dropna().values
    neg = df[df["label"] == 0]["duration_days"].dropna().values

    if len(pos) < 3 or len(neg) < 3:
        return df.copy(), {"status": "not_enough_duration_values"}

    lo = max(float(np.percentile(pos, 10)), float(np.percentile(neg, 10)))
    hi = min(float(np.percentile(pos, 90)), float(np.percentile(neg, 90)))

    if hi <= lo:
        lo = max(float(np.min(pos)), float(np.min(neg)))
        hi = min(float(np.max(pos)), float(np.max(neg)))

    sub = df[(df["duration_days"] >= lo) & (df["duration_days"] <= hi)].copy()

    status = (
        "ok"
        if sub["label"].nunique() == 2 and min(Counter(sub["label"]).values()) >= 6
        else "too_few_after_overlap"
    )

    return sub, {
        "status": status,
        "duration_lo": lo,
        "duration_hi": hi,
        "n_before": int(len(df)),
        "n_after": int(len(sub)),
        "label_counts_after": {str(k): int(v) for k, v in Counter(sub.get("label", [])).items()},
    }


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
        "Cyc_activity_turn_rate", "Cyc_M_turn_rate", "Cyc_T_turn_rate",
        "Cyc_I_turn_rate", "Cyc_TI_opposition_fraction",
    ]

    viability, reach, hysteresis = [], [], [
        "H_mismatch_rise_mean", "H_mismatch_fall_mean", "H_rise_fall_asymmetry",
    ]

    for mult in TOLERANCE_MULTIPLIERS:
        suffix = f"tol{str(mult).replace('.', 'p')}"
        viability += [
            f"B_mean_{suffix}", f"B_max_{suffix}", f"B_fraction_{suffix}",
            f"B_rate_per_hour_{suffix}",
        ]
        reach += [
            f"Reach_stable_fraction_{suffix}",
            f"Reach_breach_fraction_{suffix}",
            f"Reach_longest_stable_run_fraction_{suffix}",
            f"Reach_longest_breach_run_fraction_{suffix}",
        ]
        cycling += [
            f"Cyc_breach_transition_rate_{suffix}",
            f"Cyc_breach_reentry_rate_{suffix}",
        ]
        hysteresis += [
            f"H_breach_episode_mean_fraction_{suffix}",
            f"H_breach_episode_max_fraction_{suffix}",
        ]

    return {
        "static_activity_model": [c for c in static if c in df.columns],
        "rhythm_rate_model": [c for c in rhythm if c in df.columns],
        "cycling_rate_model": [c for c in cycling if c in df.columns],
        "viability_rate_model": [c for c in viability if c in df.columns],
        "reach_fraction_model": [c for c in reach if c in df.columns],
        "hysteresis_fraction_model": [c for c in hysteresis if c in df.columns],
        "cycle_reach_hysteresis_rate_model": [c for c in cycling + reach + hysteresis if c in df.columns],
        "combined_rate_axis_model": [c for c in static + rhythm + cycling + viability + reach + hysteresis if c in df.columns],
    }


def auc_score(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[mask], labels[mask]

    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return np.nan

    pos, neg = scores[labels == 1], scores[labels == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    ranks = stats.rankdata(np.concatenate([pos, neg]))
    rpos = np.sum(ranks[:len(pos)])
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stratified_folds(y, k, rng):
    y = np.asarray(y, dtype=int)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    p0, p1 = np.array_split(idx0, k), np.array_split(idx1, k)

    folds = []

    for i in range(k):
        test = np.concatenate([p0[i], p1[i]])
        rng.shuffle(test)
        folds.append(test)

    return folds


def residualize_train_test(X_train, X_test, Z_train, Z_test):
    if Z_train is None or Z_train.shape[1] == 0:
        return X_train, X_test

    Ztr = np.column_stack([np.ones(len(Z_train)), Z_train])
    Zte = np.column_stack([np.ones(len(Z_test)), Z_test])
    beta = np.linalg.pinv(Ztr, rcond=1e-8) @ X_train

    return X_train - Ztr @ beta, X_test - Zte @ beta


def lda_scores(X_train, y_train, X_test):
    y_train = np.asarray(y_train, dtype=int)
    X0, X1 = X_train[y_train == 0], X_train[y_train == 1]

    if len(X0) < 2 or len(X1) < 2:
        return np.zeros(len(X_test), dtype=float)

    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)

    if X_train.shape[1] == 1:
        var = float(np.var(X_train[:, 0]) + RIDGE)
        w = np.asarray([(mu1[0] - mu0[0]) / var])
    else:
        cov = np.cov(X_train.T, bias=False)
        cov = np.atleast_2d(cov) + np.eye(X_train.shape[1]) * RIDGE
        w = np.linalg.pinv(cov, rcond=1e-8) @ (mu1 - mu0)

    b = -0.5 * float((mu1 + mu0) @ w)
    return X_test @ w + b


def repeated_cv(df, feature_cols, repeats=CV_REPEATS, y_override=None, adjust_duration=False):
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return {"status": "no_features", "n": 0, "feature_cols": []}

    keep_cols = feature_cols + ["label"]
    confound_cols = [c for c in ["duration_days", "age_midpoint"] if c in df.columns]

    if adjust_duration:
        keep_cols += confound_cols

    data = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 12 or data["label"].nunique() < 2:
        return {"status": "not_enough_data", "n": int(len(data)), "feature_cols": feature_cols}

    X_raw = data[feature_cols].astype(float).values
    y = data["label"].astype(int).values

    if y_override is not None:
        y = np.asarray(y_override, dtype=int)

    if len(np.unique(y)) < 2:
        return {"status": "one_class", "n": int(len(y)), "feature_cols": feature_cols}

    Z_raw = data[confound_cols].astype(float).values if adjust_duration and confound_cols else None
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

            X_train, X_test = X_raw[train_mask], X_raw[test]
            y_train = y[train_mask]

            if Z_raw is not None:
                Z_train, Z_test = Z_raw[train_mask], Z_raw[test]
                zmu = Z_train.mean(axis=0)
                zsd = Z_train.std(axis=0)
                zsd[zsd <= 1e-12] = 1.0
                Z_train = (Z_train - zmu) / zsd
                Z_test = (Z_test - zmu) / zsd
                X_train, X_test = residualize_train_test(X_train, X_test, Z_train, Z_test)

            mu = X_train.mean(axis=0)
            sd = X_train.std(axis=0)
            sd[sd <= 1e-12] = 1.0
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
        "adjust_duration_age": bool(adjust_duration),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_min": float(np.min(aucs)),
        "auc_max": float(np.max(aucs)),
    }


def permutation_test(df, feature_cols, observed_auc, adjust_duration=False):
    feature_cols = [c for c in feature_cols if c in df.columns]
    keep_cols = feature_cols + ["label"] + (
        [c for c in ["duration_days", "age_midpoint"] if c in df.columns]
        if adjust_duration
        else []
    )
    data = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 12 or data["label"].nunique() < 2 or observed_auc is None:
        return {"status": "not_enough_data", "n_perm": 0, "p_value_ge_observed": None}

    y = data["label"].astype(int).values
    rng = np.random.default_rng(RANDOM_SEED + 99)
    perm_aucs = []

    for _ in range(PERMUTATIONS):
        yp = y.copy()
        rng.shuffle(yp)
        res = repeated_cv(data, feature_cols, repeats=PERM_REPEATS, y_override=yp, adjust_duration=adjust_duration)
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


def run_contrast_models(features_df, contrast):
    base = make_contrast_df(features_df, contrast)
    overlap, overlap_meta = duration_overlap_subset(base)

    variants = [
        ("full_rate_features", base, False),
        ("duration_adjusted", base, True),
        ("duration_overlap_rate_features", overlap, False),
        ("duration_overlap_adjusted", overlap, True),
    ]

    model_rows, perm_rows = [], []
    variant_meta = {
        "base_n": int(len(base)),
        "base_label_counts": {str(k): int(v) for k, v in Counter(base.get("label", [])).items()},
        "overlap": overlap_meta,
    }

    for variant, df, adjust in variants:
        if df.empty or df.get("label", pd.Series(dtype=int)).nunique() < 2 or min(Counter(df.get("label", [])).values() or [0]) < 6:
            model_rows.append(
                {
                    "contrast": contrast,
                    "variant": variant,
                    "model_name": "all",
                    "status": "not_enough_data",
                    "n": int(len(df)),
                }
            )
            continue

        for name, cols in model_feature_sets(df).items():
            res = repeated_cv(df, cols, adjust_duration=adjust)
            auc = res.get("auc_mean")

            model_rows.append(
                {
                    "contrast": contrast,
                    "variant": variant,
                    "model_name": name,
                    **{k: v for k, v in res.items() if k != "feature_cols"},
                    "feature_cols": " | ".join(res.get("feature_cols", [])),
                }
            )

            perm = permutation_test(df, cols, observed_auc=auc, adjust_duration=adjust)

            perm_rows.append(
                {
                    "contrast": contrast,
                    "variant": variant,
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

    return model_rows, perm_rows, variant_meta


def feature_tests(features_df, contrast):
    df = make_contrast_df(features_df, contrast)

    if df.empty or df["label"].nunique() < 2:
        return []

    raw_exclude = {"subject_key", "number", "source_file", "label"}
    cols = [
        c for c in df.columns
        if c not in raw_exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notnull().sum() >= 8
    ]

    rows = []

    for c in cols:
        if c in ["duration_hours", "duration_days", "hourly_count"]:
            continue

        pos = df[df["label"] == 1][c].dropna().astype(float).values
        neg = df[df["label"] == 0][c].dropna().astype(float).values

        if len(pos) < 3 or len(neg) < 3:
            continue

        t, p = stats.ttest_ind(pos, neg, equal_var=False, nan_policy="omit")
        pooled = math.sqrt(
            ((len(pos) - 1) * np.var(pos, ddof=1) + (len(neg) - 1) * np.var(neg, ddof=1))
            / max(len(pos) + len(neg) - 2, 1)
        )
        d = 0.0 if pooled <= 1e-12 else float((np.mean(pos) - np.mean(neg)) / pooled)

        scores = df[c].astype(float).values
        labels = df["label"].astype(int).values

        if np.nanmean(pos) < np.nanmean(neg):
            scores = -scores

        rows.append(
            {
                "contrast": contrast,
                "feature": c,
                "positive_mean": float(np.mean(pos)),
                "negative_mean": float(np.mean(neg)),
                "cohen_d_positive_minus_negative": d,
                "welch_t": float(t),
                "welch_p": float(p),
                "oriented_auc": auc_score(scores, labels),
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
            }
        )

    rows = sorted(
        rows,
        key=lambda r: (
            -(abs(r["cohen_d_positive_minus_negative"]) if np.isfinite(r["cohen_d_positive_minus_negative"]) else 0),
            r["welch_p"] if np.isfinite(r.get("welch_p", np.nan)) else 999,
        ),
    )

    return add_bh_q(rows)


def add_bh_q(rows):
    rows = list(rows)
    pvals, idxs = [], []

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


def decide_status(all_model_rows, all_perm_rows, contrast_variant_meta):
    model = {}
    perm = {}

    for r in all_model_rows:
        if r.get("status") == "ok":
            model.setdefault((r["contrast"], r["variant"]), {})[r["model_name"]] = r

    for r in all_perm_rows:
        if r.get("permutation_status") == "ok":
            perm.setdefault((r["contrast"], r["variant"]), {})[r["model_name"]] = r

    contrast_status = {}
    axis_names = [
        "rhythm_rate_model",
        "cycling_rate_model",
        "viability_rate_model",
        "reach_fraction_model",
        "hysteresis_fraction_model",
        "cycle_reach_hysteresis_rate_model",
        "combined_rate_axis_model",
    ]

    for key, models in model.items():
        contrast, variant = key
        static_auc = models.get("static_activity_model", {}).get("auc_mean")
        axis_values = [
            (models.get(n, {}).get("auc_mean"), n)
            for n in axis_names
            if models.get(n, {}).get("auc_mean") is not None
        ]
        best_axis_auc, best_axis_name = max(axis_values, key=lambda x: x[0]) if axis_values else (None, None)
        ps = [
            perm.get(key, {}).get(n, {}).get("p_value_ge_observed")
            for n in axis_names
            if perm.get(key, {}).get(n, {}).get("p_value_ge_observed") is not None
        ]
        best_p = min(ps) if ps else None
        support = (
            static_auc is not None
            and best_axis_auc is not None
            and best_axis_auc >= 0.60
            and best_axis_auc > static_auc + 0.03
        )
        locked = bool(support and best_p is not None and best_p <= 0.05)

        contrast_status.setdefault(contrast, {})[variant] = {
            "static_auc": static_auc,
            "best_axis_auc": best_axis_auc,
            "best_axis_model": best_axis_name,
            "best_axis_permutation_p": best_p,
            "axis_support": support,
            "axis_locked": locked,
        }

    for contrast, meta in contrast_variant_meta.items():
        contrast_status.setdefault(contrast, {})["variant_meta"] = meta

    def locked(contrast, variant):
        return bool(contrast_status.get(contrast, {}).get(variant, {}).get("axis_locked"))

    def support(contrast, variant):
        return bool(contrast_status.get(contrast, {}).get(variant, {}).get("axis_support"))

    bipolar_locked = locked("bipolar_any_vs_control", "duration_adjusted") or locked("bipolar_any_vs_control", "duration_overlap_adjusted")
    bipolar_supported = support("bipolar_any_vs_control", "duration_adjusted") or support("bipolar_any_vs_control", "duration_overlap_adjusted")
    mood_locked = locked("mood_any_vs_control", "duration_adjusted") or locked("mood_any_vs_control", "duration_overlap_adjusted")
    clinical_locked = locked("clinical_any_vs_control", "duration_adjusted") or locked("clinical_any_vs_control", "duration_overlap_adjusted")

    if bipolar_locked:
        return (
            "bipolar_actigraphy_label_duration_control_axis_locked",
            9,
            "Promote bipolar/mood actigraphy as the primary cycling/hysteresis lane with duration-control caveat.",
            contrast_status,
        )

    if bipolar_supported:
        return (
            "bipolar_actigraphy_supported_after_duration_control_not_locked",
            8,
            "Treat bipolar-specific cycling as supportive; inspect medication/comorbidity before locking.",
            contrast_status,
        )

    if mood_locked:
        return (
            "mood_actigraphy_duration_control_axis_locked_not_bipolar_specific",
            8,
            "Use mood actigraphy as cycling/hysteresis lane; keep bipolar-specific status unresolved.",
            contrast_status,
        )

    if clinical_locked:
        return (
            "clinical_actigraphy_duration_control_axis_locked_broad_not_mood_specific",
            7,
            "Use only as broad clinical rhythm-cycling support; refine diagnosis labels.",
            contrast_status,
        )

    return (
        "mood_actigraphy_label_duration_control_not_yet_supported",
        6,
        "Duration/label controls weakened the signal; inspect feature tables and diagnosis flags.",
        contrast_status,
    )


def plot_model_bars(rows, path):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("auc_mean") is not None]

    if not ok:
        return None

    labels = [
        f"{r['contrast']}\n{r['variant']}\n{r['model_name'].replace('_model', '')}"
        for r in ok
    ]
    vals = [float(r["auc_mean"]) for r in ok]
    x = np.arange(len(labels))

    plt.figure(figsize=(24, 9))
    plt.bar(x, vals)
    plt.axhline(0.5, linewidth=1)
    plt.xticks(x, labels, rotation=65, ha="right", fontsize=7)
    plt.ylabel("Repeated CV AUC")
    plt.title("TAIRID mood actigraphy label + duration control v2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def plot_axis_advantage(status, path):
    labels, vals = [], []

    for contrast, variants in status.items():
        for variant, s in variants.items():
            if variant == "variant_meta":
                continue

            if s.get("static_auc") is None or s.get("best_axis_auc") is None:
                continue

            labels.append(f"{contrast}\n{variant}")
            vals.append(float(s["best_axis_auc"]) - float(s["static_auc"]))

    if not labels:
        return None

    x = np.arange(len(labels))

    plt.figure(figsize=(14, 7))
    plt.bar(x, vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Best axis AUC - static AUC")
    plt.title("Duration-controlled axis advantage over static")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    return str(path)


def main():
    print("\nTAIRID mood actigraphy label + duration control v2 starting.")
    print("Boundary: operational axis test only; not proof, diagnosis, or medical advice.\n")

    record, downloads = download_zenodo_record()
    extraction = extract_archives(downloads)

    write_csv(OUTDIR / "mood_v2_download_ledger.csv", downloads)
    write_csv(OUTDIR / "mood_v2_extraction_ledger.csv", extraction)

    metadata, meta_sources, meta_rows = parse_info_files()

    write_csv(OUTDIR / "mood_v2_info_sources.csv", meta_sources)
    write_csv(OUTDIR / "mood_v2_subject_metadata_flags.csv", meta_rows)

    inventory, series_rows = load_activity_series(metadata)

    write_csv(OUTDIR / "mood_v2_table_inventory.csv", inventory)
    write_csv(OUTDIR / "mood_v2_series_rows.csv", series_rows)

    features = build_subject_features(series_rows)

    if not features.empty:
        features.to_csv(OUTDIR / "mood_v2_subject_axis_features.csv", index=False)

    contrasts = [
        "bipolar_any_vs_control",
        "unipolar_any_vs_control",
        "mood_any_vs_control",
        "bipolar_only_vs_unipolar_only",
        "bipolar_any_vs_unipolar_any_nonoverlap",
        "adhd_any_vs_control",
        "clinical_any_vs_control",
    ]

    all_model_rows, all_perm_rows, all_feature_rows = [], [], []
    contrast_variant_meta = {}

    for contrast in contrasts:
        if features.empty:
            continue

        model_rows, perm_rows, variant_meta = run_contrast_models(features, contrast)
        feats = feature_tests(features, contrast)

        all_model_rows.extend(model_rows)
        all_perm_rows.extend(perm_rows)
        all_feature_rows.extend(feats)
        contrast_variant_meta[contrast] = variant_meta

        write_csv(OUTDIR / f"mood_v2_{contrast}_model_results.csv", model_rows)
        write_csv(OUTDIR / f"mood_v2_{contrast}_permutation_results.csv", perm_rows)
        write_csv(OUTDIR / f"mood_v2_{contrast}_feature_tests_bh_fdr.csv", feats)

    model_path = write_csv(OUTDIR / "mood_v2_all_model_results.csv", all_model_rows)
    perm_path = write_csv(OUTDIR / "mood_v2_all_permutation_results.csv", all_perm_rows)
    feature_path = write_csv(OUTDIR / "mood_v2_all_feature_tests_bh_fdr.csv", all_feature_rows)

    final_status, readiness_score, next_wall, contrast_status = decide_status(
        all_model_rows,
        all_perm_rows,
        contrast_variant_meta,
    )

    plots = []

    p = plot_model_bars(all_model_rows, OUTDIR / "mood_v2_model_auc_bars.png")
    if p:
        plots.append(p)

    p = plot_axis_advantage(contrast_status, OUTDIR / "mood_v2_axis_advantage.png")
    if p:
        plots.append(p)

    flag_counts = {}

    if not features.empty:
        for col in [
            "is_control", "clinical_any", "bipolar", "unipolar", "mood_any",
            "adhd", "add", "anxiety", "substance", "other",
            "bipolar_only", "unipolar_only", "bipolar_and_unipolar",
        ]:
            if col in features.columns:
                flag_counts[col] = int(np.nansum(features[col].astype(float).values))

    summary = {
        "test_name": "TAIRID mood actigraphy label + duration control v2",
        "boundary": (
            "Operational time-series axis test only. Not proof of TAIRID, not diagnosis, "
            "not medical advice, and not a cosmology result."
        ),
        "dataset": {
            "zenodo_record_id": ZENODO_RECORD_ID,
            "zenodo_api_url": ZENODO_API_URL,
            "record_title": record.get("metadata", {}).get("title"),
            "doi": record.get("doi"),
        },
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "parser_counts": {
            "downloads_count": len(downloads),
            "extraction_count": len(extraction),
            "info_subject_metadata_rows": len(meta_rows),
            "activity_series_count": len(series_rows),
            "subject_axis_rows": int(len(features)) if not features.empty else 0,
            "flag_counts": flag_counts,
        },
        "contrast_status": contrast_status,
        "model_results": all_model_rows,
        "permutation_results": all_perm_rows,
        "top_feature_tests": all_feature_rows[:50],
        "output_files": {
            "download_ledger_csv": str(OUTDIR / "mood_v2_download_ledger.csv"),
            "extraction_ledger_csv": str(OUTDIR / "mood_v2_extraction_ledger.csv"),
            "subject_metadata_flags_csv": str(OUTDIR / "mood_v2_subject_metadata_flags.csv"),
            "table_inventory_csv": str(OUTDIR / "mood_v2_table_inventory.csv"),
            "series_rows_csv": str(OUTDIR / "mood_v2_series_rows.csv"),
            "subject_axis_features_csv": str(OUTDIR / "mood_v2_subject_axis_features.csv"),
            "all_model_results_csv": str(model_path),
            "all_permutation_results_csv": str(perm_path),
            "all_feature_tests_bh_fdr_csv": str(feature_path),
            "plots": plots,
        },
        "interpretation": {
            "what_changed_from_v1": (
                "v2 parses diagnosis flags directly, avoids numeric-key metadata collisions, "
                "removes raw count/duration features from model sets, adds duration-overlap "
                "and duration/age-adjusted sensitivity."
            ),
            "what_supports_TAIRID_here": (
                "Bipolar/mood rhythm, cycling, reach, viability, or hysteresis models outperform "
                "static activity averages after duration controls and beat permutation expectation."
            ),
            "what_weakens_the_lane": (
                "Axis models lose advantage after duration adjustment or only broad clinical-vs-control remains positive."
            ),
            "truth_boundary": (
                "A positive result supports a mood/actigraphy cycling-hysteresis lane only. "
                "It cannot prove TAIRID, diagnose bipolar disorder, or prove cosmology."
            ),
        },
    }

    (OUTDIR / "mood_actigraphy_label_duration_control_v2_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with open(OUTDIR / "mood_actigraphy_label_duration_control_v2_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID mood actigraphy label + duration control v2\n\n")
        f.write("Boundary: operational time-series axis test only. Not proof. Not diagnosis. Not medical advice.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")
        f.write("Why this test exists:\n")
        f.write("- v1 locked broad mood/clinical cycling but could not call it bipolar-specific.\n")
        f.write("- v1 also included duration/count-sensitive features.\n")
        f.write("- v2 parses diagnosis flags directly and uses rate/fraction/duration-adjusted models.\n\n")
        f.write("Contrast status:\n")
        f.write(json.dumps(contrast_status, indent=2) + "\n\n")
        f.write("Truth boundary:\n")
        f.write("- This can support mood-actigraphy as a cycling/hysteresis lane.\n")
        f.write("- It cannot prove TAIRID.\n")
        f.write("- It cannot diagnose bipolar disorder, depression, ADHD, or any condition.\n")
        f.write("- It cannot prove any cosmology claim.\n")

    print("\nTAIRID mood actigraphy label + duration control v2 complete.")
    print("Created:")
    print("  tairid_mood_actigraphy_label_duration_control_v2_outputs/mood_actigraphy_label_duration_control_v2_summary.json")
    print("  tairid_mood_actigraphy_label_duration_control_v2_outputs/mood_actigraphy_label_duration_control_v2_summary.txt")
    print("  tairid_mood_actigraphy_label_duration_control_v2_outputs/mood_v2_all_model_results.csv")
    print("  tairid_mood_actigraphy_label_duration_control_v2_outputs/mood_v2_all_permutation_results.csv")
    print("\nBoundary:")
    print("  This is not proof of TAIRID.")
    print("  This is not diagnosis or medical advice.")
    print("  This is a label/duration-control actigraphy axis test.")
    print(f"\nFinal status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
