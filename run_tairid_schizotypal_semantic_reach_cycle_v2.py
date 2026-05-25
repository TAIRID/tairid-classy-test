#!/usr/bin/env python3
"""
TAIRID schizotypal semantic reach / cycle test v2.

Purpose:
v1 found the OSF semantic-priming data and the schizotypy score columns, but it failed
to preserve enough semantic context structure to build subject-level TAIRID features.

v2 fixes the parser directly:
- PrimDir filename => direct semantic priming
- PrimInd filename => indirect semantic priming
- short / long SOA => ordered semantic timing context
- participant / part accepted as subject columns
- related / unrelated and rel / unrel normalized
- direct/indirect × related/unrelated × short/long contexts are built per subject

Core question:
Does high schizotypy show semantic/reference reach, indirect priming spread,
semantic viability-window breach, or re-entry/cycling beyond static RT features?

Boundary:
This is not proof of TAIRID.
This is not diagnosis.
This is not medical advice.
This is not a cosmology result.
It is an operational semantic-reach / cycle axis test.
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
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


OUTDIR = Path("tairid_schizotypal_semantic_reach_cycle_v2_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_DIR = OUTDIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

OSF_NODE_ID = "j29fn"
OSF_FILES_URL = f"https://api.osf.io/v2/nodes/{OSF_NODE_ID}/files/"

RANDOM_SEED = 42
CV_REPEATS = 80
PERMUTATIONS = 150
PERM_REPEATS = 15
RIDGE = 1.0e-3
TOLERANCE_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0]

SCORE_COLS = [
    "SPQPos",
    "SPQNeg",
    "SPQDis",
    "OLIFEPos",
    "OLIFENeg",
    "OLIFEDis",
    "OLIFEImp",
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
            "User-Agent": "TAIRID-schizotypal-semantic-reach-cycle-v2",
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
    seen = set()
    files = []
    inventory = []

    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        try:
            page = read_json_url(url)
        except Exception as exc:
            inventory.append({"url": url, "status": "api_read_failed", "error": str(exc)})
            continue

        for item in page.get("data", []):
            attrs = item.get("attributes", {}) or {}
            links = item.get("links", {}) or {}
            rels = item.get("relationships", {}) or {}

            row = {
                "id": item.get("id"),
                "name": attrs.get("name") or item.get("id") or "unknown",
                "kind": attrs.get("kind") or "unknown",
                "path": attrs.get("path"),
                "materialized_path": attrs.get("materialized_path"),
                "size": attrs.get("size"),
                "download": links.get("download"),
            }
            inventory.append(row)

            if row["kind"] == "file" and row["download"]:
                files.append(row)
            else:
                related = rels.get("files", {}).get("links", {}).get("related", {}).get("href")
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
            downloads.append({"name": name, "status": "no_download_url"})
            continue

        local = DOWNLOAD_DIR / safe_name(name)
        if local.exists():
            local = DOWNLOAD_DIR / f"{local.stem}_{safe_name(str(f.get('id')))}{local.suffix}"

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
                extracted.append({"archive": str(path), "status": "zip_extracted", "target": str(target)})
            except Exception as exc:
                extracted.append({"archive": str(path), "status": "zip_extract_failed", "error": str(exc)})

        elif lower.endswith((".tar", ".tar.gz", ".tgz")):
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


def list_csv_files():
    out = []
    for root in [DOWNLOAD_DIR, EXTRACT_DIR]:
        for path in root.rglob("*"):
            if path.is_file() and not is_junk_path(path) and path.name.lower().endswith(".csv"):
                out.append(path)
    return sorted(set(out))


def read_csv_flexible(path):
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df, enc, None
        except Exception as exc:
            last = str(exc)
    return None, None, last


def infer_directness_from_filename(path):
    name = Path(path).name.lower()

    if "primdir" in name:
        return "direct"
    if "primind" in name:
        return "indirect"
    return None


def english_preference_score(path):
    name = Path(path).name.lower()
    score = 0
    if "english" in name:
        score += 100
    if "111019" in name:
        score += 20
    if name.endswith(".csv"):
        score += 5
    return score


def select_priming_csvs():
    candidates = []
    inventory = []

    for path in list_csv_files():
        directness = infer_directness_from_filename(path)
        if directness is None:
            continue

        df, enc, err = read_csv_flexible(path)
        if df is None:
            inventory.append({"path": str(path), "status": "read_failed", "error": err})
            continue

        cols_norm = {norm(c): c for c in df.columns}
        has_subject = any(k in cols_norm for k in ["participant", "part"])
        has_rt = any(k in cols_norm for k in ["reactiontime", "rt"])
        has_relation = "relation" in cols_norm
        has_soa = "soa" in cols_norm

        ok = has_subject and has_rt and has_relation and has_soa and len(df) >= 100

        inventory.append(
            {
                "path": str(path),
                "status": "candidate" if ok else "not_usable",
                "directness": directness,
                "encoding": enc,
                "rows": int(len(df)),
                "columns": " | ".join(df.columns),
                "has_subject": has_subject,
                "has_rt": has_rt,
                "has_relation": has_relation,
                "has_soa": has_soa,
                "english_preference_score": english_preference_score(path),
            }
        )

        if ok:
            candidates.append((directness, english_preference_score(path), path, df))

    selected = []
    for directness in ["direct", "indirect"]:
        subset = [x for x in candidates if x[0] == directness]
        if not subset:
            continue
        subset.sort(key=lambda x: (x[1], len(x[3])), reverse=True)
        selected.append(subset[0])

    return selected, inventory


def col_by_norm(df, options):
    mapping = {norm(c): c for c in df.columns}
    for opt in options:
        if norm(opt) in mapping:
            return mapping[norm(opt)]
    return None


def normalize_soa(value):
    s = str(value).strip().lower()
    if "short" in s:
        return "short", 1.0
    if "long" in s:
        return "long", 2.0
    try:
        val = float(s)
        return str(val), val
    except Exception:
        return "unknown", np.nan


def normalize_relation(value):
    s = str(value).strip().lower()
    c = norm(s)

    if c in ["rel", "related", "r"] or "related" in c and "unrelated" not in c:
        return "related"

    if c in ["unrel", "unrelated", "un"] or "unrelated" in c:
        return "unrelated"

    return "unknown"


def normalize_lexicality(value):
    s = str(value).strip().lower()
    c = norm(s)

    if c in ["word", "pal", "realword"]:
        return "word"

    if "pseudo" in c or c in ["nonword", "pseudoword"]:
        return "pseudoword"

    return "unknown"


def build_trial_table(selected_files):
    rows = []
    sources = []

    for directness, pref, path, df in selected_files:
        subject_col = col_by_norm(df, ["participant", "part"])
        rt_col = col_by_norm(df, ["reactiontime", "rt"])
        relation_col = col_by_norm(df, ["relation"])
        soa_col = col_by_norm(df, ["soa"])
        lexicality_col = col_by_norm(df, ["lexicality", "lex"])
        prime_col = col_by_norm(df, ["prime"])
        target_col = col_by_norm(df, ["target", "targ"])
        relprop_col = col_by_norm(df, ["relationproportion", "relprop"])

        parsed = 0

        for _, r in df.iterrows():
            subject = str(r.get(subject_col, "")).strip()
            if not subject:
                continue

            rt = pd.to_numeric(pd.Series([r.get(rt_col)]), errors="coerce").iloc[0]
            if not np.isfinite(rt) or rt <= 100 or rt > 10000:
                continue

            soa_label, soa_ord = normalize_soa(r.get(soa_col))
            relation = normalize_relation(r.get(relation_col))
            lexicality = normalize_lexicality(r.get(lexicality_col)) if lexicality_col else "unknown"

            row = {
                "source_file": str(path),
                "source_name": Path(path).name,
                "subject": subject,
                "directness": directness,
                "soa_label": soa_label,
                "soa_order": soa_ord,
                "relation": relation,
                "lexicality": lexicality,
                "rt": float(rt),
                "prime": str(r.get(prime_col, "")) if prime_col else "",
                "target": str(r.get(target_col, "")) if target_col else "",
                "relationproportion": pd.to_numeric(pd.Series([r.get(relprop_col)]), errors="coerce").iloc[0] if relprop_col else np.nan,
            }

            for sc in SCORE_COLS:
                col = col_by_norm(df, [sc])
                if col:
                    row[sc] = pd.to_numeric(pd.Series([r.get(col)]), errors="coerce").iloc[0]

            rows.append(row)
            parsed += 1

        sources.append(
            {
                "path": str(path),
                "directness": directness,
                "rows_in_file": int(len(df)),
                "parsed_rows": parsed,
                "subject_col": subject_col,
                "rt_col": rt_col,
                "soa_col": soa_col,
                "relation_col": relation_col,
                "lexicality_col": lexicality_col,
            }
        )

    trial = pd.DataFrame(rows)

    if not trial.empty:
        for sc in SCORE_COLS:
            if sc in trial.columns:
                trial[sc] = pd.to_numeric(trial[sc], errors="coerce")

    return trial, sources


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
    for flag in flags:
        if bool(flag):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


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


def safe_div(a, b):
    if b is None or not np.isfinite(b) or abs(b) <= 1.0e-12:
        return np.nan
    return float(a / b)


def median_or_nan(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().astype(float).values
    if len(vals) == 0:
        return np.nan
    return float(np.median(vals))


def build_subject_features(trial):
    rows = []

    if trial.empty:
        return pd.DataFrame()

    score_cols = [sc for sc in SCORE_COLS if sc in trial.columns]

    for subject, sub in trial.groupby("subject"):
        if len(sub) < 100:
            continue

        score_values = {}
        for sc in score_cols:
            score_values[sc] = median_or_nan(sub[sc])

        rt = pd.to_numeric(sub["rt"], errors="coerce").dropna().astype(float).values
        if len(rt) < 100:
            continue

        base = {
            "subject": subject,
            "trial_count": int(len(sub)),
            "rt_mean": float(np.mean(rt)),
            "rt_median": float(np.median(rt)),
            "rt_std": float(np.std(rt)),
            "rt_cv": float(np.std(rt) / max(abs(np.mean(rt)), 1.0e-9)),
            "rt_iqr": float(np.percentile(rt, 75) - np.percentile(rt, 25)),
            **score_values,
        }

        # Contexts preserve direct/indirect, short/long, related/unrelated, and word/pseudoword.
        ctx_rows = []
        for keys, g in sub.groupby(["directness", "soa_label", "soa_order", "relation", "lexicality"], dropna=False):
            directness, soa_label, soa_order, relation, lexicality = keys
            rts = pd.to_numeric(g["rt"], errors="coerce").dropna().astype(float).values
            if len(rts) < 3:
                continue

            ctx_rows.append(
                {
                    "directness": directness,
                    "soa_label": soa_label,
                    "soa_order": float(soa_order) if np.isfinite(soa_order) else np.nan,
                    "relation": relation,
                    "lexicality": lexicality,
                    "n": int(len(rts)),
                    "rt_median": float(np.median(rts)),
                    "rt_mean": float(np.mean(rts)),
                    "rt_cv": float(np.std(rts) / max(abs(np.mean(rts)), 1.0e-9)),
                    "rt_iqr": float(np.percentile(rts, 75) - np.percentile(rts, 25)),
                }
            )

        ctx = pd.DataFrame(ctx_rows)
        if ctx.empty or len(ctx) < 6:
            continue

        direct_rank = {"direct": 0.0, "indirect": 1.0}
        rel_rank = {"related": 0.0, "unknown": 0.5, "unrelated": 1.0}
        lex_rank = {"word": 0.0, "unknown": 0.5, "pseudoword": 1.0}

        ctx["semantic_order"] = (
            ctx["directness"].map(direct_rank).fillna(0.5)
            + ctx["relation"].map(rel_rank).fillna(0.5)
            + ctx["lexicality"].map(lex_rank).fillna(0.5)
            + 0.10 * ctx["soa_order"].fillna(0.0)
        )

        ctx = ctx.sort_values(["semantic_order", "soa_order", "directness", "relation", "lexicality"]).reset_index(drop=True)

        # T = RT pacing. I = semantic constraint pressure + instability.
        T_raw = ctx["rt_median"].astype(float).values
        I_raw = (
            ctx["rt_cv"].fillna(ctx["rt_cv"].median()).astype(float).values
            + 0.20 * ctx["directness"].map({"direct": 0.0, "indirect": 1.0}).fillna(0.5).values
            + 0.20 * ctx["relation"].map({"related": 0.0, "unrelated": 1.0, "unknown": 0.5}).fillna(0.5).values
            + 0.15 * ctx["lexicality"].map({"word": 0.0, "pseudoword": 1.0, "unknown": 0.5}).fillna(0.5).values
        )

        T = zscore(T_raw)
        I = zscore(I_raw)
        M = np.abs(T - I)
        C = np.sqrt(T ** 2 + I ** 2)
        x = np.arange(len(ctx), dtype=float)

        base.update(
            {
                "context_count": int(len(ctx)),
                "M_mean": float(np.mean(M)),
                "M_median": float(np.median(M)),
                "M_max": float(np.max(M)),
                "M_range": float(np.max(M) - np.min(M)),
                "M_iqr": float(np.percentile(M, 75) - np.percentile(M, 25)),
                "M_slope_context": slope(x, M),
                "M_curvature_context": curve(x, M),
                "T_context_slope": slope(x, T),
                "I_context_slope": slope(x, I),
                "T_I_slope_gap": float(abs(slope(x, T) - slope(x, I))) if np.isfinite(slope(x, T)) and np.isfinite(slope(x, I)) else np.nan,
                "collapse_load_mean": float(np.mean(C)),
                "collapse_load_max": float(np.max(C)),
                "collapse_load_range": float(np.max(C) - np.min(C)),
                "Cyc_M_turns": sign_change_count(M),
                "Cyc_T_turns": sign_change_count(T),
                "Cyc_I_turns": sign_change_count(I),
                "Cyc_C_turns": sign_change_count(C),
                "Cyc_M_turn_rate": float(sign_change_count(M) / max(len(M) - 2, 1)),
                "Cyc_TI_opposition_fraction": float(np.mean(np.sign(np.diff(T)) * np.sign(np.diff(I)) < 0)) if len(T) > 2 else np.nan,
            }
        )

        # Semantic priming reach: unrelated minus related. Positive means facilitation.
        priming = {}

        for directness in ["direct", "indirect"]:
            for soa_label in ["short", "long"]:
                for lexicality in ["word", "pseudoword"]:
                    g = sub[
                        (sub["directness"] == directness)
                        & (sub["soa_label"] == soa_label)
                        & (sub["lexicality"] == lexicality)
                    ]

                    rel_rt = median_or_nan(g[g["relation"] == "related"]["rt"])
                    unrel_rt = median_or_nan(g[g["relation"] == "unrelated"]["rt"])

                    key = f"{directness}_{soa_label}_{lexicality}"

                    if np.isfinite(rel_rt) and np.isfinite(unrel_rt):
                        priming[key] = float(unrel_rt - rel_rt)
                    else:
                        priming[key] = np.nan

        direct_word_vals = [priming.get(f"direct_{soa}_word", np.nan) for soa in ["short", "long"]]
        indirect_word_vals = [priming.get(f"indirect_{soa}_word", np.nan) for soa in ["short", "long"]]
        direct_all_vals = [v for k, v in priming.items() if k.startswith("direct_") and np.isfinite(v)]
        indirect_all_vals = [v for k, v in priming.items() if k.startswith("indirect_") and np.isfinite(v)]
        all_prim_vals = [v for v in priming.values() if np.isfinite(v)]

        direct_word_mean = float(np.nanmean(direct_word_vals)) if np.any(np.isfinite(direct_word_vals)) else np.nan
        indirect_word_mean = float(np.nanmean(indirect_word_vals)) if np.any(np.isfinite(indirect_word_vals)) else np.nan
        direct_all_mean = float(np.nanmean(direct_all_vals)) if len(direct_all_vals) else np.nan
        indirect_all_mean = float(np.nanmean(indirect_all_vals)) if len(indirect_all_vals) else np.nan

        for key, value in priming.items():
            base[f"Priming_{key}_ms"] = value

        base.update(
            {
                "Reach_direct_word_priming_mean_ms": direct_word_mean,
                "Reach_indirect_word_priming_mean_ms": indirect_word_mean,
                "Reach_direct_all_priming_mean_ms": direct_all_mean,
                "Reach_indirect_all_priming_mean_ms": indirect_all_mean,
                "Reach_indirect_minus_direct_word_ms": float(indirect_word_mean - direct_word_mean) if np.isfinite(indirect_word_mean) and np.isfinite(direct_word_mean) else np.nan,
                "Reach_indirect_minus_direct_all_ms": float(indirect_all_mean - direct_all_mean) if np.isfinite(indirect_all_mean) and np.isfinite(direct_all_mean) else np.nan,
                "Reach_indirect_to_direct_word_ratio": safe_div(indirect_word_mean, abs(direct_word_mean)) if np.isfinite(indirect_word_mean) and np.isfinite(direct_word_mean) else np.nan,
                "Reach_priming_span_ms": float(np.nanmax(all_prim_vals) - np.nanmin(all_prim_vals)) if len(all_prim_vals) >= 2 else np.nan,
                "Reach_priming_max_ms": float(np.nanmax(all_prim_vals)) if len(all_prim_vals) else np.nan,
                "Reach_priming_min_ms": float(np.nanmin(all_prim_vals)) if len(all_prim_vals) else np.nan,
                "Cyc_priming_sign_changes": sign_change_count(all_prim_vals) if len(all_prim_vals) >= 3 else np.nan,
                "Cyc_priming_turn_rate": float(sign_change_count(all_prim_vals) / max(len(all_prim_vals) - 2, 1)) if len(all_prim_vals) >= 3 else np.nan,
            }
        )

        baseline = float(np.percentile(M, 25))
        local_sd = float(np.std(M))

        for mult in TOLERANCE_MULTIPLIERS:
            suffix = f"tol{str(mult).replace('.', 'p')}"
            W = baseline + mult * local_sd
            B = np.maximum(0.0, M - W)
            breached = B > 0.0
            stable = ~breached
            transitions = np.abs(np.diff(breached.astype(int))) if len(breached) > 1 else np.asarray([])

            base[f"W_{suffix}"] = float(W)
            base[f"B_mean_{suffix}"] = float(np.mean(B))
            base[f"B_max_{suffix}"] = float(np.max(B))
            base[f"B_fraction_{suffix}"] = float(np.mean(breached))
            base[f"Reach_stable_fraction_{suffix}"] = float(np.mean(stable))
            base[f"Reach_breach_fraction_{suffix}"] = float(np.mean(breached))
            base[f"Reach_longest_stable_run_fraction_{suffix}"] = float(longest_true_run(stable) / max(len(stable), 1))
            base[f"Reach_longest_breach_run_fraction_{suffix}"] = float(longest_true_run(breached) / max(len(breached), 1))
            base[f"Cyc_breach_transition_rate_{suffix}"] = float(np.mean(transitions)) if len(transitions) else 0.0
            base[f"Cyc_breach_reentry_rate_{suffix}"] = float(np.sum((breached[:-1] == True) & (breached[1:] == False)) / max(len(breached) - 1, 1)) if len(breached) > 1 else 0.0

        rows.append(base)

    return pd.DataFrame(rows)


def make_high_low_contrast(features, score_col):
    df = features.copy()
    score = pd.to_numeric(df[score_col], errors="coerce")
    df = df[np.isfinite(score)].copy()
    score = pd.to_numeric(df[score_col], errors="coerce")

    if len(df) < 20 or score.nunique() < 5:
        return pd.DataFrame(), {"status": "not_enough_score_range"}

    lo = float(score.quantile(1 / 3))
    hi = float(score.quantile(2 / 3))

    low = df[score <= lo].copy()
    high = df[score >= hi].copy()
    out = pd.concat([low, high], ignore_index=True)
    out["label"] = (pd.to_numeric(out[score_col], errors="coerce") >= hi).astype(int)

    return out, {
        "status": "ok",
        "score_col": score_col,
        "low_threshold": lo,
        "high_threshold": hi,
        "n_total_with_score": int(len(df)),
        "n_low": int(len(low)),
        "n_high": int(len(high)),
    }


def model_feature_sets(df):
    static = [
        "rt_mean",
        "rt_median",
        "rt_std",
        "rt_cv",
        "rt_iqr",
    ]

    semantic_reach = [
        "Reach_direct_word_priming_mean_ms",
        "Reach_indirect_word_priming_mean_ms",
        "Reach_direct_all_priming_mean_ms",
        "Reach_indirect_all_priming_mean_ms",
        "Reach_indirect_minus_direct_word_ms",
        "Reach_indirect_minus_direct_all_ms",
        "Reach_indirect_to_direct_word_ratio",
        "Reach_priming_span_ms",
        "Reach_priming_max_ms",
        "Reach_priming_min_ms",
    ]

    for directness in ["direct", "indirect"]:
        for soa in ["short", "long"]:
            for lex in ["word", "pseudoword"]:
                semantic_reach.append(f"Priming_{directness}_{soa}_{lex}_ms")

    mismatch = [
        "M_mean",
        "M_median",
        "M_max",
        "M_range",
        "M_iqr",
        "M_slope_context",
        "M_curvature_context",
        "T_I_slope_gap",
        "collapse_load_mean",
        "collapse_load_max",
        "collapse_load_range",
    ]

    viability = []
    reach_window = []
    cycling = [
        "Cyc_M_turns",
        "Cyc_T_turns",
        "Cyc_I_turns",
        "Cyc_C_turns",
        "Cyc_M_turn_rate",
        "Cyc_TI_opposition_fraction",
        "Cyc_priming_sign_changes",
        "Cyc_priming_turn_rate",
    ]

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
        "static_rt_model": static,
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


def run_models_for_score(features, score_col):
    contrast, meta = make_high_low_contrast(features, score_col)
    model_rows = []
    perm_rows = []

    if contrast.empty or meta.get("status") != "ok":
        return {
            "score_col": score_col,
            "status": "not_enough_score_data",
            "contrast_meta": meta,
            "model_rows": [],
            "permutation_rows": [],
        }

    for name, cols in model_feature_sets(contrast).items():
        res = repeated_cv(contrast, cols)
        auc = res.get("auc_mean")

        model_rows.append(
            {
                "score_col": score_col,
                "model_name": name,
                **{k: v for k, v in res.items() if k != "feature_cols"},
                "feature_cols": " | ".join(res.get("feature_cols", [])),
            }
        )

        perm = permutation_test(contrast, cols, observed_auc=auc)
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
        "status": "ok",
        "contrast_meta": meta,
        "model_rows": model_rows,
        "permutation_rows": perm_rows,
    }


def spearman_feature_tests(features, score_col):
    rows = []

    if score_col not in features.columns:
        return rows

    score = pd.to_numeric(features[score_col], errors="coerce")

    numeric_cols = [
        c for c in features.columns
        if c not in ["subject", score_col]
        and pd.api.types.is_numeric_dtype(features[c])
        and features[c].notnull().sum() >= 8
    ]

    for c in numeric_cols:
        vals = pd.to_numeric(features[c], errors="coerce")
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
    by_model = defaultdict(dict)
    by_perm = defaultdict(dict)

    for r in model_rows:
        if r.get("status") == "ok":
            by_model[r["score_col"]][r["model_name"]] = r

    for r in perm_rows:
        if r.get("permutation_status") == "ok":
            by_perm[r["score_col"]][r["model_name"]] = r

    axis_names = [
        "semantic_reach_model",
        "mismatch_dynamic_model",
        "viability_window_model",
        "reach_window_model",
        "cycling_reentry_model",
        "reach_cycle_model",
        "combined_axis_model",
    ]

    score_status = {}

    for sc in score_cols:
        models = by_model.get(sc, {})
        perms = by_perm.get(sc, {})
        static_auc = models.get("static_rt_model", {}).get("auc_mean")

        axis_values = [
            (models.get(name, {}).get("auc_mean"), name)
            for name in axis_names
            if models.get(name, {}).get("auc_mean") is not None
        ]

        best_axis_auc, best_axis_model = max(axis_values, key=lambda x: x[0]) if axis_values else (None, None)

        ps = [
            perms.get(name, {}).get("p_value_ge_observed")
            for name in axis_names
            if perms.get(name, {}).get("p_value_ge_observed") is not None
        ]
        best_p = min(ps) if ps else None

        semantic_auc = models.get("semantic_reach_model", {}).get("auc_mean")
        reach_cycle_auc = models.get("reach_cycle_model", {}).get("auc_mean")
        cycling_auc = models.get("cycling_reentry_model", {}).get("auc_mean")

        support = (
            static_auc is not None
            and best_axis_auc is not None
            and best_axis_auc >= 0.60
            and best_axis_auc > static_auc + 0.03
        )

        locked = bool(support and best_p is not None and best_p <= 0.05)

        score_status[sc] = {
            "static_auc": static_auc,
            "semantic_reach_auc": semantic_auc,
            "reach_cycle_auc": reach_cycle_auc,
            "cycling_reentry_auc": cycling_auc,
            "best_axis_auc": best_axis_auc,
            "best_axis_model": best_axis_model,
            "best_axis_permutation_p": best_p,
            "axis_support": support,
            "axis_locked": locked,
        }

    reach_locked = [
        sc for sc, st in score_status.items()
        if st.get("axis_locked")
        and st.get("best_axis_model") in ["semantic_reach_model", "reach_window_model", "reach_cycle_model"]
    ]

    locked_any = [sc for sc, st in score_status.items() if st.get("axis_locked")]
    supported_any = [sc for sc, st in score_status.items() if st.get("axis_support")]

    if reach_locked:
        return (
            "schizotypal_semantic_reach_axis_locked",
            9,
            "Add schizotypal semantic reach as the reference-field reach lane and compare reach-first vs cycle-first fingerprints.",
            score_status,
        )

    if locked_any:
        return (
            "schizotypal_semantic_axis_locked_nonreach_primary",
            8,
            "Use schizotypal data as semantic-axis support, but inspect whether mismatch, viability, or cycling is primary.",
            score_status,
        )

    if supported_any:
        return (
            "schizotypal_semantic_axis_supported_not_locked",
            7,
            "Treat the schizotypal lane as directional; inspect score mapping and semantic feature construction before promotion.",
            score_status,
        )

    return (
        "schizotypal_semantic_reach_not_supported_after_parser_correction",
        6,
        "Corrected parser ran, but semantic reach/cycle did not beat static RT strongly enough.",
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
    plt.title("TAIRID schizotypal semantic reach / cycle v2")
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
    plt.ylabel("Best axis AUC - static RT AUC")
    plt.title("Schizotypal semantic axis advantage over static")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def main():
    print("")
    print("TAIRID schizotypal semantic reach / cycle v2 starting.")
    print("Boundary: operational semantic axis test only; not proof, diagnosis, or medical advice.")
    print("")

    osf_files, osf_inventory = collect_osf_files()
    write_csv(OUTDIR / "schizotypal_v2_osf_file_inventory.csv", osf_inventory)

    downloads = download_osf_files(osf_files)
    write_csv(OUTDIR / "schizotypal_v2_download_ledger.csv", downloads)

    extraction = extract_archives(downloads)
    write_csv(OUTDIR / "schizotypal_v2_extraction_ledger.csv", extraction)

    selected_files, priming_inventory = select_priming_csvs()
    write_csv(OUTDIR / "schizotypal_v2_priming_file_inventory.csv", priming_inventory)

    selected_rows = [
        {
            "directness": directness,
            "path": str(path),
            "rows": int(len(df)),
            "columns": " | ".join(df.columns),
            "preference_score": pref,
        }
        for directness, pref, path, df in selected_files
    ]
    write_csv(OUTDIR / "schizotypal_v2_selected_priming_files.csv", selected_rows)

    trial, trial_sources = build_trial_table(selected_files)
    write_csv(OUTDIR / "schizotypal_v2_trial_sources.csv", trial_sources)

    if not trial.empty:
        trial.to_csv(OUTDIR / "schizotypal_v2_trial_table.csv", index=False)

    features = build_subject_features(trial) if not trial.empty else pd.DataFrame()

    if not features.empty:
        features.to_csv(OUTDIR / "schizotypal_v2_subject_semantic_axis_features.csv", index=False)

    score_cols = [sc for sc in SCORE_COLS if sc in features.columns and features[sc].notnull().sum() >= 20]

    all_model_rows = []
    all_perm_rows = []
    all_spearman_rows = []
    contrast_meta = []

    for sc in score_cols:
        result = run_models_for_score(features, sc)

        all_model_rows.extend(result.get("model_rows", []))
        all_perm_rows.extend(result.get("permutation_rows", []))

        contrast_meta.append(
            {
                "score_col": sc,
                "status": result.get("status"),
                **result.get("contrast_meta", {}),
            }
        )

        spearman_rows = spearman_feature_tests(features, sc)
        all_spearman_rows.extend(spearman_rows)

        write_csv(OUTDIR / f"schizotypal_v2_{safe_name(sc)}_model_results.csv", result.get("model_rows", []))
        write_csv(OUTDIR / f"schizotypal_v2_{safe_name(sc)}_permutation_results.csv", result.get("permutation_rows", []))
        write_csv(OUTDIR / f"schizotypal_v2_{safe_name(sc)}_spearman_feature_tests.csv", spearman_rows)

    model_path = write_csv(OUTDIR / "schizotypal_v2_all_model_results.csv", all_model_rows)
    perm_path = write_csv(OUTDIR / "schizotypal_v2_all_permutation_results.csv", all_perm_rows)
    spearman_path = write_csv(OUTDIR / "schizotypal_v2_all_spearman_feature_tests.csv", all_spearman_rows)
    contrast_meta_path = write_csv(OUTDIR / "schizotypal_v2_contrast_meta.csv", contrast_meta)

    final_status, readiness_score, next_wall, score_status = decide_status(
        all_model_rows,
        all_perm_rows,
        score_cols,
    )

    plots = []
    p = plot_model_bars(all_model_rows, OUTDIR / "schizotypal_v2_model_auc_bars.png")
    if p:
        plots.append(p)

    p = plot_axis_advantage(score_status, OUTDIR / "schizotypal_v2_axis_advantage.png")
    if p:
        plots.append(p)

    parser_counts = {
        "osf_file_inventory_count": len(osf_inventory),
        "download_count": len(downloads),
        "extraction_count": len(extraction),
        "selected_priming_file_count": len(selected_files),
        "trial_rows": int(len(trial)) if not trial.empty else 0,
        "trial_subjects": int(trial["subject"].nunique()) if not trial.empty else 0,
        "subject_feature_rows": int(len(features)) if not features.empty else 0,
        "score_cols_used": score_cols,
        "directness_counts": trial["directness"].value_counts().to_dict() if not trial.empty else {},
        "soa_counts": trial["soa_label"].value_counts().to_dict() if not trial.empty else {},
        "relation_counts": trial["relation"].value_counts().to_dict() if not trial.empty else {},
        "lexicality_counts": trial["lexicality"].value_counts().to_dict() if not trial.empty else {},
    }

    summary = {
        "test_name": "TAIRID schizotypal semantic reach / cycle v2",
        "boundary": (
            "Operational semantic reach/cycle axis test only. Not proof of TAIRID, "
            "not diagnosis, not medical advice, and not a cosmology result."
        ),
        "dataset": {
            "osf_node_id": OSF_NODE_ID,
            "osf_files_url": OSF_FILES_URL,
            "note": (
                "v2 parser-corrected pass: PrimDir=direct, PrimInd=indirect, short/long SOA preserved, "
                "participant/part accepted, direct/indirect × related/unrelated × short/long contexts built."
            ),
        },
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "parser_counts": parser_counts,
        "score_status": score_status,
        "contrast_meta": contrast_meta,
        "model_results": all_model_rows,
        "permutation_results": all_perm_rows,
        "top_spearman_feature_tests": all_spearman_rows[:50],
        "output_files": {
            "osf_file_inventory_csv": str(OUTDIR / "schizotypal_v2_osf_file_inventory.csv"),
            "download_ledger_csv": str(OUTDIR / "schizotypal_v2_download_ledger.csv"),
            "priming_file_inventory_csv": str(OUTDIR / "schizotypal_v2_priming_file_inventory.csv"),
            "selected_priming_files_csv": str(OUTDIR / "schizotypal_v2_selected_priming_files.csv"),
            "trial_table_csv": str(OUTDIR / "schizotypal_v2_trial_table.csv") if not trial.empty else None,
            "subject_semantic_axis_features_csv": str(OUTDIR / "schizotypal_v2_subject_semantic_axis_features.csv") if not features.empty else None,
            "model_results_csv": str(model_path),
            "permutation_results_csv": str(perm_path),
            "spearman_feature_tests_csv": str(spearman_path),
            "contrast_meta_csv": str(contrast_meta_path),
            "plots": plots,
        },
        "interpretation": {
            "what_fixed_from_v1": (
                "The v1 parser collapsed directness and SOA context. v2 forces PrimDir/PrimInd directness, "
                "converts short/long SOA, and builds enough subject-level contexts for TAIRID axis modeling."
            ),
            "what_supports_TAIRID_here": (
                "Semantic reach, reach-window, mismatch, viability, or re-entry/cycling models outperform static RT "
                "for high-vs-low schizotypy scores and beat permutation expectation."
            ),
            "what_weakens_the_lane": (
                "Static RT remains strongest, semantic reach/cycle models fail permutation, or corrected parser still "
                "cannot recover enough subject-level contexts."
            ),
            "truth_boundary": (
                "A positive result supports schizotypal semantic reach as an operational TAIRID lane. "
                "It cannot diagnose schizotypy, prove TAIRID, or prove cosmology."
            ),
        },
    }

    write_json(OUTDIR / "schizotypal_semantic_reach_cycle_v2_summary.json", summary)

    with open(OUTDIR / "schizotypal_semantic_reach_cycle_v2_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID schizotypal semantic reach / cycle v2\n\n")
        f.write("Boundary: operational semantic axis test only. Not proof. Not diagnosis. Not medical advice.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")
        f.write("Parser correction:\n")
        f.write("- PrimDir filenames are direct.\n")
        f.write("- PrimInd filenames are indirect.\n")
        f.write("- short/long SOA is preserved as ordered context.\n")
        f.write("- participant/part subject columns are accepted.\n")
        f.write("- Direct/indirect × related/unrelated × short/long contexts are built per subject.\n\n")
        f.write("Parser counts:\n")
        f.write(json.dumps(parser_counts, indent=2, default=str) + "\n\n")
        f.write("Score status:\n")
        f.write(json.dumps(score_status, indent=2, default=str) + "\n\n")
        f.write("Truth boundary:\n")
        f.write("- This can support a schizotypal semantic reach/cycle lane.\n")
        f.write("- It cannot prove TAIRID.\n")
        f.write("- It cannot diagnose schizotypy or schizotypal personality disorder.\n")
        f.write("- It cannot prove any cosmology claim.\n")

    print("")
    print("TAIRID schizotypal semantic reach / cycle v2 complete.")
    print("Created:")
    print("  tairid_schizotypal_semantic_reach_cycle_v2_outputs/schizotypal_semantic_reach_cycle_v2_summary.json")
    print("  tairid_schizotypal_semantic_reach_cycle_v2_outputs/schizotypal_semantic_reach_cycle_v2_summary.txt")
    print("  tairid_schizotypal_semantic_reach_cycle_v2_outputs/schizotypal_v2_all_model_results.csv")
    print("  tairid_schizotypal_semantic_reach_cycle_v2_outputs/schizotypal_v2_all_permutation_results.csv")
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
