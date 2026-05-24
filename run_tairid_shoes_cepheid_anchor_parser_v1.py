#!/usr/bin/env python3
"""
TAIRID SH0ES Cepheid / anchor public-data parser v1.

Purpose:
The Pantheon+SH0ES supernova-only tests reached an offset-degenerate boundary.
A calibrator-only TAIRID gate could be absorbed by calibration-offset freedom,
even with full Pantheon+SH0ES covariance and calibration-covariance isolation.

That means the next necessary wall is not another Pantheon-only test.
The next wall is source-level calibration-ladder data:
Cepheids, anchors, host labels, period-luminosity variables, parallax/anchor
terms, covariance, or an official likelihood/data vector.

This script checks whether the public PantheonPlusSH0ES/DataRelease repository
contains enough Cepheid / anchor / calibration-ladder source structure to build
a first real ladder proxy.

It does:
1. Walk the full GitHub repository tree.
2. Look for SH0ES_Data, Cepheid, anchor, host, FITS, covariance, and likelihood files.
3. Download high-priority candidate files.
4. Parse text tables, NPZ files, and FITS files where possible.
5. Score whether any public file set contains enough fields for a Cepheid-anchor
   ladder likelihood proxy.
6. Emit an honest readiness result:
   - enough_for_first_ladder_proxy
   - partial_parser_needed
   - insufficient_public_ladder_data

Boundary:
This is not a Cepheid likelihood.
This is not a SH0ES likelihood.
This is not a cosmology fit.
This does not prove TAIRID cosmology.
This is a public-data readiness and parser audit.
"""

import csv
import gzip
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits


OUTDIR = Path("shoes_cepheid_anchor_parser_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

OWNER = "PantheonPlusSH0ES"
REPO = "DataRelease"
BRANCH = "main"

TREE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/"

MAX_DOWNLOAD_BYTES = 80_000_000
MAX_TEXT_PARSE_BYTES = 12_000_000
MAX_DOWNLOADS = 140

KEYWORDS = [
    "shoes",
    "sh0es",
    "cepheid",
    "ceph",
    "anchor",
    "calibrator",
    "calib",
    "host",
    "hst",
    "wfc3",
    "lmc",
    "smc",
    "ngc4258",
    "4258",
    "mw",
    "milky",
    "parallax",
    "gaia",
    "period",
    "plr",
    "metal",
    "feh",
    "crowd",
    "cov",
    "covar",
    "matrix",
    "likelihood",
    "cosmosis",
    "h0",
]

LADDER_FIELD_KEYWORDS = {
    "cepheid": ["cepheid", "ceph", "idcep", "cep_id"],
    "host": ["host", "galaxy", "gal", "sn_host"],
    "anchor": ["anchor", "lmc", "smc", "4258", "ngc4258", "mw", "milky", "gaia", "parallax"],
    "period": ["period", "logp", "log_p", "logper"],
    "magnitude": ["mag", "m_", "f160w", "f555w", "f814w", "wfc3", "phot"],
    "uncertainty": ["err", "sigma", "unc", "cov"],
    "metallicity": ["metal", "feh", "oh", "zmet"],
    "distance": ["mu", "dist", "distance", "modulus"],
    "calibration_flag": ["calib", "calibrator", "anchor"],
}

DATA_EXTENSIONS = {
    ".txt",
    ".dat",
    ".csv",
    ".tsv",
    ".fitres",
    ".cov",
    ".ini",
    ".dataset",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".py",
    ".npz",
    ".npy",
    ".fits",
    ".fit",
    ".fits.gz",
    ".gz",
}


REPORT = {
    "boundary": "Public-data parser audit only. Not a Cepheid likelihood and not proof.",
    "repository": f"{OWNER}/{REPO}",
    "branch": BRANCH,
    "tree_url": TREE_URL,
    "raw_base": RAW_BASE,
    "files": [],
    "downloaded": [],
    "parsed_text_tables": [],
    "parsed_npz_files": [],
    "parsed_fits_files": [],
    "candidate_ladder_tables": [],
    "readiness": {},
}


def get_json(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def get_bytes(url, max_bytes=MAX_DOWNLOAD_BYTES):
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    data = response.content

    if len(data) > max_bytes:
        raise RuntimeError(f"File too large for this parser: {len(data)} bytes")

    return data


def normalized_suffix(path):
    lower = path.lower()

    if lower.endswith(".fits.gz"):
        return ".fits.gz"

    return Path(path).suffix.lower()


def score_path(path, size):
    lower = path.lower()
    score = 0

    for keyword in KEYWORDS:
        if keyword in lower:
            score += 2

    if lower.startswith("shoes_data/") or lower.startswith("sh0es_data/"):
        score += 20

    if "cepheid" in lower or "ceph" in lower:
        score += 15

    if "anchor" in lower or "4258" in lower or "lmc" in lower or "smc" in lower:
        score += 10

    if lower.endswith(".fits") or lower.endswith(".fit") or lower.endswith(".fits.gz"):
        score += 10

    if lower.endswith(".npz") or lower.endswith(".npy"):
        score += 7

    if "pantheon+_data/4_distances_and_covar" in lower:
        score += 3

    if "cosmosis_likelihood" in lower:
        score += 5

    if size is not None and size > 0 and size < MAX_DOWNLOAD_BYTES:
        score += 1

    return score


def classify_path(path):
    lower = path.lower()
    tags = []

    if lower.startswith("shoes_data/") or lower.startswith("sh0es_data/"):
        tags.append("SH0ES_Data_candidate")

    if "cepheid" in lower or "ceph" in lower:
        tags.append("cepheid_candidate")

    if "anchor" in lower or "lmc" in lower or "smc" in lower or "4258" in lower or "ngc4258" in lower:
        tags.append("anchor_candidate")

    if "host" in lower:
        tags.append("host_candidate")

    if "calib" in lower or "calibrator" in lower:
        tags.append("calibration_candidate")

    if "cov" in lower or "covar" in lower:
        tags.append("covariance_candidate")

    if "likelihood" in lower or "cosmosis" in lower:
        tags.append("likelihood_candidate")

    if lower.endswith(".fits") or lower.endswith(".fit") or lower.endswith(".fits.gz"):
        tags.append("fits_candidate")

    if lower.endswith(".npz") or lower.endswith(".npy"):
        tags.append("numpy_candidate")

    return tags


def discover_files():
    tree = get_json(TREE_URL)
    items = tree.get("tree", [])

    files = []

    for item in items:
        if item.get("type") != "blob":
            continue

        path = item.get("path", "")
        size = item.get("size")
        suffix = normalized_suffix(path)

        info = {
            "path": path,
            "raw_url": RAW_BASE + path,
            "size": size,
            "extension": suffix,
            "score": score_path(path, size),
            "tags": classify_path(path),
            "is_candidate_extension": suffix in DATA_EXTENSIONS,
        }

        files.append(info)

    files = sorted(files, key=lambda x: (-x["score"], x["path"]))
    REPORT["files"] = files

    with open(OUTDIR / "repository_candidate_inventory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "path", "size", "extension", "score", "tags", "raw_url"])

        for rank, item in enumerate(files, start=1):
            writer.writerow(
                [
                    rank,
                    item["path"],
                    item["size"],
                    item["extension"],
                    item["score"],
                    "|".join(item["tags"]),
                    item["raw_url"],
                ]
            )

    return files


def text_from_bytes(data, path):
    lower = path.lower()

    if lower.endswith(".gz") and not lower.endswith(".fits.gz"):
        try:
            return gzip.decompress(data).decode("utf-8", errors="replace")
        except Exception:
            return data.decode("utf-8", errors="replace")

    return data.decode("utf-8", errors="replace")


def parse_text_table(text):
    lines = [line for line in text.splitlines() if line.strip()]

    if not lines:
        return None, "empty"

    header_line = None
    data_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("#") and header_line is None:
            possible = stripped[1:].strip()

            if len(possible.split()) >= 2:
                header_line = possible
            continue

        if stripped.startswith("#"):
            continue

        data_lines.append(stripped)

    if not data_lines:
        return None, "no_data_lines"

    if header_line is not None:
        table_text = header_line + "\n" + "\n".join(data_lines[:20000])
    else:
        table_text = "\n".join(data_lines[:20000])

    sample = table_text[:4000]

    if sample.count(",") > 5:
        sep = ","
    elif "\t" in sample:
        sep = "\t"
    else:
        sep = r"\s+"

    try:
        if sep == r"\s+":
            df = pd.read_csv(io.StringIO(table_text), sep=sep, engine="python")
        else:
            df = pd.read_csv(io.StringIO(table_text), sep=sep)

        if df.shape[0] == 0 or df.shape[1] == 0:
            return None, "empty_dataframe"

        return df, "parsed"

    except Exception as exc:
        return None, f"parse_failed: {exc}"


def field_hit_summary(columns):
    cols = [str(c).lower() for c in columns]
    hits = {}

    for category, keys in LADDER_FIELD_KEYWORDS.items():
        hit_cols = []

        for col in cols:
            if any(key in col for key in keys):
                hit_cols.append(col)

        hits[category] = sorted(set(hit_cols))

    coverage = {
        category: bool(values)
        for category, values in hits.items()
    }

    score = sum(1 for value in coverage.values() if value)

    return hits, coverage, score


def add_candidate_table(record):
    columns = record.get("columns", [])
    hits, coverage, score = field_hit_summary(columns)

    record["ladder_field_hits"] = hits
    record["ladder_field_coverage"] = coverage
    record["ladder_field_score"] = score

    enough_core = (
        coverage.get("host", False)
        and coverage.get("period", False)
        and coverage.get("magnitude", False)
        and coverage.get("uncertainty", False)
    )

    likely_anchor = coverage.get("anchor", False) or coverage.get("distance", False)

    if enough_core or score >= 5 or likely_anchor:
        REPORT["candidate_ladder_tables"].append(record)


def parse_npz(data, path, raw_url):
    local_path = OUTDIR / "downloaded_sources" / path.replace("/", "__")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)

    info = {
        "path": path,
        "raw_url": raw_url,
        "local_path": str(local_path),
        "arrays": [],
        "status": "parsed",
    }

    try:
        npz = np.load(io.BytesIO(data), allow_pickle=True)

        for key in npz.files:
            arr = npz[key]
            info["arrays"].append(
                {
                    "name": key,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }
            )

    except Exception as exc:
        info["status"] = "parse_failed"
        info["error"] = str(exc)

    REPORT["parsed_npz_files"].append(info)

    return info


def parse_fits(data, path, raw_url):
    local_path = OUTDIR / "downloaded_sources" / path.replace("/", "__")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)

    info = {
        "path": path,
        "raw_url": raw_url,
        "local_path": str(local_path),
        "hdus": [],
        "status": "parsed",
    }

    try:
        if path.lower().endswith(".fits.gz"):
            payload = gzip.decompress(data)
        else:
            payload = data

        with fits.open(io.BytesIO(payload), memmap=False) as hdul:
            for idx, hdu in enumerate(hdul):
                hdu_info = {
                    "index": idx,
                    "name": hdu.name,
                    "class": hdu.__class__.__name__,
                    "shape": None,
                    "columns": [],
                    "n_rows": None,
                }

                if getattr(hdu, "data", None) is not None:
                    try:
                        hdu_info["shape"] = list(hdu.data.shape)
                    except Exception:
                        pass

                if hasattr(hdu, "columns") and hdu.columns is not None:
                    cols = [str(c.name) for c in hdu.columns]
                    hdu_info["columns"] = cols

                    try:
                        hdu_info["n_rows"] = int(len(hdu.data))
                    except Exception:
                        pass

                    table_record = {
                        "path": path,
                        "raw_url": raw_url,
                        "local_path": str(local_path),
                        "hdu_index": idx,
                        "hdu_name": hdu.name,
                        "n_rows": hdu_info["n_rows"],
                        "n_columns": len(cols),
                        "columns": cols,
                        "source_type": "fits",
                    }

                    add_candidate_table(table_record)

                info["hdus"].append(hdu_info)

    except Exception as exc:
        info["status"] = "parse_failed"
        info["error"] = str(exc)

    REPORT["parsed_fits_files"].append(info)

    return info


def parse_text_file(data, path, raw_url):
    local_path = OUTDIR / "downloaded_sources" / path.replace("/", "__")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)

    text = text_from_bytes(data[:MAX_TEXT_PARSE_BYTES], path)
    df, parse_status = parse_text_table(text)

    structure = {
        "line_count": len(text.splitlines()),
        "first_lines": [line for line in text.splitlines() if line.strip()][:10],
        "keyword_hits": {
            key: text.lower().count(key)
            for key in KEYWORDS
            if text.lower().count(key) > 0
        },
    }

    record = {
        "path": path,
        "raw_url": raw_url,
        "local_path": str(local_path),
        "parse_status": parse_status,
        "text_structure": structure,
        "source_type": "text",
    }

    if df is not None:
        columns = [str(c) for c in df.columns.tolist()]

        record.update(
            {
                "n_rows_preview": int(df.shape[0]),
                "n_columns": int(df.shape[1]),
                "columns": columns,
            }
        )

        table_record = {
            "path": path,
            "raw_url": raw_url,
            "local_path": str(local_path),
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "columns": columns,
            "source_type": "text",
        }

        add_candidate_table(table_record)

    REPORT["parsed_text_tables"].append(record)

    return record


def download_and_parse(files):
    candidates = [
        item for item in files
        if item["score"] > 0
        and item["is_candidate_extension"]
        and item.get("size") is not None
        and item.get("size", 0) <= MAX_DOWNLOAD_BYTES
    ]

    candidates = sorted(candidates, key=lambda x: (-x["score"], x.get("size") or 0, x["path"]))[:MAX_DOWNLOADS]

    downloaded = []

    for item in candidates:
        path = item["path"]
        raw_url = item["raw_url"]
        suffix = item["extension"]

        entry = {
            "path": path,
            "raw_url": raw_url,
            "score": item["score"],
            "tags": item["tags"],
            "size": item["size"],
            "extension": suffix,
            "status": "not_started",
        }

        try:
            data = get_bytes(raw_url)
            entry["status"] = "downloaded"
            entry["downloaded_bytes"] = len(data)

            if suffix in [".npz", ".npy"]:
                entry["parser"] = "numpy"
                parse_npz(data, path, raw_url)
            elif suffix in [".fits", ".fit", ".fits.gz"]:
                entry["parser"] = "fits"
                parse_fits(data, path, raw_url)
            else:
                entry["parser"] = "text"
                parse_text_file(data, path, raw_url)

        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)

        downloaded.append(entry)

    REPORT["downloaded"] = downloaded

    return downloaded


def write_tables():
    with open(OUTDIR / "downloaded_candidate_files.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "status", "parser", "score", "tags", "size", "downloaded_bytes", "error", "raw_url"])

        for item in REPORT["downloaded"]:
            writer.writerow(
                [
                    item.get("path"),
                    item.get("status"),
                    item.get("parser"),
                    item.get("score"),
                    "|".join(item.get("tags", [])),
                    item.get("size"),
                    item.get("downloaded_bytes"),
                    item.get("error"),
                    item.get("raw_url"),
                ]
            )

    with open(OUTDIR / "candidate_ladder_tables.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "path",
                "source_type",
                "hdu_index",
                "hdu_name",
                "n_rows",
                "n_columns",
                "ladder_field_score",
                "ladder_field_coverage",
                "ladder_field_hits",
                "columns",
                "raw_url",
            ]
        )

        sorted_tables = sorted(
            REPORT["candidate_ladder_tables"],
            key=lambda r: (-r.get("ladder_field_score", 0), r.get("path", "")),
        )

        for rank, item in enumerate(sorted_tables, start=1):
            writer.writerow(
                [
                    rank,
                    item.get("path"),
                    item.get("source_type"),
                    item.get("hdu_index"),
                    item.get("hdu_name"),
                    item.get("n_rows"),
                    item.get("n_columns"),
                    item.get("ladder_field_score"),
                    json.dumps(item.get("ladder_field_coverage", {})),
                    json.dumps(item.get("ladder_field_hits", {})),
                    json.dumps(item.get("columns", [])),
                    item.get("raw_url"),
                ]
            )

    with open(OUTDIR / "fits_inventory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "status", "hdu_index", "hdu_name", "class", "shape", "n_rows", "columns", "raw_url"])

        for item in REPORT["parsed_fits_files"]:
            if not item.get("hdus"):
                writer.writerow([item.get("path"), item.get("status"), "", "", "", "", "", "", item.get("raw_url")])
                continue

            for hdu in item["hdus"]:
                writer.writerow(
                    [
                        item.get("path"),
                        item.get("status"),
                        hdu.get("index"),
                        hdu.get("name"),
                        hdu.get("class"),
                        json.dumps(hdu.get("shape")),
                        hdu.get("n_rows"),
                        json.dumps(hdu.get("columns")),
                        item.get("raw_url"),
                    ]
                )

    with open(OUTDIR / "npz_inventory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "status", "array_name", "shape", "dtype", "raw_url"])

        for item in REPORT["parsed_npz_files"]:
            if not item.get("arrays"):
                writer.writerow([item.get("path"), item.get("status"), "", "", "", item.get("raw_url")])
                continue

            for arr in item["arrays"]:
                writer.writerow(
                    [
                        item.get("path"),
                        item.get("status"),
                        arr.get("name"),
                        json.dumps(arr.get("shape")),
                        arr.get("dtype"),
                        item.get("raw_url"),
                    ]
                )


def compute_readiness():
    files = REPORT["files"]
    tables = REPORT["candidate_ladder_tables"]

    sh0es_tree_files = [
        f for f in files
        if "SH0ES_Data_candidate" in f.get("tags", [])
    ]

    fits_files = [
        f for f in files
        if "fits_candidate" in f.get("tags", [])
    ]

    cepheid_path_files = [
        f for f in files
        if "cepheid_candidate" in f.get("tags", [])
    ]

    anchor_path_files = [
        f for f in files
        if "anchor_candidate" in f.get("tags", [])
    ]

    best_tables = sorted(tables, key=lambda r: (-r.get("ladder_field_score", 0), r.get("path", "")))
    top_table = best_tables[0] if best_tables else None

    enough_tables = [
        t for t in tables
        if t.get("ladder_field_coverage", {}).get("host")
        and t.get("ladder_field_coverage", {}).get("period")
        and t.get("ladder_field_coverage", {}).get("magnitude")
        and t.get("ladder_field_coverage", {}).get("uncertainty")
    ]

    anchor_tables = [
        t for t in tables
        if t.get("ladder_field_coverage", {}).get("anchor")
        or t.get("ladder_field_coverage", {}).get("distance")
    ]

    score = 0
    blockers = []

    if sh0es_tree_files:
        score += 2
    else:
        blockers.append("No explicit SH0ES_Data or SH0ES_Data-like folder was found in the repository tree.")

    if fits_files:
        score += 1

    if cepheid_path_files:
        score += 2
    else:
        blockers.append("No strong Cepheid filename candidates were found.")

    if enough_tables:
        score += 4
    else:
        blockers.append("No parsed table clearly contains host + period + magnitude + uncertainty fields.")

    if anchor_path_files or anchor_tables:
        score += 2
    else:
        blockers.append("No strong anchor table was identified.")

    if len(anchor_tables) > 0 and len(enough_tables) > 0:
        score += 2

    if score >= 8:
        status = "enough_for_first_ladder_proxy"
    elif score >= 5:
        status = "partial_parser_needed"
    else:
        status = "insufficient_public_ladder_data"

    next_files = []

    for item in best_tables[:20]:
        if item.get("path") not in next_files:
            next_files.append(item.get("path"))

    for item in (cepheid_path_files + anchor_path_files + fits_files)[:20]:
        if item.get("path") not in next_files:
            next_files.append(item.get("path"))

    readiness = {
        "status": status,
        "readiness_score": score,
        "blockers": blockers,
        "counts": {
            "all_repository_files": len(files),
            "explicit_sh0es_data_folder_files": len(sh0es_tree_files),
            "fits_filename_candidates": len(fits_files),
            "cepheid_filename_candidates": len(cepheid_path_files),
            "anchor_filename_candidates": len(anchor_path_files),
            "downloaded_files": len(REPORT["downloaded"]),
            "parsed_text_files": len(REPORT["parsed_text_tables"]),
            "parsed_npz_files": len(REPORT["parsed_npz_files"]),
            "parsed_fits_files": len(REPORT["parsed_fits_files"]),
            "candidate_ladder_tables": len(tables),
            "tables_with_core_cepheid_fields": len(enough_tables),
            "tables_with_anchor_or_distance_fields": len(anchor_tables),
        },
        "top_candidate_table": top_table,
        "next_recommended_files": next_files,
    }

    REPORT["readiness"] = readiness

    return readiness


def write_summary(readiness):
    report_path = OUTDIR / "shoes_cepheid_anchor_parser_v1_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))

    with open(OUTDIR / "shoes_cepheid_anchor_parser_v1_summary.txt", "w") as f:
        f.write("TAIRID SH0ES Cepheid / anchor public-data parser v1\n\n")
        f.write("Boundary: public-data parser audit only. Not a Cepheid likelihood and not proof.\n\n")
        f.write(f"Repository: {OWNER}/{REPO}\n")
        f.write(f"Branch: {BRANCH}\n")
        f.write(f"Tree URL: {TREE_URL}\n\n")
        f.write("Readiness:\n")
        f.write(json.dumps(readiness, indent=2) + "\n\n")
        f.write("Interpretation guide:\n")
        f.write("- enough_for_first_ladder_proxy means the next test can attempt a simplified Cepheid-anchor likelihood proxy.\n")
        f.write("- partial_parser_needed means some source structure exists, but we need a more targeted parser before fitting.\n")
        f.write("- insufficient_public_ladder_data means we should not fake the Cepheid wall from supernova-only offsets.\n")
        f.write("- If the best result remains only supernova/Pantheon calibration files, the honest result is that public source-level Cepheid data are not accessible enough from this repository alone.\n")


def main():
    files = discover_files()
    download_and_parse(files)
    readiness = compute_readiness()
    write_tables()
    write_summary(readiness)

    print("")
    print("TAIRID SH0ES Cepheid / anchor parser v1 complete.")
    print("Created:")
    print("  shoes_cepheid_anchor_parser_v1_outputs/shoes_cepheid_anchor_parser_v1_report.json")
    print("  shoes_cepheid_anchor_parser_v1_outputs/shoes_cepheid_anchor_parser_v1_summary.txt")
    print("  shoes_cepheid_anchor_parser_v1_outputs/repository_candidate_inventory.csv")
    print("  shoes_cepheid_anchor_parser_v1_outputs/downloaded_candidate_files.csv")
    print("  shoes_cepheid_anchor_parser_v1_outputs/candidate_ladder_tables.csv")
    print("  shoes_cepheid_anchor_parser_v1_outputs/fits_inventory.csv")
    print("  shoes_cepheid_anchor_parser_v1_outputs/npz_inventory.csv")
    print("")
    print("Boundary:")
    print("  This is not a Cepheid likelihood.")
    print("  This checks whether public source-level ladder data exist for the next real wall.")


if __name__ == "__main__":
    main()
