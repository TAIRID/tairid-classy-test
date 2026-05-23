#!/usr/bin/env python3
"""
TAIRID SH0ES ladder source-data inventory v1.

Purpose:
The Pantheon ladder-offset gate v3 became offset-degenerate:
a calibrator-only gate could be almost perfectly absorbed by calibration-offset
freedom in a supernova-only diagonal screen.

That means the next real wall is not another supernova-only gate scan.
The next wall is source-level calibration-ladder data:
Cepheids, anchors, hosts, calibrators, Hubble-flow links, design matrices,
covariances, and likelihood files.

This script inventories the public PantheonPlusSH0ES DataRelease repository,
especially SH0ES_Data, and asks:

1. What public SH0ES calibration-ladder files are available?
2. Which files appear to contain Cepheid / anchor / host / calibration data?
3. Which files appear to contain covariance or design-matrix structure?
4. Is there enough public structure to build the next likelihood test?
5. Which exact files should the next test download and parse?

Boundary:
This is not a cosmology likelihood.
This is not a Cepheid likelihood.
This is not a Planck, BAO, or Pantheon likelihood.
This does not prove TAIRID cosmology.
It is a source-data inventory and next-test readiness audit.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests


OUTDIR = Path("shoes_ladder_source_inventory_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

OWNER = "PantheonPlusSH0ES"
REPO = "DataRelease"
BRANCH = "main"

TREE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/"

KEYWORDS = [
    "cepheid",
    "ceph",
    "calib",
    "calibrator",
    "anchor",
    "host",
    "shoes",
    "sh0es",
    "hst",
    "lmc",
    "ngc",
    "4258",
    "mw",
    "milky",
    "parallax",
    "gaia",
    "cov",
    "covar",
    "matrix",
    "design",
    "likelihood",
    "cosmosis",
    "pantheon",
    "sn",
    "supernova",
    "h0",
]

DATA_EXTENSIONS = {
    ".txt",
    ".dat",
    ".csv",
    ".tsv",
    ".cov",
    ".dataset",
    ".ini",
    ".yaml",
    ".yml",
    ".py",
    ".ipynb",
    ".md",
    ".fits",
    ".fitres",
}

TEXT_EXTENSIONS = {
    ".txt",
    ".dat",
    ".csv",
    ".tsv",
    ".cov",
    ".dataset",
    ".ini",
    ".yaml",
    ".yml",
    ".py",
    ".md",
    ".fitres",
}

COLUMN_KEYWORDS = [
    "cepheid",
    "ceph",
    "host",
    "anchor",
    "calib",
    "calibrator",
    "period",
    "metal",
    "feh",
    "mag",
    "magnitude",
    "mu",
    "dist",
    "parallax",
    "gaia",
    "lmc",
    "ngc",
    "4258",
    "sn",
    "z",
    "h0",
    "is_calibrator",
    "used_in_shoes_hf",
]

REPORT = {
    "boundary": "Source-data inventory only. Not a likelihood and not proof.",
    "repository": f"{OWNER}/{REPO}",
    "branch": BRANCH,
    "tree_url": TREE_URL,
    "raw_base": RAW_BASE,
    "files": [],
    "downloaded_files": [],
    "parsed_tables": [],
    "readiness": {},
}


def safe_get_json(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def safe_get_text(url, max_bytes=5_000_000):
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    content = response.content

    if len(content) > max_bytes:
        content = content[:max_bytes]

    return content.decode("utf-8", errors="replace"), len(response.content)


def path_score(path):
    lower = path.lower()
    score = 0

    for keyword in KEYWORDS:
        if keyword in lower:
            score += 1

    if lower.startswith("shoes_data/"):
        score += 5

    if lower.startswith("pantheon+_data/4_distances_and_covar/"):
        score += 2

    if lower.startswith("cosmology/"):
        score += 1

    return score


def classify_path(path):
    lower = path.lower()
    tags = []

    if lower.startswith("shoes_data/"):
        tags.append("SH0ES_Data")

    if lower.startswith("pantheon+_data/"):
        tags.append("PantheonPlus_Data")

    if lower.startswith("cosmology/"):
        tags.append("Cosmology")

    if "cepheid" in lower or "ceph" in lower:
        tags.append("cepheid_candidate")

    if "calib" in lower or "calibrator" in lower:
        tags.append("calibrator_candidate")

    if "anchor" in lower or "lmc" in lower or "4258" in lower or "ngc" in lower or "mw" in lower or "gaia" in lower:
        tags.append("anchor_candidate")

    if "cov" in lower or "covar" in lower:
        tags.append("covariance_candidate")

    if "matrix" in lower or "design" in lower:
        tags.append("matrix_candidate")

    if "likelihood" in lower or "cosmosis" in lower:
        tags.append("likelihood_candidate")

    if "pantheon" in lower or "sn" in lower or "supernova" in lower:
        tags.append("supernova_candidate")

    return tags


def discover_repository_files():
    tree = safe_get_json(TREE_URL)
    items = tree.get("tree", [])

    files = []

    for item in items:
        if item.get("type") != "blob":
            continue

        path = item.get("path", "")
        suffix = Path(path).suffix.lower()

        file_info = {
            "path": path,
            "size": item.get("size"),
            "url": item.get("url"),
            "raw_url": RAW_BASE + path,
            "extension": suffix,
            "score": path_score(path),
            "tags": classify_path(path),
            "is_data_extension": suffix in DATA_EXTENSIONS,
            "is_text_extension": suffix in TEXT_EXTENSIONS,
        }

        files.append(file_info)

    files_sorted = sorted(files, key=lambda x: (-x["score"], x["path"]))

    REPORT["files"] = files_sorted

    return files_sorted


def save_file_inventory(files):
    with open(OUTDIR / "repository_file_inventory.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "rank",
                "path",
                "size",
                "extension",
                "score",
                "tags",
                "raw_url",
                "is_data_extension",
                "is_text_extension",
            ]
        )

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
                    item["is_data_extension"],
                    item["is_text_extension"],
                ]
            )


def detect_delimiter(sample):
    if "\t" in sample:
        return "\t"

    comma_count = sample.count(",")
    space_count = len(re.findall(r" +", sample))

    if comma_count > 3:
        return ","

    return r"\s+"


def read_table_from_text(text):
    lines = [line for line in text.splitlines() if line.strip()]

    if not lines:
        return None, "empty"

    # Try to locate a header. Many astronomy data files use # header.
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
        table_text = header_line + "\n" + "\n".join(data_lines[:5000])
    else:
        table_text = "\n".join(data_lines[:5000])

    delimiter = detect_delimiter(table_text[:2000])

    try:
        from io import StringIO

        if delimiter == r"\s+":
            df = pd.read_csv(StringIO(table_text), sep=r"\s+", engine="python")
        else:
            df = pd.read_csv(StringIO(table_text), sep=delimiter)

        if df.shape[0] == 0 or df.shape[1] == 0:
            return None, "empty_dataframe"

        return df, "parsed"

    except Exception as exc:
        return None, f"parse_failed: {exc}"


def scan_text_for_structure(text):
    lower = text.lower()

    keyword_hits = {
        keyword: lower.count(keyword)
        for keyword in KEYWORDS
        if lower.count(keyword) > 0
    }

    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]

    first_lines = nonempty[:12]

    return {
        "line_count_scanned": len(lines),
        "nonempty_line_count_scanned": len(nonempty),
        "keyword_hits": keyword_hits,
        "first_nonempty_lines": first_lines,
    }


def column_hits(columns):
    hits = []

    for col in columns:
        low = str(col).lower()

        for keyword in COLUMN_KEYWORDS:
            if keyword in low:
                hits.append({"column": str(col), "keyword": keyword})

    return hits


def download_and_parse_candidates(files):
    candidates = [
        item for item in files
        if item["score"] > 0 and item["is_text_extension"] and item.get("size", 0) is not None
    ]

    # Keep the run bounded. Prioritize SH0ES_Data and high-scoring files.
    candidates = sorted(candidates, key=lambda x: (-x["score"], x.get("size") or 0, x["path"]))[:80]

    downloaded = []
    parsed_tables = []

    for item in candidates:
        path = item["path"]
        raw_url = item["raw_url"]

        try:
            text, full_size = safe_get_text(raw_url)
        except Exception as exc:
            downloaded.append(
                {
                    "path": path,
                    "raw_url": raw_url,
                    "status": "download_failed",
                    "error": str(exc),
                    "score": item["score"],
                    "tags": item["tags"],
                }
            )
            continue

        local_name = path.replace("/", "__")
        local_path = OUTDIR / "downloaded_sources" / local_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(text, errors="replace")

        structure = scan_text_for_structure(text)
        df, parse_status = read_table_from_text(text)

        record = {
            "path": path,
            "raw_url": raw_url,
            "local_path": str(local_path),
            "status": "downloaded",
            "full_size_bytes": full_size,
            "score": item["score"],
            "tags": item["tags"],
            "text_structure": structure,
            "parse_status": parse_status,
        }

        if df is not None:
            cols = [str(c) for c in df.columns.tolist()]
            hits = column_hits(cols)

            table_record = {
                "path": path,
                "raw_url": raw_url,
                "local_path": str(local_path),
                "shape_rows_preview": int(df.shape[0]),
                "shape_columns": int(df.shape[1]),
                "columns": cols,
                "column_hits": hits,
                "tags": item["tags"],
                "score": item["score"],
                "parse_status": parse_status,
            }

            parsed_tables.append(table_record)

            record["table_shape_rows_preview"] = int(df.shape[0])
            record["table_shape_columns"] = int(df.shape[1])
            record["table_columns"] = cols
            record["column_hits"] = hits

        downloaded.append(record)

    REPORT["downloaded_files"] = downloaded
    REPORT["parsed_tables"] = parsed_tables

    return downloaded, parsed_tables


def write_download_report(downloaded, parsed_tables):
    with open(OUTDIR / "downloaded_candidate_files.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "path",
                "status",
                "score",
                "tags",
                "parse_status",
                "full_size_bytes",
                "table_rows_preview",
                "table_columns_count",
                "column_hits",
                "raw_url",
            ]
        )

        for item in downloaded:
            writer.writerow(
                [
                    item.get("path"),
                    item.get("status"),
                    item.get("score"),
                    "|".join(item.get("tags", [])),
                    item.get("parse_status"),
                    item.get("full_size_bytes"),
                    item.get("table_shape_rows_preview"),
                    item.get("table_shape_columns"),
                    json.dumps(item.get("column_hits", [])),
                    item.get("raw_url"),
                ]
            )

    with open(OUTDIR / "parsed_table_inventory.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "path",
                "score",
                "tags",
                "rows_preview",
                "columns_count",
                "columns",
                "column_hits",
                "raw_url",
            ]
        )

        for item in parsed_tables:
            writer.writerow(
                [
                    item.get("path"),
                    item.get("score"),
                    "|".join(item.get("tags", [])),
                    item.get("shape_rows_preview"),
                    item.get("shape_columns"),
                    json.dumps(item.get("columns", [])),
                    json.dumps(item.get("column_hits", [])),
                    item.get("raw_url"),
                ]
            )


def compute_readiness(files, downloaded, parsed_tables):
    shoes_files = [f for f in files if "SH0ES_Data" in f.get("tags", [])]
    cepheid_files = [f for f in files if "cepheid_candidate" in f.get("tags", [])]
    anchor_files = [f for f in files if "anchor_candidate" in f.get("tags", [])]
    covariance_files = [f for f in files if "covariance_candidate" in f.get("tags", [])]
    matrix_files = [f for f in files if "matrix_candidate" in f.get("tags", [])]
    likelihood_files = [f for f in files if "likelihood_candidate" in f.get("tags", [])]

    parsed_with_cepheid_columns = []
    parsed_with_host_anchor_columns = []
    parsed_with_matrix_like_columns = []

    for table in parsed_tables:
        hits = table.get("column_hits", [])
        hit_keywords = {hit["keyword"] for hit in hits}

        if {"cepheid"} & hit_keywords or {"ceph"} & hit_keywords:
            parsed_with_cepheid_columns.append(table)

        if {"host", "anchor", "calib", "calibrator", "lmc", "ngc", "4258", "parallax", "gaia"} & hit_keywords:
            parsed_with_host_anchor_columns.append(table)

        if {"mu", "dist", "mag", "period", "metal"} & hit_keywords:
            parsed_with_matrix_like_columns.append(table)

    readiness_score = 0
    blockers = []
    next_recommended_files = []

    if shoes_files:
        readiness_score += 2
    else:
        blockers.append("No SH0ES_Data files found in repository tree.")

    if cepheid_files or parsed_with_cepheid_columns:
        readiness_score += 2
    else:
        blockers.append("No obvious Cepheid source table found by filename or parsed columns.")

    if anchor_files or parsed_with_host_anchor_columns:
        readiness_score += 2
    else:
        blockers.append("No obvious anchor/host/calibrator table found by filename or parsed columns.")

    if covariance_files:
        readiness_score += 1
    else:
        blockers.append("No obvious covariance file found for the calibration ladder.")

    if matrix_files:
        readiness_score += 1

    if likelihood_files:
        readiness_score += 1

    for group in [
        parsed_with_cepheid_columns,
        parsed_with_host_anchor_columns,
        covariance_files,
        matrix_files,
        likelihood_files,
    ]:
        for item in group:
            path = item["path"] if isinstance(item, dict) else item["path"]

            if path not in next_recommended_files:
                next_recommended_files.append(path)

    if readiness_score >= 6:
        status = "enough_public_structure_for_next_likelihood_attempt"
    elif readiness_score >= 4:
        status = "partial_public_structure_needs_targeted_parser"
    else:
        status = "inventory_found_insufficient_structure_for_likelihood"

    readiness = {
        "status": status,
        "readiness_score": readiness_score,
        "blockers": blockers,
        "counts": {
            "all_files": len(files),
            "shoes_files": len(shoes_files),
            "cepheid_filename_candidates": len(cepheid_files),
            "anchor_filename_candidates": len(anchor_files),
            "covariance_filename_candidates": len(covariance_files),
            "matrix_filename_candidates": len(matrix_files),
            "likelihood_filename_candidates": len(likelihood_files),
            "downloaded_candidate_files": len(downloaded),
            "parsed_tables": len(parsed_tables),
            "parsed_with_cepheid_columns": len(parsed_with_cepheid_columns),
            "parsed_with_host_anchor_columns": len(parsed_with_host_anchor_columns),
            "parsed_with_matrix_like_columns": len(parsed_with_matrix_like_columns),
        },
        "next_recommended_files": next_recommended_files[:30],
    }

    REPORT["readiness"] = readiness

    return readiness


def write_summary_text(readiness):
    with open(OUTDIR / "shoes_ladder_source_inventory_v1_summary.txt", "w") as f:
        f.write("TAIRID SH0ES ladder source-data inventory v1\n\n")
        f.write("Boundary: source-data inventory only. Not a likelihood and not proof.\n\n")
        f.write(f"Repository: {OWNER}/{REPO}\n")
        f.write(f"Branch: {BRANCH}\n")
        f.write(f"Tree URL: {TREE_URL}\n\n")

        f.write("Readiness:\n")
        f.write(json.dumps(readiness, indent=2) + "\n\n")

        f.write("Interpretation guide:\n")
        f.write("- If enough public structure is found, the next test should parse the recommended Cepheid/anchor/covariance files.\n")
        f.write("- If only partial structure is found, the next test should be a targeted parser for the specific files discovered here.\n")
        f.write("- If insufficient structure is found, we should not fake a Cepheid likelihood from supernova-only offsets.\n")
        f.write("- Offset-degenerate supernova results require calibration-ladder data before stronger claims.\n")


def main():
    files = discover_repository_files()
    save_file_inventory(files)

    downloaded, parsed_tables = download_and_parse_candidates(files)
    write_download_report(downloaded, parsed_tables)

    readiness = compute_readiness(files, downloaded, parsed_tables)

    report_path = OUTDIR / "shoes_ladder_source_inventory_v1_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))

    write_summary_text(readiness)

    print("")
    print("TAIRID SH0ES ladder source-data inventory v1 complete.")
    print("Created:")
    print("  shoes_ladder_source_inventory_v1_outputs/shoes_ladder_source_inventory_v1_report.json")
    print("  shoes_ladder_source_inventory_v1_outputs/shoes_ladder_source_inventory_v1_summary.txt")
    print("  shoes_ladder_source_inventory_v1_outputs/repository_file_inventory.csv")
    print("  shoes_ladder_source_inventory_v1_outputs/downloaded_candidate_files.csv")
    print("  shoes_ladder_source_inventory_v1_outputs/parsed_table_inventory.csv")
    print("")
    print("Boundary:")
    print("  This is not a likelihood.")
    print("  This inventories public SH0ES/Pantheon+ source files for the next calibration-ladder test.")


if __name__ == "__main__":
    main()
