#!/usr/bin/env python3
"""
TAIRID External Lane Candidate Inventory v1.1
Frozen-rule replay readiness audit.

Purpose:
v1.0 locked the current SH0ES Table2 lane model:

    Variable:
        F160W-like Table2 numeric column.

    Edge rule:
        within-host high 5% F160W minus within-host low 5% F160W.

    Frozen regime rule:
        clean high-alpha hosts
        low-alpha/reference hosts = LMC + SMC + N4536
        M31 sign-break quarantine = zero transferred correction

This v1.1 test does NOT tune that model.
It does NOT add hosts.
It does NOT add variables.
It does NOT rerun another in-sample Table2 tweak.

Instead it inventories adjacent / external data lanes and asks:

    Which lane is ready for a frozen-rule replay?
    Which lane needs a parser first?
    Which lane is too different and should not be treated as validation yet?

Boundary:
This does not prove TAIRID.
This does not prove H0 correction.
This does not prove new physics.
This only prepares the next independent validation lane.
"""

import csv
import json
import math
import re
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path("tairid_external_lane_inventory_v1_1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded_text_candidates"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO = "PantheonPlusSH0ES/DataRelease"
BRANCH_CANDIDATES = ["main", "master"]

FROZEN_RULE = {
    "source_status": "locked by v1.0",
    "dataset_lane": "SH0ES Table2 residual layer only",
    "variable": "F160W-like Table2 numeric column, table2_num_7",
    "edge_rule": "within-host high 5% F160W minus within-host low 5% F160W",
    "regimes": {
        "clean_high_alpha": "all active hosts except LMC, SMC, N4536, M31",
        "low_alpha_reference": ["LMC", "SMC", "N4536"],
        "sign_break_quarantine": ["M31"],
    },
    "replay_rule": "next lane must try frozen rule as-is before any tuning",
    "anti_ad_hoc_rules": [
        "Do not add a new host to a regime during v1.1.",
        "Do not search for a new best variable during v1.1.",
        "Do not claim validation from the same Table2 lane.",
        "Do not claim H0 correction or new physics from this inventory.",
    ],
}

CLAIMS_V1_1 = {
    "battery_name": "TAIRID External Lane Candidate Inventory v1.1",
    "scope": "Frozen-rule replay readiness audit",
    "reason_for_test": (
        "v1.0 locked the SH0ES Table2 frozen model. v1.1 looks for the next adjacent or independent lane "
        "where that frozen rule can be replayed without changing it."
    ),
    "primary_question": (
        "Is there an immediately replay-ready adjacent or independent dataset layer, or must the next step first build a parser?"
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only inventories candidate lanes for future independent validation."
    ),
}


TEXT_EXTENSIONS = {
    ".tex", ".txt", ".md", ".csv", ".dat", ".data", ".json", ".yaml", ".yml", ".ini", ".log"
}

BINARY_OR_LARGE_EXTENSIONS = {
    ".fits", ".fit", ".gz", ".zip", ".npy", ".npz", ".pkl", ".pickle"
}

KEYWORDS = {
    "f160w": ["f160w", "f160", "h-band", "h band"],
    "cepheid": ["cepheid", "cepheids", "period", "p-l", "leavitt"],
    "host": ["host", "galaxy", "ngc", "lmc", "smc", "m31", "n4258"],
    "residual": ["residual", "resid", "chi2", "scatter"],
    "covariance": ["cov", "covariance", "allc", "matrix"],
    "supernova": ["supernova", "snia", "sn ia", "sne", "pantheon"],
    "distance": ["distance", "modulus", "mu", "h0", "ladder", "calibrator"],
    "photometry": ["phot", "magnitude", "mag", "color", "colour", "filter"],
    "anchor": ["anchor", "lmc", "smc", "n4258", "maser", "parallax"],
}


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:180]


def fetch_json(url, timeout=30):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-v1.1-external-lane-inventory",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def fetch_text(url, timeout=30, max_bytes=2_000_000):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-v1.1-external-lane-inventory",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read(max_bytes + 1)

    truncated = len(data) > max_bytes
    data = data[:max_bytes]

    return data.decode("utf-8", errors="replace"), truncated, len(data)


def get_repo_tree():
    errors = []

    for branch in BRANCH_CANDIDATES:
        url = f"https://api.github.com/repos/{REPO}/git/trees/{branch}?recursive=1"

        try:
            data = fetch_json(url)
            tree = data.get("tree", [])

            if tree:
                return branch, tree, errors
        except Exception as exc:
            errors.append(
                {
                    "branch": branch,
                    "url": url,
                    "error": repr(exc),
                }
            )

    return None, [], errors


def raw_url(branch, path):
    return f"https://raw.githubusercontent.com/{REPO}/{branch}/{path}"


def is_text_candidate(path):
    lower = path.lower()
    suffix = Path(lower).suffix

    if suffix in TEXT_EXTENSIONS:
        return True

    if suffix in BINARY_OR_LARGE_EXTENSIONS:
        return False

    # Some data release files may have no extension.
    if any(token in lower for token in ["readme", "table", "data", "sh0es", "pantheon"]):
        return True

    return False


def count_row_like_lines(text):
    row_like = 0

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        numeric_count = len(re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", stripped))
        delimiter_count = stripped.count("&") + stripped.count(",") + stripped.count("|") + stripped.count("\t")

        if numeric_count >= 2 or delimiter_count >= 2:
            row_like += 1

    return row_like


def keyword_hits(text):
    lower = text.lower()
    hits = {}

    for group, words in KEYWORDS.items():
        found = []

        for word in words:
            if word in lower:
                found.append(word)

        hits[group] = found

    return hits


def score_candidate(path, text, truncated=False):
    lower_path = path.lower()
    hits = keyword_hits(text + "\n" + path)
    row_like = count_row_like_lines(text)

    has = {key: bool(value) for key, value in hits.items()}

    score = 0
    reasons = []

    if has["f160w"]:
        score += 5
        reasons.append("mentions F160W/H-band")

    if has["cepheid"]:
        score += 5
        reasons.append("mentions Cepheid/period/Leavitt lane")

    if has["host"]:
        score += 3
        reasons.append("contains host/galaxy cues")

    if has["residual"]:
        score += 4
        reasons.append("contains residual/chi2/scatter cues")

    if has["covariance"]:
        score += 4
        reasons.append("contains covariance/matrix cues")

    if has["supernova"]:
        score += 3
        reasons.append("contains supernova/Pantheon cues")

    if has["distance"]:
        score += 3
        reasons.append("contains distance-ladder/H0/modulus cues")

    if has["photometry"]:
        score += 3
        reasons.append("contains photometry/magnitude/filter cues")

    if has["anchor"]:
        score += 2
        reasons.append("contains anchor cues")

    if row_like >= 50:
        score += 3
        reasons.append("has many row-like lines")

    if row_like >= 200:
        score += 2
        reasons.append("has large row-like table structure")

    if "sh0es_data/table2" in lower_path:
        score -= 20
        reasons.append("blocked: this is the already-mined Table2 lane")

    if truncated:
        reasons.append("text was truncated for inventory")

    # Classification.
    already_used_table2 = "sh0es_data/table2" in lower_path

    if already_used_table2:
        lane_class = "blocked_already_used_table2_lane"
        replay_readiness = "do_not_use_for_external_validation"
    elif has["f160w"] and has["cepheid"] and has["host"] and row_like >= 50:
        lane_class = "adjacent_cepheid_f160w_candidate"
        replay_readiness = "parser_needed_then_frozen_rule_replay_possible"
    elif has["f160w"] and has["host"] and row_like >= 20:
        lane_class = "possible_f160w_host_candidate"
        replay_readiness = "manual_schema_review_needed"
    elif has["supernova"] and has["distance"] and row_like >= 50:
        lane_class = "independent_distance_ladder_or_supernova_candidate"
        replay_readiness = "external_lane_candidate_not_direct_f160w_replay"
    elif has["cepheid"] and has["photometry"] and row_like >= 20:
        lane_class = "cepheid_photometry_candidate_without_clear_f160w"
        replay_readiness = "parser_needed_variable_may_differ"
    elif has["covariance"] or has["residual"]:
        lane_class = "supporting_residual_or_covariance_candidate"
        replay_readiness = "support_file_candidate_not_primary_lane"
    else:
        lane_class = "low_relevance_for_frozen_replay"
        replay_readiness = "not_ready"

    return {
        "path": path,
        "score": score,
        "lane_class": lane_class,
        "replay_readiness": replay_readiness,
        "row_like_lines": row_like,
        "keyword_hits": hits,
        "reason_summary": "; ".join(reasons),
        "text_truncated": bool(truncated),
    }


def inventory_files(branch, tree):
    rows = []
    download_attempts = []

    candidate_paths = []

    for item in tree:
        path = item.get("path", "")
        kind = item.get("type", "")

        if kind != "blob":
            continue

        if is_text_candidate(path):
            candidate_paths.append(path)

    # Keep inventory bounded but broad.
    priority_paths = sorted(
        candidate_paths,
        key=lambda p: (
            0 if "sh0es" in p.lower() else 1,
            0 if "table" in p.lower() else 1,
            0 if any(x in p.lower() for x in ["cepheid", "pantheon", "data"]) else 1,
            p.lower(),
        ),
    )

    for path in priority_paths:
        url = raw_url(branch, path)
        local_path = DOWNLOAD_DIR / safe_name(path)

        try:
            text, truncated, byte_count = fetch_text(url)
            local_path.write_text(text, encoding="utf-8")
            scored = score_candidate(path, text, truncated=truncated)

            row = {
                **scored,
                "status": "downloaded_text",
                "branch": branch,
                "byte_count_inventory": byte_count,
                "raw_url": url,
                "local_inventory_path": str(local_path),
            }
            rows.append(row)
            download_attempts.append(
                {
                    "path": path,
                    "status": "downloaded_text",
                    "byte_count_inventory": byte_count,
                    "truncated": truncated,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "path": path,
                    "score": 0,
                    "lane_class": "download_failed",
                    "replay_readiness": "not_ready",
                    "row_like_lines": 0,
                    "keyword_hits": {},
                    "reason_summary": f"download failed: {repr(exc)}",
                    "text_truncated": False,
                    "status": "download_failed",
                    "branch": branch,
                    "raw_url": url,
                    "local_inventory_path": "",
                }
            )
            download_attempts.append(
                {
                    "path": path,
                    "status": "download_failed",
                    "error": repr(exc),
                }
            )

    rows = sorted(
        rows,
        key=lambda r: (
            -int(r.get("score", 0)),
            r.get("lane_class", ""),
            r.get("path", ""),
        ),
    )

    return rows, download_attempts


def inventory_binary_and_support_files(branch, tree):
    rows = []

    for item in tree:
        path = item.get("path", "")
        kind = item.get("type", "")
        size = item.get("size", None)

        if kind != "blob":
            continue

        suffix = Path(path.lower()).suffix
        lower = path.lower()

        if suffix in BINARY_OR_LARGE_EXTENSIONS or any(token in lower for token in ["cov", "allc", "alll", "ally"]):
            lane_class = "binary_or_matrix_support_file"

            if "cov" in lower or "allc" in lower:
                role = "possible covariance support"
            elif "alll" in lower:
                role = "possible design-matrix support"
            elif "ally" in lower:
                role = "possible residual/response-vector support"
            elif suffix in {".fits", ".fit"}:
                role = "possible FITS table support"
            else:
                role = "binary/large support file"

            rows.append(
                {
                    "path": path,
                    "branch": branch,
                    "size": size,
                    "suffix": suffix,
                    "lane_class": lane_class,
                    "possible_role": role,
                    "raw_url": raw_url(branch, path),
                }
            )

    return sorted(rows, key=lambda r: (r["lane_class"], r["path"]))


def summarize_candidates(rows, support_rows):
    top_adjacent = [
        row for row in rows
        if row["lane_class"] in {
            "adjacent_cepheid_f160w_candidate",
            "possible_f160w_host_candidate",
            "cepheid_photometry_candidate_without_clear_f160w",
        }
    ]

    top_independent = [
        row for row in rows
        if row["lane_class"] == "independent_distance_ladder_or_supernova_candidate"
    ]

    support = [
        row for row in rows
        if row["lane_class"] == "supporting_residual_or_covariance_candidate"
    ]

    blocked = [
        row for row in rows
        if row["lane_class"] == "blocked_already_used_table2_lane"
    ]

    binary_support = support_rows

    if top_adjacent:
        final_status = "external_lane_candidates_found_parser_needed"
        readiness_score = 8
        next_wall = "Candidate adjacent Cepheid/F160W files were found, but the next step must build a frozen-rule replay parser without tuning the model."
    elif top_independent:
        final_status = "independent_distance_ladder_candidates_found_not_direct_f160w_replay"
        readiness_score = 7
        next_wall = "Independent distance-ladder/SN candidates were found, but they are not direct F160W replay lanes."
    elif support or binary_support:
        final_status = "support_files_found_no_primary_external_lane_yet"
        readiness_score = 6
        next_wall = "Support files were found, but no clear primary external replay lane is ready."
    else:
        final_status = "no_external_lane_candidate_found_in_inventory"
        readiness_score = 5
        next_wall = "Repository inventory did not expose a replay-ready external lane. Manual dataset selection is needed."

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "counts": {
            "text_candidate_count": len(rows),
            "adjacent_candidate_count": len(top_adjacent),
            "independent_candidate_count": len(top_independent),
            "support_candidate_count": len(support),
            "binary_support_count": len(binary_support),
            "blocked_table2_count": len(blocked),
        },
        "top_adjacent_candidates": top_adjacent[:15],
        "top_independent_candidates": top_independent[:15],
        "top_support_candidates": support[:15],
        "binary_support_candidates": binary_support[:30],
        "blocked_table2_candidates": blocked[:10],
    }


def build_next_step_packet(summary):
    adjacent = summary.get("top_adjacent_candidates", [])
    independent = summary.get("top_independent_candidates", [])
    support = summary.get("top_support_candidates", [])

    if adjacent:
        recommended = adjacent[0]
        next_test = {
            "recommended_next_test": "v1.2 frozen-rule parser prototype for best adjacent candidate",
            "target_path": recommended["path"],
            "target_lane_class": recommended["lane_class"],
            "instruction": (
                "Build a parser for this candidate and attempt frozen F160W within-host edge replay as-is. "
                "Do not alter regimes or add variables."
            ),
        }
    elif independent:
        recommended = independent[0]
        next_test = {
            "recommended_next_test": "v1.2 independent-lane schema audit",
            "target_path": recommended["path"],
            "target_lane_class": recommended["lane_class"],
            "instruction": (
                "Audit whether a boundary-polarity idea can be translated to this independent lane. "
                "Do not claim direct frozen F160W validation unless the needed F160W/host/residual fields exist."
            ),
        }
    elif support:
        recommended = support[0]
        next_test = {
            "recommended_next_test": "v1.2 support-file map and parser feasibility audit",
            "target_path": recommended["path"],
            "target_lane_class": recommended["lane_class"],
            "instruction": (
                "Map support files to a primary dataset. This is not validation yet."
            ),
        }
    else:
        next_test = {
            "recommended_next_test": "manual external dataset selection",
            "target_path": None,
            "target_lane_class": None,
            "instruction": (
                "Choose an external Cepheid/F160W or distance-ladder dataset before writing the next replay test."
            ),
        }

    return {
        "frozen_rule_carry_forward": FROZEN_RULE,
        "next_test": next_test,
        "hard_boundaries_for_next_test": [
            "The frozen v1.0 rule cannot be changed during the first replay.",
            "No new host regime can be added during first replay.",
            "No new variable can be selected after looking at outcomes.",
            "If parser fields are missing, classify as not directly replayable instead of forcing the rule.",
        ],
    }


def make_plots(rows):
    try:
        if not rows:
            return

        top = rows[:20]
        labels = [row["path"][-45:] for row in top]
        scores = [int(row.get("score", 0)) for row in top]
        x = np.arange(len(top))

        plt.figure(figsize=(14, 5))
        plt.bar(x, scores)
        plt.xticks(x, labels, rotation=70, ha="right", fontsize=8)
        plt.ylabel("candidate score")
        plt.title("v1.1 external lane candidate inventory")
        plt.tight_layout()
        plt.savefig(OUTDIR / "external_lane_candidate_scores_v1_1.png", dpi=160)
        plt.close()

        class_counts = {}
        for row in rows:
            cls = row.get("lane_class", "unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1

        labels = list(class_counts.keys())
        counts = [class_counts[label] for label in labels]
        x = np.arange(len(labels))

        plt.figure(figsize=(12, 5))
        plt.bar(x, counts)
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel("file count")
        plt.title("v1.1 candidate classes")
        plt.tight_layout()
        plt.savefig(OUTDIR / "external_lane_candidate_classes_v1_1.png", dpi=160)
        plt.close()

    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v1_1.json",
            {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )


def main():
    print("TAIRID External Lane Candidate Inventory v1.1 starting.")
    print("Boundary: frozen-rule replay readiness only; no Table2 tuning.")

    write_json(OUTDIR / "claims_v1_1.json", CLAIMS_V1_1)
    write_json(OUTDIR / "frozen_rule_carry_forward_v1_1.json", FROZEN_RULE)

    try:
        branch, tree, tree_errors = get_repo_tree()

        if not branch or not tree:
            summary = {
                "test_name": "TAIRID External Lane Candidate Inventory v1.1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "final_status": "repo_tree_inventory_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Could not retrieve repository tree. Check GitHub access or repo path.",
                "repo": REPO,
                "tree_errors": tree_errors,
                "truth_boundary": CLAIMS_V1_1["truth_boundary"],
            }
            write_json(OUTDIR / "external_lane_inventory_v1_1_summary.json", summary)
            print("Repository tree inventory failed.")
            return

        tree_rows = [
            {
                "path": item.get("path", ""),
                "type": item.get("type", ""),
                "size": item.get("size", ""),
                "sha": item.get("sha", ""),
            }
            for item in tree
        ]

        write_csv(OUTDIR / "repo_tree_inventory_v1_1.csv", tree_rows)

        text_rows, download_attempts = inventory_files(branch, tree)
        support_rows = inventory_binary_and_support_files(branch, tree)

        write_csv(OUTDIR / "external_lane_candidate_inventory_v1_1.csv", text_rows)
        write_csv(OUTDIR / "external_lane_support_file_inventory_v1_1.csv", support_rows)
        write_json(OUTDIR / "download_attempts_v1_1.json", download_attempts)

        candidate_summary = summarize_candidates(text_rows, support_rows)
        next_packet = build_next_step_packet(candidate_summary)

        make_plots(text_rows)

        final_status = candidate_summary["final_status"]
        readiness_score = candidate_summary["readiness_score_0_to_10"]
        next_wall = candidate_summary["next_wall"]

        summary = {
            "test_name": "TAIRID External Lane Candidate Inventory v1.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Frozen-rule replay readiness audit only. "
                "No tuning, no new regimes, no H0 claim, no new-physics claim."
            ),
            "repo": REPO,
            "branch_used": branch,
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v1_1": CLAIMS_V1_1,
            "frozen_rule": FROZEN_RULE,
            "candidate_summary": candidate_summary,
            "next_step_packet": next_packet,
            "tree_errors": tree_errors,
            "output_files": {
                "summary_json": str(OUTDIR / "external_lane_inventory_v1_1_summary.json"),
                "summary_txt": str(OUTDIR / "external_lane_inventory_v1_1_summary.txt"),
                "candidate_inventory_csv": str(OUTDIR / "external_lane_candidate_inventory_v1_1.csv"),
                "support_file_inventory_csv": str(OUTDIR / "external_lane_support_file_inventory_v1_1.csv"),
                "repo_tree_csv": str(OUTDIR / "repo_tree_inventory_v1_1.csv"),
                "next_step_packet_json": str(OUTDIR / "next_step_packet_v1_1.json"),
                "frozen_rule_json": str(OUTDIR / "frozen_rule_carry_forward_v1_1.json"),
                "plots": [
                    str(OUTDIR / "external_lane_candidate_scores_v1_1.png"),
                    str(OUTDIR / "external_lane_candidate_classes_v1_1.png"),
                ],
            },
            "interpretation": {
                "what_success_means": (
                    "v1.1 found a candidate lane or support map for replaying the frozen rule without tuning."
                ),
                "what_failure_means": (
                    "No replay-ready candidate was visible from this repository inventory; choose a new external dataset manually."
                ),
                "truth_boundary": CLAIMS_V1_1["truth_boundary"],
            },
        }

        write_json(OUTDIR / "external_lane_inventory_v1_1_summary.json", summary)
        write_json(OUTDIR / "next_step_packet_v1_1.json", next_packet)

        with open(OUTDIR / "external_lane_inventory_v1_1_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID External Lane Candidate Inventory v1.1\n\n")
            f.write("Boundary: frozen-rule replay readiness only. No tuning. No new regimes. No H0 claim.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Frozen rule carried forward:\n")
            f.write(json.dumps(FROZEN_RULE, indent=2, default=json_default) + "\n\n")
            f.write("Candidate summary:\n")
            f.write(json.dumps(candidate_summary, indent=2, default=json_default) + "\n\n")
            f.write("Next step packet:\n")
            f.write(json.dumps(next_packet, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This only inventories candidate lanes for frozen-rule replay.\n")

        print("TAIRID External Lane Candidate Inventory v1.1 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID External Lane Candidate Inventory v1.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "external_lane_inventory_v1_1_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V1_1["truth_boundary"],
        }
        write_json(OUTDIR / "external_lane_inventory_v1_1_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
