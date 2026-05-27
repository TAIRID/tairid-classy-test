#!/usr/bin/env python3
"""
TAIRID NIR Residual Layer Locator v1.3

Purpose:
v1.2 showed that the SH0ES R22 NIR Cepheid files can be parsed and can support
a frozen H/F160W-like within-host edge surface construction.

But v1.2 correctly stopped before validation because the NIR .out files do not
directly contain a residual/covariance/outcome layer.

This v1.3 test asks one narrow question:

    Does the PantheonPlusSH0ES/DataRelease repository contain a legitimate
    NIR-linked residual, outcome, covariance, fit, or model layer that can be
    connected to the NIR Cepheid H-band surface without inventing residuals?

This test does NOT validate TAIRID.
This test does NOT tune the frozen v1.0 rule.
This test does NOT create residuals from magnitudes.
This test does NOT claim H0 correction or new physics.

It only searches the release tree, classifies candidate files, and decides
whether the NIR lane is ready for a later row-alignment proof.

Truth boundary:
If no legitimate residual/outcome layer is found, the NIR lane must be classified
as surface-only context, not validation-ready.
"""

import csv
import json
import math
import re
import traceback
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


OUTDIR = Path("tairid_nir_residual_layer_locator_v1_3_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "candidate_downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPO = "PantheonPlusSH0ES/DataRelease"
BRANCH_CANDIDATES = ["main", "master"]

TARGET_NIR_FILES = [
    "SH0ES_Data/R22_orig19_NIR.out",
    "SH0ES_Data/R22_orig19_NIR.wm31.out",
]

FROZEN_RULE_CARRIED_FORWARD = {
    "source_status": "locked by v1.0 and schema-checked by v1.2",
    "locked_table2_lane": "SH0ES Table2 residual layer only",
    "frozen_variable": "F160W-like Table2 numeric column, table2_num_7",
    "external_proxy_candidate": "H column in R22 NIR files, treated only as H/F160W-like surface candidate",
    "edge_rule": "within-host high 5% H/F160W-like magnitude minus within-host low 5% H/F160W-like magnitude",
    "low_alpha_reference_hosts": ["LMC", "SMC", "N4536"],
    "sign_break_quarantine_hosts": ["M31"],
    "hard_boundary": [
        "Do not tune host regimes in v1.3.",
        "Do not invent residuals.",
        "Do not claim validation from file discovery.",
        "Do not treat a global covariance file as row-aligned NIR validation unless a later row-alignment proof succeeds.",
        "Do not transfer the Table2 result into NIR files unless a later test defines a valid outcome/residual layer.",
    ],
}

CLAIMS_V1_3 = {
    "battery_name": "TAIRID NIR Residual Layer Locator v1.3",
    "scope": "Repository-level search for NIR-linked residual/outcome/covariance layers",
    "primary_question": (
        "Does the public PantheonPlusSH0ES/DataRelease tree contain a legitimate NIR-linked residual, "
        "outcome, covariance, fit, or model layer usable for a later frozen-rule replay?"
    ),
    "truth_boundary": (
        "This is discovery and classification only. It does not validate TAIRID, H0 correction, or new physics."
    ),
}

SEARCH_TERMS_STRONG = [
    "nir",
    "cepheid",
    "f160w",
    "hband",
    "h_band",
    "h-band",
    "r22_orig19_nir",
]

OUTCOME_TERMS = [
    "resid",
    "residual",
    "cov",
    "covariance",
    "invcov",
    "inverse",
    "fit",
    "fitres",
    "model",
    "mu",
    "distance",
    "dist",
    "mag",
    "table",
    "lcparam",
    "calib",
    "calibration",
]

CONTEXT_TERMS = [
    "sh0es",
    "shoes",
    "pantheon",
    "datarelease",
    "cepheid",
    "host",
    "f160w",
    "nir",
]

LIKELY_TEXT_EXTENSIONS = {
    ".txt", ".dat", ".out", ".csv", ".tsv", ".json", ".md", ".README", ".readme",
    ".tex", ".ini", ".yaml", ".yml", ".log", ".fitres", ".cov", ".invcov"
}

MAX_DOWNLOAD_BYTES = 8_000_000


def json_default(obj):
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


def github_api_url(branch):
    return f"https://api.github.com/repos/{REPO}/git/trees/{branch}?recursive=1"


def raw_url(branch, repo_path):
    return f"https://raw.githubusercontent.com/{REPO}/{branch}/{repo_path}"


def fetch_json(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TAIRID-v1.3-nir-residual-layer-locator"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def fetch_text(url, max_bytes=MAX_DOWNLOAD_BYTES):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TAIRID-v1.3-nir-residual-layer-locator"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read(max_bytes + 1)

    if len(data) > max_bytes:
        return {
            "status": "too_large",
            "text": "",
            "bytes": len(data),
        }

    return {
        "status": "downloaded",
        "text": data.decode("utf-8", errors="replace"),
        "bytes": len(data),
    }


def fetch_repo_tree():
    errors = []

    for branch in BRANCH_CANDIDATES:
        url = github_api_url(branch)
        try:
            payload = fetch_json(url)
            tree = payload.get("tree", [])
            rows = []
            for item in tree:
                if item.get("type") != "blob":
                    continue
                rows.append(
                    {
                        "branch": branch,
                        "path": item.get("path", ""),
                        "size": item.get("size", None),
                        "sha": item.get("sha", ""),
                        "url": raw_url(branch, item.get("path", "")),
                    }
                )
            return {
                "status": "downloaded",
                "branch": branch,
                "rows": rows,
                "errors": errors,
            }
        except Exception as exc:
            errors.append({"branch": branch, "url": url, "error": repr(exc)})

    return {
        "status": "failed",
        "branch": None,
        "rows": [],
        "errors": errors,
    }


def path_score(path):
    p = path.lower()

    strong_hits = [term for term in SEARCH_TERMS_STRONG if term in p]
    outcome_hits = [term for term in OUTCOME_TERMS if term in p]
    context_hits = [term for term in CONTEXT_TERMS if term in p]

    score = 0
    score += 10 * len(strong_hits)
    score += 5 * len(outcome_hits)
    score += 2 * len(context_hits)

    if p.startswith("sh0es_data/"):
        score += 8

    if "nir" in p and ("resid" in p or "cov" in p or "fit" in p or "model" in p):
        score += 25

    if "cepheid" in p and ("resid" in p or "cov" in p or "fit" in p or "model" in p):
        score += 20

    if any(target.lower() == p for target in TARGET_NIR_FILES):
        score += 30

    return {
        "path_score": score,
        "strong_hits": strong_hits,
        "outcome_hits": outcome_hits,
        "context_hits": context_hits,
    }


def extension_of(path):
    name = Path(path).name
    suffix = Path(name).suffix
    return suffix if suffix else ""


def looks_text_like(path):
    ext = extension_of(path).lower()
    if ext in LIKELY_TEXT_EXTENSIONS:
        return True
    if path.lower().endswith(".out"):
        return True
    if path.lower().endswith(".fitres"):
        return True
    if path.lower().endswith(".cov"):
        return True
    if path.lower().endswith(".invcov"):
        return True
    return False


def sniff_content(text):
    lower = text.lower()
    first_lines = "\n".join(text.splitlines()[:25])

    content_hits = {
        "mentions_nir": "nir" in lower,
        "mentions_cepheid": "cepheid" in lower or "cepheids" in lower,
        "mentions_f160w": "f160w" in lower,
        "mentions_residual": "resid" in lower or "residual" in lower,
        "mentions_covariance": "cov" in lower or "covariance" in lower or "invcov" in lower,
        "mentions_host": "host" in lower,
        "mentions_h0": "h0" in lower,
        "mentions_distance": "distance" in lower or "dist" in lower or "mu" in lower,
        "mentions_fit": "fit" in lower or "model" in lower,
    }

    lines = [line for line in text.splitlines() if line.strip()]
    token_counts = []
    numericish_lines = 0

    for line in lines[:300]:
        tokens = line.strip().split()
        token_counts.append(len(tokens))
        numeric_count = 0
        for tok in tokens:
            try:
                float(tok)
                numeric_count += 1
            except Exception:
                pass
        if tokens and numeric_count / max(len(tokens), 1) >= 0.5:
            numericish_lines += 1

    return {
        "first_lines": first_lines[:2000],
        "line_count": len(lines),
        "median_token_count_first_300": median(token_counts) if token_counts else None,
        "numericish_lines_first_300": numericish_lines,
        "content_hits": content_hits,
    }


def median(values):
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def classify_candidate(row, sniff=None):
    path = row["path"]
    p = path.lower()
    score_info = path_score(path)

    classification = "context_or_unrelated"
    confidence = 0
    reason = []

    if any(target.lower() == p for target in TARGET_NIR_FILES):
        classification = "known_nir_surface_file"
        confidence = 95
        reason.append("This is one of the v1.2 NIR surface files, not a residual/outcome layer.")

    elif "nir" in p and ("resid" in p or "residual" in p):
        classification = "strong_nir_residual_candidate"
        confidence = 90
        reason.append("Path contains NIR and residual language.")

    elif "nir" in p and ("cov" in p or "covariance" in p or "invcov" in p):
        classification = "strong_nir_covariance_candidate"
        confidence = 88
        reason.append("Path contains NIR and covariance language.")

    elif "cepheid" in p and ("resid" in p or "residual" in p):
        classification = "cepheid_residual_candidate"
        confidence = 82
        reason.append("Path contains Cepheid and residual language.")

    elif "cepheid" in p and ("cov" in p or "covariance" in p or "invcov" in p):
        classification = "cepheid_covariance_candidate"
        confidence = 80
        reason.append("Path contains Cepheid and covariance language.")

    elif "sh0es_data" in p and ("fit" in p or "model" in p or "calib" in p or "mu" in p or "distance" in p):
        classification = "possible_sh0es_outcome_or_model_candidate"
        confidence = 65
        reason.append("Path is inside SH0ES_Data and contains model/outcome language.")

    elif "pantheon" in p and ("cov" in p or "fitres" in p or "lcparam" in p):
        classification = "global_pantheon_context_not_nir_row_aligned"
        confidence = 60
        reason.append("Likely useful global context, but not NIR Cepheid row-aligned by path alone.")

    elif score_info["path_score"] >= 25:
        classification = "weak_candidate_needs_manual_review"
        confidence = 45
        reason.append("Path has several relevant search terms but no direct NIR residual/covariance signal.")

    if sniff:
        hits = sniff.get("content_hits", {})
        if hits.get("mentions_nir") and hits.get("mentions_residual"):
            classification = "strong_content_nir_residual_candidate"
            confidence = max(confidence, 90)
            reason.append("Downloaded text mentions both NIR and residual.")
        elif hits.get("mentions_nir") and hits.get("mentions_covariance"):
            classification = "strong_content_nir_covariance_candidate"
            confidence = max(confidence, 88)
            reason.append("Downloaded text mentions both NIR and covariance.")
        elif hits.get("mentions_cepheid") and hits.get("mentions_residual"):
            classification = "content_cepheid_residual_candidate"
            confidence = max(confidence, 82)
            reason.append("Downloaded text mentions both Cepheid and residual.")
        elif hits.get("mentions_cepheid") and hits.get("mentions_covariance"):
            classification = "content_cepheid_covariance_candidate"
            confidence = max(confidence, 80)
            reason.append("Downloaded text mentions both Cepheid and covariance.")

    return {
        "classification": classification,
        "confidence_0_to_100": confidence,
        "classification_reason": " ".join(reason) if reason else "No direct residual/outcome signal found.",
        **score_info,
    }


def build_candidate_inventory(tree_rows):
    inventory = []

    for row in tree_rows:
        info = path_score(row["path"])
        p = row["path"].lower()

        include = False
        include_reason = []

        if info["path_score"] >= 10:
            include = True
            include_reason.append("path_score>=10")

        if p.startswith("sh0es_data/"):
            include = True
            include_reason.append("inside_SH0ES_Data")

        if any(term in p for term in ["resid", "residual", "cov", "covariance", "invcov", "fitres", "lcparam"]):
            include = True
            include_reason.append("outcome_or_covariance_term")

        if any(target.lower() == p for target in TARGET_NIR_FILES):
            include = True
            include_reason.append("known_v1_2_NIR_surface_file")

        if include:
            out = dict(row)
            out.update(info)
            out["include_reason"] = ",".join(include_reason)
            out["extension"] = extension_of(row["path"])
            out["looks_text_like"] = looks_text_like(row["path"])
            inventory.append(out)

    return sorted(inventory, key=lambda r: (-r["path_score"], r["path"]))


def download_and_classify_candidates(branch, inventory):
    classified = []
    downloaded_sniffs = []

    for row in inventory:
        size = row.get("size")
        should_download = (
            row.get("looks_text_like")
            and (size is None or size <= MAX_DOWNLOAD_BYTES)
            and (
                row["path_score"] >= 15
                or row["path"].lower().startswith("sh0es_data/")
                or any(term in row["path"].lower() for term in ["resid", "residual", "cov", "covariance", "invcov", "fitres"])
            )
        )

        sniff = None
        download_status = "not_downloaded"

        if should_download:
            url = raw_url(branch, row["path"])
            try:
                fetched = fetch_text(url)
                download_status = fetched["status"]

                if fetched["status"] == "downloaded":
                    local_path = DOWNLOAD_DIR / safe_name(row["path"])
                    local_path.write_text(fetched["text"], encoding="utf-8")
                    sniff = sniff_content(fetched["text"])
                    downloaded_sniffs.append(
                        {
                            "path": row["path"],
                            "url": url,
                            "bytes": fetched["bytes"],
                            "local_path": str(local_path),
                            "sniff": sniff,
                        }
                    )
                else:
                    downloaded_sniffs.append(
                        {
                            "path": row["path"],
                            "url": url,
                            "bytes": fetched["bytes"],
                            "local_path": "",
                            "sniff": None,
                            "download_status": fetched["status"],
                        }
                    )
            except Exception as exc:
                download_status = "download_failed"
                downloaded_sniffs.append(
                    {
                        "path": row["path"],
                        "url": url,
                        "bytes": None,
                        "local_path": "",
                        "sniff": None,
                        "download_status": "download_failed",
                        "error": repr(exc),
                    }
                )

        classification = classify_candidate(row, sniff=sniff)
        out = dict(row)
        out.update(classification)
        out["download_status"] = download_status
        if sniff:
            out["content_hits"] = json.dumps(sniff.get("content_hits", {}), default=json_default)
            out["line_count"] = sniff.get("line_count")
            out["numericish_lines_first_300"] = sniff.get("numericish_lines_first_300")
        else:
            out["content_hits"] = "{}"
            out["line_count"] = None
            out["numericish_lines_first_300"] = None

        classified.append(out)

    classified = sorted(
        classified,
        key=lambda r: (-r["confidence_0_to_100"], -r["path_score"], r["path"]),
    )

    return classified, downloaded_sniffs


def decide_final_status(classified):
    strong = [
        r for r in classified
        if r["classification"] in {
            "strong_nir_residual_candidate",
            "strong_nir_covariance_candidate",
            "strong_content_nir_residual_candidate",
            "strong_content_nir_covariance_candidate",
        }
    ]

    cepheid_specific = [
        r for r in classified
        if r["classification"] in {
            "cepheid_residual_candidate",
            "cepheid_covariance_candidate",
            "content_cepheid_residual_candidate",
            "content_cepheid_covariance_candidate",
        }
    ]

    possible = [
        r for r in classified
        if r["classification"] == "possible_sh0es_outcome_or_model_candidate"
    ]

    global_context = [
        r for r in classified
        if r["classification"] == "global_pantheon_context_not_nir_row_aligned"
    ]

    known_surface = [
        r for r in classified
        if r["classification"] == "known_nir_surface_file"
    ]

    if strong:
        final_status = "strong_nir_residual_or_covariance_candidate_found_manual_alignment_required"
        readiness = 8
        next_wall = (
            "A strong NIR-specific residual/covariance candidate exists. Next test must prove row alignment "
            "before any frozen-rule replay."
        )
    elif cepheid_specific:
        final_status = "cepheid_residual_or_covariance_candidate_found_manual_alignment_required"
        readiness = 7
        next_wall = (
            "A Cepheid-specific residual/covariance candidate exists, but NIR linkage is not proven. "
            "Next test must inspect schema and row alignment."
        )
    elif possible:
        final_status = "possible_sh0es_outcome_layer_found_but_nir_link_not_proven"
        readiness = 6
        next_wall = (
            "Possible SH0ES outcome/model files exist, but this does not establish NIR row alignment. "
            "Manual schema proof required."
        )
    elif global_context:
        final_status = "only_global_context_layers_found_not_nir_validation_ready"
        readiness = 5
        next_wall = (
            "Only global Pantheon/SH0ES context files appear relevant. These may help later, but they are "
            "not enough for NIR validation."
        )
    elif known_surface:
        final_status = "surface_only_context_not_validation_ready"
        readiness = 4
        next_wall = (
            "The repository exposes the known NIR surface files, but no residual/outcome/covariance layer "
            "was found for validation."
        )
    else:
        final_status = "no_relevant_nir_residual_layer_found"
        readiness = 4
        next_wall = "No relevant residual/outcome/covariance layer was found."

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "counts": {
            "strong_nir_specific_candidates": len(strong),
            "cepheid_specific_candidates": len(cepheid_specific),
            "possible_sh0es_outcome_candidates": len(possible),
            "global_context_candidates": len(global_context),
            "known_surface_files": len(known_surface),
        },
    }


def holographic_surface_ledger(decision, classified):
    top_candidates = classified[:25]

    return {
        "observable_surface": {
            "name": "Public PantheonPlusSH0ES/DataRelease repository tree",
            "target_surface_from_v1_2": "SH0ES R22 NIR Cepheid H-band row table",
            "known_nir_surface_files": TARGET_NIR_FILES,
        },
        "hidden_depth_sought": {
            "name": "NIR-linked residual/outcome/covariance depth",
            "reason_needed": (
                "A surface-only H-band table cannot validate the frozen v1.0 Table2 residual rule. "
                "A legitimate residual/outcome/covariance layer is required before replay."
            ),
        },
        "boundary_that_forms_surface": {
            "repo_boundary": REPO,
            "data_boundary": "Public release files only",
            "method_boundary": "Path/content discovery and classification only; no model fitting",
        },
        "what_information_is_lost_or_missing_if_not_found": [
            "No direct residual vector tied to NIR Cepheid rows.",
            "No row-aligned covariance for NIR Cepheid replay.",
            "No lawful outcome layer for frozen-rule prediction.",
            "No basis to translate high/low H-edge surfaces into validation.",
        ],
        "what_can_be_reconstructed_now": [
            "Repository inventory.",
            "Candidate file map.",
            "Known NIR surface availability.",
            "Candidate residual/outcome/covariance classification.",
            "Decision on whether a manual alignment test is justified.",
        ],
        "what_cannot_be_reconstructed_now": [
            "TAIRID validation.",
            "H0 correction.",
            "New physics.",
            "Residual prediction.",
            "Covariance-aware replay.",
        ],
        "surface_noise_definition": [
            "Files that mention F160W/NIR but are filters, headers, notes, or unrelated context.",
            "Global covariance files that are not row-aligned to NIR Cepheids.",
            "Magnitude tables without residual/outcome columns.",
            "Any inferred residual created by subtracting values without an explicit accepted outcome definition.",
        ],
        "decision": decision,
        "top_candidates": top_candidates,
    }


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v1.3 Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v1.3 did")
    lines.append("")
    lines.append("- Scanned the public PantheonPlusSH0ES/DataRelease repository tree.")
    lines.append("- Classified possible residual, outcome, covariance, fit, and model files.")
    lines.append("- Preserved the v1.0/v1.2 frozen-rule boundary.")
    lines.append("- Did not validate, tune, or invent residuals.")
    lines.append("")
    lines.append("## Frozen rule carried forward")
    lines.append("")
    lines.append("- F160W/H-like surface only.")
    lines.append("- Within-host high 5% H edge minus within-host low 5% H edge.")
    lines.append("- Low-alpha/reference hosts: LMC, SMC, N4536.")
    lines.append("- M31 sign-break quarantine.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v1.3 is repository discovery only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not create residuals from H magnitudes.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 7:
        lines.append(
            "v1.4 should perform manual schema inspection and row-alignment proof on the strongest candidate file(s). "
            "Only after row alignment succeeds can a frozen-rule replay be attempted."
        )
    else:
        lines.append(
            "v1.4 should classify the NIR lane as surface-only context unless a separate legitimate NIR residual/outcome "
            "layer is identified. Do not force validation from surface data."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID NIR Residual Layer Locator v1.3 starting.")
    print("Boundary: discovery/classification only; no validation and no tuning.")

    write_json(OUTDIR / "claims_v1_3.json", CLAIMS_V1_3)
    write_json(OUTDIR / "frozen_rule_carried_forward_v1_3.json", FROZEN_RULE_CARRIED_FORWARD)

    try:
        tree_result = fetch_repo_tree()
        write_json(OUTDIR / "repo_tree_fetch_v1_3.json", tree_result)

        if tree_result["status"] != "downloaded":
            decision = {
                "final_status": "repo_tree_download_failed",
                "readiness_score_0_to_10": 3,
                "next_wall": "Could not fetch repository tree. Retry or inspect repo manually.",
                "counts": {},
            }
            write_json(OUTDIR / "nir_residual_layer_locator_v1_3_summary.json", {
                "test_name": "TAIRID NIR Residual Layer Locator v1.3",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "decision": decision,
                "tree_errors": tree_result["errors"],
                "truth_boundary": CLAIMS_V1_3["truth_boundary"],
            })
            print("Repository tree download failed.")
            return

        tree_rows = tree_result["rows"]
        write_csv(OUTDIR / "repo_tree_inventory_v1_3.csv", tree_rows)

        inventory = build_candidate_inventory(tree_rows)
        write_csv(OUTDIR / "candidate_inventory_v1_3.csv", inventory)

        classified, downloaded_sniffs = download_and_classify_candidates(tree_result["branch"], inventory)
        write_csv(OUTDIR / "classified_candidates_v1_3.csv", classified)
        write_json(OUTDIR / "downloaded_candidate_sniffs_v1_3.json", downloaded_sniffs)

        decision = decide_final_status(classified)
        write_json(OUTDIR / "decision_v1_3.json", decision)

        ledger = holographic_surface_ledger(decision, classified)
        write_json(OUTDIR / "holographic_surface_ledger_v1_3.json", ledger)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v1_3.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v1_3.txt").write_text(handoff, encoding="utf-8")

        class_counts = Counter(r["classification"] for r in classified)

        summary = {
            "test_name": "TAIRID NIR Residual Layer Locator v1.3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Repository discovery/classification only. No validation, no tuning, no H0 claim, no new-physics claim."
            ),
            "repo": REPO,
            "branch": tree_result["branch"],
            "repo_file_count": len(tree_rows),
            "candidate_count": len(inventory),
            "classified_candidate_count": len(classified),
            "classification_counts": dict(class_counts),
            "decision": decision,
            "claims_v1_3": CLAIMS_V1_3,
            "frozen_rule_carried_forward": FROZEN_RULE_CARRIED_FORWARD,
            "top_50_candidates": classified[:50],
            "holographic_surface_ledger": ledger,
            "output_files": {
                "summary_json": str(OUTDIR / "nir_residual_layer_locator_v1_3_summary.json"),
                "summary_txt": str(OUTDIR / "nir_residual_layer_locator_v1_3_summary.txt"),
                "repo_tree_csv": str(OUTDIR / "repo_tree_inventory_v1_3.csv"),
                "candidate_inventory_csv": str(OUTDIR / "candidate_inventory_v1_3.csv"),
                "classified_candidates_csv": str(OUTDIR / "classified_candidates_v1_3.csv"),
                "downloaded_sniffs_json": str(OUTDIR / "downloaded_candidate_sniffs_v1_3.json"),
                "decision_json": str(OUTDIR / "decision_v1_3.json"),
                "ledger_json": str(OUTDIR / "holographic_surface_ledger_v1_3.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v1_3.md"),
                "handoff_txt": str(OUTDIR / "next_thread_handoff_after_v1_3.txt"),
            },
            "interpretation": {
                "what_success_means": (
                    "A legitimate candidate residual/outcome/covariance layer may exist and deserves manual schema/row-alignment proof."
                ),
                "what_success_does_not_mean": (
                    "Discovery of a candidate file does not validate the frozen Table2 model."
                ),
                "what_failure_means": (
                    "If no candidate is found, the NIR lane is surface-only context and should not be used for validation."
                ),
                "truth_boundary": CLAIMS_V1_3["truth_boundary"],
            },
        }

        write_json(OUTDIR / "nir_residual_layer_locator_v1_3_summary.json", summary)

        with open(OUTDIR / "nir_residual_layer_locator_v1_3_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID NIR Residual Layer Locator v1.3\n\n")
            f.write("Boundary: discovery/classification only. No validation. No tuning.\n\n")
            f.write(f"Repository: {REPO}\n")
            f.write(f"Branch: {tree_result['branch']}\n")
            f.write(f"Repository file count: {len(tree_rows)}\n")
            f.write(f"Candidate count: {len(inventory)}\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Classification counts:\n")
            f.write(json.dumps(dict(class_counts), indent=2, default=json_default) + "\n\n")
            f.write("Top candidates:\n")
            f.write(json.dumps(classified[:25], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This only searches for a legitimate residual/outcome/covariance layer.\n")
            f.write("- Do not invent residuals.\n")

        print("TAIRID NIR Residual Layer Locator v1.3 complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID NIR Residual Layer Locator v1.3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "nir_residual_layer_locator_v1_3_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V1_3["truth_boundary"],
        }
        write_json(OUTDIR / "nir_residual_layer_locator_v1_3_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
