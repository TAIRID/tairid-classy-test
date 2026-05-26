#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.4
Anchor-stratified within-host F160W polarity audit.

Purpose:
v0.2.1 showed that F160W was the primary and specific signed edge-pair polarity variable.
v0.3 showed that within-host F160W polarity was stable across host folds.
v0.3.1 showed that pooled global polarity is confounded by anchor/system-class structure:
the global bright edge is concentrated in LMC/SMC, while the global faint edge lives in other hosts.

This v0.4 test asks the next necessary question:

    Does the within-host F160W high-minus-low polarity survive after separating
    anchor systems from non-anchor hosts?

Anchor systems tested here:
    LMC
    SMC
    N4258

Core TAIRID question:
    Is the strongest field-relative F160W polarity only an anchor/calibration artifact,
    or does it survive in the non-anchor host field?

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only tests whether the stable within-host F160W polarity survives anchor stratification.
"""

import csv
import json
import math
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_tairid_table2_f160w_faint_tail_host_field_v1_6 as v16
import run_tairid_boundary_polarity_battery_v0_1 as b01
import run_tairid_boundary_polarity_battery_v0_2 as v02


OUTDIR = Path("tairid_boundary_polarity_battery_v0_4_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

FULL_DESIGN = "plus_host_top10_row_order_measurement_controls"
ORIGINAL_DESIGN = "original_47"

F160W_COLUMN = "table2_num_7"
F160W_LABEL = "f160w_like"
EDGE_PERCENTILE = 5
MIN_HOST_ROWS = 20
PERMUTATION_REPEATS = 120
SEED = 42

ANCHOR_HOSTS = {"LMC", "SMC", "N4258"}

CLAIMS_V0_4 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.4",
    "scope": "SH0ES Table2 anchor-stratified within-host F160W polarity",
    "reason_for_test": (
        "v0.3.1 showed pooled global F160W polarity is mixed with anchor/system-class structure. "
        "v0.4 therefore tests the stronger result from v0.3: within-host field-relative F160W polarity."
    ),
    "native_tairid_claim": (
        "If F160W polarity is a field-relative boundary effect rather than only an anchor artifact, "
        "then within-host high-minus-low F160W contrast should survive after excluding LMC, SMC, and N4258."
    ),
    "primary_prediction": (
        "The non-anchor within-host F160W contrast should preserve positive high-minus-low residual direction "
        "and beat host-preserving same-count permutation controls."
    ),
    "anchor_check": (
        "Anchor-only strata are audited separately. If anchors pass too, the pattern is broader. "
        "If only anchors pass, the claim narrows to calibration/anchor structure."
    ),
    "failure_rule": (
        "If non-anchor within-host polarity collapses, TAIRID must narrow this SH0ES lane to anchor/calibration geometry "
        "and stop claiming broad field-relative Cepheid polarity."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only tests whether within-host F160W polarity survives anchor stratification."
    ),
}


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Counter):
        return dict(obj)
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


def safe_name(value):
    return v16.safe_name(value)


def entity(row, key):
    return v16.get_entity(row, key)


def audit_vector(name, values, kind, y, c_factor, fit, design_name, metadata=None):
    return v16.audit_vector(
        name=name,
        values=values,
        kind=kind,
        y=y,
        c_factor=c_factor,
        fit=fit,
        design_name=design_name,
        metadata=metadata or {},
    )


def direction_stats(name, contrast_values, fit):
    return v02.direction_stats(name, contrast_values, fit)


def add_permutation_flags(observed, perm_summary):
    return v02.add_permutation_flags(observed, perm_summary)


def host_counts(mapped_rows):
    counts = Counter(entity(row, "host_guess") for row in mapped_rows)
    rows = [
        {
            "host": host,
            "row_count": count,
            "is_anchor": host in ANCHOR_HOSTS,
        }
        for host, count in counts.items()
    ]
    return sorted(rows, key=lambda r: (-r["row_count"], r["host"]))


def hosts_with_nonzero_contrast(mapped_rows, contrast):
    hosts = set()

    for row in mapped_rows:
        idx = int(row["compact_observation_index"])

        if float(contrast[idx]) != 0.0:
            hosts.add(entity(row, "host_guess"))

    return sorted(hosts)


def restrict_vector_to_hosts(vector, mapped_rows, hosts):
    hosts = set(hosts)
    out = np.zeros_like(np.asarray(vector, dtype=float))
    allowed_indices = []

    for row in mapped_rows:
        host = entity(row, "host_guess")
        idx = int(row["compact_observation_index"])

        if host in hosts:
            allowed_indices.append(idx)
            out[idx] = vector[idx]

    return out, sorted(set(allowed_indices))


def selected_rows_for_hosts(mapped_rows, contrast, hosts):
    hosts = set(hosts)
    rows = []

    for row in mapped_rows:
        host = entity(row, "host_guess")
        idx = int(row["compact_observation_index"])
        value = float(contrast[idx])

        if host in hosts and value != 0.0:
            out = dict(row)
            out["host_guess_clean"] = host
            out["contrast_value"] = value
            out["edge_side"] = "high_f160w_faint" if value > 0 else "low_f160w_bright"
            rows.append(out)

    rows = sorted(
        rows,
        key=lambda r: (
            r["host_guess_clean"],
            r["edge_side"],
            int(r["compact_observation_index"]),
        ),
    )

    return rows


def host_preserving_permutation(name, contrast, mapped_rows, allowed_hosts, y, c_factor, fit, design_name):
    rng = np.random.default_rng(SEED)
    allowed_hosts = set(allowed_hosts)
    contrast = np.asarray(contrast, dtype=float).reshape(-1)

    host_to_all_indices = defaultdict(list)
    host_to_pos_count = Counter()
    host_to_neg_count = Counter()

    for row in mapped_rows:
        host = entity(row, "host_guess")
        idx = int(row["compact_observation_index"])

        if host not in allowed_hosts:
            continue

        host_to_all_indices[host].append(idx)

        if contrast[idx] > 0:
            host_to_pos_count[host] += 1
        elif contrast[idx] < 0:
            host_to_neg_count[host] += 1

    active_hosts = sorted(
        host for host in allowed_hosts
        if host_to_pos_count[host] > 0 and host_to_neg_count[host] > 0
    )

    observed = audit_vector(
        name,
        contrast,
        "observed_anchor_stratified_within_host_contrast",
        y,
        c_factor,
        fit,
        design_name,
        metadata={
            "allowed_host_count": len(allowed_hosts),
            "active_host_count": len(active_hosts),
            "positive_count": int(np.sum(contrast > 0)),
            "negative_count": int(np.sum(contrast < 0)),
        },
    )

    if not active_hosts:
        return [], {
            "candidate": name,
            "design": design_name,
            "status": "no_active_hosts_with_both_edges",
            "repeats": PERMUTATION_REPEATS,
            "allowed_host_count": len(allowed_hosts),
            "active_host_count": 0,
            "observed_delta_chi2": observed.get("delta_chi2_score", 0.0),
            "observed_score": observed.get("score", 0.0),
            "observed_nondegenerate_ratio": observed.get("nondegenerate_ratio", 0.0),
            "permutation_delta_95": None,
            "permutation_delta_99": None,
            "permutation_abs_score_95": None,
            "permutation_abs_score_99": None,
            "observed_exceeds_95_percent_permutation_delta": None,
            "observed_exceeds_99_percent_permutation_delta": None,
            "observed_abs_score_exceeds_95_percent_permutation": None,
            "observed_abs_score_exceeds_99_percent_permutation": None,
        }

    rows = []

    for repeat in range(PERMUTATION_REPEATS):
        values = np.zeros(len(y), dtype=float)

        for host in active_hosts:
            all_indices = np.asarray(host_to_all_indices[host], dtype=int)
            pos_count = int(host_to_pos_count[host])
            neg_count = int(host_to_neg_count[host])
            total = pos_count + neg_count

            if total <= 0 or len(all_indices) < total:
                continue

            chosen = rng.choice(all_indices, size=total, replace=False)
            pos = chosen[:pos_count]
            neg = chosen[pos_count:pos_count + neg_count]

            values[pos] = 1.0
            values[neg] = -1.0

        rows.append(
            audit_vector(
                f"{name}_host_preserving_permutation_{repeat}",
                values,
                "host_preserving_same_count_permutation",
                y,
                c_factor,
                fit,
                design_name,
                metadata={
                    "source_candidate": name,
                    "repeat": repeat,
                    "allowed_host_count": len(allowed_hosts),
                    "active_host_count": len(active_hosts),
                    "positive_count": int(np.sum(values > 0)),
                    "negative_count": int(np.sum(values < 0)),
                },
            )
        )

    deltas = np.asarray([row["delta_chi2_score"] for row in rows], dtype=float)
    scores = np.asarray([row["score"] for row in rows], dtype=float)

    return rows, {
        "candidate": name,
        "design": design_name,
        "status": "ok",
        "repeats": PERMUTATION_REPEATS,
        "allowed_host_count": len(allowed_hosts),
        "active_host_count": len(active_hosts),
        "positive_count": int(np.sum(contrast > 0)),
        "negative_count": int(np.sum(contrast < 0)),
        "observed_delta_chi2": observed["delta_chi2_score"],
        "observed_score": observed["score"],
        "observed_nondegenerate_ratio": observed["nondegenerate_ratio"],
        "permutation_delta_mean": float(np.mean(deltas)),
        "permutation_delta_95": float(np.percentile(deltas, 95)),
        "permutation_delta_99": float(np.percentile(deltas, 99)),
        "permutation_abs_score_mean": float(np.mean(np.abs(scores))),
        "permutation_abs_score_95": float(np.percentile(np.abs(scores), 95)),
        "permutation_abs_score_99": float(np.percentile(np.abs(scores), 99)),
        "observed_exceeds_95_percent_permutation_delta": bool(observed["delta_chi2_score"] > float(np.percentile(deltas, 95))),
        "observed_exceeds_99_percent_permutation_delta": bool(observed["delta_chi2_score"] > float(np.percentile(deltas, 99))),
        "observed_abs_score_exceeds_95_percent_permutation": bool(abs(observed["score"]) > float(np.percentile(np.abs(scores), 95))),
        "observed_abs_score_exceeds_99_percent_permutation": bool(abs(observed["score"]) > float(np.percentile(np.abs(scores), 99))),
    }


def pass_gate(audit_row, direction_row):
    return bool(
        audit_row.get("delta_chi2_score", 0.0) >= 25.0
        and audit_row.get("p_value_chi2_one_dof", 1.0) <= 0.01
        and direction_row.get("high_side_more_positive") is True
        and audit_row.get("observed_exceeds_99_percent_permutation_delta") is True
        and audit_row.get("observed_abs_score_exceeds_99_percent_permutation") is True
    )


def directional_gate(audit_row, direction_row):
    return bool(
        audit_row.get("delta_chi2_score", 0.0) >= 10.0
        and audit_row.get("p_value_chi2_one_dof", 1.0) <= 0.05
        and direction_row.get("high_side_more_positive") is True
        and audit_row.get("observed_exceeds_95_percent_permutation_delta") is True
        and audit_row.get("observed_abs_score_exceeds_95_percent_permutation") is True
    )


def audit_stratum(stratum_name, hosts, contrast, mapped_rows, y, c_factor, fit):
    restricted, allowed_indices = restrict_vector_to_hosts(contrast, mapped_rows, hosts)

    selected_rows = selected_rows_for_hosts(mapped_rows, contrast, hosts)
    write_csv(OUTDIR / f"{safe_name(stratum_name)}_selected_rows_v0_4.csv", selected_rows)

    metadata = {
        "stratum": stratum_name,
        "host_count": len(hosts),
        "hosts": ",".join(sorted(hosts)),
        "allowed_index_count": len(allowed_indices),
        "positive_count": int(np.sum(restricted > 0)),
        "negative_count": int(np.sum(restricted < 0)),
        "nonzero_count": int(np.sum(restricted != 0)),
    }

    observed = audit_vector(
        f"within_host_f160w_high_minus_low_{stratum_name}",
        restricted,
        "anchor_stratified_within_host_f160w_contrast",
        y,
        c_factor,
        fit,
        FULL_DESIGN,
        metadata=metadata,
    )

    dstat = direction_stats(
        f"within_host_f160w_high_minus_low_{stratum_name}",
        restricted,
        fit,
    )

    perm_rows, perm_summary = host_preserving_permutation(
        f"within_host_f160w_high_minus_low_{stratum_name}",
        restricted,
        mapped_rows,
        hosts,
        y,
        c_factor,
        fit,
        FULL_DESIGN,
    )

    write_csv(OUTDIR / f"{safe_name(stratum_name)}_permutation_details_v0_4.csv", perm_rows)
    write_json(OUTDIR / f"{safe_name(stratum_name)}_permutation_summary_v0_4.json", perm_summary)

    if perm_summary.get("status") == "ok":
        observed_with_perm = add_permutation_flags(observed, perm_summary)
    else:
        observed_with_perm = dict(observed)

    observed_with_perm["stratum"] = stratum_name
    observed_with_perm["direction_positive"] = bool(dstat.get("high_side_more_positive") is True)
    observed_with_perm["signed_mean_residual_difference_high_minus_low"] = dstat.get(
        "signed_mean_residual_difference_high_minus_low"
    )
    observed_with_perm["signed_mean_cinv_residual_difference_high_minus_low"] = dstat.get(
        "signed_mean_cinv_residual_difference_high_minus_low"
    )
    observed_with_perm["stratum_pass"] = pass_gate(observed_with_perm, dstat)
    observed_with_perm["stratum_directional"] = directional_gate(observed_with_perm, dstat)
    observed_with_perm["permutation_status"] = perm_summary.get("status")

    return {
        "stratum": stratum_name,
        "hosts": sorted(hosts),
        "audit": observed_with_perm,
        "direction": dstat,
        "permutation_summary": perm_summary,
        "selected_row_count": len(selected_rows),
    }


def decide_status(stratum_results):
    by_name = {row["stratum"]: row for row in stratum_results}

    all_result = by_name.get("all_hosts", {})
    non_anchor = by_name.get("non_anchor_excluding_LMC_SMC_N4258", {})
    anchor = by_name.get("anchor_only_LMC_SMC_N4258", {})
    magellanic = by_name.get("magellanic_only_LMC_SMC", {})
    n4258 = by_name.get("n4258_only", {})

    all_pass = bool(all_result.get("audit", {}).get("stratum_pass") is True)
    non_anchor_pass = bool(non_anchor.get("audit", {}).get("stratum_pass") is True)
    anchor_pass = bool(anchor.get("audit", {}).get("stratum_pass") is True)

    non_anchor_directional = bool(non_anchor.get("audit", {}).get("stratum_directional") is True)
    anchor_directional = bool(anchor.get("audit", {}).get("stratum_directional") is True)

    best_cases = {
        "all_hosts": all_result,
        "non_anchor_excluding_LMC_SMC_N4258": non_anchor,
        "anchor_only_LMC_SMC_N4258": anchor,
        "magellanic_only_LMC_SMC": magellanic,
        "n4258_only": n4258,
        "all_pass": all_pass,
        "non_anchor_pass": non_anchor_pass,
        "anchor_pass": anchor_pass,
        "non_anchor_directional": non_anchor_directional,
        "anchor_directional": anchor_directional,
    }

    if non_anchor_pass and anchor_pass:
        return (
            "within_host_f160w_polarity_survives_anchor_and_non_anchor_strata",
            9,
            "Within-host F160W polarity survives both non-anchor and anchor strata.",
            best_cases,
        )

    if non_anchor_pass and not anchor_pass:
        return (
            "within_host_f160w_polarity_survives_non_anchor_exclusion",
            9,
            "Within-host F160W polarity survives after excluding LMC, SMC, and N4258; not only an anchor artifact.",
            best_cases,
        )

    if anchor_pass and not non_anchor_pass:
        return (
            "within_host_f160w_polarity_anchor_concentrated",
            7,
            "Within-host F160W polarity passes inside anchors but not outside them; claim narrows to anchor/calibration structure.",
            best_cases,
        )

    if non_anchor_directional:
        return (
            "within_host_f160w_polarity_non_anchor_directional_not_locked",
            7,
            "Non-anchor within-host F160W polarity remains directional, but does not meet the locked pass gate.",
            best_cases,
        )

    if all_pass and not non_anchor_directional:
        return (
            "within_host_f160w_full_sample_passes_but_non_anchor_collapses",
            7,
            "Full-sample within-host F160W polarity passes, but non-anchor polarity collapses; broad field-relative claim weakens.",
            best_cases,
        )

    return (
        "within_host_f160w_polarity_not_supported_after_anchor_stratification",
        6,
        "Anchor stratification does not support a stable non-anchor within-host F160W polarity result.",
        best_cases,
    )


def make_plots(stratum_summary_rows):
    rows = [
        r for r in stratum_summary_rows
        if r.get("status") == "ok"
    ]

    if not rows:
        return

    labels = [r["stratum"] for r in rows]
    x = np.arange(len(rows))

    plt.figure(figsize=(12, 5))
    plt.bar(x, [r.get("delta_chi2_score", 0.0) for r in rows])
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("delta chi2")
    plt.title("Anchor-stratified within-host F160W polarity")
    plt.tight_layout()
    plt.savefig(OUTDIR / "anchor_stratified_delta_chi2_v0_4.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(x, [r.get("signed_mean_residual_difference_high_minus_low", 0.0) for r in rows])
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("high-minus-low mean residual")
    plt.title("Anchor-stratified signed direction")
    plt.tight_layout()
    plt.savefig(OUTDIR / "anchor_stratified_signed_direction_v0_4.png", dpi=160)
    plt.close()


def main():
    print("TAIRID Boundary Prediction Battery v0.4 starting.")
    print("Boundary: anchor-stratified within-host F160W polarity audit only; not proof.")

    write_json(OUTDIR / "claims_v0_4.json", CLAIMS_V0_4)

    repair_summary = {}

    try:
        v16.OUTDIR = OUTDIR
        v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        b01.OUTDIR = OUTDIR
        b01.DOWNLOAD_DIR = DOWNLOAD_DIR
        v02.OUTDIR = OUTDIR
        v02.DOWNLOAD_DIR = DOWNLOAD_DIR
        b01.PERMUTATION_REPEATS = PERMUTATION_REPEATS
        v02.PERMUTATION_REPEATS = PERMUTATION_REPEATS

        ns, repair_summary = v16.load_v15_helpers()
        write_json(OUTDIR / "v15_import_repair_summary_v0_4.json", repair_summary)

        download_repo_path = ns["download_repo_path"]
        code_context_search = ns["code_context_search"]
        extract_first_numeric_fits_array = ns["extract_first_numeric_fits_array"]
        parse_table2_tex = ns["parse_table2_tex"]
        orient_design_matrix = ns["orient_design_matrix"]
        stable_cholesky = ns["stable_cholesky"]
        gls_fit = ns["gls_fit"]
        recover_compact_rows = ns["recover_compact_rows"]
        map_table2_to_spine = ns["map_table2_to_spine"]
        summarize_hosts = ns["summarize_hosts"]
        numeric_feature_summary = ns["numeric_feature_summary"]
        build_designs = ns["build_designs"]
        h0_like = ns["h0_like"]

        downloads = {}
        aux_results = []
        ledger = []

        for label, repo_path in ns["COMPACT_FILES"].items():
            result = download_repo_path(repo_path, label)
            downloads[label] = result
            ledger.append(
                {
                    "label": label,
                    "repo_path": repo_path,
                    "status": result.get("status"),
                    "local_path": result.get("local_path"),
                    "bytes": result.get("bytes"),
                    "sha256": result.get("sha256"),
                    "attempt_count": len(result.get("attempts", [])),
                }
            )

        for repo_path in ns["AUX_FILES"]:
            result = download_repo_path(repo_path, safe_name(repo_path))
            aux_results.append(result)
            ledger.append(
                {
                    "label": safe_name(repo_path),
                    "repo_path": repo_path,
                    "status": result.get("status"),
                    "local_path": result.get("local_path"),
                    "bytes": result.get("bytes"),
                    "sha256": result.get("sha256"),
                    "attempt_count": len(result.get("attempts", [])),
                }
            )

        write_csv(OUTDIR / "download_ledger_v0_4.csv", ledger)
        write_json(OUTDIR / "download_attempts_v0_4.json", {"compact": downloads, "auxiliary": aux_results})

        code_hits = code_context_search(aux_results)

        parsed = {}
        parse_meta = {}
        parse_errors = []

        for label in ["allc", "alll", "ally"]:
            result = downloads.get(label, {})

            if result.get("status") != "downloaded":
                parse_errors.append(
                    {
                        "label": label,
                        "status": "not_downloaded",
                        "download_status": result.get("status"),
                    }
                )
                continue

            try:
                arr, meta = extract_first_numeric_fits_array(Path(result["local_path"]))
                parsed[label] = arr
                parse_meta[label] = meta
            except Exception as exc:
                parse_errors.append(
                    {
                        "label": label,
                        "status": "parse_failed",
                        "error": str(exc),
                    }
                )

        write_json(OUTDIR / "parse_meta_v0_4.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v0_4.json", parse_errors)

        table2_result = next(
            (
                item for item in aux_results
                if item.get("repo_path") == "SH0ES_Data/table2.tex"
                and item.get("status") == "downloaded"
            ),
            None,
        )

        if parse_errors or not all(key in parsed for key in ["allc", "alll", "ally"]) or not table2_result:
            summary = {
                "test_name": "TAIRID Boundary Prediction Battery v0.4",
                "final_status": "boundary_polarity_battery_v0_4_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or Table2 retrieval before v0.4.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "boundary_polarity_battery_v0_4_summary.json", summary)
            print("Parse/download failed. See summary JSON.")
            return

        table2_all_rows, table2_data_rows, table2_header_rows = parse_table2_tex(
            Path(table2_result["local_path"])
        )

        C = np.asarray(parsed["allc"], dtype=float)
        L = np.asarray(parsed["alll"], dtype=float)
        y = np.asarray(parsed["ally"], dtype=float).reshape(-1)

        X, orientation = orient_design_matrix(L, len(y))

        if X is None:
            raise RuntimeError(f"Could not orient L: {orientation}")

        if C.ndim != 2 or C.shape[0] != len(y) or C.shape[1] != len(y):
            raise RuntimeError(f"C shape {C.shape} does not match y length {len(y)}")

        c_factor, C_sym, jitter, chol_attempts = stable_cholesky(C)
        baseline = gls_fit(y, X, c_factor)
        row_rows, cluster_rows = recover_compact_rows(X, y, C_sym, baseline)

        write_csv(OUTDIR / "compact_row_map_v0_4.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v0_4.csv", cluster_rows)

        mapped_rows, map_status = map_table2_to_spine(row_rows, table2_data_rows)
        host_summary = summarize_hosts(mapped_rows)
        numeric_rows, likely_numeric_labels = numeric_feature_summary(mapped_rows)

        designs, control_metadata = build_designs(X, mapped_rows, host_summary, len(y))
        design_fits = {
            name: gls_fit(y, design["D"], c_factor)
            for name, design in designs.items()
        }

        full_fit = design_fits[FULL_DESIGN]

        within_low, within_high, within_contrast, within_low_rows, within_high_rows, within_meta = v02.build_within_host_edge_pair(
            mapped_rows,
            len(y),
            F160W_COLUMN,
            F160W_LABEL,
            percentile=EDGE_PERCENTILE,
        )

        write_csv(OUTDIR / "within_host_f160w_low_rows_v0_4.csv", within_low_rows)
        write_csv(OUTDIR / "within_host_f160w_high_rows_v0_4.csv", within_high_rows)

        contrast_hosts = hosts_with_nonzero_contrast(mapped_rows, within_contrast)
        anchor_hosts_present = sorted([h for h in contrast_hosts if h in ANCHOR_HOSTS])
        non_anchor_hosts_present = sorted([h for h in contrast_hosts if h not in ANCHOR_HOSTS])
        magellanic_hosts_present = sorted([h for h in contrast_hosts if h in {"LMC", "SMC"}])
        n4258_hosts_present = sorted([h for h in contrast_hosts if h == "N4258"])

        stratum_definitions = [
            ("all_hosts", contrast_hosts),
            ("non_anchor_excluding_LMC_SMC_N4258", non_anchor_hosts_present),
            ("anchor_only_LMC_SMC_N4258", anchor_hosts_present),
            ("magellanic_only_LMC_SMC", magellanic_hosts_present),
            ("n4258_only", n4258_hosts_present),
        ]

        write_json(
            OUTDIR / "anchor_strata_inventory_v0_4.json",
            {
                "anchor_hosts_declared": sorted(ANCHOR_HOSTS),
                "contrast_hosts": contrast_hosts,
                "anchor_hosts_present": anchor_hosts_present,
                "non_anchor_hosts_present": non_anchor_hosts_present,
                "magellanic_hosts_present": magellanic_hosts_present,
                "n4258_hosts_present": n4258_hosts_present,
                "host_counts": host_counts(mapped_rows),
            },
        )

        stratum_results = []
        stratum_summary_rows = []
        all_permutation_summaries = []

        for stratum_name, hosts in stratum_definitions:
            result = audit_stratum(
                stratum_name,
                hosts,
                within_contrast,
                mapped_rows,
                y,
                c_factor,
                full_fit,
            )
            stratum_results.append(result)
            all_permutation_summaries.append(result["permutation_summary"])

            audit = result["audit"]
            stratum_summary_rows.append(
                {
                    "stratum": stratum_name,
                    "host_count": len(hosts),
                    "hosts": ",".join(hosts),
                    "selected_row_count": result["selected_row_count"],
                    "status": audit.get("status"),
                    "permutation_status": audit.get("permutation_status"),
                    "positive_count": audit.get("positive_count"),
                    "negative_count": audit.get("negative_count"),
                    "delta_chi2_score": audit.get("delta_chi2_score"),
                    "p_value_chi2_one_dof": audit.get("p_value_chi2_one_dof"),
                    "nondegenerate_ratio": audit.get("nondegenerate_ratio"),
                    "score": audit.get("score"),
                    "signed_mean_residual_difference_high_minus_low": audit.get("signed_mean_residual_difference_high_minus_low"),
                    "direction_positive": audit.get("direction_positive"),
                    "observed_exceeds_95_percent_permutation_delta": audit.get("observed_exceeds_95_percent_permutation_delta"),
                    "observed_exceeds_99_percent_permutation_delta": audit.get("observed_exceeds_99_percent_permutation_delta"),
                    "observed_abs_score_exceeds_95_percent_permutation": audit.get("observed_abs_score_exceeds_95_percent_permutation"),
                    "observed_abs_score_exceeds_99_percent_permutation": audit.get("observed_abs_score_exceeds_99_percent_permutation"),
                    "stratum_directional": audit.get("stratum_directional"),
                    "stratum_pass": audit.get("stratum_pass"),
                }
            )

        write_csv(OUTDIR / "anchor_stratified_summary_v0_4.csv", stratum_summary_rows)
        write_json(OUTDIR / "anchor_stratified_results_v0_4.json", stratum_results)
        write_json(OUTDIR / "anchor_stratified_permutation_summaries_v0_4.json", all_permutation_summaries)

        final_status, readiness_score, next_wall, best_cases = decide_status(stratum_results)
        make_plots(stratum_summary_rows)

        design_fit_rows = []

        for design_name, fit in design_fits.items():
            design_fit_rows.append(
                {
                    "design": design_name,
                    "k": fit["k"],
                    "dof": fit["dof"],
                    "chi2": fit["chi2"],
                    "reduced_chi2": fit["reduced_chi2"],
                    "aic": fit["aic"],
                    "bic": fit["bic"],
                    "delta_chi2_vs_original": baseline["chi2"] - fit["chi2"],
                    "delta_aic_vs_original": fit["aic"] - baseline["aic"],
                    "delta_bic_vs_original": fit["bic"] - baseline["bic"],
                }
            )

        write_csv(OUTDIR / "design_fit_comparison_v0_4.csv", design_fit_rows)

        residual = baseline["residual"]
        abs_residual = np.abs(residual)

        edge_counts = {
            "rows_total": int(X.shape[0]),
            "spine_38_41_43_rows": int(sum(1 for row in row_rows if row["contains_38_41_43_spine"])),
            "bridge_42_46_rows": int(sum(1 for row in row_rows if row["bridges_param42_param46"])),
            "touch_param42_rows": int(sum(1 for row in row_rows if row["touches_param42"])),
            "touch_param46_rows": int(sum(1 for row in row_rows if row["touches_param46_H0_like"])),
        }

        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.4",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Anchor-stratified within-host F160W polarity audit only. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_4": CLAIMS_V0_4,
            "repair_summary": repair_summary,
            "matrix_shapes": {
                "allc_C": list(C.shape),
                "alll_L": list(L.shape),
                "ally_y_original": list(np.asarray(parsed["ally"]).shape),
                "y_flat": list(y.shape),
                "X_design": list(X.shape),
                "L_orientation": orientation,
            },
            "covariance": {
                "cholesky_jitter": jitter,
                "cholesky_attempts": chol_attempts,
                "diag_min": float(np.min(np.diag(C_sym))),
                "diag_max": float(np.max(np.diag(C_sym))),
                "diag_nonpositive_count": int(np.sum(np.diag(C_sym) <= 0)),
            },
            "baseline_gls": {
                "chi2": baseline["chi2"],
                "dof": baseline["dof"],
                "reduced_chi2": baseline["reduced_chi2"],
                "aic": baseline["aic"],
                "bic": baseline["bic"],
                "parameter_count": int(X.shape[1]),
                "param38_value": float(baseline["beta"][38]),
                "param41_value": float(baseline["beta"][41]),
                "param43_value": float(baseline["beta"][43]),
                "param42_value": float(baseline["beta"][42]),
                "param46_fivelogH0": float(baseline["beta"][46]),
                "param46_H0_like": h0_like(baseline["beta"]),
                "normal_condition_estimate": float(np.linalg.cond(baseline["normal"])),
                "residual_mean": float(np.mean(residual)),
                "residual_std": float(np.std(residual)),
                "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
                "abs_residual_90": float(np.percentile(abs_residual, 90)),
                "abs_residual_95": float(np.percentile(abs_residual, 95)),
                "abs_residual_99": float(np.percentile(abs_residual, 99)),
            },
            "edge_counts": edge_counts,
            "table2_mapping": map_status,
            "table2_parse_counts": {
                "table2_all_rows": len(table2_all_rows),
                "table2_data_rows": len(table2_data_rows),
                "table2_header_rows": len(table2_header_rows),
                "mapped_rows": len(mapped_rows),
                "unique_host_guesses": len(host_summary),
            },
            "likely_numeric_labels": likely_numeric_labels,
            "design_fit_comparison": design_fit_rows,
            "top_numeric_correlations": numeric_rows[:40],
            "top_hosts_by_residual_pressure": host_summary[:30],
            "within_host_f160w_meta": within_meta,
            "anchor_strata_summary": stratum_summary_rows,
            "anchor_strata_results": stratum_results,
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_4_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_4_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v0_4.json"),
                "strata_inventory_json": str(OUTDIR / "anchor_strata_inventory_v0_4.json"),
                "strata_summary_csv": str(OUTDIR / "anchor_stratified_summary_v0_4.csv"),
                "strata_results_json": str(OUTDIR / "anchor_stratified_results_v0_4.json"),
                "permutation_summaries_json": str(OUTDIR / "anchor_stratified_permutation_summaries_v0_4.json"),
                "plots": [
                    str(OUTDIR / "anchor_stratified_delta_chi2_v0_4.png"),
                    str(OUTDIR / "anchor_stratified_signed_direction_v0_4.png"),
                ],
            },
            "interpretation": {
                "what_supports_non_anchor_field_relative_polarity": (
                    "The non-anchor stratum excluding LMC, SMC, and N4258 preserves positive direction and beats host-preserving permutation controls."
                ),
                "what_supports_anchor_artifact": (
                    "Anchor-only strata pass while non-anchor strata collapse."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only checks whether within-host F160W polarity survives anchor stratification."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_4_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_4_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.4\n\n")
            f.write("Boundary: anchor-stratified within-host F160W polarity audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Claims v0.4:\n")
            f.write(json.dumps(CLAIMS_V0_4, indent=2, default=json_default) + "\n\n")
            f.write("Anchor strata summary:\n")
            f.write(json.dumps(stratum_summary_rows, indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only tests within-host F160W polarity after anchor stratification.\n")

        print("TAIRID Boundary Prediction Battery v0.4 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.4",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_4_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_4_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
