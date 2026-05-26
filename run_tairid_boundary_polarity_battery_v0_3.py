#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.3
F160W host-fold replication and leave-host-out stability audit.

Purpose:
v0.2.1 found that F160W is the strongest signed edge-pair polarity variable.
This v0.3 test asks whether that F160W polarity is stable across host subsets.

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only tests whether the v0.2.1 F160W edge-pair polarity is distributed and stable.
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


OUTDIR = Path("tairid_boundary_polarity_battery_v0_3_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

FULL_DESIGN = "plus_host_top10_row_order_measurement_controls"
ORIGINAL_DESIGN = "original_47"

F160W_COLUMN = "table2_num_7"
F160W_LABEL = "f160w_like"
EDGE_PERCENTILE = 5
MIN_HOST_ROWS = 20
N_FOLDS = 5
PERMUTATION_REPEATS = 80
SEED = 42

CLAIMS_V0_3 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.3",
    "scope": "SH0ES Table2 F160W host-fold replication and leave-host-out stability",
    "native_tairid_claim": (
        "If F160W is a real boundary-polarity surface and not a single local artifact, "
        "then the high-minus-low F160W contrast should remain directionally stable across host folds "
        "and should not collapse when one host is removed."
    ),
    "primary_prediction": (
        "F160W edge-pair polarity should remain positive for most host folds and remain detectable "
        "after removing the strongest individual host contributors."
    ),
    "failure_rule": (
        "If polarity is carried by one host or a tiny host cluster, TAIRID must narrow the SH0ES claim "
        "to a local concentration rather than a distributed measurement-boundary structure."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only tests stability of the v0.2.1 F160W polarity result."
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


def host_list_from_mapped_rows(mapped_rows):
    counts = Counter(entity(row, "host_guess") for row in mapped_rows)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def make_balanced_host_folds(mapped_rows, n_folds=N_FOLDS):
    host_counts = host_list_from_mapped_rows(mapped_rows)
    folds = [set() for _ in range(n_folds)]
    fold_counts = [0 for _ in range(n_folds)]

    for host, count in host_counts:
        idx = int(np.argmin(fold_counts))
        folds[idx].add(host)
        fold_counts[idx] += count

    return [sorted(list(fold)) for fold in folds], fold_counts


def restrict_vector_to_hosts(vector, mapped_rows, hosts):
    hosts = set(hosts)
    keep_indices = set(
        int(row["compact_observation_index"])
        for row in mapped_rows
        if entity(row, "host_guess") in hosts
    )
    out = np.zeros_like(np.asarray(vector, dtype=float))
    for idx in keep_indices:
        out[idx] = vector[idx]
    return out, sorted(keep_indices)


def remove_hosts_from_vector(vector, mapped_rows, hosts):
    hosts = set(hosts)
    out = np.asarray(vector, dtype=float).copy()
    for row in mapped_rows:
        if entity(row, "host_guess") in hosts:
            out[int(row["compact_observation_index"])] = 0.0
    return out


def host_restricted_same_count_permutation(name, contrast, allowed_indices, y, c_factor, fit, design_name, repeats=PERMUTATION_REPEATS):
    rng = np.random.default_rng(SEED)
    contrast = np.asarray(contrast, dtype=float).reshape(-1)
    allowed_indices = np.asarray(sorted(set(int(i) for i in allowed_indices)), dtype=int)

    pos_count = int(np.sum(contrast > 0))
    neg_count = int(np.sum(contrast < 0))
    total = pos_count + neg_count

    rows = []
    if total <= 0 or len(allowed_indices) < total:
        observed = audit_vector(
            name,
            contrast,
            "observed_host_restricted_contrast",
            y,
            c_factor,
            fit,
            design_name,
        )
        return rows, {
            "candidate": name,
            "design": design_name,
            "repeats": repeats,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "allowed_index_count": int(len(allowed_indices)),
            "observed_delta_chi2": observed.get("delta_chi2_score", 0.0),
            "observed_score": observed.get("score", 0.0),
            "permutation_delta_95": None,
            "permutation_delta_99": None,
            "permutation_abs_score_95": None,
            "permutation_abs_score_99": None,
            "observed_exceeds_95_percent_permutation_delta": None,
            "observed_exceeds_99_percent_permutation_delta": None,
            "observed_abs_score_exceeds_95_percent_permutation": None,
            "observed_abs_score_exceeds_99_percent_permutation": None,
            "status": "not_enough_allowed_indices_or_empty_contrast",
        }

    for repeat in range(repeats):
        values = np.zeros(len(y), dtype=float)
        chosen = rng.choice(allowed_indices, size=total, replace=False)
        pos = chosen[:pos_count]
        neg = chosen[pos_count:pos_count + neg_count]
        values[pos] = 1.0
        values[neg] = -1.0
        rows.append(
            audit_vector(
                f"{name}_host_restricted_permutation_{repeat}",
                values,
                "host_restricted_same_count_permutation",
                y,
                c_factor,
                fit,
                design_name,
                metadata={
                    "source_candidate": name,
                    "repeat": repeat,
                    "positive_count": pos_count,
                    "negative_count": neg_count,
                    "allowed_index_count": int(len(allowed_indices)),
                },
            )
        )

    observed = audit_vector(
        name,
        contrast,
        "observed_host_restricted_contrast",
        y,
        c_factor,
        fit,
        design_name,
        metadata={
            "positive_count": pos_count,
            "negative_count": neg_count,
            "allowed_index_count": int(len(allowed_indices)),
        },
    )

    deltas = np.asarray([row["delta_chi2_score"] for row in rows], dtype=float)
    scores = np.asarray([row["score"] for row in rows], dtype=float)

    summary = {
        "candidate": name,
        "design": design_name,
        "repeats": repeats,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "allowed_index_count": int(len(allowed_indices)),
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
        "status": "ok",
    }

    return rows, summary


def fold_pass(row):
    return bool(
        row.get("valid_fold") is True
        and row.get("direction_positive") is True
        and row.get("delta_chi2_score", 0.0) >= 5.0
        and row.get("observed_exceeds_95_percent_permutation_delta") is True
        and row.get("observed_abs_score_exceeds_95_percent_permutation") is True
    )


def run_fold_audits(label, contrast, mapped_rows, folds, y, c_factor, fit, design_name):
    fold_rows = []
    permutation_summaries = []
    permutation_detail_rows = []

    for fold_index, hosts in enumerate(folds, start=1):
        restricted, allowed_indices = restrict_vector_to_hosts(contrast, mapped_rows, hosts)
        name = f"{label}_host_fold_{fold_index}_of_{len(folds)}"
        audit = audit_vector(
            name,
            restricted,
            "host_fold_restricted_contrast",
            y,
            c_factor,
            fit,
            design_name,
            metadata={
                "fold_index": fold_index,
                "fold_count": len(folds),
                "host_count_in_fold": len(hosts),
                "hosts": ",".join(hosts),
                "positive_count": int(np.sum(restricted > 0)),
                "negative_count": int(np.sum(restricted < 0)),
                "allowed_index_count": len(allowed_indices),
            },
        )
        dstat = direction_stats(name, restricted, fit)
        perm_rows, perm_summary = host_restricted_same_count_permutation(
            name,
            restricted,
            allowed_indices,
            y,
            c_factor,
            fit,
            design_name,
        )
        permutation_detail_rows.extend(perm_rows)
        permutation_summaries.append(perm_summary)
        audit_with_perm = add_permutation_flags(audit, perm_summary) if perm_summary.get("status") == "ok" else dict(audit)

        row = {
            **audit_with_perm,
            "fold_index": fold_index,
            "fold_count": len(folds),
            "host_count_in_fold": len(hosts),
            "hosts": ",".join(hosts),
            "positive_count": int(np.sum(restricted > 0)),
            "negative_count": int(np.sum(restricted < 0)),
            "allowed_index_count": len(allowed_indices),
            "signed_mean_residual_difference_high_minus_low": dstat.get("signed_mean_residual_difference_high_minus_low"),
            "direction_positive": bool(dstat.get("high_side_more_positive") is True),
            "valid_fold": bool(int(np.sum(restricted > 0)) > 0 and int(np.sum(restricted < 0)) > 0),
            "permutation_status": perm_summary.get("status"),
        }
        row["fold_pass_directional"] = fold_pass(row)
        fold_rows.append(row)

    return fold_rows, permutation_summaries, permutation_detail_rows


def host_component_audits(label, contrast, mapped_rows, y, c_factor, fit, design_name):
    by_host = defaultdict(list)
    for row in mapped_rows:
        idx = int(row["compact_observation_index"])
        if contrast[idx] != 0:
            by_host[entity(row, "host_guess")].append(idx)

    rows = []
    for host, indices in by_host.items():
        values = np.zeros(len(y), dtype=float)
        for idx in indices:
            values[idx] = contrast[idx]
        audit = audit_vector(
            f"{label}_host_component_{host}",
            values,
            "host_component_contrast",
            y,
            c_factor,
            fit,
            design_name,
            metadata={
                "host": host,
                "positive_count": int(np.sum(values > 0)),
                "negative_count": int(np.sum(values < 0)),
                "nonzero_count": int(np.sum(values != 0)),
            },
        )
        dstat = direction_stats(f"{label}_host_component_{host}", values, fit)
        audit["signed_mean_residual_difference_high_minus_low"] = dstat.get("signed_mean_residual_difference_high_minus_low")
        audit["direction_positive"] = bool(dstat.get("high_side_more_positive") is True)
        rows.append(audit)

    return sorted(rows, key=lambda r: (-r.get("delta_chi2_score", 0.0), -r.get("nonzero_count", 0), r.get("host", "")))


def removal_audits(label, contrast, mapped_rows, component_rows, full_delta, y, c_factor, fit, design_name):
    rows = []
    ordered_hosts = [row["host"] for row in component_rows]
    removal_counts = [1, 2, 3, 5, 10, 15]

    for k in removal_counts:
        hosts = ordered_hosts[:k]
        if not hosts:
            continue
        reduced = remove_hosts_from_vector(contrast, mapped_rows, hosts)
        audit = audit_vector(
            f"{label}_without_top_{k}_host_components",
            reduced,
            "remove_top_host_components",
            y,
            c_factor,
            fit,
            design_name,
            metadata={
                "removed_host_count": len(hosts),
                "removed_hosts": ",".join(hosts),
                "remaining_positive_count": int(np.sum(reduced > 0)),
                "remaining_negative_count": int(np.sum(reduced < 0)),
            },
        )
        dstat = direction_stats(f"{label}_without_top_{k}_host_components", reduced, fit)
        drop = float(full_delta - audit.get("delta_chi2_score", 0.0))
        audit["signed_mean_residual_difference_high_minus_low"] = dstat.get("signed_mean_residual_difference_high_minus_low")
        audit["direction_positive"] = bool(dstat.get("high_side_more_positive") is True)
        audit["delta_drop_from_full"] = drop
        audit["delta_drop_fraction_from_full"] = float(drop / full_delta) if full_delta else None
        rows.append(audit)

    return rows


def stability_summary_for(label, full_audit, fold_rows, removal_rows):
    valid = [row for row in fold_rows if row.get("valid_fold") is True]
    pass_rows = [row for row in valid if row.get("fold_pass_directional") is True]
    positive_rows = [row for row in valid if row.get("direction_positive") is True]

    top3 = next((row for row in removal_rows if row.get("removed_host_count") == 3), None)
    top5 = next((row for row in removal_rows if row.get("removed_host_count") == 5), None)
    top10 = next((row for row in removal_rows if row.get("removed_host_count") == 10), None)

    return {
        "label": label,
        "full_delta_chi2": full_audit.get("delta_chi2_score"),
        "full_score": full_audit.get("score"),
        "full_nondegenerate_ratio": full_audit.get("nondegenerate_ratio"),
        "valid_fold_count": len(valid),
        "fold_pass_count": len(pass_rows),
        "fold_pass_fraction": float(len(pass_rows) / len(valid)) if valid else None,
        "fold_positive_direction_count": len(positive_rows),
        "fold_positive_direction_fraction": float(len(positive_rows) / len(valid)) if valid else None,
        "top3_removal_remaining_delta": top3.get("delta_chi2_score") if top3 else None,
        "top3_removal_drop_fraction": top3.get("delta_drop_fraction_from_full") if top3 else None,
        "top3_removal_direction_positive": top3.get("direction_positive") if top3 else None,
        "top5_removal_remaining_delta": top5.get("delta_chi2_score") if top5 else None,
        "top5_removal_drop_fraction": top5.get("delta_drop_fraction_from_full") if top5 else None,
        "top5_removal_direction_positive": top5.get("direction_positive") if top5 else None,
        "top10_removal_remaining_delta": top10.get("delta_chi2_score") if top10 else None,
        "top10_removal_drop_fraction": top10.get("delta_drop_fraction_from_full") if top10 else None,
        "top10_removal_direction_positive": top10.get("direction_positive") if top10 else None,
    }


def decide_status(global_stability, within_stability):
    def stable(s):
        return bool(
            s.get("valid_fold_count", 0) >= 4
            and (s.get("fold_positive_direction_fraction") or 0.0) >= 0.80
            and (s.get("fold_pass_fraction") or 0.0) >= 0.60
            and (s.get("top3_removal_drop_fraction") is None or s.get("top3_removal_drop_fraction") < 0.70)
            and s.get("top3_removal_direction_positive") is True
        )

    def directional(s):
        return bool(
            s.get("valid_fold_count", 0) >= 4
            and (s.get("fold_positive_direction_fraction") or 0.0) >= 0.60
            and (s.get("top3_removal_drop_fraction") is None or s.get("top3_removal_drop_fraction") < 0.85)
        )

    global_stable = stable(global_stability)
    within_stable = stable(within_stability)
    global_directional = directional(global_stability)
    within_directional = directional(within_stability)

    best_cases = {
        "global_stability": global_stability,
        "within_host_stability": within_stability,
        "global_stable": global_stable,
        "within_host_stable": within_stable,
        "global_directional": global_directional,
        "within_host_directional": within_directional,
    }

    if global_stable and within_stable:
        return (
            "f160w_polarity_stable_global_and_within_host",
            9,
            "F160W polarity survives host folds and top-host removal at both global and within-host scales.",
            best_cases,
        )

    if within_stable and not global_stable:
        return (
            "f160w_polarity_stable_within_host_not_global",
            8,
            "Within-host F160W polarity is stable, while global host-fold stability is weaker.",
            best_cases,
        )

    if global_stable and not within_stable:
        return (
            "f160w_polarity_stable_global_not_within_host",
            8,
            "Global F160W polarity is stable, while within-host stability is weaker.",
            best_cases,
        )

    if global_directional or within_directional:
        return (
            "f160w_polarity_directional_but_not_stably_distributed",
            7,
            "F160W polarity remains directional, but host-fold or removal stability is not locked.",
            best_cases,
        )

    return (
        "f160w_polarity_not_stable_across_host_splits",
        6,
        "The F160W polarity result does not replicate stably across host folds/removals.",
        best_cases,
    )


def make_plots(global_fold_rows, within_fold_rows, global_removal_rows, within_removal_rows):
    try:
        for label, rows, filename in [
            ("Global", global_fold_rows, "global_host_fold_delta_v0_3.png"),
            ("Within-host", within_fold_rows, "within_host_fold_delta_v0_3.png"),
        ]:
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: r.get("fold_index", 0))
            x = np.arange(len(rows))
            plt.figure(figsize=(9, 5))
            plt.bar(x, [r.get("delta_chi2_score", 0.0) for r in rows])
            plt.xticks(x, [str(r.get("fold_index")) for r in rows])
            plt.xlabel("host fold")
            plt.ylabel("delta chi2")
            plt.title(f"{label} F160W polarity by host fold")
            plt.tight_layout()
            plt.savefig(OUTDIR / filename, dpi=160)
            plt.close()

        for label, rows, filename in [
            ("Global", global_removal_rows, "global_top_host_removal_v0_3.png"),
            ("Within-host", within_removal_rows, "within_host_top_host_removal_v0_3.png"),
        ]:
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: r.get("removed_host_count", 0))
            x = np.arange(len(rows))
            plt.figure(figsize=(9, 5))
            plt.bar(x, [r.get("delta_chi2_score", 0.0) for r in rows])
            plt.xticks(x, [str(r.get("removed_host_count")) for r in rows])
            plt.xlabel("top host components removed")
            plt.ylabel("remaining delta chi2")
            plt.title(f"{label} F160W polarity after top-host removal")
            plt.tight_layout()
            plt.savefig(OUTDIR / filename, dpi=160)
            plt.close()
    except Exception as exc:
        write_json(OUTDIR / "plot_error_v0_3.json", {"error": repr(exc), "traceback": traceback.format_exc()})


def main():
    print("TAIRID Boundary Prediction Battery v0.3 starting.")
    print("Boundary: F160W host-fold stability audit only; not proof.")

    write_json(OUTDIR / "claims_v0_3.json", CLAIMS_V0_3)

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
        write_json(OUTDIR / "v15_import_repair_summary_v0_3.json", repair_summary)

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

        write_csv(OUTDIR / "download_ledger_v0_3.csv", ledger)
        write_json(OUTDIR / "download_attempts_v0_3.json", {"compact": downloads, "auxiliary": aux_results})

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

        write_json(OUTDIR / "parse_meta_v0_3.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v0_3.json", parse_errors)

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
                "test_name": "TAIRID Boundary Prediction Battery v0.3",
                "final_status": "boundary_polarity_battery_v0_3_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or Table2 retrieval before v0.3.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "boundary_polarity_battery_v0_3_summary.json", summary)
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
        write_csv(OUTDIR / "compact_row_map_v0_3.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v0_3.csv", cluster_rows)

        mapped_rows, map_status = map_table2_to_spine(row_rows, table2_data_rows)
        host_summary = summarize_hosts(mapped_rows)
        numeric_rows, likely_numeric_labels = numeric_feature_summary(mapped_rows)

        designs, control_metadata = build_designs(X, mapped_rows, host_summary, len(y))
        design_fits = {
            name: gls_fit(y, design["D"], c_factor)
            for name, design in designs.items()
        }
        full_fit = design_fits[FULL_DESIGN]

        global_low, global_high, global_contrast, global_low_rows, global_high_rows, global_meta = v02.build_global_edge_pair(
            mapped_rows,
            len(y),
            F160W_COLUMN,
            F160W_LABEL,
            percentile=EDGE_PERCENTILE,
        )
        within_low, within_high, within_contrast, within_low_rows, within_high_rows, within_meta = v02.build_within_host_edge_pair(
            mapped_rows,
            len(y),
            F160W_COLUMN,
            F160W_LABEL,
            percentile=EDGE_PERCENTILE,
        )

        write_csv(OUTDIR / "global_f160w_low_rows_v0_3.csv", global_low_rows)
        write_csv(OUTDIR / "global_f160w_high_rows_v0_3.csv", global_high_rows)
        write_csv(OUTDIR / "within_host_f160w_low_rows_v0_3.csv", within_low_rows)
        write_csv(OUTDIR / "within_host_f160w_high_rows_v0_3.csv", within_high_rows)

        global_full = audit_vector(
            "f160w_global_high_minus_low_contrast_full",
            global_contrast,
            "global_high_minus_low_contrast_full",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=global_meta,
        )
        within_full = audit_vector(
            "f160w_within_host_high_minus_low_contrast_full",
            within_contrast,
            "within_host_high_minus_low_contrast_full",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=within_meta,
        )

        global_direction = direction_stats("f160w_global_high_minus_low_contrast_full", global_contrast, full_fit)
        within_direction = direction_stats("f160w_within_host_high_minus_low_contrast_full", within_contrast, full_fit)
        write_json(OUTDIR / "full_contrast_direction_stats_v0_3.json", {
            "global": global_direction,
            "within_host": within_direction,
        })

        folds, fold_counts = make_balanced_host_folds(mapped_rows, N_FOLDS)
        fold_inventory = [
            {"fold_index": i + 1, "hosts": folds[i], "row_count_estimate": fold_counts[i]}
            for i in range(len(folds))
        ]
        write_json(OUTDIR / "host_fold_inventory_v0_3.json", fold_inventory)

        global_fold_rows, global_fold_perm_summaries, global_fold_perm_details = run_fold_audits(
            "f160w_global_high_minus_low",
            global_contrast,
            mapped_rows,
            folds,
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
        )
        within_fold_rows, within_fold_perm_summaries, within_fold_perm_details = run_fold_audits(
            "f160w_within_host_high_minus_low",
            within_contrast,
            mapped_rows,
            folds,
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
        )

        write_csv(OUTDIR / "global_host_fold_audit_v0_3.csv", global_fold_rows)
        write_csv(OUTDIR / "within_host_fold_audit_v0_3.csv", within_fold_rows)
        write_json(OUTDIR / "global_fold_permutation_summaries_v0_3.json", global_fold_perm_summaries)
        write_json(OUTDIR / "within_host_fold_permutation_summaries_v0_3.json", within_fold_perm_summaries)
        write_csv(OUTDIR / "global_fold_permutation_details_v0_3.csv", global_fold_perm_details)
        write_csv(OUTDIR / "within_host_fold_permutation_details_v0_3.csv", within_fold_perm_details)

        global_components = host_component_audits(
            "f160w_global_high_minus_low",
            global_contrast,
            mapped_rows,
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
        )
        within_components = host_component_audits(
            "f160w_within_host_high_minus_low",
            within_contrast,
            mapped_rows,
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
        )
        write_csv(OUTDIR / "global_host_component_audit_v0_3.csv", global_components)
        write_csv(OUTDIR / "within_host_component_audit_v0_3.csv", within_components)

        global_removal = removal_audits(
            "f160w_global_high_minus_low",
            global_contrast,
            mapped_rows,
            global_components,
            global_full.get("delta_chi2_score", 0.0),
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
        )
        within_removal = removal_audits(
            "f160w_within_host_high_minus_low",
            within_contrast,
            mapped_rows,
            within_components,
            within_full.get("delta_chi2_score", 0.0),
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
        )
        write_csv(OUTDIR / "global_top_host_removal_audit_v0_3.csv", global_removal)
        write_csv(OUTDIR / "within_host_top_host_removal_audit_v0_3.csv", within_removal)

        global_stability = stability_summary_for("global", global_full, global_fold_rows, global_removal)
        within_stability = stability_summary_for("within_host", within_full, within_fold_rows, within_removal)
        write_json(OUTDIR / "stability_summary_v0_3.json", {
            "global": global_stability,
            "within_host": within_stability,
        })

        final_status, readiness_score, next_wall, best_cases = decide_status(global_stability, within_stability)
        make_plots(global_fold_rows, within_fold_rows, global_removal, within_removal)

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
        write_csv(OUTDIR / "design_fit_comparison_v0_3.csv", design_fit_rows)

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
            "test_name": "TAIRID Boundary Prediction Battery v0.3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "F160W host-fold replication and leave-host-out stability audit only. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_3": CLAIMS_V0_3,
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
            "full_contrasts": {
                "global": global_full,
                "within_host": within_full,
                "global_direction": global_direction,
                "within_host_direction": within_direction,
            },
            "host_folds": fold_inventory,
            "global_fold_audit": global_fold_rows,
            "within_host_fold_audit": within_fold_rows,
            "global_host_components_top30": global_components[:30],
            "within_host_components_top30": within_components[:30],
            "global_top_host_removal": global_removal,
            "within_host_top_host_removal": within_removal,
            "stability_summary": {
                "global": global_stability,
                "within_host": within_stability,
            },
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_3_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_3_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v0_3.json"),
                "stability_summary_json": str(OUTDIR / "stability_summary_v0_3.json"),
                "global_fold_csv": str(OUTDIR / "global_host_fold_audit_v0_3.csv"),
                "within_fold_csv": str(OUTDIR / "within_host_fold_audit_v0_3.csv"),
                "global_removal_csv": str(OUTDIR / "global_top_host_removal_audit_v0_3.csv"),
                "within_removal_csv": str(OUTDIR / "within_host_top_host_removal_audit_v0_3.csv"),
                "plots": [
                    str(OUTDIR / "global_host_fold_delta_v0_3.png"),
                    str(OUTDIR / "within_host_fold_delta_v0_3.png"),
                    str(OUTDIR / "global_top_host_removal_v0_3.png"),
                    str(OUTDIR / "within_host_top_host_removal_v0_3.png"),
                ],
            },
            "interpretation": {
                "what_supports_stability": (
                    "Most host folds keep positive direction and beat fold-specific same-count permutation checks; "
                    "top-host removal does not collapse the contrast."
                ),
                "what_narrows_tairid": (
                    "Polarity is real in the pooled sample but not stable across host folds/removals, implying concentration."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only checks whether F160W edge-pair polarity is stable across host partitions."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_3_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_3_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.3\n\n")
            f.write("Boundary: F160W host-fold stability audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Claims v0.3:\n")
            f.write(json.dumps(CLAIMS_V0_3, indent=2, default=json_default) + "\n\n")
            f.write("Stability summary:\n")
            f.write(json.dumps(summary["stability_summary"], indent=2, default=json_default) + "\n\n")
            f.write("Full contrasts:\n")
            f.write(json.dumps(summary["full_contrasts"], indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only tests F160W polarity stability across host partitions.\n")

        print("TAIRID Boundary Prediction Battery v0.3 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_3_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_3_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
