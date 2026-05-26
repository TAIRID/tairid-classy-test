#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.3.1
Global edge-balanced host-fold repair.

Purpose:
v0.3 showed strong within-host F160W polarity stability, but the global fold
audit was weakened because several host folds contained only one global edge
side: high/F160W-faint rows but no low/F160W-bright rows.

This v0.3.1 test repairs only the global fold construction.

It asks:
    Does the global F160W high-minus-low edge-pair polarity survive when host
    folds are constructed to contain both global bright-edge and faint-edge rows?

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only tests whether the global F160W polarity result survives edge-balanced
host-fold replication.
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
import run_tairid_boundary_polarity_battery_v0_3 as v03


OUTDIR = Path("tairid_boundary_polarity_battery_v0_3_1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

FULL_DESIGN = "plus_host_top10_row_order_measurement_controls"
ORIGINAL_DESIGN = "original_47"

F160W_COLUMN = "table2_num_7"
F160W_LABEL = "f160w_like"
EDGE_PERCENTILE = 5
N_FOLDS = 5
PERMUTATION_REPEATS = 100
SEED = 42


CLAIMS_V0_3_1 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.3.1",
    "scope": "SH0ES Table2 global F160W edge-balanced host-fold repair",
    "reason_for_test": (
        "v0.3 found strong within-host F160W polarity stability. The global fold audit was weaker "
        "because ordinary balanced host folds produced invalid folds with only one global edge side. "
        "v0.3.1 repairs that by constructing host folds with both global bright-edge and faint-edge rows when possible."
    ),
    "native_tairid_claim": (
        "If global F160W polarity is a distributed boundary-polarity surface, not only a pooled artifact, "
        "then edge-balanced host folds should preserve positive high-minus-low residual direction."
    ),
    "primary_prediction": (
        "Most edge-balanced host folds should contain both global edges, preserve positive signed direction, "
        "and beat fold-specific same-count permutation checks."
    ),
    "failure_rule": (
        "If edge-balanced folds still fail, then global polarity remains real in the pooled layer but does not "
        "earn a distributed global-fold stability claim. The stronger TAIRID claim stays at the within-host field-relative scale."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only tests whether the global F160W polarity survives better fold construction."
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


def host_edge_counts(mapped_rows, contrast):
    counts = defaultdict(
        lambda: {
            "host": None,
            "positive_count": 0,
            "negative_count": 0,
            "edge_count": 0,
            "row_count": 0,
        }
    )

    for row in mapped_rows:
        host = entity(row, "host_guess")
        idx = int(row["compact_observation_index"])
        value = float(contrast[idx])

        counts[host]["host"] = host
        counts[host]["row_count"] += 1

        if value > 0:
            counts[host]["positive_count"] += 1
            counts[host]["edge_count"] += 1
        elif value < 0:
            counts[host]["negative_count"] += 1
            counts[host]["edge_count"] += 1

    rows = list(counts.values())
    rows = sorted(
        rows,
        key=lambda r: (
            -r["edge_count"],
            -r["row_count"],
            r["host"],
        ),
    )

    return rows


def fold_score_after_add(fold, host_row):
    pos = fold["positive_count"] + host_row["positive_count"]
    neg = fold["negative_count"] + host_row["negative_count"]
    edge = fold["edge_count"] + host_row["edge_count"]
    rows = fold["row_count"] + host_row["row_count"]

    invalid_penalty = 0
    if edge > 0 and (pos == 0 or neg == 0):
        invalid_penalty = 100000

    balance_penalty = abs(pos - neg) * 100
    size_penalty = rows
    edge_penalty = abs(edge - 2 * min(pos, neg)) * 10

    return invalid_penalty + balance_penalty + edge_penalty + size_penalty


def add_host_to_fold(fold, host_row):
    fold["hosts"].append(host_row["host"])
    fold["positive_count"] += host_row["positive_count"]
    fold["negative_count"] += host_row["negative_count"]
    fold["edge_count"] += host_row["edge_count"]
    fold["row_count"] += host_row["row_count"]


def make_edge_balanced_host_folds(mapped_rows, contrast, n_folds=N_FOLDS):
    hosts = host_edge_counts(mapped_rows, contrast)

    folds = [
        {
            "fold_index": i + 1,
            "hosts": [],
            "positive_count": 0,
            "negative_count": 0,
            "edge_count": 0,
            "row_count": 0,
        }
        for i in range(n_folds)
    ]

    edge_hosts = [h for h in hosts if h["edge_count"] > 0]
    zero_edge_hosts = [h for h in hosts if h["edge_count"] == 0]

    mixed_hosts = [h for h in edge_hosts if h["positive_count"] > 0 and h["negative_count"] > 0]
    pos_only_hosts = [h for h in edge_hosts if h["positive_count"] > 0 and h["negative_count"] == 0]
    neg_only_hosts = [h for h in edge_hosts if h["negative_count"] > 0 and h["positive_count"] == 0]

    mixed_hosts = sorted(mixed_hosts, key=lambda r: (-r["edge_count"], -r["row_count"], r["host"]))
    pos_only_hosts = sorted(pos_only_hosts, key=lambda r: (-r["positive_count"], -r["row_count"], r["host"]))
    neg_only_hosts = sorted(neg_only_hosts, key=lambda r: (-r["negative_count"], -r["row_count"], r["host"]))

    assigned = set()

    # First seed folds with mixed hosts when available.
    for host_row in mixed_hosts:
        fold = min(folds, key=lambda f: (f["edge_count"], f["row_count"]))
        add_host_to_fold(fold, host_row)
        assigned.add(host_row["host"])

    # Then make sure folds missing positive edge rows get a positive host.
    for fold in folds:
        if fold["positive_count"] == 0:
            candidate = next((h for h in pos_only_hosts if h["host"] not in assigned), None)
            if candidate is not None:
                add_host_to_fold(fold, candidate)
                assigned.add(candidate["host"])

    # Then make sure folds missing negative edge rows get a negative host.
    for fold in folds:
        if fold["negative_count"] == 0:
            candidate = next((h for h in neg_only_hosts if h["host"] not in assigned), None)
            if candidate is not None:
                add_host_to_fold(fold, candidate)
                assigned.add(candidate["host"])

    remaining_edge = [h for h in edge_hosts if h["host"] not in assigned]

    for host_row in remaining_edge:
        fold = min(folds, key=lambda f: fold_score_after_add(f, host_row))
        add_host_to_fold(fold, host_row)
        assigned.add(host_row["host"])

    # Add non-edge hosts last, only to balance row counts.
    for host_row in zero_edge_hosts:
        fold = min(folds, key=lambda f: f["row_count"])
        add_host_to_fold(fold, host_row)
        assigned.add(host_row["host"])

    fold_rows = []
    for fold in folds:
        fold_rows.append(
            {
                "fold_index": fold["fold_index"],
                "hosts": sorted(fold["hosts"]),
                "host_count": len(fold["hosts"]),
                "row_count": fold["row_count"],
                "positive_count": fold["positive_count"],
                "negative_count": fold["negative_count"],
                "edge_count": fold["edge_count"],
                "valid_edge_contrast": bool(fold["positive_count"] > 0 and fold["negative_count"] > 0),
            }
        )

    return [row["hosts"] for row in fold_rows], fold_rows, hosts


def restrict_vector_to_hosts(vector, mapped_rows, hosts):
    hosts = set(hosts)
    keep_indices = set()

    for row in mapped_rows:
        if entity(row, "host_guess") in hosts:
            keep_indices.add(int(row["compact_observation_index"]))

    out = np.zeros_like(np.asarray(vector, dtype=float))

    for idx in keep_indices:
        out[idx] = vector[idx]

    return out, sorted(keep_indices)


def host_restricted_same_count_permutation(name, contrast, allowed_indices, y, c_factor, fit, design_name):
    rng = np.random.default_rng(SEED)

    contrast = np.asarray(contrast, dtype=float).reshape(-1)
    allowed_indices = np.asarray(sorted(set(int(i) for i in allowed_indices)), dtype=int)

    pos_count = int(np.sum(contrast > 0))
    neg_count = int(np.sum(contrast < 0))
    total = pos_count + neg_count

    rows = []

    observed = audit_vector(
        name,
        contrast,
        "observed_edge_balanced_host_fold_contrast",
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

    if total <= 0 or len(allowed_indices) < total or pos_count <= 0 or neg_count <= 0:
        return rows, {
            "candidate": name,
            "design": design_name,
            "status": "not_enough_allowed_indices_or_empty_contrast",
            "repeats": PERMUTATION_REPEATS,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "allowed_index_count": int(len(allowed_indices)),
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

    for repeat in range(PERMUTATION_REPEATS):
        values = np.zeros(len(y), dtype=float)
        chosen = rng.choice(allowed_indices, size=total, replace=False)
        pos = chosen[:pos_count]
        neg = chosen[pos_count:pos_count + neg_count]

        values[pos] = 1.0
        values[neg] = -1.0

        rows.append(
            audit_vector(
                f"{name}_edge_balanced_permutation_{repeat}",
                values,
                "edge_balanced_same_count_permutation",
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

    deltas = np.asarray([row["delta_chi2_score"] for row in rows], dtype=float)
    scores = np.asarray([row["score"] for row in rows], dtype=float)

    return rows, {
        "candidate": name,
        "design": design_name,
        "status": "ok",
        "repeats": PERMUTATION_REPEATS,
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
    }


def fold_pass(row):
    return bool(
        row.get("valid_fold") is True
        and row.get("direction_positive") is True
        and row.get("delta_chi2_score", 0.0) >= 5.0
        and row.get("observed_exceeds_95_percent_permutation_delta") is True
        and row.get("observed_abs_score_exceeds_95_percent_permutation") is True
    )


def run_edge_balanced_fold_audits(label, contrast, mapped_rows, folds, y, c_factor, fit, design_name):
    fold_rows = []
    permutation_summaries = []
    permutation_details = []

    for fold_index, hosts in enumerate(folds, start=1):
        restricted, allowed_indices = restrict_vector_to_hosts(contrast, mapped_rows, hosts)
        name = f"{label}_edge_balanced_host_fold_{fold_index}_of_{len(folds)}"

        audit = audit_vector(
            name,
            restricted,
            "edge_balanced_host_fold_restricted_contrast",
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

        permutation_details.extend(perm_rows)
        permutation_summaries.append(perm_summary)

        if perm_summary.get("status") == "ok":
            audit_with_perm = add_permutation_flags(audit, perm_summary)
        else:
            audit_with_perm = dict(audit)

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

    return fold_rows, permutation_summaries, permutation_details


def remove_hosts_from_vector(vector, mapped_rows, hosts):
    hosts = set(hosts)
    out = np.asarray(vector, dtype=float).copy()

    for row in mapped_rows:
        if entity(row, "host_guess") in hosts:
            out[int(row["compact_observation_index"])] = 0.0

    return out


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

    rows = sorted(
        rows,
        key=lambda r: (
            -r.get("delta_chi2_score", 0.0),
            -r.get("nonzero_count", 0),
            r.get("host", ""),
        ),
    )

    return rows


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


def stability_summary(full_audit, fold_rows, removal_rows):
    valid = [row for row in fold_rows if row.get("valid_fold") is True]
    pass_rows = [row for row in valid if row.get("fold_pass_directional") is True]
    positive_rows = [row for row in valid if row.get("direction_positive") is True]

    top3 = next((row for row in removal_rows if row.get("removed_host_count") == 3), None)
    top5 = next((row for row in removal_rows if row.get("removed_host_count") == 5), None)
    top10 = next((row for row in removal_rows if row.get("removed_host_count") == 10), None)

    return {
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


def decide_status(stability):
    stable = bool(
        stability.get("valid_fold_count", 0) >= 5
        and (stability.get("fold_positive_direction_fraction") or 0.0) >= 0.80
        and (stability.get("fold_pass_fraction") or 0.0) >= 0.60
        and (stability.get("top3_removal_drop_fraction") is None or stability.get("top3_removal_drop_fraction") < 0.70)
        and stability.get("top3_removal_direction_positive") is True
    )

    directional = bool(
        stability.get("valid_fold_count", 0) >= 4
        and (stability.get("fold_positive_direction_fraction") or 0.0) >= 0.60
        and (stability.get("top3_removal_drop_fraction") is None or stability.get("top3_removal_drop_fraction") < 0.85)
    )

    best_cases = {
        "global_edge_balanced_stability": stability,
        "stable": stable,
        "directional": directional,
    }

    if stable:
        return (
            "global_f160w_polarity_stable_after_edge_balanced_folds",
            9,
            "Global F160W polarity survives edge-balanced host folds and top-host removal.",
            best_cases,
        )

    if directional:
        return (
            "global_f160w_polarity_directional_after_edge_balanced_folds",
            8,
            "Global F160W polarity remains directional with edge-balanced folds, but the stable pass gate is not fully locked.",
            best_cases,
        )

    return (
        "global_f160w_polarity_not_stable_after_edge_balanced_folds",
        6,
        "Global F160W polarity does not survive edge-balanced host-fold replication strongly enough.",
        best_cases,
    )


def make_plots(fold_rows, removal_rows):
    try:
        if fold_rows:
            rows = sorted(fold_rows, key=lambda r: r.get("fold_index", 0))
            x = np.arange(len(rows))

            plt.figure(figsize=(9, 5))
            plt.bar(x, [r.get("delta_chi2_score", 0.0) for r in rows])
            plt.xticks(x, [str(r.get("fold_index")) for r in rows])
            plt.xlabel("edge-balanced host fold")
            plt.ylabel("delta chi2")
            plt.title("Global F160W polarity by edge-balanced host fold")
            plt.tight_layout()
            plt.savefig(OUTDIR / "global_edge_balanced_fold_delta_v0_3_1.png", dpi=160)
            plt.close()

            plt.figure(figsize=(9, 5))
            plt.bar(x, [r.get("signed_mean_residual_difference_high_minus_low", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [str(r.get("fold_index")) for r in rows])
            plt.xlabel("edge-balanced host fold")
            plt.ylabel("high-minus-low mean residual")
            plt.title("Global F160W signed direction by edge-balanced fold")
            plt.tight_layout()
            plt.savefig(OUTDIR / "global_edge_balanced_fold_direction_v0_3_1.png", dpi=160)
            plt.close()

        if removal_rows:
            rows = sorted(removal_rows, key=lambda r: r.get("removed_host_count", 0))
            x = np.arange(len(rows))

            plt.figure(figsize=(9, 5))
            plt.bar(x, [r.get("delta_chi2_score", 0.0) for r in rows])
            plt.xticks(x, [str(r.get("removed_host_count")) for r in rows])
            plt.xlabel("top host components removed")
            plt.ylabel("remaining delta chi2")
            plt.title("Global F160W polarity after top-host removal")
            plt.tight_layout()
            plt.savefig(OUTDIR / "global_edge_balanced_top_host_removal_v0_3_1.png", dpi=160)
            plt.close()

    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v0_3_1.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )


def main():
    print("TAIRID Boundary Prediction Battery v0.3.1 starting.")
    print("Boundary: global edge-balanced host-fold repair only; not proof.")

    write_json(OUTDIR / "claims_v0_3_1.json", CLAIMS_V0_3_1)

    repair_summary = {}

    try:
        v16.OUTDIR = OUTDIR
        v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        b01.OUTDIR = OUTDIR
        b01.DOWNLOAD_DIR = DOWNLOAD_DIR
        v02.OUTDIR = OUTDIR
        v02.DOWNLOAD_DIR = DOWNLOAD_DIR
        v03.OUTDIR = OUTDIR
        v03.DOWNLOAD_DIR = DOWNLOAD_DIR
        b01.PERMUTATION_REPEATS = PERMUTATION_REPEATS
        v02.PERMUTATION_REPEATS = PERMUTATION_REPEATS

        ns, repair_summary = v16.load_v15_helpers()
        write_json(OUTDIR / "v15_import_repair_summary_v0_3_1.json", repair_summary)

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

        write_csv(OUTDIR / "download_ledger_v0_3_1.csv", ledger)
        write_json(OUTDIR / "download_attempts_v0_3_1.json", {"compact": downloads, "auxiliary": aux_results})

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

        write_json(OUTDIR / "parse_meta_v0_3_1.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v0_3_1.json", parse_errors)

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
                "test_name": "TAIRID Boundary Prediction Battery v0.3.1",
                "final_status": "boundary_polarity_battery_v0_3_1_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or Table2 retrieval before v0.3.1.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "boundary_polarity_battery_v0_3_1_summary.json", summary)
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

        write_csv(OUTDIR / "compact_row_map_v0_3_1.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v0_3_1.csv", cluster_rows)

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

        write_csv(OUTDIR / "global_f160w_low_rows_v0_3_1.csv", global_low_rows)
        write_csv(OUTDIR / "global_f160w_high_rows_v0_3_1.csv", global_high_rows)

        full_audit = audit_vector(
            "f160w_global_high_minus_low_contrast_full",
            global_contrast,
            "global_high_minus_low_contrast_full",
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
            metadata=global_meta,
        )
        full_direction = direction_stats(
            "f160w_global_high_minus_low_contrast_full",
            global_contrast,
            full_fit,
        )

        folds, fold_inventory, host_edge_inventory = make_edge_balanced_host_folds(
            mapped_rows,
            global_contrast,
            N_FOLDS,
        )
        write_json(OUTDIR / "edge_balanced_host_fold_inventory_v0_3_1.json", fold_inventory)
        write_csv(OUTDIR / "host_edge_inventory_v0_3_1.csv", host_edge_inventory)

        fold_rows, fold_perm_summaries, fold_perm_details = run_edge_balanced_fold_audits(
            "f160w_global_high_minus_low",
            global_contrast,
            mapped_rows,
            folds,
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
        )

        write_csv(OUTDIR / "global_edge_balanced_host_fold_audit_v0_3_1.csv", fold_rows)
        write_json(OUTDIR / "global_edge_balanced_fold_permutation_summaries_v0_3_1.json", fold_perm_summaries)
        write_csv(OUTDIR / "global_edge_balanced_fold_permutation_details_v0_3_1.csv", fold_perm_details)

        components = host_component_audits(
            "f160w_global_high_minus_low",
            global_contrast,
            mapped_rows,
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
        )
        write_csv(OUTDIR / "global_host_component_audit_v0_3_1.csv", components)

        removal = removal_audits(
            "f160w_global_high_minus_low",
            global_contrast,
            mapped_rows,
            components,
            full_audit.get("delta_chi2_score", 0.0),
            y,
            c_factor,
            full_fit,
            FULL_DESIGN,
        )
        write_csv(OUTDIR / "global_top_host_removal_audit_v0_3_1.csv", removal)

        stability = stability_summary(full_audit, fold_rows, removal)
        write_json(OUTDIR / "global_edge_balanced_stability_summary_v0_3_1.json", stability)

        final_status, readiness_score, next_wall, best_cases = decide_status(stability)
        make_plots(fold_rows, removal)

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

        write_csv(OUTDIR / "design_fit_comparison_v0_3_1.csv", design_fit_rows)

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
            "test_name": "TAIRID Boundary Prediction Battery v0.3.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Global edge-balanced host-fold repair only. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_3_1": CLAIMS_V0_3_1,
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
            "full_global_contrast": full_audit,
            "full_global_direction": full_direction,
            "edge_balanced_host_folds": fold_inventory,
            "host_edge_inventory_top40": host_edge_inventory[:40],
            "fold_audit": fold_rows,
            "top_host_components": components[:30],
            "top_host_removal": removal,
            "stability_summary": stability,
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_3_1_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_3_1_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v0_3_1.json"),
                "stability_json": str(OUTDIR / "global_edge_balanced_stability_summary_v0_3_1.json"),
                "fold_csv": str(OUTDIR / "global_edge_balanced_host_fold_audit_v0_3_1.csv"),
                "removal_csv": str(OUTDIR / "global_top_host_removal_audit_v0_3_1.csv"),
                "host_edge_inventory_csv": str(OUTDIR / "host_edge_inventory_v0_3_1.csv"),
                "plots": [
                    str(OUTDIR / "global_edge_balanced_fold_delta_v0_3_1.png"),
                    str(OUTDIR / "global_edge_balanced_fold_direction_v0_3_1.png"),
                    str(OUTDIR / "global_edge_balanced_top_host_removal_v0_3_1.png"),
                ],
            },
            "interpretation": {
                "what_supports_global_stability": (
                    "Edge-balanced folds all contain both edge sides, preserve positive signed direction, "
                    "and beat fold-specific same-count permutation checks."
                ),
                "what_narrows_tairid": (
                    "If edge-balanced folds remain weak, global polarity remains pooled-scale only, "
                    "while the stronger claim stays within-host field-relative from v0.3."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only tests whether global F160W polarity survives edge-balanced host-fold replication."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_3_1_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_3_1_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.3.1\n\n")
            f.write("Boundary: global edge-balanced host-fold repair only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Claims v0.3.1:\n")
            f.write(json.dumps(CLAIMS_V0_3_1, indent=2, default=json_default) + "\n\n")
            f.write("Full global contrast:\n")
            f.write(json.dumps(full_audit, indent=2, default=json_default) + "\n\n")
            f.write("Full global direction:\n")
            f.write(json.dumps(full_direction, indent=2, default=json_default) + "\n\n")
            f.write("Stability summary:\n")
            f.write(json.dumps(stability, indent=2, default=json_default) + "\n\n")
            f.write("Edge-balanced host folds:\n")
            f.write(json.dumps(fold_inventory, indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only tests global F160W polarity under edge-balanced host folds.\n")

        print("TAIRID Boundary Prediction Battery v0.3.1 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.3.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_3_1_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_3_1_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
