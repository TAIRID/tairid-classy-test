#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.5
Cross-validated held-out-host prediction test.

Purpose:
v0.2.1 showed F160W was the strongest and most specific signed edge-pair polarity variable.
v0.3 showed within-host F160W polarity was stable across host folds.
v0.4 showed within-host F160W polarity survived anchor and non-anchor stratification.

This v0.5 test asks a stronger question:

    Can the within-host F160W polarity coefficient be learned on training hosts
    and predict residual direction / residual improvement on held-out hosts?

Core TAIRID prediction:
    If the within-host F160W high-minus-low edge is a real field-relative
    boundary-polarity structure, then a coefficient learned on some host fields
    should carry predictive signal into held-out host fields.

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only tests held-out-host predictive transfer of the within-host F160W polarity structure.
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
from scipy.linalg import cho_solve

import run_tairid_table2_f160w_faint_tail_host_field_v1_6 as v16
import run_tairid_boundary_polarity_battery_v0_2 as v02


OUTDIR = Path("tairid_boundary_polarity_battery_v0_5_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

FULL_DESIGN = "plus_host_top10_row_order_measurement_controls"
ORIGINAL_DESIGN = "original_47"

F160W_COLUMN = "table2_num_7"
F160W_LABEL = "f160w_like"
EDGE_PERCENTILE = 5
N_FOLDS = 5
PERMUTATION_REPEATS = 120
SEED = 42
EPS = 1.0e-12

CLAIMS_V0_5 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.5",
    "scope": "SH0ES Table2 within-host F160W held-out-host prediction",
    "native_tairid_claim": (
        "If within-host F160W polarity is a field-relative boundary structure, "
        "then the learned high-minus-low coefficient should transfer from training hosts "
        "to held-out hosts better than same-count random edge assignments."
    ),
    "primary_prediction": (
        "Most held-out host folds should show positive fixed-coefficient gain, positive signed direction, "
        "and should beat held-out same-count permutation controls."
    ),
    "anti_ad_hoc_rule": (
        "The tested variable is predeclared: within-host F160W high-minus-low contrast. "
        "The model must train on host folds and evaluate on held-out hosts without refitting the held-out coefficient."
    ),
    "failure_rule": (
        "If held-out fixed-coefficient gains fail, the claim narrows back to detected residual geometry "
        "rather than predictive transferable structure."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only tests whether the within-host F160W polarity has held-out predictive transfer."
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


def host_counts_for_contrast(mapped_rows, contrast):
    counts = defaultdict(lambda: {"host": None, "positive_count": 0, "negative_count": 0, "edge_count": 0, "row_count": 0})

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
    rows = sorted(rows, key=lambda r: (-r["edge_count"], -r["row_count"], r["host"]))

    return rows


def active_hosts_for_contrast(mapped_rows, contrast):
    rows = host_counts_for_contrast(mapped_rows, contrast)

    return [
        row for row in rows
        if row["positive_count"] > 0 and row["negative_count"] > 0
    ]


def make_balanced_active_host_folds(mapped_rows, contrast, n_folds=N_FOLDS):
    hosts = active_hosts_for_contrast(mapped_rows, contrast)

    folds = [
        {
            "fold_index": i + 1,
            "hosts": [],
            "edge_count": 0,
            "row_count": 0,
            "positive_count": 0,
            "negative_count": 0,
        }
        for i in range(n_folds)
    ]

    for host_row in hosts:
        fold = min(
            folds,
            key=lambda f: (
                f["edge_count"],
                f["row_count"],
                abs((f["positive_count"] + host_row["positive_count"]) - (f["negative_count"] + host_row["negative_count"])),
            ),
        )

        fold["hosts"].append(host_row["host"])
        fold["edge_count"] += host_row["edge_count"]
        fold["row_count"] += host_row["row_count"]
        fold["positive_count"] += host_row["positive_count"]
        fold["negative_count"] += host_row["negative_count"]

    inventory = []

    for fold in folds:
        inventory.append(
            {
                "fold_index": fold["fold_index"],
                "hosts": sorted(fold["hosts"]),
                "host_count": len(fold["hosts"]),
                "edge_count": fold["edge_count"],
                "row_count": fold["row_count"],
                "positive_count": fold["positive_count"],
                "negative_count": fold["negative_count"],
                "valid_fold": bool(fold["positive_count"] > 0 and fold["negative_count"] > 0),
            }
        )

    return [row["hosts"] for row in inventory], inventory, hosts


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


def host_index_map(mapped_rows, hosts):
    hosts = set(hosts)
    host_to_indices = defaultdict(list)

    for row in mapped_rows:
        host = entity(row, "host_guess")

        if host in hosts:
            host_to_indices[host].append(int(row["compact_observation_index"]))

    return host_to_indices


def residualized_raw_vector_stats(name, raw_values, c_factor, fit, alpha_fixed=None):
    raw = np.asarray(raw_values, dtype=float).reshape(-1)
    c_inv_raw = cho_solve(c_factor, raw, check_finite=False)

    raw_norm2 = float(raw.T @ c_inv_raw)

    base = {
        "candidate": name,
        "raw_positive_count": int(np.sum(raw > 0)),
        "raw_negative_count": int(np.sum(raw < 0)),
        "raw_nonzero_count": int(np.sum(raw != 0)),
        "raw_cinv_norm2": raw_norm2,
    }

    if raw_norm2 <= EPS:
        base.update(
            {
                "status": "zero_raw_norm",
                "residualized_norm2": 0.0,
                "score": 0.0,
                "alpha_fit": 0.0,
                "self_fit_delta_chi2": 0.0,
                "fixed_alpha": alpha_fixed,
                "fixed_alpha_gain": 0.0,
            }
        )
        return base

    x_t_cinv_raw = fit["D"].T @ c_inv_raw
    coeff = fit["normal_inv"] @ x_t_cinv_raw

    raw_perp = raw - fit["D"] @ coeff
    c_inv_raw_perp = c_inv_raw - fit["Cinv_D"] @ coeff

    norm2 = float(raw_perp.T @ c_inv_raw_perp)

    if norm2 <= EPS:
        score = 0.0
        alpha_fit = 0.0
        self_delta = 0.0
    else:
        score = float(raw_perp.T @ fit["Cinv_residual"])
        alpha_fit = float(score / norm2)
        self_delta = float((score * score) / norm2)

    if alpha_fixed is None:
        fixed_gain = None
    else:
        fixed_gain = float(2.0 * alpha_fixed * score - (alpha_fixed * alpha_fixed) * norm2)

    base.update(
        {
            "status": "ok",
            "residualized_norm2": norm2,
            "score": score,
            "alpha_fit": alpha_fit,
            "self_fit_delta_chi2": self_delta,
            "fixed_alpha": alpha_fixed,
            "fixed_alpha_gain": fixed_gain,
            "nondegenerate_ratio_raw": float(math.sqrt(max(norm2, 0.0)) / max(math.sqrt(max(raw_norm2, 0.0)), EPS)),
        }
    )

    return base


def direction_stats(name, contrast_values, fit):
    raw = np.asarray(contrast_values, dtype=float).reshape(-1)
    residual = np.asarray(fit["residual"], dtype=float).reshape(-1)
    cinv_residual = np.asarray(fit["Cinv_residual"], dtype=float).reshape(-1)

    pos = raw > 0
    neg = raw < 0

    def block(mask):
        if not np.any(mask):
            return {
                "count": 0,
                "mean_residual": None,
                "median_residual": None,
                "mean_abs_residual": None,
                "mean_cinv_residual": None,
            }

        r = residual[mask]
        cr = cinv_residual[mask]

        return {
            "count": int(np.sum(mask)),
            "mean_residual": float(np.mean(r)),
            "median_residual": float(np.median(r)),
            "mean_abs_residual": float(np.mean(np.abs(r))),
            "mean_cinv_residual": float(np.mean(cr)),
        }

    high = block(pos)
    low = block(neg)

    mean_diff = (
        float(high["mean_residual"] - low["mean_residual"])
        if high["count"] and low["count"]
        else None
    )

    cinv_diff = (
        float(high["mean_cinv_residual"] - low["mean_cinv_residual"])
        if high["count"] and low["count"]
        else None
    )

    return {
        "candidate": name,
        "high_count": high["count"],
        "low_count": low["count"],
        "high_mean_residual": high["mean_residual"],
        "low_mean_residual": low["mean_residual"],
        "high_mean_abs_residual": high["mean_abs_residual"],
        "low_mean_abs_residual": low["mean_abs_residual"],
        "signed_mean_residual_difference_high_minus_low": mean_diff,
        "signed_mean_cinv_residual_difference_high_minus_low": cinv_diff,
        "high_side_more_positive": bool(mean_diff is not None and mean_diff > 0.0),
    }


def host_preserving_test_permutations(name, observed_test_vector, test_hosts, mapped_rows, y, c_factor, fit, alpha_train):
    rng = np.random.default_rng(SEED)

    observed_test_vector = np.asarray(observed_test_vector, dtype=float).reshape(-1)
    host_to_indices = host_index_map(mapped_rows, test_hosts)

    host_to_pos = Counter()
    host_to_neg = Counter()

    for row in mapped_rows:
        host = entity(row, "host_guess")

        if host not in set(test_hosts):
            continue

        idx = int(row["compact_observation_index"])

        if observed_test_vector[idx] > 0:
            host_to_pos[host] += 1
        elif observed_test_vector[idx] < 0:
            host_to_neg[host] += 1

    rows = []

    for repeat in range(PERMUTATION_REPEATS):
        values = np.zeros(len(y), dtype=float)

        for host in test_hosts:
            indices = np.asarray(host_to_indices.get(host, []), dtype=int)
            pos_count = int(host_to_pos[host])
            neg_count = int(host_to_neg[host])
            total = pos_count + neg_count

            if total <= 0 or len(indices) < total:
                continue

            chosen = rng.choice(indices, size=total, replace=False)
            pos = chosen[:pos_count]
            neg = chosen[pos_count:pos_count + neg_count]

            values[pos] = 1.0
            values[neg] = -1.0

        stat = residualized_raw_vector_stats(
            f"{name}_permutation_{repeat}",
            values,
            c_factor,
            fit,
            alpha_fixed=alpha_train,
        )
        stat["repeat"] = repeat
        rows.append(stat)

    fixed_gains = np.asarray(
        [row.get("fixed_alpha_gain", 0.0) for row in rows],
        dtype=float,
    )
    scores = np.asarray(
        [row.get("score", 0.0) for row in rows],
        dtype=float,
    )

    if len(rows) == 0:
        return rows, {
            "candidate": name,
            "status": "no_permutation_rows",
            "repeats": PERMUTATION_REPEATS,
            "permutation_fixed_gain_95": None,
            "permutation_fixed_gain_99": None,
            "permutation_abs_score_95": None,
            "permutation_abs_score_99": None,
        }

    return rows, {
        "candidate": name,
        "status": "ok",
        "repeats": PERMUTATION_REPEATS,
        "permutation_fixed_gain_mean": float(np.mean(fixed_gains)),
        "permutation_fixed_gain_95": float(np.percentile(fixed_gains, 95)),
        "permutation_fixed_gain_99": float(np.percentile(fixed_gains, 99)),
        "permutation_abs_score_mean": float(np.mean(np.abs(scores))),
        "permutation_abs_score_95": float(np.percentile(np.abs(scores), 95)),
        "permutation_abs_score_99": float(np.percentile(np.abs(scores), 99)),
    }


def run_fold_prediction(fold_index, test_hosts, all_active_hosts, contrast, mapped_rows, y, c_factor, fit):
    test_hosts = sorted(test_hosts)
    train_hosts = sorted([host for host in all_active_hosts if host not in set(test_hosts)])

    train_vector, train_allowed = restrict_vector_to_hosts(contrast, mapped_rows, train_hosts)
    test_vector, test_allowed = restrict_vector_to_hosts(contrast, mapped_rows, test_hosts)

    train_stats = residualized_raw_vector_stats(
        f"fold_{fold_index}_train",
        train_vector,
        c_factor,
        fit,
        alpha_fixed=None,
    )

    alpha_train = float(train_stats.get("alpha_fit", 0.0))

    test_stats = residualized_raw_vector_stats(
        f"fold_{fold_index}_heldout",
        test_vector,
        c_factor,
        fit,
        alpha_fixed=alpha_train,
    )

    test_direction = direction_stats(
        f"fold_{fold_index}_heldout",
        test_vector,
        fit,
    )

    perm_rows, perm_summary = host_preserving_test_permutations(
        f"fold_{fold_index}_heldout",
        test_vector,
        test_hosts,
        mapped_rows,
        y,
        c_factor,
        fit,
        alpha_train,
    )

    fixed_gain = test_stats.get("fixed_alpha_gain", 0.0)
    score = test_stats.get("score", 0.0)

    pass_95 = bool(
        perm_summary.get("status") == "ok"
        and fixed_gain is not None
        and fixed_gain > perm_summary.get("permutation_fixed_gain_95", float("inf"))
        and abs(score) > perm_summary.get("permutation_abs_score_95", float("inf"))
        and alpha_train > 0.0
        and test_direction.get("high_side_more_positive") is True
    )

    pass_99 = bool(
        perm_summary.get("status") == "ok"
        and fixed_gain is not None
        and fixed_gain > perm_summary.get("permutation_fixed_gain_99", float("inf"))
        and abs(score) > perm_summary.get("permutation_abs_score_99", float("inf"))
        and alpha_train > 0.0
        and test_direction.get("high_side_more_positive") is True
    )

    row = {
        "fold_index": fold_index,
        "test_hosts": ",".join(test_hosts),
        "train_host_count": len(train_hosts),
        "test_host_count": len(test_hosts),
        "train_positive_count": int(np.sum(train_vector > 0)),
        "train_negative_count": int(np.sum(train_vector < 0)),
        "test_positive_count": int(np.sum(test_vector > 0)),
        "test_negative_count": int(np.sum(test_vector < 0)),
        "train_allowed_index_count": len(train_allowed),
        "test_allowed_index_count": len(test_allowed),
        "alpha_train": alpha_train,
        "train_self_fit_delta_chi2": train_stats.get("self_fit_delta_chi2"),
        "test_score": test_stats.get("score"),
        "test_self_fit_delta_chi2": test_stats.get("self_fit_delta_chi2"),
        "test_fixed_alpha_gain": fixed_gain,
        "test_nondegenerate_ratio_raw": test_stats.get("nondegenerate_ratio_raw"),
        "test_signed_mean_residual_difference_high_minus_low": test_direction.get("signed_mean_residual_difference_high_minus_low"),
        "test_direction_positive": bool(test_direction.get("high_side_more_positive") is True),
        "permutation_status": perm_summary.get("status"),
        "permutation_fixed_gain_95": perm_summary.get("permutation_fixed_gain_95"),
        "permutation_fixed_gain_99": perm_summary.get("permutation_fixed_gain_99"),
        "permutation_abs_score_95": perm_summary.get("permutation_abs_score_95"),
        "permutation_abs_score_99": perm_summary.get("permutation_abs_score_99"),
        "heldout_pass_95": pass_95,
        "heldout_pass_99": pass_99,
        "heldout_gain_positive": bool(fixed_gain is not None and fixed_gain > 0.0),
        "alpha_train_positive": bool(alpha_train > 0.0),
    }

    return row, train_stats, test_stats, test_direction, perm_summary, perm_rows


def run_individual_host_predictions(active_host_rows, contrast, mapped_rows, y, c_factor, fit):
    active_hosts = [row["host"] for row in active_host_rows]
    rows = []

    for host in active_hosts:
        train_hosts = [h for h in active_hosts if h != host]
        test_hosts = [host]

        train_vector, _ = restrict_vector_to_hosts(contrast, mapped_rows, train_hosts)
        test_vector, _ = restrict_vector_to_hosts(contrast, mapped_rows, test_hosts)

        train_stats = residualized_raw_vector_stats(
            f"loho_train_without_{host}",
            train_vector,
            c_factor,
            fit,
            alpha_fixed=None,
        )
        alpha_train = float(train_stats.get("alpha_fit", 0.0))

        test_stats = residualized_raw_vector_stats(
            f"loho_test_{host}",
            test_vector,
            c_factor,
            fit,
            alpha_fixed=alpha_train,
        )
        dstat = direction_stats(f"loho_test_{host}", test_vector, fit)

        rows.append(
            {
                "heldout_host": host,
                "host_edge_count": next((r["edge_count"] for r in active_host_rows if r["host"] == host), None),
                "host_positive_count": int(np.sum(test_vector > 0)),
                "host_negative_count": int(np.sum(test_vector < 0)),
                "alpha_train": alpha_train,
                "test_score": test_stats.get("score"),
                "test_self_fit_delta_chi2": test_stats.get("self_fit_delta_chi2"),
                "test_fixed_alpha_gain": test_stats.get("fixed_alpha_gain"),
                "test_direction_positive": bool(dstat.get("high_side_more_positive") is True),
                "test_signed_mean_residual_difference_high_minus_low": dstat.get("signed_mean_residual_difference_high_minus_low"),
                "heldout_gain_positive": bool((test_stats.get("fixed_alpha_gain") or 0.0) > 0.0),
                "alpha_train_positive": bool(alpha_train > 0.0),
            }
        )

    rows = sorted(
        rows,
        key=lambda r: (
            -r.get("test_fixed_alpha_gain", 0.0),
            -r.get("host_edge_count", 0),
            r["heldout_host"],
        ),
    )

    return rows


def decide_status(fold_rows, loho_rows):
    valid = [row for row in fold_rows if row["test_positive_count"] > 0 and row["test_negative_count"] > 0]
    pass_95 = [row for row in valid if row.get("heldout_pass_95") is True]
    pass_99 = [row for row in valid if row.get("heldout_pass_99") is True]
    positive_gain = [row for row in valid if row.get("heldout_gain_positive") is True]
    positive_direction = [row for row in valid if row.get("test_direction_positive") is True]
    positive_alpha = [row for row in valid if row.get("alpha_train_positive") is True]

    total_fixed_gain = float(np.sum([row.get("test_fixed_alpha_gain", 0.0) or 0.0 for row in valid]))
    mean_fixed_gain = float(np.mean([row.get("test_fixed_alpha_gain", 0.0) or 0.0 for row in valid])) if valid else 0.0

    loho_valid = [row for row in loho_rows if row["host_positive_count"] > 0 and row["host_negative_count"] > 0]
    loho_positive_gain = [row for row in loho_valid if row.get("heldout_gain_positive") is True]
    loho_positive_direction = [row for row in loho_valid if row.get("test_direction_positive") is True]

    summary = {
        "valid_fold_count": len(valid),
        "heldout_pass_95_count": len(pass_95),
        "heldout_pass_99_count": len(pass_99),
        "positive_gain_fold_count": len(positive_gain),
        "positive_direction_fold_count": len(positive_direction),
        "positive_alpha_fold_count": len(positive_alpha),
        "heldout_pass_95_fraction": float(len(pass_95) / len(valid)) if valid else None,
        "heldout_pass_99_fraction": float(len(pass_99) / len(valid)) if valid else None,
        "positive_gain_fraction": float(len(positive_gain) / len(valid)) if valid else None,
        "positive_direction_fraction": float(len(positive_direction) / len(valid)) if valid else None,
        "positive_alpha_fraction": float(len(positive_alpha) / len(valid)) if valid else None,
        "total_fixed_gain": total_fixed_gain,
        "mean_fixed_gain": mean_fixed_gain,
        "loho_valid_host_count": len(loho_valid),
        "loho_positive_gain_count": len(loho_positive_gain),
        "loho_positive_direction_count": len(loho_positive_direction),
        "loho_positive_gain_fraction": float(len(loho_positive_gain) / len(loho_valid)) if loho_valid else None,
        "loho_positive_direction_fraction": float(len(loho_positive_direction) / len(loho_valid)) if loho_valid else None,
    }

    best_cases = {
        "fold_summary": summary,
        "fold_rows": fold_rows,
        "top_positive_loho": loho_rows[:20],
        "bottom_loho": sorted(loho_rows, key=lambda r: r.get("test_fixed_alpha_gain", 0.0))[:20],
    }

    if (
        len(valid) >= 5
        and len(pass_99) >= 4
        and len(positive_gain) == len(valid)
        and len(positive_direction) == len(valid)
        and total_fixed_gain > 0.0
    ):
        return (
            "heldout_host_prediction_supported_strong",
            9,
            "Within-host F160W polarity coefficient transfers to held-out host folds and beats 99% permutation in most folds.",
            summary,
            best_cases,
        )

    if (
        len(valid) >= 5
        and len(pass_95) >= 4
        and len(positive_gain) >= 4
        and len(positive_direction) >= 4
        and total_fixed_gain > 0.0
    ):
        return (
            "heldout_host_prediction_supported",
            8,
            "Within-host F160W polarity coefficient transfers to held-out host folds and beats 95% permutation in most folds.",
            summary,
            best_cases,
        )

    if (
        len(valid) >= 5
        and len(positive_gain) >= 4
        and len(positive_direction) >= 4
        and total_fixed_gain > 0.0
    ):
        return (
            "heldout_host_prediction_directional_not_locked",
            7,
            "Held-out host prediction is directionally positive, but it does not beat permutation strongly enough to lock.",
            summary,
            best_cases,
        )

    return (
        "heldout_host_prediction_not_supported",
        6,
        "Within-host F160W polarity detection does not transfer strongly to held-out hosts under v0.5.",
        summary,
        best_cases,
    )


def make_plots(fold_rows, loho_rows):
    try:
        if fold_rows:
            rows = sorted(fold_rows, key=lambda r: r["fold_index"])
            x = np.arange(len(rows))

            plt.figure(figsize=(9, 5))
            plt.bar(x, [r.get("test_fixed_alpha_gain", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [str(r["fold_index"]) for r in rows])
            plt.xlabel("held-out host fold")
            plt.ylabel("fixed-coefficient held-out gain")
            plt.title("v0.5 held-out fold prediction gain")
            plt.tight_layout()
            plt.savefig(OUTDIR / "heldout_fold_fixed_gain_v0_5.png", dpi=160)
            plt.close()

            plt.figure(figsize=(9, 5))
            plt.bar(x, [r.get("test_signed_mean_residual_difference_high_minus_low", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [str(r["fold_index"]) for r in rows])
            plt.xlabel("held-out host fold")
            plt.ylabel("high-minus-low mean residual")
            plt.title("v0.5 held-out fold signed direction")
            plt.tight_layout()
            plt.savefig(OUTDIR / "heldout_fold_signed_direction_v0_5.png", dpi=160)
            plt.close()

        if loho_rows:
            rows = sorted(loho_rows, key=lambda r: -r.get("test_fixed_alpha_gain", 0.0))[:30]
            x = np.arange(len(rows))

            plt.figure(figsize=(13, 5))
            plt.bar(x, [r.get("test_fixed_alpha_gain", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [r["heldout_host"] for r in rows], rotation=60, ha="right", fontsize=8)
            plt.ylabel("fixed-coefficient gain")
            plt.title("Top leave-one-host-out prediction gains")
            plt.tight_layout()
            plt.savefig(OUTDIR / "loho_top_fixed_gain_v0_5.png", dpi=160)
            plt.close()

    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v0_5.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )


def main():
    print("TAIRID Boundary Prediction Battery v0.5 starting.")
    print("Boundary: held-out-host prediction audit only; not proof.")

    write_json(OUTDIR / "claims_v0_5.json", CLAIMS_V0_5)

    repair_summary = {}

    try:
        v16.OUTDIR = OUTDIR
        v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        v02.OUTDIR = OUTDIR
        v02.DOWNLOAD_DIR = DOWNLOAD_DIR

        ns, repair_summary = v16.load_v15_helpers()
        write_json(OUTDIR / "v15_import_repair_summary_v0_5.json", repair_summary)

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

        write_csv(OUTDIR / "download_ledger_v0_5.csv", ledger)
        write_json(OUTDIR / "download_attempts_v0_5.json", {"compact": downloads, "auxiliary": aux_results})

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

        write_json(OUTDIR / "parse_meta_v0_5.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v0_5.json", parse_errors)

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
                "test_name": "TAIRID Boundary Prediction Battery v0.5",
                "final_status": "boundary_polarity_battery_v0_5_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or Table2 retrieval before v0.5.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "boundary_polarity_battery_v0_5_summary.json", summary)
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

        write_csv(OUTDIR / "compact_row_map_v0_5.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v0_5.csv", cluster_rows)

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

        write_csv(OUTDIR / "within_host_f160w_low_rows_v0_5.csv", within_low_rows)
        write_csv(OUTDIR / "within_host_f160w_high_rows_v0_5.csv", within_high_rows)

        active_host_rows = active_hosts_for_contrast(mapped_rows, within_contrast)
        all_active_hosts = [row["host"] for row in active_host_rows]

        folds, fold_inventory, active_host_inventory = make_balanced_active_host_folds(
            mapped_rows,
            within_contrast,
            N_FOLDS,
        )

        write_json(OUTDIR / "active_host_inventory_v0_5.json", active_host_inventory)
        write_json(OUTDIR / "heldout_fold_inventory_v0_5.json", fold_inventory)

        full_stats = residualized_raw_vector_stats(
            "within_host_f160w_high_minus_low_full",
            within_contrast,
            c_factor,
            full_fit,
            alpha_fixed=None,
        )
        full_direction = direction_stats(
            "within_host_f160w_high_minus_low_full",
            within_contrast,
            full_fit,
        )

        fold_rows = []
        train_stats_rows = []
        test_stats_rows = []
        direction_rows = []
        permutation_summaries = []
        permutation_detail_rows = []

        for fold_index, test_hosts in enumerate(folds, start=1):
            row, train_stats, test_stats, test_direction, perm_summary, perm_rows = run_fold_prediction(
                fold_index,
                test_hosts,
                all_active_hosts,
                within_contrast,
                mapped_rows,
                y,
                c_factor,
                full_fit,
            )

            fold_rows.append(row)
            train_stats_rows.append(train_stats)
            test_stats_rows.append(test_stats)
            direction_rows.append(test_direction)
            permutation_summaries.append(perm_summary)
            permutation_detail_rows.extend(perm_rows)

        write_csv(OUTDIR / "heldout_fold_prediction_summary_v0_5.csv", fold_rows)
        write_csv(OUTDIR / "heldout_fold_train_stats_v0_5.csv", train_stats_rows)
        write_csv(OUTDIR / "heldout_fold_test_stats_v0_5.csv", test_stats_rows)
        write_json(OUTDIR / "heldout_fold_direction_stats_v0_5.json", direction_rows)
        write_json(OUTDIR / "heldout_fold_permutation_summaries_v0_5.json", permutation_summaries)
        write_csv(OUTDIR / "heldout_fold_permutation_details_v0_5.csv", permutation_detail_rows)

        loho_rows = run_individual_host_predictions(
            active_host_rows,
            within_contrast,
            mapped_rows,
            y,
            c_factor,
            full_fit,
        )
        write_csv(OUTDIR / "leave_one_host_out_prediction_summary_v0_5.csv", loho_rows)

        final_status, readiness_score, next_wall, heldout_summary, best_cases = decide_status(
            fold_rows,
            loho_rows,
        )
        write_json(OUTDIR / "heldout_prediction_decision_summary_v0_5.json", heldout_summary)

        make_plots(fold_rows, loho_rows)

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

        write_csv(OUTDIR / "design_fit_comparison_v0_5.csv", design_fit_rows)

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
            "test_name": "TAIRID Boundary Prediction Battery v0.5",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Held-out-host prediction audit only. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_5": CLAIMS_V0_5,
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
            "full_within_host_raw_stats": full_stats,
            "full_within_host_direction": full_direction,
            "active_host_inventory": active_host_inventory,
            "heldout_fold_inventory": fold_inventory,
            "heldout_fold_predictions": fold_rows,
            "heldout_prediction_decision_summary": heldout_summary,
            "leave_one_host_out_predictions_top30": loho_rows[:30],
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_5_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_5_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v0_5.json"),
                "heldout_fold_summary_csv": str(OUTDIR / "heldout_fold_prediction_summary_v0_5.csv"),
                "heldout_decision_json": str(OUTDIR / "heldout_prediction_decision_summary_v0_5.json"),
                "loho_summary_csv": str(OUTDIR / "leave_one_host_out_prediction_summary_v0_5.csv"),
                "permutation_summaries_json": str(OUTDIR / "heldout_fold_permutation_summaries_v0_5.json"),
                "plots": [
                    str(OUTDIR / "heldout_fold_fixed_gain_v0_5.png"),
                    str(OUTDIR / "heldout_fold_signed_direction_v0_5.png"),
                    str(OUTDIR / "loho_top_fixed_gain_v0_5.png"),
                ],
            },
            "interpretation": {
                "what_supports_predictive_transfer": (
                    "Training-host coefficient remains positive and produces positive fixed-coefficient gain "
                    "on held-out host folds, beating host-preserving same-count permutation controls."
                ),
                "what_narrows_tairid": (
                    "Detected within-host polarity remains real, but held-out fixed-coefficient transfer fails. "
                    "That would keep this lane at residual-structure detection rather than predictive correction."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only checks whether within-host F160W polarity transfers predictively across held-out host fields."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_5_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_5_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.5\n\n")
            f.write("Boundary: held-out-host prediction audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Claims v0.5:\n")
            f.write(json.dumps(CLAIMS_V0_5, indent=2, default=json_default) + "\n\n")
            f.write("Held-out decision summary:\n")
            f.write(json.dumps(heldout_summary, indent=2, default=json_default) + "\n\n")
            f.write("Held-out fold predictions:\n")
            f.write(json.dumps(fold_rows, indent=2, default=json_default) + "\n\n")
            f.write("Full within-host raw stats:\n")
            f.write(json.dumps(full_stats, indent=2, default=json_default) + "\n\n")
            f.write("Full within-host direction:\n")
            f.write(json.dumps(full_direction, indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only tests held-out-host predictive transfer of within-host F160W polarity.\n")

        print("TAIRID Boundary Prediction Battery v0.5 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.5",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_5_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_5_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
