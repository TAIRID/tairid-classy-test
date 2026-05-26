#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.7
Regime-aware two-coefficient held-out prediction audit.

Purpose:
v0.6 showed that clean-host transfer remains stable after isolating stress hosts,
and that LMC / M31 / N4536 show distinct coefficient or gain behavior.

This v0.7 test asks:

    Does a regime-aware two-coefficient model predict held-out host folds better
    than one universal coefficient?

Regimes tested before looking at v0.7 outputs:
    A. stress3 = LMC + M31 + N4536
    B. stress4 = LMC + M31 + N4536 + SMC

For each held-out fold:
    1. Train one universal alpha on all training hosts.
    2. Train clean alpha on clean training hosts.
    3. Train low/stress alpha on stress-regime training hosts.
    4. Evaluate held-out hosts without refitting held-out coefficients.
    5. Compare universal fixed-gain versus regime-aware fixed-gain.
    6. Compare against host-preserving same-count permutations.

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only tests whether the v0.6 stress-host decomposition improves held-out prediction.
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
import run_tairid_boundary_polarity_battery_v0_2 as v02
import run_tairid_boundary_polarity_battery_v0_5 as v05
import run_tairid_boundary_polarity_battery_v0_6 as v06


OUTDIR = Path("tairid_boundary_polarity_battery_v0_7_outputs")
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

REGIME_DEFINITIONS = [
    {
        "regime_name": "stress3_LMC_M31_N4536",
        "low_alpha_hosts_declared": ["LMC", "M31", "N4536"],
        "plain_meaning": "v0.5.1 / v0.6 stress hosts only.",
    },
    {
        "regime_name": "stress4_LMC_M31_N4536_SMC",
        "low_alpha_hosts_declared": ["LMC", "M31", "N4536", "SMC"],
        "plain_meaning": "stress hosts plus SMC watch-case as low-alpha anchor-like host.",
    },
]

CLAIMS_V0_7 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.7",
    "scope": "Regime-aware two-coefficient held-out prediction audit",
    "native_tairid_claim": (
        "If v0.6 correctly identified a field-relative regime split, then a clean-host coefficient "
        "plus a low-alpha/stress coefficient should predict held-out hosts better than one universal coefficient."
    ),
    "primary_prediction": (
        "A predeclared two-coefficient regime should improve held-out fixed-coefficient gain over the universal coefficient "
        "without creating new broad failures."
    ),
    "regime_A": "stress3 = LMC + M31 + N4536",
    "regime_B": "stress4 = LMC + M31 + N4536 + SMC",
    "anti_ad_hoc_rule": (
        "Only these two regime splits are tested. A new host cannot be added after seeing the v0.7 result."
    ),
    "failure_rule": (
        "If two-coefficient prediction does not improve held-out gain over the universal coefficient, "
        "the v0.6 stress-host result stays descriptive rather than becoming a predictive regime model."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only tests whether regime-aware coefficients improve held-out prediction."
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
    out = defaultdict(list)

    for row in mapped_rows:
        host = entity(row, "host_guess")
        idx = int(row["compact_observation_index"])

        if host in hosts:
            out[host].append(idx)

    return out


def active_host_rows(mapped_rows, contrast):
    return [
        row for row in v06.host_counts_for_contrast(mapped_rows, contrast)
        if row["positive_count"] > 0 and row["negative_count"] > 0
    ]


def make_balanced_folds(host_rows, n_folds=N_FOLDS):
    rows = sorted(host_rows, key=lambda r: (-r["edge_count"], -r["row_count"], r["host"]))

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

    for row in rows:
        fold = min(folds, key=lambda f: (f["edge_count"], f["row_count"]))
        fold["hosts"].append(row["host"])
        fold["edge_count"] += row["edge_count"]
        fold["row_count"] += row["row_count"]
        fold["positive_count"] += row["positive_count"]
        fold["negative_count"] += row["negative_count"]

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

    return [row["hosts"] for row in inventory], inventory


def cohort_stats(name, hosts, contrast, mapped_rows, c_factor, fit, alpha_fixed=None):
    vector, allowed = restrict_vector_to_hosts(contrast, mapped_rows, hosts)

    stats = v05.residualized_raw_vector_stats(
        name,
        vector,
        c_factor,
        fit,
        alpha_fixed=alpha_fixed,
    )
    dstat = v05.direction_stats(name, vector, fit)

    stats.update(
        {
            "cohort": name,
            "host_count": len(set(hosts)),
            "hosts": ",".join(sorted(set(hosts))),
            "allowed_index_count": len(allowed),
            "positive_count": int(np.sum(vector > 0)),
            "negative_count": int(np.sum(vector < 0)),
            "edge_count": int(np.sum(vector != 0)),
            "signed_mean_residual_difference_high_minus_low": dstat.get(
                "signed_mean_residual_difference_high_minus_low"
            ),
            "direction_positive": bool(dstat.get("high_side_more_positive") is True),
            "high_mean_residual": dstat.get("high_mean_residual"),
            "low_mean_residual": dstat.get("low_mean_residual"),
        }
    )

    return stats, vector


def split_hosts_by_regime(hosts, low_alpha_hosts):
    low_alpha_hosts = set(low_alpha_hosts)
    clean = sorted([h for h in hosts if h not in low_alpha_hosts])
    low = sorted([h for h in hosts if h in low_alpha_hosts])

    return clean, low


def fixed_gain_for_split(
    name,
    test_hosts,
    low_alpha_hosts,
    contrast,
    mapped_rows,
    c_factor,
    fit,
    alpha_universal,
    alpha_clean,
    alpha_low,
):
    clean_hosts, low_hosts = split_hosts_by_regime(test_hosts, low_alpha_hosts)

    universal_stats, universal_vector = cohort_stats(
        f"{name}_universal_test",
        test_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=alpha_universal,
    )
    clean_stats, clean_vector = cohort_stats(
        f"{name}_clean_part",
        clean_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=alpha_clean,
    )
    low_stats, low_vector = cohort_stats(
        f"{name}_low_part",
        low_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=alpha_low,
    )

    regime_gain = float((clean_stats.get("fixed_alpha_gain") or 0.0) + (low_stats.get("fixed_alpha_gain") or 0.0))
    universal_gain = float(universal_stats.get("fixed_alpha_gain") or 0.0)

    all_direction = v05.direction_stats(f"{name}_direction_all", universal_vector, fit)
    clean_direction = v05.direction_stats(f"{name}_direction_clean", clean_vector, fit)
    low_direction = v05.direction_stats(f"{name}_direction_low", low_vector, fit)

    result = {
        "name": name,
        "test_hosts": ",".join(sorted(test_hosts)),
        "test_host_count": len(test_hosts),
        "test_clean_hosts": ",".join(clean_hosts),
        "test_low_alpha_hosts": ",".join(low_hosts),
        "test_clean_host_count": len(clean_hosts),
        "test_low_host_count": len(low_hosts),
        "alpha_universal": alpha_universal,
        "alpha_clean": alpha_clean,
        "alpha_low": alpha_low,
        "universal_fixed_gain": universal_gain,
        "regime_fixed_gain": regime_gain,
        "regime_minus_universal_gain": float(regime_gain - universal_gain),
        "universal_score": universal_stats.get("score"),
        "clean_score": clean_stats.get("score"),
        "low_score": low_stats.get("score"),
        "universal_self_fit_delta_chi2": universal_stats.get("self_fit_delta_chi2"),
        "clean_self_fit_delta_chi2": clean_stats.get("self_fit_delta_chi2"),
        "low_self_fit_delta_chi2": low_stats.get("self_fit_delta_chi2"),
        "all_direction_positive": bool(all_direction.get("high_side_more_positive") is True),
        "clean_direction_positive": bool(clean_direction.get("high_side_more_positive") is True),
        "low_direction_positive": bool(low_direction.get("high_side_more_positive") is True),
        "all_signed_difference": all_direction.get("signed_mean_residual_difference_high_minus_low"),
        "clean_signed_difference": clean_direction.get("signed_mean_residual_difference_high_minus_low"),
        "low_signed_difference": low_direction.get("signed_mean_residual_difference_high_minus_low"),
        "positive_count": universal_stats.get("positive_count"),
        "negative_count": universal_stats.get("negative_count"),
        "clean_positive_count": clean_stats.get("positive_count"),
        "clean_negative_count": clean_stats.get("negative_count"),
        "low_positive_count": low_stats.get("positive_count"),
        "low_negative_count": low_stats.get("negative_count"),
    }

    return result


def train_alphas(
    name,
    train_hosts,
    low_alpha_hosts,
    contrast,
    mapped_rows,
    c_factor,
    fit,
):
    clean_train, low_train = split_hosts_by_regime(train_hosts, low_alpha_hosts)

    universal_stats, _ = cohort_stats(
        f"{name}_train_universal",
        train_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=None,
    )
    clean_stats, _ = cohort_stats(
        f"{name}_train_clean",
        clean_train,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=None,
    )
    low_stats, _ = cohort_stats(
        f"{name}_train_low",
        low_train,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=None,
    )

    alpha_universal = float(universal_stats.get("alpha_fit", 0.0))
    alpha_clean = float(clean_stats.get("alpha_fit", 0.0)) if clean_train else alpha_universal
    alpha_low = float(low_stats.get("alpha_fit", 0.0)) if low_train else alpha_universal

    return {
        "alpha_universal": alpha_universal,
        "alpha_clean": alpha_clean,
        "alpha_low": alpha_low,
        "train_host_count": len(train_hosts),
        "train_clean_host_count": len(clean_train),
        "train_low_host_count": len(low_train),
        "train_hosts": ",".join(sorted(train_hosts)),
        "train_clean_hosts": ",".join(sorted(clean_train)),
        "train_low_hosts": ",".join(sorted(low_train)),
        "universal_train_delta": universal_stats.get("self_fit_delta_chi2"),
        "clean_train_delta": clean_stats.get("self_fit_delta_chi2"),
        "low_train_delta": low_stats.get("self_fit_delta_chi2"),
        "alpha_clean_minus_low": float(alpha_clean - alpha_low),
        "alpha_clean_minus_low_abs": float(abs(alpha_clean - alpha_low)),
    }


def build_host_preserving_random_vector(test_hosts, observed_vector, mapped_rows, rng):
    test_hosts = set(test_hosts)
    host_to_indices = host_index_map(mapped_rows, test_hosts)

    pos_count = Counter()
    neg_count = Counter()

    for row in mapped_rows:
        host = entity(row, "host_guess")

        if host not in test_hosts:
            continue

        idx = int(row["compact_observation_index"])

        if observed_vector[idx] > 0:
            pos_count[host] += 1
        elif observed_vector[idx] < 0:
            neg_count[host] += 1

    values = np.zeros_like(np.asarray(observed_vector, dtype=float))

    for host in sorted(test_hosts):
        indices = np.asarray(host_to_indices.get(host, []), dtype=int)
        p = int(pos_count[host])
        n = int(neg_count[host])
        total = p + n

        if total <= 0 or len(indices) < total:
            continue

        chosen = rng.choice(indices, size=total, replace=False)
        values[chosen[:p]] = 1.0
        values[chosen[p:p + n]] = -1.0

    return values


def permutation_comparison(
    name,
    test_hosts,
    low_alpha_hosts,
    observed_vector,
    mapped_rows,
    c_factor,
    fit,
    alpha_universal,
    alpha_clean,
    alpha_low,
):
    rng = np.random.default_rng(SEED)
    rows = []

    for repeat in range(PERMUTATION_REPEATS):
        values = build_host_preserving_random_vector(
            test_hosts,
            observed_vector,
            mapped_rows,
            rng,
        )

        result = fixed_gain_for_split(
            f"{name}_permutation_{repeat}",
            test_hosts,
            low_alpha_hosts,
            values,
            mapped_rows,
            c_factor,
            fit,
            alpha_universal,
            alpha_clean,
            alpha_low,
        )
        result["repeat"] = repeat
        rows.append(result)

    if not rows:
        return rows, {
            "candidate": name,
            "status": "no_permutation_rows",
        }

    regime_gains = np.asarray([row.get("regime_fixed_gain", 0.0) for row in rows], dtype=float)
    universal_gains = np.asarray([row.get("universal_fixed_gain", 0.0) for row in rows], dtype=float)
    improvements = np.asarray([row.get("regime_minus_universal_gain", 0.0) for row in rows], dtype=float)

    summary = {
        "candidate": name,
        "status": "ok",
        "repeats": PERMUTATION_REPEATS,
        "regime_gain_mean": float(np.mean(regime_gains)),
        "regime_gain_95": float(np.percentile(regime_gains, 95)),
        "regime_gain_99": float(np.percentile(regime_gains, 99)),
        "universal_gain_mean": float(np.mean(universal_gains)),
        "universal_gain_95": float(np.percentile(universal_gains, 95)),
        "universal_gain_99": float(np.percentile(universal_gains, 99)),
        "improvement_mean": float(np.mean(improvements)),
        "improvement_95": float(np.percentile(improvements, 95)),
        "improvement_99": float(np.percentile(improvements, 99)),
    }

    return rows, summary


def evaluate_fold(
    regime_name,
    fold_index,
    test_hosts,
    active_hosts,
    low_alpha_hosts,
    contrast,
    mapped_rows,
    c_factor,
    fit,
):
    train_hosts = sorted([h for h in active_hosts if h not in set(test_hosts)])
    train = train_alphas(
        f"{regime_name}_fold_{fold_index}",
        train_hosts,
        low_alpha_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
    )

    observed = fixed_gain_for_split(
        f"{regime_name}_fold_{fold_index}_observed",
        test_hosts,
        low_alpha_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        train["alpha_universal"],
        train["alpha_clean"],
        train["alpha_low"],
    )

    observed_vector, _ = restrict_vector_to_hosts(contrast, mapped_rows, test_hosts)

    perm_rows, perm_summary = permutation_comparison(
        f"{regime_name}_fold_{fold_index}",
        test_hosts,
        low_alpha_hosts,
        observed_vector,
        mapped_rows,
        c_factor,
        fit,
        train["alpha_universal"],
        train["alpha_clean"],
        train["alpha_low"],
    )

    row = {
        "regime_name": regime_name,
        "fold_index": fold_index,
        **train,
        **observed,
        "permutation_status": perm_summary.get("status"),
        "permutation_regime_gain_95": perm_summary.get("regime_gain_95"),
        "permutation_regime_gain_99": perm_summary.get("regime_gain_99"),
        "permutation_improvement_95": perm_summary.get("improvement_95"),
        "permutation_improvement_99": perm_summary.get("improvement_99"),
    }

    row["regime_gain_positive"] = bool(row.get("regime_fixed_gain", 0.0) > 0.0)
    row["universal_gain_positive"] = bool(row.get("universal_fixed_gain", 0.0) > 0.0)
    row["regime_improves_universal"] = bool(row.get("regime_minus_universal_gain", 0.0) > 0.0)
    row["regime_beats_95_permutation"] = bool(
        perm_summary.get("status") == "ok"
        and row.get("regime_fixed_gain", 0.0) > (perm_summary.get("regime_gain_95") or float("inf"))
    )
    row["regime_beats_99_permutation"] = bool(
        perm_summary.get("status") == "ok"
        and row.get("regime_fixed_gain", 0.0) > (perm_summary.get("regime_gain_99") or float("inf"))
    )
    row["improvement_beats_95_permutation"] = bool(
        perm_summary.get("status") == "ok"
        and row.get("regime_minus_universal_gain", 0.0) > (perm_summary.get("improvement_95") or float("inf"))
    )
    row["improvement_beats_99_permutation"] = bool(
        perm_summary.get("status") == "ok"
        and row.get("regime_minus_universal_gain", 0.0) > (perm_summary.get("improvement_99") or float("inf"))
    )
    row["strict_regime_pass_95"] = bool(
        row["regime_gain_positive"]
        and row["regime_improves_universal"]
        and row["regime_beats_95_permutation"]
    )
    row["strict_regime_pass_99"] = bool(
        row["regime_gain_positive"]
        and row["regime_improves_universal"]
        and row["regime_beats_99_permutation"]
    )

    return row, perm_summary, perm_rows


def evaluate_loho(
    regime_name,
    host,
    active_hosts,
    low_alpha_hosts,
    contrast,
    mapped_rows,
    c_factor,
    fit,
):
    train_hosts = sorted([h for h in active_hosts if h != host])
    train = train_alphas(
        f"{regime_name}_loho_{host}",
        train_hosts,
        low_alpha_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
    )
    observed = fixed_gain_for_split(
        f"{regime_name}_loho_{host}_observed",
        [host],
        low_alpha_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        train["alpha_universal"],
        train["alpha_clean"],
        train["alpha_low"],
    )

    row = {
        "regime_name": regime_name,
        "heldout_host": host,
        "host_is_low_alpha_regime": bool(host in set(low_alpha_hosts)),
        **train,
        **observed,
    }
    row["regime_gain_positive"] = bool(row.get("regime_fixed_gain", 0.0) > 0.0)
    row["universal_gain_positive"] = bool(row.get("universal_fixed_gain", 0.0) > 0.0)
    row["regime_improves_universal"] = bool(row.get("regime_minus_universal_gain", 0.0) > 0.0)

    return row


def summarize_regime(regime_name, low_alpha_hosts, fold_rows, loho_rows):
    valid_folds = [
        row for row in fold_rows
        if row["regime_name"] == regime_name
        and (row.get("positive_count") or 0) > 0
        and (row.get("negative_count") or 0) > 0
    ]
    valid_loho = [
        row for row in loho_rows
        if row["regime_name"] == regime_name
        and (row.get("positive_count") or 0) > 0
        and (row.get("negative_count") or 0) > 0
    ]

    strict95 = [row for row in valid_folds if row.get("strict_regime_pass_95") is True]
    strict99 = [row for row in valid_folds if row.get("strict_regime_pass_99") is True]
    improve = [row for row in valid_folds if row.get("regime_improves_universal") is True]
    regime_positive = [row for row in valid_folds if row.get("regime_gain_positive") is True]
    universal_positive = [row for row in valid_folds if row.get("universal_gain_positive") is True]

    loho_improve = [row for row in valid_loho if row.get("regime_improves_universal") is True]
    loho_positive = [row for row in valid_loho if row.get("regime_gain_positive") is True]
    loho_low = [row for row in valid_loho if row.get("host_is_low_alpha_regime") is True]
    loho_low_improve = [row for row in loho_low if row.get("regime_improves_universal") is True]

    universal_total = float(np.sum([row.get("universal_fixed_gain") or 0.0 for row in valid_folds]))
    regime_total = float(np.sum([row.get("regime_fixed_gain") or 0.0 for row in valid_folds]))

    return {
        "regime_name": regime_name,
        "low_alpha_hosts": ",".join(sorted(low_alpha_hosts)),
        "valid_fold_count": len(valid_folds),
        "strict_pass_95_count": len(strict95),
        "strict_pass_99_count": len(strict99),
        "strict_pass_95_fraction": float(len(strict95) / len(valid_folds)) if valid_folds else None,
        "strict_pass_99_fraction": float(len(strict99) / len(valid_folds)) if valid_folds else None,
        "fold_improvement_count": len(improve),
        "fold_improvement_fraction": float(len(improve) / len(valid_folds)) if valid_folds else None,
        "regime_positive_gain_count": len(regime_positive),
        "regime_positive_gain_fraction": float(len(regime_positive) / len(valid_folds)) if valid_folds else None,
        "universal_positive_gain_count": len(universal_positive),
        "universal_positive_gain_fraction": float(len(universal_positive) / len(valid_folds)) if valid_folds else None,
        "universal_total_fold_gain": universal_total,
        "regime_total_fold_gain": regime_total,
        "regime_minus_universal_total_gain": float(regime_total - universal_total),
        "valid_loho_host_count": len(valid_loho),
        "loho_improvement_count": len(loho_improve),
        "loho_improvement_fraction": float(len(loho_improve) / len(valid_loho)) if valid_loho else None,
        "loho_positive_gain_count": len(loho_positive),
        "loho_positive_gain_fraction": float(len(loho_positive) / len(valid_loho)) if valid_loho else None,
        "low_regime_loho_host_count": len(loho_low),
        "low_regime_loho_improvement_count": len(loho_low_improve),
        "low_regime_loho_improvement_fraction": float(len(loho_low_improve) / len(loho_low)) if loho_low else None,
    }


def decide_status(regime_summaries):
    ranked = sorted(
        regime_summaries,
        key=lambda row: (
            -(row.get("regime_minus_universal_total_gain") or 0.0),
            -(row.get("strict_pass_95_count") or 0),
            -(row.get("loho_improvement_count") or 0),
        ),
    )
    best = ranked[0] if ranked else {}

    strong = bool(
        best
        and (best.get("valid_fold_count") or 0) >= 5
        and (best.get("strict_pass_95_count") or 0) >= 4
        and (best.get("fold_improvement_fraction") or 0.0) >= 0.80
        and (best.get("regime_minus_universal_total_gain") or 0.0) > 0.0
        and (best.get("loho_improvement_fraction") or 0.0) >= 0.60
    )

    directional = bool(
        best
        and (best.get("valid_fold_count") or 0) >= 5
        and (best.get("fold_improvement_fraction") or 0.0) >= 0.60
        and (best.get("regime_minus_universal_total_gain") or 0.0) > 0.0
    )

    best_cases = {
        "ranked_regime_summaries": ranked,
        "best_regime": best,
    }

    if strong:
        return (
            "regime_aware_two_coefficient_prediction_supported",
            9,
            "A predeclared two-coefficient regime improves held-out prediction over the universal coefficient.",
            best_cases,
        )

    if directional:
        return (
            "regime_aware_two_coefficient_prediction_directional_not_locked",
            7,
            "A predeclared two-coefficient regime improves held-out gain directionally, but not strongly enough to lock.",
            best_cases,
        )

    return (
        "regime_aware_two_coefficient_prediction_not_supported",
        6,
        "Two-coefficient regime-aware prediction does not improve held-out transfer over the universal coefficient.",
        best_cases,
    )


def make_plots(fold_rows, loho_rows, summaries):
    try:
        if summaries:
            rows = sorted(summaries, key=lambda r: -(r.get("regime_minus_universal_total_gain") or 0.0))
            x = np.arange(len(rows))

            plt.figure(figsize=(10, 5))
            plt.bar(x, [r.get("regime_minus_universal_total_gain", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [r["regime_name"] for r in rows], rotation=25, ha="right")
            plt.ylabel("regime total gain - universal total gain")
            plt.title("v0.7 regime-aware improvement over universal coefficient")
            plt.tight_layout()
            plt.savefig(OUTDIR / "regime_summary_improvement_v0_7.png", dpi=160)
            plt.close()

        if fold_rows:
            rows = sorted(fold_rows, key=lambda r: (r["regime_name"], r["fold_index"]))
            labels = [f"{r['regime_name']}\nF{r['fold_index']}" for r in rows]
            x = np.arange(len(rows))

            plt.figure(figsize=(14, 5))
            plt.bar(x, [r.get("regime_minus_universal_gain", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, labels, rotation=70, ha="right", fontsize=8)
            plt.ylabel("regime gain - universal gain")
            plt.title("v0.7 held-out fold improvement")
            plt.tight_layout()
            plt.savefig(OUTDIR / "fold_regime_minus_universal_gain_v0_7.png", dpi=160)
            plt.close()

        if loho_rows:
            rows = sorted(loho_rows, key=lambda r: r.get("regime_minus_universal_gain", 0.0))[:30]
            labels = [f"{r['regime_name']}:{r['heldout_host']}" for r in rows]
            x = np.arange(len(rows))

            plt.figure(figsize=(14, 5))
            plt.bar(x, [r.get("regime_minus_universal_gain", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, labels, rotation=70, ha="right", fontsize=8)
            plt.ylabel("regime gain - universal gain")
            plt.title("v0.7 worst LOHO regime-aware improvements")
            plt.tight_layout()
            plt.savefig(OUTDIR / "loho_worst_regime_minus_universal_v0_7.png", dpi=160)
            plt.close()

    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v0_7.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )


def main():
    print("TAIRID Boundary Prediction Battery v0.7 starting.")
    print("Boundary: regime-aware two-coefficient held-out prediction only; not proof.")

    write_json(OUTDIR / "claims_v0_7.json", CLAIMS_V0_7)

    repair_summary = {}

    try:
        v16.OUTDIR = OUTDIR
        v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        v02.OUTDIR = OUTDIR
        v02.DOWNLOAD_DIR = DOWNLOAD_DIR
        v05.OUTDIR = OUTDIR
        v05.DOWNLOAD_DIR = DOWNLOAD_DIR
        v06.OUTDIR = OUTDIR
        v06.DOWNLOAD_DIR = DOWNLOAD_DIR

        ns, repair_summary = v16.load_v15_helpers()
        write_json(OUTDIR / "v15_import_repair_summary_v0_7.json", repair_summary)

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

        write_csv(OUTDIR / "download_ledger_v0_7.csv", ledger)
        write_json(OUTDIR / "download_attempts_v0_7.json", {"compact": downloads, "auxiliary": aux_results})

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

        write_json(OUTDIR / "parse_meta_v0_7.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v0_7.json", parse_errors)

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
                "test_name": "TAIRID Boundary Prediction Battery v0.7",
                "final_status": "boundary_polarity_battery_v0_7_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or Table2 retrieval before v0.7.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "boundary_polarity_battery_v0_7_summary.json", summary)
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

        write_csv(OUTDIR / "compact_row_map_v0_7.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v0_7.csv", cluster_rows)

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

        write_csv(OUTDIR / "within_host_f160w_low_rows_v0_7.csv", within_low_rows)
        write_csv(OUTDIR / "within_host_f160w_high_rows_v0_7.csv", within_high_rows)

        active_rows = active_host_rows(mapped_rows, within_contrast)
        active_hosts = sorted([row["host"] for row in active_rows])
        folds, fold_inventory = make_balanced_folds(active_rows, N_FOLDS)

        write_json(
            OUTDIR / "active_host_inventory_v0_7.json",
            {
                "active_host_count": len(active_hosts),
                "active_hosts": active_hosts,
                "active_host_rows": active_rows,
                "heldout_folds": fold_inventory,
                "regime_definitions": REGIME_DEFINITIONS,
            },
        )

        all_fold_rows = []
        all_loho_rows = []
        all_perm_summaries = []
        all_perm_details = []
        regime_summaries = []

        for regime in REGIME_DEFINITIONS:
            regime_name = regime["regime_name"]
            declared_low = set(regime["low_alpha_hosts_declared"])
            low_alpha_hosts = sorted([h for h in active_hosts if h in declared_low])

            write_json(
                OUTDIR / f"{safe_name(regime_name)}_definition_v0_7.json",
                {
                    **regime,
                    "low_alpha_hosts_present": low_alpha_hosts,
                    "clean_hosts_present": sorted([h for h in active_hosts if h not in set(low_alpha_hosts)]),
                },
            )

            for fold_index, test_hosts in enumerate(folds, start=1):
                row, perm_summary, perm_rows = evaluate_fold(
                    regime_name,
                    fold_index,
                    test_hosts,
                    active_hosts,
                    low_alpha_hosts,
                    within_contrast,
                    mapped_rows,
                    c_factor,
                    full_fit,
                )
                all_fold_rows.append(row)
                all_perm_summaries.append(perm_summary)
                all_perm_details.extend(perm_rows)

            for host in active_hosts:
                row = evaluate_loho(
                    regime_name,
                    host,
                    active_hosts,
                    low_alpha_hosts,
                    within_contrast,
                    mapped_rows,
                    c_factor,
                    full_fit,
                )
                all_loho_rows.append(row)

            regime_summaries.append(
                summarize_regime(regime_name, low_alpha_hosts, all_fold_rows, all_loho_rows)
            )

        write_csv(OUTDIR / "regime_aware_fold_predictions_v0_7.csv", all_fold_rows)
        write_csv(OUTDIR / "regime_aware_loho_predictions_v0_7.csv", all_loho_rows)
        write_json(OUTDIR / "regime_aware_permutation_summaries_v0_7.json", all_perm_summaries)
        write_csv(OUTDIR / "regime_aware_permutation_details_v0_7.csv", all_perm_details)
        write_csv(OUTDIR / "regime_aware_summary_v0_7.csv", regime_summaries)

        final_status, readiness_score, next_wall, best_cases = decide_status(regime_summaries)

        make_plots(all_fold_rows, all_loho_rows, regime_summaries)

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

        write_csv(OUTDIR / "design_fit_comparison_v0_7.csv", design_fit_rows)

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
            "test_name": "TAIRID Boundary Prediction Battery v0.7",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Regime-aware two-coefficient held-out prediction audit only. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_7": CLAIMS_V0_7,
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
            "active_host_inventory": {
                "active_host_count": len(active_hosts),
                "active_hosts": active_hosts,
                "active_host_rows": active_rows,
                "fold_inventory": fold_inventory,
            },
            "regime_summaries": regime_summaries,
            "fold_predictions": all_fold_rows,
            "loho_predictions_top_negative": sorted(
                all_loho_rows,
                key=lambda r: r.get("regime_minus_universal_gain", 0.0),
            )[:30],
            "loho_predictions_top_positive": sorted(
                all_loho_rows,
                key=lambda r: -r.get("regime_minus_universal_gain", 0.0),
            )[:30],
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_7_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_7_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v0_7.json"),
                "regime_summary_csv": str(OUTDIR / "regime_aware_summary_v0_7.csv"),
                "fold_predictions_csv": str(OUTDIR / "regime_aware_fold_predictions_v0_7.csv"),
                "loho_predictions_csv": str(OUTDIR / "regime_aware_loho_predictions_v0_7.csv"),
                "permutation_summaries_json": str(OUTDIR / "regime_aware_permutation_summaries_v0_7.json"),
                "plots": [
                    str(OUTDIR / "regime_summary_improvement_v0_7.png"),
                    str(OUTDIR / "fold_regime_minus_universal_gain_v0_7.png"),
                    str(OUTDIR / "loho_worst_regime_minus_universal_v0_7.png"),
                ],
            },
            "interpretation": {
                "what_supports_regime_model": (
                    "A predeclared stress/low-alpha split improves held-out gain over the universal coefficient "
                    "across folds and many leave-one-host-out cases."
                ),
                "what_rejects_regime_model": (
                    "The two-coefficient model fails to improve held-out gain or improves only the named stress hosts while hurting the clean field."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only tests whether the stress-regime decomposition improves held-out prediction."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_7_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_7_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.7\n\n")
            f.write("Boundary: regime-aware two-coefficient held-out prediction only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Regime summaries:\n")
            f.write(json.dumps(regime_summaries, indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only tests whether regime-aware coefficients improve held-out prediction.\n")

        print("TAIRID Boundary Prediction Battery v0.7 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.7",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_7_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_7_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
