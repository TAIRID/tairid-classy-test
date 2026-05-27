#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.8
M31 sign-break and low-alpha regime validation audit.

Purpose:
v0.7 showed that a predeclared two-coefficient regime improved held-out prediction
over one universal coefficient. The best fold-level split was:

    stress4 = LMC + M31 + N4536 + SMC

But v0.7 also showed M31 remained negative even after low-alpha treatment.
This v0.8 test separates:

    low-alpha hosts: LMC + SMC + N4536
    sign-break quarantine host: M31

Core question:
    Is M31 better treated as part of the low-alpha group, or as its own
    sign-break/quarantine class where no transferred F160W correction is applied
    until a real sign-break peer class exists?

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only tests whether separating M31 from the low-alpha group improves held-out prediction.
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


OUTDIR = Path("tairid_boundary_polarity_battery_v0_8_outputs")
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

MODEL_DEFINITIONS = [
    {
        "model_name": "universal_one_coefficient",
        "model_type": "universal",
        "low_alpha_hosts": [],
        "sign_break_hosts": [],
        "sign_policy": "none",
        "plain_meaning": "One F160W coefficient for every active host.",
    },
    {
        "model_name": "two_coefficient_stress4",
        "model_type": "two_coefficient",
        "low_alpha_hosts": ["LMC", "M31", "N4536", "SMC"],
        "sign_break_hosts": [],
        "sign_policy": "none",
        "plain_meaning": "v0.7 best fold-level model: LMC + M31 + N4536 + SMC as low-alpha.",
    },
    {
        "model_name": "three_regime_m31_quarantine",
        "model_type": "three_regime",
        "low_alpha_hosts": ["LMC", "SMC", "N4536"],
        "sign_break_hosts": ["M31"],
        "sign_policy": "zero_transfer",
        "plain_meaning": "Clean high-alpha + low-alpha anchor/watch hosts + M31 sign-break quarantine with zero transferred correction.",
    },
]

CLAIMS_V0_8 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.8",
    "scope": "M31 sign-break and low-alpha regime validation",
    "native_tairid_claim": (
        "If M31 is a true sign-break rather than merely a low-alpha host, then separating it from the low-alpha group "
        "and quarantining transferred correction should improve or protect held-out prediction without damaging the clean-host field."
    ),
    "primary_prediction": (
        "The three-regime M31-quarantine model should improve M31 behavior relative to the two-coefficient stress4 model, "
        "while keeping fold-level and leave-one-host-out performance competitive."
    ),
    "models_tested": [
        "universal_one_coefficient",
        "two_coefficient_stress4",
        "three_regime_m31_quarantine",
    ],
    "anti_ad_hoc_rule": (
        "Only the predeclared models above are tested. No additional host may be added after seeing the v0.8 result."
    ),
    "failure_rule": (
        "If M31 quarantine does not improve prediction or damages the broader host field, M31 remains a difficult low-alpha/sign-break stress case, "
        "not an earned separate regime."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only tests whether M31 should be separated from the low-alpha regime."
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


def partition_hosts(hosts, low_alpha_hosts, sign_break_hosts):
    hosts = sorted(set(hosts))
    low_alpha_hosts = set(low_alpha_hosts)
    sign_break_hosts = set(sign_break_hosts)

    sign = sorted([h for h in hosts if h in sign_break_hosts])
    low = sorted([h for h in hosts if h in low_alpha_hosts and h not in sign_break_hosts])
    clean = sorted([h for h in hosts if h not in low_alpha_hosts and h not in sign_break_hosts])

    return clean, low, sign


def train_model_alphas(model, train_hosts, contrast, mapped_rows, c_factor, fit):
    model_type = model["model_type"]
    low_declared = set(model["low_alpha_hosts"])
    sign_declared = set(model["sign_break_hosts"])

    train_hosts = sorted(set(train_hosts))

    universal_stats, _ = cohort_stats(
        f"{model['model_name']}_train_universal",
        train_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=None,
    )
    alpha_universal = float(universal_stats.get("alpha_fit", 0.0))

    if model_type == "universal":
        return {
            "alpha_universal": alpha_universal,
            "alpha_clean": alpha_universal,
            "alpha_low": alpha_universal,
            "alpha_sign": alpha_universal,
            "train_host_count": len(train_hosts),
            "train_clean_host_count": len(train_hosts),
            "train_low_host_count": 0,
            "train_sign_host_count": 0,
            "train_hosts": ",".join(train_hosts),
            "train_clean_hosts": ",".join(train_hosts),
            "train_low_hosts": "",
            "train_sign_hosts": "",
            "universal_train_delta": universal_stats.get("self_fit_delta_chi2"),
            "clean_train_delta": universal_stats.get("self_fit_delta_chi2"),
            "low_train_delta": 0.0,
            "sign_train_delta": 0.0,
        }

    clean_train, low_train, sign_train = partition_hosts(train_hosts, low_declared, sign_declared)

    clean_stats, _ = cohort_stats(
        f"{model['model_name']}_train_clean",
        clean_train,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=None,
    )
    low_stats, _ = cohort_stats(
        f"{model['model_name']}_train_low",
        low_train,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=None,
    )
    sign_stats, _ = cohort_stats(
        f"{model['model_name']}_train_sign",
        sign_train,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=None,
    )

    alpha_clean = float(clean_stats.get("alpha_fit", 0.0)) if clean_train else alpha_universal
    alpha_low = float(low_stats.get("alpha_fit", 0.0)) if low_train else alpha_universal

    if model.get("sign_policy") == "zero_transfer":
        alpha_sign = 0.0
    else:
        alpha_sign = float(sign_stats.get("alpha_fit", 0.0)) if sign_train else alpha_universal

    return {
        "alpha_universal": alpha_universal,
        "alpha_clean": alpha_clean,
        "alpha_low": alpha_low,
        "alpha_sign": alpha_sign,
        "train_host_count": len(train_hosts),
        "train_clean_host_count": len(clean_train),
        "train_low_host_count": len(low_train),
        "train_sign_host_count": len(sign_train),
        "train_hosts": ",".join(train_hosts),
        "train_clean_hosts": ",".join(clean_train),
        "train_low_hosts": ",".join(low_train),
        "train_sign_hosts": ",".join(sign_train),
        "universal_train_delta": universal_stats.get("self_fit_delta_chi2"),
        "clean_train_delta": clean_stats.get("self_fit_delta_chi2"),
        "low_train_delta": low_stats.get("self_fit_delta_chi2"),
        "sign_train_delta": sign_stats.get("self_fit_delta_chi2"),
        "alpha_clean_minus_low": float(alpha_clean - alpha_low),
        "alpha_clean_minus_sign": float(alpha_clean - alpha_sign),
        "alpha_low_minus_sign": float(alpha_low - alpha_sign),
    }


def evaluate_model_gain(model, name, test_hosts, contrast, mapped_rows, c_factor, fit, alphas):
    low_declared = set(model["low_alpha_hosts"])
    sign_declared = set(model["sign_break_hosts"])
    model_type = model["model_type"]

    test_hosts = sorted(set(test_hosts))

    universal_stats, universal_vector = cohort_stats(
        f"{name}_universal_test",
        test_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=alphas["alpha_universal"],
    )

    if model_type == "universal":
        model_gain = float(universal_stats.get("fixed_alpha_gain") or 0.0)
        model_direction = v05.direction_stats(f"{name}_model_direction", universal_vector, fit)

        return {
            "name": name,
            "model_name": model["model_name"],
            "test_hosts": ",".join(test_hosts),
            "test_host_count": len(test_hosts),
            "test_clean_hosts": ",".join(test_hosts),
            "test_low_hosts": "",
            "test_sign_hosts": "",
            "test_clean_host_count": len(test_hosts),
            "test_low_host_count": 0,
            "test_sign_host_count": 0,
            "alpha_universal": alphas["alpha_universal"],
            "alpha_clean": alphas["alpha_clean"],
            "alpha_low": alphas["alpha_low"],
            "alpha_sign": alphas["alpha_sign"],
            "universal_fixed_gain": model_gain,
            "model_fixed_gain": model_gain,
            "model_minus_universal_gain": 0.0,
            "universal_score": universal_stats.get("score"),
            "model_direction_positive": bool(model_direction.get("high_side_more_positive") is True),
            "model_signed_difference": model_direction.get("signed_mean_residual_difference_high_minus_low"),
            "positive_count": universal_stats.get("positive_count"),
            "negative_count": universal_stats.get("negative_count"),
        }

    clean_hosts, low_hosts, sign_hosts = partition_hosts(test_hosts, low_declared, sign_declared)

    clean_stats, clean_vector = cohort_stats(
        f"{name}_clean_part",
        clean_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=alphas["alpha_clean"],
    )
    low_stats, low_vector = cohort_stats(
        f"{name}_low_part",
        low_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=alphas["alpha_low"],
    )
    sign_stats, sign_vector = cohort_stats(
        f"{name}_sign_part",
        sign_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alpha_fixed=alphas["alpha_sign"],
    )

    model_gain = float(
        (clean_stats.get("fixed_alpha_gain") or 0.0)
        + (low_stats.get("fixed_alpha_gain") or 0.0)
        + (sign_stats.get("fixed_alpha_gain") or 0.0)
    )
    universal_gain = float(universal_stats.get("fixed_alpha_gain") or 0.0)

    model_vector = clean_vector + low_vector + sign_vector
    model_direction = v05.direction_stats(f"{name}_model_direction", model_vector, fit)
    sign_direction = v05.direction_stats(f"{name}_sign_direction", sign_vector, fit)

    return {
        "name": name,
        "model_name": model["model_name"],
        "test_hosts": ",".join(test_hosts),
        "test_host_count": len(test_hosts),
        "test_clean_hosts": ",".join(clean_hosts),
        "test_low_hosts": ",".join(low_hosts),
        "test_sign_hosts": ",".join(sign_hosts),
        "test_clean_host_count": len(clean_hosts),
        "test_low_host_count": len(low_hosts),
        "test_sign_host_count": len(sign_hosts),
        "alpha_universal": alphas["alpha_universal"],
        "alpha_clean": alphas["alpha_clean"],
        "alpha_low": alphas["alpha_low"],
        "alpha_sign": alphas["alpha_sign"],
        "universal_fixed_gain": universal_gain,
        "model_fixed_gain": model_gain,
        "model_minus_universal_gain": float(model_gain - universal_gain),
        "universal_score": universal_stats.get("score"),
        "clean_score": clean_stats.get("score"),
        "low_score": low_stats.get("score"),
        "sign_score": sign_stats.get("score"),
        "clean_fixed_gain": clean_stats.get("fixed_alpha_gain"),
        "low_fixed_gain": low_stats.get("fixed_alpha_gain"),
        "sign_fixed_gain": sign_stats.get("fixed_alpha_gain"),
        "model_direction_positive": bool(model_direction.get("high_side_more_positive") is True),
        "model_signed_difference": model_direction.get("signed_mean_residual_difference_high_minus_low"),
        "sign_direction_positive": bool(sign_direction.get("high_side_more_positive") is True),
        "sign_signed_difference": sign_direction.get("signed_mean_residual_difference_high_minus_low"),
        "positive_count": universal_stats.get("positive_count"),
        "negative_count": universal_stats.get("negative_count"),
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


def permutation_comparison(model, name, test_hosts, observed_vector, mapped_rows, c_factor, fit, alphas):
    rng = np.random.default_rng(SEED)
    rows = []

    for repeat in range(PERMUTATION_REPEATS):
        random_values = build_host_preserving_random_vector(
            test_hosts,
            observed_vector,
            mapped_rows,
            rng,
        )
        result = evaluate_model_gain(
            model,
            f"{name}_permutation_{repeat}",
            test_hosts,
            random_values,
            mapped_rows,
            c_factor,
            fit,
            alphas,
        )
        result["repeat"] = repeat
        rows.append(result)

    if not rows:
        return rows, {"candidate": name, "status": "no_permutation_rows"}

    model_gains = np.asarray([row.get("model_fixed_gain", 0.0) for row in rows], dtype=float)
    improvements = np.asarray([row.get("model_minus_universal_gain", 0.0) for row in rows], dtype=float)

    return rows, {
        "candidate": name,
        "status": "ok",
        "repeats": PERMUTATION_REPEATS,
        "model_gain_mean": float(np.mean(model_gains)),
        "model_gain_95": float(np.percentile(model_gains, 95)),
        "model_gain_99": float(np.percentile(model_gains, 99)),
        "improvement_mean": float(np.mean(improvements)),
        "improvement_95": float(np.percentile(improvements, 95)),
        "improvement_99": float(np.percentile(improvements, 99)),
    }


def evaluate_fold(model, fold_index, test_hosts, active_hosts, contrast, mapped_rows, c_factor, fit):
    test_hosts = sorted(test_hosts)
    train_hosts = sorted([h for h in active_hosts if h not in set(test_hosts)])

    alphas = train_model_alphas(
        model,
        train_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
    )

    observed = evaluate_model_gain(
        model,
        f"{model['model_name']}_fold_{fold_index}_observed",
        test_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alphas,
    )

    observed_vector, _ = restrict_vector_to_hosts(contrast, mapped_rows, test_hosts)

    perm_rows, perm_summary = permutation_comparison(
        model,
        f"{model['model_name']}_fold_{fold_index}",
        test_hosts,
        observed_vector,
        mapped_rows,
        c_factor,
        fit,
        alphas,
    )

    row = {
        "fold_index": fold_index,
        **alphas,
        **observed,
        "permutation_status": perm_summary.get("status"),
        "permutation_model_gain_95": perm_summary.get("model_gain_95"),
        "permutation_model_gain_99": perm_summary.get("model_gain_99"),
        "permutation_improvement_95": perm_summary.get("improvement_95"),
        "permutation_improvement_99": perm_summary.get("improvement_99"),
    }

    row["model_gain_positive"] = bool(row.get("model_fixed_gain", 0.0) > 0.0)
    row["universal_gain_positive"] = bool(row.get("universal_fixed_gain", 0.0) > 0.0)
    row["model_improves_universal"] = bool(row.get("model_minus_universal_gain", 0.0) > 0.0)
    row["model_beats_95_permutation"] = bool(
        perm_summary.get("status") == "ok"
        and row.get("model_fixed_gain", 0.0) > (perm_summary.get("model_gain_95") or float("inf"))
    )
    row["model_beats_99_permutation"] = bool(
        perm_summary.get("status") == "ok"
        and row.get("model_fixed_gain", 0.0) > (perm_summary.get("model_gain_99") or float("inf"))
    )
    row["improvement_beats_95_permutation"] = bool(
        perm_summary.get("status") == "ok"
        and row.get("model_minus_universal_gain", 0.0) > (perm_summary.get("improvement_95") or float("inf"))
    )
    row["improvement_beats_99_permutation"] = bool(
        perm_summary.get("status") == "ok"
        and row.get("model_minus_universal_gain", 0.0) > (perm_summary.get("improvement_99") or float("inf"))
    )
    row["strict_model_pass_95"] = bool(
        row["model_gain_positive"]
        and (row["model_improves_universal"] or model["model_type"] == "universal")
        and row["model_beats_95_permutation"]
    )
    row["strict_model_pass_99"] = bool(
        row["model_gain_positive"]
        and (row["model_improves_universal"] or model["model_type"] == "universal")
        and row["model_beats_99_permutation"]
    )

    return row, perm_summary, perm_rows


def evaluate_loho(model, host, active_hosts, contrast, mapped_rows, c_factor, fit):
    train_hosts = sorted([h for h in active_hosts if h != host])

    alphas = train_model_alphas(
        model,
        train_hosts,
        contrast,
        mapped_rows,
        c_factor,
        fit,
    )

    observed = evaluate_model_gain(
        model,
        f"{model['model_name']}_loho_{host}_observed",
        [host],
        contrast,
        mapped_rows,
        c_factor,
        fit,
        alphas,
    )

    row = {
        "heldout_host": host,
        "host_is_low_alpha": bool(host in set(model["low_alpha_hosts"])),
        "host_is_sign_break": bool(host in set(model["sign_break_hosts"])),
        **alphas,
        **observed,
    }
    row["model_gain_positive"] = bool(row.get("model_fixed_gain", 0.0) > 0.0)
    row["universal_gain_positive"] = bool(row.get("universal_fixed_gain", 0.0) > 0.0)
    row["model_improves_universal"] = bool(row.get("model_minus_universal_gain", 0.0) > 0.0)

    return row


def local_host_diagnostics(hosts, contrast, mapped_rows, c_factor, fit):
    rows = []

    for host in hosts:
        local_stats, _ = cohort_stats(
            f"local_host_{host}",
            [host],
            contrast,
            mapped_rows,
            c_factor,
            fit,
            alpha_fixed=None,
        )
        rows.append(
            {
                "host": host,
                "local_alpha": local_stats.get("alpha_fit"),
                "local_self_fit_delta_chi2": local_stats.get("self_fit_delta_chi2"),
                "local_score": local_stats.get("score"),
                "direction_positive": local_stats.get("direction_positive"),
                "signed_mean_residual_difference_high_minus_low": local_stats.get(
                    "signed_mean_residual_difference_high_minus_low"
                ),
                "positive_count": local_stats.get("positive_count"),
                "negative_count": local_stats.get("negative_count"),
                "edge_count": local_stats.get("edge_count"),
            }
        )

    return sorted(rows, key=lambda r: (r["local_alpha"] if r["local_alpha"] is not None else 0.0))


def summarize_model(model_name, fold_rows, loho_rows):
    valid_folds = [
        row for row in fold_rows
        if row["model_name"] == model_name
        and (row.get("positive_count") or 0) > 0
        and (row.get("negative_count") or 0) > 0
    ]
    valid_loho = [
        row for row in loho_rows
        if row["model_name"] == model_name
        and (row.get("positive_count") or 0) > 0
        and (row.get("negative_count") or 0) > 0
    ]

    strict95 = [row for row in valid_folds if row.get("strict_model_pass_95") is True]
    strict99 = [row for row in valid_folds if row.get("strict_model_pass_99") is True]
    improve = [row for row in valid_folds if row.get("model_improves_universal") is True]
    positive = [row for row in valid_folds if row.get("model_gain_positive") is True]

    loho_improve = [row for row in valid_loho if row.get("model_improves_universal") is True]
    loho_positive = [row for row in valid_loho if row.get("model_gain_positive") is True]

    m31_loho = next((row for row in valid_loho if row.get("heldout_host") == "M31"), None)
    lmc_loho = next((row for row in valid_loho if row.get("heldout_host") == "LMC"), None)
    smc_loho = next((row for row in valid_loho if row.get("heldout_host") == "SMC"), None)
    n4536_loho = next((row for row in valid_loho if row.get("heldout_host") == "N4536"), None)

    universal_total = float(np.sum([row.get("universal_fixed_gain") or 0.0 for row in valid_folds]))
    model_total = float(np.sum([row.get("model_fixed_gain") or 0.0 for row in valid_folds]))

    return {
        "model_name": model_name,
        "valid_fold_count": len(valid_folds),
        "strict_pass_95_count": len(strict95),
        "strict_pass_99_count": len(strict99),
        "strict_pass_95_fraction": float(len(strict95) / len(valid_folds)) if valid_folds else None,
        "strict_pass_99_fraction": float(len(strict99) / len(valid_folds)) if valid_folds else None,
        "fold_improvement_count": len(improve),
        "fold_improvement_fraction": float(len(improve) / len(valid_folds)) if valid_folds else None,
        "model_positive_gain_count": len(positive),
        "model_positive_gain_fraction": float(len(positive) / len(valid_folds)) if valid_folds else None,
        "universal_total_fold_gain": universal_total,
        "model_total_fold_gain": model_total,
        "model_minus_universal_total_gain": float(model_total - universal_total),
        "valid_loho_host_count": len(valid_loho),
        "loho_improvement_count": len(loho_improve),
        "loho_improvement_fraction": float(len(loho_improve) / len(valid_loho)) if valid_loho else None,
        "loho_positive_gain_count": len(loho_positive),
        "loho_positive_gain_fraction": float(len(loho_positive) / len(valid_loho)) if valid_loho else None,
        "m31_loho_model_gain": m31_loho.get("model_fixed_gain") if m31_loho else None,
        "m31_loho_universal_gain": m31_loho.get("universal_fixed_gain") if m31_loho else None,
        "m31_loho_improvement": m31_loho.get("model_minus_universal_gain") if m31_loho else None,
        "lmc_loho_model_gain": lmc_loho.get("model_fixed_gain") if lmc_loho else None,
        "lmc_loho_universal_gain": lmc_loho.get("universal_fixed_gain") if lmc_loho else None,
        "lmc_loho_improvement": lmc_loho.get("model_minus_universal_gain") if lmc_loho else None,
        "smc_loho_model_gain": smc_loho.get("model_fixed_gain") if smc_loho else None,
        "smc_loho_universal_gain": smc_loho.get("universal_fixed_gain") if smc_loho else None,
        "smc_loho_improvement": smc_loho.get("model_minus_universal_gain") if smc_loho else None,
        "n4536_loho_model_gain": n4536_loho.get("model_fixed_gain") if n4536_loho else None,
        "n4536_loho_universal_gain": n4536_loho.get("universal_fixed_gain") if n4536_loho else None,
        "n4536_loho_improvement": n4536_loho.get("model_minus_universal_gain") if n4536_loho else None,
    }


def decide_status(model_summaries):
    by_name = {row["model_name"]: row for row in model_summaries}

    universal = by_name.get("universal_one_coefficient", {})
    stress4 = by_name.get("two_coefficient_stress4", {})
    quarantine = by_name.get("three_regime_m31_quarantine", {})

    stress4_total = stress4.get("model_total_fold_gain") or 0.0
    quarantine_total = quarantine.get("model_total_fold_gain") or 0.0
    stress4_m31 = stress4.get("m31_loho_model_gain")
    quarantine_m31 = quarantine.get("m31_loho_model_gain")

    m31_improved = bool(
        stress4_m31 is not None
        and quarantine_m31 is not None
        and quarantine_m31 > stress4_m31
    )

    fold_competitive = bool(
        quarantine_total >= 0.95 * stress4_total
    )

    fold_better = bool(
        quarantine_total > stress4_total
    )

    quarantine_strong = bool(
        (quarantine.get("valid_fold_count") or 0) >= 5
        and (quarantine.get("strict_pass_95_count") or 0) >= 4
        and (quarantine.get("model_positive_gain_fraction") or 0.0) >= 0.80
        and (quarantine.get("model_minus_universal_total_gain") or 0.0) > 0.0
        and m31_improved
        and fold_competitive
    )

    best_cases = {
        "universal": universal,
        "two_coefficient_stress4": stress4,
        "three_regime_m31_quarantine": quarantine,
        "m31_improved_by_quarantine": m31_improved,
        "quarantine_fold_competitive": fold_competitive,
        "quarantine_fold_better_than_stress4": fold_better,
    }

    if quarantine_strong and fold_better:
        return (
            "m31_sign_break_quarantine_supported_and_better",
            9,
            "M31 quarantine improves M31 behavior and beats the stress4 two-coefficient model at fold level.",
            best_cases,
        )

    if quarantine_strong:
        return (
            "m31_sign_break_quarantine_supported_as_protective_boundary",
            8,
            "M31 quarantine improves M31 behavior while remaining fold-competitive with the stress4 model.",
            best_cases,
        )

    if m31_improved and not fold_competitive:
        return (
            "m31_quarantine_improves_m31_but_hurts_fold_model",
            7,
            "M31 quarantine helps M31 but weakens the broader held-out fold model too much.",
            best_cases,
        )

    if not m31_improved and stress4_total > quarantine_total:
        return (
            "m31_better_left_inside_low_alpha_regime",
            7,
            "Separating M31 does not improve enough; the stress4 low-alpha model remains better.",
            best_cases,
        )

    return (
        "m31_sign_break_quarantine_not_supported",
        6,
        "The v0.8 evidence does not support separating M31 as a quarantine sign-break class.",
        best_cases,
    )


def make_plots(model_summaries, fold_rows, loho_rows, local_rows):
    try:
        if model_summaries:
            rows = sorted(model_summaries, key=lambda r: -(r.get("model_total_fold_gain") or 0.0))
            x = np.arange(len(rows))

            plt.figure(figsize=(10, 5))
            plt.bar(x, [r.get("model_total_fold_gain", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [r["model_name"] for r in rows], rotation=25, ha="right")
            plt.ylabel("total held-out fold gain")
            plt.title("v0.8 model total held-out gain")
            plt.tight_layout()
            plt.savefig(OUTDIR / "model_total_fold_gain_v0_8.png", dpi=160)
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.bar(x, [r.get("m31_loho_model_gain", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [r["model_name"] for r in rows], rotation=25, ha="right")
            plt.ylabel("M31 leave-one-host-out model gain")
            plt.title("v0.8 M31 LOHO behavior by model")
            plt.tight_layout()
            plt.savefig(OUTDIR / "m31_loho_gain_by_model_v0_8.png", dpi=160)
            plt.close()

        if local_rows:
            rows = sorted(local_rows, key=lambda r: r.get("local_alpha", 0.0))
            x = np.arange(len(rows))

            plt.figure(figsize=(14, 5))
            plt.bar(x, [r.get("local_alpha", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [r["host"] for r in rows], rotation=70, ha="right", fontsize=8)
            plt.ylabel("local alpha")
            plt.title("v0.8 local host alpha diagnostics")
            plt.tight_layout()
            plt.savefig(OUTDIR / "local_host_alpha_v0_8.png", dpi=160)
            plt.close()

    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v0_8.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )


def main():
    print("TAIRID Boundary Prediction Battery v0.8 starting.")
    print("Boundary: M31 sign-break validation only; not proof.")

    write_json(OUTDIR / "claims_v0_8.json", CLAIMS_V0_8)

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
        write_json(OUTDIR / "v15_import_repair_summary_v0_8.json", repair_summary)

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

        write_csv(OUTDIR / "download_ledger_v0_8.csv", ledger)
        write_json(OUTDIR / "download_attempts_v0_8.json", {"compact": downloads, "auxiliary": aux_results})

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

        write_json(OUTDIR / "parse_meta_v0_8.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v0_8.json", parse_errors)

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
                "test_name": "TAIRID Boundary Prediction Battery v0.8",
                "final_status": "boundary_polarity_battery_v0_8_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or Table2 retrieval before v0.8.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "boundary_polarity_battery_v0_8_summary.json", summary)
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

        write_csv(OUTDIR / "compact_row_map_v0_8.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v0_8.csv", cluster_rows)

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

        write_csv(OUTDIR / "within_host_f160w_low_rows_v0_8.csv", within_low_rows)
        write_csv(OUTDIR / "within_host_f160w_high_rows_v0_8.csv", within_high_rows)

        active_rows = active_host_rows(mapped_rows, within_contrast)
        active_hosts = sorted([row["host"] for row in active_rows])
        folds, fold_inventory = make_balanced_folds(active_rows, N_FOLDS)

        write_json(
            OUTDIR / "active_host_inventory_v0_8.json",
            {
                "active_host_count": len(active_hosts),
                "active_hosts": active_hosts,
                "active_host_rows": active_rows,
                "fold_inventory": fold_inventory,
                "model_definitions": MODEL_DEFINITIONS,
            },
        )

        all_fold_rows = []
        all_loho_rows = []
        all_perm_summaries = []
        all_perm_details = []

        for model in MODEL_DEFINITIONS:
            write_json(
                OUTDIR / f"{safe_name(model['model_name'])}_definition_v0_8.json",
                model,
            )

            for fold_index, test_hosts in enumerate(folds, start=1):
                row, perm_summary, perm_rows = evaluate_fold(
                    model,
                    fold_index,
                    test_hosts,
                    active_hosts,
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
                    model,
                    host,
                    active_hosts,
                    within_contrast,
                    mapped_rows,
                    c_factor,
                    full_fit,
                )
                all_loho_rows.append(row)

        model_summaries = [
            summarize_model(model["model_name"], all_fold_rows, all_loho_rows)
            for model in MODEL_DEFINITIONS
        ]

        local_rows = local_host_diagnostics(active_hosts, within_contrast, mapped_rows, c_factor, full_fit)

        write_csv(OUTDIR / "m31_regime_fold_predictions_v0_8.csv", all_fold_rows)
        write_csv(OUTDIR / "m31_regime_loho_predictions_v0_8.csv", all_loho_rows)
        write_csv(OUTDIR / "m31_regime_model_summary_v0_8.csv", model_summaries)
        write_csv(OUTDIR / "local_host_diagnostics_v0_8.csv", local_rows)
        write_json(OUTDIR / "m31_regime_permutation_summaries_v0_8.json", all_perm_summaries)
        write_csv(OUTDIR / "m31_regime_permutation_details_v0_8.csv", all_perm_details)

        final_status, readiness_score, next_wall, best_cases = decide_status(model_summaries)

        make_plots(model_summaries, all_fold_rows, all_loho_rows, local_rows)

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

        write_csv(OUTDIR / "design_fit_comparison_v0_8.csv", design_fit_rows)

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
            "test_name": "TAIRID Boundary Prediction Battery v0.8",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "M31 sign-break and low-alpha regime validation only. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_8": CLAIMS_V0_8,
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
            "model_summaries": model_summaries,
            "local_host_diagnostics": local_rows,
            "fold_predictions": all_fold_rows,
            "loho_predictions_top_negative": sorted(
                all_loho_rows,
                key=lambda r: r.get("model_minus_universal_gain", 0.0),
            )[:30],
            "loho_predictions_top_positive": sorted(
                all_loho_rows,
                key=lambda r: -r.get("model_minus_universal_gain", 0.0),
            )[:30],
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_8_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_8_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v0_8.json"),
                "model_summary_csv": str(OUTDIR / "m31_regime_model_summary_v0_8.csv"),
                "fold_predictions_csv": str(OUTDIR / "m31_regime_fold_predictions_v0_8.csv"),
                "loho_predictions_csv": str(OUTDIR / "m31_regime_loho_predictions_v0_8.csv"),
                "local_host_diagnostics_csv": str(OUTDIR / "local_host_diagnostics_v0_8.csv"),
                "permutation_summaries_json": str(OUTDIR / "m31_regime_permutation_summaries_v0_8.json"),
                "plots": [
                    str(OUTDIR / "model_total_fold_gain_v0_8.png"),
                    str(OUTDIR / "m31_loho_gain_by_model_v0_8.png"),
                    str(OUTDIR / "local_host_alpha_v0_8.png"),
                ],
            },
            "interpretation": {
                "what_supports_m31_sign_break": (
                    "M31 quarantine improves M31 behavior while preserving fold-level performance close to or above the stress4 model."
                ),
                "what_rejects_m31_sign_break": (
                    "M31 quarantine helps M31 only by damaging the broader held-out fold model, or fails to improve M31 at all."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only tests whether M31 should be separated from the low-alpha regime."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_8_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_8_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.8\n\n")
            f.write("Boundary: M31 sign-break and low-alpha regime validation only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Model summaries:\n")
            f.write(json.dumps(model_summaries, indent=2, default=json_default) + "\n\n")
            f.write("Local host diagnostics:\n")
            f.write(json.dumps(local_rows, indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only tests whether M31 should be separated from the low-alpha regime.\n")

        print("TAIRID Boundary Prediction Battery v0.8 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.8",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_8_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_8_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
