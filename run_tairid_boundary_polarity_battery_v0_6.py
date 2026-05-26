#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.6
Stress-host regime decomposition audit.

Purpose:
v0.5.1 showed strict held-out prediction mostly survives, but the stress hosts
LMC, M31, and N4536 break or weaken fixed-coefficient transfer.

This v0.6 test asks:

    Are LMC / M31 / N4536 random failures,
    weak-sampling failures,
    or evidence that the F160W polarity coefficient changes by host regime?

Core TAIRID question:
    Does the field-relative F160W boundary rule transfer cleanly through the
    non-stress host field while stress hosts show distinct local alpha / gain /
    signed-direction behavior?

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only decomposes stress-host behavior after v0.5.1.
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


OUTDIR = Path("tairid_boundary_polarity_battery_v0_6_outputs")
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

STRESS_HOSTS = {"LMC", "M31", "N4536"}
ANCHOR_HOSTS = {"LMC", "SMC", "N4258"}

CLAIMS_V0_6 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.6",
    "scope": "Stress-host regime decomposition for within-host F160W polarity",
    "reason_for_test": (
        "v0.5.1 showed strict held-out transfer in most hosts, but LMC, M31, and N4536 "
        "were stress cases. v0.6 tests whether those are random failures, weak-edge sampling, "
        "or distinct host-regime behavior."
    ),
    "native_tairid_claim": (
        "A field-relative boundary-polarity rule may transfer broadly while breaking at host environments "
        "with different reference structure. The failure cases must be exposed and decomposed, not hidden."
    ),
    "primary_prediction": (
        "Clean hosts excluding LMC/M31/N4536 should retain positive held-out transfer. "
        "Stress hosts should show distinguishable local alpha, fixed-gain, or signed-direction behavior."
    ),
    "failure_rule": (
        "If clean-host transfer does not improve after isolating stress hosts, or if stress hosts do not differ "
        "structurally, then the stress-host explanation is not supported."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only decomposes where the held-out F160W polarity rule breaks."
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


def host_counts_for_contrast(mapped_rows, contrast):
    counts = defaultdict(
        lambda: {
            "host": None,
            "positive_count": 0,
            "negative_count": 0,
            "edge_count": 0,
            "row_count": 0,
            "is_stress_host": False,
            "is_anchor_host": False,
        }
    )

    for row in mapped_rows:
        host = entity(row, "host_guess")
        idx = int(row["compact_observation_index"])
        val = float(contrast[idx])

        counts[host]["host"] = host
        counts[host]["row_count"] += 1
        counts[host]["is_stress_host"] = host in STRESS_HOSTS
        counts[host]["is_anchor_host"] = host in ANCHOR_HOSTS

        if val > 0:
            counts[host]["positive_count"] += 1
            counts[host]["edge_count"] += 1
        elif val < 0:
            counts[host]["negative_count"] += 1
            counts[host]["edge_count"] += 1

    rows = list(counts.values())
    rows = sorted(rows, key=lambda r: (-r["edge_count"], -r["row_count"], r["host"]))

    return rows


def active_host_rows(mapped_rows, contrast):
    return [
        row for row in host_counts_for_contrast(mapped_rows, contrast)
        if row["positive_count"] > 0 and row["negative_count"] > 0
    ]


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


def strict_transfer_pass(row, level="95"):
    threshold_gain = row.get(f"permutation_fixed_gain_{level}")
    threshold_score = row.get(f"permutation_abs_score_{level}")

    if threshold_gain is None or threshold_score is None:
        return False

    return bool(
        row.get("alpha_train", 0.0) > 0.0
        and row.get("test_direction_positive") is True
        and row.get("test_fixed_alpha_gain", 0.0) > 0.0
        and row.get("test_fixed_alpha_gain", 0.0) > threshold_gain
        and abs(row.get("test_score", 0.0)) > threshold_score
    )


def run_clean_fold_predictions(clean_hosts, contrast, mapped_rows, y, c_factor, fit):
    clean_host_rows = [
        row for row in active_host_rows(mapped_rows, contrast)
        if row["host"] in set(clean_hosts)
    ]

    folds, inventory = make_balanced_folds(clean_host_rows, N_FOLDS)

    fold_rows = []
    perm_summaries = []
    perm_details = []

    for fold_index, test_hosts in enumerate(folds, start=1):
        train_hosts = sorted([host for host in clean_hosts if host not in set(test_hosts)])

        train_stats, train_vector = cohort_stats(
            f"clean_fold_{fold_index}_train",
            train_hosts,
            contrast,
            mapped_rows,
            c_factor,
            fit,
            alpha_fixed=None,
        )
        alpha_train = float(train_stats.get("alpha_fit", 0.0))

        test_stats, test_vector = cohort_stats(
            f"clean_fold_{fold_index}_test",
            test_hosts,
            contrast,
            mapped_rows,
            c_factor,
            fit,
            alpha_fixed=alpha_train,
        )

        perm_rows, perm_summary = v05.host_preserving_test_permutations(
            f"clean_fold_{fold_index}_heldout",
            test_vector,
            test_hosts,
            mapped_rows,
            y,
            c_factor,
            fit,
            alpha_train,
        )

        perm_details.extend(perm_rows)
        perm_summaries.append(perm_summary)

        row = {
            "fold_index": fold_index,
            "test_hosts": ",".join(sorted(test_hosts)),
            "train_host_count": len(train_hosts),
            "test_host_count": len(test_hosts),
            "alpha_train": alpha_train,
            "alpha_train_positive": bool(alpha_train > 0.0),
            "train_alpha_fit": train_stats.get("alpha_fit"),
            "train_self_fit_delta_chi2": train_stats.get("self_fit_delta_chi2"),
            "test_score": test_stats.get("score"),
            "test_self_fit_delta_chi2": test_stats.get("self_fit_delta_chi2"),
            "test_fixed_alpha_gain": test_stats.get("fixed_alpha_gain"),
            "test_direction_positive": test_stats.get("direction_positive"),
            "test_signed_mean_residual_difference_high_minus_low": test_stats.get(
                "signed_mean_residual_difference_high_minus_low"
            ),
            "test_positive_count": test_stats.get("positive_count"),
            "test_negative_count": test_stats.get("negative_count"),
            "permutation_status": perm_summary.get("status"),
            "permutation_fixed_gain_95": perm_summary.get("permutation_fixed_gain_95"),
            "permutation_fixed_gain_99": perm_summary.get("permutation_fixed_gain_99"),
            "permutation_abs_score_95": perm_summary.get("permutation_abs_score_95"),
            "permutation_abs_score_99": perm_summary.get("permutation_abs_score_99"),
        }
        row["strict_pass_95"] = strict_transfer_pass(row, "95")
        row["strict_pass_99"] = strict_transfer_pass(row, "99")
        fold_rows.append(row)

    return fold_rows, inventory, perm_summaries, perm_details


def host_decomposition(active_rows, contrast, mapped_rows, c_factor, fit, alpha_all, alpha_clean, alpha_stress):
    rows = []

    for host_row in active_rows:
        host = host_row["host"]
        hosts = [host]

        local_stats, local_vector = cohort_stats(
            f"host_local_{host}",
            hosts,
            contrast,
            mapped_rows,
            c_factor,
            fit,
            alpha_fixed=None,
        )

        all_fixed, _ = cohort_stats(
            f"host_{host}_fixed_alpha_all",
            hosts,
            contrast,
            mapped_rows,
            c_factor,
            fit,
            alpha_fixed=alpha_all,
        )

        clean_fixed, _ = cohort_stats(
            f"host_{host}_fixed_alpha_clean",
            hosts,
            contrast,
            mapped_rows,
            c_factor,
            fit,
            alpha_fixed=alpha_clean,
        )

        stress_fixed, _ = cohort_stats(
            f"host_{host}_fixed_alpha_stress",
            hosts,
            contrast,
            mapped_rows,
            c_factor,
            fit,
            alpha_fixed=alpha_stress,
        )

        row = {
            **host_row,
            "local_alpha_fit": local_stats.get("alpha_fit"),
            "local_self_fit_delta_chi2": local_stats.get("self_fit_delta_chi2"),
            "local_score": local_stats.get("score"),
            "local_direction_positive": local_stats.get("direction_positive"),
            "local_signed_mean_residual_difference_high_minus_low": local_stats.get(
                "signed_mean_residual_difference_high_minus_low"
            ),
            "fixed_gain_alpha_all": all_fixed.get("fixed_alpha_gain"),
            "fixed_gain_alpha_clean": clean_fixed.get("fixed_alpha_gain"),
            "fixed_gain_alpha_stress": stress_fixed.get("fixed_alpha_gain"),
            "gain_clean_positive": bool((clean_fixed.get("fixed_alpha_gain") or 0.0) > 0.0),
            "gain_all_positive": bool((all_fixed.get("fixed_alpha_gain") or 0.0) > 0.0),
            "gain_stress_positive": bool((stress_fixed.get("fixed_alpha_gain") or 0.0) > 0.0),
            "alpha_sign_mismatch_with_clean": bool(
                (local_stats.get("alpha_fit") or 0.0) * alpha_clean < 0.0
            ),
            "stress_case_under_clean_alpha": bool(
                host in STRESS_HOSTS
                and (
                    (clean_fixed.get("fixed_alpha_gain") or 0.0) <= 0.0
                    or local_stats.get("direction_positive") is not True
                )
            ),
            "weak_edge_case": bool(host_row.get("edge_count", 0) < 6),
        }

        rows.append(row)

    rows = sorted(
        rows,
        key=lambda r: (
            not r.get("is_stress_host", False),
            r.get("fixed_gain_alpha_clean", 0.0),
            r["host"],
        ),
    )

    return rows


def transfer_matrix(active_hosts, clean_hosts, stress_hosts, contrast, mapped_rows, c_factor, fit):
    stress_hosts = sorted(stress_hosts)
    clean_hosts = sorted(clean_hosts)
    active_hosts = sorted(active_hosts)

    tests = [
        ("clean_to_stress_all", clean_hosts, stress_hosts),
        ("stress_all_to_clean", stress_hosts, clean_hosts),
        ("all_to_clean", active_hosts, clean_hosts),
        ("all_to_stress_all", active_hosts, stress_hosts),
    ]

    for host in stress_hosts:
        tests.append((f"clean_to_{host}", clean_hosts, [host]))
        tests.append((f"without_{host}_to_{host}", [h for h in active_hosts if h != host], [host]))

    rows = []

    for name, train_hosts, test_hosts in tests:
        if not train_hosts or not test_hosts:
            continue

        train_stats, _ = cohort_stats(
            f"{name}_train",
            train_hosts,
            contrast,
            mapped_rows,
            c_factor,
            fit,
            alpha_fixed=None,
        )
        alpha_train = float(train_stats.get("alpha_fit", 0.0))

        test_stats, _ = cohort_stats(
            f"{name}_test",
            test_hosts,
            contrast,
            mapped_rows,
            c_factor,
            fit,
            alpha_fixed=alpha_train,
        )

        rows.append(
            {
                "transfer_test": name,
                "train_hosts": ",".join(sorted(train_hosts)),
                "test_hosts": ",".join(sorted(test_hosts)),
                "train_host_count": len(train_hosts),
                "test_host_count": len(test_hosts),
                "alpha_train": alpha_train,
                "alpha_train_positive": bool(alpha_train > 0.0),
                "train_self_fit_delta_chi2": train_stats.get("self_fit_delta_chi2"),
                "test_positive_count": test_stats.get("positive_count"),
                "test_negative_count": test_stats.get("negative_count"),
                "test_score": test_stats.get("score"),
                "test_fixed_alpha_gain": test_stats.get("fixed_alpha_gain"),
                "test_gain_positive": bool((test_stats.get("fixed_alpha_gain") or 0.0) > 0.0),
                "test_direction_positive": test_stats.get("direction_positive"),
                "test_signed_mean_residual_difference_high_minus_low": test_stats.get(
                    "signed_mean_residual_difference_high_minus_low"
                ),
                "test_self_fit_delta_chi2": test_stats.get("self_fit_delta_chi2"),
            }
        )

    return rows


def summarize_clean_folds(clean_fold_rows):
    valid = [
        row for row in clean_fold_rows
        if (row.get("test_positive_count") or 0) > 0
        and (row.get("test_negative_count") or 0) > 0
    ]

    pass95 = [row for row in valid if row.get("strict_pass_95") is True]
    pass99 = [row for row in valid if row.get("strict_pass_99") is True]
    positive_gain = [row for row in valid if (row.get("test_fixed_alpha_gain") or 0.0) > 0.0]
    positive_direction = [row for row in valid if row.get("test_direction_positive") is True]

    return {
        "valid_fold_count": len(valid),
        "strict_pass_95_count": len(pass95),
        "strict_pass_99_count": len(pass99),
        "strict_pass_95_fraction": float(len(pass95) / len(valid)) if valid else None,
        "strict_pass_99_fraction": float(len(pass99) / len(valid)) if valid else None,
        "positive_gain_count": len(positive_gain),
        "positive_gain_fraction": float(len(positive_gain) / len(valid)) if valid else None,
        "positive_direction_count": len(positive_direction),
        "positive_direction_fraction": float(len(positive_direction) / len(valid)) if valid else None,
        "total_fixed_gain": float(np.sum([row.get("test_fixed_alpha_gain") or 0.0 for row in valid])),
        "mean_fixed_gain": float(np.mean([row.get("test_fixed_alpha_gain") or 0.0 for row in valid])) if valid else None,
    }


def decide_status(clean_fold_summary, host_rows, matrix_rows, alpha_summary):
    clean_strict = clean_fold_summary.get("strict_pass_95_fraction") or 0.0
    clean_positive_gain = clean_fold_summary.get("positive_gain_fraction") or 0.0
    clean_total_gain = clean_fold_summary.get("total_fixed_gain") or 0.0

    stress_rows = [row for row in host_rows if row.get("is_stress_host") is True]
    stress_problem_rows = [
        row for row in stress_rows
        if row.get("stress_case_under_clean_alpha") is True
    ]
    stress_weak_rows = [row for row in stress_rows if row.get("weak_edge_case") is True]

    clean_to_stress = next(
        (row for row in matrix_rows if row["transfer_test"] == "clean_to_stress_all"),
        {},
    )
    stress_to_clean = next(
        (row for row in matrix_rows if row["transfer_test"] == "stress_all_to_clean"),
        {},
    )

    clean_stable = bool(
        clean_fold_summary.get("valid_fold_count", 0) >= 5
        and clean_strict >= 0.80
        and clean_positive_gain >= 0.80
        and clean_total_gain > 0.0
    )

    stress_distinct = bool(
        len(stress_problem_rows) >= 2
        or (clean_to_stress.get("test_gain_positive") is False)
        or (stress_to_clean.get("test_gain_positive") is False)
        or alpha_summary.get("alpha_clean_minus_stress_abs", 0.0) > 0.10
    )

    mostly_weak_sampling = bool(
        stress_rows
        and len(stress_weak_rows) >= max(1, len(stress_rows) - 1)
    )

    best_cases = {
        "clean_fold_summary": clean_fold_summary,
        "alpha_summary": alpha_summary,
        "stress_host_rows": stress_rows,
        "stress_problem_rows": stress_problem_rows,
        "clean_to_stress_all": clean_to_stress,
        "stress_all_to_clean": stress_to_clean,
    }

    if clean_stable and stress_distinct and not mostly_weak_sampling:
        return (
            "stress_host_regime_decomposition_supported",
            8,
            "Clean-host transfer remains stable after isolating stress hosts, and stress hosts show distinct coefficient/gain behavior.",
            best_cases,
        )

    if clean_stable and mostly_weak_sampling:
        return (
            "stress_hosts_partly_explained_by_sampling_or_edge_count",
            7,
            "Clean-host transfer remains stable, but stress behavior may be partly explained by weak edge counts.",
            best_cases,
        )

    if clean_stable and not stress_distinct:
        return (
            "stress_hosts_not_distinct_after_decomposition",
            7,
            "Clean-host transfer is stable, but stress hosts do not form a clearly distinct regime under v0.6.",
            best_cases,
        )

    return (
        "stress_host_decomposition_not_supported",
        6,
        "Isolating stress hosts does not rescue or clarify held-out transfer enough to support a regime claim.",
        best_cases,
    )


def make_plots(host_rows, clean_fold_rows, matrix_rows):
    try:
        if host_rows:
            rows = sorted(host_rows, key=lambda r: r.get("fixed_gain_alpha_clean", 0.0))
            x = np.arange(len(rows))

            plt.figure(figsize=(14, 5))
            plt.bar(x, [r.get("fixed_gain_alpha_clean", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [r["host"] for r in rows], rotation=70, ha="right", fontsize=8)
            plt.ylabel("fixed gain under clean-host alpha")
            plt.title("v0.6 host behavior under clean-host F160W coefficient")
            plt.tight_layout()
            plt.savefig(OUTDIR / "host_fixed_gain_under_clean_alpha_v0_6.png", dpi=160)
            plt.close()

            plt.figure(figsize=(14, 5))
            plt.bar(x, [r.get("local_alpha_fit", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [r["host"] for r in rows], rotation=70, ha="right", fontsize=8)
            plt.ylabel("local alpha fit")
            plt.title("v0.6 local host alpha estimates")
            plt.tight_layout()
            plt.savefig(OUTDIR / "host_local_alpha_v0_6.png", dpi=160)
            plt.close()

        if clean_fold_rows:
            rows = sorted(clean_fold_rows, key=lambda r: r["fold_index"])
            x = np.arange(len(rows))

            plt.figure(figsize=(9, 5))
            plt.bar(x, [r.get("test_fixed_alpha_gain", 0.0) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [str(r["fold_index"]) for r in rows])
            plt.xlabel("clean held-out fold")
            plt.ylabel("fixed coefficient gain")
            plt.title("v0.6 clean-host held-out prediction")
            plt.tight_layout()
            plt.savefig(OUTDIR / "clean_host_fold_gain_v0_6.png", dpi=160)
            plt.close()

    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v0_6.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )


def main():
    print("TAIRID Boundary Prediction Battery v0.6 starting.")
    print("Boundary: stress-host regime decomposition only; not proof.")

    write_json(OUTDIR / "claims_v0_6.json", CLAIMS_V0_6)

    repair_summary = {}

    try:
        v16.OUTDIR = OUTDIR
        v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        v02.OUTDIR = OUTDIR
        v02.DOWNLOAD_DIR = DOWNLOAD_DIR
        v05.OUTDIR = OUTDIR
        v05.DOWNLOAD_DIR = DOWNLOAD_DIR
        v05.PERMUTATION_REPEATS = PERMUTATION_REPEATS

        ns, repair_summary = v16.load_v15_helpers()
        write_json(OUTDIR / "v15_import_repair_summary_v0_6.json", repair_summary)

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

        write_csv(OUTDIR / "download_ledger_v0_6.csv", ledger)
        write_json(OUTDIR / "download_attempts_v0_6.json", {"compact": downloads, "auxiliary": aux_results})

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

        write_json(OUTDIR / "parse_meta_v0_6.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v0_6.json", parse_errors)

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
                "test_name": "TAIRID Boundary Prediction Battery v0.6",
                "final_status": "boundary_polarity_battery_v0_6_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or Table2 retrieval before v0.6.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "boundary_polarity_battery_v0_6_summary.json", summary)
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

        write_csv(OUTDIR / "compact_row_map_v0_6.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v0_6.csv", cluster_rows)

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

        write_csv(OUTDIR / "within_host_f160w_low_rows_v0_6.csv", within_low_rows)
        write_csv(OUTDIR / "within_host_f160w_high_rows_v0_6.csv", within_high_rows)

        active_rows = active_host_rows(mapped_rows, within_contrast)
        active_hosts = sorted([row["host"] for row in active_rows])
        stress_hosts_present = sorted([host for host in active_hosts if host in STRESS_HOSTS])
        clean_hosts = sorted([host for host in active_hosts if host not in STRESS_HOSTS])

        write_json(
            OUTDIR / "host_regime_inventory_v0_6.json",
            {
                "stress_hosts_declared": sorted(STRESS_HOSTS),
                "stress_hosts_present": stress_hosts_present,
                "clean_host_count": len(clean_hosts),
                "clean_hosts": clean_hosts,
                "active_host_count": len(active_hosts),
                "active_host_inventory": active_rows,
            },
        )

        all_stats, all_vector = cohort_stats(
            "all_active_hosts",
            active_hosts,
            within_contrast,
            mapped_rows,
            c_factor,
            full_fit,
            alpha_fixed=None,
        )
        clean_stats, clean_vector = cohort_stats(
            "clean_hosts_excluding_LMC_M31_N4536",
            clean_hosts,
            within_contrast,
            mapped_rows,
            c_factor,
            full_fit,
            alpha_fixed=None,
        )
        stress_stats, stress_vector = cohort_stats(
            "stress_hosts_only_LMC_M31_N4536",
            stress_hosts_present,
            within_contrast,
            mapped_rows,
            c_factor,
            full_fit,
            alpha_fixed=None,
        )

        alpha_all = float(all_stats.get("alpha_fit", 0.0))
        alpha_clean = float(clean_stats.get("alpha_fit", 0.0))
        alpha_stress = float(stress_stats.get("alpha_fit", 0.0))

        alpha_summary = {
            "alpha_all": alpha_all,
            "alpha_clean_excluding_stress": alpha_clean,
            "alpha_stress_only": alpha_stress,
            "alpha_clean_minus_stress": float(alpha_clean - alpha_stress),
            "alpha_clean_minus_stress_abs": float(abs(alpha_clean - alpha_stress)),
            "alpha_clean_positive": bool(alpha_clean > 0.0),
            "alpha_stress_positive": bool(alpha_stress > 0.0),
        }

        write_json(
            OUTDIR / "cohort_alpha_summary_v0_6.json",
            {
                "alpha_summary": alpha_summary,
                "all_stats": all_stats,
                "clean_stats": clean_stats,
                "stress_stats": stress_stats,
            },
        )

        host_rows = host_decomposition(
            active_rows,
            within_contrast,
            mapped_rows,
            c_factor,
            full_fit,
            alpha_all,
            alpha_clean,
            alpha_stress,
        )
        write_csv(OUTDIR / "host_regime_decomposition_v0_6.csv", host_rows)

        clean_fold_rows, clean_fold_inventory, clean_perm_summaries, clean_perm_details = run_clean_fold_predictions(
            clean_hosts,
            within_contrast,
            mapped_rows,
            y,
            c_factor,
            full_fit,
        )
        clean_fold_summary = summarize_clean_folds(clean_fold_rows)

        write_json(OUTDIR / "clean_host_fold_inventory_v0_6.json", clean_fold_inventory)
        write_csv(OUTDIR / "clean_host_fold_prediction_v0_6.csv", clean_fold_rows)
        write_json(OUTDIR / "clean_host_fold_summary_v0_6.json", clean_fold_summary)
        write_json(OUTDIR / "clean_host_fold_permutation_summaries_v0_6.json", clean_perm_summaries)
        write_csv(OUTDIR / "clean_host_fold_permutation_details_v0_6.csv", clean_perm_details)

        matrix_rows = transfer_matrix(
            active_hosts,
            clean_hosts,
            stress_hosts_present,
            within_contrast,
            mapped_rows,
            c_factor,
            full_fit,
        )
        write_csv(OUTDIR / "stress_transfer_matrix_v0_6.csv", matrix_rows)

        final_status, readiness_score, next_wall, best_cases = decide_status(
            clean_fold_summary,
            host_rows,
            matrix_rows,
            alpha_summary,
        )

        make_plots(host_rows, clean_fold_rows, matrix_rows)

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

        write_csv(OUTDIR / "design_fit_comparison_v0_6.csv", design_fit_rows)

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
            "test_name": "TAIRID Boundary Prediction Battery v0.6",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Stress-host regime decomposition only. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_6": CLAIMS_V0_6,
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
            "alpha_summary": alpha_summary,
            "cohort_stats": {
                "all": all_stats,
                "clean_excluding_stress": clean_stats,
                "stress_only": stress_stats,
            },
            "clean_fold_summary": clean_fold_summary,
            "stress_hosts_present": stress_hosts_present,
            "host_regime_decomposition_top_negative": sorted(
                host_rows,
                key=lambda r: r.get("fixed_gain_alpha_clean", 0.0),
            )[:15],
            "host_regime_decomposition_stress_hosts": [
                row for row in host_rows if row.get("is_stress_host") is True
            ],
            "stress_transfer_matrix": matrix_rows,
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_6_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_6_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v0_6.json"),
                "host_regime_csv": str(OUTDIR / "host_regime_decomposition_v0_6.csv"),
                "clean_fold_csv": str(OUTDIR / "clean_host_fold_prediction_v0_6.csv"),
                "transfer_matrix_csv": str(OUTDIR / "stress_transfer_matrix_v0_6.csv"),
                "alpha_summary_json": str(OUTDIR / "cohort_alpha_summary_v0_6.json"),
                "plots": [
                    str(OUTDIR / "host_fixed_gain_under_clean_alpha_v0_6.png"),
                    str(OUTDIR / "host_local_alpha_v0_6.png"),
                    str(OUTDIR / "clean_host_fold_gain_v0_6.png"),
                ],
            },
            "interpretation": {
                "what_supports_regime_decomposition": (
                    "Clean-host transfer remains stable after excluding LMC/M31/N4536, while stress hosts show distinct alpha/gain/direction behavior."
                ),
                "what_supports_sampling_explanation": (
                    "Stress hosts mainly have low edge counts or weak nonzero contrast, with no distinct alpha/gain behavior."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only decomposes stress-host behavior from the v0.5.1 held-out prediction lane."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_6_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_6_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.6\n\n")
            f.write("Boundary: stress-host regime decomposition only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Alpha summary:\n")
            f.write(json.dumps(alpha_summary, indent=2, default=json_default) + "\n\n")
            f.write("Clean fold summary:\n")
            f.write(json.dumps(clean_fold_summary, indent=2, default=json_default) + "\n\n")
            f.write("Stress host rows:\n")
            f.write(json.dumps(summary["host_regime_decomposition_stress_hosts"], indent=2, default=json_default) + "\n\n")
            f.write("Transfer matrix:\n")
            f.write(json.dumps(matrix_rows, indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only decomposes stress-host behavior.\n")

        print("TAIRID Boundary Prediction Battery v0.6 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.6",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_6_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_6_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
