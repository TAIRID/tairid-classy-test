#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.5.1
Positive-gain strict held-out prediction audit.

Purpose:
v0.5 showed held-out-host predictive transfer for within-host F160W polarity,
but one fold had negative fixed-coefficient gain even though it passed the
permutation threshold.

This v0.5.1 audit reruns v0.5 and then applies a stricter rule:

    A held-out fold only passes if:
        1. training alpha is positive,
        2. held-out signed direction is positive,
        3. held-out fixed-coefficient gain is positive,
        4. held-out gain beats permutation.

It also isolates the main stress cases from v0.5:
    LMC
    M31
    N4536

Boundary:
This does not change the v0.5 science logic.
This does not prove TAIRID.
This does not prove H0 resolution.
This only tightens the held-out prediction decision rule and identifies stress hosts.
"""

import csv
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_tairid_boundary_polarity_battery_v0_5 as v05


OUTDIR = Path("tairid_boundary_polarity_battery_v0_5_1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

STRESS_HOSTS_PREDECLARED = {"LMC", "M31", "N4536"}

CLAIMS_V0_5_1 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.5.1",
    "scope": "Strict positive-gain held-out-host prediction audit",
    "reason_for_test": (
        "v0.5 found held-out transfer, but one fold had negative fixed-coefficient gain. "
        "v0.5.1 requires positive gain as part of the held-out pass condition."
    ),
    "native_tairid_claim": (
        "A predictive boundary-polarity coefficient should not only beat a weak null threshold. "
        "It should improve held-out residual structure in the predicted direction."
    ),
    "strict_pass_rule": (
        "A fold passes strictly only when alpha_train > 0, held-out direction is positive, "
        "fixed alpha gain > 0, and the gain/score beat permutation."
    ),
    "stress_case_rule": (
        "Hosts with negative fixed gain or wrong signed direction are not hidden. "
        "They are reported as stress cases that constrain the claim."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only tests whether the v0.5 held-out prediction "
        "survives a stricter positive-gain rule."
    ),
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


def read_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


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


def as_float(value, default=0.0):
    try:
        out = float(value)
    except Exception:
        return default

    if not math.isfinite(out):
        return default

    return out


def as_bool(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def split_hosts(hosts_text):
    if not hosts_text:
        return []

    return [
        item.strip()
        for item in str(hosts_text).split(",")
        if item.strip()
    ]


def strict_fold_pass(row, level="95"):
    suffix = "95" if level == "95" else "99"

    return bool(
        as_bool(row.get(f"heldout_pass_{suffix}"))
        and as_bool(row.get("alpha_train_positive"))
        and as_bool(row.get("test_direction_positive"))
        and as_float(row.get("test_fixed_alpha_gain")) > 0.0
    )


def classify_fold_rows(fold_rows):
    out = []

    for row in fold_rows:
        hosts = split_hosts(row.get("test_hosts", ""))
        stress_hosts_present = sorted([h for h in hosts if h in STRESS_HOSTS_PREDECLARED])

        new = dict(row)
        new["stress_hosts_present"] = ",".join(stress_hosts_present)
        new["contains_predeclared_stress_host"] = bool(stress_hosts_present)
        new["strict_positive_gain_pass_95"] = strict_fold_pass(row, "95")
        new["strict_positive_gain_pass_99"] = strict_fold_pass(row, "99")
        new["strict_failure_reasons"] = "; ".join(strict_failure_reasons(row))
        out.append(new)

    return out


def strict_failure_reasons(row):
    reasons = []

    if not as_bool(row.get("alpha_train_positive")):
        reasons.append("alpha_train_not_positive")

    if not as_bool(row.get("test_direction_positive")):
        reasons.append("heldout_direction_not_positive")

    if as_float(row.get("test_fixed_alpha_gain")) <= 0.0:
        reasons.append("fixed_gain_not_positive")

    if not as_bool(row.get("heldout_pass_95")):
        reasons.append("does_not_beat_95_permutation")

    if not as_bool(row.get("heldout_pass_99")):
        reasons.append("does_not_beat_99_permutation")

    return reasons


def classify_loho_rows(loho_rows):
    out = []

    for row in loho_rows:
        host = row.get("heldout_host", "")
        new = dict(row)
        new["is_predeclared_stress_host"] = host in STRESS_HOSTS_PREDECLARED
        new["strict_positive_transfer"] = bool(
            as_bool(row.get("alpha_train_positive"))
            and as_bool(row.get("test_direction_positive"))
            and as_float(row.get("test_fixed_alpha_gain")) > 0.0
        )
        new["stress_case"] = bool(
            as_float(row.get("test_fixed_alpha_gain")) <= 0.0
            or not as_bool(row.get("test_direction_positive"))
        )
        new["stress_reasons"] = "; ".join(loho_stress_reasons(row))
        out.append(new)

    return out


def loho_stress_reasons(row):
    reasons = []

    if as_float(row.get("test_fixed_alpha_gain")) <= 0.0:
        reasons.append("fixed_gain_not_positive")

    if not as_bool(row.get("test_direction_positive")):
        reasons.append("direction_not_positive")

    if not as_bool(row.get("alpha_train_positive")):
        reasons.append("alpha_train_not_positive")

    return reasons


def summarize_folds(strict_folds):
    valid = [
        row for row in strict_folds
        if as_float(row.get("test_positive_count")) > 0
        and as_float(row.get("test_negative_count")) > 0
    ]

    strict95 = [row for row in valid if as_bool(row.get("strict_positive_gain_pass_95"))]
    strict99 = [row for row in valid if as_bool(row.get("strict_positive_gain_pass_99"))]
    positive_gain = [row for row in valid if as_float(row.get("test_fixed_alpha_gain")) > 0.0]
    positive_direction = [row for row in valid if as_bool(row.get("test_direction_positive"))]
    stress_fold_rows = [row for row in valid if as_bool(row.get("contains_predeclared_stress_host"))]

    gain_values = [as_float(row.get("test_fixed_alpha_gain")) for row in valid]

    return {
        "valid_fold_count": len(valid),
        "strict_positive_gain_pass_95_count": len(strict95),
        "strict_positive_gain_pass_99_count": len(strict99),
        "strict_positive_gain_pass_95_fraction": float(len(strict95) / len(valid)) if valid else None,
        "strict_positive_gain_pass_99_fraction": float(len(strict99) / len(valid)) if valid else None,
        "positive_gain_fold_count": len(positive_gain),
        "positive_gain_fraction": float(len(positive_gain) / len(valid)) if valid else None,
        "positive_direction_fold_count": len(positive_direction),
        "positive_direction_fraction": float(len(positive_direction) / len(valid)) if valid else None,
        "total_fixed_gain": float(np.sum(gain_values)) if valid else 0.0,
        "mean_fixed_gain": float(np.mean(gain_values)) if valid else 0.0,
        "min_fixed_gain": float(np.min(gain_values)) if valid else None,
        "max_fixed_gain": float(np.max(gain_values)) if valid else None,
        "stress_fold_count": len(stress_fold_rows),
        "stress_fold_indices": [
            row.get("fold_index")
            for row in stress_fold_rows
        ],
    }


def summarize_loho(strict_loho):
    valid = [
        row for row in strict_loho
        if as_float(row.get("host_positive_count")) > 0
        and as_float(row.get("host_negative_count")) > 0
    ]

    positive_transfer = [row for row in valid if as_bool(row.get("strict_positive_transfer"))]
    stress_cases = [row for row in valid if as_bool(row.get("stress_case"))]

    non_stress_host_rows = [
        row for row in valid
        if row.get("heldout_host") not in STRESS_HOSTS_PREDECLARED
    ]

    non_stress_positive_transfer = [
        row for row in non_stress_host_rows
        if as_bool(row.get("strict_positive_transfer"))
    ]

    predeclared_stress_rows = [
        row for row in valid
        if row.get("heldout_host") in STRESS_HOSTS_PREDECLARED
    ]

    return {
        "valid_host_count": len(valid),
        "strict_positive_transfer_count": len(positive_transfer),
        "strict_positive_transfer_fraction": float(len(positive_transfer) / len(valid)) if valid else None,
        "stress_case_count": len(stress_cases),
        "stress_case_hosts": [
            row.get("heldout_host")
            for row in stress_cases
        ],
        "non_predeclared_stress_valid_host_count": len(non_stress_host_rows),
        "non_predeclared_stress_positive_transfer_count": len(non_stress_positive_transfer),
        "non_predeclared_stress_positive_transfer_fraction": (
            float(len(non_stress_positive_transfer) / len(non_stress_host_rows))
            if non_stress_host_rows else None
        ),
        "predeclared_stress_host_rows": predeclared_stress_rows,
        "worst_hosts_by_fixed_gain": sorted(
            valid,
            key=lambda row: as_float(row.get("test_fixed_alpha_gain")),
        )[:10],
        "best_hosts_by_fixed_gain": sorted(
            valid,
            key=lambda row: -as_float(row.get("test_fixed_alpha_gain")),
        )[:10],
    }


def cohort_summary(strict_loho, excluded_hosts):
    excluded_hosts = set(excluded_hosts)

    rows = [
        row for row in strict_loho
        if row.get("heldout_host") not in excluded_hosts
        and as_float(row.get("host_positive_count")) > 0
        and as_float(row.get("host_negative_count")) > 0
    ]

    positive = [row for row in rows if as_bool(row.get("strict_positive_transfer"))]
    stress = [row for row in rows if as_bool(row.get("stress_case"))]

    gains = [as_float(row.get("test_fixed_alpha_gain")) for row in rows]

    return {
        "excluded_hosts": sorted(excluded_hosts),
        "valid_host_count": len(rows),
        "positive_transfer_count": len(positive),
        "positive_transfer_fraction": float(len(positive) / len(rows)) if rows else None,
        "stress_case_count": len(stress),
        "stress_case_hosts": [row.get("heldout_host") for row in stress],
        "total_fixed_gain": float(np.sum(gains)) if rows else 0.0,
        "mean_fixed_gain": float(np.mean(gains)) if rows else 0.0,
        "min_fixed_gain": float(np.min(gains)) if rows else None,
        "max_fixed_gain": float(np.max(gains)) if rows else None,
    }


def decide_status(fold_summary, loho_summary, cohort_summaries):
    strict95 = fold_summary.get("strict_positive_gain_pass_95_count", 0)
    strict99 = fold_summary.get("strict_positive_gain_pass_99_count", 0)
    valid_folds = fold_summary.get("valid_fold_count", 0)
    positive_gain_fraction = fold_summary.get("positive_gain_fraction") or 0.0
    total_gain = fold_summary.get("total_fixed_gain") or 0.0

    loho_fraction = loho_summary.get("strict_positive_transfer_fraction") or 0.0
    non_stress_fraction = loho_summary.get("non_predeclared_stress_positive_transfer_fraction") or 0.0

    excluding_lmc_m31 = next(
        (
            item for item in cohort_summaries
            if set(item.get("excluded_hosts", [])) == {"LMC", "M31"}
        ),
        {},
    )

    excluding_lmc_m31_fraction = excluding_lmc_m31.get("positive_transfer_fraction") or 0.0

    best_cases = {
        "fold_summary": fold_summary,
        "loho_summary": loho_summary,
        "cohort_summaries": cohort_summaries,
    }

    if (
        valid_folds >= 5
        and strict99 >= 4
        and positive_gain_fraction >= 0.80
        and total_gain > 0.0
        and loho_fraction >= 0.85
    ):
        return (
            "strict_heldout_prediction_supported_with_stress_cases",
            8,
            "Strict positive-gain held-out prediction holds in most folds and most hosts, with isolated stress cases.",
            best_cases,
        )

    if (
        valid_folds >= 5
        and strict95 >= 4
        and positive_gain_fraction >= 0.80
        and total_gain > 0.0
        and excluding_lmc_m31_fraction >= 0.90
    ):
        return (
            "strict_heldout_prediction_supported_after_lmc_m31_isolation",
            8,
            "Strict held-out prediction is strong after isolating LMC/M31 stress behavior.",
            best_cases,
        )

    if (
        valid_folds >= 5
        and positive_gain_fraction >= 0.60
        and total_gain > 0.0
        and non_stress_fraction >= 0.80
    ):
        return (
            "strict_heldout_prediction_directional_not_locked",
            7,
            "Strict held-out transfer is directional but not strong enough to lock under v0.5.1.",
            best_cases,
        )

    return (
        "strict_heldout_prediction_not_supported",
        6,
        "The stricter positive-gain rule does not support held-out predictive transfer.",
        best_cases,
    )


def make_plots(strict_folds, strict_loho):
    if strict_folds:
        rows = sorted(strict_folds, key=lambda row: int(as_float(row.get("fold_index"))))
        x = np.arange(len(rows))

        plt.figure(figsize=(9, 5))
        plt.bar(x, [as_float(row.get("test_fixed_alpha_gain")) for row in rows])
        plt.axhline(0.0, linewidth=1)
        plt.xticks(x, [str(row.get("fold_index")) for row in rows])
        plt.xlabel("held-out host fold")
        plt.ylabel("fixed-coefficient gain")
        plt.title("v0.5.1 strict held-out fold gain")
        plt.tight_layout()
        plt.savefig(OUTDIR / "strict_heldout_fold_fixed_gain_v0_5_1.png", dpi=160)
        plt.close()

    if strict_loho:
        rows = sorted(strict_loho, key=lambda row: as_float(row.get("test_fixed_alpha_gain")))
        rows = rows[:15]
        x = np.arange(len(rows))

        plt.figure(figsize=(12, 5))
        plt.bar(x, [as_float(row.get("test_fixed_alpha_gain")) for row in rows])
        plt.axhline(0.0, linewidth=1)
        plt.xticks(x, [row.get("heldout_host") for row in rows], rotation=60, ha="right")
        plt.ylabel("fixed-coefficient gain")
        plt.title("v0.5.1 worst leave-one-host-out gains")
        plt.tight_layout()
        plt.savefig(OUTDIR / "strict_loho_worst_fixed_gain_v0_5_1.png", dpi=160)
        plt.close()


def main():
    patch_summary = {
        "test_name": "TAIRID Boundary Prediction Battery v0.5.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "science_logic_changed_from_v0_5": False,
        "post_audit_logic_added": True,
        "strict_requirement_added": "held-out fixed coefficient gain must be positive",
    }

    write_json(OUTDIR / "claims_v0_5_1.json", CLAIMS_V0_5_1)
    write_json(OUTDIR / "v0_5_1_patch_summary_started.json", patch_summary)

    try:
        # Redirect the v0.5 rerun into v0.5.1 outputs.
        v05.OUTDIR = OUTDIR
        v05.DOWNLOAD_DIR = DOWNLOAD_DIR
        v05.v16.OUTDIR = OUTDIR
        v05.v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        v05.v02.OUTDIR = OUTDIR
        v05.v02.DOWNLOAD_DIR = DOWNLOAD_DIR

        print("Running v0.5 inside v0.5.1 output folder.")
        print("Then applying strict positive-gain audit.")

        v05.main()

        fold_path = OUTDIR / "heldout_fold_prediction_summary_v0_5.csv"
        loho_path = OUTDIR / "leave_one_host_out_prediction_summary_v0_5.csv"
        v05_summary_path = OUTDIR / "boundary_polarity_battery_v0_5_summary.json"

        if not fold_path.exists():
            raise FileNotFoundError(f"Missing expected v0.5 fold summary: {fold_path}")

        if not loho_path.exists():
            raise FileNotFoundError(f"Missing expected v0.5 LOHO summary: {loho_path}")

        fold_rows = read_csv(fold_path)
        loho_rows = read_csv(loho_path)

        strict_folds = classify_fold_rows(fold_rows)
        strict_loho = classify_loho_rows(loho_rows)

        fold_summary = summarize_folds(strict_folds)
        loho_summary = summarize_loho(strict_loho)

        cohort_summaries = [
            cohort_summary(strict_loho, []),
            cohort_summary(strict_loho, ["LMC"]),
            cohort_summary(strict_loho, ["M31"]),
            cohort_summary(strict_loho, ["N4536"]),
            cohort_summary(strict_loho, ["LMC", "M31"]),
            cohort_summary(strict_loho, ["LMC", "M31", "N4536"]),
        ]

        final_status, readiness_score, next_wall, best_cases = decide_status(
            fold_summary,
            loho_summary,
            cohort_summaries,
        )

        write_csv(OUTDIR / "strict_heldout_fold_audit_v0_5_1.csv", strict_folds)
        write_csv(OUTDIR / "strict_leave_one_host_out_audit_v0_5_1.csv", strict_loho)
        write_json(OUTDIR / "strict_fold_summary_v0_5_1.json", fold_summary)
        write_json(OUTDIR / "strict_loho_summary_v0_5_1.json", loho_summary)
        write_json(OUTDIR / "stress_host_cohort_summaries_v0_5_1.json", cohort_summaries)

        make_plots(strict_folds, strict_loho)

        v05_summary = {}
        if v05_summary_path.exists():
            v05_summary = json.loads(v05_summary_path.read_text(encoding="utf-8"))

        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.5.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Strict positive-gain held-out prediction audit only. "
                "This reruns v0.5 and tightens the decision rule. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_5_1": CLAIMS_V0_5_1,
            "v0_5_original_status": v05_summary.get("final_status"),
            "v0_5_original_readiness_score": v05_summary.get("readiness_score_0_to_10"),
            "fold_summary": fold_summary,
            "loho_summary": loho_summary,
            "stress_host_cohort_summaries": cohort_summaries,
            "strict_heldout_folds": strict_folds,
            "strict_leave_one_host_out_top_negative": sorted(
                strict_loho,
                key=lambda row: as_float(row.get("test_fixed_alpha_gain")),
            )[:20],
            "strict_leave_one_host_out_top_positive": sorted(
                strict_loho,
                key=lambda row: -as_float(row.get("test_fixed_alpha_gain")),
            )[:20],
            "best_cases": best_cases,
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_5_1_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_5_1_summary.txt"),
                "strict_fold_csv": str(OUTDIR / "strict_heldout_fold_audit_v0_5_1.csv"),
                "strict_loho_csv": str(OUTDIR / "strict_leave_one_host_out_audit_v0_5_1.csv"),
                "fold_summary_json": str(OUTDIR / "strict_fold_summary_v0_5_1.json"),
                "loho_summary_json": str(OUTDIR / "strict_loho_summary_v0_5_1.json"),
                "cohort_summary_json": str(OUTDIR / "stress_host_cohort_summaries_v0_5_1.json"),
                "plots": [
                    str(OUTDIR / "strict_heldout_fold_fixed_gain_v0_5_1.png"),
                    str(OUTDIR / "strict_loho_worst_fixed_gain_v0_5_1.png"),
                ],
            },
            "interpretation": {
                "what_supports_predictive_transfer": (
                    "Most held-out folds and most leave-one-host-out cases retain positive gain, positive direction, "
                    "and permutation survival under the stricter rule."
                ),
                "what_identifies_stress_cases": (
                    "Hosts with negative fixed gain or wrong signed direction are listed as stress cases rather than hidden."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only checks whether v0.5 held-out prediction survives a stricter positive-gain rule."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_5_1_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_5_1_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.5.1\n\n")
            f.write("Boundary: strict positive-gain held-out prediction audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Strict fold summary:\n")
            f.write(json.dumps(fold_summary, indent=2, default=json_default) + "\n\n")
            f.write("Strict leave-one-host-out summary:\n")
            f.write(json.dumps(loho_summary, indent=2, default=json_default) + "\n\n")
            f.write("Stress host cohort summaries:\n")
            f.write(json.dumps(cohort_summaries, indent=2, default=json_default) + "\n\n")
            f.write("Strict held-out folds:\n")
            f.write(json.dumps(strict_folds, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only tightens the held-out-host prediction audit.\n")

        patch_summary["status"] = "success"
        patch_summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        patch_summary["final_status"] = final_status
        patch_summary["readiness_score_0_to_10"] = readiness_score
        write_json(OUTDIR / "v0_5_1_patch_summary_final.json", patch_summary)

        print("TAIRID Boundary Prediction Battery v0.5.1 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        patch_summary["status"] = "failed"
        patch_summary["error"] = repr(exc)
        patch_summary["traceback"] = traceback.format_exc()
        patch_summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(OUTDIR / "v0_5_1_patch_summary_final.json", patch_summary)

        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.5.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_5_1_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_5_1_summary.json", summary)

        print(patch_summary["traceback"])
        raise


if __name__ == "__main__":
    main()
