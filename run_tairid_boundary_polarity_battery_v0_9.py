#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.9
Regime penalty and anti-overfit audit.

Purpose:
v0.8 supported a three-regime M31 quarantine model:

    clean high-alpha hosts
    low-alpha hosts = LMC + SMC + N4536
    M31 = sign-break quarantine / zero transferred correction

But v0.8 added structure. This v0.9 audit asks whether that added structure
still wins after a complexity penalty.

Models audited from the v0.8 rerun:
    1. universal_one_coefficient
    2. two_coefficient_stress4
    3. three_regime_m31_quarantine

Anti-overfit question:
    Does the three-regime model still beat the two-coefficient stress4 model
    after penalizing one additional regime boundary?

Boundary:
This does not prove TAIRID.
This does not prove H0 resolution.
This does not prove new physics.
This only checks whether the v0.8 three-regime result survives complexity penalty.
"""

import csv
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_tairid_boundary_polarity_battery_v0_8 as v08


OUTDIR = Path("tairid_boundary_polarity_battery_v0_9_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

CLAIMS_V0_9 = {
    "battery_name": "TAIRID Boundary Prediction Battery v0.9",
    "scope": "Regime penalty and anti-overfit audit",
    "reason_for_test": (
        "v0.8 found that a three-regime M31 quarantine model beat the two-coefficient stress4 model. "
        "v0.9 checks whether that win survives added-complexity penalties."
    ),
    "models_compared": [
        "universal_one_coefficient",
        "two_coefficient_stress4",
        "three_regime_m31_quarantine",
    ],
    "complexity_units": {
        "universal_one_coefficient": 1,
        "two_coefficient_stress4": 2,
        "three_regime_m31_quarantine": 3,
    },
    "native_tairid_claim": (
        "A regime boundary should only be promoted if its predictive gain is larger than its added complexity. "
        "A special case that merely protects one host but weakens the broader model should remain diagnostic, not promoted."
    ),
    "primary_prediction": (
        "The three-regime M31 quarantine model should remain competitive or better than stress4 after AIC-like and BIC-like penalties."
    ),
    "failure_rule": (
        "If the three-regime model loses after penalty, it remains a protective diagnostic but should not be promoted as the current best predictive model."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only tests whether the v0.8 three-regime model survives anti-overfit penalty."
    ),
}


COMPLEXITY_UNITS = {
    "universal_one_coefficient": 1,
    "two_coefficient_stress4": 2,
    "three_regime_m31_quarantine": 3,
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


def read_csv(path):
    rows = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(dict(row))

    return rows


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


def penalty_table(model_summaries, n_eff_hosts, n_eff_edges):
    rows = []

    universal = next(
        (row for row in model_summaries if row.get("model_name") == "universal_one_coefficient"),
        None,
    )

    if not universal:
        raise RuntimeError("Missing universal_one_coefficient row in v0.8 model summary.")

    universal_gain = as_float(universal.get("model_total_fold_gain"))
    universal_k = COMPLEXITY_UNITS["universal_one_coefficient"]

    log_hosts = math.log(max(float(n_eff_hosts), 2.0))
    log_edges = math.log(max(float(n_eff_edges), 2.0))

    for row in model_summaries:
        model = row.get("model_name")
        k = COMPLEXITY_UNITS.get(model, 1)
        gain = as_float(row.get("model_total_fold_gain"))

        improvement_vs_universal = gain - universal_gain
        delta_k = k - universal_k

        aic_like_penalized_improvement = improvement_vs_universal - 2.0 * delta_k
        bic_host_penalized_improvement = improvement_vs_universal - log_hosts * delta_k
        bic_edge_penalized_improvement = improvement_vs_universal - log_edges * delta_k

        rows.append(
            {
                "model_name": model,
                "complexity_units": k,
                "delta_complexity_vs_universal": delta_k,
                "raw_total_fold_gain": gain,
                "raw_improvement_vs_universal": improvement_vs_universal,
                "aic_like_penalized_improvement_vs_universal": aic_like_penalized_improvement,
                "bic_host_penalized_improvement_vs_universal": bic_host_penalized_improvement,
                "bic_edge_penalized_improvement_vs_universal": bic_edge_penalized_improvement,
                "strict_pass_95_count": row.get("strict_pass_95_count"),
                "strict_pass_99_count": row.get("strict_pass_99_count"),
                "fold_improvement_count": row.get("fold_improvement_count"),
                "loho_improvement_count": row.get("loho_improvement_count"),
                "m31_loho_model_gain": row.get("m31_loho_model_gain"),
                "m31_loho_universal_gain": row.get("m31_loho_universal_gain"),
                "m31_loho_improvement": row.get("m31_loho_improvement"),
                "lmc_loho_model_gain": row.get("lmc_loho_model_gain"),
                "smc_loho_model_gain": row.get("smc_loho_model_gain"),
                "n4536_loho_model_gain": row.get("n4536_loho_model_gain"),
            }
        )

    rows = sorted(
        rows,
        key=lambda r: (
            -as_float(r.get("bic_edge_penalized_improvement_vs_universal")),
            -as_float(r.get("aic_like_penalized_improvement_vs_universal")),
            -as_float(r.get("raw_total_fold_gain")),
        ),
    )

    return rows


def pairwise_comparison(penalty_rows, model_a, model_b, n_eff_hosts, n_eff_edges):
    a = next((row for row in penalty_rows if row.get("model_name") == model_a), None)
    b = next((row for row in penalty_rows if row.get("model_name") == model_b), None)

    if not a or not b:
        return {
            "model_a": model_a,
            "model_b": model_b,
            "status": "missing_model",
        }

    k_a = as_float(a.get("complexity_units"))
    k_b = as_float(b.get("complexity_units"))
    gain_a = as_float(a.get("raw_total_fold_gain"))
    gain_b = as_float(b.get("raw_total_fold_gain"))
    delta_gain = gain_a - gain_b
    delta_k = k_a - k_b

    log_hosts = math.log(max(float(n_eff_hosts), 2.0))
    log_edges = math.log(max(float(n_eff_edges), 2.0))

    return {
        "model_a": model_a,
        "model_b": model_b,
        "meaning": "positive values mean model_a beats model_b",
        "raw_gain_difference": delta_gain,
        "delta_complexity_units": delta_k,
        "aic_like_penalized_difference": delta_gain - 2.0 * delta_k,
        "bic_host_penalized_difference": delta_gain - log_hosts * delta_k,
        "bic_edge_penalized_difference": delta_gain - log_edges * delta_k,
        "model_a_gain": gain_a,
        "model_b_gain": gain_b,
        "model_a_complexity": k_a,
        "model_b_complexity": k_b,
    }


def loho_anti_overfit(loho_rows):
    rows = []

    by_model = {}

    for row in loho_rows:
        model = row.get("model_name")
        by_model.setdefault(model, []).append(row)

    for model, model_rows in by_model.items():
        valid = [
            row for row in model_rows
            if as_float(row.get("positive_count")) > 0
            and as_float(row.get("negative_count")) > 0
        ]

        improved = [row for row in valid if as_float(row.get("model_minus_universal_gain")) > 0.0]
        positive = [row for row in valid if as_float(row.get("model_fixed_gain")) > 0.0]
        harmed = [row for row in valid if as_float(row.get("model_minus_universal_gain")) < 0.0]

        non_m31 = [row for row in valid if row.get("heldout_host") != "M31"]
        non_m31_improved = [
            row for row in non_m31
            if as_float(row.get("model_minus_universal_gain")) > 0.0
        ]

        m31 = next((row for row in valid if row.get("heldout_host") == "M31"), None)

        rows.append(
            {
                "model_name": model,
                "valid_loho_host_count": len(valid),
                "loho_improved_count": len(improved),
                "loho_improved_fraction": float(len(improved) / len(valid)) if valid else None,
                "loho_positive_gain_count": len(positive),
                "loho_positive_gain_fraction": float(len(positive) / len(valid)) if valid else None,
                "loho_harmed_count": len(harmed),
                "loho_harmed_hosts": ",".join([row.get("heldout_host", "") for row in harmed]),
                "non_m31_valid_host_count": len(non_m31),
                "non_m31_improved_count": len(non_m31_improved),
                "non_m31_improved_fraction": float(len(non_m31_improved) / len(non_m31)) if non_m31 else None,
                "m31_model_gain": as_float(m31.get("model_fixed_gain")) if m31 else None,
                "m31_universal_gain": as_float(m31.get("universal_fixed_gain")) if m31 else None,
                "m31_improvement": as_float(m31.get("model_minus_universal_gain")) if m31 else None,
            }
        )

    rows = sorted(rows, key=lambda r: -as_float(r.get("loho_improved_fraction")))

    return rows


def decide_status(penalty_rows, pairwise_rows, loho_rows):
    quarantine = next(
        (row for row in penalty_rows if row.get("model_name") == "three_regime_m31_quarantine"),
        {},
    )
    stress4 = next(
        (row for row in penalty_rows if row.get("model_name") == "two_coefficient_stress4"),
        {},
    )

    q_vs_stress4 = next(
        (
            row for row in pairwise_rows
            if row.get("model_a") == "three_regime_m31_quarantine"
            and row.get("model_b") == "two_coefficient_stress4"
        ),
        {},
    )

    q_loho = next(
        (row for row in loho_rows if row.get("model_name") == "three_regime_m31_quarantine"),
        {},
    )

    stress4_loho = next(
        (row for row in loho_rows if row.get("model_name") == "two_coefficient_stress4"),
        {},
    )

    beats_stress4_after_aic = as_float(q_vs_stress4.get("aic_like_penalized_difference")) > 0.0
    beats_stress4_after_bic_host = as_float(q_vs_stress4.get("bic_host_penalized_difference")) > 0.0
    beats_stress4_after_bic_edge = as_float(q_vs_stress4.get("bic_edge_penalized_difference")) > 0.0

    beats_universal_after_bic_edge = (
        as_float(quarantine.get("bic_edge_penalized_improvement_vs_universal")) > 0.0
    )

    m31_improved = as_float(quarantine.get("m31_loho_improvement")) > as_float(stress4.get("m31_loho_improvement"))

    non_m31_not_worse = (
        as_float(q_loho.get("non_m31_improved_fraction"))
        >= as_float(stress4_loho.get("non_m31_improved_fraction")) - 0.10
    )

    best_cases = {
        "three_regime_penalty": quarantine,
        "stress4_penalty": stress4,
        "three_regime_vs_stress4": q_vs_stress4,
        "three_regime_loho": q_loho,
        "stress4_loho": stress4_loho,
        "beats_stress4_after_aic": beats_stress4_after_aic,
        "beats_stress4_after_bic_host": beats_stress4_after_bic_host,
        "beats_stress4_after_bic_edge": beats_stress4_after_bic_edge,
        "beats_universal_after_bic_edge": beats_universal_after_bic_edge,
        "m31_improved_relative_to_stress4": m31_improved,
        "non_m31_not_worse": non_m31_not_worse,
    }

    if (
        beats_stress4_after_aic
        and beats_stress4_after_bic_host
        and beats_stress4_after_bic_edge
        and beats_universal_after_bic_edge
        and m31_improved
        and non_m31_not_worse
    ):
        return (
            "three_regime_model_survives_complexity_penalty",
            9,
            "The M31 quarantine model remains better than stress4 after complexity penalty and does not damage the broader LOHO field.",
            best_cases,
        )

    if (
        beats_stress4_after_aic
        and beats_stress4_after_bic_host
        and beats_universal_after_bic_edge
        and m31_improved
    ):
        return (
            "three_regime_model_supported_but_edge_bic_cautious",
            8,
            "The M31 quarantine model survives lighter penalties, but the strongest edge-count BIC penalty is cautious.",
            best_cases,
        )

    if m31_improved and not (beats_stress4_after_aic and beats_stress4_after_bic_host):
        return (
            "m31_quarantine_protective_not_promoted",
            7,
            "M31 quarantine helps the stress case but does not beat stress4 strongly enough after penalty.",
            best_cases,
        )

    return (
        "three_regime_model_not_supported_after_penalty",
        6,
        "The added M31 quarantine boundary does not survive the v0.9 anti-overfit audit.",
        best_cases,
    )


def make_plots(penalty_rows, pairwise_rows, loho_rows):
    try:
        if penalty_rows:
            rows = sorted(
                penalty_rows,
                key=lambda r: -as_float(r.get("bic_edge_penalized_improvement_vs_universal")),
            )
            x = np.arange(len(rows))

            plt.figure(figsize=(11, 5))
            plt.bar(x, [as_float(r.get("bic_edge_penalized_improvement_vs_universal")) for r in rows])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, [r["model_name"] for r in rows], rotation=25, ha="right")
            plt.ylabel("BIC-edge penalized improvement vs universal")
            plt.title("v0.9 complexity-penalized model comparison")
            plt.tight_layout()
            plt.savefig(OUTDIR / "complexity_penalized_model_comparison_v0_9.png", dpi=160)
            plt.close()

        q_pairs = [
            r for r in pairwise_rows
            if r.get("model_a") == "three_regime_m31_quarantine"
        ]

        if q_pairs:
            labels = [r["model_b"] for r in q_pairs]
            x = np.arange(len(q_pairs))

            plt.figure(figsize=(10, 5))
            plt.bar(x, [as_float(r.get("bic_edge_penalized_difference")) for r in q_pairs])
            plt.axhline(0.0, linewidth=1)
            plt.xticks(x, labels, rotation=25, ha="right")
            plt.ylabel("quarantine minus comparison, BIC-edge penalized")
            plt.title("v0.9 M31 quarantine pairwise penalty test")
            plt.tight_layout()
            plt.savefig(OUTDIR / "m31_quarantine_pairwise_penalty_v0_9.png", dpi=160)
            plt.close()

        if loho_rows:
            rows = sorted(loho_rows, key=lambda r: -as_float(r.get("loho_improved_fraction")))
            x = np.arange(len(rows))

            plt.figure(figsize=(10, 5))
            plt.bar(x, [as_float(r.get("loho_improved_fraction")) for r in rows])
            plt.xticks(x, [r["model_name"] for r in rows], rotation=25, ha="right")
            plt.ylabel("LOHO improvement fraction")
            plt.title("v0.9 leave-one-host-out anti-overfit check")
            plt.tight_layout()
            plt.savefig(OUTDIR / "loho_anti_overfit_fraction_v0_9.png", dpi=160)
            plt.close()

    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v0_9.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )


def main():
    started = {
        "test_name": "TAIRID Boundary Prediction Battery v0.9",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "runs_v0_8_first": True,
        "post_audit": "complexity penalty and anti-overfit audit",
    }

    write_json(OUTDIR / "claims_v0_9.json", CLAIMS_V0_9)
    write_json(OUTDIR / "v0_9_started.json", started)

    try:
        # Rerun v0.8 inside the v0.9 output folder so this artifact is self-contained.
        v08.OUTDIR = OUTDIR
        v08.DOWNLOAD_DIR = DOWNLOAD_DIR
        v08.v16.OUTDIR = OUTDIR
        v08.v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        v08.v02.OUTDIR = OUTDIR
        v08.v02.DOWNLOAD_DIR = DOWNLOAD_DIR
        v08.v05.OUTDIR = OUTDIR
        v08.v05.DOWNLOAD_DIR = DOWNLOAD_DIR
        v08.v06.OUTDIR = OUTDIR
        v08.v06.DOWNLOAD_DIR = DOWNLOAD_DIR

        print("Running v0.8 inside v0.9 output folder.")
        v08.main()

        summary_path = OUTDIR / "boundary_polarity_battery_v0_8_summary.json"
        model_summary_path = OUTDIR / "m31_regime_model_summary_v0_8.csv"
        loho_path = OUTDIR / "m31_regime_loho_predictions_v0_8.csv"

        if not summary_path.exists():
            raise FileNotFoundError(f"Missing expected v0.8 summary: {summary_path}")

        if not model_summary_path.exists():
            raise FileNotFoundError(f"Missing expected v0.8 model summary: {model_summary_path}")

        if not loho_path.exists():
            raise FileNotFoundError(f"Missing expected v0.8 LOHO predictions: {loho_path}")

        v08_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_summaries = read_csv(model_summary_path)
        loho_predictions = read_csv(loho_path)

        active_inventory = v08_summary.get("active_host_inventory", {})
        n_eff_hosts = int(active_inventory.get("active_host_count", 33))

        active_rows = active_inventory.get("active_host_rows", [])
        n_eff_edges = int(
            sum(
                int(row.get("edge_count", 0))
                for row in active_rows
            )
        )

        if n_eff_edges <= 0:
            n_eff_edges = 334

        penalties = penalty_table(model_summaries, n_eff_hosts, n_eff_edges)

        pairwise = [
            pairwise_comparison(
                penalties,
                "three_regime_m31_quarantine",
                "two_coefficient_stress4",
                n_eff_hosts,
                n_eff_edges,
            ),
            pairwise_comparison(
                penalties,
                "three_regime_m31_quarantine",
                "universal_one_coefficient",
                n_eff_hosts,
                n_eff_edges,
            ),
            pairwise_comparison(
                penalties,
                "two_coefficient_stress4",
                "universal_one_coefficient",
                n_eff_hosts,
                n_eff_edges,
            ),
        ]

        loho_audit = loho_anti_overfit(loho_predictions)

        final_status, readiness_score, next_wall, best_cases = decide_status(
            penalties,
            pairwise,
            loho_audit,
        )

        write_csv(OUTDIR / "complexity_penalty_table_v0_9.csv", penalties)
        write_csv(OUTDIR / "pairwise_penalty_comparison_v0_9.csv", pairwise)
        write_csv(OUTDIR / "loho_anti_overfit_audit_v0_9.csv", loho_audit)
        write_json(OUTDIR / "best_cases_v0_9.json", best_cases)

        make_plots(penalties, pairwise, loho_audit)

        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.9",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Regime complexity penalty and anti-overfit audit only. "
                "This reruns v0.8 and then tests whether the three-regime M31 quarantine survives complexity penalty. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "claims_v0_9": CLAIMS_V0_9,
            "v0_8_original_status": v08_summary.get("final_status"),
            "v0_8_original_readiness_score": v08_summary.get("readiness_score_0_to_10"),
            "effective_sample_sizes": {
                "n_eff_hosts": n_eff_hosts,
                "n_eff_edges": n_eff_edges,
                "host_log_penalty": math.log(max(float(n_eff_hosts), 2.0)),
                "edge_log_penalty": math.log(max(float(n_eff_edges), 2.0)),
            },
            "complexity_penalty_table": penalties,
            "pairwise_penalty_comparison": pairwise,
            "loho_anti_overfit_audit": loho_audit,
            "best_cases": best_cases,
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v0_9_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v0_9_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v0_9.json"),
                "complexity_penalty_csv": str(OUTDIR / "complexity_penalty_table_v0_9.csv"),
                "pairwise_penalty_csv": str(OUTDIR / "pairwise_penalty_comparison_v0_9.csv"),
                "loho_anti_overfit_csv": str(OUTDIR / "loho_anti_overfit_audit_v0_9.csv"),
                "v0_8_summary_json": str(OUTDIR / "boundary_polarity_battery_v0_8_summary.json"),
                "plots": [
                    str(OUTDIR / "complexity_penalized_model_comparison_v0_9.png"),
                    str(OUTDIR / "m31_quarantine_pairwise_penalty_v0_9.png"),
                    str(OUTDIR / "loho_anti_overfit_fraction_v0_9.png"),
                ],
            },
            "interpretation": {
                "what_supports_promotion": (
                    "The three-regime model beats universal and stress4 after AIC-like and BIC-like complexity penalties, "
                    "improves M31, and does not damage the non-M31 LOHO field."
                ),
                "what_keeps_it_diagnostic": (
                    "The model helps M31 but loses after penalty or causes broad LOHO harm."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only checks whether the v0.8 three-regime model survives an anti-overfit audit."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v0_9_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v0_9_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v0.9\n\n")
            f.write("Boundary: regime penalty and anti-overfit audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Effective sample sizes:\n")
            f.write(json.dumps(summary["effective_sample_sizes"], indent=2, default=json_default) + "\n\n")
            f.write("Complexity penalty table:\n")
            f.write(json.dumps(penalties, indent=2, default=json_default) + "\n\n")
            f.write("Pairwise penalty comparison:\n")
            f.write(json.dumps(pairwise, indent=2, default=json_default) + "\n\n")
            f.write("LOHO anti-overfit audit:\n")
            f.write(json.dumps(loho_audit, indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only checks whether v0.8 survives complexity penalty.\n")

        finished = {
            **started,
            "status": "success",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
        }
        write_json(OUTDIR / "v0_9_finished.json", finished)

        print("TAIRID Boundary Prediction Battery v0.9 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        error_summary = {
            "test_name": "TAIRID Boundary Prediction Battery v0.9",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v0_9_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(OUTDIR / "boundary_polarity_battery_v0_9_summary.json", error_summary)
        print(error_summary["traceback"])
        raise


if __name__ == "__main__":
    main()
