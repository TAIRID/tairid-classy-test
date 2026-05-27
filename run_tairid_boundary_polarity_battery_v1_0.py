#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v1.0
Frozen three-regime replay and external-lane readiness audit.

Purpose:
v0.9 showed that the three-regime M31 quarantine model survived complexity penalty.

This v1.0 test freezes the current rule exactly:

    Dataset lane:
        SH0ES Table2 residual layer only.

    Variable:
        F160W-like Table2 numeric column.

    Edge rule:
        Within-host high 5% F160W minus within-host low 5% F160W.

    Regime rule:
        Regime 1: clean high-alpha hosts.
        Regime 2: low-alpha/reference hosts = LMC + SMC + N4536.
        Regime 3: M31 sign-break quarantine = zero transferred correction.

    Training rule:
        In held-out replay, coefficients must be learned only from training hosts.
        Held-out hosts cannot be refit.

This is a freeze test, not a new search.
No new variables.
No new host additions.
No new regime discovery.

Boundary:
This does not prove TAIRID.
This does not prove H0 resolution.
This does not prove new physics.
This only locks the current SH0ES Table2 lane result if the frozen replay survives.
"""

import csv
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import run_tairid_boundary_polarity_battery_v0_9 as v09


OUTDIR = Path("tairid_boundary_polarity_battery_v1_0_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_MODEL = {
    "model_name": "TAIRID_F160W_within_host_three_regime_freeze_v1_0",
    "model_status": "frozen replay candidate",
    "dataset_lane": "SH0ES Table2 residual layer only",
    "allowed_variable": "F160W-like Table2 numeric column, table2_num_7",
    "allowed_edge_rule": "within-host high 5% F160W minus within-host low 5% F160W",
    "regime_1": {
        "name": "clean_high_alpha_hosts",
        "hosts": "all active hosts except LMC, SMC, N4536, and M31",
        "coefficient_rule": "learn alpha_clean from training clean hosts only",
    },
    "regime_2": {
        "name": "low_alpha_reference_hosts",
        "hosts": ["LMC", "SMC", "N4536"],
        "coefficient_rule": "learn alpha_low from training low-alpha hosts only; fall back only according to frozen script behavior if missing in a fold",
    },
    "regime_3": {
        "name": "m31_sign_break_quarantine",
        "hosts": ["M31"],
        "coefficient_rule": "zero transferred correction; do not force clean or low-alpha coefficient onto M31",
    },
    "training_boundary": "held-out hosts are never refit",
    "anti_ad_hoc_boundary": [
        "No new variables allowed in this freeze.",
        "No new host may be added to the low-alpha group.",
        "No new sign-break host may be added.",
        "No H0, new physics, or SH0ES-error claim may be made from this lane alone.",
    ],
}

CLAIMS_V1_0 = {
    "battery_name": "TAIRID Boundary Polarity Battery v1.0",
    "scope": "Frozen three-regime replay and external-lane readiness audit",
    "reason_for_test": (
        "v0.9 supported the three-regime M31 quarantine model after complexity penalty. "
        "v1.0 freezes the rule and reruns it as a model-card / lock test."
    ),
    "frozen_rule": FROZEN_MODEL,
    "primary_prediction": (
        "The frozen three-regime model should replay the v0.9 result, preserve the complexity-penalized win, "
        "protect M31 without harming the broader field, and produce a clear external-lane readiness packet."
    ),
    "failure_rule": (
        "If the frozen replay does not reproduce the complexity-penalized result, this model is not locked. "
        "If it reproduces only by depending on extra tuning, it remains a diagnostic, not a frozen model."
    ),
    "truth_boundary": (
        "This cannot prove TAIRID or H0 correction. It only freezes the current SH0ES Table2 result and prepares the next independent validation lane."
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


def as_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def get_model(rows, model_name):
    return next((row for row in rows if row.get("model_name") == model_name), {})


def get_pair(rows, model_a, model_b):
    return next(
        (
            row for row in rows
            if row.get("model_a") == model_a
            and row.get("model_b") == model_b
        ),
        {},
    )


def get_host(rows, host):
    return next((row for row in rows if row.get("host") == host), {})


def gate(name, passed, detail, evidence=None, severity="lock"):
    return {
        "gate": name,
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
        "evidence": evidence or {},
    }


def build_gate_table(v08_summary, v09_summary, penalties, pairwise, loho_audit, local_rows):
    v08_status = v08_summary.get("final_status")
    v09_status = v09_summary.get("final_status")

    model_summaries = v08_summary.get("model_summaries", [])

    universal = get_model(model_summaries, "universal_one_coefficient")
    stress4 = get_model(model_summaries, "two_coefficient_stress4")
    quarantine = get_model(model_summaries, "three_regime_m31_quarantine")

    q_penalty = get_model(penalties, "three_regime_m31_quarantine")
    stress4_penalty = get_model(penalties, "two_coefficient_stress4")
    q_vs_stress4 = get_pair(
        pairwise,
        "three_regime_m31_quarantine",
        "two_coefficient_stress4",
    )

    q_loho = get_model(loho_audit, "three_regime_m31_quarantine")
    stress4_loho = get_model(loho_audit, "two_coefficient_stress4")

    m31_local = get_host(local_rows, "M31")
    lmc_local = get_host(local_rows, "LMC")
    smc_local = get_host(local_rows, "SMC")
    n4536_local = get_host(local_rows, "N4536")

    q_total = as_float(quarantine.get("model_total_fold_gain"))
    stress4_total = as_float(stress4.get("model_total_fold_gain"))
    universal_total = as_float(universal.get("model_total_fold_gain"))

    q_bic_edge_vs_stress4 = as_float(q_vs_stress4.get("bic_edge_penalized_difference"))
    q_aic_vs_stress4 = as_float(q_vs_stress4.get("aic_like_penalized_difference"))
    q_bic_host_vs_stress4 = as_float(q_vs_stress4.get("bic_host_penalized_difference"))

    m31_q_gain = as_float(quarantine.get("m31_loho_model_gain"))
    m31_stress4_gain = as_float(stress4.get("m31_loho_model_gain"))

    q_non_m31 = as_float(q_loho.get("non_m31_improved_fraction"))
    stress4_non_m31 = as_float(stress4_loho.get("non_m31_improved_fraction"))

    gates = [
        gate(
            "G1_v0_8_replay_status",
            v08_status == "m31_sign_break_quarantine_supported_and_better",
            "v0.8 must reproduce the M31 quarantine positive status.",
            {"v0_8_status": v08_status},
        ),
        gate(
            "G2_v0_9_penalty_status",
            v09_status == "three_regime_model_survives_complexity_penalty",
            "v0.9 must reproduce the complexity-penalty survival status.",
            {"v0_9_status": v09_status},
        ),
        gate(
            "G3_frozen_model_beats_universal_and_stress4_raw",
            q_total > stress4_total and q_total > universal_total,
            "Frozen three-regime model must beat universal and stress4 on total held-out fold gain.",
            {
                "universal_total_fold_gain": universal_total,
                "stress4_total_fold_gain": stress4_total,
                "three_regime_total_fold_gain": q_total,
            },
        ),
        gate(
            "G4_frozen_model_survives_penalty_vs_stress4",
            q_aic_vs_stress4 > 0.0 and q_bic_host_vs_stress4 > 0.0 and q_bic_edge_vs_stress4 > 0.0,
            "Frozen three-regime model must beat stress4 after AIC-like, BIC-host, and BIC-edge penalties.",
            {
                "aic_like_penalized_difference": q_aic_vs_stress4,
                "bic_host_penalized_difference": q_bic_host_vs_stress4,
                "bic_edge_penalized_difference": q_bic_edge_vs_stress4,
            },
        ),
        gate(
            "G5_fold_pass_replay",
            as_int(quarantine.get("valid_fold_count")) >= 5
            and as_int(quarantine.get("strict_pass_95_count")) >= 5
            and as_int(quarantine.get("strict_pass_99_count")) >= 5,
            "Frozen model must preserve full fold-pass replay.",
            {
                "valid_fold_count": quarantine.get("valid_fold_count"),
                "strict_pass_95_count": quarantine.get("strict_pass_95_count"),
                "strict_pass_99_count": quarantine.get("strict_pass_99_count"),
            },
        ),
        gate(
            "G6_m31_quarantine_protects_sign_break",
            m31_q_gain >= m31_stress4_gain and as_float(m31_local.get("local_alpha")) < 0.0,
            "M31 must remain the sign-break host and quarantine must protect it better than stress4.",
            {
                "m31_quarantine_gain": m31_q_gain,
                "m31_stress4_gain": m31_stress4_gain,
                "m31_local_alpha": m31_local.get("local_alpha"),
                "m31_direction_positive": m31_local.get("direction_positive"),
            },
        ),
        gate(
            "G7_low_alpha_hosts_remain_positive_local_alpha",
            as_float(lmc_local.get("local_alpha")) > 0.0
            and as_float(smc_local.get("local_alpha")) > 0.0
            and as_float(n4536_local.get("local_alpha")) > 0.0,
            "LMC, SMC, and N4536 must remain low-alpha positive hosts, not sign-break hosts.",
            {
                "lmc_local_alpha": lmc_local.get("local_alpha"),
                "smc_local_alpha": smc_local.get("local_alpha"),
                "n4536_local_alpha": n4536_local.get("local_alpha"),
            },
        ),
        gate(
            "G8_non_m31_field_not_damaged",
            q_non_m31 >= stress4_non_m31 - 0.10,
            "M31 quarantine must not damage the broader non-M31 LOHO field by more than the allowed margin.",
            {
                "three_regime_non_m31_improved_fraction": q_non_m31,
                "stress4_non_m31_improved_fraction": stress4_non_m31,
                "allowed_margin": 0.10,
            },
        ),
        gate(
            "G9_truth_boundary_documented",
            True,
            "The model is locked only for SH0ES Table2 residual-layer prediction, not proof of H0 correction or new physics.",
            {
                "not_allowed_claims": [
                    "TAIRID proves H0 correction",
                    "TAIRID proves new physics",
                    "TAIRID proves SH0ES is wrong",
                ]
            },
            severity="documentation",
        ),
    ]

    return gates


def decide_lock_status(gates):
    lock_gates = [g for g in gates if g.get("severity") == "lock"]
    passed = [g for g in lock_gates if g.get("passed") is True]
    failed = [g for g in lock_gates if g.get("passed") is not True]

    if len(failed) == 0:
        return (
            "frozen_three_regime_model_locked_for_table2_lane",
            9,
            "The frozen three-regime F160W model replayed successfully and is ready for an independent validation lane.",
            {
                "passed_lock_gates": len(passed),
                "failed_lock_gates": len(failed),
                "failed_gate_names": [],
            },
        )

    if len(passed) >= max(1, len(lock_gates) - 2):
        return (
            "frozen_three_regime_model_supported_with_cautions",
            8,
            "Most freeze gates passed, but one or two lock gates failed. Treat as supported, not fully locked.",
            {
                "passed_lock_gates": len(passed),
                "failed_lock_gates": len(failed),
                "failed_gate_names": [g["gate"] for g in failed],
            },
        )

    return (
        "frozen_three_regime_model_not_locked",
        6,
        "The frozen replay failed too many lock gates. Do not promote this model.",
        {
            "passed_lock_gates": len(passed),
            "failed_lock_gates": len(failed),
            "failed_gate_names": [g["gate"] for g in failed],
        },
    )


def build_external_lane_readiness(final_status, gates):
    locked = final_status == "frozen_three_regime_model_locked_for_table2_lane"

    return {
        "ready_for_external_lane": bool(locked),
        "reason": (
            "Current Table2 lane is sufficiently mined. The next test should be independent rather than another Table2 model tweak."
            if locked
            else "Freeze did not fully lock. Review failed gates before leaving the Table2 lane."
        ),
        "frozen_claim_to_carry_forward": (
            "Within the SH0ES Table2 residual layer, TAIRID's predeclared F160W within-host edge-pair polarity supports a "
            "complexity-penalized three-regime model: clean high-alpha hosts, low-alpha/reference hosts LMC/SMC/N4536, "
            "and M31 sign-break quarantine."
        ),
        "claims_not_allowed": [
            "This does not prove TAIRID as physics.",
            "This does not prove an H0 correction.",
            "This does not prove SH0ES is wrong.",
            "This does not validate the model outside the SH0ES Table2 residual layer.",
        ],
        "next_independent_lanes": [
            {
                "lane": "SH0ES adjacent Cepheid layer",
                "goal": "Test whether the frozen F160W within-host polarity rule appears in a related but not identical Cepheid residual construction.",
                "rule": "No new regime tuning until the frozen rule is replayed as-is.",
            },
            {
                "lane": "independent passband or measurement layer",
                "goal": "Check whether F160W is uniquely boundary-active or whether another photometric layer has an analogous but separate boundary rule.",
                "rule": "F160W rule remains frozen; new layer must be declared before testing.",
            },
            {
                "lane": "non-Cepheid distance-ladder check",
                "goal": "Ask whether any TAIRID boundary-polarity idea transfers outside the Cepheid Table2 lane.",
                "rule": "Treat as external validation, not as automatic confirmation.",
            },
        ],
        "required_next_test_type": "independent validation lane, not further in-sample Table2 tuning",
        "gate_summary": {
            "passed": [g["gate"] for g in gates if g.get("passed") is True],
            "failed": [g["gate"] for g in gates if g.get("passed") is not True],
        },
    }


def build_model_card(final_status, readiness_score, next_wall, gates, v08_summary, v09_summary, external):
    model_summaries = v08_summary.get("model_summaries", [])
    quarantine = get_model(model_summaries, "three_regime_m31_quarantine")
    stress4 = get_model(model_summaries, "two_coefficient_stress4")
    universal = get_model(model_summaries, "universal_one_coefficient")

    lines = []
    lines.append("# TAIRID Boundary Polarity Battery v1.0 Model Card")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- Final status: `{final_status}`")
    lines.append(f"- Readiness score: `{readiness_score}/10`")
    lines.append(f"- Next wall: {next_wall}")
    lines.append("")
    lines.append("## Frozen model")
    lines.append("")
    lines.append("- Dataset lane: SH0ES Table2 residual layer only.")
    lines.append("- Variable: F160W-like Table2 numeric column, `table2_num_7`.")
    lines.append("- Edge rule: within-host high 5% F160W minus within-host low 5% F160W.")
    lines.append("- Regime 1: clean high-alpha hosts.")
    lines.append("- Regime 2: low-alpha/reference hosts = LMC, SMC, N4536.")
    lines.append("- Regime 3: M31 sign-break quarantine = zero transferred correction.")
    lines.append("- Training rule: held-out hosts are never refit.")
    lines.append("")
    lines.append("## Replay result")
    lines.append("")
    lines.append(f"- v0.8 replay status: `{v08_summary.get('final_status')}`")
    lines.append(f"- v0.9 replay status: `{v09_summary.get('final_status')}`")
    lines.append("")
    lines.append("## Fold-gain comparison")
    lines.append("")
    lines.append(f"- Universal one-coefficient total fold gain: `{universal.get('model_total_fold_gain')}`")
    lines.append(f"- Two-coefficient stress4 total fold gain: `{stress4.get('model_total_fold_gain')}`")
    lines.append(f"- Three-regime quarantine total fold gain: `{quarantine.get('model_total_fold_gain')}`")
    lines.append(f"- Three-regime improvement over universal: `{quarantine.get('model_minus_universal_total_gain')}`")
    lines.append("")
    lines.append("## Freeze gates")
    lines.append("")
    for g in gates:
        mark = "PASS" if g.get("passed") else "FAIL"
        lines.append(f"- `{mark}` — {g['gate']}: {g['detail']}")
    lines.append("")
    lines.append("## Locked claim")
    lines.append("")
    lines.append(
        "Within the SH0ES Table2 residual layer, the frozen TAIRID F160W within-host edge-pair polarity rule "
        "supports a complexity-penalized three-regime predictive model: clean high-alpha hosts, "
        "low-alpha/reference hosts LMC/SMC/N4536, and M31 sign-break quarantine."
    )
    lines.append("")
    lines.append("## Claims not allowed")
    lines.append("")
    for item in external["claims_not_allowed"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Next move")
    lines.append("")
    lines.append(external["required_next_test_type"])
    lines.append("")

    return "\n".join(lines)


def main():
    started = {
        "test_name": "TAIRID Boundary Prediction Battery v1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "runs_v0_9_first": True,
        "freeze_candidate": FROZEN_MODEL,
    }

    write_json(OUTDIR / "claims_v1_0.json", CLAIMS_V1_0)
    write_json(OUTDIR / "frozen_model_definition_v1_0.json", FROZEN_MODEL)
    write_json(OUTDIR / "v1_0_started.json", started)

    try:
        # Rerun v0.9 inside the v1.0 output folder so this artifact is self-contained.
        v09.OUTDIR = OUTDIR
        v09.DOWNLOAD_DIR = DOWNLOAD_DIR
        v09.v08.OUTDIR = OUTDIR
        v09.v08.DOWNLOAD_DIR = DOWNLOAD_DIR

        print("Running v0.9 inside v1.0 output folder.")
        print("Then applying frozen model replay gates and external-lane readiness audit.")

        v09.main()

        v09_summary_path = OUTDIR / "boundary_polarity_battery_v0_9_summary.json"
        v08_summary_path = OUTDIR / "boundary_polarity_battery_v0_8_summary.json"
        penalty_path = OUTDIR / "complexity_penalty_table_v0_9.csv"
        pairwise_path = OUTDIR / "pairwise_penalty_comparison_v0_9.csv"
        loho_audit_path = OUTDIR / "loho_anti_overfit_audit_v0_9.csv"
        local_host_path = OUTDIR / "local_host_diagnostics_v0_8.csv"

        required = [
            v09_summary_path,
            v08_summary_path,
            penalty_path,
            pairwise_path,
            loho_audit_path,
            local_host_path,
        ]

        missing = [str(path) for path in required if not path.exists()]

        if missing:
            raise FileNotFoundError(f"Missing expected replay outputs: {missing}")

        v09_summary = json.loads(v09_summary_path.read_text(encoding="utf-8"))
        v08_summary = json.loads(v08_summary_path.read_text(encoding="utf-8"))
        penalties = read_csv(penalty_path)
        pairwise = read_csv(pairwise_path)
        loho_audit = read_csv(loho_audit_path)
        local_rows = read_csv(local_host_path)

        gates = build_gate_table(
            v08_summary,
            v09_summary,
            penalties,
            pairwise,
            loho_audit,
            local_rows,
        )

        final_status, readiness_score, next_wall, lock_summary = decide_lock_status(gates)
        external = build_external_lane_readiness(final_status, gates)

        model_card_md = build_model_card(
            final_status,
            readiness_score,
            next_wall,
            gates,
            v08_summary,
            v09_summary,
            external,
        )

        write_csv(OUTDIR / "freeze_gate_table_v1_0.csv", gates)
        write_json(OUTDIR / "freeze_gate_summary_v1_0.json", lock_summary)
        write_json(OUTDIR / "external_lane_readiness_v1_0.json", external)
        (OUTDIR / "frozen_model_card_v1_0.md").write_text(model_card_md, encoding="utf-8")
        (OUTDIR / "frozen_model_card_v1_0.txt").write_text(model_card_md, encoding="utf-8")

        summary = {
            "test_name": "TAIRID Boundary Prediction Battery v1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Frozen three-regime replay and external-lane readiness audit only. "
                "This reruns v0.9 and then checks whether the current three-regime rule can be locked. "
                "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
            "frozen_model": FROZEN_MODEL,
            "claims_v1_0": CLAIMS_V1_0,
            "v0_8_replay_status": v08_summary.get("final_status"),
            "v0_8_replay_readiness": v08_summary.get("readiness_score_0_to_10"),
            "v0_9_replay_status": v09_summary.get("final_status"),
            "v0_9_replay_readiness": v09_summary.get("readiness_score_0_to_10"),
            "freeze_gates": gates,
            "freeze_gate_summary": lock_summary,
            "external_lane_readiness": external,
            "output_files": {
                "summary_json": str(OUTDIR / "boundary_polarity_battery_v1_0_summary.json"),
                "summary_txt": str(OUTDIR / "boundary_polarity_battery_v1_0_summary.txt"),
                "claims_json": str(OUTDIR / "claims_v1_0.json"),
                "frozen_model_json": str(OUTDIR / "frozen_model_definition_v1_0.json"),
                "model_card_md": str(OUTDIR / "frozen_model_card_v1_0.md"),
                "model_card_txt": str(OUTDIR / "frozen_model_card_v1_0.txt"),
                "gate_csv": str(OUTDIR / "freeze_gate_table_v1_0.csv"),
                "external_readiness_json": str(OUTDIR / "external_lane_readiness_v1_0.json"),
                "v0_9_summary_json": str(OUTDIR / "boundary_polarity_battery_v0_9_summary.json"),
                "v0_8_summary_json": str(OUTDIR / "boundary_polarity_battery_v0_8_summary.json"),
            },
            "interpretation": {
                "what_lock_means": (
                    "The frozen rule is the current best SH0ES Table2 residual-layer model and should not be further tuned on this lane before external validation."
                ),
                "what_lock_does_not_mean": (
                    "Locking does not prove TAIRID, does not prove H0 correction, and does not validate the rule outside this data lane."
                ),
                "truth_boundary": (
                    "This test only freezes the current Table2 lane result and tells us whether to move to an independent validation lane."
                ),
            },
        }

        write_json(OUTDIR / "boundary_polarity_battery_v1_0_summary.json", summary)

        with open(OUTDIR / "boundary_polarity_battery_v1_0_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Boundary Prediction Battery v1.0\n\n")
            f.write("Boundary: frozen three-regime replay and external-lane readiness audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Frozen model:\n")
            f.write(json.dumps(FROZEN_MODEL, indent=2, default=json_default) + "\n\n")
            f.write("Freeze gates:\n")
            f.write(json.dumps(gates, indent=2, default=json_default) + "\n\n")
            f.write("External lane readiness:\n")
            f.write(json.dumps(external, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only freezes the current SH0ES Table2 result and prepares the next independent validation lane.\n")

        finished = {
            **started,
            "status": "success",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
        }
        write_json(OUTDIR / "v1_0_finished.json", finished)

        print("TAIRID Boundary Prediction Battery v1.0 complete.")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        error_summary = {
            "test_name": "TAIRID Boundary Prediction Battery v1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "boundary_polarity_battery_v1_0_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(OUTDIR / "boundary_polarity_battery_v1_0_summary.json", error_summary)
        print(error_summary["traceback"])
        raise


if __name__ == "__main__":
    main()
