#!/usr/bin/env python3
"""
TAIRID Cosmology Preservation Guard Battery v3.1 Fresh

Why this test exists:
v3.0 reset the cosmology lane away from narrow anomaly hunting and built a
failure-mode atlas. The next required step was a preservation guard battery:
before TAIRID tries to explain Hubble tension, dark-energy pressure, growth
tension, or early-structure pressure, it must preserve the hard cosmology
surfaces that already constrain many alternatives.

v3.1 tests translation classes, not proof claims.

Core question:
    Which TAIRID cosmology translation branches are even admissible before
    pressure-seam testing?

This test DOES:
    - test whether candidate translation classes preserve supernova time dilation,
    - test whether they preserve distance duality,
    - test whether they preserve Tolman surface-brightness scaling,
    - test whether they keep BAO/CMB standard-ruler surfaces protected,
    - reject tired-light/static/photon-fatigue branches,
    - reject naive high-z gate extensions that would overwrite CMB/BAO surfaces,
    - identify the admissible TAIRID branch for the next pressure-map test.

This test DOES NOT:
    - validate TAIRID,
    - claim H0 correction,
    - claim new physics,
    - claim standard cosmology is wrong,
    - fit a new cosmology,
    - use anomaly chasing as proof.

Truth boundary:
v3.1 is a preservation guard. Passing means a translation branch is allowed to
move forward for pressure-map testing. It is not evidence that the branch is
true.
"""

import csv
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path("tairid_cosmology_preservation_guard_battery_v3_1_fresh_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 112221
Z_GRID = np.array([0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 5.00, 10.00, 100.00, 1100.00], dtype=float)

OBS_TIME_DILATION_B = 1.003
OBS_TIME_DILATION_STAT = 0.005
OBS_TIME_DILATION_SYS = 0.010
OBS_TIME_DILATION_SIGMA_QUAD = math.sqrt(OBS_TIME_DILATION_STAT**2 + OBS_TIME_DILATION_SYS**2)

FROZEN_GATE_V0_1 = {
    "name": "frozen_joint_gate_v0_1",
    "A": 0.6580586049,
    "zt": 0.2224541370,
    "p": 2.0,
    "truth_boundary": "Used here only as a stress case for preservation, not as proof or fit.",
}

CLAIMS_V3_1 = {
    "battery_name": "TAIRID Cosmology Preservation Guard Battery v3.1 Fresh",
    "scope": "Preservation guards for admissible TAIRID cosmology translation branches",
    "primary_question": (
        "Which cosmology translation branches preserve time dilation, distance duality/Tolman behavior, "
        "and BAO/CMB standard-ruler surfaces well enough to move into pressure-map testing?"
    ),
    "truth_boundary": (
        "This is a preservation guard only. It does not validate TAIRID, H0 correction, or new physics."
    ),
}

TAIRID_OPERATOR_GUARDS = {
    "boundary_formation": "A branch must declare which boundary forms the observable surface.",
    "retained_trace": "A branch must preserve known trace surfaces before claiming hidden depth.",
    "propagation_consolidation_split": "A branch may reinterpret accessibility only if it does not erase ruler/time surfaces.",
    "pacing_constraint_alignment": "A branch must preserve time-dilation pacing constraints.",
    "surface_depth_gap": "A branch must not mistake a compressed observable surface for the full hidden process.",
    "measurement_boundary_pressure": "A branch may discuss calibration pressure but cannot violate measurement constraints.",
}

TRANSLATION_CLASSES = [
    {
        "branch_id": "metric_accessibility_tairid_branch",
        "family": "admissible_tairid_candidate",
        "description": (
            "TAIRID interprets expansion observables as metric-like propagation/accessibility surfaces. "
            "It preserves time dilation, distance duality, Tolman dimming, BAO ruler behavior, and CMB acoustic surfaces by construction. "
            "Pressure-seam work must happen inside this metric-preserving container."
        ),
        "time_dilation_exponent_b": 1.0,
        "distance_duality_power": 2.0,
        "tolman_surface_brightness_power": -4.0,
        "preserves_metric_surface": True,
        "preserves_distance_duality": True,
        "preserves_tolman": True,
        "preserves_bao_ruler": True,
        "locks_cmb_acoustic_surface": True,
        "late_gate_allowed_only_inside_metric": True,
        "naive_high_z_gate_extension": False,
        "uses_photon_fatigue": False,
        "uses_static_redshift": False,
        "m31_or_shoes_residual_driven": False,
        "expected_result": "pass_guard",
    },
    {
        "branch_id": "static_tired_light_branch",
        "family": "rejected_alternative",
        "description": (
            "A static/no-time-dilation redshift branch. This is not allowed for TAIRID cosmology because it breaks the hard time-dilation surface."
        ),
        "time_dilation_exponent_b": 0.0,
        "distance_duality_power": 1.0,
        "tolman_surface_brightness_power": -1.0,
        "preserves_metric_surface": False,
        "preserves_distance_duality": False,
        "preserves_tolman": False,
        "preserves_bao_ruler": False,
        "locks_cmb_acoustic_surface": False,
        "late_gate_allowed_only_inside_metric": False,
        "naive_high_z_gate_extension": True,
        "uses_photon_fatigue": True,
        "uses_static_redshift": True,
        "m31_or_shoes_residual_driven": False,
        "expected_result": "fail_guard",
    },
    {
        "branch_id": "photon_fatigue_opacity_patch_branch",
        "family": "rejected_patch",
        "description": (
            "A branch that explains distance behavior through photon fatigue or opacity without preserving full metric time/ruler constraints."
        ),
        "time_dilation_exponent_b": 0.0,
        "distance_duality_power": 2.0,
        "tolman_surface_brightness_power": -3.0,
        "preserves_metric_surface": False,
        "preserves_distance_duality": False,
        "preserves_tolman": False,
        "preserves_bao_ruler": False,
        "locks_cmb_acoustic_surface": False,
        "late_gate_allowed_only_inside_metric": False,
        "naive_high_z_gate_extension": True,
        "uses_photon_fatigue": True,
        "uses_static_redshift": False,
        "m31_or_shoes_residual_driven": False,
        "expected_result": "fail_guard",
    },
    {
        "branch_id": "naive_all_redshift_gate_branch",
        "family": "rejected_naive_tairid",
        "description": (
            "A naive TAIRID gate branch that directly extends a low-z accessibility gate to all redshifts without locking CMB/BAO ruler surfaces."
        ),
        "time_dilation_exponent_b": 1.0,
        "distance_duality_power": 2.0,
        "tolman_surface_brightness_power": -4.0,
        "preserves_metric_surface": True,
        "preserves_distance_duality": True,
        "preserves_tolman": True,
        "preserves_bao_ruler": False,
        "locks_cmb_acoustic_surface": False,
        "late_gate_allowed_only_inside_metric": False,
        "naive_high_z_gate_extension": True,
        "uses_photon_fatigue": False,
        "uses_static_redshift": False,
        "m31_or_shoes_residual_driven": False,
        "expected_result": "fail_guard",
    },
    {
        "branch_id": "shoes_residual_driven_cosmology_branch",
        "family": "rejected_over_narrow_branch",
        "description": (
            "A branch that tries to promote the failed/narrow SH0ES host-residual lane into cosmology without the atlas preservation stack."
        ),
        "time_dilation_exponent_b": 1.0,
        "distance_duality_power": 2.0,
        "tolman_surface_brightness_power": -4.0,
        "preserves_metric_surface": True,
        "preserves_distance_duality": True,
        "preserves_tolman": True,
        "preserves_bao_ruler": False,
        "locks_cmb_acoustic_surface": False,
        "late_gate_allowed_only_inside_metric": False,
        "naive_high_z_gate_extension": False,
        "uses_photon_fatigue": False,
        "uses_static_redshift": False,
        "m31_or_shoes_residual_driven": True,
        "expected_result": "fail_guard",
    },
]


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
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_row(row):
    out = {}
    for key, value in row.items():
        if isinstance(value, list):
            out[key] = "|".join(str(v) for v in value)
        elif isinstance(value, dict):
            out[key] = json.dumps(value, default=json_default, sort_keys=True)
        else:
            out[key] = value
    return out


def frozen_gate(z, gate=FROZEN_GATE_V0_1):
    z = np.asarray(z, dtype=float)
    A = gate["A"]
    zt = gate["zt"]
    p = gate["p"]
    return np.exp(-A * (z / (z + zt)) ** p)


def sigma_distance(observed, expected, sigma):
    if sigma <= 0:
        return None
    return abs(float(observed) - float(expected)) / float(sigma)


def time_dilation_guard(branch):
    b = float(branch["time_dilation_exponent_b"])
    delta_sigma = sigma_distance(OBS_TIME_DILATION_B, b, OBS_TIME_DILATION_SIGMA_QUAD)
    predicted_factors = (1.0 + Z_GRID) ** b
    metric_factors = 1.0 + Z_GRID
    rel_error = np.abs(predicted_factors / metric_factors - 1.0)

    passed = (
        abs(b - 1.0) <= 0.05
        and delta_sigma is not None
        and delta_sigma <= 5.0
        and branch["preserves_metric_surface"] is True
        and branch["uses_static_redshift"] is False
        and branch["uses_photon_fatigue"] is False
    )

    return {
        "guard": "time_dilation",
        "passed": bool(passed),
        "time_dilation_exponent_b": b,
        "observed_guard_b": OBS_TIME_DILATION_B,
        "observed_guard_sigma_quad": OBS_TIME_DILATION_SIGMA_QUAD,
        "sigma_distance_from_guard": delta_sigma,
        "max_relative_error_vs_metric_grid": float(np.max(rel_error)),
        "failure_reason_if_failed": "" if passed else "Branch does not preserve metric-like b≈1 time dilation.",
    }


def distance_duality_guard(branch):
    power = float(branch["distance_duality_power"])
    ratio = (1.0 + Z_GRID) ** (power - 2.0)
    rel_error = np.abs(ratio - 1.0)

    passed = (
        branch["preserves_distance_duality"] is True
        and abs(power - 2.0) <= 1.0e-12
        and float(np.max(rel_error)) <= 1.0e-10
    )

    return {
        "guard": "distance_duality",
        "passed": bool(passed),
        "distance_duality_power": power,
        "max_relative_error_vs_duality": float(np.max(rel_error)),
        "failure_reason_if_failed": "" if passed else "Branch breaks D_L=(1+z)^2 D_A or treats opacity/fatigue as untracked depth.",
    }


def tolman_guard(branch):
    power = float(branch["tolman_surface_brightness_power"])
    ratio = (1.0 + Z_GRID) ** (power + 4.0)
    rel_error = np.abs(ratio - 1.0)

    passed = (
        branch["preserves_tolman"] is True
        and abs(power + 4.0) <= 1.0e-12
        and float(np.max(rel_error)) <= 1.0e-10
    )

    return {
        "guard": "tolman_surface_brightness",
        "passed": bool(passed),
        "tolman_power": power,
        "max_relative_error_vs_tolman": float(np.max(rel_error)),
        "failure_reason_if_failed": "" if passed else "Branch breaks Tolman (1+z)^-4 surface-brightness preservation.",
    }


def ruler_guard(branch):
    gate_values = frozen_gate(Z_GRID)
    high_z_gate_at_1100 = float(frozen_gate(np.array([1100.0]))[0])
    low_z_gate_at_1 = float(frozen_gate(np.array([1.0]))[0])

    naive_high_z_damage_flag = (
        branch["naive_high_z_gate_extension"] is True
        and high_z_gate_at_1100 < 0.99
    )

    passed = (
        branch["preserves_bao_ruler"] is True
        and branch["locks_cmb_acoustic_surface"] is True
        and branch["naive_high_z_gate_extension"] is False
        and branch["m31_or_shoes_residual_driven"] is False
    )

    return {
        "guard": "bao_cmb_ruler_preservation",
        "passed": bool(passed),
        "preserves_bao_ruler": bool(branch["preserves_bao_ruler"]),
        "locks_cmb_acoustic_surface": bool(branch["locks_cmb_acoustic_surface"]),
        "naive_high_z_gate_extension": bool(branch["naive_high_z_gate_extension"]),
        "m31_or_shoes_residual_driven": bool(branch["m31_or_shoes_residual_driven"]),
        "frozen_gate_value_z1": low_z_gate_at_1,
        "frozen_gate_value_z1100_if_naively_extended": high_z_gate_at_1100,
        "naive_high_z_damage_flag": bool(naive_high_z_damage_flag),
        "failure_reason_if_failed": "" if passed else "Branch does not protect BAO/CMB ruler surfaces before pressure-seam testing.",
    }


def anomaly_chasing_guard(branch):
    passed = (
        branch["m31_or_shoes_residual_driven"] is False
        and branch["uses_photon_fatigue"] is False
        and branch["uses_static_redshift"] is False
    )

    return {
        "guard": "no_anomaly_chasing_or_photon_fatigue",
        "passed": bool(passed),
        "uses_static_redshift": bool(branch["uses_static_redshift"]),
        "uses_photon_fatigue": bool(branch["uses_photon_fatigue"]),
        "m31_or_shoes_residual_driven": bool(branch["m31_or_shoes_residual_driven"]),
        "failure_reason_if_failed": "" if passed else "Branch relies on a rejected shortcut: tired light, photon fatigue, or narrow SH0ES residual promotion.",
    }


def evaluate_branch(branch):
    guards = [
        time_dilation_guard(branch),
        distance_duality_guard(branch),
        tolman_guard(branch),
        ruler_guard(branch),
        anomaly_chasing_guard(branch),
    ]
    failed = [g["guard"] for g in guards if not g["passed"]]
    passed_all = len(failed) == 0

    if passed_all:
        status = "admissible_for_v3_2_pressure_map"
        recommended_next_use = (
            "Allowed as the TAIRID metric-accessibility container for pressure-seam tests. "
            "Passing does not mean true; it means not immediately ruled out by preservation guards."
        )
    elif "time_dilation" in failed or "distance_duality" in failed or "tolman_surface_brightness" in failed:
        status = "rejected_by_hard_light_propagation_guard"
        recommended_next_use = "Do not use for TAIRID cosmology."
    elif "bao_cmb_ruler_preservation" in failed:
        status = "rejected_until_ruler_surfaces_are_locked"
        recommended_next_use = "Do not run pressure-seam testing until CMB/BAO surfaces are explicitly preserved."
    else:
        status = "rejected_by_method_guard"
        recommended_next_use = "Do not use as proof lane or cosmology branch."

    return {
        "branch_id": branch["branch_id"],
        "family": branch["family"],
        "description": branch["description"],
        "expected_result": branch["expected_result"],
        "status": status,
        "passed_all_guards": bool(passed_all),
        "failed_guards": failed,
        "guard_results": guards,
        "recommended_next_use": recommended_next_use,
        "truth_boundary": "Preservation admissibility only; not validation.",
    }


def build_grid_outputs(branch_results):
    rows = []
    for result in branch_results:
        branch_id = result["branch_id"]
        for z in Z_GRID:
            metric_time = 1.0 + z
            branch = next(b for b in TRANSLATION_CLASSES if b["branch_id"] == branch_id)
            b = float(branch["time_dilation_exponent_b"])
            duality_power = float(branch["distance_duality_power"])
            tolman_power = float(branch["tolman_surface_brightness_power"])

            rows.append(
                {
                    "branch_id": branch_id,
                    "z": float(z),
                    "time_dilation_factor": float((1.0 + z) ** b),
                    "metric_time_dilation_factor": float(metric_time),
                    "time_dilation_relative_error": float(((1.0 + z) ** b) / metric_time - 1.0),
                    "distance_duality_ratio": float((1.0 + z) ** (duality_power - 2.0)),
                    "tolman_ratio": float((1.0 + z) ** (tolman_power + 4.0)),
                    "frozen_gate_if_naively_extended": float(frozen_gate(np.array([z]))[0]),
                }
            )
    return rows


def decide(branch_results):
    admissible = [r for r in branch_results if r["passed_all_guards"]]
    rejected = [r for r in branch_results if not r["passed_all_guards"]]
    metric_admissible = [r for r in admissible if r["branch_id"] == "metric_accessibility_tairid_branch"]

    gates = [
        {
            "gate": "G1_at_least_one_admissible_branch",
            "passed": len(admissible) >= 1,
            "evidence": {
                "admissible_branch_count": len(admissible),
                "admissible_branches": [r["branch_id"] for r in admissible],
            },
        },
        {
            "gate": "G2_metric_accessibility_branch_survives",
            "passed": len(metric_admissible) == 1,
            "evidence": {
                "metric_accessibility_admissible": len(metric_admissible) == 1,
            },
        },
        {
            "gate": "G3_rejects_tired_light_and_photon_fatigue",
            "passed": all(
                r["passed_all_guards"] is False
                for r in branch_results
                if r["branch_id"] in {"static_tired_light_branch", "photon_fatigue_opacity_patch_branch"}
            ),
            "evidence": {
                "static_tired_light_status": next(r["status"] for r in branch_results if r["branch_id"] == "static_tired_light_branch"),
                "photon_fatigue_status": next(r["status"] for r in branch_results if r["branch_id"] == "photon_fatigue_opacity_patch_branch"),
            },
        },
        {
            "gate": "G4_rejects_naive_high_z_gate_extension",
            "passed": next(r for r in branch_results if r["branch_id"] == "naive_all_redshift_gate_branch")["passed_all_guards"] is False,
            "evidence": {
                "naive_gate_status": next(r["status"] for r in branch_results if r["branch_id"] == "naive_all_redshift_gate_branch"),
                "reason": "Low-z gate cannot be blindly extended across CMB/BAO surfaces.",
            },
        },
        {
            "gate": "G5_rejects_shoes_residual_as_cosmology_driver",
            "passed": next(r for r in branch_results if r["branch_id"] == "shoes_residual_driven_cosmology_branch")["passed_all_guards"] is False,
            "evidence": {
                "shoes_residual_branch_status": next(r["status"] for r in branch_results if r["branch_id"] == "shoes_residual_driven_cosmology_branch"),
            },
        },
        {
            "gate": "G6_no_validation_claim_allowed",
            "passed": True,
            "evidence": {
                "validation_claim": False,
                "h0_correction_claim": False,
                "new_physics_claim": False,
                "standard_cosmology_disproof_claim": False,
            },
        },
    ]

    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed:
        final_status = "preservation_guard_has_admissible_metric_accessibility_branch"
        readiness = 9
        next_wall = (
            "Proceed to v3.2 Multi-Surface Pressure Map using only the metric-accessibility TAIRID branch. "
            "Do not use tired light, photon fatigue, naive high-z gate extension, or SH0ES residual promotion."
        )
    elif len(admissible) == 0:
        final_status = "preservation_guard_no_admissible_tairid_branch"
        readiness = 4
        next_wall = "Stop cosmology pressure testing until a branch preserves hard surfaces."
    else:
        final_status = "preservation_guard_completed_with_cautions"
        readiness = 7
        next_wall = "Review failed gates before v3.2."

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "admissible_branches": [r["branch_id"] for r in admissible],
        "rejected_branches": [r["branch_id"] for r in rejected],
        "gates": gates,
        "failed_gates": failed,
        "truth_boundary": CLAIMS_V3_1["truth_boundary"],
    }


def build_surface_guard_ledger(branch_results, decision):
    return {
        "observable_surfaces_guarded": [
            {
                "surface": "SN_TIME_DILATION",
                "preservation_requirement": "Time dilation exponent must remain metric-like, b≈1.",
                "hard_failure": "No-time-dilation/static/tired-light branch.",
            },
            {
                "surface": "TOLMAN_DISTANCE_DUALITY",
                "preservation_requirement": "D_L=(1+z)^2 D_A and Tolman (1+z)^-4 behavior must be preserved unless a separate testable opacity model is declared.",
                "hard_failure": "Distance-only fit that breaks surface brightness or photon-counting constraints.",
            },
            {
                "surface": "BAO_STANDARD_RULER",
                "preservation_requirement": "BAO ruler behavior must be protected before late-time accessibility pressure is interpreted.",
                "hard_failure": "Gate or patch that destroys ruler coherence.",
            },
            {
                "surface": "CMB_ACOUSTIC_SURFACE",
                "preservation_requirement": "CMB acoustic surface must be locked before any low-z pressure-seam test is promoted.",
                "hard_failure": "Naive all-redshift extension of a low-z gate that overwrites early-universe surfaces.",
            },
        ],
        "admissible_branch": decision["admissible_branches"],
        "rejected_branches": decision["rejected_branches"],
        "method_reset": (
            "TAIRID cosmology may move forward only as a metric-preserving accessibility/pressure translation, "
            "not as tired light, photon fatigue, direct SH0ES residual promotion, or a blind all-redshift gate."
        ),
        "truth_boundary": "Surface preservation only; not validation.",
    }


def make_plots(branch_results, grid_rows):
    try:
        labels = [r["branch_id"] for r in branch_results]
        pass_counts = [sum(1 for g in r["guard_results"] if g["passed"]) for r in branch_results]
        fail_counts = [sum(1 for g in r["guard_results"] if not g["passed"]) for r in branch_results]
        x = np.arange(len(labels))

        plt.figure(figsize=(12, 5))
        plt.bar(x, pass_counts, label="passed guards")
        plt.bar(x, fail_counts, bottom=pass_counts, label="failed guards")
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel("guard count")
        plt.title("TAIRID v3.1 preservation guard outcomes by branch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / "preservation_guard_outcomes_v3_1_fresh.png", dpi=160)
        plt.close()

        z_plot = np.array([0.01, 0.05, 0.10, 0.50, 1.00, 2.00, 10.00, 1100.00], dtype=float)
        gate_values = frozen_gate(z_plot)

        plt.figure(figsize=(8, 5))
        plt.plot(z_plot, gate_values, marker="o")
        plt.xscale("log")
        plt.xlabel("redshift z")
        plt.ylabel("frozen gate value if naively extended")
        plt.title("Why low-z gates cannot be blindly extended to CMB surfaces")
        plt.tight_layout()
        plt.savefig(OUTDIR / "naive_gate_extension_stress_v3_1_fresh.png", dpi=160)
        plt.close()
    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v3_1_fresh.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )


def write_markdown_report(branch_results, decision, ledger):
    lines = []
    lines.append("# TAIRID Cosmology Preservation Guard Battery v3.1 Fresh")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("This is a preservation guard, not validation. Passing means a branch is allowed to move forward for pressure-map testing. It does not prove TAIRID, H0 correction, new physics, or standard-cosmology failure.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## Branch outcomes")
    lines.append("")
    for result in branch_results:
        lines.append(f"### {result['branch_id']}")
        lines.append(f"- Family: `{result['family']}`")
        lines.append(f"- Status: `{result['status']}`")
        lines.append(f"- Passed all guards: `{result['passed_all_guards']}`")
        lines.append(f"- Failed guards: `{', '.join(result['failed_guards']) if result['failed_guards'] else 'none'}`")
        lines.append(f"- Use: {result['recommended_next_use']}")
        lines.append("")
    lines.append("## Method reset")
    lines.append("")
    lines.append(ledger["method_reset"])
    lines.append("")
    return "\n".join(lines)


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v3.1 Fresh Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v3.1 proved methodologically")
    lines.append("")
    lines.append("- TAIRID cosmology must remain metric-surface preserving.")
    lines.append("- Tired light, static redshift, and photon fatigue branches are rejected.")
    lines.append("- A low-z gate cannot be blindly extended into the CMB/BAO domain.")
    lines.append("- The SH0ES residual lane cannot be promoted into cosmology by itself.")
    lines.append("- The only branch allowed forward is the metric-accessibility translation branch.")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v3.1 is a preservation guard only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not disprove standard cosmology.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append("Build v3.2 — Multi-Surface Pressure Map. It should use only the metric-accessibility branch and compare Hubble tension, BAO, SN, CMB, and dark-energy pressure surfaces without fitting a new free patch.")
    else:
        lines.append("Repair the preservation guard before pressure-map testing.")
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID Cosmology Preservation Guard Battery v3.1 Fresh starting.")
    print("Boundary: preservation guard only; no validation, no H0 claim, no new physics claim.")

    write_json(OUTDIR / "claims_v3_1_fresh.json", CLAIMS_V3_1)
    write_json(OUTDIR / "tairid_operator_guards_v3_1_fresh.json", TAIRID_OPERATOR_GUARDS)
    write_json(OUTDIR / "frozen_gate_v0_1_guard_stress_case_v3_1_fresh.json", FROZEN_GATE_V0_1)

    try:
        branch_results = [evaluate_branch(branch) for branch in TRANSLATION_CLASSES]
        decision = decide(branch_results)
        ledger = build_surface_guard_ledger(branch_results, decision)
        grid_rows = build_grid_outputs(branch_results)

        flat_branch_rows = []
        guard_rows = []
        for result in branch_results:
            flat = dict(result)
            flat.pop("guard_results", None)
            flat_branch_rows.append(flatten_row(flat))
            for guard in result["guard_results"]:
                row = dict(guard)
                row["branch_id"] = result["branch_id"]
                row["branch_status"] = result["status"]
                guard_rows.append(flatten_row(row))

        write_csv(OUTDIR / "translation_branch_guard_summary_v3_1_fresh.csv", flat_branch_rows)
        write_json(OUTDIR / "translation_branch_guard_summary_v3_1_fresh.json", branch_results)

        write_csv(OUTDIR / "individual_guard_results_v3_1_fresh.csv", guard_rows)
        write_csv(OUTDIR / "redshift_grid_guard_values_v3_1_fresh.csv", grid_rows)

        write_json(OUTDIR / "surface_guard_ledger_v3_1_fresh.json", ledger)
        write_json(OUTDIR / "decision_v3_1_fresh.json", decision)

        make_plots(branch_results, grid_rows)

        report = write_markdown_report(branch_results, decision, ledger)
        (OUTDIR / "preservation_guard_report_v3_1_fresh.md").write_text(report, encoding="utf-8")
        (OUTDIR / "preservation_guard_report_v3_1_fresh.txt").write_text(report, encoding="utf-8")

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v3_1_fresh.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v3_1_fresh.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID Cosmology Preservation Guard Battery v3.1 Fresh",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Preservation guard only. No validation, no H0 claim, no new-physics claim.",
            "branch_count": len(branch_results),
            "admissible_branches": decision["admissible_branches"],
            "rejected_branches": decision["rejected_branches"],
            "decision": decision,
            "surface_guard_ledger": ledger,
            "claims_v3_1": CLAIMS_V3_1,
            "output_files": {
                "summary_json": str(OUTDIR / "cosmology_preservation_guard_battery_v3_1_fresh_summary.json"),
                "summary_txt": str(OUTDIR / "cosmology_preservation_guard_battery_v3_1_fresh_summary.txt"),
                "branch_summary_csv": str(OUTDIR / "translation_branch_guard_summary_v3_1_fresh.csv"),
                "individual_guard_results_csv": str(OUTDIR / "individual_guard_results_v3_1_fresh.csv"),
                "redshift_grid_values_csv": str(OUTDIR / "redshift_grid_guard_values_v3_1_fresh.csv"),
                "surface_guard_ledger_json": str(OUTDIR / "surface_guard_ledger_v3_1_fresh.json"),
                "decision_json": str(OUTDIR / "decision_v3_1_fresh.json"),
                "report_md": str(OUTDIR / "preservation_guard_report_v3_1_fresh.md"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v3_1_fresh.md"),
                "plots": [
                    str(OUTDIR / "preservation_guard_outcomes_v3_1_fresh.png"),
                    str(OUTDIR / "naive_gate_extension_stress_v3_1_fresh.png"),
                ],
            },
            "interpretation": {
                "what_success_means": "A TAIRID metric-accessibility branch is admissible for pressure-map testing because it preserves hard cosmology surfaces by construction.",
                "what_success_does_not_mean": "This does not prove the branch is true or that TAIRID explains cosmology.",
                "what_was_rejected": "Static/tired-light, photon-fatigue, naive all-redshift gate, and SH0ES-residual-driven branches.",
                "next_required_step": "v3.2 should run a multi-surface pressure map using only the admissible metric-accessibility branch.",
                "truth_boundary": CLAIMS_V3_1["truth_boundary"],
            },
        }

        write_json(OUTDIR / "cosmology_preservation_guard_battery_v3_1_fresh_summary.json", summary)

        with open(OUTDIR / "cosmology_preservation_guard_battery_v3_1_fresh_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Cosmology Preservation Guard Battery v3.1 Fresh\n\n")
            f.write("Boundary: preservation guard only. No validation. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Admissible branches:\n")
            f.write(json.dumps(decision["admissible_branches"], indent=2, default=json_default) + "\n\n")
            f.write("Rejected branches:\n")
            f.write(json.dumps(decision["rejected_branches"], indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not disprove standard cosmology.\n")
            f.write("- This only identifies which translation branch is allowed forward.\n")

        print("TAIRID Cosmology Preservation Guard Battery v3.1 Fresh complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Cosmology Preservation Guard Battery v3.1 Fresh",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "cosmology_preservation_guard_battery_v3_1_fresh_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V3_1["truth_boundary"],
        }
        write_json(OUTDIR / "cosmology_preservation_guard_battery_v3_1_fresh_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
