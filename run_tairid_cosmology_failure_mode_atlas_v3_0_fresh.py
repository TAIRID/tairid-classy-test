#!/usr/bin/env python3
"""
TAIRID Cosmology Failure-Mode Atlas v3.0 Fresh

Why this test exists:
The SH0ES downstream lane became too narrow. It tested one surface bridge as if
that could decide a theory-of-everything claim. v3.0 resets the method.

v3.0 does not try to prove TAIRID. It builds a structured cosmology atlas that
compares where standard cosmology succeeds, where it has pressure seams, and
where TAIRID would have to preserve, translate, or fail in the same place.

Core question:
    Can we map cosmology's success surfaces and pressure seams into a TAIRID
    surface/depth/boundary/failure ledger before running another narrow test?

This test DOES:
    - create a cosmology failure-mode atlas,
    - compare standard-model success surfaces with pressure seams,
    - define TAIRID operator translations for each seam,
    - define preservation requirements TAIRID must not break,
    - define actual failure conditions,
    - rank which next data batteries are worth running,
    - write a handoff for v3.1.

This test DOES NOT:
    - validate TAIRID,
    - claim standard cosmology is wrong,
    - claim H0 correction,
    - claim new physics,
    - use anomaly chasing as proof,
    - treat one failed bridge as disproof of the whole theory.

Truth boundary:
v3.0 is an atlas/ledger builder. It decides what should be tested next and what
would count as failure. It is not the test of the physics itself.
"""

import csv
import json
import math
import re
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path("tairid_cosmology_failure_mode_atlas_v3_0_fresh_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

CLAIMS_V3_0 = {
    "battery_name": "TAIRID Cosmology Failure-Mode Atlas v3.0 Fresh",
    "scope": "Cosmology success/pressure seam ledger with TAIRID operator translation and next-test ranking",
    "primary_question": (
        "Where does current cosmology already succeed, where does it strain, and how should TAIRID be compared "
        "without collapsing the theory into one narrow anomaly test?"
    ),
    "truth_boundary": (
        "This is an atlas/ledger builder only. It does not validate TAIRID, H0 correction, or new physics."
    ),
}

TAIRID_OPERATOR_LIBRARY = {
    "boundary_formation": "A measurable surface forms only when contrast stabilizes across a boundary.",
    "retained_trace": "A surface preserves compressed prior collapse history without exposing the full hidden process.",
    "propagation_consolidation_split": "Some structure spreads while some structure stabilizes; mismatch creates pressure.",
    "pacing_constraint_alignment": "Observable stability requires local pacing and constraint to remain inside a viable relation.",
    "viability_window": "Stable structures persist inside a limited update/constraint range; outside it, regimes shift or break.",
    "phase_shift": "When pressure exceeds update capacity, the system reorganizes into a new surface regime.",
    "surface_depth_gap": "The observable is a compressed surface, not the whole depth that formed it.",
    "measurement_boundary_pressure": "Calibration, selection, and observation architecture shape what trace survives.",
}

ATLAS_ROWS = [
    {
        "atlas_id": "CMB_ACOUSTIC_SURFACE",
        "domain": "early_universe",
        "observable_surface": "CMB temperature/polarization acoustic peak structure",
        "standard_model_success": "Base cosmology fits the acoustic peak surface with high precision and gives a coherent early-universe parameter surface.",
        "standard_model_pressure_seam": "Success is very strong at the surface, but ontology remains compressed into components such as inflation initial conditions, dark matter, and dark energy.",
        "tairid_translation": "CMB peaks are a high-preservation boundary surface: TAIRID must treat them as retained trace from an early propagation/consolidation regime, not as disposable noise.",
        "primary_tairid_operators": ["boundary_formation", "retained_trace", "propagation_consolidation_split", "viability_window"],
        "preservation_requirement": "Preserve acoustic scale ordering and peak coherence before any deeper reinterpretation.",
        "failure_condition": "TAIRID translation fails if it cannot preserve acoustic peak spacing/relative structure at the same surface level current cosmology preserves.",
        "candidate_datasets": ["Planck 2018 likelihood/summary products", "ACT/SPT comparison products"],
        "next_test_kind": "preservation_first",
        "data_access_complexity_1_to_5": 4,
        "pressure_strength_1_to_5": 2,
        "preservation_importance_1_to_5": 5,
        "tairid_operator_specificity_1_to_5": 4,
        "risk_if_failed_1_to_5": 5,
    },
    {
        "atlas_id": "BAO_STANDARD_RULER",
        "domain": "large_scale_structure",
        "observable_surface": "BAO distance/ruler surface across redshift",
        "standard_model_success": "BAO behaves as a strong standard-ruler surface tied to early-universe physics and late-time geometry.",
        "standard_model_pressure_seam": "BAO now participates in late-time dark-energy and expansion-history pressure when combined with SN and CMB surfaces.",
        "tairid_translation": "BAO is a retained propagation-memory ruler: TAIRID must preserve the ruler surface while asking whether late-time accessibility changes alter the inferred path cost.",
        "primary_tairid_operators": ["retained_trace", "propagation_consolidation_split", "surface_depth_gap"],
        "preservation_requirement": "Preserve ruler-like distance behavior before interpreting late-time deviations.",
        "failure_condition": "TAIRID fails if its gate/accessibility terms destroy the BAO ruler surface or require ad hoc redshift-dependent patches.",
        "candidate_datasets": ["DESI BAO releases", "SDSS/eBOSS BAO products"],
        "next_test_kind": "cross_surface_consistency",
        "data_access_complexity_1_to_5": 3,
        "pressure_strength_1_to_5": 4,
        "preservation_importance_1_to_5": 5,
        "tairid_operator_specificity_1_to_5": 5,
        "risk_if_failed_1_to_5": 5,
    },
    {
        "atlas_id": "SN_TIME_DILATION",
        "domain": "supernova_observation",
        "observable_surface": "Supernova light-curve time dilation scaling with redshift",
        "standard_model_success": "Metric expansion naturally preserves observed time dilation.",
        "standard_model_pressure_seam": "This surface rules out many simple tired-light alternatives and forces any alternative translation to remain metric-like.",
        "tairid_translation": "TAIRID cannot be static tired light. Propagation accessibility must preserve observed time dilation as a surface constraint.",
        "primary_tairid_operators": ["pacing_constraint_alignment", "surface_depth_gap", "measurement_boundary_pressure"],
        "preservation_requirement": "Preserve observed redshift time dilation exactly enough to remain compatible with SN light-curve data.",
        "failure_condition": "TAIRID fails if it predicts no time dilation or requires non-metric photon fatigue.",
        "candidate_datasets": ["Pantheon+ light-curve products", "DES-SN time-dilation results"],
        "next_test_kind": "hard_falsification_guard",
        "data_access_complexity_1_to_5": 3,
        "pressure_strength_1_to_5": 5,
        "preservation_importance_1_to_5": 5,
        "tairid_operator_specificity_1_to_5": 5,
        "risk_if_failed_1_to_5": 5,
    },
    {
        "atlas_id": "TOLMAN_DISTANCE_DUALITY",
        "domain": "light_propagation",
        "observable_surface": "Tolman surface-brightness dimming and distance-duality relation",
        "standard_model_success": "Metric expansion preserves surface-brightness and luminosity/angular-diameter distance relations under standard assumptions.",
        "standard_model_pressure_seam": "Many alternative redshift explanations fail here, even when they fit distance curves.",
        "tairid_translation": "TAIRID must treat this as a preservation boundary: propagation-accessibility reinterpretation cannot violate photon-counting and surface-brightness constraints.",
        "primary_tairid_operators": ["pacing_constraint_alignment", "measurement_boundary_pressure", "retained_trace"],
        "preservation_requirement": "Preserve distance duality unless a specific, independently testable opacity or boundary effect is declared.",
        "failure_condition": "TAIRID fails if it matches SN distances only by breaking Tolman/distance-duality behavior.",
        "candidate_datasets": ["SN distance datasets", "galaxy surface-brightness tests", "cluster angular diameter datasets"],
        "next_test_kind": "hard_falsification_guard",
        "data_access_complexity_1_to_5": 4,
        "pressure_strength_1_to_5": 4,
        "preservation_importance_1_to_5": 5,
        "tairid_operator_specificity_1_to_5": 5,
        "risk_if_failed_1_to_5": 5,
    },
    {
        "atlas_id": "HUBBLE_TENSION",
        "domain": "expansion_history",
        "observable_surface": "Early-inferred versus local distance-ladder expansion rate mismatch",
        "standard_model_success": "The standard model fits many surfaces well, but early and late H0 routes remain difficult to reconcile cleanly.",
        "standard_model_pressure_seam": "Local distance ladder, CMB-inferred H0, and intermediate probes create a persistent boundary-pressure seam.",
        "tairid_translation": "TAIRID should ask whether H0 tension is a surface mismatch between propagation history, local calibration boundary, and retained early-ruler trace.",
        "primary_tairid_operators": ["measurement_boundary_pressure", "propagation_consolidation_split", "surface_depth_gap", "phase_shift"],
        "preservation_requirement": "Do not break CMB, BAO, SN time dilation, or distance duality while addressing H0 tension.",
        "failure_condition": "TAIRID fails if it only fits one side of H0 tension by destroying the cross-surface constraint stack.",
        "candidate_datasets": ["SH0ES/Pantheon+", "Planck", "DESI BAO", "TRGB/Cepheid alternative ladders"],
        "next_test_kind": "multi_surface_pressure_mapping",
        "data_access_complexity_1_to_5": 4,
        "pressure_strength_1_to_5": 5,
        "preservation_importance_1_to_5": 5,
        "tairid_operator_specificity_1_to_5": 5,
        "risk_if_failed_1_to_5": 4,
    },
    {
        "atlas_id": "DARK_ENERGY_EVOLUTION_HINTS",
        "domain": "late_time_acceleration",
        "observable_surface": "Late-time dark-energy equation-of-state / expansion-history surface",
        "standard_model_success": "A cosmological-constant surface remains highly effective as a compressed late-time description.",
        "standard_model_pressure_seam": "Some combined late-time datasets have hinted at possible evolution away from a constant dark-energy term.",
        "tairid_translation": "TAIRID should compare a constant surface term against a propagation-accessibility or viability-gate term that can vary without breaking early surfaces.",
        "primary_tairid_operators": ["viability_window", "phase_shift", "propagation_consolidation_split", "surface_depth_gap"],
        "preservation_requirement": "Any variable accessibility term must preserve BAO/SN/CMB cross-consistency better than a free patch.",
        "failure_condition": "TAIRID fails if its gate becomes just another unconstrained dark-energy parameterization.",
        "candidate_datasets": ["DESI BAO", "Pantheon+ or Union3 SN", "CMB priors"],
        "next_test_kind": "model_comparison_with_penalty",
        "data_access_complexity_1_to_5": 3,
        "pressure_strength_1_to_5": 4,
        "preservation_importance_1_to_5": 5,
        "tairid_operator_specificity_1_to_5": 5,
        "risk_if_failed_1_to_5": 4,
    },
    {
        "atlas_id": "S8_STRUCTURE_GROWTH",
        "domain": "structure_growth",
        "observable_surface": "Matter clustering amplitude / weak-lensing growth surface",
        "standard_model_success": "Large-scale structure is broadly modeled, but growth-amplitude comparisons have produced recurring tension across surveys and CMB-inferred parameters.",
        "standard_model_pressure_seam": "Geometry/expansion surfaces and growth/lensing surfaces do not always align cleanly.",
        "tairid_translation": "TAIRID should treat growth as consolidation-rate behavior, not merely as matter amount: structure forms when propagation memory can consolidate under viable local constraints.",
        "primary_tairid_operators": ["propagation_consolidation_split", "viability_window", "phase_shift", "retained_trace"],
        "preservation_requirement": "Preserve broad growth history while explaining mismatch between geometry and consolidation surfaces.",
        "failure_condition": "TAIRID fails if it cannot distinguish geometry distance effects from structure-consolidation effects.",
        "candidate_datasets": ["DES/KiDS/HSC weak-lensing products", "Planck CMB constraints", "cluster counts"],
        "next_test_kind": "growth_geometry_split_test",
        "data_access_complexity_1_to_5": 5,
        "pressure_strength_1_to_5": 4,
        "preservation_importance_1_to_5": 4,
        "tairid_operator_specificity_1_to_5": 4,
        "risk_if_failed_1_to_5": 4,
    },
    {
        "atlas_id": "JWST_EARLY_STRUCTURE",
        "domain": "early_galaxy_formation",
        "observable_surface": "High-redshift early galaxy abundance/mass/formation surface",
        "standard_model_success": "Galaxy formation physics has flexibility, and not every early massive candidate is fatal to the standard model.",
        "standard_model_pressure_seam": "Some early high-redshift candidates create pressure on formation timing, dust assumptions, mass estimation, and feedback modeling.",
        "tairid_translation": "TAIRID should ask whether early viable consolidation pathways opened faster or differently than expected, not simply claim impossible galaxies.",
        "primary_tairid_operators": ["viability_window", "phase_shift", "retained_trace", "propagation_consolidation_split"],
        "preservation_requirement": "Respect observational uncertainty, selection effects, photometric redshift risk, and galaxy-formation flexibility.",
        "failure_condition": "TAIRID fails if it treats tentative early candidates as proof while ignoring later spectroscopy or mass revisions.",
        "candidate_datasets": ["JWST high-redshift catalogs", "spectroscopic confirmation samples", "simulation comparison products"],
        "next_test_kind": "formation_timing_pressure_audit",
        "data_access_complexity_1_to_5": 5,
        "pressure_strength_1_to_5": 3,
        "preservation_importance_1_to_5": 4,
        "tairid_operator_specificity_1_to_5": 4,
        "risk_if_failed_1_to_5": 3,
    },
    {
        "atlas_id": "BLACK_HOLE_HORIZON_ENTROPY",
        "domain": "horizon_physics",
        "observable_surface": "Black-hole horizon, entropy, information-boundary behavior",
        "standard_model_success": "Horizon thermodynamics and GR black-hole geometry are strong surface descriptions.",
        "standard_model_pressure_seam": "Information, entropy, quantum gravity, and interior/exterior boundary relations remain conceptually unresolved.",
        "tairid_translation": "TAIRID should treat horizons as maximal boundary-pressure surfaces where propagation closure, retained trace, and inaccessible interior depth become physically central.",
        "primary_tairid_operators": ["boundary_formation", "retained_trace", "surface_depth_gap", "pacing_constraint_alignment"],
        "preservation_requirement": "Preserve known horizon thermodynamic relationships before adding TAIRID boundary language.",
        "failure_condition": "TAIRID fails if its boundary account contradicts well-tested GR horizon behavior without a replacement calculation.",
        "candidate_datasets": ["EHT horizon-scale images", "LIGO/Virgo/KAGRA ringdown constraints", "black-hole thermodynamics literature"],
        "next_test_kind": "conceptual_constraint_ledger",
        "data_access_complexity_1_to_5": 5,
        "pressure_strength_1_to_5": 5,
        "preservation_importance_1_to_5": 5,
        "tairid_operator_specificity_1_to_5": 5,
        "risk_if_failed_1_to_5": 5,
    },
    {
        "atlas_id": "COSMIC_WEB_VOIDS",
        "domain": "large_scale_topology",
        "observable_surface": "Cosmic web, voids, filaments, and environment-dependent structure",
        "standard_model_success": "ΛCDM simulations produce a cosmic web with broad success.",
        "standard_model_pressure_seam": "Detailed environment-dependent galaxy formation, void dynamics, and local-flow effects can stress simple compressed interpretations.",
        "tairid_translation": "TAIRID should treat web/void structure as a propagation-consolidation topology: filaments are retained paths, voids are low-consolidation surfaces, and nodes are boundary-saturation sites.",
        "primary_tairid_operators": ["propagation_consolidation_split", "retained_trace", "boundary_formation", "viability_window"],
        "preservation_requirement": "Preserve the broad cosmic-web success of simulations before claiming deeper topology.",
        "failure_condition": "TAIRID fails if it only redescribes the web poetically without producing measurable topology contrasts.",
        "candidate_datasets": ["SDSS/BOSS/eBOSS galaxy catalogs", "void catalogs", "simulation comparison products"],
        "next_test_kind": "topology_metric_translation",
        "data_access_complexity_1_to_5": 5,
        "pressure_strength_1_to_5": 3,
        "preservation_importance_1_to_5": 4,
        "tairid_operator_specificity_1_to_5": 4,
        "risk_if_failed_1_to_5": 3,
    },
]

SOURCE_CANDIDATES = [
    {
        "source_id": "PLANCK_2018_PARAMETERS",
        "url": "https://arxiv.org/abs/1807.06209",
        "surface_tags": ["CMB_ACOUSTIC_SURFACE", "HUBBLE_TENSION", "S8_STRUCTURE_GROWTH"],
        "source_role": "primary_or_near_primary_summary",
    },
    {
        "source_id": "PANTHEON_PLUS_SHOES_DATARELEASE",
        "url": "https://github.com/PantheonPlusSH0ES/DataRelease",
        "surface_tags": ["HUBBLE_TENSION", "SN_TIME_DILATION", "TOLMAN_DISTANCE_DUALITY"],
        "source_role": "public_data_release",
    },
    {
        "source_id": "DESI_DATA_PAGE",
        "url": "https://data.desi.lbl.gov/doc/releases/",
        "surface_tags": ["BAO_STANDARD_RULER", "DARK_ENERGY_EVOLUTION_HINTS"],
        "source_role": "public_data_release_index",
    },
    {
        "source_id": "MAST_JWST_ARCHIVE",
        "url": "https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html",
        "surface_tags": ["JWST_EARLY_STRUCTURE"],
        "source_role": "archive_entry_point",
    },
    {
        "source_id": "EHT_PUBLIC_RESULTS",
        "url": "https://eventhorizontelescope.org/",
        "surface_tags": ["BLACK_HOLE_HORIZON_ENTROPY"],
        "source_role": "project_entry_point",
    },
    {
        "source_id": "SDSS_DATA_RELEASES",
        "url": "https://www.sdss.org/dr18/",
        "surface_tags": ["COSMIC_WEB_VOIDS", "BAO_STANDARD_RULER"],
        "source_role": "public_data_release",
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
    flat = {}
    for key, value in row.items():
        if isinstance(value, list):
            flat[key] = "|".join(str(v) for v in value)
        elif isinstance(value, dict):
            flat[key] = json.dumps(value, default=json_default, sort_keys=True)
        else:
            flat[key] = value
    return flat


def score_atlas_row(row):
    pressure = float(row["pressure_strength_1_to_5"])
    preservation = float(row["preservation_importance_1_to_5"])
    specificity = float(row["tairid_operator_specificity_1_to_5"])
    risk = float(row["risk_if_failed_1_to_5"])
    complexity = float(row["data_access_complexity_1_to_5"])

    priority = (1.4 * pressure) + (1.2 * specificity) + (1.1 * preservation) + (0.6 * risk) - (0.7 * complexity)

    if row["next_test_kind"] == "hard_falsification_guard":
        priority += 1.0
    if row["next_test_kind"] == "multi_surface_pressure_mapping":
        priority += 1.2
    if row["next_test_kind"] == "model_comparison_with_penalty":
        priority += 0.8

    return round(priority, 3)


def classify_test_action(row):
    kind = row["next_test_kind"]
    if kind == "hard_falsification_guard":
        return "Run early as a guardrail before claiming any alternative cosmology."
    if kind == "multi_surface_pressure_mapping":
        return "Run as a combined pressure-map battery across multiple surfaces."
    if kind == "cross_surface_consistency":
        return "Run after preservation guards; compare BAO/SN/CMB compatibility."
    if kind == "model_comparison_with_penalty":
        return "Run only with penalty for added freedom; do not reward patching."
    if kind == "growth_geometry_split_test":
        return "Run after geometry surfaces are stable; growth data is more complex."
    if kind == "formation_timing_pressure_audit":
        return "Run as an uncertainty-aware pressure audit, not a proof claim."
    if kind == "conceptual_constraint_ledger":
        return "Build derivation constraints before numerical testing."
    if kind == "topology_metric_translation":
        return "Define measurable topology metrics before testing."
    if kind == "preservation_first":
        return "Treat as a must-preserve surface before deeper TAIRID claims."
    return "Needs manual review."


def add_derived_fields(rows):
    out = []
    for row in rows:
        r = dict(row)
        r["operator_count"] = len(r["primary_tairid_operators"])
        r["test_priority_score"] = score_atlas_row(r)
        r["recommended_action"] = classify_test_action(r)
        r["surface_depth_question"] = (
            f"What hidden boundary/trace depth is compressed into the observable surface '{r['observable_surface']}'?"
        )
        r["matched_failure_question"] = (
            "Does TAIRID fail in the same place as the current model, fail worse, or expose a more specific boundary layer?"
        )
        out.append(r)
    return sorted(out, key=lambda r: (-r["test_priority_score"], r["atlas_id"]))


def source_reachability_check(sources):
    rows = []
    for source in sources:
        status = "not_checked"
        http_code = None
        final_url = None
        error = None
        content_type = None
        byte_sample = None

        try:
            req = urllib.request.Request(
                source["url"],
                headers={"User-Agent": "TAIRID-v3.0-cosmology-atlas-source-check"},
            )
            with urllib.request.urlopen(req, timeout=25) as response:
                data = response.read(512)
                status = "reachable"
                http_code = response.status
                final_url = response.geturl()
                content_type = response.headers.get("Content-Type", "")
                byte_sample = len(data)
        except Exception as exc:
            status = "unreachable_or_blocked"
            error = repr(exc)

        row = dict(source)
        row["surface_tags"] = "|".join(source["surface_tags"])
        row["reachability_status"] = status
        row["http_code"] = http_code
        row["final_url"] = final_url
        row["content_type"] = content_type
        row["byte_sample"] = byte_sample
        row["error"] = error
        row["truth_boundary"] = "reachability_only_not_source_validation"
        rows.append(row)

    return rows


def build_operator_coverage(rows):
    counts = {}
    surfaces_by_operator = {}
    for row in rows:
        for op in row["primary_tairid_operators"]:
            counts[op] = counts.get(op, 0) + 1
            surfaces_by_operator.setdefault(op, []).append(row["atlas_id"])

    coverage_rows = []
    for op, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        coverage_rows.append(
            {
                "operator": op,
                "operator_definition": TAIRID_OPERATOR_LIBRARY.get(op, ""),
                "surface_count": count,
                "surfaces": "|".join(surfaces_by_operator[op]),
            }
        )
    return coverage_rows


def build_next_test_plan(rows):
    hard_guards = [r for r in rows if r["next_test_kind"] == "hard_falsification_guard"]
    high_priority = rows[:5]

    plan = []
    plan.append(
        {
            "step": 1,
            "test_name": "v3.1 Preservation Guard Battery",
            "purpose": "Confirm TAIRID cosmology translation preserves time dilation, distance duality/Tolman behavior, and acoustic/BAO surfaces before trying to explain pressure seams.",
            "surfaces": "|".join(sorted(set(r["atlas_id"] for r in hard_guards + high_priority if r["atlas_id"] in {
                "SN_TIME_DILATION",
                "TOLMAN_DISTANCE_DUALITY",
                "CMB_ACOUSTIC_SURFACE",
                "BAO_STANDARD_RULER",
            }))),
            "truth_boundary": "Guardrail only; passing means compatibility with must-preserve surfaces, not proof.",
        }
    )
    plan.append(
        {
            "step": 2,
            "test_name": "v3.2 Multi-Surface Pressure Map",
            "purpose": "Compare Hubble tension, BAO, SN, and CMB as pressure surfaces rather than one anomaly lane.",
            "surfaces": "HUBBLE_TENSION|BAO_STANDARD_RULER|DARK_ENERGY_EVOLUTION_HINTS|CMB_ACOUSTIC_SURFACE",
            "truth_boundary": "Pressure mapping only; no H0 correction claim.",
        }
    )
    plan.append(
        {
            "step": 3,
            "test_name": "v3.3 Growth vs Geometry Split",
            "purpose": "Separate geometry surfaces from consolidation/growth surfaces to test TAIRID's propagation-consolidation split.",
            "surfaces": "S8_STRUCTURE_GROWTH|COSMIC_WEB_VOIDS|BAO_STANDARD_RULER",
            "truth_boundary": "Association and model-translation only; not proof of new matter physics.",
        }
    )
    plan.append(
        {
            "step": 4,
            "test_name": "v3.4 Early Consolidation Pressure Audit",
            "purpose": "Audit JWST early-structure claims under uncertainty instead of using them as instant proof.",
            "surfaces": "JWST_EARLY_STRUCTURE|CMB_ACOUSTIC_SURFACE",
            "truth_boundary": "Uncertainty-aware pressure audit only.",
        }
    )
    plan.append(
        {
            "step": 5,
            "test_name": "v3.5 Horizon Boundary Constraint Ledger",
            "purpose": "Translate black-hole horizon/information boundaries into TAIRID terms only after preserving known GR/thermodynamic surface behavior.",
            "surfaces": "BLACK_HOLE_HORIZON_ENTROPY",
            "truth_boundary": "Conceptual constraint ledger first; no numerical claim until equations exist.",
        }
    )
    return plan


def build_failure_comparison_matrix(rows):
    matrix = []
    for row in rows:
        matrix.append(
            {
                "atlas_id": row["atlas_id"],
                "standard_model_surface_success": row["standard_model_success"],
                "standard_model_pressure_seam": row["standard_model_pressure_seam"],
                "tairid_translation": row["tairid_translation"],
                "tairid_must_match": row["preservation_requirement"],
                "tairid_failure_condition": row["failure_condition"],
                "same_failure_test": (
                    "If current cosmology strains here, check whether TAIRID also strains at the same observable surface "
                    "or whether it identifies a more specific hidden boundary/pressure layer."
                ),
                "do_not_claim": "Do not claim victory just because the existing model has pressure here.",
            }
        )
    return matrix


def make_plots(rows, operator_coverage):
    try:
        labels = [r["atlas_id"] for r in rows]
        scores = [r["test_priority_score"] for r in rows]
        x = np.arange(len(labels))

        plt.figure(figsize=(14, 6))
        plt.bar(x, scores)
        plt.xticks(x, labels, rotation=70, ha="right", fontsize=8)
        plt.ylabel("priority score")
        plt.title("TAIRID v3.0 cosmology atlas next-test priority")
        plt.tight_layout()
        plt.savefig(OUTDIR / "atlas_priority_scores_v3_0_fresh.png", dpi=160)
        plt.close()

        op_labels = [r["operator"] for r in operator_coverage]
        counts = [r["surface_count"] for r in operator_coverage]
        x2 = np.arange(len(op_labels))

        plt.figure(figsize=(12, 5))
        plt.bar(x2, counts)
        plt.xticks(x2, op_labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel("surface count")
        plt.title("TAIRID operator coverage across cosmology atlas")
        plt.tight_layout()
        plt.savefig(OUTDIR / "operator_coverage_v3_0_fresh.png", dpi=160)
        plt.close()
    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v3_0_fresh.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )


def write_markdown_report(rows, operator_coverage, source_rows, next_plan, decision):
    lines = []
    lines.append("# TAIRID Cosmology Failure-Mode Atlas v3.0 Fresh")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("This atlas does not validate TAIRID, disprove standard cosmology, prove H0 correction, or prove new physics. It defines the comparison surface before another data test.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## Top priority surfaces")
    lines.append("")
    for row in rows[:5]:
        lines.append(f"### {row['atlas_id']}")
        lines.append(f"- Observable surface: {row['observable_surface']}")
        lines.append(f"- Current-model pressure seam: {row['standard_model_pressure_seam']}")
        lines.append(f"- TAIRID translation: {row['tairid_translation']}")
        lines.append(f"- Must preserve: {row['preservation_requirement']}")
        lines.append(f"- Failure condition: {row['failure_condition']}")
        lines.append(f"- Priority score: `{row['test_priority_score']}`")
        lines.append("")
    lines.append("## Operator coverage")
    lines.append("")
    for op in operator_coverage:
        lines.append(f"- `{op['operator']}`: {op['surface_count']} surfaces")
    lines.append("")
    lines.append("## Next test plan")
    lines.append("")
    for step in next_plan:
        lines.append(f"{step['step']}. **{step['test_name']}** — {step['purpose']}")
    lines.append("")
    return "\n".join(lines)


def decide(rows, source_rows, next_plan, operator_coverage):
    hard_guard_count = sum(1 for r in rows if r["next_test_kind"] == "hard_falsification_guard")
    preservation_first_count = sum(1 for r in rows if r["next_test_kind"] == "preservation_first")
    reachable_sources = sum(1 for s in source_rows if s["reachability_status"] == "reachable")
    operator_count = len(operator_coverage)

    gates = [
        {
            "gate": "G1_atlas_has_enough_surfaces",
            "passed": len(rows) >= 10,
            "evidence": {"atlas_surface_count": len(rows)},
        },
        {
            "gate": "G2_preservation_guards_declared",
            "passed": hard_guard_count >= 2 and preservation_first_count >= 1,
            "evidence": {
                "hard_guard_count": hard_guard_count,
                "preservation_first_count": preservation_first_count,
            },
        },
        {
            "gate": "G3_operator_translation_not_empty",
            "passed": operator_count >= 6,
            "evidence": {"operator_count": operator_count},
        },
        {
            "gate": "G4_next_test_plan_created",
            "passed": len(next_plan) >= 3,
            "evidence": {"next_plan_count": len(next_plan)},
        },
        {
            "gate": "G5_source_reachability_checked",
            "passed": len(source_rows) >= 5,
            "evidence": {
                "source_count": len(source_rows),
                "reachable_source_count": reachable_sources,
                "note": "Reachability is not source validation.",
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
        final_status = "cosmology_failure_mode_atlas_ready_for_preservation_guard_battery"
        readiness = 9
        next_wall = (
            "Run v3.1 as a preservation guard battery before any pressure-seam model test. "
            "TAIRID must preserve time dilation, distance duality/Tolman behavior, BAO, and CMB acoustic surfaces."
        )
    elif len(failed) <= 2:
        final_status = "cosmology_failure_mode_atlas_ready_with_cautions"
        readiness = 7
        next_wall = "Review failed atlas gates before building v3.1."
    else:
        final_status = "cosmology_failure_mode_atlas_not_ready"
        readiness = 5
        next_wall = "Atlas is incomplete. Do not run a new cosmology test yet."

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "gates": gates,
        "failed_gates": failed,
        "truth_boundary": CLAIMS_V3_0["truth_boundary"],
    }


def write_handoff(decision, rows, next_plan):
    lines = []
    lines.append("# TAIRID v3.0 Fresh Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What changed")
    lines.append("")
    lines.append("- We stopped treating one narrow SH0ES residual bridge as a theory-of-everything test.")
    lines.append("- We built a cosmology failure-mode atlas instead.")
    lines.append("- Each surface now has a current-model success, pressure seam, TAIRID translation, preservation requirement, and failure condition.")
    lines.append("- The next test must be a preservation guard battery before anomaly or pressure-seam testing.")
    lines.append("")
    lines.append("## Highest-priority atlas rows")
    lines.append("")
    for row in rows[:5]:
        lines.append(f"- `{row['atlas_id']}` — score `{row['test_priority_score']}` — {row['recommended_action']}")
    lines.append("")
    lines.append("## Next planned tests")
    lines.append("")
    for step in next_plan:
        lines.append(f"- `{step['test_name']}` — {step['purpose']}")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v3.0 is an atlas/ledger builder only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not disprove standard cosmology.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append("Build v3.1 — Preservation Guard Battery. It should test whether the TAIRID cosmology translation preserves the hard surfaces: SN time dilation, Tolman/distance duality, BAO/ruler behavior, and CMB acoustic-scale constraints.")
    else:
        lines.append("Repair the atlas before building v3.1.")
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID Cosmology Failure-Mode Atlas v3.0 Fresh starting.")
    print("Boundary: atlas/ledger only; no validation, no H0 claim, no new physics claim.")

    write_json(OUTDIR / "claims_v3_0_fresh.json", CLAIMS_V3_0)
    write_json(OUTDIR / "tairid_operator_library_v3_0_fresh.json", TAIRID_OPERATOR_LIBRARY)

    try:
        rows = add_derived_fields(ATLAS_ROWS)
        source_rows = source_reachability_check(SOURCE_CANDIDATES)
        operator_coverage = build_operator_coverage(rows)
        next_plan = build_next_test_plan(rows)
        failure_matrix = build_failure_comparison_matrix(rows)
        decision = decide(rows, source_rows, next_plan, operator_coverage)

        write_csv(OUTDIR / "cosmology_failure_mode_atlas_v3_0_fresh.csv", [flatten_row(r) for r in rows])
        write_json(OUTDIR / "cosmology_failure_mode_atlas_v3_0_fresh.json", rows)

        write_csv(OUTDIR / "source_reachability_v3_0_fresh.csv", source_rows)
        write_json(OUTDIR / "source_reachability_v3_0_fresh.json", source_rows)

        write_csv(OUTDIR / "operator_coverage_v3_0_fresh.csv", operator_coverage)
        write_json(OUTDIR / "operator_coverage_v3_0_fresh.json", operator_coverage)

        write_csv(OUTDIR / "failure_comparison_matrix_v3_0_fresh.csv", failure_matrix)
        write_json(OUTDIR / "failure_comparison_matrix_v3_0_fresh.json", failure_matrix)

        write_csv(OUTDIR / "next_test_plan_v3_0_fresh.csv", next_plan)
        write_json(OUTDIR / "next_test_plan_v3_0_fresh.json", next_plan)

        write_json(OUTDIR / "decision_v3_0_fresh.json", decision)

        make_plots(rows, operator_coverage)

        report = write_markdown_report(rows, operator_coverage, source_rows, next_plan, decision)
        (OUTDIR / "cosmology_failure_mode_atlas_report_v3_0_fresh.md").write_text(report, encoding="utf-8")
        (OUTDIR / "cosmology_failure_mode_atlas_report_v3_0_fresh.txt").write_text(report, encoding="utf-8")

        handoff = write_handoff(decision, rows, next_plan)
        (OUTDIR / "next_thread_handoff_after_v3_0_fresh.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v3_0_fresh.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID Cosmology Failure-Mode Atlas v3.0 Fresh",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Atlas/ledger only. No validation, no standard-cosmology disproof, no H0 claim, no new-physics claim.",
            "atlas_surface_count": len(rows),
            "top_priority_surfaces": [
                {
                    "atlas_id": r["atlas_id"],
                    "priority_score": r["test_priority_score"],
                    "next_test_kind": r["next_test_kind"],
                    "recommended_action": r["recommended_action"],
                }
                for r in rows[:5]
            ],
            "operator_coverage": operator_coverage,
            "source_reachability_summary": {
                "source_count": len(source_rows),
                "reachable_count": sum(1 for s in source_rows if s["reachability_status"] == "reachable"),
                "truth_boundary": "reachability is not source validation",
            },
            "next_test_plan": next_plan,
            "decision": decision,
            "claims_v3_0": CLAIMS_V3_0,
            "output_files": {
                "summary_json": str(OUTDIR / "cosmology_failure_mode_atlas_v3_0_fresh_summary.json"),
                "summary_txt": str(OUTDIR / "cosmology_failure_mode_atlas_v3_0_fresh_summary.txt"),
                "atlas_csv": str(OUTDIR / "cosmology_failure_mode_atlas_v3_0_fresh.csv"),
                "atlas_json": str(OUTDIR / "cosmology_failure_mode_atlas_v3_0_fresh.json"),
                "failure_matrix_csv": str(OUTDIR / "failure_comparison_matrix_v3_0_fresh.csv"),
                "operator_coverage_csv": str(OUTDIR / "operator_coverage_v3_0_fresh.csv"),
                "source_reachability_csv": str(OUTDIR / "source_reachability_v3_0_fresh.csv"),
                "next_test_plan_csv": str(OUTDIR / "next_test_plan_v3_0_fresh.csv"),
                "decision_json": str(OUTDIR / "decision_v3_0_fresh.json"),
                "report_md": str(OUTDIR / "cosmology_failure_mode_atlas_report_v3_0_fresh.md"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v3_0_fresh.md"),
                "plots": [
                    str(OUTDIR / "atlas_priority_scores_v3_0_fresh.png"),
                    str(OUTDIR / "operator_coverage_v3_0_fresh.png"),
                ],
            },
            "interpretation": {
                "what_success_means": "The cosmology comparison has been reframed into surfaces, pressure seams, preservation requirements, and testable failure conditions.",
                "what_success_does_not_mean": "This does not prove TAIRID, disprove standard cosmology, or show new physics.",
                "next_required_step": "v3.1 should be a preservation guard battery before any pressure-seam/anomaly test.",
                "truth_boundary": CLAIMS_V3_0["truth_boundary"],
            },
        }

        write_json(OUTDIR / "cosmology_failure_mode_atlas_v3_0_fresh_summary.json", summary)

        with open(OUTDIR / "cosmology_failure_mode_atlas_v3_0_fresh_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Cosmology Failure-Mode Atlas v3.0 Fresh\n\n")
            f.write("Boundary: atlas/ledger only. No validation. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Top priority surfaces:\n")
            f.write(json.dumps(summary["top_priority_surfaces"], indent=2, default=json_default) + "\n\n")
            f.write("Next test plan:\n")
            f.write(json.dumps(next_plan, indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not disprove standard cosmology.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This only builds the comparison atlas for the next test.\n")

        print("TAIRID Cosmology Failure-Mode Atlas v3.0 Fresh complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Cosmology Failure-Mode Atlas v3.0 Fresh",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "cosmology_failure_mode_atlas_v3_0_fresh_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V3_0["truth_boundary"],
        }
        write_json(OUTDIR / "cosmology_failure_mode_atlas_v3_0_fresh_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
