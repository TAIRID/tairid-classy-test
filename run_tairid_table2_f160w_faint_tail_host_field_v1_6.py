#!/usr/bin/env python3
"""
TAIRID Table2 F160W faint-tail host/field concentration audit v1.6.

This script reuses the already-created v1.5 helper functions, but it repairs
the pasted/glued v1.5 footer at runtime before importing those helpers.

Purpose:
v1.5 localized the strongest non-overlapping F160W residual pressure to the
faintest 5% band:

    f160w_band_95_100

This test asks whether that faint-tail signal is:

1. distributed across many hosts/fields,
2. dominated by one host,
3. dominated by one Table2 field,
4. or caused by a small local cluster.

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only audits whether the SH0ES Table2 faint-tail F160W residual pressure is
distributed or concentrated.
"""

import csv
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import chi2


OUTDIR = Path("tairid_table2_f160w_faint_tail_host_field_v1_6_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_V15 = Path("run_tairid_table2_f160w_quantile_bands_v1_5.py")
CLEANED_V15 = Path("run_tairid_table2_f160w_quantile_bands_v1_5_importable_for_v1_6.py")

F160W_COLUMN = "table2_num_7"
FIELD_COLUMN = "table2_cell_0"
SEED = 42
PERMUTATION_REPEATS = 80
EPS = 1.0e-12


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


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))[:160]


def standardize(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    std = float(np.std(values))

    if not np.isfinite(std) or std <= 1.0e-14:
        return np.zeros_like(values)

    return (values - float(np.mean(values))) / std


def load_v15_helpers():
    if not SOURCE_V15.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_V15}. Create the v1.5 script file first, then run v1.6."
        )

    text = SOURCE_V15.read_text(encoding="utf-8", errors="replace")

    marker_double = 'if __name__ == "__main__":'
    marker_single = "if __name__ == '__main__':"

    positions = [idx for idx in [text.find(marker_double), text.find(marker_single)] if idx >= 0]

    if not positions:
        raise RuntimeError("Could not find the v1.5 __main__ block to remove.")

    cut = min(positions)
    cleaned = text[:cut].rstrip() + "\n"

    CLEANED_V15.write_text(cleaned, encoding="utf-8")
    compile(cleaned, str(CLEANED_V15), "exec")

    ns = {
        "__file__": str(CLEANED_V15),
        "__name__": "tairid_v15_importable_for_v16",
    }
    exec(cleaned, ns)

    ns["OUTDIR"] = OUTDIR
    ns["DOWNLOAD_DIR"] = DOWNLOAD_DIR

    return ns, {
        "source_file": str(SOURCE_V15),
        "cleaned_file": str(CLEANED_V15),
        "source_size_chars": len(text),
        "cleaned_size_chars": len(cleaned),
        "source_contains_bad_glue_main_import_json": "main()import json" in text,
        "status": "success",
    }


def get_numeric(row, key, default=np.nan):
    try:
        value = float(row.get(key, default))
    except Exception:
        value = default

    return value if np.isfinite(value) else default


def get_entity(row, key):
    value = str(row.get(key, "")).strip()

    if not value:
        return "UNKNOWN"

    return safe_name(value.upper())


def build_faint_tail_mask(mapped_rows, y_length):
    finite = []

    for row in mapped_rows:
        value = get_numeric(row, F160W_COLUMN)

        if np.isfinite(value):
            finite.append((value, row))

    finite = sorted(finite, key=lambda item: item[0])

    n = len(finite)
    start = int(math.floor(n * 0.95))
    end = n

    tail = finite[start:end]
    mask = np.zeros(y_length, dtype=float)

    tail_rows = []

    for value, row in tail:
        idx = int(row["compact_observation_index"])
        mask[idx] = 1.0

        enriched = dict(row)
        enriched["f160w_value"] = float(value)
        enriched["field_guess"] = get_entity(row, FIELD_COLUMN)
        enriched["host_guess_clean"] = get_entity(row, "host_guess")
        tail_rows.append(enriched)

    values = np.asarray([v for v, _ in tail], dtype=float)

    return mask, tail_rows, {
        "column": F160W_COLUMN,
        "band": "95_100",
        "plain_meaning": "faintest 5 percent by F160W-like magnitude",
        "finite_count": n,
        "tail_count": len(tail_rows),
        "f160w_min": float(np.min(values)) if len(values) else None,
        "f160w_max": float(np.max(values)) if len(values) else None,
        "f160w_mean": float(np.mean(values)) if len(values) else None,
        "start_rank_index": int(start),
        "end_rank_index_exclusive": int(end),
    }


def audit_vector(name, values, kind, y, c_factor, fit, design_name, metadata=None):
    metadata = metadata or {}
    raw = np.asarray(values, dtype=float).reshape(-1)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    z = standardize(raw)

    c_inv_z = cho_solve(c_factor, z, check_finite=False)
    raw_norm2 = float(z.T @ c_inv_z)

    base = {
        "design": design_name,
        "candidate": name,
        "candidate_kind": kind,
        "count_nonzero_raw": int(np.sum(np.abs(raw) > 1.0e-12)),
        **metadata,
    }

    if raw_norm2 <= EPS:
        return {
            **base,
            "status": "zero_or_near_zero_raw_norm",
            "delta_chi2_score": 0.0,
            "p_value_chi2_one_dof": 1.0,
            "nondegenerate_ratio": 0.0,
            "delta_aic_if_added_column": 2.0,
            "delta_bic_if_added_column": float(math.log(len(y))),
            "score": 0.0,
        }

    x_t_cinv_z = fit["D"].T @ c_inv_z
    coeff = fit["normal_inv"] @ x_t_cinv_z

    z_perp = z - fit["D"] @ coeff
    c_inv_z_perp = c_inv_z - fit["Cinv_D"] @ coeff

    perp_norm2 = float(z_perp.T @ c_inv_z_perp)
    raw_norm = float(math.sqrt(max(raw_norm2, 0.0)))
    perp_norm = float(math.sqrt(max(perp_norm2, 0.0)))
    ratio = float(perp_norm / max(raw_norm, EPS))

    if perp_norm2 <= EPS:
        score = 0.0
        delta = 0.0
        alpha = 0.0
    else:
        score = float(z_perp.T @ fit["Cinv_residual"])
        delta = float((score * score) / perp_norm2)
        alpha = float(score / perp_norm2)

    p_value = float(chi2.sf(max(delta, 0.0), 1))

    return {
        **base,
        "status": "ok",
        "base_design_k": int(fit["k"]),
        "base_chi2": float(fit["chi2"]),
        "base_reduced_chi2": float(fit["reduced_chi2"]),
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "raw_Cinv_norm": raw_norm,
        "residualized_Cinv_norm": perp_norm,
        "nondegenerate_ratio": ratio,
        "projection_absorption_fraction": float(1.0 - ratio),
        "score": score,
        "alpha_hat_added_column": alpha,
        "delta_chi2_score": delta,
        "p_value_chi2_one_dof": p_value,
        "delta_aic_if_added_column": float(2.0 - delta),
        "delta_bic_if_added_column": float(math.log(len(y)) - delta),
        "would_improve_aic": bool(2.0 - delta < 0.0),
        "would_improve_bic": bool(math.log(len(y)) - delta < 0.0),
    }


def entity_masks(tail_rows, y_length, entity_key):
    grouped = defaultdict(list)

    for row in tail_rows:
        grouped[row[entity_key]].append(row)

    masks = {}

    for entity, rows in grouped.items():
        mask = np.zeros(y_length, dtype=float)

        for row in rows:
            mask[int(row["compact_observation_index"])] = 1.0

        masks[entity] = {
            "mask": mask,
            "rows": rows,
            "count": int(np.sum(mask > 0)),
        }

    return masks


def total_counts_by_entity(mapped_rows, entity_kind):
    counts = Counter()

    if entity_kind == "host":
        for row in mapped_rows:
            counts[get_entity(row, "host_guess")] += 1

    elif entity_kind == "field":
        for row in mapped_rows:
            counts[get_entity(row, FIELD_COLUMN)] += 1

    else:
        raise ValueError(entity_kind)

    return counts


def analyze_entity_concentration(entity_kind, entity_key, tail_rows, full_tail_mask, mapped_rows, y, c_factor, fit, design_name):
    y_length = len(y)
    masks = entity_masks(tail_rows, y_length, entity_key)
    totals = total_counts_by_entity(mapped_rows, entity_kind)

    component_rows = []

    for entity, info in masks.items():
        mask = info["mask"]
        count = info["count"]
        total = totals.get(entity, 0)

        audit = audit_vector(
            name=f"faint_tail_{entity_kind}_component_{entity}",
            values=mask,
            kind=f"{entity_kind}_component",
            y=y,
            c_factor=c_factor,
            fit=fit,
            design_name=design_name,
            metadata={
                entity_kind: entity,
                "faint_tail_count": count,
                "total_entity_count": int(total),
                "faint_tail_fraction_within_entity": float(count / total) if total else None,
            },
        )

        component_rows.append(audit)

    component_rows = sorted(
        component_rows,
        key=lambda r: (-r.get("delta_chi2_score", 0.0), -r.get("faint_tail_count", 0), r.get(entity_kind, "")),
    )

    leaveout_rows = []

    for row in component_rows:
        entity = row[entity_kind]
        leave_mask = full_tail_mask.copy()
        leave_mask[masks[entity]["mask"] > 0] = 0.0

        audit = audit_vector(
            name=f"faint_tail_without_{entity_kind}_{entity}",
            values=leave_mask,
            kind=f"leave_one_{entity_kind}_out",
            y=y,
            c_factor=c_factor,
            fit=fit,
            design_name=design_name,
            metadata={
                entity_kind: entity,
                "removed_faint_tail_count": row["faint_tail_count"],
                "total_entity_count": row["total_entity_count"],
            },
        )

        leaveout_rows.append(audit)

    return component_rows, leaveout_rows


def cumulative_removal(entity_kind, component_rows, tail_rows, full_tail_mask, y, c_factor, fit, design_name, top_values=(1, 2, 3, 5, 10)):
    y_length = len(y)
    entity_key = "host_guess_clean" if entity_kind == "host" else "field_guess"
    masks = entity_masks(tail_rows, y_length, entity_key)

    ordered_entities = [row[entity_kind] for row in component_rows]
    rows = []

    for k in top_values:
        selected = ordered_entities[:k]

        if not selected:
            continue

        reduced = full_tail_mask.copy()
        removed_count = 0

        for entity in selected:
            if entity not in masks:
                continue

            removed_count += int(np.sum(masks[entity]["mask"] > 0))
            reduced[masks[entity]["mask"] > 0] = 0.0

        audit = audit_vector(
            name=f"faint_tail_without_top_{k}_{entity_kind}_components",
            values=reduced,
            kind=f"remove_top_{k}_{entity_kind}_components",
            y=y,
            c_factor=c_factor,
            fit=fit,
            design_name=design_name,
            metadata={
                "entity_kind": entity_kind,
                "removed_entities": ",".join(selected),
                "removed_entity_count": len(selected),
                "removed_faint_tail_rows": removed_count,
            },
        )

        rows.append(audit)

    return rows


def permutation_control(name, values, y, c_factor, fit, design_name, mapped_rows, repeats=PERMUTATION_REPEATS):
    rng = np.random.default_rng(SEED)
    raw = np.asarray(values, dtype=float).reshape(-1)

    spine_indices = np.asarray(
        [int(row["compact_observation_index"]) for row in mapped_rows],
        dtype=int,
    )

    spine_values = raw[spine_indices].copy()
    rows = []

    for i in range(repeats):
        permuted = raw.copy()
        pvals = spine_values.copy()
        rng.shuffle(pvals)
        permuted[spine_indices] = pvals

        rows.append(
            audit_vector(
                name=f"{name}_permutation_{i}",
                values=permuted,
                kind="permutation_control",
                y=y,
                c_factor=c_factor,
                fit=fit,
                design_name=design_name,
            )
        )

    observed = audit_vector(
        name=name,
        values=raw,
        kind="observed",
        y=y,
        c_factor=c_factor,
        fit=fit,
        design_name=design_name,
    )

    deltas = np.asarray([r["delta_chi2_score"] for r in rows], dtype=float)
    ratios = np.asarray([r["nondegenerate_ratio"] for r in rows], dtype=float)

    summary = {
        "design": design_name,
        "candidate": name,
        "repeats": repeats,
        "observed_delta_chi2": observed["delta_chi2_score"],
        "observed_nondegenerate_ratio": observed["nondegenerate_ratio"],
        "permutation_delta_mean": float(np.mean(deltas)),
        "permutation_delta_95": float(np.percentile(deltas, 95)),
        "permutation_delta_99": float(np.percentile(deltas, 99)),
        "permutation_ratio_mean": float(np.mean(ratios)),
        "permutation_ratio_95": float(np.percentile(ratios, 95)),
        "observed_exceeds_95_percent_permutation_delta": bool(observed["delta_chi2_score"] > float(np.percentile(deltas, 95))),
        "observed_exceeds_99_percent_permutation_delta": bool(observed["delta_chi2_score"] > float(np.percentile(deltas, 99))),
    }

    return rows, summary


def add_drops(rows, full_delta):
    out = []

    for row in rows:
        r = dict(row)
        drop = float(full_delta - r.get("delta_chi2_score", 0.0))
        r["delta_drop_from_full_faint_tail"] = drop
        r["delta_drop_fraction_from_full_faint_tail"] = float(drop / full_delta) if full_delta else None
        out.append(r)

    return out


def decide_status(full_audit, host_component_rows, host_leaveout_rows, host_removal_rows, field_component_rows, field_leaveout_rows, field_removal_rows):
    top_host = host_component_rows[0] if host_component_rows else None
    top_field = field_component_rows[0] if field_component_rows else None

    leaveout_by_host = {row.get("host"): row for row in host_leaveout_rows}
    leaveout_by_field = {row.get("field"): row for row in field_leaveout_rows}

    top_host_leave = leaveout_by_host.get(top_host["host"]) if top_host else None
    top_field_leave = leaveout_by_field.get(top_field["field"]) if top_field else None

    top_host_drop_frac = top_host_leave.get("delta_drop_fraction_from_full_faint_tail") if top_host_leave else None
    top_field_drop_frac = top_field_leave.get("delta_drop_fraction_from_full_faint_tail") if top_field_leave else None

    host_top3 = next((r for r in host_removal_rows if r.get("removed_entity_count") == 3), None)
    field_top3 = next((r for r in field_removal_rows if r.get("removed_entity_count") == 3), None)

    host_top3_drop = host_top3.get("delta_drop_fraction_from_full_faint_tail") if host_top3 else None
    field_top3_drop = field_top3.get("delta_drop_fraction_from_full_faint_tail") if field_top3 else None

    host_count = len(host_component_rows)
    field_count = len(field_component_rows)

    best_cases = {
        "full_faint_tail_audit": full_audit,
        "top_host_component": top_host,
        "top_host_leaveout": top_host_leave,
        "top_field_component": top_field,
        "top_field_leaveout": top_field_leave,
        "host_top3_removal": host_top3,
        "field_top3_removal": field_top3,
        "host_count_in_faint_tail": host_count,
        "field_count_in_faint_tail": field_count,
    }

    if top_field_drop_frac is not None and top_field_drop_frac >= 0.50:
        return (
            "faint_tail_signal_field_concentrated",
            7,
            "The faint-tail signal drops heavily when the top field component is removed; next test should be field-level geometry.",
            best_cases,
        )

    if top_host_drop_frac is not None and top_host_drop_frac >= 0.50:
        return (
            "faint_tail_signal_host_concentrated",
            7,
            "The faint-tail signal drops heavily when the top host component is removed; next test should be host-specific artifact checks.",
            best_cases,
        )

    if field_top3_drop is not None and field_top3_drop >= 0.70:
        return (
            "faint_tail_signal_small_field_cluster_concentrated",
            7,
            "The faint-tail signal is not one field only, but top fields explain most of it; next test should be field cluster geometry.",
            best_cases,
        )

    if host_top3_drop is not None and host_top3_drop >= 0.70:
        return (
            "faint_tail_signal_small_host_cluster_concentrated",
            7,
            "The faint-tail signal is not one host only, but top hosts explain most of it; next test should be host cluster checks.",
            best_cases,
        )

    if host_count >= 5 and field_count >= 10 and (top_host_drop_frac is None or top_host_drop_frac < 0.35) and (top_field_drop_frac is None or top_field_drop_frac < 0.35):
        return (
            "faint_tail_signal_distributed_across_hosts_and_fields",
            8,
            "The faint-tail signal survives top host and top field removal enough to look distributed rather than local.",
            best_cases,
        )

    return (
        "faint_tail_concentration_inconclusive",
        6,
        "The faint-tail signal remains, but host/field concentration is not clean enough to classify.",
        best_cases,
    )


def make_plots(host_component_rows, host_leaveout_rows, field_component_rows, field_leaveout_rows):
    host_top = host_component_rows[:20]
    field_top = field_component_rows[:25]

    if host_top:
        labels = [r["host"] for r in host_top]
        values = [r["delta_chi2_score"] for r in host_top]

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(labels)), values)
        plt.xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=8)
        plt.ylabel("delta chi2 score")
        plt.title("Faint-tail component pressure by host v1.6")
        plt.tight_layout()
        plt.savefig(OUTDIR / "faint_tail_component_pressure_by_host_v1_6.png", dpi=160)
        plt.close()

    if field_top:
        labels = [r["field"] for r in field_top]
        values = [r["delta_chi2_score"] for r in field_top]

        plt.figure(figsize=(14, 5))
        plt.bar(np.arange(len(labels)), values)
        plt.xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        plt.ylabel("delta chi2 score")
        plt.title("Faint-tail component pressure by field v1.6")
        plt.tight_layout()
        plt.savefig(OUTDIR / "faint_tail_component_pressure_by_field_v1_6.png", dpi=160)
        plt.close()

    if host_leaveout_rows:
        rows = sorted(host_leaveout_rows, key=lambda r: -r.get("delta_drop_fraction_from_full_faint_tail", 0.0))[:20]
        labels = [r["host"] for r in rows]
        values = [r.get("delta_drop_fraction_from_full_faint_tail", 0.0) for r in rows]

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(labels)), values)
        plt.xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=8)
        plt.ylabel("fraction of full delta removed")
        plt.title("Leave-one-host-out drop from faint-tail signal v1.6")
        plt.tight_layout()
        plt.savefig(OUTDIR / "leave_one_host_out_drop_v1_6.png", dpi=160)
        plt.close()

    if field_leaveout_rows:
        rows = sorted(field_leaveout_rows, key=lambda r: -r.get("delta_drop_fraction_from_full_faint_tail", 0.0))[:25]
        labels = [r["field"] for r in rows]
        values = [r.get("delta_drop_fraction_from_full_faint_tail", 0.0) for r in rows]

        plt.figure(figsize=(14, 5))
        plt.bar(np.arange(len(labels)), values)
        plt.xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        plt.ylabel("fraction of full delta removed")
        plt.title("Leave-one-field-out drop from faint-tail signal v1.6")
        plt.tight_layout()
        plt.savefig(OUTDIR / "leave_one_field_out_drop_v1_6.png", dpi=160)
        plt.close()


def main():
    print("")
    print("TAIRID Table2 F160W faint-tail host/field concentration audit v1.6 starting.")
    print("Boundary: distribution/concentration audit only; not proof.")
    print("")

    repair_summary = {}

    try:
        ns, repair_summary = load_v15_helpers()
        write_json(OUTDIR / "v15_import_repair_summary_v1_6.json", repair_summary)

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

        compact_files = ns["COMPACT_FILES"]
        aux_files = ns["AUX_FILES"]

        downloads = {}
        aux_results = []
        ledger = []

        for label, repo_path in compact_files.items():
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

        for repo_path in aux_files:
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

        write_csv(OUTDIR / "download_ledger_v1_6.csv", ledger)
        write_json(OUTDIR / "download_attempts_v1_6.json", {"compact": downloads, "auxiliary": aux_results})

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

        write_json(OUTDIR / "parse_meta_v1_6.json", parse_meta)
        write_json(OUTDIR / "parse_errors_v1_6.json", parse_errors)

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
                "test_name": "TAIRID Table2 F160W faint-tail host/field concentration audit v1.6",
                "final_status": "table2_f160w_faint_tail_host_field_v1_6_parse_or_download_failed",
                "readiness_score_0_to_10": 4,
                "next_wall": "Fix compact matrix or table2 retrieval before host/field concentration audit.",
                "parse_errors": parse_errors,
                "table2_downloaded": bool(table2_result),
                "repair_summary": repair_summary,
            }
            write_json(OUTDIR / "table2_f160w_faint_tail_host_field_v1_6_summary.json", summary)
            print("Parse/download failed. See summary JSON.")
            return

        table2_all_rows, table2_data_rows, table2_header_rows = parse_table2_tex(Path(table2_result["local_path"]))

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

        write_csv(OUTDIR / "compact_row_map_v1_6.csv", row_rows)
        write_csv(OUTDIR / "compact_cluster_map_v1_6.csv", cluster_rows)

        mapped_rows, map_status = map_table2_to_spine(row_rows, table2_data_rows)
        host_summary = summarize_hosts(mapped_rows)
        numeric_rows, likely_numeric_labels = numeric_feature_summary(mapped_rows)

        designs, control_metadata = build_designs(X, mapped_rows, host_summary, len(y))
        design_fits = {}

        for design_name, design in designs.items():
            design_fits[design_name] = gls_fit(y, design["D"], c_factor)

        full_design_name = "plus_host_top10_row_order_measurement_controls"
        full_fit = design_fits[full_design_name]
        original_fit = design_fits["original_47"]

        faint_tail_mask, tail_rows, tail_meta = build_faint_tail_mask(mapped_rows, len(y))
        write_csv(OUTDIR / "faint_tail_rows_v1_6.csv", tail_rows)
        write_json(OUTDIR / "faint_tail_meta_v1_6.json", tail_meta)

        full_audit = audit_vector(
            name="f160w_faint_tail_95_100_full",
            values=faint_tail_mask,
            kind="faint_tail_full_mask",
            y=y,
            c_factor=c_factor,
            fit=full_fit,
            design_name=full_design_name,
            metadata=tail_meta,
        )

        original_audit = audit_vector(
            name="f160w_faint_tail_95_100_original",
            values=faint_tail_mask,
            kind="faint_tail_full_mask",
            y=y,
            c_factor=c_factor,
            fit=original_fit,
            design_name="original_47",
            metadata=tail_meta,
        )

        host_component_rows, host_leaveout_rows = analyze_entity_concentration(
            entity_kind="host",
            entity_key="host_guess_clean",
            tail_rows=tail_rows,
            full_tail_mask=faint_tail_mask,
            mapped_rows=mapped_rows,
            y=y,
            c_factor=c_factor,
            fit=full_fit,
            design_name=full_design_name,
        )

        field_component_rows, field_leaveout_rows = analyze_entity_concentration(
            entity_kind="field",
            entity_key="field_guess",
            tail_rows=tail_rows,
            full_tail_mask=faint_tail_mask,
            mapped_rows=mapped_rows,
            y=y,
            c_factor=c_factor,
            fit=full_fit,
            design_name=full_design_name,
        )

        host_leaveout_rows = add_drops(host_leaveout_rows, full_audit["delta_chi2_score"])
        field_leaveout_rows = add_drops(field_leaveout_rows, full_audit["delta_chi2_score"])

        host_removal_rows = cumulative_removal(
            entity_kind="host",
            component_rows=host_component_rows,
            tail_rows=tail_rows,
            full_tail_mask=faint_tail_mask,
            y=y,
            c_factor=c_factor,
            fit=full_fit,
            design_name=full_design_name,
        )
        host_removal_rows = add_drops(host_removal_rows, full_audit["delta_chi2_score"])

        field_removal_rows = cumulative_removal(
            entity_kind="field",
            component_rows=field_component_rows,
            tail_rows=tail_rows,
            full_tail_mask=faint_tail_mask,
            y=y,
            c_factor=c_factor,
            fit=full_fit,
            design_name=full_design_name,
        )
        field_removal_rows = add_drops(field_removal_rows, full_audit["delta_chi2_score"])

        write_csv(OUTDIR / "host_component_audit_v1_6.csv", host_component_rows)
        write_csv(OUTDIR / "host_leave_one_out_audit_v1_6.csv", host_leaveout_rows)
        write_csv(OUTDIR / "host_cumulative_removal_audit_v1_6.csv", host_removal_rows)

        write_csv(OUTDIR / "field_component_audit_v1_6.csv", field_component_rows)
        write_csv(OUTDIR / "field_leave_one_out_audit_v1_6.csv", field_leaveout_rows)
        write_csv(OUTDIR / "field_cumulative_removal_audit_v1_6.csv", field_removal_rows)

        perm_rows, perm_summary = permutation_control(
            name="f160w_faint_tail_95_100_full",
            values=faint_tail_mask,
            y=y,
            c_factor=c_factor,
            fit=full_fit,
            design_name=full_design_name,
            mapped_rows=mapped_rows,
        )
        write_csv(OUTDIR / "faint_tail_permutation_details_v1_6.csv", perm_rows)
        write_json(OUTDIR / "faint_tail_permutation_summary_v1_6.json", perm_summary)

        final_status, readiness_score, next_wall, best_cases = decide_status(
            full_audit=full_audit,
            host_component_rows=host_component_rows,
            host_leaveout_rows=host_leaveout_rows,
            host_removal_rows=host_removal_rows,
            field_component_rows=field_component_rows,
            field_leaveout_rows=field_leaveout_rows,
            field_removal_rows=field_removal_rows,
        )

        make_plots(
            host_component_rows=host_component_rows,
            host_leaveout_rows=host_leaveout_rows,
            field_component_rows=field_component_rows,
            field_leaveout_rows=field_leaveout_rows,
        )

        residual = baseline["residual"]
        abs_residual = np.abs(residual)

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

        write_csv(OUTDIR / "design_fit_comparison_v1_6.csv", design_fit_rows)

        edge_counts = {
            "rows_total": int(X.shape[0]),
            "spine_38_41_43_rows": int(sum(1 for r in row_rows if r["contains_38_41_43_spine"])),
            "bridge_42_46_rows": int(sum(1 for r in row_rows if r["bridges_param42_param46"])),
            "touch_param42_rows": int(sum(1 for r in row_rows if r["touches_param42"])),
            "touch_param46_rows": int(sum(1 for r in row_rows if r["touches_param46_H0_like"])),
        }

        summary = {
            "test_name": "TAIRID Table2 F160W faint-tail host/field concentration audit v1.6",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Host/field concentration audit only. Not proof of TAIRID, not H0 resolution, "
                "not BAO, not Planck, and not a full cosmology model."
            ),
            "final_status": final_status,
            "readiness_score_0_to_10": readiness_score,
            "next_wall": next_wall,
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
            "faint_tail_meta": tail_meta,
            "full_faint_tail_original_design_audit": original_audit,
            "full_faint_tail_full_control_audit": full_audit,
            "faint_tail_permutation_summary": perm_summary,
            "top_hosts_in_faint_tail": host_component_rows[:30],
            "top_host_leave_one_out": sorted(
                host_leaveout_rows,
                key=lambda r: -r.get("delta_drop_fraction_from_full_faint_tail", 0.0),
            )[:30],
            "top_fields_in_faint_tail": field_component_rows[:40],
            "top_field_leave_one_out": sorted(
                field_leaveout_rows,
                key=lambda r: -r.get("delta_drop_fraction_from_full_faint_tail", 0.0),
            )[:40],
            "host_cumulative_removal": host_removal_rows,
            "field_cumulative_removal": field_removal_rows,
            "best_cases": best_cases,
            "code_context_hits_count": len(code_hits),
            "output_files": {
                "summary_json": str(OUTDIR / "table2_f160w_faint_tail_host_field_v1_6_summary.json"),
                "summary_txt": str(OUTDIR / "table2_f160w_faint_tail_host_field_v1_6_summary.txt"),
                "faint_tail_rows_csv": str(OUTDIR / "faint_tail_rows_v1_6.csv"),
                "host_component_csv": str(OUTDIR / "host_component_audit_v1_6.csv"),
                "host_leave_one_out_csv": str(OUTDIR / "host_leave_one_out_audit_v1_6.csv"),
                "field_component_csv": str(OUTDIR / "field_component_audit_v1_6.csv"),
                "field_leave_one_out_csv": str(OUTDIR / "field_leave_one_out_audit_v1_6.csv"),
                "permutation_summary_json": str(OUTDIR / "faint_tail_permutation_summary_v1_6.json"),
                "plots": [
                    str(OUTDIR / "faint_tail_component_pressure_by_host_v1_6.png"),
                    str(OUTDIR / "faint_tail_component_pressure_by_field_v1_6.png"),
                    str(OUTDIR / "leave_one_host_out_drop_v1_6.png"),
                    str(OUTDIR / "leave_one_field_out_drop_v1_6.png"),
                ],
            },
            "interpretation": {
                "distributed_condition": (
                    "Faint-tail signal survives top host and top field removal; pressure is spread across many hosts/fields."
                ),
                "host_artifact_condition": (
                    "Faint-tail signal collapses when one host or a few hosts are removed."
                ),
                "field_artifact_condition": (
                    "Faint-tail signal collapses when one field or a few fields are removed."
                ),
                "truth_boundary": (
                    "This test cannot prove TAIRID. It only checks whether the v1.5 faint-tail F160W residual localization is distributed or local."
                ),
            },
        }

        write_json(OUTDIR / "table2_f160w_faint_tail_host_field_v1_6_summary.json", summary)

        with open(OUTDIR / "table2_f160w_faint_tail_host_field_v1_6_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Table2 F160W faint-tail host/field concentration audit v1.6\n\n")
            f.write("Boundary: host/field concentration audit only. Not proof. Not H0 resolution.\n\n")
            f.write(f"Final status: {final_status}\n")
            f.write(f"Readiness score: {readiness_score}/10\n")
            f.write(f"Next wall: {next_wall}\n\n")
            f.write("Faint-tail meta:\n")
            f.write(json.dumps(tail_meta, indent=2, default=json_default) + "\n\n")
            f.write("Full faint-tail full-control audit:\n")
            f.write(json.dumps(full_audit, indent=2, default=json_default) + "\n\n")
            f.write("Permutation summary:\n")
            f.write(json.dumps(perm_summary, indent=2, default=json_default) + "\n\n")
            f.write("Top host components:\n")
            f.write(json.dumps(host_component_rows[:20], indent=2, default=json_default) + "\n\n")
            f.write("Top host leave-one-out drops:\n")
            f.write(json.dumps(summary["top_host_leave_one_out"][:20], indent=2, default=json_default) + "\n\n")
            f.write("Top field components:\n")
            f.write(json.dumps(field_component_rows[:25], indent=2, default=json_default) + "\n\n")
            f.write("Top field leave-one-out drops:\n")
            f.write(json.dumps(summary["top_field_leave_one_out"][:25], indent=2, default=json_default) + "\n\n")
            f.write("Best cases:\n")
            f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 resolution.\n")
            f.write("- This only checks whether the faint-tail F160W residual localization is distributed or local.\n")

        print("")
        print("TAIRID Table2 F160W faint-tail host/field concentration audit v1.6 complete.")
        print("Created:")
        print("  tairid_table2_f160w_faint_tail_host_field_v1_6_outputs/table2_f160w_faint_tail_host_field_v1_6_summary.json")
        print("  tairid_table2_f160w_faint_tail_host_field_v1_6_outputs/table2_f160w_faint_tail_host_field_v1_6_summary.txt")
        print("  tairid_table2_f160w_faint_tail_host_field_v1_6_outputs/host_component_audit_v1_6.csv")
        print("  tairid_table2_f160w_faint_tail_host_field_v1_6_outputs/field_component_audit_v1_6.csv")
        print("")
        print(f"Final status: {final_status}")
        print(f"Readiness score: {readiness_score}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Table2 F160W faint-tail host/field concentration audit v1.6",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "v1_6_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "repair_summary": repair_summary,
        }
        write_json(OUTDIR / "table2_f160w_faint_tail_host_field_v1_6_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
