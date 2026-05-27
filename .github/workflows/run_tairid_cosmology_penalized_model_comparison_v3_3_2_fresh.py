#!/usr/bin/env python3
"""
TAIRID Cosmology Penalized Model Comparison v3.3.2 Fresh

This is a fresh-file retry after the v3.3.1 workflow failed before running
because the script file was not present in the repository.

Purpose:
Compare three metric-preserving SN distance-surface prototypes on Pantheon+SH0ES:

1. flat-LCDM shape + free intercept
2. simple wCDM shape + free intercept
3. flat-LCDM shape + frozen TAIRID gate basis + free intercept + one amplitude

Truth boundary:
This does not validate TAIRID, prove H0 correction, prove new physics, or show
standard cosmology is wrong. It only asks whether the frozen gate earns follow-up
after AIC/BIC penalties. If it does not, stop using it as an SN distance
correction term.
"""

from __future__ import annotations

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


OUTDIR = Path("tairid_cosmology_penalized_model_comparison_v3_3_2_fresh_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

C_LIGHT = 299792.458
H0_SHAPE = 70.0

PANTHEON_REPO = "PantheonPlusSH0ES/DataRelease"
PANTHEON_BRANCHES = ["main", "master"]
PANTHEON_DISTANCE_PATH = "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
PANTHEON_COV_PATH = "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"

BAO_REPO = "CobayaSampler/bao_data"
BAO_BRANCHES = ["master", "main"]
BAO_TARGETS = [
    "desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_mean.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt",
    "desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt",
    "desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt",
    "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt",
    "desi_2024_gaussian_bao_Lya_GCcomb_mean.txt",
]

FROZEN_GATE = {
    "name": "frozen_joint_gate_v0_1",
    "A": 0.6580586049,
    "zt": 0.2224541370,
    "p": 2.0,
    "boundary": "Frozen basis only. Not refit. Not proof. Not H0 correction.",
}

CLAIMS = {
    "test_name": "TAIRID Cosmology Penalized Model Comparison v3.3.2 Fresh",
    "primary_question": "Does the frozen TAIRID gate basis survive AIC/BIC penalties on the SN distance surface?",
    "truth_boundary": "Penalized SN-surface prototype only; no validation, no H0 correction, no new physics.",
}


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
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


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:180]


def raw_url(repo: str, branch: str, repo_path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{repo_path}"


def fetch_text(repo: str, branches: list[str], repo_path: str, timeout: int = 120) -> dict:
    errors = []

    for branch in branches:
        url = raw_url(repo, branch, repo_path)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "TAIRID-v3.3.2-penalized-model-comparison"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()

            text = data.decode("utf-8", errors="replace")

            if text.lower().lstrip().startswith("<!doctype html") or "404: not found" in text.lower()[:200]:
                errors.append({"branch": branch, "url": url, "error": "html_or_not_found_payload"})
                continue

            return {
                "status": "downloaded",
                "branch": branch,
                "url": url,
                "bytes": len(data),
                "text": text,
                "errors": errors,
            }

        except Exception as exc:
            errors.append({"branch": branch, "url": url, "error": repr(exc)})

    return {
        "status": "failed",
        "branch": None,
        "url": None,
        "bytes": 0,
        "text": "",
        "errors": errors,
    }


def to_float(value):
    try:
        out = float(str(value).replace("D", "E"))
    except Exception:
        return None
    return out if math.isfinite(out) else None


def parse_distance_table(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"columns": [], "rows": []}

    columns = lines[0].split()
    float_cols = {
        "zHD",
        "zHDERR",
        "zCMB",
        "zCMBERR",
        "zHEL",
        "zHELERR",
        "m_b_corr",
        "m_b_corr_err_DIAG",
        "MU_SH0ES",
        "MU_SH0ES_ERR_DIAG",
        "CEPH_DIST",
        "IS_CALIBRATOR",
        "USED_IN_SH0ES_HF",
        "HOST_LOGMASS",
    }

    rows = []
    for idx, line in enumerate(lines[1:]):
        parts = line.split()
        if len(parts) < len(columns):
            continue

        row = dict(zip(columns, parts[: len(columns)]))
        row["row_index"] = idx

        for key in float_cols:
            if key in row:
                val = to_float(row[key])
                if val is not None:
                    row[key] = val

        rows.append(row)

    return {"columns": columns, "rows": rows}


def parse_cov_shape(text: str) -> dict:
    tokens = text.split()
    first = None

    if tokens:
        try:
            first = int(float(tokens[0]))
        except Exception:
            first = None

    expected = first * first if first is not None else None

    return {
        "dimension": first,
        "payload_count_after_first": len(tokens) - 1 if tokens else 0,
        "expected_square": expected,
        "shape_valid": expected is not None and len(tokens) - 1 == expected,
    }


def select_sn_rows(rows: list[dict]) -> list[dict]:
    selected = []

    for row in rows:
        z = row.get("zHD")
        mu = row.get("MU_SH0ES")
        sig = row.get("MU_SH0ES_ERR_DIAG")

        if z is None or mu is None or sig is None:
            continue

        if not (0.01 < float(z) < 2.5 and float(sig) > 0):
            continue

        if int(float(row.get("IS_CALIBRATOR", 0))) == 1:
            continue

        selected.append(row)

    return selected


def e_model(z, om, w=-1.0):
    z = np.asarray(z, dtype=float)
    mat = om * (1.0 + z) ** 3
    de = (1.0 - om) * (1.0 + z) ** (3.0 * (1.0 + w))
    return np.sqrt(np.maximum(mat + de, 1e-12))


def cumulative_trapezoid(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if len(y) < 2:
        return np.zeros_like(y)

    area = (x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5
    return np.concatenate([[0.0], np.cumsum(area)])


def distance_modulus(z_values, om, w=-1.0):
    z_values = np.asarray(z_values, dtype=float)
    zmax = max(0.02, float(np.max(z_values)) * 1.001)

    grid = np.linspace(0.0, zmax, 5000)
    integral = cumulative_trapezoid(1.0 / e_model(grid, om, w), grid)

    dc = C_LIGHT / H0_SHAPE * np.interp(z_values, grid, integral)
    dl = (1.0 + z_values) * np.maximum(dc, 1e-12)

    return 5.0 * np.log10(dl) + 25.0


def frozen_gate(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-FROZEN_GATE["A"] * (z / (z + FROZEN_GATE["zt"])) ** FROZEN_GATE["p"])


def gate_basis(z, weights):
    raw = -np.log(np.maximum(frozen_gate(z), 1e-12))
    mean = np.average(raw, weights=weights)
    centered = raw - mean
    var = np.average(centered * centered, weights=weights)
    return centered / math.sqrt(max(var, 1e-12))


def weighted_offset_fit(target_offset, weights, basis=None):
    y = np.asarray(target_offset, dtype=float)
    w = np.asarray(weights, dtype=float)

    if basis is None:
        intercept = float(np.average(y, weights=w))
        predicted_offset = np.full_like(y, intercept)
        params = {"intercept": intercept, "gate_gamma": None}
    else:
        x = np.asarray(basis, dtype=float)
        X = np.column_stack([np.ones_like(y), x])
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        predicted_offset = X @ coef
        params = {"intercept": float(coef[0]), "gate_gamma": float(coef[1])}

    residual = y - predicted_offset
    chi2 = float(np.sum(w * residual * residual))

    return params, predicted_offset, residual, chi2


def fit_grid(z, mu, sigma, kind):
    weights = 1.0 / np.maximum(sigma, 1e-6) ** 2
    phi = gate_basis(z, weights)

    grid_rows = []
    best = None

    if kind in {"lcdm", "tairid"}:
        for om in np.linspace(0.10, 0.50, 81):
            shape = distance_modulus(z, om, -1.0)
            basis = phi if kind == "tairid" else None
            params, offset, resid, chi2 = weighted_offset_fit(mu - shape, weights, basis=basis)

            row = {
                "model": "flat_lcdm" if kind == "lcdm" else "tairid_fixed_gate_pressure_basis",
                "Omega_m": float(om),
                "w": -1.0,
                "intercept": params["intercept"],
                "gate_gamma": params["gate_gamma"],
                "chi2": chi2,
                "k_params": 2 if kind == "lcdm" else 3,
                "n": int(len(z)),
            }

            grid_rows.append(row)

            if best is None or chi2 < best["chi2"]:
                best = dict(row)
                best["_shape"] = shape
                best["_offset"] = offset
                best["_residual"] = resid

    elif kind == "wcdm":
        for om in np.linspace(0.10, 0.50, 51):
            for ww in np.linspace(-1.40, -0.60, 51):
                shape = distance_modulus(z, om, ww)
                params, offset, resid, chi2 = weighted_offset_fit(mu - shape, weights, basis=None)

                row = {
                    "model": "wcdm",
                    "Omega_m": float(om),
                    "w": float(ww),
                    "intercept": params["intercept"],
                    "gate_gamma": None,
                    "chi2": chi2,
                    "k_params": 3,
                    "n": int(len(z)),
                }

                grid_rows.append(row)

                if best is None or chi2 < best["chi2"]:
                    best = dict(row)
                    best["_shape"] = shape
                    best["_offset"] = offset
                    best["_residual"] = resid
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    n = best["n"]
    k = best["k_params"]

    best["aic"] = float(best["chi2"] + 2.0 * k)
    best["bic"] = float(best["chi2"] + k * math.log(n))
    best["chi2_per_dof"] = float(best["chi2"] / max(n - k, 1))
    best["residual_std"] = float(np.std(best["_residual"]))
    best["residual_max_abs"] = float(np.max(np.abs(best["_residual"])))

    public_best = {key: value for key, value in best.items() if not key.startswith("_")}

    return public_best, grid_rows, best


def compare_models(models):
    best_aic = min(m["aic"] for m in models)
    best_bic = min(m["bic"] for m in models)

    out = []
    for model in models:
        row = dict(model)
        row["delta_aic_vs_best"] = float(row["aic"] - best_aic)
        row["delta_bic_vs_best"] = float(row["bic"] - best_bic)
        out.append(row)

    return sorted(out, key=lambda r: (r["bic"], r["aic"]))


def residual_table(sn_rows, z, mu, weights, best_objects):
    phi = gate_basis(z, weights)
    residuals = {}

    for name, obj in best_objects.items():
        residuals[name] = mu - (obj["_shape"] + obj["_offset"])

    rows = []
    for i, row in enumerate(sn_rows):
        rows.append(
            {
                "row_index": row.get("row_index"),
                "CID": row.get("CID"),
                "zHD": float(z[i]),
                "MU_SH0ES": float(mu[i]),
                "MU_SH0ES_ERR_DIAG": float(row["MU_SH0ES_ERR_DIAG"]),
                "gate_basis_standardized": float(phi[i]),
                "lcdm_residual": float(residuals["lcdm"][i]),
                "wcdm_residual": float(residuals["wcdm"][i]),
                "tairid_fixed_gate_residual": float(residuals["tairid"][i]),
                "truth_boundary": "Residual comparison only; not H0 correction or validation.",
            }
        )

    return rows


def fetch_bao_availability():
    rows = []

    for target in BAO_TARGETS:
        fetched = fetch_text(BAO_REPO, BAO_BRANCHES, target, timeout=60)

        rows.append(
            {
                "file": target,
                "status": fetched["status"],
                "branch": fetched["branch"],
                "url": fetched["url"],
                "bytes": fetched["bytes"],
                "errors": json.dumps(fetched["errors"], default=json_default),
                "truth_boundary": "Availability check only; BAO is not fitted in v3.3.2.",
            }
        )

        if fetched["status"] == "downloaded":
            (DOWNLOAD_DIR / safe_name(target)).write_text(fetched["text"], encoding="utf-8")

    return rows


def decide(comparison, bao_rows, cov_shape):
    by_model = {row["model"]: row for row in comparison}

    lcdm = by_model["flat_lcdm"]
    wcdm = by_model["wcdm"]
    tairid = by_model["tairid_fixed_gate_pressure_basis"]

    delta_bic_tairid_minus_lcdm = tairid["bic"] - lcdm["bic"]
    delta_aic_tairid_minus_lcdm = tairid["aic"] - lcdm["aic"]
    delta_bic_tairid_minus_wcdm = tairid["bic"] - wcdm["bic"]

    tairid_clears_lcdm = delta_bic_tairid_minus_lcdm <= -6.0
    tairid_competes_wcdm = delta_bic_tairid_minus_wcdm <= 2.0

    bao_available = sum(1 for r in bao_rows if r["status"] == "downloaded") >= 5
    cov_ok = cov_shape.get("shape_valid") is True

    gates = [
        {
            "gate": "G1_sn_surface_fit_completed",
            "passed": all(m.get("n", 0) >= 1000 for m in comparison),
            "evidence": {m["model"]: m.get("n") for m in comparison},
        },
        {
            "gate": "G2_pantheon_covariance_shape_available",
            "passed": cov_ok,
            "evidence": cov_shape,
        },
        {
            "gate": "G3_bao_surface_lock_available",
            "passed": bao_available,
            "evidence": {
                "downloaded": sum(1 for r in bao_rows if r["status"] == "downloaded"),
                "target_count": len(bao_rows),
            },
        },
        {
            "gate": "G4_tairid_fixed_gate_survives_lcdm_penalty",
            "passed": tairid_clears_lcdm,
            "evidence": {
                "delta_bic_tairid_minus_lcdm": delta_bic_tairid_minus_lcdm,
                "delta_aic_tairid_minus_lcdm": delta_aic_tairid_minus_lcdm,
                "threshold": "BIC must be <= -6",
            },
        },
        {
            "gate": "G5_tairid_fixed_gate_competitive_with_wcdm",
            "passed": tairid_competes_wcdm,
            "evidence": {
                "delta_bic_tairid_minus_wcdm": delta_bic_tairid_minus_wcdm,
                "threshold": "BIC must be <= 2",
            },
        },
        {
            "gate": "G6_no_validation_claim_allowed",
            "passed": True,
            "evidence": {
                "validation_claim": False,
                "h0_correction_claim": False,
                "new_physics_claim": False,
            },
        },
    ]

    failed = [gate["gate"] for gate in gates if not gate["passed"]]

    if not failed:
        status = "tairid_fixed_gate_survives_penalized_sn_prototype_not_validation"
        readiness = 9
        next_wall = "Add BAO/CMB likelihood locks before any larger claim."
    elif "G4_tairid_fixed_gate_survives_lcdm_penalty" in failed:
        status = "tairid_fixed_gate_not_supported_as_sn_distance_pressure_term"
        readiness = 8
        next_wall = "Do not use the frozen gate as an SN distance correction. Keep it diagnostic or move surfaces."
    elif len(failed) <= 2:
        status = "penalized_model_comparison_partial_with_cautions"
        readiness = 7
        next_wall = "Review failed gates before expanding this lane."
    else:
        status = "penalized_model_comparison_not_ready"
        readiness = 5
        next_wall = "Core model-comparison surfaces failed."

    return {
        "final_status": status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "failed_gates": failed,
        "gates": gates,
        "model_ranking_by_bic": comparison,
        "truth_boundary": CLAIMS["truth_boundary"],
    }


def make_plots(comparison, residuals):
    try:
        labels = [m["model"] for m in comparison]
        deltas = [m["delta_bic_vs_best"] for m in comparison]
        x = np.arange(len(labels))

        plt.figure(figsize=(9, 5))
        plt.bar(x, deltas)
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("Delta BIC vs best")
        plt.title("v3.3.2 penalized model comparison")
        plt.tight_layout()
        plt.savefig(OUTDIR / "model_delta_bic_v3_3_2_fresh.png", dpi=160)
        plt.close()

        z = np.asarray([r["zHD"] for r in residuals], dtype=float)
        lcdm = np.asarray([r["lcdm_residual"] for r in residuals], dtype=float)
        tairid = np.asarray([r["tairid_fixed_gate_residual"] for r in residuals], dtype=float)
        gate = np.asarray([r["gate_basis_standardized"] for r in residuals], dtype=float)

        plt.figure(figsize=(9, 5))
        plt.scatter(z, lcdm, s=6, alpha=0.35, label="LCDM residual")
        plt.scatter(z, tairid, s=6, alpha=0.35, label="TAIRID fixed-gate residual")
        plt.xscale("log")
        plt.xlabel("redshift z")
        plt.ylabel("SN residual")
        plt.title("v3.3.2 SN residual comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / "sn_residual_model_comparison_v3_3_2_fresh.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter(gate, lcdm, s=6, alpha=0.45)
        plt.xlabel("standardized frozen gate basis")
        plt.ylabel("LCDM residual")
        plt.title("v3.3.2 gate basis vs LCDM residual")
        plt.tight_layout()
        plt.savefig(OUTDIR / "gate_basis_vs_lcdm_residual_v3_3_2_fresh.png", dpi=160)
        plt.close()

    except Exception as exc:
        write_json(
            OUTDIR / "plot_error_v3_3_2_fresh.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )


def write_handoff(decision):
    lines = [
        "# TAIRID v3.3.2 Fresh Handoff",
        "",
        "## Current status",
        "",
        f"- Final status: `{decision['final_status']}`",
        f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`",
        f"- Next wall: {decision['next_wall']}",
        "",
        "## Model ranking by BIC",
        "",
    ]

    for model in decision["model_ranking_by_bic"]:
        lines.append(
            f"- `{model['model']}`: BIC `{model['bic']:.6f}`, "
            f"ΔBIC `{model['delta_bic_vs_best']:.6f}`, "
            f"chi2 `{model['chi2']:.6f}`, params `{model['k_params']}`"
        )

    lines.extend(
        [
            "",
            "## Truth boundary",
            "",
            "- v3.3.2 is a penalized SN-surface prototype only.",
            "- It does not validate TAIRID.",
            "- It does not prove H0 correction.",
            "- It does not prove new physics.",
            "- It does not disprove standard cosmology.",
            "",
            "## Next test",
            "",
        ]
    )

    if decision["final_status"] == "tairid_fixed_gate_survives_penalized_sn_prototype_not_validation":
        lines.append("Build v3.4 — BAO/CMB-Locked Follow-Up. Add ruler constraints before any larger claim.")
    elif decision["final_status"] == "tairid_fixed_gate_not_supported_as_sn_distance_pressure_term":
        lines.append("Stop pushing the frozen gate as an SN distance correction. Pivot to BAO/CMB ruler pressure, growth/geometry split, or derivation-level rework.")
    else:
        lines.append("Repair failed gates before deciding the next surface.")

    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID Cosmology Penalized Model Comparison v3.3.2 Fresh starting.")
    print("Boundary: penalized SN-surface prototype only; no validation, no H0 claim, no new physics claim.")

    write_json(OUTDIR / "claims_v3_3_2_fresh.json", CLAIMS)
    write_json(OUTDIR / "frozen_gate_v0_1_penalty_test_v3_3_2_fresh.json", FROZEN_GATE)

    try:
        distance_fetch = fetch_text(PANTHEON_REPO, PANTHEON_BRANCHES, PANTHEON_DISTANCE_PATH)
        cov_fetch = fetch_text(PANTHEON_REPO, PANTHEON_BRANCHES, PANTHEON_COV_PATH, timeout=180)

        write_csv(
            OUTDIR / "download_ledger_v3_3_2_fresh.csv",
            [
                {
                    "path": PANTHEON_DISTANCE_PATH,
                    "status": distance_fetch["status"],
                    "branch": distance_fetch["branch"],
                    "url": distance_fetch["url"],
                    "bytes": distance_fetch["bytes"],
                    "errors": json.dumps(distance_fetch["errors"], default=json_default),
                },
                {
                    "path": PANTHEON_COV_PATH,
                    "status": cov_fetch["status"],
                    "branch": cov_fetch["branch"],
                    "url": cov_fetch["url"],
                    "bytes": cov_fetch["bytes"],
                    "errors": json.dumps(cov_fetch["errors"], default=json_default),
                },
            ],
        )

        if distance_fetch["status"] == "downloaded":
            (DOWNLOAD_DIR / safe_name(PANTHEON_DISTANCE_PATH)).write_text(distance_fetch["text"], encoding="utf-8")

        if cov_fetch["status"] == "downloaded":
            (DOWNLOAD_DIR / safe_name(PANTHEON_COV_PATH)).write_text(cov_fetch["text"], encoding="utf-8")

        distance = parse_distance_table(distance_fetch["text"])
        cov_shape = parse_cov_shape(cov_fetch["text"])
        write_json(OUTDIR / "pantheon_covariance_shape_v3_3_2_fresh.json", cov_shape)

        sn_rows = select_sn_rows(distance["rows"])

        if len(sn_rows) < 1000:
            raise RuntimeError(f"Not enough SN rows selected: {len(sn_rows)}")

        z = np.asarray([float(r["zHD"]) for r in sn_rows], dtype=float)
        mu = np.asarray([float(r["MU_SH0ES"]) for r in sn_rows], dtype=float)
        sigma = np.asarray([float(r["MU_SH0ES_ERR_DIAG"]) for r in sn_rows], dtype=float)
        weights = 1.0 / np.maximum(sigma, 1e-6) ** 2

        lcdm_best, lcdm_grid, lcdm_obj = fit_grid(z, mu, sigma, "lcdm")
        wcdm_best, wcdm_grid, wcdm_obj = fit_grid(z, mu, sigma, "wcdm")
        tairid_best, tairid_grid, tairid_obj = fit_grid(z, mu, sigma, "tairid")

        comparison = compare_models([lcdm_best, wcdm_best, tairid_best])

        write_csv(OUTDIR / "model_comparison_v3_3_2_fresh.csv", comparison)
        write_csv(OUTDIR / "lcdm_grid_scores_v3_3_2_fresh.csv", lcdm_grid)
        write_csv(OUTDIR / "wcdm_grid_scores_v3_3_2_fresh.csv", wcdm_grid)
        write_csv(OUTDIR / "tairid_fixed_gate_grid_scores_v3_3_2_fresh.csv", tairid_grid)

        residuals = residual_table(
            sn_rows,
            z,
            mu,
            weights,
            {"lcdm": lcdm_obj, "wcdm": wcdm_obj, "tairid": tairid_obj},
        )
        write_csv(OUTDIR / "best_model_residuals_v3_3_2_fresh.csv", residuals)

        bao_rows = fetch_bao_availability()
        write_csv(OUTDIR / "bao_availability_v3_3_2_fresh.csv", bao_rows)

        decision = decide(comparison, bao_rows, cov_shape)
        write_json(OUTDIR / "decision_v3_3_2_fresh.json", decision)

        make_plots(comparison, residuals)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v3_3_2_fresh.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v3_3_2_fresh.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": CLAIMS["test_name"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": CLAIMS["truth_boundary"],
            "sn_surface": {
                "selected_noncalibrator_count": len(sn_rows),
                "z_min": float(np.min(z)),
                "z_max": float(np.max(z)),
                "sigma_source": "MU_SH0ES_ERR_DIAG only; full covariance parsed for shape availability but not used in grid prototype.",
            },
            "model_comparison": comparison,
            "covariance_shape": cov_shape,
            "bao_availability": {
                "target_count": len(bao_rows),
                "downloaded_count": sum(1 for r in bao_rows if r["status"] == "downloaded"),
            },
            "decision": decision,
            "output_files": {
                "summary_json": str(OUTDIR / "cosmology_penalized_model_comparison_v3_3_2_fresh_summary.json"),
                "summary_txt": str(OUTDIR / "cosmology_penalized_model_comparison_v3_3_2_fresh_summary.txt"),
                "model_comparison_csv": str(OUTDIR / "model_comparison_v3_3_2_fresh.csv"),
                "residuals_csv": str(OUTDIR / "best_model_residuals_v3_3_2_fresh.csv"),
                "decision_json": str(OUTDIR / "decision_v3_3_2_fresh.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v3_3_2_fresh.md"),
            },
            "interpretation": {
                "what_success_means": "The frozen TAIRID gate earned follow-up as a fixed SN pressure basis after model penalties.",
                "what_failure_means": "The frozen gate should not be used as an SN distance correction term.",
                "truth_boundary": CLAIMS["truth_boundary"],
            },
        }

        write_json(OUTDIR / "cosmology_penalized_model_comparison_v3_3_2_fresh_summary.json", summary)

        with open(OUTDIR / "cosmology_penalized_model_comparison_v3_3_2_fresh_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Cosmology Penalized Model Comparison v3.3.2 Fresh\n\n")
            f.write("Boundary: penalized SN-surface prototype only. No validation. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Model comparison:\n")
            f.write(json.dumps(comparison, indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not disprove standard cosmology.\n")
            f.write("- This only tests whether the frozen gate survives SN-surface model penalties.\n")

        print("TAIRID Cosmology Penalized Model Comparison v3.3.2 Fresh complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": CLAIMS["test_name"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "cosmology_penalized_model_comparison_v3_3_2_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS["truth_boundary"],
        }
        write_json(OUTDIR / "cosmology_penalized_model_comparison_v3_3_2_fresh_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
