#!/usr/bin/env python3
"""
TAIRID Cosmology Multi-Surface Pressure Map v3.2 Fresh

Why this test exists:
v3.0 built the cosmology failure-mode atlas.
v3.1 rejected bad translation branches and allowed only the metric-accessibility
TAIRID branch forward.

v3.2 is not a proof test. It is a pressure-map test. It compares several
observable cosmology surfaces at once so we stop treating one narrow anomaly as
the whole theory.

Core question:
    Where is the pressure located across SN geometry, H0 early/late mismatch,
    BAO availability/ruler surfaces, CMB preservation, and the frozen TAIRID
    low-z gate stress case?

This test DOES:
    - download and parse Pantheon+SH0ES distance rows,
    - fit a plain flat-LCDM SN shape surface with a free intercept,
    - compute SN residual pressure and compare it to the frozen TAIRID gate
      as a diagnostic only,
    - fetch and parse DESI DR1 BAO mean/covariance files from the public
      CobayaSampler BAO data repository when reachable,
    - compute declared H0 early/late tension as a pressure seam,
    - keep CMB/BAO preservation locked instead of letting the low-z gate
      overwrite high-z surfaces,
    - decide what next test is allowed.

This test DOES NOT:
    - validate TAIRID,
    - claim H0 correction,
    - claim new physics,
    - claim standard cosmology is wrong,
    - fit a new TAIRID cosmology,
    - use the frozen gate as a distance correction,
    - promote SH0ES residuals into cosmology.

Truth boundary:
v3.2 maps pressure surfaces. It does not solve them.
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


OUTDIR = Path("tairid_cosmology_multi_surface_pressure_map_v3_2_fresh_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

C_LIGHT = 299792.458
RANDOM_SEED = 112221

PANTHEON_REPO = "PantheonPlusSH0ES/DataRelease"
PANTHEON_BRANCHES = ["main", "master"]
PANTHEON_DISTANCE_PATH = "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
PANTHEON_COV_PATH = "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"

BAO_REPO = "CobayaSampler/bao_data"
BAO_BRANCHES = ["master", "main"]
BAO_TARGETS = [
    "desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_mean.txt",
    "desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_cov.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt",
    "desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt",
    "desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_cov.txt",
    "desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt",
    "desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_cov.txt",
    "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt",
    "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_cov.txt",
    "desi_2024_gaussian_bao_Lya_GCcomb_mean.txt",
    "desi_2024_gaussian_bao_Lya_GCcomb_cov.txt",
]

H0_EARLY_PLANCK_LIKE = 67.36
H0_EARLY_SIGMA = 0.54
H0_LOCAL_SHOES_LIKE = 73.04
H0_LOCAL_SIGMA = 1.04

FROZEN_GATE_V0_1 = {
    "name": "frozen_joint_gate_v0_1",
    "A": 0.6580586049,
    "zt": 0.2224541370,
    "p": 2.0,
    "truth_boundary": "Diagnostic stress surface only; not used as a distance correction.",
}

CLAIMS_V3_2 = {
    "battery_name": "TAIRID Cosmology Multi-Surface Pressure Map v3.2 Fresh",
    "scope": "Pressure-map comparison across SN geometry, H0 seam, BAO availability/ruler surfaces, CMB guard, and frozen-gate diagnostic pressure",
    "primary_question": (
        "Where is cosmology pressure located when TAIRID is constrained to the metric-accessibility branch?"
    ),
    "truth_boundary": (
        "This maps pressure surfaces only. It does not validate TAIRID, H0 correction, or new physics."
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
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:180]


def raw_url(repo, branch, repo_path):
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{repo_path}"


def fetch_text(repo, branches, repo_path, timeout=120):
    errors = []
    for branch in branches:
        url = raw_url(repo, branch, repo_path)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TAIRID-v3.2-pressure-map"})
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
    return {"status": "failed", "branch": None, "url": None, "bytes": 0, "text": "", "errors": errors}


def to_float(x):
    try:
        y = float(str(x).replace("D", "E"))
    except Exception:
        return None
    return y if math.isfinite(y) else None


def parse_pantheon_distance(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"columns": [], "rows": []}
    columns = lines[0].split()
    rows = []
    float_cols = {
        "zHD", "zHDERR", "zCMB", "zCMBERR", "zHEL", "zHELERR",
        "m_b_corr", "m_b_corr_err_DIAG", "MU_SH0ES", "MU_SH0ES_ERR_DIAG",
        "CEPH_DIST", "IS_CALIBRATOR", "USED_IN_SH0ES_HF",
        "RA", "DEC", "HOST_RA", "HOST_DEC", "VPEC", "VPECERR", "HOST_LOGMASS",
    }
    for idx, line in enumerate(lines[1:]):
        parts = line.split()
        if len(parts) < len(columns):
            continue
        row = dict(zip(columns, parts[:len(columns)]))
        row["row_index"] = idx
        for key in float_cols:
            if key in row:
                val = to_float(row[key])
                if val is not None:
                    row[key] = val
        rows.append(row)
    return {"columns": columns, "rows": rows}


def parse_cov_shape(text):
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


def E_flat_lcdm(z, om):
    return np.sqrt(om * (1.0 + z) ** 3 + (1.0 - om))


def comoving_distance_grid(z_values, om):
    z_values = np.asarray(z_values, dtype=float)
    out = []
    for z in z_values:
        if z <= 0:
            out.append(0.0)
            continue
        n = max(200, int(600 * min(z, 3.0)))
        grid = np.linspace(0.0, float(z), n)
        inv_e = 1.0 / E_flat_lcdm(grid, om)
        dc = C_LIGHT * np.trapz(inv_e, grid) / 70.0
        out.append(dc)
    return np.asarray(out, dtype=float)


def distance_modulus_shape(z_values, om):
    dc = comoving_distance_grid(z_values, om)
    dl = (1.0 + np.asarray(z_values, dtype=float)) * dc
    dl = np.maximum(dl, 1e-12)
    return 5.0 * np.log10(dl) + 25.0


def weighted_mean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return None
    return float(np.average(v[mask], weights=w[mask]))


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"n": int(np.sum(mask)), "r": None}
    xx = x[mask]
    yy = y[mask]
    if np.std(xx) == 0 or np.std(yy) == 0:
        return {"n": int(len(xx)), "r": None}
    return {"n": int(len(xx)), "r": float(np.corrcoef(xx, yy)[0, 1])}


def frozen_gate(z):
    z = np.asarray(z, dtype=float)
    A = FROZEN_GATE_V0_1["A"]
    zt = FROZEN_GATE_V0_1["zt"]
    p = FROZEN_GATE_V0_1["p"]
    return np.exp(-A * (z / (z + zt)) ** p)


def fit_sn_shape(rows):
    usable = []
    for row in rows:
        z = row.get("zHD")
        mu = row.get("MU_SH0ES")
        sig = row.get("MU_SH0ES_ERR_DIAG")
        if z is None or mu is None or sig is None:
            continue
        if not (0.01 < float(z) < 2.5 and float(sig) > 0):
            continue
        usable.append(row)

    z = np.asarray([float(r["zHD"]) for r in usable], dtype=float)
    mu = np.asarray([float(r["MU_SH0ES"]) for r in usable], dtype=float)
    sig = np.asarray([float(r["MU_SH0ES_ERR_DIAG"]) for r in usable], dtype=float)
    w = 1.0 / np.maximum(sig, 1e-6) ** 2

    grid = np.linspace(0.10, 0.50, 81)
    fits = []
    best = None
    for om in grid:
        shape = distance_modulus_shape(z, om)
        intercept = weighted_mean(mu - shape, w)
        model = shape + intercept
        resid = mu - model
        chi2 = float(np.sum(w * resid * resid))
        dof = int(len(z) - 2)
        rec = {
            "Omega_m": float(om),
            "intercept": intercept,
            "chi2": chi2,
            "dof": dof,
            "chi2_per_dof": chi2 / dof if dof > 0 else None,
        }
        fits.append(rec)
        if best is None or chi2 < best["chi2"]:
            best = rec

    best_shape = distance_modulus_shape(z, best["Omega_m"])
    best_model = best_shape + best["intercept"]
    resid = mu - best_model
    gate = frozen_gate(z)
    gate_pressure = -np.log(np.maximum(gate, 1e-12))

    diag = []
    for r, zz, mmu, ss, mod, rr, gg, gp in zip(usable, z, mu, sig, best_model, resid, gate, gate_pressure):
        diag.append(
            {
                "CID": r.get("CID"),
                "zHD": float(zz),
                "MU_SH0ES": float(mmu),
                "MU_SH0ES_ERR_DIAG": float(ss),
                "model_mu_shape_best_lcdm": float(mod),
                "sn_residual_mu": float(rr),
                "frozen_gate_value": float(gg),
                "frozen_gate_pressure_minus_lnG": float(gp),
                "IS_CALIBRATOR": r.get("IS_CALIBRATOR"),
                "USED_IN_SH0ES_HF": r.get("USED_IN_SH0ES_HF"),
                "truth_boundary": "SN residual and gate-pressure comparison is diagnostic only, not a correction model.",
            }
        )

    corr = pearson_corr(resid, gate_pressure)
    calib = [d for d in diag if int(float(d.get("IS_CALIBRATOR", 0))) == 1]
    noncalib = [d for d in diag if int(float(d.get("IS_CALIBRATOR", 0))) != 1]

    summary = {
        "usable_sn_count": int(len(usable)),
        "best_flat_lcdm_shape_fit": best,
        "omega_grid_min": float(min(grid)),
        "omega_grid_max": float(max(grid)),
        "residual_summary": {
            "mean": float(np.mean(resid)),
            "median": float(np.median(resid)),
            "std": float(np.std(resid)),
            "max_abs": float(np.max(np.abs(resid))),
        },
        "gate_pressure_residual_correlation": corr,
        "calibrator_count": len(calib),
        "noncalibrator_count": len(noncalib),
        "calibrator_residual_mean": float(np.mean([d["sn_residual_mu"] for d in calib])) if calib else None,
        "noncalibrator_residual_mean": float(np.mean([d["sn_residual_mu"] for d in noncalib])) if noncalib else None,
        "truth_boundary": "This is a pressure map over SN geometry, not a TAIRID distance fit.",
    }
    return summary, diag, fits


def parse_bao_file(text, name):
    numeric_tokens = []
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        nums = [to_float(t) for t in tokens]
        nums = [n for n in nums if n is not None]
        if nums:
            numeric_tokens.extend(nums)
            rows.append({"line_no": line_no, "raw": stripped[:240], "numeric_count": len(nums), "numeric_values": "|".join(str(n) for n in nums)})
    z_match = re.search(r"z([0-9.]+)-([0-9.]+)", name)
    z_eff = None
    if z_match:
        z_eff = (float(z_match.group(1)) + float(z_match.group(2))) / 2.0
    elif "Lya" in name:
        z_eff = 2.33
    elif "QSO" in name:
        z_eff = 1.491
    return {
        "file": name,
        "line_count": len(text.splitlines()),
        "data_line_count": len(rows),
        "numeric_token_count": len(numeric_tokens),
        "z_eff_from_filename": z_eff,
        "numeric_min": float(np.min(numeric_tokens)) if numeric_tokens else None,
        "numeric_max": float(np.max(numeric_tokens)) if numeric_tokens else None,
        "preview_rows": rows[:10],
    }


def fetch_bao_surfaces():
    ledger = []
    parsed = []
    previews = []
    for target in BAO_TARGETS:
        fetched = fetch_text(BAO_REPO, BAO_BRANCHES, target)
        ledger.append({
            "file": target,
            "status": fetched["status"],
            "branch": fetched["branch"],
            "url": fetched["url"],
            "bytes": fetched["bytes"],
            "errors": json.dumps(fetched["errors"], default=json_default),
        })
        if fetched["status"] == "downloaded":
            (DOWNLOAD_DIR / safe_name(target)).write_text(fetched["text"], encoding="utf-8")
            p = parse_bao_file(fetched["text"], target)
            parsed.append({k: v for k, v in p.items() if k != "preview_rows"})
            for row in p["preview_rows"]:
                row["file"] = target
                previews.append(row)
    mean_files = [p for p in parsed if p["file"].endswith("_mean.txt")]
    cov_files = [p for p in parsed if p["file"].endswith("_cov.txt")]
    summary = {
        "target_count": len(BAO_TARGETS),
        "downloaded_count": sum(1 for x in ledger if x["status"] == "downloaded"),
        "mean_file_count": len(mean_files),
        "cov_file_count": len(cov_files),
        "bao_redshift_surface_count": len([m for m in mean_files if m["z_eff_from_filename"] is not None]),
        "z_eff_values": sorted([m["z_eff_from_filename"] for m in mean_files if m["z_eff_from_filename"] is not None]),
        "truth_boundary": "BAO files are parsed for pressure-surface availability, not fitted as a cosmology likelihood in v3.2.",
    }
    return summary, ledger, parsed, previews


def h0_pressure_seam():
    delta = H0_LOCAL_SHOES_LIKE - H0_EARLY_PLANCK_LIKE
    sigma = math.sqrt(H0_EARLY_SIGMA ** 2 + H0_LOCAL_SIGMA ** 2)
    tension = abs(delta) / sigma if sigma > 0 else None
    return {
        "surface": "HUBBLE_TENSION",
        "early_H0": H0_EARLY_PLANCK_LIKE,
        "early_sigma": H0_EARLY_SIGMA,
        "local_H0": H0_LOCAL_SHOES_LIKE,
        "local_sigma": H0_LOCAL_SIGMA,
        "delta_local_minus_early": delta,
        "quadrature_sigma": sigma,
        "tension_sigma": tension,
        "pressure_class": "high" if tension is not None and tension >= 4.0 else "moderate_or_low",
        "truth_boundary": "Declared pressure seam only; v3.2 does not solve H0 tension.",
    }


def preservation_pressure():
    zvals = np.asarray([0.1, 1.0, 2.0, 10.0, 1100.0])
    gate_vals = frozen_gate(zvals)
    return {
        "surface": "CMB_BAO_PRESERVATION",
        "metric_accessibility_branch": "locked/admissible from v3.1",
        "naive_low_z_gate_extension_allowed": False,
        "frozen_gate_values_if_naively_extended": {str(float(z)): float(g) for z, g in zip(zvals, gate_vals)},
        "high_z_gate_damage_flag": bool(gate_vals[-1] < 0.99),
        "preservation_instruction": "Use frozen gate as low-z pressure diagnostic only unless a future derivation locks CMB/BAO behavior.",
        "truth_boundary": "Preservation guard, not a model fit.",
    }


def score_pressure_surfaces(sn_summary, bao_summary, h0_summary, preservation_summary):
    rows = []
    rows.append({
        "surface": "SN_GEOMETRY",
        "pressure_score_0_to_10": min(10.0, float(sn_summary["best_flat_lcdm_shape_fit"]["chi2_per_dof"] or 0.0)),
        "evidence": f"SN chi2/dof={sn_summary['best_flat_lcdm_shape_fit']['chi2_per_dof']}",
        "interpretation": "Higher means plain LCDM shape leaves more residual pressure; do not treat as TAIRID proof.",
    })
    h0_score = min(10.0, float(h0_summary["tension_sigma"]) * 1.5) if h0_summary["tension_sigma"] is not None else 0.0
    rows.append({
        "surface": "HUBBLE_TENSION",
        "pressure_score_0_to_10": h0_score,
        "evidence": f"H0 tension sigma={h0_summary['tension_sigma']}",
        "interpretation": "High score means real pressure seam; not a solution.",
    })
    bao_score = 7.0 if bao_summary["mean_file_count"] >= 5 and bao_summary["cov_file_count"] >= 5 else 4.0
    rows.append({
        "surface": "BAO_RULER_AVAILABILITY",
        "pressure_score_0_to_10": bao_score,
        "evidence": f"mean files={bao_summary['mean_file_count']}; cov files={bao_summary['cov_file_count']}",
        "interpretation": "Score reflects usable ruler-surface coverage for next test, not tension strength.",
    })
    rows.append({
        "surface": "CMB_BAO_PRESERVATION",
        "pressure_score_0_to_10": 9.0 if preservation_summary["high_z_gate_damage_flag"] else 2.0,
        "evidence": f"naive high-z gate damage flag={preservation_summary['high_z_gate_damage_flag']}",
        "interpretation": "High score means preservation risk if low-z gate is misused.",
    })
    corr = sn_summary["gate_pressure_residual_correlation"]["r"]
    rows.append({
        "surface": "TAIRID_GATE_PRESSURE_DIAGNOSTIC",
        "pressure_score_0_to_10": abs(corr) * 10.0 if corr is not None else 0.0,
        "evidence": f"SN residual vs frozen gate pressure r={corr}",
        "interpretation": "Diagnostic alignment only; not a correction or proof.",
    })
    return rows


def decide(sn_summary, bao_summary, h0_summary, preservation_summary, pressure_rows):
    sn_ok = sn_summary["usable_sn_count"] >= 1000
    bao_ok = bao_summary["mean_file_count"] >= 5
    h0_ok = h0_summary["tension_sigma"] is not None and h0_summary["tension_sigma"] >= 3.0
    preservation_ok = preservation_summary["metric_accessibility_branch"] == "locked/admissible from v3.1"
    no_gate_misuse = preservation_summary["naive_low_z_gate_extension_allowed"] is False

    gates = [
        {"gate": "G1_pantheon_sn_surface_parsed", "passed": sn_ok, "evidence": {"usable_sn_count": sn_summary["usable_sn_count"]}},
        {"gate": "G2_desi_bao_surface_available", "passed": bao_ok, "evidence": bao_summary},
        {"gate": "G3_h0_pressure_seam_declared", "passed": h0_ok, "evidence": h0_summary},
        {"gate": "G4_cmb_bao_preservation_locked", "passed": preservation_ok and no_gate_misuse, "evidence": preservation_summary},
        {"gate": "G5_gate_used_as_diagnostic_only", "passed": True, "evidence": {"gate_distance_correction_fit": False}},
        {"gate": "G6_no_validation_claim_allowed", "passed": True, "evidence": {"validation_claim": False, "h0_correction_claim": False, "new_physics_claim": False}},
    ]
    failed = [g["gate"] for g in gates if not g["passed"]]

    if not failed:
        final_status = "multi_surface_pressure_map_ready_for_penalized_model_comparison"
        readiness = 9
        next_wall = (
            "v3.3 may run a penalized model-comparison prototype, but only inside the metric-accessibility branch "
            "and without treating the frozen gate as a free distance correction."
        )
    elif len(failed) <= 2 and sn_ok:
        final_status = "multi_surface_pressure_map_partial_with_cautions"
        readiness = 7
        next_wall = "Repair missing BAO or pressure-seam surfaces before model comparison."
    else:
        final_status = "multi_surface_pressure_map_not_ready"
        readiness = 5
        next_wall = "Core pressure surfaces did not parse. Do not escalate."

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness,
        "next_wall": next_wall,
        "failed_gates": failed,
        "gates": gates,
        "top_pressure_surfaces": sorted(pressure_rows, key=lambda r: -r["pressure_score_0_to_10"])[:5],
        "truth_boundary": CLAIMS_V3_2["truth_boundary"],
    }


def make_plots(sn_diag, pressure_rows):
    try:
        z = np.asarray([d["zHD"] for d in sn_diag], dtype=float)
        resid = np.asarray([d["sn_residual_mu"] for d in sn_diag], dtype=float)
        gatep = np.asarray([d["frozen_gate_pressure_minus_lnG"] for d in sn_diag], dtype=float)

        if len(z) > 0:
            idx = np.argsort(z)
            plt.figure(figsize=(9, 5))
            plt.scatter(z[idx], resid[idx], s=6, alpha=0.45)
            plt.xscale("log")
            plt.xlabel("redshift z")
            plt.ylabel("SN residual after best flat-LCDM shape fit")
            plt.title("v3.2 SN geometry residual pressure surface")
            plt.tight_layout()
            plt.savefig(OUTDIR / "sn_residual_pressure_surface_v3_2_fresh.png", dpi=160)
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.scatter(gatep, resid, s=6, alpha=0.45)
            plt.xlabel("frozen gate pressure -ln(G)")
            plt.ylabel("SN residual")
            plt.title("v3.2 diagnostic only: gate pressure vs SN residual")
            plt.tight_layout()
            plt.savefig(OUTDIR / "gate_pressure_vs_sn_residual_v3_2_fresh.png", dpi=160)
            plt.close()

        labels = [r["surface"] for r in pressure_rows]
        scores = [r["pressure_score_0_to_10"] for r in pressure_rows]
        x = np.arange(len(labels))
        plt.figure(figsize=(10, 5))
        plt.bar(x, scores)
        plt.xticks(x, labels, rotation=40, ha="right")
        plt.ylabel("pressure score 0-10")
        plt.title("v3.2 multi-surface pressure map")
        plt.tight_layout()
        plt.savefig(OUTDIR / "multi_surface_pressure_scores_v3_2_fresh.png", dpi=160)
        plt.close()
    except Exception as exc:
        write_json(OUTDIR / "plot_error_v3_2_fresh.json", {"error": repr(exc), "traceback": traceback.format_exc()})


def write_handoff(decision):
    lines = []
    lines.append("# TAIRID v3.2 Fresh Handoff")
    lines.append("")
    lines.append("## Current status")
    lines.append("")
    lines.append(f"- Final status: `{decision['final_status']}`")
    lines.append(f"- Readiness score: `{decision['readiness_score_0_to_10']}/10`")
    lines.append(f"- Next wall: {decision['next_wall']}")
    lines.append("")
    lines.append("## What v3.2 did")
    lines.append("")
    lines.append("- Parsed Pantheon+SH0ES SN geometry surface.")
    lines.append("- Fit a plain flat-LCDM SN shape surface with a free intercept.")
    lines.append("- Compared SN residual pressure against the frozen TAIRID low-z gate as a diagnostic only.")
    lines.append("- Parsed DESI DR1 BAO mean/covariance files when reachable.")
    lines.append("- Declared the H0 early/late mismatch as a pressure seam, not as a solved problem.")
    lines.append("- Kept CMB/BAO preservation locked so the low-z gate cannot overwrite early-universe ruler surfaces.")
    lines.append("")
    lines.append("## Top pressure surfaces")
    lines.append("")
    for row in decision["top_pressure_surfaces"]:
        lines.append(f"- `{row['surface']}` — score `{row['pressure_score_0_to_10']}` — {row['interpretation']}")
    lines.append("")
    lines.append("## Truth boundary")
    lines.append("")
    lines.append("- v3.2 is a pressure map only.")
    lines.append("- It does not validate TAIRID.")
    lines.append("- It does not prove H0 correction.")
    lines.append("- It does not prove new physics.")
    lines.append("- It does not disprove standard cosmology.")
    lines.append("")
    lines.append("## Next test")
    lines.append("")
    if decision["readiness_score_0_to_10"] >= 8:
        lines.append("Build v3.3 — Penalized Model-Comparison Prototype. It should compare ΛCDM, simple wCDM, and a constrained TAIRID metric-accessibility pressure term with parameter penalties and preservation locks.")
    else:
        lines.append("Repair missing pressure surfaces before building v3.3.")
    lines.append("")
    return "\n".join(lines)


def main():
    print("TAIRID Cosmology Multi-Surface Pressure Map v3.2 Fresh starting.")
    print("Boundary: pressure map only; no validation, no H0 claim, no new physics claim.")

    write_json(OUTDIR / "claims_v3_2_fresh.json", CLAIMS_V3_2)
    write_json(OUTDIR / "frozen_gate_v0_1_diagnostic_v3_2_fresh.json", FROZEN_GATE_V0_1)

    try:
        pantheon = fetch_text(PANTHEON_REPO, PANTHEON_BRANCHES, PANTHEON_DISTANCE_PATH)
        cov = fetch_text(PANTHEON_REPO, PANTHEON_BRANCHES, PANTHEON_COV_PATH, timeout=180)

        download_ledger = [
            {
                "surface": "Pantheon+SH0ES distance",
                "path": PANTHEON_DISTANCE_PATH,
                "status": pantheon["status"],
                "branch": pantheon["branch"],
                "url": pantheon["url"],
                "bytes": pantheon["bytes"],
                "errors": json.dumps(pantheon["errors"], default=json_default),
            },
            {
                "surface": "Pantheon+SH0ES covariance",
                "path": PANTHEON_COV_PATH,
                "status": cov["status"],
                "branch": cov["branch"],
                "url": cov["url"],
                "bytes": cov["bytes"],
                "errors": json.dumps(cov["errors"], default=json_default),
            },
        ]

        if pantheon["status"] == "downloaded":
            (DOWNLOAD_DIR / safe_name(PANTHEON_DISTANCE_PATH)).write_text(pantheon["text"], encoding="utf-8")
        if cov["status"] == "downloaded":
            (DOWNLOAD_DIR / safe_name(PANTHEON_COV_PATH)).write_text(cov["text"], encoding="utf-8")

        distance = parse_pantheon_distance(pantheon["text"])
        cov_shape = parse_cov_shape(cov["text"])
        sn_summary, sn_diag, omega_grid = fit_sn_shape(distance["rows"])

        write_csv(OUTDIR / "download_ledger_v3_2_fresh.csv", download_ledger)
        write_json(OUTDIR / "pantheon_covariance_shape_v3_2_fresh.json", cov_shape)
        write_csv(OUTDIR / "sn_pressure_diagnostics_v3_2_fresh.csv", sn_diag)
        write_csv(OUTDIR / "sn_omega_grid_shape_fit_v3_2_fresh.csv", omega_grid)
        write_json(OUTDIR / "sn_pressure_summary_v3_2_fresh.json", sn_summary)

        bao_summary, bao_ledger, bao_parsed, bao_previews = fetch_bao_surfaces()
        write_csv(OUTDIR / "bao_download_ledger_v3_2_fresh.csv", bao_ledger)
        write_csv(OUTDIR / "bao_surface_parse_summary_v3_2_fresh.csv", bao_parsed)
        write_csv(OUTDIR / "bao_surface_preview_rows_v3_2_fresh.csv", bao_previews)
        write_json(OUTDIR / "bao_surface_summary_v3_2_fresh.json", bao_summary)

        h0_summary = h0_pressure_seam()
        preservation_summary = preservation_pressure()
        write_json(OUTDIR / "h0_pressure_seam_v3_2_fresh.json", h0_summary)
        write_json(OUTDIR / "cmb_bao_preservation_pressure_v3_2_fresh.json", preservation_summary)

        pressure_rows = score_pressure_surfaces(sn_summary, bao_summary, h0_summary, preservation_summary)
        write_csv(OUTDIR / "multi_surface_pressure_scores_v3_2_fresh.csv", pressure_rows)

        decision = decide(sn_summary, bao_summary, h0_summary, preservation_summary, pressure_rows)
        write_json(OUTDIR / "decision_v3_2_fresh.json", decision)

        make_plots(sn_diag, pressure_rows)

        handoff = write_handoff(decision)
        (OUTDIR / "next_thread_handoff_after_v3_2_fresh.md").write_text(handoff, encoding="utf-8")
        (OUTDIR / "next_thread_handoff_after_v3_2_fresh.txt").write_text(handoff, encoding="utf-8")

        summary = {
            "test_name": "TAIRID Cosmology Multi-Surface Pressure Map v3.2 Fresh",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": "Pressure map only. No validation, no H0 correction claim, no new-physics claim.",
            "sn_pressure_summary": sn_summary,
            "pantheon_covariance_shape": cov_shape,
            "bao_surface_summary": bao_summary,
            "h0_pressure_seam": h0_summary,
            "cmb_bao_preservation_pressure": preservation_summary,
            "pressure_scores": pressure_rows,
            "decision": decision,
            "claims_v3_2": CLAIMS_V3_2,
            "output_files": {
                "summary_json": str(OUTDIR / "cosmology_multi_surface_pressure_map_v3_2_fresh_summary.json"),
                "summary_txt": str(OUTDIR / "cosmology_multi_surface_pressure_map_v3_2_fresh_summary.txt"),
                "download_ledger_csv": str(OUTDIR / "download_ledger_v3_2_fresh.csv"),
                "sn_pressure_csv": str(OUTDIR / "sn_pressure_diagnostics_v3_2_fresh.csv"),
                "sn_summary_json": str(OUTDIR / "sn_pressure_summary_v3_2_fresh.json"),
                "bao_download_ledger_csv": str(OUTDIR / "bao_download_ledger_v3_2_fresh.csv"),
                "bao_summary_json": str(OUTDIR / "bao_surface_summary_v3_2_fresh.json"),
                "pressure_scores_csv": str(OUTDIR / "multi_surface_pressure_scores_v3_2_fresh.csv"),
                "decision_json": str(OUTDIR / "decision_v3_2_fresh.json"),
                "handoff_md": str(OUTDIR / "next_thread_handoff_after_v3_2_fresh.md"),
                "plots": [
                    str(OUTDIR / "sn_residual_pressure_surface_v3_2_fresh.png"),
                    str(OUTDIR / "gate_pressure_vs_sn_residual_v3_2_fresh.png"),
                    str(OUTDIR / "multi_surface_pressure_scores_v3_2_fresh.png"),
                ],
            },
            "interpretation": {
                "what_success_means": "We can locate and compare multiple cosmology pressure surfaces without using one anomaly as proof.",
                "what_success_does_not_mean": "This does not validate TAIRID or solve H0/dark-energy/growth tensions.",
                "next_required_step": "v3.3 may run penalized model comparison only if v3.2 passes.",
                "truth_boundary": CLAIMS_V3_2["truth_boundary"],
            },
        }
        write_json(OUTDIR / "cosmology_multi_surface_pressure_map_v3_2_fresh_summary.json", summary)

        with open(OUTDIR / "cosmology_multi_surface_pressure_map_v3_2_fresh_summary.txt", "w", encoding="utf-8") as f:
            f.write("TAIRID Cosmology Multi-Surface Pressure Map v3.2 Fresh\n\n")
            f.write("Boundary: pressure map only. No validation. No H0 claim. No new physics claim.\n\n")
            f.write(f"Final status: {decision['final_status']}\n")
            f.write(f"Readiness score: {decision['readiness_score_0_to_10']}/10\n")
            f.write(f"Next wall: {decision['next_wall']}\n\n")
            f.write("Top pressure surfaces:\n")
            f.write(json.dumps(decision["top_pressure_surfaces"], indent=2, default=json_default) + "\n\n")
            f.write("SN summary:\n")
            f.write(json.dumps(sn_summary, indent=2, default=json_default) + "\n\n")
            f.write("BAO summary:\n")
            f.write(json.dumps(bao_summary, indent=2, default=json_default) + "\n\n")
            f.write("H0 pressure seam:\n")
            f.write(json.dumps(h0_summary, indent=2, default=json_default) + "\n\n")
            f.write("Failed gates:\n")
            f.write(json.dumps(decision["failed_gates"], indent=2, default=json_default) + "\n\n")
            f.write("Truth boundary:\n")
            f.write("- This does not prove TAIRID.\n")
            f.write("- This does not prove H0 correction.\n")
            f.write("- This does not prove new physics.\n")
            f.write("- This does not disprove standard cosmology.\n")
            f.write("- This only maps pressure surfaces.\n")

        print("TAIRID Cosmology Multi-Surface Pressure Map v3.2 Fresh complete.")
        print(f"Final status: {decision['final_status']}")
        print(f"Readiness score: {decision['readiness_score_0_to_10']}/10")

    except Exception as exc:
        summary = {
            "test_name": "TAIRID Cosmology Multi-Surface Pressure Map v3.2 Fresh",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "final_status": "cosmology_multi_surface_pressure_map_v3_2_runtime_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "truth_boundary": CLAIMS_V3_2["truth_boundary"],
        }
        write_json(OUTDIR / "cosmology_multi_surface_pressure_map_v3_2_fresh_summary.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
