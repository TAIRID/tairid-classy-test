#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder parameter-map audit v1.

Purpose:
The previous targeted FITS parser confirmed that the public SH0ES compact ladder system is
real and aligned:
- C = allc covariance matrix, 3492 x 3492
- L = alll equation/design matrix, 47 x 3492
- y = ally data vector, 3492

This test does not insert the TAIRID gate yet. It reconstructs the compact-ladder baseline
likelihood and builds a candidate semantic map of the 47 fitted parameters before any gate
perturbation is attempted.

This test asks:
1. Can we recompute the public compact-ladder chi-square from y, L, C, and lstsq_results.txt?
2. Do the 47 theta parameters have stable numerical/signature profiles?
3. Can support files such as README.md, table2.*, R22 outputs, and scripts help identify
   host distances, anchors, Cepheid relation terms, SN terms, nuisance terms, and H0-like terms?
4. Which parameters are safe candidates for a later calibrator-boundary perturbation test?

Boundary:
This is not a TAIRID gate test.
This is not a new SH0ES result.
This is not a cosmology fit.
This does not prove TAIRID cosmology.
It is a baseline compact-ladder audit and candidate parameter-map reconstruction.
"""

import csv
import hashlib
import json
import math
import os
import re
import shutil
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.linalg import cho_factor, cho_solve


OUTDIR = Path("shoes_compact_ladder_parameter_map_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_DIR = OUTDIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

REPO_OWNER = "PantheonPlusSH0ES"
REPO_NAME = "DataRelease"
BRANCH = "main"
SHOES_DIR = "SH0ES_Data"

GITHUB_API_SHOES_DIR = (
    f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{SHOES_DIR}?ref={BRANCH}"
)
RAW_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{SHOES_DIR}"
MEDIA_BASE = f"https://media.githubusercontent.com/media/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{SHOES_DIR}"
GITHUB_RAW_BASE = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/raw/{BRANCH}/{SHOES_DIR}"

PRIMARY_FILES = [
    "allc_shoes_ceph_topantheonwt6.0_112221.fits",
    "alll_shoes_ceph_topantheonwt6.0_112221.fits",
    "ally_shoes_ceph_topantheonwt6.0_112221.fits",
    "lstsq_results.txt",
]

SUPPORT_FILES = [
    "README.md",
    "table2.README",
    "table2.tex",
    "R22_orig19_NIR.out",
    "R22_orig19_NIR.wm31.out",
    "optical_wes_R22_for19fromR16.dat",
    "optical_wes_R22_for19fromR16.wM31.dat",
    "read_chains_example.py",
    "MCMC_utils.py",
    "run_mcmc.py",
]

ALL_TARGETS = PRIMARY_FILES + SUPPORT_FILES

PARAMETER_HINTS = [
    "theta", "parameter", "distance", "modulus", "mu", "host", "anchor", "cepheid", "ceph",
    "period", "logp", "metal", "slope", "intercept", "zeropoint", "zero point", "zpt",
    "M_H", "m_h", "M_B", "supernova", "snia", "sn", "H0", "lmc", "n4258", "mw", "m31",
    "parallax", "maser", "calib", "calibrator", "cov", "matrix", "lstsq", "R22", "SH0ES",
]


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        if value.size > 60:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sample_first_60": json_safe(value.ravel()[:60]),
            }
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_url(url, timeout=180):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-SH0ES-compact-ladder-parameter-map-v1",
            "Accept": "application/vnd.github.v3+json, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        final_url = response.geturl()
        status = getattr(response, "status", None)
        content_type = response.headers.get("Content-Type", "")
    return data, final_url, status, content_type


def is_git_lfs_pointer_bytes(data):
    head = data[:300].decode("utf-8", errors="ignore")
    return "version https://git-lfs.github.com/spec/v1" in head and "oid sha256:" in head


def parse_git_lfs_pointer(data):
    text = data.decode("utf-8", errors="replace")
    out = {"raw_text": text}
    oid_match = re.search(r"oid\s+sha256:([0-9a-fA-F]+)", text)
    size_match = re.search(r"size\s+([0-9]+)", text)
    if oid_match:
        out["oid_sha256"] = oid_match.group(1)
    if size_match:
        out["size"] = int(size_match.group(1))
    return out


def github_api_list_shoes_dir():
    try:
        data, final_url, status, content_type = read_url(GITHUB_API_SHOES_DIR)
        listing = json.loads(data.decode("utf-8"))
        if not isinstance(listing, list):
            raise RuntimeError("GitHub API listing did not return a list.")
        return listing
    except Exception as exc:
        return {"error": str(exc), "url": GITHUB_API_SHOES_DIR}


def build_url_candidates(filename, api_listing_by_name):
    encoded = urllib.parse.quote(filename)
    candidates = []
    api_entry = api_listing_by_name.get(filename)
    if isinstance(api_entry, dict):
        if api_entry.get("download_url"):
            candidates.append(api_entry["download_url"])
        if api_entry.get("html_url"):
            candidates.append(api_entry["html_url"].replace("/blob/", "/raw/"))
    candidates.extend(
        [
            f"{MEDIA_BASE}/{encoded}",
            f"{GITHUB_RAW_BASE}/{encoded}",
            f"{RAW_BASE}/{encoded}",
        ]
    )
    out = []
    seen = set()
    for url in candidates:
        if url and url not in seen:
            out.append(url)
            seen.add(url)
    return out


def download_or_copy(filename, api_listing_by_name):
    local_path = DOWNLOAD_DIR / filename

    local_data_dir = os.environ.get("LOCAL_SHOES_DATA_DIR")
    if local_data_dir:
        source = Path(local_data_dir) / filename
        if source.exists():
            shutil.copy2(source, local_path)
            return {
                "filename": filename,
                "status": "copied_from_LOCAL_SHOES_DATA_DIR",
                "path": str(local_path),
                "size_bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
                "attempts": [],
            }

    candidates = build_url_candidates(filename, api_listing_by_name)
    attempts = []
    pointer_best = None

    for url in candidates:
        try:
            data, final_url, status, content_type = read_url(url)
            is_pointer = is_git_lfs_pointer_bytes(data)
            attempt = {
                "url": url,
                "final_url": final_url,
                "status": status,
                "content_type": content_type,
                "size_bytes": len(data),
                "git_lfs_pointer": is_pointer,
            }

            if is_pointer:
                attempt["pointer"] = parse_git_lfs_pointer(data)
                pointer_best = data
                attempts.append(attempt)
                continue

            local_path.write_bytes(data)
            attempt["sha256"] = sha256_file(local_path)
            attempts.append(attempt)

            return {
                "filename": filename,
                "status": "downloaded",
                "path": str(local_path),
                "size_bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
                "attempts": attempts,
            }

        except Exception as exc:
            attempts.append({"url": url, "status": "error", "error": str(exc)})

    if pointer_best is not None:
        pointer_path = DOWNLOAD_DIR / f"{filename}.git_lfs_pointer.txt"
        pointer_path.write_bytes(pointer_best)
        return {
            "filename": filename,
            "status": "git_lfs_pointer_only",
            "path": str(pointer_path),
            "size_bytes": pointer_path.stat().st_size,
            "pointer": parse_git_lfs_pointer(pointer_best),
            "attempts": attempts,
        }

    return {
        "filename": filename,
        "status": "missing_or_download_failed",
        "path": None,
        "attempts": attempts,
    }


def load_primary_matrices(downloads_by_name):
    paths = {}

    for name in PRIMARY_FILES:
        item = downloads_by_name.get(name)
        if not item or item.get("status") not in ["downloaded", "copied_from_LOCAL_SHOES_DATA_DIR"]:
            raise RuntimeError(f"Required file was not downloaded/copied: {name}. Status: {item}")
        paths[name] = Path(item["path"])

    C = np.asarray(
        fits.open(paths["allc_shoes_ceph_topantheonwt6.0_112221.fits"], memmap=True)[0].data,
        dtype=np.float64,
    )
    L = np.asarray(
        fits.open(paths["alll_shoes_ceph_topantheonwt6.0_112221.fits"], memmap=True)[0].data,
        dtype=np.float64,
    )
    Y = np.asarray(
        fits.open(paths["ally_shoes_ceph_topantheonwt6.0_112221.fits"], memmap=True)[0].data,
        dtype=np.float64,
    )

    q_sigma = np.loadtxt(paths["lstsq_results.txt"])
    if q_sigma.ndim != 2 or q_sigma.shape[1] < 2:
        raise RuntimeError(f"Could not parse lstsq_results.txt shape: {q_sigma.shape}")

    theta = np.asarray(q_sigma[:, 0], dtype=np.float64)
    theta_sigma = np.asarray(q_sigma[:, 1], dtype=np.float64)

    if L.shape[0] != theta.size and L.shape[1] == theta.size:
        L = L.T

    if L.shape[0] != theta.size:
        raise RuntimeError(f"L/theta mismatch: L shape {L.shape}, theta length {theta.size}")
    if L.shape[1] != Y.size:
        raise RuntimeError(f"L/Y mismatch: L shape {L.shape}, Y length {Y.size}")
    if C.shape != (Y.size, Y.size):
        raise RuntimeError(f"C/Y mismatch: C shape {C.shape}, Y length {Y.size}")

    return Y, L, C, theta, theta_sigma, {k: str(v) for k, v in paths.items()}


def compute_baseline_likelihood(Y, L, C, theta):
    model = np.dot(theta, L)
    residual = Y - model

    c_factor = cho_factor(C, lower=True, check_finite=False)
    solved = cho_solve(c_factor, residual, check_finite=False)

    chi2 = float(np.dot(residual, solved))
    dof = int(Y.size - theta.size)
    loglike = -0.5 * chi2

    return {
        "model": model,
        "residual": residual,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": float(chi2 / dof),
        "log_likelihood_unnormalized": loglike,
    }


def classify_theta_value(index, theta_value, sigma, row, Y):
    nz = np.where(np.abs(row) > 1.0e-10)[0]
    nnz = int(nz.size)
    absval = abs(float(theta_value))

    if sigma == 0.0 and absval < 1.0e-12:
        return "fixed_or_gauge_parameter_candidate", "medium"
    if 29.0 <= theta_value <= 35.5:
        return "host_distance_modulus_candidate", "medium"
    if 23.0 <= theta_value <= 25.5:
        return "nearby_anchor_distance_modulus_candidate", "medium"
    if -6.8 <= theta_value <= -5.0:
        return "cepheid_relation_intercept_or_zero_point_candidate", "medium"
    if -20.5 <= theta_value <= -18.0:
        return "supernova_absolute_magnitude_intercept_candidate", "medium"
    if 8.0 <= theta_value <= 10.5:
        return "H0_or_distance_ladder_intercept_candidate", "low"
    if absval <= 0.35 and nnz > 1000:
        return "global_slope_color_metallicity_or_cross_calibration_candidate", "low"
    if absval <= 0.35:
        return "local_slope_offset_anchor_or_nuisance_candidate", "low"

    return "unmapped_parameter_candidate", "low"


def rounded_top_values(values, limit=12):
    if values.size == 0:
        return []
    rounded = np.round(values.astype(float), 6)
    counts = Counter(rounded.tolist())
    return [{"value": float(k), "count": int(v)} for k, v in counts.most_common(limit)]


def build_parameter_map(theta, theta_sigma, L, Y, residual):
    rows = []

    for i in range(theta.size):
        coeff = L[i, :]
        nz = np.where(np.abs(coeff) > 1.0e-10)[0]
        nz_coeff = coeff[nz]
        nz_y = Y[nz] if nz.size else np.asarray([])
        contribution = theta[i] * coeff

        role, confidence = classify_theta_value(i, theta[i], theta_sigma[i], coeff, Y)

        row = {
            "theta_index": int(i),
            "theta_value": float(theta[i]),
            "theta_sigma_lstsq": float(theta_sigma[i]),
            "heuristic_role_candidate": role,
            "role_confidence": confidence,
            "nonzero_count": int(nz.size),
            "nonzero_fraction": float(nz.size / Y.size),
            "obs_index_min": int(nz.min()) if nz.size else None,
            "obs_index_max": int(nz.max()) if nz.size else None,
            "coeff_min": float(np.min(nz_coeff)) if nz.size else None,
            "coeff_max": float(np.max(nz_coeff)) if nz.size else None,
            "coeff_mean": float(np.mean(nz_coeff)) if nz.size else None,
            "coeff_std": float(np.std(nz_coeff)) if nz.size else None,
            "coeff_abs_max": float(np.max(np.abs(nz_coeff))) if nz.size else None,
            "coeff_top_values_json": json.dumps(rounded_top_values(nz_coeff)),
            "y_min_where_active": float(np.min(nz_y)) if nz.size else None,
            "y_max_where_active": float(np.max(nz_y)) if nz.size else None,
            "y_mean_where_active": float(np.mean(nz_y)) if nz.size else None,
            "y_std_where_active": float(np.std(nz_y)) if nz.size else None,
            "contribution_mean": float(np.mean(contribution)),
            "contribution_rms": float(np.sqrt(np.mean(contribution * contribution))),
            "contribution_abs_max": float(np.max(np.abs(contribution))),
            "residual_mean_where_active": float(np.mean(residual[nz])) if nz.size else None,
            "residual_rms_where_active": float(np.sqrt(np.mean(residual[nz] ** 2))) if nz.size else None,
        }
        rows.append(row)

    return rows


def write_parameter_map_csv(rows):
    path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_parameter_map.csv"
    fieldnames = list(rows[0].keys()) if rows else []

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def classify_observation(Y_value, active_count):
    if Y_value > 10.0:
        y_class = "positive_magnitude_or_distance_like_row"
    elif Y_value < -10.0:
        y_class = "negative_absolute_magnitude_or_transformed_SN_row"
    elif abs(Y_value) <= 1.0:
        y_class = "near_zero_constraint_or_anchor_relation_row"
    else:
        y_class = "intermediate_transformed_row"

    if active_count <= 2:
        structure_class = "sparse_equation"
    elif active_count <= 5:
        structure_class = "moderate_equation"
    else:
        structure_class = "dense_equation"

    return f"{y_class}__{structure_class}"


def build_observation_profile(Y, L, residual):
    rows = []
    LT = L.T

    for j in range(Y.size):
        active = np.where(np.abs(LT[j]) > 1.0e-10)[0]
        coeffs = LT[j, active]

        rows.append(
            {
                "obs_index": int(j),
                "Y_value": float(Y[j]),
                "residual": float(residual[j]),
                "active_theta_count": int(active.size),
                "active_theta_indices": " ".join(str(int(x)) for x in active),
                "active_coefficients": " ".join(f"{float(x):.6g}" for x in coeffs),
                "observation_class_candidate": classify_observation(Y[j], active.size),
            }
        )

    return rows


def write_observation_profile_csv(rows):
    path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_observation_profile.csv"
    fieldnames = list(rows[0].keys()) if rows else []

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def parse_optical_wes(path):
    rows = []

    if not path.exists():
        return rows

    lines = path.read_text(errors="replace").splitlines()

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("-") or stripped.lower().startswith("host"):
            continue

        parts = stripped.split()

        if len(parts) < 11:
            continue

        try:
            rows.append(
                {
                    "Host": parts[0],
                    "ra": float(parts[1]),
                    "dec": float(parts[2]),
                    "ID": parts[3],
                    "period": float(parts[4]),
                    "V_minus_I": float(parts[5]),
                    "sigma_VI": float(parts[6]),
                    "I": float(parts[7]),
                    "sigma_I": float(parts[8]),
                    "metal_minus_8p69": float(parts[9]),
                    "HST": parts[10],
                }
            )
        except Exception:
            continue

    return rows


def write_optical_inventory(downloads_by_name):
    files = [
        "optical_wes_R22_for19fromR16.dat",
        "optical_wes_R22_for19fromR16.wM31.dat",
    ]

    all_summaries = []
    rows_out = []
    host_rows = []

    for name in files:
        item = downloads_by_name.get(name)

        if not item or not item.get("path"):
            all_summaries.append({"filename": name, "status": "missing"})
            continue

        path = Path(item["path"])
        rows = parse_optical_wes(path)

        periods = np.asarray([r["period"] for r in rows], dtype=float) if rows else np.asarray([])
        mags = np.asarray([r["I"] for r in rows], dtype=float) if rows else np.asarray([])
        host_counts = Counter(r["Host"] for r in rows)

        all_summaries.append(
            {
                "filename": name,
                "status": "parsed" if rows else "no_rows_parsed",
                "row_count": len(rows),
                "host_count": len(host_counts),
                "period_min": float(np.min(periods)) if periods.size else None,
                "period_max": float(np.max(periods)) if periods.size else None,
                "I_mag_min": float(np.min(mags)) if mags.size else None,
                "I_mag_max": float(np.max(mags)) if mags.size else None,
                "top_hosts": host_counts.most_common(20),
            }
        )

        for host, count in host_counts.most_common():
            host_rows.append({"filename": name, "host": host, "count": int(count)})

        for r in rows[:200]:
            rows_out.append({"filename": name, **r})

    host_path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_optical_host_counts.csv"

    with open(host_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "host", "count"])
        writer.writeheader()
        writer.writerows(host_rows)

    sample_path = SAMPLES_DIR / "shoes_compact_ladder_parameter_map_v1_optical_sample_rows.csv"

    if rows_out:
        with open(sample_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

    return {
        "summaries": all_summaries,
        "host_counts_csv": str(host_path),
        "sample_rows_csv": str(sample_path) if rows_out else None,
    }


def inspect_support_text_files(downloads_by_name):
    support_summary = []
    hit_rows = []

    for name in SUPPORT_FILES + ["lstsq_results.txt"]:
        item = downloads_by_name.get(name)

        if not item or not item.get("path"):
            support_summary.append({"filename": name, "status": "missing"})
            continue

        path = Path(item["path"])

        if not path.exists() or path.suffix.lower() == ".fits":
            continue

        text = path.read_text(errors="replace")
        lines = text.splitlines()
        lower = text.lower()

        hint_counts = {
            hint: lower.count(hint.lower())
            for hint in PARAMETER_HINTS
            if lower.count(hint.lower())
        }

        first_lines_path = SAMPLES_DIR / f"{name}_first_250_lines.txt"
        first_lines_path.write_text("\n".join(lines[:250]))

        matched = 0

        for idx, line in enumerate(lines, start=1):
            lowered = line.lower()

            if any(h.lower() in lowered for h in PARAMETER_HINTS):
                hit_rows.append({"filename": name, "line": idx, "text": line[:800]})
                matched += 1

            if matched >= 250:
                break

        support_summary.append(
            {
                "filename": name,
                "status": "inspected",
                "line_count": len(lines),
                "size_bytes": path.stat().st_size,
                "hint_counts": hint_counts,
                "first_250_lines_file": str(first_lines_path),
                "matched_hint_line_count_capped": matched,
            }
        )

    hits_path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_support_text_hits.csv"

    with open(hits_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "line", "text"])
        writer.writeheader()
        writer.writerows(hit_rows)

    return support_summary, str(hits_path)


def write_download_ledger(downloads):
    path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_download_ledger.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "status",
                "path",
                "size_bytes",
                "sha256",
                "git_lfs_pointer",
                "pointer_size",
                "pointer_oid_sha256",
                "attempt_count",
            ],
        )
        writer.writeheader()

        for item in downloads:
            pointer = item.get("pointer", {})

            writer.writerow(
                {
                    "filename": item.get("filename"),
                    "status": item.get("status"),
                    "path": item.get("path"),
                    "size_bytes": item.get("size_bytes"),
                    "sha256": item.get("sha256"),
                    "git_lfs_pointer": bool(pointer),
                    "pointer_size": pointer.get("size"),
                    "pointer_oid_sha256": pointer.get("oid_sha256"),
                    "attempt_count": len(item.get("attempts", [])),
                }
            )

    return path


def summarize_matrix_system(Y, L, C, theta, theta_sigma):
    diag = np.diag(C)

    return {
        "Y_shape": list(Y.shape),
        "L_shape": list(L.shape),
        "C_shape": list(C.shape),
        "theta_count": int(theta.size),
        "observation_count": int(Y.size),
        "C_diag_min": float(np.min(diag)),
        "C_diag_max": float(np.max(diag)),
        "C_diag_mean": float(np.mean(diag)),
        "C_max_relative_asymmetry": float(
            np.max(np.abs(C - C.T)) / max(1.0e-30, np.max(np.abs(C)))
        ),
        "theta_min": float(np.min(theta)),
        "theta_max": float(np.max(theta)),
        "theta_sigma_min": float(np.min(theta_sigma)),
        "theta_sigma_max": float(np.max(theta_sigma)),
    }


def make_plots(theta, theta_sigma, parameter_rows, Y, residual):
    plot_paths = []
    indices = np.arange(theta.size)

    plt.figure(figsize=(12, 6))
    plt.errorbar(indices, theta, yerr=theta_sigma, fmt="o", markersize=3, capsize=2)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("theta index")
    plt.ylabel("theta value from lstsq_results.txt")
    plt.title("SH0ES compact ladder parameter-map v1: theta values")
    plt.tight_layout()
    path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_theta_values.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    nonzero = [r["nonzero_count"] for r in parameter_rows]

    plt.figure(figsize=(12, 6))
    plt.bar(indices, nonzero)
    plt.xlabel("theta index")
    plt.ylabel("nonzero coefficient count in L row")
    plt.title("SH0ES compact ladder parameter-map v1: coefficient support by parameter")
    plt.tight_layout()
    path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_nonzero_counts.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    plt.figure(figsize=(12, 6))
    plt.scatter(np.arange(Y.size), Y, s=4, alpha=0.35)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("observation/equation index")
    plt.ylabel("Y value")
    plt.title("SH0ES compact ladder parameter-map v1: y vector by observation index")
    plt.tight_layout()
    path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_y_vector_profile.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    plt.figure(figsize=(12, 6))
    plt.scatter(np.arange(residual.size), residual, s=4, alpha=0.35)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("observation/equation index")
    plt.ylabel("baseline residual")
    plt.title("SH0ES compact ladder parameter-map v1: baseline residuals")
    plt.tight_layout()
    path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_baseline_residuals.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    return plot_paths


def main():
    print("")
    print("TAIRID SH0ES compact ladder parameter-map audit v1 starting.")
    print("Boundary: baseline compact-ladder audit only; no TAIRID gate insertion.")
    print("")

    api_listing = github_api_list_shoes_dir()

    if isinstance(api_listing, dict) and "error" in api_listing:
        api_listing_by_name = {}
        api_summary = api_listing
    else:
        api_listing_by_name = {
            item.get("name"): item
            for item in api_listing
            if isinstance(item, dict)
        }
        api_summary = {
            "status": "ok",
            "url": GITHUB_API_SHOES_DIR,
            "file_count": len(api_listing_by_name),
            "files": sorted(api_listing_by_name.keys()),
        }

    (OUTDIR / "shoes_compact_ladder_parameter_map_v1_github_api_listing.json").write_text(
        json.dumps(api_summary, indent=2)
    )

    downloads = []

    for filename in ALL_TARGETS:
        print(f"Downloading / locating {filename} ...")
        result = download_or_copy(filename, api_listing_by_name)
        downloads.append(result)
        print(f"  {result.get('status')}")

    downloads_by_name = {item["filename"]: item for item in downloads}
    download_ledger = write_download_ledger(downloads)

    Y, L, C, theta, theta_sigma, primary_paths = load_primary_matrices(downloads_by_name)

    baseline = compute_baseline_likelihood(Y, L, C, theta)
    residual = baseline["residual"]

    matrix_summary = summarize_matrix_system(Y, L, C, theta, theta_sigma)
    parameter_rows = build_parameter_map(theta, theta_sigma, L, Y, residual)
    observation_rows = build_observation_profile(Y, L, residual)

    parameter_map_path = write_parameter_map_csv(parameter_rows)
    observation_profile_path = write_observation_profile_csv(observation_rows)

    optical_inventory = write_optical_inventory(downloads_by_name)
    support_summary, support_hits_csv = inspect_support_text_files(downloads_by_name)

    baseline_audit = {
        "likelihood_form": "res = Y - np.dot(theta, L); chi2 = res.T @ C^-1 @ res",
        "theta_source": "lstsq_results.txt first column",
        "theta_sigma_source": "lstsq_results.txt second column",
        "chi2": baseline["chi2"],
        "dof": baseline["dof"],
        "reduced_chi2": baseline["reduced_chi2"],
        "log_likelihood_unnormalized": baseline["log_likelihood_unnormalized"],
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "residual_min": float(np.min(residual)),
        "residual_max": float(np.max(residual)),
    }

    baseline_audit_path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_baseline_chi2_audit.json"
    baseline_audit_path.write_text(json.dumps(baseline_audit, indent=2))

    role_counts = Counter(r["heuristic_role_candidate"] for r in parameter_rows)

    high_value_roles = [
        r for r in parameter_rows
        if r["heuristic_role_candidate"] in [
            "host_distance_modulus_candidate",
            "nearby_anchor_distance_modulus_candidate",
            "cepheid_relation_intercept_or_zero_point_candidate",
            "supernova_absolute_magnitude_intercept_candidate",
            "H0_or_distance_ladder_intercept_candidate",
        ]
    ]

    all_primary_ok = all(
        downloads_by_name.get(name, {}).get("status")
        in ["downloaded", "copied_from_LOCAL_SHOES_DATA_DIR"]
        for name in PRIMARY_FILES
    )

    baseline_ok = math.isfinite(baseline["chi2"]) and baseline["dof"] == int(Y.size - theta.size)

    support_downloaded_count = sum(
        1 for name in SUPPORT_FILES
        if downloads_by_name.get(name, {}).get("status")
        in ["downloaded", "copied_from_LOCAL_SHOES_DATA_DIR"]
    )

    if all_primary_ok and baseline_ok and support_downloaded_count >= 8:
        final_status = "baseline_chi_square_reconstructed_candidate_parameter_map_built"
        readiness_score = 8
    elif all_primary_ok and baseline_ok:
        final_status = "baseline_chi_square_reconstructed_support_mapping_partial"
        readiness_score = 7
    elif all_primary_ok:
        final_status = "matrix_system_loaded_but_chi_square_audit_failed"
        readiness_score = 5
    else:
        final_status = "required_compact_ladder_files_missing"
        readiness_score = 3

    plot_paths = make_plots(theta, theta_sigma, parameter_rows, Y, residual)

    summary = {
        "test_name": "TAIRID SH0ES compact ladder parameter-map audit v1",
        "boundary": (
            "Baseline compact-ladder audit only. No TAIRID gate insertion. "
            "Not a new SH0ES result and not proof."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "primary_paths": primary_paths,
        "github_api_listing": api_summary,
        "downloads": downloads,
        "matrix_summary": matrix_summary,
        "baseline_chi2_audit": baseline_audit,
        "role_counts": dict(role_counts),
        "high_value_role_candidates": high_value_roles,
        "optical_wes_inventory": optical_inventory,
        "support_text_summaries": support_summary,
        "output_files": {
            "download_ledger_csv": str(download_ledger),
            "parameter_map_csv": str(parameter_map_path),
            "observation_profile_csv": str(observation_profile_path),
            "baseline_chi2_audit_json": str(baseline_audit_path),
            "support_text_hits_csv": support_hits_csv,
            "optical_host_counts_csv": optical_inventory.get("host_counts_csv"),
            "plots": plot_paths,
        },
        "next_test_recommendation": {
            "name": "SH0ES compact ladder gate-insertion dry run v1",
            "only_after": (
                "Use this parameter map to select safe rows/terms. "
                "Do not perturb y/L/C blindly."
            ),
            "goal": (
                "Add a calibrator-boundary vector to the compact system and compare delta chi2 "
                "against ordinary offset/host-distance freedom."
            ),
        },
    }

    summary_path = OUTDIR / "shoes_compact_ladder_parameter_map_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    with open(OUTDIR / "shoes_compact_ladder_parameter_map_v1_summary.txt", "w") as f:
        f.write("TAIRID SH0ES compact ladder parameter-map audit v1\n\n")
        f.write("Boundary: baseline compact-ladder audit only. No TAIRID gate insertion. Not proof.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n\n")

        f.write("Matrix system:\n")
        f.write(json.dumps(matrix_summary, indent=2) + "\n\n")

        f.write("Baseline compact ladder chi-square audit:\n")
        f.write(json.dumps(baseline_audit, indent=2) + "\n\n")

        f.write("Heuristic role counts:\n")
        f.write(json.dumps(dict(role_counts), indent=2) + "\n\n")

        f.write("High-value role candidates:\n")
        f.write(json.dumps(high_value_roles, indent=2) + "\n\n")

        f.write("Optical WES source-table inventory:\n")
        f.write(json.dumps(optical_inventory, indent=2) + "\n\n")

        f.write("Interpretation guide:\n")
        f.write("- If baseline chi-square reconstructs, the compact SH0ES likelihood is runnable.\n")
        f.write("- Parameter roles in the CSV are candidate labels, not confirmed SH0ES parameter names.\n")
        f.write("- Host-distance and anchor-like candidates are likely the safest place to inspect first.\n")
        f.write("- The next gate test must compare any TAIRID perturbation against ordinary offset/host-distance freedom.\n")
        f.write("- Do not claim TAIRID solved Hubble tension from this audit.\n")

    print("")
    print("TAIRID SH0ES compact ladder parameter-map audit v1 complete.")
    print("Created:")
    print("  shoes_compact_ladder_parameter_map_v1_outputs/shoes_compact_ladder_parameter_map_v1_summary.json")
    print("  shoes_compact_ladder_parameter_map_v1_outputs/shoes_compact_ladder_parameter_map_v1_summary.txt")
    print("  shoes_compact_ladder_parameter_map_v1_outputs/shoes_compact_ladder_parameter_map_v1_baseline_chi2_audit.json")
    print("  shoes_compact_ladder_parameter_map_v1_outputs/shoes_compact_ladder_parameter_map_v1_parameter_map.csv")
    print("  shoes_compact_ladder_parameter_map_v1_outputs/shoes_compact_ladder_parameter_map_v1_observation_profile.csv")
    print("  shoes_compact_ladder_parameter_map_v1_outputs/shoes_compact_ladder_parameter_map_v1_support_text_hits.csv")
    print("")
    print("Boundary:")
    print("  This is not a TAIRID gate test.")
    print("  This is not a new SH0ES result.")
    print("  This is a compact-ladder baseline and parameter-map audit.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
