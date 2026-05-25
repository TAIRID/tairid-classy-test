#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder gate-insertion dry run v1.

Purpose:
The previous compact-ladder parameter-map audit confirmed that the public SH0ES matrix
system is runnable:

- C = 3492 x 3492 covariance matrix
- L = 47 x 3492 equation/design matrix
- y = 3492 data vector
- theta = 47 fitted ladder parameters from lstsq_results.txt

It also identified candidate parameter roles:
- theta 0-36: host-distance-modulus-like candidates
- theta 38: Cepheid relation intercept / zero-point-like candidate
- theta 40: nearby anchor-distance-modulus-like candidate
- theta 42: SN absolute-magnitude intercept-like candidate
- theta 46: five-log-H0 parameter, with H0 = 10^(theta_46/5)

This test finally performs the first controlled TAIRID gate-insertion dry run.

What it asks:
If a structured calibrator-boundary / ladder-boundary perturbation is inserted into the
compact SH0ES y/L/C system, does it survive after the ordinary SH0ES ladder parameters
are refit, or is it absorbed by normal ladder freedom?

Method:
1. Load y, L, C, theta.
2. Recompute baseline generalized least-squares chi-square.
3. Build candidate gate vectors in observation/equation space.
4. Project each gate vector through different refit freedoms:
   - H0-only
   - SN-intercept-only
   - host-distance-only
   - host + H0
   - host + SN intercept
   - host + SN intercept + H0
   - all 47 parameters
5. Measure absorption fraction, residual fraction, delta chi-square, theta shifts,
   H0 shift, and whether the gate is distinguishable from ordinary offset freedom.

Boundary:
This is not a final SH0ES likelihood analysis.
This is not a claim that TAIRID solves Hubble tension.
This is not a cosmology proof.
This is a dry-run perturbation/projection test inside the public compact ladder system.
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


OUTDIR = Path("shoes_compact_ladder_gate_insertion_dryrun_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    "read_chains_example.py",
    "MCMC_utils.py",
]

HOST_THETA = list(range(0, 37))
GLOBAL_THETA = list(range(37, 47))
CEPHEID_ZEROPOINT_THETA = [38]
ANCHOR_THETA = [40]
SN_INTERCEPT_THETA = [42]
FIVELOGH0_THETA = [46]

FREE_SETS = {
    "h0_only": FIVELOGH0_THETA,
    "sn_intercept_only": SN_INTERCEPT_THETA,
    "host_distance_only": HOST_THETA,
    "host_plus_h0": HOST_THETA + FIVELOGH0_THETA,
    "host_plus_sn_intercept": HOST_THETA + SN_INTERCEPT_THETA,
    "host_plus_sn_intercept_plus_h0": HOST_THETA + SN_INTERCEPT_THETA + FIVELOGH0_THETA,
    "host_plus_anchor_plus_cepheid_zp_plus_sn_plus_h0": (
        HOST_THETA + ANCHOR_THETA + CEPHEID_ZEROPOINT_THETA + SN_INTERCEPT_THETA + FIVELOGH0_THETA
    ),
    "global_only_37_to_46": GLOBAL_THETA,
    "all_47": list(range(47)),
}

GATE_AMPLITUDES = [0.010, 0.025, 0.050, 0.085]
COLUMN_SHIFT_MAGNITUDES = [0.010, 0.050, 0.100]


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        if value.size > 80:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sample_first_80": json_safe(value.ravel()[:80]),
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
            "User-Agent": "TAIRID-SH0ES-compact-ladder-gate-insertion-dryrun-v1",
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


def load_primary_system(downloads_by_name):
    required = [
        "allc_shoes_ceph_topantheonwt6.0_112221.fits",
        "alll_shoes_ceph_topantheonwt6.0_112221.fits",
        "ally_shoes_ceph_topantheonwt6.0_112221.fits",
        "lstsq_results.txt",
    ]

    paths = {}
    for name in required:
        item = downloads_by_name.get(name)
        if not item or item.get("status") not in ["downloaded", "copied_from_LOCAL_SHOES_DATA_DIR"]:
            raise RuntimeError(f"Required file missing: {name}. Item: {item}")
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
    theta_public = np.asarray(q_sigma[:, 0], dtype=np.float64)
    theta_sigma = np.asarray(q_sigma[:, 1], dtype=np.float64)

    if L.shape[0] != theta_public.size and L.shape[1] == theta_public.size:
        L = L.T

    if L.shape[0] != theta_public.size:
        raise RuntimeError(f"L/theta mismatch: L shape {L.shape}, theta length {theta_public.size}")
    if L.shape[1] != Y.size:
        raise RuntimeError(f"L/Y mismatch: L shape {L.shape}, Y length {Y.size}")
    if C.shape != (Y.size, Y.size):
        raise RuntimeError(f"C/Y mismatch: C shape {C.shape}, Y length {Y.size}")

    M = L.T

    return Y, L, M, C, theta_public, theta_sigma, {k: str(v) for k, v in paths.items()}


def generalized_least_squares_setup(Y, M, C):
    c_factor = cho_factor(C, lower=True, check_finite=False)

    Cinv_Y = cho_solve(c_factor, Y, check_finite=False)
    Cinv_M = cho_solve(c_factor, M, check_finite=False)

    A = M.T @ Cinv_M
    b = M.T @ Cinv_Y

    theta_gls = np.linalg.solve(A, b)
    residual = Y - M @ theta_gls
    Cinv_residual = cho_solve(c_factor, residual, check_finite=False)

    chi2 = float(residual @ Cinv_residual)
    dof = int(Y.size - theta_gls.size)

    return {
        "c_factor": c_factor,
        "Cinv_M": Cinv_M,
        "A": A,
        "b": b,
        "theta_gls": theta_gls,
        "residual": residual,
        "Cinv_residual": Cinv_residual,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": float(chi2 / dof),
    }


def chi2_for_theta(Y, M, c_factor, theta):
    residual = Y - M @ theta
    solved = cho_solve(c_factor, residual, check_finite=False)
    return {
        "chi2": float(residual @ solved),
        "residual": residual,
    }


def h0_from_theta46(theta46):
    return float(10.0 ** (theta46 / 5.0))


def active_theta_masks(M):
    active = np.abs(M) > 1.0e-10

    host = np.any(active[:, HOST_THETA], axis=1)
    global_any = np.any(active[:, GLOBAL_THETA], axis=1)
    cepheid_zp = np.any(active[:, CEPHEID_ZEROPOINT_THETA], axis=1)
    anchor = np.any(active[:, ANCHOR_THETA], axis=1)
    sn_intercept = np.any(active[:, SN_INTERCEPT_THETA], axis=1)
    h0 = np.any(active[:, FIVELOGH0_THETA], axis=1)

    active_count = np.sum(active, axis=1)

    y_dummy = None

    return {
        "host_active_rows": host,
        "global_active_rows": global_any,
        "cepheid_zp_active_rows": cepheid_zp,
        "anchor_active_rows": anchor,
        "sn_intercept_active_rows": sn_intercept,
        "h0_active_rows": h0,
        "host_and_global_rows": host & global_any,
        "host_and_anchor_rows": host & anchor,
        "host_and_cepheid_zp_rows": host & cepheid_zp,
        "host_and_sn_rows": host & sn_intercept,
        "host_and_h0_rows": host & h0,
        "sn_or_h0_rows": sn_intercept | h0,
        "host_sn_or_h0_rows": host & (sn_intercept | h0),
        "sparse_rows_active_1_to_2": active_count <= 2,
        "moderate_rows_active_3_to_5": (active_count >= 3) & (active_count <= 5),
        "dense_rows_active_6_plus": active_count >= 6,
    }


def build_gate_masks(Y, M):
    masks = active_theta_masks(M)

    positive_y = Y > 10.0
    near_zero_y = np.abs(Y) <= 1.0
    negative_y = Y < -10.0
    intermediate_y = ~(positive_y | near_zero_y | negative_y)

    masks.update(
        {
            "positive_y_distance_like_rows": positive_y,
            "near_zero_constraint_rows": near_zero_y,
            "negative_y_absolute_mag_like_rows": negative_y,
            "intermediate_y_rows": intermediate_y,
            "host_positive_y_rows": masks["host_active_rows"] & positive_y,
            "anchor_positive_y_rows": masks["anchor_active_rows"] & positive_y,
            "sn_positive_y_rows": masks["sn_intercept_active_rows"] & positive_y,
            "h0_positive_y_rows": masks["h0_active_rows"] & positive_y,
            "host_global_positive_y_rows": masks["host_and_global_rows"] & positive_y,
            "host_sparse_rows": masks["host_active_rows"] & masks["sparse_rows_active_1_to_2"],
            "host_moderate_rows": masks["host_active_rows"] & masks["moderate_rows_active_3_to_5"],
            "host_dense_rows": masks["host_active_rows"] & masks["dense_rows_active_6_plus"],
        }
    )

    cleaned = {}
    for name, mask in masks.items():
        if int(np.sum(mask)) > 0:
            cleaned[name] = mask.astype(bool)

    return cleaned


def make_uniform_gate_delta(mask, amplitude, sign):
    """
    Convert a fractional distance gate into a magnitude-space perturbation.

    For a local shortening gate G = 1 - A:
    delta_mu = 5 log10(G), which is negative.

    sign = -1 means distance-shortening direction.
    sign = +1 means distance-lengthening mirror check.
    """
    if sign < 0:
        delta_mag = 5.0 * math.log10(max(1.0e-9, 1.0 - amplitude))
    else:
        delta_mag = -5.0 * math.log10(max(1.0e-9, 1.0 - amplitude))

    return delta_mag * mask.astype(float), delta_mag


def make_column_direction_delta(M, theta_index, theta_shift):
    return M[:, theta_index] * theta_shift


def weighted_norm(vec, c_factor):
    solved = cho_solve(c_factor, vec, check_finite=False)
    return float(vec @ solved), solved


def project_gate_vector(
    delta_y,
    free_indices,
    M,
    A,
    c_factor,
    baseline_residual,
    baseline_Cinv_residual,
    baseline_chi2,
    theta_gls,
):
    free_indices = list(free_indices)
    delta_norm, Cinv_delta = weighted_norm(delta_y, c_factor)
    cross = float(2.0 * baseline_residual @ Cinv_delta)

    b_delta_full = M.T @ Cinv_delta

    if len(free_indices) == 0:
        projection_norm = 0.0
        delta_theta_free = np.asarray([], dtype=float)
        delta_theta_full = np.zeros_like(theta_gls)
    else:
        A_ff = A[np.ix_(free_indices, free_indices)]
        b_f = b_delta_full[free_indices]
        delta_theta_free = np.linalg.solve(A_ff, b_f)

        delta_theta_full = np.zeros_like(theta_gls)
        delta_theta_full[free_indices] = delta_theta_free

        projection_norm = float(b_f @ delta_theta_free)

    residual_norm_after_projection = max(0.0, float(delta_norm - projection_norm))
    chi2_after = float(baseline_chi2 + cross + residual_norm_after_projection)
    delta_chi2 = float(chi2_after - baseline_chi2)

    if delta_norm > 0.0:
        absorption_fraction = float(max(0.0, min(1.0, projection_norm / delta_norm)))
        residual_fraction = float(max(0.0, residual_norm_after_projection / delta_norm))
    else:
        absorption_fraction = float("nan")
        residual_fraction = float("nan")

    theta_new = theta_gls + delta_theta_full

    return {
        "delta_norm": delta_norm,
        "cross_with_baseline_residual": cross,
        "projection_norm": projection_norm,
        "residual_norm_after_projection": residual_norm_after_projection,
        "absorption_fraction": absorption_fraction,
        "residual_fraction": residual_fraction,
        "chi2_after_refit": chi2_after,
        "delta_chi2_after_refit": delta_chi2,
        "delta_theta_full": delta_theta_full,
        "theta_new": theta_new,
    }


def classify_projection_result(row):
    absorption = row["absorption_fraction"]
    delta_chi2 = row["delta_chi2_after_refit"]
    free_set = row["free_set"]

    if not math.isfinite(absorption):
        return "invalid_zero_norm_vector"

    if absorption >= 0.995 and abs(delta_chi2) <= 1.0:
        return "fully_absorbed_by_" + free_set

    if absorption >= 0.98 and abs(delta_chi2) <= 5.0:
        return "mostly_absorbed_by_" + free_set

    if absorption >= 0.90 and abs(delta_chi2) <= 20.0:
        return "partly_absorbed_by_" + free_set

    if delta_chi2 > 20.0:
        return "survives_as_non_offset_residual_against_" + free_set

    return "mixed_projection_response"


def summarize_theta_shift(delta_theta, theta_gls):
    host_delta = delta_theta[HOST_THETA]
    global_delta = delta_theta[GLOBAL_THETA]

    theta46_before = float(theta_gls[46])
    theta46_after = float(theta_gls[46] + delta_theta[46])
    h0_before = h0_from_theta46(theta46_before)
    h0_after = h0_from_theta46(theta46_after)

    return {
        "host_delta_rms": float(np.sqrt(np.mean(host_delta * host_delta))),
        "host_delta_abs_max": float(np.max(np.abs(host_delta))),
        "global_delta_rms": float(np.sqrt(np.mean(global_delta * global_delta))),
        "theta38_cepheid_zp_delta": float(delta_theta[38]),
        "theta40_anchor_delta": float(delta_theta[40]),
        "theta42_sn_intercept_delta": float(delta_theta[42]),
        "theta46_fivelogH0_delta": float(delta_theta[46]),
        "theta46_fivelogH0_before": theta46_before,
        "theta46_fivelogH0_after": theta46_after,
        "H0_before_from_theta46": h0_before,
        "H0_after_from_theta46": h0_after,
        "H0_delta_from_theta46": float(h0_after - h0_before),
    }


def run_projection_suite(Y, M, gls):
    c_factor = gls["c_factor"]
    A = gls["A"]
    theta_gls = gls["theta_gls"]
    baseline_residual = gls["residual"]
    baseline_Cinv_residual = gls["Cinv_residual"]
    baseline_chi2 = gls["chi2"]

    rows = []
    gate_masks = build_gate_masks(Y, M)

    mask_summary = {
        name: int(np.sum(mask))
        for name, mask in gate_masks.items()
    }

    candidate_vectors = []

    for mask_name, mask in gate_masks.items():
        for amplitude in GATE_AMPLITUDES:
            for sign_name, sign in [
                ("distance_shortening_negative_delta_mu", -1),
                ("distance_lengthening_positive_delta_mu", +1),
            ]:
                delta_y, delta_mag = make_uniform_gate_delta(mask, amplitude, sign)
                candidate_vectors.append(
                    {
                        "vector_name": f"{mask_name}__A{amplitude:.3f}__{sign_name}",
                        "vector_family": "uniform_mask_gate",
                        "mask_name": mask_name,
                        "row_count": int(np.sum(mask)),
                        "amplitude": float(amplitude),
                        "delta_mag_per_selected_row": float(delta_mag),
                        "theta_index_source": None,
                        "theta_shift_source": None,
                        "delta_y": delta_y,
                    }
                )

    for theta_index in [38, 40, 42, 46]:
        for theta_shift in COLUMN_SHIFT_MAGNITUDES:
            for signed_shift in [-theta_shift, theta_shift]:
                delta_y = make_column_direction_delta(M, theta_index, signed_shift)
                candidate_vectors.append(
                    {
                        "vector_name": f"exact_column_direction_theta{theta_index}__shift{signed_shift:+.3f}",
                        "vector_family": "exact_L_column_direction_sanity_check",
                        "mask_name": None,
                        "row_count": int(np.sum(np.abs(delta_y) > 1.0e-12)),
                        "amplitude": None,
                        "delta_mag_per_selected_row": None,
                        "theta_index_source": int(theta_index),
                        "theta_shift_source": float(signed_shift),
                        "delta_y": delta_y,
                    }
                )

    for candidate in candidate_vectors:
        delta_y = candidate["delta_y"]

        for free_set_name, free_indices in FREE_SETS.items():
            projection = project_gate_vector(
                delta_y=delta_y,
                free_indices=free_indices,
                M=M,
                A=A,
                c_factor=c_factor,
                baseline_residual=baseline_residual,
                baseline_Cinv_residual=baseline_Cinv_residual,
                baseline_chi2=baseline_chi2,
                theta_gls=theta_gls,
            )

            theta_shift_summary = summarize_theta_shift(projection["delta_theta_full"], theta_gls)

            row = {
                "vector_name": candidate["vector_name"],
                "vector_family": candidate["vector_family"],
                "mask_name": candidate["mask_name"],
                "row_count": candidate["row_count"],
                "amplitude": candidate["amplitude"],
                "delta_mag_per_selected_row": candidate["delta_mag_per_selected_row"],
                "theta_index_source": candidate["theta_index_source"],
                "theta_shift_source": candidate["theta_shift_source"],
                "free_set": free_set_name,
                "free_indices": " ".join(str(i) for i in free_indices),
                "free_parameter_count": len(free_indices),
                "delta_norm": projection["delta_norm"],
                "cross_with_baseline_residual": projection["cross_with_baseline_residual"],
                "projection_norm": projection["projection_norm"],
                "residual_norm_after_projection": projection["residual_norm_after_projection"],
                "absorption_fraction": projection["absorption_fraction"],
                "residual_fraction": projection["residual_fraction"],
                "chi2_after_refit": projection["chi2_after_refit"],
                "delta_chi2_after_refit": projection["delta_chi2_after_refit"],
                **theta_shift_summary,
            }

            row["projection_diagnostic"] = classify_projection_result(row)
            rows.append(row)

    return rows, mask_summary


def write_projection_csv(rows):
    path = OUTDIR / "shoes_compact_ladder_gate_insertion_dryrun_v1_projection_results.csv"

    fieldnames = list(rows[0].keys()) if rows else []

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_best_survivors_csv(rows):
    path = OUTDIR / "shoes_compact_ladder_gate_insertion_dryrun_v1_survivors_and_absorbed.csv"

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r["vector_family"] != "uniform_mask_gate",
            r["free_set"] != "all_47",
            -r["residual_fraction"],
            -r["delta_chi2_after_refit"],
        ),
    )

    selected = []

    for row in sorted_rows:
        if row["vector_family"] == "uniform_mask_gate" and row["free_set"] in [
            "all_47",
            "host_plus_sn_intercept_plus_h0",
            "host_plus_anchor_plus_cepheid_zp_plus_sn_plus_h0",
            "host_distance_only",
            "h0_only",
        ]:
            selected.append(row)

        if len(selected) >= 300:
            break

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    return path


def write_download_ledger(downloads):
    path = OUTDIR / "shoes_compact_ladder_gate_insertion_dryrun_v1_download_ledger.csv"

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


def inspect_support_files(downloads_by_name):
    out = []

    for name in ["read_chains_example.py", "MCMC_utils.py"]:
        item = downloads_by_name.get(name)
        if not item or not item.get("path"):
            out.append({"filename": name, "status": "missing"})
            continue

        path = Path(item["path"])
        text = path.read_text(errors="replace")
        lines = text.splitlines()

        hits = []
        for idx, line in enumerate(lines, start=1):
            lowered = line.lower()
            if any(term in lowered for term in ["h0", "theta", "fivelog", "log_likelihood", "chi2", "np.dot"]):
                hits.append({"line": idx, "text": line[:500]})

        out_path = OUTDIR / f"shoes_compact_ladder_gate_insertion_dryrun_v1_{name}_hits.json"
        out_path.write_text(json.dumps(hits, indent=2))

        out.append(
            {
                "filename": name,
                "status": "inspected",
                "line_count": len(lines),
                "hit_count": len(hits),
                "hits_json": str(out_path),
            }
        )

    return out


def summarize_results(rows):
    uniform_rows = [r for r in rows if r["vector_family"] == "uniform_mask_gate"]
    all47_uniform = [r for r in uniform_rows if r["free_set"] == "all_47"]

    if all47_uniform:
        mean_absorption_all47 = float(np.mean([r["absorption_fraction"] for r in all47_uniform]))
        min_absorption_all47 = float(np.min([r["absorption_fraction"] for r in all47_uniform]))
        max_delta_chi2_all47 = float(np.max([r["delta_chi2_after_refit"] for r in all47_uniform]))
        max_residual_fraction_all47 = float(np.max([r["residual_fraction"] for r in all47_uniform]))
    else:
        mean_absorption_all47 = None
        min_absorption_all47 = None
        max_delta_chi2_all47 = None
        max_residual_fraction_all47 = None

    by_free_set = {}
    for free_set in sorted(set(r["free_set"] for r in rows)):
        subset = [r for r in uniform_rows if r["free_set"] == free_set]
        if not subset:
            continue

        by_free_set[free_set] = {
            "count": len(subset),
            "mean_absorption_fraction": float(np.mean([r["absorption_fraction"] for r in subset])),
            "min_absorption_fraction": float(np.min([r["absorption_fraction"] for r in subset])),
            "max_absorption_fraction": float(np.max([r["absorption_fraction"] for r in subset])),
            "mean_residual_fraction": float(np.mean([r["residual_fraction"] for r in subset])),
            "max_delta_chi2": float(np.max([r["delta_chi2_after_refit"] for r in subset])),
            "min_delta_chi2": float(np.min([r["delta_chi2_after_refit"] for r in subset])),
        }

    top_survivors = sorted(
        [r for r in all47_uniform if r["amplitude"] == 0.085],
        key=lambda r: (-r["residual_fraction"], -r["delta_chi2_after_refit"]),
    )[:25]

    top_absorbed = sorted(
        [r for r in all47_uniform if r["amplitude"] == 0.085],
        key=lambda r: (-r["absorption_fraction"], abs(r["delta_chi2_after_refit"])),
    )[:25]

    if (
        min_absorption_all47 is not None
        and min_absorption_all47 >= 0.98
        and max_delta_chi2_all47 is not None
        and max_delta_chi2_all47 <= 5.0
    ):
        final_status = "tested_gate_vectors_are_mostly_offset_absorbed_by_all_47_ladder_refit"
        readiness_score = 8
        next_wall = "Need a physically mapped Cepheid/host-row gate, not generic row masks, to beat offset freedom."
    elif (
        max_residual_fraction_all47 is not None
        and max_residual_fraction_all47 >= 0.10
        and max_delta_chi2_all47 is not None
        and max_delta_chi2_all47 > 20.0
    ):
        final_status = "some_gate_vectors_survive_all_47_refit_as_non_offset_residuals"
        readiness_score = 8
        next_wall = "Inspect surviving vectors and map them to real Cepheid/anchor semantics before treating them as TAIRID-relevant."
    else:
        final_status = "mixed_gate_projection_response_needs_semantic_row_mapping"
        readiness_score = 7
        next_wall = "Map observation rows to Cepheid/anchor/host meaning before a physics interpretation."

    return {
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "mean_absorption_all47_uniform_gates": mean_absorption_all47,
        "min_absorption_all47_uniform_gates": min_absorption_all47,
        "max_delta_chi2_all47_uniform_gates": max_delta_chi2_all47,
        "max_residual_fraction_all47_uniform_gates": max_residual_fraction_all47,
        "by_free_set": by_free_set,
        "top_surviving_uniform_A0p085_all47": top_survivors,
        "top_absorbed_uniform_A0p085_all47": top_absorbed,
        "next_wall": next_wall,
    }


def make_plots(rows):
    plot_paths = []

    uniform = [r for r in rows if r["vector_family"] == "uniform_mask_gate"]
    all47 = [r for r in uniform if r["free_set"] == "all_47" and r["amplitude"] == 0.085]

    if all47:
        top = sorted(all47, key=lambda r: r["row_count"], reverse=True)[:30]
        labels = [r["mask_name"][:35] for r in top]
        absorption = [r["absorption_fraction"] for r in top]
        residual = [r["residual_fraction"] for r in top]
        pos = np.arange(len(top))

        plt.figure(figsize=(14, 6))
        plt.bar(pos - 0.2, absorption, width=0.4, label="absorbed")
        plt.bar(pos + 0.2, residual, width=0.4, label="residual")
        plt.xticks(pos, labels, rotation=35, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("fraction of weighted gate norm")
        plt.title("SH0ES compact ladder gate dry run v1: A=0.085 all-47 refit")
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = OUTDIR / "shoes_compact_ladder_gate_insertion_dryrun_v1_all47_absorption_A0p085.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths.append(str(path))

    free_sets = sorted(set(r["free_set"] for r in uniform))
    fs_labels = []
    fs_abs = []

    for fs in free_sets:
        subset = [r for r in uniform if r["free_set"] == fs]
        if subset:
            fs_labels.append(fs[:35])
            fs_abs.append(float(np.mean([r["absorption_fraction"] for r in subset])))

    if fs_labels:
        pos = np.arange(len(fs_labels))

        plt.figure(figsize=(14, 6))
        plt.bar(pos, fs_abs)
        plt.xticks(pos, fs_labels, rotation=35, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("mean absorption fraction")
        plt.title("SH0ES compact ladder gate dry run v1: mean absorption by refit freedom")
        plt.tight_layout()
        path = OUTDIR / "shoes_compact_ladder_gate_insertion_dryrun_v1_absorption_by_free_set.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths.append(str(path))

    h0_rows = [
        r for r in uniform
        if r["free_set"] == "all_47" and r["amplitude"] == 0.085
    ]

    if h0_rows:
        labels = [r["mask_name"][:35] for r in h0_rows[:40]]
        h0_delta = [r["H0_delta_from_theta46"] for r in h0_rows[:40]]
        pos = np.arange(len(labels))

        plt.figure(figsize=(14, 6))
        plt.bar(pos, h0_delta)
        plt.xticks(pos, labels, rotation=35, ha="right")
        plt.ylabel("H0 shift from theta46 after all-47 refit")
        plt.title("SH0ES compact ladder gate dry run v1: induced H0 shift")
        plt.tight_layout()
        path = OUTDIR / "shoes_compact_ladder_gate_insertion_dryrun_v1_h0_shift_A0p085.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths.append(str(path))

    return plot_paths


def main():
    print("")
    print("TAIRID SH0ES compact ladder gate insertion dry run v1 starting.")
    print("Boundary: projection/perturbation dry run only, not proof and not a final SH0ES result.")
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

    (OUTDIR / "shoes_compact_ladder_gate_insertion_dryrun_v1_github_api_listing.json").write_text(
        json.dumps(api_summary, indent=2)
    )

    downloads = []

    for filename in PRIMARY_FILES:
        print(f"Downloading / locating {filename} ...")
        result = download_or_copy(filename, api_listing_by_name)
        downloads.append(result)
        print(f"  {result.get('status')}")

    downloads_by_name = {item["filename"]: item for item in downloads}
    download_ledger_path = write_download_ledger(downloads)

    Y, L, M, C, theta_public, theta_sigma, primary_paths = load_primary_system(downloads_by_name)

    gls = generalized_least_squares_setup(Y, M, C)
    public_fit = chi2_for_theta(Y, M, gls["c_factor"], theta_public)

    projection_rows, mask_summary = run_projection_suite(Y, M, gls)

    projection_csv = write_projection_csv(projection_rows)
    survivors_csv = write_best_survivors_csv(projection_rows)

    support_summary = inspect_support_files(downloads_by_name)
    result_summary = summarize_results(projection_rows)
    plot_paths = make_plots(projection_rows)

    theta_difference = gls["theta_gls"] - theta_public

    baseline_summary = {
        "Y_shape": list(Y.shape),
        "L_shape": list(L.shape),
        "M_shape": list(M.shape),
        "C_shape": list(C.shape),
        "theta_count": int(theta_public.size),
        "public_theta_chi2": public_fit["chi2"],
        "gls_refit_chi2": gls["chi2"],
        "gls_refit_dof": gls["dof"],
        "gls_refit_reduced_chi2": gls["reduced_chi2"],
        "public_minus_gls_chi2": float(public_fit["chi2"] - gls["chi2"]),
        "theta_public_46_fivelogH0": float(theta_public[46]),
        "theta_gls_46_fivelogH0": float(gls["theta_gls"][46]),
        "H0_public_from_theta46": h0_from_theta46(theta_public[46]),
        "H0_gls_from_theta46": h0_from_theta46(gls["theta_gls"][46]),
        "theta_public_vs_gls_delta_rms": float(np.sqrt(np.mean(theta_difference ** 2))),
        "theta_public_vs_gls_delta_abs_max": float(np.max(np.abs(theta_difference))),
    }

    baseline_path = OUTDIR / "shoes_compact_ladder_gate_insertion_dryrun_v1_baseline_audit.json"
    baseline_path.write_text(json.dumps(baseline_summary, indent=2))

    summary = {
        "test_name": "TAIRID SH0ES compact ladder gate insertion dry run v1",
        "boundary": (
            "Projection/perturbation dry run only. Not a final SH0ES likelihood analysis, "
            "not proof of TAIRID cosmology, and not a Hubble-tension solution claim."
        ),
        "final_status": result_summary["final_status"],
        "readiness_score_0_to_10": result_summary["readiness_score_0_to_10"],
        "primary_paths": primary_paths,
        "github_api_listing": api_summary,
        "downloads": downloads,
        "baseline": baseline_summary,
        "mask_summary": mask_summary,
        "result_summary": result_summary,
        "support_summary": support_summary,
        "free_sets": FREE_SETS,
        "output_files": {
            "download_ledger_csv": str(download_ledger_path),
            "baseline_audit_json": str(baseline_path),
            "projection_results_csv": str(projection_csv),
            "survivors_and_absorbed_csv": str(survivors_csv),
            "plots": plot_paths,
        },
    }

    summary_path = OUTDIR / "shoes_compact_ladder_gate_insertion_dryrun_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    with open(OUTDIR / "shoes_compact_ladder_gate_insertion_dryrun_v1_summary.txt", "w") as f:
        f.write("TAIRID SH0ES compact ladder gate insertion dry run v1\n\n")
        f.write("Boundary: dry-run perturbation/projection test only. Not proof. Not a final SH0ES result.\n\n")
        f.write(f"Final status: {result_summary['final_status']}\n")
        f.write(f"Readiness score: {result_summary['readiness_score_0_to_10']}/10\n\n")

        f.write("Baseline compact ladder audit:\n")
        f.write(json.dumps(baseline_summary, indent=2) + "\n\n")

        f.write("Mask summary:\n")
        f.write(json.dumps(mask_summary, indent=2) + "\n\n")

        f.write("Projection result summary:\n")
        f.write(json.dumps(result_summary, indent=2) + "\n\n")

        f.write("Interpretation guide:\n")
        f.write("- High all-47 absorption means the candidate gate is ordinary ladder freedom unless better row semantics are added.\n")
        f.write("- High residual after all-47 refit means the vector is not just an offset, but it still needs physical row mapping.\n")
        f.write("- H0-only absorption tests whether the vector is basically theta46/five-log-H0 movement.\n")
        f.write("- Host-only absorption tests whether the vector is basically host-distance freedom.\n")
        f.write("- Host + SN + H0 absorption tests whether it is only ordinary ladder calibration freedom.\n")
        f.write("- Do not claim TAIRID solved Hubble tension from this dry run.\n")

    print("")
    print("TAIRID SH0ES compact ladder gate insertion dry run v1 complete.")
    print("Created:")
    print("  shoes_compact_ladder_gate_insertion_dryrun_v1_outputs/shoes_compact_ladder_gate_insertion_dryrun_v1_summary.json")
    print("  shoes_compact_ladder_gate_insertion_dryrun_v1_outputs/shoes_compact_ladder_gate_insertion_dryrun_v1_summary.txt")
    print("  shoes_compact_ladder_gate_insertion_dryrun_v1_outputs/shoes_compact_ladder_gate_insertion_dryrun_v1_baseline_audit.json")
    print("  shoes_compact_ladder_gate_insertion_dryrun_v1_outputs/shoes_compact_ladder_gate_insertion_dryrun_v1_projection_results.csv")
    print("  shoes_compact_ladder_gate_insertion_dryrun_v1_outputs/shoes_compact_ladder_gate_insertion_dryrun_v1_survivors_and_absorbed.csv")
    print("")
    print("Boundary:")
    print("  This is not a final SH0ES likelihood analysis.")
    print("  This is not proof of TAIRID cosmology.")
    print("  This is a compact-ladder gate projection dry run.")
    print("")
    print(f"Final status: {result_summary['final_status']}")
    print(f"Readiness score: {result_summary['readiness_score_0_to_10']}/10")


if __name__ == "__main__":
    main()
