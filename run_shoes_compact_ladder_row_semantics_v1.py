#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder row-semantics audit v1.

Purpose:
The previous compact-ladder gate-insertion dry run found a mixed projection response.
Most generic gate vectors were absorbed by ordinary SH0ES ladder refitting, but some
sparse/moderate equation-row classes left non-offset residual pressure.

This test does not push the gate harder.

It asks:
Which of the 3492 compact SH0ES rows/equations are driving that behavior?

Method:
1. Load the public SH0ES compact system:
   - C = allc covariance matrix
   - L = alll equation/design matrix
   - y = ally data vector
   - theta = lstsq_results first column
2. Recompute baseline generalized least-squares fit.
3. Classify all 3492 rows by:
   - active theta indices
   - coefficient pattern
   - y-value class
   - host-distance involvement
   - anchor involvement
   - Cepheid-zero-point involvement
   - SN-intercept involvement
   - H0/five-log-H0 involvement
   - sparse / moderate / dense active-parameter structure
4. Rebuild the key A=0.085 dry-run vectors:
   - sparse_rows_active_1_to_2
   - moderate_rows_active_3_to_5
   - host_sparse_rows
   - host_moderate_rows
   - positive_y_distance_like_rows
   - near_zero_constraint_rows
5. Project each vector through all 47 parameters and identify the rows that dominate
   the surviving non-offset residual.
6. Produce row-semantics and row-driver CSV files for the next physically mapped gate.

Boundary:
This is not a TAIRID gate claim.
This is not a new SH0ES result.
This is not a cosmology fit.
This does not prove TAIRID cosmology.
It is a row-semantics audit before physical interpretation.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.linalg import cho_factor, cho_solve


OUTDIR = Path("shoes_compact_ladder_row_semantics_v1_outputs")
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

HOST_THETA = list(range(0, 37))
GLOBAL_THETA = list(range(37, 47))
CEPHEID_ZEROPOINT_THETA = [38]
ANCHOR_THETA = [40]
SN_INTERCEPT_THETA = [42]
FIVELOGH0_THETA = [46]

A_DRYRUN = 0.085


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
            "User-Agent": "TAIRID-SH0ES-row-semantics-v1",
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
    paths = {}

    for name in PRIMARY_FILES:
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
    A_inv = np.linalg.inv(A)
    b = M.T @ Cinv_Y

    theta_gls = A_inv @ b
    residual = Y - M @ theta_gls
    Cinv_residual = cho_solve(c_factor, residual, check_finite=False)

    chi2 = float(residual @ Cinv_residual)
    dof = int(Y.size - theta_gls.size)

    leverage = np.einsum("ij,jk,ik->i", M, A_inv, Cinv_M)

    return {
        "c_factor": c_factor,
        "Cinv_M": Cinv_M,
        "A": A,
        "A_inv": A_inv,
        "b": b,
        "theta_gls": theta_gls,
        "residual": residual,
        "Cinv_residual": Cinv_residual,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": float(chi2 / dof),
        "leverage_diag": leverage,
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


def y_class(Y_value):
    if Y_value > 10.0:
        return "positive_distance_or_magnitude_like"
    if Y_value < -10.0:
        return "negative_absolute_magnitude_like"
    if abs(Y_value) <= 1.0:
        return "near_zero_constraint_like"
    return "intermediate_transformed_like"


def active_count_class(active_count):
    if active_count <= 2:
        return "sparse_1_to_2"
    if active_count <= 5:
        return "moderate_3_to_5"
    return "dense_6_plus"


def relation_candidate(active_indices):
    active_set = set(active_indices)
    host = sorted(active_set.intersection(HOST_THETA))
    global_terms = sorted(active_set.intersection(GLOBAL_THETA))
    has_host = len(host) > 0
    has_cepheid_zp = 38 in active_set
    has_anchor = 40 in active_set
    has_sn = 42 in active_set
    has_h0 = 46 in active_set

    if has_h0 and has_sn and has_host:
        return "host_SN_H0_link_candidate"
    if has_h0 and not has_host:
        return "H0_global_constraint_candidate"
    if has_sn and has_host:
        return "host_SN_calibration_relation_candidate"
    if has_anchor and has_host:
        return "host_anchor_relation_candidate"
    if has_cepheid_zp and has_host:
        return "host_Cepheid_relation_candidate"
    if has_host and global_terms:
        return "host_global_ladder_relation_candidate"
    if has_host:
        return "host_distance_only_or_host_prior_candidate"
    if global_terms:
        return "global_or_nuisance_relation_candidate"
    return "unmapped_relation_candidate"


def active_signature(active_indices):
    hosts = [i for i in active_indices if i in HOST_THETA]
    globals_ = [i for i in active_indices if i in GLOBAL_THETA]
    return f"H[{','.join(map(str, hosts))}]__G[{','.join(map(str, globals_))}]"


def classify_rows(Y, M, C_diag, gls):
    rows = []
    active_bool = np.abs(M) > 1.0e-10
    residual = gls["residual"]
    leverage = gls["leverage_diag"]

    for j in range(Y.size):
        active = np.where(active_bool[j])[0]
        coeffs = M[j, active]

        host_terms = [int(i) for i in active if i in HOST_THETA]
        global_terms = [int(i) for i in active if i in GLOBAL_THETA]

        flags = {
            "has_host": len(host_terms) > 0,
            "has_global": len(global_terms) > 0,
            "has_cepheid_zp_theta38": 38 in active,
            "has_anchor_theta40": 40 in active,
            "has_sn_intercept_theta42": 42 in active,
            "has_fivelogH0_theta46": 46 in active,
        }

        rows.append(
            {
                "row_index": int(j),
                "Y_value": float(Y[j]),
                "Y_class": y_class(Y[j]),
                "baseline_residual": float(residual[j]),
                "baseline_residual_abs": float(abs(residual[j])),
                "C_diag": float(C_diag[j]),
                "C_sigma": float(math.sqrt(C_diag[j])) if C_diag[j] >= 0 else None,
                "leverage_diag_approx": float(leverage[j]),
                "active_theta_count": int(active.size),
                "active_count_class": active_count_class(active.size),
                "active_theta_indices": " ".join(str(int(i)) for i in active),
                "active_coefficients": " ".join(f"{float(c):.8g}" for c in coeffs),
                "host_theta_indices": " ".join(str(i) for i in host_terms),
                "global_theta_indices": " ".join(str(i) for i in global_terms),
                "active_signature": active_signature(active),
                "relation_candidate": relation_candidate(active),
                **flags,
                "coeff_min": float(np.min(coeffs)) if coeffs.size else None,
                "coeff_max": float(np.max(coeffs)) if coeffs.size else None,
                "coeff_abs_max": float(np.max(np.abs(coeffs))) if coeffs.size else None,
                "coeff_sum": float(np.sum(coeffs)) if coeffs.size else None,
                "coeff_l2": float(np.sqrt(np.sum(coeffs * coeffs))) if coeffs.size else None,
            }
        )

    return rows


def make_gate_masks(row_rows):
    n = len(row_rows)

    def mask_from(predicate):
        return np.asarray([bool(predicate(r)) for r in row_rows], dtype=bool)

    masks = {
        "sparse_rows_active_1_to_2": mask_from(lambda r: r["active_count_class"] == "sparse_1_to_2"),
        "moderate_rows_active_3_to_5": mask_from(lambda r: r["active_count_class"] == "moderate_3_to_5"),
        "dense_rows_active_6_plus": mask_from(lambda r: r["active_count_class"] == "dense_6_plus"),
        "host_sparse_rows": mask_from(lambda r: r["has_host"] and r["active_count_class"] == "sparse_1_to_2"),
        "host_moderate_rows": mask_from(lambda r: r["has_host"] and r["active_count_class"] == "moderate_3_to_5"),
        "host_dense_rows": mask_from(lambda r: r["has_host"] and r["active_count_class"] == "dense_6_plus"),
        "positive_y_distance_like_rows": mask_from(lambda r: r["Y_class"] == "positive_distance_or_magnitude_like"),
        "near_zero_constraint_rows": mask_from(lambda r: r["Y_class"] == "near_zero_constraint_like"),
        "negative_absolute_mag_like_rows": mask_from(lambda r: r["Y_class"] == "negative_absolute_magnitude_like"),
        "host_cepheid_relation_candidate_rows": mask_from(lambda r: r["relation_candidate"] == "host_Cepheid_relation_candidate"),
        "host_anchor_relation_candidate_rows": mask_from(lambda r: r["relation_candidate"] == "host_anchor_relation_candidate"),
        "host_SN_calibration_relation_candidate_rows": mask_from(lambda r: r["relation_candidate"] == "host_SN_calibration_relation_candidate"),
        "host_SN_H0_link_candidate_rows": mask_from(lambda r: r["relation_candidate"] == "host_SN_H0_link_candidate"),
        "H0_global_constraint_candidate_rows": mask_from(lambda r: r["relation_candidate"] == "H0_global_constraint_candidate"),
    }

    return {name: mask for name, mask in masks.items() if int(np.sum(mask)) > 0}


def uniform_gate_delta(mask, amplitude):
    delta_mag = 5.0 * math.log10(max(1.0e-9, 1.0 - amplitude))
    return delta_mag * mask.astype(float), delta_mag


def project_delta_all47(delta_y, M, gls):
    c_factor = gls["c_factor"]
    A = gls["A"]
    theta = gls["theta_gls"]

    Cinv_delta = cho_solve(c_factor, delta_y, check_finite=False)
    delta_norm = float(delta_y @ Cinv_delta)

    b_delta = M.T @ Cinv_delta
    delta_theta = np.linalg.solve(A, b_delta)

    projected = M @ delta_theta
    surviving = delta_y - projected
    Cinv_surviving = cho_solve(c_factor, surviving, check_finite=False)

    projection_norm = float(projected @ cho_solve(c_factor, projected, check_finite=False))
    surviving_norm = float(surviving @ Cinv_surviving)
    absorption_fraction = projection_norm / delta_norm if delta_norm > 0 else float("nan")
    residual_fraction = surviving_norm / delta_norm if delta_norm > 0 else float("nan")

    baseline_cross = float(2.0 * gls["residual"] @ Cinv_surviving)
    delta_chi2_after_refit = float(baseline_cross + surviving_norm)

    theta_new = theta + delta_theta
    h0_before = h0_from_theta46(theta[46])
    h0_after = h0_from_theta46(theta_new[46])

    per_row_signed_survival_contrib = surviving * Cinv_surviving
    per_row_abs_survival_contrib = np.abs(per_row_signed_survival_contrib)

    return {
        "delta_norm": delta_norm,
        "projection_norm": projection_norm,
        "surviving_norm": surviving_norm,
        "absorption_fraction": absorption_fraction,
        "residual_fraction": residual_fraction,
        "baseline_cross_after_projection": baseline_cross,
        "delta_chi2_after_refit": delta_chi2_after_refit,
        "delta_theta": delta_theta,
        "theta46_delta": float(delta_theta[46]),
        "H0_before": h0_before,
        "H0_after": h0_after,
        "H0_delta": float(h0_after - h0_before),
        "surviving_vector": surviving,
        "Cinv_surviving_vector": Cinv_surviving,
        "per_row_signed_survival_contrib": per_row_signed_survival_contrib,
        "per_row_abs_survival_contrib": per_row_abs_survival_contrib,
    }


def row_driver_audit(row_rows, M, gls):
    masks = make_gate_masks(row_rows)
    summaries = []
    drivers = []

    for mask_name, mask in masks.items():
        delta_y, delta_mag = uniform_gate_delta(mask, A_DRYRUN)
        projection = project_delta_all47(delta_y, M, gls)

        selected = np.where(mask)[0]
        abs_contrib = projection["per_row_abs_survival_contrib"]
        signed_contrib = projection["per_row_signed_survival_contrib"]
        surviving = projection["surviving_vector"]

        top_indices = np.argsort(-abs_contrib)[:250]

        selected_contrib_abs_sum = float(np.sum(abs_contrib[selected])) if selected.size else 0.0
        total_contrib_abs_sum = float(np.sum(abs_contrib))

        summaries.append(
            {
                "mask_name": mask_name,
                "selected_row_count": int(np.sum(mask)),
                "delta_mag_per_selected_row": float(delta_mag),
                "delta_norm": projection["delta_norm"],
                "projection_norm": projection["projection_norm"],
                "surviving_norm": projection["surviving_norm"],
                "absorption_fraction_all47": projection["absorption_fraction"],
                "residual_fraction_all47": projection["residual_fraction"],
                "baseline_cross_after_projection": projection["baseline_cross_after_projection"],
                "delta_chi2_after_all47_refit": projection["delta_chi2_after_refit"],
                "theta46_delta": projection["theta46_delta"],
                "H0_before": projection["H0_before"],
                "H0_after": projection["H0_after"],
                "H0_delta": projection["H0_delta"],
                "selected_rows_abs_contribution_fraction": (
                    selected_contrib_abs_sum / total_contrib_abs_sum
                    if total_contrib_abs_sum > 0 else None
                ),
            }
        )

        for rank, idx in enumerate(top_indices, start=1):
            r = row_rows[int(idx)]
            drivers.append(
                {
                    "mask_name": mask_name,
                    "rank": rank,
                    "row_index": int(idx),
                    "selected_by_mask": bool(mask[int(idx)]),
                    "surviving_value": float(surviving[int(idx)]),
                    "signed_survival_contribution": float(signed_contrib[int(idx)]),
                    "abs_survival_contribution": float(abs_contrib[int(idx)]),
                    "fraction_of_abs_contribution": (
                        float(abs_contrib[int(idx)] / total_contrib_abs_sum)
                        if total_contrib_abs_sum > 0 else None
                    ),
                    "Y_value": r["Y_value"],
                    "Y_class": r["Y_class"],
                    "baseline_residual": r["baseline_residual"],
                    "C_diag": r["C_diag"],
                    "C_sigma": r["C_sigma"],
                    "leverage_diag_approx": r["leverage_diag_approx"],
                    "active_theta_count": r["active_theta_count"],
                    "active_count_class": r["active_count_class"],
                    "active_theta_indices": r["active_theta_indices"],
                    "active_coefficients": r["active_coefficients"],
                    "host_theta_indices": r["host_theta_indices"],
                    "global_theta_indices": r["global_theta_indices"],
                    "active_signature": r["active_signature"],
                    "relation_candidate": r["relation_candidate"],
                    "has_host": r["has_host"],
                    "has_cepheid_zp_theta38": r["has_cepheid_zp_theta38"],
                    "has_anchor_theta40": r["has_anchor_theta40"],
                    "has_sn_intercept_theta42": r["has_sn_intercept_theta42"],
                    "has_fivelogH0_theta46": r["has_fivelogH0_theta46"],
                }
            )

    return summaries, drivers, masks


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return path

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def summarize_row_classes(row_rows):
    groups = defaultdict(list)

    for r in row_rows:
        key = (
            r["active_count_class"],
            r["Y_class"],
            r["relation_candidate"],
        )
        groups[key].append(r)

    out = []

    for key, rows in groups.items():
        active_class, yclass, relation = key
        residuals = np.asarray([r["baseline_residual"] for r in rows], dtype=float)
        leverage = np.asarray([r["leverage_diag_approx"] for r in rows], dtype=float)
        cdiag = np.asarray([r["C_diag"] for r in rows], dtype=float)

        out.append(
            {
                "active_count_class": active_class,
                "Y_class": yclass,
                "relation_candidate": relation,
                "row_count": len(rows),
                "residual_mean": float(np.mean(residuals)),
                "residual_rms": float(np.sqrt(np.mean(residuals * residuals))),
                "residual_abs_max": float(np.max(np.abs(residuals))),
                "leverage_mean": float(np.mean(leverage)),
                "leverage_max": float(np.max(leverage)),
                "C_diag_mean": float(np.mean(cdiag)),
                "C_diag_min": float(np.min(cdiag)),
                "C_diag_max": float(np.max(cdiag)),
            }
        )

    return sorted(out, key=lambda r: (-r["row_count"], r["active_count_class"], r["relation_candidate"]))


def inspect_support_files(downloads_by_name):
    rows = []
    summaries = []

    terms = [
        "h0", "theta", "fivelog", "chi2", "likelihood", "cepheid", "host",
        "anchor", "table", "parameter", "m_h", "m_b", "period", "metal",
        "n4258", "lmc", "mw", "m31", "calibrator"
    ]

    for name in SUPPORT_FILES:
        item = downloads_by_name.get(name)
        if not item or not item.get("path"):
            summaries.append({"filename": name, "status": "missing"})
            continue

        path = Path(item["path"])
        if not path.exists():
            summaries.append({"filename": name, "status": "missing_path"})
            continue

        try:
            text = path.read_text(errors="replace")
        except Exception as exc:
            summaries.append({"filename": name, "status": "read_failed", "error": str(exc)})
            continue

        lines = text.splitlines()
        hit_count = 0

        for idx, line in enumerate(lines, start=1):
            lowered = line.lower()
            if any(t in lowered for t in terms):
                rows.append({"filename": name, "line": idx, "text": line[:800]})
                hit_count += 1
            if hit_count >= 250:
                break

        summaries.append(
            {
                "filename": name,
                "status": "inspected",
                "line_count": len(lines),
                "size_bytes": path.stat().st_size,
                "hit_count_capped": hit_count,
            }
        )

    hits_path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_support_hits.csv"
    write_csv(hits_path, rows)

    return summaries, str(hits_path)


def parse_optical_wes(downloads_by_name):
    files = [
        "optical_wes_R22_for19fromR16.dat",
        "optical_wes_R22_for19fromR16.wM31.dat",
    ]

    summaries = []
    host_rows = []

    for name in files:
        item = downloads_by_name.get(name)
        if not item or not item.get("path"):
            summaries.append({"filename": name, "status": "missing"})
            continue

        path = Path(item["path"])
        text = path.read_text(errors="replace")
        rows = []

        for line in text.splitlines():
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

        host_counts = Counter(r["Host"] for r in rows)

        summaries.append(
            {
                "filename": name,
                "status": "parsed" if rows else "no_rows_parsed",
                "row_count": len(rows),
                "host_count": len(host_counts),
                "top_hosts": host_counts.most_common(25),
            }
        )

        for host, count in host_counts.most_common():
            host_rows.append(
                {
                    "filename": name,
                    "host": host,
                    "count": int(count),
                }
            )

    host_counts_path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_optical_host_counts.csv"
    write_csv(host_counts_path, host_rows)

    return summaries, str(host_counts_path)


def write_download_ledger(downloads):
    rows = []

    for item in downloads:
        pointer = item.get("pointer", {})
        rows.append(
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

    path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_download_ledger.csv"
    write_csv(path, rows)
    return path


def make_plots(row_rows, driver_summaries):
    plot_paths = []

    row_index = np.asarray([r["row_index"] for r in row_rows])
    y_values = np.asarray([r["Y_value"] for r in row_rows])
    active_counts = np.asarray([r["active_theta_count"] for r in row_rows])
    residuals = np.asarray([r["baseline_residual"] for r in row_rows])
    leverage = np.asarray([r["leverage_diag_approx"] for r in row_rows])

    plt.figure(figsize=(12, 6))
    plt.scatter(row_index, y_values, s=5, alpha=0.35)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("row index")
    plt.ylabel("Y value")
    plt.title("SH0ES compact ladder row semantics v1: y vector by row")
    plt.tight_layout()
    path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_y_by_row.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    plt.figure(figsize=(12, 6))
    plt.scatter(row_index, active_counts, s=5, alpha=0.35)
    plt.xlabel("row index")
    plt.ylabel("active theta count")
    plt.title("SH0ES compact ladder row semantics v1: active parameter count")
    plt.tight_layout()
    path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_active_count_by_row.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    plt.figure(figsize=(12, 6))
    plt.scatter(row_index, residuals, s=5, alpha=0.35)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("row index")
    plt.ylabel("baseline residual")
    plt.title("SH0ES compact ladder row semantics v1: baseline residual by row")
    plt.tight_layout()
    path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_residual_by_row.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    plt.figure(figsize=(12, 6))
    plt.scatter(row_index, leverage, s=5, alpha=0.35)
    plt.xlabel("row index")
    plt.ylabel("weighted leverage diag approximation")
    plt.title("SH0ES compact ladder row semantics v1: row leverage")
    plt.tight_layout()
    path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_leverage_by_row.png"
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    if driver_summaries:
        labels = [r["mask_name"][:36] for r in driver_summaries]
        residual_fraction = [r["residual_fraction_all47"] for r in driver_summaries]
        delta_chi2 = [r["delta_chi2_after_all47_refit"] for r in driver_summaries]
        pos = np.arange(len(labels))

        plt.figure(figsize=(14, 6))
        plt.bar(pos, residual_fraction)
        plt.xticks(pos, labels, rotation=35, ha="right")
        plt.ylabel("residual fraction after all-47 projection")
        plt.title("SH0ES row semantics v1: surviving non-offset fraction by vector")
        plt.tight_layout()
        path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_residual_fraction_by_vector.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths.append(str(path))

        plt.figure(figsize=(14, 6))
        plt.bar(pos, delta_chi2)
        plt.xticks(pos, labels, rotation=35, ha="right")
        plt.ylabel("delta chi-square after all-47 refit")
        plt.title("SH0ES row semantics v1: delta chi-square by vector")
        plt.tight_layout()
        path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_delta_chi2_by_vector.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths.append(str(path))

    return plot_paths


def main():
    print("")
    print("TAIRID SH0ES compact ladder row-semantics audit v1 starting.")
    print("Boundary: row semantics only, no new gate claim.")
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

    (OUTDIR / "shoes_compact_ladder_row_semantics_v1_github_api_listing.json").write_text(
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

    Y, L, M, C, theta_public, theta_sigma, primary_paths = load_primary_system(downloads_by_name)

    gls = generalized_least_squares_setup(Y, M, C)
    public_fit = chi2_for_theta(Y, M, gls["c_factor"], theta_public)

    C_diag = np.diag(C)
    row_rows = classify_rows(Y, M, C_diag, gls)

    row_semantics_path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_rows.csv"
    write_csv(row_semantics_path, row_rows)

    class_summary_rows = summarize_row_classes(row_rows)
    class_summary_path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_class_summary.csv"
    write_csv(class_summary_path, class_summary_rows)

    driver_summaries, driver_rows, masks = row_driver_audit(row_rows, M, gls)

    driver_summary_path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_driver_summary.csv"
    write_csv(driver_summary_path, driver_summaries)

    driver_rows_path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_top_row_drivers.csv"
    write_csv(driver_rows_path, driver_rows)

    support_summaries, support_hits_csv = inspect_support_files(downloads_by_name)
    optical_summaries, optical_host_counts_csv = parse_optical_wes(downloads_by_name)

    plot_paths = make_plots(row_rows, driver_summaries)

    role_counts = Counter(r["relation_candidate"] for r in row_rows)
    active_counts = Counter(r["active_count_class"] for r in row_rows)
    y_counts = Counter(r["Y_class"] for r in row_rows)

    key_vectors = {
        r["mask_name"]: r
        for r in driver_summaries
        if r["mask_name"] in [
            "sparse_rows_active_1_to_2",
            "moderate_rows_active_3_to_5",
            "host_sparse_rows",
            "host_moderate_rows",
            "positive_y_distance_like_rows",
            "near_zero_constraint_rows",
        ]
    }

    moderate = key_vectors.get("moderate_rows_active_3_to_5")
    sparse = key_vectors.get("sparse_rows_active_1_to_2")

    if moderate and moderate["residual_fraction_all47"] >= 0.02:
        final_status = "row_semantics_built_moderate_rows_retain_non_offset_residual_pressure"
        readiness_score = 8
        next_wall = "Build a physically mapped gate using only rows whose relation candidate is identified, then compare against host/SN/H0 freedom."
    elif sparse and sparse["delta_chi2_after_all47_refit"] > 20:
        final_status = "row_semantics_built_sparse_rows_are_stiff_but_need_physical_mapping"
        readiness_score = 7
        next_wall = "Inspect sparse-row drivers manually before interpreting any TAIRID survival."
    else:
        final_status = "row_semantics_built_generic_gate_residuals_mostly_absorbed"
        readiness_score = 7
        next_wall = "Use semantic row groups rather than generic sparse/moderate masks."

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
    }

    summary = {
        "test_name": "TAIRID SH0ES compact ladder row-semantics audit v1",
        "boundary": (
            "Row semantics only. No TAIRID gate claim. Not a new SH0ES result. "
            "Not a cosmology fit and not proof."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "primary_paths": primary_paths,
        "github_api_listing": api_summary,
        "downloads": downloads,
        "baseline_summary": baseline_summary,
        "row_counts": {
            "total_rows": len(row_rows),
            "active_count_class_counts": dict(active_counts),
            "Y_class_counts": dict(y_counts),
            "relation_candidate_counts": dict(role_counts),
        },
        "driver_key_vectors": key_vectors,
        "driver_summaries": driver_summaries,
        "support_summaries": support_summaries,
        "optical_wes_summaries": optical_summaries,
        "output_files": {
            "download_ledger_csv": str(download_ledger),
            "row_semantics_csv": str(row_semantics_path),
            "class_summary_csv": str(class_summary_path),
            "driver_summary_csv": str(driver_summary_path),
            "top_row_drivers_csv": str(driver_rows_path),
            "support_hits_csv": support_hits_csv,
            "optical_host_counts_csv": optical_host_counts_csv,
            "plots": plot_paths,
        },
    }

    summary_path = OUTDIR / "shoes_compact_ladder_row_semantics_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    with open(OUTDIR / "shoes_compact_ladder_row_semantics_v1_summary.txt", "w") as f:
        f.write("TAIRID SH0ES compact ladder row-semantics audit v1\n\n")
        f.write("Boundary: row semantics only. No TAIRID gate claim. Not proof.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Baseline compact ladder audit:\n")
        f.write(json.dumps(baseline_summary, indent=2) + "\n\n")

        f.write("Row counts:\n")
        f.write(json.dumps(summary["row_counts"], indent=2) + "\n\n")

        f.write("Key vector driver summaries:\n")
        f.write(json.dumps(key_vectors, indent=2) + "\n\n")

        f.write("Interpretation guide:\n")
        f.write("- Sparse-row survival may mean stiff constraints, not TAIRID structure.\n")
        f.write("- Moderate-row residual pressure is only useful if the row semantics identify real ladder relations.\n")
        f.write("- A physically meaningful next gate must be built from row classes such as host-Cepheid, host-anchor, or host-SN relations.\n")
        f.write("- Do not treat generic sparse/moderate masks as physical evidence.\n")

    print("")
    print("TAIRID SH0ES compact ladder row-semantics audit v1 complete.")
    print("Created:")
    print("  shoes_compact_ladder_row_semantics_v1_outputs/shoes_compact_ladder_row_semantics_v1_summary.json")
    print("  shoes_compact_ladder_row_semantics_v1_outputs/shoes_compact_ladder_row_semantics_v1_summary.txt")
    print("  shoes_compact_ladder_row_semantics_v1_outputs/shoes_compact_ladder_row_semantics_v1_rows.csv")
    print("  shoes_compact_ladder_row_semantics_v1_outputs/shoes_compact_ladder_row_semantics_v1_class_summary.csv")
    print("  shoes_compact_ladder_row_semantics_v1_outputs/shoes_compact_ladder_row_semantics_v1_driver_summary.csv")
    print("  shoes_compact_ladder_row_semantics_v1_outputs/shoes_compact_ladder_row_semantics_v1_top_row_drivers.csv")
    print("")
    print("Boundary:")
    print("  This is not a TAIRID gate claim.")
    print("  This is row semantics before physical interpretation.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
