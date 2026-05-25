#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder global-parameter semantics v1.

Purpose:
The previous row-semantics audit showed that direct gates on obvious physical ladder
groups are absorbed almost perfectly by ordinary SH0ES refitting. The only notable
residual pressure came from stiff global/nuisance rows around rows 3207-3214 and
parameters theta 37, 38, 39, 43, and 45.

This test asks:
1. What are the structural roles of theta 37-46?
2. Which rows are pure global/nuisance/prior-like constraints?
3. Are rows 3207-3214 physical ladder rows or stiff prior/nuisance rows?
4. If we exclude pure global/nuisance rows from physical gate interpretation, does any
   non-offset residual survive in real Cepheid/anchor/SN ladder rows?
5. Is the remaining gate behavior still mostly offset freedom?

Boundary:
This is not a TAIRID proof.
This is not a new SH0ES result.
This is not a full cosmology fit.
This is a nuisance/prior-row separation audit before any physical gate interpretation.
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
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.linalg import cho_factor, cho_solve


OUTDIR = Path("shoes_compact_ladder_global_parameter_semantics_v1_outputs")
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
KNOWN_ROLE_HINTS = {
    38: "Cepheid intercept / zero-point-like candidate; README notes an initial slope term -3.285 was used, but exact parameter names still need paper-level mapping.",
    40: "anchor-distance-modulus-like candidate from previous parameter map",
    42: "SN absolute magnitude / SN intercept-like candidate",
    46: "five-log-H0 parameter; read_chains_example.py computes H0 = 10^(theta_46/5)",
}

A_DRYRUN = 0.085


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
            "User-Agent": "TAIRID-SH0ES-global-parameter-semantics-v1",
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

    attempts = []
    pointer_best = None

    for url in build_url_candidates(filename, api_listing_by_name):
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


def h0_from_theta46(theta46):
    return float(10.0 ** (theta46 / 5.0))


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
        "theta_gls": theta_gls,
        "residual": residual,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": float(chi2 / dof),
        "leverage_diag": leverage,
    }


def chi2_for_theta(Y, M, c_factor, theta):
    residual = Y - M @ theta
    solved = cho_solve(c_factor, residual, check_finite=False)
    return float(residual @ solved), residual


def y_class(y):
    if y > 10.0:
        return "positive_distance_or_magnitude_like"
    if y < -10.0:
        return "negative_absolute_magnitude_like"
    if abs(y) <= 1.0:
        return "near_zero_constraint_like"
    return "intermediate_transformed_like"


def active_count_class(n):
    if n <= 2:
        return "sparse_1_to_2"
    if n <= 5:
        return "moderate_3_to_5"
    return "dense_6_plus"


def relation_candidate(active):
    s = set(active)
    has_host = any(i in HOST_THETA for i in s)
    has_global = any(i in GLOBAL_THETA for i in s)
    has_theta38 = 38 in s
    has_theta40 = 40 in s
    has_theta42 = 42 in s
    has_theta46 = 46 in s

    if has_host and has_theta38:
        return "host_Cepheid_relation_candidate"
    if has_host and has_theta40:
        return "host_anchor_relation_candidate"
    if has_host and has_theta42 and has_theta46:
        return "host_SN_H0_link_candidate"
    if has_host and has_theta42:
        return "host_SN_calibration_relation_candidate"
    if has_theta42 and has_theta46 and not has_host:
        return "H0_global_constraint_candidate"
    if has_global and not has_host:
        return "pure_global_or_nuisance_constraint_candidate"
    if has_host and has_global:
        return "host_global_ladder_relation_candidate"
    if has_host:
        return "host_only_relation_candidate"
    return "unmapped_relation_candidate"


def classify_row(Y, M, C_diag, gls, j):
    active = np.where(np.abs(M[j]) > 1.0e-10)[0]
    coeffs = M[j, active]
    hosts = [int(i) for i in active if i in HOST_THETA]
    globals_ = [int(i) for i in active if i in GLOBAL_THETA]
    relation = relation_candidate(active)

    pure_global_single = (
        len(active) == 1
        and len(globals_) == 1
        and len(hosts) == 0
    )

    prior_like = False
    prior_reason = ""

    if pure_global_single and abs(float(Y[j])) <= 1.0:
        prior_like = True
        prior_reason = "single global parameter near-zero constraint"
    elif pure_global_single and C_diag[j] < 0.002:
        prior_like = True
        prior_reason = "single global parameter high-weight constraint"
    elif pure_global_single:
        prior_like = True
        prior_reason = "single global parameter direct constraint"
    elif relation == "pure_global_or_nuisance_constraint_candidate" and C_diag[j] < 0.002:
        prior_like = True
        prior_reason = "pure global/nuisance high-weight row"

    return {
        "row_index": int(j),
        "Y_value": float(Y[j]),
        "Y_class": y_class(float(Y[j])),
        "baseline_residual": float(gls["residual"][j]),
        "C_diag": float(C_diag[j]),
        "C_sigma": float(math.sqrt(C_diag[j])) if C_diag[j] >= 0 else None,
        "leverage_diag_approx": float(gls["leverage_diag"][j]),
        "active_theta_count": int(active.size),
        "active_count_class": active_count_class(int(active.size)),
        "active_theta_indices": " ".join(str(int(i)) for i in active),
        "active_coefficients": " ".join(f"{float(c):.8g}" for c in coeffs),
        "host_theta_indices": " ".join(str(i) for i in hosts),
        "global_theta_indices": " ".join(str(i) for i in globals_),
        "relation_candidate": relation,
        "is_pure_global_single_parameter_row": bool(pure_global_single),
        "is_prior_or_nuisance_like_row": bool(prior_like),
        "prior_or_nuisance_reason": prior_reason,
        "has_theta37": 37 in active,
        "has_theta38": 38 in active,
        "has_theta39": 39 in active,
        "has_theta40": 40 in active,
        "has_theta41": 41 in active,
        "has_theta42": 42 in active,
        "has_theta43": 43 in active,
        "has_theta44": 44 in active,
        "has_theta45": 45 in active,
        "has_theta46": 46 in active,
    }


def classify_all_rows(Y, M, C_diag, gls):
    return [classify_row(Y, M, C_diag, gls, j) for j in range(Y.size)]


def support_text_audit(downloads_by_name):
    terms = [
        "theta", "parameter", "h0", "fivelog", "slope", "-3.285", "zeropoint",
        "zero point", "metal", "cepheid", "host", "anchor", "m_h", "m_b",
        "lmc", "n4258", "mw", "m31", "prior", "lstsq", "likelihood", "chi2",
    ]

    hits = []
    summaries = []

    for name in SUPPORT_FILES:
        item = downloads_by_name.get(name)
        if not item or not item.get("path"):
            summaries.append({"filename": name, "status": "missing"})
            continue

        path = Path(item["path"])
        try:
            text = path.read_text(errors="replace")
        except Exception as exc:
            summaries.append({"filename": name, "status": "read_failed", "error": str(exc)})
            continue

        lines = text.splitlines()
        lower = text.lower()
        term_counts = {t: lower.count(t.lower()) for t in terms if lower.count(t.lower())}

        hit_count = 0
        for i, line in enumerate(lines, start=1):
            if any(t.lower() in line.lower() for t in terms):
                hits.append({"filename": name, "line": i, "text": line[:900]})
                hit_count += 1
            if hit_count >= 400:
                break

        summaries.append(
            {
                "filename": name,
                "status": "inspected",
                "line_count": len(lines),
                "size_bytes": path.stat().st_size,
                "term_counts": term_counts,
                "hit_count_capped": hit_count,
            }
        )

    return summaries, hits


def parse_read_chains_indices(downloads_by_name):
    item = downloads_by_name.get("read_chains_example.py")
    if not item or not item.get("path"):
        return {"status": "missing"}

    text = Path(item["path"]).read_text(errors="replace")

    idx_match = re.search(r"idx\s*=\s*\[([^\]]+)\]", text)
    h0_formula = "H0_samples = 10**(fivelogH0/5)" in text

    out = {
        "status": "parsed",
        "H0_formula_found": bool(h0_formula),
        "raw_idx_match": idx_match.group(0) if idx_match else None,
    }

    if idx_match:
        nums = [int(x.strip()) for x in idx_match.group(1).split(",") if x.strip().isdigit()]
        out["selected_indices"] = nums
        if nums:
            out["inferred_H0_index"] = nums[-1]
            out["inference"] = "The script uses H0_idx=-1 after selecting idx, so the final selected theta index is the H0-bearing parameter."
    return out


def README_parameter_notes(downloads_by_name):
    item = downloads_by_name.get("README.md")
    if not item or not item.get("path"):
        return {"status": "missing"}

    text = Path(item["path"]).read_text(errors="replace")
    notes = []

    for line in text.splitlines():
        low = line.lower()
        if "initial slope term" in low or "free parameter slope" in low or "allc" in low or "alll" in low or "ally" in low:
            notes.append(line.strip())

    return {"status": "parsed", "notes": notes}


def global_parameter_profile(theta, theta_sigma, M, Y, C_diag, rows):
    profiles = []

    for k in GLOBAL_THETA:
        active_rows = np.where(np.abs(M[:, k]) > 1.0e-10)[0]
        coeffs = M[active_rows, k] if active_rows.size else np.asarray([])
        row_subset = [rows[int(j)] for j in active_rows]

        relation_counts = Counter(r["relation_candidate"] for r in row_subset)
        yclass_counts = Counter(r["Y_class"] for r in row_subset)
        count_class_counts = Counter(r["active_count_class"] for r in row_subset)
        direct_rows = [
            int(r["row_index"])
            for r in row_subset
            if r["is_pure_global_single_parameter_row"]
        ]

        candidate_role = KNOWN_ROLE_HINTS.get(k, "")

        if k == 46:
            role_confidence = "confirmed_by_read_chains_example"
        elif k in [38, 40, 42]:
            role_confidence = "medium_from_value_and_row_structure"
        elif direct_rows:
            role_confidence = "low_to_medium_direct_constraint_rows_detected"
        else:
            role_confidence = "low_unmapped_global_parameter"

        profiles.append(
            {
                "theta_index": int(k),
                "theta_value_public": float(theta[k]),
                "theta_sigma_lstsq": float(theta_sigma[k]),
                "candidate_role_note": candidate_role,
                "role_confidence": role_confidence,
                "active_row_count": int(active_rows.size),
                "direct_single_parameter_rows": " ".join(str(x) for x in direct_rows),
                "direct_single_parameter_row_count": len(direct_rows),
                "relation_counts_json": json.dumps(dict(relation_counts)),
                "Y_class_counts_json": json.dumps(dict(yclass_counts)),
                "active_count_class_counts_json": json.dumps(dict(count_class_counts)),
                "coeff_min": float(np.min(coeffs)) if coeffs.size else None,
                "coeff_max": float(np.max(coeffs)) if coeffs.size else None,
                "coeff_mean": float(np.mean(coeffs)) if coeffs.size else None,
                "coeff_std": float(np.std(coeffs)) if coeffs.size else None,
                "Y_mean_active": float(np.mean(Y[active_rows])) if active_rows.size else None,
                "Y_min_active": float(np.min(Y[active_rows])) if active_rows.size else None,
                "Y_max_active": float(np.max(Y[active_rows])) if active_rows.size else None,
                "C_diag_min_active": float(np.min(C_diag[active_rows])) if active_rows.size else None,
                "C_diag_median_active": float(np.median(C_diag[active_rows])) if active_rows.size else None,
            }
        )

    return profiles


def uniform_gate_delta(mask, amplitude=A_DRYRUN):
    delta_mag = 5.0 * math.log10(max(1.0e-9, 1.0 - amplitude))
    return delta_mag * mask.astype(float), delta_mag


def project_all47(delta_y, M, gls):
    c_factor = gls["c_factor"]
    Cinv_delta = cho_solve(c_factor, delta_y, check_finite=False)
    delta_norm = float(delta_y @ Cinv_delta)

    b_delta = M.T @ Cinv_delta
    delta_theta = np.linalg.solve(gls["A"], b_delta)

    projected = M @ delta_theta
    surviving = delta_y - projected
    Cinv_surviving = cho_solve(c_factor, surviving, check_finite=False)

    projected_norm = float(projected @ cho_solve(c_factor, projected, check_finite=False))
    surviving_norm = float(surviving @ Cinv_surviving)

    baseline_cross = float(2.0 * gls["residual"] @ Cinv_surviving)
    delta_chi2 = float(baseline_cross + surviving_norm)

    theta_new = gls["theta_gls"] + delta_theta

    return {
        "delta_norm": delta_norm,
        "projected_norm": projected_norm,
        "surviving_norm": surviving_norm,
        "absorption_fraction": projected_norm / delta_norm if delta_norm > 0 else None,
        "residual_fraction": surviving_norm / delta_norm if delta_norm > 0 else None,
        "baseline_cross_after_projection": baseline_cross,
        "delta_chi2_after_all47_refit": delta_chi2,
        "delta_theta": delta_theta,
        "surviving": surviving,
        "Cinv_surviving": Cinv_surviving,
        "theta46_delta": float(delta_theta[46]),
        "H0_before": h0_from_theta46(gls["theta_gls"][46]),
        "H0_after": h0_from_theta46(theta_new[46]),
        "H0_delta": float(h0_from_theta46(theta_new[46]) - h0_from_theta46(gls["theta_gls"][46])),
    }


def build_masks(rows):
    n = len(rows)

    def arr(pred):
        return np.asarray([bool(pred(r)) for r in rows], dtype=bool)

    prior_like = arr(lambda r: r["is_prior_or_nuisance_like_row"])
    physical_ladder = ~prior_like

    masks = {
        "all_rows": np.ones(n, dtype=bool),
        "all_rows_excluding_prior_or_nuisance_like": physical_ladder,
        "moderate_rows_active_3_to_5": arr(lambda r: r["active_count_class"] == "moderate_3_to_5"),
        "moderate_rows_excluding_prior_or_nuisance_like": arr(lambda r: r["active_count_class"] == "moderate_3_to_5") & physical_ladder,
        "sparse_rows_active_1_to_2": arr(lambda r: r["active_count_class"] == "sparse_1_to_2"),
        "sparse_rows_excluding_prior_or_nuisance_like": arr(lambda r: r["active_count_class"] == "sparse_1_to_2") & physical_ladder,
        "host_cepheid_rows": arr(lambda r: r["relation_candidate"] == "host_Cepheid_relation_candidate"),
        "host_sn_rows": arr(lambda r: r["relation_candidate"] == "host_SN_calibration_relation_candidate"),
        "h0_global_rows": arr(lambda r: r["relation_candidate"] == "H0_global_constraint_candidate"),
        "pure_prior_or_nuisance_like_rows": prior_like,
        "near_zero_prior_or_nuisance_like_rows": arr(lambda r: r["is_prior_or_nuisance_like_row"] and r["Y_class"] == "near_zero_constraint_like"),
        "theta37_39_43_45_direct_constraint_rows": arr(
            lambda r: r["is_pure_global_single_parameter_row"]
            and any(r[f"has_theta{k}"] for k in [37, 39, 43, 45])
        ),
        "theta38_direct_constraint_rows": arr(lambda r: r["is_pure_global_single_parameter_row"] and r["has_theta38"]),
    }

    return {k: v for k, v in masks.items() if int(np.sum(v)) > 0}


def projection_audit(rows, M, gls):
    masks = build_masks(rows)
    out_rows = []
    driver_rows = []

    for name, mask in masks.items():
        delta_y, delta_mag = uniform_gate_delta(mask)
        proj = project_all47(delta_y, M, gls)

        signed = proj["surviving"] * proj["Cinv_surviving"]
        abs_contrib = np.abs(signed)
        total_abs = float(np.sum(abs_contrib))

        selected = np.where(mask)[0]
        selected_abs = float(np.sum(abs_contrib[selected])) if selected.size else 0.0

        out_rows.append(
            {
                "mask_name": name,
                "selected_row_count": int(np.sum(mask)),
                "delta_mag_per_selected_row": float(delta_mag),
                "delta_norm": proj["delta_norm"],
                "projected_norm": proj["projected_norm"],
                "surviving_norm": proj["surviving_norm"],
                "absorption_fraction_all47": proj["absorption_fraction"],
                "residual_fraction_all47": proj["residual_fraction"],
                "baseline_cross_after_projection": proj["baseline_cross_after_projection"],
                "delta_chi2_after_all47_refit": proj["delta_chi2_after_all47_refit"],
                "theta46_delta": proj["theta46_delta"],
                "H0_before": proj["H0_before"],
                "H0_after": proj["H0_after"],
                "H0_delta": proj["H0_delta"],
                "selected_rows_abs_contribution_fraction": selected_abs / total_abs if total_abs > 0 else None,
            }
        )

        for rank, idx in enumerate(np.argsort(-abs_contrib)[:200], start=1):
            r = rows[int(idx)]
            driver_rows.append(
                {
                    "mask_name": name,
                    "rank": rank,
                    "row_index": int(idx),
                    "selected_by_mask": bool(mask[int(idx)]),
                    "signed_survival_contribution": float(signed[int(idx)]),
                    "abs_survival_contribution": float(abs_contrib[int(idx)]),
                    "fraction_of_abs_contribution": float(abs_contrib[int(idx)] / total_abs) if total_abs > 0 else None,
                    "Y_value": r["Y_value"],
                    "Y_class": r["Y_class"],
                    "C_diag": r["C_diag"],
                    "C_sigma": r["C_sigma"],
                    "leverage_diag_approx": r["leverage_diag_approx"],
                    "active_theta_count": r["active_theta_count"],
                    "active_theta_indices": r["active_theta_indices"],
                    "relation_candidate": r["relation_candidate"],
                    "is_prior_or_nuisance_like_row": r["is_prior_or_nuisance_like_row"],
                    "prior_or_nuisance_reason": r["prior_or_nuisance_reason"],
                }
            )

    return out_rows, driver_rows


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_plots(parameter_profiles, projection_rows):
    paths = []

    p = pd.DataFrame(parameter_profiles)
    plt.figure(figsize=(10, 6))
    plt.bar(p["theta_index"].astype(str), p["active_row_count"])
    plt.xlabel("global theta index")
    plt.ylabel("active row count")
    plt.title("SH0ES global parameter semantics v1: active row count by theta")
    plt.tight_layout()
    path = OUTDIR / "global_theta_active_row_counts.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(str(path))

    q = pd.DataFrame(projection_rows)
    plt.figure(figsize=(14, 6))
    plt.bar(q["mask_name"], q["residual_fraction_all47"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("residual fraction after all-47 refit")
    plt.title("SH0ES global parameter semantics v1: residual fraction after prior-row separation")
    plt.tight_layout()
    path = OUTDIR / "projection_residual_fraction_after_prior_separation.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(str(path))

    plt.figure(figsize=(14, 6))
    plt.bar(q["mask_name"], q["delta_chi2_after_all47_refit"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("delta chi-square after all-47 refit")
    plt.title("SH0ES global parameter semantics v1: delta chi-square after prior-row separation")
    plt.tight_layout()
    path = OUTDIR / "projection_delta_chi2_after_prior_separation.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(str(path))

    return paths


def main():
    print("")
    print("TAIRID SH0ES compact ladder global-parameter semantics v1 starting.")
    print("Boundary: nuisance/prior-row separation only; no TAIRID gate claim.")
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

    (OUTDIR / "github_api_listing.json").write_text(json.dumps(api_summary, indent=2))

    downloads = []
    for filename in ALL_TARGETS:
        print(f"Downloading / locating {filename} ...")
        result = download_or_copy(filename, api_listing_by_name)
        downloads.append(result)
        print(f"  {result.get('status')}")

    downloads_by_name = {item["filename"]: item for item in downloads}

    Y, L, M, C, theta_public, theta_sigma, primary_paths = load_primary_system(downloads_by_name)
    gls = generalized_least_squares_setup(Y, M, C)
    public_chi2, public_residual = chi2_for_theta(Y, M, gls["c_factor"], theta_public)

    C_diag = np.diag(C)
    rows = classify_all_rows(Y, M, C_diag, gls)
    parameter_profiles = global_parameter_profile(theta_public, theta_sigma, M, Y, C_diag, rows)

    support_summaries, support_hits = support_text_audit(downloads_by_name)
    read_chains_parse = parse_read_chains_indices(downloads_by_name)
    readme_notes = README_parameter_notes(downloads_by_name)

    projection_rows, driver_rows = projection_audit(rows, M, gls)

    row_path = write_csv(OUTDIR / "global_parameter_semantics_v1_rows.csv", rows)
    profile_path = write_csv(OUTDIR / "global_parameter_semantics_v1_theta37_to_46_profiles.csv", parameter_profiles)
    projection_path = write_csv(OUTDIR / "global_parameter_semantics_v1_projection_after_prior_separation.csv", projection_rows)
    driver_path = write_csv(OUTDIR / "global_parameter_semantics_v1_top_projection_drivers.csv", driver_rows)
    support_hits_path = write_csv(OUTDIR / "global_parameter_semantics_v1_support_hits.csv", support_hits)

    download_ledger_rows = []
    for item in downloads:
        pointer = item.get("pointer", {})
        download_ledger_rows.append(
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
    download_ledger = write_csv(OUTDIR / "global_parameter_semantics_v1_download_ledger.csv", download_ledger_rows)

    role_counts = Counter(r["relation_candidate"] for r in rows)
    prior_count = sum(1 for r in rows if r["is_prior_or_nuisance_like_row"])
    pure_global_single_count = sum(1 for r in rows if r["is_pure_global_single_parameter_row"])

    projection_by_name = {r["mask_name"]: r for r in projection_rows}
    physical = projection_by_name.get("all_rows_excluding_prior_or_nuisance_like")
    moderate_physical = projection_by_name.get("moderate_rows_excluding_prior_or_nuisance_like")
    prior_only = projection_by_name.get("pure_prior_or_nuisance_like_rows")

    if physical and physical["residual_fraction_all47"] < 0.01 and abs(physical["delta_chi2_after_all47_refit"]) < 5.0:
        final_status = "physical_ladder_rows_mostly_absorbed_after_prior_nuisance_exclusion"
        readiness_score = 8
        next_wall = "Gate branch likely remains calibration-degenerate unless a more specific derived physical vector is provided."
    elif moderate_physical and moderate_physical["residual_fraction_all47"] >= 0.02:
        final_status = "non_offset_residual_persists_after_prior_nuisance_exclusion"
        readiness_score = 8
        next_wall = "Inspect top physical row drivers and build a narrowly mapped gate using only those row semantics."
    elif prior_only and prior_only["delta_chi2_after_all47_refit"] > 20.0:
        final_status = "residual_pressure_localizes_to_prior_nuisance_rows"
        readiness_score = 8
        next_wall = "Treat previous residual as nuisance/prior sensitivity, not physical TAIRID evidence."
    else:
        final_status = "mixed_global_parameter_semantics_needs_manual_review"
        readiness_score = 7
        next_wall = "Manual review of theta37-46 profiles and row drivers required before next gate."

    plot_paths = make_plots(parameter_profiles, projection_rows)

    baseline_summary = {
        "Y_shape": list(Y.shape),
        "L_shape": list(L.shape),
        "M_shape": list(M.shape),
        "C_shape": list(C.shape),
        "theta_count": int(theta_public.size),
        "public_theta_chi2": public_chi2,
        "gls_refit_chi2": gls["chi2"],
        "gls_refit_dof": gls["dof"],
        "gls_refit_reduced_chi2": gls["reduced_chi2"],
        "public_minus_gls_chi2": float(public_chi2 - gls["chi2"]),
        "theta46_public": float(theta_public[46]),
        "theta46_gls": float(gls["theta_gls"][46]),
        "H0_public_from_theta46": h0_from_theta46(theta_public[46]),
        "H0_gls_from_theta46": h0_from_theta46(gls["theta_gls"][46]),
    }

    summary = {
        "test_name": "TAIRID SH0ES compact ladder global-parameter semantics v1",
        "boundary": (
            "Nuisance/prior-row separation only. No TAIRID gate claim. "
            "Not a new SH0ES result, not a full cosmology fit, and not proof."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "primary_paths": primary_paths,
        "baseline_summary": baseline_summary,
        "row_counts": {
            "total_rows": int(len(rows)),
            "relation_candidate_counts": dict(role_counts),
            "pure_global_single_parameter_row_count": int(pure_global_single_count),
            "prior_or_nuisance_like_row_count": int(prior_count),
        },
        "read_chains_example_parse": read_chains_parse,
        "README_parameter_notes": readme_notes,
        "theta37_to_46_profiles": parameter_profiles,
        "projection_after_prior_separation": projection_rows,
        "support_summaries": support_summaries,
        "output_files": {
            "download_ledger_csv": str(download_ledger),
            "rows_csv": str(row_path),
            "theta37_to_46_profiles_csv": str(profile_path),
            "projection_after_prior_separation_csv": str(projection_path),
            "top_projection_drivers_csv": str(driver_path),
            "support_hits_csv": str(support_hits_path),
            "plots": plot_paths,
        },
    }

    summary_path = OUTDIR / "global_parameter_semantics_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    with open(OUTDIR / "global_parameter_semantics_v1_summary.txt", "w") as f:
        f.write("TAIRID SH0ES compact ladder global-parameter semantics v1\n\n")
        f.write("Boundary: nuisance/prior-row separation only. No TAIRID gate claim. Not proof.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Baseline compact ladder audit:\n")
        f.write(json.dumps(baseline_summary, indent=2) + "\n\n")

        f.write("Read chains parse:\n")
        f.write(json.dumps(read_chains_parse, indent=2) + "\n\n")

        f.write("README parameter notes:\n")
        f.write(json.dumps(readme_notes, indent=2) + "\n\n")

        f.write("Row counts:\n")
        f.write(json.dumps(summary["row_counts"], indent=2) + "\n\n")

        f.write("Projection after prior/nuisance separation:\n")
        f.write(json.dumps(projection_rows, indent=2) + "\n\n")

        f.write("Interpretation guide:\n")
        f.write("- If physical ladder rows are absorbed after excluding prior/nuisance rows, the gate remains calibration-degenerate.\n")
        f.write("- If residual pressure localizes to prior/nuisance rows, it should not be interpreted as TAIRID physical structure.\n")
        f.write("- If residual survives in physical Cepheid/anchor/SN rows, the next test should build a narrowly mapped physical gate vector.\n")
        f.write("- Do not claim TAIRID solved Hubble tension from this audit.\n")

    print("")
    print("TAIRID SH0ES compact ladder global-parameter semantics v1 complete.")
    print("Created:")
    print("  shoes_compact_ladder_global_parameter_semantics_v1_outputs/global_parameter_semantics_v1_summary.json")
    print("  shoes_compact_ladder_global_parameter_semantics_v1_outputs/global_parameter_semantics_v1_summary.txt")
    print("  shoes_compact_ladder_global_parameter_semantics_v1_outputs/global_parameter_semantics_v1_theta37_to_46_profiles.csv")
    print("  shoes_compact_ladder_global_parameter_semantics_v1_outputs/global_parameter_semantics_v1_projection_after_prior_separation.csv")
    print("  shoes_compact_ladder_global_parameter_semantics_v1_outputs/global_parameter_semantics_v1_top_projection_drivers.csv")
    print("")
    print("Boundary:")
    print("  This is nuisance/prior-row separation before physical interpretation.")
    print("  This is not a TAIRID gate claim.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
