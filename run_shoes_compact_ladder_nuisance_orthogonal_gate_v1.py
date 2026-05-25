#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder nuisance-orthogonal gate v1.

Purpose:
The global-parameter semantics audit showed that residual pressure persisted after
excluding direct prior/nuisance rows from the perturbation mask, but most surviving
pressure still localized into stiff global/nuisance constraint rows.

This test asks a sharper question:
If a physical-looking ladder gate is explicitly orthogonalized against nuisance/prior
directions before projection through the all-47 SH0ES refit, does any non-offset residual
remain in real ladder rows?

Method:
1. Load public SH0ES compact matrix system:
   - C covariance matrix
   - L equation matrix
   - y data vector
   - theta from lstsq_results.txt
2. Reconstruct GLS baseline.
3. Classify rows and detect direct prior/nuisance rows.
4. Build physical gate vectors:
   - all non-prior rows
   - moderate non-prior rows
   - host-Cepheid rows
   - host-SN rows
   - H0 global rows
5. Orthogonalize each vector against:
   - nothing
   - nuisance/global direct row basis
   - nuisance parameter columns theta 37,38,39,41,43,44,45
   - both nuisance row basis and nuisance parameter columns
6. Refit all 47 parameters after each cleaned vector.
7. Measure:
   - absorption fraction
   - residual fraction
   - delta chi-square
   - H0 shift
   - where surviving residual localizes

Boundary:
This is not a TAIRID proof.
This is not a new SH0ES result.
This is not a full cosmology fit.
This is a nuisance-orthogonal projection test before any physical gate interpretation.
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
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.linalg import cho_factor, cho_solve


OUTDIR = Path("shoes_compact_ladder_nuisance_orthogonal_gate_v1_outputs")
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
    "README.md",
]

HOST_THETA = list(range(0, 37))
GLOBAL_THETA = list(range(37, 47))
NUISANCE_THETA = [37, 38, 39, 41, 43, 44, 45]
SN_THETA = [42]
H0_THETA = [46]

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
            "User-Agent": "TAIRID-SH0ES-nuisance-orthogonal-gate-v1",
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


def h0_from_theta46(theta46):
    return float(10.0 ** (theta46 / 5.0))


def gls_setup(Y, M, C):
    c_factor = cho_factor(C, lower=True, check_finite=False)
    Cinv_Y = cho_solve(c_factor, Y, check_finite=False)
    Cinv_M = cho_solve(c_factor, M, check_finite=False)

    A = M.T @ Cinv_M
    A_inv = np.linalg.inv(A)
    b = M.T @ Cinv_Y

    theta = A_inv @ b
    residual = Y - M @ theta
    Cinv_residual = cho_solve(c_factor, residual, check_finite=False)
    chi2 = float(residual @ Cinv_residual)
    dof = int(Y.size - theta.size)

    return {
        "c_factor": c_factor,
        "Cinv_M": Cinv_M,
        "A": A,
        "A_inv": A_inv,
        "theta": theta,
        "residual": residual,
        "Cinv_residual": Cinv_residual,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": float(chi2 / dof),
    }


def chi2_for_theta(Y, M, c_factor, theta):
    residual = Y - M @ theta
    solved = cho_solve(c_factor, residual, check_finite=False)
    return float(residual @ solved)


def relation_candidate(active):
    s = set(active)
    has_host = any(i in HOST_THETA for i in s)
    has_theta38 = 38 in s
    has_theta40 = 40 in s
    has_theta42 = 42 in s
    has_theta46 = 46 in s
    has_global = any(i in GLOBAL_THETA for i in s)

    if has_host and has_theta38:
        return "host_Cepheid_relation_candidate"
    if has_host and has_theta40:
        return "host_anchor_relation_candidate"
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


def classify_rows(Y, M, C):
    C_diag = np.diag(C)
    rows = []

    for j in range(Y.size):
        active = np.where(np.abs(M[j]) > 1.0e-10)[0]
        globals_ = [int(i) for i in active if i in GLOBAL_THETA]
        hosts = [int(i) for i in active if i in HOST_THETA]
        relation = relation_candidate(active)
        pure_global_single = len(active) == 1 and len(globals_) == 1 and len(hosts) == 0

        prior_like = False
        reason = ""

        if pure_global_single and abs(float(Y[j])) <= 1.0:
            prior_like = True
            reason = "single global parameter near-zero constraint"
        elif pure_global_single and C_diag[j] < 0.002:
            prior_like = True
            reason = "single global parameter high-weight constraint"
        elif pure_global_single:
            prior_like = True
            reason = "single global parameter direct constraint"
        elif relation == "pure_global_or_nuisance_constraint_candidate" and C_diag[j] < 0.002:
            prior_like = True
            reason = "pure global/nuisance high-weight row"

        rows.append(
            {
                "row_index": int(j),
                "Y_value": float(Y[j]),
                "Y_class": y_class(float(Y[j])),
                "C_diag": float(C_diag[j]),
                "C_sigma": float(math.sqrt(C_diag[j])) if C_diag[j] >= 0 else None,
                "active_theta_count": int(active.size),
                "active_count_class": active_count_class(int(active.size)),
                "active_theta_indices": " ".join(str(int(i)) for i in active),
                "host_theta_indices": " ".join(str(i) for i in hosts),
                "global_theta_indices": " ".join(str(i) for i in globals_),
                "relation_candidate": relation,
                "is_prior_or_nuisance_like_row": bool(prior_like),
                "prior_or_nuisance_reason": reason,
                "has_host": bool(hosts),
                "has_theta38": 38 in active,
                "has_theta40": 40 in active,
                "has_theta42": 42 in active,
                "has_theta46": 46 in active,
            }
        )

    return rows


def build_physical_masks(rows):
    n = len(rows)

    def arr(pred):
        return np.asarray([bool(pred(r)) for r in rows], dtype=bool)

    prior = arr(lambda r: r["is_prior_or_nuisance_like_row"])
    nonprior = ~prior

    masks = {
        "all_nonprior_rows": nonprior,
        "moderate_nonprior_rows": arr(lambda r: r["active_count_class"] == "moderate_3_to_5") & nonprior,
        "sparse_nonprior_rows": arr(lambda r: r["active_count_class"] == "sparse_1_to_2") & nonprior,
        "host_cepheid_rows": arr(lambda r: r["relation_candidate"] == "host_Cepheid_relation_candidate"),
        "host_sn_rows": arr(lambda r: r["relation_candidate"] == "host_SN_calibration_relation_candidate"),
        "h0_global_rows": arr(lambda r: r["relation_candidate"] == "H0_global_constraint_candidate"),
        "prior_or_nuisance_rows_only": prior,
    }

    return {name: mask for name, mask in masks.items() if int(np.sum(mask)) > 0}


def uniform_gate_delta(mask, amplitude=A_DRYRUN):
    delta_mag = 5.0 * math.log10(max(1.0e-9, 1.0 - amplitude))
    return delta_mag * mask.astype(float), delta_mag


def weighted_project_onto_basis(vec, B, c_factor, ridge=1.0e-12):
    if B is None or B.size == 0 or B.shape[1] == 0:
        return np.zeros_like(vec), vec, 0

    Cinv_vec = cho_solve(c_factor, vec, check_finite=False)
    Cinv_B = cho_solve(c_factor, B, check_finite=False)

    G = B.T @ Cinv_B
    h = B.T @ Cinv_vec

    try:
        coeff = np.linalg.solve(G, h)
    except np.linalg.LinAlgError:
        coeff = np.linalg.solve(G + np.eye(G.shape[0]) * ridge, h)

    projected = B @ coeff
    cleaned = vec - projected
    return projected, cleaned, B.shape[1]


def build_nuisance_bases(rows, M):
    prior_indices = [r["row_index"] for r in rows if r["is_prior_or_nuisance_like_row"]]
    n = M.shape[0]

    prior_basis = np.zeros((n, len(prior_indices)), dtype=float)
    for k, idx in enumerate(prior_indices):
        prior_basis[idx, k] = 1.0

    nuisance_cols = M[:, NUISANCE_THETA]

    both = np.column_stack([prior_basis, nuisance_cols]) if prior_basis.size else nuisance_cols

    return {
        "none": None,
        "prior_row_basis": prior_basis,
        "nuisance_parameter_columns": nuisance_cols,
        "prior_rows_plus_nuisance_parameter_columns": both,
    }, prior_indices


def weighted_norm(vec, c_factor):
    solved = cho_solve(c_factor, vec, check_finite=False)
    return float(vec @ solved), solved


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

    theta_new = gls["theta"] + delta_theta

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
        "H0_before": h0_from_theta46(gls["theta"][46]),
        "H0_after": h0_from_theta46(theta_new[46]),
        "H0_delta": float(h0_from_theta46(theta_new[46]) - h0_from_theta46(gls["theta"][46])),
    }


def run_orthogonal_tests(rows, M, gls):
    masks = build_physical_masks(rows)
    bases, prior_indices = build_nuisance_bases(rows, M)

    results = []
    drivers = []

    for mask_name, mask in masks.items():
        raw_delta, raw_delta_mag = uniform_gate_delta(mask)

        for clean_method, B in bases.items():
            nuisance_projected, clean_delta, removed_basis_count = weighted_project_onto_basis(
                raw_delta, B, gls["c_factor"]
            )

            raw_norm, _ = weighted_norm(raw_delta, gls["c_factor"])
            clean_norm, _ = weighted_norm(clean_delta, gls["c_factor"])
            removed_norm, _ = weighted_norm(nuisance_projected, gls["c_factor"])

            proj = project_all47(clean_delta, M, gls)

            signed = proj["surviving"] * proj["Cinv_surviving"]
            abs_contrib = np.abs(signed)
            total_abs = float(np.sum(abs_contrib))

            selected = np.where(mask)[0]
            prior = np.asarray(prior_indices, dtype=int)
            selected_abs = float(np.sum(abs_contrib[selected])) if selected.size else 0.0
            prior_abs = float(np.sum(abs_contrib[prior])) if prior.size else 0.0

            results.append(
                {
                    "mask_name": mask_name,
                    "clean_method": clean_method,
                    "selected_row_count": int(np.sum(mask)),
                    "removed_basis_count": removed_basis_count,
                    "raw_delta_mag_per_selected_row": float(raw_delta_mag),
                    "raw_delta_norm": raw_norm,
                    "removed_nuisance_norm": removed_norm,
                    "clean_delta_norm": clean_norm,
                    "clean_norm_fraction_of_raw": clean_norm / raw_norm if raw_norm > 0 else None,
                    "delta_norm_entering_all47_refit": proj["delta_norm"],
                    "absorption_fraction_all47": proj["absorption_fraction"],
                    "residual_fraction_all47": proj["residual_fraction"],
                    "surviving_norm": proj["surviving_norm"],
                    "delta_chi2_after_all47_refit": proj["delta_chi2_after_all47_refit"],
                    "theta46_delta": proj["theta46_delta"],
                    "H0_before": proj["H0_before"],
                    "H0_after": proj["H0_after"],
                    "H0_delta": proj["H0_delta"],
                    "selected_rows_abs_contribution_fraction": selected_abs / total_abs if total_abs > 0 else None,
                    "prior_rows_abs_contribution_fraction": prior_abs / total_abs if total_abs > 0 else None,
                }
            )

            for rank, idx in enumerate(np.argsort(-abs_contrib)[:100], start=1):
                r = rows[int(idx)]
                drivers.append(
                    {
                        "mask_name": mask_name,
                        "clean_method": clean_method,
                        "rank": rank,
                        "row_index": int(idx),
                        "selected_by_mask": bool(mask[int(idx)]),
                        "is_prior_or_nuisance_like_row": r["is_prior_or_nuisance_like_row"],
                        "signed_survival_contribution": float(signed[int(idx)]),
                        "abs_survival_contribution": float(abs_contrib[int(idx)]),
                        "fraction_of_abs_contribution": abs_contrib[int(idx)] / total_abs if total_abs > 0 else None,
                        "Y_value": r["Y_value"],
                        "Y_class": r["Y_class"],
                        "C_diag": r["C_diag"],
                        "C_sigma": r["C_sigma"],
                        "active_theta_count": r["active_theta_count"],
                        "active_theta_indices": r["active_theta_indices"],
                        "relation_candidate": r["relation_candidate"],
                        "prior_or_nuisance_reason": r["prior_or_nuisance_reason"],
                    }
                )

    return results, drivers, prior_indices


def support_audit(downloads_by_name):
    out = {}

    rc = downloads_by_name.get("read_chains_example.py")
    if rc and rc.get("path"):
        text = Path(rc["path"]).read_text(errors="replace")
        idx_match = re.search(r"idx\s*=\s*\[([^\]]+)\]", text)
        out["read_chains_example"] = {
            "idx_match": idx_match.group(0) if idx_match else None,
            "H0_formula_found": "10**(fivelogH0/5)" in text,
        }

    readme = downloads_by_name.get("README.md")
    if readme and readme.get("path"):
        text = Path(readme["path"]).read_text(errors="replace")
        notes = []
        for line in text.splitlines():
            low = line.lower()
            if "allc" in low or "alll" in low or "ally" in low or "initial slope" in low or "prior-free" in low:
                notes.append(line.strip())
        out["README_notes"] = notes

    return out


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_plots(results):
    paths = []
    df = pd.DataFrame(results)

    focus = df[df["mask_name"].isin(["all_nonprior_rows", "moderate_nonprior_rows", "host_cepheid_rows", "host_sn_rows"])]

    if len(focus):
        label = focus["mask_name"] + "\n" + focus["clean_method"]
        x = np.arange(len(focus))

        plt.figure(figsize=(16, 7))
        plt.bar(x, focus["residual_fraction_all47"])
        plt.xticks(x, label, rotation=45, ha="right")
        plt.ylabel("residual fraction after all-47 refit")
        plt.title("Nuisance-orthogonal gate v1: residual fraction after cleaning")
        plt.tight_layout()
        path = OUTDIR / "nuisance_orthogonal_gate_v1_residual_fraction.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

        plt.figure(figsize=(16, 7))
        plt.bar(x, focus["delta_chi2_after_all47_refit"])
        plt.xticks(x, label, rotation=45, ha="right")
        plt.ylabel("delta chi-square after all-47 refit")
        plt.title("Nuisance-orthogonal gate v1: delta chi-square after cleaning")
        plt.tight_layout()
        path = OUTDIR / "nuisance_orthogonal_gate_v1_delta_chi2.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

        plt.figure(figsize=(16, 7))
        plt.bar(x, focus["prior_rows_abs_contribution_fraction"])
        plt.xticks(x, label, rotation=45, ha="right")
        plt.ylabel("fraction of surviving contribution in prior/nuisance rows")
        plt.title("Nuisance-orthogonal gate v1: prior-row localization after cleaning")
        plt.tight_layout()
        path = OUTDIR / "nuisance_orthogonal_gate_v1_prior_localization.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

    return paths


def main():
    print("")
    print("TAIRID SH0ES compact ladder nuisance-orthogonal gate v1 starting.")
    print("Boundary: nuisance-orthogonal projection only, not a TAIRID proof.")
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

    downloads = []
    for filename in PRIMARY_FILES:
        print(f"Downloading / locating {filename} ...")
        result = download_or_copy(filename, api_listing_by_name)
        downloads.append(result)
        print(f"  {result.get('status')}")

    downloads_by_name = {d["filename"]: d for d in downloads}

    Y, L, M, C, theta_public, theta_sigma, primary_paths = load_primary_system(downloads_by_name)
    gls = gls_setup(Y, M, C)
    public_chi2 = chi2_for_theta(Y, M, gls["c_factor"], theta_public)

    rows = classify_rows(Y, M, C)
    results, drivers, prior_indices = run_orthogonal_tests(rows, M, gls)

    support = support_audit(downloads_by_name)

    row_counts = {
        "total_rows": len(rows),
        "prior_or_nuisance_like_rows": int(sum(r["is_prior_or_nuisance_like_row"] for r in rows)),
        "prior_indices": prior_indices,
        "relation_candidate_counts": dict(Counter(r["relation_candidate"] for r in rows)),
    }

    by_key = {(r["mask_name"], r["clean_method"]): r for r in results}

    decisive = by_key.get(("moderate_nonprior_rows", "prior_rows_plus_nuisance_parameter_columns"))
    all_nonprior_clean = by_key.get(("all_nonprior_rows", "prior_rows_plus_nuisance_parameter_columns"))

    if decisive and decisive["residual_fraction_all47"] < 0.01 and abs(decisive["delta_chi2_after_all47_refit"]) < 5:
        final_status = "non_offset_residual_collapses_after_nuisance_orthogonalization"
        readiness_score = 8
        next_wall = "Record gate branch as calibration/nuisance-degenerate unless a derived physical vector is supplied."
    elif decisive and decisive["residual_fraction_all47"] >= 0.02 and decisive["prior_rows_abs_contribution_fraction"] < 0.25:
        final_status = "nuisance_orthogonal_physical_residual_survives"
        readiness_score = 9
        next_wall = "Inspect surviving physical row drivers and build a narrowly mapped TAIRID boundary vector."
    elif all_nonprior_clean and all_nonprior_clean["prior_rows_abs_contribution_fraction"] >= 0.50:
        final_status = "residual_still_localizes_to_prior_nuisance_sector"
        readiness_score = 8
        next_wall = "Treat apparent survival as nuisance-sector sensitivity, not physical evidence."
    else:
        final_status = "mixed_nuisance_orthogonal_response_needs_review"
        readiness_score = 7
        next_wall = "Review projection drivers before another gate test."

    result_path = write_csv(OUTDIR / "nuisance_orthogonal_gate_v1_results.csv", results)
    driver_path = write_csv(OUTDIR / "nuisance_orthogonal_gate_v1_top_drivers.csv", drivers)
    rows_path = write_csv(OUTDIR / "nuisance_orthogonal_gate_v1_rows.csv", rows)

    download_ledger = []
    for item in downloads:
        pointer = item.get("pointer", {})
        download_ledger.append(
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
    download_ledger_path = write_csv(OUTDIR / "nuisance_orthogonal_gate_v1_download_ledger.csv", download_ledger)

    plot_paths = make_plots(results)

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
        "theta46_public": float(theta_public[46]),
        "theta46_gls": float(gls["theta"][46]),
        "H0_public_from_theta46": h0_from_theta46(theta_public[46]),
        "H0_gls_from_theta46": h0_from_theta46(gls["theta"][46]),
    }

    summary = {
        "test_name": "TAIRID SH0ES compact ladder nuisance-orthogonal gate v1",
        "boundary": (
            "Nuisance-orthogonal projection test only. Not a TAIRID proof, not a new SH0ES result, "
            "and not a full cosmology fit."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "baseline_summary": baseline_summary,
        "row_counts": row_counts,
        "support_audit": support,
        "key_results": {
            "moderate_nonprior_cleaned_by_both": decisive,
            "all_nonprior_cleaned_by_both": all_nonprior_clean,
        },
        "all_results": results,
        "output_files": {
            "results_csv": str(result_path),
            "top_drivers_csv": str(driver_path),
            "rows_csv": str(rows_path),
            "download_ledger_csv": str(download_ledger_path),
            "plots": plot_paths,
        },
    }

    summary_path = OUTDIR / "nuisance_orthogonal_gate_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    with open(OUTDIR / "nuisance_orthogonal_gate_v1_summary.txt", "w") as f:
        f.write("TAIRID SH0ES compact ladder nuisance-orthogonal gate v1\n\n")
        f.write("Boundary: nuisance-orthogonal projection only. Not proof. Not a new SH0ES result.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")
        f.write("Baseline:\n")
        f.write(json.dumps(baseline_summary, indent=2) + "\n\n")
        f.write("Row counts:\n")
        f.write(json.dumps(row_counts, indent=2) + "\n\n")
        f.write("Key results:\n")
        f.write(json.dumps(summary["key_results"], indent=2) + "\n\n")
        f.write("Interpretation guide:\n")
        f.write("- If residual collapses after nuisance orthogonalization, the previous gate survival was nuisance-degenerate.\n")
        f.write("- If residual survives and no longer localizes to prior rows, inspect physical row drivers.\n")
        f.write("- Do not claim a TAIRID Hubble solution from this test alone.\n")

    print("")
    print("TAIRID SH0ES compact ladder nuisance-orthogonal gate v1 complete.")
    print("Created:")
    print("  shoes_compact_ladder_nuisance_orthogonal_gate_v1_outputs/nuisance_orthogonal_gate_v1_summary.json")
    print("  shoes_compact_ladder_nuisance_orthogonal_gate_v1_outputs/nuisance_orthogonal_gate_v1_summary.txt")
    print("  shoes_compact_ladder_nuisance_orthogonal_gate_v1_outputs/nuisance_orthogonal_gate_v1_results.csv")
    print("  shoes_compact_ladder_nuisance_orthogonal_gate_v1_outputs/nuisance_orthogonal_gate_v1_top_drivers.csv")
    print("")
    print("Boundary:")
    print("  This is nuisance-orthogonal projection before physical interpretation.")
    print("  This is not a TAIRID gate proof.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
