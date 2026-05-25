#!/usr/bin/env python3
"""
TAIRID SH0ES compact ladder matrix retrieval + parser v1.

Purpose:
The calibrator-boundary residual geometry test found that Pantheon+SH0ES
supernova residuals still contain group-specific geometry after ladder-boundary
offsets are included.

The next wall is not another Pantheon-only residual screen.

The next wall is the compact SH0ES ladder system:

    ally = y data vector
    alll = L equation/design matrix
    allc = C covariance matrix

This test does not fit TAIRID yet.

It only asks:
1. Can we retrieve the real Git LFS FITS payloads, especially allc?
2. Can we parse allc, alll, and ally?
3. Do the matrix dimensions line up?
4. Is C usable as a covariance matrix?
5. Can the compact linear system run a baseline GLS sanity solve?

Boundary:
This is not proof of TAIRID.
This is not the final SH0ES likelihood test.
This is not BAO.
This is not Planck.
This is a data-readiness and compact-matrix parser test.

If successful, the next test can ask:
Can a calibrator-boundary gate enter the compact ladder system without merely
renaming itself as an offset?
"""

import csv
import json
import math
import re
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve


OUTDIR = Path("tairid_shoes_compact_ladder_matrix_parser_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

OWNER = "PantheonPlusSH0ES"
REPO = "DataRelease"
BRANCH = "main"

GITHUB_API_CONTENTS = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/SH0ES_Data?ref={BRANCH}"

COMPACT_FILES = {
    "allc": "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits",
    "alll": "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits",
    "ally": "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits",
}

AUX_FILES = [
    "SH0ES_Data/MCMC_utils.py",
    "SH0ES_Data/run_mcmc.py",
    "SH0ES_Data/lstsq_results.txt",
    "SH0ES_Data/optical_wes_R22_for19fromR16.dat",
    "SH0ES_Data/README.md",
    "SH0ES_Data/README.txt",
    "SH0ES_Data/readme.md",
    "SH0ES_Data/readme.txt",
]


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")
    return path


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

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

    return path


def safe_name(path):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(path))[:220]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def fetch_url(url, timeout=900):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-SH0ES-compact-ladder-parser-v1",
            "Accept": "*/*",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        status = getattr(response, "status", None)

    return data, final_url, content_type, status


def is_git_lfs_pointer(data):
    if data is None:
        return False

    head = data[:200].decode("utf-8", errors="replace")
    return "version https://git-lfs.github.com/spec/v1" in head and "oid sha256:" in head


def parse_lfs_pointer(data):
    text = data.decode("utf-8", errors="replace")
    out = {
        "is_lfs_pointer": True,
        "raw_text": text,
    }

    oid = re.search(r"oid sha256:([a-fA-F0-9]+)", text)
    size = re.search(r"size\s+([0-9]+)", text)

    if oid:
        out["oid_sha256"] = oid.group(1)

    if size:
        out["declared_size"] = int(size.group(1))

    return out


def candidate_urls(repo_path):
    quoted_path = urllib.parse.quote(repo_path, safe="/._-+")

    return [
        {
            "kind": "raw_githubusercontent",
            "url": f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{quoted_path}",
        },
        {
            "kind": "media_githubusercontent",
            "url": f"https://media.githubusercontent.com/media/{OWNER}/{REPO}/{BRANCH}/{quoted_path}",
        },
        {
            "kind": "github_raw",
            "url": f"https://github.com/{OWNER}/{REPO}/raw/{BRANCH}/{quoted_path}",
        },
    ]


def download_repo_path(repo_path, label=None, required=False):
    label = label or safe_name(repo_path)
    local = DOWNLOAD_DIR / safe_name(repo_path)

    attempts = []
    pointer_info = None

    for cand in candidate_urls(repo_path):
        url = cand["url"]

        try:
            data, final_url, content_type, status = fetch_url(url)

            attempt = {
                "repo_path": repo_path,
                "label": label,
                "candidate_kind": cand["kind"],
                "url": url,
                "final_url": final_url,
                "http_status": status,
                "content_type": content_type,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }

            if is_git_lfs_pointer(data):
                p = parse_lfs_pointer(data)
                attempt.update(p)
                attempt["status"] = "git_lfs_pointer_not_payload"
                pointer_info = p
                attempts.append(attempt)
                continue

            local.write_bytes(data)

            attempt["status"] = "downloaded_real_payload"
            attempt["local_path"] = str(local)
            attempt["file_sha256"] = sha256_file(local)
            attempts.append(attempt)

            return {
                "repo_path": repo_path,
                "label": label,
                "status": "downloaded",
                "local_path": str(local),
                "bytes": local.stat().st_size,
                "sha256": sha256_file(local),
                "attempts": attempts,
                "pointer_info": pointer_info,
            }

        except urllib.error.HTTPError as exc:
            attempts.append(
                {
                    "repo_path": repo_path,
                    "label": label,
                    "candidate_kind": cand["kind"],
                    "url": url,
                    "status": "http_error",
                    "http_code": exc.code,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "repo_path": repo_path,
                    "label": label,
                    "candidate_kind": cand["kind"],
                    "url": url,
                    "status": "download_failed",
                    "error": str(exc),
                }
            )

    return {
        "repo_path": repo_path,
        "label": label,
        "status": "failed_required" if required else "failed_optional",
        "local_path": None,
        "attempts": attempts,
        "pointer_info": pointer_info,
    }


def try_github_folder_inventory():
    try:
        data, final_url, content_type, status = fetch_url(GITHUB_API_CONTENTS, timeout=300)
        text = data.decode("utf-8", errors="replace")
        obj = json.loads(text)

        write_json(OUTDIR / "shoes_data_github_folder_inventory.json", obj)

        rows = []
        for item in obj:
            rows.append(
                {
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "size": item.get("size"),
                    "type": item.get("type"),
                    "download_url": item.get("download_url"),
                    "html_url": item.get("html_url"),
                }
            )

        write_csv(OUTDIR / "shoes_data_github_folder_inventory.csv", rows)

        return {
            "status": "ok",
            "count": len(rows),
            "final_url": final_url,
            "content_type": content_type,
            "http_status": status,
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
        }


def fits_hdu_inventory(path):
    from astropy.io import fits

    rows = []

    with fits.open(path, memmap=True) as hdul:
        for idx, hdu in enumerate(hdul):
            data = hdu.data
            shape = None
            dtype = None
            fields = None

            if data is not None:
                try:
                    shape = tuple(data.shape)
                except Exception:
                    shape = None

                try:
                    dtype = str(data.dtype)
                except Exception:
                    dtype = None

                try:
                    if getattr(data.dtype, "fields", None):
                        fields = list(data.dtype.fields.keys())
                except Exception:
                    fields = None

            rows.append(
                {
                    "file": str(path),
                    "hdu_index": idx,
                    "hdu_name": hdu.name,
                    "hdu_class": hdu.__class__.__name__,
                    "shape": str(shape),
                    "dtype": dtype,
                    "fields": " | ".join(fields) if fields else "",
                    "naxis": hdu.header.get("NAXIS"),
                    "bitpix": hdu.header.get("BITPIX"),
                    "xtension": hdu.header.get("XTENSION"),
                }
            )

    return rows


def extract_first_numeric_fits_array(path):
    from astropy.io import fits

    inventory = fits_hdu_inventory(path)

    with fits.open(path, memmap=True) as hdul:
        for idx, hdu in enumerate(hdul):
            data = hdu.data

            if data is None:
                continue

            arr = None

            try:
                if getattr(data.dtype, "fields", None):
                    field_names = list(data.dtype.fields.keys())

                    if len(field_names) == 1:
                        arr = np.asarray(data[field_names[0]])
                    else:
                        numeric_fields = []
                        for name in field_names:
                            vals = np.asarray(data[name])
                            if np.issubdtype(vals.dtype, np.number):
                                numeric_fields.append(vals)

                        if len(numeric_fields) == 1:
                            arr = np.asarray(numeric_fields[0])
                        elif len(numeric_fields) > 1:
                            try:
                                arr = np.column_stack([np.asarray(v).reshape(len(v), -1) for v in numeric_fields])
                            except Exception:
                                arr = None
                else:
                    arr = np.asarray(data)

                if arr is None:
                    continue

                arr = np.asarray(arr)

                if not np.issubdtype(arr.dtype, np.number):
                    continue

                arr = np.squeeze(arr).astype(np.float64)

                if arr.size == 0:
                    continue

                return arr, {
                    "selected_hdu_index": idx,
                    "selected_hdu_name": hdu.name,
                    "selected_shape": list(arr.shape),
                    "selected_dtype": str(arr.dtype),
                    "fits_inventory": inventory,
                }

            except Exception:
                continue

    raise RuntimeError(f"No numeric FITS array found in {path}")


def matrix_basic_stats(name, arr):
    arr = np.asarray(arr)

    finite = np.isfinite(arr)
    vals = arr[finite]

    out = {
        "name": name,
        "shape": list(arr.shape),
        "ndim": int(arr.ndim),
        "size": int(arr.size),
        "finite_count": int(np.sum(finite)),
        "nan_count": int(np.sum(np.isnan(arr))),
        "inf_count": int(np.sum(np.isinf(arr))),
    }

    if vals.size:
        out.update(
            {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
            }
        )

    return out


def covariance_diagnostics(C):
    C = np.asarray(C, dtype=np.float64)

    out = {
        "shape": list(C.shape),
        "is_square": bool(C.ndim == 2 and C.shape[0] == C.shape[1]),
    }

    if not out["is_square"]:
        return out

    diag = np.diag(C)
    asym = C - C.T

    out.update(
        {
            "diag_min": float(np.min(diag)),
            "diag_max": float(np.max(diag)),
            "diag_mean": float(np.mean(diag)),
            "diag_median": float(np.median(diag)),
            "diag_nonpositive_count": int(np.sum(diag <= 0)),
            "max_abs_asymmetry": float(np.max(np.abs(asym))),
            "mean_abs_asymmetry": float(np.mean(np.abs(asym))),
            "relative_max_asymmetry": float(np.max(np.abs(asym)) / max(np.max(np.abs(C)), 1.0e-300)),
        }
    )

    return out


def stable_cholesky_cov(C):
    C = np.asarray(C, dtype=np.float64)
    C_sym = 0.5 * (C + C.T)
    diag = np.diag(C_sym)

    scale = float(np.median(diag[diag > 0])) if np.any(diag > 0) else 1.0
    jitter = 0.0

    attempts = []

    for i in range(12):
        try:
            if jitter == 0.0:
                cf = cho_factor(C_sym, lower=True, check_finite=False)
            else:
                cf = cho_factor(C_sym + np.eye(C_sym.shape[0]) * jitter, lower=True, check_finite=False)

            attempts.append(
                {
                    "attempt": i,
                    "jitter": jitter,
                    "status": "success",
                }
            )

            return cf, C_sym, jitter, attempts

        except Exception as exc:
            attempts.append(
                {
                    "attempt": i,
                    "jitter": jitter,
                    "status": "failed",
                    "error": str(exc),
                }
            )

            if jitter == 0.0:
                jitter = max(scale * 1.0e-12, 1.0e-14)
            else:
                jitter *= 10.0

    raise RuntimeError("Cholesky failed even with jitter.")


def determine_design_orientation(L, y_length):
    L = np.asarray(L, dtype=np.float64)

    if L.ndim != 2:
        return None, {
            "status": "L_not_2d",
            "L_shape": list(L.shape),
            "y_length": int(y_length),
        }

    if L.shape[0] == y_length and L.shape[1] != y_length:
        return L, {
            "status": "ok",
            "orientation": "L_is_observation_by_parameter",
            "X_shape": list(L.shape),
            "observation_axis": 0,
            "parameter_count": int(L.shape[1]),
        }

    if L.shape[1] == y_length and L.shape[0] != y_length:
        X = L.T
        return X, {
            "status": "ok",
            "orientation": "L_transposed_to_observation_by_parameter",
            "original_L_shape": list(L.shape),
            "X_shape": list(X.shape),
            "observation_axis": 1,
            "parameter_count": int(X.shape[1]),
        }

    if L.shape[0] == y_length and L.shape[1] == y_length:
        return L, {
            "status": "ambiguous_square_L",
            "orientation": "using_L_as_observation_by_parameter",
            "X_shape": list(L.shape),
            "parameter_count": int(L.shape[1]),
        }

    return None, {
        "status": "no_L_axis_matches_y_length",
        "L_shape": list(L.shape),
        "y_length": int(y_length),
    }


def gls_sanity_solve(y, L, C):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X, orient = determine_design_orientation(L, len(y))

    if X is None:
        return {
            "status": "failed_orientation",
            "orientation": orient,
        }, None

    if C.ndim != 2 or C.shape[0] != len(y) or C.shape[1] != len(y):
        return {
            "status": "failed_covariance_shape",
            "orientation": orient,
            "C_shape": list(C.shape),
            "y_length": int(len(y)),
        }, None

    cf, C_sym, jitter, chol_attempts = stable_cholesky_cov(C)

    Cinv_y = cho_solve(cf, y, check_finite=False)
    Cinv_X = cho_solve(cf, X, check_finite=False)

    normal = X.T @ Cinv_X
    rhs = X.T @ Cinv_y

    normal_inv = np.linalg.pinv(normal, rcond=1.0e-12)
    beta = normal_inv @ rhs

    residual = y - X @ beta
    Cinv_res = cho_solve(cf, residual, check_finite=False)
    chi2_value = float(residual.T @ Cinv_res)
    dof = int(len(y) - X.shape[1])

    beta_err = np.sqrt(np.maximum(np.diag(normal_inv), 0.0))

    diagnostics = {
        "status": "ok",
        "orientation": orient,
        "y_length": int(len(y)),
        "X_shape": list(X.shape),
        "C_shape": list(C.shape),
        "parameter_count": int(X.shape[1]),
        "cholesky_jitter": jitter,
        "cholesky_attempts": chol_attempts,
        "normal_shape": list(normal.shape),
        "normal_condition_estimate": float(np.linalg.cond(normal)) if normal.size <= 10000 else None,
        "chi2": chi2_value,
        "dof": dof,
        "reduced_chi2": float(chi2_value / dof) if dof > 0 else float("nan"),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "beta_count": int(len(beta)),
        "beta_preview_first_20": [float(v) for v in beta[:20]],
        "beta_err_preview_first_20": [float(v) for v in beta_err[:20]],
    }

    result_arrays = {
        "X": X,
        "C_sym": C_sym,
        "beta": beta,
        "beta_err": beta_err,
        "residual": residual,
        "normal": normal,
    }

    return diagnostics, result_arrays


def write_matrix_samples(mats):
    rows = []

    for name, arr in mats.items():
        arr = np.asarray(arr)

        if arr.ndim == 1:
            for i in range(min(50, arr.shape[0])):
                rows.append(
                    {
                        "matrix": name,
                        "i": i,
                        "j": "",
                        "value": float(arr[i]),
                    }
                )

        elif arr.ndim == 2:
            imax = min(15, arr.shape[0])
            jmax = min(15, arr.shape[1])

            for i in range(imax):
                for j in range(jmax):
                    rows.append(
                        {
                            "matrix": name,
                            "i": i,
                            "j": j,
                            "value": float(arr[i, j]),
                        }
                    )

    write_csv(OUTDIR / "compact_matrix_samples_v1.csv", rows)


def write_gls_outputs(result_arrays):
    if result_arrays is None:
        return []

    output_files = []

    beta = result_arrays["beta"]
    beta_err = result_arrays["beta_err"]
    residual = result_arrays["residual"]
    normal = result_arrays["normal"]

    beta_rows = []
    for i, (b, e) in enumerate(zip(beta, beta_err)):
        beta_rows.append(
            {
                "parameter_index": i,
                "beta": float(b),
                "beta_err": float(e),
            }
        )

    write_csv(OUTDIR / "compact_ladder_gls_beta_v1.csv", beta_rows)
    output_files.append(str(OUTDIR / "compact_ladder_gls_beta_v1.csv"))

    residual_rows = []
    for i in range(min(len(residual), 500)):
        residual_rows.append(
            {
                "observation_index": i,
                "residual": float(residual[i]),
            }
        )

    write_csv(OUTDIR / "compact_ladder_gls_residual_preview_v1.csv", residual_rows)
    output_files.append(str(OUTDIR / "compact_ladder_gls_residual_preview_v1.csv"))

    normal_rows = []
    nmax = min(30, normal.shape[0])
    mmax = min(30, normal.shape[1])

    for i in range(nmax):
        for j in range(mmax):
            normal_rows.append(
                {
                    "i": i,
                    "j": j,
                    "value": float(normal[i, j]),
                }
            )

    write_csv(OUTDIR / "compact_ladder_normal_matrix_preview_v1.csv", normal_rows)
    output_files.append(str(OUTDIR / "compact_ladder_normal_matrix_preview_v1.csv"))

    plt.figure(figsize=(10, 6))
    plt.hist(residual, bins=80)
    plt.xlabel("GLS residual")
    plt.ylabel("count")
    plt.title("SH0ES compact ladder GLS residual preview")
    plt.tight_layout()
    plt.savefig(OUTDIR / "compact_ladder_gls_residual_hist_v1.png", dpi=160)
    plt.close()
    output_files.append(str(OUTDIR / "compact_ladder_gls_residual_hist_v1.png"))

    plt.figure(figsize=(10, 6))
    plt.plot(beta, marker="o", linewidth=1)
    plt.xlabel("parameter index")
    plt.ylabel("GLS beta")
    plt.title("SH0ES compact ladder GLS beta preview")
    plt.tight_layout()
    plt.savefig(OUTDIR / "compact_ladder_gls_beta_plot_v1.png", dpi=160)
    plt.close()
    output_files.append(str(OUTDIR / "compact_ladder_gls_beta_plot_v1.png"))

    return output_files


def inspect_aux_text(download_results):
    rows = []

    for item in download_results:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["local_path"])
        lower = path.name.lower()

        if not lower.endswith((".txt", ".md", ".py", ".dat")):
            continue

        try:
            text = path.read_text(errors="replace")
            lines = text.splitlines()
            rows.append(
                {
                    "repo_path": item["repo_path"],
                    "local_path": str(path),
                    "line_count": len(lines),
                    "char_count": len(text),
                    "first_20_lines": "\n".join(lines[:20]),
                }
            )

            preview_path = OUTDIR / f"preview_{safe_name(item['repo_path'])}.txt"
            preview_path.write_text("\n".join(lines[:120]), encoding="utf-8")

        except Exception as exc:
            rows.append(
                {
                    "repo_path": item["repo_path"],
                    "local_path": str(path),
                    "error": str(exc),
                }
            )

    write_csv(OUTDIR / "auxiliary_text_file_previews_v1.csv", rows)

    return rows


def decide_status(downloads, parsed, dimension_report, cov_diag, gls_diag):
    all_compact_downloaded = all(downloads.get(k, {}).get("status") == "downloaded" for k in ["allc", "alll", "ally"])
    all_compact_parsed = all(k in parsed for k in ["allc", "alll", "ally"])

    C_ok = bool(cov_diag.get("is_square")) if cov_diag else False
    dims_ok = dimension_report.get("status") in ["matrix_axes_align", "matrix_axes_align_but_C_shape_problem"]
    gls_ok = gls_diag.get("status") == "ok" if gls_diag else False

    if all_compact_downloaded and all_compact_parsed and C_ok and dims_ok and gls_ok:
        return (
            "compact_shoes_ladder_matrices_ready_for_likelihood_test",
            9,
            "Build the next test: SH0ES compact ladder boundary-gate likelihood v1 using y, L, and C directly.",
        )

    if all_compact_downloaded and all_compact_parsed and dims_ok:
        return (
            "compact_shoes_ladder_matrices_retrieved_partial_solver_issue",
            8,
            "Inspect covariance/cholesky/GLS diagnostics, then retry likelihood setup with regularization if needed.",
        )

    if downloads.get("allc", {}).get("status") != "downloaded" and downloads.get("alll", {}).get("status") == "downloaded" and downloads.get("ally", {}).get("status") == "downloaded":
        return (
            "partial_parser_needed_allc_covariance_not_retrieved",
            7,
            "The L and y sides are reachable but C is still blocked; use Git LFS/media retrieval or release asset mirror.",
        )

    return (
        "compact_shoes_ladder_not_ready",
        5,
        "Resolve download/parser failures before attempting a ladder likelihood.",
    )


def main():
    print("")
    print("TAIRID SH0ES compact ladder matrix retrieval + parser v1 starting.")
    print("Boundary: data-readiness and compact-matrix parser only; not a TAIRID fit.")
    print("")

    folder_inventory_status = try_github_folder_inventory()

    download_results = {}
    download_ledger_rows = []

    for label, repo_path in COMPACT_FILES.items():
        result = download_repo_path(repo_path, label=label, required=True)
        download_results[label] = result

        row = {
            "label": label,
            "repo_path": repo_path,
            "status": result.get("status"),
            "local_path": result.get("local_path"),
            "bytes": result.get("bytes"),
            "sha256": result.get("sha256"),
            "pointer_declared_size": (result.get("pointer_info") or {}).get("declared_size"),
            "pointer_oid_sha256": (result.get("pointer_info") or {}).get("oid_sha256"),
            "attempt_count": len(result.get("attempts", [])),
        }
        download_ledger_rows.append(row)

    aux_results = []

    for repo_path in AUX_FILES:
        result = download_repo_path(repo_path, label=safe_name(repo_path), required=False)
        aux_results.append(result)

        download_ledger_rows.append(
            {
                "label": safe_name(repo_path),
                "repo_path": repo_path,
                "status": result.get("status"),
                "local_path": result.get("local_path"),
                "bytes": result.get("bytes"),
                "sha256": result.get("sha256"),
                "pointer_declared_size": (result.get("pointer_info") or {}).get("declared_size"),
                "pointer_oid_sha256": (result.get("pointer_info") or {}).get("oid_sha256"),
                "attempt_count": len(result.get("attempts", [])),
            }
        )

    write_csv(OUTDIR / "shoes_compact_download_ledger_v1.csv", download_ledger_rows)
    write_json(OUTDIR / "shoes_compact_download_attempts_v1.json", {"compact": download_results, "auxiliary": aux_results})

    aux_previews = inspect_aux_text(aux_results)

    parsed = {}
    fits_inventory_rows = []
    matrix_stats = {}
    parse_errors = []

    for label in ["allc", "alll", "ally"]:
        result = download_results.get(label, {})

        if result.get("status") != "downloaded":
            parse_errors.append(
                {
                    "label": label,
                    "status": "not_downloaded",
                    "download_status": result.get("status"),
                }
            )
            continue

        path = Path(result["local_path"])

        try:
            arr, meta = extract_first_numeric_fits_array(path)
            parsed[label] = arr
            matrix_stats[label] = matrix_basic_stats(label, arr)

            for r in meta["fits_inventory"]:
                out = {"label": label}
                out.update(r)
                fits_inventory_rows.append(out)

            write_json(OUTDIR / f"{label}_fits_parse_meta_v1.json", meta)

        except Exception as exc:
            parse_errors.append(
                {
                    "label": label,
                    "status": "parse_failed",
                    "path": str(path),
                    "error": str(exc),
                }
            )

    write_csv(OUTDIR / "shoes_compact_fits_inventory_v1.csv", fits_inventory_rows)
    write_json(OUTDIR / "shoes_compact_parse_errors_v1.json", parse_errors)
    write_json(OUTDIR / "shoes_compact_matrix_basic_stats_v1.json", matrix_stats)

    dimension_report = {
        "status": "not_enough_matrices",
        "available_labels": sorted(parsed.keys()),
    }

    cov_diag = {}
    gls_diag = {}
    gls_output_files = []

    if all(k in parsed for k in ["allc", "alll", "ally"]):
        C = np.asarray(parsed["allc"], dtype=np.float64)
        L = np.asarray(parsed["alll"], dtype=np.float64)
        y = np.asarray(parsed["ally"], dtype=np.float64).reshape(-1)

        write_matrix_samples({"allc": C, "alll": L, "ally": y})

        cov_diag = covariance_diagnostics(C)

        X, orient = determine_design_orientation(L, len(y))

        dimension_report = {
            "status": "matrix_axes_align",
            "y_length": int(len(y)),
            "allc_shape": list(C.shape),
            "alll_shape": list(L.shape),
            "ally_shape_original": list(np.asarray(parsed["ally"]).shape),
            "ally_shape_flat": list(y.shape),
            "L_orientation": orient,
            "C_matches_y": bool(C.ndim == 2 and C.shape[0] == len(y) and C.shape[1] == len(y)),
            "L_has_y_axis": bool(orient.get("status") in ["ok", "ambiguous_square_L"]),
        }

        if not dimension_report["C_matches_y"] or not dimension_report["L_has_y_axis"]:
            dimension_report["status"] = "matrix_axes_do_not_align"

        if dimension_report["C_matches_y"] and dimension_report["L_has_y_axis"]:
            try:
                gls_diag, gls_arrays = gls_sanity_solve(y, L, C)
                gls_output_files = write_gls_outputs(gls_arrays)
            except Exception as exc:
                gls_diag = {
                    "status": "gls_failed",
                    "error": str(exc),
                }

        write_json(OUTDIR / "shoes_compact_dimension_report_v1.json", dimension_report)
        write_json(OUTDIR / "shoes_compact_covariance_diagnostics_v1.json", cov_diag)
        write_json(OUTDIR / "shoes_compact_gls_sanity_diagnostics_v1.json", gls_diag)

        if C.ndim == 2 and C.shape[0] == C.shape[1]:
            diag = np.diag(C)
            plt.figure(figsize=(10, 6))
            plt.hist(diag[np.isfinite(diag)], bins=80)
            plt.xlabel("covariance diagonal")
            plt.ylabel("count")
            plt.title("SH0ES compact covariance diagonal")
            plt.tight_layout()
            plt.savefig(OUTDIR / "shoes_compact_covariance_diag_hist_v1.png", dpi=160)
            plt.close()
            gls_output_files.append(str(OUTDIR / "shoes_compact_covariance_diag_hist_v1.png"))

        plt.figure(figsize=(10, 6))
        plt.hist(y[np.isfinite(y)], bins=80)
        plt.xlabel("y value")
        plt.ylabel("count")
        plt.title("SH0ES compact y-vector distribution")
        plt.tight_layout()
        plt.savefig(OUTDIR / "shoes_compact_y_hist_v1.png", dpi=160)
        plt.close()
        gls_output_files.append(str(OUTDIR / "shoes_compact_y_hist_v1.png"))

    else:
        write_json(OUTDIR / "shoes_compact_dimension_report_v1.json", dimension_report)
        write_json(OUTDIR / "shoes_compact_covariance_diagnostics_v1.json", cov_diag)
        write_json(OUTDIR / "shoes_compact_gls_sanity_diagnostics_v1.json", gls_diag)

    final_status, readiness_score, next_wall = decide_status(
        download_results,
        parsed,
        dimension_report,
        cov_diag,
        gls_diag,
    )

    summary = {
        "test_name": "TAIRID SH0ES compact ladder matrix retrieval + parser v1",
        "boundary": (
            "Data-readiness and compact-matrix parser only. Not a TAIRID fit, "
            "not proof, not BAO, not Planck, and not the final SH0ES likelihood test."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
        "github": {
            "owner": OWNER,
            "repo": REPO,
            "branch": BRANCH,
            "folder_inventory_status": folder_inventory_status,
        },
        "compact_files": COMPACT_FILES,
        "download_summary": {
            label: {
                "status": result.get("status"),
                "local_path": result.get("local_path"),
                "bytes": result.get("bytes"),
                "sha256": result.get("sha256"),
                "pointer_info": result.get("pointer_info"),
            }
            for label, result in download_results.items()
        },
        "parse_errors": parse_errors,
        "matrix_basic_stats": matrix_stats,
        "dimension_report": dimension_report,
        "covariance_diagnostics": cov_diag,
        "gls_sanity_diagnostics": gls_diag,
        "auxiliary_file_previews": aux_previews,
        "output_files": {
            "summary_json": str(OUTDIR / "shoes_compact_ladder_matrix_parser_v1_summary.json"),
            "summary_txt": str(OUTDIR / "shoes_compact_ladder_matrix_parser_v1_summary.txt"),
            "download_ledger_csv": str(OUTDIR / "shoes_compact_download_ledger_v1.csv"),
            "download_attempts_json": str(OUTDIR / "shoes_compact_download_attempts_v1.json"),
            "fits_inventory_csv": str(OUTDIR / "shoes_compact_fits_inventory_v1.csv"),
            "matrix_basic_stats_json": str(OUTDIR / "shoes_compact_matrix_basic_stats_v1.json"),
            "dimension_report_json": str(OUTDIR / "shoes_compact_dimension_report_v1.json"),
            "covariance_diagnostics_json": str(OUTDIR / "shoes_compact_covariance_diagnostics_v1.json"),
            "gls_sanity_diagnostics_json": str(OUTDIR / "shoes_compact_gls_sanity_diagnostics_v1.json"),
            "matrix_samples_csv": str(OUTDIR / "compact_matrix_samples_v1.csv"),
            "gls_output_files": gls_output_files,
        },
        "interpretation": {
            "what_success_means": (
                "The compact SH0ES ladder matrices were retrieved and parsed, including C. "
                "The next test can operate on y, L, and C directly."
            ),
            "what_failure_means": (
                "A failed run is a retrieval/parser wall, not a cosmology failure."
            ),
            "next_likelihood_question": (
                "Can a calibrator-boundary gate enter the compact ladder system without simply becoming an offset?"
            ),
            "truth_boundary": (
                "Even a successful parser does not prove TAIRID. It only opens the next real likelihood wall."
            ),
        },
    }

    write_json(OUTDIR / "shoes_compact_ladder_matrix_parser_v1_summary.json", summary)

    with open(OUTDIR / "shoes_compact_ladder_matrix_parser_v1_summary.txt", "w", encoding="utf-8") as f:
        f.write("TAIRID SH0ES compact ladder matrix retrieval + parser v1\n\n")
        f.write("Boundary: data-readiness and compact-matrix parser only.\n")
        f.write("Not proof of TAIRID. Not the final SH0ES likelihood test. Not BAO. Not Planck.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")

        f.write("Compact files:\n")
        f.write(json.dumps(COMPACT_FILES, indent=2) + "\n\n")

        f.write("Download summary:\n")
        f.write(json.dumps(summary["download_summary"], indent=2, default=json_default) + "\n\n")

        f.write("Dimension report:\n")
        f.write(json.dumps(dimension_report, indent=2, default=json_default) + "\n\n")

        f.write("Covariance diagnostics:\n")
        f.write(json.dumps(cov_diag, indent=2, default=json_default) + "\n\n")

        f.write("GLS sanity diagnostics:\n")
        f.write(json.dumps(gls_diag, indent=2, default=json_default) + "\n\n")

        f.write("Truth boundary:\n")
        f.write("- A successful run only means the compact matrix wall is open.\n")
        f.write("- It does not prove TAIRID.\n")
        f.write("- It does not prove H0 resolution.\n")
        f.write("- The next test must be a bounded compact-ladder likelihood test.\n")

    print("")
    print("TAIRID SH0ES compact ladder matrix retrieval + parser v1 complete.")
    print("Created:")
    print("  tairid_shoes_compact_ladder_matrix_parser_v1_outputs/shoes_compact_ladder_matrix_parser_v1_summary.json")
    print("  tairid_shoes_compact_ladder_matrix_parser_v1_outputs/shoes_compact_ladder_matrix_parser_v1_summary.txt")
    print("  tairid_shoes_compact_ladder_matrix_parser_v1_outputs/shoes_compact_download_ledger_v1.csv")
    print("  tairid_shoes_compact_ladder_matrix_parser_v1_outputs/shoes_compact_fits_inventory_v1.csv")
    print("  tairid_shoes_compact_ladder_matrix_parser_v1_outputs/shoes_compact_gls_sanity_diagnostics_v1.json")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is a compact SH0ES matrix retrieval/parser test.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
