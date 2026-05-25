#!/usr/bin/env python3
"""
TAIRID SH0ES equation-edge likelihood v2.1.

Corrected replacement for v2.

Purpose:
The row-mask candidate-boundary likelihood v2 failed because blunt 277/77 row
membership masks were mostly absorbed by the original 47-parameter SH0ES compact
ladder model.

The equation-edge recovery v1 found the sharper structure:

    277 rows bridge parameter 42 and parameter 46
    active columns = 42,46
    sign pattern = +,-

Parameter 46 is treated as the H0-like compact-ladder parameter because the
SH0ES utility code uses:

    H0 = 10^(fivelogH0 / 5)

This test asks whether a constrained deformation of the recovered 42<->46 edge
improves the compact SH0ES ladder likelihood beyond the baseline 47-parameter
model.

Boundary:
This is not proof of TAIRID.
This is not H0 resolution.
This is not BAO, Planck, or a full cosmology model.
This only tests whether the recovered 42<->46 equation edge carries independent
likelihood pressure under a constrained deformation.
"""

import csv
import json
import math
import re
import hashlib
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


OUTDIR = Path("tairid_shoes_equation_edge_likelihood_v2_1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

OWNER = "PantheonPlusSH0ES"
REPO = "DataRelease"
BRANCH = "main"

COMPACT = {
    "allc": "SH0ES_Data/allc_shoes_ceph_topantheonwt6.0_112221.fits",
    "alll": "SH0ES_Data/alll_shoes_ceph_topantheonwt6.0_112221.fits",
    "ally": "SH0ES_Data/ally_shoes_ceph_topantheonwt6.0_112221.fits",
}

AUX = [
    "SH0ES_Data/MCMC_utils.py",
    "SH0ES_Data/run_mcmc.py",
    "SH0ES_Data/lstsq_results.txt",
    "SH0ES_Data/README.md",
]

P42 = 42
P46 = 46
EPS = 1.0e-12
SEED = 42

LAMBDA_GRID = np.linspace(-0.08, 0.08, 33)
CONTROL_LAMBDA_GRID = np.linspace(-0.08, 0.08, 17)
RANDOM_CONTROL_REPEATS = 20


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


def candidate_urls(repo_path):
    quoted = urllib.parse.quote(repo_path, safe="/._-+")

    return [
        (
            "raw_githubusercontent",
            f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{quoted}",
        ),
        (
            "media_githubusercontent",
            f"https://media.githubusercontent.com/media/{OWNER}/{REPO}/{BRANCH}/{quoted}",
        ),
        (
            "github_raw",
            f"https://github.com/{OWNER}/{REPO}/raw/{BRANCH}/{quoted}",
        ),
    ]


def is_lfs_pointer(data):
    head = data[:220].decode("utf-8", errors="replace")
    return "version https://git-lfs.github.com/spec/v1" in head and "oid sha256:" in head


def parse_lfs_pointer(data):
    text = data.decode("utf-8", errors="replace")
    oid = re.search(r"oid sha256:([a-fA-F0-9]+)", text)
    size = re.search(r"size\s+([0-9]+)", text)

    return {
        "oid_sha256": oid.group(1) if oid else None,
        "declared_size": int(size.group(1)) if size else None,
        "raw_text": text,
    }


def fetch_url(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TAIRID-SH0ES-edge-v2-1",
            "Accept": "*/*",
        },
    )

    with urllib.request.urlopen(req, timeout=900) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        status = getattr(response, "status", None)

    return data, final_url, content_type, status


def download_repo_path(repo_path, label):
    local = DOWNLOAD_DIR / safe_name(repo_path)
    attempts = []
    pointer_info = None

    for kind, url in candidate_urls(repo_path):
        try:
            data, final_url, content_type, status = fetch_url(url)

            attempt = {
                "label": label,
                "repo_path": repo_path,
                "candidate_kind": kind,
                "url": url,
                "final_url": final_url,
                "http_status": status,
                "content_type": content_type,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }

            if is_lfs_pointer(data):
                pointer_info = parse_lfs_pointer(data)
                attempt.update(pointer_info)
                attempt["status"] = "git_lfs_pointer_not_payload"
                attempts.append(attempt)
                continue

            local.write_bytes(data)

            attempt["status"] = "downloaded_real_payload"
            attempt["local_path"] = str(local)
            attempt["file_sha256"] = sha256_file(local)
            attempts.append(attempt)

            return {
                "label": label,
                "repo_path": repo_path,
                "status": "downloaded",
                "local_path": str(local),
                "bytes": local.stat().st_size,
                "sha256": sha256_file(local),
                "pointer_info": pointer_info,
                "attempts": attempts,
            }

        except urllib.error.HTTPError as exc:
            attempts.append(
                {
                    "label": label,
                    "repo_path": repo_path,
                    "candidate_kind": kind,
                    "url": url,
                    "status": "http_error",
                    "http_code": exc.code,
                    "error": str(exc),
                }
            )

        except Exception as exc:
            attempts.append(
                {
                    "label": label,
                    "repo_path": repo_path,
                    "candidate_kind": kind,
                    "url": url,
                    "status": "download_failed",
                    "error": str(exc),
                }
            )

    return {
        "label": label,
        "repo_path": repo_path,
        "status": "failed",
        "local_path": None,
        "pointer_info": pointer_info,
        "attempts": attempts,
    }


def extract_first_numeric_fits_array(path):
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        for index, hdu in enumerate(hdul):
            data = hdu.data

            if data is None:
                continue

            try:
                if getattr(data.dtype, "fields", None):
                    numeric = []

                    for name in data.dtype.fields:
                        values = np.asarray(data[name])

                        if np.issubdtype(values.dtype, np.number):
                            numeric.append(values)

                    if len(numeric) == 1:
                        arr = np.asarray(numeric[0])
                    elif len(numeric) > 1:
                        arr = np.column_stack(
                            [np.asarray(v).reshape(len(v), -1) for v in numeric]
                        )
                    else:
                        continue

                else:
                    arr = np.asarray(data)

                arr = np.squeeze(arr)

                if not np.issubdtype(arr.dtype, np.number):
                    continue

                arr = arr.astype(float)

                if arr.size:
                    return arr, {
                        "hdu_index": index,
                        "hdu_name": hdu.name,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }

            except Exception:
                continue

    raise RuntimeError(f"No numeric FITS array found in {path}")


def orient_design_matrix(L, y_length):
    L = np.asarray(L, float)

    if L.ndim != 2:
        return None, {"status": "L_not_2d", "L_shape": list(L.shape)}

    if L.shape[0] == y_length and L.shape[1] != y_length:
        return L, {
            "status": "ok",
            "orientation": "L_is_observation_by_parameter",
            "X_shape": list(L.shape),
        }

    if L.shape[1] == y_length and L.shape[0] != y_length:
        X = L.T

        return X, {
            "status": "ok",
            "orientation": "L_transposed_to_observation_by_parameter",
            "original_L_shape": list(L.shape),
            "X_shape": list(X.shape),
        }

    if L.shape[0] == y_length and L.shape[1] == y_length:
        return L, {
            "status": "ambiguous_square_L",
            "X_shape": list(L.shape),
        }

    return None, {
        "status": "no_axis_matches_y",
        "L_shape": list(L.shape),
        "y_length": int(y_length),
    }


def stable_cholesky(C):
    C_sym = 0.5 * (C + C.T)
    diag = np.diag(C_sym)
    scale = float(np.median(diag[diag > 0])) if np.any(diag > 0) else 1.0

    jitter = 0.0
    attempts = []

    for attempt in range(12):
        try:
            if jitter == 0.0:
                factor = cho_factor(C_sym, lower=True, check_finite=False)
            else:
                factor = cho_factor(
                    C_sym + np.eye(C_sym.shape[0]) * jitter,
                    lower=True,
                    check_finite=False,
                )

            attempts.append(
                {
                    "attempt": attempt,
                    "jitter": jitter,
                    "status": "success",
                }
            )

            return factor, C_sym, jitter, attempts

        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "jitter": jitter,
                    "status": "failed",
                    "error": str(exc),
                }
            )

            if jitter == 0.0:
                jitter = max(scale * 1.0e-12, 1.0e-14)
            else:
                jitter *= 10.0

    raise RuntimeError("Cholesky failed")


def base_gls_fit(y, X, c_factor):
    c_inv_y = cho_solve(c_factor, y, check_finite=False)
    c_inv_x = cho_solve(c_factor, X, check_finite=False)

    y_cinv_y = float(y.T @ c_inv_y)
    normal = X.T @ c_inv_x
    rhs = X.T @ c_inv_y

    normal_inv = np.linalg.pinv(normal, rcond=1.0e-12)
    beta = normal_inv @ rhs
    chi2_value = float(y_cinv_y - 2.0 * beta.T @ rhs + beta.T @ normal @ beta)

    residual = y - X @ beta

    n = len(y)
    k = X.shape[1]
    beta_err = np.sqrt(np.maximum(np.diag(normal_inv), 0.0))

    return {
        "Cinv_y": c_inv_y,
        "Cinv_X": c_inv_x,
        "y_Cinv_y": y_cinv_y,
        "normal": normal,
        "rhs": rhs,
        "normal_inv": normal_inv,
        "beta": beta,
        "beta_err": beta_err,
        "residual": residual,
        "chi2": chi2_value,
        "dof": int(n - k),
        "k": int(k),
        "aic": float(chi2_value + 2 * k),
        "bic": float(chi2_value + k * math.log(n)),
        "reduced_chi2": float(chi2_value / (n - k)),
    }


def h0_like_from_beta(beta):
    if len(beta) <= P46:
        return None

    return float(10.0 ** (beta[P46] / 5.0))


def row_signature(row):
    active = np.where(np.abs(row) > 1.0e-12)[0]
    signs = np.sign(row[active]).astype(int)

    full_key = ",".join(f"{int(i)}:{int(s)}" for i, s in zip(active, signs))
    active_key = ",".join(str(int(i)) for i in active)
    sign_key = ",".join(str(int(s)) for s in signs)

    return active, signs, full_key, active_key, sign_key


def recover_edge_rows(X, y, baseline):
    keys = [row_signature(X[i, :])[2] for i in range(X.shape[0])]
    cluster_counts = {key: keys.count(key) for key in set(keys)}

    row_rows = []
    bridge_mask = np.zeros(X.shape[0], dtype=bool)
    touch42_mask = np.zeros(X.shape[0], dtype=bool)
    touch46_mask = np.zeros(X.shape[0], dtype=bool)

    for i in range(X.shape[0]):
        active, signs, full_key, active_key, sign_key = row_signature(X[i, :])
        active_set = set(map(int, active))

        touches_42 = P42 in active_set
        touches_46 = P46 in active_set
        bridge = full_key == "42:1,46:-1"

        if not bridge and active_set == {P42, P46} and len(active) == 2:
            bridge = True

        bridge_mask[i] = bridge
        touch42_mask[i] = touches_42
        touch46_mask[i] = touches_46

        row_rows.append(
            {
                "observation_index": i,
                "active_cols": active_key,
                "sign_pattern": sign_key,
                "full_signature": full_key,
                "signature_cluster_size": int(cluster_counts[full_key]),
                "nonzero_count": int(len(active)),
                "touches_param42": bool(touches_42),
                "touches_param46_H0_like": bool(touches_46),
                "bridges_param42_param46": bool(bridge),
                "y": float(y[i]),
                "baseline_residual": float(baseline["residual"][i]),
                "abs_baseline_residual": float(abs(baseline["residual"][i])),
            }
        )

    return row_rows, bridge_mask, touch42_mask, touch46_mask


def deformation_matrix(X, mask, kind):
    D = np.zeros_like(X)
    mask = np.asarray(mask, dtype=bool)

    if kind in ("bridge_param46_scale", "param46_scale_on_mask"):
        D[mask, P46] = X[mask, P46]

    elif kind in ("bridge_param42_scale", "param42_scale_on_mask"):
        D[mask, P42] = X[mask, P42]

    elif kind == "bridge_antisym_42_46":
        D[mask, P42] = X[mask, P42]
        D[mask, P46] = -X[mask, P46]

    elif kind == "bridge_common_42_46":
        D[mask, P42] = X[mask, P42]
        D[mask, P46] = X[mask, P46]

    elif kind == "global_param46_scale":
        D[:, P46] = X[:, P46]

    else:
        raise ValueError(kind)

    return D


def solve_from_normal(y_length, y_cinv_y, normal, rhs):
    normal_inv = np.linalg.pinv(normal, rcond=1.0e-12)
    beta = normal_inv @ rhs
    chi2_value = float(y_cinv_y - 2.0 * beta.T @ rhs + beta.T @ normal @ beta)
    dof = int(y_length - len(beta))

    return beta, chi2_value, dof


def fit_deformation_grid_fast(y, X, c_factor, baseline, mask, kind, model_name, lambdas):
    D = deformation_matrix(X, mask, kind)

    c_inv_d = cho_solve(c_factor, D, check_finite=False)

    normal_0 = baseline["normal"]
    rhs_0 = baseline["rhs"]

    normal_1 = D.T @ baseline["Cinv_X"] + X.T @ c_inv_d
    normal_2 = D.T @ c_inv_d
    rhs_1 = D.T @ baseline["Cinv_y"]

    rows = []
    best = None

    y_length = len(y)
    k_eff = X.shape[1] + 1

    for lam in lambdas:
        lam = float(lam)

        normal = normal_0 + lam * normal_1 + lam * lam * normal_2
        rhs = rhs_0 + lam * rhs_1

        beta, chi2_value, dof = solve_from_normal(
            y_length,
            baseline["y_Cinv_y"],
            normal,
            rhs,
        )

        effective_aic = float(chi2_value + 2 * k_eff)
        effective_bic = float(chi2_value + k_eff * math.log(y_length))
        delta_chi2 = float(baseline["chi2"] - chi2_value)

        beta_shift_sigma = np.abs(beta - baseline["beta"]) / np.maximum(
            baseline["beta_err"],
            EPS,
        )

        h0_base = h0_like_from_beta(baseline["beta"])
        h0_model = h0_like_from_beta(beta)

        row = {
            "model_name": model_name,
            "deformation_kind": kind,
            "lambda": lam,
            "mask_count": int(np.sum(mask)),
            "chi2": chi2_value,
            "delta_chi2_improvement": delta_chi2,
            "p_value_chi2_improvement_one_param": float(chi2.sf(max(delta_chi2, 0.0), 1)),
            "effective_aic": effective_aic,
            "delta_aic_model_minus_baseline": float(effective_aic - baseline["aic"]),
            "effective_bic": effective_bic,
            "delta_bic_model_minus_baseline": float(effective_bic - baseline["bic"]),
            "model_dof": dof,
            "model_reduced_chi2": float(chi2_value / dof) if dof > 0 else float("nan"),
            "param42_baseline": float(baseline["beta"][P42]),
            "param42_model": float(beta[P42]),
            "param42_shift_sigma": float(beta_shift_sigma[P42]),
            "param46_fivelogH0_baseline": float(baseline["beta"][P46]),
            "param46_fivelogH0_model": float(beta[P46]),
            "param46_shift_sigma": float(beta_shift_sigma[P46]),
            "param46_H0_like_baseline": h0_base,
            "param46_H0_like_model": h0_model,
            "param46_H0_like_shift": (
                float(h0_model - h0_base)
                if h0_base is not None and h0_model is not None
                else None
            ),
            "max_baseline_param_shift_sigma": float(np.max(beta_shift_sigma)),
            "median_baseline_param_shift_sigma": float(np.median(beta_shift_sigma)),
        }

        rows.append(row)

        if best is None or row["chi2"] < best["chi2"]:
            best = dict(row)

    best["lambda_at_grid_boundary"] = bool(
        abs(best["lambda"] - float(np.min(lambdas))) < 1.0e-14
        or abs(best["lambda"] - float(np.max(lambdas))) < 1.0e-14
    )

    return best, rows


def random_mask_controls(
    y,
    X,
    c_factor,
    baseline,
    observed_best,
    kind,
    count,
    lambdas,
    repeats,
    control_type,
):
    rng = np.random.default_rng(SEED + (101 if control_type == "random_same_count" else 202))

    n = X.shape[0]
    count = int(count)

    rows = []

    if count <= 0 or count >= n:
        return {
            "control_type": control_type,
            "count": count,
            "repeats": 0,
            "p_control_delta_ge_observed": None,
        }, rows

    best_deltas = []

    for repeat_index in range(repeats):
        mask = np.zeros(n, dtype=bool)

        if control_type == "random_same_count":
            mask[rng.choice(n, size=count, replace=False)] = True

        elif control_type == "random_contiguous_block":
            start = int(rng.integers(0, n - count + 1))
            mask[start : start + count] = True

        else:
            raise ValueError(control_type)

        best, _ = fit_deformation_grid_fast(
            y,
            X,
            c_factor,
            baseline,
            mask,
            kind,
            f"{control_type}_{repeat_index}",
            lambdas,
        )

        best_deltas.append(best["delta_chi2_improvement"])

        rows.append(
            {
                "control_type": control_type,
                "control_index": repeat_index,
                "count": count,
                "best_delta_chi2": best["delta_chi2_improvement"],
                "best_lambda": best["lambda"],
                "best_delta_aic": best["delta_aic_model_minus_baseline"],
                "best_delta_bic": best["delta_bic_model_minus_baseline"],
                "best_h0_shift": best["param46_H0_like_shift"],
            }
        )

    best_deltas = np.asarray(best_deltas, dtype=float)

    summary = {
        "control_type": control_type,
        "count": count,
        "repeats": repeats,
        "observed_delta_chi2": observed_best["delta_chi2_improvement"],
        "control_delta_mean": float(np.mean(best_deltas)),
        "control_delta_std": float(np.std(best_deltas)),
        "control_delta_90": float(np.percentile(best_deltas, 90)),
        "control_delta_95": float(np.percentile(best_deltas, 95)),
        "control_delta_99": float(np.percentile(best_deltas, 99)),
        "p_control_delta_ge_observed": float(
            (1.0 + np.sum(best_deltas >= observed_best["delta_chi2_improvement"]))
            / (1.0 + len(best_deltas))
        ),
    }

    return summary, rows


def code_context_search(aux_results):
    terms = [
        "fivelogH0",
        "H0",
        "alll",
        "ally",
        "allc",
        "theta",
        "samples",
        "parameter",
        "muhat",
        "intercept",
        "ceph",
        "cepheid",
        "anchor",
        "host",
        "calibrator",
        "pantheon",
        "hubble",
        "supernova",
        "sn",
    ]

    rows = []

    for item in aux_results:
        if item.get("status") != "downloaded":
            continue

        path = Path(item["local_path"])

        if not path.name.lower().endswith(
            (".py", ".txt", ".md", ".dat", ".out", ".tex", ".readme")
        ):
            continue

        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue

        preview_path = OUTDIR / f"preview_{safe_name(item['repo_path'])}.txt"
        preview_path.write_text("\n".join(lines[:180]), encoding="utf-8")

        for i, line in enumerate(lines, start=1):
            hits = [term for term in terms if term.lower() in line.lower()]

            if hits:
                lo = max(1, i - 2)
                hi = min(len(lines), i + 2)
                context = "\n".join(f"{j}: {lines[j - 1]}" for j in range(lo, hi + 1))

                rows.append(
                    {
                        "repo_path": item["repo_path"],
                        "line_number": i,
                        "hit_terms": " | ".join(hits),
                        "line": line[:700],
                        "context": context[:1600],
                    }
                )

    write_csv(OUTDIR / "equation_edge_likelihood_code_context_hits_v2_1.csv", rows)

    return rows


def make_plots(row_rows, bridge_mask, touch46_mask, baseline, model_rows, grid_rows):
    residual = np.asarray([r["baseline_residual"] for r in row_rows], dtype=float)

    plt.figure(figsize=(11, 5))
    plt.plot(residual, linewidth=0.8, label="baseline residual")

    if np.any(touch46_mask):
        plt.scatter(
            np.where(touch46_mask)[0],
            residual[touch46_mask],
            s=8,
            label="touches param46",
        )

    if np.any(bridge_mask):
        plt.scatter(
            np.where(bridge_mask)[0],
            residual[bridge_mask],
            s=10,
            label="42-46 bridge",
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("compact row index")
    plt.ylabel("baseline residual")
    plt.title("Equation-edge likelihood v2.1 residual map")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTDIR / "equation_edge_likelihood_residual_map_v2_1.png", dpi=160)
    plt.close()

    labels = [r["model_name"] for r in model_rows]
    values = [r["delta_chi2_improvement"] for r in model_rows]

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(labels)), values)
    plt.axhline(0, linewidth=1)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("best delta chi2 improvement")
    plt.title("Equation-edge deformation best improvements v2.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "equation_edge_deformation_delta_chi2_v2_1.png", dpi=160)
    plt.close()

    for model_name in sorted(set(r["model_name"] for r in grid_rows)):
        rows = sorted(
            [r for r in grid_rows if r["model_name"] == model_name],
            key=lambda r: r["lambda"],
        )

        plt.figure(figsize=(9, 5))
        plt.plot(
            [r["lambda"] for r in rows],
            [r["delta_chi2_improvement"] for r in rows],
            marker="o",
            linewidth=1.5,
        )
        plt.axhline(0, linewidth=1)
        plt.axvline(0, linewidth=1)
        plt.xlabel("lambda")
        plt.ylabel("delta chi2 improvement")
        plt.title(f"Grid profile: {model_name}")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"grid_profile_{safe_name(model_name)}_v2_1.png", dpi=160)
        plt.close()


def decide_status(model_rows, control_summaries):
    by_name = {r["model_name"]: r for r in model_rows}

    candidate_names = [
        "bridge_antisym_42_46",
        "bridge_param46_scale",
        "bridge_param42_scale",
        "bridge_common_42_46",
    ]

    candidates = [by_name[name] for name in candidate_names if name in by_name]

    best_candidate = (
        max(candidates, key=lambda r: r["delta_chi2_improvement"]) if candidates else None
    )
    best_overall = (
        max(model_rows, key=lambda r: r["delta_chi2_improvement"])
        if model_rows
        else None
    )

    controls_for_best = [
        c
        for c in control_summaries
        if best_candidate and c.get("model_name") == best_candidate["model_name"]
    ]

    random_p = next(
        (
            c.get("p_control_delta_ge_observed")
            for c in controls_for_best
            if c["control_type"] == "random_same_count"
        ),
        None,
    )
    block_p = next(
        (
            c.get("p_control_delta_ge_observed")
            for c in controls_for_best
            if c["control_type"] == "random_contiguous_block"
        ),
        None,
    )

    random_ok = True if random_p is None else random_p <= 0.05
    block_ok = True if block_p is None else block_p <= 0.10

    strong = bool(
        best_candidate
        and best_candidate["p_value_chi2_improvement_one_param"] <= 0.01
        and best_candidate["delta_bic_model_minus_baseline"] <= -2.0
        and not best_candidate["lambda_at_grid_boundary"]
        and random_ok
        and block_ok
    )

    directional = bool(
        best_candidate
        and best_candidate["p_value_chi2_improvement_one_param"] <= 0.05
        and best_candidate["delta_aic_model_minus_baseline"] < 0.0
        and not best_candidate["lambda_at_grid_boundary"]
    )

    best_cases = {
        "best_candidate": best_candidate,
        "best_overall": best_overall,
        "controls_for_best_candidate": controls_for_best,
    }

    if strong:
        return (
            "equation_edge_deformation_supported_provisional",
            8,
            "Recover stronger code/parameter labels, then rerun v3 with explicit SH0ES equation labels before theory promotion.",
            best_cases,
        )

    if directional:
        return (
            "equation_edge_deformation_directional_not_locked",
            7,
            "Treat as directional only. Inspect controls, lambda profiles, and H0-shift audit before continuing.",
            best_cases,
        )

    return (
        "equation_edge_deformation_not_supported",
        6,
        "The recovered 42-46 bridge is real matrix structure, but this constrained deformation does not add enough likelihood support.",
        best_cases,
    )


def main():
    print("")
    print("TAIRID SH0ES compact ladder equation-edge likelihood v2.1 starting.")
    print("Boundary: constrained equation-edge deformation screen only; not proof.")
    print("")

    downloads = {}
    ledger = []

    for label, repo_path in COMPACT.items():
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
                "pointer_declared_size": (result.get("pointer_info") or {}).get(
                    "declared_size"
                ),
                "pointer_oid_sha256": (result.get("pointer_info") or {}).get(
                    "oid_sha256"
                ),
                "attempt_count": len(result.get("attempts", [])),
            }
        )

    aux_results = []

    for repo_path in AUX:
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
                "pointer_declared_size": (result.get("pointer_info") or {}).get(
                    "declared_size"
                ),
                "pointer_oid_sha256": (result.get("pointer_info") or {}).get(
                    "oid_sha256"
                ),
                "attempt_count": len(result.get("attempts", [])),
            }
        )

    write_csv(OUTDIR / "equation_edge_likelihood_download_ledger_v2_1.csv", ledger)
    write_json(
        OUTDIR / "equation_edge_likelihood_download_attempts_v2_1.json",
        {"compact": downloads, "auxiliary": aux_results},
    )

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

    write_json(OUTDIR / "equation_edge_likelihood_parse_meta_v2_1.json", parse_meta)
    write_json(OUTDIR / "equation_edge_likelihood_parse_errors_v2_1.json", parse_errors)

    if parse_errors or not all(k in parsed for k in ["allc", "alll", "ally"]):
        summary = {
            "test_name": "TAIRID SH0ES compact ladder equation-edge likelihood v2.1",
            "boundary": "Matrix parse/download failure. No likelihood result.",
            "final_status": "equation_edge_likelihood_matrix_parse_failed",
            "readiness_score_0_to_10": 4,
            "next_wall": "Fix compact matrix retrieval/parsing before likelihood testing.",
            "parse_errors": parse_errors,
        }
        write_json(OUTDIR / "equation_edge_likelihood_v2_1_summary.json", summary)
        print("Parse failed. See summary JSON.")
        return

    C = np.asarray(parsed["allc"], float)
    L = np.asarray(parsed["alll"], float)
    y = np.asarray(parsed["ally"], float).reshape(-1)

    X, orientation = orient_design_matrix(L, len(y))

    if X is None:
        raise RuntimeError(f"Could not orient L: {orientation}")

    if C.ndim != 2 or C.shape[0] != len(y) or C.shape[1] != len(y):
        raise RuntimeError(f"C shape {C.shape} does not match y length {len(y)}")

    c_factor, C_sym, jitter, chol_attempts = stable_cholesky(C)
    baseline = base_gls_fit(y, X, c_factor)

    row_rows, bridge_mask, touch42_mask, touch46_mask = recover_edge_rows(
        X,
        y,
        baseline,
    )

    write_csv(OUTDIR / "equation_edge_likelihood_row_map_v2_1.csv", row_rows)

    candidate_models = [
        ("bridge_antisym_42_46", "bridge_antisym_42_46", bridge_mask),
        ("bridge_param46_scale", "bridge_param46_scale", bridge_mask),
        ("bridge_param42_scale", "bridge_param42_scale", bridge_mask),
        ("bridge_common_42_46", "bridge_common_42_46", bridge_mask),
        ("touch46_param46_scale", "param46_scale_on_mask", touch46_mask),
        ("touch42_param42_scale", "param42_scale_on_mask", touch42_mask),
        (
            "global_param46_scale_control",
            "global_param46_scale",
            np.ones(X.shape[0], dtype=bool),
        ),
    ]

    model_rows = []
    grid_rows = []

    for model_name, kind, mask in candidate_models:
        best, grid = fit_deformation_grid_fast(
            y,
            X,
            c_factor,
            baseline,
            mask,
            kind,
            model_name,
            LAMBDA_GRID,
        )
        model_rows.append(best)
        grid_rows.extend(grid)

    bridge_candidate_rows = [
        r for r in model_rows if r["model_name"].startswith("bridge_")
    ]
    candidate_for_controls = max(
        bridge_candidate_rows,
        key=lambda r: r["delta_chi2_improvement"],
    )

    control_summaries = []
    control_detail_rows = []

    for control_type in ["random_same_count", "random_contiguous_block"]:
        summary, rows = random_mask_controls(
            y,
            X,
            c_factor,
            baseline,
            candidate_for_controls,
            candidate_for_controls["deformation_kind"],
            int(np.sum(bridge_mask)),
            CONTROL_LAMBDA_GRID,
            RANDOM_CONTROL_REPEATS,
            control_type,
        )
        summary["model_name"] = candidate_for_controls["model_name"]
        control_summaries.append(summary)
        control_detail_rows.extend(rows)

    final_status, readiness_score, next_wall, best_cases = decide_status(
        model_rows,
        control_summaries,
    )

    write_csv(OUTDIR / "equation_edge_deformation_model_results_v2_1.csv", model_rows)
    write_csv(OUTDIR / "equation_edge_deformation_grid_results_v2_1.csv", grid_rows)
    write_csv(
        OUTDIR / "equation_edge_deformation_control_summaries_v2_1.csv",
        control_summaries,
    )
    write_csv(
        OUTDIR / "equation_edge_deformation_control_details_v2_1.csv",
        control_detail_rows,
    )

    make_plots(row_rows, bridge_mask, touch46_mask, baseline, model_rows, grid_rows)

    edge_counts = {
        "rows_total": int(X.shape[0]),
        "bridge_42_46_rows": int(np.sum(bridge_mask)),
        "touch_param42_rows": int(np.sum(touch42_mask)),
        "touch_param46_rows": int(np.sum(touch46_mask)),
    }

    summary = {
        "test_name": "TAIRID SH0ES compact ladder equation-edge likelihood v2.1",
        "boundary": (
            "Constrained equation-edge deformation screen only. "
            "Not proof of TAIRID, not H0 resolution, not BAO, not Planck, "
            "and not a full cosmology model."
        ),
        "final_status": final_status,
        "readiness_score_0_to_10": readiness_score,
        "next_wall": next_wall,
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
            "parameter_count": int(len(baseline["beta"])),
            "param42_value": float(baseline["beta"][P42]),
            "param42_err": float(baseline["beta_err"][P42]),
            "param46_fivelogH0": float(baseline["beta"][P46]),
            "param46_err": float(baseline["beta_err"][P46]),
            "param46_H0_like": h0_like_from_beta(baseline["beta"]),
            "normal_condition_estimate": float(np.linalg.cond(baseline["normal"])),
            "residual_mean": float(np.mean(baseline["residual"])),
            "residual_std": float(np.std(baseline["residual"])),
            "residual_rms": float(np.sqrt(np.mean(baseline["residual"] ** 2))),
        },
        "edge_counts": edge_counts,
        "lambda_grid": [float(v) for v in LAMBDA_GRID],
        "control_lambda_grid": [float(v) for v in CONTROL_LAMBDA_GRID],
        "model_results": model_rows,
        "control_summaries": control_summaries,
        "best_cases": best_cases,
        "code_context_hits_count": len(code_hits),
        "output_files": {
            "summary_json": str(OUTDIR / "equation_edge_likelihood_v2_1_summary.json"),
            "summary_txt": str(OUTDIR / "equation_edge_likelihood_v2_1_summary.txt"),
            "row_map_csv": str(OUTDIR / "equation_edge_likelihood_row_map_v2_1.csv"),
            "model_results_csv": str(
                OUTDIR / "equation_edge_deformation_model_results_v2_1.csv"
            ),
            "grid_results_csv": str(
                OUTDIR / "equation_edge_deformation_grid_results_v2_1.csv"
            ),
            "control_summaries_csv": str(
                OUTDIR / "equation_edge_deformation_control_summaries_v2_1.csv"
            ),
            "control_details_csv": str(
                OUTDIR / "equation_edge_deformation_control_details_v2_1.csv"
            ),
            "code_context_hits_csv": str(
                OUTDIR / "equation_edge_likelihood_code_context_hits_v2_1.csv"
            ),
            "plots": [
                str(OUTDIR / "equation_edge_likelihood_residual_map_v2_1.png"),
                str(OUTDIR / "equation_edge_deformation_delta_chi2_v2_1.png"),
            ],
        },
        "interpretation": {
            "what_supports_TAIRID_here": (
                "A 42<->46 edge deformation improves likelihood enough to overcome "
                "an effective one-parameter BIC penalty, is not at the grid boundary, "
                "and beats random/contiguous same-count controls."
            ),
            "what_weakens_TAIRID_here": (
                "The deformation does not improve likelihood, improves only at the grid edge, "
                "fails BIC/AIC, or performs no better than same-count random or contiguous controls."
            ),
            "truth_boundary": (
                "Even a positive result would only justify stronger row/parameter-label "
                "recovery and a v3 edge-likelihood test. It would not prove TAIRID "
                "or resolve H0."
            ),
        },
    }

    write_json(OUTDIR / "equation_edge_likelihood_v2_1_summary.json", summary)

    with open(
        OUTDIR / "equation_edge_likelihood_v2_1_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("TAIRID SH0ES compact ladder equation-edge likelihood v2.1\n\n")
        f.write("Boundary: constrained equation-edge deformation screen only.\n")
        f.write("Not proof of TAIRID. Not H0 resolution. Not BAO. Not Planck.\n\n")
        f.write(f"Final status: {final_status}\n")
        f.write(f"Readiness score: {readiness_score}/10\n")
        f.write(f"Next wall: {next_wall}\n\n")
        f.write("Edge counts:\n")
        f.write(json.dumps(edge_counts, indent=2, default=json_default) + "\n\n")
        f.write("Baseline GLS:\n")
        f.write(json.dumps(summary["baseline_gls"], indent=2, default=json_default) + "\n\n")
        f.write("Best cases:\n")
        f.write(json.dumps(best_cases, indent=2, default=json_default) + "\n\n")
        f.write("Model results:\n")
        f.write(json.dumps(model_rows, indent=2, default=json_default) + "\n\n")
        f.write("Control summaries:\n")
        f.write(json.dumps(control_summaries, indent=2, default=json_default) + "\n\n")
        f.write("Truth boundary:\n")
        f.write("- This does not prove TAIRID.\n")
        f.write("- This does not prove H0 resolution.\n")
        f.write("- This only tests constrained deformation of the recovered 42-46 equation edge.\n")

    print("")
    print("TAIRID SH0ES compact ladder equation-edge likelihood v2.1 complete.")
    print("Created:")
    print("  tairid_shoes_equation_edge_likelihood_v2_1_outputs/equation_edge_likelihood_v2_1_summary.json")
    print("  tairid_shoes_equation_edge_likelihood_v2_1_outputs/equation_edge_likelihood_v2_1_summary.txt")
    print("  tairid_shoes_equation_edge_likelihood_v2_1_outputs/equation_edge_deformation_model_results_v2_1.csv")
    print("  tairid_shoes_equation_edge_likelihood_v2_1_outputs/equation_edge_deformation_grid_results_v2_1.csv")
    print("")
    print("Boundary:")
    print("  This is not proof of TAIRID.")
    print("  This is an equation-edge deformation screen only.")
    print("")
    print(f"Final status: {final_status}")
    print(f"Readiness score: {readiness_score}/10")


if __name__ == "__main__":
    main()
