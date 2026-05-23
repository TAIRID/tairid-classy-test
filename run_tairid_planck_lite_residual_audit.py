#!/usr/bin/env python3
"""
TAIRID Planck-lite CMB residual audit.

Purpose:
The Planck-ready packet showed that TT and EE remain close to CDM, but TE ratio
metrics can look misleading because TE crosses near zero.

This script fixes that by using normalized residual metrics instead of
point-by-point ratios.

For TT, TE, EE, and CMB lensing, it computes:

    normalized_rms_residual = RMS(candidate - CDM) / RMS(CDM)

    normalized_mean_abs_residual = mean(abs(candidate - CDM)) / RMS(CDM)

    normalized_max_abs_residual = max(abs(candidate - CDM)) / RMS(CDM)

It also computes a correlation-style shape agreement score.

Boundary:
This is not a Planck likelihood.
This does not use Planck covariance.
This does not include official Planck nuisance parameters, foregrounds, beams, or clik likelihoods.
This is a Planck-lite residual sanity audit only.
"""

import csv
import json
import math
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from classy import Class


OUTDIR = Path("planck_lite_residual_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

OMEGA_NEUTRAL_PHYSICAL = 0.1200000000
H = 0.66893180

ELL_MAX = 2500
ELL_SHAPE_MIN = 30
ELL_SHAPE_MAX = 2000
ELL_LENS_MIN = 40
ELL_LENS_MAX = 1000

PEAK_WINDOWS = [
    (100, 350),
    (350, 650),
    (650, 1000),
]

BASE = {
    "output": "tCl,pCl,lCl,mPk",
    "lensing": "yes",
    "h": H,
    "omega_b": 0.0223700000,
    "omega_cdm": OMEGA_NEUTRAL_PHYSICAL,
    "N_ur": 3.046,
    "Omega_k": 0.0,
    "Omega_Lambda": 0.6817397872,
    "n_s": 0.9649,
    "A_s": 2.100549e-9,
    "tau_reio": 0.0544,
    "YHe": 0.245,
    "T_cmb": 2.7255,
    "l_max_scalars": ELL_MAX,
    "P_k_max_1/Mpc": 50.0,
    "z_max_pk": 3.0,
    "z_pk": "0",
}

CASES = [
    {
        "name": "cdm_baseline",
        "label": "CDM baseline",
        "warm_fraction": 0.0,
        "m_ncdm_eV": None,
    },
    {
        "name": "strict_anchor_1p5pct_12p5eV",
        "label": "Strict safety anchor",
        "warm_fraction": 0.015,
        "m_ncdm_eV": 12.5,
    },
    {
        "name": "best_score_1p75pct_15eV",
        "label": "Best matrix score anchor",
        "warm_fraction": 0.0175,
        "m_ncdm_eV": 15.0,
    },
    {
        "name": "stronger_s8_2pct_15eV",
        "label": "Stronger S8 but lensing warning",
        "warm_fraction": 0.020,
        "m_ncdm_eV": 15.0,
    },
    {
        "name": "warning_old_5pct_20eV",
        "label": "Old S8-helpful lensing-warning case",
        "warm_fraction": 0.050,
        "m_ncdm_eV": 20.0,
    },
]


def make_params(case):
    params = dict(BASE)

    f = float(case["warm_fraction"])
    if f <= 0:
        return params

    omega_warm = OMEGA_NEUTRAL_PHYSICAL * f
    omega_cold = OMEGA_NEUTRAL_PHYSICAL * (1.0 - f)

    params["omega_cdm"] = omega_cold
    params["N_ncdm"] = 1
    params["m_ncdm"] = float(case["m_ncdm_eV"])
    params["Omega_ncdm"] = omega_warm / (H * H)

    return params


def omega_total_matter_proxy(params):
    h = float(params["h"])
    h2 = h * h

    omega_b = float(params.get("omega_b", 0.0))
    omega_cdm = float(params.get("omega_cdm", 0.0))
    omega_ncdm = float(params.get("Omega_ncdm", 0.0)) * h2

    omega_m = omega_b + omega_cdm + omega_ncdm
    Omega_m = omega_m / h2

    return omega_m, Omega_m


def s8_from_sigma8(sigma8, Omega_m):
    return float(sigma8 * math.sqrt(Omega_m / 0.3))


def peak_by_windows(ell, dl_tt):
    peak_ells = []
    peak_heights = []

    for lo, hi in PEAK_WINDOWS:
        mask = (ell >= lo) & (ell <= hi)

        if not np.any(mask):
            peak_ells.append(None)
            peak_heights.append(None)
            continue

        ell_window = ell[mask]
        dl_window = dl_tt[mask]
        idx = int(np.argmax(dl_window))

        peak_ells.append(int(ell_window[idx]))
        peak_heights.append(float(dl_window[idx]))

    return peak_ells, peak_heights


def get_lensing_pp(cosmo, lmax):
    try:
        cl = cosmo.lensed_cl(lmax)
        if "pp" in cl:
            return np.asarray(cl["ell"]), np.asarray(cl["pp"])
    except Exception:
        pass

    try:
        cl = cosmo.raw_cl(lmax)
        if "pp" in cl:
            return np.asarray(cl["ell"]), np.asarray(cl["pp"])
    except Exception:
        pass

    raise RuntimeError("Could not find pp lensing spectrum in CLASS output.")


def residual_metrics(base_x, base_y, case_x, case_y, lo, hi):
    if not np.array_equal(base_x, case_x):
        case_interp = np.interp(base_x, case_x, case_y)
    else:
        case_interp = case_y

    mask = (
        (base_x >= lo)
        & (base_x <= hi)
        & np.isfinite(base_y)
        & np.isfinite(case_interp)
    )

    if np.sum(mask) < 2:
        return {
            "base_rms": float("nan"),
            "normalized_rms_residual": float("nan"),
            "normalized_mean_abs_residual": float("nan"),
            "normalized_max_abs_residual": float("nan"),
            "shape_correlation": float("nan"),
        }

    b = base_y[mask]
    c = case_interp[mask]
    d = c - b

    base_rms = float(np.sqrt(np.nanmean(b * b)))

    if base_rms <= 0 or not np.isfinite(base_rms):
        norm_rms = float("nan")
        norm_mean_abs = float("nan")
        norm_max_abs = float("nan")
    else:
        norm_rms = float(np.sqrt(np.nanmean(d * d)) / base_rms)
        norm_mean_abs = float(np.nanmean(np.abs(d)) / base_rms)
        norm_max_abs = float(np.nanmax(np.abs(d)) / base_rms)

    b_centered = b - np.nanmean(b)
    c_centered = c - np.nanmean(c)

    denom = np.sqrt(np.nansum(b_centered * b_centered) * np.nansum(c_centered * c_centered))

    if denom <= 0 or not np.isfinite(denom):
        corr = float("nan")
    else:
        corr = float(np.nansum(b_centered * c_centered) / denom)

    return {
        "base_rms": base_rms,
        "normalized_rms_residual": norm_rms,
        "normalized_mean_abs_residual": norm_mean_abs,
        "normalized_max_abs_residual": norm_max_abs,
        "shape_correlation": corr,
    }


def write_curve(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def run_case(case):
    name = case["name"]
    params = make_params(case)
    omega_m, Omega_m = omega_total_matter_proxy(params)

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    sigma8 = float(cosmo.sigma8())
    S8 = s8_from_sigma8(sigma8, Omega_m)

    cl = cosmo.lensed_cl(ELL_MAX)

    ell = np.asarray(cl["ell"])
    tt = np.asarray(cl["tt"])
    te = np.asarray(cl["te"])
    ee = np.asarray(cl["ee"])

    dl_tt = ell * (ell + 1) * tt / (2.0 * math.pi)
    dl_te = ell * (ell + 1) * te / (2.0 * math.pi)
    dl_ee = ell * (ell + 1) * ee / (2.0 * math.pi)

    peak_ell, peak_height = peak_by_windows(ell, dl_tt)

    ell_pp, pp = get_lensing_pp(cosmo, ELL_MAX)

    lensing_scaled = np.zeros_like(pp, dtype=float)
    good = ell_pp > 0
    lensing_scaled[good] = (ell_pp[good] ** 4) * pp[good] / (2.0 * math.pi)

    case_dir = OUTDIR / name
    case_dir.mkdir(parents=True, exist_ok=True)

    write_curve(
        case_dir / "TT_TE_EE_Dell.csv",
        ["ell", "D_ell_TT", "D_ell_TE", "D_ell_EE"],
        [[int(e), float(a), float(b), float(c)] for e, a, b, c in zip(ell, dl_tt, dl_te, dl_ee)],
    )

    write_curve(
        case_dir / "lensing_scaled.csv",
        ["ell", "C_ell_pp", "ell4_C_ell_pp_over_2pi"],
        [[int(e), float(p), float(ls)] for e, p, ls in zip(ell_pp, pp, lensing_scaled)],
    )

    summary = {
        "name": name,
        "label": case["label"],
        "status": "success",
        "warm_fraction": float(case["warm_fraction"]),
        "m_ncdm_eV": None if case["m_ncdm_eV"] is None else float(case["m_ncdm_eV"]),
        "Omega_m_total": float(Omega_m),
        "sigma8": sigma8,
        "S8": S8,
        "peak_ell": peak_ell,
        "peak_height": peak_height,
        "boundary": "Planck-lite residual audit only. Not a Planck likelihood.",
    }

    (case_dir / "case_summary.json").write_text(json.dumps(summary, indent=2))

    cosmo.struct_cleanup()
    cosmo.empty()

    return {
        "summary": summary,
        "ell": ell,
        "dl_tt": dl_tt,
        "dl_te": dl_te,
        "dl_ee": dl_ee,
        "ell_pp": ell_pp,
        "lensing_scaled": lensing_scaled,
    }


results = {}
errors = {}

for case in CASES:
    name = case["name"]

    print("")
    print("Running Planck-lite residual case:", name)

    try:
        results[name] = run_case(case)
        print("  success")
        print("  S8:", results[name]["summary"]["S8"])
        print("  peaks:", results[name]["summary"]["peak_ell"])

    except Exception as exc:
        errors[name] = {
            "name": name,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print("  FAILED:", exc)


baseline_name = "cdm_baseline"

if baseline_name not in results:
    raise RuntimeError("CDM baseline failed. Cannot compute residual audit.")

base = results[baseline_name]

summary_rows = []

for name, data in results.items():
    s = data["summary"]

    tt = residual_metrics(base["ell"], base["dl_tt"], data["ell"], data["dl_tt"], ELL_SHAPE_MIN, ELL_SHAPE_MAX)
    te = residual_metrics(base["ell"], base["dl_te"], data["ell"], data["dl_te"], ELL_SHAPE_MIN, ELL_SHAPE_MAX)
    ee = residual_metrics(base["ell"], base["dl_ee"], data["ell"], data["dl_ee"], ELL_SHAPE_MIN, ELL_SHAPE_MAX)
    lens = residual_metrics(base["ell_pp"], base["lensing_scaled"], data["ell_pp"], data["lensing_scaled"], ELL_LENS_MIN, ELL_LENS_MAX)

    if name == baseline_name:
        diagnostic = "baseline"
    elif (
        tt["normalized_rms_residual"] <= 0.003
        and te["normalized_rms_residual"] <= 0.010
        and ee["normalized_rms_residual"] <= 0.005
        and lens["normalized_rms_residual"] <= 0.035
        and s["S8"] <= 0.833
    ):
        diagnostic = "planck_lite_close"
    elif (
        tt["normalized_rms_residual"] <= 0.005
        and te["normalized_rms_residual"] <= 0.020
        and ee["normalized_rms_residual"] <= 0.008
        and lens["normalized_rms_residual"] <= 0.050
        and s["S8"] <= 0.835
    ):
        diagnostic = "planck_lite_warning_but_survives"
    else:
        diagnostic = "planck_lite_pressure"

    row = {
        "case": name,
        "label": s["label"],
        "diagnostic": diagnostic,
        "warm_fraction": s["warm_fraction"],
        "m_ncdm_eV": s["m_ncdm_eV"],
        "S8": s["S8"],
        "ell_peak1": s["peak_ell"][0],
        "ell_peak2": s["peak_ell"][1],
        "ell_peak3": s["peak_ell"][2],
        "TT_norm_rms_residual": tt["normalized_rms_residual"],
        "TT_norm_max_abs_residual": tt["normalized_max_abs_residual"],
        "TT_shape_correlation": tt["shape_correlation"],
        "TE_norm_rms_residual": te["normalized_rms_residual"],
        "TE_norm_max_abs_residual": te["normalized_max_abs_residual"],
        "TE_shape_correlation": te["shape_correlation"],
        "EE_norm_rms_residual": ee["normalized_rms_residual"],
        "EE_norm_max_abs_residual": ee["normalized_max_abs_residual"],
        "EE_shape_correlation": ee["shape_correlation"],
        "lensing_norm_rms_residual": lens["normalized_rms_residual"],
        "lensing_norm_max_abs_residual": lens["normalized_max_abs_residual"],
        "lensing_shape_correlation": lens["shape_correlation"],
    }

    summary_rows.append(row)


with open(OUTDIR / "planck_lite_residual_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)


packet_summary = {
    "boundary": "Planck-lite residual audit only. Not a Planck likelihood.",
    "important_note": "TE is measured by normalized residuals, not ratios, because TE crosses near zero.",
    "rows": summary_rows,
    "errors": errors,
    "next_step": "If TE residuals are stable, attempt a real Planck/Cobaya likelihood bridge.",
}

(OUTDIR / "planck_lite_residual_summary.json").write_text(json.dumps(packet_summary, indent=2))


plt.figure(figsize=(10, 6))
for name, data in results.items():
    plt.plot(data["ell"], data["dl_te"], label=name, linewidth=1.0)
plt.axhline(0.0, linewidth=1)
plt.xlabel("ell")
plt.ylabel("D_ell TE")
plt.title("Planck-lite audit: TE spectra")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUTDIR / "planck_lite_TE_spectra.png", dpi=160)
plt.close()


plt.figure(figsize=(10, 6))
for name, data in results.items():
    if name == baseline_name:
        continue

    diff = data["dl_te"] - base["dl_te"]
    plt.plot(data["ell"], diff, label=name, linewidth=1.0)

plt.axhline(0.0, linewidth=1)
plt.xlabel("ell")
plt.ylabel("D_ell TE residual versus CDM")
plt.title("Planck-lite audit: TE residuals")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUTDIR / "planck_lite_TE_residuals.png", dpi=160)
plt.close()


plt.figure(figsize=(10, 6))
for name, data in results.items():
    if name == baseline_name:
        continue

    diff = data["dl_tt"] - base["dl_tt"]
    plt.plot(data["ell"], diff, label=name, linewidth=1.0)

plt.axhline(0.0, linewidth=1)
plt.xlabel("ell")
plt.ylabel("D_ell TT residual versus CDM")
plt.title("Planck-lite audit: TT residuals")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUTDIR / "planck_lite_TT_residuals.png", dpi=160)
plt.close()


plt.figure(figsize=(10, 6))
for name, data in results.items():
    if name == baseline_name:
        continue

    diff = data["dl_ee"] - base["dl_ee"]
    plt.plot(data["ell"], diff, label=name, linewidth=1.0)

plt.axhline(0.0, linewidth=1)
plt.xlabel("ell")
plt.ylabel("D_ell EE residual versus CDM")
plt.title("Planck-lite audit: EE residuals")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUTDIR / "planck_lite_EE_residuals.png", dpi=160)
plt.close()


plt.figure(figsize=(10, 6))
for name, data in results.items():
    if name == baseline_name:
        continue

    if not np.array_equal(base["ell_pp"], data["ell_pp"]):
        curve = np.interp(base["ell_pp"], data["ell_pp"], data["lensing_scaled"])
    else:
        curve = data["lensing_scaled"]

    diff = curve - base["lensing_scaled"]
    mask = (base["ell_pp"] >= ELL_LENS_MIN) & (base["ell_pp"] <= ELL_LENS_MAX)

    plt.plot(base["ell_pp"][mask], diff[mask], label=name, linewidth=1.0)

plt.axhline(0.0, linewidth=1)
plt.xlabel("ell")
plt.ylabel("lensing scaled residual versus CDM")
plt.title("Planck-lite audit: CMB lensing residuals")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUTDIR / "planck_lite_lensing_residuals.png", dpi=160)
plt.close()


print("")
print("TAIRID Planck-lite residual audit complete.")
print("Created folder:")
print("  planck_lite_residual_outputs/")
print("")
print("Created key files:")
print("  planck_lite_residual_outputs/planck_lite_residual_summary.csv")
print("  planck_lite_residual_outputs/planck_lite_residual_summary.json")
print("  planck_lite_residual_outputs/planck_lite_TE_spectra.png")
print("  planck_lite_residual_outputs/planck_lite_TE_residuals.png")
print("")
print("Boundary:")
print("  This is not a Planck likelihood.")
print("  TE is measured with normalized residuals instead of bad ratio math.")
