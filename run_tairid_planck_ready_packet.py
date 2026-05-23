#!/usr/bin/env python3
"""
TAIRID Planck-ready CMB packet.

Purpose:
Generate a clean CMB output packet for the current TAIRID proxy corridor.

This is the bridge step before attempting any full Planck likelihood.

It outputs:
- TT spectrum
- TE spectrum
- EE spectrum
- CMB lensing pp spectrum
- candidate summary
- baseline-relative drift metrics

Boundary:
This is not a Planck likelihood.
This does not use Planck covariance.
This does not include nuisance parameters, foregrounds, beams, or official clik likelihoods.
This is a Planck-readiness packet only.
"""

import csv
import json
import math
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from classy import Class


OUTDIR = Path("planck_ready_packet_outputs")
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


def drift_metrics(base_x, base_y, case_x, case_y, lo, hi):
    if not np.array_equal(base_x, case_x):
        case_interp = np.interp(base_x, case_x, case_y)
    else:
        case_interp = case_y

    mask = (
        (base_x >= lo)
        & (base_x <= hi)
        & np.isfinite(base_y)
        & np.isfinite(case_interp)
        & (base_y != 0)
    )

    if np.sum(mask) < 2:
        return {
            "mean_ratio": float("nan"),
            "rms_drift": float("nan"),
            "max_abs_drift": float("nan"),
            "mean_abs_drift": float("nan"),
        }

    ratio = case_interp[mask] / base_y[mask]
    drift = ratio - 1.0

    return {
        "mean_ratio": float(np.nanmean(ratio)),
        "rms_drift": float(np.sqrt(np.nanmean(drift * drift))),
        "max_abs_drift": float(np.nanmax(np.abs(drift))),
        "mean_abs_drift": float(np.nanmean(np.abs(drift))),
    }


def write_spectrum_csv(path, ell, values, header_name):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ell", header_name])

        for e, v in zip(ell, values):
            writer.writerow([int(e), float(v)])


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

    write_spectrum_csv(case_dir / "TT_Dell.csv", ell, dl_tt, "D_ell_TT")
    write_spectrum_csv(case_dir / "TE_Dell.csv", ell, dl_te, "D_ell_TE")
    write_spectrum_csv(case_dir / "EE_Dell.csv", ell, dl_ee, "D_ell_EE")
    write_spectrum_csv(case_dir / "lensing_pp.csv", ell_pp, pp, "C_ell_pp")
    write_spectrum_csv(case_dir / "lensing_scaled.csv", ell_pp, lensing_scaled, "ell4_C_ell_pp_over_2pi")

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
        "files": {
            "TT": str(case_dir / "TT_Dell.csv"),
            "TE": str(case_dir / "TE_Dell.csv"),
            "EE": str(case_dir / "EE_Dell.csv"),
            "lensing_pp": str(case_dir / "lensing_pp.csv"),
            "lensing_scaled": str(case_dir / "lensing_scaled.csv"),
        },
        "boundary": "Planck-ready packet only. Not a Planck likelihood.",
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
        "pp": pp,
        "lensing_scaled": lensing_scaled,
    }


results = {}
errors = {}

for case in CASES:
    name = case["name"]

    print("")
    print("Running Planck-ready packet case:", name)

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
    raise RuntimeError("CDM baseline failed. Cannot compute baseline-relative packet.")

base = results[baseline_name]

summary_rows = []
index_rows = []

for name, data in results.items():
    s = data["summary"]

    if name == baseline_name:
        tt_metrics = {"rms_drift": 0.0, "max_abs_drift": 0.0, "mean_abs_drift": 0.0}
        te_metrics = {"rms_drift": 0.0, "max_abs_drift": 0.0, "mean_abs_drift": 0.0}
        ee_metrics = {"rms_drift": 0.0, "max_abs_drift": 0.0, "mean_abs_drift": 0.0}
        lens_metrics = {"mean_ratio": 1.0, "rms_drift": 0.0, "max_abs_drift": 0.0, "mean_abs_drift": 0.0}
    else:
        tt_metrics = drift_metrics(
            base["ell"],
            base["dl_tt"],
            data["ell"],
            data["dl_tt"],
            ELL_SHAPE_MIN,
            ELL_SHAPE_MAX,
        )

        te_metrics = drift_metrics(
            base["ell"],
            base["dl_te"],
            data["ell"],
            data["dl_te"],
            ELL_SHAPE_MIN,
            ELL_SHAPE_MAX,
        )

        ee_metrics = drift_metrics(
            base["ell"],
            base["dl_ee"],
            data["ell"],
            data["dl_ee"],
            ELL_SHAPE_MIN,
            ELL_SHAPE_MAX,
        )

        lens_metrics = drift_metrics(
            base["ell_pp"],
            base["lensing_scaled"],
            data["ell_pp"],
            data["lensing_scaled"],
            ELL_LENS_MIN,
            ELL_LENS_MAX,
        )

    row = {
        "case": name,
        "label": s["label"],
        "status": s["status"],
        "warm_fraction": s["warm_fraction"],
        "m_ncdm_eV": s["m_ncdm_eV"],
        "Omega_m_total": s["Omega_m_total"],
        "sigma8": s["sigma8"],
        "S8": s["S8"],
        "ell_peak1": s["peak_ell"][0],
        "ell_peak2": s["peak_ell"][1],
        "ell_peak3": s["peak_ell"][2],
        "TT_rms_drift": tt_metrics["rms_drift"],
        "TT_max_abs_drift": tt_metrics["max_abs_drift"],
        "TE_rms_drift": te_metrics["rms_drift"],
        "TE_max_abs_drift": te_metrics["max_abs_drift"],
        "EE_rms_drift": ee_metrics["rms_drift"],
        "EE_max_abs_drift": ee_metrics["max_abs_drift"],
        "lensing_mean_ratio": lens_metrics["mean_ratio"],
        "lensing_rms_drift": lens_metrics["rms_drift"],
        "lensing_max_abs_drift": lens_metrics["max_abs_drift"],
    }

    summary_rows.append(row)

    index_rows.append(
        {
            "case": name,
            "label": s["label"],
            "TT_file": s["files"]["TT"],
            "TE_file": s["files"]["TE"],
            "EE_file": s["files"]["EE"],
            "lensing_pp_file": s["files"]["lensing_pp"],
            "lensing_scaled_file": s["files"]["lensing_scaled"],
            "case_summary_file": str(OUTDIR / name / "case_summary.json"),
        }
    )


with open(OUTDIR / "planck_ready_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)


with open(OUTDIR / "planck_ready_index.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
    writer.writeheader()
    writer.writerows(index_rows)


packet_summary = {
    "boundary": "Planck-ready packet only. Not a Planck likelihood.",
    "cases": summary_rows,
    "errors": errors,
    "next_step": "Use this packet to connect to a real Planck/Cobaya/clik likelihood workflow.",
}

(OUTDIR / "planck_ready_packet_summary.json").write_text(json.dumps(packet_summary, indent=2))


plt.figure(figsize=(10, 6))
for name, data in results.items():
    plt.plot(data["ell"], data["dl_tt"], label=name, linewidth=1.1)
plt.xlabel("ell")
plt.ylabel("D_ell TT")
plt.title("Planck-ready packet: TT spectra")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUTDIR / "planck_ready_TT_spectra.png", dpi=160)
plt.close()


plt.figure(figsize=(10, 6))
for name, data in results.items():
    plt.plot(data["ell"], data["dl_ee"], label=name, linewidth=1.1)
plt.xlabel("ell")
plt.ylabel("D_ell EE")
plt.title("Planck-ready packet: EE spectra")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUTDIR / "planck_ready_EE_spectra.png", dpi=160)
plt.close()


plt.figure(figsize=(10, 6))
base_lens = base["lensing_scaled"]

for name, data in results.items():
    if name == baseline_name:
        continue

    if not np.array_equal(base["ell_pp"], data["ell_pp"]):
        curve = np.interp(base["ell_pp"], data["ell_pp"], data["lensing_scaled"])
    else:
        curve = data["lensing_scaled"]

    ratio = curve / base_lens
    mask = (base["ell_pp"] >= ELL_LENS_MIN) & (base["ell_pp"] <= ELL_LENS_MAX)
    plt.plot(base["ell_pp"][mask], ratio[mask], label=name, linewidth=1.1)

plt.axhline(1.0, linewidth=1)
plt.xlabel("ell")
plt.ylabel("lensing ratio to CDM")
plt.title("Planck-ready packet: CMB lensing ratios")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUTDIR / "planck_ready_lensing_ratios.png", dpi=160)
plt.close()


print("")
print("TAIRID Planck-ready packet complete.")
print("Created folder:")
print("  planck_ready_packet_outputs/")
print("")
print("Created key files:")
print("  planck_ready_packet_outputs/planck_ready_summary.csv")
print("  planck_ready_packet_outputs/planck_ready_index.csv")
print("  planck_ready_packet_outputs/planck_ready_packet_summary.json")
print("")
print("Boundary:")
print("  This is not a Planck likelihood.")
print("  This is the packet needed before attempting a real Planck likelihood bridge.")
