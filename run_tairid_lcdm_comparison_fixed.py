#!/usr/bin/env python3
"""
Compare TAIRID neutral-substrate CLASS proxies against a standard LCDM-like baseline.

Boundary:
This is still a CLASS proxy test, not final TAIRID.
The TAIRID neutral substrate is treated as CDM-like at early times.

Outputs:
- lcdm_baseline_cl_tt.txt
- tairid_h0698_cl_tt.txt
- tairid_physical_match_cl_tt.txt
- peak_comparison.csv
- peak_comparison.json
- cmb_tt_comparison.png
- cmb_tt_difference.png
- cmb_tt_ratio.png
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from classy import Class


COSMOLOGIES = {
    "LCDM_baseline": {
        "output": "tCl,pCl,lCl",
        "lensing": "yes",
        "h": 0.6736,
        "omega_b": 0.02237,
        "omega_cdm": 0.1200000000,
        "N_ur": 3.046,
        "Omega_k": 0.0,
        "Omega_Lambda": 0.6861,
        "n_s": 0.9649,
        "A_s": 2.100549e-9,
        "tau_reio": 0.0544,
        "YHe": 0.245,
        "T_cmb": 2.7255,
        "l_max_scalars": 2500,
    },
    "TAIRID_h0698": {
        "output": "tCl,pCl,lCl",
        "lensing": "yes",
        "h": 0.69800000,
        "omega_b": 0.0223700000,
        "omega_cdm": 0.1306557128,
        "N_ur": 3.046,
        "Omega_k": 0.0,
        "Omega_Lambda": 0.6858245566,
        "n_s": 0.9649,
        "A_s": 2.100549e-9,
        "tau_reio": 0.0544,
        "YHe": 0.245,
        "T_cmb": 2.7255,
        "l_max_scalars": 2500,
    },
    "TAIRID_physical_match": {
        "output": "tCl,pCl,lCl",
        "lensing": "yes",
        "h": 0.66893180,
        "omega_b": 0.0223700000,
        "omega_cdm": 0.1200000000,
        "N_ur": 3.046,
        "Omega_k": 0.0,
        "Omega_Lambda": 0.6817397872,
        "n_s": 0.9649,
        "A_s": 2.100549e-9,
        "tau_reio": 0.0544,
        "YHe": 0.245,
        "T_cmb": 2.7255,
        "l_max_scalars": 2500,
    },
}


PEAK_WINDOWS = [
    (100, 350),
    (350, 650),
    (650, 1000),
]


def run_class(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    cl = cosmo.lensed_cl(2500)
    ell = np.asarray(cl["ell"])
    tt = np.asarray(cl["tt"])

    cosmo.struct_cleanup()
    cosmo.empty()

    return ell, tt


def dl_from_cl(ell, cl):
    return ell * (ell + 1) * cl / (2.0 * np.pi)


def peak_by_windows(ell, dl):
    peak_ells = []
    peak_heights = []

    for lo, hi in PEAK_WINDOWS:
        mask = (ell >= lo) & (ell <= hi)
        if not np.any(mask):
            peak_ells.append(None)
            peak_heights.append(None)
            continue

        ell_window = ell[mask]
        dl_window = dl[mask]
        idx = int(np.argmax(dl_window))

        peak_ells.append(int(ell_window[idx]))
        peak_heights.append(float(dl_window[idx]))

    return peak_ells, peak_heights


def safe_ratio(a, b):
    if b is None or b == 0:
        return None
    return float(a / b)


results = {}
spectra = {}

for name, params in COSMOLOGIES.items():
    print(f"\nRunning CLASS case: {name}")
    ell, tt = run_class(params)
    dl = dl_from_cl(ell, tt)

    peaks_ell, peaks_height = peak_by_windows(ell, dl)

    ratio_21 = safe_ratio(peaks_height[1], peaks_height[0])
    ratio_31 = safe_ratio(peaks_height[2], peaks_height[0])

    out_txt = f"{name.lower()}_cl_tt.txt"
    np.savetxt(
        out_txt,
        np.column_stack([ell, tt, dl]),
        header="ell C_l_TT_raw D_l_TT_raw"
    )

    spectra[name] = {
        "ell": ell,
        "tt": tt,
        "dl": dl,
    }

    results[name] = {
        "output_file": out_txt,
        "peak_ell": peaks_ell,
        "peak_Dl_raw": peaks_height,
        "ratio_peak2_to_peak1": ratio_21,
        "ratio_peak3_to_peak1": ratio_31,
        "params": params,
    }

    print(f"Saved {out_txt}")
    print(f"Peaks ell: {peaks_ell}")
    print(f"Peak ratios P2/P1={ratio_21}, P3/P1={ratio_31}")


# Compare deltas relative to LCDM
lcdm = results["LCDM_baseline"]

for name, data in results.items():
    if name == "LCDM_baseline":
        data["delta_vs_LCDM"] = {
            "peak_ell_delta": [0, 0, 0],
            "peak_height_fractional_delta": [0.0, 0.0, 0.0],
            "ratio_21_delta": 0.0,
            "ratio_31_delta": 0.0,
        }
        continue

    ell_delta = []
    height_frac_delta = []

    for i in range(3):
        model_ell = data["peak_ell"][i]
        base_ell = lcdm["peak_ell"][i]
        model_h = data["peak_Dl_raw"][i]
        base_h = lcdm["peak_Dl_raw"][i]

        ell_delta.append(None if model_ell is None or base_ell is None else int(model_ell - base_ell))
        height_frac_delta.append(None if model_h is None or base_h in (None, 0) else float((model_h - base_h) / base_h))

    data["delta_vs_LCDM"] = {
        "peak_ell_delta": ell_delta,
        "peak_height_fractional_delta": height_frac_delta,
        "ratio_21_delta": None if data["ratio_peak2_to_peak1"] is None else float(data["ratio_peak2_to_peak1"] - lcdm["ratio_peak2_to_peak1"]),
        "ratio_31_delta": None if data["ratio_peak3_to_peak1"] is None else float(data["ratio_peak3_to_peak1"] - lcdm["ratio_peak3_to_peak1"]),
    }


# Save JSON
Path("peak_comparison.json").write_text(json.dumps(results, indent=2))


# Save CSV
with open("peak_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Model",
        "ell_peak1", "ell_peak2", "ell_peak3",
        "Dl_peak1_raw", "Dl_peak2_raw", "Dl_peak3_raw",
        "ratio21", "ratio31",
        "delta_ell1_vs_LCDM", "delta_ell2_vs_LCDM", "delta_ell3_vs_LCDM",
        "frac_delta_height1_vs_LCDM", "frac_delta_height2_vs_LCDM", "frac_delta_height3_vs_LCDM",
        "delta_ratio21_vs_LCDM", "delta_ratio31_vs_LCDM",
    ])

    for name, data in results.items():
        d = data["delta_vs_LCDM"]
        writer.writerow([
            name,
            *data["peak_ell"],
            *data["peak_Dl_raw"],
            data["ratio_peak2_to_peak1"],
            data["ratio_peak3_to_peak1"],
            *d["peak_ell_delta"],
            *d["peak_height_fractional_delta"],
            d["ratio_21_delta"],
            d["ratio_31_delta"],
        ])


# Plots
plt.figure(figsize=(12, 6))
for name, spec in spectra.items():
    plt.plot(spec["ell"], spec["dl"], label=name, linewidth=1.4)
plt.xlabel("multipole ell")
plt.ylabel("raw D_ell^TT")
plt.title("CLASS TT proxy spectra: TAIRID vs LCDM")
plt.xlim(0, 2000)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("cmb_tt_comparison.png", dpi=160)
plt.close()


lcdm_ell = spectra["LCDM_baseline"]["ell"]
lcdm_dl = spectra["LCDM_baseline"]["dl"]

plt.figure(figsize=(12, 4.5))
for name, spec in spectra.items():
    if name == "LCDM_baseline":
        continue
    model_dl_interp = np.interp(lcdm_ell, spec["ell"], spec["dl"])
    plt.plot(lcdm_ell, model_dl_interp - lcdm_dl, label=f"{name} - LCDM", linewidth=1.2)
plt.xlabel("multipole ell")
plt.ylabel("raw Delta D_ell^TT")
plt.title("Difference from LCDM baseline")
plt.xlim(0, 2000)
plt.axhline(0, linewidth=1)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("cmb_tt_difference.png", dpi=160)
plt.close()


plt.figure(figsize=(12, 4.5))
for name, spec in spectra.items():
    if name == "LCDM_baseline":
        continue
    model_dl_interp = np.interp(lcdm_ell, spec["ell"], spec["dl"])
    ratio = np.divide(model_dl_interp, lcdm_dl, out=np.full_like(model_dl_interp, np.nan), where=lcdm_dl != 0)
    plt.plot(lcdm_ell, ratio, label=f"{name} / LCDM", linewidth=1.2)
plt.xlabel("multipole ell")
plt.ylabel("raw D_ell ratio")
plt.title("TT ratio relative to LCDM baseline")
plt.xlim(2, 2000)
plt.axhline(1, linewidth=1)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("cmb_tt_ratio.png", dpi=160)
plt.close()


print("\nComparison complete.")
print("Created:")
print("  peak_comparison.json")
print("  peak_comparison.csv")
print("  cmb_tt_comparison.png")
print("  cmb_tt_difference.png")
print("  cmb_tt_ratio.png")
