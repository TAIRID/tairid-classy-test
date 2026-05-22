#!/usr/bin/env python3
"""
TAIRID neutral-substrate density tolerance scan.

Purpose:
The previous CLASS comparison showed that the TAIRID physical-density matched
proxy is essentially LCDM-like at the acoustic-peak level when:

    omega_cdm = 0.1200000000

This script asks the next pressure question:

    How much can the neutral substrate's physical density move away from the
    CDM-like value before the first three TT acoustic peaks drift?

Boundary:
This is still a CLASS proxy test. It does not add a true sound speed c_N^2 or
a source-code-level TAIRID perturbation equation. It is the clean next test that
can be done without patching CLASS.
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from classy import Class


BASE = {
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
}

DELTAS = [
    -0.10,
    -0.05,
    -0.02,
    -0.01,
    -0.005,
    0.0,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
]

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

        ell_window = ell[mask]
        dl_window = dl[mask]

        idx = int(np.argmax(dl_window))

        peak_ells.append(int(ell_window[idx]))
        peak_heights.append(float(dl_window[idx]))

    return peak_ells, peak_heights


def peak_ratios(heights):
    return float(heights[1] / heights[0]), float(heights[2] / heights[0])


def make_case(delta):
    params = dict(BASE)

    omega_cdm = BASE["omega_cdm"] * (1.0 + delta)
    params["omega_cdm"] = omega_cdm

    h2 = params["h"] ** 2
    delta_omega = omega_cdm - BASE["omega_cdm"]
    delta_omega_as_omega = delta_omega / h2

    params["Omega_Lambda"] = BASE["Omega_Lambda"] - delta_omega_as_omega

    return params


results = {}
spectra = {}

for delta in DELTAS:
    label = f"delta_{delta:+.3f}"
    label = label.replace("+", "plus_")
    label = label.replace("-", "minus_")
    label = label.replace(".", "p")

    params = make_case(delta)

    print(f"\nRunning density scan case {label}")
    print(f"  omega_cdm = {params['omega_cdm']}")
    print(f"  Omega_Lambda = {params['Omega_Lambda']}")

    ell, tt = run_class(params)
    dl = dl_from_cl(ell, tt)

    peaks_ell, peaks_height = peak_by_windows(ell, dl)
    ratio21, ratio31 = peak_ratios(peaks_height)

    out_txt = f"neutral_density_scan_{label}_cl_tt.txt"

    np.savetxt(
        out_txt,
        np.column_stack([ell, tt, dl]),
        header="ell C_l_TT_raw D_l_TT_raw",
    )

    results[label] = {
        "delta_fraction": float(delta),
        "omega_cdm": float(params["omega_cdm"]),
        "Omega_Lambda": float(params["Omega_Lambda"]),
        "peak_ell": peaks_ell,
        "peak_Dl_raw": peaks_height,
        "ratio_peak2_to_peak1": ratio21,
        "ratio_peak3_to_peak1": ratio31,
        "output_file": out_txt,
    }

    spectra[label] = {
        "ell": ell,
        "dl": dl,
    }


base_label = "delta_plus_0p000"
base = results[base_label]

for label, data in results.items():
    data["delta_vs_baseline"] = {
        "peak_ell_delta": [
            int(data["peak_ell"][i] - base["peak_ell"][i])
            for i in range(3)
        ],
        "peak_height_fractional_delta": [
            float(
                (data["peak_Dl_raw"][i] - base["peak_Dl_raw"][i])
                / base["peak_Dl_raw"][i]
            )
            for i in range(3)
        ],
        "ratio21_delta": float(
            data["ratio_peak2_to_peak1"] - base["ratio_peak2_to_peak1"]
        ),
        "ratio31_delta": float(
            data["ratio_peak3_to_peak1"] - base["ratio_peak3_to_peak1"]
        ),
    }


Path("neutral_density_scan_summary.json").write_text(
    json.dumps(results, indent=2)
)


with open("neutral_density_scan_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(
        [
            "case",
            "delta_fraction",
            "omega_cdm",
            "Omega_Lambda",
            "ell_peak1",
            "ell_peak2",
            "ell_peak3",
            "Dl_peak1_raw",
            "Dl_peak2_raw",
            "Dl_peak3_raw",
            "ratio21",
            "ratio31",
            "delta_ell1",
            "delta_ell2",
            "delta_ell3",
            "frac_delta_height1",
            "frac_delta_height2",
            "frac_delta_height3",
            "delta_ratio21",
            "delta_ratio31",
        ]
    )

    for label, data in results.items():
        d = data["delta_vs_baseline"]

        writer.writerow(
            [
                label,
                data["delta_fraction"],
                data["omega_cdm"],
                data["Omega_Lambda"],
                *data["peak_ell"],
                *data["peak_Dl_raw"],
                data["ratio_peak2_to_peak1"],
                data["ratio_peak3_to_peak1"],
                *d["peak_ell_delta"],
                *d["peak_height_fractional_delta"],
                d["ratio21_delta"],
                d["ratio31_delta"],
            ]
        )


plt.figure(figsize=(12, 6))

for label in [
    "delta_minus_0p100",
    "delta_minus_0p050",
    "delta_minus_0p010",
    "delta_plus_0p000",
    "delta_plus_0p010",
    "delta_plus_0p050",
    "delta_plus_0p100",
]:
    spec = spectra[label]
    plt.plot(spec["ell"], spec["dl"], label=label, linewidth=1.2)

plt.xlabel("multipole ell")
plt.ylabel("raw D_ell_TT")
plt.title("Neutral substrate physical-density scan: TT spectra")
plt.xlim(0, 1600)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("neutral_density_scan_tt_spectra.png", dpi=160)
plt.close()


base_ell = spectra[base_label]["ell"]
base_dl = spectra[base_label]["dl"]

plt.figure(figsize=(12, 5))

for label in [
    "delta_minus_0p100",
    "delta_minus_0p050",
    "delta_minus_0p010",
    "delta_plus_0p010",
    "delta_plus_0p050",
    "delta_plus_0p100",
]:
    spec = spectra[label]
    interp = np.interp(base_ell, spec["ell"], spec["dl"])
    ratio = np.divide(
        interp,
        base_dl,
        out=np.full_like(base_dl, np.nan),
        where=base_dl != 0,
    )
    plt.plot(base_ell, ratio, label=label, linewidth=1.2)

plt.axhline(1.0, linewidth=1)
plt.xlabel("multipole ell")
plt.ylabel("D_ell ratio to baseline")
plt.title("Neutral density scan: TT ratio to physical-match baseline")
plt.xlim(2, 1600)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("neutral_density_scan_tt_ratio.png", dpi=160)
plt.close()


deltas = np.array([results[k]["delta_fraction"] for k in results])
order = np.argsort(deltas)
deltas = deltas[order]

labels_ordered = [list(results.keys())[i] for i in order]

plt.figure(figsize=(9, 5))

for peak_idx in range(3):
    y = [
        results[label]["delta_vs_baseline"]["peak_ell_delta"][peak_idx]
        for label in labels_ordered
    ]
    plt.plot(deltas * 100, y, marker="o", label=f"peak {peak_idx + 1}")

plt.axhline(0, linewidth=1)
plt.xlabel("omega_cdm change from baseline percent")
plt.ylabel("peak ell shift")
plt.title("Acoustic peak shifts under neutral-density changes")
plt.legend()
plt.tight_layout()
plt.savefig("neutral_density_scan_peak_shifts.png", dpi=160)
plt.close()


print("\nNeutral density scan complete.")
print("Created:")
print("  neutral_density_scan_summary.json")
print("  neutral_density_scan_summary.csv")
print("  neutral_density_scan_tt_spectra.png")
print("  neutral_density_scan_tt_ratio.png")
print("  neutral_density_scan_peak_shifts.png")
