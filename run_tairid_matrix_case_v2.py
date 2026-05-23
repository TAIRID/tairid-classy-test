#!/usr/bin/env python3
"""
TAIRID combined matrix scan case runner v2.

This file runs ONE CLASS proxy case and saves the files needed by the aggregator.

Boundary:
Internal CLASS proxy test only.
Not proof of TAIRID cosmology.
"""

import argparse
import csv
import json
import math
import re
import traceback
from pathlib import Path

import numpy as np
from classy import Class


OBS_FSIGMA8 = [
    {"label": "6dFGS", "z": 0.067, "fs8": 0.423, "sigma": 0.055},
    {"label": "SDSS_MGS", "z": 0.150, "fs8": 0.530, "sigma": 0.160},
    {"label": "GAMA", "z": 0.180, "fs8": 0.360, "sigma": 0.090},
    {"label": "BOSS_LOWZ", "z": 0.320, "fs8": 0.384, "sigma": 0.095},
    {"label": "BOSS_DR12_z038", "z": 0.380, "fs8": 0.497, "sigma": 0.045},
    {"label": "WiggleZ_z044", "z": 0.440, "fs8": 0.413, "sigma": 0.080},
    {"label": "BOSS_DR12_z051", "z": 0.510, "fs8": 0.458, "sigma": 0.038},
    {"label": "WiggleZ_z060", "z": 0.600, "fs8": 0.390, "sigma": 0.063},
    {"label": "BOSS_DR12_z061", "z": 0.610, "fs8": 0.436, "sigma": 0.034},
    {"label": "eBOSS_LRG", "z": 0.698, "fs8": 0.473, "sigma": 0.044},
    {"label": "WiggleZ_z073", "z": 0.730, "fs8": 0.437, "sigma": 0.072},
    {"label": "VIPERS", "z": 0.800, "fs8": 0.470, "sigma": 0.080},
    {"label": "eBOSS_ELG", "z": 0.850, "fs8": 0.315, "sigma": 0.095},
    {"label": "eBOSS_QSO", "z": 1.480, "fs8": 0.462, "sigma": 0.045},
]

OBS_Z = np.array([x["z"] for x in OBS_FSIGMA8], dtype=float)

Z_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 2.0, 61),
            OBS_Z,
            np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00]),
        ]
    )
)
Z_GRID.sort()

BAO_Z = [0.106, 0.150, 0.380, 0.510, 0.610, 0.700, 0.850, 1.480, 2.330]
K_VALUES = np.logspace(-4, 1.2, 220)

OMEGA_NEUTRAL_PHYSICAL = 0.1200000000
H = 0.66893180

PEAK_WINDOWS = [(100, 350), (350, 650), (650, 1000)]

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
    "l_max_scalars": 2500,
    "P_k_max_1/Mpc": 50.0,
    "z_max_pk": 3.0,
    "z_pk": ",".join(str(float(z)) for z in Z_GRID),
}


def safe_name(text):
    text = str(text).replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_")


def automatic_case_name(warm_fraction, m_ncdm_eV):
    if warm_fraction <= 0:
        return "cdm_baseline"
    pct = safe_name(f"{warm_fraction * 100.0:g}")
    mass = safe_name(f"{m_ncdm_eV:g}")
    return f"matrix_{pct}pct_{mass}eV"


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def make_params(warm_fraction, m_ncdm_eV):
    params = dict(BASE)

    f = float(warm_fraction)
    if f <= 0:
        return params

    omega_warm = OMEGA_NEUTRAL_PHYSICAL * f
    omega_cold = OMEGA_NEUTRAL_PHYSICAL * (1.0 - f)

    params["omega_cdm"] = omega_cold
    params["N_ncdm"] = 1
    params["m_ncdm"] = float(m_ncdm_eV)
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


def top_hat_window(x):
    x = np.asarray(x)
    w = np.ones_like(x)

    small = np.abs(x) < 1.0e-4
    large = ~small

    xl = x[large]
    w[large] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / (xl ** 3)

    xs = x[small]
    w[small] = 1.0 - (xs ** 2) / 10.0

    return w


def sigma_R_from_pk(k_values, pk_values, R_mpc):
    k = np.asarray(k_values)
    pk = np.asarray(pk_values)

    good = np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0)
    k = k[good]
    pk = pk[good]

    if len(k) < 10:
        return float("nan")

    x = k * R_mpc
    w = top_hat_window(x)

    integrand = (k ** 3) * pk * (w ** 2) / (2.0 * math.pi * math.pi)
    ln_k = np.log(k)

    sigma2 = integrate_trapezoid(integrand, ln_k)

    if sigma2 <= 0 or not np.isfinite(sigma2):
        return float("nan")

    return float(math.sqrt(sigma2))


def compute_growth_quantities(sigma8_values):
    sigma = np.asarray(sigma8_values)
    sigma = np.where(sigma > 1.0e-30, sigma, np.nan)

    ln_sigma = np.log(sigma)
    dlnsigma_dz = np.gradient(ln_sigma, Z_GRID)

    f_growth = -(1.0 + Z_GRID) * dlnsigma_dz
    f_sigma8 = f_growth * sigma

    return f_growth, f_sigma8


def chi_square_to_observations(model_fs8_obs):
    rows = []
    chi2 = 0.0

    for obs, pred in zip(OBS_FSIGMA8, model_fs8_obs):
        residual = float(pred - obs["fs8"])
        pull = float(residual / obs["sigma"])
        contribution = float(pull * pull)
        chi2 += contribution

        rows.append(
            [
                obs["label"],
                obs["z"],
                obs["fs8"],
                obs["sigma"],
                float(pred),
                residual,
                pull,
                contribution,
            ]
        )

    return float(chi2), rows


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


def get_rs_drag(cosmo):
    try:
        return float(cosmo.rs_drag())
    except Exception:
        pass

    try:
        derived = cosmo.get_current_derived_parameters(["rs_drag"])
        return float(derived["rs_drag"])
    except Exception:
        pass

    return float("nan")


def run_case(case_name, warm_fraction, m_ncdm_eV, outdir):
    params = make_params(warm_fraction, m_ncdm_eV)
    omega_m, Omega_m = omega_total_matter_proxy(params)

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    sigma8_z0 = float(cosmo.sigma8())
    S8 = s8_from_sigma8(sigma8_z0, Omega_m)

    cl = cosmo.lensed_cl(2500)
    ell_tt = np.asarray(cl["ell"])
    tt = np.asarray(cl["tt"])
    dl_tt = ell_tt * (ell_tt + 1) * tt / (2.0 * math.pi)
    peak_ell, peak_height = peak_by_windows(ell_tt, dl_tt)

    ell_lensing, pp = get_lensing_pp(cosmo, 2500)
    lensing_scaled = np.zeros_like(pp, dtype=float)
    good_lensing = ell_lensing > 0
    lensing_scaled[good_lensing] = (ell_lensing[good_lensing] ** 4) * pp[good_lensing] / (2.0 * math.pi)

    pk_z0 = []
    for k in K_VALUES:
        try:
            pk_z0.append(float(cosmo.pk(float(k), 0.0)))
        except Exception:
            pk_z0.append(float("nan"))
    pk_z0 = np.asarray(pk_z0)

    h = float(params["h"])
    R8_mpc = 8.0 / h
    sigma8_values = []

    for z in Z_GRID:
        pk_values = []

        for k in K_VALUES:
            try:
                pk_values.append(float(cosmo.pk(float(k), float(z))))
            except Exception:
                pk_values.append(float("nan"))

        pk_values = np.asarray(pk_values)
        sigma8_values.append(sigma_R_from_pk(K_VALUES, pk_values, R8_mpc))

    sigma8_values = np.asarray(sigma8_values)
    f_growth, f_sigma8 = compute_growth_quantities(sigma8_values)
    model_fs8_obs = np.interp(OBS_Z, Z_GRID, f_sigma8)
    growth_chi2, residual_rows = chi_square_to_observations(model_fs8_obs)
    growth_dof = len(OBS_FSIGMA8)
    growth_reduced_chi2 = growth_chi2 / growth_dof

    rd = get_rs_drag(cosmo)
    bao_rows = []

    for z in BAO_Z:
        da = float(cosmo.angular_distance(z))
        dm = da * (1.0 + z)
        hubble = float(cosmo.Hubble(z))
        dh = 1.0 / hubble
        dv = (z * dm * dm * dh) ** (1.0 / 3.0)

        bao_rows.append(
            [
                float(z),
                da,
                dm,
                dh,
                dv,
                rd,
                dm / rd,
                dh / rd,
                dv / rd,
            ]
        )

    cosmo.struct_cleanup()
    cosmo.empty()

    write_csv(
        outdir / "pk_z0.csv",
        ["k_1_per_Mpc", "Pk_z0"],
        [[float(k), float(pk)] for k, pk in zip(K_VALUES, pk_z0)],
    )

    write_csv(
        outdir / "cmb_tt_spectrum.csv",
        ["ell", "D_ell_TT"],
        [[int(e), float(v)] for e, v in zip(ell_tt, dl_tt)],
    )

    write_csv(
        outdir / "cmb_lensing_curve.csv",
        ["ell", "lensing_scaled"],
        [[int(e), float(v)] for e, v in zip(ell_lensing, lensing_scaled)],
    )

    write_csv(
        outdir / "bao_distances.csv",
        ["z", "D_A_Mpc", "D_M_Mpc", "D_H_Mpc", "D_V_Mpc", "r_d_Mpc", "D_M_over_rd", "D_H_over_rd", "D_V_over_rd"],
        bao_rows,
    )

    write_csv(
        outdir / "growth_curve.csv",
        ["z", "sigma8_integral", "f_growth", "f_sigma8"],
        [[float(z), float(sig), float(fg), float(fs8)] for z, sig, fg, fs8 in zip(Z_GRID, sigma8_values, f_growth, f_sigma8)],
    )

    write_csv(
        outdir / "growth_observation_residuals.csv",
        ["label", "z", "observed_fs8", "sigma", "model_fs8", "residual_model_minus_obs", "pull_sigma", "chi2_contribution"],
        residual_rows,
    )

    summary = {
        "name": case_name,
        "status": "success",
        "warm_fraction": float(warm_fraction),
        "m_ncdm_eV": None if m_ncdm_eV is None else float(m_ncdm_eV),
        "Omega_m_total": float(Omega_m),
        "omega_m_physical": float(omega_m),
        "sigma8_z0": float(sigma8_z0),
        "S8": float(S8),
        "growth_chi2": float(growth_chi2),
        "growth_dof": int(growth_dof),
        "growth_reduced_chi2": float(growth_reduced_chi2),
        "peak_ell": peak_ell,
        "peak_height": peak_height,
        "r_d_Mpc": float(rd),
        "boundary": "Internal CLASS proxy case. Not a final likelihood and not proof of TAIRID cosmology.",
    }

    (outdir / "case_summary.json").write_text(json.dumps(summary, indent=2))

    write_csv(
        outdir / "case_summary.csv",
        ["name", "status", "warm_fraction", "m_ncdm_eV", "Omega_m_total", "sigma8_z0", "S8", "growth_chi2", "growth_reduced_chi2", "ell_peak1", "ell_peak2", "ell_peak3", "r_d_Mpc"],
        [[summary["name"], summary["status"], summary["warm_fraction"], summary["m_ncdm_eV"], summary["Omega_m_total"], summary["sigma8_z0"], summary["S8"], summary["growth_chi2"], summary["growth_reduced_chi2"], peak_ell[0], peak_ell[1], peak_ell[2], summary["r_d_Mpc"]]],
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)
    parser.add_argument("--warm-fraction", type=float, required=True)
    parser.add_argument("--m-ncdm-ev", type=float, default=None)
    args = parser.parse_args()

    if args.warm_fraction > 0 and args.m_ncdm_ev is None:
        raise ValueError("--m-ncdm-ev is required when --warm-fraction is greater than zero.")

    name = args.name or automatic_case_name(args.warm_fraction, args.m_ncdm_ev)
    outdir = Path("matrix_case_outputs") / safe_name(name)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running combined matrix case:", name)
    print("warm_fraction:", args.warm_fraction)
    print("m_ncdm_eV:", args.m_ncdm_ev)

    try:
        summary = run_case(name, args.warm_fraction, args.m_ncdm_ev, outdir)
        print("success")
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        error_summary = {
            "name": name,
            "status": "failed",
            "warm_fraction": float(args.warm_fraction),
            "m_ncdm_eV": None if args.m_ncdm_ev is None else float(args.m_ncdm_ev),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (outdir / "case_summary.json").write_text(json.dumps(error_summary, indent=2))
        print("failed")
        print(json.dumps(error_summary, indent=2))
        raise


if __name__ == "__main__":
    main()
