#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np

try:
    from classy import Class
except Exception as exc:
    raise SystemExit(f"Could not import classy: {exc}")

CASES = {
    "h0698": {
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
    "physical_match": {
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

summary = {}

for label, params in CASES.items():
    print(f"\nRunning fixed TAIRID proxy case: {label}")
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    cl = cosmo.lensed_cl(2500)
    ell = cl["ell"]
    tt = cl["tt"]
    Dl_raw = ell * (ell + 1) * tt / (2.0 * np.pi)

    out = Path(f"tairid_fixed_{label}_cl_tt.txt")
    np.savetxt(
        out,
        np.column_stack([ell, tt, Dl_raw]),
        header="ell C_l_TT_raw D_l_TT_raw"
    )

    peak_mask = (ell >= 50) & (ell <= 500)
    peak_ell = int(ell[peak_mask][np.argmax(Dl_raw[peak_mask])])

    summary[label] = {
        "output_file": str(out),
        "lmax": int(np.max(ell)),
        "first_peak_ell_raw": peak_ell,
        "params": params,
    }

    print(f"Saved {out}")
    print(f"Approx raw first peak ell: {peak_ell}")

    cosmo.struct_cleanup()
    cosmo.empty()

Path("tairid_fixed_classy_summary.json").write_text(json.dumps(summary, indent=2))
print("\nDone. Download artifact: tairid-classy-fixed-output")
