#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np

try:
    from classy import Class
except Exception as exc:
    raise SystemExit(f"Could not import classy. Install failed or CLASS wrapper unavailable: {exc}")

CASES = [
    ("h0698", "tairid_neutral_proxy_h0698.ini"),
    ("physical_match", "tairid_neutral_proxy_physical_match.ini"),
]

def read_ini(path):
    params = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        params[key.strip()] = val.strip()
    return params

summary = {}

for label, ini in CASES:
    print(f"\nRunning {label}: {ini}")
    params = read_ini(ini)
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    cl = cosmo.lensed_cl(2500)
    ell = cl["ell"]
    tt = cl["tt"]
    Dl_raw = ell * (ell + 1) * tt / (2.0 * np.pi)

    out = Path(f"tairid_neutral_proxy_{label}_cl_tt.txt")
    np.savetxt(out, np.column_stack([ell, tt, Dl_raw]), header="ell C_l_TT_raw D_l_TT_raw")
    print(f"Saved {out}")

    summary[label] = {
        "ini": ini,
        "lmax": int(np.max(ell)),
        "first_ell": int(ell[0]),
        "last_ell": int(ell[-1]),
        "output": str(out),
    }

    cosmo.struct_cleanup()
    cosmo.empty()

Path("tairid_classy_run_summary.json").write_text(json.dumps(summary, indent=2))
print("\nDone. Download the workflow artifact named tairid-classy-output.")
