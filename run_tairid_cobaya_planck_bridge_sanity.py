#!/usr/bin/env python3
"""
TAIRID Cobaya / Planck bridge sanity check.

Purpose:
Check whether GitHub Actions can install and import the tooling needed for a
future real Planck likelihood bridge.

This does NOT run Planck likelihoods yet.

It checks:
- Python environment
- Cobaya import
- Cobaya version if available
- classy import
- CLASS can compute a small CMB packet for the current best TAIRID proxy anchor
- YAML bridge files can be written for the future Planck likelihood attempt

Boundary:
This is not a Planck likelihood.
This does not download or install official Planck data.
This does not use clik.
This does not prove TAIRID cosmology.
This is only an environment and bridge-readiness check.
"""

import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml


OUTDIR = Path("cobaya_planck_bridge_sanity_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

REPORT = {
    "boundary": "Environment sanity check only. Not a Planck likelihood and not proof of TAIRID cosmology.",
    "python": {
        "version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    },
    "checks": {},
    "next_step_if_passed": "Attempt a real Cobaya/Planck likelihood installation or prepare a clik data bridge.",
}


def record_check(name, status, detail=None, error=None):
    REPORT["checks"][name] = {
        "status": status,
        "detail": detail,
        "error": error,
    }


def try_imports():
    try:
        import cobaya

        version = getattr(cobaya, "__version__", "unknown")
        record_check("import_cobaya", "passed", {"version": version})
    except Exception as exc:
        record_check("import_cobaya", "failed", error=str(exc))
        REPORT["cobaya_traceback"] = traceback.format_exc()

    try:
        from cobaya.run import run as cobaya_run

        record_check("import_cobaya_run", "passed", {"callable": str(cobaya_run)})
    except Exception as exc:
        record_check("import_cobaya_run", "failed", error=str(exc))
        REPORT["cobaya_run_traceback"] = traceback.format_exc()

    try:
        import classy
        from classy import Class

        record_check("import_classy", "passed", {"Class": str(Class)})
    except Exception as exc:
        record_check("import_classy", "failed", error=str(exc))
        REPORT["classy_traceback"] = traceback.format_exc()


def run_command_check(name, command):
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )

        record_check(
            name,
            "passed" if proc.returncode == 0 else "warning_nonzero_exit",
            {
                "command": command,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            },
        )
    except Exception as exc:
        record_check(name, "failed", {"command": command}, error=str(exc))


def run_small_class_anchor():
    try:
        from classy import Class

        h = 0.66893180
        omega_neutral = 0.1200000000
        warm_fraction = 0.0175
        m_ncdm_eV = 15.0

        omega_warm = omega_neutral * warm_fraction
        omega_cold = omega_neutral * (1.0 - warm_fraction)

        params = {
            "output": "tCl,pCl,lCl,mPk",
            "lensing": "yes",
            "h": h,
            "omega_b": 0.0223700000,
            "omega_cdm": omega_cold,
            "N_ur": 3.046,
            "N_ncdm": 1,
            "m_ncdm": m_ncdm_eV,
            "Omega_ncdm": omega_warm / (h * h),
            "Omega_k": 0.0,
            "Omega_Lambda": 0.6817397872,
            "n_s": 0.9649,
            "A_s": 2.100549e-9,
            "tau_reio": 0.0544,
            "YHe": 0.245,
            "T_cmb": 2.7255,
            "l_max_scalars": 1200,
            "P_k_max_1/Mpc": 20.0,
            "z_max_pk": 1.0,
            "z_pk": "0",
        }

        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()

        sigma8 = float(cosmo.sigma8())

        cl = cosmo.lensed_cl(1200)
        ell = np.asarray(cl["ell"])
        tt = np.asarray(cl["tt"])
        te = np.asarray(cl["te"])
        ee = np.asarray(cl["ee"])

        dl_tt = ell * (ell + 1) * tt / (2.0 * np.pi)
        dl_te = ell * (ell + 1) * te / (2.0 * np.pi)
        dl_ee = ell * (ell + 1) * ee / (2.0 * np.pi)

        packet_file = OUTDIR / "small_class_anchor_packet.csv"

        with open(packet_file, "w") as f:
            f.write("ell,D_ell_TT,D_ell_TE,D_ell_EE\n")
            for e, a, b, c in zip(ell, dl_tt, dl_te, dl_ee):
                f.write(f"{int(e)},{float(a)},{float(b)},{float(c)}\n")

        cosmo.struct_cleanup()
        cosmo.empty()

        record_check(
            "small_class_anchor_compute",
            "passed",
            {
                "case": "best_score_anchor_1p75pct_15eV",
                "sigma8": sigma8,
                "packet_file": str(packet_file),
                "ell_count": int(len(ell)),
            },
        )

    except Exception as exc:
        record_check("small_class_anchor_compute", "failed", error=str(exc))
        REPORT["small_class_anchor_traceback"] = traceback.format_exc()


def write_future_bridge_yaml_files():
    future_planck_note = {
        "boundary": "Template only. Do not treat as a working Planck likelihood run.",
        "purpose": "Future Cobaya/Planck bridge file once official Planck likelihood data are installed.",
        "important": [
            "This file is not run in the sanity check.",
            "Official Planck likelihood data are not included here.",
            "A real run needs Planck likelihood installation, data path, nuisance parameters, and covariance handling.",
        ],
    }

    template = {
        "theory": {
            "classy": {
                "extra_args": {
                    "output": "tCl,pCl,lCl,mPk",
                    "lensing": "yes",
                    "l_max_scalars": 2500,
                    "P_k_max_1/Mpc": 50.0,
                }
            }
        },
        "likelihood": {
            "PLACEHOLDER_planck_likelihood_goes_here": None
        },
        "params": {
            "h": 0.66893180,
            "omega_b": 0.0223700000,
            "omega_cdm": 0.1179000000,
            "N_ur": 3.046,
            "N_ncdm": 1,
            "m_ncdm": 15.0,
            "Omega_ncdm": 0.004697,
            "Omega_k": 0.0,
            "Omega_Lambda": 0.6817397872,
            "n_s": 0.9649,
            "A_s": 2.100549e-9,
            "tau_reio": 0.0544,
        },
        "sampler": {
            "evaluate": None
        },
    }

    note_path = OUTDIR / "future_planck_bridge_note.json"
    yaml_path = OUTDIR / "future_cobaya_planck_template.yml"

    note_path.write_text(json.dumps(future_planck_note, indent=2))

    with open(yaml_path, "w") as f:
        yaml.safe_dump(template, f, sort_keys=False)

    record_check(
        "write_future_bridge_templates",
        "passed",
        {
            "note": str(note_path),
            "yaml_template": str(yaml_path),
        },
    )


def main():
    try_imports()

    run_command_check("python_m_cobaya_help", [sys.executable, "-m", "cobaya", "--help"])

    run_small_class_anchor()

    write_future_bridge_yaml_files()

    report_path = OUTDIR / "cobaya_planck_bridge_sanity_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))

    with open(OUTDIR / "cobaya_planck_bridge_sanity_summary.txt", "w") as f:
        f.write("TAIRID Cobaya / Planck bridge sanity check\n")
        f.write("\n")
        f.write("Boundary: environment sanity check only. Not a Planck likelihood.\n")
        f.write("\n")

        for name, check in REPORT["checks"].items():
            f.write(f"{name}: {check['status']}\n")

        f.write("\n")
        f.write("Next step if passed: attempt real Planck likelihood installation or clik bridge.\n")

    print("")
    print("TAIRID Cobaya / Planck bridge sanity check complete.")
    print("Created:")
    print("  cobaya_planck_bridge_sanity_outputs/cobaya_planck_bridge_sanity_report.json")
    print("  cobaya_planck_bridge_sanity_outputs/cobaya_planck_bridge_sanity_summary.txt")
    print("  cobaya_planck_bridge_sanity_outputs/future_cobaya_planck_template.yml")
    print("  cobaya_planck_bridge_sanity_outputs/small_class_anchor_packet.csv")
    print("")
    print("Boundary:")
    print("  This is not a Planck likelihood.")
    print("  This only checks whether the bridge environment is ready.")


if __name__ == "__main__":
    main()
