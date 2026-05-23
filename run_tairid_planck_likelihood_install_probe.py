#!/usr/bin/env python3
"""
TAIRID Planck likelihood install probe.

Purpose:
Check whether GitHub Actions can install or prepare Planck likelihood components
through Cobaya.

Boundary:
This is not a Planck likelihood run.
This does not evaluate TAIRID against Planck.
This only tests whether the likelihood installation path is available in this environment.

Expected possibilities:
- PASS: Cobaya installs the requested Planck likelihood package.
- PARTIAL: Cobaya installs some support files but not all Planck clik components.
- FAIL: GitHub cannot install the Planck likelihood package in this runner.

A fail here does not kill the cosmology test. It means we need a different Planck route,
such as a local Planck data package, NPIPE/native likelihood route, or external likelihood bridge.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


OUTDIR = Path("planck_likelihood_install_probe_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

PACKAGES_DIR = OUTDIR / "cobaya_packages"
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

REPORT = {
    "boundary": "Planck likelihood install probe only. Not a Planck likelihood run.",
    "python": {
        "version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    },
    "paths": {
        "packages_dir": str(PACKAGES_DIR),
    },
    "commands": [],
    "checks": {},
    "interpretation": "",
}


def run_command(label, command, timeout=1800):
    print("")
    print("Running:", label)
    print("Command:", " ".join(command))

    entry = {
        "label": label,
        "command": command,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "status": "not_run",
    }

    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )

        entry["returncode"] = proc.returncode
        entry["stdout_tail"] = proc.stdout[-6000:]
        entry["stderr_tail"] = proc.stderr[-6000:]
        entry["status"] = "passed" if proc.returncode == 0 else "nonzero_exit"

        print("Return code:", proc.returncode)
        print("STDOUT tail:")
        print(entry["stdout_tail"])
        print("STDERR tail:")
        print(entry["stderr_tail"])

    except Exception as exc:
        entry["status"] = "failed_exception"
        entry["error"] = str(exc)
        entry["traceback"] = traceback.format_exc()

        print("FAILED:", exc)

    REPORT["commands"].append(entry)
    return entry


def check_imports():
    try:
        import cobaya
        REPORT["checks"]["import_cobaya"] = {
            "status": "passed",
            "version": getattr(cobaya, "__version__", "unknown"),
        }
    except Exception as exc:
        REPORT["checks"]["import_cobaya"] = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    try:
        from classy import Class
        REPORT["checks"]["import_classy"] = {
            "status": "passed",
            "Class": str(Class),
        }
    except Exception as exc:
        REPORT["checks"]["import_classy"] = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def list_packages_dir():
    files = []

    for path in PACKAGES_DIR.rglob("*"):
        try:
            rel = path.relative_to(PACKAGES_DIR)
            files.append(
                {
                    "path": str(rel),
                    "is_dir": path.is_dir(),
                    "size": path.stat().st_size if path.is_file() else None,
                }
            )
        except Exception:
            pass

    REPORT["packages_dir_listing_sample"] = files[:300]
    REPORT["packages_dir_file_count"] = len(files)


def write_probe_yaml():
    yaml_text = """
theory:
  classy:
    extra_args:
      output: tCl,pCl,lCl,mPk
      lensing: 'yes'
      l_max_scalars: 2500
      P_k_max_1/Mpc: 50.0

likelihood:
  planck_2018_lowl.TT: null
  planck_2018_lowl.EE: null
  planck_2018_lensing.clik: null

params:
  h: 0.6689318
  omega_b: 0.02237
  omega_cdm: 0.1179
  N_ur: 3.046
  N_ncdm: 1
  m_ncdm: 15.0
  Omega_ncdm: 0.004697
  Omega_k: 0.0
  Omega_Lambda: 0.6817397872
  n_s: 0.9649
  A_s: 2.100549e-09
  tau_reio: 0.0544

sampler:
  evaluate: null

packages_path: planck_likelihood_install_probe_outputs/cobaya_packages
"""

    path = OUTDIR / "planck_probe_template.yml"
    path.write_text(yaml_text.strip() + "\n")

    REPORT["checks"]["write_probe_yaml"] = {
        "status": "passed",
        "path": str(path),
    }

    return path


def main():
    check_imports()

    run_command(
        "cobaya_help",
        [sys.executable, "-m", "cobaya", "--help"],
        timeout=120,
    )

    run_command(
        "cobaya_install_help",
        ["cobaya-install", "--help"],
        timeout=120,
    )

    yaml_path = write_probe_yaml()

    run_command(
        "install_planck_lowl_TT",
        ["cobaya-install", "planck_2018_lowl.TT", "-p", str(PACKAGES_DIR)],
        timeout=1800,
    )

    run_command(
        "install_planck_lowl_EE",
        ["cobaya-install", "planck_2018_lowl.EE", "-p", str(PACKAGES_DIR)],
        timeout=1800,
    )

    run_command(
        "install_planck_lensing_clik",
        ["cobaya-install", "planck_2018_lensing.clik", "-p", str(PACKAGES_DIR)],
        timeout=1800,
    )

    run_command(
        "install_from_probe_yaml",
        ["cobaya-install", str(yaml_path), "-p", str(PACKAGES_DIR)],
        timeout=1800,
    )

    list_packages_dir()

    statuses = [cmd["status"] for cmd in REPORT["commands"]]

    if all(status == "passed" for status in statuses):
        REPORT["interpretation"] = (
            "All install probe commands passed. The GitHub runner appears able to prepare "
            "the Planck likelihood package path for the next bridge test."
        )
    elif any(status == "passed" for status in statuses):
        REPORT["interpretation"] = (
            "The install probe partially passed. Some Cobaya/Planck pieces are available, "
            "but at least one Planck install command failed or exited nonzero. Inspect command tails."
        )
    else:
        REPORT["interpretation"] = (
            "The install probe failed. This does not falsify TAIRID. It means GitHub cannot prepare "
            "the Planck likelihood route this way, so we need another route."
        )

    report_path = OUTDIR / "planck_likelihood_install_probe_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))

    with open(OUTDIR / "planck_likelihood_install_probe_summary.txt", "w") as f:
        f.write("TAIRID Planck likelihood install probe\n")
        f.write("\n")
        f.write("Boundary: install probe only. Not a Planck likelihood run.\n")
        f.write("\n")

        for cmd in REPORT["commands"]:
            f.write(f"{cmd['label']}: {cmd['status']} returncode={cmd['returncode']}\n")

        f.write("\n")
        f.write("Interpretation:\n")
        f.write(REPORT["interpretation"] + "\n")

    print("")
    print("TAIRID Planck likelihood install probe complete.")
    print("Created:")
    print("  planck_likelihood_install_probe_outputs/planck_likelihood_install_probe_report.json")
    print("  planck_likelihood_install_probe_outputs/planck_likelihood_install_probe_summary.txt")
    print("  planck_likelihood_install_probe_outputs/planck_probe_template.yml")
    print("")
    print("Boundary:")
    print("  This is not a Planck likelihood run.")
    print("  This only tests the install path.")


if __name__ == "__main__":
    main()
