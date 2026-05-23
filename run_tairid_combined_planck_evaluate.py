#!/usr/bin/env python3
"""
TAIRID combined Planck evaluate probe.

Purpose:
Evaluate fixed TAIRID proxy candidates against the combined available
Planck 2018 Cobaya likelihood bridge:

- planck_2018_lowl.TT
- planck_2018_lowl.EE
- planck_2018_lensing.clik
- planck_2018_highl_plik.TTTEEE

Boundary:
This is not a full Planck proof.
This is not an MCMC.
This does not refit nuisance parameters.
This does not establish TAIRID cosmology.
This is a fixed-candidate combined Planck likelihood bridge test.
"""

import csv
import json
import platform
import re
import subprocess
import sys
import traceback
from pathlib import Path

import yaml


OUTDIR = Path("combined_planck_evaluate_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = OUTDIR / "candidate_yamls"
RUN_DIR.mkdir(parents=True, exist_ok=True)

PACKAGES_DIR = Path("combined_planck_packages_runtime")
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

H = 0.66893180
H0 = H * 100.0
OMEGA_NEUTRAL_PHYSICAL = 0.1200000000

BASE_CONSTANTS = {
    "H0": H0,
    "omega_b": 0.0223700000,
    "n_s": 0.9649,
    "A_s": 2.100549e-9,
    "tau_reio": 0.0544,
}

COMMON_CLASS_EXTRA = {
    "output": "tCl,pCl,lCl,mPk",
    "lensing": "yes",
    "l_max_scalars": 2500,
    "P_k_max_1/Mpc": 50.0,
    "z_max_pk": 3.0,
    "z_pk": "0",
    "N_ur": 3.046,
    "Omega_k": 0.0,
    "Omega_Lambda": 0.6817397872,
    "YHe": 0.245,
    "T_cmb": 2.7255,
}

LIKELIHOODS = {
    "planck_2018_lowl.TT": None,
    "planck_2018_lowl.EE": None,
    "planck_2018_lensing.clik": None,
    "planck_2018_highl_plik.TTTEEE": None,
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
        "label": "Best matrix score anchor from internal proxy scan",
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
        "label": "Old S8-helpful warning case",
        "warm_fraction": 0.050,
        "m_ncdm_eV": 20.0,
    },
]

REPORT = {
    "boundary": "Fixed-candidate combined Planck evaluate probe only. Not MCMC and not proof of TAIRID cosmology.",
    "python": {
        "version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    },
    "packages_dir": str(PACKAGES_DIR),
    "likelihoods": list(LIKELIHOODS.keys()),
    "install_commands": [],
    "cases": [],
}


def run_command(label, command, timeout=2400):
    print("")
    print("Running:", label)
    print("Command:", " ".join(command))

    entry = {
        "label": label,
        "command": command,
        "returncode": None,
        "status": "not_run",
        "stdout_tail": "",
        "stderr_tail": "",
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
        entry["status"] = "passed" if proc.returncode == 0 else "nonzero_exit"
        entry["stdout_tail"] = proc.stdout[-14000:]
        entry["stderr_tail"] = proc.stderr[-14000:]

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

    return entry


def install_planck_components():
    for component in LIKELIHOODS.keys():
        entry = run_command(
            "install_" + component.replace(".", "_"),
            ["cobaya-install", component, "-p", str(PACKAGES_DIR)],
            timeout=2400,
        )
        REPORT["install_commands"].append(entry)


def make_candidate_info(case):
    warm_fraction = float(case["warm_fraction"])
    m_ncdm_eV = case["m_ncdm_eV"]

    omega_cdm = OMEGA_NEUTRAL_PHYSICAL
    extra_args = dict(COMMON_CLASS_EXTRA)

    if warm_fraction > 0.0:
        omega_warm = OMEGA_NEUTRAL_PHYSICAL * warm_fraction
        omega_cold = OMEGA_NEUTRAL_PHYSICAL * (1.0 - warm_fraction)

        omega_cdm = omega_cold

        extra_args["N_ncdm"] = 1
        extra_args["m_ncdm"] = float(m_ncdm_eV)
        extra_args["Omega_ncdm"] = omega_warm / (H * H)

    params = dict(BASE_CONSTANTS)
    params["omega_cdm"] = omega_cdm

    info = {
        "debug": False,
        "packages_path": str(PACKAGES_DIR),
        "theory": {
            "classy": {
                "extra_args": extra_args,
            }
        },
        "likelihood": LIKELIHOODS,
        "params": params,
        "sampler": {
            "evaluate": None,
        },
        "output": str(OUTDIR / "cobaya_runs" / case["name"]),
    }

    return info


def write_yaml(case, info):
    path = RUN_DIR / f"{case['name']}.yml"
    with open(path, "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)
    return path


def extract_likelihood_lines(text):
    lines = []

    for line in text.splitlines():
        low = line.lower()
        if (
            "logpost" in low
            or "loglike" in low
            or "chi2" in low
            or "minuslogpost" in low
            or "minus log" in low
            or "posterior" in low
        ):
            lines.append(line)

    return lines[-120:]


def extract_numbers_from_likelihood_lines(lines):
    values = []

    for line in lines:
        low = line.lower()
        if "chi2" not in low:
            continue

        tokens = re.split(r"[=:\s,\[\]\(\)]+", line)

        for token in tokens:
            try:
                value = float(token)
                if value > 0:
                    values.append(value)
            except Exception:
                pass

    return values


def evaluate_candidate(case):
    info = make_candidate_info(case)
    yaml_path = write_yaml(case, info)

    entry = run_command(
        "evaluate_" + case["name"],
        [sys.executable, "-m", "cobaya", "run", str(yaml_path), "--force"],
        timeout=2400,
    )

    combined_text = (entry.get("stdout_tail") or "") + "\n" + (entry.get("stderr_tail") or "")
    likelihood_lines = extract_likelihood_lines(combined_text)
    chi2_values = extract_numbers_from_likelihood_lines(likelihood_lines)

    result = {
        "name": case["name"],
        "label": case["label"],
        "warm_fraction": case["warm_fraction"],
        "m_ncdm_eV": case["m_ncdm_eV"],
        "yaml_file": str(yaml_path),
        "status": entry["status"],
        "returncode": entry["returncode"],
        "stdout_tail": entry["stdout_tail"],
        "stderr_tail": entry["stderr_tail"],
        "likelihood_related_lines": likelihood_lines,
        "extracted_chi2_values": chi2_values,
        "last_extracted_chi2_value": chi2_values[-1] if chi2_values else None,
    }

    return result


def write_summary_csv():
    rows = REPORT["cases"]

    with open(OUTDIR / "combined_planck_evaluate_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "case",
                "label",
                "status",
                "returncode",
                "warm_fraction",
                "m_ncdm_eV",
                "last_extracted_chi2_value",
                "likelihood_line_count",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row.get("name"),
                    row.get("label"),
                    row.get("status"),
                    row.get("returncode"),
                    row.get("warm_fraction"),
                    row.get("m_ncdm_eV"),
                    row.get("last_extracted_chi2_value"),
                    len(row.get("likelihood_related_lines", [])),
                ]
            )


def add_deltas_if_possible():
    baseline = None

    for row in REPORT["cases"]:
        if row.get("name") == "cdm_baseline":
            baseline = row
            break

    if baseline is None:
        return

    base_chi2 = baseline.get("last_extracted_chi2_value")

    if base_chi2 is None:
        return

    for row in REPORT["cases"]:
        chi2 = row.get("last_extracted_chi2_value")

        if chi2 is None:
            row["delta_chi2_vs_cdm"] = None
        else:
            row["delta_chi2_vs_cdm"] = chi2 - base_chi2


def main():
    try:
        import cobaya
        REPORT["cobaya_version"] = getattr(cobaya, "__version__", "unknown")
    except Exception as exc:
        REPORT["cobaya_import_error"] = str(exc)

    install_planck_components()

    for case in CASES:
        print("")
        print("Evaluating combined Planck candidate:", case["name"])

        try:
            result = evaluate_candidate(case)
        except Exception as exc:
            result = {
                "name": case["name"],
                "label": case["label"],
                "warm_fraction": case["warm_fraction"],
                "m_ncdm_eV": case["m_ncdm_eV"],
                "status": "failed_exception",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        REPORT["cases"].append(result)

    add_deltas_if_possible()

    passed_cases = [case for case in REPORT["cases"] if case.get("returncode") == 0]

    if passed_cases:
        REPORT["interpretation"] = (
            "At least one fixed candidate evaluated successfully against the combined Planck bridge. "
            "This is still not a full Planck fit. Compare extracted likelihood lines and deltas cautiously."
        )
    else:
        REPORT["interpretation"] = (
            "No candidate completed the combined Planck bridge successfully. This is a bridge failure, not a cosmology failure. "
            "Inspect stderr tails for nuisance, foreground, parameter, or likelihood-path issues."
        )

    report_path = OUTDIR / "combined_planck_evaluate_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))

    write_summary_csv()

    with open(OUTDIR / "combined_planck_evaluate_summary.txt", "w") as f:
        f.write("TAIRID combined Planck evaluate probe\n")
        f.write("\n")
        f.write("Boundary: fixed-candidate combined Planck bridge only. Not MCMC and not proof.\n")
        f.write("\n")
        f.write("Likelihoods:\n")
        for likelihood in LIKELIHOODS.keys():
            f.write(f"- {likelihood}\n")

        f.write("\n")
        f.write("Install commands:\n")
        for cmd in REPORT["install_commands"]:
            f.write(f"{cmd['label']}: {cmd['status']} returncode={cmd['returncode']}\n")

        f.write("\n")
        f.write("Candidate evaluate results:\n")
        for case in REPORT["cases"]:
            f.write(
                f"{case.get('name')}: {case.get('status')} returncode={case.get('returncode')} "
                f"last_chi2={case.get('last_extracted_chi2_value')} "
                f"delta_vs_cdm={case.get('delta_chi2_vs_cdm')}\n"
            )

        f.write("\n")
        f.write("Interpretation:\n")
        f.write(REPORT["interpretation"] + "\n")

    print("")
    print("TAIRID combined Planck evaluate probe complete.")
    print("Created:")
    print("  combined_planck_evaluate_outputs/combined_planck_evaluate_report.json")
    print("  combined_planck_evaluate_outputs/combined_planck_evaluate_summary.csv")
    print("  combined_planck_evaluate_outputs/combined_planck_evaluate_summary.txt")
    print("")
    print("Boundary:")
    print("  This is not a full Planck fit.")
    print("  This is a fixed-candidate combined Planck bridge test.")


if __name__ == "__main__":
    main()
