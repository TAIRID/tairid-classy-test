#!/usr/bin/env python3
"""
TAIRID combined Planck fixed-nuisance evaluate.

Purpose:
The previous combined Planck evaluate worked, but Cobaya chose different
nuisance parameter reference values for different candidates. That makes the
candidate ranking unsafe.

This test fixes that.

Method:
1. Run one CDM seed evaluate.
2. Read the nuisance/calibration values from the CDM output table.
3. Freeze those same nuisance/calibration values for every candidate.
4. Re-run all candidates against the same combined Planck likelihood bridge.

Likelihoods:
- planck_2018_lowl.TT
- planck_2018_lowl.EE
- planck_2018_lensing.clik
- planck_2018_highl_plik.TTTEEE

Boundary:
This is still not a full Planck fit.
This is not MCMC.
This does not optimize nuisance parameters.
This does not prove TAIRID cosmology.
It is a fairer fixed-candidate comparison because nuisance settings are held constant.
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


OUTDIR = Path("combined_planck_fixed_nuisance_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = OUTDIR / "candidate_yamls"
RUN_DIR.mkdir(parents=True, exist_ok=True)

PACKAGES_DIR = Path("combined_planck_fixed_nuisance_packages")
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
        "label": "Best internal matrix score anchor",
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
    "boundary": "Fixed-candidate combined Planck evaluate with fixed nuisance settings. Not MCMC and not proof.",
    "python": {
        "version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    },
    "packages_dir": str(PACKAGES_DIR),
    "likelihoods": list(LIKELIHOODS.keys()),
    "install_commands": [],
    "seed_run": {},
    "fixed_nuisance_values": {},
    "cases": [],
}


NON_NUISANCE_COLUMNS = {
    "weight",
    "minuslogpost",
    "chi2__CMB",
    "minuslogprior",
    "minuslogprior__0",
    "minuslogprior__SZ",
    "chi2",
    "chi2__planck_2018_lowl.TT",
    "chi2__planck_2018_lowl.EE",
    "chi2__planck_2018_lensing.clik",
    "chi2__planck_2018_highl_plik.TTTEEE",
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


def make_candidate_info(case, fixed_nuisance=None, output_name=None):
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

    if fixed_nuisance:
        for key, value in fixed_nuisance.items():
            params[key] = value

    out_name = output_name or case["name"]

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
        "output": str(OUTDIR / "cobaya_runs" / out_name),
    }

    return info


def write_yaml(name, info):
    path = RUN_DIR / f"{name}.yml"
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


def parse_output_table(path):
    text = Path(path).read_text().splitlines()

    header_line = None
    data_line = None

    for line in text:
        if line.startswith("#"):
            header_line = line[1:].strip()
        elif line.strip():
            data_line = line.strip()
            break

    if not header_line or not data_line:
        raise RuntimeError(f"Could not parse Cobaya output table: {path}")

    header = header_line.split()
    values = data_line.split()

    if len(values) < len(header):
        raise RuntimeError(f"Output table length mismatch in {path}")

    parsed = {}

    for key, value in zip(header, values):
        try:
            parsed[key] = float(value)
        except Exception:
            parsed[key] = value

    return parsed


def find_output_table(output_prefix):
    candidates = sorted(Path(output_prefix).parent.glob(Path(output_prefix).name + ".*.txt"))

    if not candidates:
        raise RuntimeError(f"No Cobaya output table found for prefix {output_prefix}")

    return candidates[0]


def fixed_nuisance_from_seed_table(parsed):
    nuisance = {}

    for key, value in parsed.items():
        if key in NON_NUISANCE_COLUMNS:
            continue

        if key.startswith("chi2"):
            continue

        if key in BASE_CONSTANTS:
            continue

        if key == "omega_cdm":
            continue

        if isinstance(value, float):
            nuisance[key] = value

    return nuisance


def run_seed_cdm():
    seed_case = CASES[0]
    info = make_candidate_info(seed_case, fixed_nuisance=None, output_name="seed_cdm_for_nuisance")
    yaml_path = write_yaml("seed_cdm_for_nuisance", info)

    entry = run_command(
        "evaluate_seed_cdm_for_nuisance",
        [sys.executable, "-m", "cobaya", "run", str(yaml_path), "--force"],
        timeout=2400,
    )

    output_prefix = str(OUTDIR / "cobaya_runs" / "seed_cdm_for_nuisance")
    table_path = find_output_table(output_prefix)
    parsed = parse_output_table(table_path)
    nuisance = fixed_nuisance_from_seed_table(parsed)

    REPORT["seed_run"] = {
        "yaml_file": str(yaml_path),
        "output_table": str(table_path),
        "status": entry["status"],
        "returncode": entry["returncode"],
        "parsed_chi2_CMB": parsed.get("chi2__CMB"),
        "parsed_highl_chi2": parsed.get("chi2__planck_2018_highl_plik.TTTEEE"),
        "parsed_lensing_chi2": parsed.get("chi2__planck_2018_lensing.clik"),
        "parsed_lowl_TT_chi2": parsed.get("chi2__planck_2018_lowl.TT"),
        "parsed_lowl_EE_chi2": parsed.get("chi2__planck_2018_lowl.EE"),
    }

    REPORT["fixed_nuisance_values"] = nuisance

    fixed_path = OUTDIR / "fixed_nuisance_values_from_seed_cdm.json"
    fixed_path.write_text(json.dumps(nuisance, indent=2))

    return nuisance


def evaluate_candidate(case, fixed_nuisance):
    info = make_candidate_info(case, fixed_nuisance=fixed_nuisance, output_name=case["name"])
    yaml_path = write_yaml(case["name"], info)

    entry = run_command(
        "evaluate_fixed_nuisance_" + case["name"],
        [sys.executable, "-m", "cobaya", "run", str(yaml_path), "--force"],
        timeout=2400,
    )

    output_prefix = str(OUTDIR / "cobaya_runs" / case["name"])

    parsed = {}
    table_path = None

    try:
        table_path = find_output_table(output_prefix)
        parsed = parse_output_table(table_path)
    except Exception as exc:
        parsed = {"parse_error": str(exc)}

    combined_text = (entry.get("stdout_tail") or "") + "\n" + (entry.get("stderr_tail") or "")
    likelihood_lines = extract_likelihood_lines(combined_text)

    result = {
        "name": case["name"],
        "label": case["label"],
        "warm_fraction": case["warm_fraction"],
        "m_ncdm_eV": case["m_ncdm_eV"],
        "yaml_file": str(yaml_path),
        "output_table": str(table_path) if table_path else None,
        "status": entry["status"],
        "returncode": entry["returncode"],
        "stdout_tail": entry["stdout_tail"],
        "stderr_tail": entry["stderr_tail"],
        "likelihood_related_lines": likelihood_lines,
        "chi2_CMB": parsed.get("chi2__CMB"),
        "chi2_highl_plik_TTTEEE": parsed.get("chi2__planck_2018_highl_plik.TTTEEE"),
        "chi2_lensing": parsed.get("chi2__planck_2018_lensing.clik"),
        "chi2_lowl_TT": parsed.get("chi2__planck_2018_lowl.TT"),
        "chi2_lowl_EE": parsed.get("chi2__planck_2018_lowl.EE"),
        "minuslogpost": parsed.get("minuslogpost"),
        "minuslogprior": parsed.get("minuslogprior"),
    }

    if "parse_error" in parsed:
        result["parse_error"] = parsed["parse_error"]

    return result


def add_deltas():
    baseline = None

    for row in REPORT["cases"]:
        if row.get("name") == "cdm_baseline":
            baseline = row
            break

    if baseline is None:
        return

    for key in [
        "chi2_CMB",
        "chi2_highl_plik_TTTEEE",
        "chi2_lensing",
        "chi2_lowl_TT",
        "chi2_lowl_EE",
    ]:
        base_value = baseline.get(key)

        for row in REPORT["cases"]:
            value = row.get(key)

            if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                row["delta_" + key + "_vs_cdm"] = value - base_value
            else:
                row["delta_" + key + "_vs_cdm"] = None


def write_summary_csv():
    rows = REPORT["cases"]

    header = [
        "case",
        "label",
        "status",
        "returncode",
        "warm_fraction",
        "m_ncdm_eV",
        "chi2_CMB",
        "delta_chi2_CMB_vs_cdm",
        "chi2_highl_plik_TTTEEE",
        "delta_chi2_highl_plik_TTTEEE_vs_cdm",
        "chi2_lensing",
        "delta_chi2_lensing_vs_cdm",
        "chi2_lowl_TT",
        "delta_chi2_lowl_TT_vs_cdm",
        "chi2_lowl_EE",
        "delta_chi2_lowl_EE_vs_cdm",
        "minuslogpost",
        "minuslogprior",
    ]

    with open(OUTDIR / "combined_planck_fixed_nuisance_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            writer.writerow(
                [
                    row.get("name"),
                    row.get("label"),
                    row.get("status"),
                    row.get("returncode"),
                    row.get("warm_fraction"),
                    row.get("m_ncdm_eV"),
                    row.get("chi2_CMB"),
                    row.get("delta_chi2_CMB_vs_cdm"),
                    row.get("chi2_highl_plik_TTTEEE"),
                    row.get("delta_chi2_highl_plik_TTTEEE_vs_cdm"),
                    row.get("chi2_lensing"),
                    row.get("delta_chi2_lensing_vs_cdm"),
                    row.get("chi2_lowl_TT"),
                    row.get("delta_chi2_lowl_TT_vs_cdm"),
                    row.get("chi2_lowl_EE"),
                    row.get("delta_chi2_lowl_EE_vs_cdm"),
                    row.get("minuslogpost"),
                    row.get("minuslogprior"),
                ]
            )


def main():
    try:
        import cobaya
        REPORT["cobaya_version"] = getattr(cobaya, "__version__", "unknown")
    except Exception as exc:
        REPORT["cobaya_import_error"] = str(exc)

    install_planck_components()

    fixed_nuisance = run_seed_cdm()

    for case in CASES:
        print("")
        print("Evaluating fixed-nuisance candidate:", case["name"])

        try:
            result = evaluate_candidate(case, fixed_nuisance)
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

    add_deltas()

    passed_cases = [case for case in REPORT["cases"] if case.get("returncode") == 0]

    if passed_cases:
        REPORT["interpretation"] = (
            "Fixed-nuisance combined Planck bridge completed for at least one candidate. "
            "This is a fairer fixed-candidate comparison than the previous mixed-nuisance run, "
            "but it is still not an optimized Planck fit."
        )
    else:
        REPORT["interpretation"] = (
            "No fixed-nuisance candidate completed successfully. Inspect stderr tails."
        )

    report_path = OUTDIR / "combined_planck_fixed_nuisance_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))

    write_summary_csv()

    with open(OUTDIR / "combined_planck_fixed_nuisance_summary.txt", "w") as f:
        f.write("TAIRID combined Planck fixed-nuisance evaluate\n")
        f.write("\n")
        f.write("Boundary: fixed-candidate, fixed-nuisance comparison only. Not MCMC and not proof.\n")
        f.write("\n")
        f.write("Seed CDM nuisance chi2_CMB: ")
        f.write(str(REPORT["seed_run"].get("parsed_chi2_CMB")) + "\n")
        f.write("Fixed nuisance count: ")
        f.write(str(len(REPORT["fixed_nuisance_values"])) + "\n")
        f.write("\n")
        f.write("Candidate results:\n")

        for case in REPORT["cases"]:
            f.write(
                f"{case.get('name')}: status={case.get('status')} "
                f"chi2_CMB={case.get('chi2_CMB')} "
                f"delta_CMB={case.get('delta_chi2_CMB_vs_cdm')} "
                f"highl={case.get('chi2_highl_plik_TTTEEE')} "
                f"delta_highl={case.get('delta_chi2_highl_plik_TTTEEE_vs_cdm')}\n"
            )

        f.write("\n")
        f.write("Interpretation:\n")
        f.write(REPORT["interpretation"] + "\n")

    print("")
    print("TAIRID combined Planck fixed-nuisance evaluate complete.")
    print("Created:")
    print("  combined_planck_fixed_nuisance_outputs/combined_planck_fixed_nuisance_report.json")
    print("  combined_planck_fixed_nuisance_outputs/combined_planck_fixed_nuisance_summary.csv")
    print("  combined_planck_fixed_nuisance_outputs/combined_planck_fixed_nuisance_summary.txt")
    print("  combined_planck_fixed_nuisance_outputs/fixed_nuisance_values_from_seed_cdm.json")
    print("")
    print("Boundary:")
    print("  This is not a full Planck fit.")
    print("  This is a fairer fixed-nuisance bridge comparison.")


if __name__ == "__main__":
    main()
