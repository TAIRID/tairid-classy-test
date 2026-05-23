#!/usr/bin/env python3
"""
TAIRID Planck matched relief control scan.

Purpose:
The prior relief scan showed a large apparent improvement when A_s was lowered
by 1%. That may not be TAIRID-specific. It may simply mean the fixed baseline
A_s was too high for this fixed-nuisance bridge.

This control scan gives CDM the same standard-parameter relief options as the
TAIRID proxy candidates.

Question:
Does the TAIRID proxy still help after CDM receives the same A_s / n_s /
neutral-density relief?

Likelihoods:
- planck_2018_lowl.TT
- planck_2018_lowl.EE
- planck_2018_lensing.clik
- planck_2018_highl_plik.TTTEEE

Boundary:
This is not a full Planck fit.
This is not MCMC.
This does not optimize nuisance parameters.
This does not prove TAIRID cosmology.
It is a matched fixed-nuisance control scan.
"""

import csv
import json
import math
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import yaml


OUTDIR = Path("planck_relief_control_scan_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = OUTDIR / "candidate_yamls"
RUN_DIR.mkdir(parents=True, exist_ok=True)

PACKAGES_DIR = Path("planck_relief_control_packages")
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

H = 0.66893180
H0 = H * 100.0

OMEGA_B = 0.0223700000
OMEGA_NEUTRAL_BASE = 0.1200000000
N_S_BASE = 0.9649
A_S_BASE = 2.100549e-9
TAU_REIO = 0.0544

BASE_CONSTANTS = {
    "H0": H0,
    "omega_b": OMEGA_B,
    "n_s": N_S_BASE,
    "A_s": A_S_BASE,
    "tau_reio": TAU_REIO,
}

LIKELIHOODS = {
    "planck_2018_lowl.TT": None,
    "planck_2018_lowl.EE": None,
    "planck_2018_lensing.clik": None,
    "planck_2018_highl_plik.TTTEEE": None,
}

NON_NUISANCE_COLUMNS = {
    "weight",
    "minuslogpost",
    "minuslogprior",
    "minuslogprior__0",
    "minuslogprior__SZ",
    "chi2",
    "chi2__CMB",
    "chi2__planck_2018_lowl.TT",
    "chi2__planck_2018_lowl.EE",
    "chi2__planck_2018_lensing.clik",
    "chi2__planck_2018_highl_plik.TTTEEE",
}

BASE_CLASS_EXTRA = {
    "output": "tCl,pCl,lCl,mPk",
    "lensing": "yes",
    "l_max_scalars": 2500,
    "P_k_max_1/Mpc": 50.0,
    "z_max_pk": 3.0,
    "z_pk": "0",
    "N_ur": 3.046,
    "Omega_k": 0.0,
    "YHe": 0.245,
    "T_cmb": 2.7255,
}

TRIAL_SPECS = [
    ("base", 1.000, 0.0000, 1.000),
    ("As_down_1pct", 0.990, 0.0000, 1.000),
    ("As_up_1pct", 1.010, 0.0000, 1.000),
    ("ns_down_0p003", 1.000, -0.0030, 1.000),
    ("ns_up_0p003", 1.000, 0.0030, 1.000),
    ("neutral_down_1pct", 1.000, 0.0000, 0.990),
    ("neutral_up_1pct", 1.000, 0.0000, 1.010),
    ("combo_soft", 0.990, -0.0030, 0.990),
]

ANCHORS = [
    {
        "anchor": "cdm_control",
        "warm_fraction": 0.0,
        "m_ncdm_eV": None,
    },
    {
        "anchor": "best_planck_penalty",
        "warm_fraction": 0.0125,
        "m_ncdm_eV": 17.5,
    },
    {
        "anchor": "strict_s8_tradeoff",
        "warm_fraction": 0.015,
        "m_ncdm_eV": 15.0,
    },
    {
        "anchor": "near_tradeoff",
        "warm_fraction": 0.0175,
        "m_ncdm_eV": 17.5,
    },
]

CASES = []

for anchor in ANCHORS:
    wf = anchor["warm_fraction"]
    mass = anchor["m_ncdm_eV"]
    anchor_name = anchor["anchor"]

    for label, A_s_scale, n_s_shift, neutral_scale in TRIAL_SPECS:
        if wf <= 0:
            name = f"cdm_{label}"
        else:
            pct = str(wf * 100.0).replace(".", "p")
            mass_text = str(mass).replace(".", "p")
            name = f"{anchor_name}_{pct}pct_{mass_text}eV_{label}"

        CASES.append(
            {
                "name": name,
                "anchor": anchor_name,
                "warm_fraction": wf,
                "m_ncdm_eV": mass,
                "A_s_scale": A_s_scale,
                "n_s_shift": n_s_shift,
                "neutral_scale": neutral_scale,
            }
        )

REPORT = {
    "boundary": "Matched fixed-nuisance control scan. Not MCMC and not proof.",
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
        entry["stdout_tail"] = proc.stdout[-12000:]
        entry["stderr_tail"] = proc.stderr[-12000:]

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


def omega_lambda_for_flatish(omega_neutral):
    omega_m = OMEGA_B + omega_neutral
    return 1.0 - (omega_m / (H * H))


def make_candidate_info(case, fixed_nuisance=None, output_name=None):
    warm_fraction = float(case["warm_fraction"])
    m_ncdm_eV = case["m_ncdm_eV"]

    A_s_scale = float(case.get("A_s_scale", 1.0))
    n_s_shift = float(case.get("n_s_shift", 0.0))
    neutral_scale = float(case.get("neutral_scale", 1.0))

    omega_neutral = OMEGA_NEUTRAL_BASE * neutral_scale

    omega_cdm = omega_neutral
    extra_args = dict(BASE_CLASS_EXTRA)
    extra_args["Omega_Lambda"] = omega_lambda_for_flatish(omega_neutral)

    if warm_fraction > 0.0:
        omega_warm = omega_neutral * warm_fraction
        omega_cold = omega_neutral * (1.0 - warm_fraction)

        omega_cdm = omega_cold

        extra_args["N_ncdm"] = 1
        extra_args["m_ncdm"] = float(m_ncdm_eV)
        extra_args["Omega_ncdm"] = omega_warm / (H * H)

    params = dict(BASE_CONSTANTS)
    params["omega_cdm"] = omega_cdm
    params["A_s"] = A_S_BASE * A_s_scale
    params["n_s"] = N_S_BASE + n_s_shift

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


def parse_output_table(path):
    lines = Path(path).read_text().splitlines()

    header_line = None
    data_line = None

    for line in lines:
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
    prefix = Path(output_prefix)
    candidates = sorted(prefix.parent.glob(prefix.name + ".*.txt"))

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

        if isinstance(value, float) and math.isfinite(value):
            nuisance[key] = value

    return nuisance


def run_seed_cdm():
    seed_case = {
        "name": "seed_cdm_for_nuisance",
        "anchor": "seed",
        "warm_fraction": 0.0,
        "m_ncdm_eV": None,
        "A_s_scale": 1.0,
        "n_s_shift": 0.0,
        "neutral_scale": 1.0,
    }

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


def compute_s8_for_case(case):
    try:
        from classy import Class

        warm_fraction = float(case["warm_fraction"])
        m_ncdm_eV = case["m_ncdm_eV"]

        A_s_scale = float(case.get("A_s_scale", 1.0))
        n_s_shift = float(case.get("n_s_shift", 0.0))
        neutral_scale = float(case.get("neutral_scale", 1.0))

        omega_neutral = OMEGA_NEUTRAL_BASE * neutral_scale

        params = dict(BASE_CLASS_EXTRA)
        params.update(
            {
                "h": H,
                "omega_b": OMEGA_B,
                "omega_cdm": omega_neutral,
                "n_s": N_S_BASE + n_s_shift,
                "A_s": A_S_BASE * A_s_scale,
                "tau_reio": TAU_REIO,
                "Omega_Lambda": omega_lambda_for_flatish(omega_neutral),
            }
        )

        if warm_fraction > 0.0:
            omega_warm = omega_neutral * warm_fraction
            omega_cold = omega_neutral * (1.0 - warm_fraction)

            params["omega_cdm"] = omega_cold
            params["N_ncdm"] = 1
            params["m_ncdm"] = float(m_ncdm_eV)
            params["Omega_ncdm"] = omega_warm / (H * H)

        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()

        sigma8 = float(cosmo.sigma8())
        omega_m_physical = OMEGA_B + omega_neutral
        Omega_m = omega_m_physical / (H * H)
        S8 = sigma8 * math.sqrt(Omega_m / 0.3)

        cosmo.struct_cleanup()
        cosmo.empty()

        return sigma8, S8

    except Exception:
        return None, None


def evaluate_candidate(case, fixed_nuisance):
    info = make_candidate_info(case, fixed_nuisance=fixed_nuisance, output_name=case["name"])
    yaml_path = write_yaml(case["name"], info)

    entry = run_command(
        "evaluate_control_" + case["name"],
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

    sigma8, S8 = compute_s8_for_case(case)

    result = {
        "name": case["name"],
        "anchor": case["anchor"],
        "warm_fraction": case["warm_fraction"],
        "m_ncdm_eV": case["m_ncdm_eV"],
        "A_s_scale": case.get("A_s_scale", 1.0),
        "n_s_shift": case.get("n_s_shift", 0.0),
        "neutral_scale": case.get("neutral_scale", 1.0),
        "sigma8": sigma8,
        "S8": S8,
        "yaml_file": str(yaml_path),
        "output_table": str(table_path) if table_path else None,
        "status": entry["status"],
        "returncode": entry["returncode"],
        "stdout_tail": entry["stdout_tail"],
        "stderr_tail": entry["stderr_tail"],
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


def add_deltas_and_rank():
    seed_baseline = None

    for row in REPORT["cases"]:
        if row.get("name") == "cdm_base":
            seed_baseline = row
            break

    cdm_rows = [
        row for row in REPORT["cases"]
        if row.get("anchor") == "cdm_control"
        and isinstance(row.get("chi2_CMB"), (int, float))
    ]

    best_cdm = min(cdm_rows, key=lambda row: row["chi2_CMB"]) if cdm_rows else None

    REPORT["best_cdm_control"] = best_cdm

    for row in REPORT["cases"]:
        if seed_baseline:
            for key in ["chi2_CMB", "chi2_highl_plik_TTTEEE", "chi2_lensing", "chi2_lowl_TT", "chi2_lowl_EE"]:
                base_value = seed_baseline.get(key)
                value = row.get(key)

                if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                    row["delta_" + key + "_vs_base_cdm"] = value - base_value
                else:
                    row["delta_" + key + "_vs_base_cdm"] = None

        if best_cdm:
            for key in ["chi2_CMB", "chi2_highl_plik_TTTEEE", "chi2_lensing", "chi2_lowl_TT", "chi2_lowl_EE"]:
                base_value = best_cdm.get(key)
                value = row.get(key)

                if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                    row["delta_" + key + "_vs_best_cdm_control"] = value - base_value
                else:
                    row["delta_" + key + "_vs_best_cdm_control"] = None

    for row in REPORT["cases"]:
        chi2 = row.get("chi2_CMB")
        S8 = row.get("S8")

        if not isinstance(chi2, (int, float)) or not isinstance(S8, (int, float)):
            row["control_score"] = 9999.0
            row["diagnostic"] = "failed_or_unparsed"
            continue

        if row.get("anchor") == "cdm_control":
            row["control_score"] = chi2
            row["diagnostic"] = "cdm_control"
            continue

        delta_best = row.get("delta_chi2_CMB_vs_best_cdm_control")

        if not isinstance(delta_best, (int, float)):
            row["control_score"] = 9999.0
            row["diagnostic"] = "missing_best_cdm_delta"
            continue

        row["control_score"] = delta_best + 80.0 * abs(S8 - 0.830)

        if delta_best <= 0 and S8 <= 0.833:
            row["diagnostic"] = "beats_best_cdm_control_and_lowers_s8"
        elif delta_best <= 3 and S8 <= 0.833:
            row["diagnostic"] = "near_best_cdm_control_and_lowers_s8"
        elif delta_best <= 8 and S8 <= 0.833:
            row["diagnostic"] = "survives_but_control_penalty"
        elif S8 <= 0.833:
            row["diagnostic"] = "lowers_s8_but_loses_to_cdm_control"
        else:
            row["diagnostic"] = "weak_s8_or_loses_to_cdm_control"


def write_outputs():
    add_deltas_and_rank()

    sorted_rows = sorted(
        REPORT["cases"],
        key=lambda row: (
            row.get("anchor") == "cdm_control",
            row.get("control_score", 9999.0),
        ),
    )

    REPORT["ranked_cases"] = sorted_rows
    REPORT["best_non_cdm_case"] = next(
        (row for row in sorted_rows if row.get("anchor") != "cdm_control"),
        None,
    )

    report_path = OUTDIR / "planck_relief_control_scan_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))

    header = [
        "rank",
        "case",
        "anchor",
        "diagnostic",
        "control_score",
        "status",
        "returncode",
        "warm_fraction",
        "m_ncdm_eV",
        "A_s_scale",
        "n_s_shift",
        "neutral_scale",
        "sigma8",
        "S8",
        "chi2_CMB",
        "delta_chi2_CMB_vs_base_cdm",
        "delta_chi2_CMB_vs_best_cdm_control",
        "chi2_highl_plik_TTTEEE",
        "delta_chi2_highl_plik_TTTEEE_vs_best_cdm_control",
        "chi2_lensing",
        "delta_chi2_lensing_vs_best_cdm_control",
        "chi2_lowl_TT",
        "chi2_lowl_EE",
        "minuslogpost",
        "minuslogprior",
    ]

    with open(OUTDIR / "planck_relief_control_scan_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rank, row in enumerate(sorted_rows, start=1):
            writer.writerow(
                [
                    rank,
                    row.get("name"),
                    row.get("anchor"),
                    row.get("diagnostic"),
                    row.get("control_score"),
                    row.get("status"),
                    row.get("returncode"),
                    row.get("warm_fraction"),
                    row.get("m_ncdm_eV"),
                    row.get("A_s_scale"),
                    row.get("n_s_shift"),
                    row.get("neutral_scale"),
                    row.get("sigma8"),
                    row.get("S8"),
                    row.get("chi2_CMB"),
                    row.get("delta_chi2_CMB_vs_base_cdm"),
                    row.get("delta_chi2_CMB_vs_best_cdm_control"),
                    row.get("chi2_highl_plik_TTTEEE"),
                    row.get("delta_chi2_highl_plik_TTTEEE_vs_best_cdm_control"),
                    row.get("chi2_lensing"),
                    row.get("delta_chi2_lensing_vs_best_cdm_control"),
                    row.get("chi2_lowl_TT"),
                    row.get("chi2_lowl_EE"),
                    row.get("minuslogpost"),
                    row.get("minuslogprior"),
                ]
            )

    with open(OUTDIR / "planck_relief_control_scan_summary.txt", "w") as f:
        f.write("TAIRID Planck matched relief control scan\n")
        f.write("\n")
        f.write("Boundary: matched fixed-nuisance control scan only. Not MCMC and not proof.\n")
        f.write("\n")

        best_cdm = REPORT.get("best_cdm_control")
        if best_cdm:
            f.write("Best CDM control:\n")
            f.write(
                f"{best_cdm.get('name')} S8={best_cdm.get('S8')} "
                f"chi2_CMB={best_cdm.get('chi2_CMB')} "
                f"As_scale={best_cdm.get('A_s_scale')} "
                f"ns_shift={best_cdm.get('n_s_shift')} "
                f"neutral_scale={best_cdm.get('neutral_scale')}\n"
            )
            f.write("\n")

        f.write("Top non-CDM ranked candidates:\n")
        non_cdm = [row for row in sorted_rows if row.get("anchor") != "cdm_control"]

        for rank, row in enumerate(non_cdm[:10], start=1):
            f.write(
                f"{rank}. {row.get('name')} "
                f"S8={row.get('S8')} "
                f"chi2_CMB={row.get('chi2_CMB')} "
                f"delta_vs_best_cdm={row.get('delta_chi2_CMB_vs_best_cdm_control')} "
                f"As_scale={row.get('A_s_scale')} "
                f"ns_shift={row.get('n_s_shift')} "
                f"neutral_scale={row.get('neutral_scale')} "
                f"diagnostic={row.get('diagnostic')}\n"
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
        print("Evaluating control candidate:", case["name"])

        try:
            result = evaluate_candidate(case, fixed_nuisance)
        except Exception as exc:
            result = {
                "name": case["name"],
                "anchor": case.get("anchor"),
                "warm_fraction": case["warm_fraction"],
                "m_ncdm_eV": case["m_ncdm_eV"],
                "A_s_scale": case.get("A_s_scale", 1.0),
                "n_s_shift": case.get("n_s_shift", 0.0),
                "neutral_scale": case.get("neutral_scale", 1.0),
                "status": "failed_exception",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        REPORT["cases"].append(result)

    write_outputs()

    print("")
    print("TAIRID Planck relief control scan complete.")
    print("Created:")
    print("  planck_relief_control_scan_outputs/planck_relief_control_scan_report.json")
    print("  planck_relief_control_scan_outputs/planck_relief_control_scan_summary.csv")
    print("  planck_relief_control_scan_outputs/planck_relief_control_scan_summary.txt")
    print("  planck_relief_control_scan_outputs/fixed_nuisance_values_from_seed_cdm.json")
    print("")
    print("Boundary:")
    print("  This is not a full Planck fit.")
    print("  This is the matched control test to see whether the relief belongs to TAIRID or to ordinary parameter movement.")


if __name__ == "__main__":
    main()
