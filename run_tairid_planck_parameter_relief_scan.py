#!/usr/bin/env python3
"""
TAIRID Planck fixed-nuisance standard-parameter relief scan.

Purpose:
The fixed-nuisance Planck refinement showed that the TAIRID proxy corridor
survives evaluation but does not currently beat fixed CDM. The penalty mainly
comes from high-l Plik TTTEEE.

This scan keeps Planck nuisance/calibration values fixed from a CDM seed run,
but lets nearby standard spectrum-shape parameters move slightly:

- A_s scale
- n_s shift
- neutral matter density scale

Goal:
Check whether the high-l Planck penalty relaxes near the surviving TAIRID
corridor when ordinary cosmology parameters are allowed a small compensation.

Boundary:
This is not a full Planck fit.
This is not MCMC.
This does not optimize nuisance parameters.
This does not prove TAIRID cosmology.
This is a limited fixed-nuisance relief scan.
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


OUTDIR = Path("planck_parameter_relief_scan_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = OUTDIR / "candidate_yamls"
RUN_DIR.mkdir(parents=True, exist_ok=True)

PACKAGES_DIR = Path("planck_parameter_relief_packages")
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

ANCHORS = [
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

RELIEF_CASES = []

RELIEF_CASES.append(
    {
        "name": "cdm_baseline",
        "anchor": "cdm",
        "warm_fraction": 0.0,
        "m_ncdm_eV": None,
        "A_s_scale": 1.0,
        "n_s_shift": 0.0,
        "neutral_scale": 1.0,
    }
)

for anchor in ANCHORS:
    wf = anchor["warm_fraction"]
    mass = anchor["m_ncdm_eV"]
    anchor_name = anchor["anchor"]

    trial_specs = [
        ("base", 1.000, 0.0000, 1.000),
        ("As_down_1pct", 0.990, 0.0000, 1.000),
        ("As_up_1pct", 1.010, 0.0000, 1.000),
        ("ns_down_0p003", 1.000, -0.0030, 1.000),
        ("ns_up_0p003", 1.000, 0.0030, 1.000),
        ("neutral_down_1pct", 1.000, 0.0000, 0.990),
        ("neutral_up_1pct", 1.000, 0.0000, 1.010),
        ("combo_soft", 0.990, -0.0030, 0.990),
        ("combo_blue_lowmatter", 0.990, 0.0030, 0.990),
        ("combo_highAs_lowmatter", 1.010, -0.0030, 0.990),
    ]

    for label, A_s_scale, n_s_shift, neutral_scale in trial_specs:
        pct = str(wf * 100.0).replace(".", "p")
        mass_text = str(mass).replace(".", "p")
        name = f"{anchor_name}_{pct}pct_{mass_text}eV_{label}"

        RELIEF_CASES.append(
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
    "boundary": "Limited fixed-nuisance standard-parameter relief scan. Not MCMC and not proof.",
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
    seed_case = RELIEF_CASES[0]
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
        "evaluate_relief_" + case["name"],
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

    for row in REPORT["cases"]:
        if row.get("name") == "cdm_baseline":
            row["relief_score"] = 999.0
            row["diagnostic"] = "baseline"
            continue

        delta_cmb = row.get("delta_chi2_CMB_vs_cdm")
        S8 = row.get("S8")

        if not isinstance(delta_cmb, (int, float)) or not isinstance(S8, (int, float)):
            row["relief_score"] = 9999.0
            row["diagnostic"] = "failed_or_unparsed"
            continue

        score = delta_cmb + 80.0 * abs(S8 - 0.830)
        row["relief_score"] = score

        if delta_cmb <= 0 and S8 <= 0.833:
            row["diagnostic"] = "beats_cdm_and_lowers_s8"
        elif delta_cmb <= 3 and S8 <= 0.833:
            row["diagnostic"] = "near_cdm_and_lowers_s8"
        elif delta_cmb <= 8 and S8 <= 0.833:
            row["diagnostic"] = "survives_but_planck_penalty"
        elif S8 <= 0.833:
            row["diagnostic"] = "lowers_s8_but_planck_penalty"
        else:
            row["diagnostic"] = "weak_s8_or_disfavored"


def write_outputs():
    add_deltas_and_rank()

    sorted_rows = sorted(
        REPORT["cases"],
        key=lambda row: (
            row.get("name") == "cdm_baseline",
            row.get("relief_score", 9999.0),
        ),
    )

    REPORT["ranked_cases"] = sorted_rows
    REPORT["best_non_cdm_case"] = next(
        (row for row in sorted_rows if row.get("name") != "cdm_baseline"),
        None,
    )

    report_path = OUTDIR / "planck_parameter_relief_scan_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))

    header = [
        "rank",
        "case",
        "anchor",
        "diagnostic",
        "relief_score",
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

    with open(OUTDIR / "planck_parameter_relief_scan_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rank, row in enumerate(sorted_rows, start=1):
            writer.writerow(
                [
                    rank,
                    row.get("name"),
                    row.get("anchor"),
                    row.get("diagnostic"),
                    row.get("relief_score"),
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

    with open(OUTDIR / "planck_parameter_relief_scan_summary.txt", "w") as f:
        f.write("TAIRID Planck fixed-nuisance standard-parameter relief scan\n")
        f.write("\n")
        f.write("Boundary: limited fixed-nuisance relief scan only. Not MCMC and not proof.\n")
        f.write("\n")
        f.write("Seed CDM chi2_CMB: ")
        f.write(str(REPORT["seed_run"].get("parsed_chi2_CMB")) + "\n")
        f.write("Fixed nuisance count: ")
        f.write(str(len(REPORT["fixed_nuisance_values"])) + "\n")
        f.write("\n")
        f.write("Top ranked candidates:\n")

        for rank, row in enumerate(sorted_rows[:10], start=1):
            f.write(
                f"{rank}. {row.get('name')} "
                f"S8={row.get('S8')} "
                f"delta_CMB={row.get('delta_chi2_CMB_vs_cdm')} "
                f"highl_delta={row.get('delta_chi2_highl_plik_TTTEEE_vs_cdm')} "
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

    for case in RELIEF_CASES:
        print("")
        print("Evaluating relief candidate:", case["name"])

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
    print("TAIRID Planck parameter relief scan complete.")
    print("Created:")
    print("  planck_parameter_relief_scan_outputs/planck_parameter_relief_scan_report.json")
    print("  planck_parameter_relief_scan_outputs/planck_parameter_relief_scan_summary.csv")
    print("  planck_parameter_relief_scan_outputs/planck_parameter_relief_scan_summary.txt")
    print("  planck_parameter_relief_scan_outputs/fixed_nuisance_values_from_seed_cdm.json")
    print("")
    print("Boundary:")
    print("  This is not a full Planck fit.")
    print("  This is a small fixed-nuisance standard-parameter relief scan.")


if __name__ == "__main__":
    main()
