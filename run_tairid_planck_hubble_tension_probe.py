#!/usr/bin/env python3
"""
TAIRID Planck Hubble tension probe.

Purpose:
The previous branch mostly tested S8 relief while holding H0 near the Planck-side
value. This test asks a different question:

Can CDM or a TAIRID proxy move toward local-H0 / SH0ES-like values without
taking too much Planck penalty?

Method:
1. Install the same Planck likelihood bridge.
2. Run a CDM seed evaluate.
3. Extract nuisance/calibration values from the seed.
4. Freeze those nuisance values.
5. Evaluate CDM and TAIRID proxy anchors across several H0 values.
6. Compare:
   - Planck chi-square penalty
   - SH0ES H0 prior penalty
   - combined diagnostic score

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
It is a fixed-nuisance Hubble-tension diagnostic.
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


OUTDIR = Path("planck_hubble_tension_probe_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = OUTDIR / "candidate_yamls"
RUN_DIR.mkdir(parents=True, exist_ok=True)

PACKAGES_DIR = Path("planck_hubble_tension_packages")
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

OMEGA_B = 0.0223700000
OMEGA_NEUTRAL_BASE = 0.1200000000
N_S_BASE = 0.9649
A_S_BASE = 2.100549e-9
TAU_REIO = 0.0544

SH0ES_H0 = 73.04
SH0ES_SIGMA = 1.04

BASE_H0 = 66.89318

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

MODEL_SPECS = [
    {
        "model": "cdm",
        "warm_fraction": 0.0,
        "m_ncdm_eV": None,
    },
    {
        "model": "tairid_best_planck_penalty",
        "warm_fraction": 0.0125,
        "m_ncdm_eV": 17.5,
    },
    {
        "model": "tairid_strict_s8_tradeoff",
        "warm_fraction": 0.015,
        "m_ncdm_eV": 15.0,
    },
    {
        "model": "tairid_near_tradeoff",
        "warm_fraction": 0.0175,
        "m_ncdm_eV": 17.5,
    },
]

H0_VALUES = [
    BASE_H0,
    69.0,
    71.0,
    SH0ES_H0,
]

CASES = []

for model in MODEL_SPECS:
    for H0 in H0_VALUES:
        h0_text = str(H0).replace(".", "p")
        mass = model["m_ncdm_eV"]

        if mass is None:
            mass_text = "none"
        else:
            mass_text = str(mass).replace(".", "p")

        name = f"{model['model']}_H0_{h0_text}_m_{mass_text}"

        CASES.append(
            {
                "name": name,
                "model": model["model"],
                "H0": H0,
                "warm_fraction": model["warm_fraction"],
                "m_ncdm_eV": model["m_ncdm_eV"],
                "A_s_scale": 1.0,
                "n_s_shift": 0.0,
                "neutral_scale": 1.0,
            }
        )

REPORT = {
    "boundary": "Fixed-nuisance Hubble-tension diagnostic only. Not MCMC and not proof.",
    "python": {
        "version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    },
    "packages_dir": str(PACKAGES_DIR),
    "likelihoods": list(LIKELIHOODS.keys()),
    "SH0ES_H0_prior": {
        "H0": SH0ES_H0,
        "sigma": SH0ES_SIGMA,
    },
    "install_commands": [],
    "seed_run": {},
    "fixed_nuisance_values": {},
    "cases": [],
    "hubble_questions": {},
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
        entry["stdout_tail"] = proc.stdout[-10000:]
        entry["stderr_tail"] = proc.stderr[-10000:]

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


def omega_lambda_for_flatish(H0, omega_neutral):
    h = H0 / 100.0
    omega_m = OMEGA_B + omega_neutral
    return 1.0 - (omega_m / (h * h))


def make_candidate_info(case, fixed_nuisance=None, output_name=None):
    H0 = float(case["H0"])
    h = H0 / 100.0

    warm_fraction = float(case["warm_fraction"])
    m_ncdm_eV = case["m_ncdm_eV"]

    A_s_scale = float(case.get("A_s_scale", 1.0))
    n_s_shift = float(case.get("n_s_shift", 0.0))
    neutral_scale = float(case.get("neutral_scale", 1.0))

    omega_neutral = OMEGA_NEUTRAL_BASE * neutral_scale

    omega_cdm = omega_neutral
    extra_args = dict(BASE_CLASS_EXTRA)
    extra_args["Omega_Lambda"] = omega_lambda_for_flatish(H0, omega_neutral)

    if warm_fraction > 0.0:
        omega_warm = omega_neutral * warm_fraction
        omega_cold = omega_neutral * (1.0 - warm_fraction)

        omega_cdm = omega_cold

        extra_args["N_ncdm"] = 1
        extra_args["m_ncdm"] = float(m_ncdm_eV)
        extra_args["Omega_ncdm"] = omega_warm / (h * h)

    params = {
        "H0": H0,
        "omega_b": OMEGA_B,
        "omega_cdm": omega_cdm,
        "n_s": N_S_BASE + n_s_shift,
        "A_s": A_S_BASE * A_s_scale,
        "tau_reio": TAU_REIO,
    }

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

    base_param_names = {
        "H0",
        "omega_b",
        "omega_cdm",
        "n_s",
        "A_s",
        "tau_reio",
    }

    for key, value in parsed.items():
        if key in NON_NUISANCE_COLUMNS:
            continue

        if key.startswith("chi2"):
            continue

        if key in base_param_names:
            continue

        if isinstance(value, float) and math.isfinite(value):
            nuisance[key] = value

    return nuisance


def run_seed_cdm():
    seed_case = {
        "name": "seed_cdm_for_nuisance",
        "model": "seed",
        "H0": BASE_H0,
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

        H0 = float(case["H0"])
        h = H0 / 100.0

        warm_fraction = float(case["warm_fraction"])
        m_ncdm_eV = case["m_ncdm_eV"]

        A_s_scale = float(case.get("A_s_scale", 1.0))
        n_s_shift = float(case.get("n_s_shift", 0.0))
        neutral_scale = float(case.get("neutral_scale", 1.0))

        omega_neutral = OMEGA_NEUTRAL_BASE * neutral_scale

        params = dict(BASE_CLASS_EXTRA)
        params.update(
            {
                "H0": H0,
                "omega_b": OMEGA_B,
                "omega_cdm": omega_neutral,
                "n_s": N_S_BASE + n_s_shift,
                "A_s": A_S_BASE * A_s_scale,
                "tau_reio": TAU_REIO,
                "Omega_Lambda": omega_lambda_for_flatish(H0, omega_neutral),
            }
        )

        if warm_fraction > 0.0:
            omega_warm = omega_neutral * warm_fraction
            omega_cold = omega_neutral * (1.0 - warm_fraction)

            params["omega_cdm"] = omega_cold
            params["N_ncdm"] = 1
            params["m_ncdm"] = float(m_ncdm_eV)
            params["Omega_ncdm"] = omega_warm / (h * h)

        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()

        sigma8 = float(cosmo.sigma8())

        omega_m_physical = OMEGA_B + omega_neutral
        Omega_m = omega_m_physical / (h * h)
        S8 = sigma8 * math.sqrt(Omega_m / 0.3)

        cosmo.struct_cleanup()
        cosmo.empty()

        return sigma8, S8

    except Exception:
        return None, None


def h0_prior_chi2(H0):
    return ((float(H0) - SH0ES_H0) / SH0ES_SIGMA) ** 2


def evaluate_candidate(case, fixed_nuisance):
    info = make_candidate_info(case, fixed_nuisance=fixed_nuisance, output_name=case["name"])
    yaml_path = write_yaml(case["name"], info)

    entry = run_command(
        "evaluate_hubble_" + case["name"],
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
        "model": case["model"],
        "H0": case["H0"],
        "warm_fraction": case["warm_fraction"],
        "m_ncdm_eV": case["m_ncdm_eV"],
        "sigma8": sigma8,
        "S8": S8,
        "H0_prior_chi2_SH0ES": h0_prior_chi2(case["H0"]),
        "yaml_file": str(yaml_path),
        "output_table": str(table_path) if table_path else None,
        "status": entry["status"],
        "returncode": entry["returncode"],
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


def add_deltas_and_questions():
    valid = [
        row for row in REPORT["cases"]
        if isinstance(row.get("chi2_CMB"), (int, float))
    ]

    if not valid:
        return

    best_planck = min(valid, key=lambda row: row["chi2_CMB"])
    REPORT["best_planck_case"] = best_planck

    best_combined = min(
        valid,
        key=lambda row: row["chi2_CMB"] + row["H0_prior_chi2_SH0ES"],
    )
    REPORT["best_combined_planck_plus_SH0ES_prior_case"] = best_combined

    base_chi2 = best_planck["chi2_CMB"]

    for row in REPORT["cases"]:
        chi2 = row.get("chi2_CMB")

        if isinstance(chi2, (int, float)):
            row["delta_chi2_CMB_vs_best_planck_case"] = chi2 - base_chi2
            row["combined_chi2_CMB_plus_SH0ES_prior"] = chi2 + row["H0_prior_chi2_SH0ES"]
        else:
            row["delta_chi2_CMB_vs_best_planck_case"] = None
            row["combined_chi2_CMB_plus_SH0ES_prior"] = None

        delta = row.get("delta_chi2_CMB_vs_best_planck_case")
        h0_prior = row.get("H0_prior_chi2_SH0ES")

        if isinstance(delta, (int, float)) and isinstance(h0_prior, (int, float)):
            if delta <= 3 and h0_prior <= 4:
                row["diagnostic"] = "overlap_zone_planck_close_and_SH0ES_close"
            elif delta <= 8 and h0_prior <= 4:
                row["diagnostic"] = "loose_overlap_zone"
            elif delta <= 3:
                row["diagnostic"] = "planck_close_but_low_H0"
            elif h0_prior <= 4:
                row["diagnostic"] = "SH0ES_close_but_planck_penalty"
            else:
                row["diagnostic"] = "outside_overlap"
        else:
            row["diagnostic"] = "failed_or_unparsed"

    planck_close = [
        row for row in valid
        if row.get("delta_chi2_CMB_vs_best_planck_case") is not None
        and row["delta_chi2_CMB_vs_best_planck_case"] <= 3
    ]

    sh0es_close = [
        row for row in valid
        if row["H0_prior_chi2_SH0ES"] <= 4
    ]

    overlap = [
        row for row in valid
        if row.get("delta_chi2_CMB_vs_best_planck_case") is not None
        and row["delta_chi2_CMB_vs_best_planck_case"] <= 3
        and row["H0_prior_chi2_SH0ES"] <= 4
    ]

    loose_overlap = [
        row for row in valid
        if row.get("delta_chi2_CMB_vs_best_planck_case") is not None
        and row["delta_chi2_CMB_vs_best_planck_case"] <= 8
        and row["H0_prior_chi2_SH0ES"] <= 4
    ]

    REPORT["hubble_questions"] = {
        "best_planck_case": best_planck,
        "best_combined_planck_plus_SH0ES_prior_case": best_combined,
        "highest_H0_among_delta_planck_le_3": max(planck_close, key=lambda row: row["H0"]) if planck_close else None,
        "best_planck_penalty_among_SH0ES_close": min(sh0es_close, key=lambda row: row["delta_chi2_CMB_vs_best_planck_case"]) if sh0es_close else None,
        "overlap_count_delta_planck_le_3_and_SH0ES_prior_le_4": len(overlap),
        "loose_overlap_count_delta_planck_le_8_and_SH0ES_prior_le_4": len(loose_overlap),
    }


def write_outputs():
    add_deltas_and_questions()

    sorted_rows = sorted(
        REPORT["cases"],
        key=lambda row: (
            row.get("combined_chi2_CMB_plus_SH0ES_prior")
            if isinstance(row.get("combined_chi2_CMB_plus_SH0ES_prior"), (int, float))
            else 999999.0
        ),
    )

    REPORT["ranked_by_combined_planck_plus_SH0ES_prior"] = sorted_rows

    report_path = OUTDIR / "planck_hubble_tension_probe_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))

    header = [
        "rank",
        "case",
        "model",
        "diagnostic",
        "H0",
        "H0_prior_chi2_SH0ES",
        "warm_fraction",
        "m_ncdm_eV",
        "sigma8",
        "S8",
        "chi2_CMB",
        "delta_chi2_CMB_vs_best_planck_case",
        "combined_chi2_CMB_plus_SH0ES_prior",
        "chi2_highl_plik_TTTEEE",
        "chi2_lensing",
        "chi2_lowl_TT",
        "chi2_lowl_EE",
        "status",
        "returncode",
    ]

    with open(OUTDIR / "planck_hubble_tension_probe_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rank, row in enumerate(sorted_rows, start=1):
            writer.writerow(
                [
                    rank,
                    row.get("name"),
                    row.get("model"),
                    row.get("diagnostic"),
                    row.get("H0"),
                    row.get("H0_prior_chi2_SH0ES"),
                    row.get("warm_fraction"),
                    row.get("m_ncdm_eV"),
                    row.get("sigma8"),
                    row.get("S8"),
                    row.get("chi2_CMB"),
                    row.get("delta_chi2_CMB_vs_best_planck_case"),
                    row.get("combined_chi2_CMB_plus_SH0ES_prior"),
                    row.get("chi2_highl_plik_TTTEEE"),
                    row.get("chi2_lensing"),
                    row.get("chi2_lowl_TT"),
                    row.get("chi2_lowl_EE"),
                    row.get("status"),
                    row.get("returncode"),
                ]
            )

    with open(OUTDIR / "planck_hubble_tension_probe_summary.txt", "w") as f:
        f.write("TAIRID Planck Hubble tension probe\n")
        f.write("\n")
        f.write("Boundary: fixed-nuisance Hubble diagnostic only. Not MCMC and not proof.\n")
        f.write("\n")
        f.write("SH0ES prior used:\n")
        f.write(f"H0 = {SH0ES_H0}, sigma = {SH0ES_SIGMA}\n\n")

        f.write("Hubble questions:\n")
        f.write(json.dumps(REPORT["hubble_questions"], indent=2) + "\n\n")

        f.write("Top ranked by Planck + SH0ES prior:\n")
        for rank, row in enumerate(sorted_rows[:10], start=1):
            f.write(
                f"{rank}. {row.get('name')} "
                f"H0={row.get('H0')} "
                f"S8={row.get('S8')} "
                f"Planck_delta={row.get('delta_chi2_CMB_vs_best_planck_case')} "
                f"SH0ES_prior={row.get('H0_prior_chi2_SH0ES')} "
                f"combined={row.get('combined_chi2_CMB_plus_SH0ES_prior')} "
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
        print("Evaluating Hubble tension case:", case["name"])

        try:
            result = evaluate_candidate(case, fixed_nuisance)
        except Exception as exc:
            result = {
                "name": case["name"],
                "model": case.get("model"),
                "H0": case["H0"],
                "warm_fraction": case["warm_fraction"],
                "m_ncdm_eV": case["m_ncdm_eV"],
                "status": "failed_exception",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        REPORT["cases"].append(result)

    write_outputs()

    print("")
    print("TAIRID Planck Hubble tension probe complete.")
    print("Created:")
    print("  planck_hubble_tension_probe_outputs/planck_hubble_tension_probe_report.json")
    print("  planck_hubble_tension_probe_outputs/planck_hubble_tension_probe_summary.csv")
    print("  planck_hubble_tension_probe_outputs/planck_hubble_tension_probe_summary.txt")
    print("  planck_hubble_tension_probe_outputs/fixed_nuisance_values_from_seed_cdm.json")
    print("")
    print("Boundary:")
    print("  This is not a full Planck fit.")
    print("  This checks whether the Hubble tension branch has an overlap zone in this fixed-nuisance probe.")


if __name__ == "__main__":
    main()
