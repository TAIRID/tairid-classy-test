#!/usr/bin/env python3
"""
TAIRID Planck Pareto overlap checkpoint.

Purpose:
The matched relief control scan showed that ordinary CDM parameter relief can
explain much of the apparent improvement. This checkpoint asks the cleaner
Pareto questions:

1. Among cases with S8 <= 0.833, what is the smallest Planck penalty versus
   the best matched CDM control?

2. Among cases close to the best CDM control, how low can S8 get?

3. Is there any overlap zone where Planck closeness and S8 reduction are both
   acceptable?

Boundary:
This is not a full Planck fit.
This is not MCMC.
This does not optimize nuisance parameters.
This does not prove TAIRID cosmology.
It is a fixed-nuisance Pareto checkpoint.
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


OUTDIR = Path("planck_pareto_overlap_checkpoint_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = OUTDIR / "candidate_yamls"
RUN_DIR.mkdir(parents=True, exist_ok=True)

PACKAGES_DIR = Path("planck_pareto_overlap_packages")
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
    {"anchor": "cdm_control", "warm_fraction": 0.0, "m_ncdm_eV": None},
    {"anchor": "best_planck_penalty", "warm_fraction": 0.0125, "m_ncdm_eV": 17.5},
    {"anchor": "strict_s8_tradeoff", "warm_fraction": 0.015, "m_ncdm_eV": 15.0},
    {"anchor": "near_tradeoff", "warm_fraction": 0.0175, "m_ncdm_eV": 17.5},
    {"anchor": "target_s8_probe", "warm_fraction": 0.020, "m_ncdm_eV": 17.5},
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
    "boundary": "Fixed-nuisance Pareto checkpoint only. Not MCMC and not proof.",
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
    "pareto_questions": {},
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
        "evaluate_pareto_" + case["name"],
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


def compute_pareto_front(rows):
    good = [
        row for row in rows
        if row.get("anchor") != "cdm_control"
        and isinstance(row.get("S8"), (int, float))
        and isinstance(row.get("delta_chi2_CMB_vs_best_cdm_control"), (int, float))
    ]

    front = []

    for row in good:
        dominated = False

        for other in good:
            if other is row:
                continue

            other_better_or_equal_s8 = other["S8"] <= row["S8"]
            other_better_or_equal_chi2 = other["delta_chi2_CMB_vs_best_cdm_control"] <= row["delta_chi2_CMB_vs_best_cdm_control"]

            other_strictly_better = (
                other["S8"] < row["S8"]
                or other["delta_chi2_CMB_vs_best_cdm_control"] < row["delta_chi2_CMB_vs_best_cdm_control"]
            )

            if other_better_or_equal_s8 and other_better_or_equal_chi2 and other_strictly_better:
                dominated = True
                break

        if not dominated:
            front.append(row)

    return sorted(front, key=lambda row: (row["S8"], row["delta_chi2_CMB_vs_best_cdm_control"]))


def add_control_deltas_and_pareto():
    cdm_rows = [
        row for row in REPORT["cases"]
        if row.get("anchor") == "cdm_control"
        and isinstance(row.get("chi2_CMB"), (int, float))
    ]

    best_cdm = min(cdm_rows, key=lambda row: row["chi2_CMB"]) if cdm_rows else None

    REPORT["best_cdm_control"] = best_cdm

    if best_cdm:
        for row in REPORT["cases"]:
            for key in ["chi2_CMB", "chi2_highl_plik_TTTEEE", "chi2_lensing", "chi2_lowl_TT", "chi2_lowl_EE"]:
                base_value = best_cdm.get(key)
                value = row.get(key)

                if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                    row["delta_" + key + "_vs_best_cdm_control"] = value - base_value
                else:
                    row["delta_" + key + "_vs_best_cdm_control"] = None

    for row in REPORT["cases"]:
        delta = row.get("delta_chi2_CMB_vs_best_cdm_control")
        S8 = row.get("S8")

        if row.get("anchor") == "cdm_control":
            row["diagnostic"] = "cdm_control"
        elif isinstance(delta, (int, float)) and isinstance(S8, (int, float)):
            if S8 <= 0.833 and delta <= 3:
                row["diagnostic"] = "overlap_zone"
            elif S8 <= 0.833:
                row["diagnostic"] = "s8_zone_planck_penalty"
            elif delta <= 3:
                row["diagnostic"] = "planck_close_weak_s8"
            else:
                row["diagnostic"] = "outside_overlap"
        else:
            row["diagnostic"] = "failed_or_unparsed"

    non_cdm_good = [
        row for row in REPORT["cases"]
        if row.get("anchor") != "cdm_control"
        and isinstance(row.get("S8"), (int, float))
        and isinstance(row.get("delta_chi2_CMB_vs_best_cdm_control"), (int, float))
    ]

    s8_zone = [row for row in non_cdm_good if row["S8"] <= 0.833]
    planck_close_zone = [row for row in non_cdm_good if row["delta_chi2_CMB_vs_best_cdm_control"] <= 3]
    loose_planck_zone = [row for row in non_cdm_good if row["delta_chi2_CMB_vs_best_cdm_control"] <= 8]
    overlap_zone = [row for row in non_cdm_good if row["S8"] <= 0.833 and row["delta_chi2_CMB_vs_best_cdm_control"] <= 3]

    REPORT["pareto_questions"] = {
        "best_cdm_control": best_cdm,
        "best_planck_penalty_among_S8_le_0p833": min(s8_zone, key=lambda row: row["delta_chi2_CMB_vs_best_cdm_control"]) if s8_zone else None,
        "lowest_S8_among_delta_chi2_le_3": min(planck_close_zone, key=lambda row: row["S8"]) if planck_close_zone else None,
        "lowest_S8_among_delta_chi2_le_8": min(loose_planck_zone, key=lambda row: row["S8"]) if loose_planck_zone else None,
        "overlap_zone_count_S8_le_0p833_and_delta_le_3": len(overlap_zone),
        "pareto_front": compute_pareto_front(REPORT["cases"]),
    }


def write_outputs():
    add_control_deltas_and_pareto()

    rows = sorted(
        REPORT["cases"],
        key=lambda row: (
            row.get("anchor") == "cdm_control",
            row.get("S8") if isinstance(row.get("S8"), (int, float)) else 999.0,
            row.get("delta_chi2_CMB_vs_best_cdm_control") if isinstance(row.get("delta_chi2_CMB_vs_best_cdm_control"), (int, float)) else 9999.0,
        ),
    )

    REPORT["ranked_by_s8_then_planck_penalty"] = rows

    report_path = OUTDIR / "planck_pareto_overlap_checkpoint_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))

    header = [
        "rank",
        "case",
        "anchor",
        "diagnostic",
        "warm_fraction",
        "m_ncdm_eV",
        "A_s_scale",
        "n_s_shift",
        "neutral_scale",
        "sigma8",
        "S8",
        "chi2_CMB",
        "delta_chi2_CMB_vs_best_cdm_control",
        "chi2_highl_plik_TTTEEE",
        "delta_chi2_highl_plik_TTTEEE_vs_best_cdm_control",
        "chi2_lensing",
        "delta_chi2_lensing_vs_best_cdm_control",
        "chi2_lowl_TT",
        "chi2_lowl_EE",
        "status",
        "returncode",
    ]

    with open(OUTDIR / "planck_pareto_overlap_checkpoint_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    rank,
                    row.get("name"),
                    row.get("anchor"),
                    row.get("diagnostic"),
                    row.get("warm_fraction"),
                    row.get("m_ncdm_eV"),
                    row.get("A_s_scale"),
                    row.get("n_s_shift"),
                    row.get("neutral_scale"),
                    row.get("sigma8"),
                    row.get("S8"),
                    row.get("chi2_CMB"),
                    row.get("delta_chi2_CMB_vs_best_cdm_control"),
                    row.get("chi2_highl_plik_TTTEEE"),
                    row.get("delta_chi2_highl_plik_TTTEEE_vs_best_cdm_control"),
                    row.get("chi2_lensing"),
                    row.get("delta_chi2_lensing_vs_best_cdm_control"),
                    row.get("chi2_lowl_TT"),
                    row.get("chi2_lowl_EE"),
                    row.get("status"),
                    row.get("returncode"),
                ]
            )

    pareto = REPORT["pareto_questions"]

    with open(OUTDIR / "planck_pareto_overlap_checkpoint_summary.txt", "w") as f:
        f.write("TAIRID Planck Pareto overlap checkpoint\n")
        f.write("\n")
        f.write("Boundary: fixed-nuisance Pareto checkpoint only. Not MCMC and not proof.\n")
        f.write("\n")

        best_cdm = pareto.get("best_cdm_control")
        f.write("Best CDM control:\n")
        f.write(json.dumps(best_cdm, indent=2) + "\n\n")

        f.write("Best Planck penalty among S8 <= 0.833:\n")
        f.write(json.dumps(pareto.get("best_planck_penalty_among_S8_le_0p833"), indent=2) + "\n\n")

        f.write("Lowest S8 among delta chi2 <= 3:\n")
        f.write(json.dumps(pareto.get("lowest_S8_among_delta_chi2_le_3"), indent=2) + "\n\n")

        f.write("Lowest S8 among delta chi2 <= 8:\n")
        f.write(json.dumps(pareto.get("lowest_S8_among_delta_chi2_le_8"), indent=2) + "\n\n")

        f.write("Overlap zone count, S8 <= 0.833 and delta chi2 <= 3:\n")
        f.write(str(pareto.get("overlap_zone_count_S8_le_0p833_and_delta_le_3")) + "\n\n")

        f.write("Pareto front:\n")
        f.write(json.dumps(pareto.get("pareto_front"), indent=2) + "\n")


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
        print("Evaluating Pareto checkpoint candidate:", case["name"])

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
    print("TAIRID Planck Pareto overlap checkpoint complete.")
    print("Created:")
    print("  planck_pareto_overlap_checkpoint_outputs/planck_pareto_overlap_checkpoint_report.json")
    print("  planck_pareto_overlap_checkpoint_outputs/planck_pareto_overlap_checkpoint_summary.csv")
    print("  planck_pareto_overlap_checkpoint_outputs/planck_pareto_overlap_checkpoint_summary.txt")
    print("  planck_pareto_overlap_checkpoint_outputs/fixed_nuisance_values_from_seed_cdm.json")
    print("")
    print("Boundary:")
    print("  This is not a full Planck fit.")
    print("  This is a Pareto checkpoint for Planck closeness versus S8 reduction.")


if __name__ == "__main__":
    main()
