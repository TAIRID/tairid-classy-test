#!/usr/bin/env python3
"""
TAIRID Boundary Prediction Battery v0.2.1

Patch-only runner for v0.2.

The v0.2 run failed after the analysis stage during plotting:

    KeyError: 'global_direction'

That was caused by the plot function looking for old short key names.
The actual v0.2 summary rows use:

    global_signed_mean_residual_difference_high_minus_low
    within_host_signed_mean_residual_difference_high_minus_low

This runner does not change the science logic.
It patches only make_plots() and reruns v0.2 into a new output folder.
"""

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_tairid_boundary_polarity_battery_v0_2 as v02


OUTDIR = Path("tairid_boundary_polarity_battery_v0_2_1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = OUTDIR / "downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def value(row, key, default=0.0):
    raw = row.get(key, default)

    try:
        val = float(raw)
    except Exception:
        return default

    if not np.isfinite(val):
        return default

    return val


def patched_make_plots(variable_summary_rows):
    rows = [
        row for row in variable_summary_rows
        if row.get("status") == "ok"
    ]

    if not rows:
        return

    rows = sorted(
        rows,
        key=lambda r: -value(r, "combined_delta_chi2"),
    )

    labels = [str(r.get("label", r.get("column", "unknown"))) for r in rows]
    x = np.arange(len(rows))

    # Plot 1: combined strength.
    plt.figure(figsize=(12, 5))
    plt.bar(x, [value(r, "combined_delta_chi2") for r in rows])
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("combined delta chi2")
    plt.title("TAIRID v0.2.1 combined edge-pair polarity strength")
    plt.tight_layout()
    plt.savefig(OUTDIR / "variable_combined_delta_chi2_v0_2_1.png", dpi=160)
    plt.close()

    # Plot 2: global vs within-host strength.
    width = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(
        x - width / 2,
        [value(r, "global_delta_chi2") for r in rows],
        width,
        label="global",
    )
    plt.bar(
        x + width / 2,
        [value(r, "within_host_delta_chi2") for r in rows],
        width,
        label="within-host",
    )
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("delta chi2")
    plt.title("Global vs within-host edge-pair polarity v0.2.1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "variable_global_vs_within_delta_chi2_v0_2_1.png", dpi=160)
    plt.close()

    # Plot 3: global signed direction.
    plt.figure(figsize=(12, 5))
    plt.bar(
        x,
        [
            value(r, "global_signed_mean_residual_difference_high_minus_low")
            for r in rows
        ],
    )
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("high-minus-low mean residual")
    plt.title("Global signed direction by variable v0.2.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "variable_global_signed_direction_v0_2_1.png", dpi=160)
    plt.close()

    # Plot 4: within-host signed direction.
    plt.figure(figsize=(12, 5))
    plt.bar(
        x,
        [
            value(r, "within_host_signed_mean_residual_difference_high_minus_low")
            for r in rows
        ],
    )
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("within-host high-minus-low mean residual")
    plt.title("Within-host signed direction by variable v0.2.1")
    plt.tight_layout()
    plt.savefig(OUTDIR / "variable_within_host_signed_direction_v0_2_1.png", dpi=160)
    plt.close()


def main():
    patch_summary = {
        "patch_name": "TAIRID Boundary Polarity Battery v0.2.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Fix v0.2 plot-only KeyError: 'global_direction'.",
        "science_logic_changed": False,
        "patched_function": "make_plots",
        "new_output_dir": str(OUTDIR),
        "status": "started",
    }

    write_json(OUTDIR / "v0_2_1_patch_summary_started.json", patch_summary)

    try:
        # Redirect v0.2 output into the v0.2.1 output folder.
        v02.OUTDIR = OUTDIR
        v02.DOWNLOAD_DIR = DOWNLOAD_DIR

        # Also redirect helper module output folders used inside v0.2.
        v02.v16.OUTDIR = OUTDIR
        v02.v16.DOWNLOAD_DIR = DOWNLOAD_DIR
        v02.b01.OUTDIR = OUTDIR
        v02.b01.DOWNLOAD_DIR = DOWNLOAD_DIR

        # Patch only the plotting function.
        v02.make_plots = patched_make_plots

        print("Running TAIRID Boundary Polarity Battery v0.2.1")
        print("Patch: plotting only. Science logic unchanged.")

        v02.main()

        patch_summary["status"] = "success"
        patch_summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(OUTDIR / "v0_2_1_patch_summary_final.json", patch_summary)

        print("TAIRID Boundary Polarity Battery v0.2.1 complete.")
        print("Plot-only bug patched successfully.")

    except Exception as exc:
        patch_summary["status"] = "failed"
        patch_summary["error"] = repr(exc)
        patch_summary["traceback"] = traceback.format_exc()
        patch_summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(OUTDIR / "v0_2_1_patch_summary_final.json", patch_summary)
        print(patch_summary["traceback"])
        raise


if __name__ == "__main__":
    main()
