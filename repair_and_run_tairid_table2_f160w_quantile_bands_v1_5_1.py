#!/usr/bin/env python3
"""
TAIRID Table2 F160W quantile bands v1.5.1 repair runner.

Purpose:
The v1.5 science script failed because the end of the file appears to contain
a pasted/glued fragment:

    main()import json

That is a syntax/paste error, not a failed TAIRID or SH0ES result.

This repair runner:
1. Reads the existing v1.5 script.
2. Keeps everything before the first if __name__ == "__main__": block.
3. Writes a clean runtime copy ending with:
       if __name__ == "__main__":
           main()
4. Compiles the cleaned copy to catch syntax errors.
5. Runs the cleaned copy.
"""

import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


SRC = Path("run_tairid_table2_f160w_quantile_bands_v1_5.py")
DST = Path("run_tairid_table2_f160w_quantile_bands_v1_5_cleaned_runtime.py")

REPAIR_OUTDIR = Path("tairid_table2_f160w_quantile_bands_v1_5_1_repair_outputs")
REPAIR_OUTDIR.mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    started = datetime.now(timezone.utc).isoformat()

    summary = {
        "repair_name": "TAIRID Table2 F160W quantile bands v1.5.1 repair runner",
        "started_utc": started,
        "source_file": str(SRC),
        "cleaned_runtime_file": str(DST),
        "status": "started",
        "boundary": [
            "This repairs a syntax/paste error only.",
            "This does not change the scientific logic before the __main__ block.",
            "This does not prove TAIRID or H0 resolution.",
        ],
    }

    try:
        if not SRC.exists():
            raise FileNotFoundError(f"Missing source script: {SRC}")

        text = SRC.read_text(encoding="utf-8", errors="replace")
        summary["source_size_chars"] = len(text)
        summary["source_contains_bad_glue_main_import_json"] = "main()import json" in text

        marker_double = 'if __name__ == "__main__":'
        marker_single = "if __name__ == '__main__':"

        idx_double = text.find(marker_double)
        idx_single = text.find(marker_single)

        indexes = [i for i in [idx_double, idx_single] if i >= 0]

        if not indexes:
            raise RuntimeError(
                "Could not find an if __name__ == '__main__' block to repair from."
            )

        idx = min(indexes)
        kept = text[:idx].rstrip()

        cleaned = kept + '\n\nif __name__ == "__main__":\n    main()\n'

        DST.write_text(cleaned, encoding="utf-8")

        summary["cleaned_size_chars"] = len(cleaned)
        summary["repair_method"] = (
            "Truncated source at the first __main__ block and appended a clean main() footer."
        )

        # Compile before running so syntax errors are reported clearly.
        compile(cleaned, str(DST), "exec")
        summary["compile_status"] = "success"

        write_json(REPAIR_OUTDIR / "repair_summary_before_run.json", summary)

        result = subprocess.run(
            [sys.executable, str(DST)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        summary["runtime_returncode"] = result.returncode
        summary["runtime_stdout_tail"] = result.stdout[-6000:]
        summary["runtime_stderr_tail"] = result.stderr[-6000:]

        (REPAIR_OUTDIR / "cleaned_runtime_stdout.txt").write_text(
            result.stdout,
            encoding="utf-8",
        )
        (REPAIR_OUTDIR / "cleaned_runtime_stderr.txt").write_text(
            result.stderr,
            encoding="utf-8",
        )

        if result.returncode != 0:
            summary["status"] = "cleaned_runtime_failed"
            write_json(REPAIR_OUTDIR / "repair_summary_final.json", summary)
            raise SystemExit(result.returncode)

        summary["status"] = "success"
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(REPAIR_OUTDIR / "repair_summary_final.json", summary)

        print("TAIRID v1.5.1 repair runner complete.")
        print("Cleaned runtime executed successfully.")
        print("Created:")
        print("  run_tairid_table2_f160w_quantile_bands_v1_5_cleaned_runtime.py")
        print("  tairid_table2_f160w_quantile_bands_v1_5_1_repair_outputs/repair_summary_final.json")

    except Exception as exc:
        summary["status"] = "repair_failed"
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(REPAIR_OUTDIR / "repair_summary_final.json", summary)
        print(summary["traceback"])
        raise


if __name__ == "__main__":
    main()
