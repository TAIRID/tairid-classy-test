#!/usr/bin/env python3
"""
TAIRID Queue Probe v1.

Purpose:
This is not a science test.
This only checks whether GitHub Actions can queue, start Python, create files,
and upload an artifact.

If this does not run automatically after commit, the problem is GitHub Actions
queue/settings/branch behavior, not the TAIRID Table2 script.
"""

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


OUTDIR = Path("tairid_queue_probe_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    now = datetime.now(timezone.utc).isoformat()

    files = []
    for path in sorted(Path(".").glob("*")):
        files.append(
            {
                "name": str(path),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
            }
        )

    summary = {
        "test_name": "TAIRID Queue Probe v1",
        "status": "success",
        "meaning": "GitHub Actions queued, Python ran, output files were created.",
        "utc_time": now,
        "python_version": sys.version,
        "platform": platform.platform(),
        "github_actions": os.environ.get("GITHUB_ACTIONS"),
        "github_workflow": os.environ.get("GITHUB_WORKFLOW"),
        "github_ref": os.environ.get("GITHUB_REF"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "repository_files_seen": files,
        "next_step_if_success": "Run the Table2 host residual test using a new auto workflow.",
        "next_step_if_failure": "Check repository Actions settings or whether commits are going to the default branch.",
    }

    (OUTDIR / "queue_probe_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (OUTDIR / "queue_probe_summary.txt").write_text(
        "\n".join(
            [
                "TAIRID Queue Probe v1",
                "",
                "STATUS: success",
                "",
                "GitHub Actions queued successfully.",
                "Python ran successfully.",
                "The artifact uploader should attach this output folder.",
                "",
                f"UTC time: {now}",
                f"Python: {sys.version}",
                f"Platform: {platform.platform()}",
                "",
                "Next step:",
                "If this worked, the GitHub queue is okay and we can rebuild the Table2 test with the same clean auto-trigger style.",
            ]
        ),
        encoding="utf-8",
    )

    print("TAIRID Queue Probe v1 complete.")
    print("Created tairid_queue_probe_v1_outputs/queue_probe_summary.json")
    print("Created tairid_queue_probe_v1_outputs/queue_probe_summary.txt")


if __name__ == "__main__":
    main()
