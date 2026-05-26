#!/usr/bin/env python3
"""
TAIRID Action Pair Probe v1.

This is not a TAIRID science test.

Purpose:
This checks whether the normal two-file pattern still works:

1. A workflow file in .github/workflows/
2. A Python script at the top level of the repository

If this runs, GitHub Actions can still:
- see the workflow,
- start a runner,
- find the Python script,
- execute Python,
- create output files,
- upload an artifact.

If this does not run after the Python file is committed, the issue is not the
TAIRID Table2 test. It is GitHub Actions triggering/settings/branch behavior.
"""

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


OUTDIR = Path("tairid_action_pair_probe_v1_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)


def list_repo_files():
    rows = []
    for path in sorted(Path(".").glob("*")):
        rows.append(
            {
                "name": str(path),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return rows


def read_workflow_folder():
    workflow_dir = Path(".github/workflows")
    rows = []

    if not workflow_dir.exists():
        return [
            {
                "status": "missing",
                "message": ".github/workflows folder was not found by the runner.",
            }
        ]

    for path in sorted(workflow_dir.glob("*")):
        rows.append(
            {
                "name": str(path),
                "is_file": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )

    return rows


def main():
    now = datetime.now(timezone.utc).isoformat()

    env_keys = [
        "GITHUB_ACTIONS",
        "GITHUB_WORKFLOW",
        "GITHUB_WORKSPACE",
        "GITHUB_REPOSITORY",
        "GITHUB_REF",
        "GITHUB_REF_NAME",
        "GITHUB_SHA",
        "GITHUB_EVENT_NAME",
        "RUNNER_OS",
        "RUNNER_ARCH",
    ]

    github_env = {key: os.environ.get(key) for key in env_keys}

    summary = {
        "test_name": "TAIRID Action Pair Probe v1",
        "status": "success",
        "meaning": "GitHub Actions started, found the Python script, ran Python, and created output files.",
        "utc_time": now,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "github_environment": github_env,
        "repo_top_level_files_seen": list_repo_files(),
        "workflow_files_seen": read_workflow_folder(),
        "truth_boundary": [
            "This is not a TAIRID science test.",
            "This does not test cosmology.",
            "This only tests whether the two-file GitHub Actions pattern can run.",
        ],
        "next_step_if_success": "Rebuild the Table2 host residual test using this same two-file pattern.",
        "next_step_if_failure": "Check whether GitHub Actions is disabled, blocked, or whether commits are not landing on the default branch.",
    }

    summary_json = OUTDIR / "action_pair_probe_summary.json"
    summary_txt = OUTDIR / "action_pair_probe_summary.txt"
    repo_files_txt = OUTDIR / "repo_files_seen.txt"
    workflow_files_txt = OUTDIR / "workflow_files_seen.txt"

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_txt.write_text(
        "\n".join(
            [
                "TAIRID Action Pair Probe v1",
                "",
                "STATUS: success",
                "",
                "GitHub Actions started this workflow.",
                "Python ran successfully.",
                "The output folder was created successfully.",
                "",
                f"UTC time: {now}",
                f"Python executable: {sys.executable}",
                f"Platform: {platform.platform()}",
                "",
                "GitHub environment:",
                json.dumps(github_env, indent=2),
                "",
                "Truth boundary:",
                "- This is not a TAIRID science test.",
                "- This only checks GitHub Actions execution.",
                "",
                "Next step if this worked:",
                "Rebuild the Table2 host residual test using this same two-file pattern.",
            ]
        ),
        encoding="utf-8",
    )

    repo_files_txt.write_text(
        "\n".join(json.dumps(row) for row in summary["repo_top_level_files_seen"]),
        encoding="utf-8",
    )

    workflow_files_txt.write_text(
        "\n".join(json.dumps(row) for row in summary["workflow_files_seen"]),
        encoding="utf-8",
    )

    print("TAIRID Action Pair Probe v1 complete.")
    print(f"Created {summary_json}")
    print(f"Created {summary_txt}")
    print(f"Created {repo_files_txt}")
    print(f"Created {workflow_files_txt}")


if __name__ == "__main__":
    main()
