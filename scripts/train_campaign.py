#!/usr/bin/env python
"""Run a sequence of training runs back-to-back, auto-versioning each so every
run lands in its own ``checkpoints/v<version>/`` folder.

Versions are resolved from what already exists on disk, not from pyproject: each
run's version is the LARGEST existing version bumped at the requested level, and
if that lands on a folder that already exists it keeps bumping. Gaps are never
filled -- it always grows past the largest version -- so a campaign can never
clobber a finished run. Versions chosen for earlier runs in the same campaign are
reserved for the later ones.

Each run is launched as a subprocess (``deepnash-train-async``), so it re-reads
the freshly written version from pyproject.toml at startup (that is what
``checkpoints.py`` keys every checkpoint and metrics file on). A run trains to
completion (``total_iters``); the nightly idle schedule (see config) lives inside
the trainer, so a single run already spans multiple nights -- this script just
chains distinct versioned runs, it does not manage the daily on/off cycle.

EDIT ``RUNS`` below: each entry is (bump_level, [extra CLI args]). bump_level is
"major" | "minor" | "patch". The args pass straight through to
deepnash-train-async (see ``--help``).

    uv run python scripts/train_campaign.py            # run the whole campaign
    uv run python scripts/train_campaign.py --dry-run  # print the plan only

Version detection scans the default ``checkpoints/`` dir; if a run overrides
--checkpoint-dir, that folder isn't considered when picking versions.

Note: this rewrites the ``version`` field in pyproject.toml as it goes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from deepnash_rbc.checkpoints import existing_versions, next_free_version
from deepnash_rbc.version import get_version

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CHECKPOINTS = ROOT / "checkpoints"

# --- edit me -----------------------------------------------------------------
# (bump_level, args) per run; args pass straight to deepnash-train-async.
# Example: sweep the observation history length across three versions.
RUNS: list[tuple[str, list[str]]] = [
    ("minor", ["--history", "4"]),
    ("minor", ["--history", "8"]),
    ("minor", ["--history", "16"]),
]
# -----------------------------------------------------------------------------


def write_version(version: str) -> None:
    text = PYPROJECT.read_text()
    # replace the first (project-table) version assignment only
    new, n = re.subn(r'(?m)^version = "[^"]*"', f'version = "{version}"', text, count=1)
    if n != 1:
        raise RuntimeError("could not find a 'version = \"...\"' line in pyproject.toml")
    PYPROJECT.write_text(new)


def plan_versions() -> list[tuple[str, list[str]]]:
    """Resolve each run's version, reserving earlier picks for later runs."""
    reserved = set(existing_versions(str(CHECKPOINTS)))
    base = get_version()  # fallback only when nothing exists on disk yet
    plan = []
    for level, run_args in RUNS:
        version = next_free_version(level, reserved, base)
        reserved.add(version)
        plan.append((version, run_args))
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description="Chain auto-versioned training runs.")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    found = existing_versions(str(CHECKPOINTS))
    plan = plan_versions()
    print(f"[campaign] existing versions: {found or '(none)'}")
    print(f"[campaign] {len(plan)} run(s) planned:")
    for i, (version, run_args) in enumerate(plan, 1):
        print(f"  {i}. v{version}: deepnash-train-async {' '.join(run_args)}")
    if args.dry_run:
        return

    for i, (version, run_args) in enumerate(plan, 1):
        write_version(version)  # what get_version() reads in the subprocess
        cmd = ["uv", "run", "deepnash-train-async", *run_args]
        print(f"\n[campaign] === run {i}/{len(plan)}  v{version} ===\n[campaign] $ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"[campaign] run {i} (v{version}) exited {result.returncode}; stopping.")
            sys.exit(result.returncode)
    print("\n[campaign] all runs complete.")


if __name__ == "__main__":
    main()
