#!/usr/bin/env python
"""Resume already-trained runs from their last checkpoint and extend the horizon.

Unlike ``scripts/train_campaign.py`` -- which starts *fresh* runs, each auto-
versioned into a new ``checkpoints/v<version>/`` folder -- this script continues
*existing* runs in place: it points the project version at each target run,
resumes from that run's latest checkpoint, and trains on to a larger
``total_iters`` (checkpoints keep landing in the same ``v<version>/`` folder).

Why it can't be a `train_campaign` sweep: that framework treats ``train.resume``
as non-sweepable and always bumps to a new version, so a "resume" run would
write a new folder from scratch. Here we instead:

  1. Read the run's pinned ``checkpoints/v<version>/config.json`` and replay it as
     ``--set`` overrides, so the continuation uses the *exact* hyper-parameters
     that produced the checkpoint (drifted `config.py` defaults can't leak in) --
     only ``train.total_iters`` is raised to the new horizon.
  2. Point ``pyproject.toml`` at that version (what ``get_version()`` reads, which
     decides the ``v<version>/`` checkpoint folder and picks the resume target),
     run ``deepnash-train-async --resume auto``, then restore pyproject.

The async trainer counts ``total_iters`` in LEARNER STEPS and resumes the loop
counter from the checkpoint's step, so a run saved at step 80k with
``--total-iters 240000`` trains 160k more steps, saving every
``checkpoint_every`` as usual.

    uv run python scripts/resume_campaign.py --dry-run
    uv run python scripts/resume_campaign.py            # resume all defaults to 240k
    uv run python scripts/resume_campaign.py v0.43.0 --to 300000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Reuse the campaign's config-flattening / version-writing machinery so the two
# scripts stay in lock-step on what "a sweepable param" is and how pyproject is
# edited.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_campaign import (  # noqa: E402
    CHECKPOINTS,
    PYPROJECT,
    all_params,
    fmt_set_value,
    write_version,
)

from deepnash_rbc.checkpoints import find_latest_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Runs to extend, and the horizon to extend them to (learner steps). These are
# the four temporal nets picked out of the internal-Elo ladder as still climbing
# at 80k -- one GRU, one LSTM, two Transformers.
DEFAULT_VERSIONS = ["v0.43.0", "v0.39.0", "v0.40.0", "v0.42.0"]
DEFAULT_TOTAL_ITERS = 240_000

# Default for --ignore-idle / --honour-idle (mirrors train_campaign): True runs
# back-to-back ignoring the idle window.
IGNORE_IDLE = True


def config_overrides(version: str, total_iters: int) -> list[str]:
    """`--set` args replaying a version's saved config, with the new horizon.

    Reads ``checkpoints/v<version>/config.json`` and emits every *sweepable*
    param as ``--set section.field=value`` (non-sweepable/derived fields -- action
    counts, frame channels, metrics/checkpoint paths -- are skipped), then forces
    ``train.total_iters`` to the requested horizon.
    """
    cfg_path = CHECKPOINTS / version / "config.json"
    if not cfg_path.exists():
        sys.exit(f"[resume] no config.json for {version} at {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    valid = set(all_params())
    flat = {
        f"{section}.{field}": value
        for section, sub in cfg.items()
        for field, value in sub.items()
        if f"{section}.{field}" in valid
    }
    flat["train.total_iters"] = total_iters  # the one thing we change

    args: list[str] = []
    for key, value in flat.items():
        args += ["--set", f"{key}={fmt_set_value(value)}"]
    return args


def resume_step(version: str) -> int | None:
    """Learner step of the checkpoint ``--resume auto`` would pick, or None."""
    latest = find_latest_checkpoint(str(CHECKPOINTS), version=version[1:])
    if latest is None:
        return None
    stem = Path(latest).stem  # deepnash_async_v0.43.0_80000
    return int(stem.rsplit("_", 1)[1])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resume existing runs and extend their training horizon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "versions", nargs="*", default=DEFAULT_VERSIONS,
        help=f"version dirs to resume (default: {' '.join(DEFAULT_VERSIONS)})",
    )
    ap.add_argument(
        "--to", "--total-iters", dest="total_iters", type=int,
        default=DEFAULT_TOTAL_ITERS,
        help=f"learner-step horizon to train to (default: {DEFAULT_TOTAL_ITERS})",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, run nothing")
    idle = ap.add_mutually_exclusive_group()
    idle.add_argument("--ignore-idle", dest="ignore_idle", action="store_true",
                      default=IGNORE_IDLE,
                      help="run back-to-back 24/7, ignoring the idle schedule")
    idle.add_argument("--honour-idle", dest="ignore_idle", action="store_false",
                      help="pause during the quiet hours configured in config")
    args = ap.parse_args()

    # Validate targets and their resume points up front.
    plan = []
    for version in args.versions:
        step = resume_step(version)
        if step is None:
            sys.exit(f"[resume] {version}: no checkpoint to resume from in "
                     f"{CHECKPOINTS / version}")
        if step >= args.total_iters:
            print(f"[resume] {version}: already at step {step:,} >= horizon "
                  f"{args.total_iters:,}; skipping")
            continue
        plan.append((version, step))

    if not plan:
        sys.exit("[resume] nothing to do (every target already past the horizon)")

    print(f"[resume] {len(plan)} run(s) -> total_iters {args.total_iters:,}:")
    for version, step in plan:
        print(f"  {version}: resume @ {step:,} -> {args.total_iters:,} "
              f"(+{args.total_iters - step:,} steps)")
    if args.dry_run:
        return

    env = os.environ.copy()
    if args.ignore_idle:
        env["DEEPNASH_IGNORE_IDLE"] = "1"
    print(f"[resume] idle schedule: "
          f"{'ignored (24/7)' if args.ignore_idle else 'honoured'}")

    original_version = PYPROJECT.read_text()  # restore verbatim when done
    try:
        for i, (version, step) in enumerate(plan, 1):
            write_version(version[1:])  # get_version() -> this version's folder
            overrides = config_overrides(version, args.total_iters)
            cmd = ["uv", "run", "deepnash-train-async", "--resume", "auto",
                   *overrides]
            print(f"\n[resume] === run {i}/{len(plan)}  {version} "
                  f"(@ {step:,} -> {args.total_iters:,}) ===")
            result = subprocess.run(cmd, cwd=ROOT, env=env)
            if result.returncode != 0:
                print(f"[resume] {version} exited {result.returncode}; stopping.")
                sys.exit(result.returncode)
    finally:
        PYPROJECT.write_text(original_version)
        print("[resume] restored pyproject.toml version")

    print("\n[resume] all runs complete.")


if __name__ == "__main__":
    main()
