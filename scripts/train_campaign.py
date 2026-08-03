#!/usr/bin/env python
"""Run a sequence of training runs back-to-back, auto-versioning each so every
run lands in its own ``checkpoints/v<version>/`` folder.

Two ways to define the campaign:

1. ``--sweep sweep.json`` (preferred): a manifest of runs with config overrides
   by dotted path (anything ``--set`` accepts). Generate a template listing
   every available parameter with its default via::

       uv run python scripts/train_campaign.py --write-template sweep.json

   Manifest structure::

       {
         "defaults":   {"encoding.history": 16, ...},   # applied to every FRESH run
         "runs": [
           # fresh run: auto-versioned into a new v<version>/ folder
           {"name": "eta-0.3", "bump": "minor", "overrides": {"rnad.eta": 0.3}},
           # resume run: continue an existing folder to a new horizon
           {"name": "extend-gru", "resume": "v0.43.0",
            "overrides": {"train.total_iters": 300000}},
           ...
         ],
         "tournament": {                # after each run (omit to disable)
           "pair_games": 8,
           "baselines": ["random", "trout", "mht"],
           "incumbents": 5,             # top-N existing checkpoints to gauntlet
           "keep_top": 2,               # prune run to its best N checkpoints
           "workers": 8,
           "threshold": 0.05,
           "out": "results/tournament.jsonl"
         }
       }

   Pinning params in "defaults" makes runs reproducible even if config.py
   defaults drift later. Overrides win over defaults. After each run the new
   checkpoints play a gauntlet (vs the baselines + current top incumbents,
   appended to the shared tournament JSONL), and if "keep_top" is set, all but
   the run's best N checkpoints are DELETED to keep storage bounded
   (config.json and metrics are always kept).

   A run with a "resume": "v<version>" key CONTINUES that existing run in place
   instead of starting fresh: it points the project version at that folder,
   resumes from its latest checkpoint (deepnash-train-async --resume auto), and
   trains on to the "train.total_iters" set in the run's "overrides" (each resume
   thus gets its OWN horizon). A resume replays that version's pinned config.json
   as its base -- so its "overrides" carry only what to change, "defaults" and
   "bump" do NOT apply, and architecture keys (network.*/encoding.*) are rejected
   because a folder's checkpoints are shape-locked. Resumes already at/past their
   horizon are skipped.

2. Legacy: edit ``RUNS`` below; each entry is (bump_level, [extra CLI args])
   passed straight to deepnash-train-async.

Versions are resolved from what already exists on disk: each run's version is
the LARGEST existing version bumped at the requested level; gaps are never
filled, so a campaign can never clobber a finished run.

    uv run python scripts/train_campaign.py --sweep sweep.json --dry-run

Note: this rewrites the ``version`` field in pyproject.toml as it goes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path

from deepnash_rbc.checkpoints import (
    existing_versions,
    find_latest_checkpoint,
    next_free_version,
)
from deepnash_rbc.config import Config
from deepnash_rbc.version import get_version

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CHECKPOINTS = ROOT / "checkpoints"

# Default for --ignore-idle / --honour-idle: True runs the whole campaign
# back-to-back ignoring the idle window (exports DEEPNASH_IGNORE_IDLE=1 to every
# run), False honours the schedule in config.
IGNORE_IDLE = True

# --- legacy inline campaign (used when --sweep is not given) -------------------
RUNS: list[tuple[str, list[str]]] = []
# -------------------------------------------------------------------------------

# Derived/bookkeeping fields that make no sense to sweep.
NON_SWEEPABLE = {
    "encoding.frame_channels",   # fixed by the observation encoding
    "network.move_actions",      # fixed by the action encoding
    "network.sense_actions",     # fixed by the action encoding
    "train.metrics_path",        # derived from checkpoint_dir + version
    "train.checkpoint_dir",      # campaign owns the layout
    "train.resume",              # resuming is not a sweep axis
}


def all_params() -> dict:
    """Every sweepable config field as dotted path -> default value."""
    cfg = Config()
    out = {}
    for section in ("encoding", "network", "rnad", "train"):
        obj = getattr(cfg, section)
        for f in fields(obj):
            path = f"{section}.{f.name}"
            if path in NON_SWEEPABLE:
                continue
            v = getattr(obj, f.name)
            out[path] = list(v) if isinstance(v, tuple) else v
    return out


def fmt_set_value(v) -> str:
    """Format a manifest value the way ``--set`` expects it."""
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def set_args(overrides: dict) -> list[str]:
    """Turn a dotted-path override dict into repeated ``--set k=v`` CLI args."""
    args: list[str] = []
    for k, v in overrides.items():
        args += ["--set", f"{k}={fmt_set_value(v)}"]
    return args


# A resume continues an *existing* v<version>/ folder, whose checkpoints are
# shape-locked to the architecture that produced them (see checkpoints.ARCH_KEYS).
# Changing these on resume would make the trainer's config drift-guard raise; we
# reject it up front instead so the failure is early and clear.
ARCH_PREFIXES = ("network.", "encoding.")


def resume_step(version: str) -> int | None:
    """Learner step of the checkpoint ``--resume auto`` would pick, or None.

    ``version`` is bare (e.g. ``"0.43.0"``), matching ``find_latest_checkpoint``.
    """
    latest = find_latest_checkpoint(str(CHECKPOINTS), version=version)
    if latest is None:
        return None
    return int(Path(latest).stem.rsplit("_", 1)[1])  # deepnash_async_v0.43.0_80000


def pinned_flat(version: str) -> dict:
    """A version's pinned ``config.json`` flattened to sweepable ``section.field``.

    Replays the exact hyper-parameters that produced the checkpoint so drifted
    ``config.py`` defaults can't leak into a continuation. ``version`` is bare.
    """
    cfg_path = CHECKPOINTS / f"v{version}" / "config.json"
    if not cfg_path.exists():
        sys.exit(f"[campaign] no config.json for v{version} at {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    valid = set(all_params())
    return {
        f"{section}.{field}": value
        for section, sub in cfg.items()
        for field, value in sub.items()
        if f"{section}.{field}" in valid
    }


@dataclass
class Run:
    """One manifest entry: a fresh run (``resume is None``) or a continuation."""

    name: str
    bump: str
    resume: str | None = None          # bare version to continue, else fresh
    overrides: dict | None = None      # fresh: defaults+overrides; resume: overrides
    raw_args: list[str] | None = None  # legacy inline RUNS: verbatim CLI args


@dataclass
class Planned:
    """A resolved run ready to launch."""

    version: str            # bare version the run writes to
    name: str
    resume: bool
    args: list[str]        # deepnash-train-async args (resume adds --resume auto)
    step: int | None = None    # resume start step (resume only, for display)
    total: int | None = None   # target train.total_iters (resume only)


def write_template(path: Path) -> None:
    manifest = {
        "defaults": all_params(),
        "runs": [
            {"name": "example-eta-0.3", "bump": "minor",
             "overrides": {"rnad.eta": 0.3}},
            {"name": "example-anchor-2000", "bump": "minor",
             "overrides": {"rnad.iteration_steps": 2000}},
            {"name": "example-resume", "resume": "v0.0.0",
             "overrides": {"train.total_iters": 240000}},
        ],
        "tournament": {
            "pair_games": 8,
            "baselines": ["random", "trout", "mht"],
            "incumbents": 5,
            "keep_top": 2,
            "workers": 8,
            "threshold": 0.05,
            "out": "results/tournament.jsonl",
        },
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[campaign] template with all {len(manifest['defaults'])} sweepable "
          f"params written to {path}")


def load_sweep(path: Path) -> tuple[list[Run], dict | None]:
    """Manifest -> ([Run], tournament cfg or None).

    A run with a ``"resume": "v<version>"`` key continues that existing folder;
    without it the run is fresh and auto-versioned. Manifest ``defaults`` layer
    onto fresh runs only -- a resume derives its base from the version's own
    pinned config (see ``pinned_flat``), so its ``overrides`` carry just the
    knobs to change (typically ``train.total_iters``).
    """
    data = json.loads(path.read_text())
    valid = set(all_params())
    defaults = data.get("defaults", {})
    runs: list[Run] = []
    for i, run in enumerate(data.get("runs", []), 1):
        name = run.get("name", f"run-{i}")
        raw = run.get("overrides", {})
        resume = run.get("resume")
        overrides = raw if resume else {**defaults, **raw}
        unknown = set(overrides) - valid
        if unknown:
            sys.exit(f"[campaign] {name}: unknown/non-sweepable params: "
                     f"{sorted(unknown)}")
        if resume:
            arch = [k for k in raw if k.startswith(ARCH_PREFIXES)]
            if arch:
                sys.exit(f"[campaign] {name}: a resume can't change architecture "
                         f"{sorted(arch)} -- its checkpoints are shape-locked. "
                         f"Bump a fresh version for a new layout instead.")
            resume = resume[1:] if resume.startswith("v") else resume
        runs.append(Run(name=name, bump=run.get("bump", "minor"),
                        resume=resume, overrides=overrides))
    if not runs:
        sys.exit(f"[campaign] no runs in {path}")
    return runs, data.get("tournament")


def write_version(version: str) -> None:
    text = PYPROJECT.read_text()
    # replace the first (project-table) version assignment only
    new, n = re.subn(r'(?m)^version = "[^"]*"', f'version = "{version}"', text, count=1)
    if n != 1:
        raise RuntimeError(
            "could not find a 'version = \"...\"' line in pyproject.toml"
        )
    PYPROJECT.write_text(new)


def plan_versions(runs: list[Run]) -> list[Planned]:
    """Resolve each run's version and args, reserving fresh picks for later runs.

    Fresh runs take the next unused version at their bump level; resume runs keep
    their existing version (already on disk, so never collides with a fresh pick).
    A resume already at/past its target horizon is dropped with a note.
    """
    reserved = set(existing_versions(str(CHECKPOINTS)))
    base = get_version()  # fallback only when nothing exists on disk yet
    plan: list[Planned] = []
    for run in runs:
        if run.resume:
            version = run.resume
            step = resume_step(version)
            if step is None:
                sys.exit(f"[campaign] {run.name}: no checkpoint to resume from in "
                         f"{CHECKPOINTS / f'v{version}'}")
            merged = {**pinned_flat(version), **(run.overrides or {})}
            total = int(merged["train.total_iters"])
            if step >= total:
                print(f"[campaign] {run.name} (v{version}): already at step "
                      f"{step:,} >= horizon {total:,}; skipping")
                continue
            args = ["--resume", "auto", *set_args(merged)]
            plan.append(Planned(version=version, name=run.name, resume=True,
                                args=args, step=step, total=total))
        else:
            version = next_free_version(run.bump, reserved, base)
            reserved.add(version)
            args = run.raw_args if run.raw_args is not None \
                else set_args(run.overrides or {})
            plan.append(Planned(version=version, name=run.name, resume=False,
                                args=args))
    return plan


# ----------------------------------------------------------- tournament stage
def _tournament_mod():
    sys.path.insert(0, str(ROOT / "tools"))
    import tournament
    return tournament


def _elo_on_disk(rows: list[dict], T) -> dict[str, float]:
    """BT-Elo over players whose checkpoints still exist (plus baselines)."""
    ok = [r for r in rows if r["winner"] != "error"]
    names = {r[side] for r in ok for side in ("white", "black")}
    alive = [n for n in names
             if n in T.BASELINES or _checkpoint_path(n).exists()]
    return T.bradley_terry(ok, alive) if alive else {}


def _checkpoint_path(name: str) -> Path:
    ver = name.rsplit("_", 1)[0]
    return CHECKPOINTS / ver / f"deepnash_async_{name}.pt"


def tournament_and_prune(version: str, tcfg: dict) -> None:
    T = _tournament_mod()
    out = ROOT / tcfg.get("out", "results/tournament.jsonl")
    new_pts = sorted((CHECKPOINTS / f"v{version}").glob("*.pt"))
    if not new_pts:
        print(f"[campaign] v{version}: no checkpoints found, skipping tournament")
        return

    rows = []
    if out.exists():
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]

    players = [str(p) for p in new_pts] + list(tcfg.get("baselines",
                                               ["random", "trout", "mht"]))
    elo = _elo_on_disk(rows, T)
    incumbents = [n for n in sorted(elo, key=lambda n: -elo[n])
                  if n not in T.BASELINES][: int(tcfg.get("incumbents", 5))]
    players += [str(_checkpoint_path(n)) for n in incumbents]

    cmd = ["uv", "run", "python", "tools/tournament.py", *players,
           "--pair-games", str(tcfg.get("pair_games", 8)),
           "--workers", str(tcfg.get("workers", 8)),
           "--threshold", str(tcfg.get("threshold", 0.05)),
           "--out", str(out)]
    print(f"[campaign] v{version}: gauntlet vs {len(players) - len(new_pts)} "
          f"opponents\n[campaign] $ {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[campaign] tournament exited {result.returncode}; NOT pruning.")
        return

    keep_top = tcfg.get("keep_top")
    if not keep_top:
        return
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    elo = _elo_on_disk(rows, T)
    ranked = sorted((p for p in new_pts),
                    key=lambda p: -elo.get(p.stem.replace("deepnash_async_", ""),
                                           float("-inf")))
    keep, drop = ranked[:int(keep_top)], ranked[int(keep_top):]
    print(f"[campaign] v{version}: keeping "
          f"{[p.name for p in keep]}, pruning {len(drop)} checkpoint(s)")
    for p in drop:
        p.unlink()


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Chain auto-versioned training runs.")
    ap.add_argument("--sweep", type=Path, default=None,
                    help="JSON sweep manifest (see module docstring)")
    ap.add_argument("--write-template", type=Path, default=None, metavar="PATH",
                    help="write a manifest template with ALL sweepable params "
                         "and their defaults, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, run nothing")
    idle = ap.add_mutually_exclusive_group()
    idle.add_argument("--ignore-idle", dest="ignore_idle", action="store_true",
                      default=IGNORE_IDLE,
                      help="run back-to-back 24/7, ignoring the idle schedule")
    idle.add_argument("--honour-idle", dest="ignore_idle", action="store_false",
                      help="pause during the quiet hours configured in config")
    args = ap.parse_args()

    if args.write_template:
        write_template(args.write_template)
        return

    tournament_cfg = None
    if args.sweep:
        runs, tournament_cfg = load_sweep(args.sweep)
    else:
        runs = [Run(name=f"run-{i}", bump=level, raw_args=run_args)
                for i, (level, run_args) in enumerate(RUNS, 1)]
        if not runs:
            sys.exit("[campaign] no runs: pass --sweep or edit RUNS in this file")

    found = existing_versions(str(CHECKPOINTS))
    plan = plan_versions(runs)
    if not plan:
        sys.exit("[campaign] nothing to do (every resume target past its horizon)")
    print(f"[campaign] existing versions: {found or '(none)'}")
    print(f"[campaign] {len(plan)} run(s) planned"
          + (" + gauntlet/prune per run:" if tournament_cfg else ":"))
    for i, p in enumerate(plan, 1):
        if p.resume:
            print(f"  {i}. v{p.version} [{p.name}]: resume @ {p.step:,} -> "
                  f"{p.total:,} (+{p.total - p.step:,} steps)")
        else:
            shown = " ".join(p.args) if len(p.args) < 12 else \
                " ".join(p.args[:12]) + f" ... (+{(len(p.args) - 12) // 2} params)"
            print(f"  {i}. v{p.version} [{p.name}]: deepnash-train-async {shown}")
    if args.dry_run:
        return

    env = os.environ.copy()
    if args.ignore_idle:
        env["DEEPNASH_IGNORE_IDLE"] = "1"
    print(f"[campaign] idle schedule: "
          f"{'ignored (24/7)' if args.ignore_idle else 'honoured'}")
    # A resume points pyproject at an OLD version; restore the original verbatim
    # afterward so the campaign leaves the working tree's version field untouched.
    original_pyproject = PYPROJECT.read_text()
    try:
        for i, p in enumerate(plan, 1):
            write_version(p.version)  # what get_version() reads in the subprocess
            cmd = ["uv", "run", "deepnash-train-async", *p.args]
            tag = "resume" if p.resume else "fresh"
            print(f"\n[campaign] === run {i}/{len(plan)}  v{p.version} "
                  f"[{p.name}] ({tag}) ===")
            result = subprocess.run(cmd, cwd=ROOT, env=env)
            if result.returncode != 0:
                print(f"[campaign] run {i} (v{p.version}) exited "
                      f"{result.returncode}; stopping.")
                sys.exit(result.returncode)
            if tournament_cfg:
                tournament_and_prune(p.version, tournament_cfg)
    finally:
        PYPROJECT.write_text(original_pyproject)
        print("[campaign] restored pyproject.toml version")
    print("\n[campaign] all runs complete.")


if __name__ == "__main__":
    main()
