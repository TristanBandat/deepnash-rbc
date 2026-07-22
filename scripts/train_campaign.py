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
         "defaults":   {"encoding.history": 16, ...},   # applied to every run
         "runs": [
           {"name": "eta-0.3", "bump": "minor", "overrides": {"rnad.eta": 0.3}},
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
from dataclasses import fields
from pathlib import Path

from deepnash_rbc.checkpoints import existing_versions, next_free_version
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


def write_template(path: Path) -> None:
    manifest = {
        "defaults": all_params(),
        "runs": [
            {"name": "example-eta-0.3", "bump": "minor",
             "overrides": {"rnad.eta": 0.3}},
            {"name": "example-anchor-2000", "bump": "minor",
             "overrides": {"rnad.iteration_steps": 2000}},
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


def load_sweep(path: Path) -> tuple[list[tuple[str, str, list[str]]], dict | None]:
    """Manifest -> [(bump, name, cli_args)], tournament cfg (or None)."""
    data = json.loads(path.read_text())
    valid = set(all_params())
    defaults = data.get("defaults", {})
    runs = []
    for i, run in enumerate(data.get("runs", []), 1):
        overrides = {**defaults, **run.get("overrides", {})}
        unknown = set(overrides) - valid
        if unknown:
            sys.exit(f"[campaign] run {i}: unknown/non-sweepable params: "
                     f"{sorted(unknown)}")
        args = []
        for k, v in overrides.items():
            args += ["--set", f"{k}={fmt_set_value(v)}"]
        runs.append((run.get("bump", "minor"), run.get("name", f"run-{i}"), args))
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


def plan_versions(runs: list[tuple[str, str, list[str]]]) -> list[tuple[str, str, list[str]]]:
    """Resolve each run's version, reserving earlier picks for later runs."""
    reserved = set(existing_versions(str(CHECKPOINTS)))
    base = get_version()  # fallback only when nothing exists on disk yet
    plan = []
    for level, name, run_args in runs:
        version = next_free_version(level, reserved, base)
        reserved.add(version)
        plan.append((version, name, run_args))
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
        runs = [(level, f"run-{i}", run_args)
                for i, (level, run_args) in enumerate(RUNS, 1)]
        if not runs:
            sys.exit("[campaign] no runs: pass --sweep or edit RUNS in this file")

    found = existing_versions(str(CHECKPOINTS))
    plan = plan_versions(runs)
    print(f"[campaign] existing versions: {found or '(none)'}")
    print(f"[campaign] {len(plan)} run(s) planned"
          + (" + gauntlet/prune per run:" if tournament_cfg else ":"))
    for i, (version, name, run_args) in enumerate(plan, 1):
        shown = " ".join(run_args) if len(run_args) < 12 else \
            " ".join(run_args[:12]) + f" ... (+{(len(run_args) - 12) // 2} params)"
        print(f"  {i}. v{version} [{name}]: deepnash-train-async {shown}")
    if args.dry_run:
        return

    env = os.environ.copy()
    if args.ignore_idle:
        env["DEEPNASH_IGNORE_IDLE"] = "1"
    print(f"[campaign] idle schedule: "
          f"{'ignored (24/7)' if args.ignore_idle else 'honoured'}")
    for i, (version, name, run_args) in enumerate(plan, 1):
        write_version(version)  # what get_version() reads in the subprocess
        cmd = ["uv", "run", "deepnash-train-async", *run_args]
        print(f"\n[campaign] === run {i}/{len(plan)}  v{version} [{name}] ===")
        result = subprocess.run(cmd, cwd=ROOT, env=env)
        if result.returncode != 0:
            print(f"[campaign] run {i} (v{version}) exited {result.returncode}; stopping.")
            sys.exit(result.returncode)
        if tournament_cfg:
            tournament_and_prune(version, tournament_cfg)
    print("\n[campaign] all runs complete.")


if __name__ == "__main__":
    main()
