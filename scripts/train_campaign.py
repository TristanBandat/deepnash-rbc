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

   A run with a "from": "v<version>@<step>" key FORKS a specific checkpoint into a
   brand-new auto-versioned folder (bump applies) -- ideal for minmaxing: branch a
   strong checkpoint and try different schedules/hyper-parameters without touching
   the source run. "from" also accepts "v<version>" (its latest checkpoint) or a
   literal .pt path. Like resume it replays the source's pinned config for the
   architecture (arch overrides rejected) and applies the run's "overrides" on top;
   the forked run's step counter starts at the source checkpoint's step, so a
   step-based lr schedule (rnad.lr_schedule) is measured on the absolute step. The
   source is stamped into the new folder's config.json as
   ``train.fork_source = "v<version>@<step>"`` (resolved to an exact step) so eval
   can reconstruct lineage from config.json alone; it is None for fresh/resume runs.

       {"name": "tf-wsd-from-v0.42-peak", "bump": "minor",
        "from": "v0.42.0@80000",
        "overrides": {"rnad.lr_schedule": "wsd", "rnad.lr_decay_start": 120000,
                      "train.total_iters": 240000}}

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
    "train.fork_source",         # provenance, stamped by the fork path (below)
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


def resolve_checkpoint(spec: str) -> tuple[str, str, int]:
    """Resolve a fork source ``spec`` -> (checkpoint path, source version, step).

    Accepts ``"v0.42.0@80000"`` (that exact checkpoint), ``"v0.42.0"`` (the
    version's latest checkpoint), or a literal ``.pt`` path. The bare source
    version is returned so the fork can replay that checkpoint's pinned config
    and match its architecture. Step is parsed from the filename.
    """
    if spec.endswith(".pt") or "/" in spec:
        p = Path(spec)
        if not p.is_absolute():
            p = ROOT / p
        parent = p.parent.name
        version = parent[1:] if parent.startswith("v") else parent
    elif "@" in spec:
        ver, _, step_s = spec.partition("@")
        version = ver[1:] if ver.startswith("v") else ver
        p = CHECKPOINTS / f"v{version}" / f"deepnash_async_v{version}_{step_s}.pt"
    else:
        version = spec[1:] if spec.startswith("v") else spec
        latest = find_latest_checkpoint(str(CHECKPOINTS), version=version)
        p = Path(latest) if latest else CHECKPOINTS / f"v{version}"
    if not p.exists():
        sys.exit(f"[campaign] fork source not found: {spec} -> {p}")
    m = re.search(r"_(\d+)\.pt$", p.name)
    if not m:
        sys.exit(f"[campaign] can't parse a step from fork source {p.name}")
    return str(p), version, int(m.group(1))


@dataclass
class Run:
    """One manifest entry: a fresh run, an in-place resume, or a fork."""

    name: str
    bump: str
    resume: str | None = None          # bare version to continue in place, else None
    fork_from: str | None = None       # checkpoint spec to fork a NEW version from
    overrides: dict | None = None      # fresh: defaults+overrides; else: overrides only
    raw_args: list[str] | None = None  # legacy inline RUNS: verbatim CLI args


@dataclass
class Planned:
    """A resolved run ready to launch."""

    version: str            # bare version the run writes to
    name: str
    kind: str              # "fresh" | "resume" | "fork"
    args: list[str]        # deepnash-train-async args (resume/fork add --resume ...)
    step: int | None = None    # resume/fork start step (for display)
    total: int | None = None   # target train.total_iters (resume/fork)
    source: str | None = None  # fork: source checkpoint spec (for display)


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
            {"name": "example-fork", "from": "v0.0.0@80000",
             "overrides": {"rnad.lr_schedule": "wsd",
                           "train.total_iters": 240000}},
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

    A run continues a checkpoint if it has a ``"resume": "v<version>"`` key (extend
    that folder in place) or a ``"from": "v<version>@<step>"`` key (fork a NEW
    auto-versioned folder off that checkpoint -- for minmaxing variants without
    touching the source run). Otherwise the run is fresh and auto-versioned.
    Manifest ``defaults`` layer onto fresh runs only -- a resume/fork derives its
    base from the source's own pinned config (see ``pinned_flat``), so its
    ``overrides`` carry just the knobs to change (schedule, lr, total_iters, ...).
    """
    data = json.loads(path.read_text())
    valid = set(all_params())
    defaults = data.get("defaults", {})
    runs: list[Run] = []
    for i, run in enumerate(data.get("runs", []), 1):
        name = run.get("name", f"run-{i}")
        raw = run.get("overrides", {})
        resume = run.get("resume")
        fork_from = run.get("from")
        if resume and fork_from:
            sys.exit(f"[campaign] {name}: use either 'resume' or 'from', not both")
        cont = bool(resume or fork_from)  # continues an existing checkpoint
        overrides = raw if cont else {**defaults, **raw}
        unknown = set(overrides) - valid
        if unknown:
            sys.exit(f"[campaign] {name}: unknown/non-sweepable params: "
                     f"{sorted(unknown)}")
        if cont:
            arch = [k for k in raw if k.startswith(ARCH_PREFIXES)]
            if arch:
                sys.exit(f"[campaign] {name}: can't change architecture {sorted(arch)} "
                         f"when continuing a checkpoint -- its weights are shape-locked. "
                         f"Use a fresh run for a new layout.")
        if resume:
            resume = resume[1:] if resume.startswith("v") else resume
        runs.append(Run(name=name, bump=run.get("bump", "minor"), resume=resume,
                        fork_from=fork_from, overrides=overrides))
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

    Fresh/fork runs take the next unused version at their bump level; resume runs
    keep their existing version (already on disk, so never collides with a fresh
    pick). A resume/fork already at/past its target horizon is dropped with a note.
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
            plan.append(Planned(version=version, name=run.name, kind="resume",
                                args=args, step=step, total=total))
        elif run.fork_from:
            version = next_free_version(run.bump, reserved, base)
            reserved.add(version)
            ckpt, src_ver, step = resolve_checkpoint(run.fork_from)
            # arch comes from the source (so the loaded weights fit); overrides win
            merged = {**pinned_flat(src_ver), **(run.overrides or {})}
            # stamp lineage into the new folder's config.json (resolved to an exact
            # version@step) so eval can trace the fork without external bookkeeping
            merged["train.fork_source"] = f"v{src_ver}@{step}"
            total = int(merged["train.total_iters"])
            if step >= total:
                print(f"[campaign] {run.name}: fork source at {step:,} >= horizon "
                      f"{total:,}; nothing to train; skipping")
                continue
            args = ["--resume", ckpt, *set_args(merged)]
            plan.append(Planned(version=version, name=run.name, kind="fork",
                                args=args, step=step, total=total,
                                source=run.fork_from))
        else:
            version = next_free_version(run.bump, reserved, base)
            reserved.add(version)
            args = run.raw_args if run.raw_args is not None \
                else set_args(run.overrides or {})
            plan.append(Planned(version=version, name=run.name, kind="fresh",
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
        sys.exit("[campaign] nothing to do (every resume/fork target past its horizon)")
    print(f"[campaign] existing versions: {found or '(none)'}")
    print(f"[campaign] {len(plan)} run(s) planned"
          + (" + gauntlet/prune per run:" if tournament_cfg else ":"))
    for i, p in enumerate(plan, 1):
        if p.kind == "resume":
            print(f"  {i}. v{p.version} [{p.name}]: resume @ {p.step:,} -> "
                  f"{p.total:,} (+{p.total - p.step:,} steps)")
        elif p.kind == "fork":
            print(f"  {i}. v{p.version} [{p.name}]: fork {p.source} @ {p.step:,} -> "
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
            print(f"\n[campaign] === run {i}/{len(plan)}  v{p.version} "
                  f"[{p.name}] ({p.kind}) ===")
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
