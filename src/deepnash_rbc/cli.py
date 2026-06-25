"""Shared CLI / env overrides for the training entrypoints.

This exists so the all-actions-NeuRD ablation runs as two single commands
instead of hand-editing ``config.py`` between arms, and so each arm keeps its
checkpoints + metrics in a separate directory rather than clobbering the other:

    uv run deepnash-train --full-action-neurd    --checkpoint-dir runs/allactions
    uv run deepnash-train --no-full-action-neurd --checkpoint-dir runs/single

Precedence for ``full_action_neurd``: explicit CLI flag > ``DEEPNASH_FULL_ACTION_NEURD``
env var > the config default. Passing a ``cfg`` directly to a trainer ``main()``
(as the tests do) bypasses argument parsing entirely.
"""

from __future__ import annotations

import argparse
import os

from .config import Config

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str) -> bool | None:
    val = os.environ.get(name)
    if val is None:
        return None
    v = val.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{name}={val!r} is not a boolean (use one of {_TRUE | _FALSE})")


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="DeepNash-RBC trainer")
    p.add_argument(
        "--full-action-neurd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="All-actions NeuRD (--full-action-neurd) vs single-sample "
        "(--no-full-action-neurd). Overrides the config default and the "
        "DEEPNASH_FULL_ACTION_NEURD env var. Default: use the config value.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoints; also redirects metrics to "
        "<dir>/metrics.jsonl unless --metrics-path is given. Use a distinct "
        "dir per ablation arm so the runs don't overwrite each other.",
    )
    p.add_argument(
        "--metrics-path",
        default=None,
        help="Path to the metrics JSONL (overrides the checkpoint-dir default).",
    )
    p.add_argument(
        "--channels",
        type=int,
        default=None,
        help="ResNet torso width (NetworkConfig.channels). Bump with --blocks for "
        "more capacity / stronger play. A larger net is shape-incompatible with "
        "existing checkpoints, so bump the version in pyproject.toml too. Default: "
        "config value.",
    )
    p.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="ResNet torso depth (NetworkConfig.blocks). See --channels. Default: "
        "config value.",
    )
    p.add_argument("--device", default=None, help="cuda or cpu (default: config value).")
    p.add_argument("--seed", type=int, default=None, help="Override the training seed.")
    p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Run the in-training skill eval every N learner steps; 0 PAUSES eval "
        "entirely (it runs synchronously on the learner and stalls the GPU while it "
        "plays CPU games, so pause it when the learner is fast). Default: config value.",
    )
    p.add_argument(
        "--async-actors",
        type=int,
        default=None,
        help="Persistent CPU self-play workers for deepnash-train-async. Each is "
        "pinned to one torch thread, so set this near your core count minus a few "
        "for the learner (e.g. ~56 on a 60-core box). Default: use the config value.",
    )
    p.add_argument(
        "--num-actors",
        type=int,
        default=None,
        help="Self-play actors for the sync deepnash-train trainer (>1 spawns a "
        "torch.multiprocessing pool). Default: use the config value.",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume deepnash-train-async from a checkpoint. Bare --resume picks "
        "the latest checkpoint in --checkpoint-dir/v<version>/ (current project "
        "version); pass a path to pick a specific one. Restores weights+step "
        "(and optimizer/pi_reg if the "
        "checkpoint has them; older checkpoints don't, so those reset with a warning).",
    )
    p.add_argument(
        "--fast-learner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Vectorized (bit-identical) learner step. Default: config value (on).",
    )
    p.add_argument(
        "--compile",
        dest="compile_net",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="torch.compile the forward (opt-in; changes numerics slightly).",
    )
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="bf16 autocast on the forward (opt-in; changes numerics; CUDA only).",
    )
    return p


def config_from_args(argv: list[str] | None = None, prog: str | None = None) -> Config:
    """Build a Config from argv, applying CLI/env overrides onto the defaults."""
    args = build_parser(prog).parse_args(argv)
    cfg = Config()

    if args.full_action_neurd is not None:
        cfg.rnad.full_action_neurd = args.full_action_neurd
    else:
        env = _env_bool("DEEPNASH_FULL_ACTION_NEURD")
        if env is not None:
            cfg.rnad.full_action_neurd = env

    if args.channels is not None:
        cfg.network.channels = args.channels
    if args.blocks is not None:
        cfg.network.blocks = args.blocks

    if args.checkpoint_dir is not None:
        cfg.train.checkpoint_dir = args.checkpoint_dir
        cfg.train.metrics_path = os.path.join(args.checkpoint_dir, "metrics.jsonl")
    if args.metrics_path is not None:
        cfg.train.metrics_path = args.metrics_path
    if args.device is not None:
        cfg.train.device = args.device
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.eval_every is not None:
        cfg.train.eval_every = args.eval_every
    if args.async_actors is not None:
        cfg.train.async_actors = args.async_actors
    if args.num_actors is not None:
        cfg.train.num_actors = args.num_actors
    if args.resume is not None:
        cfg.train.resume = args.resume
    if args.fast_learner is not None:
        cfg.rnad.fast_learner = args.fast_learner
    if args.compile_net is not None:
        cfg.rnad.compile_net = args.compile_net
    if args.amp is not None:
        cfg.rnad.amp = args.amp

    return cfg
