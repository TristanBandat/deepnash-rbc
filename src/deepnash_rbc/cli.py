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
from dataclasses import fields, is_dataclass

from .checkpoints import metrics_path
from .config import Config

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{value!r} is not a boolean (use one of {_TRUE | _FALSE})")


def _env_bool(name: str) -> bool | None:
    val = os.environ.get(name)
    if val is None:
        return None
    return _parse_bool(val)


def _coerce_scalar(text: str):
    """Best-effort scalar parse for tuple elements: int, else float, else str."""
    text = text.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _coerce(value: str, annotation, current) -> object:
    """Coerce a --set string to a config field's type, using its annotation
    (a string under ``from __future__ import annotations``) and current value."""
    ann = str(annotation or type(current).__name__).lower()
    v = value.strip()
    if "none" in ann and v.lower() in {"none", "null"}:
        return None
    if "bool" in ann:
        return _parse_bool(v)
    if "tuple" in ann or "list" in ann:
        return tuple(_coerce_scalar(x) for x in v.split(",")) if v else ()
    if "int" in ann:
        return int(v)
    if "float" in ann:
        return float(v)
    return v  # str, str | None with a value, or unknown


def _apply_overrides(cfg: Config, assignments: list[str]) -> None:
    """Apply ``--set section.field=value`` overrides onto nested config dataclasses."""
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"--set expects PATH=VALUE, got {item!r}")
        path, value = item.split("=", 1)
        parts = path.strip().split(".")
        obj = cfg
        for seg in parts[:-1]:
            if not is_dataclass(obj) or not hasattr(obj, seg):
                raise ValueError(f"--set: no config section {seg!r} in {path!r}")
            obj = getattr(obj, seg)
        field = parts[-1]
        names = {f.name: f for f in fields(obj)} if is_dataclass(obj) else {}
        if field not in names:
            raise ValueError(f"--set: unknown config field {path!r}")
        if is_dataclass(getattr(obj, field)):
            raise ValueError(f"--set: {path!r} is a section, not a leaf field")
        setattr(obj, field, _coerce(value, names[field].type, getattr(obj, field)))


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
        "<dir>/v<version>/metrics_v<version>.jsonl unless --metrics-path is "
        "given. Use a distinct dir per ablation arm so the runs don't overwrite "
        "each other.",
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
    p.add_argument(
        "--history",
        type=int,
        default=None,
        help="Past observation frames stacked into the input (EncodingConfig.history); "
        "sets the network's input channels. Like --channels/--blocks this is "
        "shape-locked to existing checkpoints, so bump the version in pyproject.toml "
        "for each distinct value. Default: config value.",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=None,
        metavar="PATH=VALUE",
        help="Override any config field by dotted path, e.g. --set encoding.history=4 "
        "--set rnad.eta=0.3 --set train.train_days=0,1,2. Repeatable; applied AFTER "
        "the named flags so it wins. The value is coerced to the field's type "
        "(int/float/bool/str/tuple; 'none' clears an optional field).",
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
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a tqdm progress bar during training (--no-progress to disable). "
        "Auto-suppressed when stdout isn't a TTY. Default: config value (on).",
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
    if args.history is not None:
        cfg.encoding.history = args.history

    if args.checkpoint_dir is not None:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    # Derive the versioned metrics path from whichever checkpoint_dir is in effect
    # (default or overridden); --metrics-path takes precedence over the default.
    cfg.train.metrics_path = metrics_path(cfg.train.checkpoint_dir)
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
    if args.progress is not None:
        cfg.train.progress = args.progress
    if args.fast_learner is not None:
        cfg.rnad.fast_learner = args.fast_learner
    if args.compile_net is not None:
        cfg.rnad.compile_net = args.compile_net
    if args.amp is not None:
        cfg.rnad.amp = args.amp

    # Generic overrides last so they win over the named flags above.
    if args.overrides:
        _apply_overrides(cfg, args.overrides)

    return cfg
