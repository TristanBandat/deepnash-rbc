"""reconchess entry point for playing DeepNash-RBC on the JHU APL server.

rc-connect loads this file, finds the bot class via get_player(), and
instantiates it with no arguments once per accepted game (in a subprocess),
so the checkpoint is loaded at game start, before the clock matters.

Preferred usage: set the checkpoint in rbc_connect.json and credentials in
.env (both repo root), then run scripts/rc_connect.py. This file can also be
used directly:

    uv run rc-connect --username <user> --password <pass> scripts/deepnash_bot.py

Model settings come from rbc_connect.json ("checkpoint", "device", "greedy",
checkpoint path relative to the repo root). Environment variables override
the config for one-off experiments:

    DEEPNASH_CKPT    path to a checkpoint .pt
    DEEPNASH_DEVICE  torch device string, e.g. "cuda" or "cpu"
    DEEPNASH_GREEDY  set to 1 to play argmax instead of sampling from the
                     policy (default: sample -- the RNaD policy is a mixed
                     strategy, sampling is the intended way to play it)
    DEEPNASH_SAMPLE_THRESHOLD
                     overrides "sample_threshold" from the config: when
                     sampling, actions with probability below this are dropped
                     and the rest renormalized (DeepNash-style truncated
                     sampling). 0.0 = raw policy.

With neither set, falls back to the highest step of the newest version under
checkpoints/, and cuda if available.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

from deepnash_rbc.agent import RNaDPlayer
from deepnash_rbc.play_session import load_net

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "rbc_connect.json"


def _config() -> dict:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text())
    return {}

_STEP_RE = re.compile(r"_(\d+)\.pt$")
_VERSION_RE = re.compile(r"v\d+(\.\d+)*$")


def _latest_checkpoint() -> Path:
    root = _REPO_ROOT / "checkpoints"
    dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and _VERSION_RE.fullmatch(d.name)),
        key=lambda d: tuple(int(x) for x in d.name[1:].split(".")),
    )
    for d in reversed(dirs):
        steps = sorted(
            (p for p in d.glob("*.pt") if _STEP_RE.search(p.name)),
            key=lambda p: int(_STEP_RE.search(p.name).group(1)),
        )
        if steps:
            return steps[-1]
    raise FileNotFoundError(f"no checkpoints with a _<step>.pt suffix under {root}")


class DeepNashBot(RNaDPlayer):
    def __init__(self):
        cfg = _config()
        ckpt_str = os.environ.get("DEEPNASH_CKPT") or cfg.get("checkpoint")
        ckpt = _REPO_ROOT / ckpt_str if ckpt_str else _latest_checkpoint()
        device = torch.device(
            os.environ.get("DEEPNASH_DEVICE")
            or cfg.get("device")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        net, enc = load_net(str(ckpt), device)
        greedy = os.environ.get("DEEPNASH_GREEDY", "").lower() in ("1", "true", "yes")
        sample = not (greedy or ("DEEPNASH_GREEDY" not in os.environ and cfg.get("greedy")))
        threshold = float(
            os.environ.get("DEEPNASH_SAMPLE_THRESHOLD", cfg.get("sample_threshold", 0.0) or 0.0)
        )
        print(f"[deepnash_bot] checkpoint={ckpt.name} device={device} "
              f"history={enc.history} sample={sample} threshold={threshold}", flush=True)
        super().__init__(net, device, history=enc.history, sample=sample,
                         sample_threshold=threshold)


def get_player():
    # this module holds two Player subclasses (RNaDPlayer is imported above),
    # so reconchess.load_player needs this to disambiguate
    return DeepNashBot
