"""Metrics logging.

Two kinds of signal get logged, and they answer different questions:

  - TRAINING metrics (loss/policy_loss/value_loss/entropy): stability telemetry.
    In self-play these are all relative to a moving target, so they tell you
    whether learning is *stable*, NOT whether the agent is getting better at RBC.
  - SKILL metrics (win-rate vs fixed baseline bots): the actual learning curve.
    This is the number that should rise over training.

Everything is appended as JSON lines so you can tail/plot it without a DB:
    {"iter": 50, "type": "skill", "vs_random": 0.85, "vs_attacker": 0.40, ...}
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict

import torch


class MetricsLogger:
    def __init__(self, path: str, resume_wall_s: float = 0.0):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Seed the clock so wall_s CONTINUES from a resume rather than restarting
        # at 0: a fresh start passes 0.0; a resume passes the wall_s recorded at
        # the checkpoint (see resume_metrics).
        self._t0 = time.time() - resume_wall_s

    def log(self, record: Dict) -> None:
        record = {"wall_s": round(time.time() - self._t0, 1), **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def _record_step(rec: Dict) -> int | None:
    """The step counter a record is tied to: ``step`` (async) or ``iter`` (sync)."""
    if "step" in rec:
        return int(rec["step"])
    if "iter" in rec:
        return int(rec["iter"])
    return None


def _reverse_lines(f, block: int = 65536):
    """Yield ``(start_offset, line_bytes)`` for each line of ``f``, last first.

    ``line_bytes`` excludes the trailing newline; ``start_offset`` is the byte
    offset where the line's content begins. The file is read in blocks from the
    end so only the tail that is actually consumed gets touched."""
    f.seek(0, os.SEEK_END)
    pos = f.tell()  # bytes [pos, data_end) are buffered in `data`
    data = b""
    while pos > 0:
        read = min(block, pos)
        pos -= read
        f.seek(pos)
        data = f.read(read) + data
        idx = len(data)
        nl = data.rfind(b"\n", 0, idx)
        while nl != -1:
            yield pos + nl + 1, data[nl + 1 : idx]
            idx = nl
            nl = data.rfind(b"\n", 0, idx)
        data = data[:idx]  # leading partial line; may continue in the next block
    if data:
        yield 0, data


def resume_metrics(path: str, step: int) -> float:
    """Prepare a metrics log for a resume at learner ``step`` and return the
    ``wall_s`` recorded there (0.0 if the file or a matching entry is missing).

    A run keeps logging after its last checkpoint, so the file can hold entries
    NEWER than the checkpoint being resumed from. Replaying past them would make
    step/wall_s jump backwards and confuse any reader, so every entry after the
    resume point is dropped (the file is truncated). The returned wall_s seeds
    the logger so the clock continues instead of restarting.

    Scans from the end -- the resume point is almost always near the tail -- so
    cost tracks how much is discarded, not the file size.
    """
    if not os.path.exists(path):
        return 0.0
    size = os.path.getsize(path)
    truncate_at: int | None = None
    wall = 0.0
    with open(path, "rb+") as f:
        for start, raw in _reverse_lines(f):
            text = raw.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except json.JSONDecodeError:
                continue
            s = _record_step(rec)
            if s is None or s > step:
                continue  # stale: logged after the checkpoint we resume from
            truncate_at = start + len(raw) + 1  # include this line's newline
            wall = float(rec.get("wall_s", 0.0))
            break
        if truncate_at is not None and truncate_at < size:
            f.truncate(truncate_at)
    return wall


def masked_entropy(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy (nats) of the masked policy over a batch of decisions.
    Falling entropy = the policy is sharpening; near-zero very early can signal the
    logit-runaway / collapse that the NeuRD threshold is meant to prevent."""
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(legal_mask, logits, torch.full_like(logits, neg_inf))
    logp = torch.log_softmax(masked, dim=-1)
    p = logp.exp()
    ent = -(p * logp).masked_fill(~legal_mask, 0.0).sum(dim=-1)
    return ent.mean()
