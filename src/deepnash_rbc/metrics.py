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
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._t0 = time.time()

    def log(self, record: Dict) -> None:
        record = {"wall_s": round(time.time() - self._t0, 1), **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


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
