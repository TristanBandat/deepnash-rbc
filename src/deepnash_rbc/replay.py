"""Trajectory records and replay buffer.

Each *decision point* (a sense choice or a move choice) is one Step. A game
produces one Trajectory per player. Rewards are sparse: +1/-1/0 assigned to the
whole trajectory at game end (z), with the per-step reward 0 except the last.
The R-NaD reward transform and v-trace operate over these per-player sequences.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

import numpy as np

SENSE = 0
MOVE = 1


@dataclass
class Step:
    obs: np.ndarray            # [in_channels, 8, 8] uint8 (binary planes)
    head: int                  # SENSE or MOVE
    legal: np.ndarray          # int64 array of legal action indices for this head
    action: int                # chosen action index
    behavior_logprob: float    # log prob under the actor's (behavior) policy


@dataclass
class Trajectory:
    steps: List[Step] = field(default_factory=list)
    z: float = 0.0  # terminal return from this player's perspective (+1/-1/0)

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: List[Trajectory] = []

    def add(self, traj: Trajectory) -> None:
        if len(traj) == 0:
            return
        self._buf.append(traj)
        if len(self._buf) > self.capacity:
            self._buf.pop(0)

    def sample(self, n: int) -> List[Trajectory]:
        if not self._buf:
            return []
        n = min(n, len(self._buf))
        return random.sample(self._buf, n)

    def __len__(self) -> int:
        return len(self._buf)
