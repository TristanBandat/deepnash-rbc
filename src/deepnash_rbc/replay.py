"""Trajectory records and replay buffer.

Each *decision point* (a sense choice or a move choice) is one Step. A game
produces one Trajectory per player. Rewards are sparse: +1/-1/0 assigned to the
whole trajectory at game end (z), with the per-step reward 0 except the last.
The R-NaD reward transform and v-trace operate over these per-player sequences.

Observations are stored *deduplicated*: the network input for a step is the
last (history-1) committed frames plus the step's in-progress frame, so
consecutive steps share almost their whole stack. Storing the full stack per
step made trajectory size scale with ``history`` (at history=128 each frame was
stored ~128x -- dominating replay RAM, the actor->learner queue, and the H2D
copy). Instead each Trajectory keeps every committed frame once (``frames``)
and each Step keeps only its in-progress frame plus a ``turn`` index; the
learner reconstructs the stacks on-device (see rnad.trainer.collate).
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from typing import List

import numpy as np

SENSE = 0
MOVE = 1


@dataclass
class Step:
    obs: np.ndarray            # [FRAME_CHANNELS, 8, 8] uint8 -- in-progress frame only
    head: int                  # SENSE or MOVE
    legal: np.ndarray          # int64 array of legal action indices for this head
    action: int                # chosen action index
    behavior_logprob: float    # log prob under the actor's (behavior) policy
    turn: int                  # committed frames before this decision (index into
    #                            Trajectory.frames; the stack is frames[turn-H+1:turn]
    #                            left-padded with blanks, then obs)


@dataclass
class Trajectory:
    steps: List[Step] = field(default_factory=list)
    # committed [FRAME_CHANNELS, 8, 8] uint8 frame per finished turn, in order;
    # shared by every Step whose history window covers that turn.
    frames: List[np.ndarray] = field(default_factory=list)
    z: float = 0.0  # terminal return from this player's perspective (+1/-1/0)

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


class ReplayBuffer:
    """FIFO trajectory buffer. add/sample are locked so a prefetch thread can
    sample while the main thread drains new trajectories in (async_train)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: List[Trajectory] = []
        self._lock = threading.Lock()

    def add(self, traj: Trajectory) -> None:
        if len(traj) == 0:
            return
        with self._lock:
            self._buf.append(traj)
            if len(self._buf) > self.capacity:
                self._buf.pop(0)

    def sample(self, n: int) -> List[Trajectory]:
        with self._lock:
            if not self._buf:
                return []
            n = min(n, len(self._buf))
            return random.sample(self._buf, n)

    def __len__(self) -> int:
        return len(self._buf)
