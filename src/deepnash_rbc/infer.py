"""In-process batched inference for self-play actors.

A self-play actor spends ~85-90% of its wall clock inside the network forward,
queried one position at a time -- and batch 1 is the worst possible shape for
these models on CPU. Measured single-threaded, per-sample forward cost falls by
2.3x (resnet, history=128), 2.9x (resnet, history=16), 3.0x (gru) and 3.4x
(transformer) going from batch 1 to batch 8-16, after which it flattens.

So instead of one game per process, an actor runs ``actor_games`` games on
``actor_games`` threads and funnels every query through one
:class:`InferenceBatcher`: a single background thread that collects the pending
requests, runs *one* forward over them, and hands each caller its own slice.
Threads (not processes) because torch releases the GIL inside the forward, so
the game threads' Python -- the reconchess arbiter, move indexing, encoding,
which together are <10% of an actor -- overlaps it.

The batcher never waits for a game that cannot answer: it collects until either
``max_batch`` requests are in hand or every in-flight game is already waiting on
it, whichever comes first, with ``max_wait_s`` only bounding the straggler case.

:class:`BatchedNet` is a drop-in stand-in for the network object itself, so
``RNaDPlayer`` and ``selfplay.play_one_game`` need no changes -- they only ever
touch the net through ``net.eval()``, ``net(x)`` and ``net.step(x, state)``.

End to end through the real ``actor_loop`` (8 actor processes, 16-core desktop),
decisions/s relative to ``actor_games=1``, as a median of 3 paired repeats --
conditions interleaved within a repeat, ratio taken inside the repeat, because
the standalone throughput of a fixed config drifts 7-14% between runs:

    actor_games        1                2                4
    resnet h=16     1.00x   1.78x [1.74, 1.78]   2.06x [2.06, 2.10]
    transformer     1.00x   1.01x [0.98, 1.14]   1.11x [1.03, 1.20]

The resnet gain is solid and reproducible. The transformer's is not: its mixer
replays the whole token prefix per step, so ``group_by_prefix`` (correctly) keeps
out-of-phase games out of the same forward, and there is little batching left to
do. Batching removed the regression that naive padding caused there, but it does
not make the transformer meaningfully faster -- that needs the prefix replay
itself fixed (an incremental/KV-cached ``step``), not a bigger batch.

Tune on the training box with ``deepnash-bench --actor-games``; the best value
moves with core count, arch, and how fast the learner drains the queue.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch


@dataclass
class _Request:
    x: torch.Tensor                      # [1, C, 8, 8]
    state: Any                           # per-game mixer state (temporal archs)
    done: threading.Event = field(default_factory=threading.Event)
    out: Optional[tuple] = None
    err: Optional[BaseException] = None


class BatcherError(RuntimeError):
    """The batcher thread died; every waiting game thread is unblocked with this."""


def _prefix_len(state) -> int:
    return 0 if state is None else int(state.shape[0])


# Roughly how many mixer tokens the length-independent part of a step costs: the
# per-frame conv encoder, FiLM and the heads all run once per row regardless of
# how long the game is. Only used to weigh padding waste against that fixed cost,
# so an order of magnitude is enough -- it just stops the grouping below from
# splitting hairs over prefixes of 1 vs 2 tokens, where padding is free in practice.
_STEP_BASE_COST = 8.0


def group_by_prefix(batch: List[_Request], waste_tol: float = 0.25) -> List[List[_Request]]:
    """Split a batch so that padding never costs more than ``waste_tol`` extra.

    The transformer and xLSTM mixers keep the whole token prefix as their acting
    state and replay it every step, so a step at move 100 costs far more than a
    step at move 1. Batching those two together pads the short game out to the
    long one's length, and the padded rows are real work: measured on the
    transformer, naively batching a whole actor's games made it *slower* than one
    game per process (0.82x at 2 games, 0.52x at 4) -- games drift out of phase
    as each finishes and restarts, and then most of every batch is padding.

    So group by prefix length instead. Modelling a row as ``base + length``, a
    game (sorted ascending) joins the current group while the padded cost
    ``n * (base + longest)`` stays within ``waste_tol`` of the true cost
    ``sum(base + length)``; otherwise it opens a new group. Games in the same
    phase -- the common case, since games in a process start together -- still
    batch; a freshly restarted game is served on its own rather than dragging the
    whole batch out to its length.
    """
    ordered = sorted(batch, key=lambda r: _prefix_len(r.state))
    groups: List[List[_Request]] = []
    current: List[_Request] = []
    total = 0.0
    for req in ordered:
        length = _prefix_len(req.state)
        padded = (len(current) + 1) * (_STEP_BASE_COST + length)
        real = total + _STEP_BASE_COST + length
        if current and padded > real * (1 + waste_tol):
            groups.append(current)
            current, total = [], 0.0
        current.append(req)
        total += _STEP_BASE_COST + length
    if current:
        groups.append(current)
    return groups


class InferenceBatcher:
    """Serves network queries from several game threads as batched forwards."""

    def __init__(self, net, device: torch.device, max_batch: int,
                 max_wait_s: float = 0.002, on_batch=None):
        self.net = net
        self.device = device
        self.is_temporal = bool(getattr(net, "is_temporal", False))
        self._state_is_prefix = bool(getattr(net, "state_is_prefix", False))
        self.max_batch = max(1, max_batch)
        self.max_wait_s = max_wait_s
        # run on the batcher thread before each forward, i.e. at the one point
        # where no forward is in flight -- that is where a weight refresh is safe
        self._on_batch = on_batch
        # requests submitted by game threads, answered by the batcher thread
        self._q: "queue.SimpleQueue[_Request]" = queue.SimpleQueue()
        # number of games currently inside a game loop, i.e. that can still
        # produce a request for the batch being collected (see _collect)
        self._active = 0
        self._active_lock = threading.Lock()
        self._stop = threading.Event()
        self.batches = 0
        self.queries = 0
        self._thread = threading.Thread(
            target=self._loop, name="inference-batcher", daemon=True
        )
        self._thread.start()

    # -- game-thread side ----------------------------------------------------
    @contextlib.contextmanager
    def slot(self):
        """Mark this thread as playing a game for the duration of the block.

        The batcher stops collecting once it holds a request from every active
        game, so this is what lets it fire immediately instead of always paying
        ``max_wait_s``. A thread between games is not active and is not waited on.
        """
        with self._active_lock:
            self._active += 1
        try:
            yield
        finally:
            with self._active_lock:
                self._active -= 1

    def submit(self, x: torch.Tensor, state: Any) -> tuple:
        """Queue one query and block until the batch containing it has run."""
        req = _Request(x=x, state=state)
        self._q.put(req)
        while not req.done.wait(timeout=1.0):
            if not self._thread.is_alive():
                raise BatcherError("inference batcher thread died")
        if req.err is not None:
            raise req.err
        return req.out

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    # -- batcher thread ------------------------------------------------------
    def _collect(self) -> Optional[List[_Request]]:
        try:
            batch = [self._q.get(timeout=0.1)]
        except queue.Empty:
            return None
        while len(batch) < self.max_batch:
            with self._active_lock:
                active = self._active
            if len(batch) >= active:
                break  # everyone who could ask has asked; don't wait on nobody
            try:
                batch.append(self._q.get(timeout=self.max_wait_s))
            except queue.Empty:
                break  # a straggler is still in Python; go without it
        return batch

    def _loop(self) -> None:
        while not self._stop.is_set():
            batch = self._collect()
            if batch is None:
                continue
            try:
                if self._on_batch is not None:
                    self._on_batch()
                self._forward(batch)
            except BaseException as e:  # noqa: BLE001 -- re-raised in the callers
                for req in batch:
                    if req.out is None:  # don't fail a request that did get served
                        req.err = e
                    req.done.set()
                if not isinstance(e, Exception):
                    raise
            self.batches += 1
            self.queries += len(batch)

    def _forward(self, batch: List[_Request]) -> None:
        for group in group_by_prefix(batch) if self._state_is_prefix else [batch]:
            self._forward_one(group)

    def _forward_one(self, batch: List[_Request]) -> None:
        x = torch.cat([r.x for r in batch]).to(self.device)
        with torch.no_grad():
            if self.is_temporal:
                value, sense, move, states = self.net.step_batched(
                    x, [r.state for r in batch]
                )
            else:
                value, sense, move = self.net(x)
                states = [None] * len(batch)
        for i, req in enumerate(batch):
            # keep the [1, ...] leading dim each caller's own step() would return
            req.out = (value[i: i + 1], sense[i: i + 1], move[i: i + 1], states[i])
            req.done.set()


class BatchedNet:
    """Network-shaped façade over an :class:`InferenceBatcher`.

    Implements exactly the surface ``RNaDPlayer`` and ``selfplay.play_one_game``
    use, so nothing downstream knows the forward is shared with other games.
    """

    def __init__(self, batcher: InferenceBatcher):
        self._batcher = batcher
        self.is_temporal = batcher.is_temporal

    def eval(self):  # play_one_game calls net.eval()
        return self

    def __call__(self, x: torch.Tensor):
        value, sense, move, _ = self._batcher.submit(x, None)
        return value, sense, move

    def step(self, x: torch.Tensor, state):
        return self._batcher.submit(x, state)


def play_games_forever(batcher: InferenceBatcher, cfg, n_games: int, on_trajectory,
                       should_stop) -> None:
    """Run ``n_games`` self-play games concurrently until ``should_stop()``.

    Each thread plays whole games back to back through the shared batcher and
    hands finished trajectories to ``on_trajectory``; a thread that finishes a
    short game starts the next one immediately instead of waiting on the others,
    so the batch stays full. Blocks until every game thread has exited.
    """
    from .selfplay import play_one_game

    net = BatchedNet(batcher)
    device = torch.device("cpu")

    def worker() -> None:
        while not should_stop():
            with batcher.slot():
                trajectories = play_one_game(net, device, cfg)
            for traj in trajectories:
                if should_stop():
                    return
                on_trajectory(traj)

    threads = [
        threading.Thread(target=worker, name=f"selfplay-{i}", daemon=True)
        for i in range(n_games)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def resolve_batch_size(cfg) -> int:
    """Max requests per forward: ``actor_max_batch`` if set, else one per game."""
    configured = getattr(cfg.train, "actor_max_batch", 0)
    return configured if configured > 0 else max(1, cfg.train.actor_games)
