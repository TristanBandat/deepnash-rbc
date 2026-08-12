"""Batched acting must agree with one-game-at-a-time acting.

Two guarantees are pinned here, because everything the multi-game actor buys is
worthless if it quietly changes the behavior policy:

  1. ``TemporalNet.step_batched`` over B games with ragged histories equals B
     independent ``step`` chains. The mixers are causal, so padding the shorter
     games on the right cannot leak into the row we read back -- this test is
     what makes that claim checkable rather than asserted.
  2. ``InferenceBatcher`` + ``BatchedNet`` return, to each caller, exactly what
     calling the net directly would have returned.
"""

from __future__ import annotations

import threading

import pytest
import torch

from deepnash_rbc.config import Config
from deepnash_rbc.encoding.observation import FRAME_CHANNELS
from deepnash_rbc.infer import BatchedNet, InferenceBatcher
from deepnash_rbc.network import make_net

TEMPORAL_ARCHS = ["gru", "lstm", "transformer", "xlstm"]


def _cfg(arch: str) -> Config:
    cfg = Config()
    cfg.encoding.history = 1 if arch != "resnet" else 4
    cfg.network.arch = arch
    cfg.network.channels = 16
    cfg.network.blocks = 1
    cfg.network.enc_blocks = 1
    cfg.network.mixer_dim = 32
    cfg.network.mixer_layers = 2
    cfg.network.nhead = 2
    return cfg


def _net(arch: str):
    torch.manual_seed(0)
    cfg = _cfg(arch)
    try:
        net = make_net(cfg.encoding, cfg.network)
    except (ImportError, OSError) as e:  # xlstm is an optional dependency
        pytest.skip(f"{arch} unavailable: {e}")
    return net.eval(), cfg


@pytest.mark.parametrize("arch", TEMPORAL_ARCHS)
def test_step_batched_matches_sequential_steps(arch):
    """Ragged game lengths batched together == the same games stepped alone."""
    net, _ = _net(arch)
    torch.manual_seed(1)
    # three games at different points in their history, so every call pads
    lengths = [5, 2, 4]
    frames = [
        [torch.randn(1, FRAME_CHANNELS, 8, 8) for _ in range(n)] for n in lengths
    ]

    # reference: each game stepped on its own, exactly as a single-game actor does
    reference = []
    for game in frames:
        state = None
        outs = []
        with torch.no_grad():
            for frame in game:
                value, sense, move, state = net.step(frame, state)
                outs.append((value, sense, move))
        reference.append(outs)

    # batched: at step t only the games that are still running take part, so the
    # batch shrinks and the histories stay ragged
    states = [None] * len(frames)
    batched = [[] for _ in frames]
    for t in range(max(lengths)):
        live = [i for i, n in enumerate(lengths) if t < n]
        x = torch.cat([frames[i][t] for i in live])
        with torch.no_grad():
            value, sense, move, new_states = net.step_batched(x, [states[i] for i in live])
        for slot, i in enumerate(live):
            states[i] = new_states[slot]
            batched[i].append(
                (value[slot: slot + 1], sense[slot: slot + 1], move[slot: slot + 1])
            )

    for i, (ref_game, bat_game) in enumerate(zip(reference, batched)):
        for t, (ref, bat) in enumerate(zip(ref_game, bat_game)):
            for name, a, b in zip(("value", "sense", "move"), ref, bat):
                torch.testing.assert_close(
                    a, b, rtol=1e-4, atol=1e-5,
                    msg=lambda m, n=name, i=i, t=t: f"game {i} step {t} {n}: {m}",
                )


@pytest.mark.parametrize("arch", ["resnet", "gru", "transformer"])
def test_batcher_serves_each_caller_its_own_result(arch):
    """Concurrent callers through the batcher get the direct-call answers."""
    net, cfg = _net(arch)
    channels = FRAME_CHANNELS if arch != "resnet" else cfg.encoding.in_channels
    torch.manual_seed(2)
    inputs = [torch.randn(1, channels, 8, 8) for _ in range(6)]

    with torch.no_grad():
        if arch == "resnet":
            expected = [net(x) for x in inputs]
        else:
            expected = [net.step(x, None)[:3] for x in inputs]

    batcher = InferenceBatcher(net, torch.device("cpu"), max_batch=len(inputs),
                               max_wait_s=0.05)
    proxy = BatchedNet(batcher)
    results: dict = {}
    barrier = threading.Barrier(len(inputs))

    def call(i):
        barrier.wait()  # force the requests to land together so they batch
        with batcher.slot():
            if arch == "resnet":
                results[i] = proxy(inputs[i])
            else:
                results[i] = proxy.step(inputs[i], None)[:3]

    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(inputs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    batcher.stop()

    assert len(results) == len(inputs)
    assert batcher.batches < len(inputs), "requests were not batched at all"
    for i, want in enumerate(expected):
        for name, a, b in zip(("value", "sense", "move"), want, results[i]):
            torch.testing.assert_close(
                a, b, rtol=1e-4, atol=1e-5,
                msg=lambda m, n=name, i=i: f"request {i} {n}: {m}",
            )


def _req(prefix_len: int):
    from deepnash_rbc.infer import _Request

    state = None if prefix_len == 0 else torch.zeros(prefix_len, 1, 4)
    return _Request(x=torch.zeros(1, 4, 8, 8), state=state)


@pytest.mark.parametrize(
    "lengths, expected",
    [
        ([], []),
        ([10, 11, 12], [[10, 11, 12]]),          # same phase -> one forward
        ([0, 60], [[0], [60]]),                  # restarted game must not drag
        ([58, 60, 1, 2], [[1, 2], [58, 60]]),    # two phases -> two forwards
        ([0, 0, 0], [[0, 0, 0]]),                # all fresh -> nothing to waste
    ],
)
def test_group_by_prefix_bounds_padding_waste(lengths, expected):
    from deepnash_rbc.infer import _prefix_len, group_by_prefix

    groups = group_by_prefix([_req(n) for n in lengths])
    assert [[_prefix_len(r.state) for r in g] for g in groups] == expected
    from deepnash_rbc.infer import _STEP_BASE_COST

    for group in groups:
        lens = [_prefix_len(r.state) for r in group]
        padded = len(lens) * (_STEP_BASE_COST + max(lens))
        real = sum(_STEP_BASE_COST + n for n in lens)
        assert padded <= real * 1.25


def test_prefix_grouping_only_applies_to_prefix_state_archs():
    """GRU's acting state is fixed-size, so grouping would only cost batch size."""
    gru, _ = _net("gru")
    transformer, _ = _net("transformer")
    assert gru.state_is_prefix is False
    assert transformer.state_is_prefix is True

    batcher = InferenceBatcher(gru, torch.device("cpu"), max_batch=4)
    assert batcher._state_is_prefix is False
    batcher.stop()


def test_batcher_propagates_forward_errors():
    """A failing forward must raise in the game thread, not hang it."""

    class Boom:
        is_temporal = False

        def __call__(self, x):
            raise ValueError("boom")

    batcher = InferenceBatcher(Boom(), torch.device("cpu"), max_batch=1)
    with batcher.slot(), pytest.raises(ValueError, match="boom"):
        BatchedNet(batcher)(torch.zeros(1, 4, 8, 8))
    batcher.stop()
