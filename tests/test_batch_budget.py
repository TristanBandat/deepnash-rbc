"""Padded-frame budget (micro-batching) and length-bucketed sampling.

Both exist because a temporal learner batch is padded to ``[Tmax, B]``: activation
memory scales with ``Tmax * B``, not with the number of real steps, so one outlier
trajectory pads all B columns to its length. RBC game lengths are heavy-tailed, so
without a bound the learner's peak VRAM is set by whichever batch happens to draw
the longest game.

What has to hold:

  * ``max_batch_frames`` must be a pure memory/scheduling change -- the split
    batch's accumulated gradient must equal the unsplit batch's gradient, and the
    logged scalars must match, or the knob silently changes what is trained.
  * ``length_bucket_pool`` must leave every trajectory's marginal probability of
    being sampled unchanged (in particular it must not under-sample long games,
    the ones it exists to isolate) while actually cutting the padding.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from deepnash_rbc.config import (
    Config,
    EncodingConfig,
    NetworkConfig,
    RNaDConfig,
    TrainConfig,
)
from deepnash_rbc.encoding.moves import MOVE_ACTIONS
from deepnash_rbc.network import make_net
from deepnash_rbc.replay import MOVE, SENSE, ReplayBuffer, Step, Trajectory
from deepnash_rbc.rnad.trainer import RNaDLearner

FRAME_CHANNELS = 19


def _cfg(max_batch_frames: int = 0, full_action: bool = True) -> Config:
    return Config(
        encoding=EncodingConfig(history=1),
        network=NetworkConfig(
            arch="gru", channels=8, enc_blocks=1, mixer_dim=8, mixer_layers=1,
            value_hidden=8,
        ),
        rnad=RNaDConfig(iteration_steps=10_000, full_action_neurd=full_action),
        train=TrainConfig(device="cpu", max_batch_frames=max_batch_frames),
    )


def _fake_traj(n_steps: int, rng: np.random.Generator) -> Trajectory:
    """A trajectory of the right *shape* for the learner: alternating sense/move
    decisions, a legal set per step with the taken action inside it. The learner
    never inspects game semantics, only these fields."""
    traj = Trajectory()
    for i in range(n_steps):
        head = SENSE if i % 2 == 0 else MOVE
        n_actions = 64 if head == SENSE else MOVE_ACTIONS
        legal = np.sort(rng.choice(n_actions, size=6, replace=False)).astype(np.int64)
        obs = rng.integers(0, 2, size=(FRAME_CHANNELS, 8, 8), dtype=np.uint8)
        traj.add(Step(
            obs=obs, head=head, legal=legal, action=int(legal[0]),
            behavior_logprob=-float(np.log(len(legal))), turn=i // 2,
        ))
        if head == MOVE:
            traj.frames.append(obs)
    traj.z = 1.0 if n_steps % 2 else -1.0
    return traj


def _grads_after_update(cfg: Config, trajs, net_state) -> tuple[dict, dict]:
    torch.manual_seed(1)
    net = make_net(cfg.encoding, cfg.network)
    net.load_state_dict(net_state)
    learner = RNaDLearner(cfg, net, torch.device("cpu"))
    stats = learner.update(trajs)
    # .grad survives the step (zero_grad runs at the START of the next update),
    # so this is the gradient the optimizer actually consumed
    grads = {n: p.grad.detach().clone() for n, p in learner.net.named_parameters()}
    return grads, stats


@pytest.mark.parametrize("full_action", [True, False])
def test_micro_batching_reproduces_the_unsplit_gradient(full_action):
    rng = np.random.default_rng(0)
    # one outlier plus a spread of short games: the shape that forces a split
    trajs = [_fake_traj(n, rng) for n in (4, 6, 9, 11, 40, 5, 7)]

    base = _cfg(0, full_action)
    torch.manual_seed(1)
    net_state = copy.deepcopy(make_net(base.encoding, base.network).state_dict())

    whole_grads, whole_stats = _grads_after_update(base, trajs, net_state)
    # 40 * 7 = 280 padded frames unsplit; 60 forces several micro-batches
    split_grads, split_stats = _grads_after_update(_cfg(60, full_action), trajs, net_state)

    assert split_stats["micro_batches"] > 1, "budget did not split the batch"
    assert "micro_batches" not in whole_stats, "unsplit step must not log a split"

    for name in whole_grads:
        torch.testing.assert_close(
            split_grads[name], whole_grads[name], rtol=1e-5, atol=1e-7,
            msg=lambda m, n=name: f"gradient differs for {n}:\n{m}",
        )
    for key in ("loss", "policy_loss", "value_loss", "entropy"):
        assert split_stats[key] == pytest.approx(whole_stats[key], rel=1e-5, abs=1e-7)


def test_split_partitions_the_batch_and_respects_the_budget():
    rng = np.random.default_rng(1)
    lengths = [3, 50, 7, 4, 20, 9, 6, 31]
    trajs = [_fake_traj(n, rng) for n in lengths]
    learner = RNaDLearner(_cfg(60), make_net(_cfg().encoding, _cfg().network),
                          torch.device("cpu"))
    col = learner.collate(trajs)
    parts = learner._split_for_budget(col)

    assert len(parts) > 1
    for part in parts:
        padded = max(part.lengths) * len(part.lengths)
        assert padded <= 60 or len(part.lengths) == 1  # a lone long traj can't split
        # the flat per-step rows must still line up with the sub-batch's lengths
        assert part.cur.shape[0] == sum(part.lengths)
        assert part.heads.shape[0] == sum(part.lengths)
        assert len(part.legals) == sum(part.lengths)

    assert sorted(l for p in parts for l in p.lengths) == sorted(lengths)
    assert sum(len(p.trajectories) for p in parts) == len(trajs)


def test_budget_off_is_a_no_op():
    rng = np.random.default_rng(2)
    trajs = [_fake_traj(n, rng) for n in (4, 60, 6)]
    learner = RNaDLearner(_cfg(0), make_net(_cfg().encoding, _cfg().network),
                          torch.device("cpu"))
    col = learner.collate(trajs)
    assert learner._split_for_budget(col) == [col]


# -- length-bucketed sampling ------------------------------------------------
class _StubTraj:
    """The buffer only ever asks a trajectory for its length."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n


def _heavy_tailed_buffer(rng, size=512) -> tuple[ReplayBuffer, list]:
    # lognormal-ish: mostly short games, a thin tail of very long ones -- the RBC
    # shape (median ~26 steps, p99.9 ~1400 in the public server archive)
    lengths = np.clip(rng.lognormal(3.3, 1.0, size).astype(int), 2, 4000).tolist()
    buf = ReplayBuffer(capacity=size)
    for n in lengths:
        buf.add(_StubTraj(n))
    return buf, lengths


def test_bucketing_keeps_marginals_uniform_across_lengths():
    rng = np.random.default_rng(3)
    buf, lengths = _heavy_tailed_buffer(rng)
    order = np.argsort(lengths)
    shortest, longest = order[:128], order[-128:]

    import random
    random.seed(0)
    counts = np.zeros(len(lengths))
    draws = 4000
    for _ in range(draws):
        for traj in buf.sample(16, pool=4):
            counts[buf._buf.index(traj)] += 1

    expected = draws * 16 / len(lengths)
    # the bias this must not have: long games sampled less often than short ones
    assert counts[longest].mean() == pytest.approx(expected, rel=0.08)
    assert counts[shortest].mean() == pytest.approx(expected, rel=0.08)
    assert counts.sum() == draws * 16


def test_bucketing_cuts_the_padded_batch():
    rng = np.random.default_rng(4)
    buf, _ = _heavy_tailed_buffer(rng)

    import random
    random.seed(0)

    def mean_padded(pool):
        tot = 0.0
        for _ in range(500):
            batch = buf.sample(64, pool=pool)
            tot += max(len(t) for t in batch) * len(batch)
        return tot / 500

    uniform, bucketed = mean_padded(1), mean_padded(8)
    assert bucketed < uniform / 3, f"uniform {uniform:.0f} -> bucketed {bucketed:.0f}"


def test_pool_one_is_plain_uniform_sampling():
    import random
    buf = ReplayBuffer(capacity=64)
    for n in range(1, 33):
        buf.add(_StubTraj(n))

    random.seed(7)
    a = buf.sample(8)
    random.seed(7)
    b = buf.sample(8, pool=1)
    assert [len(t) for t in a] == [len(t) for t in b]
    assert len({id(t) for t in a}) == 8  # distinct trajectories, as before
