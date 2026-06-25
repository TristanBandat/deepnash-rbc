"""The fast (vectorized) learner step must be numerically identical to the legacy
per-trajectory one. They differ only in *how* the work is scheduled -- batched
v-trace vs a per-trajectory loop, scatter-built legal masks vs a per-row loop,
fused scalar readback vs four .item() calls -- not in the math.

We prove it end to end: from an identical initial network + optimizer, feeding
identical trajectories, one update step must leave both networks (and the logged
scalars) equal. This is the guard that lets us flip on `fast_learner` without
touching model performance. Runs on CPU; no GPU required.
"""

from __future__ import annotations

import copy

import torch

from deepnash_rbc.config import (
    Config,
    EncodingConfig,
    NetworkConfig,
    RNaDConfig,
    TrainConfig,
)
from deepnash_rbc.network import DeepNashNet
from deepnash_rbc.rnad.trainer import RNaDLearner
from deepnash_rbc.selfplay import collect


def _tiny_cfg(full_action: bool) -> Config:
    # short games + a tiny net so the whole equivalence check runs in ~seconds
    return Config(
        encoding=EncodingConfig(history=4),
        network=NetworkConfig(channels=16, blocks=1, value_hidden=16),
        rnad=RNaDConfig(iteration_steps=1000, full_action_neurd=full_action),
        train=TrainConfig(device="cpu", games_per_iter=1, seconds_per_player=12.0),
    )


def _make_trajectories(cfg: Config):
    device = torch.device("cpu")
    torch.manual_seed(0)
    net = DeepNashNet(cfg.encoding, cfg.network).to(device)
    # a few games -> trajectories of differing length, exercising v-trace padding
    trajs = [t for t in collect(net, device, cfg, n_games=3) if len(t) > 0]
    assert trajs, "no trajectories produced"
    assert len({len(t) for t in trajs}) > 1 or len(trajs) > 1
    return trajs


def _run_one_step(cfg: Config, trajs, fast: bool):
    device = torch.device("cpu")
    torch.manual_seed(1)
    net = DeepNashNet(cfg.encoding, cfg.network).to(device)
    learner = RNaDLearner(cfg, copy.deepcopy(net), device, fast=fast)
    stats = learner.update(trajs)
    return learner.net, stats


def _assert_equivalent(full_action: bool):
    cfg = _tiny_cfg(full_action)
    trajs = _make_trajectories(cfg)

    net_legacy, s_legacy = _run_one_step(cfg, trajs, fast=False)
    net_fast, s_fast = _run_one_step(cfg, trajs, fast=True)

    for k in ("loss", "policy_loss", "value_loss", "entropy"):
        assert abs(s_legacy[k] - s_fast[k]) < 1e-5, (
            f"{k} differs (full_action={full_action}): "
            f"legacy={s_legacy[k]} fast={s_fast[k]}"
        )

    for (n, p_leg), (_, p_fast) in zip(
        net_legacy.named_parameters(), net_fast.named_parameters()
    ):
        assert torch.allclose(p_leg, p_fast, atol=1e-5, rtol=1e-4), (
            f"param {n} diverged after one step (full_action={full_action}), "
            f"max abs diff {(p_leg - p_fast).abs().max().item():.2e}"
        )


def test_fast_matches_legacy_all_actions_neurd():
    _assert_equivalent(full_action=True)


def test_fast_matches_legacy_single_sample_neurd():
    _assert_equivalent(full_action=False)
