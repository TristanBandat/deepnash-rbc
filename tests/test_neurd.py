"""Tests for the all-actions NeuRD policy update and its ablation plumbing.

Covers the three things that make the full_action_neurd ablation trustworthy:
  1. the all-actions loss math (the pi-weighted advantage is a true advantage,
     illegal logits get no gradient, the beta-clip thresholding fires);
  2. the CLI/env override that selects the flag and redirects output dirs;
  3. that BOTH flag values run a real end-to-end learner step.
"""

from __future__ import annotations

import torch

from deepnash_rbc.checkpoints import metrics_path
from deepnash_rbc.cli import config_from_args
from deepnash_rbc.config import (
    Config,
    EncodingConfig,
    NetworkConfig,
    RNaDConfig,
    TrainConfig,
)
from deepnash_rbc.network import DeepNashNet
from deepnash_rbc.rnad.neurd import all_actions_neurd_loss
from deepnash_rbc.rnad.trainer import RNaDLearner
from deepnash_rbc.selfplay import collect


def _make_inputs(seed: int = 0):
    """A small legal-action batch + a backward-ready logits tensor."""
    torch.manual_seed(seed)
    n, H = 4, 6
    logits = torch.randn(n, H, requires_grad=True)
    legal = torch.rand(n, H) > 0.3
    taken = torch.tensor([0, 1, 2, 3])
    legal[:, 0] = True  # guarantee >=1 legal action
    for i, t in enumerate(taken):
        legal[i, t] = True
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(legal, logits, torch.full_like(logits, neg))
    logp_all = torch.log_softmax(masked, dim=1)
    adv = torch.tensor([1.5, -2.0, 0.7, -0.3])
    return logits, logp_all, legal, taken, adv


def test_pi_weighted_advantage_sums_to_zero():
    """E_pi[adv_act] == 0 per row: the baseline makes it a proper advantage."""
    logits, logp_all, legal, taken, adv = _make_inputs()
    pi = logp_all.exp().detach()
    rows = torch.arange(logits.shape[0])
    qrel = torch.zeros_like(pi)
    qrel[rows, taken] = adv
    baseline = (pi[rows, taken] * adv).unsqueeze(1)
    adv_act = (qrel - baseline).masked_fill(~legal, 0.0)
    assert torch.allclose((pi * adv_act).sum(1), torch.zeros(4), atol=1e-6)


def test_illegal_logits_get_no_gradient():
    logits, logp_all, legal, taken, adv = _make_inputs()
    loss = all_actions_neurd_loss(logits, logp_all, legal, taken, adv, beta=2.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[~legal] == 0).all()


def test_gradient_equals_negative_per_action_advantage():
    """With a large beta (no clipping), d loss / d logit_a == -adv_act[a] on every
    legal action, where adv_act is the pi-baselined advantage. In particular the
    taken logit's gradient is -adv*(1-pi(taken)), so descent pushes it in the
    advantage's direction; non-taken legal logits move the opposite way."""
    logits, logp_all, legal, taken, adv = _make_inputs()
    loss = all_actions_neurd_loss(logits, logp_all, legal, taken, adv, beta=1e9)
    loss.backward()

    pi = logp_all.exp().detach()
    rows = torch.arange(logits.shape[0])
    qrel = torch.zeros_like(pi)
    qrel[rows, taken] = adv
    baseline = (pi[rows, taken] * adv).unsqueeze(1)
    adv_act = (qrel - baseline).masked_fill(~legal, 0.0)

    assert torch.allclose(logits.grad[legal], -adv_act[legal], atol=1e-5)
    # taken-logit gradient points opposite to its advantage (descent follows adv)
    nonzero = adv != 0
    assert (logits.grad[rows[nonzero], taken[nonzero]] * adv[nonzero] < 0).all()


def test_beta_clip_zeroes_runaway_gradient():
    """A logit already past +beta with a positive advantage gets clipped to 0."""
    logits = torch.tensor([[5.0, 0.0]], requires_grad=True)
    legal = torch.tensor([[True, True]])
    logp_all = torch.log_softmax(logits.detach(), dim=1)
    taken = torch.tensor([0])
    adv = torch.tensor([3.0])  # positive -> would push the already-large logit further
    loss = all_actions_neurd_loss(logits, logp_all, legal, taken, adv, beta=2.5)
    loss.backward()
    assert logits.grad[0, 0] == 0.0


# ---- CLI / env override plumbing ----------------------------------------


def test_cli_selects_flag_and_redirects_dirs():
    cfg = config_from_args(
        ["--no-full-action-neurd", "--checkpoint-dir", "runs/single"]
    )
    assert cfg.rnad.full_action_neurd is False
    assert cfg.train.checkpoint_dir == "runs/single"
    assert cfg.train.metrics_path == metrics_path("runs/single")

    cfg2 = config_from_args(["--full-action-neurd"])
    assert cfg2.rnad.full_action_neurd is True


def test_explicit_metrics_path_overrides_derived():
    cfg = config_from_args(
        ["--checkpoint-dir", "runs/a", "--metrics-path", "elsewhere/m.jsonl"]
    )
    assert cfg.train.metrics_path == "elsewhere/m.jsonl"


def test_env_var_sets_flag_and_cli_wins(monkeypatch):
    monkeypatch.setenv("DEEPNASH_FULL_ACTION_NEURD", "false")
    assert config_from_args([]).rnad.full_action_neurd is False
    # explicit CLI flag overrides the env var
    assert config_from_args(["--full-action-neurd"]).rnad.full_action_neurd is True


def test_no_flag_keeps_config_default():
    # default config has the flag True; bare args must not change it
    assert config_from_args([]).rnad.full_action_neurd == Config().rnad.full_action_neurd


# ---- both trainer paths run end to end ----------------------------------


def _tiny_cfg(full_action: bool) -> Config:
    return Config(
        encoding=EncodingConfig(history=4),
        network=NetworkConfig(channels=16, blocks=1, value_hidden=16),
        rnad=RNaDConfig(iteration_steps=2, full_action_neurd=full_action),
        train=TrainConfig(device="cpu", games_per_iter=1, seconds_per_player=20.0),
    )


def test_both_flag_paths_run_a_learner_step():
    for flag in (True, False):
        cfg = _tiny_cfg(flag)
        device = torch.device("cpu")
        torch.manual_seed(0)
        net = DeepNashNet(cfg.encoding, cfg.network).to(device)
        trajs = collect(net, device, cfg, n_games=1)
        assert trajs, "no trajectories produced"
        learner = RNaDLearner(cfg, net, device)
        before = next(net.parameters()).clone()
        out = learner.update([t for t in trajs if len(t) > 0])
        after = next(net.parameters())
        assert out and torch.isfinite(torch.tensor(out["loss"]))
        assert not torch.allclose(before, after), f"weights unchanged (flag={flag})"
