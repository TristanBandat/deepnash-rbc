"""Deduplicated observation storage must be lossless.

Steps store only their in-progress frame + a turn index; the trajectory stores
each committed frame once (replay.py), and the learner rebuilds the full
[history * FRAME_CHANNELS, 8, 8] stack per step in collate/_to_device. This
test plays a real self-play game while capturing what ``encoder.tensor()``
emitted at every decision point (exactly what the old format stored per step)
and asserts the reconstruction is bit-identical. Runs on CPU; no GPU required.
"""

from __future__ import annotations

import numpy as np
import torch

from deepnash_rbc.agent import RNaDPlayer
from deepnash_rbc.config import Config, EncodingConfig, NetworkConfig, RNaDConfig, TrainConfig
from deepnash_rbc.network import DeepNashNet
from deepnash_rbc.rnad.trainer import RNaDLearner
from deepnash_rbc.selfplay import collect


def _tiny_cfg() -> Config:
    # history=4 keeps the game fast while still exercising blank left-padding
    # (turn < history-1) AND full windows (turn >= history-1) in one game
    return Config(
        encoding=EncodingConfig(history=4),
        network=NetworkConfig(channels=16, blocks=1, value_hidden=16),
        rnad=RNaDConfig(iteration_steps=1000),
        train=TrainConfig(device="cpu", games_per_iter=1, seconds_per_player=12.0),
    )


def test_reconstruction_matches_encoder_tensor(monkeypatch):
    cfg = _tiny_cfg()
    device = torch.device("cpu")
    torch.manual_seed(0)
    net = DeepNashNet(cfg.encoding, cfg.network).to(device)

    # capture the full stack at every decision point, attached to the Step it
    # belongs to (capture per player instance, so both colors are covered)
    orig_record = RNaDPlayer._record

    def record_with_stack(self, head, legal, action, logp):
        stack = self.encoder.tensor().astype(np.uint8)
        orig_record(self, head, legal, action, logp)
        self.trajectory.steps[-1]._full_stack = stack

    monkeypatch.setattr(RNaDPlayer, "_record", record_with_stack)

    trajs = [t for t in collect(net, device, cfg, n_games=1) if len(t) > 0]
    assert trajs, "no trajectories produced"
    steps = [s for t in trajs for s in t.steps]
    assert any(s.turn >= cfg.encoding.history - 1 for s in steps), (
        "game too short to exercise a full history window"
    )

    learner = RNaDLearner(cfg, net, device)
    col = learner.collate(trajs)
    obs, *_ = learner._to_device(col)

    expected = torch.from_numpy(
        np.stack([s._full_stack for s in steps])
    ).float()
    assert obs.shape == expected.shape, f"{obs.shape} != {expected.shape}"
    assert torch.equal(obs, expected), (
        f"reconstructed stacks differ from encoder output in "
        f"{(obs != expected).sum().item()} elements"
    )
