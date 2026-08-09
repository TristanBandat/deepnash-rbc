"""The bench must tune the config a run actually uses, against the right target.

Two failure modes this pins down, both of which produce confidently wrong advice:

  1. Benchmarking the library defaults instead of the run's own manifest -- a
     different architecture entirely, so every number is about a model nobody is
     training.
  2. Treating surplus actor throughput as waste when the run is learner-bound.
     Past the point where the GPU is fed, extra trajectories buy a shorter
     off-policy horizon, and that is a real objective, not slack to reclaim.
"""

from __future__ import annotations

import json

import pytest

from deepnash_rbc.bench import (
    _horizon,
    _reuse,
    apply_to_config,
    describe_config,
    recommend,
)
from deepnash_rbc.config import Config, config_from_dict


# -- loading a run's own config ----------------------------------------------
def test_config_from_dict_round_trips():
    from dataclasses import asdict

    cfg = Config()
    cfg.network.arch = "transformer"
    cfg.network.mixer_dim = 384
    cfg.network.mixer_layers = 6
    cfg.encoding.history = 1
    cfg.train.batch_trajectories = 64

    restored = config_from_dict(json.loads(json.dumps(asdict(cfg))))
    assert describe_config(restored) == describe_config(cfg)
    assert restored.network.mixer_dim == 384
    assert restored.train.batch_trajectories == 64


def test_config_from_dict_restores_tuples():
    """JSON has no tuples; fields like eval_opponents/xlstm_slstm_at must not
    come back as lists or downstream tuple handling breaks."""
    data = {"train": {"eval_opponents": ["random", "trout"]},
            "network": {"xlstm_slstm_at": [1, 3]}}
    cfg = config_from_dict(data)
    assert cfg.train.eval_opponents == ("random", "trout")
    assert cfg.network.xlstm_slstm_at == (1, 3)


def test_config_from_dict_tolerates_old_and_unknown_fields():
    """A manifest pinned before actor_games existed must still load, and a
    manifest from a newer schema must not crash the bench."""
    cfg = config_from_dict({
        "train": {"async_actors": 24, "not_a_real_field": 5},
        "network": {"arch": "gru"},
    })
    assert cfg.train.async_actors == 24
    assert cfg.train.actor_games == Config().train.actor_games  # default kept
    assert cfg.network.arch == "gru"


# -- off-policy metrics -------------------------------------------------------
def test_horizon_and_reuse_match_the_documented_formulas():
    cfg = Config()
    cfg.train.buffer_capacity = 4096
    cfg.train.batch_trajectories = 64
    # v0.59.0's measured operating point
    assert _horizon(cfg, 14.35) == pytest.approx(4096 / 14.35)
    assert _reuse(cfg, 2.46, 35.29) == pytest.approx(64 * 2.46 / 35.29)


def test_horizon_is_linear_in_actor_throughput():
    """Doubling fresh/step must halve the horizon -- that is the whole reason
    extra actors are worth buying on a learner-bound run."""
    cfg = Config()
    cfg.train.buffer_capacity = 4096
    assert _horizon(cfg, 20.0) == pytest.approx(_horizon(cfg, 10.0) / 2)


def _row(actors, games, fresh, *, steps=2.5, horizon=None, q=0.2, wait=0.05):
    return {
        "actors": actors, "actor_games": games, "concurrent_games": actors * games,
        "fresh_per_step": fresh, "steps_per_s": steps, "traj_per_s": fresh * steps,
        "horizon_steps": horizon if horizon is not None else 4096 / fresh,
        "reuse": 64 * steps / (fresh * steps), "buffer_capacity": 4096,
        "q_occ_max": q, "gpu_busy_frac": 0.8, "data_wait_frac": wait,
    }


# -- recommendation -----------------------------------------------------------
def test_without_horizon_target_picks_the_cheapest_that_keeps_up():
    """Old behavior: surplus is waste, so take the smallest actor footprint."""
    rows = [_row(4, 1, 3.0), _row(16, 1, 14.0), _row(30, 1, 26.0)]
    rec = recommend(rows, fresh_target=1.0)
    assert rec["async_actors"] == 4
    assert rec["actor_bound"] is False


def test_horizon_target_makes_surplus_throughput_worth_buying():
    """With a horizon target the same grid picks a bigger config, because the
    extra trajectories are buying freshness rather than being discarded."""
    rows = [_row(4, 1, 3.0), _row(16, 1, 14.0), _row(30, 1, 26.0)]
    cheap = recommend(rows, fresh_target=1.0)
    fresh = recommend(rows, fresh_target=1.0, horizon_target=200.0)
    # horizons: 4 actors -> 1365 steps, 16 -> 293, 30 -> 158. Only 30 clears 200.
    assert cheap["async_actors"] == 4
    assert fresh["async_actors"] == 30
    assert fresh["horizon_steps"] <= 200


def test_cheapest_config_wins_ties_on_actor_processes_not_games():
    """8x1 and 2x4 both run 8 games; the 2-process one costs 6 fewer cores."""
    rows = [_row(8, 1, 14.0), _row(2, 4, 14.0)]
    rec = recommend(rows, fresh_target=1.0)
    assert (rec["async_actors"], rec["actor_games"]) == (2, 4)


def test_actor_bound_grid_reports_the_knee():
    rows = [_row(4, 1, 0.20, wait=0.6), _row(8, 1, 0.38, wait=0.5),
            _row(16, 1, 0.39, wait=0.5)]
    rec = recommend(rows, fresh_target=1.0)
    assert rec["actor_bound"] is True
    assert rec["async_actors"] == 8            # 16 buys ~nothing over 8
    assert "actor-bound" in rec["reason"]


def test_unreachable_horizon_suggests_the_free_lever():
    """buffer_capacity moves the horizon without spending a single core, so the
    advice must mention it rather than just failing."""
    rows = [_row(30, 1, 26.0)]
    rec = recommend(rows, fresh_target=1.0, horizon_target=50.0)
    assert rec["actor_bound"] is False
    assert "buffer_capacity" in rec["reason"]


def test_saturated_queue_is_not_recommended():
    """A config whose queue is pegged is dropping games; freshness there is a lie."""
    rows = [_row(30, 1, 26.0, q=0.99), _row(16, 1, 14.0, q=0.3)]
    rec = recommend(rows, fresh_target=1.0)
    assert rec["async_actors"] == 16


# -- --apply ------------------------------------------------------------------
def test_apply_to_config_writes_both_knobs(tmp_path, monkeypatch):
    from deepnash_rbc import config as config_mod

    src = tmp_path / "config.py"
    src.write_text(
        "    async_actors: int = 16  # persistent CPU self-play workers\n"
        "    actor_games: int = 1\n"
    )
    monkeypatch.setattr(config_mod, "__file__", str(src))
    apply_to_config(30, 4)
    text = src.read_text()
    assert "async_actors: int = 30  # persistent CPU self-play workers" in text
    assert "actor_games: int = 4" in text


def test_apply_to_config_leaves_file_untouched_when_field_missing(tmp_path, monkeypatch):
    from deepnash_rbc import config as config_mod

    src = tmp_path / "config.py"
    original = "    async_actors: int = 16\n"  # no actor_games line
    src.write_text(original)
    monkeypatch.setattr(config_mod, "__file__", str(src))
    with pytest.raises(RuntimeError, match="actor_games"):
        apply_to_config(30, 4)
