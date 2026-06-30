"""Tests for the generic ``--set PATH=VALUE`` config override."""

from __future__ import annotations

import pytest

from deepnash_rbc.cli import config_from_args


def test_set_coerces_by_field_type():
    cfg = config_from_args(
        [
            "--set", "encoding.history=4",       # int
            "--set", "rnad.eta=0.35",            # float
            "--set", "rnad.full_action_neurd=false",  # bool
            "--set", "train.total_iters=200000",
        ]
    )
    assert cfg.encoding.history == 4 and isinstance(cfg.encoding.history, int)
    assert cfg.rnad.eta == 0.35
    assert cfg.rnad.full_action_neurd is False
    assert cfg.train.total_iters == 200000


def test_set_tuple_fields():
    cfg = config_from_args(
        ["--set", "train.train_days=0,1,2", "--set", "train.eval_opponents=random,trout"]
    )
    assert cfg.train.train_days == (0, 1, 2)  # ints
    assert cfg.train.eval_opponents == ("random", "trout")  # strs


def test_set_optional_field_none_and_value():
    assert config_from_args(["--set", "train.metrics_path=none"]).train.metrics_path is None
    cfg = config_from_args(["--set", "train.resume=auto"])
    assert cfg.train.resume == "auto"


def test_set_overrides_named_flag():
    # --set is applied after the named flags, so it wins
    cfg = config_from_args(["--history", "8", "--set", "encoding.history=2"])
    assert cfg.encoding.history == 2


@pytest.mark.parametrize(
    "bad",
    [
        "encoding.history",          # missing '='
        "nope.field=1",              # unknown section
        "train.does_not_exist=1",    # unknown field
        "encoding=1",                # targets a section, not a leaf
    ],
)
def test_set_rejects_bad_specs(bad):
    with pytest.raises(ValueError):
        config_from_args(["--set", bad])
