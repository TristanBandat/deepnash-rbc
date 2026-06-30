"""Tests for the nightly training-window schedule (sound/electricity gating)."""

from __future__ import annotations

from datetime import datetime

import pytest

from deepnash_rbc.config import Config
from deepnash_rbc.schedule import next_window_start, training_allowed


def _cfg(**over):
    cfg = Config()
    for k, v in over.items():
        setattr(cfg.train, k, v)
    return cfg


# Default window: 19:00 -> 06:00, nights starting Mon-Fri (Mon=0).
# 2024-01-01 is a Monday, so weekday() maps cleanly onto that week.
MON, TUE, WED, THU, FRI, SAT, SUN = (datetime(2024, 1, d) for d in range(1, 8))


@pytest.mark.parametrize(
    "when, allowed",
    [
        (MON.replace(hour=20), True),    # Mon evening -> train
        (MON.replace(hour=10), False),   # Mon daytime -> idle
        (TUE.replace(hour=2), True),     # Tue early AM = Mon-night tail -> train
        (TUE.replace(hour=6), False),    # exactly 06:00 -> stop
        (FRI.replace(hour=23), True),    # Fri night -> train
        (SAT.replace(hour=2), True),     # Sat early AM = Fri-night tail -> train
        (SAT.replace(hour=20), False),   # Sat evening -> idle (weekend)
        (SUN.replace(hour=2), False),    # Sun early AM -> idle (no Sat-night run)
        (SUN.replace(hour=23), False),   # Sun night -> idle
    ],
)
def test_default_window(when, allowed):
    assert training_allowed(_cfg(), when) is allowed


def test_master_switch_and_override(monkeypatch):
    daytime = MON.replace(hour=12)  # normally idle
    assert training_allowed(_cfg(), daytime) is False
    # master switch off -> always allowed
    assert training_allowed(_cfg(idle_schedule=False), daytime) is True
    # env override -> always allowed without editing config
    monkeypatch.setenv("DEEPNASH_IGNORE_IDLE", "1")
    assert training_allowed(_cfg(), daytime) is True


def test_same_day_window():
    # non-wrapping window 9->17 on weekdays
    cfg = _cfg(train_start_hour=9, train_stop_hour=17)
    assert training_allowed(cfg, MON.replace(hour=12)) is True
    assert training_allowed(cfg, MON.replace(hour=8)) is False
    assert training_allowed(cfg, MON.replace(hour=17)) is False
    assert training_allowed(cfg, SAT.replace(hour=12)) is False


def test_next_window_start_from_daytime():
    # Monday noon -> next opening is Monday 19:00
    nxt = next_window_start(_cfg(), MON.replace(hour=12))
    assert nxt == MON.replace(hour=19)
