"""Recurring training window (sound / electricity).

Training should only run when the rig is allowed to be loud and drawing power.
``Config.train`` defines a recurring window; outside it the trainers idle -- the
async actors are paused and the learner sleeps -- until the window reopens.

The window is ``[train_start_hour, train_stop_hour)`` on the configured weekdays
and may wrap past midnight (start > stop), so the default 19:00->06:00 on Mon-Fri
means "train on weekday nights": Friday night runs into Saturday 06:00, then the
weekend is idle. Hour granularity matches the on-the-hour schedule.

Set ``DEEPNASH_IGNORE_IDLE=1`` to bypass the schedule for a one-off run without
editing the config.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .config import Config

_TRUE = {"1", "true", "yes", "on"}


def _override() -> bool:
    return os.environ.get("DEEPNASH_IGNORE_IDLE", "").strip().lower() in _TRUE


def _in_window(cfg: "Config", now: datetime) -> bool:
    """Raw schedule test (ignores the master switch / env override)."""
    start = cfg.train.train_start_hour
    stop = cfg.train.train_stop_hour
    days = set(cfg.train.train_days)
    wd, h = now.weekday(), now.hour
    if start > stop:  # window wraps midnight, e.g. 19 -> 6
        # evening half on a listed day, or morning half of a window that opened
        # the previous (listed) day
        return (wd in days and h >= start) or ((wd - 1) % 7 in days and h < stop)
    return wd in days and start <= h < stop


def training_allowed(cfg: "Config", now: Optional[datetime] = None) -> bool:
    """True if training may run now (schedule disabled, overridden, or in window)."""
    if not cfg.train.idle_schedule or _override():
        return True
    return _in_window(cfg, now or datetime.now())


def next_window_start(
    cfg: "Config", now: Optional[datetime] = None
) -> Optional[datetime]:
    """Next hour boundary at which the window reopens (for human-readable logs)."""
    now = now or datetime.now()
    probe = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    for _ in range(24 * 8):  # at most ~a week out
        if _in_window(cfg, probe):
            return probe
        probe += timedelta(hours=1)
    return None


def wait_until_allowed(
    cfg: "Config",
    log: Callable[[str], None] = print,
    poll: float = 60.0,
    on_idle: Optional[Callable[[], None]] = None,
    on_resume: Optional[Callable[[], None]] = None,
) -> bool:
    """Block until training is allowed; return True iff it actually idled.

    ``on_idle`` runs once when entering the idle state and ``on_resume`` once
    just before returning, so callers can pause/unpause workers. The polling
    sleep keeps it responsive to KeyboardInterrupt.
    """
    if training_allowed(cfg):
        return False
    if on_idle is not None:
        on_idle()
    nxt = next_window_start(cfg)
    until = f" until {nxt:%a %H:%M}" if nxt else ""
    log(f"[schedule] outside training window -> idling{until}")
    while not training_allowed(cfg):
        time.sleep(poll)
    if on_resume is not None:
        on_resume()
    log("[schedule] training window open -> resuming")
    return True
