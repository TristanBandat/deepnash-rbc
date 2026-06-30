# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.2", "tensorboard>=2.14"]
# ///
"""Convert a metrics JSONL into a TensorBoard run.

Replays the records into event files so the curves the trainer logged can be
browsed in TensorBoard. Scalars are written at their ``step`` (the global-step
axis) with ``wall_s`` as the event walltime, so the RELATIVE-time axis is
meaningful too (works on the cleaned, wall-stitched file).

Every numeric field is logged -- nothing is dropped, and fields added to the
metrics format in the future are picked up automatically. Related families are
overlaid on a single chart via ``add_scalars`` so they can be compared directly:

  * ``loss``               -- total / policy / value on one chart
  * ``train/entropy``      -- policy entropy
  * ``diag/*``             -- buffer, drained, games_seen, iteration
  * ``skill/win_rate``     -- per opponent (random, attacker, ...)
  * ``skill/draw_rate``    -- per opponent
  * ``skill/game_length``  -- mean plies, per opponent
  * ``skill/n_games``      -- eval games per opponent
  * ``throughput/*``       -- steps/s and games/s, from step & games_seen deltas

Usage:
    uv run --group notebooks python notebooks/metrics_to_tensorboard.py \
        checkpoints/v0.3.0/metrics_v0.3.0.jsonl runs/v0.3.0 [--stride N]

Then:  uv run --group notebooks tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json

from torch.utils.tensorboard import SummaryWriter

# Axis / bookkeeping keys that are not metrics to plot (``steps`` duplicates
# ``step``; ``wall_s``/``step``/``iter`` are the axes; ``type`` is the router).
SKIP_KEYS = {"wall_s", "step", "iter", "steps", "type"}

# Train loss components overlaid on one "loss" chart: metric key -> line name.
LOSS_COMPONENTS = {"loss": "total", "policy_loss": "policy", "value_loss": "value"}

# vs_<opponent><suffix> -> (chart, suffix). No suffix == win rate.
SKILL_SUFFIXES = {"_draw": "draw_rate", "_plies": "game_length", "_n": "n_games"}


def _write_train(w: SummaryWriter, rec: dict, step: int, wall: float) -> None:
    losses = {
        name: float(rec[key])
        for key, name in LOSS_COMPONENTS.items()
        if key in rec
    }
    if losses:
        w.add_scalars("loss", losses, step, walltime=wall)
    # everything else numeric: entropy -> train/, the rest -> diag/
    for key, val in rec.items():
        if key in SKIP_KEYS or key in LOSS_COMPONENTS or isinstance(val, (bool, str)):
            continue
        tag = "train/entropy" if key == "entropy" else f"diag/{key}"
        w.add_scalar(tag, float(val), step, walltime=wall)


def _write_skill(w: SummaryWriter, rec: dict, step: int, wall: float) -> None:
    # group vs_<opponent>* fields into one overlaid chart per family
    charts: dict[str, dict[str, float]] = {}
    for key, val in rec.items():
        if not key.startswith("vs_") or isinstance(val, (bool, str)):
            continue
        rest = key[len("vs_"):]
        chart, opp = "win_rate", rest
        for suffix, name in SKILL_SUFFIXES.items():
            if rest.endswith(suffix):
                chart, opp = name, rest[: -len(suffix)]
                break
        charts.setdefault(chart, {})[opp] = float(val)
    for chart, series in charts.items():
        w.add_scalars(f"skill/{chart}", series, step, walltime=wall)


def _write_throughput(w, rec, step, wall, prev) -> None:
    """Rates between this and the previously written train row. Relies on the
    cleaned file's monotonic counters, so deltas are non-negative (the warmup gap
    after a resume shows as a brief dip -- training really did pause there)."""
    dt = wall - prev["wall"]
    if dt <= 0:
        return
    w.add_scalar("throughput/steps_per_s", (step - prev["step"]) / dt, step, walltime=wall)
    games = rec.get("games_seen")
    if games is not None and prev["games"] is not None:
        w.add_scalar(
            "throughput/games_per_s", (games - prev["games"]) / dt, step, walltime=wall
        )


def convert(src: str, logdir: str, stride: int = 1) -> None:
    writer = SummaryWriter(log_dir=logdir)
    n_train = n_skill = 0
    prev = None  # last written train row: {"step", "wall", "games"}
    with open(src) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            step = rec.get("step", rec.get("iter"))
            if step is None:
                continue
            wall = float(rec.get("wall_s", 0.0))
            kind = rec.get("type")

            if kind == "skill":
                _write_skill(writer, rec, step, wall)  # sparse: always written
                n_skill += 1
            elif kind == "train":
                # 1M train rows; thin by stride (TensorBoard downsamples anyway)
                if stride > 1 and (i % stride):
                    continue
                _write_train(writer, rec, step, wall)
                if prev is not None:
                    _write_throughput(writer, rec, step, wall, prev)
                prev = {"step": step, "wall": wall, "games": rec.get("games_seen")}
                n_train += 1

    writer.flush()
    writer.close()
    print(
        f"{src} -> {logdir}\n"
        f"  train rows written: {n_train:,} (stride={stride})  skill rows: {n_skill:,}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="metrics JSONL")
    p.add_argument("logdir", help="TensorBoard run directory to create")
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="write every Nth train row (default 1 = all; skill rows always all)",
    )
    args = p.parse_args()
    convert(args.src, args.logdir, args.stride)
