# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reconstruct a clean, versioned metrics log from the legacy flat file.

The old (pre-versioned) trainer appended every run and resume to a single
``checkpoints/metrics.jsonl``, so it interleaves false starts and resume segments
that overlap wherever a run logged past its last saved checkpoint. This script
applies the same rule the trainer now applies on resume -- a resume from
checkpoint step X invalidates everything after X from earlier segments -- to emit
one gapless, monotonic run in the new format:

  * aborted/false-start segments and stale post-checkpoint rows are dropped;
  * ``wall_s`` is stitched into a single cumulative clock across resumes, exactly
    as the resume-aware MetricsLogger now records it (per-resume warmup included);
  * every other field is copied verbatim from the original record (so train rows
    stay train-shaped and skill rows skill-shaped -- no column unioning).

Usage:
    uv run python notebooks/clean_legacy_metrics.py \
        checkpoints/metrics.jsonl checkpoints/v0.3.0/metrics_v0.3.0.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _step(rec: dict) -> int | None:
    if "step" in rec:
        return int(rec["step"])
    if "iter" in rec:
        return int(rec["iter"])
    return None


def clean(src: str, dst: str) -> None:
    # Pass 1: read minimally (step, wall_s) to find segment boundaries -- a fresh
    # start or resume begins a new segment, marked by wall_s or step going back.
    recs = []  # (wall_s, step)
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            recs.append((float(r["wall_s"]), _step(r), r.get("games_seen")))

    n = len(recs)
    seg_of = [0] * n
    seg = 0
    for i in range(1, n):
        w, s, _ = recs[i]
        pw, ps, _ = recs[i - 1]
        if w < pw or (s is not None and ps is not None and s < ps):
            seg += 1
        seg_of[i] = seg

    n_segs = seg + 1
    seg_base = [None] * n_segs  # resume checkpoint step (= first step - 1)
    seg_first_idx = [None] * n_segs
    for i in range(n):
        sg = seg_of[i]
        if seg_first_idx[sg] is None:
            seg_first_idx[sg] = i
            seg_base[sg] = (recs[i][1] or 1) - 1  # recs[i] = (wall, step, games)

    # A segment's rows survive only up to the smallest resume-base of any LATER
    # segment (a later resume from B truncates everything after B).
    suffix_min = [float("inf")] * n_segs
    running = float("inf")
    for sg in range(n_segs - 1, -1, -1):
        suffix_min[sg] = running
        running = min(running, seg_base[sg])

    # Per-segment cumulative offsets for the counters that reset each process:
    # each surviving segment continues where the previous one's last kept row left
    # off. wall_s -> the resume warmup becomes a real gap in the clock; games_seen
    # -> a single monotonic "trajectories consumed" total. Both match what the
    # resume-aware trainer now records natively. (Both counters rise within a
    # segment, so a segment's last-kept value is its max over kept rows.)
    kept_max_wall = [0.0] * n_segs
    kept_max_games = [0] * n_segs
    for i in range(n):
        sg = seg_of[i]
        w, s, g = recs[i]
        if s is not None and s <= suffix_min[sg]:
            kept_max_wall[sg] = max(kept_max_wall[sg], w)
            if g is not None:
                kept_max_games[sg] = max(kept_max_games[sg], int(g))

    def _offsets(kept_max):
        off = [0] * n_segs
        acc = 0
        for sg in range(n_segs):
            off[sg] = acc
            if kept_max[sg] > 0:  # only advance for segments that kept rows
                acc += kept_max[sg]
        return off, acc

    offset, acc = _offsets(kept_max_wall)
    games_offset, _ = _offsets(kept_max_games)

    # Pass 2: stream the original file again, keep survivors in file order (= step
    # order), rewrite wall_s to the stitched cumulative value, copy the rest.
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    last_step = -1
    with open(src) as fin, open(dst, "w") as fout:
        for i, line in enumerate(fin):
            sg = seg_of[i]
            w, s, _ = recs[i]
            if s is None or s > suffix_min[sg]:
                dropped += 1
                continue
            rec = json.loads(line)
            rec["wall_s"] = round(w + offset[sg], 1)
            if "games_seen" in rec:
                rec["games_seen"] = int(rec["games_seen"]) + games_offset[sg]
            fout.write(json.dumps(rec) + "\n")
            kept += 1
            last_step = max(last_step, s)

    print(
        f"{src} -> {dst}\n"
        f"  segments: {n_segs}  kept: {kept:,}  dropped: {dropped:,}\n"
        f"  final step: {last_step:,}  total wall: {acc / 3600:.1f} h"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    clean(sys.argv[1], sys.argv[2])
