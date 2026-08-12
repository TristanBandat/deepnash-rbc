"""Softmax analysis of the sense head -- what the *distribution* looks like.

Every other sense analysis in this repo looks at sensing through the squares that
were actually sensed: ``make_experiment_figures.fig_senses`` counts the sampled
choice per ladder game, the ``sense_behavior`` notebook averages the policy over
the opening senses.  Both throw away the shape of the distribution the choice was
drawn from -- and in an R-NaD agent that shape *is* the object of interest, because
the algorithm converges to a mixed strategy, not to a best square.

This module queries the checkpoint directly: it plays games, and at every sense
decision stores the full 64-way masked softmax the network emitted, together with
the context it was emitted in (color, which sense of the game it was, whether the
opponent had just captured a piece and where).  From those vectors it computes:

  * **Mixedness.**  Shannon entropy per decision, in bits, against the 6-bit
    uniform-over-64 ceiling; the implied effective support 2^H; top-1 and top-5
    mass; and the concentration curve (cumulative mass over rank-sorted squares).

  * **State dependence.**  With pi-bar the mean of the per-decision distributions,

        H(pi-bar) = E[H(pi_d)] + I(context ; square)

    -- the aggregate heatmap is *at least* as spread out as a single decision, and
    the gap I is exactly the mutual information between the decision context and
    the square sensed.  It separates a genuinely state-conditional sense head from
    one that has memorized a single mixed prior and replays it every turn.  A
    counting analysis cannot see this difference at all: both look identical once
    the choices are pooled.

  * **Geometry.**  The sensed square is the *center* of a 3x3 window, so mass on
    the border is partly wasted (a corner reveals 4 squares, an interior square
    9).  ``expected_revealed`` = sum_s pi_s |window(s)| scores that directly:
    7.56 under the uniform policy, 9.0 for a policy that never touches the border.
    ``border_mass`` reports the same thing as a share (43.75 % under uniform).

  * **Targeting.**  ``opp_half_mass``, the share of mass on the opponent's four
    ranks (50 % under uniform), and the capture response: when the opponent has
    just captured one of our pieces on square c, how much mass lands on a center
    whose window covers c, against the coverage that square would get by chance.

The output is a ``.npz`` of the raw per-decision vectors plus a ``.json`` summary,
written under ``results/sense_softmax/``, and the thesis figure script reads the
npz.  Collection needs a GPU-ish machine and a few minutes; plotting must not, so
the two stay separate the way the Elo fit and its figures do.

Usage:
    uv run python -m deepnash_rbc.analysis.sense_softmax \\
        --checkpoint checkpoints/v0.14.0/deepnash_async_v0.14.0_70000.pt \\
        --games 200 --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import chess
import numpy as np
import torch
from reconchess import LocalGame, play_local_game

from ..agent import RNaDPlayer
from ..play_session import load_net
from ..replay import SENSE

# ---------------------------------------------------------------- board geometry
# Sensing at square s reveals the 3x3 window centered on s, clipped at the board
# edge. Precompute, for every center, its window and its size -- both the geometry
# metrics and the capture-coverage metric are sums over these.
WINDOW: List[np.ndarray] = []
for _sq in range(64):
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    WINDOW.append(np.array(
        [chess.square(f, r)
         for r in range(max(0, _r - 1), min(8, _r + 2))
         for f in range(max(0, _f - 1), min(8, _f + 2))],
        dtype=np.int64,
    ))
WINDOW_SIZE = np.array([len(w) for w in WINDOW], dtype=np.float64)  # 4, 6 or 9
IS_BORDER = WINDOW_SIZE < 9  # the 28 squares whose window is clipped

# covers[c] = centers whose window contains c. The relation is symmetric
# (cheb(s, c) <= 1), so this is the same table read the other way round.
COVERS: List[np.ndarray] = [
    np.array([s for s in range(64) if c in set(WINDOW[s].tolist())], dtype=np.int64)
    for c in range(64)
]

UNIFORM_BITS = np.log2(64)  # 6.0


# ------------------------------------------------------------------- collection
@dataclass
class SenseRecord:
    """One sense decision: the full masked policy plus the context it saw."""

    game: int
    color: bool          # the deciding player's color (True = White)
    ordinal: int         # 1-based sense number *for that player* in that game
    capture_square: int  # square the opponent just captured on, -1 if none
    sampled: int         # the square actually played
    probs: np.ndarray    # float32[64], sums to 1


class SenseProbe(RNaDPlayer):
    """RNaDPlayer that keeps the sense distribution it sampled from.

    ``choose_sense`` is reimplemented rather than wrapped: calling ``_forward``
    a second time to "read off" the policy would advance a temporal net's
    recurrent state twice per turn and quietly change the very thing measured.
    Sampling still goes through the parent's ``_sample_from`` so the recorded
    policy is exactly the one that was played, deployment threshold included.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.records: List[SenseRecord] = []
        self.game_index: int = 0
        self._pending_capture: int = -1

    def handle_game_start(self, color, board, opponent_name):  # noqa: D102
        super().handle_game_start(color, board, opponent_name)
        self.records = []
        self._pending_capture = -1

    def handle_opponent_move_result(self, captured_my_piece, capture_square):  # noqa: D102
        self._pending_capture = int(capture_square) if captured_my_piece else -1
        super().handle_opponent_move_result(captured_my_piece, capture_square)

    def choose_sense(self, sense_actions, move_actions, seconds_left):  # noqa: D102
        if not sense_actions:
            return None
        _, sense_logits, _ = self._forward()
        legal = np.asarray(sense_actions, dtype=np.int64)
        action, logp = self._sample_from(sense_logits, legal)
        self.records.append(SenseRecord(
            game=self.game_index,
            color=bool(self.color),
            ordinal=len(self.records) + 1,
            capture_square=self._pending_capture,
            sampled=int(action),
            probs=masked_probs(sense_logits, legal),
        ))
        self._record(SENSE, legal, action, logp)
        return int(action)


def masked_probs(logits: torch.Tensor, legal: np.ndarray) -> np.ndarray:
    """The softmax over the legal sense squares, as a dense 64-vector."""
    flat = logits.squeeze(0).float().cpu()
    masked = torch.full_like(flat, float("-inf"))
    idx = torch.from_numpy(legal)
    masked[idx] = flat[idx]
    return torch.softmax(masked, dim=0).numpy().astype(np.float32)


def build_opponent(name: str, net, device, history: int, sample: bool):
    """The other seat. ``self`` shares the net, so one game probes both colors."""
    if name == "self":
        return SenseProbe(net, device, history=history, sample=sample)
    from ..eval import _make_opponent
    return _make_opponent(name)


def collect(
    net,
    device: torch.device,
    history: int,
    games: int,
    opponent: str = "self",
    sample: bool = True,
    seconds_per_player: float = 900.0,
    progress: bool = True,
) -> List[SenseRecord]:
    """Play ``games`` full games and return every sense decision made by the net.

    Against an external bot the probe alternates seats so White and Black are
    sampled evenly; in self-play both seats are the probe and one game yields
    both colors at once.
    """
    records: List[SenseRecord] = []
    t0 = time.time()
    for g in range(games):
        probe = SenseProbe(net, device, history=history, sample=sample)
        probe.game_index = g
        opp = build_opponent(opponent, net, device, history, sample)
        probes = [probe] + ([opp] if isinstance(opp, SenseProbe) else [])
        for p in probes:
            p.game_index = g
        if opponent == "self" or g % 2 == 0:
            white, black = probe, opp
        else:
            white, black = opp, probe
        game = LocalGame(seconds_per_player=seconds_per_player)
        try:
            play_local_game(white, black, game=game)
        except Exception as exc:  # a bot blowing up must not lose the whole run
            print(f"[warn] game {g} aborted: {type(exc).__name__}: {exc}", flush=True)
        for p in probes:
            records.extend(p.records)
        if progress and (g + 1) % max(1, games // 20) == 0:
            rate = (g + 1) / max(1e-9, time.time() - t0)
            print(f"  {g + 1}/{games} games, {len(records)} sense decisions "
                  f"({rate:.2f} games/s)", flush=True)
    return records


# ---------------------------------------------------------------------- metrics
def own_side_down(vec: np.ndarray, color: bool) -> np.ndarray:
    """A 64-vector rotated so the deciding player's back rank is at the bottom.

    Black's board is the 180-degree flip s -> 63-s, the same convention the
    thesis sense heatmap uses, so White and Black panels are comparable.
    """
    v = np.asarray(vec, dtype=np.float64)
    return v if color else v[..., ::-1]


def entropy_bits(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits along the last axis (uniform over 64 = 6)."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return -terms.sum(axis=-1)


def opp_half_mass(p: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Share of mass on the opponent's four ranks, per decision (50 % = uniform)."""
    p = np.asarray(p, dtype=np.float64)
    flipped = np.where(np.asarray(color, bool)[:, None], p, p[:, ::-1])
    return flipped.reshape(-1, 8, 8)[:, 4:, :].sum(axis=(1, 2))


def border_mass(p: np.ndarray) -> np.ndarray:
    """Share of mass on centers whose 3x3 window is clipped (43.75 % = uniform)."""
    return np.asarray(p, dtype=np.float64)[:, IS_BORDER].sum(axis=-1)


def expected_revealed(p: np.ndarray) -> np.ndarray:
    """Expected number of squares a sense reveals: 9 at best, 7.5625 if uniform."""
    return np.asarray(p, dtype=np.float64) @ WINDOW_SIZE


def cumulative_mass(p: np.ndarray) -> np.ndarray:
    """Cumulative mass over rank-sorted squares -- the concentration curve.

    Row k of the result is the mass the k+1 most likely squares carry, which is
    where "the policy is mixed but not uniform" becomes a number a reader can
    check: uniform is the straight line k/64, a deterministic policy is flat 1.
    """
    s = np.sort(np.asarray(p, dtype=np.float64), axis=-1)[..., ::-1]
    return np.cumsum(s, axis=-1)


def capture_coverage(p: np.ndarray, capture_square: np.ndarray) -> np.ndarray:
    """P(the sensed 3x3 window covers square c), per decision with a capture."""
    out = np.empty(len(p), dtype=np.float64)
    for i, c in enumerate(capture_square):
        out[i] = p[i, COVERS[int(c)]].sum()
    return out


def summarize(records: Sequence[SenseRecord], meta: Optional[dict] = None) -> dict:
    """Reduce the raw decisions to the numbers the thesis section talks about."""
    probs = np.stack([r.probs for r in records]).astype(np.float64)
    color = np.array([r.color for r in records], dtype=bool)
    ordinal = np.array([r.ordinal for r in records], dtype=np.int64)
    cap = np.array([r.capture_square for r in records], dtype=np.int64)

    h = entropy_bits(probs)
    pooled = probs.mean(axis=0)
    h_pooled = float(entropy_bits(pooled))
    mi = h_pooled - float(h.mean())  # H(pi-bar) - E[H(pi_d)] >= 0

    cum = cumulative_mass(probs).mean(axis=0)
    oh = opp_half_mass(probs, color)

    out: dict = {
        "meta": dict(meta or {}),
        "n_decisions": int(len(records)),
        "n_games": int(len({r.game for r in records})),
        "entropy_bits_mean": float(h.mean()),
        "entropy_bits_std": float(h.std()),
        "entropy_bits_pooled": h_pooled,
        "entropy_bits_uniform": float(UNIFORM_BITS),
        "state_dependence_bits": float(mi),
        "effective_support_mean": float(np.mean(2.0 ** h)),
        "top1_mass_mean": float(probs.max(axis=1).mean()),
        "top5_mass_mean": float(cum[4]),
        "squares_for_half_mass": int(np.argmax(cum >= 0.5) + 1),
        "squares_for_90_mass": int(np.argmax(cum >= 0.9) + 1),
        "opp_half_mass_mean": float(oh.mean()),
        "border_mass_mean": float(border_mass(probs).mean()),
        "border_mass_uniform": float(IS_BORDER.sum() / 64),
        "expected_revealed_mean": float(expected_revealed(probs).mean()),
        "expected_revealed_uniform": float(WINDOW_SIZE.mean()),
        "argmax_agreement": float(np.mean(probs.argmax(axis=1)
                                          == np.array([r.sampled for r in records]))),
    }

    # per-ordinal profile (the first senses are the ones with a story: White's
    # sense 1 fires before the opponent has ever moved and cannot be informative)
    profile = []
    for o in range(1, int(ordinal.max()) + 1):
        for c in (True, False):
            m = (ordinal == o) & (color == c)
            if m.sum() < 5:
                continue
            profile.append({
                "ordinal": o,
                "color": "white" if c else "black",
                "n": int(m.sum()),
                "entropy_bits": float(h[m].mean()),
                "opp_half_mass": float(oh[m].mean()),
                "border_mass": float(border_mass(probs[m]).mean()),
            })
    out["per_ordinal"] = profile

    # capture response: coverage of the square we were just captured on, against
    # what an arbitrary square would get from the same distributions by chance.
    has_cap = cap >= 0
    if has_cap.any():
        cov = capture_coverage(probs[has_cap], cap[has_cap])
        chance = expected_revealed(probs[has_cap]) / 64.0
        out["capture"] = {
            "n": int(has_cap.sum()),
            "coverage_mean": float(cov.mean()),
            "chance_mean": float(chance.mean()),
            "lift": float(cov.mean() / chance.mean()) if chance.mean() > 0 else float("nan"),
        }
    else:
        out["capture"] = {"n": 0}
    return out


# ------------------------------------------------------------------------- io
def checkpoint_label(path: str | Path) -> str:
    """``.../deepnash_async_v0.14.0_70000.pt`` -> ``v0.14.0_70000``."""
    m = re.search(r"v(\d+\.\d+\.\d+)_(\d+)\.pt$", str(path))
    return f"v{m.group(1)}_{m.group(2)}" if m else Path(path).stem


def save(records: Sequence[SenseRecord], summary: dict, out_dir: Path, label: str) -> Path:
    """Write the raw vectors (npz) and the summary (json); returns the npz path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{label}.npz"
    np.savez_compressed(
        npz,
        probs=np.stack([r.probs for r in records]).astype(np.float32),
        color=np.array([r.color for r in records], dtype=np.bool_),
        ordinal=np.array([r.ordinal for r in records], dtype=np.int16),
        capture_square=np.array([r.capture_square for r in records], dtype=np.int16),
        sampled=np.array([r.sampled for r in records], dtype=np.int16),
        game=np.array([r.game for r in records], dtype=np.int32),
        meta=json.dumps(summary.get("meta", {})),
    )
    (out_dir / f"{label}.json").write_text(json.dumps(summary, indent=1) + "\n")
    return npz


def load_npz(path: str | Path) -> dict:
    """Read back a collection as plain arrays (what the figure script uses)."""
    z = np.load(path, allow_pickle=False)
    return {
        "probs": z["probs"].astype(np.float64),
        "color": z["color"].astype(bool),
        "ordinal": z["ordinal"].astype(int),
        "capture_square": z["capture_square"].astype(int),
        "sampled": z["sampled"].astype(int),
        "game": z["game"].astype(int),
        "meta": json.loads(str(z["meta"])) if "meta" in z else {},
    }


def _print_summary(s: dict) -> None:
    m = s["meta"]
    print(f"\n=== Sense softmax: {m.get('label', '?')} vs {m.get('opponent', '?')} "
          f"({s['n_decisions']} decisions in {s['n_games']} games) ===")
    print(f"  entropy            {s['entropy_bits_mean']:.2f} +- {s['entropy_bits_std']:.2f} bits "
          f"(uniform {s['entropy_bits_uniform']:.2f}); effective support "
          f"{s['effective_support_mean']:.1f} squares")
    print(f"  pooled entropy     {s['entropy_bits_pooled']:.2f} bits -> state dependence "
          f"I = {s['state_dependence_bits']:.2f} bits")
    print(f"  concentration      top-1 {s['top1_mass_mean'] * 100:.1f} %, top-5 "
          f"{s['top5_mass_mean'] * 100:.1f} %, {s['squares_for_half_mass']} squares hold half")
    print(f"  targeting          opponent half {s['opp_half_mass_mean'] * 100:.1f} % (uniform 50)")
    print(f"  geometry           border {s['border_mass_mean'] * 100:.1f} % "
          f"(uniform {s['border_mass_uniform'] * 100:.1f}); reveals "
          f"{s['expected_revealed_mean']:.2f}/9 squares (uniform "
          f"{s['expected_revealed_uniform']:.2f})")
    c = s["capture"]
    if c.get("n"):
        print(f"  capture response   covers the captured square {c['coverage_mean'] * 100:.1f} % "
              f"vs {c['chance_mean'] * 100:.1f} % by chance ({c['lift']:.2f}x, n={c['n']})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m deepnash_rbc.analysis.sense_softmax",
        description="Collect and summarize a checkpoint's sense-policy softmax.")
    p.add_argument("-c", "--checkpoint", required=True, help="path to a .pt checkpoint")
    p.add_argument("-n", "--games", type=int, default=200, help="games to play (default 200)")
    p.add_argument("--opponent", default="self",
                   choices=["self", "random", "attacker", "trout", "mht", "strangefish2"],
                   help="who the probe plays (default self)")
    p.add_argument("--device", default="cuda", help="torch device (default cuda)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true",
                   help="play argmax instead of sampling (the policy is recorded either way)")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default <repo>/results/sense_softmax)")
    p.add_argument("--label", default=None, help="output basename (default <version>_<step>)")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu")
        args.device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    net, enc = load_net(args.checkpoint, device)
    label = args.label or f"{checkpoint_label(args.checkpoint)}_{args.opponent}"
    print(f"Loaded {args.checkpoint} ({type(net).__name__}, history={enc.history}) "
          f"on {device}; {args.games} games vs {args.opponent}", flush=True)

    t0 = time.time()
    records = collect(net, device, enc.history, args.games,
                      opponent=args.opponent, sample=not args.greedy)
    if not records:
        raise SystemExit("no sense decisions recorded -- did every game abort?")

    summary = summarize(records, meta={
        "label": label,
        "checkpoint": str(args.checkpoint),
        "run": checkpoint_label(args.checkpoint),
        "opponent": args.opponent,
        "games": args.games,
        "sample": not args.greedy,
        "seed": args.seed,
        "arch": type(net).__name__,
        "history": enc.history,
        "device": str(device),
        "seconds": round(time.time() - t0, 1),
    })
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(__file__).resolve().parents[3] / "results" / "sense_softmax"
    npz = save(records, summary, out_dir, label)
    _print_summary(summary)
    print(f"\nwrote {npz} and {npz.with_suffix('.json')}")


if __name__ == "__main__":
    main()
