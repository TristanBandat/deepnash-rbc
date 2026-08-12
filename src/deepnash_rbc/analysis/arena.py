"""Arena for comparing *deployment policy modes* of trained checkpoints.

The question this exists to answer: at evaluation time, should a checkpoint play
its **argmax** action or **sample** from its policy (optionally with DeepNash's
truncation threshold)? R-NaD converges to an approximate Nash equilibrium, whose
whole point is being *mixed* -- taking the argmax throws the mixture away and
makes the agent exploitable, but it also removes the low-probability blunder
tail. Which effect dominates is an empirical question, and this module measures
it in Elo.

How a mode is measured
----------------------
An **arm** is one (checkpoint, policy mode) pair -- the same weights entered
twice under different modes are two arms. Arms are played against an **anchor
pool** drawn from the existing internal ladder (``results/tournament.jsonl`` and
its rendered leaderboard). That ladder is treated as strictly **read-only**: new
games land in a separate JSONL, and the anchors' ratings are held FIXED while
each arm's rating is fit conditionally (see :mod:`.elo`). Freezing the yardstick
is what makes "argmax is +N Elo over sampling" a statement about the two modes
rather than about the fit shifting under them.

Two caveats the caller must respect, both enforceable through the config:

  * **Anchors must replay the mode the ladder was built with.** The round-robin
    was run at ``sample_threshold=0.05``, sampled (``tools/tournament.py``
    defaults), so ``REFERENCE_MODE`` is that. Anchoring against differently-
    configured opponents would silently re-scale the ratings.
  * **Anchors are re-played, not re-used.** We never mine the old ladder rows
    for an arm's record -- an arm is new, so it has no rows there.

Beyond anchored Elo the arena also runs direct head-to-head schedules (argmax vs
sampled, same weights), cross-products (every checkpoint of run A vs every
checkpoint of run B), and free-form round robins, and it records per-decision
policy statistics (entropy, argmax agreement, top-1 mass) so a rating difference
can be tied back to *how much* the two modes actually diverge in behaviour.

Driven by ``notebooks/policy_eval.py`` (interactive) and ``tools/policy_eval.py``
(headless, for long runs).
"""

from __future__ import annotations

import glob
import json
import math
import multiprocessing as mp
import os
import random as pyrandom
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from .elo import AnchoredFit, anchored_elo, bradley_terry, delta_se, score_to_elo, wilson_ci

BASELINES = ("random", "attacker", "trout", "mht", "strangefish2")

# The policy mode ``tools/tournament.py`` used to build the internal ladder.
# Anchors MUST play in this mode or the frozen ratings do not apply to them.
REFERENCE_MODE = "sample@0.05"

_CKPT_RE = re.compile(r"deepnash_async_v(\d+\.\d+\.\d+)_(\d+)\.pt$")
_LADDER_NAME_RE = re.compile(r"^v(\d+\.\d+(?:\.\d+)?)_(\d+)$")

_WORKER: dict = {}


# --------------------------------------------------------------------- arms
@dataclass(frozen=True)
class Arm:
    """One competitor: a checkpoint (or baseline bot) plus a policy mode.

    ``name`` is the identity used in result rows, so it must be stable across
    resumed runs -- that is why the arms of a run are written to a sidecar file
    and validated on resume.
    """

    name: str
    spec: str                 # baseline name, or path to a .pt checkpoint
    greedy: bool = False      # True -> argmax; False -> sample from the policy
    threshold: float = 0.0    # sampling truncation (ignored when greedy)

    @property
    def is_net(self) -> bool:
        return self.spec not in BASELINES

    @property
    def mode(self) -> str:
        return "argmax" if self.greedy else format_mode(False, self.threshold)


def format_mode(greedy: bool, threshold: float) -> str:
    if greedy:
        return "argmax"
    return "sample" if threshold <= 0 else f"sample@{threshold:g}"


def parse_mode(mode: str) -> tuple[bool, float]:
    """``"argmax"`` / ``"sample"`` / ``"sample@0.05"`` -> ``(greedy, threshold)``."""
    mode = mode.strip()
    if mode in ("argmax", "greedy"):
        return True, 0.0
    if mode == "sample":
        return False, 0.0
    m = re.fullmatch(r"sample@([0-9.eE+-]+)", mode)
    if not m:
        raise ValueError(
            f"unknown policy mode {mode!r} (use 'argmax', 'sample', 'sample@<threshold>')"
        )
    return False, float(m.group(1))


def make_arm(label: str, spec: str, mode: str, name: Optional[str] = None) -> Arm:
    """Build an arm named ``<label>·<mode>`` unless ``name`` overrides it."""
    greedy, threshold = parse_mode(mode)
    return Arm(name or f"{label}·{format_mode(greedy, threshold)}",
               spec, greedy, threshold)


def make_arms(checkpoints: dict[str, str], modes: Sequence[str]) -> list[Arm]:
    """Cross a ``{label: path}`` mapping with a list of modes."""
    return [make_arm(label, path, mode)
            for label, path in checkpoints.items() for mode in modes]


def anchor_arm(name: str, checkpoints: dict[str, str],
               mode: str = REFERENCE_MODE) -> Optional[Arm]:
    """Arm for a ladder player, keeping its **ladder name** so the frozen rating
    still applies. Returns None when the checkpoint file is gone."""
    if name in BASELINES:
        greedy, threshold = parse_mode(mode)
        return Arm(name, name, greedy, threshold)
    if name in checkpoints:
        greedy, threshold = parse_mode(mode)
        return Arm(name, checkpoints[name], greedy, threshold)
    return None


# ------------------------------------------------------------- discovery/IO
def discover_checkpoints(root: str | os.PathLike) -> dict[str, str]:
    """Every ``checkpoints/v*/deepnash_async_v*_*.pt`` as ``{v<ver>_<step>: path}``,
    ordered by (version, step).

    Paths are fully resolved: an ``Arm``'s identity includes its spec string, so
    a run driven from the notebook and one driven from ``tools/policy_eval.py``
    must agree character-for-character or :func:`check_arms` reports a phantom
    clash on the shared results file.
    """
    found: list[tuple[tuple[int, ...], str, str]] = []
    for p in Path(root).glob("v*/deepnash_async_v*_*.pt"):
        m = _CKPT_RE.search(p.name)
        if not m:
            continue
        ver, step = m.group(1), int(m.group(2))
        found.append(((*(int(x) for x in ver.split(".")), step),
                      f"v{ver}_{step}", str(p.resolve())))
    found.sort()
    return {label: path for _, label, path in found}


def group_by_version(checkpoints: dict[str, str]) -> dict[str, dict[str, str]]:
    """``{v<ver>: {label: path}}`` -- the training runs, each with its steps."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for label, path in checkpoints.items():
        m = _LADDER_NAME_RE.match(label)
        if m:
            out[f"v{m.group(1)}"][label] = path
    return dict(out)


def parse_leaderboard(path: str | os.PathLike) -> dict[str, dict]:
    """Parse a rendered ``=== Leaderboard ===`` block into ``{player: row}``.

    Preferred over refitting from the raw JSONL: it is the canonical published
    fit, and reading it costs milliseconds against ~100 MB / 750k rows.
    """
    out: dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        name, elo, games, score, se = parts
        if not re.fullmatch(r"-?\d+", elo):  # header / decoration
            continue
        out[name] = {"player": name, "elo": float(elo), "games": int(games),
                     "score": float(score), "se": float(se)}
    return out


def anchor_pool(
    leaderboard: dict[str, dict],
    checkpoints: dict[str, str],
    top: Optional[int] = None,
    include_baselines: bool = True,
    elo_range: Optional[tuple[float, float]] = None,
    min_games: int = 0,
    explicit: Optional[Sequence[str]] = None,
) -> list[str]:
    """Pick anchor names from the ladder.

    An anchor is only usable if we can actually *build* the player, so ladder
    entries whose checkpoint file no longer exists are dropped. Ordering is by
    descending Elo, which is also the order ``top`` truncates in.
    """
    if explicit:
        return [n for n in explicit
                if n in leaderboard and (n in BASELINES or n in checkpoints)]
    rows = [r for r in leaderboard.values()
            if r["player"] in BASELINES or r["player"] in checkpoints]
    rows = [r for r in rows if r["games"] >= min_games]
    if not include_baselines:
        rows = [r for r in rows if r["player"] not in BASELINES]
    if elo_range is not None:
        lo, hi = elo_range
        rows = [r for r in rows if lo <= r["elo"] <= hi]
    rows.sort(key=lambda r: -r["elo"])
    if top is not None:
        rows = rows[:top]
    return [r["player"] for r in rows]


def ladder_elo_from_games(path: str | os.PathLike, names: Sequence[str]) -> dict[str, float]:
    """Refit anchor Elo from the raw ladder JSONL (fallback when no rendered
    leaderboard exists). Streams the file -- it is ~100 MB."""
    keep = set(names)
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("white") in keep and r.get("black") in keep:
                rows.append(r)
    return bradley_terry(rows, list(names))


# ----------------------------------------------------------------- schedule
def pairs_round_robin(arms: Sequence[Arm]) -> list[tuple[str, str]]:
    """Every unordered pair within one set."""
    return [(a.name, b.name) for a, b in combinations(arms, 2)]


def pairs_cross(a: Sequence[Arm], b: Sequence[Arm]) -> list[tuple[str, str]]:
    """Every A-vs-B pair (self-pairings and duplicates removed).

    This is the "all checkpoints of run X vs all checkpoints of run Y" schedule,
    and also the arm-vs-anchor schedule.
    """
    seen: set[frozenset[str]] = set()
    out = []
    for x in a:
        for y in b:
            if x.name == y.name:
                continue
            key = frozenset((x.name, y.name))
            if key in seen:
                continue
            seen.add(key)
            out.append((x.name, y.name))
    return out


def pairs_paired_modes(arms: Sequence[Arm]) -> list[tuple[str, str]]:
    """Pairs of arms that share the same weights but differ in mode.

    The tightest possible comparison: identical network, identical opponent,
    only the action-selection rule differs, so nothing else can explain the
    result.
    """
    by_spec: dict[str, list[Arm]] = defaultdict(list)
    for a in arms:
        by_spec[a.spec].append(a)
    out = []
    for group in by_spec.values():
        out += [(x.name, y.name) for x, y in combinations(group, 2)]
    return out


def existing_counts(rows: Iterable[dict]) -> dict[frozenset[str], int]:
    """Completed (non-errored) games per unordered pair, for resume."""
    done: dict[frozenset[str], int] = defaultdict(int)
    for r in rows:
        if r.get("winner") != "error":
            done[frozenset((r["white"], r["black"]))] += 1
    return dict(done)


def build_tasks(
    pairs: Sequence[tuple[str, str]],
    games_per_pair: int,
    done: Optional[dict[frozenset[str], int]] = None,
    seconds: float = 900.0,
    seed: int = 0,
    shuffle: bool = True,
) -> list[tuple[str, str, float, int]]:
    """Expand pairs into per-game tasks with alternating colors, skipping games
    already recorded in ``done``.

    Colors alternate *within* a pair so an odd ``games_per_pair`` is the only way
    to end up with a color imbalance; the caller is nudged toward even counts
    because RBC has a real first-move advantage.

    The per-game seed depends on the game index but **not** on the pair, so the
    n-th game of every pair starts from the same RNG state. Against a stochastic
    opponent that is a common-random-numbers design: an argmax arm and a sampled
    arm meet an identically-seeded opponent, which cancels part of the opponent's
    variance out of their difference -- the quantity this module exists to
    measure -- for free.

    Shuffling matters for wall-clock feedback, not correctness: engine-backed
    anchors (trout/mht) are ~50x slower than net-vs-net, so leaving them
    contiguous makes the progress bar's ETA useless for most of the run.
    """
    done = dict(done or {})
    tasks = []
    for a, b in pairs:
        have = done.get(frozenset((a, b)), 0)
        for g in range(have, games_per_pair):
            white, black = (a, b) if g % 2 == 0 else (b, a)
            tasks.append((white, black, seconds, seed + 1000 * g))
    if shuffle:
        pyrandom.Random(seed).shuffle(tasks)
    return tasks


def count_engine_games(tasks: Sequence[tuple], arms: dict[str, Arm]) -> int:
    """How many scheduled games involve a Stockfish-backed bot (the slow ones)."""
    slow = {"trout", "mht", "strangefish2"}
    return sum(1 for t in tasks
               if arms[t[0]].spec in slow or arms[t[1]].spec in slow)


# ----------------------------------------------------- instrumented player
@dataclass
class PolicyStats:
    """Per-decision policy telemetry, accumulated over a game.

    Recorded for both heads because sense and move policies behave very
    differently: the sense head is often near-uniform (many squares are
    equivalent), so argmax there is a much bigger intervention than on the move
    head, where the policy is usually peaked. An Elo gap that shows up only in
    one head is a different finding from one spread across both.
    """

    decisions: int = 0
    argmax_agree: int = 0      # sampled action happened to be the argmax
    entropy_bits: float = 0.0
    top1_prob: float = 0.0
    legal: int = 0
    # Probability mass the sampling threshold discarded, summed over decisions.
    # The *count* of truncated decisions would be useless here: with thousands
    # of legal moves practically every decision has some action below any
    # threshold, so that count pins at 100%. The mass says how much of the
    # policy the truncation is actually throwing away.
    truncated_mass: float = 0.0
    # Decisions where NO action cleared the threshold, so RNaDPlayer fell back
    # to the raw policy and the threshold did nothing. Tracked because this makes
    # truncation non-monotone in a way that is otherwise baffling to read: a
    # *higher* threshold can discard *less* mass, since it trips the fallback far
    # more often. On the near-uniform sense head (top-1 ~0.05) a threshold of 0.2
    # is bypassed at essentially every decision, i.e. it is not truncating at all.
    threshold_bypassed: int = 0

    def add(self, other: "PolicyStats") -> None:
        self.decisions += other.decisions
        self.argmax_agree += other.argmax_agree
        self.entropy_bits += other.entropy_bits
        self.top1_prob += other.top1_prob
        self.legal += other.legal
        self.truncated_mass += other.truncated_mass
        self.threshold_bypassed += other.threshold_bypassed

    def as_dict(self) -> dict:
        return asdict(self)


def _make_stats_player(net, device, history: int, arm: Arm):
    """RNaDPlayer that also records the policy it acted on.

    ``_sample_from`` is re-derived rather than intercepted around the network
    call: recomputing the masked softmax is cheap, whereas calling ``_forward``
    a second time would double-advance a temporal net's recurrent state and
    silently corrupt the game.
    """
    import numpy as np
    import torch

    from ..agent import RNaDPlayer
    from ..replay import MOVE, SENSE

    class _StatsPlayer(RNaDPlayer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.stats = {SENSE: PolicyStats(), MOVE: PolicyStats()}
            self._head = SENSE

        def choose_sense(self, *a, **k):
            self._head = SENSE
            return super().choose_sense(*a, **k)

        def choose_move(self, *a, **k):
            self._head = MOVE
            return super().choose_move(*a, **k)

        def _sample_from(self, logits, legal):
            action, logp = super()._sample_from(logits, legal)
            with torch.no_grad():
                flat = logits.squeeze(0).float().cpu()
                masked = torch.full_like(flat, float("-inf"))
                idx = torch.from_numpy(legal)
                masked[idx] = flat[idx]
                probs = torch.softmax(masked, dim=0)
                p = probs[idx].numpy().astype(np.float64)
            p = p / max(p.sum(), 1e-12)
            nz = p[p > 0]
            s = self.stats[self._head]
            s.decisions += 1
            s.argmax_agree += int(action == int(idx[int(np.argmax(p))]))
            s.entropy_bits += float(-(nz * np.log2(nz)).sum())
            s.top1_prob += float(p.max())
            s.legal += int(len(p))
            if self.sample_threshold > 0:
                below = p < self.sample_threshold
                # a policy flatter than the threshold keeps everything (see
                # RNaDPlayer._sample_from), so nothing is discarded there
                if below.all():
                    s.threshold_bypassed += 1
                else:
                    s.truncated_mass += float(p[below].sum())
            return action, logp

    return _StatsPlayer(net, device, history=history,
                        sample=not arm.greedy, sample_threshold=arm.threshold)


# ------------------------------------------------------------------ workers
def _worker_init(arms: dict, device: Optional[str], cache_size: int,
                 collect_stats: bool):
    import torch

    from ..eval import _ensure_stockfish

    arms = {name: Arm(**d) for name, d in arms.items()}
    if any(a.spec in ("trout", "mht", "strangefish2") for a in arms.values()):
        _ensure_stockfish()
    _WORKER.update(
        arms=arms,
        device=torch.device(device) if device else
               torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        nets=OrderedDict(),  # LRU: path -> (net, enc)
        cache_size=cache_size,
        collect_stats=collect_stats,
    )


def _get_net(path: str):
    from ..play_session import load_net

    nets: OrderedDict = _WORKER["nets"]
    if path in nets:
        nets.move_to_end(path)
        return nets[path]
    if len(nets) >= _WORKER["cache_size"]:
        nets.popitem(last=False)
    nets[path] = load_net(path, _WORKER["device"])
    return nets[path]


def _build_player(name: str):
    from ..agent import RNaDPlayer
    from ..eval import _make_opponent

    arm: Arm = _WORKER["arms"][name]
    if not arm.is_net:
        return _make_opponent(arm.spec)
    net, enc = _get_net(arm.spec)
    if _WORKER["collect_stats"]:
        return _make_stats_player(net, _WORKER["device"], enc.history, arm)
    return RNaDPlayer(net, _WORKER["device"], history=enc.history,
                      sample=not arm.greedy, sample_threshold=arm.threshold)


def _play_game(task) -> dict:
    """``(white, black, seconds, seed)`` -> one result row."""
    import random

    import chess
    import numpy as np
    import torch
    from reconchess import LocalGame, play_local_game

    white_name, black_name, seconds, seed = task
    # Seed every RNG the game can touch: the policy's multinomial (torch), the
    # baseline bots (random), and anything numpy-side. Without this a resumed
    # run is not reproducible, which would make small Elo gaps unfalsifiable.
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

    row = {"white": white_name, "black": black_name,
           "ts": round(time.time(), 1), "seed": seed}
    t0 = time.time()
    try:
        white = _build_player(white_name)
        black = _build_player(black_name)
        winner, reason, hist = play_local_game(
            white, black, game=LocalGame(seconds_per_player=seconds))
    except Exception as e:
        row.update(winner="error", reason=f"{type(e).__name__}: {e}")
        return row
    row["winner"] = "draw" if winner is None else (
        "white" if winner == chess.WHITE else "black")
    row["reason"] = str(reason)
    row["turns"] = hist.num_turns()
    row["secs"] = round(time.time() - t0, 2)

    from ..replay import MOVE, SENSE

    stats = {}
    for name, player in ((white_name, white), (black_name, black)):
        acc = getattr(player, "stats", None)
        if acc is None:
            continue
        stats[name] = {"sense": acc[SENSE].as_dict(), "move": acc[MOVE].as_dict()}
    if stats:
        row["stats"] = stats
    return row


def run_games(
    tasks: Sequence[tuple],
    arms: dict[str, Arm],
    workers: int = 8,
    device: Optional[str] = None,
    net_cache: int = 8,
    collect_stats: bool = True,
    out_path: Optional[str | os.PathLike] = None,
) -> Iterator[dict]:
    """Play ``tasks`` in a process pool, yielding rows as they finish.

    A generator so the caller owns the progress display (a marimo progress bar,
    a tqdm, a print loop) instead of this module guessing. Rows are appended and
    flushed to ``out_path`` before being yielded, so an interrupted run resumes
    with everything it had already played.
    """
    if not tasks:
        return
    payload = {name: asdict(a) for name, a in arms.items()}
    ctx = mp.get_context("spawn")
    fh = None
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fh = open(out_path, "a")
    try:
        with ctx.Pool(workers, initializer=_worker_init,
                      initargs=(payload, device, net_cache, collect_stats)) as pool:
            for row in pool.imap_unordered(_play_game, list(tasks)):
                if fh is not None:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                yield row
    finally:
        if fh is not None:
            fh.close()


# --------------------------------------------------------------- run store
def load_rows(path: str | os.PathLike) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def arms_sidecar(path: str | os.PathLike) -> Path:
    return Path(str(path) + ".arms.json")


def save_arms(path: str | os.PathLike, arms: dict[str, Arm]) -> None:
    """Persist arm definitions next to the results.

    Arm *names* are the only identity in a result row, so resuming a run with a
    changed definition (a different threshold under the same name) would blend
    two different agents into one rating. The sidecar lets
    :func:`check_arms` catch that instead.
    """
    p = arms_sidecar(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    known = load_arms(path)
    known.update(arms)
    p.write_text(json.dumps({n: asdict(a) for n, a in sorted(known.items())}, indent=2))


def load_arms(path: str | os.PathLike) -> dict[str, Arm]:
    p = arms_sidecar(path)
    if not p.exists():
        return {}
    return {n: Arm(**d) for n, d in json.loads(p.read_text()).items()}


def check_arms(path: str | os.PathLike, arms: dict[str, Arm]) -> list[str]:
    """Names whose stored definition disagrees with ``arms``."""
    stored = load_arms(path)
    return [n for n, a in arms.items() if n in stored and stored[n] != a]


# -------------------------------------------------------------- aggregation
@dataclass
class ArmResult:
    name: str
    arm: Optional[Arm] = None
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    errors: int = 0
    white_games: int = 0
    white_score: float = 0.0
    turns: float = 0.0            # summed; averaged in as_dict
    vs: dict[str, list[float]] = field(default_factory=dict)  # opp -> [score, games]
    sense: PolicyStats = field(default_factory=PolicyStats)
    move: PolicyStats = field(default_factory=PolicyStats)
    fit: Optional[AnchoredFit] = None

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def score_rate(self) -> float:
        return self.score / self.games if self.games else float("nan")

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else float("nan")

    @property
    def avg_turns(self) -> float:
        return self.turns / self.games if self.games else float("nan")

    @property
    def white_rate(self) -> float:
        return self.white_score / self.white_games if self.white_games else float("nan")


def summarize(rows: Iterable[dict], arms: dict[str, Arm]) -> dict[str, ArmResult]:
    """Fold result rows into a per-arm record (including per-opponent splits)."""
    out = {name: ArmResult(name=name, arm=arm) for name, arm in arms.items()}
    for r in rows:
        w, b = r.get("white"), r.get("black")
        if r.get("winner") == "error":
            for n in (w, b):
                if n in out:
                    out[n].errors += 1
            continue
        for me, opp, is_white in ((w, b, True), (b, w, False)):
            if me not in out:
                continue
            res = out[me]
            res.games += 1
            res.turns += r.get("turns", 0) or 0
            if r["winner"] == "draw":
                res.draws += 1
                gained = 0.5
            elif r[r["winner"]] == me:
                res.wins += 1
                gained = 1.0
            else:
                res.losses += 1
                gained = 0.0
            if is_white:
                res.white_games += 1
                res.white_score += gained
            slot = res.vs.setdefault(opp, [0.0, 0.0])
            slot[0] += gained
            slot[1] += 1
        for name, st in (r.get("stats") or {}).items():
            if name not in out:
                continue
            out[name].sense.add(PolicyStats(**st["sense"]))
            out[name].move.add(PolicyStats(**st["move"]))
    return out


def fit_arms(
    results: dict[str, ArmResult],
    anchors: dict[str, float],
) -> dict[str, ArmResult]:
    """Attach an anchored Elo fit to every arm, in place.

    Arm-vs-arm games are excluded outright -- feeding them in would reintroduce
    exactly the coupling that freezing the anchors exists to remove, and they are
    reported separately by :func:`head_to_head`. Games against a *non-arm*
    opponent that has no ladder rating are passed through to
    :func:`~.elo.anchored_elo` so they surface as ``fit.dropped`` rather than
    vanishing silently -- a large ``dropped`` means the anchor pool is not
    actually rating this arm.
    """
    for res in results.values():
        if res.name in anchors:
            continue  # an anchor's rating is fixed by definition
        record = {opp: (s, n) for opp, (s, n) in res.vs.items()
                  if opp not in results or opp in anchors}
        if record:
            res.fit = anchored_elo(record, anchors)
    return results


def head_to_head(rows: Iterable[dict], a: str, b: str) -> dict:
    """Direct match report for ``a`` vs ``b``: score, Wilson CI, implied Elo.

    Reported alongside the anchored fit rather than instead of it. The direct
    match has no anchor-pool dependency and is the cleanest paired test, but it
    only measures how the two modes do *against each other* -- a mode can beat
    its twin and still fare worse against the field, which is exactly what
    "exploitable but sharper" would look like.
    """
    score = games = 0.0
    a_white = 0
    for r in rows:
        if r.get("winner") == "error":
            continue
        names = {r.get("white"), r.get("black")}
        if names != {a, b}:
            continue
        games += 1
        a_white += int(r["white"] == a)
        if r["winner"] == "draw":
            score += 0.5
        elif r[r["winner"]] == a:
            score += 1.0
    rate = score / games if games else float("nan")
    lo, hi = wilson_ci(score, games)
    return {
        "a": a, "b": b, "games": int(games), "score": score, "rate": rate,
        "ci_low": lo, "ci_high": hi,
        "elo": score_to_elo(rate) if games else float("nan"),
        "elo_low": score_to_elo(lo) if games else float("nan"),
        "elo_high": score_to_elo(hi) if games else float("nan"),
        "a_white_games": a_white,
        # significant iff the interval excludes an even match
        "significant": bool(games and (lo > 0.5 or hi < 0.5)),
    }


def mode_delta(results: dict[str, ArmResult], a: str, b: str) -> dict:
    """``a - b`` in anchored Elo, with the 1-sigma error of the difference."""
    fa, fb = results[a].fit, results[b].fit
    if fa is None or fb is None:
        return {"a": a, "b": b, "delta": float("nan"), "se": float("inf"),
                "z": float("nan"), "significant": False}
    d = fa.elo - fb.elo
    se = delta_se(fa, fb)
    z = d / se if se and math.isfinite(se) else float("nan")
    return {"a": a, "b": b, "delta": d, "se": se, "z": z,
            "a_elo": fa.elo, "b_elo": fb.elo,
            "significant": bool(math.isfinite(z) and abs(z) >= 1.96),
            "degenerate": fa.degenerate or fb.degenerate}


def eta_string(done: int, total: int, elapsed: float) -> str:
    if not done:
        return "estimating…"
    rate = elapsed / done
    remain = rate * (total - done)
    return f"{rate:.1f}s/game · ETA {remain / 60:.0f} min"
