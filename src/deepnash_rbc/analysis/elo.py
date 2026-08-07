"""Elo / Bradley-Terry fitting, shared by the round-robin ladder and the arena.

Two different fits live here, and the distinction matters:

  * :func:`bradley_terry` -- the **joint** MM fit used by ``tools/tournament.py``.
    Every player's strength is a free parameter and the scale is pinned by
    anchoring ``random = 0``. This is the right fit when the whole field is being
    ranked at once.

  * :func:`anchored_elo` -- a **conditional** fit for dropping one new player
    into an *existing* scale. The opponents' ratings are held FIXED at their
    ladder values and only the new player's rating is estimated.

The second is what makes an argmax-vs-sampling comparison honest. If two
variants of the same checkpoint were added to a joint fit, both would shift the
field (they beat the same opponents, so the whole scale slides) and their Elo
difference would be confounded by that shift. Measured against a frozen
yardstick they are not: each variant's rating is a one-parameter MLE over its
own games, the two fits are independent, and the difference of the two has a
closed-form standard error (:func:`delta_se`).

Elo convention throughout: expected score ``1 / (1 + 10 ** ((r_opp - r) / 400))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

# d/dR of the logistic in Elo units: p'(R) = LN10_400 * p * (1 - p)
LN10_400 = math.log(10.0) / 400.0


# ----------------------------------------------------------------- joint fit
def bradley_terry(rows: list[dict], names: list[str]) -> dict[str, float]:
    """Fit BT strengths over ``names`` (draws = half a win each); return Elo.

    ``rows`` are per-game dicts ``{"white", "black", "winner"}`` where ``winner``
    is ``"white"``/``"black"``/``"draw"``/``"error"``. Games touching a player
    outside ``names`` are ignored, so a sub-field can be fit in isolation.
    Anchored at ``random = 0`` when present, else at mean 0.
    """
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    score = [0.0] * n                       # total score of player i
    pair_n = [[0.0] * n for _ in range(n)]  # games between i and j
    for r in rows:
        if r["winner"] == "error" or r["white"] not in idx or r["black"] not in idx:
            continue
        w, b = idx[r["white"]], idx[r["black"]]
        pair_n[w][b] += 1
        pair_n[b][w] += 1
        if r["winner"] == "draw":
            score[w] += 0.5
            score[b] += 0.5
        else:
            score[idx[r[r["winner"]]]] += 1.0

    p = [1.0] * n
    for _ in range(500):  # MM iterations
        new = []
        for i in range(n):
            denom = sum(pair_n[i][j] / (p[i] + p[j]) for j in range(n) if j != i)
            # clamp: undefeated/never-scoring players have no finite MLE
            s = min(max(score[i], 0.5), sum(pair_n[i]) - 0.5) if sum(pair_n[i]) else 0.5
            new.append(s / denom if denom else p[i])
        norm = math.exp(sum(math.log(x) for x in new) / n)
        p = [x / norm for x in new]

    elo = {name: 400.0 * math.log10(p[i]) for name, i in idx.items()}
    anchor = elo.get("random", sum(elo.values()) / len(elo))
    return {name: e - anchor for name, e in elo.items()}


# ----------------------------------------------------------- conditional fit
@dataclass(frozen=True)
class AnchoredFit:
    """One player's rating on a frozen scale."""

    elo: float
    se: float          # 1-sigma, from the Fisher information of the 1-D MLE
    games: float       # games counted (i.e. against known anchors)
    score: float       # wins + 0.5 * draws
    dropped: int = 0   # games discarded: opponent had no anchor rating
    degenerate: bool = False  # perfect / null score -> rating is a lower bound

    @property
    def score_rate(self) -> float:
        return self.score / self.games if self.games else float("nan")


def expected_score(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def anchored_elo(
    results: Mapping[str, tuple[float, float]],
    anchors: Mapping[str, float],
) -> AnchoredFit:
    """Estimate one rating against opponents whose ratings are held fixed.

    ``results`` maps opponent name -> ``(score, games)``; ``anchors`` maps
    opponent name -> its fixed Elo. Opponents missing from ``anchors`` are
    dropped (and counted), since there is no scale on which to score them.

    The log-likelihood ``sum_j s_j log p_j + (n_j - s_j) log(1 - p_j)`` is
    strictly concave in the rating, so the score equation
    ``sum_j (s_j - n_j p_j) = 0`` has a unique root; we bisect for it. A perfect
    (or null) record has no finite MLE, so the total score is nudged by half a
    game -- the same clamp :func:`bradley_terry` uses -- and ``degenerate`` is
    set so callers can render the number as a bound rather than an estimate.
    """
    obs: list[tuple[float, float, float]] = []  # (opp_rating, score, games)
    dropped = 0
    for opp, (s, n) in results.items():
        if n <= 0:
            continue
        if opp not in anchors:
            dropped += int(n)
            continue
        obs.append((float(anchors[opp]), float(s), float(n)))

    total_n = sum(n for _, _, n in obs)
    total_s = sum(s for _, s, _ in obs)
    if not obs or total_n <= 0:
        return AnchoredFit(float("nan"), float("inf"), 0.0, 0.0, dropped, True)

    # half-game clamp for perfect/null records, spread over opponents in
    # proportion to games played so the per-opponent shape is preserved
    degenerate = total_s < 0.5 or total_s > total_n - 0.5
    if degenerate:
        eps = 0.5
        if total_s < 0.5:
            obs = [(r, eps * n / total_n, n) for r, _, n in obs]
        else:
            obs = [(r, n - eps * n / total_n, n) for r, _, n in obs]

    def gradient(rating: float) -> float:
        return sum(s - n * expected_score(rating, r) for r, s, n in obs)

    lo, hi = -4000.0, 6000.0
    for _ in range(200):  # bisection: ~1e-27 Elo, i.e. exact for our purposes
        mid = 0.5 * (lo + hi)
        if gradient(mid) > 0:
            lo = mid
        else:
            hi = mid
    rating = 0.5 * (lo + hi)

    info = LN10_400**2 * sum(
        n * (p := expected_score(rating, r)) * (1.0 - p) for r, _, n in obs
    )
    se = 1.0 / math.sqrt(info) if info > 0 else float("inf")
    return AnchoredFit(rating, se, total_n, total_s, dropped, degenerate)


def delta_se(a: AnchoredFit, b: AnchoredFit) -> float:
    """1-sigma error of ``a.elo - b.elo``.

    Valid because both are fit against *fixed* anchors from disjoint sets of
    games, so the two estimates are independent -- the anchors contribute no
    shared estimation error. (In a joint BT fit they would be correlated and
    this would understate the uncertainty.)
    """
    return math.hypot(a.se, b.se)


# -------------------------------------------------------------- small stats
def wilson_ci(score: float, games: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a score rate. Handles 0/n and n/n gracefully, which
    the normal approximation does not -- and both happen constantly in short
    head-to-head matches."""
    if games <= 0:
        return (float("nan"), float("nan"))
    p = score / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def score_to_elo(rate: float) -> float:
    """Elo difference implied by a head-to-head score rate (+inf/-inf clipped)."""
    if rate <= 0.0:
        return float("-inf")
    if rate >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / rate - 1.0)
