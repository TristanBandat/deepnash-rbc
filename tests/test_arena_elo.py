"""Tests for the policy-mode arena and its Elo estimators.

The argmax-vs-sampling conclusion is only as trustworthy as the conditional fit
underneath it, so the estimator is pinned against closed-form values rather than
regression snapshots: a 50% score must return the opponent's exact rating, 75%
must return it plus 400*log10(3), and the standard error must reproduce the
classic 347/sqrt(n) rule.
"""

from __future__ import annotations

import math

import pytest

from deepnash_rbc.analysis import arena
from deepnash_rbc.analysis.elo import (
    anchored_elo,
    bradley_terry,
    delta_se,
    expected_score,
    score_to_elo,
    wilson_ci,
)


# ------------------------------------------------------------- anchored fit
def test_even_score_returns_the_anchor_rating():
    fit = anchored_elo({"a": (50.0, 100.0)}, {"a": 700.0})
    assert fit.elo == pytest.approx(700.0, abs=1e-6)
    assert not fit.degenerate


def test_score_maps_to_the_textbook_elo_offset():
    # 75% expected score is exactly +400*log10(0.75/0.25) = +190.85 Elo
    fit = anchored_elo({"a": (75.0, 100.0)}, {"a": 700.0})
    assert fit.elo == pytest.approx(700.0 + 400.0 * math.log10(3.0), abs=1e-6)


def test_standard_error_matches_the_347_over_sqrt_n_rule():
    # at p=0.5 the Fisher information gives se = 800/(ln10*sqrt(n)), i.e. the
    # familiar 347/sqrt(n) rule of thumb (the exact constant is 347.4356)
    fit = anchored_elo({"a": (50.0, 100.0)}, {"a": 700.0})
    assert fit.se == pytest.approx(800.0 / (math.log(10) * math.sqrt(100)))
    assert fit.se == pytest.approx(34.7, abs=0.05)
    # and it must shrink as sqrt(n)
    wide = anchored_elo({"a": (25.0, 50.0)}, {"a": 700.0})
    assert wide.se == pytest.approx(fit.se * math.sqrt(2), rel=1e-6)


def test_total_score_is_the_sufficient_statistic():
    """Same opponents and game counts, same total -> same rating.

    A property of the model, not an accident: the score equation reduces to
    ``sum_j s_j = sum_j n_j p_j(R)``, whose left side sees only the total. Worth
    pinning because it is exactly what makes two arms with identical records
    land on identical Elo (and looks like a bug when it happens).
    """
    anchors = {"x": 800.0, "y": 700.0, "z": 600.0}
    a = anchored_elo({"x": (5.0, 6.0), "y": (5.0, 6.0), "z": (4.0, 6.0)}, anchors)
    b = anchored_elo({"x": (6.0, 6.0), "y": (5.0, 6.0), "z": (3.0, 6.0)}, anchors)
    assert a.elo == pytest.approx(b.elo)
    assert a.se == pytest.approx(b.se)


def test_stronger_anchors_imply_a_higher_rating_for_the_same_score():
    weak = anchored_elo({"a": (30.0, 40.0)}, {"a": 400.0})
    strong = anchored_elo({"a": (30.0, 40.0)}, {"a": 900.0})
    assert strong.elo - weak.elo == pytest.approx(500.0, abs=1e-6)


def test_perfect_and_null_records_are_finite_but_flagged():
    perfect = anchored_elo({"a": (100.0, 100.0)}, {"a": 700.0})
    null = anchored_elo({"a": (0.0, 100.0)}, {"a": 700.0})
    assert perfect.degenerate and null.degenerate
    assert math.isfinite(perfect.elo) and math.isfinite(null.elo)
    assert perfect.elo > 700.0 > null.elo


def test_unrated_opponents_are_dropped_and_counted():
    fit = anchored_elo({"a": (5.0, 10.0), "ghost": (7.0, 10.0)}, {"a": 700.0})
    assert fit.dropped == 10
    assert fit.games == 10
    assert fit.elo == pytest.approx(700.0, abs=1e-6)


def test_no_rated_opponents_yields_an_empty_fit():
    fit = anchored_elo({"ghost": (5.0, 10.0)}, {"a": 700.0})
    assert fit.games == 0 and fit.dropped == 10
    assert math.isnan(fit.elo) and fit.se == float("inf")


def test_delta_error_is_the_quadrature_sum():
    a = anchored_elo({"x": (50.0, 100.0)}, {"x": 700.0})
    b = anchored_elo({"x": (25.0, 50.0)}, {"x": 700.0})
    assert delta_se(a, b) == pytest.approx(math.hypot(a.se, b.se))


# ----------------------------------------------------------- small stats
def test_expected_score_and_its_inverse_round_trip():
    for diff in (-400.0, -50.0, 0.0, 137.0, 400.0):
        rate = expected_score(diff, 0.0)
        assert score_to_elo(rate) == pytest.approx(diff, abs=1e-9)


def test_wilson_interval_stays_in_range_at_the_extremes():
    lo, hi = wilson_ci(0.0, 10.0)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = wilson_ci(10.0, 10.0)
    assert hi == 1.0 and 0.0 < lo < 1.0


def test_bradley_terry_anchors_random_at_zero():
    rows = [{"white": "a", "black": "random", "winner": "white"} for _ in range(10)]
    rows += [{"white": "random", "black": "a", "winner": "white"} for _ in range(3)]
    elo = bradley_terry(rows, ["a", "random"])
    assert elo["random"] == pytest.approx(0.0)
    assert elo["a"] > 0.0


# ------------------------------------------------------------------- arms
@pytest.mark.parametrize(
    "text, expected",
    [
        ("argmax", (True, 0.0)),
        ("greedy", (True, 0.0)),
        ("sample", (False, 0.0)),
        ("sample@0.05", (False, 0.05)),
    ],
)
def test_parse_mode(text, expected):
    assert arena.parse_mode(text) == expected


def test_parse_mode_rejects_nonsense():
    with pytest.raises(ValueError):
        arena.parse_mode("mostly argmax")


def test_mode_formatting_round_trips():
    for mode in ("argmax", "sample", "sample@0.05", "sample@0.2"):
        greedy, threshold = arena.parse_mode(mode)
        assert arena.format_mode(greedy, threshold) == mode


# -------------------------------------------------------------- schedules
def _arm(name, spec, mode="sample"):
    return arena.make_arm(name, spec, mode)


def test_paired_modes_pairs_only_within_shared_weights():
    arms = [
        _arm("c1", "/w1.pt", "argmax"),
        _arm("c1", "/w1.pt", "sample@0.05"),
        _arm("c2", "/w2.pt", "argmax"),
    ]
    pairs = arena.pairs_paired_modes(arms)
    assert pairs == [("c1·argmax", "c1·sample@0.05")]


def test_cross_skips_self_pairings_and_duplicates():
    a = [_arm("c1", "/w1.pt", "argmax"), _arm("c2", "/w2.pt", "argmax")]
    b = [_arm("c2", "/w2.pt", "argmax"), _arm("c3", "/w3.pt", "argmax")]
    pairs = arena.pairs_cross(a, b)
    assert len(pairs) == len({frozenset(p) for p in pairs})  # no duplicates
    assert all(x != y for x, y in pairs)                      # no self-play


def test_build_tasks_alternates_colors_and_resumes():
    pairs = [("a", "b")]
    tasks = arena.build_tasks(pairs, 4, {}, shuffle=False)
    assert [(t[0], t[1]) for t in tasks] == [
        ("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")
    ]
    # two already played -> only the remaining two, keeping the color cadence
    resumed = arena.build_tasks(pairs, 4, {frozenset(("a", "b")): 2}, shuffle=False)
    assert [(t[0], t[1]) for t in resumed] == [("a", "b"), ("b", "a")]


def test_task_seeds_are_shared_across_pairs_for_common_random_numbers():
    """The n-th game of every pair starts from the same seed, so competing arms
    meet an identically-seeded opponent (see build_tasks)."""
    one = arena.build_tasks([("a", "anchor")], 3, {}, shuffle=False)
    two = arena.build_tasks([("b", "anchor")], 3, {}, shuffle=False)
    assert [t[3] for t in one] == [t[3] for t in two]
    assert len(set(t[3] for t in one)) == 3  # but distinct within a pair


# ------------------------------------------------------------ aggregation
def _rows():
    return [
        {"white": "arm", "black": "anchor", "winner": "white", "turns": 20},
        {"white": "anchor", "black": "arm", "winner": "draw", "turns": 30},
        {"white": "arm", "black": "anchor", "winner": "black", "turns": 40},
        {"white": "arm", "black": "twin", "winner": "white", "turns": 10},
        {"white": "arm", "black": "anchor", "winner": "error", "reason": "boom"},
    ]


def _arms():
    return {
        "arm": arena.Arm("arm", "/w.pt", greedy=True),
        "twin": arena.Arm("twin", "/w.pt", greedy=False, threshold=0.05),
        "anchor": arena.Arm("anchor", "random"),
    }


def test_summarize_splits_results_colors_and_errors():
    res = arena.summarize(_rows(), _arms())["arm"]
    assert (res.games, res.wins, res.draws, res.losses, res.errors) == (4, 2, 1, 1, 1)
    assert res.score == pytest.approx(2.5)
    assert res.white_games == 3                     # 3 of its 4 games as White
    assert res.white_score == pytest.approx(2.0)    # win + loss + win
    assert res.avg_turns == pytest.approx((20 + 30 + 40 + 10) / 4)
    assert res.vs["anchor"] == [1.5, 3.0]
    assert res.vs["twin"] == [1.0, 1.0]


def test_fit_excludes_arm_versus_arm_games():
    """Arm-vs-arm games must not enter the anchored fit -- including them would
    couple the two ratings the frozen anchors exist to keep independent."""
    results = arena.fit_arms(arena.summarize(_rows(), _arms()), {"anchor": 500.0})
    fit = results["arm"].fit
    assert fit.games == 3            # the 3 completed games vs the anchor only
    assert fit.score == pytest.approx(1.5)
    assert fit.dropped == 0          # the twin is an arm, not an unrated opponent
    assert fit.elo == pytest.approx(500.0, abs=1e-6)  # even score vs a 500 anchor
    assert results["anchor"].fit is None               # anchors are never re-fit


def test_head_to_head_reads_only_the_direct_match():
    match = arena.head_to_head(_rows(), "arm", "twin")
    assert match["games"] == 1
    assert match["rate"] == pytest.approx(1.0)
    assert match["a_white_games"] == 1
    assert not match["significant"]  # one game can never clear 95%


def test_mode_delta_orients_and_propagates_error():
    results = arena.fit_arms(arena.summarize(_rows(), _arms()), {"anchor": 500.0})
    # give the twin a record so it also gets a fit
    results["twin"].vs["anchor"] = [6.0, 8.0]
    results = arena.fit_arms(results, {"anchor": 500.0})
    delta = arena.mode_delta(results, "arm", "twin")
    assert delta["delta"] == pytest.approx(
        results["arm"].fit.elo - results["twin"].fit.elo
    )
    assert delta["se"] == pytest.approx(
        delta_se(results["arm"].fit, results["twin"].fit)
    )


# -------------------------------------------------------------- anchor pool
def _leaderboard():
    return {
        "top": {"player": "top", "elo": 900.0, "games": 500},
        "mid": {"player": "mid", "elo": 600.0, "games": 500},
        "thin": {"player": "thin", "elo": 700.0, "games": 4},
        "gone": {"player": "gone", "elo": 800.0, "games": 500},
        "random": {"player": "random", "elo": 0.0, "games": 500},
    }


def test_anchor_pool_filters_and_orders():
    known = {"top": "/top.pt", "mid": "/mid.pt", "thin": "/thin.pt"}
    pool = arena.anchor_pool(_leaderboard(), known, top=None, min_games=100)
    assert pool == ["top", "mid", "random"]  # descending Elo; "gone" unbuildable,
    #                                          "thin" too few games


def test_anchor_pool_respects_elo_range_and_baseline_switch():
    known = {"top": "/top.pt", "mid": "/mid.pt"}
    pool = arena.anchor_pool(
        _leaderboard(), known, min_games=100,
        include_baselines=False, elo_range=(500.0, 800.0),
    )
    assert pool == ["mid"]


def test_anchor_arm_keeps_the_ladder_name_so_the_frozen_rating_applies():
    arm = arena.anchor_arm("top", {"top": "/top.pt"}, "sample@0.05")
    assert arm.name == "top"          # NOT "top·sample@0.05"
    assert arm.spec == "/top.pt" and arm.threshold == 0.05
    assert arena.anchor_arm("gone", {}) is None
