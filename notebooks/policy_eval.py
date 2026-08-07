# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.9", "polars>=1.0", "matplotlib>=3.7"]
# ///
"""Argmax vs sampling at evaluation time — an Elo arena over the internal ladder.

R-NaD converges to an approximate **Nash equilibrium**, and the point of an
equilibrium in an imperfect-information game is that it is *mixed*. Playing the
argmax throws the mixture away: the agent becomes deterministic and therefore
exploitable, but it also stops sampling the low-probability blunder tail. Which
effect wins is not answerable from theory — it depends on how peaked the trained
policy actually is, and on whether the opponents present can exploit
determinism. This notebook measures it in Elo.

**How the measurement works.** An *arm* is one (checkpoint, policy mode) pair —
the same weights entered twice under different modes are two arms. Arms play a
pool of **anchors** taken from the existing internal ladder
(`results/tournament.jsonl` + its rendered leaderboard). The ladder is used
strictly **read-only**: new games go to a separate results file, and the
anchors' Elo is held *fixed* while each arm's rating is fit conditionally
against it (`deepnash_rbc.analysis.elo.anchored_elo`). Freezing the yardstick is
what makes the comparison honest — in a joint re-fit both arms would drag the
whole field, and their difference would partly measure the fit moving.

> **The one thing to get right:** anchors must replay the policy mode the ladder
> was built with (`tools/tournament.py` defaults: sampled, threshold 0.05).
> Anchoring against differently-configured opponents silently re-scales every
> rating. The default anchor mode below is exactly that, and the schedule
> preview warns if you change it.

Four schedules, all configurable:

| schedule | what it answers |
|---|---|
| **modes vs anchors** | where each mode lands on the ladder's Elo scale |
| **paired head-to-head** | same weights, argmax vs sampled, playing each other |
| **cross: run A × run B** | every checkpoint of one run against every checkpoint of another |
| **round robin** | free-form: every selected arm against every other |

Alongside Elo the arena records per-decision **policy statistics** — entropy,
top-1 mass, and how often the sampled action *was* the argmax — separately for
the sense and move heads. These tie a rating gap back to how much the two modes
actually diverge in behaviour: if sampling agrees with argmax 95% of the time on
the move head, no Elo gap there can be large, and a gap that appears only on the
near-uniform sense head is a different finding entirely.

Run:  uv run --group notebooks marimo edit notebooks/policy_eval.py
      uv run --group notebooks marimo run  notebooks/policy_eval.py   # app mode

For long runs prefer the headless driver and reopen this notebook on its output
file to analyse — both share the same on-disk format and resume logic:

      uv run python tools/policy_eval.py --help
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import torch

    from deepnash_rbc.analysis import arena

    return Path, arena, mo, pl, plt, time, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Evaluation policy: argmax vs sampling

    Enter the same checkpoint twice — once playing **argmax**, once **sampling**
    — against a frozen slice of the internal ladder, and read off the Elo
    difference. Anchor ratings never move, so the gap is a property of the two
    modes and not of the fit.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Data sources
    """)
    return


@app.cell
def _(Path, mo):
    # Resolve defaults relative to this notebook so the launch directory doesn't matter.
    _root = Path(__file__).resolve().parent.parent

    lb_input = mo.ui.text(
        value=str(_root / "results" / "tournament_leaderboard.txt"),
        label="ladder leaderboard .txt (frozen anchor ratings)",
        full_width=True,
    )
    ckpt_input = mo.ui.text(
        value=str(_root / "checkpoints"),
        label="checkpoints dir",
        full_width=True,
    )
    out_input = mo.ui.text(
        value=str(_root / "results" / "policy_eval.jsonl"),
        label="arena results .jsonl (written here — the ladder is never modified)",
        full_width=True,
    )
    mo.vstack([lb_input, ckpt_input, out_input])
    return ckpt_input, lb_input, out_input


@app.cell
def _(arena, ckpt_input, lb_input):
    leaderboard = arena.parse_leaderboard(lb_input.value)
    checkpoints = arena.discover_checkpoints(ckpt_input.value)
    versions = arena.group_by_version(checkpoints)
    version_names = sorted(
        versions, key=lambda v: tuple(int(x) for x in v[1:].split("."))
    )
    return checkpoints, leaderboard, version_names, versions


@app.cell
def _(arena, checkpoints, leaderboard, mo, version_names):
    _usable = [
        n for n in leaderboard if n in checkpoints or n in arena.BASELINES
    ]
    mo.md(
        f"""
        **Loaded.** {len(checkpoints)} checkpoints across {len(version_names)} runs ·
        {len(leaderboard)} rated ladder players, {len(_usable)} of them still
        buildable and therefore usable as anchors.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Experiment
    """)
    return


@app.cell
def _(mo):
    schedule_select = mo.ui.radio(
        options={
            "modes vs anchors": "anchors",
            "paired head-to-head": "paired",
            "cross: run A × run B": "cross",
            "round robin": "round_robin",
        },
        value="modes vs anchors",
        label="schedule",
        inline=True,
    )
    mo.vstack([
        schedule_select,
        mo.md(
            """
            * **modes vs anchors** — each arm plays the anchor pool; yields an
              anchored Elo per arm. The headline measurement.
            * **paired head-to-head** — arms sharing weights play *each other*.
              No anchor-pool dependency and perfectly paired, but it only says
              which mode beats its twin, not how each fares against the field.
            * **cross: run A × run B** — every checkpoint of one run against
              every checkpoint of another, each side with its own mode.
            * **round robin** — every selected arm against every other.

            Anchors are added on top of *any* schedule while *play anchors* is
            on, so a cross or round robin still yields ladder-scale Elo.
            """
        ),
    ])
    return (schedule_select,)


@app.cell
def _(mo, version_names, versions):
    _default_a = version_names[-1] if version_names else None
    _default_b = version_names[-2] if len(version_names) > 1 else _default_a

    run_a = mo.ui.dropdown(
        options=version_names, value=_default_a, label="run A", searchable=True
    )
    run_b = mo.ui.dropdown(
        options=version_names, value=_default_b, label="run B (cross only)",
        searchable=True,
    )
    _modes = ["argmax", "sample", "sample@0.02", "sample@0.05", "sample@0.1", "sample@0.2"]
    modes_a = mo.ui.multiselect(
        options=_modes, value=["argmax", "sample@0.05"], label="modes for A"
    )
    modes_b = mo.ui.multiselect(
        options=_modes, value=["sample@0.05"], label="modes for B (cross only)"
    )
    extra_modes = mo.ui.text(
        value="", label="extra modes (comma-separated, e.g. sample@0.01)",
        full_width=True,
    )
    mo.vstack([
        mo.hstack([run_a, modes_a], justify="start", gap=1.5),
        mo.hstack([run_b, modes_b], justify="start", gap=1.5),
        extra_modes,
    ])
    return extra_modes, modes_a, modes_b, run_a, run_b


@app.cell
def _(mo, run_a, run_b, versions):
    # Separate cell from the one that creates run_a/run_b: marimo forbids reading
    # a UI element's value in the cell that built it, and this must react to the
    # dropdowns anyway so the step lists follow the selected runs.
    def _steps(version):
        return sorted(
            versions.get(version or "", {}),
            key=lambda s: int(s.rsplit("_", 1)[1]),
        )

    _a, _b = _steps(run_a.value), _steps(run_b.value)
    steps_a = mo.ui.multiselect(
        options=_a, value=_a[-1:], label="A checkpoints", full_width=True
    )
    steps_b = mo.ui.multiselect(
        options=_b, value=_b[-1:], label="B checkpoints (cross only)", full_width=True
    )
    mo.vstack([
        steps_a,
        steps_b,
        mo.md(
            f"*Run A has {len(_a)} checkpoints, run B has {len(_b)}. Select every "
            "step of a run to trace how the argmax/sampling gap evolves over "
            "training — it should widen as the policy sharpens.*"
        ),
    ])
    return steps_a, steps_b


@app.cell
def _(mo):
    mo.md(r"""
    ### Anchor pool
    """)
    return


@app.cell
def _(arena, mo):
    use_anchors = mo.ui.switch(value=True, label="play anchors (needed for ladder-scale Elo)")
    anchor_top = mo.ui.slider(
        2, 60, value=12, step=1, label="top-N ladder players", show_value=True
    )
    anchor_baselines = mo.ui.switch(value=True, label="include baseline bots")
    anchor_lo = mo.ui.number(value=300, start=-500, stop=2000, label="anchor Elo ≥")
    anchor_hi = mo.ui.number(value=2000, start=-500, stop=2000, label="anchor Elo ≤")
    anchor_min_games = mo.ui.number(
        value=200, start=0, stop=10000, label="anchor min ladder games"
    )
    anchor_mode = mo.ui.text(value=arena.REFERENCE_MODE, label="anchor policy mode")
    mo.vstack([
        mo.hstack([use_anchors, anchor_baselines], justify="start", gap=1.5),
        mo.hstack([anchor_top, anchor_min_games], justify="start", gap=1.5),
        mo.hstack([anchor_lo, anchor_hi, anchor_mode], justify="start", gap=1.5),
        mo.md(
            f"""
            Anchors must replay the mode the ladder was built with —
            `{arena.REFERENCE_MODE}` — or their frozen ratings no longer describe
            the players actually on the board. Spread the pool around your arms'
            expected strength: anchors far above or below contribute almost no
            information per game (they win or lose nearly all of them), so a
            *narrow, well-matched* band beats a wide one at equal cost.
            `anchor min ladder games` filters out anchors whose own rating is
            noisy — their uncertainty is ignored by the frozen fit, so a
            thinly-played anchor quietly injects bias.
            """
        ),
    ])
    return (
        anchor_baselines,
        anchor_hi,
        anchor_lo,
        anchor_min_games,
        anchor_mode,
        anchor_top,
        use_anchors,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Compute
    """)
    return


@app.cell
def _(mo, torch):
    games_per_pair = mo.ui.slider(
        2, 200, value=20, step=2, label="games per pair", show_value=True
    )
    workers = mo.ui.slider(1, 32, value=8, step=1, label="workers", show_value=True)
    seconds = mo.ui.number(value=900, start=10, stop=3600, label="clock / player (s)")
    net_cache = mo.ui.slider(1, 32, value=8, step=1, label="nets cached / worker", show_value=True)
    seed_input = mo.ui.number(value=0, start=0, stop=10**6, label="seed")
    _has_cuda = torch.cuda.is_available()
    device_select = mo.ui.radio(
        options=["auto", "cpu", "cuda"] if _has_cuda else ["cpu"],
        value="auto" if _has_cuda else "cpu",
        label="device", inline=True,
    )
    collect_stats = mo.ui.switch(value=True, label="record policy statistics")
    run_button = mo.ui.run_button(label="▶ Play scheduled games")

    mo.vstack([
        mo.hstack([games_per_pair, workers, net_cache], justify="start", gap=1.5),
        mo.hstack([seconds, seed_input, device_select], justify="start", gap=1.5),
        mo.hstack([collect_stats, run_button], justify="start", gap=1.5),
        mo.md(
            "*Games per pair is per **unordered** pair, colors alternating; keep "
            "it even so White's first-move advantage cancels. Elo precision goes "
            "as ~347/√games — 20 games is ±78 Elo, 100 is ±35, 400 is ±17. A "
            "mode difference is typically tens of Elo, so budget accordingly.*"
        ),
    ])
    return (
        collect_stats,
        device_select,
        games_per_pair,
        net_cache,
        run_button,
        seconds,
        seed_input,
        workers,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Schedule preview
    """)
    return


@app.cell
def _(
    anchor_baselines,
    anchor_hi,
    anchor_lo,
    anchor_min_games,
    anchor_mode,
    anchor_top,
    arena,
    checkpoints,
    extra_modes,
    leaderboard,
    modes_a,
    modes_b,
    run_a,
    run_b,
    schedule_select,
    steps_a,
    steps_b,
    use_anchors,
    versions,
):
    def _mode_list(selected):
        extra = [m.strip() for m in extra_modes.value.split(",") if m.strip()]
        return list(dict.fromkeys(list(selected) + extra))

    def build_plan():
        """Arms + unordered pairs implied by the current controls.

        Returns ``(arms, pairs, anchor_names, problems)``. ``problems`` are
        human-readable reasons the plan is not runnable, surfaced *before* any
        game is played rather than after an hour of compute.
        """
        problems: list[str] = []
        ckpts_a = {s: versions[run_a.value][s] for s in steps_a.value} if run_a.value else {}
        ckpts_b = {s: versions[run_b.value][s] for s in steps_b.value} if run_b.value else {}
        try:
            arms_a = arena.make_arms(ckpts_a, _mode_list(modes_a.value))
            arms_b = arena.make_arms(ckpts_b, _mode_list(modes_b.value))
        except ValueError as e:
            return {}, [], [], [str(e)]

        mode = schedule_select.value
        arms = list(arms_a) + (list(arms_b) if mode == "cross" else [])
        by_name = {a.name: a for a in arms}
        if len(by_name) != len(arms):
            problems.append("duplicate arm names — A and B overlap; vary the mode")
        arms = list(by_name.values())
        if not arms:
            problems.append("no arms selected")

        if mode == "anchors":
            pairs = []
            if not use_anchors.value:
                problems.append("'modes vs anchors' needs the anchor pool switched on")
        elif mode == "paired":
            pairs = arena.pairs_paired_modes(arms)
            if not pairs:
                problems.append("paired head-to-head needs ≥2 modes on one checkpoint")
        elif mode == "cross":
            pairs = arena.pairs_cross(arms_a, arms_b)
            if not pairs:
                problems.append("cross needs checkpoints selected on both sides")
        else:
            pairs = arena.pairs_round_robin(arms)

        anchor_names: list[str] = []
        if use_anchors.value:
            anchor_names = arena.anchor_pool(
                leaderboard, checkpoints,
                top=int(anchor_top.value),
                include_baselines=anchor_baselines.value,
                elo_range=(float(anchor_lo.value), float(anchor_hi.value)),
                min_games=int(anchor_min_games.value),
            )
            # An arm's own checkpoint may sit in the pool under its ladder name.
            # A different name and mode makes it a legitimate opponent, but an
            # exact name collision would be self-play, so drop those.
            anchor_names = [n for n in anchor_names if n not in by_name]
            try:
                anchor_arms = [
                    a for n in anchor_names
                    if (a := arena.anchor_arm(n, checkpoints, anchor_mode.value))
                ]
            except ValueError as e:
                return {}, [], [], [f"anchor mode: {e}"]
            pairs = pairs + arena.pairs_cross(arms, anchor_arms)
            arms = arms + anchor_arms
            if not anchor_names:
                problems.append("anchor pool is empty — widen the Elo range or top-N")
            if anchor_mode.value.strip() != arena.REFERENCE_MODE:
                problems.append(
                    f"anchor mode is {anchor_mode.value!r}, not the ladder's "
                    f"{arena.REFERENCE_MODE!r} — the frozen ratings will not apply"
                )
        return {a.name: a for a in arms}, pairs, anchor_names, problems

    plan_arms, plan_pairs, plan_anchors, plan_problems = build_plan()
    return plan_anchors, plan_arms, plan_pairs, plan_problems


@app.cell
def _(
    arena,
    games_per_pair,
    mo,
    out_input,
    plan_anchors,
    plan_arms,
    plan_pairs,
    plan_problems,
    seconds,
    seed_input,
    workers,
):
    _prior = arena.load_rows(out_input.value)
    _preview_tasks = arena.build_tasks(
        plan_pairs, int(games_per_pair.value), arena.existing_counts(_prior),
        seconds=float(seconds.value), seed=int(seed_input.value),
    )
    _engine = arena.count_engine_games(_preview_tasks, plan_arms) if _preview_tasks else 0
    _fast = len(_preview_tasks) - _engine
    # rough wall clock: engine-backed bots ~40 s/game, net-vs-net ~2 s/game
    _est = (_engine * 40 + _fast * 2) / max(int(workers.value), 1) / 60
    _clash = arena.check_arms(out_input.value, plan_arms)

    _warn = "\n".join(f"* ⚠️ {p}" for p in plan_problems)
    if _clash:
        _warn += (
            "\n* ⚠️ these names are already in the results file under a "
            f"**different** definition: `{', '.join(_clash)}` — use a fresh "
            "results file or rename, otherwise two agents merge into one rating"
        )

    mo.md(
        f"""
        **{len(plan_arms)}** players ({len(plan_anchors)} of them anchors) ·
        **{len(plan_pairs)}** pairs · **{len(_preview_tasks)}** games left to play
        ({_engine} against engine bots) · **~{_est:.0f} min** wall clock at
        {workers.value} workers.

        {len(_prior)} games already recorded in `{out_input.value}`; those pairs
        resume instead of replaying. The exact schedule is recomputed from disk
        when you press run, so this preview being a little stale is harmless.
        {_warn}
        """
    )
    return


@app.cell
def _(mo, plan_anchors, plan_arms):
    mo.accordion({
        f"Arms ({len(plan_arms) - len(plan_anchors)})": mo.md(
            "\n".join(
                f"* `{a.name}` — {a.mode}"
                for a in plan_arms.values()
                if a.name not in plan_anchors
            ) or "*none*"
        ),
        f"Anchors ({len(plan_anchors)})": mo.md(
            ", ".join(f"`{n}`" for n in plan_anchors) or "*none*"
        ),
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Play
    """)
    return


@app.cell
def _(mo):
    # Bumped when a run finishes, so the analysis below refreshes without being a
    # *descendant* of the run cell -- descendants of a `mo.stop` never execute,
    # which would make the results unreadable until a run happened.
    get_stamp, set_stamp = mo.state(0)
    return get_stamp, set_stamp


@app.cell
def _(
    arena,
    collect_stats,
    device_select,
    games_per_pair,
    mo,
    net_cache,
    out_input,
    plan_arms,
    plan_pairs,
    plan_problems,
    run_button,
    seconds,
    seed_input,
    set_stamp,
    time,
    workers,
):
    mo.stop(
        not run_button.value,
        mo.md("⏸️ Review the schedule above, then press **▶ Play scheduled games**."),
    )
    mo.stop(
        bool(plan_problems),
        mo.md(f"🛑 Fix the schedule first: {'; '.join(plan_problems)}"),
    )

    # Recompute what is missing straight from disk: the preview above may be
    # stale (it does not re-read after a run), and scheduling from a stale count
    # would replay games that are already recorded.
    _tasks = arena.build_tasks(
        plan_pairs,
        int(games_per_pair.value),
        arena.existing_counts(arena.load_rows(out_input.value)),
        seconds=float(seconds.value),
        seed=int(seed_input.value),
    )
    arena.save_arms(out_input.value, plan_arms)

    _played = _errors = 0
    if _tasks:
        _t0 = time.time()
        with mo.status.progress_bar(
            total=len(_tasks), title="Playing games", show_eta=True, show_rate=True
        ) as _bar:
            for _row in arena.run_games(
                _tasks, plan_arms,
                workers=int(workers.value),
                device=None if device_select.value == "auto" else device_select.value,
                net_cache=int(net_cache.value),
                collect_stats=collect_stats.value,
                out_path=out_input.value,
            ):
                _played += 1
                _errors += int(_row["winner"] == "error")
                _bar.update(subtitle=(
                    f"{_played}/{len(_tasks)} · "
                    f"{arena.eta_string(_played, len(_tasks), time.time() - _t0)}"
                    + (f" · {_errors} errored" if _errors else "")
                ))
    set_stamp(time.time())

    mo.md(
        (
            f"✅ Played **{_played}** games"
            + (f", **{_errors}** errored" if _errors else "")
            + f". Appended to `{out_input.value}`."
        )
        if _played
        else "✅ Nothing left to play — this schedule is already complete on disk."
    )
    return


@app.cell
def _(arena, get_stamp, out_input):
    # Re-read from disk rather than keeping the streamed rows, so the analysis
    # also covers games played by tools/policy_eval.py or in an earlier session.
    _ = get_stamp()
    rows = arena.load_rows(out_input.value)
    all_arms = arena.load_arms(out_input.value)
    return all_arms, rows


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Results
    """)
    return


@app.cell
def _(mo):
    scope_select = mo.ui.radio(
        options={"this schedule": "plan", "everything in the file": "all"},
        value="this schedule", label="analyse", inline=True,
    )
    scope_select
    return (scope_select,)


@app.cell
def _(
    all_arms,
    arena,
    checkpoints,
    leaderboard,
    plan_arms,
    rows,
    scope_select,
):
    scoped_arms = dict(plan_arms) if scope_select.value == "plan" else dict(all_arms)
    # Frozen ratings for every anchor present. Anchors keep their ladder name
    # precisely so this lookup works.
    anchor_elo = {
        name: leaderboard[name]["elo"]
        for name in scoped_arms
        if name in leaderboard and (name in checkpoints or name in arena.BASELINES)
    }
    results = arena.fit_arms(arena.summarize(rows, scoped_arms), anchor_elo)
    played_arms = {n: r for n, r in results.items() if r.games or r.errors}
    return anchor_elo, played_arms, results


@app.cell
def _(anchor_elo, mo, played_arms, rows):
    _err = sum(r.errors for r in played_arms.values()) // 2
    mo.md(
        f"""
        {len(rows)} rows in the file · {len(played_arms)} players with games ·
        {len(anchor_elo)} of them rated anchors (Elo held fixed)
        {f"· ⚠️ {_err} errored games excluded" if _err else ""}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Anchored Elo per arm
    """)
    return


@app.cell
def _(anchor_elo, mo, pl, played_arms):
    _rows = []
    for _name, _r in played_arms.items():
        _anchored = _name in anchor_elo
        _rows.append({
            "arm": _name,
            "mode": _r.arm.mode if _r.arm else "",
            "role": "anchor" if _anchored else "arm",
            "elo": round(
                anchor_elo[_name] if _anchored
                else (_r.fit.elo if _r.fit else float("nan")), 1
            ),
            "±": round(_r.fit.se, 1) if _r.fit and not _anchored else None,
            "games": _r.games,
            "score": round(_r.score_rate, 3),
            "draw%": round(100 * _r.draw_rate, 1),
            "white%": round(100 * _r.white_rate, 1),
            "turns": round(_r.avg_turns, 1),
            "note": "unbounded (perfect/null record)"
            if _r.fit and _r.fit.degenerate else "",
        })
    elo_table = (
        pl.DataFrame(_rows).sort("elo", descending=True, nulls_last=True)
        if _rows else pl.DataFrame()
    )
    mo.vstack([
        mo.ui.table(elo_table, selection=None, page_size=25)
        if _rows else mo.md("*No games yet.*"),
        mo.md(
            "`elo` for an arm is the conditional MLE against the frozen anchors "
            "and `±` its 1σ error; anchors show their fixed ladder rating. "
            "`white%` is the score as White — a large departure from the arm's "
            "overall `score` flags a color imbalance in the schedule, which "
            "biases the rating."
        ),
    ])
    return (elo_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Mode deltas

    The headline number: same weights, two modes, difference in anchored Elo.
    The two fits are independent — disjoint games, fixed anchors — so the error
    of the difference is the quadrature sum, and `z ≥ 1.96` means the gap clears
    95% significance. A wide interval is not a null result; it means the game
    budget was too small to resolve the gap.
    """)
    return


@app.cell
def _(arena, mo, pl, played_arms, results):
    def _short(spec):
        return spec.rsplit("/", 1)[-1].replace("deepnash_async_", "").replace(".pt", "")

    _by_spec = {}
    for _n, _r in played_arms.items():
        if _r.arm and _r.arm.is_net and _r.fit is not None:
            _by_spec.setdefault(_r.arm.spec, []).append(_n)

    _rows = []
    for _spec, _names in _by_spec.items():
        for _i in range(len(_names)):
            for _j in range(_i + 1, len(_names)):
                _a, _b = _names[_i], _names[_j]
                # orient as argmax − sampled so the sign always reads the same way
                if results[_b].arm.greedy and not results[_a].arm.greedy:
                    _a, _b = _b, _a
                _d = arena.mode_delta(results, _a, _b)
                _rows.append({
                    "checkpoint": _short(_spec),
                    "A": results[_a].arm.mode,
                    "B": results[_b].arm.mode,
                    "Δ elo (A−B)": round(_d["delta"], 1),
                    "±": round(_d["se"], 1),
                    "z": round(_d["z"], 2),
                    "95%": "✓" if _d["significant"] else "",
                    "A elo": round(_d["a_elo"], 1),
                    "B elo": round(_d["b_elo"], 1),
                    "note": "unbounded fit" if _d.get("degenerate") else "",
                })
    delta_table = pl.DataFrame(_rows).sort("checkpoint") if _rows else pl.DataFrame()
    (
        mo.ui.table(delta_table, selection=None, page_size=25)
        if _rows
        else mo.md("*Needs ≥2 modes on one checkpoint, both fit against anchors.*")
    )
    return (delta_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Direct head-to-head

    Same weights facing each other. Perfectly paired — identical network,
    identical opponent, only the selection rule differs — but it measures only
    which mode beats *its twin*. A mode can beat its twin and still do worse
    against the field; that combination is exactly what "sharper but more
    exploitable" looks like, so read this next to the anchored deltas rather
    than instead of them.
    """)
    return


@app.cell
def _(arena, mo, pl, played_arms, rows):
    def _short(spec):
        return spec.rsplit("/", 1)[-1].replace("deepnash_async_", "").replace(".pt", "")

    _by_spec = {}
    for _n, _r in played_arms.items():
        if _r.arm and _r.arm.is_net:
            _by_spec.setdefault(_r.arm.spec, []).append(_n)

    _h2h = []
    for _spec, _names in _by_spec.items():
        for _i in range(len(_names)):
            for _j in range(_i + 1, len(_names)):
                _a, _b = _names[_i], _names[_j]
                if played_arms[_b].arm.greedy and not played_arms[_a].arm.greedy:
                    _a, _b = _b, _a
                _m = arena.head_to_head(rows, _a, _b)
                if not _m["games"]:
                    continue
                _elo = _m["elo"]
                _h2h.append({
                    "checkpoint": _short(_spec),
                    "A": played_arms[_a].arm.mode,
                    "B": played_arms[_b].arm.mode,
                    "games": _m["games"],
                    "A score": round(_m["rate"], 3),
                    "95% CI": f"{_m['ci_low']:.3f}–{_m['ci_high']:.3f}",
                    "Δ elo": round(_elo, 1) if abs(_elo) != float("inf") else None,
                    "sig": "✓" if _m["significant"] else "",
                    "A as white": _m["a_white_games"],
                })
    h2h_table = pl.DataFrame(_h2h).sort("checkpoint") if _h2h else pl.DataFrame()
    (
        mo.ui.table(h2h_table, selection=None, page_size=25)
        if _h2h
        else mo.md("*No direct arm-vs-arm games — use the 'paired head-to-head' schedule.*")
    )
    return (h2h_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Policy statistics

    How far apart the modes actually are, per decision. `agree%` is how often the
    sampled action *was* the argmax — the ceiling on any behavioural difference.
    `cut mass%` is how much policy mass the truncation threshold discarded, which
    is the meaningful measure of how aggressive a threshold is (the *count* of
    truncated decisions is useless — with thousands of legal moves, essentially
    every decision has something below any threshold).

    `bypass%` is the catch. When *no* action clears the threshold, `RNaDPlayer`
    falls back to the raw policy and the threshold does nothing, so truncation is
    **not monotone**: a higher threshold can discard *less* mass because it trips
    the fallback far more often. Read `cut mass%` and `bypass%` together — a
    threshold with high `bypass%` is not the setting you think you configured.

    Sense and move heads are split because they behave nothing alike: the sense
    policy is frequently near-uniform (many squares are genuinely equivalent), so
    argmax there is a far larger intervention than on the usually-peaked move
    head — and a threshold that works on the move head may be bypassed entirely
    on the sense head.
    """)
    return


@app.cell
def _(mo, pl, played_arms):
    _rows = []
    for _name, _r in played_arms.items():
        for _head, _s in (("sense", _r.sense), ("move", _r.move)):
            if not _s.decisions:
                continue
            _rows.append({
                "arm": _name,
                "mode": _r.arm.mode if _r.arm else "",
                "head": _head,
                "decisions": _s.decisions,
                "agree%": round(100 * _s.argmax_agree / _s.decisions, 1),
                "entropy (bits)": round(_s.entropy_bits / _s.decisions, 3),
                "top-1 p": round(_s.top1_prob / _s.decisions, 3),
                "legal": round(_s.legal / _s.decisions, 1),
                "cut mass%": round(100 * _s.truncated_mass / _s.decisions, 1),
                "bypass%": round(100 * _s.threshold_bypassed / _s.decisions, 1),
            })
    stats_table = (
        pl.DataFrame(_rows).sort(["arm", "head"]) if _rows else pl.DataFrame()
    )
    (
        mo.ui.table(stats_table, selection=None, page_size=25)
        if _rows
        else mo.md("*No policy statistics recorded — switch on 'record policy statistics'.*")
    )
    return (stats_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Elo across training steps
    """)
    return


@app.cell
def _(mo, played_arms, plt):
    def _step_of(name):
        try:
            return int(name.split("·")[0].rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return None

    _series = {}
    for _name, _r in played_arms.items():
        _step = _step_of(_name)
        if _r.fit is None or not _r.arm or not _r.arm.is_net or _step is None:
            continue
        _run = _name.split("·")[0].rsplit("_", 1)[0]
        _series.setdefault((_run, _r.arm.mode), []).append((_step, _r.fit.elo, _r.fit.se))

    if not _series:
        _out = mo.md("*Nothing to plot yet.*")
    else:
        _fig, _ax = plt.subplots(figsize=(8, 4.5))
        for (_run, _mode), _pts in sorted(_series.items()):
            _pts.sort()
            _ax.errorbar(
                [p[0] for p in _pts], [p[1] for p in _pts], yerr=[p[2] for p in _pts],
                marker="o", capsize=3, label=f"{_run} · {_mode}",
                linestyle="--" if _mode == "argmax" else "-",
            )
        _ax.set_xlabel("training step")
        _ax.set_ylabel("anchored Elo (ladder scale)")
        _ax.set_title("Evaluation policy mode across training")
        _ax.grid(alpha=0.3)
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Elo vs sampling threshold

    Argmax is the limit of truncated sampling as the threshold approaches the
    top-1 probability, so this curve spans the whole design space: threshold 0 is
    the raw policy, argmax the far end. A flat curve means the mode simply does
    not matter for this checkpoint; a peak in the middle means truncation buys
    the blunder-tail removal without paying the full exploitability cost of
    determinism.
    """)
    return


@app.cell
def _(mo, played_arms, plt):
    _curves = {}
    for _name, _r in played_arms.items():
        if _r.fit is None or not _r.arm or not _r.arm.is_net:
            continue
        _key = _name.split("·")[0]
        _c = _curves.setdefault(_key, {"pts": [], "argmax": None})
        if _r.arm.greedy:
            _c["argmax"] = (_r.fit.elo, _r.fit.se)
        else:
            _c["pts"].append((_r.arm.threshold, _r.fit.elo, _r.fit.se))

    _plottable = {k: v for k, v in _curves.items() if len(v["pts"]) >= 2 or v["argmax"]}
    if not _plottable:
        _out = mo.md("*Needs several sampling thresholds on one checkpoint.*")
    else:
        _fig, _ax = plt.subplots(figsize=(8, 4.5))
        for _key, _v in sorted(_plottable.items()):
            _v["pts"].sort()
            _color = None
            if _v["pts"]:
                _bars = _ax.errorbar(
                    [p[0] for p in _v["pts"]], [p[1] for p in _v["pts"]],
                    yerr=[p[2] for p in _v["pts"]], marker="o", capsize=3, label=_key,
                )
                _color = _bars.lines[0].get_color()
            if _v["argmax"]:
                _ax.axhline(_v["argmax"][0], linestyle=":", color=_color, alpha=0.8)
                _ax.annotate(f"{_key} argmax", (0.0, _v["argmax"][0]),
                             fontsize=7, color=_color, va="bottom")
        _ax.set_xlabel("sample_threshold  (0 = raw policy)")
        _ax.set_ylabel("anchored Elo (ladder scale)")
        _ax.set_title("Truncated sampling; dotted lines = argmax")
        _ax.grid(alpha=0.3)
        _ax.legend(fontsize=8)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Export
    """)
    return


@app.cell
def _(delta_table, elo_table, h2h_table, mo, stats_table):
    def _csv(df):
        return (lambda: df.write_csv().encode()) if df.height else (lambda: b"")

    mo.hstack([
        mo.download(_csv(elo_table), filename="policy_eval_elo.csv", label="Elo table"),
        mo.download(_csv(delta_table), filename="policy_eval_deltas.csv", label="Mode deltas"),
        mo.download(_csv(h2h_table), filename="policy_eval_h2h.csv", label="Head-to-head"),
        mo.download(_csv(stats_table), filename="policy_eval_stats.csv", label="Policy stats"),
    ], justify="start", gap=1)
    return


if __name__ == "__main__":
    app.run()
