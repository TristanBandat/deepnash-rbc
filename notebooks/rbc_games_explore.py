# /// script
# dependencies = [
#     "chess==1.11.2",
#     "marimo",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import chess
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from deepnash_rbc.analysis import rbc_games

    return chess, mo, pd, plt, rbc_games


@app.cell
def _(mo, rbc_games):
    mo.md(f"""
    # RBC ladder game explorer

    Games are cached in `{rbc_games.DEFAULT_CACHE}`. **Sync** scans the
    server's game log back to the date below and downloads any new
    histories (only new games are fetched; a full re-scan is a few hundred
    requests). Map new `version_id`s to checkpoints in
    `rbc_games.VERSION_CHECKPOINTS`.
    """)
    return


@app.cell
def _(mo):
    since_input = mo.ui.text(value="2026-07-06", label="scan back to (ISO date)")
    sync_button = mo.ui.run_button(label="Sync games from server")
    mo.hstack([since_input, sync_button], justify="start")
    return since_input, sync_button


@app.cell
def _(rbc_games, since_input, sync_button):
    if sync_button.value:
        rbc_games.sync(since=since_input.value)
    games_df = rbc_games.features_frame()
    # unversioned games are manual guest invites, not ladder play
    games_df = games_df[
        games_df.finished & games_df.won.notna() & games_df.version_id.notna()
    ].copy()
    games_df["version"] = games_df.version_id.astype(int)
    games_df["won"] = games_df.won.astype(bool)
    return (games_df,)


@app.cell
def _(games_df, mo, pd):
    overview = games_df.groupby(["version", "checkpoint"]).agg(
        games=("game_id", "count"),
        win_rate=("won", "mean"),
        first_game=("time_created", "min"),
        last_game=("time_created", "max"),
    ).round(3).reset_index()
    by_opp = pd.pivot_table(
        games_df, index=["version", "checkpoint"], columns="opponent",
        values="won", aggfunc="mean",
    ).round(2)
    mo.vstack([
        mo.md("## Versions on the ladder"),
        mo.ui.table(overview, selection=None),
        mo.md(
            "**Win rate by opponent.** The ladder Elo is dominated by the"
            " ever-present server bots, so a version that drops `attacker`"
            " or `trout` pays for it directly."
        ),
        mo.ui.table(by_opp.reset_index(), selection=None),
    ])
    return


@app.cell
def _(games_df, mo):
    loss_anatomy = games_df[~games_df.won].groupby(["version", "opponent"]).agg(
        losses=("game_id", "count"),
        median_death_turn=("our_moves", "median"),
        king_captured=("win_reason", lambda s: (s == "KING_CAPTURE").mean()),
        timeout=("win_reason", lambda s: (s == "TIMEOUT").mean()),
        material_at_end=("material_diff_end", "mean"),
    ).round(2).reset_index()
    mo.vstack([
        mo.md(
            """
            ## Loss anatomy

            `median_death_turn` is our move count when the game ended: ~3-4
            against `attacker` means we lose to its fixed early queen/knight
            assault on the king; double-digit turns with negative
            `material_at_end` (our material minus theirs) means we were ground
            down positionally.
            """
        ),
        mo.ui.table(loss_anatomy, selection=None),
    ])
    return


@app.cell
def _(games_df, mo, plt):
    _vs_att = games_df[games_df.opponent == "attacker"]
    _versions = sorted(_vs_att.version.unique())
    _fig, _axes = plt.subplots(1, len(_versions), figsize=(2.4 * len(_versions), 2.6),
                               sharey=True, sharex=True)
    for _ax, _v in zip(_axes, _versions):
        _g = _vs_att[_vs_att.version == _v]
        for _won, _color in ((True, "tab:green"), (False, "tab:red")):
            _sel = _g[_g.won == _won]
            _ax.hist(_sel.our_moves, bins=range(0, 30, 2), alpha=0.6, color=_color)
        _ax.set_title(f"V{_v} ({_g.won.mean():.0%} win)")
        _ax.set_xlabel("our moves")
    _axes[0].set_ylabel("games")
    _fig.suptitle("vs attacker: game length, wins (green) vs losses (red)")
    _fig.tight_layout()
    mo.vstack([
        mo.md(
            """
            ## The attacker race

            `attacker` ignores everything and runs a scripted piece straight at
            the king; there is no check in RBC, so whoever captures the enemy
            king first wins. Every game against it is a race (or a dodge):
            losses at ~3-4 moves mean our own attack was a tempo too slow *and*
            we never moved our king out of the scripted mating square.
            """
        ),
        _fig,
    ])
    return


@app.cell
def _(chess, games_df, mo, pd, rbc_games):
    def _first_own_moves(game_id, us_white, n=4):
        h = rbc_games.load_history(game_id)
        if h is None:
            return None
        _us = chess.WHITE if us_white else chess.BLACK
        moves = []
        for t in h.turns(_us):
            if h.has_move(t):
                m = h.taken_move(t)
                moves.append(m.uci() if m else "pass")
            if len(moves) >= n:
                break
        return " ".join(moves)

    _rows = []
    for _, _r in games_df.iterrows():
        _sig = _first_own_moves(_r.game_id, _r.color == "white")
        if _sig:
            _rows.append({"version": _r.version, "color": _r.color, "opening": _sig})
    _sig_df = pd.DataFrame(_rows)
    opening_signature = (
        _sig_df.groupby(["version", "color"]).opening
        .agg(lambda s: s.value_counts().head(3).to_dict())
        .reset_index()
    )
    mo.vstack([
        mo.md(
            """
            ## Opening signature (our first 4 moves)

            A compact fingerprint of each version's policy. It also identifies
            *which checkpoint* an unlabeled server version was: compare its
            signature against candidates in the self-play tournament data.
            """
        ),
        mo.ui.table(opening_signature, selection=None),
    ])
    return


@app.cell
def _(chess, games_df, mo, plt, rbc_games):
    def _material_curve(game_id, us_white, max_turn=25):
        h = rbc_games.load_history(game_id)
        if h is None:
            return None
        _us = chess.WHITE if us_white else chess.BLACK
        curve = {}
        for t in h.turns(_us):
            if t.turn_number > max_turn or not h.has_move(t):
                break
            try:
                board = h.truth_board_after_move(t)
            except Exception:
                break
            curve[t.turn_number] = (rbc_games._material(board, _us)
                                    - rbc_games._material(board, not _us))
        return curve

    _vs_trout = games_df[games_df.opponent == "trout"]
    _fig2, _ax2 = plt.subplots(figsize=(7, 3.2))
    for _v in sorted(_vs_trout.version.unique()):
        _g = _vs_trout[_vs_trout.version == _v]
        _acc = {}
        for _, _r in _g.iterrows():
            _c = _material_curve(_r.game_id, _r.color == "white")
            for _t, _m in (_c or {}).items():
                _acc.setdefault(_t, []).append(_m)
        _turns = sorted(_acc)
        _ax2.plot(_turns, [sum(_acc[_t]) / len(_acc[_t]) for _t in _turns],
                  marker=".", label=f"V{_v}")
    _ax2.axhline(0, color="grey", lw=0.5)
    _ax2.set_xlabel("our move number")
    _ax2.set_ylabel("material diff (us - them)")
    _ax2.legend()
    _fig2.tight_layout()
    mo.vstack([
        mo.md(
            """
            ## Material vs trout

            `trout` is Stockfish behind a sense-tracked board: it punishes
            unsound aggression. A curve that dives negative early means the
            version donates pieces (typically the queen) to positions a real
            engine refutes, then gets ground down and king-captured.
            """
        ),
        _fig2,
    ])
    return


@app.cell
def _(games_df, mo):
    version_pick = mo.ui.dropdown(
        options=[str(v) for v in sorted(games_df.version.unique())],
        value=str(int(games_df.version.max())), label="version",
    )
    opponent_pick = mo.ui.dropdown(
        options=sorted(games_df.opponent.unique()), value="attacker", label="opponent",
    )
    result_pick = mo.ui.dropdown(options=["losses", "wins", "all"], value="losses",
                                 label="show")
    mo.hstack([version_pick, opponent_pick, result_pick], justify="start")
    return opponent_pick, result_pick, version_pick


@app.cell
def _(
    chess,
    games_df,
    mo,
    opponent_pick,
    rbc_games,
    result_pick,
    version_pick,
):
    def _movetext(game_id):
        h = rbc_games.load_history(game_id)
        if h is None:
            return "(no history cached)"
        _turns = sorted(t for c in (chess.WHITE, chess.BLACK)
                        for t in h.turns(c) if h.has_move(t))
        parts = []
        for t in _turns:
            m, r = h.taken_move(t), h.requested_move(t)
            _txt = m.uci() if m else "pass"
            if r is not None and m is not None and r != m:
                _txt += f"(req {r.uci()})"
            parts.append(f"{t.turn_number}{'W' if t.color else 'B'}:{_txt}")
        return " ".join(parts)

    _sel = games_df[(games_df.version == int(version_pick.value))
                    & (games_df.opponent == opponent_pick.value)]
    if result_pick.value != "all":
        _sel = _sel[_sel.won == (result_pick.value == "wins")]
    _lines = [
        f"**{_r.game_id}** ({_r.color}, {_r.win_reason}):  {_movetext(_r.game_id)}"
        for _, _r in _sel.head(20).iterrows()
    ]
    mo.vstack([
        mo.md("## Game drill-down\nTaken moves, both sides; `(req ...)` marks a"
              " move the server altered (blocked slide / illegal request)."),
        mo.md("\n\n".join(_lines) if _lines else "*no games match*"),
    ])
    return


if __name__ == "__main__":
    app.run()
