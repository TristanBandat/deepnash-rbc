# /// script
# dependencies = [
#     "marimo",
#     "polars",
#     "matplotlib",
# ]
# requires-python = ">=3.12"
# ///
"""Aggregate statistics over the full RBC game corpus.

The server publishes every game as one periodically-updated zip of
``<game_id>.json`` GameHistory files (public, ~1.3 GB, ~800k games). This
notebook downloads that archive once and computes population statistics --
game length, outcomes, win reasons, per-bot activity. The first full scan
parses the zip in parallel and caches the flat rows as parquet next to it;
every later run loads that cache in well under a second (polars).

For a focused view of deepnash-rbc's *own* ladder games, use
``rbc_games_explore.py`` instead.

Run:  uv run --group notebooks marimo edit notebooks/rbc_stats.py
"""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    from deepnash_rbc.analysis import rbc_games

    return Path, mo, pl, plt, rbc_games


@app.cell
def _(mo, rbc_games):
    mo.md(f"""
    # RBC corpus statistics

    Source: the server's full game-log archive
    [`{rbc_games.ARCHIVE_URL}`]({rbc_games.ARCHIVE_URL}) — every game as a
    `<game_id>.json` history (public, no login, ~1.3 GB, periodically
    updated). **Download** fetches it once to `{rbc_games.DEFAULT_ARCHIVE}`
    (resumable); statistics then come from a parallel scan of the zip.

    The **whole corpus** (~800k games) is analysed by default. The first
    scan takes a minute or two and caches its rows as parquet next to the
    zip — after that, recomputing is effectively instant. Untick **use all
    games** to draw a random subset instead.
    """)
    return


@app.cell
def _(mo, rbc_games):
    archive_input = mo.ui.text(value=str(rbc_games.DEFAULT_ARCHIVE),
                               label="archive zip", full_width=True)
    download_button = mo.ui.run_button(label="Download / resume archive")
    sample_input = mo.ui.number(value=25000, start=100, stop=1_000_000, step=1000,
                                label="sample size")
    all_games = mo.ui.checkbox(value=True, label="use all games")
    seed_input = mo.ui.number(value=0, start=0, stop=9999, step=1, label="seed")
    compute_button = mo.ui.run_button(label="Compute statistics")
    mo.vstack([
        mo.hstack([archive_input, download_button], justify="start", align="end"),
        mo.hstack([sample_input, all_games, seed_input, compute_button],
                  justify="start", align="end"),
    ])
    return (
        all_games,
        archive_input,
        compute_button,
        download_button,
        sample_input,
        seed_input,
    )


@app.cell
def _(Path, archive_input, download_button, mo, rbc_games):
    archive_path = Path(archive_input.value)
    if download_button.value:
        with mo.status.spinner(title="Downloading the RBC archive (~1.3 GB)…"):
            rbc_games.download_archive(archive_path, progress=False)
    _size = archive_path.stat().st_size / 1e9 if archive_path.exists() else 0
    mo.md(
        f"**Archive:** `{archive_path}` — "
        + (f"{_size:.2f} GB on disk." if archive_path.exists()
           else "*not downloaded yet — click Download.*")
    )
    return (archive_path,)


@app.cell
def _(
    all_games,
    archive_path,
    compute_button,
    mo,
    pl,
    rbc_games,
    sample_input,
    seed_input,
):
    games_df = pl.DataFrame()
    if compute_button.value and archive_path.exists():
        _sample = None if all_games.value else int(sample_input.value)
        with mo.status.spinner(
                title="Scanning the archive (first full scan takes a minute, "
                      "then it's cached)…"):
            games_df = rbc_games.archive_frame(
                archive_path, sample=_sample, seed=int(seed_input.value))
    _msg = (f"Loaded **{games_df.height:,}** games "
            f"({'all' if all_games.value else 'sampled'})."
            if not games_df.is_empty()
            else "*Download the archive, then click **Compute statistics**.*")
    mo.md(_msg)
    return (games_df,)


@app.cell
def _(games_df, mo, pl):
    if games_df.is_empty():
        _out = mo.md("")
    else:
        _dec = games_df.filter(pl.col("outcome").is_in(["white", "black"]))
        _draws = (games_df["outcome"] == "draw").sum()
        _L = games_df["total_plies"]
        _tbl = [
            ("games analysed", f"{games_df.height:,}"),
            ("decisive / draws",
             f"{_dec.height:,} ({_dec.height / games_df.height:.0%}) / "
             f"{_draws:,}"),
            ("median game length", f"{_L.median():.0f} plies"),
            ("mean game length", f"{_L.mean():.1f} plies"),
            ("longest game", f"{_L.max():.0f} plies"),
            ("captures / game (median)", f"{games_df['captures'].median():.0f}"),
        ]
        _out = mo.vstack([
            mo.md("## Headline numbers\nOne *ply* = one player's turn (a sense **and** a move)."),
            mo.md("| metric | value |\n|---|---|\n"
                  + "\n".join(f"| {k} | {v} |" for k, v in _tbl)),
        ])
    _out
    return


@app.cell
def _(games_df, mo, pl, plt):
    if games_df.is_empty():
        _out = mo.md("")
    else:
        _len = games_df["total_plies"]
        _qs = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        _desc = pl.DataFrame({
            "statistic": (["count", "mean", "std", "min"]
                          + [f"{int(q * 100)}%" for q in _qs] + ["max"]),
            "plies": ([float(_len.len()), round(_len.mean(), 1),
                       round(_len.std(), 1), float(_len.min())]
                      + [float(_len.quantile(q)) for q in _qs]
                      + [float(_len.max())]),
        })
        _fig, _ax = plt.subplots(figsize=(7, 3.2))
        _hi = 257  # just past the training code's maximum history size of 256
        _ax.hist(_len.clip(upper_bound=_hi).to_numpy(), bins=range(0, _hi + 2, 2),
                 color="tab:blue", alpha=0.8)
        _ax.axvline(_len.median(), color="black", ls="--", lw=1,
                    label=f"median {_len.median():.0f}")
        _ax.axvline(_len.mean(), color="tab:red", ls=":", lw=1,
                    label=f"mean {_len.mean():.1f}")
        _ax.axvline(256, color="tab:green", ls="-.", lw=1, label="history cap 256")
        _ax.set_xlabel(f"game length (plies, clipped at {_hi})")
        _ax.set_ylabel("games")
        _ax.legend()
        _fig.tight_layout()
        _p256 = (_len <= 256).mean()
        _p257 = (_len <= 257).mean()
        _out = mo.vstack([
            mo.md("## Game-length distribution\nRight-skewed: most games end "
                  "quickly by king capture, a long tail grinds on.\n\n"
                  f"A length of **256 plies is the {_p256:.4%} percentile**, "
                  f"**257 the {_p257:.4%} percentile** — so only "
                  f"**{1 - _p256:.4%}** of corpus games exceed the maximum "
                  "history size of 256 used in the training code."),
            mo.ui.table(_desc, selection=None),
            _fig,
        ])
    _out
    return


@app.cell
def _(games_df, mo, pl, plt):
    if games_df.is_empty():
        _out = mo.md("")
    else:
        _wr = games_df["win_reason"].value_counts(sort=True)
        _fig, _ax = plt.subplots(figsize=(6, 2.8))
        _ax.barh(_wr["win_reason"].cast(pl.Utf8).to_list()[::-1],
                 _wr["count"].to_list()[::-1],
                 color="tab:purple", alpha=0.8)
        _ax.set_xlabel("games")
        _fig.tight_layout()
        _dec = games_df.filter(pl.col("outcome").is_in(["white", "black"]))
        _white = ((_dec["outcome"] == "white").mean()
                  if _dec.height else float("nan"))
        _out = mo.vstack([
            mo.md(
                f"""
                ## Outcomes

                How games end, and the first-move (White) edge. White wins
                **{_white:.1%}** of decisive games; draws are
                **{(games_df["outcome"] == "draw").mean():.1%}** of all games
                (RBC's automatic 50-turn no-progress rule).
                """
            ),
            _fig,
        ])
    _out
    return


@app.cell
def _(games_df, mo, pl):
    if games_df.is_empty():
        _out = mo.md("")
    else:
        _long = pl.concat([
            games_df.select(pl.col("white").alias("bot"),
                            (pl.col("outcome") == "white").alias("won")),
            games_df.select(pl.col("black").alias("bot"),
                            (pl.col("outcome") == "black").alias("won")),
        ])
        _by_bot = (
            _long.group_by("bot")
            .agg(pl.len().alias("games"),
                 pl.col("won").mean().round(3).alias("win_rate"))
            .sort("games", descending=True)
        )
        _out = mo.vstack([
            mo.md("## Busiest bots\nGames played (either colour) and overall win "
                  "rate across the analysed corpus."),
            mo.ui.table(_by_bot, selection=None, page_size=15),
        ])
    _out
    return


if __name__ == "__main__":
    app.run()
