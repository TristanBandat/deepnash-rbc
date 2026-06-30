# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.9", "pandas>=2.0", "matplotlib>=3.7"]
# ///
"""Interactive explorer for a DeepNash-RBC metrics JSONL.

Built to make sense of the legacy flat ``checkpoints/metrics.jsonl`` -- a single
file the old (pre-versioned) trainer appended to across every run and resume, so
it interleaves false starts, full runs, and overlapping step ranges left behind
when a run logged past its last saved checkpoint.

The notebook splits the file into segments (a fresh start or a resume each begins
a new one), then reconstructs the *canonical* training curve by applying the same
rule the trainer now applies on resume: when a run resumes from checkpoint step
X, everything after X from earlier segments is stale and dropped.

Run:  uv run --group notebooks marimo edit notebooks/metrics_explore.py
"""

from __future__ import annotations

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return Path, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Metrics explorer

        Point at a metrics JSONL below. The legacy default is the flat
        `checkpoints/metrics.jsonl` that mixes every run and resume together; this
        notebook untangles it into segments and a single canonical training curve.
        """
    )
    return


@app.cell
def _(Path, mo):
    # Default to the repo's legacy flat file, resolved relative to this notebook so
    # it works regardless of the directory marimo is launched from.
    _default = Path(__file__).resolve().parent.parent / "checkpoints" / "metrics.jsonl"
    path_input = mo.ui.text(
        value=str(_default), label="metrics JSONL path", full_width=True
    )
    path_input
    return (path_input,)


@app.cell
def _(np, pd, path_input):
    def load_metrics(path: str) -> pd.DataFrame:
        """Read a metrics JSONL into a DataFrame, unioning the train/skill columns.

        Adds ``line`` (file order) and a unified ``step`` (async ``step`` falling
        back to the sync trainer's ``iter``)."""
        df = pd.read_json(path, lines=True)
        df["line"] = np.arange(len(df))
        if "step" not in df.columns:
            df["step"] = np.nan
        if "iter" in df.columns:
            df["step"] = df["step"].fillna(df["iter"])
        return df

    df_raw = load_metrics(path_input.value)
    return (df_raw,)


@app.cell
def _(df_raw, mo):
    mo.md(
        f"""
        **Loaded** `{df_raw.shape[0]:,}` rows x `{df_raw.shape[1]}` columns.
        Types present: `{df_raw['type'].value_counts().to_dict()}`.
        """
    )
    return


@app.cell
def _(df_raw, np, pd):
    def add_segments(df: pd.DataFrame) -> pd.DataFrame:
        """Tag each row with a ``segment`` id. A new segment begins wherever wall
        time or step goes backwards -- i.e. at every fresh start and every resume."""
        df = df.copy()
        reset = (df["wall_s"].diff() < 0) | (df["step"].diff() < 0)
        df["segment"] = reset.cumsum().astype(int)
        return df

    def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("segment")
        s = pd.DataFrame(
            {
                "lines": g.size(),
                "step_min": g["step"].min().astype(int),
                "step_max": g["step"].max().astype(int),
                "wall_min_s": g["wall_s"].min(),
                "wall_max_s": g["wall_s"].max(),
            }
        )
        s["base"] = s["step_min"] - 1  # checkpoint step a resume started from
        s["kind"] = np.where(s["step_min"] <= 1, "fresh start", "resume")
        s["dur_h"] = (s["wall_max_s"] - s["wall_min_s"]) / 3600
        return s

    df_seg = add_segments(df_raw)
    summary = segment_summary(df_seg)
    return add_segments, df_seg, segment_summary, summary


@app.cell
def _(mo, summary):
    mo.md("## Segments")
    return


@app.cell
def _(mo, summary):
    mo.ui.table(
        summary.reset_index().round({"wall_min_s": 1, "wall_max_s": 1, "dur_h": 2}),
        selection=None,
    )
    return


@app.cell
def _(np, summary):
    def reconstruct(df_seg, summary):
        """Drop stale rows and return the canonical, deduplicated training curve.

        A segment's rows survive only up to the smallest resume-base of any LATER
        segment (a later resume from step B invalidates everything after B). The
        survivors then concatenate into one gapless, monotonic step series. Also
        adds ``cum_wall_s`` / ``cum_wall_h``: real cumulative wall-clock stitched
        across segments (each segment's clock restarts at 0 in the raw file)."""
        bases = summary["base"].to_numpy()
        seg_ids = summary.index.to_numpy()
        # suffix-min of base over strictly-later segments (inf for the last)
        suffix_min = np.empty(len(bases), dtype=float)
        running = np.inf
        for i in range(len(bases) - 1, -1, -1):
            suffix_min[i] = running
            running = min(running, bases[i])
        ceil_map = dict(zip(seg_ids, suffix_min))

        df = df_seg.copy()
        df["ceil"] = df["segment"].map(ceil_map)
        canon = (
            df[df["step"] <= df["ceil"]]
            .drop_duplicates("step", keep="last")
            .sort_values("step")
            .reset_index(drop=True)
        )

        # stitch wall-clock: each surviving segment continues from where the
        # previous one's last kept row ended (its per-resume warmup becomes a real
        # gap, exactly as the resume-aware MetricsLogger now records it).
        seg_max = canon.groupby("segment")["wall_s"].max()
        offset, acc = {}, 0.0
        for s in sorted(canon["segment"].unique()):
            offset[s] = acc
            acc += float(seg_max[s])
        canon["cum_wall_s"] = canon["wall_s"] + canon["segment"].map(offset)
        canon["cum_wall_h"] = canon["cum_wall_s"] / 3600
        return canon

    return (reconstruct,)


@app.cell
def _(df_seg, reconstruct, summary):
    canon = reconstruct(df_seg, summary)
    return (canon,)


@app.cell
def _(canon, df_raw, mo, summary):
    _n_fresh = int((summary["kind"] == "fresh start").sum())
    _n_resume = int((summary["kind"] == "resume").sum())
    _wasted = len(df_raw) - len(canon)
    _hours = canon["cum_wall_h"].max()
    mo.md(
        f"""
        ## What's in this file

        - **{len(summary)} segments**: {_n_fresh} fresh start(s), {_n_resume} resume(s).
        - Canonical run after dropping stale/overlap rows:
          **step {int(canon['step'].min()):,} -> {int(canon['step'].max()):,}**,
          `{len(canon):,}` rows, monotonic & gapless = `{canon['step'].is_monotonic_increasing and canon['step'].nunique() == len(canon)}`.
        - **{_wasted:,} rows discarded** as logged-past-checkpoint or false-start cruft.
        - Stitched real training time: **{_hours:.1f} h**
          (the raw file's per-segment clocks hide this; the versioned trainer now
          continues `wall_s` across resumes so this is recorded directly).
        - Skill evals kept: `{int((canon['type'] == 'skill').sum())}`.
        """
    )
    return


@app.cell
def _(df_seg, mo, np, plt):
    mo.md("### Raw timeline — step vs file position, colored by segment")
    # Decimate: 1M+ points would choke the renderer; stride to ~20k.
    _stride = max(1, len(df_seg) // 20000)
    _d = df_seg.iloc[::_stride]
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _sc = _ax.scatter(_d["line"], _d["step"], c=_d["segment"], s=4, cmap="tab10")
    _ax.set_xlabel("file line")
    _ax.set_ylabel("step")
    _ax.set_title("resumes show as drops; overlaps are where a run logged past its checkpoint")
    _fig.colorbar(_sc, ax=_ax, label="segment")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(canon, mo):
    metric_select = mo.ui.dropdown(
        options=["loss", "policy_loss", "value_loss", "entropy"],
        value="loss",
        label="training metric",
    )
    smooth_slider = mo.ui.slider(
        start=1, stop=5000, value=500, step=1, label="smoothing window (steps)"
    )
    x_axis = mo.ui.radio(
        options=["step", "cum_wall_h"], value="step", label="x-axis", inline=True
    )
    mo.hstack([metric_select, smooth_slider, x_axis], justify="start", gap=2)
    return metric_select, smooth_slider, x_axis


@app.cell
def _(canon, metric_select, np, plt, smooth_slider, x_axis):
    _train = canon[canon["type"] == "train"]
    _m = metric_select.value
    _y = _train[_m].rolling(smooth_slider.value, min_periods=1).mean()
    _x = _train[x_axis.value]
    # decimate for rendering after smoothing
    _stride = max(1, len(_train) // 8000)
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.plot(_x.iloc[::_stride], _y.iloc[::_stride], lw=0.9)
    _ax.set_xlabel(x_axis.value)
    _ax.set_ylabel(_m)
    _ax.set_title(f"{_m} (rolling mean, window={smooth_slider.value})")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(canon, mo):
    mo.md("### Skill — win-rate vs fixed baselines (the curve that should rise)")
    return


@app.cell
def _(canon, plt):
    _skill = canon[canon["type"] == "skill"].sort_values("step")
    _cols = [c for c in _skill.columns if c.startswith("vs_") and not c.endswith(("_draw", "_plies", "_n"))]
    _fig, _ax = plt.subplots(figsize=(9, 4))
    for _c in _cols:
        _s = _skill.dropna(subset=[_c])
        _ax.plot(_s["step"], _s[_c], marker="o", ms=3, lw=1, label=_c)
    _ax.set_xlabel("step")
    _ax.set_ylabel("win rate")
    _ax.set_ylim(0, 1.02)
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
