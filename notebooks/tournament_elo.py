# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.9", "polars>=1.0", "matplotlib>=3.7"]
# ///
"""Internal-tournament Elo explorer for the DeepNash-RBC checkpoint ladder.

The round-robin tournament (`tools/tournament.py`) plays every saved checkpoint
`v<version>_<step>` against every other and against the fixed baselines
(random / trout / mht), then fits a Bradley-Terry/Elo leaderboard anchored at
random = 0. This notebook reads that leaderboard and reconstructs, per training
run, how internal Elo *rises across the saved checkpoints* -- i.e. the learning
curve measured against the whole field rather than against fixed baselines.

Runs are grouped for plotting:

  * **Seed replicates** -- versions whose configs match once the RNG seed, paths,
    idle/electricity schedule and pure schema-drift defaults are normalised out
    share one graph (each seed a separate line). This includes the channel-stacked
    `resnet` runs; `network.arch == "resnet"` is NOT a sequence model even though
    newer configs always serialise an `arch` field.
  * **Sequence-model families** -- only the streaming-state archs
    (gru / lstm / transformer / xlstm) are grouped by *model type*; every version
    of a type shares one graph regardless of its other hyper-params.

Config schema has drifted over the project's life (later runs serialise fields
that early runs never had: `arch`, the temporal-mixer knobs, the lr-schedule
knobs, `selfplay_sample`, ...). To keep a run grouped with its same-experiment
replicates across that drift, the signature drops the temporal-only fields and
back-fills the post-hoc fields with the value an old run implicitly ran at
(e.g. `selfplay_sample=True`, `lr_schedule="constant"`) before hashing. So
v0.14.0 and its later argmax-sweep re-runs v0.52/0.53 land in one group, while
the greedy arm v0.54/0.55 (`selfplay_sample=False`) forms its own.

Elo is read from the rendered leaderboard text (the canonical, anchored fit);
grouping is read from each run's `checkpoints/v*/config.json`. All frames are
polars.

Run:  uv run --group notebooks marimo edit notebooks/tournament_elo.py
"""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    return Path, json, mo, pl, plt, re


@app.cell
def _(mo):
    mo.md(r"""
    # Internal-tournament Elo across training

    Elo comes from the round-robin ladder over **all** saved checkpoints plus
    the fixed baselines; it is anchored so **random = 0**. For each run we plot
    Elo against checkpoint step -- the internal skill curve -- and overlay the
    `trout` and `mht` baselines as reference lines.
    """)
    return


@app.cell
def _(Path, mo):
    # Resolve defaults relative to this notebook so launch directory doesn't matter.
    _root = Path(__file__).resolve().parent.parent
    lb_input = mo.ui.text(
        value=str(_root / "results" / "tournament_leaderboard.txt"),
        label="leaderboard .txt",
        full_width=True,
    )
    ckpt_input = mo.ui.text(
        value=str(_root / "checkpoints"),
        label="checkpoints dir (for config.json grouping)",
        full_width=True,
    )
    mo.vstack([lb_input, ckpt_input])
    return ckpt_input, lb_input


@app.cell
def _(Path, pl, re):
    def parse_leaderboard(path: str) -> pl.DataFrame:
        """Parse a rendered `=== Leaderboard ===` block into a tidy frame.

        Each body row is `name elo games score +/-`; the player name carries no
        spaces. Net players are `v<version>_<step>`; everything else is a baseline.
        """
        rows = []
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            name, elo, games, score, se = parts
            if not re.fullmatch(r"-?\d+", elo):  # skip header / decoration lines
                continue
            m = re.fullmatch(r"(v\d+\.\d+(?:\.\d+)?)_(\d+)", name)
            rows.append(
                {
                    "player": name,
                    "version": m.group(1) if m else None,
                    "step": int(m.group(2)) if m else None,
                    "is_net": m is not None,
                    "elo": int(elo),
                    "games": int(games),
                    "score": float(score),
                    "se": int(se),
                }
            )
        return pl.DataFrame(rows)

    return (parse_leaderboard,)


@app.cell
def _(lb_input, parse_leaderboard):
    lb = parse_leaderboard(lb_input.value)
    nets_lb = lb.filter(lb["is_net"])
    baselines_lb = lb.filter(~lb["is_net"])
    return baselines_lb, lb, nets_lb


@app.cell
def _(baselines_lb, lb, mo, nets_lb):
    _b = {r["player"]: r["elo"] for r in baselines_lb.iter_rows(named=True)}
    mo.md(
        f"""
        **Leaderboard loaded.** {lb.height} players
        (`{nets_lb.height}` checkpoints across
        `{nets_lb['version'].n_unique()}` runs, `{baselines_lb.height}` baselines).
        Baseline anchors: {', '.join(f'`{k}={v}`' for k, v in sorted(_b.items(), key=lambda kv: -kv[1]))}.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Grouping runs from their configs

    A run's *param signature* is its `config.json` with seed, paths, device and
    bookkeeping fields stripped. Non-sequence runs sharing a signature are seed
    replicates and share a graph. Sequence runs (those with `network.arch`) are
    instead grouped by **model type**.
    """)
    return


@app.cell
def _(ckpt_input, json, pl):
    # Only these archs are streaming-state "sequence models"; resnet is the plain
    # channel-stacked net even though newer configs always serialise arch="resnet".
    SEQ_ARCHS = {"gru", "lstm", "transformer", "xlstm"}

    # Leaf config keys that never define a resnet "experiment", so they must not
    # split a run from its replicates:
    #   * seed / paths / device / bookkeeping (progress, cadence, run length)
    #   * fork_source (provenance only)
    #   * the idle/electricity schedule -- WHEN the rig trains, not what it learns
    #   * eval-only knobs -- how the fitted net is *scored*, not how it trained
    #   * the temporal-mixer fields -- ignored by resnet; sequence runs are grouped
    #     by arch (their signature is never consulted), so drop them everywhere
    #   * arch itself -- handled by the seq/net group prefix, not the signature
    _DROP = {
        "seed", "metrics_path", "checkpoint_dir", "resume", "device",
        "progress", "checkpoint_every", "total_iters", "fork_source",
        "idle_schedule", "train_start_hour", "train_stop_hour", "train_days",
        "eval_every", "eval_games", "eval_sample", "eval_opponents",
        "arch", "enc_blocks", "mixer_dim", "mixer_layers", "nhead", "max_seq",
        "xlstm_slstm_at", "xlstm_conv_kernel",
    }

    # Fields added to the config schema after the earliest runs. A config that
    # predates a field implicitly ran at this value, so back-fill it before hashing
    # -- otherwise pure schema drift would split a run from its newer-schema, same-
    # experiment replicates (this is exactly what put v0.14.0 and its argmax-sweep
    # re-runs v0.52/0.53 in different groups). Keyed by "<section>.<leaf>".
    _BACKFILL = {
        "rnad.lr_schedule": "constant",
        "rnad.lr_warmup": 0,
        "rnad.lr_decay_start": 0,
        "rnad.lr_min": 0.0,
        "train.selfplay_sample": True,
    }

    def _signature(cfg: dict) -> str:
        flat = {
            f"{sec}.{k}": v
            for sec, sub in cfg.items()
            for k, v in sub.items()
            if k not in _DROP
        }
        for key, default in _BACKFILL.items():
            flat.setdefault(key, default)
        return json.dumps(flat, sort_keys=True)

    def _vkey(v: str) -> int:
        # "v0.14.0" -> 14000, "v0.7.1" -> 7001; a sortable release-order key
        # (polars can't min() a tuple column, so keep it an int).
        maj, mnr, pat = (int(x) for x in v[1:].split("."))
        return maj * 1_000_000 + mnr * 1_000 + pat

    def load_versions(ckpt_dir: str) -> pl.DataFrame:
        """One row per run that has a config.json, with its group assignment."""
        from pathlib import Path as _P

        recs = []
        for cfg_path in sorted(_P(ckpt_dir).glob("v*/config.json")):
            version = cfg_path.parent.name
            cfg = json.loads(cfg_path.read_text())
            net, rnad, enc, train = (
                cfg["network"], cfg["rnad"], cfg["encoding"], cfg["train"]
            )
            arch = net.get("arch")  # None on pre-arch configs; "resnet" or a seq arch
            recs.append(
                {
                    "version": version,
                    "vsort": _vkey(version),
                    "seed": train.get("seed"),
                    # resnet (and pre-arch configs) are NOT sequence models
                    "is_seq": arch in SEQ_ARCHS,
                    "arch": arch,
                    "signature": _signature(cfg),
                    "blocks": net.get("blocks"),
                    "channels": net.get("channels"),
                    "history": enc.get("history"),
                    "eta": rnad.get("eta"),
                    "lr": rnad.get("lr"),
                    "mixer_layers": net.get("mixer_layers"),
                    "enc_blocks": net.get("enc_blocks"),
                }
            )
        df = pl.DataFrame(recs)

        # Group id: sequence runs by arch; others (incl. resnet) by shared signature.
        df = df.with_columns(
            pl.when(pl.col("is_seq"))
            .then("seq:" + pl.col("arch"))
            .otherwise("net:" + pl.col("signature"))
            .alias("group_id")
        )
        return df

    versions = load_versions(ckpt_input.value)
    return (versions,)


@app.cell
def _(nets_lb, pl, versions):
    # Join Elo (per checkpoint) onto its run's grouping. Inner join: only runs that
    # actually appear in the leaderboard survive.
    nets = nets_lb.join(versions, on="version", how="inner").sort(["vsort", "step"])

    # Build human labels per group_id.
    def _group_label(gid: str, members) -> str:
        rows = members.sort("vsort")
        if gid.startswith("seq:"):
            return gid.split(":", 1)[1].upper()  # GRU / LSTM / TRANSFORMER / XLSTM
        vs = rows["version"].to_list()
        return "+".join(vs)  # e.g. v0.14.0+v0.27.0+v0.52.0+v0.53.0

    def _group_kind(gid: str) -> str:
        return "sequence-model family" if gid.startswith("seq:") else "seed replicates"

    _labels, _kinds, _sort = {}, {}, {}
    for gid, sub in nets.group_by("group_id"):
        gid = gid[0] if isinstance(gid, tuple) else gid
        _labels[gid] = _group_label(gid, sub.unique("version"))
        _kinds[gid] = _group_kind(gid)
        _sort[gid] = (0 if gid.startswith("net:") else 1, sub["vsort"].min())

    nets = nets.with_columns(
        pl.col("group_id").replace_strict(_labels).alias("group"),
        pl.col("group_id").replace_strict(_kinds).alias("group_kind"),
    )
    group_order = sorted(_labels, key=lambda g: _sort[g])
    return group_order, nets


@app.cell
def _(mo, nets, pl):
    _summary = (
        nets.group_by("group", "group_kind")
        .agg(
            pl.col("version").n_unique().alias("runs"),
            pl.col("step").n_unique().alias("checkpoints"),
            pl.col("elo").max().alias("peak_elo"),
        )
        .sort(["group_kind", "peak_elo"], descending=[False, True])
    )
    mo.vstack(
        [
            mo.md("### Groups discovered"),
            mo.ui.table(_summary, selection=None, page_size=40),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Full leaderboard
    """)
    return


@app.cell
def _(lb, mo):
    mo.ui.table(
        lb.select("player", "elo", "games", "score", "se", "version", "step")
        .sort("elo", descending=True),
        selection=None,
        page_size=20,
    )
    return


@app.cell
def _(mo):
    band_toggle = mo.ui.switch(value=True, label="show ±1σ Elo band")
    ref_toggle = mo.ui.switch(value=True, label="show trout / mht reference lines")
    mo.hstack([band_toggle, ref_toggle], justify="start", gap=2)
    return band_toggle, ref_toggle


@app.cell
def _(baselines_lb):
    # Validated, CVD-safe categorical hues (dataviz skill, light mode), fixed order.
    PALETTE = [
        "#2a78d6", "#1baf7a", "#eda100", "#008300",
        "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
    ]
    INK = "#0b0b0b"
    MUTED = "#898781"
    GRID = "#e1e0d9"
    REF = {  # baseline -> (color, linestyle)
        "mht": ("#52514e", "--"),
        "trout": ("#898781", ":"),
    }
    BASE_ELO = {r["player"]: r["elo"] for r in baselines_lb.iter_rows(named=True)}

    def plot_group(ax, sub, band: bool, refs: bool, legend_versions: bool):
        """Draw one group's Elo-vs-step curves onto `ax`.

        `sub` is the joined `nets` rows for a single group. One line per version
        (a seed for replicate groups, a model variant for sequence families).
        """
        versions = sub.sort("vsort")["version"].unique(maintain_order=True).to_list()
        for i, ver in enumerate(versions):
            v = sub.filter(sub["version"] == ver).sort("step")
            color = PALETTE[i % len(PALETTE)]
            xs = v["step"].to_list()
            ys = v["elo"].to_list()
            seed = v["seed"][0]
            if v["is_seq"][0]:  # sequence model: label by mixer/encoder depth
                lbl = f"{ver} · mix{v['mixer_layers'][0]}/enc{v['enc_blocks'][0]}"
            else:
                lbl = f"{ver} · seed {seed}" if legend_versions else ver
            ax.plot(xs, ys, "-o", ms=4, lw=2, color=color, label=lbl, zorder=3)
            if band:
                lo = [e - s for e, s in zip(ys, v["se"].to_list())]
                hi = [e + s for e, s in zip(ys, v["se"].to_list())]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.12, lw=0, zorder=1)
        if refs:
            for name, (c, ls) in REF.items():
                if name in BASE_ELO:
                    ax.axhline(BASE_ELO[name], color=c, ls=ls, lw=1.3, zorder=2)
                    ax.text(
                        ax.get_xlim()[1], BASE_ELO[name], f" {name}",
                        va="center", ha="left", fontsize=7, color=c,
                    )
        ax.axhline(0, color=GRID, lw=1, zorder=0)  # random = 0 anchor
        ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)

    return REF, plot_group


@app.cell
def _(mo):
    mo.md(r"""
    ## Seed-replicate runs — internal Elo across checkpoints

    One panel per config; each line is a seed. Spread between seeds is the
    run-to-run noise of the training procedure at fixed hyper-params.
    """)
    return


@app.cell
def _(band_toggle, group_order, mo, nets, plot_group, plt, ref_toggle):
    def small_multiples(kind: str, ncols: int):
        gids = [g for g in group_order]
        subs = []
        for gid in gids:
            sub = nets.filter(nets["group_id"] == gid)
            if sub.height and sub["group_kind"][0] == kind:
                subs.append((sub["group"][0], sub))
        if not subs:
            return mo.md(f"_no groups of kind {kind}._")
        nrows = (len(subs) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 3.4, nrows * 2.7),
            squeeze=False, sharey=False,
        )
        flat = axes.ravel()
        for ax, (title, sub) in zip(flat, subs):
            plot_group(
                ax, sub, band=band_toggle.value, refs=ref_toggle.value,
                legend_versions=(kind == "seed replicates"),
            )
            ax.set_title(title, fontsize=9, color="#0b0b0b")
            ax.legend(fontsize=6.5, frameon=False, loc="lower right")
        for ax in flat[len(subs):]:
            ax.set_visible(False)
        fig.supxlabel("checkpoint step", fontsize=9)
        fig.supylabel("internal Elo (random = 0)", fontsize=9)
        fig.tight_layout()
        return fig

    small_multiples("seed replicates", ncols=3)
    return (small_multiples,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Sequence-model families — grouped by model type

    Every `gru` / `lstm` / `transformer` run of a type shares a panel; lines are
    the individual runs (labelled with their mixer/encoder depth).
    """)
    return


@app.cell
def _(small_multiples):
    small_multiples("sequence-model family", ncols=3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Single-group explorer

    Pick a group for a large view of its Elo curve.
    """)
    return


@app.cell
def _(group_order, mo, nets):
    _label_by_gid = {
        gid: f"{nets.filter(nets['group_id'] == gid)['group'][0]}  "
        f"[{nets.filter(nets['group_id'] == gid)['group_kind'][0]}]"
        for gid in group_order
    }
    group_dd = mo.ui.dropdown(
        options={v: k for k, v in _label_by_gid.items()},
        value=next(iter(_label_by_gid.values())),
        label="group",
    )
    group_dd
    return (group_dd,)


@app.cell
def _(band_toggle, group_dd, nets, plot_group, plt, ref_toggle):
    _gid = group_dd.value
    _sub = nets.filter(nets["group_id"] == _gid)
    _fig, _ax = plt.subplots(figsize=(9, 5))
    plot_group(
        _ax, _sub, band=band_toggle.value, refs=ref_toggle.value,
        legend_versions=(_sub["group_kind"][0] == "seed replicates"),
    )
    _ax.set_title(
        f"{_sub['group'][0]}  —  {_sub['group_kind'][0]}",
        fontsize=12, color="#0b0b0b",
    )
    _ax.set_xlabel("checkpoint step")
    _ax.set_ylabel("internal Elo (random = 0)")
    _ax.legend(fontsize=9, frameon=False, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Free version comparison

    Pick **any** set of versions to overlay their Elo-vs-step curves on one axis,
    regardless of group — e.g. select the argmax-sweep runs
    `v0.52 / v0.53 / v0.54 / v0.55` (the default) to see the four together, or add
    `v0.14.0` to compare the sweep against its reference run. The table lists each
    selection's seed, its assigned group, **how many checkpoints** it has on the
    ladder, and its peak Elo.
    """)
    return


@app.cell
def _(mo, nets):
    _all = nets.sort("vsort")["version"].unique(maintain_order=True).to_list()
    # Default to the argmax self-play sweep (sampled 52/53 vs greedy 54/55).
    _pref = [v for v in ("v0.52.0", "v0.53.0", "v0.54.0", "v0.55.0") if v in _all]
    ver_ms = mo.ui.multiselect(
        options=_all,
        value=_pref or _all[-4:],
        label="versions to compare",
    )
    ver_ms
    return (ver_ms,)


@app.cell
def _(band_toggle, mo, nets, plot_group, plt, ref_toggle, ver_ms):
    _sel = list(ver_ms.value)
    _sub = nets.filter(nets["version"].is_in(_sel)).sort(["vsort", "step"])
    if _sub.height == 0:
        _out = mo.md("_pick at least one version above._")
    else:
        _fig, _ax = plt.subplots(figsize=(10, 5.5))
        plot_group(
            _ax, _sub, band=band_toggle.value, refs=ref_toggle.value,
            legend_versions=True,
        )
        _ax.set_title("Version comparison", fontsize=12, color="#0b0b0b")
        _ax.set_xlabel("checkpoint step")
        _ax.set_ylabel("internal Elo (random = 0)")
        _ncol = 2 if len(_sel) > 4 else 1
        _ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=_ncol)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(mo, nets, pl, ver_ms):
    _sel = list(ver_ms.value)
    _tbl = (
        nets.filter(nets["version"].is_in(_sel))
        .group_by("version")
        .agg(
            pl.col("seed").first(),
            pl.col("group").first().alias("group"),
            pl.col("group_kind").first(),
            pl.col("step").n_unique().alias("checkpoints"),
            pl.col("step").max().alias("last_step"),
            pl.col("elo").max().alias("peak_elo"),
            pl.col("vsort").first(),
        )
        .sort("vsort")
        .drop("vsort")
    )
    mo.ui.table(_tbl, selection=None, page_size=20)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cross-group comparison — peak Elo reached

    Best checkpoint Elo per run, so every experiment is comparable on one axis.
    """)
    return


@app.cell
def _(REF, baselines_lb, nets, pl, plt):
    _base_elo = {r["player"]: r["elo"] for r in baselines_lb.iter_rows(named=True)}
    _peak = (
        nets.group_by("version")
        .agg(
            pl.col("elo").max().alias("peak_elo"),
            pl.col("group").first(),
            pl.col("group_kind").first(),
            pl.col("vsort").first(),
            pl.col("se").first(),
        )
        .sort("peak_elo", descending=False)
    )
    _n = _peak.height
    _fig, _ax = plt.subplots(figsize=(9, max(4, _n * 0.28)))
    _colors = [
        "#2a78d6" if k == "seed replicates" else "#eb6834"
        for k in _peak["group_kind"].to_list()
    ]
    _y = range(_n)
    _ax.barh(
        list(_y), _peak["peak_elo"].to_list(), color=_colors,
        xerr=_peak["se"].to_list(), error_kw=dict(ecolor="#898781", lw=0.8),
        height=0.72, zorder=3,
    )
    _ax.set_yticks(list(_y))
    _ax.set_yticklabels(_peak["version"].to_list(), fontsize=7)
    from matplotlib.patches import Patch

    _ax.legend(
        handles=[
            Patch(color="#2a78d6", label="seed-replicate run"),
            Patch(color="#eb6834", label="sequence-model run"),
        ],
        fontsize=8, frameon=False, loc="lower right",
    )
    for _name, (_c, _ls) in REF.items():
        if _name in _base_elo:
            _ax.axvline(_base_elo[_name], color=_c, ls=_ls, lw=1.2, zorder=2)
            _ax.text(
                _base_elo[_name], _n - 0.5, f" {_name}",
                va="top", ha="left", fontsize=7, color=_c,
            )
    _ax.set_xlabel("peak internal Elo (random = 0)")
    _ax.grid(True, axis="x", color="#e1e0d9", lw=0.6)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
