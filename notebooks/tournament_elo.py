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
def _():
    # ================= ALIAS SCHEME (edit here) =============================
    # Every run is shown by a short, config-derived *alias* instead of its bare
    # version number, in all plots and tables. The alias is:
    #
    #     <family>[·<deviation>...]·s<seed>
    #
    # where <family> is the model family and each <deviation> is a hyper-param
    # that differs from that family's baseline (nothing is printed when a run is
    # at baseline, so a plain "CNN·s0" is the reference config):
    #
    #     family   CNN (resnet / channel-stacked) | GRU | LSTM | TFM | xLSTM
    #     η<e>     regularisation eta      (baseline 0.2)
    #     H<h>     observation history     (CNN baseline 16; sequence nets are 1)
    #     c<c>     conv channels           (baseline 128)
    #     b<n>     residual blocks         (baseline 6)
    #     lr1e-4   peak learning rate      (baseline 5e-5)
    #     greedy   argmax self-play        (baseline = sampled)
    #     L<n>     mixer layers   (sequence only, baseline 2)
    #     e<n>     encoder blocks (sequence only, baseline 4)
    #
    # Genuine re-runs that share an identical config+seed get the bare version
    # appended so aliases stay unique, e.g. "CNN·η0.5·s0 (v0.14.0)".
    #
    # To hand-name a run, add "version": "my name" to ALIAS_OVERRIDE below; that
    # string is used verbatim and wins over the auto scheme.
    FAMILY = {
        None: "CNN", "resnet": "CNN",
        "gru": "GRU", "lstm": "LSTM", "transformer": "TFM", "xlstm": "xLSTM",
    }
    ALIAS_OVERRIDE = {
        "v0.14.0": "CNN·η0.5·best·s0",  # top of the seed distribution (see thesis)
    }

    def core_alias(cfg: dict) -> str:
        """Family + deviations-from-baseline, no seed. All members of a seed-
        replicate group share this string, so it doubles as the group title."""
        net, rnad, enc, train = (
            cfg["network"], cfg["rnad"], cfg["encoding"], cfg["train"]
        )
        fam = FAMILY.get(net.get("arch"), net.get("arch") or "CNN")
        num = lambda x: f"{x:g}"
        tags = []
        if rnad.get("eta") != 0.2:
            tags.append(f"η{num(rnad.get('eta'))}")
        if fam == "CNN" and enc.get("history") != 16:
            tags.append(f"H{enc.get('history')}")
        if net.get("channels") != 128:
            tags.append(f"c{net.get('channels')}")
        if net.get("blocks") != 6:
            tags.append(f"b{net.get('blocks')}")
        if rnad.get("lr") != 5e-5:
            tags.append("lr1e-4")
        if train.get("selfplay_sample", True) is False:
            tags.append("greedy")
        if fam != "CNN":
            if net.get("mixer_layers") not in (2, None):
                tags.append(f"L{net.get('mixer_layers')}")
            if net.get("enc_blocks") not in (4, None):
                tags.append(f"e{net.get('enc_blocks')}")
        return fam + ("·" + "·".join(tags) if tags else "")

    return ALIAS_OVERRIDE, core_alias


@app.cell
def _(ALIAS_OVERRIDE, ckpt_input, core_alias, json, pl):
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
                    "core": core_alias(cfg),  # family + deviations, no seed
                    "blocks": net.get("blocks"),
                    "channels": net.get("channels"),
                    "history": enc.get("history"),
                    "eta": rnad.get("eta"),
                    "lr": rnad.get("lr"),
                    "mixer_layers": net.get("mixer_layers"),
                    "enc_blocks": net.get("enc_blocks"),
                }
            )

        # Per-version alias: "<core>·s<seed>", uniquified by appending the bare
        # version when two genuine re-runs share config+seed; overrides win.
        from collections import Counter as _Counter

        _auto = {
            r["version"]: f"{r['core']}·s{r['seed']}"
            for r in recs if r["version"] not in ALIAS_OVERRIDE
        }
        _clash = _Counter(_auto.values())
        for r in recs:
            v = r["version"]
            if v in ALIAS_OVERRIDE:
                r["alias"] = ALIAS_OVERRIDE[v]
            else:
                base = _auto[v]
                r["alias"] = base if _clash[base] == 1 else f"{base} ({v})"

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

    # A net group's title is the config core its members share (e.g. "CNN·η0.5").
    # A few early baseline configs collapse to the same core while differing on
    # fields the alias doesn't surface (iteration_steps, batch, amp, ...), so when
    # a core is used by more than one group we anchor it with the earliest version.
    from collections import Counter as _Counter

    _net_core = {
        (gid[0] if isinstance(gid, tuple) else gid): sub["core"][0]
        for gid, sub in nets.group_by("group_id")
        if (gid[0] if isinstance(gid, tuple) else gid).startswith("net:")
    }
    _dupe_core = {c for c, k in _Counter(_net_core.values()).items() if k > 1}

    # Build human labels per group_id.
    def _group_label(gid: str, members) -> str:
        rows = members.sort("vsort")
        if gid.startswith("seq:"):
            return gid.split(":", 1)[1].upper()  # GRU / LSTM / TRANSFORMER / XLSTM
        core = rows["core"][0]
        return f"{core} ({rows['version'][0]})" if core in _dupe_core else core

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
def _(mo, nets, pl):
    # The alias table: the config-derived name every plot/table uses in place of
    # the version number. One row per run, with the key hyper-params it encodes
    # and how many checkpoints of it are on the ladder.
    _alias_tbl = (
        nets.group_by("version")
        .agg(
            pl.col("alias").first(),
            pl.col("group").first().alias("group"),
            pl.col("arch").first(),
            pl.col("eta").first(),
            pl.col("history").first(),
            pl.col("lr").first(),
            pl.col("seed").first(),
            pl.col("step").n_unique().alias("checkpoints"),
            pl.col("elo").max().alias("peak_elo"),
            pl.col("vsort").first(),
        )
        .sort("vsort")
        .select(
            "version", "alias", "group", "arch", "eta", "history", "lr",
            "seed", "checkpoints", "peak_elo",
        )
    )
    mo.vstack(
        [
            mo.md("### Alias table — version ↔ name used everywhere below"),
            mo.ui.table(_alias_tbl, selection=None, page_size=60),
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
def _(lb, mo, nets, pl):
    # Attach the run alias to each checkpoint row; baselines keep their own name.
    _alias_of = dict(
        nets.select("version", "alias").unique().iter_rows()
    )
    _lb2 = lb.with_columns(
        pl.col("version")
        .replace_strict(_alias_of, default=None)
        .fill_null(pl.col("player"))
        .alias("alias")
    )
    mo.ui.table(
        _lb2.select("alias", "player", "elo", "games", "score", "se", "version", "step")
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

    def plot_group(ax, sub, band: bool, refs: bool, legend_versions: bool = True):
        """Draw one group's Elo-vs-step curves onto `ax`.

        `sub` is the joined `nets` rows for a single group. One line per version
        (a seed for replicate groups, a model variant for sequence families),
        labelled by its config-derived alias. (`legend_versions` is retained for
        call-site compatibility; the alias already carries seed/variant.)
        """
        versions = sub.sort("vsort")["version"].unique(maintain_order=True).to_list()
        for i, ver in enumerate(versions):
            v = sub.filter(sub["version"] == ver).sort("step")
            color = PALETTE[i % len(PALETTE)]
            xs = v["step"].to_list()
            ys = v["elo"].to_list()
            lbl = v["alias"][0]
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

    When the selected runs have **vastly different checkpoint lengths** the plot
    gets lopsided, so the *checkpoint alignment* control trims the x-axis:

      * **all checkpoints** — no trimming (every run drawn to its last step).
      * **up to a chosen step** — drag the slider to a common cutoff; every run is
        clipped to `step ≤ cutoff`.
      * **shared range** — clip all runs to the shortest run's last step, so the
        overlay only spans steps *every* selected run reached.
      * **up to each run's best** — clip **each** run at its own peak-Elo step, so
        no line extends past where that run stopped improving.
    """)
    return


@app.cell
def _(mo, nets):
    _rows = (
        nets.select("version", "alias", "vsort").unique().sort("vsort")
    )
    _alias_by_ver = dict(zip(_rows["version"].to_list(), _rows["alias"].to_list()))
    # Options map the displayed alias (label) -> version; `.value` returns versions.
    _options = {a: v for v, a in _alias_by_ver.items()}
    # Initial selection must be given as option LABELS (the aliases), not versions.
    # Default to the argmax self-play sweep (sampled 52/53 vs greedy 54/55).
    _pref = [
        _alias_by_ver[v]
        for v in ("v0.52.0", "v0.53.0", "v0.54.0", "v0.55.0")
        if v in _alias_by_ver
    ]
    ver_ms = mo.ui.multiselect(
        options=_options,
        value=_pref or _rows["alias"].to_list()[-4:],
        label="versions to compare",
    )
    ver_ms
    return (ver_ms,)


@app.cell
def _(mo, nets, ver_ms):
    # Checkpoint-alignment control for the overlay below. The slider is only used
    # by the "up to a chosen step" mode; it snaps to exactly the checkpoint steps
    # that actually exist across the selected runs (their union), so every stop is
    # a real checkpoint of at least one selected model.
    _sel = list(ver_ms.value)
    _steps = sorted(
        nets.filter(nets["version"].is_in(_sel))["step"].unique().to_list()
    ) if _sel else []

    align_dd = mo.ui.dropdown(
        options={
            "all checkpoints": "all",
            "up to a chosen step": "step",
            "shared range (shortest run)": "shared",
            "up to each run's best": "best",
        },
        value="all checkpoints",
        label="checkpoint alignment",
    )
    cutoff_slider = mo.ui.slider(
        steps=_steps or [0], value=(_steps[-1] if _steps else 0),
        label="cutoff step", show_value=True,
    )
    mo.hstack([align_dd, cutoff_slider], justify="start", gap=2)
    return align_dd, cutoff_slider


@app.cell
def _(align_dd, cutoff_slider, nets, pl, ver_ms):
    # Apply the chosen alignment to the selected versions -> `aligned_sub`, the
    # single frame consumed by both the overlay plot and the comparison table.
    _sel = list(ver_ms.value)
    aligned_sub = nets.filter(nets["version"].is_in(_sel)).sort(["vsort", "step"])
    _mode = align_dd.value
    if aligned_sub.height:
        if _mode == "step":
            aligned_sub = aligned_sub.filter(pl.col("step") <= cutoff_slider.value)
        elif _mode == "shared":
            # Trim every run to the shortest run's last step.
            _cut = (
                aligned_sub.group_by("version")
                .agg(pl.col("step").max().alias("m"))["m"].min()
            )
            aligned_sub = aligned_sub.filter(pl.col("step") <= _cut)
        elif _mode == "best":
            # Trim each run at its own earliest peak-Elo step.
            _best = aligned_sub.group_by("version").agg(
                pl.col("step")
                .filter(pl.col("elo") == pl.col("elo").max())
                .min()
                .alias("_bstep")
            )
            aligned_sub = (
                aligned_sub.join(_best, on="version")
                .filter(pl.col("step") <= pl.col("_bstep"))
                .drop("_bstep")
            )
    return (aligned_sub,)


@app.cell
def _(aligned_sub, band_toggle, mo, plot_group, plt, ref_toggle, ver_ms):
    _sel = list(ver_ms.value)
    _sub = aligned_sub
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
def _(aligned_sub, mo, pl):
    # Counts/peaks are computed over `aligned_sub`, so they reflect the trimmed
    # x-range currently shown in the overlay above.
    _tbl = (
        aligned_sub
        .group_by("version")
        .agg(
            pl.col("alias").first(),
            pl.col("seed").first(),
            pl.col("group").first().alias("group"),
            pl.col("group_kind").first(),
            pl.col("step").n_unique().alias("checkpoints"),
            pl.col("step").max().alias("last_step"),
            pl.col("elo").max().alias("peak_elo"),
            pl.col("vsort").first(),
        )
        .sort("vsort")
        .select(
            "version", "alias", "seed", "group", "group_kind",
            "checkpoints", "last_step", "peak_elo",
        )
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
            pl.col("alias").first(),
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
    _ax.set_yticklabels(_peak["alias"].to_list(), fontsize=7)
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
