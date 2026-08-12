# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.9", "matplotlib>=3.7", "numpy>=1.24"]
# ///
"""Model-based analysis of DeepNash-RBC sense behavior.

The thesis figure ``ladder-sense-heatmap`` (scripts/make_experiment_figures.py)
is *data-based*: it counts, from played ladder games, which square each deployed
checkpoint sensed, aggregated over the whole game. This notebook is the
*model-based* counterpart. Instead of counting sampled choices from a game log,
it queries a checkpoint's **sense policy** directly -- the full 64-way softmax the
network puts over the board at each sense decision -- and resolves it *per sense
ordinal within the game* rather than lumping every sense together.

That per-ordinal split is the point. In RBC the player senses, then moves. White's
very first sense (ordinal 1) happens before the opponent has moved, so nothing is
yet unknown and the policy there is near-uniform / uninformative. The informative
opening senses are therefore:

    Black:  sense 1, 2, 3   (Black senses *after* White's first move)
    White:        2, 3      (White's sense 1 is skipped as uninformative)

To get a model-based distribution we roll the agent through the opening a number
of times (against itself or a baseline bot), and average the sense-policy vector
the network emits at each ordinal. Averaging the *policy mass* (not just the one
sampled square) is far lower variance than counting choices, so a few dozen short
rollouts already give a clean picture.

Everything is dynamic: pick a checkpoint, colors, opponent, how many senses, how
many rollouts, then render the per-ordinal heatmaps (thesis colormap + own-side-
down orientation) plus a handful of companion plots.

Run:  uv run --group notebooks marimo edit notebooks/sense_behavior.py
      uv run --group notebooks marimo run  notebooks/sense_behavior.py   # app mode
"""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    from collections import defaultdict
    from pathlib import Path

    import chess
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from reconchess import LocalGame
    from reconchess.bots.attacker_bot import AttackerBot
    from reconchess.bots.random_bot import RandomBot
    from reconchess.play import play_turn

    from deepnash_rbc.agent import RNaDPlayer
    from deepnash_rbc.play_session import load_net
    from deepnash_rbc.replay import SENSE

    return (
        AttackerBot,
        LocalGame,
        Path,
        RNaDPlayer,
        RandomBot,
        SENSE,
        chess,
        defaultdict,
        load_net,
        mo,
        mpl,
        np,
        play_turn,
        plt,
        re,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Sense behavior — model-based

    Query a checkpoint's **sense policy** at each opening sense and see *where*
    the network chooses to look. Unlike the thesis heatmap (which counts sensed
    squares over whole ladder games), this reads the network's full 64-way sense
    distribution directly and splits it **per sense ordinal**, so the trivial
    first White sense is separated from the informative ones.

    > Informative opening senses: **Black 1–3**, **White 2–3**
    > (White's sense 1 fires before the opponent has moved — near-uniform).
    """)
    return


@app.cell
def _(Path, re):
    # Discover every checkpoint under checkpoints/ and label it v<version>_<step>.
    def discover_checkpoints() -> dict[str, str]:
        root = Path(__file__).resolve().parent.parent / "checkpoints"
        pat = re.compile(r"deepnash_async_v(\d+\.\d+\.\d+)_(\d+)\.pt$")
        found: list[tuple[tuple[int, int, int, int], str, str]] = []
        for p in root.glob("v*/deepnash_async_v*_*.pt"):
            m = pat.search(p.name)
            if not m:
                continue
            ver, step = m.group(1), int(m.group(2))
            key = (*(int(x) for x in ver.split(".")), step)
            found.append((key, f"v{ver}_{step}", str(p)))
        found.sort()
        return {label: path for _, label, path in found}

    checkpoints = discover_checkpoints()
    return (checkpoints,)


@app.cell
def _(checkpoints, mo, torch):
    # Default to the thesis main model if present, else the newest discovered.
    _labels = list(checkpoints)
    _default = "v0.27.0_80000" if "v0.27.0_80000" in checkpoints else (_labels[-1] if _labels else None)

    ckpt_select = mo.ui.dropdown(
        options=_labels, value=_default, label="checkpoint", searchable=True
    )
    color_select = mo.ui.radio(
        options=["Both", "White", "Black"], value="Both", label="model color", inline=True
    )
    opponent_select = mo.ui.dropdown(
        options=["self", "random", "attacker", "trout"], value="self", label="opponent"
    )
    n_senses = mo.ui.slider(1, 8, value=3, step=1, label="senses per color (N)", show_value=True)
    n_games = mo.ui.slider(4, 200, value=40, step=4, label="rollout games / color", show_value=True)
    mode_select = mo.ui.radio(
        options=["policy mass (mean π)", "sampled counts"],
        value="policy mass (mean π)",
        label="quantity",
        inline=True,
    )
    sample_toggle = mo.ui.switch(value=True, label="stochastic rollouts (sample play)")
    seed_input = mo.ui.number(value=0, start=0, stop=10_000, label="seed")
    _has_cuda = torch.cuda.is_available()
    device_select = mo.ui.radio(
        options=["cpu", "cuda"] if _has_cuda else ["cpu"],
        value="cpu",
        label="device",
        inline=True,
    )
    run_button = mo.ui.run_button(label="▶ Run rollouts")

    mo.vstack(
        [
            mo.hstack([ckpt_select, opponent_select, device_select], justify="start", gap=1.5),
            mo.hstack([color_select, mode_select], justify="start", gap=1.5),
            mo.hstack([n_senses, n_games], justify="start", gap=1.5),
            mo.hstack([sample_toggle, seed_input, run_button], justify="start", gap=1.5),
        ]
    )
    return (
        ckpt_select,
        color_select,
        device_select,
        mode_select,
        n_games,
        n_senses,
        opponent_select,
        run_button,
        sample_toggle,
        seed_input,
    )


@app.cell
def _(
    AttackerBot,
    LocalGame,
    RNaDPlayer,
    RandomBot,
    SENSE,
    chess,
    defaultdict,
    np,
    play_turn,
):
    # ----------------------------------------------------------------- recorder
    class RecordingPlayer(RNaDPlayer):
        """RNaDPlayer that also stores the *full* masked sense distribution and the
        sampled square at every sense decision, tagged by its ordinal within the
        game (1 = this color's first sense). Reimplements ``choose_sense`` rather
        than wrapping it so the network is queried exactly once per sense -- calling
        ``_forward`` twice would double-advance a temporal net's recurrent state."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.sense_dists: list[tuple[int, np.ndarray]] = []  # (ordinal, probs[64])
            self.sense_picks: list[tuple[int, int]] = []  # (ordinal, square)

        def choose_sense(self, sense_actions, move_actions, seconds_left):
            if not sense_actions:
                return None
            _, sense_logits, _ = self._forward()
            legal = np.asarray(sense_actions, dtype=np.int64)
            probs = masked_sense_probs(sense_logits, legal)
            ordinal = len(self.sense_dists) + 1
            action, logp = self._sample_from(sense_logits, legal)
            self.sense_dists.append((ordinal, probs))
            self.sense_picks.append((ordinal, int(action)))
            self._record(SENSE, legal, action, logp)
            return int(action)

    def masked_sense_probs(sense_logits, legal: np.ndarray) -> np.ndarray:
        import torch

        logits = sense_logits.squeeze(0).float().cpu()
        mask = torch.full_like(logits, float("-inf"))
        idx = torch.from_numpy(legal)
        mask[idx] = logits[idx]
        return torch.softmax(mask, dim=0).numpy()

    # ----------------------------------------------------------------- opponents
    def build_opponent(name: str, net, device, history):
        if name == "self":
            return RecordingPlayer(net, device, history=history, sample=True)
        if name == "random":
            return RandomBot()
        if name == "attacker":
            return AttackerBot()
        if name == "trout":
            from deepnash_rbc.eval import _make_trout

            return _make_trout()
        raise ValueError(f"unknown opponent {name!r}")

    # ----------------------------------------------------------------- one game
    def play_opening(net, device, history, model_color, opponent, n_senses, sample):
        """Play a single game far enough for the recording side(s) to reach
        ``n_senses`` senses, then stop. Returns the recording players keyed by the
        color they actually played (both sides record in self-play)."""
        agent = RecordingPlayer(net, device, history=history, sample=sample)
        opp = build_opponent(opponent, net, device, history)
        white, black = (agent, opp) if model_color == chess.WHITE else (opp, agent)

        game = LocalGame(seconds_per_player=900)
        game.store_players(type(white).__name__, type(black).__name__)
        white.handle_game_start(chess.WHITE, game.board.copy(), type(black).__name__)
        black.handle_game_start(chess.BLACK, game.board.copy(), type(white).__name__)
        game.start()

        recorders = {c: p for c, p in ((chess.WHITE, white), (chess.BLACK, black))
                     if isinstance(p, RecordingPlayer)}
        players = [black, white]  # indexed by game.turn (0=black? see reconchess: True=White)
        # reconchess: game.turn is a chess color bool; players list is [black, white]
        cap = 2 * n_senses + 2  # safety: never loop forever
        plies = 0
        while not game.is_over() and plies < cap:
            play_turn(game, players[game.turn], end_turn_last=True)
            plies += 1
            if all(len(p.sense_dists) >= n_senses for p in recorders.values()):
                break
        return recorders

    # -------------------------------------------------- rollout batch (shared)
    def run_rollouts(net, device, history, want_colors, opponent, n_senses,
                     n_games, sample, progress=None):
        """Roll ``n_games`` openings and collect the sense-policy vectors and the
        sampled squares, keyed by ``(color, ordinal)``. In self-play one game
        yields both colors; otherwise the model's seat alternates so a two-color
        request gets balanced White/Black samples. Shared by the single-checkpoint
        deep-dive and the checkpoint comparison so both measure the same thing."""
        self_play = opponent == "self"
        vecs: dict[tuple, list] = defaultdict(list)
        picks: dict[tuple, list] = defaultdict(list)
        for g in range(n_games):
            model_color = chess.WHITE if self_play else want_colors[g % len(want_colors)]
            recorders = play_opening(net, device, history, model_color,
                                     opponent, n_senses, sample)
            for color, rp in recorders.items():
                if color not in want_colors:
                    continue
                for ordinal, p in rp.sense_dists:
                    vecs[(color, ordinal)].append(p)
                for ordinal, sq in rp.sense_picks:
                    picks[(color, ordinal)].append(sq)
            if progress is not None:
                progress()
        return vecs, picks

    return play_opening, run_rollouts


@app.cell
def _(
    checkpoints,
    chess,
    ckpt_select,
    color_select,
    device_select,
    load_net,
    mo,
    n_games,
    n_senses,
    np,
    opponent_select,
    run_button,
    run_rollouts,
    sample_toggle,
    seed_input,
    torch,
):
    mo.stop(
        not run_button.value or ckpt_select.value is None,
        mo.md("⏳ Choose settings and press **▶ Run rollouts**."),
    )

    torch.manual_seed(int(seed_input.value))
    np.random.seed(int(seed_input.value))

    _device = torch.device("cuda" if device_select.value == "cuda" else "cpu")
    _net, _enc = load_net(checkpoints[ckpt_select.value], _device)
    _history = _enc.history

    # which model colors to gather
    if color_select.value == "Both":
        _want = [chess.WHITE, chess.BLACK]
    elif color_select.value == "White":
        _want = [chess.WHITE]
    else:
        _want = [chess.BLACK]

    N = int(n_senses.value)
    with mo.status.progress_bar(total=int(n_games.value)) as _bar:
        vecs, picks = run_rollouts(
            _net, _device, _history, _want, opponent_select.value, N,
            int(n_games.value), sample_toggle.value, progress=_bar.update,
        )

    # aggregate: mean policy mass, sampled-count histogram, entropy per (color, ord)
    mass = {k: np.mean(np.stack(v), axis=0) for k, v in vecs.items() if v}
    counts = {}
    for k, v in picks.items():
        h = np.zeros(64)
        for sq in v:
            h[sq] += 1
        counts[k] = h
    entropy = {
        k: float(np.mean([-(p[p > 0] * np.log2(p[p > 0])).sum() for p in v]))
        for k, v in vecs.items() if v
    }
    nobs = {k: len(v) for k, v in vecs.items()}
    meta = {
        "checkpoint": ckpt_select.value,
        "opponent": opponent_select.value,
        "N": N,
        "arch": type(_net).__name__,
        "history": _history,
        "device": str(_device),
    }
    return N, counts, entropy, mass, meta, nobs


@app.cell
def _(meta, mo, nobs):
    _rows = " · ".join(
        f"{'W' if c else 'B'}#{o}: n={n}" for (c, o), n in sorted(nobs.items(), key=lambda x: (not x[0][0], x[0][1]))
    )
    mo.md(
        f"""
        **{meta['checkpoint']}** · {meta['arch']} (history={meta['history']}) ·
        opponent = `{meta['opponent']}` · device = `{meta['device']}`
        Samples per (color, ordinal): {_rows}
        """
    )
    return


@app.cell
def _(mpl, np):
    # ----- shared plotting helpers (thesis colormap + own-side-down orientation)
    TERRA = mpl.colors.LinearSegmentedColormap.from_list(
        "terra", ["#FCFCFB", "#F2CC8F", "#C4512F", "#7E2F16"]
    )

    def own_side_down(vec: np.ndarray, color: bool) -> np.ndarray:
        """Rotate a flat 64-vector to the own-side-down perspective. Black is a 180°
        board flip (flat index s -> 63-s), matching the thesis figure."""
        return vec if color else vec[::-1]

    def opp_half_mass(vec: np.ndarray, color: bool) -> float:
        """Percentage of sense mass on the opponent's half. In the own-side-down
        board the own ranks (1-4) are the bottom four rows and the opponent's ranks
        (5-8) the top four, so this is the mass on rows 4-7. 50 % = uniform; higher
        means the model looks toward the opponent (the mark of learned sensing)."""
        v = own_side_down(np.asarray(vec, float), color).reshape(8, 8)
        return float(v[4:].sum() / v.sum() * 100) if v.sum() > 0 else 0.0

    def policy_entropy(vec: np.ndarray) -> float:
        """Shannon entropy of a sense distribution in bits (uniform-over-64 = 6)."""
        p = np.asarray(vec, float)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum()) if p.size else 0.0

    def draw_board(ax, vec, color, title, vmax=None, as_pct=True):
        v = own_side_down(np.asarray(vec, float), color)
        if v.sum() > 0:
            v = v / v.sum()
        if as_pct:
            v = v * 100
        m = v.reshape(8, 8)
        im = ax.imshow(m, origin="lower", cmap=TERRA, vmin=0, vmax=vmax)
        ax.set_title(title, loc="left", fontsize=8, pad=8)
        ax.set_xticks(range(8), list("abcdefgh"), fontsize=6)
        ax.set_yticks(range(8), [str(i) for i in range(1, 9)], fontsize=6)
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        return im

    return draw_board, opp_half_mass, policy_entropy


@app.cell
def _(mo):
    mo.md(r"""
    ## Per-ordinal sense heatmaps

    Own-side-down perspective (own back rank at the bottom, as in the thesis).
    Each panel is one sense ordinal; White's sense 1 is flagged *uninformative*.
    """)
    return


@app.cell
def _(
    N,
    chess,
    color_select,
    counts,
    draw_board,
    mass,
    meta,
    mode_select,
    nobs,
    opp_half_mass,
    plt,
):
    _use_counts = mode_select.value == "sampled counts"
    _data = counts if _use_counts else mass
    _label = "sampled-choice share (%)" if _use_counts else "mean sense-policy mass (%)"

    if color_select.value == "Both":
        _colors = [chess.WHITE, chess.BLACK]
    elif color_select.value == "White":
        _colors = [chess.WHITE]
    else:
        _colors = [chess.BLACK]

    # shared vmax over displayed panels for cross-panel comparability
    def _pct(vec):
        v = vec.astype(float)
        return (v / v.sum() * 100) if v.sum() > 0 else v

    _vmax = max(
        (float(_pct(_data[(c, o)]).max()) for c in _colors for o in range(1, N + 1) if (c, o) in _data),
        default=1.0,
    )

    _fig, _axes = plt.subplots(
        len(_colors), N, figsize=(1.9 * N + 0.7, 2.15 * len(_colors)), squeeze=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.3},
    )
    for _ri, _c in enumerate(_colors):
        for _oi, _o in enumerate(range(1, N + 1)):
            _ax = _axes[_ri][_oi]
            _key = (_c, _o)
            if _key not in _data:
                _ax.axis("off")
                continue
            _name = "White" if _c else "Black"
            _flag = "  (uninformative)" if (_c == chess.WHITE and _o == 1) else ""
            _oh = opp_half_mass(_data[_key], _c)
            _im = draw_board(
                _ax, _data[_key], _c,
                f"{_name} · sense {_o}{_flag}\n(n={nobs.get(_key, 0)}, opp½ {_oh:.0f}%)",
                vmax=_vmax,
            )
    _cbar = _fig.colorbar(_im, ax=_axes, fraction=0.02, pad=0.02)
    _cbar.set_label(_label, fontsize=8)
    _cbar.ax.tick_params(labelsize=7)
    _cbar.outline.set_visible(False)
    _fig.suptitle(
        f"{meta['checkpoint']} — sense policy by ordinal (vs {meta['opponent']})",
        fontsize=9, x=0.02, ha="left",
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Companion views

    **Left:** policy entropy per ordinal — how *focused* the sense is. A high
    first-White-sense entropy (near the 6-bit uniform ceiling) is the signature
    of an uninformative sense; entropy should drop as the agent learns to target.
    **Right:** the aggregate over all N senses per color — directly comparable to
    the thesis `ladder-sense-heatmap` figure.
    """)
    return


@app.cell
def _(N, chess, color_select, draw_board, entropy, mass, meta, np, plt):
    if color_select.value == "Both":
        _colors = [chess.WHITE, chess.BLACK]
    elif color_select.value == "White":
        _colors = [chess.WHITE]
    else:
        _colors = [chess.BLACK]

    _fig = plt.figure(figsize=(3.0 + 2.0 * len(_colors), 2.6))
    _gs = _fig.add_gridspec(1, 1 + len(_colors), width_ratios=[1.4] + [1] * len(_colors))

    # entropy-per-ordinal line plot
    _axe = _fig.add_subplot(_gs[0, 0])
    _ords = list(range(1, N + 1))
    for _c in _colors:
        _y = [entropy.get((_c, _o), np.nan) for _o in _ords]
        _axe.plot(_ords, _y, marker="o", ms=4, lw=1.2, label="White" if _c else "Black")
    _axe.axhline(np.log2(64), ls="--", lw=0.8, color="grey", label="uniform (6 bits)")
    _axe.set_xlabel("sense ordinal")
    _axe.set_ylabel("policy entropy (bits)")
    _axe.set_xticks(_ords)
    _axe.set_ylim(0, np.log2(64) * 1.05)
    _axe.legend(fontsize=6)
    _axe.grid(True, alpha=0.3)
    _axe.set_title("sense focus by ordinal", loc="left", fontsize=8)

    # aggregate-over-N heatmaps (thesis-style)
    _agg = {}
    for _c in _colors:
        _vs = [mass[(_c, _o)] for _o in _ords if (_c, _o) in mass]
        if _vs:
            _agg[_c] = np.mean(np.stack(_vs), axis=0)
    _vmax = max((float((a / a.sum() * 100).max()) for a in _agg.values()), default=1.0)
    for _i, _c in enumerate(_colors):
        _ax = _fig.add_subplot(_gs[0, 1 + _i])
        if _c in _agg:
            _im = draw_board(
                _ax, _agg[_c], _c,
                f"{'White' if _c else 'Black'} · senses 1–{N}", vmax=_vmax,
            )
    _fig.suptitle(f"{meta['checkpoint']} — companion views", fontsize=9, x=0.02, ha="left")
    _fig.tight_layout(rect=(0, 0, 1, 0.94))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Top squares per ordinal

    The squares carrying the most sense-policy mass at each ordinal, in
    algebraic notation (true board coordinates, not rotated).
    """)
    return


@app.cell
def _(N, chess, color_select, mass, mo, np, opp_half_mass, policy_entropy):
    import chess as _chess

    if color_select.value == "Both":
        _colors = [chess.WHITE, chess.BLACK]
    elif color_select.value == "White":
        _colors = [chess.WHITE]
    else:
        _colors = [chess.BLACK]

    _rows = []
    for _c in _colors:
        for _o in range(1, N + 1):
            _key = (_c, _o)
            if _key not in mass:
                continue
            _p = mass[_key]
            _top = np.argsort(_p)[::-1][:5]
            _rows.append(
                {
                    "color": "White" if _c else "Black",
                    "sense": _o,
                    "top squares (share %)": ", ".join(
                        f"{_chess.square_name(int(s))} {_p[s] * 100:.1f}" for s in _top
                    ),
                    "opp-half mass (%)": round(opp_half_mass(_p, _c), 1),
                    "entropy (bits)": round(policy_entropy(_p), 2),
                }
            )
    mo.ui.table(_rows, selection=None) if _rows else mo.md("*no data*")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Checkpoint comparison — sensing quality vs training

    Sensing *at the opponent* is a learned skill: an under-trained sense head looks
    at its own board (low **opponent-half mass**, sometimes with deceptively low
    entropy because it is confidently wrong), while a trained one puts most of its
    mass on the opponent's ranks — and, being near-Nash, does so as a deliberately
    *mixed* (higher-entropy) distribution rather than a predictable spike.

    Pick two or more checkpoints (a weak and a strong one make the trend obvious).
    Opponent, N, sampling and seed are inherited from the controls at the top; each
    checkpoint is rolled out fresh below. **Both colors are always measured.**
    """)
    return


@app.cell
def _(checkpoints, mo):
    _labels = list(checkpoints)
    _defaults = [c for c in ("v0.12.0_30000", "v0.27.0_80000") if c in checkpoints] or _labels[:2]
    compare_select = mo.ui.multiselect(
        options=_labels, value=_defaults, label="checkpoints to compare", full_width=True
    )
    compare_games = mo.ui.slider(
        4, 120, value=20, step=4, label="rollout games / checkpoint", show_value=True
    )
    compare_button = mo.ui.run_button(label="▶ Run comparison")
    mo.vstack([compare_select, mo.hstack([compare_games, compare_button], justify="start", gap=1.5)])
    return compare_button, compare_games, compare_select


@app.cell
def _(
    checkpoints,
    chess,
    compare_button,
    compare_games,
    compare_select,
    device_select,
    load_net,
    mo,
    n_senses,
    np,
    opp_half_mass,
    opponent_select,
    policy_entropy,
    run_rollouts,
    sample_toggle,
    seed_input,
    torch,
):
    mo.stop(
        not compare_button.value or not compare_select.value,
        mo.md("⏳ Select checkpoints and press **▶ Run comparison**."),
    )

    _device = torch.device("cuda" if device_select.value == "cuda" else "cpu")
    _want = [chess.WHITE, chess.BLACK]
    Ncmp = int(n_senses.value)

    cmp_oh: dict[str, dict] = {}   # label -> {(color, ordinal): opp-half mass %}
    cmp_ent: dict[str, dict] = {}  # label -> {(color, ordinal): mean entropy bits}
    with mo.status.progress_bar(total=len(compare_select.value)) as _bar:
        for _label in compare_select.value:
            # Re-seed per checkpoint so each sees the same opponent openings.
            torch.manual_seed(int(seed_input.value))
            np.random.seed(int(seed_input.value))
            _net, _enc = load_net(checkpoints[_label], _device)
            _vecs, _ = run_rollouts(
                _net, _device, _enc.history, _want, opponent_select.value,
                Ncmp, int(compare_games.value), sample_toggle.value,
            )
            cmp_oh[_label] = {
                k: float(np.mean([opp_half_mass(p, k[0]) for p in v])) for k, v in _vecs.items()
            }
            cmp_ent[_label] = {
                k: float(np.mean([policy_entropy(p) for p in v])) for k, v in _vecs.items()
            }
            _bar.update()

    cmp_meta = {"opponent": opponent_select.value, "N": Ncmp, "games": int(compare_games.value)}
    return Ncmp, cmp_ent, cmp_meta, cmp_oh


@app.cell
def _(Ncmp, chess, cmp_ent, cmp_meta, cmp_oh, np, plt):
    _ords = list(range(1, Ncmp + 1))
    _colors = [chess.WHITE, chess.BLACK]
    _fig, _axes = plt.subplots(2, 2, figsize=(8.2, 5.4), sharex=True)
    # row 0: opponent-half mass; row 1: mean policy entropy. columns: White, Black.
    for _ci, _c in enumerate(_colors):
        _name = "White" if _c else "Black"
        for _label in cmp_oh:
            _y = [cmp_oh[_label].get((_c, _o), np.nan) for _o in _ords]
            _axes[0][_ci].plot(_ords, _y, marker="o", ms=4, lw=1.4, label=_label)
        _axes[0][_ci].axhline(50, ls="--", lw=0.8, color="grey")
        _axes[0][_ci].set_title(_name, fontsize=9)
        _axes[0][_ci].set_ylim(0, 100)
        _axes[0][_ci].grid(True, alpha=0.3)

        for _label in cmp_ent:
            _y = [cmp_ent[_label].get((_c, _o), np.nan) for _o in _ords]
            _axes[1][_ci].plot(_ords, _y, marker="o", ms=4, lw=1.4, label=_label)
        _axes[1][_ci].axhline(np.log2(64), ls="--", lw=0.8, color="grey")
        _axes[1][_ci].set_ylim(0, 6.3)
        _axes[1][_ci].set_xlabel("sense ordinal")
        _axes[1][_ci].set_xticks(_ords)
        _axes[1][_ci].grid(True, alpha=0.3)

    _axes[0][0].set_ylabel("opponent-half mass (%)\n(50 % = uniform ↑ better)")
    _axes[1][0].set_ylabel("mean policy entropy (bits)\n(dashed = 6-bit uniform)")
    _axes[0][0].legend(fontsize=6, loc="best")
    _fig.suptitle(
        f"Sensing quality vs training — opponent={cmp_meta['opponent']}, "
        f"{cmp_meta['games']} games/ckpt",
        fontsize=10,
    )
    _fig.tight_layout(rect=(0, 0, 1, 0.95))
    _fig
    return


if __name__ == "__main__":
    app.run()
