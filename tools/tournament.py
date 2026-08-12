"""Round-robin tournament between checkpoints and baseline bots, with a
Bradley-Terry/Elo leaderboard.

With no player arguments, every checkpoint under checkpoints/v*/ plus the
random, trout and mht baselines is entered automatically. Players can also be
given explicitly as checkpoint paths/globs and/or baseline names (random,
attacker, trout, mht). Every unordered pair plays --pair-games games with
alternating colors; results are appended to a JSONL file, and re-running the
same command resumes (only the games still missing for each pair are
scheduled). New checkpoints are picked up on the next run and only their
missing pairings are played -- finished pairs are never repeated.

Checkpoint agents play in deployment mode by default: sampled with
--threshold truncation (see RNaDPlayer.sample_threshold); use --greedy to
evaluate argmax play instead.

To enter (or finish entering) one training run on the ladder, use
--model VERSION: every checkpoint of that version plays every other player --
the whole existing field plus the baselines -- and its own checkpoints play each
other, while pairs that touch neither are skipped. The version's checkpoints are
added to the field automatically, so no globs are needed and the flag can be
repeated to do several runs at once.

To add a few fresh checkpoints to a big existing ladder without replaying the
whole O(N^2) field, use --vs-top N: players that have no games yet in --out
play against the current top-N established players (ranked by Elo from --out),
every baseline, and each other; established-vs-established pairs are skipped
entirely. Combining it with --model swaps that "no games yet" guess for the
named run, which is what you want when the run already has some games on the
ladder (e.g. it was extended with new checkpoints, or a previous --model pass
was interrupted).

Examples:

    uv run python tools/tournament.py               # everything, auto
    uv run python tools/tournament.py \
        'checkpoints/v0.12.0/*.pt' random trout mht \
        --pair-games 8 --out results/tournament_v12.jsonl
    uv run python tools/tournament.py --dry-run     # show the schedule size
    uv run python tools/tournament.py \
        'checkpoints/v0.13.0/*.pt' --vs-top 20      # new nets vs current top 20
    uv run python tools/tournament.py --model v0.58.0 --dry-run
    uv run python tools/tournament.py --model v0.58.0            # vs whole field
    uv run python tools/tournament.py --model v0.58.0 --vs-top 20  # cheap version

The leaderboard is printed at the end (and can be recomputed any time with
--leaderboard-only). Elo is anchored so random = 0 when present.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import random as pyrandom
import sys
import time
from collections import OrderedDict, defaultdict
from itertools import combinations
from pathlib import Path

BASELINES = ("random", "attacker", "trout", "mht")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINES = ["random", "trout", "mht"]

_WORKER: dict = {}


# ------------------------------------------------------------------ players
def resolve_model(spec: str) -> dict[str, str]:
    """Map display name -> path for every checkpoint of one training run.

    `spec` is a version as it appears under checkpoints/ ("v0.58.0"), or a path
    to such a directory ("checkpoints/v0.58.0"); both forms are accepted so the
    flag works with shell completion.
    """
    d = Path(spec)
    if not d.is_dir():
        d = ROOT / "checkpoints" / Path(spec.rstrip("/")).name
    if not d.is_dir():
        sys.exit(f"--model {spec}: no such checkpoint directory ({d})")
    paths = sorted(d.glob("*.pt"))
    if not paths:
        sys.exit(f"--model {spec}: no .pt checkpoints in {d}")
    return {p.stem.replace("deepnash_async_", ""): str(p.resolve()) for p in paths}


def resolve_players(specs: list[str]) -> dict[str, str]:
    """Map display name -> spec (baseline name or checkpoint path)."""
    players: dict[str, str] = {}
    for spec in specs:
        if spec in BASELINES:
            players[spec] = spec
            continue
        paths = sorted(glob.glob(spec)) or [spec]
        for p in paths:
            path = Path(p)
            if not path.exists():
                sys.exit(f"player spec matched nothing: {spec}")
            name = path.stem.replace("deepnash_async_", "")
            players[name] = str(path.resolve())
    return players


# ------------------------------------------------------------------ workers
def _worker_init(players: dict, device: str, threshold: float, greedy: bool,
                 cache_size: int):
    import torch
    from deepnash_rbc.eval import _ensure_stockfish

    if any(spec in ("trout", "mht") for spec in players.values()):
        _ensure_stockfish()
    _WORKER.update(
        players=players,
        device=torch.device(device) if device else
               torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        threshold=threshold,
        greedy=greedy,
        nets=OrderedDict(),  # LRU: path -> (net, enc)
        cache_size=cache_size,
    )


def _get_net(path: str):
    from deepnash_rbc.play_session import load_net

    nets: OrderedDict = _WORKER["nets"]
    if path in nets:
        nets.move_to_end(path)
        return nets[path]
    if len(nets) >= _WORKER["cache_size"]:
        nets.popitem(last=False)
    nets[path] = load_net(path, _WORKER["device"])
    return nets[path]


def _make_player(name: str):
    from deepnash_rbc.agent import RNaDPlayer
    from deepnash_rbc.eval import _make_opponent

    spec = _WORKER["players"][name]
    if spec in BASELINES:
        return _make_opponent(spec)
    net, enc = _get_net(spec)
    return RNaDPlayer(net, _WORKER["device"], history=enc.history,
                      sample=not _WORKER["greedy"],
                      sample_threshold=_WORKER["threshold"])


def _play_game(task):
    """(white_name, black_name, seconds) -> result row dict."""
    import chess
    from reconchess import LocalGame, play_local_game

    white_name, black_name, seconds = task
    row = {"white": white_name, "black": black_name, "ts": round(time.time(), 1)}
    try:
        white = _make_player(white_name)
        black = _make_player(black_name)
        winner, reason, hist = play_local_game(
            white, black, game=LocalGame(seconds_per_player=seconds))
    except Exception as e:
        row.update(winner="error", reason=f"{type(e).__name__}: {e}")
        return row
    if winner is None:
        row["winner"] = "draw"
    else:
        row["winner"] = "white" if winner == chess.WHITE else "black"
    row["reason"] = str(reason)
    row["turns"] = hist.num_turns()
    return row


# -------------------------------------------------------------- leaderboard
# The BT fit lives in the package (deepnash_rbc.analysis.elo) so the policy-mode
# arena, which places new agents on *this* ladder's scale, fits with exactly the
# same estimator instead of a drifting copy.
from deepnash_rbc.analysis.elo import bradley_terry  # noqa: E402


def print_leaderboard(rows: list[dict], players: dict[str, str]):
    played = defaultdict(float)
    score = defaultdict(float)
    errors = 0
    for r in rows:
        if r["winner"] == "error":
            errors += 1
            continue
        for side in ("white", "black"):
            played[r[side]] += 1
        if r["winner"] == "draw":
            score[r["white"]] += 0.5
            score[r["black"]] += 0.5
        else:
            score[r[r["winner"]]] += 1.0

    names = [n for n in players if played[n]]
    if not names:
        print("no completed games yet")
        return
    elo = bradley_terry(rows, names)
    print(f"\n=== Leaderboard ({int(sum(played.values()) / 2)} games"
          + (f", {errors} errored" if errors else "") + ") ===")
    print(f"{'player':<24} {'elo':>7} {'games':>6} {'score':>7} {'+/-':>5}")
    for name in sorted(names, key=lambda x: -elo[x]):
        n = played[name]
        # 1-sigma Elo uncertainty from a balanced binomial approximation
        se = 347.0 / math.sqrt(n) if n else float("inf")
        print(f"{name:<24} {elo[name]:>7.0f} {int(n):>6} "
              f"{score[name] / n:>7.2f} {se:>5.0f}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("players", nargs="*",
                    help="checkpoint paths/globs and/or baseline names "
                         f"{BASELINES}; default: all checkpoints/v*/*.pt "
                         f"+ {' '.join(DEFAULT_BASELINES)}")
    ap.add_argument("--pair-games", type=int, default=8,
                    help="games per unordered pair (colors alternate)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=900,
                    help="clock per player per game (server default 900)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="sample_threshold for checkpoint agents")
    ap.add_argument("--greedy", action="store_true",
                    help="checkpoint agents play argmax instead of sampling")
    ap.add_argument("--net-cache", type=int, default=8,
                    help="nets kept in memory per worker (LRU)")
    ap.add_argument("--out", default="results/tournament.jsonl")
    ap.add_argument("--model", action="append", default=None, metavar="VERSION",
                    help="play every checkpoint of this run (e.g. v0.58.0) "
                         "against the whole field and against each other; "
                         "pairs touching no --model checkpoint are skipped. "
                         "Repeatable")
    ap.add_argument("--vs-top", type=int, default=None, metavar="N",
                    help="only schedule players with no games yet in --out "
                         "against the top-N established players (by Elo from "
                         "--out); skip established-vs-established pairs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print how many games would be played, then exit")
    ap.add_argument("--leaderboard-only", action="store_true",
                    help="recompute the leaderboard from --out and exit")
    args = ap.parse_args()

    specs = args.players or (
        sorted(str(p) for p in (ROOT / "checkpoints").glob("v*/*.pt"))
        + DEFAULT_BASELINES
    )
    players = resolve_players(specs)

    # --model: the focus set. Its checkpoints join the field even if the
    # positional specs (or the auto default, on a run whose files landed after
    # the glob) would not have picked them up.
    focus: set[str] = set()
    for spec in args.model or []:
        ckpts = resolve_model(spec)
        players.update(ckpts)
        focus |= set(ckpts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():
        try:
            rows = [json.loads(line) for line in out.read_text().splitlines() if line]
        except json.JSONDecodeError:
            sys.exit(f"{out} is not a JSONL results file -- --out expects the "
                     f"per-game results (e.g. results/tournament.jsonl), not a "
                     f"rendered leaderboard")

    if args.leaderboard_only:
        print_leaderboard(rows, players)
        return

    # resume: schedule only what is missing per pair
    done = defaultdict(int)
    opponents = defaultdict(set)
    for r in rows:
        if r["winner"] != "error":
            done[frozenset((r["white"], r["black"]))] += 1
            opponents[r["white"]].add(r["black"])
            opponents[r["black"]].add(r["white"])

    # which unordered pairs are eligible for scheduling
    if args.vs_top is not None:
        baselines = {n for n in players if n in BASELINES}
        # Classify "new" players so a stopped run resumes to the same schedule.
        # A new checkpoint only ever plays anchors (top-N + baselines) or other
        # new checkpoints, so "every opponent seen so far is an anchor or new"
        # stays true no matter how many of its games are already recorded --
        # unlike "has zero games", which flips after the first game. Solve for
        # it as a fixpoint; the top-N is fit from established-vs-established
        # games only (bradley_terry ignores games touching a player outside its
        # name list), so the anchor set does not drift as new games land.
        #
        # --model states the focus set outright, so no inference is needed (and
        # a run that already has games -- extended, or a resumed --model pass --
        # is still scheduled in full).
        top: list[str] = []
        if focus:
            new = set(focus)
            established = [n for n in players if n not in new]
            if not established:
                sys.exit("--vs-top needs existing results in --out to rank opponents")
            elo = bradley_terry(rows, established)
            top = sorted(established, key=lambda n: -elo[n])[:args.vs_top]
            anchors = (set(top) | baselines) - new
        else:
            new = {n for n in players if not opponents[n]}
            while True:
                established = [n for n in players if n not in new]
                if not established:
                    sys.exit("--vs-top needs existing results in --out to rank "
                             "opponents")
                elo = bradley_terry(rows, established)
                top = sorted(established, key=lambda n: -elo[n])[:args.vs_top]
                anchors = (set(top) | baselines) - new
                grown = {n for n in players
                         if opponents[n] and opponents[n] <= (anchors | new)}
                if grown <= new:
                    break
                new |= grown
            # new checkpoints play the top-N, every baseline, and each other
            if not new:
                print("--vs-top: no new players (all already have games in --out)")
        print(f"--vs-top {args.vs_top}: {len(new)} "
              f"{'--model' if focus else 'new'} player(s) vs "
              f"{len(anchors)} anchors (top-{len(top)} + baselines) "
              f"and each other")
        pairs = [(a, b) for a in new for b in anchors]
        pairs += list(combinations(new, 2))
    elif focus:
        # Every pair that touches the focus set: its checkpoints vs the whole
        # field, plus the focus set's own internal ladder.
        _focus = sorted(focus)  # stable schedule across invocations
        rest = [n for n in players if n not in focus]
        print(f"--model {' '.join(args.model)}: {len(_focus)} checkpoint(s) vs "
              f"{len(rest)} other player(s) and each other")
        pairs = [(a, b) for a in _focus for b in rest]
        pairs += list(combinations(_focus, 2))
    else:
        pairs = list(combinations(players, 2))

    tasks = []
    for a, b in pairs:
        have = done[frozenset((a, b))]
        for g in range(have, args.pair_games):
            white, black = (a, b) if g % 2 == 0 else (b, a)
            tasks.append((white, black, args.seconds))
    pyrandom.Random(0).shuffle(tasks)  # spread engine games across the run
    engine = sum(1 for t in tasks
                 if players[t[0]] in ("trout", "mht")
                 or players[t[1]] in ("trout", "mht"))
    print(f"{len(players)} players, {len(tasks)} games to play "
          f"({engine} vs engine bots, ~30-60s each; rest are fast) "
          f"({len(rows)} already in {out})")

    if args.dry_run:
        return

    if tasks:
        ctx = mp.get_context("spawn")
        t0 = time.time()
        with ctx.Pool(args.workers, initializer=_worker_init,
                      initargs=(players, args.device, args.threshold,
                                args.greedy, args.net_cache)) as pool, \
                out.open("a") as fh:
            for i, row in enumerate(pool.imap_unordered(_play_game, tasks), 1):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                if row["winner"] == "error":
                    print(f"[{i}/{len(tasks)}] ERROR {row['white']} vs "
                          f"{row['black']}: {row['reason']}", flush=True)
                elif i % 10 == 0 or i == len(tasks):
                    rate = (time.time() - t0) / i
                    eta = rate * (len(tasks) - i) / 60
                    print(f"[{i}/{len(tasks)}] {rate:.1f}s/game, "
                          f"ETA {eta:.0f} min", flush=True)

    print_leaderboard(rows, players)


if __name__ == "__main__":
    main()
