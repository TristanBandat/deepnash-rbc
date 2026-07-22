"""Focused tournament between the top-N checkpoints of a previous tournament.

Recomputes the Bradley-Terry/Elo leaderboard from an existing tournament JSONL
(default results/tournament.jsonl), selects the --top strongest players
(baselines like mht/trout compete for slots too, unless --exclude-baselines),
and plays a schedule in which every player gets exactly --games-per-player
games. The
schedule is built from perfect-matching rounds (circle method), so games are
spread evenly across all pairs with alternating colors.

Results are appended to their own JSONL file (default
results/tournament_top20.jsonl); re-running resumes and only plays the games
still missing per pair, exactly like tools/tournament.py.

Examples:

    uv run python tools/tournament_top20.py --dry-run   # show selection only
    uv run python tools/tournament_top20.py             # play it
    uv run python tools/tournament_top20.py --leaderboard-only
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tournament as T  # noqa: E402

ROOT = T.ROOT


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rank_players(rows: list[dict]) -> list[tuple[str, float, int]]:
    """All players from rows, as (name, elo, games) sorted best-first."""
    games = defaultdict(int)
    for r in rows:
        if r["winner"] == "error":
            continue
        games[r["white"]] += 1
        games[r["black"]] += 1
    names = sorted(games)
    elo = T.bradley_terry(rows, names)
    return sorted(((n, elo[n], games[n]) for n in names), key=lambda x: -x[1])


def checkpoint_path(name: str) -> Path:
    """Invert tournament.py's naming: v0.8.0_70000 -> checkpoints/v0.8.0/..."""
    version = name.rsplit("_", 1)[0]
    return ROOT / "checkpoints" / version / f"deepnash_async_{name}.pt"


def select_top(rows: list[dict], top: int, include_baselines: bool) -> dict[str, str]:
    """Top players as a name -> spec dict (resolved paths / baseline names)."""
    players: dict[str, str] = {}
    print(f"{'rank':<5} {'player':<24} {'elo':>7} {'games':>7}")
    for name, elo, games in rank_players(rows):
        if name in T.BASELINES:
            if not include_baselines:
                continue
            players[name] = name
        else:
            path = checkpoint_path(name)
            if not path.exists():
                sys.exit(f"checkpoint for ranked player {name} not found: {path}")
            players[name] = str(path)
        print(f"{len(players):<5} {name:<24} {elo:>7.0f} {games:>7}")
        if len(players) == top:
            return players
    sys.exit(f"only {len(players)} eligible players in source, need {top}")


def build_schedule(names: list[str], games_per_player: int) -> list[tuple[str, str]]:
    """Perfect-matching rounds (circle method) until every player has exactly
    games_per_player games; colors alternate within each pair."""
    order = list(names)
    if len(order) % 2:
        order.append(None)  # bye
    n = len(order)
    need = {name: games_per_player for name in names}
    pair_count: defaultdict = defaultdict(int)
    schedule = []
    while any(need.values()):
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            if a is None or b is None or not (need[a] and need[b]):
                continue
            pair = frozenset((a, b))
            white, black = (a, b) if pair_count[pair] % 2 == 0 else (b, a)
            pair_count[pair] += 1
            schedule.append((white, black))
            need[a] -= 1
            need[b] -= 1
        order = [order[0]] + [order[-1]] + order[1:-1]  # rotate, fix first
    return schedule


def write_leaderboard(rows: list[dict], players: dict[str, str], path: str):
    """Print the leaderboard and mirror it to a txt file."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.print_leaderboard(rows, players)
    print(buf.getvalue(), end="")
    if path:
        Path(path).write_text(buf.getvalue())
        print(f"leaderboard written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="results/tournament.jsonl",
                    help="JSONL with past results used to rank the players")
    ap.add_argument("--top", type=int, default=20,
                    help="number of players to advance")
    ap.add_argument("--games-per-player", type=int, default=2000)
    ap.add_argument("--exclude-baselines", action="store_true",
                    help=f"keep baselines {T.BASELINES} out of the top-N slots")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=900,
                    help="clock per player per game (server default 900)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="sample_threshold for checkpoint agents")
    ap.add_argument("--greedy", action="store_true",
                    help="checkpoint agents play argmax instead of sampling")
    ap.add_argument("--net-cache", type=int, default=20,
                    help="nets kept in memory per worker (LRU)")
    ap.add_argument("--out", default="results/tournament_top20.jsonl")
    ap.add_argument("--leaderboard-out",
                    default="results/tournament_top20_leaderboard.txt",
                    help="txt file the leaderboard is mirrored to ('' disables)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the selection and schedule size, play nothing")
    ap.add_argument("--leaderboard-only", action="store_true",
                    help="recompute the leaderboard from --out and exit")
    args = ap.parse_args()

    source_rows = load_rows(Path(args.source))
    if not source_rows:
        sys.exit(f"no results in {args.source}")
    players = select_top(source_rows, args.top, not args.exclude_baselines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(out)

    if args.leaderboard_only:
        write_leaderboard(rows, players, args.leaderboard_out)
        return

    # resume: drop the games each pair has already completed
    done = defaultdict(int)
    for r in rows:
        if r["winner"] != "error":
            done[frozenset((r["white"], r["black"]))] += 1
    remaining = dict(done)
    tasks = []
    for white, black in build_schedule(list(players), args.games_per_player):
        pair = frozenset((white, black))
        if remaining.get(pair, 0) > 0:
            remaining[pair] -= 1
        else:
            tasks.append((white, black, args.seconds))

    total = args.top * args.games_per_player // 2
    print(f"\n{len(players)} players, {args.games_per_player} games each "
          f"({total} total): {len(tasks)} left to play "
          f"({len(rows)} rows already in {out})")
    if args.dry_run:
        return

    if tasks:
        ctx = mp.get_context("spawn")
        t0 = time.time()
        with ctx.Pool(args.workers, initializer=T._worker_init,
                      initargs=(players, args.device, args.threshold,
                                args.greedy, args.net_cache)) as pool, \
                out.open("a") as fh:
            for i, row in enumerate(pool.imap_unordered(T._play_game, tasks), 1):
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

    write_leaderboard(rows, players, args.leaderboard_out)


if __name__ == "__main__":
    main()
