"""Stockfish-backed performance report for a trained checkpoint.

Two analyses, both driven by a Stockfish binary (set STOCKFISH_EXECUTABLE or put
`stockfish` on PATH):

  1. MOVE QUALITY -- play the agent for N games against each chosen opponent, then
     replay every game against the ground-truth board and grade its moves with
     Stockfish: ACPL, blunder rate, top-1 match, plus two information-cost figures
     (see analysis/move_quality.py).  This separates *chess skill* from *information
     skill*.  --mq-opponent takes one or more opponents; each gets its own section
     reporting win/draw/loss (from those same games) and the move-quality table, so
     the opponent drives every number, not just the grading.

  2. STRENGTH LADDER -- play the agent vs TroutBot at a sweep of UCI Skill
     Levels and report win-rate per level, a cheap strength curve.

Usage:
  uv run deepnash-stockfish-eval --checkpoint checkpoints/latest.pt
  uv run deepnash-stockfish-eval -c ckpt.pt --num-games 20 --mq-opponent mht strangefish2 \
      --ladder-levels 15,20 --json report.json

Games are independent and played across a process pool (-j / --workers, default
auto). Each worker loads its own net on --device and its own Stockfish (spawn start
method), so grading happens in-worker and only small results cross the boundary.

Defaults: device cuda; move-quality opponents trout/mht (strangefish2 opt-in); ladder at
Skill Level 20 only; --num-games (default 10) drives both the per-opponent and
per-ladder-level game counts unless --mq-games / --ladder-games override it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import chess
import torch
from reconchess import LocalGame, play_local_game

from .analysis.engine import StockfishAnalyst, resolve_engine_path
from .analysis.ladder import make_ladder_opponent
from .analysis.move_quality import Aggregate, aggregate, analyze_game, merge
from .agent import RNaDPlayer
from .eval import _make_opponent
from .play_session import load_net


def _default_workers() -> int:
    """Auto worker count: ~half the cores, capped so the default doesn't spawn a
    huge number of CUDA contexts / Stockfish processes. Override with -j."""
    return max(1, min((os.cpu_count() or 2) // 2, 8))


# --------------------------------------------------------------------------
# Parallel game workers.
#
# Games are independent, so we play them in a process Pool. We benchmarked two
# strategies -- fork+net-on-CPU vs spawn+net-on-CUDA -- and spawn+CUDA won (~1.5x)
# because the eval net does an unbatched single-sample forward every move, which is
# slow on CPU. fork is also unsafe here: torch's OpenMP/intraop threads hold locks
# a fork can't inherit, deadlocking every worker on its first forward. So we always
# use the "spawn" start method and load the net -- and, for move quality, a
# per-worker StockfishAnalyst -- once per worker in the initializer, since a CUDA
# net can't be shipped across the process boundary anyway.
# --------------------------------------------------------------------------
_WORKER: Dict[str, object] = {}


def _worker_init(checkpoint: str, device: str, analyst_kwargs: Optional[dict]) -> None:
    """Runs once per worker: resolve Stockfish, load the net on `device`, and (for the
    move-quality path) open a StockfishAnalyst that lives for the worker's lifetime."""
    import atexit

    from .eval import _ensure_stockfish
    _ensure_stockfish()
    dev = torch.device(device)
    net, enc = load_net(checkpoint, dev)
    _WORKER.update(net=net, device=dev, history=enc.history, analyst=None)
    if analyst_kwargs is not None:
        analyst = StockfishAnalyst(resolve_engine_path(), **analyst_kwargs)
        analyst.__enter__()
        _WORKER["analyst"] = analyst
        atexit.register(lambda: analyst.__exit__(None, None, None))


def _pool(args, analyst_kwargs: Optional[dict]):
    """A spawn Pool whose workers have the net (and optional analyst) preloaded."""
    ctx = mp.get_context("spawn")
    return ctx.Pool(args.workers, initializer=_worker_init,
                    initargs=(args.checkpoint, args.device, analyst_kwargs))


def _ladder_game(task):
    """Worker: play one ladder game vs trout@level -> (level, status, payload)."""
    level, g, seconds = task
    try:
        opponent = make_ladder_opponent(level)
    except Exception as e:
        return (level, "build_error", f"{type(e).__name__}: {e}")
    agent = RNaDPlayer(_WORKER["net"], _WORKER["device"],
                       history=_WORKER["history"], sample=False)
    agent_is_white = (g % 2 == 0)
    white, black = (agent, opponent) if agent_is_white else (opponent, agent)
    game = LocalGame(seconds_per_player=seconds)
    try:
        winner, _reason, _hist = play_local_game(white, black, game=game)
    except Exception as e:
        return (level, "game_error", f"{type(e).__name__}: {e}")
    color = chess.WHITE if agent_is_white else chess.BLACK
    outcome = "draw" if winner is None else ("win" if winner == color else "loss")
    return (level, "ok", outcome)


def _mq_game(task):
    """Worker: play one move-quality game vs `opp_name`, grade it with the worker's own
    analyst -> (opp_name, status, payload). On success payload is (color, outcome, Aggregate)."""
    opp_name, g, seconds, belief = task
    try:
        opponent = _make_opponent(opp_name)
    except Exception as e:
        return (opp_name, "build_error", f"{type(e).__name__}: {e}")
    agent = RNaDPlayer(_WORKER["net"], _WORKER["device"],
                       history=_WORKER["history"], sample=False, track_belief=True)
    agent_is_white = (g % 2 == 0)
    white, black = (agent, opponent) if agent_is_white else (opponent, agent)
    game = LocalGame(seconds_per_player=seconds)
    try:
        winner, _reason, hist = play_local_game(white, black, game=game)
    except Exception as e:
        return (opp_name, "game_error", f"{type(e).__name__}: {e}")
    color = chess.WHITE if agent_is_white else chess.BLACK
    outcome = "draw" if winner is None else ("win" if winner == color else "loss")
    records = analyze_game(hist, color, _WORKER["analyst"],
                           belief_fens=list(agent.belief_fens) if belief else None)
    return (opp_name, "ok", (color, outcome, aggregate(records)))


def run_move_quality(args) -> Dict:
    """Play args.mq_games vs each opponent in args.mq_opponent (across a worker pool)
    and grade them. Returns {opp: report} with win/draw/loss (the outcome of those same
    games) plus per-colour and overall move-quality aggregates."""
    analyst_kwargs = dict(depth=args.depth, movetime=args.movetime, blunder_cp=args.blunder_cp)
    tasks = [(opp, g, args.seconds_per_player, args.belief)
             for opp in args.mq_opponent for g in range(args.mq_games)]
    per: Dict[str, dict] = {opp: {chess.WHITE: [], chess.BLACK: [],
                                  "wins": 0, "draws": 0, "losses": 0, "n": 0}
                            for opp in args.mq_opponent}
    with _pool(args, analyst_kwargs) as pool:
        for opp, status, payload in pool.imap_unordered(_mq_game, tasks):
            if status != "ok":
                print(f"[mq:{opp}] game skipped ({status}): {payload}")
                continue
            color, outcome, agg = payload
            d = per[opp]
            d["n"] += 1
            d[{"win": "wins", "draw": "draws", "loss": "losses"}[outcome]] += 1
            d[color].append(agg)
    out: Dict[str, Dict] = {}
    for opp in args.mq_opponent:
        d = per[opp]
        played = d["n"]
        rep: Dict[str, object] = {
            "n": played, "wins": d["wins"], "draws": d["draws"], "losses": d["losses"],
            "win_rate": round(d["wins"] / played, 4) if played else None,
            "draw_rate": round(d["draws"] / played, 4) if played else None,
        }
        all_aggs: List[Aggregate] = []
        for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            if d[color]:
                rep[label] = asdict(merge(d[color]))
                all_aggs.extend(d[color])
        if all_aggs:
            rep["overall"] = asdict(merge(all_aggs))
        out[opp] = rep
    return out


def run_ladder(args) -> List[Dict]:
    tasks = [(level, g, args.seconds_per_player)
             for level in args.ladder_levels for g in range(args.ladder_games)]
    acc: Dict[int, dict] = {level: {"win": 0, "draw": 0, "loss": 0, "n": 0}
                            for level in args.ladder_levels}
    with _pool(args, None) as pool:
        for level, status, payload in pool.imap_unordered(_ladder_game, tasks):
            if status != "ok":
                print(f"[ladder] game@{level} skipped ({status}): {payload}")
                continue
            acc[level][payload] += 1
            acc[level]["n"] += 1
    rows: List[Dict] = []
    for level in args.ladder_levels:
        a = acc[level]
        n = a["n"]
        row = {
            "skill_level": level,
            "n": n,
            "win_rate": round(a["win"] / n, 4) if n else None,
            "draw_rate": round(a["draw"] / n, 4) if n else None,
        }
        rows.append(row)
        print(f"[ladder] skill={level:>2}  win={row['win_rate']}  "
              f"draw={row['draw_rate']}  (n={n})")
    return rows


def _print_move_quality(mq: Dict) -> None:
    """Print one labelled section per opponent: win/draw/loss over the games played,
    the per-colour ACPL table, and the information-cost summary."""
    for opp_name, res in mq.items():
        print(f"\n=== vs {opp_name}  (n={res['n']}  "
              f"W/D/L={res['wins']}/{res['draws']}/{res['losses']}  "
              f"win={res['win_rate']}  draw={res['draw_rate']}) ===")
        print(f"{'':8} {'ACPL':>7} {'blunder%':>9} {'top1%':>7} {'scored':>7} {'no-eval':>8}")
        for label in ("white", "black", "overall"):
            a = res.get(label)
            if not a:
                continue
            print(f"{label:8} {a['acpl']:>7} {a['blunder_rate'] * 100:>8.1f}% "
                  f"{a['top1_rate'] * 100:>6.1f}% {a['n_scored']:>7} {a['n_no_eval']:>8}")
        ov = res.get("overall")
        if ov:
            print("  Information cost (centipawns, higher = imperfect info hurt more):")
            if ov.get("cost_A_mean") is not None:
                print(f"    A  taken move: true board vs naive-belief board   "
                      f"{ov['cost_A_mean']:+.1f} cp  (n={ov['cost_A_n']}, proxy belief)")
            if ov.get("cost_B_mean") is not None:
                print(f"    B  on true board: taken vs requested move         "
                      f"{ov['cost_B_mean']:+.1f} cp  (n={ov['cost_B_n']}, truncation cost)")
            print(f"    truncated moves (requested != taken): {ov['n_truncated']}/{ov['n_moves']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stockfish performance report for a checkpoint.")
    p.add_argument("-c", "--checkpoint", required=True, help="path to a .pt checkpoint")
    p.add_argument("--history", type=int, default=8, help="observation history frames (match training)")
    p.add_argument("--device", default="cuda", help="torch device (cuda/cpu)")
    p.add_argument("-j", "--workers", type=int, default=None,
                   help="parallel worker processes for game play (spawn start method; each "
                        "worker loads its own net on --device and its own Stockfish). "
                        "Default: auto (~half the cores, capped at 8). 1 = serial.")
    p.add_argument("--seconds-per-player", type=float, default=900.0)
    p.add_argument("-n", "--num-games", type=int, default=10,
                   help="games to play per move-quality opponent AND per ladder level. "
                        "Overridden per-section by --mq-games / --ladder-games.")
    # move-quality
    p.add_argument("--mq-games", type=int, default=None,
                   help="override --num-games for the move-quality report (0 to skip)")
    p.add_argument("--mq-opponent", nargs="+", default=["trout", "mht"], metavar="NAME",
                   choices=["random", "attacker", "trout", "mht", "strangefish2"],
                   help="one or more opponents for the move-quality report; each is played "
                        "for --num-games games and gets its own win/draw/loss + ACPL section "
                        "(default 'trout mht'; mht and strangefish2 are strong, slow, and "
                        "print their own progress output -- strangefish2 is opt-in as it is "
                        "especially slow)")
    p.add_argument("--depth", type=int, default=12, help="Stockfish analysis depth")
    p.add_argument("--movetime", type=float, default=None, help="seconds/position (overrides --depth)")
    p.add_argument("--blunder-cp", type=int, default=200, help="centipawn-loss threshold for a blunder")
    p.add_argument("--no-belief", dest="belief", action="store_false",
                   help="skip the constructed-belief cost (analysis A)")
    # ladder
    p.add_argument("--ladder-levels", default=None,
                   help="comma-separated UCI Skill Levels, e.g. 0,5,10,20 (default: 20 only; empty to skip)")
    p.add_argument("--ladder-games", type=int, default=None,
                   help="override --num-games for the ladder")
    # output
    p.add_argument("--json", default=None, help="also write the full report to this path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    # A single --num-games sets every game count; the per-section flags override it.
    if args.mq_games is None:
        args.mq_games = args.num_games
    if args.ladder_games is None:
        args.ladder_games = args.num_games
    if args.ladder_levels is None:
        args.ladder_levels = [20]  # full strength only unless a sweep is given
    elif args.ladder_levels.strip() == "":
        args.ladder_levels = []
    else:
        args.ladder_levels = [int(x) for x in args.ladder_levels.split(",") if x.strip()]

    engine_path = resolve_engine_path()
    if engine_path is None:
        raise SystemExit(
            "No Stockfish binary found. Install one (e.g. `dnf install stockfish`) and set "
            "STOCKFISH_EXECUTABLE, or put `stockfish` on PATH. This whole tool needs an engine."
        )

    # Resolve the device once; workers each load a net on this (passed as a string).
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu")
        args.device = "cpu"
    if args.workers is None:
        args.workers = _default_workers()
    args.workers = max(1, args.workers)

    # Load once on CPU here only to validate the checkpoint and read its history for
    # the log line; the nets that actually play are loaded per worker on args.device.
    _net, enc = load_net(args.checkpoint, torch.device("cpu"))
    del _net
    args.history = enc.history  # trust the checkpoint's own history depth
    print(f"Loaded {args.checkpoint} (history={args.history}); engine={engine_path}; "
          f"device={args.device}; workers={args.workers}")

    report: Dict = {"checkpoint": args.checkpoint, "engine": engine_path}

    if args.mq_games > 0:
        report["move_quality"] = mq = run_move_quality(args)
        print("\n=== Move quality & win-rate (vs ground-truth board) ===")
        _print_move_quality(mq)

    if args.ladder_levels:
        print("\n=== Strength ladder (win-rate vs TroutBot @ Skill Level) ===")
        report["ladder"] = run_ladder(args)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report to {args.json}")


if __name__ == "__main__":
    main()
