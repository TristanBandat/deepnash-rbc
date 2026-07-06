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

Defaults: device cuda; move-quality opponents trout/mht (strangefish2 opt-in); ladder at
Skill Level 20 only; --num-games (default 10) drives both the per-opponent and
per-ladder-level game counts unless --mq-games / --ladder-games override it.
"""

from __future__ import annotations

import argparse
import json
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


def _play_game(net, device, history, opponent_name, agent_is_white, seconds_per_player):
    """Play one game; return (history, agent_color, belief_fens, winner_color) or None."""
    agent = RNaDPlayer(net, device, history=history, sample=False, track_belief=True)
    opponent = _make_opponent(opponent_name)
    white, black = (agent, opponent) if agent_is_white else (opponent, agent)
    game = LocalGame(seconds_per_player=seconds_per_player)
    try:
        winner, _reason, hist = play_local_game(white, black, game=game)
    except Exception as e:
        print(f"[mq] game vs {opponent_name} aborted: {type(e).__name__}: {e}")
        return None
    color = chess.WHITE if agent_is_white else chess.BLACK
    return hist, color, list(agent.belief_fens), winner


def _run_one_opponent(net, device, args, analyst, opp_name) -> Dict:
    """Play args.mq_games vs one opponent and grade them. Returns a report dict with
    win/draw/loss (the outcome of those same games) plus per-colour and overall
    move-quality aggregates -- so choosing an opponent drives *all* the numbers."""
    per_color: Dict[bool, List[Aggregate]] = {chess.WHITE: [], chess.BLACK: []}
    wins = draws = losses = played = 0
    for g in range(args.mq_games):
        agent_is_white = (g % 2 == 0)
        got = _play_game(net, device, args.history, opp_name,
                         agent_is_white, args.seconds_per_player)
        if got is None:
            continue
        hist, color, belief_fens, winner = got
        played += 1
        outcome = "draw" if winner is None else ("win" if winner == color else "loss")
        if outcome == "win":
            wins += 1
        elif outcome == "draw":
            draws += 1
        else:
            losses += 1
        records = analyze_game(hist, color, analyst,
                               belief_fens=belief_fens if args.belief else None)
        per_color[color].append(aggregate(records))
        print(f"[mq:{opp_name}] game {g + 1}/{args.mq_games} "
              f"({'W' if color else 'B'}, {outcome}): {len(records)} moves analysed")
    out: Dict[str, object] = {
        "n": played, "wins": wins, "draws": draws, "losses": losses,
        "win_rate": round(wins / played, 4) if played else None,
        "draw_rate": round(draws / played, 4) if played else None,
    }
    all_aggs: List[Aggregate] = []
    for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        if per_color[color]:
            out[label] = asdict(merge(per_color[color]))
            all_aggs.extend(per_color[color])
    if all_aggs:
        out["overall"] = asdict(merge(all_aggs))
    return out


def run_move_quality(net, device, args, analyst: StockfishAnalyst) -> Dict:
    """Grade the agent against each opponent in args.mq_opponent, keyed by name."""
    return {opp: _run_one_opponent(net, device, args, analyst, opp)
            for opp in args.mq_opponent}


def run_ladder(net, device, args) -> List[Dict]:
    rows: List[Dict] = []
    for level in args.ladder_levels:
        wins = draws = losses = played = 0
        for g in range(args.ladder_games):
            agent = RNaDPlayer(net, device, history=args.history, sample=False)
            try:
                opponent = make_ladder_opponent(level)
            except Exception as e:
                print(f"[ladder] cannot build trout@{level}: {e}")
                break
            agent_is_white = (g % 2 == 0)
            white, black = (agent, opponent) if agent_is_white else (opponent, agent)
            game = LocalGame(seconds_per_player=args.seconds_per_player)
            try:
                winner, _reason, _hist = play_local_game(white, black, game=game)
            except Exception as e:
                print(f"[ladder] game@{level} aborted: {type(e).__name__}: {e}")
                continue
            played += 1
            agent_color = chess.WHITE if agent_is_white else chess.BLACK
            if winner is None:
                draws += 1
            elif winner == agent_color:
                wins += 1
            else:
                losses += 1
        row = {
            "skill_level": level,
            "n": played,
            "win_rate": round(wins / played, 4) if played else None,
            "draw_rate": round(draws / played, 4) if played else None,
        }
        rows.append(row)
        print(f"[ladder] skill={level:>2}  win={row['win_rate']}  "
              f"draw={row['draw_rate']}  (n={played})")
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

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    net, enc = load_net(args.checkpoint, device)
    args.history = enc.history  # trust the checkpoint's own history depth
    print(f"Loaded {args.checkpoint} (history={args.history}); engine={engine_path}")

    report: Dict = {"checkpoint": args.checkpoint, "engine": engine_path}

    if args.mq_games > 0:
        with StockfishAnalyst(engine_path, depth=args.depth, movetime=args.movetime,
                              blunder_cp=args.blunder_cp) as analyst:
            mq = run_move_quality(net, device, args, analyst)
        report["move_quality"] = mq
        print("\n=== Move quality & win-rate (vs ground-truth board) ===")
        _print_move_quality(mq)

    if args.ladder_levels:
        print("\n=== Strength ladder (win-rate vs TroutBot @ Skill Level) ===")
        report["ladder"] = run_ladder(net, device, args)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report to {args.json}")


if __name__ == "__main__":
    main()
