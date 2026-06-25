"""Step 1 learning curve: move quality vs training step.

For each checkpoint, play N greedy games vs AttackerBot (alternating colours),
grade every agent move against the ground-truth board with Stockfish, and report
overall ACPL / blunder% / top-1% plus win-rate. A descending ACPL means training
is still extracting chess skill; a flat line means it has plateaued (=> the
bottleneck is algorithmic/representational, not compute).
"""

from __future__ import annotations

import chess
import torch
from reconchess import LocalGame, play_local_game

from deepnash_rbc.agent import RNaDPlayer
from deepnash_rbc.analysis import move_quality as mq
from deepnash_rbc.analysis.engine import StockfishAnalyst
from deepnash_rbc.eval import _make_opponent
from deepnash_rbc.checkpoints import checkpoint_path
from deepnash_rbc.play_session import load_net

STEPS = [1000, 10000, 25000, 50000, 75000, 100000]
OPPONENT = "attacker"
N_GAMES = 20
DEPTH = 16
device = torch.device("cpu")


def run_ckpt(step: int, analyst: StockfishAnalyst) -> dict:
    net, enc = load_net(checkpoint_path("checkpoints", step), device)
    H = enc.history
    aggs = []
    wins = draws = losses = played = 0
    for g in range(N_GAMES):
        agent_is_white = (g % 2 == 0)
        agent = RNaDPlayer(net, device, history=H, sample=False)
        opp = _make_opponent(OPPONENT)
        white, black = (agent, opp) if agent_is_white else (opp, agent)
        try:
            winner, _r, hist = play_local_game(white, black, game=LocalGame(seconds_per_player=900.0))
        except Exception as e:
            print(f"  [{step}] game aborted: {type(e).__name__}: {e}", flush=True)
            continue
        played += 1
        color = chess.WHITE if agent_is_white else chess.BLACK
        if winner is None:
            draws += 1
        elif winner == color:
            wins += 1
        else:
            losses += 1
        try:
            aggs.append(mq.aggregate(mq.analyze_game(hist, color, analyst)))
        except Exception as e:
            print(f"  [{step}] analyze failed: {e}", flush=True)
    m = mq.merge(aggs)
    print(f"  done {step}: ACPL={m.acpl} blunder={m.blunder_rate*100:.1f}% "
          f"top1={m.top1_rate*100:.1f}% win={wins}/{played}", flush=True)
    return {"step": step, "acpl": m.acpl, "blunder": m.blunder_rate,
            "top1": m.top1_rate, "scored": m.n_scored, "no_eval": m.n_no_eval,
            "win_rate": round(wins / played, 3) if played else None, "n": played}


def main() -> None:
    rows = []
    with StockfishAnalyst(depth=DEPTH) as analyst:
        for step in STEPS:
            rows.append(run_ckpt(step, analyst))
    print(f"\n{'step':>7} {'ACPL':>7} {'blunder%':>9} {'top1%':>7} {'win%':>6} {'scored':>7} {'no-eval':>8}")
    for r in rows:
        wr = "n/a" if r["win_rate"] is None else f"{r['win_rate']*100:.0f}"
        print(f"{r['step']:>7} {r['acpl']:>7} {r['blunder']*100:>8.1f}% "
              f"{r['top1']*100:>6.1f}% {wr:>6} {r['scored']:>7} {r['no_eval']:>8}")


if __name__ == "__main__":
    main()
