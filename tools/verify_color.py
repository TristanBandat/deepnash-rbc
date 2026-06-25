"""Verify the White/Black move-quality asymmetry across opponents.

Within each condition both colours are graded identically, so the W-vs-B
comparison is valid even though conditions differ in setup:
  random    agent greedy (sample=False); RandomBot supplies position variety.
            alternating colours.
  self      agent vs itself, sampled (sample=True) so games aren't identical.
            both sides are the SAME network -> removes any opponent-by-colour
            bias, isolating the model. each game grades both colours.

Prints a table; if Black ACPL >> White in *self-play* too, the asymmetry is the
model (encoding), not AttackerBot sharpness.
"""

from __future__ import annotations

import chess
import torch
from reconchess import LocalGame, play_local_game

from deepnash_rbc.agent import RNaDPlayer
from deepnash_rbc.analysis import move_quality as mq
from deepnash_rbc.analysis.engine import StockfishAnalyst
from deepnash_rbc.eval import _make_opponent
from deepnash_rbc.play_session import load_net

CKPT = "checkpoints/deepnash_async_100000.pt"
DEPTH = 16
N_RANDOM = 40   # games vs RandomBot (alternating colours)
N_SELF = 30     # self-play games (each yields both colours)

device = torch.device("cpu")
net, enc = load_net(CKPT, device)
H = enc.history


def play_vs(opp_name: str, agent_is_white: bool):
    agent = RNaDPlayer(net, device, history=H, sample=False)
    opp = _make_opponent(opp_name)
    white, black = (agent, opp) if agent_is_white else (opp, agent)
    try:
        _w, _r, hist = play_local_game(white, black, game=LocalGame(seconds_per_player=900.0))
    except Exception as e:
        print(f"  vs {opp_name} aborted: {type(e).__name__}: {e}", flush=True)
        return None
    return hist, (chess.WHITE if agent_is_white else chess.BLACK)


def play_self():
    w = RNaDPlayer(net, device, history=H, sample=True)
    b = RNaDPlayer(net, device, history=H, sample=True)
    try:
        _w, _r, hist = play_local_game(w, b, game=LocalGame(seconds_per_player=900.0))
    except Exception as e:
        print(f"  self-play aborted: {type(e).__name__}: {e}", flush=True)
        return None
    return hist


def main() -> None:
    aggs = {"random": {chess.WHITE: [], chess.BLACK: []},
            "self":   {chess.WHITE: [], chess.BLACK: []}}
    with StockfishAnalyst(depth=DEPTH) as analyst:
        for g in range(N_RANDOM):
            r = play_vs("random", agent_is_white=(g % 2 == 0))
            if r is None:
                continue
            hist, color = r
            try:
                aggs["random"][color].append(mq.aggregate(mq.analyze_game(hist, color, analyst)))
            except Exception as e:
                print(f"  analyze(random) failed: {e}", flush=True)
            if (g + 1) % 10 == 0:
                print(f"  random {g + 1}/{N_RANDOM}", flush=True)
        for g in range(N_SELF):
            hist = play_self()
            if hist is None:
                continue
            for color in (chess.WHITE, chess.BLACK):
                try:
                    aggs["self"][color].append(mq.aggregate(mq.analyze_game(hist, color, analyst)))
                except Exception as e:
                    print(f"  analyze(self) failed: {e}", flush=True)
            if (g + 1) % 10 == 0:
                print(f"  self {g + 1}/{N_SELF}", flush=True)

    print(f"\n{'condition/color':16} {'ACPL':>7} {'blunder%':>9} {'top1%':>7} {'scored':>7} {'no-eval':>8} {'games':>6}")
    for cond in ("random", "self"):
        for color, lbl in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            games = aggs[cond][color]
            m = mq.merge(games)
            print(f"{cond + ' ' + lbl:16} {m.acpl:>7} {m.blunder_rate * 100:>8.1f}% "
                  f"{m.top1_rate * 100:>6.1f}% {m.n_scored:>7} {m.n_no_eval:>8} {len(games):>6}")


if __name__ == "__main__":
    main()
