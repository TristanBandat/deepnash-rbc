"""Replay a finished game and grade the agent's moves against the oracle.

For each of the agent's move decisions we use the arbiter's GROUND-TRUTH board
(``GameHistory.truth_board_before_move``) -- information the agent never had --
and compute three conditions:

  taken_on_true    the move that actually happened, scored on the true board.
                   This is *raw chess quality*: it charges the agent for moves
                   that walked into a capture it couldn't see, or slid into a
                   wall.  Headline ACPL / blunder / top-1 come from here.

  requested_on_true the move the agent *asked* for, scored on the true board
                   (analysis B).  Where it differs from taken, RBC truncated or
                   blocked it -- so (taken vs requested) isolates one concrete
                   information cost: misjudging the board into a worse outcome.
                   Requested moves illegal on the true board are bucketed as
                   such (you literally cannot make that move) and reported.

  taken_on_belief  the taken move scored on the agent's *constructed naive
                   belief* board (analysis A).  cost_A = cpl_true - cpl_belief:
                   how much better the move looked under the agent's (stale /
                   incomplete) picture than it really was.  Proxy -- see belief.py.

Aggregates per colour and overall: ACPL (mean cpl over scored moves), blunder
rate, top-1 match rate, and a no-eval count (positions/moves we couldn't score,
reported rather than hidden so coverage is honest).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chess
from reconchess.history import GameHistory

from .engine import MoveEval, StockfishAnalyst


@dataclass
class MoveRecord:
    turn_index: int
    taken_on_true: MoveEval
    requested_on_true: MoveEval
    taken_on_belief: Optional[MoveEval] = None
    truncated: bool = False  # requested != taken (RBC blocked/cut the move)


@dataclass
class Aggregate:
    n_moves: int = 0
    n_scored: int = 0          # taken_on_true was scorable
    n_no_eval: int = 0
    acpl: float = 0.0          # mean cpl over scored taken-on-true moves
    blunder_rate: float = 0.0
    top1_rate: float = 0.0
    n_truncated: int = 0
    # information-cost summaries (means over moves scorable in both conditions)
    cost_A_mean: Optional[float] = None   # cpl_true - cpl_belief (analysis A)
    cost_A_n: int = 0
    cost_B_mean: Optional[float] = None   # cpl_taken - cpl_requested (analysis B)
    cost_B_n: int = 0


def analyze_game(
    history: GameHistory,
    color: chess.Color,
    analyst: StockfishAnalyst,
    belief_fens: Optional[List[str]] = None,
) -> List[MoveRecord]:
    """Grade every move ``color`` made in ``history``.

    ``belief_fens`` (if given) must be the per-move-decision FENs logged by
    ``RNaDPlayer(track_belief=True)``, aligned 1:1 with this colour's moves.
    """
    records: List[MoveRecord] = []
    move_turns = [t for t in history.turns(color) if history.has_move(t)]
    for i, turn in enumerate(move_turns):
        true_board = history.truth_board_before_move(turn)
        requested = history.requested_move(turn)
        taken = history.taken_move(turn)

        taken_eval = analyst.evaluate_move(true_board, taken, color)
        req_eval = analyst.evaluate_move(true_board, requested, color)

        belief_eval = None
        if belief_fens is not None and i < len(belief_fens):
            try:
                bb = chess.Board(belief_fens[i])
                belief_eval = analyst.evaluate_move(bb, taken, color)
            except ValueError:
                belief_eval = MoveEval(ok=False, reason="bad_belief_fen")

        records.append(MoveRecord(
            turn_index=i,
            taken_on_true=taken_eval,
            requested_on_true=req_eval,
            taken_on_belief=belief_eval,
            truncated=(requested != taken),
        ))
    return records


def aggregate(records: List[MoveRecord]) -> Aggregate:
    agg = Aggregate(n_moves=len(records))
    cpls: List[int] = []
    blunders = tops = 0
    cost_a: List[int] = []
    cost_b: List[int] = []
    for r in records:
        if r.truncated:
            agg.n_truncated += 1
        t = r.taken_on_true
        if t.ok:
            agg.n_scored += 1
            cpls.append(t.cpl)
            blunders += int(bool(t.is_blunder))
            tops += int(bool(t.is_top1))
            # cost A: same taken move, true vs believed board
            if r.taken_on_belief is not None and r.taken_on_belief.ok:
                cost_a.append(t.cpl - r.taken_on_belief.cpl)
            # cost B: requested vs taken, both on the true board
            if r.requested_on_true.ok:
                cost_b.append(t.cpl - r.requested_on_true.cpl)
        else:
            agg.n_no_eval += 1
    if cpls:
        agg.acpl = round(sum(cpls) / len(cpls), 1)
        agg.blunder_rate = round(blunders / len(cpls), 4)
        agg.top1_rate = round(tops / len(cpls), 4)
    if cost_a:
        agg.cost_A_mean = round(sum(cost_a) / len(cost_a), 1)
        agg.cost_A_n = len(cost_a)
    if cost_b:
        agg.cost_B_mean = round(sum(cost_b) / len(cost_b), 1)
        agg.cost_B_n = len(cost_b)
    return agg


def merge(aggs: List[Aggregate]) -> Aggregate:
    """Pool several per-game aggregates into one (weighting by move counts)."""
    out = Aggregate()
    cpl_w = blunder_w = top_w = 0.0
    a_w = b_w = 0.0
    for a in aggs:
        out.n_moves += a.n_moves
        out.n_scored += a.n_scored
        out.n_no_eval += a.n_no_eval
        out.n_truncated += a.n_truncated
        cpl_w += a.acpl * a.n_scored
        blunder_w += a.blunder_rate * a.n_scored
        top_w += a.top1_rate * a.n_scored
        if a.cost_A_mean is not None:
            a_w += a.cost_A_mean * a.cost_A_n
            out.cost_A_n += a.cost_A_n
        if a.cost_B_mean is not None:
            b_w += a.cost_B_mean * a.cost_B_n
            out.cost_B_n += a.cost_B_n
    if out.n_scored:
        out.acpl = round(cpl_w / out.n_scored, 1)
        out.blunder_rate = round(blunder_w / out.n_scored, 4)
        out.top1_rate = round(top_w / out.n_scored, 4)
    if out.cost_A_n:
        out.cost_A_mean = round(a_w / out.cost_A_n, 1)
    if out.cost_B_n:
        out.cost_B_mean = round(b_w / out.cost_B_n, 1)
    return out
