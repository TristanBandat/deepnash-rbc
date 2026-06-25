"""Action encodings for RBC.

Move policy uses the AlphaZero 8x8x73 chess move encoding (4672 planes) plus one
extra index for the explicit "pass" action that RBC allows -> 4673 logits total.
Sense policy is a flat distribution over the 64 board squares (the sense *center*);
reconchess hands us the list of legal sense squares each turn, which we use as a mask.

Important property: within the set of legal moves produced by reconchess for a
single board state, no two distinct moves map to the same index (queen-promotions
fold into the distance-1 queen-move planes; under-promotions get their own planes).
That is all sampling needs -- we mask the head to the candidate indices, sample,
and map the chosen index back to the concrete chess.Move object we already hold.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import chess

# ---- move action space ----------------------------------------------------
_QUEEN_DIRS = [  # (file_delta, rank_delta), N, NE, E, SE, S, SW, W, NW
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
]
_KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

PLANES_PER_SQUARE = 73  # 56 queen + 8 knight + 9 underpromotion
MOVE_PLANES = PLANES_PER_SQUARE * 64  # 4672
PASS_INDEX = MOVE_PLANES  # 4672
MOVE_ACTIONS = MOVE_PLANES + 1  # 4673

SENSE_ACTIONS = 64


def _sq_file(sq: int) -> int:
    return sq % 8


def _sq_rank(sq: int) -> int:
    return sq // 8


def move_to_index(move: Optional[chess.Move]) -> int:
    """Map a chess.Move (or None=pass) to a flat policy index in [0, MOVE_ACTIONS)."""
    if move is None or move == chess.Move.null():
        return PASS_INDEX

    frm, to = move.from_square, move.to_square
    df = _sq_file(to) - _sq_file(frm)
    dr = _sq_rank(to) - _sq_rank(frm)

    # under-promotions get dedicated planes; queen promos fall through to queen moves
    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = _UNDERPROMO_PIECES.index(move.promotion)
        # direction by file delta: straight=0, left=1, right=2
        dir_idx = {0: 0, -1: 1, 1: 2}[df]
        plane = 64 + piece_idx * 3 + dir_idx
        return frm * PLANES_PER_SQUARE + plane

    # knight moves
    if (abs(df), abs(dr)) in ((1, 2), (2, 1)):
        k_idx = _KNIGHT_DELTAS.index((df, dr))
        plane = 56 + k_idx
        return frm * PLANES_PER_SQUARE + plane

    # queen-like (rook/bishop/king/pawn pushes, and queen-promotions) moves
    dist = max(abs(df), abs(dr))
    step = (0 if df == 0 else df // abs(df), 0 if dr == 0 else dr // abs(dr))
    dir_idx = _QUEEN_DIRS.index(step)
    plane = dir_idx * 7 + (dist - 1)
    return frm * PLANES_PER_SQUARE + plane


def build_move_index(moves: List[chess.Move]) -> Dict[int, chess.Move]:
    """Index -> Move map for a single turn's candidate moves (collision-checked)."""
    out: Dict[int, chess.Move] = {}
    for mv in moves:
        idx = move_to_index(mv)
        if idx in out and out[idx] != mv:
            # Defensive: should not happen for a single legal move set. Keep first.
            continue
        out[idx] = mv
    return out
