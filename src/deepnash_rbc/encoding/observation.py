"""Observation encoding for RBC -- model-free, observation-only (no belief state).

The agent always knows its own pieces exactly (RBC notifies you of every capture
and the true result of your own moves), so we maintain an exact map of *our*
pieces. Opponent information enters ONLY through (a) the latest 3x3 sense result
and (b) capture-square pings. We never reconstruct or track a distribution over
the opponent's full board -- that implicit belief is what the network learns to
carry across the stacked history, exactly as in DeepNash.

We track our own pieces with a plain {square: piece_type} dict rather than a
chess.Board, because RBC moves are frequently illegal under full chess rules
(moving into check, sliding captures that truncate, etc.) and python-chess's
legality machinery would reject them.

Per-frame channel layout (FRAME_CHANNELS planes, each 8x8):
  0-5    our pieces by type (P,N,B,R,Q,K)
  6-11   opponent pieces revealed by the MOST RECENT sense, by type
  12     squares covered by the most recent sense (the 3x3 window)
  13     square where one of our pieces was just captured
  14     square where we just captured an opponent piece
  15     our last move: from-square
  16     our last move: to-square
  17     move-was-truncated flag (requested != taken), broadcast over the board
  18     side-to-move color plane (all 1 if we are White, else all 0)
A richer encoder would *age* opponent observations instead of only keeping the
last sense; left as a clearly-marked extension point.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import chess
import numpy as np

FRAME_CHANNELS = 19

C_OWN = 0          # 0..5
C_OPP_SENSED = 6   # 6..11
C_SENSE_MASK = 12
C_CAPTURED_MINE = 13
C_CAPTURED_THEIRS = 14
C_LAST_FROM = 15
C_LAST_TO = 16
C_TRUNCATED = 17
C_COLOR = 18

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_PIECE_PLANE = {p: i for i, p in enumerate(_PIECE_ORDER)}


def _rc(sq: int) -> Tuple[int, int]:
    return sq // 8, sq % 8


class ObservationEncoder:
    """Stateful encoder. Driven by the agent's reconchess callbacks; emits a
    stacked [history * FRAME_CHANNELS, 8, 8] float32 tensor on demand."""

    def __init__(self, history: int = 8):
        self.history = history
        self.color: bool = chess.WHITE
        self.own: Dict[int, int] = {}  # square -> piece_type (our pieces only)
        self._frames: Deque[np.ndarray] = deque(maxlen=history)
        self._cur = self._blank()
        self.reset(chess.WHITE)

    # -- lifecycle -----------------------------------------------------------
    def reset(self, color: bool) -> None:
        self.color = color
        self.own = self._starting_pieces(color)
        self._frames.clear()
        for _ in range(self.history):
            self._frames.append(self._blank())
        self._cur = self._new_frame()

    @staticmethod
    def _starting_pieces(color: bool) -> Dict[int, int]:
        b = chess.Board()
        return {sq: p.piece_type for sq, p in b.piece_map().items() if p.color == color}

    def _blank(self) -> np.ndarray:
        return np.zeros((FRAME_CHANNELS, 8, 8), dtype=np.float32)

    def _new_frame(self) -> np.ndarray:
        f = self._blank()
        self._write_own(f)
        f[C_COLOR, :, :] = 1.0 if self.color == chess.WHITE else 0.0
        return f

    # -- per-turn observation updates ----------------------------------------
    def _write_own(self, frame: np.ndarray) -> None:
        frame[C_OWN:C_OWN + 6] = 0.0
        for sq, ptype in self.own.items():
            r, c = _rc(sq)
            frame[C_OWN + _PIECE_PLANE[ptype], r, c] = 1.0

    def begin_turn(self) -> None:
        self._cur = self._new_frame()

    def opponent_captured_my_piece(self, capture_square: Optional[int]) -> None:
        if capture_square is not None:
            r, c = _rc(capture_square)
            self._cur[C_CAPTURED_MINE, r, c] = 1.0
            self.own.pop(capture_square, None)
            self._write_own(self._cur)

    def sense_result(self, result: List[Tuple[int, Optional[chess.Piece]]]) -> None:
        for sq, piece in result:
            r, c = _rc(sq)
            self._cur[C_SENSE_MASK, r, c] = 1.0
            if piece is not None and piece.color != self.color:
                self._cur[C_OPP_SENSED + _PIECE_PLANE[piece.piece_type], r, c] = 1.0

    def move_result(
        self,
        requested: Optional[chess.Move],
        taken: Optional[chess.Move],
        captured_opponent_piece: bool,
        capture_square: Optional[int],
    ) -> None:
        if taken is not None:
            fr, fc = _rc(taken.from_square)
            tr, tc = _rc(taken.to_square)
            self._cur[C_LAST_FROM, fr, fc] = 1.0
            self._cur[C_LAST_TO, tr, tc] = 1.0
            # update our own-piece map
            ptype = self.own.pop(taken.from_square, chess.PAWN)
            self.own[taken.to_square] = taken.promotion if taken.promotion else ptype
            self._write_own(self._cur)
        if requested != taken:
            self._cur[C_TRUNCATED, :, :] = 1.0
        if captured_opponent_piece and capture_square is not None:
            r, c = _rc(capture_square)
            self._cur[C_CAPTURED_THEIRS, r, c] = 1.0

    def commit_turn(self) -> None:
        self._frames.append(self._cur.copy())

    # -- output --------------------------------------------------------------
    def tensor(self) -> np.ndarray:
        frames = list(self._frames)[1:] + [self._cur]
        return np.concatenate(frames, axis=0)
