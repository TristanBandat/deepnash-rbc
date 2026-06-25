"""BeliefBoardTracker -- a constructed, *naive* belief board (analysis A).

The network is model-free: it carries no explicit board, only the latest sense
snapshot plus its own exact pieces, with the real belief living implicitly in
its activations.  That implicit belief is **not extractable** as a FEN.  So to
get a concrete "what did the agent think the board looked like?" for the
cost-of-information comparison, we *construct* a deliberately simple proxy and
label it as such.

The proxy = the most naive observation-consistent single board:
  * our own pieces, tracked exactly (RBC tells us every capture and the true
    result of our own moves);
  * the opponent's pieces left wherever we last *observed* them -- the standard
    start position, overwritten by 3x3 sense reveals and cleared on sensed-empty
    squares; unobserved opponent moves are simply not applied.

THIS IS A PROXY, NOT THE NETWORK'S BELIEF, and it has documented distortions:

  * **Material conservation (handled).** The naive "assume home pieces + overlay
    what we sense" double-counts: sensing a pawn that has moved to e5 while we
    still assume one on e7 yields nine black pawns and an *illegal* board, which
    Stockfish would reject -- silently gutting analysis A.  So whenever an
    observation pushes an opponent piece type over its legal starting count, we
    evict the *stalest* assumed piece of that type (the one we have gone longest
    without observing), modelling "that home piece must be the one that moved."
    This conserves material and keeps the board legal far more often.  It is a
    heuristic: e.g. it cannot represent under-/over-promotion faithfully.
  * **Unknown capturers.** When the opponent captures one of our pieces, an enemy
    of unknown type now sits there; we cannot identify it, so we drop the square.
  * **Staleness.** Opponent pieces we have not sensed recently sit where last
    seen; the longer since an observation, the less the belief reflects reality.

The gap between a move's quality on this board and on the true board is then a
(noisy) signal of how much the agent's outdated/incomplete picture costs it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess

# legal maximum count per piece type in a standard army (pre-promotion).
_START_COUNTS = {
    chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,
    chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1,
}


class BeliefBoardTracker:
    """Driven by the same reconchess callbacks as the agent's encoder."""

    def __init__(self, color: chess.Color = chess.WHITE):
        self.color: chess.Color = color
        self.pieces: Dict[int, chess.Piece] = {}
        # observation recency per square (higher = more recently confirmed).
        # Assumed home pieces start at clock 0; every sense bumps the clock so
        # eviction always sacrifices the least-recently-seen assumption.
        self._stamp: Dict[int, int] = {}
        self._clock: int = 0
        self.reset(color)

    def reset(self, color: chess.Color) -> None:
        self.color = color
        self.pieces = dict(chess.Board().piece_map())
        self._stamp = {sq: 0 for sq in self.pieces}
        self._clock = 0

    # -- observation updates -------------------------------------------------
    def opponent_captured_my_piece(self, capture_square: Optional[int]) -> None:
        # We lose the piece; an enemy of unknown type now occupies the square.
        # We cannot identify it, so we drop the square (documented limitation).
        if capture_square is not None:
            self.pieces.pop(capture_square, None)
            self._stamp.pop(capture_square, None)

    def sense_result(self, result: List[Tuple[int, Optional[chess.Piece]]]) -> None:
        self._clock += 1
        for sq, piece in result:
            if piece is not None and piece.color != self.color:
                self.pieces[sq] = piece                      # observed enemy piece
                self._stamp[sq] = self._clock
            else:
                # sensed empty (or our own piece) -> no enemy lives here; clear
                # any stale enemy belief. Never touch our own exact pieces.
                existing = self.pieces.get(sq)
                if existing is not None and existing.color != self.color:
                    self.pieces.pop(sq, None)
                    self._stamp.pop(sq, None)
        self._enforce_material()

    def apply_my_move(self, taken: Optional[chess.Move]) -> None:
        if taken is None:
            return
        piece = self.pieces.pop(taken.from_square, None)
        self._stamp.pop(taken.from_square, None)
        if piece is None:                                    # defensive
            piece = chess.Piece(chess.PAWN, self.color)
        if taken.promotion:
            piece = chess.Piece(taken.promotion, self.color)
        self.pieces[taken.to_square] = piece                 # may capture stale enemy
        self._stamp[taken.to_square] = self._clock

    def _enforce_material(self) -> None:
        """Evict stalest assumed opponent pieces so no type exceeds its legal
        starting count -- the fix that keeps the belief board valid."""
        by_type: Dict[int, List[int]] = {}
        for sq, piece in self.pieces.items():
            if piece.color != self.color:
                by_type.setdefault(piece.piece_type, []).append(sq)
        for ptype, squares in by_type.items():
            limit = _START_COUNTS[ptype]
            if len(squares) <= limit:
                continue
            # drop the oldest (lowest stamp) extras: those are home assumptions
            # the moved piece left behind.
            squares.sort(key=lambda s: self._stamp.get(s, 0))
            for sq in squares[: len(squares) - limit]:
                self.pieces.pop(sq, None)
                self._stamp.pop(sq, None)

    # -- output --------------------------------------------------------------
    def board(self) -> chess.Board:
        """The believed position with our colour to move. May still be illegal
        as standard chess; the analyst guards every position with is_valid()."""
        b = chess.Board(None)            # empty board, no castling/ep
        b.set_piece_map(dict(self.pieces))
        b.turn = self.color
        b.clear_stack()
        return b

    def fen(self) -> str:
        return self.board().fen()
