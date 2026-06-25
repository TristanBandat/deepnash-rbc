"""StockfishAnalyst -- a thin, RBC-tolerant wrapper over a UCI engine.

Why not just call ``chess.engine`` directly?  RBC throws three things at a chess
engine that vanilla analysis code does not expect:

  1. **Illegal positions.** The arbiter's true board can be illegal as standard
     chess (the side to move's king is capturable, a king is missing because it
     was just taken, etc.).  Stockfish rejects these.  We detect them up front
     (``board.is_valid()``) and report a *no-eval* rather than crashing the run.
  2. **Illegal / null "moves".** A requested move may be illegal on the true
     board (it got truncated -- that is precisely what condition B measures); a
     turn may be a pass (``None``).  We score what we can and bucket the rest.
  3. **Mate scores.** Left unclamped, a single forced mate dwarfs hundreds of
     centipawns of ordinary error and wrecks any average.  We clamp every score
     to +/-``max_cp`` before differencing, the standard ACPL convention.

Scores are always returned from a fixed point of view (the agent's colour), in
centipawns, so positive = good for the agent.  Results are cached by FEN because
the same position recurs across the best-move probe and the played-move probe,
and across games.
"""

from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import chess
import chess.engine

STOCKFISH_ENV_VAR = "STOCKFISH_EXECUTABLE"


def bundled_stockfish() -> Optional[str]:
    """Look for a Stockfish binary shipped in ``tools/stockfish/`` (the repo bundles
    one, e.g. ``stockfish-ubuntu-x86-64-bmi2``). Checks the repo root inferred from
    this file and the current working directory, so it works under ``uv run`` from
    the project dir even though ``tools/`` is not part of the installed package."""
    roots = [Path(__file__).resolve().parents[3], Path.cwd()]
    seen: set[str] = set()
    for root in roots:
        d = str(root)
        if d in seen:
            continue
        seen.add(d)
        for cand in sorted(glob.glob(os.path.join(d, "tools", "stockfish", "stockfish*"))):
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


def resolve_engine_path(path: Optional[str] = None) -> Optional[str]:
    """Find a Stockfish binary: explicit arg, then env var, then PATH, then the
    binary bundled in ``tools/stockfish/``."""
    if path:
        return path
    env = os.environ.get(STOCKFISH_ENV_VAR)
    if env:
        return env
    return shutil.which("stockfish") or bundled_stockfish()


@dataclass
class MoveEval:
    """Grade of one (position, move) pair from the agent's point of view."""
    ok: bool                       # False => position/move could not be scored
    reason: str = ""               # populated when ok is False
    best_cp: Optional[int] = None  # eval of the engine's best move (agent POV)
    move_cp: Optional[int] = None  # eval after the move actually considered
    cpl: Optional[int] = None      # centipawn loss = best_cp - move_cp (>= 0)
    is_top1: Optional[bool] = None  # did the move match the engine's first choice?
    is_blunder: Optional[bool] = None  # cpl >= blunder threshold


class StockfishAnalyst:
    """Context manager around a UCI engine. Open once, score many positions.

    Strength of the *analysis* (not of an opponent) is set by ``depth`` or, if
    given, ``movetime`` seconds. Use a fixed limit for reproducibility.
    """

    def __init__(
        self,
        engine_path: Optional[str] = None,
        depth: int = 12,
        movetime: Optional[float] = None,
        max_cp: int = 1500,
        mate_score: int = 100_000,
        blunder_cp: int = 200,
    ):
        self.engine_path = resolve_engine_path(engine_path)
        if not self.engine_path or not os.path.exists(self.engine_path):
            raise FileNotFoundError(
                "No Stockfish binary found. Install one (e.g. `dnf install stockfish`) "
                f"and set {STOCKFISH_ENV_VAR}, or pass engine_path=..."
            )
        self.limit = (
            chess.engine.Limit(time=movetime) if movetime is not None
            else chess.engine.Limit(depth=depth)
        )
        self.max_cp = max_cp
        self.mate_score = mate_score
        self.blunder_cp = blunder_cp
        self._engine: Optional[chess.engine.SimpleEngine] = None
        # fen -> (best_move_uci or None, white_pov_cp). Keyed white-POV so the
        # same cache serves analysis of either colour's decisions.
        self._cache: Dict[str, Tuple[Optional[str], int]] = {}

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "StockfishAnalyst":
        self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        return self

    def __exit__(self, *exc) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    # -- core analysis -------------------------------------------------------
    def _analyse_white_pov(self, board: chess.Board) -> Tuple[Optional[str], int]:
        """Return (best_move_uci, score in centipawns from WHITE's POV), cached.

        Raises if the engine genuinely fails; callers guard with is_valid().
        """
        key = board.fen()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        info = self._engine.analyse(board, self.limit)  # type: ignore[union-attr]
        score = info["score"].white().score(mate_score=self.mate_score)
        score = max(-self.max_cp, min(self.max_cp, int(score)))
        pv = info.get("pv")
        best = pv[0].uci() if pv else None
        self._cache[key] = (best, score)
        return best, score

    def _score_pov(self, board: chess.Board, pov: chess.Color) -> int:
        _, white_cp = self._analyse_white_pov(board)
        return white_cp if pov == chess.WHITE else -white_cp

    # -- public API ----------------------------------------------------------
    def evaluate_move(
        self,
        board: chess.Board,
        move: Optional[chess.Move],
        pov: chess.Color,
        blunder_cp: Optional[int] = None,
    ) -> MoveEval:
        """Grade ``move`` played in ``board`` from ``pov``'s perspective.

        ``board`` is taken as-is except that its side to move is forced to
        ``pov`` (the agent is the one deciding).  A ``None`` move is treated as a
        pass and scored via a null move when legal.
        """
        thresh = self.blunder_cp if blunder_cp is None else blunder_cp
        b = board.copy(stack=False)
        b.turn = pov
        if not b.is_valid():
            return MoveEval(ok=False, reason="illegal_position")
        try:
            best_uci, _ = self._analyse_white_pov(b)
            best_cp = self._score_pov(b, pov)
        except chess.engine.EngineError as e:
            return MoveEval(ok=False, reason=f"engine_error:{type(e).__name__}")

        # apply the move (or a pass) on a fresh copy
        nb = b.copy(stack=False)
        if move is None:
            if b.is_check():
                return MoveEval(ok=False, reason="pass_in_check")
            nb.push(chess.Move.null())
            move_uci = None
        else:
            if move not in nb.legal_moves:
                # requested move illegal on this board (e.g. truncated in RBC)
                return MoveEval(ok=False, reason="illegal_move", best_cp=best_cp)
            nb.push(move)
            move_uci = move.uci()
        try:
            move_cp = self._score_pov(nb, pov)
        except chess.engine.EngineError as e:
            return MoveEval(ok=False, reason=f"engine_error:{type(e).__name__}", best_cp=best_cp)

        cpl = max(0, best_cp - move_cp)
        return MoveEval(
            ok=True,
            best_cp=best_cp,
            move_cp=move_cp,
            cpl=cpl,
            is_top1=(best_uci is not None and move_uci == best_uci),
            is_blunder=(cpl >= thresh),
        )
