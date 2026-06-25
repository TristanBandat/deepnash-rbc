"""A tunable-strength Stockfish opponent, for a win-rate-vs-strength curve.

reconchess's ``TroutBot`` already wraps Stockfish with RBC belief handling (it
tracks a board from sense results and queries the engine).  Rather than
reimplement that, we subclass it and dial the engine's UCI ``Skill Level``
(0 = weakest, 20 = full strength).  Sweeping the level turns a single
"win-rate vs trout" number into a curve -- far more informative for a thesis
than one data point.

But stock ``TroutBot`` is fragile in exactly the way RBC makes inevitable: the
single board it reconstructs from sense results is often *illegal* as standard
chess (wrong king count, own king in check, ...).  Handing such a FEN to
Stockfish makes the engine process **terminate**, and stock TroutBot never
restarts it -- so it silently degrades to passing for the rest of the game,
flattening the whole skill ladder.  We override ``choose_move`` to:

  * skip the engine entirely on an invalid board (fall back to a legal move),
  * **resurrect** the engine (re-applying the skill level) if it has died,

so the configured strength actually holds for the whole game.

Note: Skill Level changes search/eval noise, not a calibrated Elo.  Report it as
an ordinal strength axis, not as Elo.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional

import chess
import chess.engine
from reconchess.bots.trout_bot import STOCKFISH_ENV_VAR, TroutBot


class TunableTroutBot(TroutBot):
    """TroutBot whose engine plays at a fixed UCI Skill Level (0-20) and which
    revives its engine instead of dying silently."""

    def __init__(self, skill_level: int = 20, movetime: float = 0.5):
        super().__init__()  # opens the engine (needs STOCKFISH_EXECUTABLE)
        self.skill_level = max(0, min(20, int(skill_level)))
        self.movetime = movetime
        self._engine_path = os.environ[STOCKFISH_ENV_VAR]
        self._apply_skill()

    def _apply_skill(self) -> None:
        try:
            self.engine.configure({"Skill Level": self.skill_level})
        except Exception:
            pass  # option absent on some builds -> plays full strength

    def _revive_engine(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass
        self.engine = chess.engine.SimpleEngine.popen_uci(self._engine_path, setpgrp=True)
        self._apply_skill()

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # king-capture shortcut, identical to stock TroutBot
        enemy_king_square = self.board.king(not self.color)
        if enemy_king_square:
            attackers = self.board.attackers(self.color, enemy_king_square)
            if attackers:
                return chess.Move(attackers.pop(), enemy_king_square)

        self.board.turn = self.color
        self.board.clear_stack()
        if not self.board.is_valid():
            # don't crash the engine on an illegal RBC board; play a legal move
            return random.choice(move_actions) if move_actions else None

        for attempt in (1, 2):  # one retry after reviving a dead engine
            try:
                return self.engine.play(self.board, chess.engine.Limit(time=self.movetime)).move
            except chess.engine.EngineTerminatedError:
                if attempt == 1:
                    self._revive_engine()
                    continue
            except chess.engine.EngineError:
                break  # bad position state -> fall through to a safe move
        return random.choice(move_actions) if move_actions else None


def make_ladder_opponent(skill_level: int) -> TunableTroutBot:
    return TunableTroutBot(skill_level=skill_level)


DEFAULT_LEVELS: List[int] = [0, 3, 5, 10, 15, 20]
