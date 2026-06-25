"""Human-vs-checkpoint game session (framework-agnostic core).

Drives a reconchess LocalGame one decision at a time so a human can play through
a UI: the human's turn pauses for sense + move input, the bot's turn is driven
immediately. Builds the human's *fogged* view of the board -- own pieces are
known exactly, enemy pieces are shown only as last-known reconnaissance intel
(with an age), never the ground truth. Emits a human-readable notification log.

This module has no web/UI dependency so it can be unit-tested headlessly; the
Flask layer in play_server.py is a thin wrapper over it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import chess
import torch

from .agent import RNaDPlayer
from .network import DeepNashNet

try:
    from reconchess import LocalGame
except Exception:  # pragma: no cover
    LocalGame = None


class GameSession:
    def __init__(self, net: DeepNashNet, human_color: bool, history: int,
                 sample: bool = False, seconds_per_player: float = 900.0):
        self.net = net
        self.human_color = human_color
        self.bot_color = not human_color
        self.device = torch.device("cpu")
        self.bot = RNaDPlayer(net, self.device, history=history, sample=sample)
        self.bot.handle_game_start(self.bot_color, chess.Board(), "human")

        self.game = LocalGame(seconds_per_player=seconds_per_player)
        self.game.start()

        self.revealed: Dict[int, Dict] = {}   # enemy intel: square -> {symbol, seen_turn}
        self.log: List[str] = []
        self.human_turn_no = 0
        self.phase = "sense"                   # 'sense' | 'move' | 'over'

        self._log(f"New game. You are {'White' if human_color else 'Black'}.")
        # if the bot is White it moves first; play bot turns until it's our turn
        if self.game.turn == self.bot_color:
            self._run_bot_turn()
        self._begin_human_turn()

    # ------------------------------------------------------------------ state
    def state(self) -> Dict:
        truth = self.game.board
        squares = []
        for sq in range(64):
            own_sym = None
            piece = truth.piece_at(sq)
            if piece is not None and piece.color == self.human_color:
                own_sym = piece.symbol()
            intel = None
            if own_sym is None and sq in self.revealed:
                r = self.revealed[sq]
                intel = {"symbol": r["symbol"], "age": self.human_turn_no - r["seen_turn"]}
            squares.append({"square": sq, "own": own_sym, "enemy": intel})

        over = self.game.is_over()
        out: Dict = {
            "squares": squares,
            "phase": "over" if over else self.phase,
            "turn": "human",
            "human_color": "white" if self.human_color else "black",
            "log": self.log[-200:],
            "over": over,
            "winner": None,
            "reason": None,
            "seconds_left": round(self.game.get_seconds_left(), 1),
        }
        if over:
            w = self.game.get_winner_color()
            out["winner"] = ("draw" if w is None else
                             ("human" if w == self.human_color else "bot"))
            reason = self.game.get_win_reason()
            out["reason"] = getattr(reason, "name", str(reason)) if reason else None
            return out

        if self.phase == "sense":
            out["sense_actions"] = list(self.game.sense_actions())
        elif self.phase == "move":
            out["moves"] = self._grouped_moves()
        return out

    def _grouped_moves(self) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for mv in self.game.move_actions():
            grouped.setdefault(str(mv.from_square), []).append({
                "to": mv.to_square,
                "uci": mv.uci(),
                "promotion": chess.piece_symbol(mv.promotion) if mv.promotion else None,
            })
        return grouped

    # ------------------------------------------------------------ human turn
    def _begin_human_turn(self) -> None:
        if self.game.is_over():
            self.phase = "over"
            self._finish()
            return
        self.human_turn_no += 1
        cap = self.game.opponent_move_results()
        if cap is not None:
            self._log(f"The opponent captured your piece on {chess.square_name(cap)}.")
            self.revealed.pop(cap, None)
        self.phase = "sense"

    def do_sense(self, square: Optional[int]) -> Dict:
        if self.phase != "sense":
            raise ValueError("not the sense phase")
        result = self.game.sense(square)
        found = []
        for sq, piece in result:
            if piece is not None and piece.color == self.bot_color:
                self.revealed[sq] = {"symbol": piece.symbol(), "seen_turn": self.human_turn_no}
                found.append(f"{piece.symbol()}@{chess.square_name(sq)}")
            elif piece is None or piece.color == self.human_color:
                # square is empty (of enemy) -> clear any stale intel there
                self.revealed.pop(sq, None)
        where = chess.square_name(square) if square is not None else "nowhere"
        self._log(f"Sensed around {where}: " + (", ".join(found) if found else "no enemy pieces."))
        self.phase = "move"
        return self.state()

    def do_move(self, uci: Optional[str]) -> Dict:
        if self.phase != "move":
            raise ValueError("not the move phase")
        requested = chess.Move.from_uci(uci) if uci else None
        req, taken, capsq = self.game.move(requested)
        if requested is None:
            self._log("You passed.")
        elif taken is None:
            self._log("Your move did not go through.")
        elif taken != requested:
            self._log(f"Your move was cut short — stopped at {chess.square_name(taken.to_square)}.")
        else:
            self._log(f"You moved {taken.uci()}.")
        if capsq is not None:
            self._log(f"You captured a piece on {chess.square_name(capsq)}!")
            self.revealed.pop(capsq, None)
        self.game.end_turn()

        if self.game.is_over():
            self._finish()
            return self.state()

        self._run_bot_turn()
        self._begin_human_turn()
        return self.state()

    # -------------------------------------------------------------- bot turn
    def _run_bot_turn(self) -> None:
        g, bot = self.game, self.bot
        t = g.get_seconds_left()
        cap = g.opponent_move_results()
        bot.handle_opponent_move_result(cap is not None, cap)

        sense_sq = bot.choose_sense(g.sense_actions(), g.move_actions(), t)
        bot.handle_sense_result(g.sense(sense_sq))

        mv = bot.choose_move(g.move_actions(), t)
        req, taken, capsq = g.move(mv)
        bot.handle_move_result(req, taken, capsq is not None, capsq)
        if capsq is not None:
            # the human learns of this loss at the start of their turn (via
            # opponent_move_results); no need to leak it here.
            pass
        g.end_turn()
        self._log("The opponent moved.")

    # --------------------------------------------------------------- finish
    def _finish(self) -> None:
        self.phase = "over"
        try:
            self.bot.handle_game_end(self.game.get_winner_color(),
                                     self.game.get_win_reason(),
                                     self.game.get_game_history())
        except Exception:
            pass
        w = self.game.get_winner_color()
        if w is None:
            self._log("Game over — draw.")
        elif w == self.human_color:
            self._log("Game over — you win!")
        else:
            self._log("Game over — the opponent wins.")

    def _log(self, msg: str) -> None:
        self.log.append(msg)


# ----------------------------------------------------------------- loading
def load_net(path: str, device: torch.device):
    """Load a checkpoint, rebuilding the exact architecture it was trained with
    (falls back to current defaults if the checkpoint predates config saving)."""
    from .config import EncodingConfig, NetworkConfig
    ckpt = torch.load(path, map_location=device, weights_only=False)
    enc = EncodingConfig(**ckpt["enc_cfg"]) if "enc_cfg" in ckpt else EncodingConfig()
    net_cfg = NetworkConfig(**ckpt["net_cfg"]) if "net_cfg" in ckpt else NetworkConfig()
    net = DeepNashNet(enc, net_cfg).to(device)
    state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    net.load_state_dict(state)
    net.eval()
    return net, enc
