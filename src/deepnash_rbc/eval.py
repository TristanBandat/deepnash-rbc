"""Skill evaluation against fixed baseline opponents.

This is the metric that actually measures whether the agent is getting better at
RBC (unlike the self-play losses, which are relative to a moving target). We play
the current network -- GREEDILY (sample=False), so evaluation is deterministic
given the opponent -- against reconchess's bundled bots, as both White and Black,
and report win/draw/loss rates plus average game length.

Baselines:
  random   -- RandomBot: random legal sense + move. The floor; should be crushed
              quickly once anything is learned.
  attacker -- AttackerBot: makes a beeline for the king. Weak but not trivial.
  trout    -- TroutBot: Stockfish-backed. Used whenever a Stockfish binary can be
              resolved -- STOCKFISH_EXECUTABLE, `stockfish` on PATH, or the copy
              bundled in tools/stockfish/ -- which is then exported as
              STOCKFISH_EXECUTABLE so reconchess's TroutBot picks it up.
              This is the meaningful "is it actually playing chess" bar.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

import chess
import torch
from reconchess import LocalGame, play_local_game
from reconchess.bots.attacker_bot import AttackerBot
from reconchess.bots.random_bot import RandomBot

from .agent import RNaDPlayer
from .config import Config
from .network import DeepNashNet


def _ensure_stockfish() -> str | None:
    """Resolve a Stockfish binary (env var, PATH, or the bundled tools/stockfish/)
    and export STOCKFISH_EXECUTABLE so reconchess's TroutBot finds it. Returns the
    path, or None if no engine is available. A user-set env var is left untouched."""
    from .analysis.engine import STOCKFISH_ENV_VAR, resolve_engine_path

    path = resolve_engine_path()
    if path:
        os.environ.setdefault(STOCKFISH_ENV_VAR, path)
    return path


def _make_trout():
    """Build a TroutBot that never hands Stockfish an illegal position.

    RBC belief boards are frequently illegal as standard chess (missing enemy
    king, side-not-to-move left in check, ...). Stockfish does not validate its
    input and *segfaults* on many such positions, killing the engine mid-eval.
    reconchess's TroutBot only guards the king-capture case, so we subclass it to
    add the same ``board.is_valid()`` gate that StockfishAnalyst already relies on
    (see analysis/engine.py); on an illegal board we fall back to a random legal
    move instead of feeding it to the engine."""
    from reconchess.bots.trout_bot import TroutBot

    class SafeTroutBot(TroutBot):
        def choose_move(self, move_actions, seconds_left):
            # keep reconchess's king-capture shortcut
            enemy_king = self.board.king(not self.color)
            if enemy_king:
                attackers = self.board.attackers(self.color, enemy_king)
                if attackers:
                    return chess.Move(attackers.pop(), enemy_king)
            # TroutBot sets turn/clear_stack right before engine.play; mirror that
            # so is_valid() checks the exact position we'd otherwise send.
            self.board.turn = self.color
            self.board.clear_stack()
            if not self.board.is_valid():
                return random.choice(move_actions) if move_actions else None
            return super().choose_move(move_actions, seconds_left)

    return SafeTroutBot()


def _make_opponent(name: str):
    if name == "random":
        return RandomBot()
    if name == "attacker":
        return AttackerBot()
    if name == "trout":
        return _make_trout()
    raise ValueError(f"unknown opponent: {name}")


def evaluate(
    net: DeepNashNet,
    device: torch.device,
    cfg: Config,
    opponents: List[str] | None = None,
    games_per_opponent: int | None = None,
) -> Dict[str, float]:
    net.eval()
    opponents = list(opponents or cfg.train.eval_opponents)
    if "trout" in opponents and not _ensure_stockfish():
        opponents = [o for o in opponents if o != "trout"]  # skip only if no engine
    n = games_per_opponent or cfg.train.eval_games
    history = cfg.encoding.history

    results: Dict[str, float] = {}
    for opp_name in opponents:
        wins = draws = losses = 0
        total_plies = 0
        played = 0
        for g in range(n):
            agent = RNaDPlayer(net, device, history=history, sample=False)  # greedy
            try:
                opponent = _make_opponent(opp_name)
            except Exception as e:
                print(f"[eval] cannot build opponent {opp_name}: {e}")
                break
            agent_is_white = (g % 2 == 0)  # alternate colors
            white, black = (agent, opponent) if agent_is_white else (opponent, agent)
            game = LocalGame(seconds_per_player=cfg.train.seconds_per_player)
            try:
                winner, _reason, hist = play_local_game(white, black, game=game)
            except Exception as e:
                print(f"[eval] game vs {opp_name} aborted: {type(e).__name__}: {e}")
                continue
            played += 1
            total_plies += _num_moves(hist)
            agent_color = chess.WHITE if agent_is_white else chess.BLACK
            if winner is None:
                draws += 1
            elif winner == agent_color:
                wins += 1
            else:
                losses += 1
        if played:
            results[f"vs_{opp_name}"] = round(wins / played, 4)
            results[f"vs_{opp_name}_draw"] = round(draws / played, 4)
            results[f"vs_{opp_name}_plies"] = round(total_plies / played, 1)
            results[f"vs_{opp_name}_n"] = played
    return results


def _num_moves(history) -> int:
    try:
        return int(history.num_turns())
    except Exception:
        return 0
