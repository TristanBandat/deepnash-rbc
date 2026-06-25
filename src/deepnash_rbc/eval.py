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
  trout    -- TroutBot: Stockfish-backed. Only used if a Stockfish binary is
              available (STOCKFISH_EXECUTABLE env var or `stockfish` on PATH).
              This is the meaningful "is it actually playing chess" bar.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List

import chess
import torch
from reconchess import LocalGame, play_local_game
from reconchess.bots.attacker_bot import AttackerBot
from reconchess.bots.random_bot import RandomBot

from .agent import RNaDPlayer
from .config import Config
from .network import DeepNashNet


def _stockfish_available() -> bool:
    return bool(os.environ.get("STOCKFISH_EXECUTABLE")) or shutil.which("stockfish") is not None


def _make_opponent(name: str):
    if name == "random":
        return RandomBot()
    if name == "attacker":
        return AttackerBot()
    if name == "trout":
        from reconchess.bots.trout_bot import TroutBot
        return TroutBot()
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
    if "trout" in opponents and not _stockfish_available():
        opponents = [o for o in opponents if o != "trout"]  # silently skip if no engine
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
