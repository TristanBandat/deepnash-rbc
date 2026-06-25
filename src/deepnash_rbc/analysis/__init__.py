"""Stockfish-backed evaluation of the trained agent.

This package answers a different question than ``selfplay`` losses or even the
win-rate ``eval``: *given a perfect-information oracle, how good is the agent's
play, and how much does imperfect information cost it?*  RBC win-rate conflates
two skills -- tracking the hidden board (information skill) and choosing good
moves given a belief (chess skill).  Stockfish lets us separate them, because
the reconchess arbiter hands us the GROUND-TRUTH board at every ply via
``GameHistory`` -- information the agent itself never had.

Modules:
  engine.py        StockfishAnalyst: cached, illegal-tolerant position scoring.
  belief.py        BeliefBoardTracker: a constructed *naive* belief board (A).
  move_quality.py  replay a game, grade moves vs the oracle, aggregate.
  ladder.py        a tunable-strength Stockfish opponent for a win-rate curve.
"""
