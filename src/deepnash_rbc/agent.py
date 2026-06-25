"""The reconchess Player that drives self-play.

Wraps the network behind the reconchess callback interface. On each turn it:
  begin_turn -> (opponent capture ping) -> choose_sense -> sense_result
             -> choose_move -> move_result -> commit_turn
and records a Step for the sense choice and the move choice. Sampling is masked
to the legal actions reconchess provides, so we never emit an illegal action and
never need our own legal-move generator.

The same network instance is shared by both players in a self-play game; it is
queried under torch.no_grad in eval mode (the actor is the behavior policy).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import chess
import numpy as np
import torch
from reconchess import Player, WinReason
from reconchess.history import GameHistory

from .encoding.moves import build_move_index, move_to_index, PASS_INDEX
from .encoding.observation import ObservationEncoder
from .replay import MOVE, SENSE, Step, Trajectory


class RNaDPlayer(Player):
    def __init__(self, net, device: torch.device, history: int = 8, sample: bool = True,
                 track_belief: bool = False):
        self.net = net
        self.device = device
        self.encoder = ObservationEncoder(history=history)
        self.sample = sample
        self.trajectory = Trajectory()
        self.color = chess.WHITE
        self._last_requested: Optional[chess.Move] = None
        # Optional, off by default (zero cost to training/self-play): maintain a
        # constructed naive belief board for offline Stockfish cost-of-info
        # analysis. belief_fens[i] is the believed position at the i-th move
        # decision, aligned 1:1 with GameHistory.turns(color). See analysis/.
        self._track_belief = track_belief
        self.belief = None
        self.belief_fens: List[str] = []
        if track_belief:
            from .analysis.belief import BeliefBoardTracker
            self.belief = BeliefBoardTracker(self.color)

    # -- lifecycle -----------------------------------------------------------
    def handle_game_start(self, color: bool, board: chess.Board, opponent_name: str):
        self.color = color
        self.encoder.reset(color)
        self.trajectory = Trajectory()
        if self.belief is not None:
            self.belief.reset(color)
            self.belief_fens = []

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[int]):
        self.encoder.begin_turn()
        cap = capture_square if captured_my_piece else None
        self.encoder.opponent_captured_my_piece(cap)
        if self.belief is not None:
            self.belief.opponent_captured_my_piece(cap)

    # -- sense ---------------------------------------------------------------
    def choose_sense(self, sense_actions: List[int], move_actions: List[chess.Move],
                     seconds_left: float) -> Optional[int]:
        if not sense_actions:
            return None
        _, sense_logits, _ = self._forward()
        legal = np.asarray(sense_actions, dtype=np.int64)
        action, logp = self._sample_from(sense_logits, legal)
        self._record(SENSE, legal, action, logp)
        return int(action)

    def handle_sense_result(self, sense_result: List[Tuple[int, Optional[chess.Piece]]]):
        self.encoder.sense_result(sense_result)
        if self.belief is not None:
            self.belief.sense_result(sense_result)

    # -- move ----------------------------------------------------------------
    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # snapshot the believed board *before* we move -- this is what the agent
        # "thought" the position was when it decided, our-colour to move.
        if self.belief is not None:
            self.belief_fens.append(self.belief.fen())
        _, _, move_logits = self._forward()
        index_to_move = build_move_index(move_actions)
        legal_indices = list(index_to_move.keys()) + [PASS_INDEX]  # pass always allowed
        legal = np.asarray(sorted(set(legal_indices)), dtype=np.int64)
        action, logp = self._sample_from(move_logits, legal)
        self._record(MOVE, legal, action, logp)
        mv = index_to_move.get(int(action), None)  # None => pass
        self._last_requested = mv
        return mv

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[int]):
        self.encoder.move_result(requested_move, taken_move, captured_opponent_piece, capture_square)
        self.encoder.commit_turn()
        if self.belief is not None:
            self.belief.apply_my_move(taken_move)

    def handle_game_end(self, winner_color: Optional[bool], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        if winner_color is None:
            self.trajectory.z = 0.0
        else:
            self.trajectory.z = 1.0 if winner_color == self.color else -1.0

    # -- internals -----------------------------------------------------------
    def _forward(self):
        x = torch.from_numpy(self.encoder.tensor()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.net(x)

    def _sample_from(self, logits: torch.Tensor, legal: np.ndarray) -> Tuple[int, float]:
        logits = logits.squeeze(0).float().cpu()
        mask = torch.full_like(logits, float("-inf"))
        idx = torch.from_numpy(legal)
        mask[idx] = logits[idx]
        logp_all = torch.log_softmax(mask, dim=0)
        if self.sample:
            probs = logp_all.exp()
            action = int(torch.multinomial(probs, 1).item())
        else:
            action = int(torch.argmax(logp_all).item())
        return action, float(logp_all[action].item())

    def _record(self, head: int, legal: np.ndarray, action: int, logp: float):
        # planes are all binary -> store as uint8 (4x smaller than float32 for the
        # replay buffer and, crucially, for the actor->learner queue in async mode)
        self.trajectory.add(Step(
            obs=self.encoder.tensor().astype(np.uint8),
            head=head,
            legal=legal,
            action=action,
            behavior_logprob=logp,
        ))
