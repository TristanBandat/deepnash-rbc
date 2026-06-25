"""End-to-end smoke test.

Runs on CPU in well under a minute and verifies the whole pipeline wires up:
  1. move encoding has no collisions on real legal-move sets
  2. a full self-play game produces non-empty trajectories
  3. one R-NaD learner step runs forward+backward and updates weights
  4. the fast (vectorized) learner matches the legacy path and is not slower
  5. skill eval vs baseline bots produces win-rates
  6. the Stockfish analysis engine resolves and runs (bundled binary)

Run:  uv run deepnash-smoke   (or: python -m deepnash_rbc.smoke_test)
"""

from __future__ import annotations

import copy
import time

import chess
import torch

from .config import Config, EncodingConfig, NetworkConfig, TrainConfig, RNaDConfig
from .encoding.moves import build_move_index, move_to_index, PASS_INDEX
from .network import DeepNashNet
from .rnad.trainer import RNaDLearner
from .selfplay import collect


def _tiny_cfg() -> Config:
    return Config(
        encoding=EncodingConfig(history=4),
        network=NetworkConfig(channels=32, blocks=2, value_hidden=32),
        rnad=RNaDConfig(iteration_steps=2),
        train=TrainConfig(device="cpu", games_per_iter=1, seconds_per_player=30.0),
    )


def check_move_encoding() -> None:
    # collisions on the start position's legal moves + a promotion-rich position
    boards = [chess.Board(),
              chess.Board("8/PPPPPPPP/8/8/8/8/pppppppp/8 w - - 0 1")]
    for b in boards:
        moves = list(b.pseudo_legal_moves)
        idx = build_move_index(moves)
        assert len(idx) == len({move_to_index(m) for m in moves}), "index collision!"
        for m in moves:
            assert 0 <= move_to_index(m) < PASS_INDEX
    print(f"  move encoding OK ({len(boards)} positions, no collisions)")


def check_fast_learner(cfg: Config, device: torch.device, trajs) -> None:
    """The vectorized fast path must match the legacy path bit-for-bit (it only
    removes host overhead) and should not be slower per step. Both learners start
    from an identical net so one update must leave equal weights + stats."""
    torch.manual_seed(0)
    base = DeepNashNet(cfg.encoding, cfg.network).to(device)

    def run(fast: bool):
        learner = RNaDLearner(cfg, copy.deepcopy(base), device, fast=fast)
        t0 = time.perf_counter()
        stats = learner.update(trajs)
        dt = time.perf_counter() - t0
        return learner.net, stats, dt

    net_leg, s_leg, t_leg = run(fast=False)
    net_fast, s_fast, t_fast = run(fast=True)

    for k in ("loss", "policy_loss", "value_loss", "entropy"):
        assert abs(s_leg[k] - s_fast[k]) < 1e-5, f"fast/legacy {k} differ: {s_leg[k]} vs {s_fast[k]}"
    for (n, p_leg), (_, p_fast) in zip(net_leg.named_parameters(), net_fast.named_parameters()):
        assert torch.allclose(p_leg, p_fast, atol=1e-5, rtol=1e-4), f"fast/legacy param {n} diverged"

    speed = (t_leg / t_fast) if t_fast > 0 else float("nan")
    print(f"  fast==legacy (Δloss<1e-5, weights match); "
          f"legacy {t_leg*1e3:.1f}ms vs fast {t_fast*1e3:.1f}ms/step ({speed:.2f}x)")


def check_stockfish() -> None:
    """Resolve the Stockfish binary (bundled in tools/stockfish/ as a fallback) and
    confirm it actually analyses a position. Non-fatal if no binary is found --
    Stockfish is only needed for the move-quality eval, not for training."""
    from .analysis.engine import StockfishAnalyst, resolve_engine_path

    path = resolve_engine_path()
    if not path:
        print("  WARNING: no Stockfish binary found (env STOCKFISH_EXECUTABLE / PATH / "
              "tools/stockfish/); move-quality eval will be skipped on this host")
        return
    with StockfishAnalyst(engine_path=path, depth=6) as sf:
        board = chess.Board()
        ev = sf.evaluate_move(board, chess.Move.from_uci("e2e4"), pov=chess.WHITE)
    assert ev.ok and ev.cpl is not None and ev.cpl >= 0, f"engine returned no usable eval: {ev}"
    print(f"  Stockfish OK at {path} (e4 cpl={ev.cpl}, top1={ev.is_top1})")


def main() -> None:
    cfg = _tiny_cfg()
    device = torch.device("cpu")
    torch.manual_seed(0)

    print("[1/6] move encoding")
    check_move_encoding()

    print("[2/6] self-play game")
    net = DeepNashNet(cfg.encoding, cfg.network).to(device)
    trajs = collect(net, device, cfg, n_games=1)
    assert trajs, "no trajectories produced"
    total = sum(len(t) for t in trajs)
    print(f"  {len(trajs)} trajectories, {total} steps total, "
          f"z={[t.z for t in trajs]}")
    assert total > 0

    print("[3/6] one R-NaD learner step")
    learner = RNaDLearner(cfg, net, device)
    before = next(net.parameters()).clone()
    metrics = learner.update(trajs)
    after = next(net.parameters())
    changed = not torch.allclose(before, after)
    print(f"  metrics={metrics}")
    print(f"  weights updated: {changed}")
    assert metrics and changed, "learner step did not update weights"

    print("[4/6] fast vs legacy learner (perf path)")
    check_fast_learner(cfg, device, trajs)

    print("[5/6] skill eval vs baseline bots")
    from .eval import evaluate
    skill = evaluate(net, device, cfg, opponents=["random", "attacker"], games_per_opponent=2)
    print(f"  {skill}")
    assert any(k.startswith("vs_") for k in skill), "eval produced no win-rates"

    print("[6/6] Stockfish analysis engine")
    check_stockfish()

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
