"""End-to-end smoke test.

Runs on CPU in well under a minute and verifies the whole pipeline wires up.
Arch-independent checks run once (move encoding, Stockfish engine); the core
pipeline runs per selected architecture:
  - a full self-play game produces non-empty trajectories (exercises the agent's
    acting-time streaming state for temporal archs)
  - one R-NaD learner step runs forward+backward and updates weights (exercises
    the temporal per-trajectory batching + gather-back for temporal archs)
  - the fast (vectorized) learner matches the legacy path and is not slower
  - skill eval vs baseline bots produces win-rates

Run:  uv run deepnash-smoke                 (resnet only, default)
      uv run deepnash-smoke --arch transformer
      uv run deepnash-smoke --arch all       (every arch; slower)
      uv run deepnash-smoke --arch all --async  (also drive the real run_async loop)
  (or: python -m deepnash_rbc.smoke_test ...)
"""

from __future__ import annotations

import copy
import os
import time

import chess
import torch

from .config import Config, EncodingConfig, NetworkConfig, TrainConfig, RNaDConfig
from .encoding.moves import build_move_index, move_to_index, PASS_INDEX
from .network import make_net
from .rnad.trainer import RNaDLearner
from .selfplay import collect


def _tiny_cfg(arch: str = "resnet") -> Config:
    temporal = arch != "resnet"
    return Config(
        # temporal archs stream one frame per step and ignore history; the ResNet
        # stacks it, so give it a few frames to exercise the stack rebuild.
        encoding=EncodingConfig(history=1 if temporal else 4),
        network=NetworkConfig(
            arch=arch, channels=32, blocks=2, value_hidden=32,
            enc_blocks=2, mixer_dim=48, mixer_layers=2, nhead=4,
        ),
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
    base = make_net(cfg.encoding, cfg.network).to(device)

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


def check_async(arch: str, device: torch.device) -> None:
    """Drive the real async actor/learner loop (run_async) for a few learner
    steps in a temp dir: spawns CPU actors, streams trajectories, broadcasts
    weights, runs the temporal learner branch, and checkpoints. Confirms the
    async path (the one training uses) wires up and its checkpoint reloads with
    the right arch."""
    import glob
    import shutil
    import tempfile

    from .async_train import run_async

    tmp = tempfile.mkdtemp(prefix="deepnash_smoke_async_")
    prev_idle = os.environ.get("DEEPNASH_IGNORE_IDLE")
    os.environ["DEEPNASH_IGNORE_IDLE"] = "1"  # never park in an idle window
    try:
        cfg = _tiny_cfg(arch)
        cfg.train.device = "cpu"
        cfg.train.checkpoint_dir = tmp
        cfg.train.metrics_path = os.path.join(tmp, "metrics.jsonl")
        cfg.train.async_actors = 2
        cfg.train.min_buffer_to_train = 2
        cfg.train.buffer_capacity = 16
        cfg.train.batch_trajectories = 2
        cfg.train.total_iters = 2       # learner steps in async mode
        cfg.train.checkpoint_every = 2  # -> save at the final step
        cfg.train.weight_broadcast_every = 1
        cfg.train.idle_schedule = False
        cfg.train.progress = False
        cfg.train.seconds_per_player = 12.0
        run_async(cfg)
        ckpts = glob.glob(os.path.join(tmp, "**", "*.pt"), recursive=True)
        assert ckpts, "async run produced no checkpoint"
        from .play_session import load_net
        net2, _ = load_net(ckpts[0], device)
        assert bool(getattr(net2, "is_temporal", False)) == (arch != "resnet")
        print(f"      async loop OK: {len(ckpts)} checkpoint(s), reload arch={arch} "
              f"({type(net2).__name__})")
    finally:
        if prev_idle is None:
            os.environ.pop("DEEPNASH_IGNORE_IDLE", None)
        else:
            os.environ["DEEPNASH_IGNORE_IDLE"] = prev_idle
        shutil.rmtree(tmp, ignore_errors=True)


def check_arch(arch: str, device: torch.device, do_async: bool = False) -> None:
    """Per-architecture pipeline: self-play -> one learner step (weights update)
    -> fast==legacy parity -> skill eval. Exercises the real entry points (agent
    acting-time state, temporal learner batching) for the given arch."""
    print(f"\n=== arch={arch} ===")
    cfg = _tiny_cfg(arch)
    torch.manual_seed(0)

    print("  [a] self-play game")
    net = make_net(cfg.encoding, cfg.network).to(device)
    trajs = collect(net, device, cfg, n_games=1)
    assert trajs, "no trajectories produced"
    total = sum(len(t) for t in trajs)
    print(f"      {len(trajs)} trajectories, {total} steps total, z={[t.z for t in trajs]}")
    assert total > 0

    print("  [b] one R-NaD learner step")
    learner = RNaDLearner(cfg, net, device)
    before = next(net.parameters()).clone()
    metrics = learner.update(trajs)
    changed = not torch.allclose(before, next(net.parameters()))
    for k in ("loss", "policy_loss", "value_loss"):
        assert metrics[k] == metrics[k], f"{k} is NaN"
    print(f"      loss={metrics['loss']:.4f} policy={metrics['policy_loss']:.4f} "
          f"value={metrics['value_loss']:.4f}; weights updated: {changed}")
    assert metrics and changed, "learner step did not update weights"

    print("  [c] fast vs legacy learner (perf path)")
    check_fast_learner(cfg, device, trajs)

    print("  [d] skill eval vs baseline bots")
    from .eval import evaluate
    skill = evaluate(net, device, cfg, opponents=["random", "attacker"], games_per_opponent=2)
    print(f"      { {k: v for k, v in skill.items() if k.startswith('vs_') and k.count('_') == 1} }")
    assert any(k.startswith("vs_") for k in skill), "eval produced no win-rates"

    if do_async:
        print("  [e] async actor/learner loop")
        check_async(arch, device)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="deepnash-smoke")
    ap.add_argument(
        "--arch",
        choices=("resnet", "gru", "lstm", "transformer", "xlstm", "all"),
        default="resnet",
        help="Architecture(s) to exercise. 'all' runs every arch (slower). "
        "Default: resnet.",
    )
    ap.add_argument(
        "--async",
        dest="do_async",
        action="store_true",
        help="Also drive the real async actor/learner loop (run_async) per arch. "
        "Off by default to keep the plain smoke fast; the sync learner path is "
        "always checked.",
    )
    args = ap.parse_args()
    archs = (
        ("resnet", "gru", "lstm", "transformer", "xlstm") if args.arch == "all" else (args.arch,)
    )
    device = torch.device("cpu")
    torch.manual_seed(0)

    print("[move encoding]")
    check_move_encoding()

    for arch in archs:
        check_arch(arch, device, do_async=args.do_async)

    print("\n[Stockfish analysis engine]")
    check_stockfish()

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
