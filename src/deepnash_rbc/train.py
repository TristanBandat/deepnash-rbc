"""Training entrypoint.

The loop, per iteration:
  1. collect self-play games with the current network (behavior policy)
  2. push trajectories into the replay buffer
  3. run a few R-NaD learner steps on batches sampled from the buffer
  4. periodically swap the regularization policy (handled inside the learner)
  5. checkpoint

Run:  uv run deepnash-train      (or: python -m deepnash_rbc.train)
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict

import torch

from .checkpoints import checkpoint_path, version_dir
from .cli import config_from_args
from .config import Config
from .eval import evaluate
from .metrics import MetricsLogger
from .network import DeepNashNet
from .replay import ReplayBuffer
from .rnad.trainer import RNaDLearner
from .selfplay import collect, parallel_self_play


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA not available -> using CPU")
        return torch.device("cpu")
    if requested == "cuda":
        # TF32 matmuls: big speedup on Ampere/Ada (L40S, RTX 30/40) at negligible
        # precision cost. No effect on Turing (RTX 2080) but harmless to set.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return torch.device(requested)


def main(cfg: Config | None = None) -> None:
    cfg = cfg or config_from_args(prog="deepnash-train")
    torch.manual_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    net = DeepNashNet(cfg.encoding, cfg.network).to(device)
    learner = RNaDLearner(cfg, net, device)
    buffer = ReplayBuffer(cfg.train.buffer_capacity)
    os.makedirs(version_dir(cfg.train.checkpoint_dir), exist_ok=True)
    metrics = MetricsLogger(cfg.train.metrics_path)

    print(f"[train] device={device} params={sum(p.numel() for p in net.parameters()):,}")

    for it in range(cfg.train.total_iters):
        t0 = time.time()
        if cfg.train.num_actors > 1:
            trajs = parallel_self_play(net, cfg, cfg.train.games_per_iter)
        else:
            trajs = collect(net, device, cfg, cfg.train.games_per_iter)
        for tr in trajs:
            buffer.add(tr)

        last = {}
        for _ in range(cfg.train.learner_steps_per_iter):
            batch = buffer.sample(cfg.train.batch_trajectories)
            if batch:
                last = learner.update(batch)

        if last:
            metrics.log({"iter": it, "type": "train", **last})

        if it % 10 == 0:
            dt = time.time() - t0
            print(f"[iter {it}] buffer={len(buffer)} {last} ({dt:.1f}s)")

        # ---- skill evaluation: the curve that should actually rise ----
        if cfg.train.eval_every and it > 0 and it % cfg.train.eval_every == 0:
            skill = evaluate(net, device, cfg)
            skill_row = {"iter": it, "type": "skill", **skill}
            metrics.log(skill_row)
            wr = " ".join(f"{k}={v}" for k, v in skill.items() if k.startswith("vs_")
                          and not k.endswith(("_draw", "_plies", "_n")))
            print(f"[eval {it}] {wr}")

        if it > 0 and it % cfg.train.checkpoint_every == 0:
            path = checkpoint_path(cfg.train.checkpoint_dir, it, prefix="deepnash")
            torch.save({"net": net.state_dict(), "iter": it,
                        "net_cfg": asdict(cfg.network), "enc_cfg": asdict(cfg.encoding)}, path)
            print(f"[train] saved {path}")


if __name__ == "__main__":
    main()
