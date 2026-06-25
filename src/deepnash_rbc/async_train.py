"""Persistent async actor/learner training (IMPALA / DeepNash topology).

Workers are spawned ONCE and run continuously: each holds a CPU copy of the
network, plays self-play games, and pushes finished trajectories into a shared
queue. The GPU learner pulls from the queue into its replay buffer and trains
without interruption, periodically broadcasting fresh weights back to the actors.

This decouples environment throughput (CPU-bound, parallel across many cores)
from the learner (GPU-bound). The actors necessarily play with slightly stale
weights -- that off-policy lag is exactly what v-trace's importance correction is
for, so it is principled here rather than a hack.

Why this and not `selfplay.parallel_self_play`: that helper rebuilds its process
pool every iteration (spawn + torch re-import each time), which dominates cost at
this scale. Here the pool is persistent and the learner never blocks on it.

Counters note: in async mode `total_iters`, `eval_every`, and `checkpoint_every`
count LEARNER STEPS (not outer iterations).

Run:  uv run deepnash-train-async   (or: python -m deepnash_rbc.async_train)
"""

from __future__ import annotations

import os
import queue
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
import torch.multiprocessing as mp

from .checkpoints import (
    checkpoint_path,
    ensure_version_config,
    find_latest_checkpoint,
    version_dir,
)
from .cli import config_from_args
from .config import Config
from .eval import evaluate
from .metrics import MetricsLogger
from .network import DeepNashNet
from .replay import ReplayBuffer
from .rnad.trainer import RNaDLearner
from .selfplay import play_one_game
from .train import resolve_device


# -- weight transport (numpy avoids torch shared-memory fd issues across procs) --
def _sd_to_numpy(net: DeepNashNet) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


def _load_numpy_sd(net: DeepNashNet, sd: Dict[str, np.ndarray]) -> None:
    net.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})


def _drain_latest(q: "mp.Queue"):
    """Return the most recent item in a maxsize-1 queue, discarding older ones."""
    item = None
    try:
        while True:
            item = q.get_nowait()
    except queue.Empty:
        pass
    return item


# -- checkpoint resume -------------------------------------------------------
def load_resume(path: str, learner: RNaDLearner, device: torch.device) -> int:
    """Restore a learner from a checkpoint, tolerating older (net+step only)
    formats. Returns the learner step to resume from."""
    ck = torch.load(path, map_location=device)
    learner.net.load_state_dict(ck["net"])
    step = int(ck.get("step", ck.get("iter", 0)))
    learner.steps = step

    if "opt" in ck:
        learner.opt.load_state_dict(ck["opt"])
    else:
        print("[resume] checkpoint has no optimizer state; Adam moments reset")
    if "reg_net" in ck:
        learner.reg_net.load_state_dict(ck["reg_net"])
    else:
        learner.reg_net.load_state_dict(learner.net.state_dict())
        print("[resume] checkpoint has no pi_reg; seeding it from the resumed net")
    learner.reg_net.eval()
    learner.iteration = int(
        ck.get("iteration", step // max(1, learner.cfg.rnad.iteration_steps))
    )
    return step


# -- actor process -----------------------------------------------------------
def actor_loop(actor_id, init_sd, cfg: Config, traj_q, weight_q, stop_event):
    torch.set_num_threads(1)  # one thread per actor: avoid oversubscription
    device = torch.device("cpu")
    net = DeepNashNet(cfg.encoding, cfg.network)
    _load_numpy_sd(net, init_sd)
    net.eval()

    while not stop_event.is_set():
        latest = _drain_latest(weight_q)
        if latest is not None:
            _load_numpy_sd(net, latest)
        for traj in play_one_game(net, device, cfg):
            if stop_event.is_set():
                break
            try:
                traj_q.put(traj, timeout=1.0)
            except queue.Full:
                pass  # learner is behind -> drop (backpressure), keep playing


# -- learner (main process) --------------------------------------------------
def run_async(cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    torch.manual_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    ctx = mp.get_context("spawn")

    net = DeepNashNet(cfg.encoding, cfg.network).to(device)
    learner = RNaDLearner(cfg, net, device)
    buffer = ReplayBuffer(cfg.train.buffer_capacity)
    metrics = MetricsLogger(cfg.train.metrics_path)
    os.makedirs(version_dir(cfg.train.checkpoint_dir), exist_ok=True)
    cfg_path = ensure_version_config(cfg.train.checkpoint_dir, cfg)
    print(f"[async] version config: {cfg_path}")

    # resume before snapshotting init weights so actors start from the resumed net
    start_step = 0
    if cfg.train.resume:
        path = cfg.train.resume
        if path == "auto":
            path = find_latest_checkpoint(cfg.train.checkpoint_dir)
            if path is None:
                raise RuntimeError(
                    "--resume: no deepnash_async_v*_*.pt found in "
                    f"{cfg.train.checkpoint_dir} for the current version"
                )
        start_step = load_resume(path, learner, device)
        print(f"[async] resumed from {path} at step {start_step}")

    n_actors = max(1, cfg.train.async_actors)
    traj_q = ctx.Queue(maxsize=cfg.train.traj_queue_size)
    weight_qs = [ctx.Queue(maxsize=1) for _ in range(n_actors)]
    stop_event = ctx.Event()

    init_sd = _sd_to_numpy(net)
    procs: List[mp.Process] = []
    for i in range(n_actors):
        p = ctx.Process(target=actor_loop,
                        args=(i, init_sd, cfg, traj_q, weight_qs[i], stop_event),
                        daemon=True)
        p.start()
        procs.append(p)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"[async] device={device} actors={n_actors} params={n_params:,}")

    step = start_step
    last: Dict = {}
    last_log = 0.0
    games_seen = 0
    warmup_start = time.time()
    try:
        while step < cfg.train.total_iters:
            # 1) drain finished trajectories into the buffer
            drained = 0
            while drained < cfg.train.drain_per_cycle:
                try:
                    buffer.add(traj_q.get_nowait())
                    drained += 1
                    games_seen += 1
                except queue.Empty:
                    break

            # 2) warmup gate (with deadlock detection)
            if len(buffer) < cfg.train.min_buffer_to_train:
                alive = [p.is_alive() for p in procs]
                if not any(alive):
                    codes = [p.exitcode for p in procs]
                    raise RuntimeError(f"all actors died before warmup (exitcodes={codes})")
                if games_seen == 0 and time.time() - warmup_start > 120:
                    raise RuntimeError("no trajectories after 120s; actors may be stuck")
                time.sleep(0.05)
                continue

            # 3) one learner step
            batch = buffer.sample(cfg.train.batch_trajectories)
            if batch:
                last = learner.update(batch)
                step = last["steps"]
                metrics.log({"step": step, "type": "train", "drained": drained,
                             "games_seen": games_seen, "buffer": len(buffer), **last})

            # 4) broadcast fresh weights to actors
            if step % cfg.train.weight_broadcast_every == 0:
                sd = _sd_to_numpy(net)
                for wq in weight_qs:
                    _drain_latest(wq)
                    try:
                        wq.put_nowait(sd)
                    except queue.Full:
                        pass

            # 5) skill eval (pauses training briefly; runs on the learner net)
            if cfg.train.eval_every and step > 0 and step % cfg.train.eval_every == 0:
                skill = evaluate(net, device, cfg)
                metrics.log({"step": step, "type": "skill", **skill})
                wr = " ".join(f"{k}={v}" for k, v in skill.items()
                              if k.startswith("vs_") and not k.endswith(("_draw", "_plies", "_n")))
                print(f"[eval step {step}] {wr}")

            # 6) checkpoint
            if step > 0 and step % cfg.train.checkpoint_every == 0:
                path = checkpoint_path(cfg.train.checkpoint_dir, step,
                                       prefix="deepnash_async")
                torch.save({"net": net.state_dict(), "step": step,
                            "opt": learner.opt.state_dict(),
                            "reg_net": learner.reg_net.state_dict(),
                            "iteration": learner.iteration,
                            "net_cfg": asdict(cfg.network), "enc_cfg": asdict(cfg.encoding)}, path)
                print(f"[async] saved {path}")

            if time.time() - last_log > 10:
                print(f"[async step {step}] buffer={len(buffer)} "
                      f"qsize~{_qsize(traj_q)} games_seen={games_seen} {last}")
                last_log = time.time()
    finally:
        stop_event.set()
        # terminate actors directly (they may be mid-game and not checking the
        # event); they are stateless self-play workers so this is safe.
        for p in procs:
            p.terminate()
        # avoid feeder-thread join deadlocks on the queues
        for q in [traj_q, *weight_qs]:
            try:
                q.cancel_join_thread()
            except Exception:
                pass
        for p in procs:
            p.join(timeout=3)
        print("[async] stopped")


def _qsize(q) -> int:
    try:
        return q.qsize()
    except (NotImplementedError, Exception):
        return -1


def main(cfg: Config | None = None) -> None:
    run_async(cfg or config_from_args(prog="deepnash-train-async"))


if __name__ == "__main__":
    main()
