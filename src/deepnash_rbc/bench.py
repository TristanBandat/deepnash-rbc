"""Auto-tune the async trainer's actor count for this machine.

The only knob that thread/core count controls in async training is
``TrainConfig.async_actors``: each actor is a process pinned to one torch thread
(see ``async_train.actor_loop``), so #actors is effectively #cores spent on
self-play. Too few and the GPU learner trains on stale data (the actors can't
refresh the buffer fast enough); too many and you oversubscribe cores and the
trajectory queue saturates, so the surplus actors just drop games (wasted cores).

Rather than time the actors and the learner in isolation, this benchmark runs the
*real* async training loop -- the GPU learner and N CPU actors together -- for a
short window at each actor count. Running them together is the point: it captures
the actual GPU step time, the time the loop spends *waiting on data* vs. *busy on
the GPU*, the per-step cost of draining the queue and broadcasting weights, and
the trajectory-queue occupancy (empty => actor-bound, full => over-provisioned).
From those real timings it recommends the smallest actor count that keeps the
buffer fresh without saturating the queue.

Run:  uv run deepnash-bench                         (sweep up to the core count)
       uv run deepnash-bench --counts 16,32,56 --measure 30 --json bench.json
       uv run deepnash-bench --apply                (write the pick into config.py)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import time
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp

from . import config as config_mod
from .async_train import _drain_latest, _sd_to_numpy, actor_loop
from .config import Config
from .network import DeepNashNet
from .replay import ReplayBuffer
from .rnad.trainer import RNaDLearner
from .train import resolve_device


def _now() -> float:
    return time.perf_counter()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


# -- one real-training-loop measurement at a fixed actor count ----------------
def measure_training(
    cfg: Config,
    n_actors: int,
    warmup_s: float,
    measure_s: float,
    fast: Optional[bool] = None,
) -> Dict[str, float]:
    """Run the actual async loop (GPU learner + ``n_actors`` CPU actors) and time it.

    This is a faithful, instrumented copy of ``async_train.run_async``'s inner
    loop (minus eval/checkpoint): actors push trajectories into a bounded queue,
    the learner drains them into a replay buffer and runs GPU update steps, and
    fresh weights are broadcast periodically. We separate, in absolute terms:

      * ``gpu_step_ms``    -- forward+backward on the GPU, measured with cuda sync
      * ``data_wait_ms``   -- learner idle because the buffer hasn't warmed (the
                              GPU waiting on the CPU actors)
      * ``host_ms``        -- draining the queue, sampling, and weight broadcast
                              (CPU-side loop overhead that steals from the GPU)

    plus throughput (learner steps/s, trajectories/s) and the trajectory-queue
    occupancy that tells us whether actors are under- or over-provisioned.
    """
    ctx = mp.get_context("spawn")
    device = resolve_device(cfg.train.device)

    net = DeepNashNet(cfg.encoding, cfg.network).to(device)
    learner = RNaDLearner(cfg, net, device, fast=fast)
    buffer = ReplayBuffer(cfg.train.buffer_capacity)

    traj_q = ctx.Queue(maxsize=cfg.train.traj_queue_size)
    weight_qs = [ctx.Queue(maxsize=1) for _ in range(n_actors)]
    stop_event = ctx.Event()

    init_sd = _sd_to_numpy(net)
    procs: List[mp.Process] = []
    for i in range(n_actors):
        p = ctx.Process(
            target=actor_loop,
            args=(i, init_sd, cfg, traj_q, weight_qs[i], stop_event),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # accumulators (seconds), reset when the measurement window opens
    acc = {"gpu": 0.0, "wait": 0.0, "host": 0.0}
    steps = traj = q_samples = 0
    q_occ_sum = q_occ_max = 0.0
    qmax = float(cfg.train.traj_queue_size)
    cap = cfg.train.traj_queue_size

    def drain() -> int:
        n = 0
        while n < cfg.train.drain_per_cycle:
            try:
                buffer.add(traj_q.get_nowait())
                n += 1
            except queue.Empty:
                break
        return n

    measuring = False
    warm_deadline = _now() + warmup_s
    measure_start = warm_deadline
    win_end = warm_deadline + measure_s
    step_for_broadcast = 0
    try:
        while True:
            t = _now()
            if not measuring and t >= warm_deadline:
                measuring = True       # window opens; accumulators already zero
                measure_start = _now()
                win_end = measure_start + measure_s
            if measuring and t >= win_end:
                break
            if not any(p.is_alive() for p in procs):
                codes = [p.exitcode for p in procs]
                raise RuntimeError(f"all actors died (exitcodes={codes})")

            # 1) drain queue into buffer (host-side)
            t0 = _now()
            d = drain()
            occ = _qsize(traj_q)
            host = _now() - t0

            # 2) warmup gate: if the buffer hasn't filled, the GPU is idle waiting
            #    on the actors -- this is real CPU->GPU wait time.
            if len(buffer) < cfg.train.min_buffer_to_train:
                if measuring:
                    acc["host"] += host
                    acc["wait"] += 0.02
                time.sleep(0.02)
                continue

            # 3) sample (host) then one GPU learner step, timed with device sync
            t0 = _now()
            batch = buffer.sample(cfg.train.batch_trajectories)
            sample_host = _now() - t0
            _sync(device)              # flush any prior work so we time only this step
            tg = _now()
            if batch:
                learner.update(batch)
            _sync(device)
            gpu = _now() - tg

            # 4) periodic weight broadcast (host-side, includes state_dict copy)
            t0 = _now()
            step_for_broadcast += 1
            if step_for_broadcast % cfg.train.weight_broadcast_every == 0:
                sd = _sd_to_numpy(net)
                for wq in weight_qs:
                    _drain_latest(wq)
                    try:
                        wq.put_nowait(sd)
                    except queue.Full:
                        pass
            broadcast = _now() - t0

            if measuring:
                steps += 1
                traj += d
                acc["gpu"] += gpu
                acc["host"] += host + broadcast + sample_host
                q_samples += 1
                frac = occ / qmax if qmax else 0.0
                q_occ_sum += frac
                q_occ_max = max(q_occ_max, frac)
        wall = _now() - measure_start
    finally:
        stop_event.set()
        for p in procs:
            p.terminate()
        for q in [traj_q, *weight_qs]:
            try:
                q.cancel_join_thread()
            except Exception:
                pass
        for p in procs:
            p.join(timeout=3)

    wall = max(wall, 1e-9)
    steps = max(steps, 0)
    sps = steps / wall if wall else 0.0
    variant = (cfg.rnad.fast_learner if fast is None else fast)
    return {
        "actors": n_actors,
        "learner": "fast" if variant else "legacy",
        "device": str(device),
        "wall_s": round(wall, 2),
        "learner_steps": steps,
        "steps_per_s": round(sps, 3),
        "traj_per_s": round(traj / wall, 3),
        "fresh_per_step": round((traj / steps) if steps else 0.0, 3),
        "gpu_step_ms": round(1000 * acc["gpu"] / steps, 2) if steps else float("nan"),
        "gpu_busy_frac": round(acc["gpu"] / wall, 3),
        "data_wait_frac": round(acc["wait"] / wall, 3),
        "host_ms": round(1000 * acc["host"] / steps, 2) if steps else float("nan"),
        "q_occ_avg": round(q_occ_sum / q_samples, 3) if q_samples else 0.0,
        "q_occ_max": round(q_occ_max, 3),
        "q_cap": cap,
    }


def _qsize(q) -> int:
    try:
        return q.qsize()
    except (NotImplementedError, Exception):
        return 0


# -- sweep + recommendation --------------------------------------------------
def default_counts(max_actors: int) -> List[int]:
    """A coarse-to-fine ladder from a few actors up to the core count."""
    pts = {2, 4, 8}
    step = max(4, max_actors // 8)
    n = 8
    while n < max_actors:
        pts.add(n)
        n += step
    pts.add(max_actors)
    return sorted(c for c in pts if 1 <= c <= max_actors)


def recommend(rows: List[Dict[str, float]], fresh_target: float) -> Dict:
    """Choose the smallest actor count that keeps the buffer fresh without
    saturating the trajectory queue, using the real combined-loop timings.

    Signals (all from a real training window):
      * data_wait_frac high or queue chronically empty -> actor-bound: the GPU
        is waiting on the CPU, more actors help.
      * fresh_per_step >= target and queue not saturated -> enough actors; the
        smallest such count wins (extra cores would just be dropped games).
      * queue saturated (q_occ_max ~ 1) -> over-provisioned at that count.
    """
    rows = sorted(rows, key=lambda r: r["actors"])
    best_fresh = max(rows, key=lambda r: r["fresh_per_step"])

    # "enough" = refreshes the buffer at the target rate AND isn't already
    # saturating the queue (which would mean actors are dropping trajectories).
    enough = [r for r in rows
              if r["fresh_per_step"] >= fresh_target and r["q_occ_max"] < 0.95]
    if enough:
        choice = min(enough, key=lambda r: r["actors"])
        return {
            "async_actors": choice["actors"],
            "reason": (
                f"smallest actor count keeping the buffer fresh "
                f"({choice['fresh_per_step']} fresh traj/step >= {fresh_target}) while the GPU "
                f"runs {choice['steps_per_s']} steps/s at {choice['gpu_busy_frac']*100:.0f}% busy; "
                f"queue peaks at {choice['q_occ_max']*100:.0f}% so actors aren't dropping games. "
                f"More actors would mostly be idle/dropped cores."
            ),
            "actor_bound": False,
        }

    # Nothing hits the freshness target -> actor-bound. Take the knee of the
    # freshness curve (more cores still help until it flattens).
    knee = rows[0]
    for prev, cur in zip(rows, rows[1:]):
        base = max(prev["fresh_per_step"], 1e-9)
        if (cur["fresh_per_step"] - prev["fresh_per_step"]) / base < 0.08:
            knee = prev
            break
        knee = cur
    return {
        "async_actors": knee["actors"],
        "reason": (
            f"actor-bound: even at {best_fresh['actors']} actors the buffer only refreshes "
            f"{best_fresh['fresh_per_step']} fresh traj/step (< {fresh_target}) and the GPU waits "
            f"{best_fresh['data_wait_frac']*100:.0f}% of the time on data; production is the limit, "
            f"so use the knee at {knee['actors']} actors (more cores barely help past here)."
        ),
        "actor_bound": True,
    }


def run_compare(cfg: Config, counts: List[int], warmup_s: float, measure_s: float) -> Dict:
    """A/B the legacy vs fast learner on the real training loop at each actor
    count, reporting the throughput win. Same data path, same actors -- only the
    learner step changes -- so the steps/s ratio isolates the learner speedup."""
    cores = os.cpu_count() or 0
    device = resolve_device(cfg.train.device)
    print(f"[bench] COMPARE legacy vs fast learner  cores={cores} device={device} "
          f"counts={counts} warmup={warmup_s}s measure={measure_s}s")
    if device.type != "cuda":
        print("[bench] WARNING: no CUDA device -> learner runs on CPU; this validates "
              "the speedup direction but not the absolute L40S numbers.")

    rows: List[Dict[str, float]] = []
    print(f"\n{'actors':>6} {'legacy s/s':>11} {'fast s/s':>9} {'speedup':>8} "
          f"{'legacy ms':>10} {'fast ms':>8}")
    for c in counts:
        leg = measure_training(cfg, c, warmup_s, measure_s, fast=False)
        fas = measure_training(cfg, c, warmup_s, measure_s, fast=True)
        speed = (fas["steps_per_s"] / leg["steps_per_s"]) if leg["steps_per_s"] else float("nan")
        rows.extend([leg, fas])
        print(f"{c:>6} {leg['steps_per_s']:>11} {fas['steps_per_s']:>9} "
              f"{speed:>7.2f}x {leg['gpu_step_ms']:>10} {fas['gpu_step_ms']:>8}")
    return {"cores": cores, "device": str(device), "rows": rows, "compare": True}


def run_bench(cfg: Config, counts: List[int], warmup_s: float, measure_s: float,
              fresh_target: float) -> Dict:
    cores = os.cpu_count() or 0
    device = resolve_device(cfg.train.device)
    print(f"[bench] cores={cores} learner_device={device} counts={counts} "
          f"warmup={warmup_s}s measure={measure_s}s")
    if device.type != "cuda":
        print("[bench] WARNING: no CUDA device -> learner runs on CPU; absolute GPU "
              "timings will not reflect the L40S. Run this on the GPU server.")
    if any(c > cores for c in counts) and cores:
        print(f"[bench] note: counts above {cores} cores will oversubscribe (expected to show "
              "as a saturated queue / no extra freshness).")

    rows: List[Dict[str, float]] = []
    for c in counts:
        print(f"[bench] running real training loop with {c} actors ...", flush=True)
        row = measure_training(cfg, c, warmup_s, measure_s)
        rows.append(row)
        print(f"        steps/s={row['steps_per_s']}  gpu_step={row['gpu_step_ms']}ms "
              f"({row['gpu_busy_frac']*100:.0f}% busy, {row['data_wait_frac']*100:.0f}% waiting on data)  "
              f"host={row['host_ms']}ms/step  fresh/step={row['fresh_per_step']}  "
              f"queue avg/max={row['q_occ_avg']*100:.0f}/{row['q_occ_max']*100:.0f}%")
        if row["learner_steps"] == 0:
            print("        !! no learner steps in the window; raise --measure or --warmup")

    rec = recommend(rows, fresh_target)
    return {"cores": cores, "device": str(device), "rows": rows, "recommendation": rec}


# -- write the pick into config.py -------------------------------------------
def apply_to_config(n_actors: int) -> str:
    """Rewrite the ``async_actors`` default in config.py to ``n_actors`` in place.

    Edits the source file the running package was imported from (via the module's
    __file__), so it works regardless of CWD. Preserves the trailing comment.
    Returns the path written. Raises if the field can't be found unambiguously.
    """
    path = os.path.abspath(config_mod.__file__)
    with open(path) as f:
        src = f.read()
    pat = re.compile(r"^(?P<pre>\s*async_actors:\s*int\s*=\s*)\d+(?P<post>.*)$", re.MULTILINE)
    new_src, n = pat.subn(rf"\g<pre>{n_actors}\g<post>", src)
    if n != 1:
        raise RuntimeError(
            f"expected exactly one 'async_actors: int = ...' in {path}, found {n}; "
            f"not editing -- set --async-actors {n_actors} on the run instead"
        )
    with open(path, "w") as f:
        f.write(new_src)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deepnash-bench",
        description="Benchmark the real async training loop (GPU + actors) and recommend async_actors.",
    )
    p.add_argument("--counts", default=None,
                   help="Comma-separated actor counts to test (default: ladder up to core count).")
    p.add_argument("--max-actors", type=int, default=None,
                   help="Upper bound for the default ladder (default: os.cpu_count()).")
    p.add_argument("--warmup", type=float, default=12.0,
                   help="Warmup seconds per count: lets actors fill the buffer past the warmup "
                        "gate before timing (default: 12).")
    p.add_argument("--measure", type=float, default=25.0,
                   help="Measurement-window seconds per count (default: 25). Raise if games are long.")
    p.add_argument("--fresh-per-step", type=float, default=1.0,
                   help="Target fresh trajectories added to the buffer per GPU step; the pick is the "
                        "smallest actor count meeting it (default: 1.0).")
    p.add_argument("--device", default=None, help="Learner device override (cuda/cpu).")
    p.add_argument("--json", default=None, help="Write the full report to this path.")
    p.add_argument("--apply", action="store_true",
                   help="Rewrite the async_actors default in config.py to the recommendation.")
    p.add_argument("--compare", action="store_true",
                   help="A/B the legacy vs fast (vectorized) learner step at each count "
                        "and report the throughput speedup, instead of tuning actors.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config()
    if args.device is not None:
        cfg.train.device = args.device

    if args.counts:
        counts = sorted({int(x) for x in args.counts.split(",") if x.strip()})
    else:
        counts = default_counts(args.max_actors or os.cpu_count() or 8)

    if args.compare:
        report = run_compare(cfg, counts, args.warmup, args.measure)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(report, f, indent=2)
            print(f"[bench] wrote {args.json}")
        return

    report = run_bench(cfg, counts, args.warmup, args.measure, args.fresh_per_step)

    rec = report["recommendation"]
    print("\n=== recommendation ===")
    print(f"  async_actors = {rec['async_actors']}")
    print(f"  why: {rec['reason']}")
    print(f"  run: uv run deepnash-train-async --async-actors {rec['async_actors']}")

    if args.apply:
        old = Config().train.async_actors
        path = apply_to_config(rec["async_actors"])
        print(f"[bench] set async_actors {old} -> {rec['async_actors']} in {path}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[bench] wrote {args.json}")


if __name__ == "__main__":
    main()
