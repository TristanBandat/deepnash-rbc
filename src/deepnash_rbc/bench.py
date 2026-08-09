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

Two things decide whether the answer is useful:

**Benchmark the right model.** Pass ``--version``/``--config`` to load a run's own
``config.json``. Without it the *library defaults* are measured -- a different
architecture from anything you are training, so every number is about a model
that does not exist in your campaign.

**Optimize for the right thing.** Once the actors out-run the learner the buffer
is permanently full and extra trajectories do not make training step faster --
so a pure-throughput objective calls them waste and tells you to cut cores. But
they are still buying something: a shorter *off-policy horizon*
(``buffer_capacity / fresh_per_step``, the learner steps of policy drift spanned
by the buffer, linear in actor throughput) and lower sample reuse. If that
freshness is what you want the cores for, pass ``--max-horizon`` and the pick
becomes the cheapest config reaching it. ``buffer_capacity`` is the other lever
on the same quantity and costs no cores -- sweep it with ``--set``.

Run:  uv run deepnash-bench --version 0.59.0 --counts 8,16,30,48 \
          --actor-games 1,2,4 --max-horizon 150 --json bench.json
      uv run deepnash-bench --version 0.59.0 --set train.buffer_capacity=2048 ...
      uv run deepnash-bench --apply          (write the pick into config.py)
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
from .async_train import BatchPrefetcher, actor_loop
from .checkpoints import existing_versions, read_version_config
from .cli import _apply_overrides
from .config import Config, config_from_dict
from .network import make_net
from .replay import ReplayBuffer
from .rnad.trainer import RNaDLearner
from .train import resolve_device
from .weights import WeightBus


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
    loop (minus eval/checkpoint/idle-schedule): actors push trajectories into a
    bounded queue, the learner drains them into a replay buffer, batches are
    sampled + collated on the prefetch thread (when ``prefetch_depth > 0``,
    matching training), and fresh weights are broadcast periodically. We
    separate, in absolute terms:

      * ``gpu_step_ms``    -- forward+backward on the GPU, measured with cuda sync
      * ``data_wait_ms``   -- learner idle waiting on data: buffer warmup plus
                              time blocked on the prefetch queue
      * ``host_ms``        -- draining the queue, weight broadcast, and (only
                              with prefetch disabled) inline sample+collate
                              (CPU-side loop overhead that steals from the GPU)

    plus throughput (learner steps/s, trajectories/s) and the trajectory-queue
    occupancy that tells us whether actors are under- or over-provisioned.
    """
    ctx = mp.get_context("spawn")
    device = resolve_device(cfg.train.device)

    net = make_net(cfg.encoding, cfg.network).to(device)
    learner = RNaDLearner(cfg, net, device, fast=fast)
    buffer = ReplayBuffer(cfg.train.buffer_capacity)

    traj_q = ctx.Queue(maxsize=cfg.train.traj_queue_size)
    stop_event = ctx.Event()
    pause_event = ctx.Event()  # never set here; actor_loop requires it

    bus = WeightBus.create(net, ctx)
    bus.publish(net)
    procs: List[mp.Process] = []
    for i in range(n_actors):
        p = ctx.Process(
            target=actor_loop,
            args=(i, bus, cfg, traj_q, stop_event, pause_event),
            daemon=True,
        )
        p.start()
        procs.append(p)

    prefetcher = (
        BatchPrefetcher(learner, buffer, cfg.train.batch_trajectories,
                        cfg.train.min_buffer_to_train, cfg.train.prefetch_depth)
        if cfg.train.prefetch_depth > 0 else None
    )

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

            # 3) next batch: prefetched (sample+collate overlapped the previous
            #    GPU step, matching run_async) or built inline when disabled.
            #    Time blocked on the prefetch queue as data wait: the GPU is
            #    starved there, whether by actors or by collation throughput.
            t0 = _now()
            if prefetcher is not None:
                col = prefetcher.get(timeout=1.0)
                fetch_wait = _now() - t0
                sample_host = 0.0
            else:
                col = learner.collate(buffer.sample(cfg.train.batch_trajectories))
                fetch_wait = 0.0
                sample_host = _now() - t0
            _sync(device)              # flush any prior work so we time only this step
            tg = _now()
            if col is not None:
                learner.update_collated(col)
            _sync(device)
            gpu = _now() - tg

            # 4) periodic weight broadcast (host-side, includes state_dict copy)
            t0 = _now()
            step_for_broadcast += 1
            if step_for_broadcast % cfg.train.weight_broadcast_every == 0:
                bus.publish(net)
            broadcast = _now() - t0

            if measuring:
                if col is not None:
                    steps += 1
                traj += d
                acc["gpu"] += gpu
                acc["wait"] += fetch_wait
                acc["host"] += host + broadcast + sample_host
                q_samples += 1
                frac = occ / qmax if qmax else 0.0
                q_occ_sum += frac
                q_occ_max = max(q_occ_max, frac)
        wall = _now() - measure_start
    finally:
        if prefetcher is not None:
            prefetcher.stop()
        stop_event.set()
        for p in procs:
            p.terminate()
        try:
            traj_q.cancel_join_thread()
        except Exception:
            pass
        for p in procs:
            p.join(timeout=3)

    wall = max(wall, 1e-9)
    steps = max(steps, 0)
    sps = steps / wall if wall else 0.0
    tps = traj / wall
    fresh = (traj / steps) if steps else 0.0
    variant = (cfg.rnad.fast_learner if fast is None else fast)
    return {
        "actors": n_actors,
        "actor_games": max(1, cfg.train.actor_games),
        "concurrent_games": n_actors * max(1, cfg.train.actor_games),
        "learner": "fast" if variant else "legacy",
        "prefetch_depth": cfg.train.prefetch_depth,
        "device": str(device),
        "wall_s": round(wall, 2),
        "learner_steps": steps,
        "steps_per_s": round(sps, 3),
        "traj_per_s": round(tps, 3),
        "fresh_per_step": round(fresh, 3),
        # --- how off-policy the learner's data is (see off_policy_metrics) ---
        "horizon_steps": round(_horizon(cfg, fresh), 1),
        "reuse": round(_reuse(cfg, sps, tps), 2),
        "buffer_capacity": cfg.train.buffer_capacity,
        "batch_trajectories": cfg.train.batch_trajectories,
        "gpu_step_ms": round(1000 * acc["gpu"] / steps, 2) if steps else float("nan"),
        "gpu_busy_frac": round(acc["gpu"] / wall, 3),
        "data_wait_frac": round(acc["wait"] / wall, 3),
        "host_ms": round(1000 * acc["host"] / steps, 2) if steps else float("nan"),
        "q_occ_avg": round(q_occ_sum / q_samples, 3) if q_samples else 0.0,
        "q_occ_max": round(q_occ_max, 3),
        "q_cap": cap,
    }


# -- off-policy metrics -------------------------------------------------------
# Throughput alone is the wrong objective once the actors out-run the learner:
# the surplus does not train the net faster, it makes the net's data *fresher*.
# Both quantities below are linear in actor throughput, which is what makes the
# actor knobs worth tuning even on a run whose buffer is permanently full.
#
#   horizon_steps -- learner steps of policy drift spanned by the buffer,
#                    buffer_capacity / fresh_per_step. The oldest trajectory the
#                    learner samples was generated by a policy this many steps
#                    behind the current one; it is the off-policy distance
#                    v-trace has to correct for.
#   reuse         -- expected times a trajectory is sampled before it is evicted,
#                    (batch_trajectories * steps/s) / trajectories/s.
#
# ``buffer_capacity`` is the *other* lever on horizon_steps and costs no cores at
# all -- shrinking it tightens the horizon but also narrows data diversity, so
# the sweep exposes both axes rather than assuming which one you want to spend.
def _horizon(cfg: Config, fresh_per_step: float) -> float:
    if fresh_per_step <= 0:
        return float("inf")
    return cfg.train.buffer_capacity / fresh_per_step


def _reuse(cfg: Config, steps_per_s: float, traj_per_s: float) -> float:
    if traj_per_s <= 0:
        return float("inf")
    return cfg.train.batch_trajectories * steps_per_s / traj_per_s


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


def _cost(row: Dict[str, float]) -> tuple:
    """Machine cost of a config: actor processes first (that is what eats cores),
    then total concurrent games (memory and queue churn)."""
    return (row["actors"], row["concurrent_games"])


def recommend(rows: List[Dict[str, float]], fresh_target: float,
              horizon_target: Optional[float] = None) -> Dict:
    """Pick the cheapest actor config that hits the data-quality targets.

    Two different objectives live here, and which one applies depends on whether
    the actors can out-run the learner:

    * **Actor-bound** (nothing reaches ``fresh_target``): the GPU is idling on
      data. Throughput is the objective -- take the knee of the freshness curve,
      past which more cores stop buying trajectories.

    * **Learner-bound** (the usual case here): the buffer is permanently full and
      extra trajectories do not make the learner step faster. What they *do* buy
      is a shorter off-policy horizon -- ``buffer_capacity / fresh_per_step``
      learner steps of policy drift across the buffer, linear in actor
      throughput. So the objective becomes: hit ``horizon_target`` at the lowest
      core cost. Without a horizon target this degrades to the old behavior
      (cheapest config that merely keeps up), which spends the surplus on
      nothing -- pass ``--max-horizon`` when data freshness is what you care about.
    """
    rows = sorted(rows, key=_cost)
    best_fresh = max(rows, key=lambda r: r["fresh_per_step"])

    def viable(r):
        if r["fresh_per_step"] < fresh_target or r["q_occ_max"] >= 0.95:
            return False
        return horizon_target is None or r["horizon_steps"] <= horizon_target

    enough = [r for r in rows if viable(r)]
    if enough:
        choice = min(enough, key=_cost)
        goal = (f"horizon {choice['horizon_steps']:.0f} <= {horizon_target:.0f} steps, "
                if horizon_target is not None else "")
        return {
            "async_actors": choice["actors"],
            "actor_games": choice["actor_games"],
            "reason": (
                f"cheapest config meeting the targets ({goal}"
                f"{choice['fresh_per_step']} fresh traj/step >= {fresh_target}) at "
                f"{choice['actors']} processes x {choice['actor_games']} games "
                f"({choice['concurrent_games']} concurrent). Learner runs "
                f"{choice['steps_per_s']} steps/s, {choice['gpu_busy_frac']*100:.0f}% GPU-busy, "
                f"queue peaks at {choice['q_occ_max']*100:.0f}%; reuse {choice['reuse']}x."
            ),
            "actor_bound": False,
            "horizon_steps": choice["horizon_steps"],
            "reuse": choice["reuse"],
        }

    # Nothing meets the targets. If even freshness is unmet we are actor-bound;
    # if only the horizon is unmet, more actors still help -- both point at the
    # config with the most production, so report the best achievable.
    unmet_fresh = best_fresh["fresh_per_step"] < fresh_target
    best_horizon = min(rows, key=lambda r: r["horizon_steps"])
    if unmet_fresh:
        knee = rows[0]
        by_actors = sorted(rows, key=lambda r: r["actors"])
        for prev, cur in zip(by_actors, by_actors[1:]):
            base = max(prev["fresh_per_step"], 1e-9)
            if (cur["fresh_per_step"] - prev["fresh_per_step"]) / base < 0.08:
                knee = prev
                break
            knee = cur
        return {
            "async_actors": knee["actors"],
            "actor_games": knee["actor_games"],
            "reason": (
                f"actor-bound: even at {best_fresh['actors']}x{best_fresh['actor_games']} the buffer "
                f"only refreshes {best_fresh['fresh_per_step']} fresh traj/step (< {fresh_target}) "
                f"and the GPU waits {best_fresh['data_wait_frac']*100:.0f}% of the time on data; "
                f"production is the limit, so use the knee at {knee['actors']}x{knee['actor_games']} "
                f"(more cores barely help past here)."
            ),
            "actor_bound": True,
            "horizon_steps": knee["horizon_steps"],
            "reuse": knee["reuse"],
        }
    return {
        "async_actors": best_horizon["actors"],
        "actor_games": best_horizon["actor_games"],
        "reason": (
            f"no config reached the {horizon_target:.0f}-step horizon target; the best is "
            f"{best_horizon['horizon_steps']:.0f} steps at "
            f"{best_horizon['actors']}x{best_horizon['actor_games']} (reuse {best_horizon['reuse']}x). "
            f"Actor throughput is one lever on the horizon; buffer_capacity is the other and "
            f"costs no cores -- try --set train.buffer_capacity="
            f"{int(best_horizon['buffer_capacity'] * horizon_target / max(best_horizon['horizon_steps'], 1e-9))}."
        ),
        "actor_bound": False,
        "horizon_steps": best_horizon["horizon_steps"],
        "reuse": best_horizon["reuse"],
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


def describe_config(cfg: Config) -> str:
    """One line identifying what is actually being benchmarked."""
    net = cfg.network
    arch = net.arch
    shape = (f"ch={net.channels} blocks={net.blocks} history={cfg.encoding.history}"
             if arch == "resnet" else
             f"enc={net.enc_blocks} mixer={net.mixer_dim}x{net.mixer_layers} heads={net.nhead}")
    return (f"arch={arch} {shape} batch={cfg.train.batch_trajectories} "
            f"buffer={cfg.train.buffer_capacity} amp={cfg.rnad.amp}")


def run_bench(cfg: Config, counts: List[int], warmup_s: float, measure_s: float,
              fresh_target: float, games_counts: Optional[List[int]] = None,
              horizon_target: Optional[float] = None) -> Dict:
    cores = os.cpu_count() or 0
    device = resolve_device(cfg.train.device)
    games_counts = games_counts or [max(1, cfg.train.actor_games)]
    print(f"[bench] cores={cores} learner_device={device} counts={counts} "
          f"actor_games={games_counts} warmup={warmup_s}s measure={measure_s}s")
    print(f"[bench] config: {describe_config(cfg)}")
    if horizon_target is not None:
        print(f"[bench] targets: fresh/step >= {fresh_target}, "
              f"off-policy horizon <= {horizon_target:.0f} learner steps")
    if device.type != "cuda":
        print("[bench] WARNING: no CUDA device -> learner runs on CPU; absolute GPU "
              "timings will not reflect the L40S. Run this on the GPU server.")
    if any(c > cores for c in counts) and cores:
        print(f"[bench] note: counts above {cores} cores will oversubscribe (expected to show "
              "as a saturated queue / no extra freshness).")

    rows: List[Dict[str, float]] = []
    for games in games_counts:
        cfg.train.actor_games = games
        for c in counts:
            print(f"[bench] running real training loop with {c} actors "
                  f"x {games} games ...", flush=True)
            row = measure_training(cfg, c, warmup_s, measure_s)
            rows.append(row)
            print(f"        steps/s={row['steps_per_s']}  gpu_step={row['gpu_step_ms']}ms "
                  f"({row['gpu_busy_frac']*100:.0f}% busy, {row['data_wait_frac']*100:.0f}% waiting on data)  "
                  f"host={row['host_ms']}ms/step  traj/s={row['traj_per_s']}  "
                  f"fresh/step={row['fresh_per_step']}  horizon={row['horizon_steps']:.0f} steps  "
                  f"reuse={row['reuse']}x  "
                  f"queue avg/max={row['q_occ_avg']*100:.0f}/{row['q_occ_max']*100:.0f}%")
            if row["learner_steps"] == 0:
                print("        !! no learner steps in the window; raise --measure or --warmup")

    print_grid(rows, cores)
    rec = recommend(rows, fresh_target, horizon_target)
    return {"cores": cores, "device": str(device), "rows": rows,
            "config": describe_config(cfg), "recommendation": rec}


def print_grid(rows: List[Dict[str, float]], cores: int) -> None:
    """The whole grid in one table, so the trade-off is visible, not just the pick."""
    print(f"\n=== grid ({len(rows)} configs) ===")
    print(f"{'actors':>6} {'games':>6} {'conc':>5} {'cores%':>7} {'steps/s':>8} "
          f"{'traj/s':>8} {'fresh':>7} {'horizon':>8} {'reuse':>6} {'gpu%':>5} "
          f"{'wait%':>6} {'queue%':>7}  verdict")
    for r in sorted(rows, key=_cost):
        pct = 100 * r["actors"] / cores if cores else float("nan")
        if r["data_wait_frac"] > 0.25:
            verdict = "actor-bound (GPU idle on data)"
        elif r["q_occ_max"] >= 0.95:
            verdict = "queue saturated (dropping games)"
        elif r["fresh_per_step"] > 4:
            verdict = "learner-bound (surplus -> freshness)"
        else:
            verdict = "balanced"
        print(f"{r['actors']:>6} {r['actor_games']:>6} {r['concurrent_games']:>5} "
              f"{pct:>6.0f}% {r['steps_per_s']:>8.2f} {r['traj_per_s']:>8.2f} "
              f"{r['fresh_per_step']:>7.2f} {r['horizon_steps']:>8.0f} {r['reuse']:>5.1f}x "
              f"{r['gpu_busy_frac']*100:>4.0f}% {r['data_wait_frac']*100:>5.0f}% "
              f"{r['q_occ_max']*100:>6.0f}%  {verdict}")


# -- write the pick into config.py -------------------------------------------
def apply_to_config(n_actors: int, n_games: int = 1) -> str:
    """Rewrite the ``async_actors`` / ``actor_games`` defaults in config.py in place.

    Edits the source file the running package was imported from (via the module's
    __file__), so it works regardless of CWD. Preserves the trailing comments.
    Returns the path written. Raises if a field can't be found unambiguously,
    leaving the file untouched.
    """
    path = os.path.abspath(config_mod.__file__)
    with open(path) as f:
        src = f.read()
    for name, value in (("async_actors", n_actors), ("actor_games", n_games)):
        pat = re.compile(rf"^(?P<pre>\s*{name}:\s*int\s*=\s*)\d+(?P<post>.*)$", re.MULTILINE)
        src, n = pat.subn(rf"\g<pre>{value}\g<post>", src)
        if n != 1:
            raise RuntimeError(
                f"expected exactly one '{name}: int = ...' in {path}, found {n}; "
                f"not editing -- pass --async-actors {n_actors} --actor-games {n_games} "
                f"on the run instead"
            )
    with open(path, "w") as f:
        f.write(src)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deepnash-bench",
        description="Benchmark the real async training loop (GPU + actors) and recommend "
                    "async_actors / actor_games for a given run config.",
        epilog="Tune the config a run actually uses, e.g.:\n"
               "  deepnash-bench --version 0.59.0 --counts 8,16,30,48 --actor-games 1,2,4 "
               "--max-horizon 150\n"
               "Benchmarking without --version/--config measures the library defaults, "
               "which are almost certainly not your run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", default=None,
                   help="Benchmark the config of an existing run, read from "
                        "<checkpoint-dir>/v<VERSION>/config.json (e.g. --version 0.59.0). "
                        "Without this the library defaults are measured, which usually "
                        "means a different architecture than the run you care about.")
    p.add_argument("--config", default=None,
                   help="Path to a config.json manifest to benchmark (alternative to --version).")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Where to look for v<VERSION>/config.json (default: checkpoints).")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE",
                   help="Override any config field by dotted path after loading, e.g. "
                        "--set train.buffer_capacity=2048. Repeatable. Useful for sweeping "
                        "the non-actor lever on the off-policy horizon.")
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
                   help="Minimum fresh trajectories added to the buffer per learner step "
                        "(default: 1.0). Below this the GPU is starved.")
    p.add_argument("--max-horizon", type=float, default=None,
                   help="Target off-policy horizon in LEARNER STEPS: buffer_capacity / "
                        "fresh_per_step, i.e. how far the policy drifts across the buffer. "
                        "Set this when data freshness (not just keeping the GPU fed) is what "
                        "you are buying with cores -- the pick becomes the cheapest config "
                        "reaching it. Without it, surplus actor throughput is treated as waste.")
    p.add_argument("--actor-games", default=None,
                   help="Comma-separated concurrent-games-per-actor values to sweep "
                        "(TrainConfig.actor_games; default: just the config value). "
                        "Total concurrent games is actors x games, so e.g. "
                        "--counts 16,30 --actor-games 1,4,8 explores the whole grid.")
    p.add_argument("--device", default=None, help="Learner device override (cuda/cpu).")
    p.add_argument("--json", default=None, help="Write the full report to this path.")
    p.add_argument("--apply", action="store_true",
                   help="Rewrite the async_actors and actor_games defaults in config.py to "
                        "the recommendation.")
    p.add_argument("--compare", action="store_true",
                   help="A/B the legacy vs fast (vectorized) learner step at each count "
                        "and report the throughput speedup, instead of tuning actors.")
    return p


def load_bench_config(args) -> Config:
    """Build the Config to benchmark: a run's manifest if given, else defaults."""
    if args.config and args.version:
        raise SystemExit("pass --config or --version, not both")
    if args.config:
        with open(args.config) as f:
            cfg = config_from_dict(json.load(f))
        print(f"[bench] loaded config from {args.config}")
    elif args.version:
        data = read_version_config(args.checkpoint_dir, args.version)
        if data is None:
            raise SystemExit(
                f"no config.json for version {args.version} under {args.checkpoint_dir}/; "
                f"available: {', '.join(existing_versions(args.checkpoint_dir)) or 'none'}"
            )
        cfg = config_from_dict(data)
        print(f"[bench] loaded config for v{args.version}")
    else:
        cfg = Config()
        print("[bench] NOTE: benchmarking library defaults -- pass --version/--config to "
              "measure the model your run actually trains.")
    if args.overrides:
        _apply_overrides(cfg, args.overrides)
        print(f"[bench] overrides: {', '.join(args.overrides)}")
    if args.device is not None:
        cfg.train.device = args.device
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_bench_config(args)

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

    games_counts = (
        sorted({int(x) for x in args.actor_games.split(",") if x.strip()})
        if args.actor_games else None
    )
    report = run_bench(cfg, counts, args.warmup, args.measure, args.fresh_per_step,
                       games_counts, args.max_horizon)

    rec = report["recommendation"]
    print("\n=== recommendation ===")
    print(f"  async_actors = {rec['async_actors']}   actor_games = {rec['actor_games']}")
    print(f"  off-policy horizon = {rec['horizon_steps']:.0f} learner steps, "
          f"reuse = {rec['reuse']}x")
    print(f"  why: {rec['reason']}")
    print(f"  run: uv run deepnash-train-async --async-actors {rec['async_actors']} "
          f"--actor-games {rec['actor_games']}")
    if args.max_horizon is None:
        print("  note: no --max-horizon given, so surplus actor throughput counted as "
              "waste. If shorter off-policy horizon is what you want the cores for, "
              "re-run with --max-horizon <steps>.")

    if args.apply:
        old = Config().train
        path = apply_to_config(rec["async_actors"], rec["actor_games"])
        print(f"[bench] set async_actors {old.async_actors} -> {rec['async_actors']}, "
              f"actor_games {old.actor_games} -> {rec['actor_games']} in {path}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[bench] wrote {args.json}")


if __name__ == "__main__":
    main()
