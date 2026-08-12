"""Headless driver for the evaluation-policy arena (argmax vs sampling).

Same engine and same on-disk format as ``notebooks/policy_eval.py`` -- use this
for the long runs and open the notebook on the same ``--out`` file to explore
the result. Both resume: re-running a command only plays the games still missing
for each pair.

Arms are the cross product of ``--checkpoints`` (paths, globs or ``v<ver>_<step>``
labels) with ``--modes``. They are measured against a pool of **anchors** taken
from the existing internal ladder, whose Elo is held FIXED -- the ladder itself
is only ever read. See ``deepnash_rbc.analysis.arena`` for why that matters.

Examples:

    # where does argmax land vs sampled, for one checkpoint
    uv run python tools/policy_eval.py --checkpoints v0.14.0_80000 \
        --modes argmax,sample@0.05 --pair-games 40

    # threshold sweep across a whole run
    uv run python tools/policy_eval.py --checkpoints 'checkpoints/v0.14.0/*.pt' \
        --modes argmax,sample,sample@0.05,sample@0.2 --anchor-top 8

    # the two modes playing each other directly, no anchors
    uv run python tools/policy_eval.py --checkpoints v0.14.0_80000 \
        --modes argmax,sample@0.05 --schedule paired --no-anchors --pair-games 200

    # every checkpoint of one run against every checkpoint of another
    uv run python tools/policy_eval.py --checkpoints 'checkpoints/v0.14.0/*.pt' \
        --cross-b 'checkpoints/v0.8.0/*.pt' --modes argmax --modes-b sample@0.05 \
        --schedule cross

    uv run python tools/policy_eval.py ... --dry-run     # size the run first
    uv run python tools/policy_eval.py --out ... --report-only
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deepnash_rbc.analysis import arena  # noqa: E402


def resolve_specs(specs: list[str], checkpoints: dict[str, str]) -> dict[str, str]:
    """``{label: path}`` from ladder labels, file paths and globs."""
    out: dict[str, str] = {}
    for spec in specs:
        if spec in checkpoints:
            out[spec] = checkpoints[spec]
            continue
        paths = sorted(glob.glob(spec)) or ([spec] if Path(spec).exists() else [])
        if not paths:
            sys.exit(f"checkpoint spec matched nothing: {spec}")
        for p in paths:
            label = Path(p).stem.replace("deepnash_async_", "")
            out[label] = str(Path(p).resolve())
    return out


def report(out_path: str, leaderboard: dict, checkpoints: dict) -> None:
    rows = arena.load_rows(out_path)
    arms = arena.load_arms(out_path)
    if not rows:
        print("no games recorded yet")
        return
    anchor_elo = {
        n: leaderboard[n]["elo"]
        for n in arms
        if n in leaderboard and (n in checkpoints or n in arena.BASELINES)
    }
    results = arena.fit_arms(arena.summarize(rows, arms), anchor_elo)
    played = {n: r for n, r in results.items() if r.games}

    print(f"\n=== Arena ({len(rows)} games, {len(anchor_elo)} frozen anchors) ===")
    print(f"{'arm':<34} {'mode':<13} {'elo':>7} {'+/-':>5} {'games':>6} "
          f"{'score':>6} {'draw%':>6}")
    for name, r in sorted(
        played.items(),
        key=lambda kv: -(anchor_elo.get(kv[0]) or (kv[1].fit.elo if kv[1].fit else -1e9)),
    ):
        anchored = name in anchor_elo
        # an arm with no games against a rated anchor has no rating at all --
        # print a dash rather than a nan, which reads as a broken fit
        elo = (f"{anchor_elo[name]:.0f}" if anchored
               else (f"{r.fit.elo:.0f}" if r.fit else "-"))
        se = "" if anchored or not r.fit else f"{r.fit.se:.0f}"
        flag = " *" if r.fit and r.fit.degenerate and not anchored else ""
        print(f"{name:<34} {(r.arm.mode if r.arm else ''):<13} {elo:>7} {se:>5} "
              f"{r.games:>6} {r.score_rate:>6.2f} {100 * r.draw_rate:>6.1f}{flag}")

    # mode deltas: same weights, different selection rule
    by_spec: dict[str, list[str]] = {}
    for n, r in played.items():
        if r.arm and r.arm.is_net and r.fit is not None:
            by_spec.setdefault(r.arm.spec, []).append(n)
    deltas = [
        (spec, a, b)
        for spec, names in by_spec.items()
        for i, a in enumerate(names)
        for b in names[i + 1:]
    ]
    if deltas:
        print("\n=== Mode deltas (anchored Elo) ===")
        print(f"{'checkpoint':<28} {'A':<13} {'B':<13} {'dElo':>7} {'+/-':>5} {'z':>6}")
        for spec, a, b in deltas:
            if results[b].arm.greedy and not results[a].arm.greedy:
                a, b = b, a  # orient argmax - sampled
            d = arena.mode_delta(results, a, b)
            label = Path(spec).stem.replace("deepnash_async_", "")
            print(f"{label:<28} {results[a].arm.mode:<13} {results[b].arm.mode:<13} "
                  f"{d['delta']:>7.0f} {d['se']:>5.0f} {d['z']:>6.2f}"
                  + ("  *" if d["significant"] else ""))

    # direct matches, where they exist
    h2h = []
    for spec, names in by_spec.items():
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                m = arena.head_to_head(rows, a, b)
                if m["games"]:
                    h2h.append((spec, a, b, m))
    if h2h:
        print("\n=== Direct head-to-head ===")
        print(f"{'checkpoint':<28} {'A':<13} {'B':<13} {'games':>6} {'A score':>8} {'95% CI':>15}")
        for spec, a, b, m in h2h:
            label = Path(spec).stem.replace("deepnash_async_", "")
            print(f"{label:<28} {results[a].arm.mode:<13} {results[b].arm.mode:<13} "
                  f"{m['games']:>6} {m['rate']:>8.3f} "
                  f"{m['ci_low']:>7.3f}-{m['ci_high']:<7.3f}"
                  + ("  *" if m["significant"] else ""))

    stats = [(n, r) for n, r in played.items() if r.move.decisions or r.sense.decisions]
    if stats:
        print("\n=== Policy statistics (per decision) ===")
        print(f"{'arm':<34} {'head':<6} {'n':>8} {'agree%':>7} {'entropy':>8} "
              f"{'top1':>6} {'cut%':>6} {'bypass%':>8}")
        for name, r in sorted(stats):
            for head, s in (("sense", r.sense), ("move", r.move)):
                if not s.decisions:
                    continue
                print(f"{name:<34} {head:<6} {s.decisions:>8} "
                      f"{100 * s.argmax_agree / s.decisions:>7.1f} "
                      f"{s.entropy_bits / s.decisions:>8.3f} "
                      f"{s.top1_prob / s.decisions:>6.3f} "
                      f"{100 * s.truncated_mass / s.decisions:>6.1f} "
                      f"{100 * s.threshold_bypassed / s.decisions:>8.1f}")
        print("  cut% = policy mass the threshold discarded; bypass% = decisions "
              "where nothing\n  cleared the threshold, so it was ignored entirely "
              "(truncation is not monotone).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoints", default="",
                    help="comma-separated paths, globs or v<ver>_<step> labels")
    ap.add_argument("--modes", default="argmax,sample@0.05",
                    help="comma-separated policy modes for the A side")
    ap.add_argument("--cross-b", default="", help="B-side checkpoints (schedule=cross)")
    ap.add_argument("--modes-b", default="sample@0.05", help="B-side policy modes")
    ap.add_argument("--schedule", default="anchors",
                    choices=("anchors", "paired", "cross", "round_robin"))

    ap.add_argument("--no-anchors", action="store_true",
                    help="skip the anchor pool (no ladder-scale Elo)")
    ap.add_argument("--anchor-top", type=int, default=12,
                    help="how many top ladder players to anchor against")
    ap.add_argument("--no-baselines", action="store_true",
                    help="exclude random/trout/mht from the anchor pool")
    ap.add_argument("--anchor-elo-min", type=float, default=300.0)
    ap.add_argument("--anchor-elo-max", type=float, default=2000.0)
    ap.add_argument("--anchor-min-games", type=int, default=200,
                    help="drop anchors with a thinly-played (noisy) ladder rating")
    ap.add_argument("--anchor-mode", default=arena.REFERENCE_MODE,
                    help="policy mode anchors replay; must match the ladder's")
    ap.add_argument("--anchors", default="",
                    help="explicit comma-separated anchor names (overrides the pool)")

    ap.add_argument("--pair-games", type=int, default=20,
                    help="games per unordered pair (colors alternate; keep it even)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=900.0)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--net-cache", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stats", action="store_true",
                    help="skip per-decision policy statistics")

    ap.add_argument("--out", default=str(ROOT / "results" / "policy_eval.jsonl"))
    ap.add_argument("--leaderboard",
                    default=str(ROOT / "results" / "tournament_leaderboard.txt"))
    ap.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the schedule size and exit")
    ap.add_argument("--report-only", action="store_true",
                    help="recompute the report from --out and exit")
    args = ap.parse_args()

    checkpoints = arena.discover_checkpoints(args.checkpoint_dir)
    leaderboard = (
        arena.parse_leaderboard(args.leaderboard)
        if Path(args.leaderboard).exists() else {}
    )
    if args.report_only:
        report(args.out, leaderboard, checkpoints)
        return

    def _split(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    if not _split(args.checkpoints):
        sys.exit("--checkpoints is required (paths, globs or v<ver>_<step> labels)")
    ckpts_a = resolve_specs(_split(args.checkpoints), checkpoints)
    ckpts_b = resolve_specs(_split(args.cross_b), checkpoints) if args.cross_b else {}
    arms_a = arena.make_arms(ckpts_a, _split(args.modes))
    arms_b = arena.make_arms(ckpts_b, _split(args.modes_b)) if ckpts_b else []

    arms = list(arms_a) + (list(arms_b) if args.schedule == "cross" else [])
    by_name = {a.name: a for a in arms}
    if len(by_name) != len(arms):
        sys.exit("duplicate arm names -- the A and B sides overlap; vary the mode")

    if args.schedule == "anchors":
        pairs = []
    elif args.schedule == "paired":
        pairs = arena.pairs_paired_modes(arms)
        if not pairs:
            sys.exit("--schedule paired needs >=2 modes on the same checkpoint")
    elif args.schedule == "cross":
        if not arms_b:
            sys.exit("--schedule cross needs --cross-b")
        pairs = arena.pairs_cross(arms_a, arms_b)
    else:
        pairs = arena.pairs_round_robin(arms)

    anchor_names: list[str] = []
    if not args.no_anchors:
        if not leaderboard:
            sys.exit(f"no leaderboard at {args.leaderboard} -- pass --leaderboard "
                     f"or --no-anchors")
        anchor_names = arena.anchor_pool(
            leaderboard, checkpoints,
            top=args.anchor_top,
            include_baselines=not args.no_baselines,
            elo_range=(args.anchor_elo_min, args.anchor_elo_max),
            min_games=args.anchor_min_games,
            explicit=_split(args.anchors) or None,
        )
        anchor_names = [n for n in anchor_names if n not in by_name]
        if not anchor_names:
            sys.exit("anchor pool is empty -- widen --anchor-elo-min/max or --anchor-top")
        if args.anchor_mode != arena.REFERENCE_MODE:
            print(f"WARNING: --anchor-mode {args.anchor_mode!r} is not the ladder's "
                  f"{arena.REFERENCE_MODE!r}; the frozen ratings will not apply",
                  file=sys.stderr)
        anchor_arms = [
            a for n in anchor_names
            if (a := arena.anchor_arm(n, checkpoints, args.anchor_mode))
        ]
        pairs = pairs + arena.pairs_cross(arms, anchor_arms)
        arms = arms + anchor_arms
    elif args.schedule == "anchors":
        sys.exit("--schedule anchors is meaningless with --no-anchors")

    all_arms = {a.name: a for a in arms}
    clash = arena.check_arms(args.out, all_arms)
    if clash:
        sys.exit(f"{args.out} already holds these names with a DIFFERENT definition: "
                 f"{', '.join(clash)}\nUse a fresh --out or rename the arms -- "
                 f"otherwise two different agents merge into one rating.")

    prior = arena.load_rows(args.out)
    tasks = arena.build_tasks(
        pairs, args.pair_games, arena.existing_counts(prior),
        seconds=args.seconds, seed=args.seed,
    )
    engine = arena.count_engine_games(tasks, all_arms)
    print(f"{len(all_arms)} players ({len(anchor_names)} anchors), {len(pairs)} pairs, "
          f"{len(tasks)} games to play ({engine} vs engine bots) "
          f"({len(prior)} already in {args.out})")
    if args.dry_run:
        return

    if tasks:
        arena.save_arms(args.out, all_arms)
        t0 = time.time()
        errors = 0
        for i, row in enumerate(arena.run_games(
            tasks, all_arms, workers=args.workers, device=args.device,
            net_cache=args.net_cache, collect_stats=not args.no_stats,
            out_path=args.out,
        ), 1):
            if row["winner"] == "error":
                errors += 1
                print(f"[{i}/{len(tasks)}] ERROR {row['white']} vs {row['black']}: "
                      f"{row['reason']}", flush=True)
            elif i % 10 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] "
                      f"{arena.eta_string(i, len(tasks), time.time() - t0)}"
                      + (f" · {errors} errored" if errors else ""), flush=True)

    report(args.out, leaderboard, checkpoints)


if __name__ == "__main__":
    main()
