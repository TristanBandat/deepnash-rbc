"""Benchmark the xLSTM learner step at different batch sizes (and with/without the
sync-free sLSTM patch), to test the 'batch-independent' throughput claim on GPU.

Mirrors the real learner: TemporalNet(arch=xlstm).forward(seq[T,B,F,8,8], mask) ->
CNN per-frame encoder -> mixer.sequence (the sLSTM Python time-loop) -> FiLM ->
heads, then a scalar loss + backward + optimizer step, under bf16 autocast (the
production default amp=True). Times the full step with CUDA events.

    uv run python scripts/bench_slstm_step.py --seq 512 --batches 32 64 --iters 15

Reports ms/step, games/sec, the 64/32 wall-clock ratio (≈1 => batch-independent =>
2x batch is ~free 2x samples; ≈2 => batch-linear), and the patch on/off speedup.
Run on whatever CUDA device is present; absolute ms are device-specific (here an
RTX 2080, not the L40S), but the *ratios* transfer.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

os.environ.setdefault("CUDA_HOME", "/nonexistent/xlstm-vanilla-backend-no-cuda")

import torch

from deepnash_rbc import network
from deepnash_rbc.config import EncodingConfig, NetworkConfig

# The pristine original still lives in src.vanilla (the patch only rebinds the name
# in the cell module); import it so we can toggle patch on/off for A/B timing.
import xlstm.blocks.slstm.cell as slstm_cell
from xlstm.blocks.slstm.src.vanilla import slstm_forward as ORIG_FORWARD


def build_net(slstm_at, device):
    enc = EncodingConfig(history=1)
    net = NetworkConfig(
        arch="xlstm", channels=128, enc_blocks=4, mixer_dim=128,
        mixer_layers=2, nhead=4, max_seq=512,
        xlstm_slstm_at=tuple(slstm_at), xlstm_conv_kernel=4,
    )
    return network.make_net(enc, net).to(device), enc.frame_channels


def time_step(model, opt, seq, mask, iters, warmup, amp, device, mixer_only=False):
    """Median ms for a full forward+backward+opt.step, CUDA-event timed.

    ``mixer_only`` times just ``mixer.sequence`` (the sLSTM loop) on token inputs,
    isolating the sLSTM part from the batch-linear CNN encoder + heads."""
    def one_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            if mixer_only:
                loss = model.mixer.sequence(seq, mask).float().mean()
            else:
                value, sense, move = model(seq, mask)
                loss = value.float().mean() + sense.float().mean() + move.float().mean()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        one_step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512, help="T (worst case = max_seq)")
    ap.add_argument("--batches", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--slstm-at", type=int, nargs="*", default=[1],
                    help="sLSTM block indices; [1]=canonical mixed, [0 1]=all-sLSTM")
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--mixer-only", action="store_true",
                    help="time only mixer.sequence (the sLSTM loop), no CNN/heads")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; this benchmark measures GPU behaviour. Aborting.")
        return 1
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = not args.no_amp
    T = args.seq
    print(f"device={torch.cuda.get_device_name(0)}  T={T}  amp={amp}  "
          f"slstm_at={args.slstm_at}  mixer_only={args.mixer_only}  "
          f"iters={args.iters} (warmup {args.warmup})\n")

    model, F = build_net(args.slstm_at, device)
    D = 128  # mixer_dim, for mixer-only token inputs
    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()

    variants = [("patched (sync-free)", network._slstm_forward_nosync),
                ("original (per-step sync)", ORIG_FORWARD)]

    results = {}  # (variant, batch) -> ms
    for vname, fwd in variants:
        slstm_cell.slstm_forward = fwd
        for B in args.batches:
            torch.manual_seed(0)
            if args.mixer_only:
                seq = torch.randn(T, B, D, device=device)
            else:
                seq = torch.randn(T, B, F, 8, 8, device=device)
            mask = torch.ones(T, B, dtype=torch.bool, device=device)
            try:
                ms = time_step(model, opt, seq, mask, args.iters, args.warmup, amp,
                               device, mixer_only=args.mixer_only)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM at variant={vname} B={B}; skipping")
                continue
            results[(vname, B)] = ms
            games_per_s = B / (ms / 1000.0)
            print(f"  {vname:<26} B={B:<4} {ms:8.1f} ms/step   {games_per_s:7.1f} games/s")
            del seq, mask
            torch.cuda.empty_cache()
        print()

    # --- interpretation ----------------------------------------------------
    print("=" * 60)
    for vname, _ in variants:
        bs = [b for b in args.batches if (vname, b) in results]
        if len(bs) >= 2:
            lo, hi = min(bs), max(bs)
            ratio = results[(vname, hi)] / results[(vname, lo)]
            samp = (hi / results[(vname, hi)]) / (lo / results[(vname, lo)])
            print(f"[{vname}] B{hi}/B{lo} wall-clock ratio = {ratio:.2f}x  "
                  f"(1.0=batch-independent, 2.0=batch-linear); "
                  f"samples/sec gain = {samp:.2f}x")
    for B in args.batches:
        if ("original (per-step sync)", B) in results and ("patched (sync-free)", B) in results:
            spd = results[("original (per-step sync)", B)] / results[("patched (sync-free)", B)]
            print(f"[patch] B={B}: sync-free is {spd:.2f}x faster than original")
    return 0


if __name__ == "__main__":
    sys.exit(main())
