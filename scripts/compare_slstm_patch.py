"""Verify the sync-free sLSTM patch is a bit-identical drop-in.

The vanilla NX-AI sLSTM forward runs ``if torch.all(n == 0.0):`` on every
timestep, forcing a device->host sync each step of the GPU learner loop.
``network._slstm_forward_nosync`` selects that step-0 branch from a Python flag
instead (see the rationale in network.py). This script proves the replacement
produces *identical* numbers -- not merely close -- at three levels:

  1. raw ``slstm_forward`` function (original vs patched)
  2. a full vanilla ``sLSTMCell.forward`` (toggling the module-level function)
  3. the end-to-end ``_XLSTMMixer`` (``sequence()`` and the acting ``step()``)

Bit-identity is checked with ``torch.equal`` (exact). Run:

    uv run python scripts/compare_slstm_patch.py [--seq 512] [--batch 8] [--seed 0]

Exits non-zero on any mismatch. Runs on CPU (the math graph is identical on GPU;
the sync it removes is a GPU-only stall, so CPU parity implies GPU parity).
"""
from __future__ import annotations

import argparse
import os
import sys

# xlstm touches torch.utils.cpp_extension at import when CUDA is visible; the same
# harmless placeholder network._build_xlstm_stack uses keeps import CPU-runnable.
os.environ.setdefault("CUDA_HOME", "/nonexistent/xlstm-vanilla-backend-no-cuda")

import torch

# Import order matters: grab the pristine original BEFORE the patch is installed.
import xlstm.blocks.slstm.cell as slstm_cell
from xlstm.blocks.slstm.src.vanilla import slstm_pointwise_function_registry

slstm_forward_pointwise = slstm_pointwise_function_registry["slstm"]

from deepnash_rbc import network


def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    exact = torch.equal(a, b)
    max_abs = (a.double() - b.double()).abs().max().item() if a.numel() else 0.0
    flag = "OK " if exact else "FAIL"
    print(f"  [{flag}] {name:<38} exact={exact}  max_abs_diff={max_abs:.3e}")
    return exact


def check_raw_forward(seq: int, batch: int, num_heads: int, head_dim: int) -> bool:
    """Level 1: call the original and patched module functions on identical inputs."""
    print("Level 1 -- raw slstm_forward(x, states, R, b, pointwise):")
    hidden = num_heads * head_dim
    num_gates = 4
    x = torch.randn(seq, batch, num_gates * hidden, dtype=torch.float32)
    R = torch.randn(num_heads, num_gates * head_dim, head_dim, dtype=torch.float32) * 0.1
    b = torch.randn(num_gates * hidden, dtype=torch.float32)
    states = torch.zeros(4, batch, hidden, dtype=torch.float32)  # canonical zero init

    orig_all, orig_last, orig_g = _ORIG_FORWARD(
        x, states, R, b, slstm_forward_pointwise
    )
    patch_all, patch_last, patch_g = network._slstm_forward_nosync(
        x, states, R, b, slstm_forward_pointwise
    )
    ok = _report("states_all", orig_all, patch_all)
    ok &= _report("final_state", orig_last, patch_last)
    ok &= _report("gates", orig_g, patch_g)
    return ok


def check_cell(seq: int, batch: int) -> bool:
    """Level 2: a real vanilla sLSTMCell.forward, toggling the module function."""
    print("Level 2 -- sLSTMCell.forward (vanilla backend):")
    from xlstm.blocks.slstm.cell import sLSTMCell, sLSTMCellConfig

    torch.manual_seed(1234)
    cfg = sLSTMCellConfig(hidden_size=64, num_heads=4, backend="vanilla")
    cell = sLSTMCell(cfg).eval()
    x = torch.randn(seq, batch, 4 * 64, dtype=torch.float32)

    slstm_cell.slstm_forward = _ORIG_FORWARD  # original
    with torch.no_grad():
        out_orig, st_orig = cell(x)
    slstm_cell.slstm_forward = network._slstm_forward_nosync  # patched
    with torch.no_grad():
        out_patch, st_patch = cell(x)

    ok = _report("output", out_orig, out_patch)
    ok &= _report("state", st_orig, st_patch)
    return ok


def check_mixer(seq: int, batch: int) -> bool:
    """Level 3: end-to-end _XLSTMMixer.sequence() and .step(), same weights."""
    print("Level 3 -- _XLSTMMixer.sequence() and .step():")
    torch.manual_seed(4242)
    dim = 64
    # slstm_at=(0, 1): both blocks are sLSTM -> exercises the patched path maximally.
    mixer = network._XLSTMMixer(
        dim=dim, num_blocks=2, num_heads=4, max_seq=max(seq, 8),
        slstm_at=(0, 1), conv_kernel=4,
    ).eval()
    tok = torch.randn(seq, batch, dim, dtype=torch.float32)
    mask = torch.ones(seq, batch, dtype=torch.bool)

    slstm_cell.slstm_forward = _ORIG_FORWARD
    with torch.no_grad():
        seq_orig = mixer.sequence(tok, mask)
    slstm_cell.slstm_forward = network._slstm_forward_nosync
    with torch.no_grad():
        seq_patch = mixer.sequence(tok, mask)
    ok = _report("sequence()", seq_orig, seq_patch)

    # acting path: feed one token at a time, compare the streamed context.
    def run_step():
        state, outs = None, []
        for t in range(seq):
            out, state = mixer.step(tok[t], state)
            outs.append(out)
        return torch.stack(outs, dim=0)

    slstm_cell.slstm_forward = _ORIG_FORWARD
    with torch.no_grad():
        step_orig = run_step()
    slstm_cell.slstm_forward = network._slstm_forward_nosync
    with torch.no_grad():
        step_patch = run_step()
    ok &= _report("step() stream", step_orig, step_patch)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512, help="sequence length (games are <= max_seq=512)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"seq={args.seq} batch={args.batch} seed={args.seed}\n")

    ok = True
    ok &= check_raw_forward(args.seq, args.batch, num_heads=4, head_dim=16)
    print()
    ok &= check_cell(args.seq, args.batch)
    print()
    ok &= check_mixer(args.seq, args.batch)
    print()

    if ok:
        print("ALL CHECKS PASSED -- sync-free sLSTM patch is bit-identical.")
        return 0
    print("MISMATCH -- patch is NOT bit-identical. Do not ship.")
    return 1


# Capture the pristine original before importing network installs anything.
_ORIG_FORWARD = slstm_cell.slstm_forward
assert _ORIG_FORWARD is not network._slstm_forward_nosync, (
    "patch already installed at import time -- cannot capture the original"
)

if __name__ == "__main__":
    sys.exit(main())
