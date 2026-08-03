"""DeepNash-style network for RBC.

A shared convolutional torso (residual blocks at constant 8x8 resolution, the
AlphaZero board-game design) feeds three heads:
  - value:  scalar in [-1, 1]  (tanh)
  - sense:  64 logits          (distribution over sense-window centers)
  - move:   4673 logits        (AlphaZero 8x8x73 move planes + 1 pass action)

Constant-resolution ResNet is used rather than DeepNash's U-Net: on an 8x8 board
the multi-scale benefit of a U-Net is marginal and the ResNet is simpler/faster
to train on a single GPU. Swap in a U-Net torso here if local+global integration
becomes the bottleneck.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import NetworkConfig, EncodingConfig
from .encoding.moves import MOVE_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class DeepNashNet(nn.Module):
    is_temporal = False

    def __init__(self, enc: EncodingConfig, net: NetworkConfig):
        super().__init__()
        c = net.channels
        self.stem = nn.Sequential(
            nn.Conv2d(enc.in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.torso = nn.Sequential(*[ResidualBlock(c) for _ in range(net.blocks)])

        # value head: 1x1 conv -> flatten -> MLP -> tanh
        self.value_conv = nn.Sequential(
            nn.Conv2d(c, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(64, net.value_hidden), nn.ReLU(inplace=True),
            nn.Linear(net.value_hidden, 1), nn.Tanh(),
        )

        # sense head: 1x1 conv -> [B,1,8,8] -> 64 logits
        self.sense_conv = nn.Conv2d(c, 1, 1)

        # move head: conv -> [B,73,8,8] -> 4672 logits, + scalar pass logit
        self.move_conv = nn.Conv2d(c, MOVE_PLANES // 64, 1)  # 73 planes
        self.pass_fc = nn.Linear(c, 1)

    def forward(self, x: torch.Tensor):
        h = self.torso(self.stem(x))
        b = h.size(0)

        value = self.value_fc(self.value_conv(h).reshape(b, 64)).squeeze(-1)

        sense_logits = self.sense_conv(h).reshape(b, 64)

        move_planes = self.move_conv(h).reshape(b, -1)  # [B, 4672]
        pooled = h.mean(dim=(2, 3))  # [B, C]
        pass_logit = self.pass_fc(pooled)  # [B, 1]
        move_logits = torch.cat([move_planes, pass_logit], dim=1)  # [B, 4673]

        return value, sense_logits, move_logits


# ---------------------------------------------------------------------------- #
#  Temporal (streaming-state) architectures: RNN / Transformer                 #
# ---------------------------------------------------------------------------- #
#
# TemporalNet processes ONE 19-channel frame per decision step and carries a
# recurrent/attention state across the whole game -- it does not stack history
# into channels, so it ignores encoding.history. A shared per-frame encoder maps
# each frame [19,8,8] -> a spatial map [C,8,8] and a pooled token [D]; a swappable
# temporal mixer (GRU/LSTM or causal Transformer) runs over the token sequence to
# a per-step context [D], which is FiLM-injected back into the spatial map before
# the (own copies of the) conv heads. GroupNorm is used throughout instead of
# BatchNorm: the training forward flattens padded [T,B,...] sequences, and acting
# runs one frame at a time -- per-sample norm keeps padded rows from touching real
# stats and makes train/eval/act numerics identical.


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = max_groups
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResidualBlockGN(ResidualBlock):
    """ResidualBlock with GroupNorm (see TemporalNet rationale)."""

    def __init__(self, channels: int):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = _group_norm(channels)


class _RecurrentMixer(nn.Module):
    """GRU/LSTM over the token sequence. Same weights drive the batched training
    ``sequence`` path and the one-step acting ``step`` path."""

    def __init__(self, dim: int, layers: int, kind: str):
        super().__init__()
        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(dim, dim, num_layers=layers)  # [T, B, D]

    def sequence(self, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # padded tail steps are causally after every real step -> their outputs
        # are discarded by the flat<->[T,B] gather, so no packing is needed.
        out, _ = self.rnn(tok)
        return out

    def step(self, tok: torch.Tensor, state):
        out, new_state = self.rnn(tok.unsqueeze(0), state)  # [1,B,D]
        return out.squeeze(0), new_state


class _TransformerMixer(nn.Module):
    """Causal Transformer encoder over the token sequence."""

    def __init__(self, dim: int, nhead: int, layers: int, max_seq: int):
        super().__init__()
        self.max_seq = max_seq
        # dropout=0: keep the forward deterministic (as the ResNet/GRU paths are),
        # so train/eval/act match and the fast==legacy learner guarantee holds.
        # dim_feedforward = 4*dim (standard ratio) rather than PyTorch's fixed 2048,
        # so the FFN scales with mixer_dim instead of dominating a small model.
        layer = nn.TransformerEncoderLayer(
            dim, nhead, dim_feedforward=4 * dim, dropout=0.0, batch_first=False
        )
        # enable_nested_tensor is a batch_first-only padding-skip optimization;
        # with batch_first=False PyTorch disables it at runtime anyway, so set it
        # explicitly to match reality and silence the spurious startup warning.
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.register_buffer("pos", self._sinusoidal(max_seq, dim), persistent=False)

    @staticmethod
    def _sinusoidal(max_seq: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_seq, dim)
        pos = torch.arange(max_seq).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    @staticmethod
    def _causal_mask(t: int, device) -> torch.Tensor:
        # bool [T,T]; True = disallowed (upper triangle, excluding diagonal)
        return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)

    def sequence(self, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t = tok.shape[0]
        assert t <= self.max_seq, f"sequence length {t} exceeds max_seq {self.max_seq}"
        x = tok + self.pos[:t].unsqueeze(1)
        attn = self._causal_mask(t, tok.device)
        key_pad = ~mask.transpose(0, 1)  # [B, T]; True = padding
        return self.enc(x, mask=attn, src_key_padding_mask=key_pad)

    def step(self, tok: torch.Tensor, state):
        buf = tok.unsqueeze(0) if state is None else torch.cat([state, tok.unsqueeze(0)], 0)
        t = buf.shape[0]
        assert t <= self.max_seq, f"sequence length {t} exceeds max_seq {self.max_seq}"
        x = buf + self.pos[:t].unsqueeze(1)
        out = self.enc(x, mask=self._causal_mask(t, buf.device))
        return out[-1], buf  # last position is the current step's context


# --- sLSTM vanilla-backend perf patch (bit-identical) -----------------------
# The NX-AI vanilla sLSTM forward is a sequential Python time-loop; its pointwise
# runs ``if torch.all(n == 0.0):`` every step, which forces a device->host sync on
# every timestep of the GPU learner (~T stalls per sequence). The step-0 branch is
# the *only* place it differs, and it is fully determined by position: the initial
# n is 0 (step 0) and n = fgate*n + igate is strictly > 0 afterwards (igate =
# min(exp(.), 1) > 0). So selecting the branch from a Python ``first`` flag is
# provably identical output with no per-step sync. We patch only the training
# ``slstm_forward`` (the GPU learner loop); the CPU-actor ``slstm_forward_step`` is
# left untouched (no GPU sync there, and train==act parity is preserved because the
# patched forward stays bit-identical to the original forward).
_SLSTM_NOSYNC_INSTALLED = False


def _slstm_pointwise_nosync(Wx, Ry, b, states, first):
    """Bit-identical reimplementation of xlstm's vanilla ``slstm_forward_pointwise``
    that picks the step-0 branch from the Python ``first`` flag instead of the
    ``torch.all(n == 0.0)`` GPU reduction (which syncs). Same ops, same order."""
    from torch.nn.functional import logsigmoid

    raw = Wx + Ry + b
    y, c, n, m = torch.unbind(states.view(4, states.shape[1], -1), dim=0)
    iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)
    logfplusm = m + logsigmoid(fraw)
    mnew = iraw if first else torch.max(iraw, logfplusm)
    ogate = torch.sigmoid(oraw)
    igate = torch.minimum(torch.exp(iraw - mnew), torch.ones_like(iraw))
    fgate = torch.minimum(torch.exp(logfplusm - mnew), torch.ones_like(iraw))
    cnew = fgate * c + igate * torch.tanh(zraw)
    nnew = fgate * n + igate
    ynew = ogate * cnew / nnew
    return (
        torch.stack((ynew, cnew, nnew, mnew), dim=0),
        torch.stack((igate, fgate, zraw, ogate), dim=0),
    )


def _slstm_forward_nosync(x, states, R, b, pointwise_forward, constants={}):
    """Drop-in for ``xlstm.blocks.slstm.cell.slstm_forward`` with the per-timestep
    sync removed. Identical loop, shapes and return value; only the pointwise is
    swapped for the sync-free variant (valid because arch=='xlstm' blocks only ever
    use function=='slstm'). ``pointwise_forward`` is intentionally ignored."""
    del pointwise_forward, constants
    num_states = states.shape[0]
    sequence_dim = x.shape[0]
    num_gates_r = R.shape[1] // R.shape[2]
    hidden_dim = R.shape[2] * R.shape[0]
    num_gates_t = b.shape[0] // hidden_dim
    batch_dim = x.shape[1]
    num_heads = R.shape[0]
    head_dim = R.shape[2]

    assert batch_dim == states.shape[1]
    assert hidden_dim == states.shape[2]

    g = torch.zeros(
        [sequence_dim + 1, num_gates_t, batch_dim, hidden_dim],
        device=x.device, dtype=x.dtype,
    )
    states_all = torch.zeros(
        [num_states, sequence_dim + 1, batch_dim, hidden_dim],
        device=x.device, dtype=x.dtype,
    )
    states_all[:, 0] = states
    for i, Wx_t in enumerate(x.unbind(dim=0)):
        Ry = (
            states[0]
            .reshape(batch_dim, num_heads, 1, -1)
            .matmul(
                R.transpose(1, 2).reshape(1, num_heads, head_dim, num_gates_r * head_dim)
            )
            .reshape(batch_dim, num_heads, num_gates_r, -1)
            .transpose(1, 2)
            .reshape(batch_dim, -1)
        )
        sdtype = states.dtype
        states, gates = _slstm_pointwise_nosync(Wx_t, Ry, b, states, first=(i == 0))
        states = states.to(dtype=sdtype)
        g[i] = gates
        states_all[:, i + 1] = states

    return states_all, states, g


def _install_slstm_nosync_patch():
    """Idempotently swap the vanilla sLSTM training forward for the sync-free one."""
    global _SLSTM_NOSYNC_INSTALLED
    if _SLSTM_NOSYNC_INSTALLED:
        return
    from xlstm.blocks.slstm import cell as _slstm_cell

    _slstm_cell.slstm_forward = _slstm_forward_nosync
    _SLSTM_NOSYNC_INSTALLED = True


def _build_xlstm_stack(dim, num_blocks, num_heads, max_seq, slstm_at, conv_kernel):
    """Construct an NX-AI ``xLSTMBlockStack`` of alternating mLSTM/sLSTM blocks.

    Imported lazily (only when arch=="xlstm") so the other archs never pay the
    import cost. The xlstm package touches ``torch.utils.cpp_extension`` at import
    time whenever CUDA is visible (to locate the sLSTM CUDA kernel), which raises if
    CUDA_HOME is unset -- so set a harmless placeholder first. We never compile the
    kernel: the sLSTM ``backend="vanilla"`` (pure-PyTorch) path keeps the forward
    deterministic and CPU-runnable, preserving the train==act / fast==legacy
    guarantees; the placeholder is only consulted by the (unused) CUDA compiler.
    """
    import os

    os.environ.setdefault("CUDA_HOME", "/nonexistent/xlstm-vanilla-backend-no-cuda")
    from xlstm import (
        xLSTMBlockStack, xLSTMBlockStackConfig,
        mLSTMBlockConfig, mLSTMLayerConfig,
        sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig,
    )

    _install_slstm_nosync_patch()  # remove the vanilla sLSTM per-timestep GPU sync

    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(num_heads=num_heads, conv1d_kernel_size=conv_kernel),
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=num_heads, backend="vanilla",
                conv1d_kernel_size=conv_kernel, dropout=0.0,
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu", dropout=0.0),
        ),
        num_blocks=num_blocks, embedding_dim=dim, context_length=max_seq,
        slstm_at=list(slstm_at), dropout=0.0, bias=True,
    )
    return xLSTMBlockStack(cfg)


class _XLSTMMixer(nn.Module):
    """Full xLSTM stack (alternating mLSTM/sLSTM blocks, NX-AI package) over the
    token sequence. Pure-PyTorch 'vanilla' backend + dropout=0 keep the forward
    deterministic and CPU-runnable (train==act, fast==legacy guarantees)."""

    def __init__(self, dim, num_blocks, num_heads, max_seq, slstm_at, conv_kernel):
        super().__init__()
        self.max_seq = max_seq
        self.stack = _build_xlstm_stack(
            dim, num_blocks, num_heads, max_seq, slstm_at, conv_kernel
        )

    def sequence(self, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # xLSTM is causal and batch_first; like _RecurrentMixer the padded tail steps
        # are causally after every real step, so their outputs are discarded by the
        # flat<->[T,B] gather and no explicit mask is needed.
        t = tok.shape[0]
        assert t <= self.max_seq, f"sequence length {t} exceeds max_seq {self.max_seq}"
        return self.stack(tok.transpose(0, 1)).transpose(0, 1)  # [T,B,D]

    def step(self, tok: torch.Tensor, state):
        # buffer-reuse (as _TransformerMixer.step) for exact train==act parity: keep
        # the growing token buffer and re-run the causal stack, returning the last
        # position. Cheap because RBC games are short and acting is one game at a time.
        buf = tok.unsqueeze(0) if state is None else torch.cat([state, tok.unsqueeze(0)], 0)
        t = buf.shape[0]
        assert t <= self.max_seq, f"sequence length {t} exceeds max_seq {self.max_seq}"
        out = self.stack(buf.transpose(0, 1)).transpose(0, 1)
        return out[-1], buf


class TemporalNet(nn.Module):
    is_temporal = True

    def __init__(self, enc: EncodingConfig, net: NetworkConfig):
        super().__init__()
        f = enc.frame_channels
        c = net.channels
        d = net.mixer_dim
        self.c, self.d = c, d

        # per-frame encoder (shared; applied to every frame independently)
        self.stem = nn.Sequential(
            nn.Conv2d(f, c, 3, padding=1, bias=False),
            _group_norm(c),
            nn.ReLU(inplace=True),
        )
        self.enc_blocks = nn.Sequential(*[ResidualBlockGN(c) for _ in range(net.enc_blocks)])
        self.token_proj = nn.Linear(c, d)

        # temporal mixer over tokens
        if net.arch in ("gru", "lstm"):
            self.mixer = _RecurrentMixer(d, net.mixer_layers, net.arch)
        elif net.arch == "xlstm":
            self.mixer = _XLSTMMixer(
                d, net.mixer_layers, net.nhead, net.max_seq,
                net.xlstm_slstm_at, net.xlstm_conv_kernel,
            )
        else:
            self.mixer = _TransformerMixer(d, net.nhead, net.mixer_layers, net.max_seq)

        # FiLM: context -> per-channel (gamma, beta)
        self.film = nn.Linear(d, 2 * c)

        # heads (own params; same design as DeepNashNet but GroupNorm not BatchNorm)
        self.value_conv = nn.Sequential(
            nn.Conv2d(c, 1, 1, bias=False), _group_norm(1), nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(64, net.value_hidden), nn.ReLU(inplace=True),
            nn.Linear(net.value_hidden, 1), nn.Tanh(),
        )
        self.sense_conv = nn.Conv2d(c, 1, 1)
        self.move_conv = nn.Conv2d(c, MOVE_PLANES // 64, 1)  # 73 planes
        self.pass_fc = nn.Linear(c, 1)

    # -- shared pieces -------------------------------------------------------
    def _encode(self, frames: torch.Tensor):
        # frames: [M, F, 8, 8] -> map [M, C, 8, 8], token [M, D]
        h = self.enc_blocks(self.stem(frames))
        tok = self.token_proj(h.mean(dim=(2, 3)))
        return h, tok

    def _apply_film(self, spatial: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # spatial: [..., C, 8, 8]; ctx: [..., D]
        gamma, beta = self.film(ctx).chunk(2, dim=-1)  # [..., C] each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return spatial * (1 + gamma) + beta

    def _heads(self, fused: torch.Tensor):
        # fused: [M, C, 8, 8]
        m = fused.size(0)
        value = self.value_fc(self.value_conv(fused).reshape(m, 64)).squeeze(-1)
        sense_logits = self.sense_conv(fused).reshape(m, 64)
        move_planes = self.move_conv(fused).reshape(m, -1)  # [M, 4672]
        pass_logit = self.pass_fc(fused.mean(dim=(2, 3)))  # [M, 1]
        move_logits = torch.cat([move_planes, pass_logit], dim=1)  # [M, 4673]
        return value, sense_logits, move_logits

    # -- training entry (batched sequences) ----------------------------------
    def forward(self, seq: torch.Tensor, mask: torch.Tensor):
        # seq: [T, B, F, 8, 8]; mask (valid): [T, B] bool
        t, b = seq.shape[0], seq.shape[1]
        spatial, tok = self._encode(seq.reshape(t * b, *seq.shape[2:]))
        tok = tok.view(t, b, self.d)
        ctx = self.mixer.sequence(tok, mask)  # [T, B, D] (causal)
        spatial = spatial.view(t, b, self.c, 8, 8)
        fused = self._apply_film(spatial, ctx).reshape(t * b, self.c, 8, 8)
        value, sense_logits, move_logits = self._heads(fused)
        return (
            value.view(t, b),
            sense_logits.view(t, b, -1),
            move_logits.view(t, b, -1),
        )

    # -- acting entry (one frame + carried state) ----------------------------
    def step(self, frame: torch.Tensor, state):
        # frame: [B, F, 8, 8] (B is 1 at acting time)
        spatial, tok = self._encode(frame)
        ctx, new_state = self.mixer.step(tok, state)  # [B, D]
        fused = self._apply_film(spatial, ctx)  # [B, C, 8, 8]
        value, sense_logits, move_logits = self._heads(fused)
        return value, sense_logits, move_logits, new_state


def make_net(enc: EncodingConfig, net: NetworkConfig) -> nn.Module:
    """Build the network selected by ``net.arch``.

    ``resnet`` -> the original channel-stacked :class:`DeepNashNet`.
    ``gru`` / ``lstm`` / ``transformer`` / ``xlstm`` -> the whole-game
    streaming-state :class:`TemporalNet`, whose mixer is chosen by the same
    ``arch`` string.
    """
    if net.arch == "resnet":
        return DeepNashNet(enc, net)
    if net.arch in ("gru", "lstm", "transformer", "xlstm"):
        return TemporalNet(enc, net)
    raise ValueError(
        f"unknown network.arch {net.arch!r} "
        "(expected one of: resnet, gru, lstm, transformer, xlstm)"
    )
