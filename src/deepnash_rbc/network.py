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
