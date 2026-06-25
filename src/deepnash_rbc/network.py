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
