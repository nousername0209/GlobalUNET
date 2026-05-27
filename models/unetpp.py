from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .unet import ConvBlock


class UNetPlusPlus(nn.Module):
    """UNet++ with nested skip pathways and optional deep supervision."""

    def __init__(self, num_classes: int, in_channels: int = 3, channels: tuple[int, ...] = (64, 128, 256, 512, 1024), deep_supervision: bool = False) -> None:
        super().__init__()
        c = channels
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = ConvBlock(in_channels, c[0])
        self.conv1_0 = ConvBlock(c[0], c[1])
        self.conv2_0 = ConvBlock(c[1], c[2])
        self.conv3_0 = ConvBlock(c[2], c[3])
        self.conv4_0 = ConvBlock(c[3], c[4])
        self.conv0_1 = ConvBlock(c[0] + c[1], c[0])
        self.conv1_1 = ConvBlock(c[1] + c[2], c[1])
        self.conv2_1 = ConvBlock(c[2] + c[3], c[2])
        self.conv3_1 = ConvBlock(c[3] + c[4], c[3])
        self.conv0_2 = ConvBlock(c[0] * 2 + c[1], c[0])
        self.conv1_2 = ConvBlock(c[1] * 2 + c[2], c[1])
        self.conv2_2 = ConvBlock(c[2] * 2 + c[3], c[2])
        self.conv0_3 = ConvBlock(c[0] * 3 + c[1], c[0])
        self.conv1_3 = ConvBlock(c[1] * 3 + c[2], c[1])
        self.conv0_4 = ConvBlock(c[0] * 4 + c[1], c[0])
        if deep_supervision:
            self.heads = nn.ModuleList([nn.Conv2d(c[0], num_classes, 1) for _ in range(4)])
        else:
            self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))
        if self.deep_supervision:
            return [head(feature) for head, feature in zip(self.heads, [x0_1, x0_2, x0_3, x0_4])]
        return self.head(x0_4)

    def _up(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=True)
