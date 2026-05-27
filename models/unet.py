from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    """Simple U-Net, reimplemented in PyTorch from the common encoder-decoder design."""

    def __init__(self, num_classes: int, in_channels: int = 3, channels: tuple[int, ...] = (64, 128, 256, 512)) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        current = in_channels
        for width in channels:
            self.encoders.append(ConvBlock(current, width))
            self.pools.append(nn.MaxPool2d(2))
            current = width
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = channels[-1] * 2
        for width in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(current, width, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(width * 2, width))
            current = width
        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
        return self.head(x)
