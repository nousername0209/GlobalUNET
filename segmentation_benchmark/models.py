from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, features: tuple[int, ...] = (64, 128, 256, 512)) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        current_channels = in_channels

        for feature in features:
            self.down_blocks.append(ConvBlock(current_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current_channels = features[-1] * 2
        for feature in reversed(features):
            self.up_transpose.append(nn.ConvTranspose2d(current_channels, feature, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(feature * 2, feature))
            current_channels = feature

        self.classifier = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for transpose, block, skip in zip(self.up_transpose, self.up_blocks, skips):
            x = transpose(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.classifier(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, features: tuple[int, ...] = (64, 128, 256, 512, 1024)) -> None:
        super().__init__()
        filters = features
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])

        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[3], filters[2])

        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[2], filters[1])

        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[1], filters[0])
        self.classifier = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up_to(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up_to(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up_to(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up_to(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up_to(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up_to(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up_to(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up_to(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up_to(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up_to(x1_3, x0_0)], dim=1))

        return self.classifier(x0_4)

    def _up_to(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=True)


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "unet":
        return UNet(num_classes=num_classes)
    if name == "unetpp":
        return UNetPlusPlus(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
