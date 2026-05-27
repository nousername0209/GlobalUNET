from __future__ import annotations

import torch

from models.unet import UNet
from models.unetpp import UNetPlusPlus


def test_unet_forward_shape() -> None:
    model = UNet(num_classes=21, channels=(8, 16, 32, 64))
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 21, 64, 64)


def test_unetpp_forward_shape_without_deep_supervision() -> None:
    model = UNetPlusPlus(num_classes=19, channels=(8, 16, 32, 64, 128), deep_supervision=False)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 19, 64, 64)


def test_unetpp_forward_shape_with_deep_supervision() -> None:
    model = UNetPlusPlus(num_classes=19, channels=(8, 16, 32, 64, 128), deep_supervision=True)
    outputs = model(torch.randn(2, 3, 64, 64))
    assert isinstance(outputs, list)
    assert len(outputs) == 4
    assert all(output.shape == (2, 19, 64, 64) for output in outputs)
