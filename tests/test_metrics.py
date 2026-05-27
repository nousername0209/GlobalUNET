from __future__ import annotations

import torch

from utils.metrics import SegmentationMetrics


def test_metrics_perfect_prediction() -> None:
    logits = torch.tensor([[[[10.0, 0.0], [0.0, 10.0]], [[0.0, 10.0], [10.0, 0.0]]]])
    target = torch.tensor([[[0, 1], [1, 0]]])
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(logits, target)
    result = metrics.compute()
    assert result["miou"] == 1.0
    assert result["dice_f1"] == 1.0
    assert result["pixel_accuracy"] == 1.0
