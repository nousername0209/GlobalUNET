from __future__ import annotations

import torch
from torch import nn


def cross_entropy_loss(outputs: torch.Tensor | list[torch.Tensor], target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    if isinstance(outputs, list):
        return sum(criterion(output, target) for output in outputs) / len(outputs)
    return criterion(outputs, target)
