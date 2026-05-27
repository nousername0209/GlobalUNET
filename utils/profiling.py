from __future__ import annotations

import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.no_grad()
def inference_latency_ms(model: nn.Module, sample: torch.Tensor, device: torch.device, warmup: int = 10, iterations: int = 50) -> float:
    model.eval()
    sample = sample.to(device)
    for _ in range(warmup):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    images = 0
    for _ in range(iterations):
        _ = model(sample)
        images += sample.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / max(images, 1)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)
