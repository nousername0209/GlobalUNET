from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import ExperimentConfig
from .datasets import build_dataloaders
from .metrics import SegmentationMetrics
from .models import build_model, count_parameters
from .seed import seed_everything


def run_experiment(config: ExperimentConfig) -> dict[str, float | int | str]:
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    config.run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config.model, config.num_classes).to(device)
    parameter_count = count_parameters(model)

    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")

    best_miou = -1.0
    last_metrics: dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config.amp)
        scheduler.step()
        last_metrics = evaluate(model, val_loader, config.num_classes, config.ignore_index, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **last_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        _append_csv(config.run_dir / "history.csv", row)

        if last_metrics["miou"] > best_miou:
            best_miou = last_metrics["miou"]
            torch.save(model.state_dict(), config.run_dir / "best_model.pt")

    inference_ms = measure_inference_time(model, val_loader, device, config.inference_warmup, config.inference_iters)
    result = {
        "dataset": config.dataset,
        "model": config.model,
        "seed": config.seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "image_size": config.image_size,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "miou": last_metrics.get("miou", 0.0),
        "dice_f1": last_metrics.get("dice_f1", 0.0),
        "pixel_accuracy": last_metrics.get("pixel_accuracy", 0.0),
        "parameter_count": parameter_count,
        "inference_time_ms_per_image": inference_ms,
    }
    _write_csv(config.run_dir / "metrics.csv", result)
    return result


def _train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_images = 0

    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)

    return total_loss / max(total_images, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics = SegmentationMetrics(num_classes, ignore_index)
    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        metrics.update(logits, masks)
    return metrics.compute()


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    model.eval()
    sample_images, _ = next(iter(loader))
    sample_images = sample_images.to(device)

    for _ in range(warmup):
        _ = model(sample_images)
    if device.type == "cuda":
        torch.cuda.synchronize()

    image_count = 0
    start = time.perf_counter()
    for index in range(iterations):
        _ = model(sample_images)
        image_count += sample_images.size(0)
        if index + 1 >= iterations:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / max(image_count, 1)) * 1000


def _append_csv(path: Path, row: dict[str, float | int | str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_csv(path: Path, row: dict[str, float | int | str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
