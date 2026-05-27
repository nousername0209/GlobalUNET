from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def train(config: dict) -> dict[str, float | int | str]:
    import torch
    from torch.cuda.amp import GradScaler
    from datasets import build_dataloader
    from models import build_model
    from utils.profiling import count_parameters, peak_memory_mb
    from utils.seed import seed_everything

    training_cfg = config["training"]
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    seed_everything(training_cfg["seed"])
    device = torch.device(training_cfg["device"] if torch.cuda.is_available() or training_cfg["device"] == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_loader = build_dataloader(config, config["dataset"]["train_split"])
    val_loader = build_dataloader(config, config["dataset"]["val_split"])
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg["learning_rate"], weight_decay=training_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["epochs"])
    scaler = GradScaler(enabled=training_cfg["amp"] and device.type == "cuda")
    best_miou = -1.0
    last_metrics = {}
    epoch_times = []

    for epoch in range(1, training_cfg["epochs"] + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, config, device)
        scheduler.step()
        last_metrics = evaluate_loop(model, val_loader, config, device)
        epoch_time = time.perf_counter() - start
        epoch_times.append(epoch_time)
        row = {"epoch": epoch, "train_loss": train_loss, "training_time_sec_per_epoch": epoch_time, "lr": scheduler.get_last_lr()[0], **last_metrics}
        append_csv(run_dir / "logs" / "history.csv", row)
        if last_metrics["miou"] > best_miou:
            best_miou = last_metrics["miou"]
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pt")
        torch.save(model.state_dict(), run_dir / "checkpoints" / "last.pt")

    summary = {"experiment_name": config["experiment_name"], "dataset": config["dataset"]["name"], "model": config["model"]["name"], "deep_supervision": config["model"].get("deep_supervision", False), "seed": training_cfg["seed"], "epochs": training_cfg["epochs"], "batch_size": training_cfg["batch_size"], "image_size": training_cfg["image_size"], "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "learning_rate": training_cfg["learning_rate"], "weight_decay": training_cfg["weight_decay"], "miou": last_metrics.get("miou", 0.0), "dice_f1": last_metrics.get("dice_f1", 0.0), "pixel_accuracy": last_metrics.get("pixel_accuracy", 0.0), "parameter_count": count_parameters(model), "training_time_sec_per_epoch": sum(epoch_times) / max(len(epoch_times), 1), "gpu_memory_mb": peak_memory_mb(device)}
    write_csv(run_dir / "train_summary.csv", summary)
    return summary


def train_one_epoch(model, loader, optimizer, scaler, config: dict, device) -> float:
    from torch.cuda.amp import autocast
    from tqdm import tqdm
    from utils.losses import cross_entropy_loss
    model.train()
    total_loss = 0.0
    total_images = 0
    amp = config["training"]["amp"] and device.type == "cuda"
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            outputs = model(images)
            loss = cross_entropy_loss(outputs, masks, config["dataset"]["ignore_index"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)
    return total_loss / max(total_images, 1)


def evaluate_loop(model, loader, config: dict, device) -> dict[str, float]:
    import torch
    from tqdm import tqdm
    from utils.metrics import SegmentationMetrics
    model.eval()
    metrics = SegmentationMetrics(config["dataset"]["num_classes"], config["dataset"]["ignore_index"])
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            logits = outputs[-1] if isinstance(outputs, list) else outputs
            metrics.update(logits, masks)
    return metrics.compute()


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, row: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a segmentation model from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    from utils.config import load_config
    train(load_config(args.config))


if __name__ == "__main__":
    main()
