from __future__ import annotations

import argparse
import csv
from pathlib import Path


def evaluate(config: dict) -> dict[str, float | int | str]:
    import torch
    from datasets import build_dataloader
    from models import build_model
    from train import evaluate_loop
    from utils.profiling import count_parameters, inference_latency_ms, peak_memory_mb
    from utils.seed import seed_everything

    training_cfg = config["training"]
    run_dir = Path(config["output"]["run_dir"])
    seed_everything(training_cfg["seed"])
    device = torch.device(training_cfg["device"] if torch.cuda.is_available() or training_cfg["device"] == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loader = build_dataloader(config, config["dataset"]["val_split"])
    model = build_model(config).to(device)
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    metrics = evaluate_loop(model, loader, config, device)
    sample, _ = next(iter(loader))
    result = {"experiment_name": config["experiment_name"], "dataset": config["dataset"]["name"], "model": config["model"]["name"], "deep_supervision": config["model"].get("deep_supervision", False), "miou": metrics["miou"], "dice_f1": metrics["dice_f1"], "pixel_accuracy": metrics["pixel_accuracy"], "parameter_count": count_parameters(model), "inference_latency_ms_per_image": inference_latency_ms(model, sample, device, warmup=config["profiling"]["warmup"], iterations=config["profiling"]["iterations"]), "gpu_memory_mb": peak_memory_mb(device)}
    write_csv(run_dir / "metrics.csv", result)
    return result


def write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    from utils.config import load_config
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
