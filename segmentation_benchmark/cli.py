from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig


def parse_experiment_args(dataset: str, model: str) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=f"Run {model} on {dataset}.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--download-voc", action="store_true")
    parser.add_argument("--cityscapes-split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--inference-warmup", type=int, default=10)
    parser.add_argument("--inference-iters", type=int, default=50)
    args = parser.parse_args()

    return ExperimentConfig(
        dataset=dataset,
        model=model,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        amp=not args.no_amp,
        download_voc=args.download_voc,
        cityscapes_split=args.cityscapes_split,
        inference_warmup=args.inference_warmup,
        inference_iters=args.inference_iters,
    )
