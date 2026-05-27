from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from utils.seed import seed_worker

from .cityscapes import CityscapesSegmentation
from .voc2012 import VOC2012Segmentation


def build_dataset(config: dict, split: str):
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    train = split == dataset_cfg["train_split"]
    common = {"root": dataset_cfg["root"], "split": split, "image_size": training_cfg["image_size"], "train": train, "augment": config.get("augmentation", {})}
    if dataset_cfg["name"] == "voc2012":
        return VOC2012Segmentation(**common, download=dataset_cfg.get("download", False))
    if dataset_cfg["name"] == "cityscapes":
        return CityscapesSegmentation(**common)
    raise ValueError(f"Unknown dataset: {dataset_cfg['name']}")


def build_dataloader(config: dict, split: str) -> DataLoader:
    dataset = build_dataset(config, split)
    training_cfg = config["training"]
    generator = torch.Generator().manual_seed(training_cfg["seed"])
    shuffle = split == config["dataset"]["train_split"]
    return DataLoader(dataset, batch_size=training_cfg["batch_size"], shuffle=shuffle, num_workers=training_cfg["num_workers"], pin_memory=True, worker_init_fn=seed_worker, generator=generator if shuffle else None)
