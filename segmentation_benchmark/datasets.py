from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, VOCSegmentation

from .config import ExperimentConfig
from .seed import seed_worker
from .transforms import SegmentationTransform


CITYSCAPES_ID_TO_TRAIN_ID = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}


class VOC2012Segmentation(VOCSegmentation):
    def __init__(self, root: Path, image_set: str, transform_pair: SegmentationTransform, download: bool) -> None:
        super().__init__(
            root=str(root),
            year="2012",
            image_set=image_set,
            download=download,
            transforms=None,
        )
        self.transform_pair = transform_pair

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])
        return self.transform_pair(image, mask)


class CityscapesSegmentation(Cityscapes):
    def __init__(self, root: Path, split: str, transform_pair: SegmentationTransform) -> None:
        super().__init__(
            root=str(root),
            split=split,
            mode="fine",
            target_type="semantic",
            transforms=None,
        )
        self.transform_pair = transform_pair

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.targets[index][0])
        mask = _cityscapes_to_train_ids(mask)
        return self.transform_pair(image, mask)


def _cityscapes_to_train_ids(mask: Image.Image) -> Image.Image:
    array = np.array(mask, dtype=np.uint8)
    mapped = np.full_like(array, 255, dtype=np.uint8)
    for raw_id, train_id in CITYSCAPES_ID_TO_TRAIN_ID.items():
        mapped[array == raw_id] = train_id
    return Image.fromarray(mapped)


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    train_transform = SegmentationTransform(config.image_size, train=True)
    eval_transform = SegmentationTransform(config.image_size, train=False)
    generator = torch.Generator().manual_seed(config.seed)

    if config.dataset == "voc2012":
        train_dataset = VOC2012Segmentation(config.data_root, "train", train_transform, config.download_voc)
        val_dataset = VOC2012Segmentation(config.data_root, "val", eval_transform, config.download_voc)
    elif config.dataset == "cityscapes":
        train_dataset = CityscapesSegmentation(config.data_root, "train", train_transform)
        val_dataset = CityscapesSegmentation(config.data_root, config.cityscapes_split, eval_transform)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader
