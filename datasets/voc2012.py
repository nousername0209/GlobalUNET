from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation

from .transforms import SegmentationTransform


class VOC2012Segmentation(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, train: bool, augment: dict | None = None, download: bool = False) -> None:
        self.dataset = VOCSegmentation(root=str(root), year="2012", image_set=split, download=download, transforms=None)
        self.transform = SegmentationTransform(image_size=image_size, train=train, augment=augment)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.dataset.images[index]).convert("RGB")
        mask = Image.open(self.dataset.masks[index])
        return self.transform(image, mask)
