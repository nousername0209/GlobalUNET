from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes

from .transforms import SegmentationTransform


CITYSCAPES_ID_TO_TRAIN_ID = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}


class CityscapesSegmentation(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, train: bool, augment: dict | None = None) -> None:
        self.dataset = Cityscapes(root=str(root), split=split, mode="fine", target_type="semantic", transforms=None)
        self.transform = SegmentationTransform(image_size=image_size, train=train, augment=augment)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.dataset.images[index]).convert("RGB")
        mask = Image.open(self.dataset.targets[index][0])
        mask = map_cityscapes_to_train_ids(mask)
        return self.transform(image, mask)


def map_cityscapes_to_train_ids(mask: Image.Image) -> Image.Image:
    array = np.array(mask, dtype=np.uint8)
    mapped = np.full_like(array, 255, dtype=np.uint8)
    for label_id, train_id in CITYSCAPES_ID_TO_TRAIN_ID.items():
        mapped[array == label_id] = train_id
    return Image.fromarray(mapped)
