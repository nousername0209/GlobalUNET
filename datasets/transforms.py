from __future__ import annotations

import random

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class SegmentationTransform:
    def __init__(self, image_size: int, train: bool, augment: dict | None = None) -> None:
        self.image_size = image_size
        self.train = train
        self.augment = augment or {}

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train and self.augment.get("horizontal_flip", True) and random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        image = F.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.image_size, self.image_size], InterpolationMode.NEAREST)
        if self.train and self.augment.get("color_jitter", True):
            image = F.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = F.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = F.adjust_saturation(image, random.uniform(0.8, 1.2))
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_tensor = torch.as_tensor(list(mask.getdata()), dtype=torch.long).view(mask.height, mask.width)
        return image_tensor, mask_tensor
