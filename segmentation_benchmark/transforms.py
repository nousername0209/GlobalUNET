from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


@dataclass
class SegmentationTransform:
    image_size: int
    train: bool

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train and random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        image = F.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.image_size, self.image_size], InterpolationMode.NEAREST)

        if self.train:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            image = F.adjust_brightness(image, brightness)
            image = F.adjust_contrast(image, contrast)
            image = F.adjust_saturation(image, saturation)

        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        mask_tensor = torch.as_tensor(list(mask.getdata()), dtype=torch.long).view(mask.height, mask.width)
        return image_tensor, mask_tensor
