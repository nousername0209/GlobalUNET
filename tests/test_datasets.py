from __future__ import annotations

import torch
from PIL import Image

from datasets.cityscapes import map_cityscapes_to_train_ids
from datasets.transforms import SegmentationTransform


def test_transform_output_format() -> None:
    image = Image.new("RGB", (32, 32), color=(128, 64, 32))
    mask = Image.new("L", (32, 32), color=1)
    transform = SegmentationTransform(image_size=16, train=False)
    image_tensor, mask_tensor = transform(image, mask)
    assert image_tensor.shape == (3, 16, 16)
    assert mask_tensor.shape == (16, 16)
    assert image_tensor.dtype == torch.float32
    assert mask_tensor.dtype == torch.long


def test_cityscapes_label_mapping() -> None:
    mask = Image.fromarray(torch.tensor([[7, 8], [0, 255]], dtype=torch.uint8).numpy())
    mapped = torch.as_tensor(list(map_cityscapes_to_train_ids(mask).getdata()), dtype=torch.long).view(2, 2)
    assert mapped.tolist() == [[0, 1], [255, 255]]
