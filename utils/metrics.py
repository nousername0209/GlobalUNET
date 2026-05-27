from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.argmax(logits, dim=1).detach().cpu()
        target = target.detach().cpu()
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        if target.numel() == 0:
            return
        encoded = self.num_classes * target + pred
        bincount = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        matrix = self.confusion_matrix.float()
        tp = torch.diag(matrix)
        fp = matrix.sum(dim=0) - tp
        fn = matrix.sum(dim=1) - tp
        union = tp + fp + fn
        iou = tp / union.clamp_min(1)
        dice = (2 * tp) / (2 * tp + fp + fn).clamp_min(1)
        pixel_accuracy = tp.sum() / matrix.sum().clamp_min(1)
        return {"miou": iou[union > 0].mean().item() if (union > 0).any() else 0.0, "dice_f1": dice[(2 * tp + fp + fn) > 0].mean().item() if ((2 * tp + fp + fn) > 0).any() else 0.0, "pixel_accuracy": pixel_accuracy.item()}
