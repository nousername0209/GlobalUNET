from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = torch.argmax(logits, dim=1).detach().cpu()
        target = target.detach().cpu()
        valid = target != self.ignore_index
        target = target[valid]
        prediction = prediction[valid]

        if target.numel() == 0:
            return

        indices = self.num_classes * target + prediction
        matrix = torch.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += matrix.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        matrix = self.confusion_matrix.float()
        true_positive = torch.diag(matrix)
        false_positive = matrix.sum(dim=0) - true_positive
        false_negative = matrix.sum(dim=1) - true_positive

        union = true_positive + false_positive + false_negative
        iou = true_positive / union.clamp_min(1)
        dice = (2 * true_positive) / (2 * true_positive + false_positive + false_negative).clamp_min(1)
        pixel_accuracy = true_positive.sum() / matrix.sum().clamp_min(1)

        valid_iou = union > 0
        valid_dice = (2 * true_positive + false_positive + false_negative) > 0
        return {
            "miou": iou[valid_iou].mean().item() if valid_iou.any() else 0.0,
            "dice_f1": dice[valid_dice].mean().item() if valid_dice.any() else 0.0,
            "pixel_accuracy": pixel_accuracy.item(),
        }
