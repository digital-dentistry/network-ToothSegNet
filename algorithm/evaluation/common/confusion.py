"""Per-point class metrics from a confusion matrix (port of `train.SegMeter`)."""
from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(pred: torch.Tensor, gt: torch.Tensor, n_classes: int) -> torch.Tensor:
    """[C,C] int64 counts, rows = true class, columns = predicted.  Points whose
    true class is outside 0..C-1 (17, a tooth of unknown position) are left out."""
    known = (gt >= 0) & (gt < n_classes)
    k = gt[known] * n_classes + pred[known]
    return torch.bincount(k, minlength=n_classes * n_classes).view(n_classes, n_classes)


class ConfusionMeter:
    def __init__(self, n_classes: int) -> None:
        self._cm = np.zeros((n_classes, n_classes), np.int64)

    def add(self, confusion: torch.Tensor) -> None:
        self._cm += confusion.cpu().numpy()

    def result(self) -> dict[str, object]:
        cm = self._cm
        tp = np.diag(cm).astype(np.float64)
        iou = tp / np.maximum(cm.sum(1) + cm.sum(0) - tp, 1)
        seen = cm.sum(1) > 0
        # TSA: the binary tooth / gingiva F1, the challenge's segmentation metric
        gt_t, pr_t, tp_t = cm[1:, :].sum(), cm[:, 1:].sum(), cm[1:, 1:].sum()
        return {"miou": float(iou[seen].mean()) if seen.any() else 0.0,
                "acc": float(tp.sum() / max(cm.sum(), 1)),
                "tsa": float(2 * tp_t / max(gt_t + pr_t, 1)), "iou": iou.tolist(), "seen": seen.tolist()}
