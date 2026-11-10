from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .average_precision import ScanRecord
from .challenge_metrics import jaw_metrics
from .confusion import confusion_matrix
from .instance_metrics import instance_iou, matched_summary, tooth_dice

if TYPE_CHECKING:
    from ..evaluation import EvaluationInput, EvaluationOutput


def instance_metrics(inp: EvaluationInput, out: EvaluationOutput, device: torch.device) -> bool:
    pred, gt = inp.pred_instances.to(device), inp.gt_instances.to(device)
    iou, pred_ids, _ = instance_iou(pred, gt, inp.min_pred_points, inp.min_gt_points)
    if inp.scores is None:
        scores = torch.ones(pred_ids.numel(), dtype=torch.float64)
    else:
        scores = inp.scores.to(device, torch.float64)[pred_ids - 1]
    pred_tooth = torch.isin(pred, pred_ids)              # the specks are not predictions
    out.record = ScanRecord(scores.cpu().numpy(), iou.cpu().numpy(), iou.shape[1])
    out.metrics = {"tsa": tooth_dice(pred_tooth, gt != 0), **matched_summary(iou)}
    return True


def challenge_metrics(inp: EvaluationInput, out: EvaluationOutput, device: torch.device) -> bool:
    if inp.positions is None or inp.gt_classes is None:
        raise ValueError("CHALLENGE needs positions and gt_classes")
    out.record = None
    out.metrics = jaw_metrics(
        inp.positions.to(device), inp.gt_classes.to(device), inp.gt_instances.to(device),
        inp.pred_instances.to(device), None if inp.pred_classes is None else inp.pred_classes.to(device))
    return True


def semantic_metrics(inp: EvaluationInput, out: EvaluationOutput, device: torch.device) -> bool:
    if inp.point_classes is None or inp.gt_classes is None:
        raise ValueError("SEMANTIC needs point_classes and gt_classes")
    out.record = None
    out.confusion = confusion_matrix(inp.point_classes.to(device), inp.gt_classes.to(device), inp.n_classes)
    cm = out.confusion.double()
    out.metrics = {"acc": float(cm.diagonal().sum() / cm.sum().clamp_min(1)),
                   "tsa": tooth_dice(inp.point_classes.to(device) > 0, inp.gt_classes.to(device) > 0)}
    return True
