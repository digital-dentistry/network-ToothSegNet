from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy

from .common.average_precision import ScanRecord


@dataclass
class EvaluationInput:
    pred_instances: torch.Tensor             # [V] int64 per vertex: 0 gingiva, i = instance i
    gt_instances: torch.Tensor               # [V] int64, same vertices
    scores: torch.Tensor | None = None       # [P] score of instance i at [i - 1]; None scores every one 1
    min_pred_points: int = 30                # smaller predictions are specks (train_inst's validation)
    min_gt_points: int = 1
    # CHALLENGE only
    positions: torch.Tensor | None = None    # [V,3] mm
    gt_classes: torch.Tensor | None = None   # [V] int64 tooth label per vertex (FDI number or class)
    pred_classes: torch.Tensor | None = None  # [P] label of instance i at [i - 1]; None = identity-free model
    # SEMANTIC only (gt_classes above is the target)
    point_classes: torch.Tensor | None = None   # [V] int64 predicted class per point
    n_classes: int = 17


@dataclass
class EvaluationOutput:
    record: ScanRecord | None = None         # INSTANCE: feed InstanceAPMeter.add for mAP over scans
    confusion: torch.Tensor | None = None    # SEMANTIC: [C,C], feed ConfusionMeter.add for mIoU over scans
    metrics: dict[str, float] = field(default_factory=dict)


class EvaluationType(Enum):
    INSTANCE_CPU = 0             # IoU vs every true tooth, TSA, matched mean IoU; AP record
    INSTANCE_GPU = 1
    CHALLENGE_CPU = 2            # 3DTeethSeg'22 TSA / TLA / TIR, centroid error, misses
    CHALLENGE_GPU = 3
    SEMANTIC_CPU = 4             # per-point class confusion matrix; mIoU, accuracy, TSA
    SEMANTIC_GPU = 5


EvaluationStrategy = IStrategy[EvaluationInput, EvaluationOutput]


def _make_instance_cpu() -> EvaluationStrategy:
    from .strategy.instance_metrics_cpu import InstanceMetricsCPU
    return InstanceMetricsCPU()


def _make_instance_gpu() -> EvaluationStrategy:
    from .strategy.instance_metrics_gpu import InstanceMetricsGPU
    return InstanceMetricsGPU()


def _make_challenge_cpu() -> EvaluationStrategy:
    from .strategy.challenge_metrics_cpu import ChallengeMetricsCPU
    return ChallengeMetricsCPU()


def _make_challenge_gpu() -> EvaluationStrategy:
    from .strategy.challenge_metrics_gpu import ChallengeMetricsGPU
    return ChallengeMetricsGPU()


def _make_semantic_cpu() -> EvaluationStrategy:
    from .strategy.semantic_metrics_cpu import SemanticMetricsCPU
    return SemanticMetricsCPU()


def _make_semantic_gpu() -> EvaluationStrategy:
    from .strategy.semantic_metrics_gpu import SemanticMetricsGPU
    return SemanticMetricsGPU()


class Evaluation(StrategyAlgorithm[EvaluationInput, EvaluationOutput, EvaluationType]):
    """One scan's predicted instances against its ground truth -> metrics.

    Scores are per scan; mAP ranks predictions across scans, so INSTANCE also
    returns a record for `common.average_precision.InstanceAPMeter`, and
    SEMANTIC a confusion matrix for `common.confusion.ConfusionMeter`.
    """

    def __init__(self) -> None:
        super().__init__({
            EvaluationType.INSTANCE_CPU: _make_instance_cpu,
            EvaluationType.INSTANCE_GPU: _make_instance_gpu,
            EvaluationType.CHALLENGE_CPU: _make_challenge_cpu,
            EvaluationType.CHALLENGE_GPU: _make_challenge_gpu,
            EvaluationType.SEMANTIC_CPU: _make_semantic_cpu,
            EvaluationType.SEMANTIC_GPU: _make_semantic_gpu,
        })
