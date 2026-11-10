from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy


@dataclass
class LossWeights:
    """MASK_TRANSFORMER: `train_inst.py` defaults, InstanceCriterion(w_cls=0.5,
    w_bce=5.0, w_dice=2.0, no_obj=0.1) and --w-aux / --w-sem / --w-off.
    SEMANTIC: `train.py` defaults (--w-dice / --w-off / --w-dir); use
    `LossWeights.semantic()`, its offset weight differs."""

    cls: float = 0.5
    bce: float = 5.0
    dice: float = 2.0
    no_object: float = 0.1       # class weight of "nothing here": most queries own no tooth
    aux: float = 0.5             # deep supervision on every decoder layer's coarse mask
    position: float = 0.5        # auxiliary arch-position head
    offset: float = 0.05         # centroid-offset head
    class_dice: float = 1.0      # SEMANTIC: soft dice over the 17 classes
    direction: float = 0.2       # SEMANTIC: cosine on the offset direction

    @staticmethod
    def semantic() -> LossWeights:
        return LossWeights(offset=0.1)


@dataclass
class LossInput:
    instances: torch.Tensor                  # [N] int64 target: 0 gingiva, one positive id per tooth of a scan
    counts: list[int]                        # L0 points per scan
    # MASK_TRANSFORMER: the query head
    cls_logits: torch.Tensor | None = None   # [B,K,2] per query: [is a tooth, nothing here]
    mask_logits: torch.Tensor | None = None  # [N,K] per L0 point, packed scan by scan
    # deep supervision: each decoder layer's coarse masks and the level they live on
    aux_mask_logits: list[torch.Tensor] = field(default_factory=list)   # per layer [B,K,M]
    coarse_inverse: torch.Tensor | None = None       # [N] L0 point -> packed cross-attention point
    pad_index: torch.Tensor | None = None            # [B,M] that level as a rectangular batch
    pad_valid: torch.Tensor | None = None            # [B,M]
    # auxiliary heads
    position_logits: torch.Tensor | None = None      # [N,9]
    arch_positions: torch.Tensor | None = None       # [N] int64 0..8, -1 unknown (targets.arch_position)
    offsets: torch.Tensor | None = None              # [N,3]
    offset_targets: torch.Tensor | None = None       # [N,3] (targets.offset_targets)
    # SEMANTIC: one class per point
    class_logits: torch.Tensor | None = None         # [N,17]
    classes: torch.Tensor | None = None              # [N] int64 0..16 (17, unknown position, is not trained here)
    class_weights: torch.Tensor | None = None        # [17] cross-entropy weights (targets.class_weights)
    weights: LossWeights = field(default_factory=LossWeights)


@dataclass
class LossOutput:
    total: torch.Tensor = field(default_factory=lambda: torch.zeros(()))     # backward() this
    # detached, in the units the toothseg3d training scripts log.  MASK_TRANSFORMER:
    # bce / dice / cls already weighted, aux / position / offset not.  SEMANTIC:
    # none weighted.  `total` applies whatever is not.
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    matches: list[tuple[torch.Tensor, torch.Tensor] | None] = field(default_factory=list)


class LossType(Enum):
    MASK_TRANSFORMER = 0         # Hungarian matching + BCE + dice + query class + deep supervision + aux heads
    SEMANTIC = 1                 # weighted CE + class dice + offset L1 + offset direction (pre-training)


LossStrategy = IStrategy[LossInput, LossOutput]


def _make_mask_transformer() -> LossStrategy:
    from .strategy.mask_transformer_loss import MaskTransformerLoss
    return MaskTransformerLoss()


def _make_semantic() -> LossStrategy:
    from .strategy.semantic_loss import SemanticLoss
    return SemanticLoss()


class Loss(StrategyAlgorithm[LossInput, LossOutput, LossType]):
    """Network outputs + per-point targets -> a scalar loss to backpropagate.

    No CPU / GPU split: a loss runs where its inputs and their autograd graph
    already are.  Only the Hungarian solve visits the host.
    """

    def __init__(self) -> None:
        super().__init__({
            LossType.MASK_TRANSFORMER: _make_mask_transformer,
            LossType.SEMANTIC: _make_semantic,
        })
