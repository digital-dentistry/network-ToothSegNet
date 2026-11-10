from __future__ import annotations

from common.device import cuda_device
from interface.strategy import IStrategy

from ..common.matching import fuse_poses, query_confidence
from ..query_fusion import QueryFusionInput, QueryFusionOutput


class HungarianFusionGPU(IStrategy[QueryFusionInput, QueryFusionOutput]):
    """IoU matmuls on the device; only each 32 x 32 IoU matrix goes to the host
    for the assignment."""

    def __init__(self) -> None:
        self._device = cuda_device()

    def execute(self, inp: QueryFusionInput, out: QueryFusionOutput) -> bool:
        conf, prob = query_confidence(inp.cls_logits.to(self._device),
                                      inp.mask_logits.to(self._device), inp.n_points)
        # scores from pose 0 alone, as the original (see the CPU strategy)
        out.confidence, out.scores = fuse_poses(conf), prob[0]
        return True
