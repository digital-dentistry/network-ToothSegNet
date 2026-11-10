from __future__ import annotations

from common.device import CPU
from interface.strategy import IStrategy

from ..common.matching import fuse_poses, query_confidence
from ..query_fusion import QueryFusionInput, QueryFusionOutput


class HungarianFusionCPU(IStrategy[QueryFusionInput, QueryFusionOutput]):
    def execute(self, inp: QueryFusionInput, out: QueryFusionOutput) -> bool:
        conf, prob = query_confidence(inp.cls_logits.to(CPU), inp.mask_logits.to(CPU), inp.n_points)
        # Scores come from pose 0 alone.  The original meant to average them
        # too, but that line is disabled (`... if False else scores`); kept so
        # the AP ranking matches.
        out.confidence, out.scores = fuse_poses(conf), prob[0]
        return True
