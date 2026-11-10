from __future__ import annotations

import torch

from ..instance_extraction import InstanceExtractionInput, InstanceExtractionOutput
from .labels import assign, erode, instance_scores, renumber


def extract(inp: InstanceExtractionInput, out: InstanceExtractionOutput, device: torch.device) -> None:
    conf = inp.confidence.to(device)
    active = inp.active.to(device)
    scores = inp.query_scores.to(device)
    n_vertex = inp.positions.shape[0]

    ins = assign(conf, active, n_vertex, inp.score_thresh)
    if inp.erode_mm > 0:
        ins = erode(ins, inp.positions.to(device), inp.erode_mm)
    ins, kept = renumber(ins, scores.numel(), inp.min_frac)
    out.instances = ins
    out.scores = instance_scores(ins, kept, scores, conf, active, inp.mask_confidence)
    out.vertex_confidence = (conf.amax(1) if active.numel()
                             else torch.zeros(n_vertex, device=device))
