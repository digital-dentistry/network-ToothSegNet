from __future__ import annotations

import torch.nn.functional as F

from interface.strategy import IStrategy

from ..common.class_dice_loss import class_dice_loss
from ..common.offset_direction_loss import offset_direction_loss
from ..common.offset_loss import offset_loss
from ..loss import LossInput, LossOutput


class SemanticLoss(IStrategy[LossInput, LossOutput]):
    """The loss of `toothseg3d/train.py`: the 17-class model whose backbone
    initialises the instance model.

    Weighted cross-entropy + class dice for the labels, L1 + cosine direction
    for the centroid offsets on tooth points.  Points of class 17 (a tooth of
    unknown position) are left out: the 17-class target has no slot for them.
    """

    def execute(self, inp: LossInput, out: LossOutput) -> bool:
        if inp.class_logits is None or inp.classes is None:
            raise ValueError("SEMANTIC needs class_logits and classes")
        w = inp.weights
        logits = inp.class_logits.float()
        n_cls = logits.shape[1]
        known = inp.classes < n_cls
        logits, classes = logits[known], inp.classes[known]
        cw = None if inp.class_weights is None else inp.class_weights.to(logits)
        l_ce = F.cross_entropy(logits, classes, weight=cw)
        l_dice = class_dice_loss(logits, classes, n_cls)
        l_off = l_dir = logits.sum() * 0
        if inp.offsets is not None and inp.offset_targets is not None:
            off, tgt = inp.offsets.float()[known], inp.offset_targets.float()[known]
            tooth = classes > 0
            l_off = offset_loss(off, tgt, tooth)
            l_dir = offset_direction_loss(off, tgt, tooth)
        out.total = l_ce + w.class_dice * l_dice + w.offset * l_off + w.direction * l_dir
        out.terms = {k: v.detach() for k, v in (("ce", l_ce), ("class_dice", l_dice),
                                                ("offset", l_off), ("direction", l_dir))}
        out.matches = []
        return True
