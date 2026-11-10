from __future__ import annotations

import torch
import torch.nn.functional as F

from interface.strategy import IStrategy

from ..common.dice_loss import dice_loss
from ..common.mask_bce_loss import mask_bce_loss
from ..common.matching import hungarian, match_cost
from ..common.offset_loss import offset_loss
from ..common.targets import coarse_targets, instance_targets
from ..loss import LossInput, LossOutput


class MaskTransformerLoss(IStrategy[LossInput, LossOutput]):
    """`InstanceCriterion` plus the auxiliary terms of `train_inst.py`'s loop.

    Each query is matched to at most one tooth by the Hungarian algorithm on
    mask BCE + dice - P(tooth); matched queries learn their tooth's mask and
    "tooth", the rest learn "nothing here".  Tooth identity never enters, so
    there is no left/right or neighbour confusion to learn around.
    """

    def execute(self, inp: LossInput, out: LossOutput) -> bool:
        if inp.cls_logits is None or inp.mask_logits is None:
            raise ValueError("MASK_TRANSFORMER needs cls_logits and mask_logits")
        w = inp.weights
        cls_logit = inp.cls_logits.float()                          # [B,K,2]
        b_n, k_n, _ = cls_logit.shape
        prob = cls_logit.softmax(-1)
        targets = instance_targets(inp.instances, inp.counts)

        l_bce = cls_logit.new_zeros(())
        l_dice = cls_logit.new_zeros(())
        tgt_cls = torch.ones(b_n, k_n, dtype=torch.long, device=cls_logit.device)
        matches: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        n_match = 0
        for b, mlog in enumerate(torch.split(inp.mask_logits, inp.counts)):
            mlog, t = mlog.t().float(), targets[b]                  # [K,n], [G,n]
            if t.numel() == 0:
                matches.append(None)
                continue
            r, c = hungarian(match_cost(mlog, prob[b, :, 0], t, w.bce, w.dice, w.cls))
            matches.append((r, c))
            tgt_cls[b, r] = 0
            l_bce = l_bce + mask_bce_loss(mlog[r], t[c])
            l_dice = l_dice + dice_loss(mlog[r], t[c])
            n_match += 1
        n_match = max(n_match, 1)
        cls_w = torch.tensor([1.0, w.no_object], device=cls_logit.device)
        l_cls = F.cross_entropy(cls_logit.reshape(-1, 2), tgt_cls.reshape(-1), weight=cls_w)

        l_aux = self._deep_supervision(inp, targets, matches, cls_logit)
        l_pos = l_off = cls_logit.new_zeros(())
        if inp.position_logits is not None and inp.arch_positions is not None:
            l_pos = F.cross_entropy(inp.position_logits.float(), inp.arch_positions, ignore_index=-1)
        if inp.offsets is not None and inp.offset_targets is not None:
            # only teeth of known position: an in-house tooth (-1) trains the
            # instance head but not the auxiliary heads
            tooth = (inp.arch_positions > 0 if inp.arch_positions is not None else inp.instances > 0)
            l_off = offset_loss(inp.offsets.float(), inp.offset_targets.float(), tooth)

        bce, dice, cls = w.bce * l_bce / n_match, w.dice * l_dice / n_match, w.cls * l_cls
        out.total = bce + dice + cls + w.aux * l_aux + w.position * l_pos + w.offset * l_off
        out.terms = {k: v.detach() for k, v in (("bce", bce), ("dice", dice), ("cls", cls), ("aux", l_aux),
                                                ("position", l_pos), ("offset", l_off))}
        out.matches = matches
        return True

    @staticmethod
    def _deep_supervision(inp: LossInput, targets: list[torch.Tensor],
                          matches: list[tuple[torch.Tensor, torch.Tensor] | None],
                          like: torch.Tensor) -> torch.Tensor:
        """Every decoder layer's coarse masks against the coarse targets, reusing
        the final layer's matching -- without it the early layers get no direct
        signal and the stack learns slowly (Mask2Former)."""
        l_aux = like.new_zeros(())
        if not inp.aux_mask_logits or inp.coarse_inverse is None or inp.pad_index is None:
            return l_aux
        coarse = coarse_targets(targets, inp.coarse_inverse, inp.counts, inp.pad_index)
        n_aux = 0
        for layer in inp.aux_mask_logits:                           # [B,K,M]
            for b, mt in enumerate(matches):
                if mt is None:
                    continue
                r, c = mt
                v = inp.pad_valid[b]
                sel, gt = layer[b].float()[r][:, v], coarse[b][c][:, v]
                l_aux = l_aux + mask_bce_loss(sel, gt) + dice_loss(sel, gt)
                n_aux += 1
        return l_aux / max(n_aux, 1)
