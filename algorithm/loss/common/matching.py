"""Hungarian assignment of queries to ground-truth teeth (port of
`InstanceCriterion.match`)."""
from __future__ import annotations

import torch

from .dice_loss import dice_cost
from .mask_bce_loss import bce_cost


def match_cost(mask_logit: torch.Tensor, tooth_prob: torch.Tensor, target: torch.Tensor,
               w_bce: float, w_dice: float, w_cls: float) -> torch.Tensor:
    """[K,G] cost of giving query k tooth g: mask BCE + mask dice - P(query is a tooth).

    mask_logit [K,n], tooth_prob [K], target [G,n].
    """
    return (w_bce * bce_cost(mask_logit, target) + w_dice * dice_cost(mask_logit, target)
            - w_cls * tooth_prob[:, None])


@torch.no_grad()
def hungarian(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimum-cost one-to-one assignment -> (query rows, target columns).

    scipy, on the host: torch has no assignment solver, and 32 x 16 is nothing.
    """
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(cost.float().cpu().numpy())
    return torch.as_tensor(r, device=cost.device), torch.as_tensor(c, device=cost.device)
