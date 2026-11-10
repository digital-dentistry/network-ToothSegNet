from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from .params import AugmentationParams
from .sample import AugmentationSample

if TYPE_CHECKING:
    from ..augmentation import AugmentationInput, AugmentationOutput

AugmentationOp = Callable[[AugmentationSample, AugmentationParams, torch.Generator | None],
                          AugmentationSample | None]


def apply_op(op: AugmentationOp, inp: AugmentationInput, out: AugmentationOutput,
             device: torch.device) -> bool:
    """Run `op` on the sample moved to `device`.  An op that declines -- too few
    points or teeth left -- returns None, and the sample passes through."""
    if inp.generator is not None and inp.generator.device.type != "cpu":
        raise ValueError("pass a CPU torch.Generator; GPU strategies seed their own from it")
    sample = inp.sample.to(device)
    result = op(sample, inp.params, inp.generator)
    out.applied = result is not None
    out.sample = sample if result is None else result
    return True
