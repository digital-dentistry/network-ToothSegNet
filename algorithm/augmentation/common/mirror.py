from __future__ import annotations

import torch

from .params import AugmentationParams
from .sample import AugmentationSample

UNKNOWN_TOOTH = 17
# mirroring a jaw swaps the two quadrants: class c in 1..8 <-> c + 8.  Class 17
# is "a tooth, position unknown" (in-house scans) and stays unknown.
MIRROR_MAP = torch.tensor([0] + list(range(9, 17)) + list(range(1, 9)) + [UNKNOWN_TOOTH])


def mirror(sample: AugmentationSample, params: AugmentationParams,
           gen: torch.Generator | None) -> AugmentationSample:
    """Reflect across the mid-sagittal plane (x -> -x); the quadrants swap with it.

    A reflection is not a rotation, so this still adds something on top of the
    SO(3) augmentation -- it is the one transform that changes which side of the
    patient a tooth is on.
    """
    flip = torch.tensor([-1.0, 1.0, 1.0], device=sample.device)
    return sample.replace(positions=sample.positions * flip, normals=sample.normals * flip,
                          classes=MIRROR_MAP.to(sample.device)[sample.classes])
