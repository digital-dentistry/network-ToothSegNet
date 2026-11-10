from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from interface.algorithm import StrategyAlgorithm
from interface.strategy import IStrategy

from .common.params import AugmentationParams
from .common.sample import AugmentationSample


@dataclass
class AugmentationInput:
    sample: AugmentationSample
    params: AugmentationParams = field(default_factory=AugmentationParams)
    generator: torch.Generator | None = None     # CPU; None draws from torch's global RNG


@dataclass
class AugmentationOutput:
    sample: AugmentationSample | None = None
    applied: bool = False            # False: the op declined and `sample` is the input, unchanged


class AugmentationType(Enum):
    MIRROR_CPU = 0                   # x -> -x, quadrants swap
    MIRROR_GPU = 1
    RANDOM_SO3_CPU = 2               # uniform rotation + scale + jitter
    RANDOM_SO3_GPU = 3
    SMALL_ROTATION_CPU = 4           # +-10 deg tilt, +-25 deg turn + scale + jitter
    SMALL_ROTATION_GPU = 5
    WEDGE_CROP_CPU = 6               # angular wedge of the arch
    WEDGE_CROP_GPU = 7
    SEGMENT_CROP_CPU = 8             # bridge / quadrant / half arch, whole teeth
    SEGMENT_CROP_GPU = 9
    GINGIVA_TRIM_CPU = 10            # drop gingiva beyond a random reach of the teeth
    GINGIVA_TRIM_GPU = 11
    SYNTH_COLOUR_CPU = 12            # paint vertex colour
    SYNTH_COLOUR_GPU = 13
    TRAIN_RECIPE_CPU = 14            # all of the above in toothseg3d's order and probabilities
    TRAIN_RECIPE_GPU = 15


AugmentationStrategy = IStrategy[AugmentationInput, AugmentationOutput]


def _make_mirror_cpu() -> AugmentationStrategy:
    from .strategy.mirror_cpu import MirrorCPU
    return MirrorCPU()


def _make_mirror_gpu() -> AugmentationStrategy:
    from .strategy.mirror_gpu import MirrorGPU
    return MirrorGPU()


def _make_random_so3_cpu() -> AugmentationStrategy:
    from .strategy.random_so3_cpu import RandomSO3CPU
    return RandomSO3CPU()


def _make_random_so3_gpu() -> AugmentationStrategy:
    from .strategy.random_so3_gpu import RandomSO3GPU
    return RandomSO3GPU()


def _make_small_rotation_cpu() -> AugmentationStrategy:
    from .strategy.small_rotation_cpu import SmallRotationCPU
    return SmallRotationCPU()


def _make_small_rotation_gpu() -> AugmentationStrategy:
    from .strategy.small_rotation_gpu import SmallRotationGPU
    return SmallRotationGPU()


def _make_wedge_crop_cpu() -> AugmentationStrategy:
    from .strategy.wedge_crop_cpu import WedgeCropCPU
    return WedgeCropCPU()


def _make_wedge_crop_gpu() -> AugmentationStrategy:
    from .strategy.wedge_crop_gpu import WedgeCropGPU
    return WedgeCropGPU()


def _make_segment_crop_cpu() -> AugmentationStrategy:
    from .strategy.segment_crop_cpu import SegmentCropCPU
    return SegmentCropCPU()


def _make_segment_crop_gpu() -> AugmentationStrategy:
    from .strategy.segment_crop_gpu import SegmentCropGPU
    return SegmentCropGPU()


def _make_gingiva_trim_cpu() -> AugmentationStrategy:
    from .strategy.gingiva_trim_cpu import GingivaTrimCPU
    return GingivaTrimCPU()


def _make_gingiva_trim_gpu() -> AugmentationStrategy:
    from .strategy.gingiva_trim_gpu import GingivaTrimGPU
    return GingivaTrimGPU()


def _make_synth_colour_cpu() -> AugmentationStrategy:
    from .strategy.synth_colour_cpu import SynthColourCPU
    return SynthColourCPU()


def _make_synth_colour_gpu() -> AugmentationStrategy:
    from .strategy.synth_colour_gpu import SynthColourGPU
    return SynthColourGPU()


def _make_train_recipe_cpu() -> AugmentationStrategy:
    from .strategy.train_recipe_cpu import TrainRecipeCPU
    return TrainRecipeCPU()


def _make_train_recipe_gpu() -> AugmentationStrategy:
    from .strategy.train_recipe_gpu import TrainRecipeGPU
    return TrainRecipeGPU()


class Augmentation(StrategyAlgorithm[AugmentationInput, AugmentationOutput, AugmentationType]):
    """One labelled full-resolution scan -> the same scan, augmented.

    Each strategy is one augmentation and always applies it (with its own
    random parameters); TRAIN_RECIPE chains them with the training
    probabilities.  Strategies draw every scalar from the CPU generator, so a
    seed picks the same crop, reach and rotation on the CPU and the GPU.
    """

    def __init__(self) -> None:
        super().__init__({
            AugmentationType.MIRROR_CPU: _make_mirror_cpu,
            AugmentationType.MIRROR_GPU: _make_mirror_gpu,
            AugmentationType.RANDOM_SO3_CPU: _make_random_so3_cpu,
            AugmentationType.RANDOM_SO3_GPU: _make_random_so3_gpu,
            AugmentationType.SMALL_ROTATION_CPU: _make_small_rotation_cpu,
            AugmentationType.SMALL_ROTATION_GPU: _make_small_rotation_gpu,
            AugmentationType.WEDGE_CROP_CPU: _make_wedge_crop_cpu,
            AugmentationType.WEDGE_CROP_GPU: _make_wedge_crop_gpu,
            AugmentationType.SEGMENT_CROP_CPU: _make_segment_crop_cpu,
            AugmentationType.SEGMENT_CROP_GPU: _make_segment_crop_gpu,
            AugmentationType.GINGIVA_TRIM_CPU: _make_gingiva_trim_cpu,
            AugmentationType.GINGIVA_TRIM_GPU: _make_gingiva_trim_gpu,
            AugmentationType.SYNTH_COLOUR_CPU: _make_synth_colour_cpu,
            AugmentationType.SYNTH_COLOUR_GPU: _make_synth_colour_gpu,
            AugmentationType.TRAIN_RECIPE_CPU: _make_train_recipe_cpu,
            AugmentationType.TRAIN_RECIPE_GPU: _make_train_recipe_gpu,
        })
