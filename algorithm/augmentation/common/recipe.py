from __future__ import annotations

import torch

from .colour import synth_colour
from .crop import gingiva_trim, segment_crop, wedge_crop
from .mirror import mirror
from .params import AugmentationParams
from .rng import chance
from .rotation import rotate_small, rotate_so3
from .sample import AugmentationSample


def train_recipe(sample: AugmentationSample, params: AugmentationParams,
                 gen: torch.Generator | None) -> AugmentationSample:
    """The training branch of `Teeth3DS.__getitem__`, step for step.

    mirror -> crop (while the arch is still canonically oriented) -> gingiva
    trim -> rotation -> colour.  Colour is decided last, as the original decides
    it after downsampling: with probability colour_p the scan keeps its real
    colour or gets a painted one, otherwise it goes in colourless -- real colour
    included, so the network never comes to depend on it.
    """
    if chance(gen, params.mirror_p):
        sample = mirror(sample, params, gen)
    if chance(gen, params.crop_p):
        crop = wedge_crop if chance(gen, params.wedge_p) else segment_crop
        sample = crop(sample, params, gen) or sample
    if chance(gen, params.trim_p):
        sample = gingiva_trim(sample, params, gen) or sample
    sample = (rotate_so3 if params.rotation == "so3" else rotate_small)(sample, params, gen)
    if not chance(gen, params.colour_p):
        return sample.replace(colour=None)
    if sample.colour is None:
        return sample.replace(colour=synth_colour(sample.positions, sample.classes, sample.instances, gen))
    return sample
