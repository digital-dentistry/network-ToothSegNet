"""Plausible vertex colour for colourless scans (port of `Teeth3DS._synth_colour`
and `toothseg3d/colour_palette.py`).

The palettes are linear sRGB in 0..1.  Measured on the in-house L-SCAN scans,
enamel and gingiva are only 0.7-1.1 standard deviations apart per channel.  An
earlier toothseg3d version painted them far apart and the network read the
label straight off the colour (val mAP 66.8 -> 38.7); matching the real overlap
is what keeps colour a hint rather than an answer.
"""
from __future__ import annotations

import torch

from .params import AugmentationParams
from .rng import chance, normal, randint, uniform
from .sample import AugmentationSample

# VITA Classical enamel shades (B1 .. C4), a few bleached ones, and the in-house scanner
TOOTH_SHADES = torch.tensor([
    (0.97, 0.96, 0.93), (0.95, 0.93, 0.88), (0.94, 0.91, 0.84), (0.93, 0.89, 0.80),
    (0.87, 0.84, 0.77), (0.89, 0.84, 0.76), (0.91, 0.85, 0.73), (0.92, 0.87, 0.75),
    (0.87, 0.81, 0.71), (0.84, 0.79, 0.68), (0.89, 0.82, 0.67), (0.84, 0.78, 0.68),
    (0.89, 0.82, 0.64), (0.81, 0.75, 0.63), (0.86, 0.77, 0.60), (0.86, 0.77, 0.59),
    (0.82, 0.72, 0.55), (0.75, 0.68, 0.55), (0.884, 0.668, 0.582)])
# healthy coral through inflamed red to physiologic pigmentation, and the in-house scanner
GINGIVA_SHADES = torch.tensor([
    (0.95, 0.80, 0.78), (0.93, 0.68, 0.65), (0.91, 0.60, 0.56), (0.90, 0.52, 0.49),
    (0.85, 0.42, 0.40), (0.76, 0.33, 0.32), (0.68, 0.36, 0.36), (0.55, 0.38, 0.33),
    (0.40, 0.28, 0.26), (0.944, 0.535, 0.478)])
# amalgam, metal, polished metal, gold, zirconia, dark composite: these break
# the "enamel is pale, gingiva is red" rule on purpose
RESTORATION_SHADES = torch.tensor([
    (0.45, 0.46, 0.48), (0.62, 0.63, 0.66), (0.78, 0.79, 0.82), (0.83, 0.69, 0.35),
    (0.93, 0.92, 0.90), (0.70, 0.62, 0.52)])
# measured per-channel spread
TOOTH_SD = torch.tensor([0.101, 0.143, 0.184])
GINGIVA_SD = torch.tensor([0.054, 0.099, 0.117])

RESTORATION_P = 0.35


def synth_colour(positions: torch.Tensor, classes: torch.Tensor, instances: torch.Tensor,
                 gen: torch.Generator | None) -> torch.Tensor:
    """[V,3] colour in 0..1: two base shades, a per-tooth shift, smooth spatial
    drift, measured per-point noise, specular highlights, the occasional metal
    or gold crown, and a white balance."""
    dev, n = positions.device, positions.shape[0]
    t = classes > 0

    base_t = TOOTH_SHADES[randint(gen, 0, len(TOOTH_SHADES))] + normal(gen, (3,), 0.04)
    base_g = GINGIVA_SHADES[randint(gen, 0, len(GINGIVA_SHADES))] + normal(gen, (3,), 0.05)
    # A shade guide separates the two the way a scanner's white balance does
    # not; shrink the gap toward the midpoint by a random factor.
    mix = uniform(gen, 0.08, 0.70)
    mid = (base_t + base_g) / 2
    base_t, base_g = mid + (base_t - mid) * mix, mid + (base_g - mid) * mix
    c = torch.where(t[:, None], base_t.to(dev), base_g.to(dev))

    # every tooth in a mouth is a slightly different shade
    ids = torch.unique(instances)
    ids = ids[ids > 0]
    tooth = instances > 0
    slot = torch.searchsorted(ids, instances[tooth])
    c[tooth] += normal(gen, (ids.numel(), 3), 0.025).to(dev)[slot]

    # smooth spatial variation -- real colour drifts, it does not speckle
    q = (positions - positions.mean(0)) / 10.0
    for ch in range(3):
        w = torch.zeros(n, device=dev)
        for _ in range(3):
            k = normal(gen, (3,), 0.9).to(dev)
            w += torch.sin(q @ k + uniform(gen, 0.0, 6.283))
        c[:, ch] += 0.035 * w / 3.0

    # per-point noise at the measured spread
    sd = torch.where(t[:, None], TOOTH_SD.to(dev), GINGIVA_SD.to(dev))
    c += normal(gen, (n, 3), 1.0, dev) * sd * uniform(gen, 0.6, 1.1)

    # specular highlights wash colour out wherever the light caught
    for _ in range(randint(gen, 0, 4)):
        c0 = positions[randint(gen, 0, n)]
        d2 = (positions - c0).square().sum(1)
        c += (0.35 * torch.exp(-d2 / (2 * uniform(gen, 1.5, 5.0) ** 2)))[:, None]

    # restorations: a crown that is metal, gold or zirconia, not enamel
    if ids.numel() and chance(gen, RESTORATION_P):
        count = min(ids.numel(), randint(gen, 1, 3))
        for j in ids[torch.randperm(ids.numel(), generator=gen)[:count].to(dev)].tolist():
            shade = RESTORATION_SHADES[randint(gen, 0, len(RESTORATION_SHADES))] + normal(gen, (3,), 0.03)
            c[instances == j] = shade.to(dev)

    # white balance / exposure differ between scanners and sessions
    gain = torch.tensor([uniform(gen, 0.85, 1.15) for _ in range(3)], device=dev)
    c = c * gain + float(normal(gen, (), 0.03))
    return c.clamp(0.0, 1.0)


def paint_colour(sample: AugmentationSample, params: AugmentationParams,
                 gen: torch.Generator | None) -> AugmentationSample:
    return sample.replace(colour=synth_colour(sample.positions, sample.classes, sample.instances, gen))
