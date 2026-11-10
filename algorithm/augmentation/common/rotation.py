from __future__ import annotations

import math

import torch

from .params import AugmentationParams
from .rng import normal, uniform
from .sample import AugmentationSample


def random_so3(gen: torch.Generator | None) -> torch.Tensor:
    """A rotation drawn uniformly from SO(3) (Shoemake's quaternion method), [3,3] float64.

    This is what makes the model indifferent to the frame a scanner writes:
    Teeth3DS puts the occlusal plane in xy, the in-house scanner in xz.
    """
    u1, u2, u3 = torch.rand(3, generator=gen, dtype=torch.float64).tolist()
    a, b = math.sqrt(1 - u1), math.sqrt(u1)
    x, y = a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2)
    z, w = b * math.sin(2 * math.pi * u3), b * math.cos(2 * math.pi * u3)
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=torch.float64)


def small_rotation(gen: torch.Generator | None, rot_xy_deg: float, rot_z_deg: float) -> torch.Tensor:
    """Rx @ Ry @ Rz with a small tilt and a larger turn about z, [3,3] float64."""
    ax = math.radians(uniform(gen, -rot_xy_deg, rot_xy_deg))
    ay = math.radians(uniform(gen, -rot_xy_deg, rot_xy_deg))
    az = math.radians(uniform(gen, -rot_z_deg, rot_z_deg))
    cx, cy, cz = math.cos(ax), math.cos(ay), math.cos(az)
    sx, sy, sz = math.sin(ax), math.sin(ay), math.sin(az)
    rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float64)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float64)
    rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float64)
    return rx @ ry @ rz


def rotate(sample: AugmentationSample, rotation: torch.Tensor, params: AugmentationParams,
           gen: torch.Generator | None) -> AugmentationSample:
    """Rotate, scale and jitter, in float64 as the original does.

    No translation: every scan is centred on the origin after downsampling, at
    inference too, so a translation would be removed again right away.
    """
    dev = sample.device
    r = rotation.to(dev, torch.float64)
    s = uniform(gen, 1 - params.scale, 1 + params.scale)
    pos = (sample.positions.double() @ r.T) * s
    if params.jitter_mm > 0:
        pos = pos + normal(gen, tuple(pos.shape), params.jitter_mm, dev, torch.float64)
    nrm = sample.normals.double() @ r.T                  # normals follow the same matrix
    return sample.replace(positions=pos.float(), normals=nrm.float())


def rotate_so3(sample: AugmentationSample, params: AugmentationParams,
               gen: torch.Generator | None) -> AugmentationSample:
    return rotate(sample, random_so3(gen), params, gen)


def rotate_small(sample: AugmentationSample, params: AugmentationParams,
                 gen: torch.Generator | None) -> AugmentationSample:
    return rotate(sample, small_rotation(gen, params.rot_xy_deg, params.rot_z_deg), params, gen)
