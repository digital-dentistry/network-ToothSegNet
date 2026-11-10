from __future__ import annotations

import torch

ARCH_RADIUS_MM = 35.0       # xyz / 35 mm puts an arch in roughly [-1, 1]


def standardise_colour(rgb: torch.Tensor) -> torch.Tensor:
    """Per-scan standardised RGB (port of `dataset.standardise_colour`).

    Only relative colour matters -- scanners differ in white balance and
    exposure.  STL carries no colour and Teeth3DS OBJ is a flat grey with
    ~0.004 of quantisation wobble; both standardise to exactly zero, the
    network's "no information" value.
    """
    sd = rgb.std(0, correction=0)
    if float(sd.mean()) < 0.01:
        return torch.zeros_like(rgb)
    return ((rgb - rgb.mean(0)) / sd.clamp_min(1e-6)).clamp(-4.0, 4.0)


def unit_normals(normals: torch.Tensor) -> torch.Tensor:
    return normals / torch.linalg.vector_norm(normals, dim=1, keepdim=True).clamp_min(1e-9)


def point_features(coord: torch.Tensor, normals: torch.Tensor, jaw_upper: bool,
                   colour: torch.Tensor | None) -> torch.Tensor:
    """[xyz / 35 mm, unit normal, jaw flag (, standardised RGB)] -> [N, 7 or 10].

    `normals` must already be unit length; `colour` is the standardised RGB,
    or None for a 7-channel model.
    """
    jaw = torch.full((coord.shape[0], 1), 1.0 if jaw_upper else 0.0, device=coord.device)
    cols = [coord / ARCH_RADIUS_MM, normals, jaw]
    if colour is not None:
        cols.append(colour)
    return torch.cat(cols, 1)
