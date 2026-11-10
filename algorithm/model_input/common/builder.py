from __future__ import annotations

from enum import Enum

import torch

from .features import point_features
from .geometry import Geometry, Level, collate, pad_index
from .hierarchy import legacy_hierarchy, morton3d_hierarchy
from .layout import block_plan, build_layouts, curve_order
from .morton import legacy_key, morton3d_key
from .sampling import farthest_point_seeds


class CurveKey(Enum):
    LEGACY = 0          # what every trained checkpoint expects
    MORTON3D = 1        # real 3D Morton key; needs retraining


def build_model_input(coord: torch.Tensor, normals: torch.Tensor, jaw_upper: bool,
                      colour: torch.Tensor | None, rotations: torch.Tensor,
                      cells: torch.Tensor | None, voxel_mm: float, enc_depths: tuple[int, ...],
                      dec_depths: tuple[int, ...], patch: int, pad_level: int, n_query: int,
                      curve: CurveKey, device: torch.device) -> tuple[torch.Tensor, Geometry]:
    """Features and packed geometry for one cloud seen in `len(rotations)` poses.

    Each pose is one scan of the packed batch, so TTA runs as a single
    forward.  With the legacy key every pose is built in the same sorts (a
    scan prefix on the key); the morton3d builder goes pose by pose.
    """
    coord, normals = coord.to(device), normals.to(device)
    colour = None if colour is None else colour.to(device)
    plan = block_plan(enc_depths, dec_depths)
    n_levels = len(enc_depths)
    poses = [_pose(coord, normals, rot) for rot in rotations]
    feats = torch.cat([point_features(c, n, jaw_upper, colour) for c, n, _ in poses])

    if curve is CurveKey.LEGACY:
        levels = legacy_hierarchy([c for c, _, _ in poses], voxel_mm, n_levels)
        layouts = build_layouts(levels, plan, patch, lambda lvl, var: curve_order(
            levels[lvl].grid, var, legacy_key, levels[lvl].scan))
    else:
        scans = []
        for c, _, identity in poses:
            grid0 = cells.to(device) if (identity and cells is not None) else _regrid(c, voxel_mm)
            lv, order0 = morton3d_hierarchy(c, grid0, n_levels, morton3d_key(grid0))
            layouts = build_layouts(lv, plan, patch,
                                    lambda lvl, var, lv=lv, o=order0: _morton3d_order(lv, o, lvl, var))
            scans.append((lv, layouts))
        levels, layouts = collate(scans)

    for lv in levels:
        lv.grid = lv.scan = None          # build-time only; the network never reads them
    index, valid = pad_index(levels[pad_level].counts, device)
    seed = farthest_point_seeds(levels[pad_level].coord, index, valid, n_query)
    return feats, Geometry(levels, layouts, pad_level, seed)


def _pose(coord: torch.Tensor, normals: torch.Tensor,
          rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """A pose as the original applies it -- `(p @ R.T)` in float64, then cast.
    The identity skips the round trip, so pass 0 is bit-identical to no TTA."""
    if torch.equal(rot, torch.eye(3, dtype=rot.dtype)):
        return coord, normals, True
    r = rot.to(coord.device, torch.float64)
    return (coord.double() @ r.T).float(), (normals.double() @ r.T).float(), False


def _morton3d_order(levels: list[Level], order0: torch.Tensor, lvl: int, var: int) -> torch.Tensor:
    if var == 0:                          # the hierarchy's own order -- no sort
        return order0 if lvl == 0 else torch.arange(levels[lvl].n, device=order0.device)
    return curve_order(levels[lvl].grid, var, morton3d_key)


def _regrid(coord: torch.Tensor, voxel_mm: float) -> torch.Tensor:
    """A rotated pose no longer sits on the downsampling grid; grid it afresh."""
    g = torch.floor(coord / voxel_mm).long()
    return g - g.amin(0)
