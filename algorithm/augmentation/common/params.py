"""Augmentation knobs.  The defaults are what `toothseg3d/train_inst.py` trained
the shipped instance models with."""
from __future__ import annotations

from dataclasses import dataclass, field

CROP_MODES = ("bridge", "quadrant", "half")


@dataclass
class AugmentationParams:
    # rotation (RANDOM_SO3 / SMALL_ROTATION): scale and jitter apply to both
    scale: float = 0.1                       # uniform in [1 - scale, 1 + scale]
    jitter_mm: float = 0.05                  # per-vertex Gaussian noise
    rot_xy_deg: float = 10.0                 # SMALL_ROTATION: tilt about x and y
    rot_z_deg: float = 25.0                  # SMALL_ROTATION: turn about z

    # partial-scan crops (WEDGE_CROP / SEGMENT_CROP)
    crop_mode: str | None = None             # "bridge" / "quadrant" / "half"; None draws from crop_mix
    crop_mix: tuple[float, float, float] = (0.4, 0.35, 0.25)
    crop_margin_mm: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "bridge": (1.0, 3.0),                # a bridge prep is scanned tight
        "quadrant": (1.5, 4.5),
        "half": (2.0, 6.0),                  # a half arch keeps more of the ridge
    })
    wedge_keep: tuple[float, float] = (0.30, 0.95)   # fraction of the full turn a wedge keeps
    crop_min_points: int = 3000              # a crop that leaves fewer is not applied

    # GINGIVA_TRIM
    trim_reach_mm: tuple[float, float] = (2.5, 14.0)
    trim_min_points: int = 5000

    # TRAIN_RECIPE: how often each step runs
    mirror_p: float = 0.5
    crop_p: float = 0.35
    wedge_p: float = 0.25                    # of the crops, the share that is a plain wedge
    trim_p: float = 0.5
    colour_p: float = 0.6                    # keep real colour or paint one; otherwise colourless
    rotation: str = "so3"                    # "so3" or "limited"
