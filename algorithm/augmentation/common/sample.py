from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class AugmentationSample:
    """One labelled full-resolution scan, as the augmentations see it.

    Augmentation runs before downsampling, as in toothseg3d: the voxel grid then
    lands differently on every epoch, a mild augmentation of its own.
    """

    positions: torch.Tensor                  # [V,3] float32, mm
    normals: torch.Tensor                    # [V,3] float32, unit length
    classes: torch.Tensor                    # [V] int64: 0 gingiva, 1..16 FDI class, 17 tooth of unknown position
    instances: torch.Tensor                  # [V] int64: 0 gingiva, one positive id per tooth
    colour: torch.Tensor | None = None       # [V,3] float32 in 0..1; None = colourless

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def to(self, device: torch.device) -> AugmentationSample:
        return AugmentationSample(
            self.positions.to(device), self.normals.to(device), self.classes.to(device),
            self.instances.to(device), None if self.colour is None else self.colour.to(device))

    def select(self, keep: torch.Tensor) -> AugmentationSample:
        """The vertices where `keep` [V] bool is set."""
        return AugmentationSample(
            self.positions[keep], self.normals[keep], self.classes[keep], self.instances[keep],
            None if self.colour is None else self.colour[keep])

    def replace(self, **changes: object) -> AugmentationSample:
        return replace(self, **changes)
