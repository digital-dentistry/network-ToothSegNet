"""Cached scans -> network-ready training items (port of `toothseg3d/dataset.py`).

One item: load the full-resolution scan, augment it, downsample to the
training resolution, centre it, and build the features and geometry with the
same algorithms the inference pipeline runs -- so training and inference can
never disagree on the input.  Everything runs on the host, in DataLoader
workers.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from algorithm.augmentation.augmentation import (Augmentation, AugmentationInput, AugmentationOutput,
                                                 AugmentationType)
from algorithm.augmentation.common.colour import synth_colour
from algorithm.augmentation.common.params import AugmentationParams
from algorithm.augmentation.common.sample import AugmentationSample
from algorithm.loss.common.targets import arch_position, offset_targets
from algorithm.model_input.common.builder import CurveKey
from algorithm.model_input.common.features import standardise_colour, unit_normals
from algorithm.model_input.common.geometry import Geometry, collate as collate_levels
from algorithm.model_input.model_input import (GeometrySpec, ModelInput, ModelInputInput, ModelInputOutput,
                                               model_input_type)
from algorithm.voxel_downsample.voxel_downsample import (VoxelDownsample, VoxelDownsampleInput,
                                                         VoxelDownsampleOutput, VoxelDownsampleType)
from layer.checkpoint import DEC_DEPTHS, ENC_DEPTHS, KV_LEVEL, PATCH

EYE = torch.eye(3, dtype=torch.float64)[None]


@dataclass(frozen=True)
class DatasetConfig:
    voxel_mm: float = 0.25
    in_ch: int = 10                          # 7: xyz, normal, jaw; 10: + standardised RGB
    n_query: int = 32                        # 0 for the semantic stage (no query seeds)
    curve: CurveKey = CurveKey.MORTON3D
    max_points: int | None = 70000           # randomly thin a scan with more L0 points
    train: bool = True                       # augment (TRAIN_RECIPE)
    augmentation: AugmentationParams = field(default_factory=AugmentationParams)
    colour_force: bool = False               # validation: real colour, or painted when there is none
    crop_force: str | None = None            # validation: always crop this way ("bridge" / ...)

    @property
    def geometry_spec(self) -> GeometrySpec:
        return GeometrySpec(enc_depths=ENC_DEPTHS, dec_depths=DEC_DEPTHS, patch=PATCH,
                            pad_level=KV_LEVEL, n_query=self.n_query)


@dataclass
class TrainingItem:
    name: str
    features: torch.Tensor                   # [n, in_ch]
    geometry: Geometry
    classes: torch.Tensor                    # [n] 0..17
    instances: torch.Tensor                  # [n] 0 gingiva, 1..G
    arch_positions: torch.Tensor             # [n] 0..8, -1 unknown
    offset_targets: torch.Tensor             # [n,3]


@dataclass
class TrainingBatch:
    names: list[str]
    counts: list[int]
    features: torch.Tensor                   # [N, in_ch], packed scan by scan
    geometry: Geometry
    classes: torch.Tensor
    instances: torch.Tensor                  # ids per scan; the loss groups scan by scan
    arch_positions: torch.Tensor
    offset_targets: torch.Tensor


def read_split(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _name_generator(name: str) -> torch.Generator:
    """Validation draws (forced crop, painted colour) are fixed per scan."""
    return torch.Generator().manual_seed(zlib.crc32(name.encode()))


class TeethDataset(Dataset):
    def __init__(self, cache_dirs: list[Path], names: list[str], config: DatasetConfig) -> None:
        # several caches may be mixed -- Teeth3DS plus in-house scans -- so a
        # name resolves to whichever one holds it
        self.config = config
        self._path: dict[str, Path] = {}
        for n in names:
            for d in cache_dirs:
                if (d / f"{n}.npz").exists():
                    self._path[n] = d / f"{n}.npz"
                    break
        self.names = [n for n in names if n in self._path]
        self._algorithms: tuple[Augmentation, VoxelDownsample, ModelInput] | None = None

    def __len__(self) -> int:
        return len(self.names)

    def _algos(self) -> tuple[Augmentation, VoxelDownsample, ModelInput]:
        """Built on first use, inside the worker that uses them."""
        if self._algorithms is None:
            down = VoxelDownsample()
            down.set_strategy(VoxelDownsampleType.GRID_CPU)
            model_input = ModelInput()
            model_input.set_strategy(model_input_type(self.config.curve, gpu=False))
            self._algorithms = (Augmentation(), down, model_input)
        return self._algorithms

    def load(self, name: str) -> tuple[AugmentationSample, bool]:
        d = np.load(self._path[name])
        rgb = torch.from_numpy(d["rgb"].astype(np.float32) / 255.0) if "rgb" in d.files else None
        sample = AugmentationSample(torch.from_numpy(d["xyz"].astype(np.float32)),
                                    torch.from_numpy(d["nrm"].astype(np.float32)),
                                    torch.from_numpy(d["cls"].astype(np.int64)),
                                    torch.from_numpy(d["ins"].astype(np.int64)), rgb)
        return sample, bool(d["jaw"])

    def __getitem__(self, i: int) -> TrainingItem:
        cfg = self.config
        augment, down, model_input = self._algos()
        name = self.names[i]
        sample, jaw_upper = self.load(name)
        # training draws from the worker's own torch RNG, which the DataLoader
        # seeds differently per worker; validation draws are fixed per scan
        gen = None if cfg.train else _name_generator(name)

        if cfg.crop_force:
            params = AugmentationParams(crop_mode=cfg.crop_force)
            sample = self._augment(augment, AugmentationType.SEGMENT_CROP_CPU, sample, params, gen)
        if cfg.train:
            sample = self._augment(augment, AugmentationType.TRAIN_RECIPE_CPU, sample, cfg.augmentation, gen)
        elif cfg.colour_force and sample.colour is None:
            sample = sample.replace(colour=synth_colour(sample.positions, sample.classes, sample.instances, gen))
        elif not cfg.colour_force:
            sample = sample.replace(colour=None)

        ds = VoxelDownsampleOutput()
        if not down.execute(VoxelDownsampleInput(sample.positions, cfg.voxel_mm), ds):
            raise ValueError(f"{name}: nothing to downsample")
        keep = torch.arange(ds.representative.numel())
        if cfg.max_points and keep.numel() > cfg.max_points:
            keep = torch.randperm(keep.numel(), generator=gen)[:cfg.max_points]
        rep = ds.representative[keep]
        # centred on the origin, as inference does: no translation to learn
        points = sample.positions[rep] - sample.positions[rep].double().mean(0).float()
        normals = unit_normals(sample.normals[rep])
        classes, instances = sample.classes[rep], sample.instances[rep]
        colour = None
        if cfg.in_ch >= 10:
            colour = (standardise_colour(sample.colour[rep]) if sample.colour is not None
                      else torch.zeros_like(points))          # colourless standardises to zero

        out = ModelInputOutput()
        if not model_input.execute(ModelInputInput(points, normals, jaw_upper, colour, EYE, cfg.voxel_mm,
                                                   cfg.geometry_spec, cells=ds.cells[keep]), out):
            raise ValueError(f"{name}: could not build the network input")
        n = points.shape[0]
        return TrainingItem(name, out.features, out.geometry, classes, instances, arch_position(classes),
                            offset_targets(points, instances, [n]))

    @staticmethod
    def _augment(augment: Augmentation, typ: AugmentationType, sample: AugmentationSample,
                 params: AugmentationParams, gen: torch.Generator | None) -> AugmentationSample:
        augment.set_strategy(typ)
        res = AugmentationOutput()
        augment.execute(AugmentationInput(sample, params, gen), res)
        return res.sample


def collate(items: list[TrainingItem]) -> TrainingBatch:
    """Pack scans into one batch: points concatenated, every geometry index shifted."""
    levels, layouts = collate_levels([(it.geometry.levels, it.geometry.layouts) for it in items])
    seed = torch.cat([it.geometry.seed for it in items])   # offsets into each scan's own row
    geometry = Geometry(levels, layouts, items[0].geometry.pad_level, seed)
    return TrainingBatch(
        names=[it.name for it in items], counts=[it.features.shape[0] for it in items],
        features=torch.cat([it.features for it in items]), geometry=geometry,
        classes=torch.cat([it.classes for it in items]), instances=torch.cat([it.instances for it in items]),
        arch_positions=torch.cat([it.arch_positions for it in items]),
        offset_targets=torch.cat([it.offset_targets for it in items]))
